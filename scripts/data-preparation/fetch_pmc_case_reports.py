"""
Fetch open-access medical case reports from PubMed Central (PMC).

Uses NCBI E-utilities API to search for case reports, filter by length,
and download plain-text full articles. Results are saved in a format
compatible with the existing data pipeline.

Usage:
    # Search and preview candidates (dry run)
    python fetch_pmc_case_reports.py --query "rare diseases" --max_results 20 --dry_run

    # Download top candidates
    python fetch_pmc_case_reports.py --query "rare diseases" --max_results 10

    # Search a specific journal known for case reports
    python fetch_pmc_case_reports.py --journal "J Med Case Rep" --max_results 10

    # Filter by minimum word count in the extracted text
    python fetch_pmc_case_reports.py --query "neurology" --min_words 1500 --max_words 8000

    # Use your own NCBI API key for higher rate limits
    python fetch_pmc_case_reports.py --query "cardiology" --api_key YOUR_KEY

Notes:
    - Without an API key, NCBI limits you to 3 requests/second.
      With a key (free, register at https://www.ncbi.nlm.nih.gov/account/),
      the limit is 10 requests/second.
    - The script respects rate limits with a configurable delay between requests.
"""

import argparse
import json
import os
import re
import sys
import time
from xml.etree import ElementTree as ET

import requests

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
PMC_OA_URL = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

# Default output dir relative to project root (two levels up from scripts/data-preparation/)
DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "medical", "raw"
)


def build_search_query(
    query: str | None = None,
    journal: str | None = None,
    min_date: str | None = None,
    max_date: str | None = None,
) -> str:
    """Build an NCBI ESearch query string for open-access case reports."""
    parts = ['"Case Reports"[Publication Type]', '"open access"[Filter]']
    if query:
        parts.append(f"({query})")
    if journal:
        parts.append(f'"{journal}"[Journal]')
    if min_date and max_date:
        parts.append(f'("{min_date}"[PDAT] : "{max_date}"[PDAT])')
    elif min_date:
        parts.append(f'("{min_date}"[PDAT] : "3000"[PDAT])')
    elif max_date:
        parts.append(f'("1900"[PDAT] : "{max_date}"[PDAT])')
    return " AND ".join(parts)


def esearch(query: str, max_results: int, api_key: str | None = None) -> list[str]:
    """Search PMC and return a list of PMC IDs."""
    params = {
        "db": "pmc",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{EUTILS_BASE}/esearch.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    id_list = data.get("esearchresult", {}).get("idlist", [])
    count = data.get("esearchresult", {}).get("count", "0")
    print(f"Search returned {count} total results, fetching top {len(id_list)}")
    return id_list


def esummary(pmc_ids: list[str], api_key: str | None = None) -> dict:
    """Fetch summary metadata for a list of PMC IDs."""
    if not pmc_ids:
        return {}
    params = {
        "db": "pmc",
        "id": ",".join(pmc_ids),
        "retmode": "json",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{EUTILS_BASE}/esummary.fcgi", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("result", {})


def efetch_fulltext_xml(pmc_id: str, api_key: str | None = None) -> str | None:
    """Fetch the full-text JATS XML for a single PMC article."""
    params = {
        "db": "pmc",
        "id": pmc_id,
        "rettype": "xml",
        "retmode": "xml",
    }
    if api_key:
        params["api_key"] = api_key

    resp = requests.get(f"{EUTILS_BASE}/efetch.fcgi", params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def extract_text_from_jats(xml_text: str) -> dict:
    """
    Parse JATS XML and extract structured plain text.

    Returns a dict with keys:
        title, authors, abstract, sections (list of (heading, text)),
        full_text (concatenated plain text for word counting).
    """
    root = ET.fromstring(xml_text)

    # Handle both <article> root and <pmc-articleset><article>...
    article = root if root.tag == "article" else root.find(".//article")
    if article is None:
        return {"title": "", "authors": [], "abstract": "", "sections": [], "full_text": ""}

    def get_all_text(elem):
        """Recursively extract all text from an element, skipping figures/tables."""
        if elem is None:
            return ""
        parts = []
        if elem.text:
            parts.append(elem.text)
        for child in elem:
            tag = child.tag
            # Skip figures, tables, and graphics
            if tag in ("fig", "table-wrap", "graphic", "inline-graphic", "supplementary-material"):
                if child.tail:
                    parts.append(child.tail)
                continue
            # Handle xref (citations, references) — keep the text
            if tag == "xref":
                if child.text:
                    parts.append(child.text)
                if child.tail:
                    parts.append(child.tail)
                continue
            parts.append(get_all_text(child))
            if child.tail:
                parts.append(child.tail)
        return "".join(parts)

    # Title
    title_elem = article.find(".//article-title")
    title = get_all_text(title_elem).strip() if title_elem is not None else ""

    # Authors
    authors = []
    for contrib in article.findall(".//contrib[@contrib-type='author']"):
        surname = contrib.findtext(".//surname", "")
        given = contrib.findtext(".//given-names", "")
        if surname:
            authors.append(f"{given} {surname}".strip())

    # Abstract
    abstract_elem = article.find(".//abstract")
    abstract = get_all_text(abstract_elem).strip() if abstract_elem is not None else ""

    # Body sections
    sections = []
    body = article.find(".//body")
    if body is not None:
        for sec in body.findall(".//sec"):
            heading_elem = sec.find("title")
            heading = get_all_text(heading_elem).strip() if heading_elem is not None else ""
            # Get paragraphs directly in this section (not nested sub-sections)
            paragraphs = []
            for p in sec.findall("p"):
                p_text = get_all_text(p).strip()
                if p_text:
                    paragraphs.append(p_text)
            if paragraphs:
                sections.append((heading, "\n\n".join(paragraphs)))

    # If no sections found, try to get all paragraphs from body directly
    if not sections and body is not None:
        paragraphs = []
        for p in body.findall(".//p"):
            p_text = get_all_text(p).strip()
            if p_text:
                paragraphs.append(p_text)
        if paragraphs:
            sections.append(("", "\n\n".join(paragraphs)))

    # Build full text
    full_parts = []
    if title:
        full_parts.append(title)
    if abstract:
        full_parts.append(abstract)
    for heading, text in sections:
        if heading:
            full_parts.append(heading)
        full_parts.append(text)
    full_text = "\n\n".join(full_parts)

    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "sections": sections,
        "full_text": full_text,
    }


def format_as_plain_text(parsed: dict) -> str:
    """Format parsed article as clean plain text suitable for the training pipeline."""
    parts = []

    if parsed["title"]:
        parts.append(parsed["title"])
        parts.append("")

    if parsed["authors"]:
        parts.append(", ".join(parsed["authors"]))
        parts.append("")

    if parsed["abstract"]:
        parts.append("Abstract")
        parts.append("")
        parts.append(parsed["abstract"])
        parts.append("")

    for heading, text in parsed["sections"]:
        if heading:
            parts.append(heading)
            parts.append("")
        parts.append(text)
        parts.append("")

    return "\n".join(parts).strip() + "\n"


def sanitize_filename(title: str, pmc_id: str) -> str:
    """Create a short, filesystem-safe filename from the article title."""
    # Take first ~60 chars of title, keep only alphanumeric and spaces
    clean = re.sub(r"[^a-zA-Z0-9 ]", "", title)
    clean = re.sub(r"\s+", "_", clean.strip())
    clean = clean[:60].rstrip("_")
    if not clean:
        clean = f"PMC{pmc_id}"
    return clean


def fetch_and_process_articles(
    pmc_ids: list[str],
    summaries: dict,
    min_words: int,
    max_words: int,
    api_key: str | None,
    delay: float,
    dry_run: bool,
    output_dir: str,
) -> list[dict]:
    """Fetch full text for each PMC ID, filter by word count, and optionally save."""
    results = []

    for i, pmc_id in enumerate(pmc_ids):
        summary = summaries.get(pmc_id, {})
        title = summary.get("title", f"PMC{pmc_id}")
        journal = summary.get("fulljournalname", summary.get("source", "Unknown"))

        print(f"\n[{i+1}/{len(pmc_ids)}] PMC{pmc_id}: {title[:80]}")
        print(f"  Journal: {journal}")

        if dry_run:
            results.append({
                "pmc_id": pmc_id,
                "title": title,
                "journal": journal,
                "status": "dry_run",
            })
            continue

        try:
            xml_text = efetch_fulltext_xml(pmc_id, api_key=api_key)
        except requests.RequestException as e:
            print(f"  ERROR fetching full text: {e}")
            results.append({
                "pmc_id": pmc_id,
                "title": title,
                "journal": journal,
                "status": f"fetch_error: {e}",
            })
            time.sleep(delay)
            continue

        parsed = extract_text_from_jats(xml_text)
        word_count = len(parsed["full_text"].split())
        print(f"  Words: {word_count}")

        if word_count < min_words:
            print(f"  SKIPPED: below minimum word count ({min_words})")
            results.append({
                "pmc_id": pmc_id,
                "title": title,
                "journal": journal,
                "word_count": word_count,
                "status": "too_short",
            })
            time.sleep(delay)
            continue

        if word_count > max_words:
            print(f"  SKIPPED: above maximum word count ({max_words})")
            results.append({
                "pmc_id": pmc_id,
                "title": title,
                "journal": journal,
                "word_count": word_count,
                "status": "too_long",
            })
            time.sleep(delay)
            continue

        plain_text = format_as_plain_text(parsed)
        filename = sanitize_filename(parsed["title"] or title, pmc_id)

        os.makedirs(output_dir, exist_ok=True)

        # Save plain text
        text_path = os.path.join(output_dir, f"{filename}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(plain_text)

        # Save metadata
        meta_path = os.path.join(output_dir, f"{filename}.meta.json")
        meta = {
            "pmc_id": pmc_id,
            "title": parsed["title"] or title,
            "authors": parsed["authors"],
            "journal": journal,
            "word_count": word_count,
            "num_sections": len(parsed["sections"]),
            "filename": filename,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"  SAVED: {text_path}")
        results.append({**meta, "status": "saved", "path": text_path})

        time.sleep(delay)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch open-access case reports from PubMed Central",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Topic search terms (e.g., 'rare diseases', 'neurology')",
    )
    parser.add_argument(
        "--journal", type=str, default=None,
        help='Journal filter (e.g., "J Med Case Rep", "BMJ Case Rep")',
    )
    parser.add_argument(
        "--max_results", type=int, default=20,
        help="Maximum number of PMC articles to fetch (default: 20)",
    )
    parser.add_argument(
        "--min_words", type=int, default=1000,
        help="Minimum word count for extracted text (default: 1000)",
    )
    parser.add_argument(
        "--max_words", type=int, default=10000,
        help="Maximum word count for extracted text (default: 10000)",
    )
    parser.add_argument(
        "--min_date", type=str, default=None,
        help="Minimum publication date (YYYY/MM/DD or YYYY)",
    )
    parser.add_argument(
        "--max_date", type=str, default=None,
        help="Maximum publication date (YYYY/MM/DD or YYYY)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for downloaded articles",
    )
    parser.add_argument(
        "--api_key", type=str, default=None,
        help="NCBI API key for higher rate limits (get one at ncbi.nlm.nih.gov/account)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.4,
        help="Delay in seconds between API requests (default: 0.4)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Only search and print results, don't download full text",
    )

    args = parser.parse_args()

    if not args.query and not args.journal:
        parser.error("At least one of --query or --journal is required")

    # Build and run search
    query = build_search_query(
        query=args.query,
        journal=args.journal,
        min_date=args.min_date,
        max_date=args.max_date,
    )
    print(f"Search query: {query}")
    print()

    pmc_ids = esearch(query, max_results=args.max_results, api_key=args.api_key)
    if not pmc_ids:
        print("No results found.")
        sys.exit(0)

    time.sleep(args.delay)

    # Fetch summaries for metadata
    summaries = esummary(pmc_ids, api_key=args.api_key)
    time.sleep(args.delay)

    # Fetch and process articles
    results = fetch_and_process_articles(
        pmc_ids=pmc_ids,
        summaries=summaries,
        min_words=args.min_words,
        max_words=args.max_words,
        api_key=args.api_key,
        delay=args.delay,
        dry_run=args.dry_run,
        output_dir=args.output_dir,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    saved = [r for r in results if r.get("status") == "saved"]
    skipped_short = [r for r in results if r.get("status") == "too_short"]
    skipped_long = [r for r in results if r.get("status") == "too_long"]
    errors = [r for r in results if "error" in r.get("status", "")]

    print(f"  Saved:         {len(saved)}")
    print(f"  Too short:     {len(skipped_short)}")
    print(f"  Too long:      {len(skipped_long)}")
    print(f"  Errors:        {len(errors)}")

    if saved:
        print(f"\n  Output dir: {os.path.abspath(args.output_dir)}")
        print("\n  Saved articles:")
        for r in saved:
            print(f"    - {r['title'][:70]} ({r['word_count']} words)")

    # Save manifest
    if saved and not args.dry_run:
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
