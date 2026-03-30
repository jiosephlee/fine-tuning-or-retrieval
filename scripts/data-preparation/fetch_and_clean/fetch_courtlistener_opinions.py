"""
Fetch court opinions and their cited precedent cases from CourtListener.

Uses the CourtListener REST API v4 to:
1. Search for opinions by court, date, query terms
2. Download full plain text
3. Fetch the citation graph (which cases does this opinion cite?)
4. Download cited precedent cases

This supports the RQ3 experimental design: comparing training on
the specific cited precedents vs. random cases.

Usage:
    # First, get a free API token at https://www.courtlistener.com/register/
    # Then set it as an environment variable:
    export COURTLISTENER_TOKEN="your_token_here"

    # Search and preview short appellate opinions (dry run)
    python fetch_courtlistener_opinions.py --court ca9 --query "habeas corpus" \
        --max_results 20 --dry_run

    # Download opinions in a token range
    python fetch_courtlistener_opinions.py --court ca9 --query "immigration" \
        --max_results 10 --min_words 1500 --max_words 4000

    # Download an opinion plus all its cited precedents
    python fetch_courtlistener_opinions.py --court ca9 --query "immigration" \
        --max_results 5 --fetch_cited --min_words 1500 --max_words 4000

    # Target unpublished (shorter) memorandum opinions
    python fetch_courtlistener_opinions.py --court ca9 --status Unpublished \
        --max_results 15

    # Filter by date
    python fetch_courtlistener_opinions.py --court scotus \
        --filed_after 2020-01-01 --max_results 10

Notes:
    - Authentication is required for fetching full opinion text.
    - Free accounts get 5,000 queries/hour.
    - Set COURTLISTENER_TOKEN env var or use --token flag.
    - Register at https://www.courtlistener.com/register/
"""

import argparse
import json
import os
import re
import sys
import time
from html.parser import HTMLParser

import requests

API_BASE = "https://www.courtlistener.com/api/rest/v4"

DEFAULT_OUTPUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "legal", "raw"
)


class HTMLTextExtractor(HTMLParser):
    """Strip HTML tags and extract plain text."""

    def __init__(self):
        super().__init__()
        self.parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("p", "br", "div", "h1", "h2", "h3", "h4", "li"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        elif tag == "p":
            self.parts.append("\n")

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def get_text(self):
        return "".join(self.parts)


def html_to_text(html: str) -> str:
    """Convert HTML to plain text."""
    extractor = HTMLTextExtractor()
    extractor.feed(html)
    text = extractor.get_text()
    # Normalize whitespace: collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_session(token: str | None) -> requests.Session:
    """Create an authenticated requests session."""
    session = requests.Session()
    if token:
        session.headers["Authorization"] = f"Token {token}"
    return session


def search_opinions(
    session: requests.Session,
    query: str | None = None,
    court: str | None = None,
    status: str | None = None,
    filed_after: str | None = None,
    filed_before: str | None = None,
    max_results: int = 20,
) -> list[dict]:
    """Search for opinion clusters via the CourtListener search API."""
    params = {"type": "o"}

    # Build query parts
    q_parts = []
    if query:
        q_parts.append(query)
    if q_parts:
        params["q"] = " ".join(q_parts)

    if court:
        params["court"] = court
    if filed_after:
        params["filed_after"] = filed_after
    if filed_before:
        params["filed_before"] = filed_before

    # Status filters
    if status:
        params[f"stat_{status}"] = "on"
    else:
        params["stat_Published"] = "on"

    results = []
    url = f"{API_BASE}/search/"
    page = 0

    while len(results) < max_results and url:
        page += 1
        resp = session.get(url, params=params if page == 1 else None, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("results", []):
            if len(results) >= max_results:
                break
            results.append(item)

        url = data.get("next")
        params = None  # next URL includes params
        time.sleep(0.3)

    count = data.get("count", len(results)) if 'data' in dir() else len(results)
    print(f"Search returned {count} total results, fetching top {len(results)}")
    return results


def fetch_opinion_text(session: requests.Session, opinion_id: int) -> str | None:
    """Fetch the full text of an opinion by its ID."""
    url = f"{API_BASE}/opinions/{opinion_id}/"
    params = {"fields": "id,plain_text,html_with_citations,html,html_columbia,html_lawbox,xml_harvard"}

    resp = session.get(url, params=params, timeout=30)
    if resp.status_code == 403:
        print("  ERROR: Authentication required. Set COURTLISTENER_TOKEN env var.")
        return None
    resp.raise_for_status()
    data = resp.json()

    # Try fields in order of preference
    if data.get("plain_text"):
        return data["plain_text"].strip()

    for field in ("html_with_citations", "html", "html_columbia", "html_lawbox"):
        if data.get(field):
            return html_to_text(data[field])

    if data.get("xml_harvard"):
        return html_to_text(data["xml_harvard"])

    return None


def fetch_cited_opinions(session: requests.Session, opinion_id: int) -> list[dict]:
    """Fetch the list of opinions cited by a given opinion."""
    results = []
    url = f"{API_BASE}/opinions-cited/"
    params = {"citing_opinion": opinion_id}

    while url:
        resp = session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for item in data.get("results", []):
            # Extract the cited opinion ID from the URL
            cited_url = item.get("cited_opinion", "")
            match = re.search(r"/opinions/(\d+)/", cited_url)
            if match:
                results.append({
                    "opinion_id": int(match.group(1)),
                    "depth": item.get("depth", 1),
                })

        url = data.get("next")
        params = None
        time.sleep(0.3)

    return results


def fetch_cluster_detail(session: requests.Session, cluster_id: int) -> dict:
    """Fetch cluster (case) metadata."""
    url = f"{API_BASE}/clusters/{cluster_id}/"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()


def sanitize_filename(name: str, fallback_id: int) -> str:
    """Create a filesystem-safe filename."""
    clean = re.sub(r"[^a-zA-Z0-9 ]", "", name)
    clean = re.sub(r"\s+", "_", clean.strip())
    clean = clean[:60].rstrip("_")
    if not clean:
        clean = f"opinion_{fallback_id}"
    return clean


def save_opinion(
    output_dir: str,
    filename: str,
    text: str,
    metadata: dict,
    subfolder: str | None = None,
) -> str:
    """Save opinion text and metadata to disk."""
    target_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
    os.makedirs(target_dir, exist_ok=True)

    text_path = os.path.join(target_dir, f"{filename}.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")

    meta_path = os.path.join(target_dir, f"{filename}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return text_path


def process_search_results(
    session: requests.Session,
    search_results: list[dict],
    min_words: int,
    max_words: int,
    delay: float,
    dry_run: bool,
    fetch_cited: bool,
    output_dir: str,
) -> list[dict]:
    """Process search results: fetch text, filter, optionally fetch cited cases."""
    results = []

    for i, item in enumerate(search_results):
        case_name = item.get("caseName", "Unknown")
        court = item.get("court_id", "unknown")
        date_filed = item.get("dateFiled", "unknown")
        cluster_id = item.get("cluster_id")

        # Get opinion IDs from search results
        opinions = item.get("opinions", [])
        if not opinions:
            print(f"\n[{i+1}/{len(search_results)}] {case_name[:70]}")
            print(f"  SKIPPED: no opinions in search result")
            continue

        # Use the first (lead) opinion
        opinion_id = opinions[0].get("id")
        if not opinion_id:
            continue

        print(f"\n[{i+1}/{len(search_results)}] {case_name[:70]}")
        print(f"  Court: {court} | Filed: {date_filed} | Opinion ID: {opinion_id}")

        if dry_run:
            results.append({
                "opinion_id": opinion_id,
                "cluster_id": cluster_id,
                "case_name": case_name,
                "court": court,
                "date_filed": date_filed,
                "status": "dry_run",
            })
            continue

        # Fetch full text
        try:
            text = fetch_opinion_text(session, opinion_id)
        except requests.RequestException as e:
            print(f"  ERROR: {e}")
            results.append({
                "opinion_id": opinion_id,
                "case_name": case_name,
                "status": f"fetch_error: {e}",
            })
            time.sleep(delay)
            continue

        if not text:
            print(f"  SKIPPED: no text available")
            results.append({
                "opinion_id": opinion_id,
                "case_name": case_name,
                "status": "no_text",
            })
            time.sleep(delay)
            continue

        word_count = len(text.split())
        print(f"  Words: {word_count}")

        if word_count < min_words:
            print(f"  SKIPPED: below minimum ({min_words})")
            results.append({
                "opinion_id": opinion_id,
                "case_name": case_name,
                "word_count": word_count,
                "status": "too_short",
            })
            time.sleep(delay)
            continue

        if word_count > max_words:
            print(f"  SKIPPED: above maximum ({max_words})")
            results.append({
                "opinion_id": opinion_id,
                "case_name": case_name,
                "word_count": word_count,
                "status": "too_long",
            })
            time.sleep(delay)
            continue

        # Save the opinion
        filename = sanitize_filename(case_name, opinion_id)
        meta = {
            "opinion_id": opinion_id,
            "cluster_id": cluster_id,
            "case_name": case_name,
            "court": court,
            "date_filed": date_filed,
            "word_count": word_count,
            "filename": filename,
        }

        path = save_opinion(output_dir, filename, text, meta)
        print(f"  SAVED: {path}")
        meta["status"] = "saved"
        meta["path"] = path

        # Fetch cited cases if requested
        if fetch_cited:
            print(f"  Fetching citation graph...")
            time.sleep(delay)
            try:
                cited = fetch_cited_opinions(session, opinion_id)
                meta["num_cited"] = len(cited)
                print(f"  Found {len(cited)} cited opinions")

                # Save cited opinion texts
                cited_dir = os.path.join(output_dir, filename, "cited")
                cited_saved = 0
                cited_meta_list = []

                for j, cite in enumerate(cited):
                    cited_id = cite["opinion_id"]
                    time.sleep(delay)

                    try:
                        cited_text = fetch_opinion_text(session, cited_id)
                    except requests.RequestException:
                        continue

                    if not cited_text:
                        continue

                    cited_words = len(cited_text.split())
                    # Save all cited opinions regardless of length —
                    # length filtering is for target opinions only
                    cited_filename = f"cited_{cited_id}"

                    cited_meta = {
                        "opinion_id": cited_id,
                        "depth": cite["depth"],
                        "word_count": cited_words,
                        "cited_by": opinion_id,
                    }

                    save_opinion(cited_dir, cited_filename, cited_text, cited_meta)
                    cited_saved += 1
                    cited_meta_list.append(cited_meta)

                    if (j + 1) % 10 == 0:
                        print(f"    Fetched {j+1}/{len(cited)} cited opinions...")

                meta["cited_saved"] = cited_saved
                meta["cited_opinions"] = cited_meta_list
                print(f"  Saved {cited_saved}/{len(cited)} cited opinion texts")

            except requests.RequestException as e:
                print(f"  ERROR fetching citations: {e}")
                meta["num_cited"] = 0

        results.append(meta)
        time.sleep(delay)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Fetch court opinions and cited precedents from CourtListener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--query", type=str, default=None,
        help="Search terms (e.g., 'habeas corpus', 'immigration')",
    )
    parser.add_argument(
        "--court", type=str, default=None,
        help="Court ID (e.g., 'scotus', 'ca9', 'ca5'). Comma-separated for multiple.",
    )
    parser.add_argument(
        "--status", type=str, default=None,
        choices=["Published", "Unpublished"],
        help="Precedential status filter (default: Published)",
    )
    parser.add_argument(
        "--filed_after", type=str, default=None,
        help="Minimum filing date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--filed_before", type=str, default=None,
        help="Maximum filing date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--max_results", type=int, default=20,
        help="Maximum number of opinions to fetch (default: 20)",
    )
    parser.add_argument(
        "--min_words", type=int, default=1500,
        help="Minimum word count (default: 1500)",
    )
    parser.add_argument(
        "--max_words", type=int, default=5000,
        help="Maximum word count (default: 5000)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--token", type=str, default=None,
        help="CourtListener API token (or set COURTLISTENER_TOKEN env var)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.5,
        help="Delay between API requests in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="Search only, don't download full text",
    )
    parser.add_argument(
        "--fetch_cited", action="store_true",
        help="Also download all cases cited by each target opinion",
    )

    args = parser.parse_args()

    if not args.query and not args.court:
        parser.error("At least one of --query or --court is required")

    token = args.token or os.environ.get("COURTLISTENER_TOKEN")
    if not token and not args.dry_run:
        print("WARNING: No API token provided. Full text fetching requires authentication.")
        print("  Register at https://www.courtlistener.com/register/")
        print("  Then set: export COURTLISTENER_TOKEN='your_token'")
        print("  Or use: --token your_token")
        print()

    session = get_session(token)

    # Search
    print(f"Searching CourtListener...")
    print(f"  Court: {args.court or 'all'}")
    print(f"  Query: {args.query or 'none'}")
    print(f"  Status: {args.status or 'Published'}")
    print(f"  Date range: {args.filed_after or 'any'} to {args.filed_before or 'any'}")
    print()

    search_results = search_opinions(
        session,
        query=args.query,
        court=args.court,
        status=args.status,
        filed_after=args.filed_after,
        filed_before=args.filed_before,
        max_results=args.max_results,
    )

    if not search_results:
        print("No results found.")
        sys.exit(0)

    # Process results
    results = process_search_results(
        session=session,
        search_results=search_results,
        min_words=args.min_words,
        max_words=args.max_words,
        delay=args.delay,
        dry_run=args.dry_run,
        fetch_cited=args.fetch_cited,
        output_dir=args.output_dir,
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    saved = [r for r in results if r.get("status") == "saved"]
    too_short = [r for r in results if r.get("status") == "too_short"]
    too_long = [r for r in results if r.get("status") == "too_long"]
    no_text = [r for r in results if r.get("status") == "no_text"]
    errors = [r for r in results if "error" in r.get("status", "")]

    print(f"  Saved:         {len(saved)}")
    print(f"  Too short:     {len(too_short)}")
    print(f"  Too long:      {len(too_long)}")
    print(f"  No text:       {len(no_text)}")
    print(f"  Errors:        {len(errors)}")

    if saved:
        print(f"\n  Output dir: {os.path.abspath(args.output_dir)}")
        print("\n  Saved opinions:")
        for r in saved:
            cited_info = f", {r.get('cited_saved', '?')} cited" if args.fetch_cited else ""
            print(f"    - {r['case_name'][:60]} ({r['word_count']} words{cited_info})")

    # Save manifest
    if saved and not args.dry_run:
        manifest_path = os.path.join(args.output_dir, "manifest.json")
        os.makedirs(args.output_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
