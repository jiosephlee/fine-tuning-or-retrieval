#!/usr/bin/env python3
"""Build human_blog_*.txt files from curated link lists with depth checks.

Usage:
  source .venv_blog_extract/bin/activate
  python scripts/data_prep/build_human_blogs.py
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path
from typing import Iterable
from urllib.parse import urljoin, urlparse

import html2text
import requests
from bs4 import BeautifulSoup


BASE_DIR = Path("data/arxiv/explanations")
DEFAULT_DOMAINS = ["1_58", "BOFT", "OFT", "QLoRA"]
MIN_WORDS = 850
REQUEST_TIMEOUT = 45
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def _word_count(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9_'-]+", text))


def _normalize_spaces(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _line_is_noise(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if "View all docs" in s:
        return True
    if s.startswith("Transformers documentation"):
        return True
    if s.startswith("PEFT documentation"):
        return True
    if s.startswith("Join the Hugging Face community"):
        return True
    if s == "to get started":
        return True
    if s.startswith("You are viewing main version"):
        return True
    if s in {
        "and get access to the augmented documentation experience",
        "Collaborate on models, datasets and Spaces",
        "Faster examples with accelerated inference",
        "Switch between documentation themes",
    }:
        return True
    if re.fullmatch(r"\[\]\(/join\)", s):
        return True
    if re.fullmatch(r"[-_=*#`~\s]+", s):
        return True
    if len(re.findall(r"v\d+\.\d+\.\d+", s)) >= 3:
        return True
    if re.search(r"mainv\d+\.\d+\.\d+", s):
        return True
    if s.startswith("[←") and s.endswith(")"):
        return True
    if s.startswith("[Back to Articles]") or s.startswith("[ Back to Articles]"):
        return True
    # Many site nav lines appear as a long, punctuation-light token soup.
    if len(s) > 160 and s.count(".") + s.count(",") + s.count(":") < 2:
        return True
    return False


def _slug_keywords(url: str) -> set[str]:
    path = urlparse(url).path.lower()
    words = set(re.findall(r"[a-z0-9]+", path))
    stop = {
        "docs",
        "main",
        "en",
        "blog",
        "package",
        "reference",
        "conceptual",
        "guides",
        "guide",
        "quantization",
        "model",
        "doc",
        "source",
        "category",
        "pages",
        "community",
        "p",
        "ai",
    }
    return {w for w in words if len(w) >= 3 and w not in stop}


def _clean_markdown(md: str, url: str) -> str:
    lines: list[str] = []
    prev = None
    for raw in md.splitlines():
        line = raw.strip()
        if _line_is_noise(line):
            continue
        if line == "":
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if line == prev:
            continue
        lines.append(line)
        prev = line
    # Drop front-matter nav/menus until a relevant heading.
    keywords = _slug_keywords(url)
    first_relevant_heading = None
    for i, ln in enumerate(lines):
        if not re.match(r"^#{1,6}\s+\S", ln):
            continue
        lower_ln = ln.lower()
        if any(k in lower_ln for k in keywords):
            first_relevant_heading = i
            break
    if first_relevant_heading is not None and first_relevant_heading > 0:
        lines = lines[first_relevant_heading:]

    # Fallback: trim until first heading if no relevant heading was found.
    first_heading = None
    for i, ln in enumerate(lines):
        if re.match(r"^#{1,6}\s+\S", ln):
            first_heading = i
            break
    if first_heading is not None and first_heading > 0:
        lines = lines[first_heading:]
    return _normalize_spaces("\n".join(lines))


def _best_content_node(soup: BeautifulSoup):
    selectors = [
        "article",
        "main",
        "div[class*=article]",
        "div[class*=post]",
        "div[class*=content]",
        "section[class*=article]",
        "section[class*=content]",
        "body",
    ]
    candidates = []
    for sel in selectors:
        for node in soup.select(sel):
            wc = _word_count(node.get_text(" ", strip=True))
            if wc >= 120:
                candidates.append((wc, node, sel))
    if not candidates:
        return soup, "soup"
    candidates.sort(key=lambda x: x[0], reverse=True)
    _, node, sel = candidates[0]
    return node, sel


def _fetch_html(url: str) -> str:
    r = requests.get(
        url, headers={"User-Agent": USER_AGENT}, timeout=REQUEST_TIMEOUT, allow_redirects=True
    )
    r.raise_for_status()
    return r.text


def extract_text(url: str) -> tuple[str, int, BeautifulSoup]:
    html = _fetch_html(url)
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(
        ["script", "style", "noscript", "svg", "canvas", "iframe", "form", "button", "input"]
    ):
        tag.decompose()

    node, _ = _best_content_node(soup)
    for sub in node.select("nav,header,footer,aside"):
        sub.decompose()

    parser = html2text.HTML2Text()
    parser.ignore_links = False
    parser.ignore_images = True
    parser.body_width = 0
    parser.single_line_break = True
    parser.ignore_emphasis = False

    md = parser.handle(str(node))
    text = _clean_markdown(md, url)
    return text, _word_count(text), soup


def _expand_arxiv_links(urls: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for url in urls:
        # Prefer full arXiv HTML when an abs/pdf URL is present.
        m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{5})", url)
        if m:
            arxiv_id = m.group(1)
            html_url = f"https://arxiv.org/html/{arxiv_id}"
            if html_url not in seen:
                seen.add(html_url)
                out.append(html_url)

        if url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def supplemental_links(url: str, soup: BeautifulSoup) -> list[str]:
    link_scores: list[tuple[int, str]] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href", "").strip()
        if not href:
            continue
        full = urljoin(url, href)
        parsed = urlparse(full)
        if parsed.scheme not in {"http", "https"}:
            continue
        if "/papers/" in parsed.path:
            # Hugging Face papers pages include AI-generated summaries; avoid
            # mixing these into the "human blog" corpus.
            continue
        score = 0
        lowered = f"{a.get_text(' ', strip=True)} {full}".lower()
        if "arxiv.org" in lowered:
            score += 5
        if "openreview.net" in lowered:
            score += 4
        if "proceedings.neurips.cc" in lowered:
            score += 4
        if "paper" in lowered:
            score += 3
        if "blog" in lowered:
            score += 2
        if "huggingface.co/docs" in lowered:
            score += 2
        if score > 0:
            link_scores.append((score, full))

    link_scores.sort(key=lambda x: x[0], reverse=True)
    deduped: list[str] = []
    seen: set[str] = set([url])
    for _, candidate in link_scores:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return _expand_arxiv_links(deduped)


def build_one(url: str, min_words: int) -> tuple[str, int]:
    base_text, base_words, soup = extract_text(url)
    if base_words >= min_words:
        return base_text, base_words

    parts = [base_text]
    total = base_words
    for extra_url in supplemental_links(url, soup):
        try:
            extra_text, extra_words, _ = extract_text(extra_url)
        except Exception:
            continue
        if extra_words < 200:
            continue
        parts.append(f"\n\n## Supplemental Source: {extra_url}\n\n{extra_text}")
        total = _word_count("\n".join(parts))
        if total >= min_words:
            break
        time.sleep(0.3)

    return _normalize_spaces("\n".join(parts)), total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains",
        nargs="+",
        default=DEFAULT_DOMAINS,
        help="Domain folders under data/arxiv/explanations to process.",
    )
    parser.add_argument(
        "--min-words",
        type=int,
        default=MIN_WORDS,
        help="Minimum extracted word count per output file.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    domains = args.domains
    min_words = args.min_words

    for domain in domains:
        links_path = BASE_DIR / domain / "human_blog_links.txt"
        out_dir = BASE_DIR / domain / "human"
        out_dir.mkdir(parents=True, exist_ok=True)

        urls = [u.strip() for u in links_path.read_text().splitlines() if u.strip()]
        if len(urls) < 3:
            raise RuntimeError(f"{links_path} must contain at least 3 URLs.")

        print(f"\n=== {domain}")
        for i, url in enumerate(urls[:3], start=1):
            text, words = build_one(url, min_words)
            if words < min_words:
                raise RuntimeError(
                    f"{domain} human_blog_{i} below threshold: {words} < {min_words} ({url})"
                )

            header = (
                f"# Source URL\n{url}\n\n"
                f"# Extracted Word Count\n{words}\n\n"
                f"# Extracted Text\n\n"
            )
            output = header + text + "\n"
            out_path = out_dir / f"human_blog_{i}.txt"
            out_path.write_text(output)
            print(f"wrote {out_path} ({words} words)")


if __name__ == "__main__":
    main()
