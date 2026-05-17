#!/usr/bin/env python3
"""Resolve local arXiv seed papers and download TeX sources for cited arXiv works."""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import os
import re
import shutil
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote
from xml.etree import ElementTree

import requests


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
S2_API_URL = "https://api.semanticscholar.org/graph/v1"
OPENALEX_API_URL = "https://api.openalex.org"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

DEFAULT_TITLE_OVERRIDES = {
    "1_58": "The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits",
    "BOFT": "Parameter-Efficient Orthogonal Finetuning via Butterfly Factorization",
    "ByteLatent": "Byte Latent Transformer: Patches Scale Better Than Tokens",
    "DPO": "Direct Preference Optimization: Your Language Model is Secretly a Reward Model",
    "FeatLLM": "Large Language Models Can Automatically Engineer Features for Few-Shot Tabular Learning",
    "GRPO": "DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
    "GSPO": "Group Sequence Policy Optimization",
    "LongRoPE": "LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens",
    "OFT": "Controlling Text-to-Image Diffusion by Orthogonal Finetuning",
    "QLoRA": "QLoRA: Efficient Finetuning of Quantized LLMs",
    "fa3": "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision",
    "xLSTM": "xLSTM: Extended Long Short-Term Memory",
}

DEFAULT_ARXIV_ID_OVERRIDES = {
    "1_58": "2402.17764",
    "BOFT": "2311.06243",
    "ByteLatent": "2412.09871",
    "DPO": "2305.18290",
    "FeatLLM": "2404.09491",
    "GRPO": "2402.03300",
    "GSPO": "2507.18071",
    "LongRoPE": "2402.13753",
    "OFT": "2306.07280",
    "QLoRA": "2305.14314",
    "fa3": "2407.08608",
    "xLSTM": "2405.04517",
}


@dataclass(frozen=True)
class SeedPaper:
    local_id: str
    path: Path
    title: str


@dataclass
class RateLimiter:
    min_interval_seconds: float
    last_request_at: float = 0.0

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        now = time.monotonic()
        elapsed = now - self.last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self.last_request_at = time.monotonic()


class ApiError(RuntimeError):
    pass


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def request_json(
    session: requests.Session,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = 60.0,
    max_retries: int = 6,
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(max_retries + 1):
        if rate_limiter is not None:
            rate_limiter.wait()
        response = session.get(url, headers=headers, params=params, timeout=timeout)
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == max_retries:
                break
            retry_after = response.headers.get("Retry-After")
            try:
                wait = float(retry_after) if retry_after else delay
            except ValueError:
                wait = delay
            time.sleep(wait)
            delay *= 2
            continue
        if response.status_code >= 400:
            raise ApiError(f"{response.status_code} from {response.url}: {response.text[:300]}")
        return response.json()
    raise ApiError(f"{response.status_code} from {response.url}: {response.text[:300]}")


def request_text(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    timeout: float = 60.0,
    max_retries: int = 4,
) -> str:
    delay = 2.0
    for attempt in range(max_retries + 1):
        response = session.get(url, params=params, timeout=timeout)
        if response.status_code == 429 or 500 <= response.status_code < 600:
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay *= 2
            continue
        if response.status_code >= 400:
            raise ApiError(f"{response.status_code} from {response.url}: {response.text[:300]}")
        return response.text
    raise ApiError(f"{response.status_code} from {response.url}: {response.text[:300]}")


def extract_braced_argument(text: str, command: str) -> str | None:
    pattern = "\\" + command
    start = text.find(pattern)
    if start < 0:
        return None

    index = start + len(pattern)
    while index < len(text) and text[index].isspace():
        index += 1
    if index < len(text) and text[index] == "[":
        depth = 1
        index += 1
        while index < len(text) and depth:
            if text[index] == "[":
                depth += 1
            elif text[index] == "]":
                depth -= 1
            index += 1
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        return None

    index += 1
    depth = 1
    out: list[str] = []
    while index < len(text) and depth:
        char = text[index]
        if char == "\\":
            if index + 1 < len(text):
                out.append(char)
                index += 1
                out.append(text[index])
                index += 1
                continue
        if char == "{":
            depth += 1
            out.append(char)
        elif char == "}":
            depth -= 1
            if depth:
                out.append(char)
        else:
            out.append(char)
        index += 1
    return "".join(out) if depth == 0 else None


def simple_latex_macros(text: str) -> dict[str, str]:
    macros: dict[str, str] = {}
    for match in re.finditer(r"\\newcommand\s*\{\\([A-Za-z]+)\}\s*\{([^{}]*)\}", text):
        macros[match.group(1)] = match.group(2)
    return macros


def clean_latex_title(raw_title: str, macros: dict[str, str]) -> str:
    title = raw_title
    for name, value in macros.items():
        title = re.sub(rf"\\{re.escape(name)}\b\s*~?", value, title)

    title = title.replace("\\\\", " ")
    title = title.replace("\\linebreak", " ")
    title = title.replace("~", " ")
    title = re.sub(r"\\(?:vspace|hspace)\*?\{[^{}]*\}", " ", title)
    title = re.sub(r"\\fontsize\{[^{}]*\}\{[^{}]*\}\\selectfont", " ", title)
    title = re.sub(r"\\(?:centering|selectfont|xspace)\b", " ", title)
    title = re.sub(r"\\text(?:sc|sf|bf|it|normal)\{([^{}]*)\}", r"\1", title)
    title = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", title)
    title = title.replace("{", " ").replace("}", " ")
    title = re.sub(r"\s+", " ", title)
    return title.strip(" -\t\n")


def seed_title_from_tex(path: Path) -> str | None:
    text = read_text(path)
    macros = simple_latex_macros(text)
    for command in ("title", "icmltitle"):
        raw_title = extract_braced_argument(text, command)
        if raw_title:
            cleaned = clean_latex_title(raw_title, macros)
            if cleaned:
                return cleaned
    return None


def discover_seed_papers(raw_dir: Path, selected: set[str] | None) -> list[SeedPaper]:
    papers: list[SeedPaper] = []
    for path in sorted(raw_dir.glob("*.tex")):
        local_id = path.stem
        if selected and local_id not in selected and path.name not in selected:
            continue
        title = DEFAULT_TITLE_OVERRIDES.get(local_id) or seed_title_from_tex(path)
        if not title:
            raise ValueError(f"Could not infer title for {path}")
        papers.append(SeedPaper(local_id=local_id, path=path, title=title))
    return papers


def normalize_title(title: str) -> str:
    return re.sub(r"\W+", " ", title).strip().lower()


def strip_arxiv_version(arxiv_id: str) -> str:
    return re.sub(r"v\d+$", "", arxiv_id)


def safe_arxiv_id(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_")


def parse_arxiv_entry(entry: ElementTree.Element) -> dict[str, Any]:
    entry_id = entry.findtext("atom:id", default="", namespaces=ATOM_NS)
    arxiv_id = entry_id.rstrip("/").split("/")[-1]
    title = entry.findtext("atom:title", default="", namespaces=ATOM_NS)
    published = entry.findtext("atom:published", default="", namespaces=ATOM_NS)
    authors = [
        author.findtext("atom:name", default="", namespaces=ATOM_NS)
        for author in entry.findall("atom:author", ATOM_NS)
    ]
    return {
        "arxiv_id": strip_arxiv_version(arxiv_id),
        "arxiv_versioned_id": arxiv_id,
        "title": re.sub(r"\s+", " ", title).strip(),
        "authors": [author for author in authors if author],
        "published": published,
        "year": int(published[:4]) if published[:4].isdigit() else None,
        "abstract_url": f"https://arxiv.org/abs/{strip_arxiv_version(arxiv_id)}",
        "pdf_url": f"https://arxiv.org/pdf/{strip_arxiv_version(arxiv_id)}",
    }


def search_arxiv_by_title(
    session: requests.Session,
    title: str,
    *,
    sleep_seconds: float,
) -> dict[str, Any] | None:
    queries = [f'ti:"{title}"', f'all:"{title}"']
    best: dict[str, Any] | None = None
    best_score = -1
    target = normalize_title(title)

    for query in queries:
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 5,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        text = request_text(session, ARXIV_API_URL, params=params)
        root = ElementTree.fromstring(text)
        for entry in root.findall("atom:entry", ATOM_NS):
            candidate = parse_arxiv_entry(entry)
            candidate_title = normalize_title(candidate["title"])
            common = len(set(target.split()) & set(candidate_title.split()))
            exact_bonus = 100 if target == candidate_title else 0
            containment_bonus = 20 if target in candidate_title or candidate_title in target else 0
            score = common + exact_bonus + containment_bonus
            if score > best_score:
                best = candidate
                best_score = score
        if best and best_score >= 100:
            break
        time.sleep(sleep_seconds)
    return best


def s2_headers(api_key: str | None) -> dict[str, str]:
    headers = {"User-Agent": "fine-tuning-or-retrieval cited arxiv fetcher"}
    if api_key:
        headers["x-api-key"] = api_key
    return headers


def openalex_params(email: str | None) -> dict[str, str]:
    return {"mailto": email} if email else {}


def openalex_work_id_suffix(openalex_id: str) -> str:
    return openalex_id.rstrip("/").split("/")[-1]


def openalex_authors(work: dict[str, Any]) -> list[str]:
    authors: list[str] = []
    for authorship in work.get("authorships") or []:
        author = authorship.get("author") or {}
        name = author.get("display_name") or authorship.get("raw_author_name")
        if name:
            authors.append(name)
    return authors


def extract_arxiv_id_from_text(value: str | None) -> str | None:
    if not value:
        return None
    patterns = [
        r"arxiv[.:/ ]+([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
        r"arxiv[.:/ ]+([a-z-]+(?:\.[A-Z]{2})?/[0-9]{7}(?:v[0-9]+)?)",
        r"arxiv\.org/(?:abs|pdf|e-print)/([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)",
        r"arxiv\.org/(?:abs|pdf|e-print)/([a-z-]+(?:\.[A-Z]{2})?/[0-9]{7}(?:v[0-9]+)?)",
    ]
    for pattern in patterns:
        match = re.search(pattern, value, flags=re.IGNORECASE)
        if match:
            return strip_arxiv_version(match.group(1).removesuffix(".pdf"))
    return None


def extract_arxiv_id_from_openalex_work(work: dict[str, Any]) -> str | None:
    ids = work.get("ids") or {}
    for value in ids.values():
        arxiv_id = extract_arxiv_id_from_text(str(value))
        if arxiv_id:
            return arxiv_id

    for key in ("doi", "id"):
        arxiv_id = extract_arxiv_id_from_text(work.get(key))
        if arxiv_id:
            return arxiv_id

    locations = []
    if work.get("primary_location"):
        locations.append(work["primary_location"])
    locations.extend(work.get("locations") or [])
    for location in locations:
        arxiv_id = extract_arxiv_id_from_text(location.get("landing_page_url"))
        if arxiv_id:
            return arxiv_id
        arxiv_id = extract_arxiv_id_from_text(location.get("pdf_url"))
        if arxiv_id:
            return arxiv_id
    return None


def openalex_work_to_manifest_entry(work: dict[str, Any]) -> dict[str, Any] | None:
    arxiv_id = extract_arxiv_id_from_openalex_work(work)
    if not arxiv_id:
        return None
    return {
        "arxiv_id": arxiv_id,
        "openalex_id": work.get("id"),
        "title": work.get("display_name"),
        "year": work.get("publication_year"),
        "venue": ((work.get("primary_location") or {}).get("source") or {}).get("display_name"),
        "url": work.get("id"),
        "external_ids": work.get("ids") or {},
        "authors": openalex_authors(work),
        "source_status": "pending",
    }


def fetch_s2_paper_by_arxiv(
    session: requests.Session,
    arxiv_id: str,
    *,
    headers: dict[str, str],
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any] | None:
    fields = ",".join(
        [
            "paperId",
            "title",
            "year",
            "venue",
            "url",
            "externalIds",
            "authors.name",
            "references.paperId",
            "references.title",
            "references.year",
            "references.venue",
            "references.url",
            "references.externalIds",
        ]
    )
    url = f"{S2_API_URL}/paper/ARXIV:{quote(arxiv_id)}"
    try:
        return request_json(
            session,
            url,
            headers=headers,
            params={"fields": fields},
            rate_limiter=rate_limiter,
        )
    except ApiError as exc:
        if "404" in str(exc):
            return None
        raise


def search_s2_paper_by_title(
    session: requests.Session,
    title: str,
    *,
    headers: dict[str, str],
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any] | None:
    fields = "paperId,title,year,venue,url,externalIds,authors.name"
    data = request_json(
        session,
        f"{S2_API_URL}/paper/search",
        headers=headers,
        params={"query": title, "limit": 5, "fields": fields},
        rate_limiter=rate_limiter,
    )
    target = normalize_title(title)
    best = None
    best_score = -1
    for candidate in data.get("data", []):
        candidate_title = normalize_title(candidate.get("title") or "")
        common = len(set(target.split()) & set(candidate_title.split()))
        exact_bonus = 100 if target == candidate_title else 0
        score = common + exact_bonus
        if score > best_score:
            best = candidate
            best_score = score
    if best and best.get("paperId"):
        return fetch_s2_paper_by_id(
            session,
            best["paperId"],
            headers=headers,
            rate_limiter=rate_limiter,
        )
    return None


def search_s2_seed_by_title(
    session: requests.Session,
    title: str,
    *,
    headers: dict[str, str],
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any] | None:
    fields = "paperId,title,year,venue,url,externalIds,authors.name"
    data = request_json(
        session,
        f"{S2_API_URL}/paper/search",
        headers=headers,
        params={"query": title, "limit": 5, "fields": fields},
        rate_limiter=rate_limiter,
    )
    target = normalize_title(title)
    target_words = set(target.split())
    best = None
    best_score = -1
    for candidate in data.get("data", []):
        candidate_title = normalize_title(candidate.get("title") or "")
        common = len(target_words & set(candidate_title.split()))
        exact_bonus = 100 if target == candidate_title else 0
        containment_bonus = 20 if target in candidate_title or candidate_title in target else 0
        score = common + exact_bonus + containment_bonus
        if score > best_score:
            best = candidate
            best_score = score
    threshold = max(8, len(target_words) // 2)
    return best if best_score >= threshold else None


def fetch_s2_paper_by_id(
    session: requests.Session,
    paper_id: str,
    *,
    headers: dict[str, str],
    rate_limiter: RateLimiter | None = None,
) -> dict[str, Any]:
    fields = ",".join(
        [
            "paperId",
            "title",
            "year",
            "venue",
            "url",
            "externalIds",
            "authors.name",
            "references.paperId",
            "references.title",
            "references.year",
            "references.venue",
            "references.url",
            "references.externalIds",
        ]
    )
    return request_json(
        session,
        f"{S2_API_URL}/paper/{quote(paper_id)}",
        headers=headers,
        params={"fields": fields},
        rate_limiter=rate_limiter,
    )


def fetch_openalex_work_by_arxiv(
    session: requests.Session,
    arxiv_id: str,
    *,
    mailto: str | None,
) -> dict[str, Any] | None:
    fields = ",".join(
        [
            "id",
            "doi",
            "display_name",
            "publication_year",
            "ids",
            "referenced_works",
            "authorships",
            "primary_location",
            "locations",
        ]
    )
    params = {
        "filter": f"doi:10.48550/arxiv.{arxiv_id}",
        "select": fields,
        "per-page": 1,
        **openalex_params(mailto),
    }
    data = request_json(session, f"{OPENALEX_API_URL}/works", params=params)
    results = data.get("results") or []
    return results[0] if results else None


def search_openalex_work_by_title(
    session: requests.Session,
    title: str,
    *,
    mailto: str | None,
) -> dict[str, Any] | None:
    fields = ",".join(
        [
            "id",
            "doi",
            "display_name",
            "publication_year",
            "ids",
            "referenced_works",
            "authorships",
            "primary_location",
            "locations",
        ]
    )
    params = {
        "search": title,
        "select": fields,
        "per-page": 5,
        **openalex_params(mailto),
    }
    data = request_json(session, f"{OPENALEX_API_URL}/works", params=params)
    target = normalize_title(title)
    target_words = set(target.split())
    best = None
    best_score = -1
    for candidate in data.get("results") or []:
        candidate_title = normalize_title(candidate.get("display_name") or "")
        common = len(target_words & set(candidate_title.split()))
        exact_bonus = 100 if target == candidate_title else 0
        containment_bonus = 20 if target in candidate_title or candidate_title in target else 0
        score = common + exact_bonus + containment_bonus
        if score > best_score:
            best = candidate
            best_score = score
    threshold = max(8, len(target_words) // 2)
    return best if best_score >= threshold else None


def fetch_openalex_works_by_ids(
    session: requests.Session,
    openalex_ids: list[str],
    *,
    mailto: str | None,
) -> list[dict[str, Any]]:
    if not openalex_ids:
        return []
    fields = ",".join(
        [
            "id",
            "doi",
            "display_name",
            "publication_year",
            "ids",
            "authorships",
            "primary_location",
            "locations",
        ]
    )
    results: list[dict[str, Any]] = []
    for start in range(0, len(openalex_ids), 50):
        batch = [openalex_work_id_suffix(openalex_id) for openalex_id in openalex_ids[start : start + 50]]
        params = {
            "filter": "openalex:" + "|".join(batch),
            "select": fields,
            "per-page": len(batch),
            **openalex_params(mailto),
        }
        data = request_json(session, f"{OPENALEX_API_URL}/works", params=params)
        results.extend(data.get("results") or [])
    return results


def reference_to_manifest_entry(reference: dict[str, Any]) -> dict[str, Any] | None:
    external_ids = reference.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv")
    if not arxiv_id:
        return None
    arxiv_id = strip_arxiv_version(str(arxiv_id))
    authors = reference.get("authors") or []
    return {
        "arxiv_id": arxiv_id,
        "semantic_scholar_paper_id": reference.get("paperId"),
        "title": reference.get("title"),
        "year": reference.get("year"),
        "venue": reference.get("venue"),
        "url": reference.get("url"),
        "external_ids": external_ids,
        "authors": [author.get("name") for author in authors if author.get("name")],
        "source_status": "pending",
    }


def looks_like_pdf(content: bytes) -> bool:
    return content[:5] == b"%PDF-"


def looks_like_tex(content: bytes) -> bool:
    sample = content[:20000].decode("utf-8", errors="ignore")
    return "\\documentclass" in sample or "\\begin{document}" in sample


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def extract_source_archive(content: bytes, destination: Path) -> tuple[str, list[str]]:
    remove_dir(destination)
    destination.mkdir(parents=True, exist_ok=True)

    extracted_files: list[str] = []
    buffer = io.BytesIO(content)

    try:
        with tarfile.open(fileobj=buffer, mode="r:*") as tar:
            tar.extractall(destination, filter="data")
            extracted_files = [str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()]
            tex_files = [path for path in extracted_files if path.lower().endswith(".tex")]
            return ("ok" if tex_files else "no_tex_files", tex_files)
    except tarfile.TarError:
        pass

    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            archive.extractall(destination)
            extracted_files = [str(path.relative_to(destination)) for path in destination.rglob("*") if path.is_file()]
            tex_files = [path for path in extracted_files if path.lower().endswith(".tex")]
            return ("ok" if tex_files else "no_tex_files", tex_files)
    except zipfile.BadZipFile:
        pass

    try:
        decompressed = gzip.decompress(content)
    except OSError:
        decompressed = content

    if looks_like_pdf(decompressed):
        remove_dir(destination)
        return "pdf_response", []

    if looks_like_tex(decompressed):
        tex_path = destination / "main.tex"
        tex_path.write_bytes(decompressed)
        return "ok", ["main.tex"]

    raw_path = destination / "source"
    raw_path.write_bytes(decompressed)
    return "unrecognized_source_format", []


def download_arxiv_source(
    session: requests.Session,
    arxiv_id: str,
    *,
    output_dir: Path,
    force: bool,
    timeout: float,
) -> dict[str, Any]:
    safe_id = safe_arxiv_id(arxiv_id)
    source_dir = output_dir / "sources" / safe_id
    archive_dir = output_dir / "source_archives"
    archive_path = archive_dir / f"{safe_id}.eprint"

    existing_tex = sorted(str(path.relative_to(source_dir)) for path in source_dir.rglob("*.tex")) if source_dir.exists() else []
    if existing_tex and not force:
        return {
            "source_status": "ok",
            "source_dir": str(source_dir),
            "archive_path": str(archive_path) if archive_path.exists() else None,
            "tex_files": existing_tex,
            "downloaded": False,
        }

    url = ARXIV_EPRINT_URL.format(arxiv_id=quote(arxiv_id, safe="/"))
    response = session.get(url, timeout=timeout)
    if response.status_code >= 400:
        return {
            "source_status": f"http_{response.status_code}",
            "source_dir": None,
            "archive_path": None,
            "tex_files": [],
            "downloaded": False,
        }

    content = response.content
    if looks_like_pdf(content):
        return {
            "source_status": "pdf_response",
            "source_dir": None,
            "archive_path": None,
            "tex_files": [],
            "downloaded": False,
        }

    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path.write_bytes(content)
    status, tex_files = extract_source_archive(content, source_dir)
    return {
        "source_status": status,
        "source_dir": str(source_dir) if source_dir.exists() else None,
        "archive_path": str(archive_path),
        "tex_files": tex_files,
        "downloaded": True,
    }


def write_edges_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed_local_id",
        "seed_arxiv_id",
        "seed_title",
        "cited_arxiv_id",
        "cited_title",
        "cited_year",
        "semantic_scholar_paper_id",
        "openalex_id",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_failures_csv(path: Path, cited_papers: dict[str, dict[str, Any]]) -> None:
    failures = [
        paper for paper in cited_papers.values() if paper.get("source_status") not in {"ok", "pending"}
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["arxiv_id", "title", "source_status", "source_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for paper in sorted(failures, key=lambda item: item["arxiv_id"]):
            writer.writerow({field: paper.get(field) for field in fieldnames})


def resolve_seeds(
    seeds: list[SeedPaper],
    *,
    session: requests.Session,
    output_dir: Path,
    sleep_arxiv: float,
    force_refresh: bool,
) -> dict[str, dict[str, Any]]:
    manifest_path = output_dir / "seed_papers.json"
    manifest: dict[str, dict[str, Any]] = load_json(manifest_path, {})
    for seed in seeds:
        cached = manifest.get(seed.local_id)
        if cached and cached.get("arxiv_id") and not force_refresh:
            continue
        print(f"Resolving seed: {seed.local_id} -> {seed.title}")
        match = search_arxiv_by_title(session, seed.title, sleep_seconds=sleep_arxiv)
        if not match:
            manifest[seed.local_id] = {
                "local_id": seed.local_id,
                "path": str(seed.path),
                "title": seed.title,
                "resolution_status": "not_found",
            }
            continue
        manifest[seed.local_id] = {
            "local_id": seed.local_id,
            "path": str(seed.path),
            "title": seed.title,
            "resolution_status": "ok",
            **match,
        }
        write_json(manifest_path, manifest)
        time.sleep(sleep_arxiv)
    write_json(manifest_path, manifest)
    return manifest


def resolve_seeds_with_openalex(
    seeds: list[SeedPaper],
    *,
    session: requests.Session,
    output_dir: Path,
    mailto: str | None,
    sleep_openalex: float,
    force_refresh: bool,
) -> dict[str, dict[str, Any]]:
    manifest_path = output_dir / "seed_papers.json"
    manifest: dict[str, dict[str, Any]] = load_json(manifest_path, {})
    for seed in seeds:
        cached = manifest.get(seed.local_id)
        if cached and cached.get("arxiv_id") and not force_refresh:
            continue
        print(f"Resolving seed with OpenAlex: {seed.local_id} -> {seed.title}")
        try:
            work = search_openalex_work_by_title(session, seed.title, mailto=mailto)
        except ApiError as exc:
            manifest[seed.local_id] = {
                "local_id": seed.local_id,
                "path": str(seed.path),
                "title": seed.title,
                "resolution_status": "error",
                "resolution_error": str(exc),
            }
            write_json(manifest_path, manifest)
            continue
        arxiv_id = extract_arxiv_id_from_openalex_work(work or {})
        if not work or not arxiv_id:
            manifest[seed.local_id] = {
                "local_id": seed.local_id,
                "path": str(seed.path),
                "title": seed.title,
                "resolution_status": "not_found",
                "openalex_id": work.get("id") if work else None,
            }
            write_json(manifest_path, manifest)
            continue
        manifest[seed.local_id] = {
            "local_id": seed.local_id,
            "path": str(seed.path),
            "title": seed.title,
            "resolution_status": "ok",
            "resolution_provider": "openalex",
            "arxiv_id": arxiv_id,
            "arxiv_versioned_id": arxiv_id,
            "openalex_id": work.get("id"),
            "doi": work.get("doi"),
            "openalex_title": work.get("display_name"),
            "authors": openalex_authors(work),
            "year": work.get("publication_year"),
            "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        }
        write_json(manifest_path, manifest)
        time.sleep(sleep_openalex)
    write_json(manifest_path, manifest)
    return manifest


def resolve_seeds_with_semantic_scholar(
    seeds: list[SeedPaper],
    *,
    session: requests.Session,
    output_dir: Path,
    headers: dict[str, str],
    sleep_s2: float,
    force_refresh: bool,
) -> dict[str, dict[str, Any]]:
    manifest_path = output_dir / "seed_papers.json"
    manifest: dict[str, dict[str, Any]] = load_json(manifest_path, {})
    rate_limiter = RateLimiter(sleep_s2)
    for seed in seeds:
        cached = manifest.get(seed.local_id)
        if cached and cached.get("arxiv_id") and not force_refresh:
            continue
        arxiv_override = DEFAULT_ARXIV_ID_OVERRIDES.get(seed.local_id)
        if arxiv_override:
            arxiv_id = strip_arxiv_version(arxiv_override)
            manifest[seed.local_id] = {
                "local_id": seed.local_id,
                "path": str(seed.path),
                "title": seed.title,
                "resolution_status": "ok",
                "resolution_provider": "manual-arxiv-id",
                "arxiv_id": arxiv_id,
                "arxiv_versioned_id": arxiv_id,
                "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
            }
            write_json(manifest_path, manifest)
            continue
        print(f"Resolving seed with Semantic Scholar: {seed.local_id} -> {seed.title}")
        try:
            paper = search_s2_seed_by_title(
                session,
                seed.title,
                headers=headers,
                rate_limiter=rate_limiter,
            )
        except ApiError as exc:
            manifest[seed.local_id] = {
                "local_id": seed.local_id,
                "path": str(seed.path),
                "title": seed.title,
                "resolution_status": "error",
                "resolution_error": str(exc),
            }
            write_json(manifest_path, manifest)
            continue

        external_ids = (paper or {}).get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv")
        if not paper or not arxiv_id:
            manifest[seed.local_id] = {
                "local_id": seed.local_id,
                "path": str(seed.path),
                "title": seed.title,
                "resolution_status": "not_found",
                "semantic_scholar_paper_id": paper.get("paperId") if paper else None,
                "semantic_scholar_title": paper.get("title") if paper else None,
            }
            write_json(manifest_path, manifest)
            continue

        arxiv_id = strip_arxiv_version(str(arxiv_id))
        authors = paper.get("authors") or []
        manifest[seed.local_id] = {
            "local_id": seed.local_id,
            "path": str(seed.path),
            "title": seed.title,
            "resolution_status": "ok",
            "resolution_provider": "semantic-scholar",
            "arxiv_id": arxiv_id,
            "arxiv_versioned_id": arxiv_id,
            "semantic_scholar_paper_id": paper.get("paperId"),
            "semantic_scholar_title": paper.get("title"),
            "authors": [author.get("name") for author in authors if author.get("name")],
            "year": paper.get("year"),
            "venue": paper.get("venue"),
            "url": paper.get("url"),
            "external_ids": external_ids,
            "abstract_url": f"https://arxiv.org/abs/{arxiv_id}",
            "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
        }
        write_json(manifest_path, manifest)
    write_json(manifest_path, manifest)
    return manifest


def collect_citations(
    seeds: list[SeedPaper],
    seed_manifest: dict[str, dict[str, Any]],
    *,
    session: requests.Session,
    output_dir: Path,
    headers: dict[str, str],
    sleep_s2: float,
    force_refresh: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cited_manifest_path = output_dir / "cited_papers.json"
    edges_path = output_dir / "citation_edges.csv"
    cited_papers: dict[str, dict[str, Any]] = load_json(cited_manifest_path, {})
    existing_edges: list[dict[str, Any]] = []
    rate_limiter = RateLimiter(sleep_s2)

    s2_seed_cache_path = output_dir / "seed_semantic_scholar.json"
    s2_seed_cache: dict[str, dict[str, Any]] = load_json(s2_seed_cache_path, {})

    for seed in seeds:
        seed_record = seed_manifest.get(seed.local_id) or {}
        seed_arxiv_id = seed_record.get("arxiv_id")
        if not seed_arxiv_id:
            continue
        s2_paper = s2_seed_cache.get(seed.local_id)
        if not s2_paper or force_refresh:
            print(f"Fetching references: {seed.local_id} ({seed_arxiv_id})")
            try:
                s2_paper = fetch_s2_paper_by_arxiv(
                    session,
                    seed_arxiv_id,
                    headers=headers,
                    rate_limiter=rate_limiter,
                )
                if not s2_paper:
                    s2_paper = search_s2_paper_by_title(
                        session,
                        seed.title,
                        headers=headers,
                        rate_limiter=rate_limiter,
                    )
            except ApiError as exc:
                s2_seed_cache[seed.local_id] = {
                    "status": "error",
                    "error": str(exc),
                    "arxiv_id": seed_arxiv_id,
                    "title": seed.title,
                }
                write_json(s2_seed_cache_path, s2_seed_cache)
                print(f"Reference fetch failed for {seed.local_id}: {exc}")
                continue
            if not s2_paper:
                s2_seed_cache[seed.local_id] = {"status": "not_found"}
                write_json(s2_seed_cache_path, s2_seed_cache)
                continue
            s2_seed_cache[seed.local_id] = s2_paper
            write_json(s2_seed_cache_path, s2_seed_cache)
            time.sleep(sleep_s2)

        references = s2_paper.get("references") or []
        for reference in references:
            entry = reference_to_manifest_entry(reference)
            if not entry:
                continue
            cited_arxiv_id = entry["arxiv_id"]
            if cited_arxiv_id not in cited_papers:
                cited_papers[cited_arxiv_id] = entry
            else:
                cited_papers[cited_arxiv_id].update(
                    {key: value for key, value in entry.items() if value and not cited_papers[cited_arxiv_id].get(key)}
                )
            existing_edges.append(
                {
                    "seed_local_id": seed.local_id,
                    "seed_arxiv_id": seed_arxiv_id,
                    "seed_title": seed_record.get("title") or seed.title,
                    "cited_arxiv_id": cited_arxiv_id,
                    "cited_title": entry.get("title"),
                    "cited_year": entry.get("year"),
                    "semantic_scholar_paper_id": entry.get("semantic_scholar_paper_id"),
                }
            )

    unique_edges = {
        (row["seed_arxiv_id"], row["cited_arxiv_id"]): row
        for row in existing_edges
    }
    edges = sorted(unique_edges.values(), key=lambda row: (row["seed_local_id"], row["cited_arxiv_id"]))
    write_json(cited_manifest_path, cited_papers)
    write_edges_csv(edges_path, edges)
    return cited_papers, edges


def collect_openalex_citations(
    seeds: list[SeedPaper],
    seed_manifest: dict[str, dict[str, Any]],
    *,
    session: requests.Session,
    output_dir: Path,
    mailto: str | None,
    sleep_openalex: float,
    force_refresh: bool,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    cited_manifest_path = output_dir / "cited_papers.json"
    edges_path = output_dir / "citation_edges.csv"
    cited_papers: dict[str, dict[str, Any]] = load_json(cited_manifest_path, {})
    openalex_seed_cache_path = output_dir / "seed_openalex.json"
    openalex_seed_cache: dict[str, dict[str, Any]] = load_json(openalex_seed_cache_path, {})
    edges: list[dict[str, Any]] = []

    for seed in seeds:
        seed_record = seed_manifest.get(seed.local_id) or {}
        seed_arxiv_id = seed_record.get("arxiv_id")
        if not seed_arxiv_id:
            continue

        work = openalex_seed_cache.get(seed.local_id)
        if not work or force_refresh:
            print(f"Fetching OpenAlex references: {seed.local_id} ({seed_arxiv_id})")
            try:
                work = fetch_openalex_work_by_arxiv(session, seed_arxiv_id, mailto=mailto)
                if not work:
                    work = search_openalex_work_by_title(session, seed.title, mailto=mailto)
            except ApiError as exc:
                openalex_seed_cache[seed.local_id] = {
                    "status": "error",
                    "error": str(exc),
                    "arxiv_id": seed_arxiv_id,
                    "title": seed.title,
                }
                write_json(openalex_seed_cache_path, openalex_seed_cache)
                print(f"OpenAlex fetch failed for {seed.local_id}: {exc}")
                continue
            if not work:
                openalex_seed_cache[seed.local_id] = {"status": "not_found"}
                write_json(openalex_seed_cache_path, openalex_seed_cache)
                continue
            openalex_seed_cache[seed.local_id] = work
            write_json(openalex_seed_cache_path, openalex_seed_cache)
            time.sleep(sleep_openalex)

        referenced_work_ids = work.get("referenced_works") or []
        if not referenced_work_ids:
            print(f"OpenAlex has no referenced_works for {seed.local_id} ({seed_arxiv_id})")
            continue

        try:
            referenced_works = fetch_openalex_works_by_ids(
                session,
                referenced_work_ids,
                mailto=mailto,
            )
        except ApiError as exc:
            print(f"OpenAlex reference expansion failed for {seed.local_id}: {exc}")
            continue
        time.sleep(sleep_openalex)

        for reference in referenced_works:
            entry = openalex_work_to_manifest_entry(reference)
            if not entry:
                continue
            cited_arxiv_id = entry["arxiv_id"]
            if cited_arxiv_id not in cited_papers:
                cited_papers[cited_arxiv_id] = entry
            else:
                cited_papers[cited_arxiv_id].update(
                    {key: value for key, value in entry.items() if value and not cited_papers[cited_arxiv_id].get(key)}
                )
            edges.append(
                {
                    "seed_local_id": seed.local_id,
                    "seed_arxiv_id": seed_arxiv_id,
                    "seed_title": seed_record.get("title") or seed.title,
                    "cited_arxiv_id": cited_arxiv_id,
                    "cited_title": entry.get("title"),
                    "cited_year": entry.get("year"),
                    "semantic_scholar_paper_id": None,
                    "openalex_id": entry.get("openalex_id"),
                }
            )

    unique_edges = {
        (row["seed_arxiv_id"], row["cited_arxiv_id"]): row
        for row in edges
    }
    edges = sorted(unique_edges.values(), key=lambda row: (row["seed_local_id"], row["cited_arxiv_id"]))
    write_json(cited_manifest_path, cited_papers)
    write_edges_csv(edges_path, edges)
    return cited_papers, edges


def download_sources(
    cited_papers: dict[str, dict[str, Any]],
    *,
    session: requests.Session,
    output_dir: Path,
    max_downloads: int | None,
    force_download: bool,
    sleep_arxiv: float,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    downloaded = 0
    for arxiv_id in sorted(cited_papers):
        paper = cited_papers[arxiv_id]
        if paper.get("source_status") == "ok" and not force_download:
            continue
        if max_downloads is not None and downloaded >= max_downloads:
            break
        print(f"Downloading TeX source: {arxiv_id} - {paper.get('title')}")
        try:
            result = download_arxiv_source(
                session,
                arxiv_id,
                output_dir=output_dir,
                force=force_download,
                timeout=timeout,
            )
            paper.update(result)
        except Exception as exc:  # noqa: BLE001 - persist per-paper failures and continue.
            paper.update(
                {
                    "source_status": "error",
                    "source_error": str(exc),
                    "tex_files": [],
                    "downloaded": False,
                }
            )
        downloaded += 1
        write_json(output_dir / "cited_papers.json", cited_papers)
        time.sleep(sleep_arxiv)
    write_json(output_dir / "cited_papers.json", cited_papers)
    write_failures_csv(output_dir / "source_failures.csv", cited_papers)
    return cited_papers


def parse_args() -> argparse.Namespace:
    repo_root = find_repo_root(Path(__file__).resolve())
    parser = argparse.ArgumentParser(
        description="Download TeX source packages for arXiv papers cited by local seed arXiv papers.",
    )
    parser.add_argument("--raw-dir", type=Path, default=repo_root / "data" / "arxiv" / "raw")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "arxiv" / "cited")
    parser.add_argument("--only-seed", nargs="*", help="Optional local seed IDs or filenames to process.")
    parser.add_argument("--max-seeds", type=int, help="Limit seed count for testing.")
    parser.add_argument("--max-downloads", type=int, help="Limit cited source downloads for testing.")
    parser.add_argument("--skip-download", action="store_true", help="Resolve citations but do not download sources.")
    parser.add_argument(
        "--citation-provider",
        choices=("openalex", "semantic-scholar"),
        default="openalex",
        help="Citation metadata provider. OpenAlex is the default and does not require an API key.",
    )
    parser.add_argument("--force-refresh", action="store_true", help="Refresh arXiv/Semantic Scholar metadata.")
    parser.add_argument("--force-download", action="store_true", help="Redownload and re-extract source packages.")
    parser.add_argument("--sleep-arxiv", type=float, default=3.1, help="Delay between arXiv requests.")
    parser.add_argument("--sleep-s2", type=float, default=1.0, help="Delay between Semantic Scholar requests.")
    parser.add_argument("--sleep-openalex", type=float, default=1.0, help="Delay between OpenAlex requests.")
    parser.add_argument("--timeout", type=float, default=90.0, help="HTTP timeout in seconds.")
    parser.add_argument(
        "--semantic-scholar-api-key",
        default=os.environ.get("SEMANTIC_SCHOLAR_API_KEY") or os.environ.get("S2_API_KEY"),
        help="Semantic Scholar API key. Defaults to SEMANTIC_SCHOLAR_API_KEY or S2_API_KEY.",
    )
    parser.add_argument(
        "--openalex-email",
        default=os.environ.get("OPENALEX_EMAIL"),
        help="Email to send as OpenAlex mailto parameter for the polite pool. Defaults to OPENALEX_EMAIL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.citation_provider == "semantic-scholar" and not args.semantic_scholar_api_key:
        args.sleep_s2 = max(args.sleep_s2, 3.5)
    selected = set(args.only_seed) if args.only_seed else None
    seeds = discover_seed_papers(args.raw_dir, selected)
    if args.max_seeds is not None:
        seeds = seeds[: args.max_seeds]
    if not seeds:
        raise SystemExit("No seed papers found.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "fine-tuning-or-retrieval cited arxiv tex fetcher"})

    if args.citation_provider == "openalex":
        seed_manifest = resolve_seeds_with_openalex(
            seeds,
            session=session,
            output_dir=args.output_dir,
            mailto=args.openalex_email,
            sleep_openalex=args.sleep_openalex,
            force_refresh=args.force_refresh,
        )
        cited_papers, edges = collect_openalex_citations(
            seeds,
            seed_manifest,
            session=session,
            output_dir=args.output_dir,
            mailto=args.openalex_email,
            sleep_openalex=args.sleep_openalex,
            force_refresh=args.force_refresh,
        )
    else:
        s2_request_headers = s2_headers(args.semantic_scholar_api_key)
        seed_manifest = resolve_seeds_with_semantic_scholar(
            seeds,
            session=session,
            output_dir=args.output_dir,
            headers=s2_request_headers,
            sleep_s2=args.sleep_s2,
            force_refresh=args.force_refresh,
        )
        cited_papers, edges = collect_citations(
            seeds,
            seed_manifest,
            session=session,
            output_dir=args.output_dir,
            headers=s2_request_headers,
            sleep_s2=args.sleep_s2,
            force_refresh=args.force_refresh,
        )
    if not args.skip_download:
        cited_papers = download_sources(
            cited_papers,
            session=session,
            output_dir=args.output_dir,
            max_downloads=args.max_downloads,
            force_download=args.force_download,
            sleep_arxiv=args.sleep_arxiv,
            timeout=args.timeout,
        )

    ok_sources = sum(1 for paper in cited_papers.values() if paper.get("source_status") == "ok")
    failures = sum(
        1 for paper in cited_papers.values() if paper.get("source_status") not in {"ok", "pending"}
    )
    print(
        "Done: "
        f"{len(seeds)} seeds, "
        f"{len(edges)} citation edges, "
        f"{len(cited_papers)} unique cited arXiv papers, "
        f"{ok_sources} TeX source packages available, "
        f"{failures} source failures."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
