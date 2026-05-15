"""Analyze redundancy in granular explanation folders.

Reports both lexical bag-of-words overlap and optional transformer embedding
overlap. Embeddings are cached by content hash to keep repeated runs cheap.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))


STOPWORDS = set(
    """
    a an the and or but if then than that this these those to of in on for from
    with without into onto by as at is are was were be been being it its they
    them their there here we you your our he she his her not no yes do does did
    can could should would may might must will shall also such through across
    about above below between among within each other more most less least very
    just only same both because while where when how what which who whom whose
    why into out up down over under again further once case chapter section
    question answer title clinical legal medical court opinion paper article
    study report patient patients model data results approach method analysis
    using use used based important key main first second
    """.split()
)


def words(text: str, content_only: bool = False) -> list[str]:
    tokens = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-']+", text)]
    if content_only:
        tokens = [t for t in tokens if len(t) > 3 and t not in STOPWORDS]
    return tokens


def cosine_from_counters(c1: collections.Counter, c2: collections.Counter) -> float:
    dot = sum(v * c2.get(k, 0) for k, v in c1.items())
    n1 = math.sqrt(sum(v * v for v in c1.values()))
    n2 = math.sqrt(sum(v * v for v in c2.values()))
    return dot / (n1 * n2) if n1 and n2 else 0.0


def pairwise_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "p90": 0.0, "max": 0.0}
    values = sorted(values)
    return {
        "mean": float(sum(values) / len(values)),
        "p90": float(values[int(0.9 * (len(values) - 1))]),
        "max": float(values[-1]),
    }


def repeated_ngram_rate(texts: list[str], n: int = 7) -> tuple[float, int]:
    all_ngrams = []
    total = 0
    for text in texts:
        cleaned = "\n".join(line for line in text.splitlines() if not line.startswith("Title:"))
        tokens = words(cleaned)
        ngrams = [tuple(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))]
        total += len(ngrams)
        all_ngrams.extend(ngrams)
    if total == 0:
        return 0.0, 0
    counts = collections.Counter(all_ngrams)
    repeated = sum(v for v in counts.values() if v > 1)
    return repeated / total, sum(1 for v in counts.values() if v > 1)


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def save_embedding_cache(path: Path, cache: dict[str, list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(cache, f)
    tmp_path.replace(path)


def get_openai_client():
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        try:
            from utils.keys import OPENAI_API_KEY  # type: ignore

            api_key = OPENAI_API_KEY
        except (ImportError, ModuleNotFoundError):
            api_key = None
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or utils.keys")
    return OpenAI(api_key=api_key)


def chunk_for_embedding(text: str, max_chars: int) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start, end)
            if boundary > start + max_chars // 2:
                end = boundary
        chunks.append(text[start:end].strip())
        start = end
    return [chunk for chunk in chunks if chunk]


def embedding_for_text(
    text: str,
    client,
    model: str,
    cache: dict[str, list[float]],
    max_chars: int,
) -> np.ndarray:
    chunks = chunk_for_embedding(text, max_chars)
    vectors = []
    for chunk in chunks:
        key = f"{model}:{hashlib.sha256(chunk.encode('utf-8')).hexdigest()}"
        if key not in cache:
            response = client.embeddings.create(model=model, input=chunk)
            cache[key] = response.data[0].embedding
        vectors.append(np.array(cache[key], dtype=np.float32))
    vector = np.mean(vectors, axis=0)
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def discover_view_dirs(base_dir: Path, domains: list[str] | None, views: list[str]) -> list[tuple[str, str, str, list[Path]]]:
    view_dirs = []
    domain_dirs = [base_dir / d for d in domains] if domains else [p for p in base_dir.iterdir() if p.is_dir()]
    for domain_dir in domain_dirs:
        exp_dir = domain_dir / "explanations"
        if not exp_dir.is_dir():
            continue
        for doc_dir in exp_dir.iterdir():
            if not doc_dir.is_dir() or ".bak_" in doc_dir.name:
                continue
            for view in views:
                granular_dir = doc_dir / view
                if granular_dir.is_dir() and ".bak_" not in granular_dir.name:
                    files = sorted(granular_dir.glob("*.txt"))
                    if len(files) >= 2:
                        view_dirs.append((domain_dir.name, doc_dir.name, view, files))
    return view_dirs


def analyze_view(files: list[Path], args, client=None, cache=None) -> dict[str, float | int]:
    texts = [p.read_text(errors="ignore") for p in files]
    counters = [collections.Counter(words(text, content_only=True)) for text in texts]
    pairs = list(itertools.combinations(range(len(files)), 2))
    bow_scores = [cosine_from_counters(counters[i], counters[j]) for i, j in pairs]
    bow = pairwise_stats(bow_scores)
    repeated_rate, repeated_unique = repeated_ngram_rate(texts)

    result: dict[str, float | int] = {
        "files": len(files),
        "bow_mean": bow["mean"],
        "bow_p90": bow["p90"],
        "bow_max": bow["max"],
        "rep7_rate": repeated_rate,
        "rep7_unique": repeated_unique,
    }

    if args.embeddings:
        vectors = [
            embedding_for_text(text, client, args.embedding_model, cache, args.embedding_max_chars)
            for text in texts
        ]
        embedding_scores = [float(np.dot(vectors[i], vectors[j])) for i, j in pairs]
        emb = pairwise_stats(embedding_scores)
        result.update(
            {
                "emb_mean": emb["mean"],
                "emb_p90": emb["p90"],
                "emb_max": emb["max"],
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default="data", help="Repository data directory.")
    parser.add_argument("--domains", nargs="+", help="Domains to scan, e.g. legal medical arxiv.")
    parser.add_argument("--views", nargs="+", default=["textbooks", "blogs", "stackexchange"])
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--embeddings", action="store_true", help="Also compute transformer embedding cosine overlap.")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--embedding-max-chars", type=int, default=24000)
    parser.add_argument("--cache-path", default="reports/redundancy_embedding_cache.json")
    parser.add_argument("--output-json", help="Optional path to write full results JSON.")
    args = parser.parse_args()

    view_dirs = discover_view_dirs(Path(args.base_dir), args.domains, args.views)
    client = get_openai_client() if args.embeddings else None
    cache = load_embedding_cache(Path(args.cache_path)) if args.embeddings else {}

    rows = []
    for domain, doc, view, files in view_dirs:
        metrics = analyze_view(files, args, client=client, cache=cache)
        rows.append({"domain": domain, "doc": doc, "view": view, **metrics})
        if args.embeddings and len(rows) % 10 == 0:
            save_embedding_cache(Path(args.cache_path), cache)

    if args.embeddings:
        save_embedding_cache(Path(args.cache_path), cache)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(rows, f, indent=2)

    print(f"scanned_view_dirs={len(rows)}")
    print("\nTOP embedding topical overlap" if args.embeddings else "\nTOP lexical topical overlap")
    sort_key = "emb_mean" if args.embeddings else "bow_mean"
    for row in sorted(rows, key=lambda r: r[sort_key], reverse=True)[: args.top_k]:
        emb = (
            f" emb={row['emb_mean']:.3f} p90={row['emb_p90']:.3f} max={row['emb_max']:.3f}"
            if args.embeddings
            else ""
        )
        print(
            f"{emb} bow={row['bow_mean']:.3f} rep7={row['rep7_rate']:.2%} "
            f"files={row['files']:2d} {row['domain']}/{row['doc']}/{row['view']}"
        )

    print("\nTOP exact-ish repeated phrase rate")
    for row in sorted(rows, key=lambda r: r["rep7_rate"], reverse=True)[: args.top_k]:
        emb = f" emb={row['emb_mean']:.3f}" if args.embeddings else ""
        print(
            f"rep7={row['rep7_rate']:.2%}{emb} bow={row['bow_mean']:.3f} "
            f"files={row['files']:2d} {row['domain']}/{row['doc']}/{row['view']}"
        )


if __name__ == "__main__":
    main()
