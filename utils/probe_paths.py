from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROBES_ROOT = PROJECT_ROOT / "probes"
SUPPORTED_SOURCES = ("arxiv", "legal", "medical")


def _version_number(version: str) -> int:
    digits = "".join(ch for ch in version if ch.isdigit())
    return int(digits) if digits else 0


def _candidate_domain_locations(source: str, domain: str) -> Iterable[Path]:
    data_root = PROJECT_ROOT / "data" / source
    yield data_root / "cleaned" / f"{domain}.txt"
    yield data_root / "cleaned" / f"{domain}.tex"
    yield data_root / "raw" / f"{domain}.txt"
    yield data_root / "raw" / f"{domain}.tex"
    yield data_root / "prior_knowledge" / domain

    for semicleaned_dir in data_root.glob("semicleaned_*"):
        yield semicleaned_dir / f"{domain}.txt"
        yield semicleaned_dir / f"{domain}.tex"


def infer_domain_source(domain: str, default: str = "arxiv") -> str:
    for source in ("legal", "medical", "arxiv"):
        if any(path.exists() for path in _candidate_domain_locations(source, domain)):
            return source
    return default


def canonical_probe_dir(probe_kind: str, domain: str, domain_source: Optional[str] = None) -> Path:
    source = domain_source or infer_domain_source(domain)
    return CANONICAL_PROBES_ROOT / source / domain / probe_kind


def resolve_probe_dir(probe_kind: str, domain: str, domain_source: Optional[str] = None) -> Path:
    return canonical_probe_dir(probe_kind, domain, domain_source)


def resolve_probe_kind_root(probe_kind: str, domain_source: Optional[str] = None) -> Path:
    if domain_source is None:
        return CANONICAL_PROBES_ROOT
    return CANONICAL_PROBES_ROOT / domain_source


def get_all_domains_from_probe_kind(probe_kind: str = "facts") -> List[str]:
    domains = set()

    if CANONICAL_PROBES_ROOT.exists():
        for source in SUPPORTED_SOURCES:
            source_root = CANONICAL_PROBES_ROOT / source
            if not source_root.exists():
                continue
            for domain_dir in source_root.iterdir():
                if (domain_dir / probe_kind).exists():
                    domains.add(domain_dir.name)

    return sorted(domains)


def resolve_knowledge_probe_path(
    domain: str,
    version: str,
    domain_source: Optional[str] = None,
    filename_suffix: str = "",
) -> Path:
    base_dir = resolve_probe_dir("facts", domain, domain_source)
    if _version_number(version) >= 8:
        return base_dir / f"probes_{version}{filename_suffix}.csv"
    return base_dir / f"{domain}_knowledge_probes_{version}{filename_suffix}.csv"


def resolve_inference_probe_candidates(
    domain: str,
    version: str,
    domain_source: Optional[str] = None,
    filename_suffix: str = "",
) -> List[Path]:
    base_dir = resolve_probe_dir("inference", domain, domain_source)
    return [
        base_dir / f"probes_{version}{filename_suffix}.csv",
        base_dir / f"{domain.lower()}_high_level_probes_{version}{filename_suffix}.csv",
    ]


def resolve_generation_prompt_path(
    domain: str,
    filename: str,
    domain_source: Optional[str] = None,
) -> Path:
    return resolve_probe_dir("generation", domain, domain_source) / filename


def resolve_probe_path(
    probe_kind: str,
    domain: str,
    version: str,
    domain_source: Optional[str] = None,
) -> Path:
    if probe_kind == "facts":
        return resolve_knowledge_probe_path(domain, version, domain_source)
    if probe_kind == "inference":
        candidates = resolve_inference_probe_candidates(domain, version, domain_source)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return resolve_probe_dir(probe_kind, domain, domain_source) / f"probes_{version}.csv"


def resolve_mcqa_probe_path(
    probe_kind: str,
    domain: str,
    version: str,
    domain_source: Optional[str] = None,
) -> Path:
    """Return path to the MCQA variant of a probe CSV, e.g. probes_v11_mcqa.csv."""
    base_dir = resolve_probe_dir(probe_kind, domain, domain_source)
    return base_dir / f"probes_{version}_mcqa.csv"


def resolve_filter_path(
    probe_kind: str,
    domain: str,
    exact_match: bool = False,
    domain_source: Optional[str] = None,
) -> Path:
    filename = "filter_em.json" if exact_match else "filter.json"
    return resolve_probe_dir(probe_kind, domain, domain_source) / filename
