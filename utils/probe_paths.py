from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_PROBES_ROOT = PROJECT_ROOT / "probes"
LEGACY_PROBES_ROOT = PROJECT_ROOT / "data" / "probes"
SUPPORTED_SOURCES = ("arxiv", "legal", "medical")


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


def legacy_probe_dir(probe_kind: str, domain: str) -> Path:
    return LEGACY_PROBES_ROOT / probe_kind / domain


def resolve_probe_dir(probe_kind: str, domain: str, domain_source: Optional[str] = None) -> Path:
    canonical = canonical_probe_dir(probe_kind, domain, domain_source)
    if canonical.exists():
        return canonical
    return legacy_probe_dir(probe_kind, domain)


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

    legacy_root = LEGACY_PROBES_ROOT / probe_kind
    if legacy_root.exists():
        for domain_dir in legacy_root.iterdir():
            if domain_dir.is_dir():
                domains.add(domain_dir.name)

    return sorted(domains)


def resolve_knowledge_probe_path(domain: str, version: str, domain_source: Optional[str] = None) -> Path:
    base_dir = resolve_probe_dir("facts", domain, domain_source)
    if version[-1].isdigit() and int(version[-1]) >= 8:
        return base_dir / f"probes_{version}.csv"
    return base_dir / f"{domain}_knowledge_probes_{version}.csv"


def resolve_inference_probe_candidates(
    domain: str,
    version: str,
    domain_source: Optional[str] = None,
) -> List[Path]:
    base_dir = resolve_probe_dir("inference", domain, domain_source)
    return [
        base_dir / f"probes_{version}.csv",
        base_dir / f"{domain.lower()}_high_level_probes_{version}.csv",
    ]


def resolve_generation_prompt_path(
    domain: str,
    filename: str,
    domain_source: Optional[str] = None,
) -> Path:
    return resolve_probe_dir("generation", domain, domain_source) / filename
