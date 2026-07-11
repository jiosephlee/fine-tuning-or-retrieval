"""Consolidate sibling explanation-generation runs into one canonical slug per corpus.

Multiple partial generation runs exist per gpt_oss corpus (e.g. gpt_oss_120b_low,
_low_max, _low_max_v2, _low_64k). For each (domain, explanation type) this picks the
best flat file across the variant runs and copies it — together with its paired
outline JSON, which split_explanations_to_subfolders.py needs — into the canonical
target slug. Source variant directories are left untouched; a merge_manifest.json
records provenance.

Best-file rule per (domain, type):
  1. eligible = size >= --min_bytes AND no leaked harmony channel markers
  2. winner   = largest eligible file (largest non-empty with a warning if none eligible)
"""
import argparse
import json
import os
import shutil

DOMAINS = ["1_58", "BOFT", "ByteLatent", "DPO", "FeatLLM", "GRPO",
           "GSPO", "LongRoPE", "OFT", "QLoRA", "fa3", "xLSTM"]

TYPES = {
    "textbook.txt": "textbook_outline.json",
    "blogs.txt": "blog_outline.json",
    "stackexchange.txt": "stack_exchange_outline.json",
}

CHANNEL_MARKERS = ("<|channel|>", "<|message|>", "assistantfinal")

FAMILIES = {
    "gpt_oss_20b_low": ["gpt_oss_20b_low", "gpt_oss_20b_low_max", "gpt_oss_20b_low_max_v2", "gpt_oss_20b_low_64k"],
    "gpt_oss_120b_low": ["gpt_oss_120b_low", "gpt_oss_120b_low_max", "gpt_oss_120b_low_max_v2", "gpt_oss_120b_low_64k"],
    "gpt_oss_20b_high": ["gpt_oss_20b_high", "gpt_oss_20b_high_v2", "gpt_oss_20b_high_64k"],
    "gpt_oss_120b_high": ["gpt_oss_120b_high", "gpt_oss_120b_high_v2", "gpt_oss_120b_high_64k"],
}


def has_channel_markers(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        return True
    return any(m in text for m in CHANNEL_MARKERS)


def pick_winner(base_dir, variants, domain, flat_name, min_bytes):
    candidates = []
    for variant in variants:
        path = os.path.join(base_dir, variant, domain, flat_name)
        if os.path.isfile(path):
            size = os.path.getsize(path)
            if size > 0:
                candidates.append((variant, path, size))
    if not candidates:
        return None, "absent"
    eligible = [c for c in candidates
                if c[2] >= min_bytes and not has_channel_markers(c[1])]
    if eligible:
        return max(eligible, key=lambda c: c[2]), "ok"
    return max(candidates, key=lambda c: c[2]), "fallback"


def merge_family(base_dir, target, variants, min_bytes, dry_run):
    manifest = {}
    print(f"\n=== merging into {target} (candidates: {', '.join(variants)})")
    for domain in DOMAINS:
        manifest[domain] = {}
        target_domain_dir = os.path.join(base_dir, target, domain)
        for flat_name, outline_name in TYPES.items():
            winner, status = pick_winner(base_dir, variants, domain, flat_name, min_bytes)
            if winner is None:
                print(f"  {domain}/{flat_name}: GAP — no candidate in any variant")
                manifest[domain][flat_name] = {"source": None, "bytes": 0, "status": "absent"}
                continue
            variant, src_path, size = winner
            if status == "fallback":
                print(f"  {domain}/{flat_name}: WARNING — no clean candidate >= {min_bytes}B; "
                      f"using {variant} ({size}B) anyway")
            manifest[domain][flat_name] = {"source": variant, "bytes": size, "status": status}
            if variant == target:
                continue  # already in place
            if not dry_run:
                os.makedirs(target_domain_dir, exist_ok=True)
                shutil.copy2(src_path, os.path.join(target_domain_dir, flat_name))
                src_outline = os.path.join(base_dir, variant, domain, outline_name)
                if os.path.isfile(src_outline):
                    shutil.copy2(src_outline, os.path.join(target_domain_dir, outline_name))
            print(f"  {domain}/{flat_name}: {variant} ({size // 1000}k)")
    if not dry_run:
        manifest_path = os.path.join(base_dir, target, "merge_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  manifest -> {manifest_path}")
    complete = sum(
        1 for domain in DOMAINS
        if all(manifest[domain][t]["status"] == "ok" for t in TYPES)
    )
    print(f"  --> {target}: {complete}/12 domains complete (clean files >= {min_bytes}B)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="../../data/arxiv/explanations/")
    parser.add_argument("--targets", nargs="+", choices=list(FAMILIES), default=list(FAMILIES))
    parser.add_argument("--min_bytes", type=int, default=10_000)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    for target in args.targets:
        merge_family(args.base_dir, target, FAMILIES[target], args.min_bytes, args.dry_run)
