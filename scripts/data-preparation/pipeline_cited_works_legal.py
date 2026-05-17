import os
import json
import sys
import glob

sys.path.append('../../')


def load_cited_opinions(case_dir):
    """Load all cited opinion texts and metadata from a case's cited/ directory.

    Returns a list of (meta, text) tuples sorted by opinion_id for deterministic ordering.
    """
    cited_dir = os.path.join(case_dir, "cited")
    if not os.path.isdir(cited_dir):
        return []

    cited_metas = sorted(glob.glob(os.path.join(cited_dir, "cited_*.meta.json")))
    results = []
    for meta_path in cited_metas:
        opinion_id = os.path.basename(meta_path).replace("cited_", "").replace(".meta.json", "")
        text_path = os.path.join(cited_dir, f"cited_{opinion_id}.txt")
        if not os.path.exists(text_path):
            print(f"  Warning: missing text for cited opinion {opinion_id}, skipping")
            continue

        with open(meta_path, 'r') as f:
            meta = json.load(f)
        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        if not text:
            print(f"  Warning: empty text for cited opinion {opinion_id}, skipping")
            continue

        results.append((meta, text))

    results.sort(key=lambda x: x[0].get("opinion_id", 0))
    return results


def build_cited_works_text(case_name, case_meta, cited_opinions):
    """Concatenate cited opinions into a single document with clear boundary markers."""
    parts = []

    case_display = case_meta.get('case_name', case_name)
    parts.append(f"### CITED OPINIONS FOR {case_display}\n\n")

    for i, (meta, text) in enumerate(cited_opinions, 1):
        parts.append(f"##### Cited Opinion {i}\n\n")
        parts.append(text)
        parts.append("\n\n")

    return "".join(parts)


def generate_cited_works_legal(case_name, manifest_entry):
    """Process a single legal case to compile its cited opinions."""
    print(f"Processing {case_name} for cited works compilation...")

    RAW_DIR = f"../../data/legal/raw/"
    OUTPUT_DIR = f"../../data/legal/cited_works/{case_name}/"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    case_meta_path = os.path.join(RAW_DIR, f"{case_name}.meta.json")
    with open(case_meta_path, 'r') as f:
        case_meta = json.load(f)

    case_dir = os.path.join(RAW_DIR, case_name)
    cited_opinions = load_cited_opinions(case_dir)

    if not cited_opinions:
        print(f"  No cited opinions found for {case_name}. Skipping.")
        return

    print(f"  Found {len(cited_opinions)} cited opinions.")

    # Save each cited opinion individually
    for meta, text in cited_opinions:
        opinion_id = meta.get("opinion_id", "unknown")
        individual_path = os.path.join(OUTPUT_DIR, f"cited_{opinion_id}.txt")
        with open(individual_path, 'w', encoding='utf-8') as f:
            f.write(text)

    # Save metadata index
    cited_index = []
    for meta, text in cited_opinions:
        entry = {
            "opinion_id": meta.get("opinion_id"),
            "depth": meta.get("depth"),
            "word_count": meta.get("word_count"),
            "cited_by": meta.get("cited_by"),
        }
        cited_index.append(entry)

    index_path = os.path.join(OUTPUT_DIR, "cited_index.json")
    with open(index_path, 'w') as f:
        json.dump(cited_index, f, indent=2)
    print(f"  Saved cited index to {index_path}")

    # Build and save combined cited works text
    combined_text = build_cited_works_text(case_name, case_meta, cited_opinions)
    output_path = os.path.join(OUTPUT_DIR, "cited_opinions.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined_text)

    total_words = sum(m.get("word_count", 0) for m, _ in cited_opinions)
    print(f"  Saved combined cited works ({len(cited_opinions)} opinions, "
          f"~{total_words} words) to {output_path}")


def process_cases():
    manifest_path = "../../data/legal/raw/manifest.json"
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    saved_cases = [entry for entry in manifest if entry.get("status") == "saved"]
    print(f"Found {len(saved_cases)} saved cases in manifest.\n")

    for entry in saved_cases:
        case_name = entry["filename"]
        generate_cited_works_legal(case_name, entry)
        print()


if __name__ == "__main__":
    process_cases()
