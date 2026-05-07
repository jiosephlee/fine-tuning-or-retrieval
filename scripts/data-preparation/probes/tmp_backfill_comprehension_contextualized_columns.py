import argparse
from pathlib import Path

import pandas as pd

from pipeline_generate_comprehension_mcqa import (
    build_contextualized_fewshot_question,
    build_contextualized_question,
    get_paper_title,
    read_paper_text,
)


def backfill_file(csv_path: Path) -> bool:
    domain = csv_path.parent.parent.name
    paper_text = read_paper_text(domain)
    if not paper_text:
        print(f"Skipping {csv_path}: could not load paper text for domain '{domain}'")
        return False

    paper_title = get_paper_title(paper_text, domain)
    df = pd.read_csv(csv_path)

    if 'question' not in df.columns or 'excerpt' not in df.columns:
        print(f"Skipping {csv_path}: missing required columns")
        return False

    df['contextualized_question'] = df.apply(
        lambda row: build_contextualized_question(row['excerpt'], row['question'], paper_title),
        axis=1,
    )
    df['contextualized_fewshot_question'] = df.apply(
        lambda row: build_contextualized_fewshot_question(row['excerpt'], row['question'], paper_title),
        axis=1,
    )
    df.to_csv(csv_path, index=False)
    print(f"Updated {csv_path}")
    return True


def iter_target_files(root: Path):
    yield from root.glob('probes/*/*/inference/comprehension_mcqa.csv')


def main():
    parser = argparse.ArgumentParser(description='Backfill contextualized comprehension MCQA prompt columns.')
    parser.add_argument(
        '--root',
        type=Path,
        default=Path(__file__).resolve().parents[3],
        help='Project root',
    )
    args = parser.parse_args()

    updated = 0
    for csv_path in sorted(iter_target_files(args.root)):
        updated += int(backfill_file(csv_path))

    print(f"Backfilled {updated} comprehension MCQA files.")


if __name__ == '__main__':
    main()
