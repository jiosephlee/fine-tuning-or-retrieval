from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports" / "v13_rubric_run"


def expected_docs() -> list[tuple[str, str, int]]:
    docs: list[tuple[str, str, int]] = []
    for path in sorted((ROOT / "probes").glob("*/*/facts/probes_v12.csv")):
        domain, doc = path.parts[-4], path.parts[-3]
        docs.append((domain, doc, len(pd.read_csv(path))))
    return docs


def count_csv(path: Path) -> int | None:
    if not path.exists():
        return None
    return len(pd.read_csv(path))


def main() -> None:
    summary_rows: list[dict[str, object]] = []
    for domain, doc, v12_rows in expected_docs():
        first = RUN / "outputs_first_pass" / domain / doc / "facts" / "probes_v13_rubric_first_pass.csv"
        final = RUN / "outputs" / domain / doc / "facts" / "probes_v13_rubric.csv"
        d1 = RUN / "reviews" / domain / doc / "decisions_first_pass.csv"
        d2 = RUN / "reviews" / domain / doc / "decisions_target_source.csv"
        first_rows = count_csv(first)
        final_rows = count_csv(final)
        summary_rows.append(
            {
                "domain": domain,
                "doc": doc,
                "v12_rows": v12_rows,
                "first_pass_rows": first_rows if first_rows is not None else "",
                "final_rows": final_rows if final_rows is not None else "",
                "first_pass_drop_count": "" if first_rows is None else v12_rows - first_rows,
                "target_source_drop_count": ""
                if first_rows is None or final_rows is None
                else first_rows - final_rows,
                "has_decisions_first_pass": d1.exists(),
                "has_decisions_target_source": d2.exists(),
                "has_first_pass_output": first.exists(),
                "has_final_output": final.exists(),
            }
        )

    out = RUN / "summaries" / "summary_by_doc.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    done = [r for r in summary_rows if r["first_pass_rows"] != ""]
    final_done = [r for r in summary_rows if r["final_rows"] != ""]
    totals = {
        "expected_docs": len(summary_rows),
        "completed_first_pass_docs": len(done),
        "completed_final_docs": len(final_done),
        "v12_rows_total": sum(int(r["v12_rows"]) for r in summary_rows),
        "first_pass_rows_total": sum(int(r["first_pass_rows"]) for r in done),
        "final_rows_total": sum(int(r["final_rows"]) for r in final_done),
        "historical_initial_v13_rows": 6483,
        "historical_final_v13_rows": 6419,
    }
    totals_path = RUN / "summaries" / "totals.csv"
    with totals_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in totals.items():
            writer.writerow({"metric": key, "value": value})

    print(totals)
    print(out)
    print(totals_path)


if __name__ == "__main__":
    main()
