from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "reports" / "v13_rubric_run"


KEY_COLS = ["section", "subsection", "raw_knowledge_statement"]
CONTENT_COLS = ["target", "probe", "fact"]


def norm(x: object) -> str:
    if pd.isna(x):
        return ""
    return str(x).replace("\r\n", "\n").strip()


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in df.columns:
        df[col] = df[col].map(norm)
    df["_key"] = df[KEY_COLS].agg("\u241f".join, axis=1)
    df["_content"] = df[CONTENT_COLS].agg("\u241f".join, axis=1)
    return df


def docs() -> list[tuple[str, str]]:
    out = []
    for p in sorted((RUN / "outputs").glob("*/*/facts/probes_v13_rubric.csv")):
        out.append((p.parts[-4], p.parts[-3]))
    return out


def main() -> None:
    summary = []
    detail = []
    for domain, doc in docs():
        current_path = ROOT / "probes" / domain / doc / "facts" / "probes_v13.csv"
        rubric_path = RUN / "outputs" / domain / doc / "facts" / "probes_v13_rubric.csv"
        current = load(current_path)
        rubric = load(rubric_path)

        cur_by_key = {k: g for k, g in current.groupby("_key", sort=False)}
        rub_by_key = {k: g for k, g in rubric.groupby("_key", sort=False)}
        keys = set(cur_by_key) | set(rub_by_key)
        identical = repaired = only_current = only_rubric = duplicate_key = 0

        for key in sorted(keys):
            cg = cur_by_key.get(key)
            rg = rub_by_key.get(key)
            if cg is None:
                only_rubric += len(rg)
                for _, row in rg.iterrows():
                    detail.append(row_detail(domain, doc, "only_rubric", "", row))
                continue
            if rg is None:
                only_current += len(cg)
                for _, row in cg.iterrows():
                    detail.append(row_detail(domain, doc, "only_current", row, ""))
                continue
            if len(cg) != 1 or len(rg) != 1:
                duplicate_key += max(len(cg), len(rg))
                continue
            crow = cg.iloc[0]
            rrow = rg.iloc[0]
            if crow["_content"] == rrow["_content"]:
                identical += 1
            else:
                repaired += 1
                d = row_detail(domain, doc, "content_diff", crow, rrow)
                d.update(
                    {
                        "target_same": crow["target"] == rrow["target"],
                        "probe_same": crow["probe"] == rrow["probe"],
                        "fact_same": crow["fact"] == rrow["fact"],
                    }
                )
                detail.append(d)

        summary.append(
            {
                "domain": domain,
                "doc": doc,
                "current_rows": len(current),
                "rubric_rows": len(rubric),
                "identical_retained": identical,
                "content_diff_same_source_key": repaired,
                "only_current": only_current,
                "only_rubric": only_rubric,
                "duplicate_key_unclassified": duplicate_key,
            }
        )

    out_dir = RUN / "summaries"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "current_vs_rubric_summary.csv", summary)
    write_csv(out_dir / "current_vs_rubric_detail.csv", detail)
    totals = {
        "docs": len(summary),
        "current_rows": sum(r["current_rows"] for r in summary),
        "rubric_rows": sum(r["rubric_rows"] for r in summary),
        "identical_retained": sum(r["identical_retained"] for r in summary),
        "content_diff_same_source_key": sum(r["content_diff_same_source_key"] for r in summary),
        "only_current": sum(r["only_current"] for r in summary),
        "only_rubric": sum(r["only_rubric"] for r in summary),
        "duplicate_key_unclassified": sum(r["duplicate_key_unclassified"] for r in summary),
    }
    write_csv(out_dir / "current_vs_rubric_totals.csv", [{"metric": k, "value": v} for k, v in totals.items()])
    print(totals)


def row_detail(domain: str, doc: str, status: str, current: object, rubric: object) -> dict[str, object]:
    def get(row: object, col: str) -> str:
        if isinstance(row, str):
            return ""
        return row.get(col, "")

    return {
        "domain": domain,
        "doc": doc,
        "status": status,
        "current_target": get(current, "target"),
        "rubric_target": get(rubric, "target"),
        "current_probe": get(current, "probe"),
        "rubric_probe": get(rubric, "probe"),
        "current_fact": get(current, "fact"),
        "rubric_fact": get(rubric, "fact"),
        "raw_knowledge_statement": get(current, "raw_knowledge_statement") or get(rubric, "raw_knowledge_statement"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
