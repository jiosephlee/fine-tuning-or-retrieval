import os

from scripts.plotting.plot_utils import find_latest_run, load_metrics


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows)


def _classic_probe(root, domain="DPO", probe_type="knowledge", value=0.25):
    probe_dir = root / f"{domain}_{probe_type}_probe"
    _write_csv(
        probe_dir / f"{domain}_{probe_type}_probe_metrics.csv",
        "step,probe_index,log_prob,target_rank\n"
        f"0,0,{value},5\n"
        f"1,0,{value + 1},3\n",
    )


def _mcqa_probe(root, domain="DPO", probe_type="knowledge", suffix="", value=0.5):
    if probe_type == "knowledge":
        probe_dir = root / f"{domain}_mcqa_probe"
    else:
        probe_dir = root / f"{domain}_inference_mcqa_probe{suffix}"
    _write_csv(
        probe_dir / f"{probe_dir.name}_metrics.csv",
        "step,probe_index,mcqa_accuracy\n"
        f"0,0,{value}\n"
        f"1,0,{value + 0.25}\n",
    )


def test_find_latest_run_accepts_exact_experiment_leaf(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _classic_probe(leaf)

    assert find_latest_run(str(leaf)) == str(leaf)


def test_find_latest_run_descends_from_overlap_parent(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _classic_probe(leaf)

    assert find_latest_run(str(tmp_path / "overlap_1_16")) == str(leaf)


def test_find_latest_run_descends_from_parent_above_overlap(tmp_path):
    leaf = tmp_path / "bs256_lr4e-05" / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _classic_probe(leaf)

    assert find_latest_run(str(tmp_path / "bs256_lr4e-05")) == str(leaf)


def test_load_metrics_classic_default_is_unchanged(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _classic_probe(leaf, value=-2.0)

    df = load_metrics(str(tmp_path / "overlap_1_16"), "knowledge", ["DPO"], ".")

    assert list(df.columns) == ["step", "log_prob"]
    assert df["log_prob"].tolist() == [-2.0, -1.0]


def test_load_metrics_mcqa_uses_mcqa_accuracy_default(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _mcqa_probe(leaf, value=0.25)

    df = load_metrics(
        str(tmp_path / "overlap_1_16"),
        "knowledge",
        ["DPO"],
        ".",
        probe_family="mcqa",
    )

    assert list(df.columns) == ["step", "mcqa_accuracy"]
    assert df["mcqa_accuracy"].tolist() == [0.25, 0.5]


def test_load_metrics_auto_falls_back_to_mcqa_when_classic_missing(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _mcqa_probe(leaf, probe_type="inference", value=0.0)

    df = load_metrics(
        str(tmp_path / "overlap_1_16"),
        "inference",
        ["DPO"],
        ".",
        probe_family="auto",
    )

    assert list(df.columns) == ["step", "mcqa_accuracy"]
    assert df["mcqa_accuracy"].tolist() == [0.0, 0.25]


def test_load_metrics_inference_mcqa_prefers_reviewed_folder(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12", value=0.0)
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12_reviewed", value=0.75)

    df = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
    )

    assert df["mcqa_accuracy"].tolist() == [0.75, 1.0]


def test_load_metrics_inference_mcqa_regular_ignores_reviewed_folder(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12", value=0.0)
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12_reviewed", value=0.75)

    df = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
        mcqa_variant="regular",
    )

    assert df["mcqa_accuracy"].tolist() == [0.0, 0.25]


def test_load_metrics_inference_mcqa_reviewed_ignores_regular_folder(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12", value=0.0)
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12_reviewed", value=0.75)

    df = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
        mcqa_variant="reviewed",
    )

    assert df["mcqa_accuracy"].tolist() == [0.75, 1.0]


def test_load_metrics_inference_mcqa_reviewed_missing_returns_none(tmp_path):
    leaf = tmp_path / "overlap_1_16" / "E5_paraphrase_1b_all_domains"
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12", value=0.0)

    df = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
        mcqa_variant="reviewed",
    )

    assert df is None


def test_load_metrics_inference_mcqa_reviewed_root_accepts_unversioned_folder(tmp_path):
    leaf = (
        tmp_path
        / "probes_v13_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed"
        / "overlap_1_16"
        / "E5_paraphrase_13b_all_domains"
    )
    _mcqa_probe(leaf, probe_type="inference", value=0.75)

    df = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
        mcqa_variant="reviewed",
    )

    assert df["mcqa_accuracy"].tolist() == [0.75, 1.0]


def test_load_metrics_inference_mcqa_regular_ignores_unversioned_reviewed_root(tmp_path):
    leaf = (
        tmp_path
        / "probes_v13_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed"
        / "overlap_1_16"
        / "E5_paraphrase_13b_all_domains"
    )
    _mcqa_probe(leaf, probe_type="inference", value=0.75)

    df = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
        mcqa_variant="regular",
    )

    assert df is None


def test_load_metrics_inference_mcqa_mixed_reviewed_root_uses_explicit_variants(tmp_path):
    leaf = (
        tmp_path
        / "probes_v13_prompt_formatted_question_5shot_inf_mcqa_v12_reviewed+v12"
        / "overlap_1_16"
        / "E10_source_32b_all_domains"
    )
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12", value=0.0)
    _mcqa_probe(leaf, probe_type="inference", suffix="_v12_reviewed", value=0.75)

    regular = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
        mcqa_variant="regular",
    )
    reviewed = load_metrics(
        str(leaf),
        "inference",
        ["DPO"],
        ".",
        probe_family="mcqa",
        mcqa_variant="reviewed",
    )

    assert regular["mcqa_accuracy"].tolist() == [0.0, 0.25]
    assert reviewed["mcqa_accuracy"].tolist() == [0.75, 1.0]
