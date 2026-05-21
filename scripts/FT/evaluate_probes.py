#!/usr/bin/env python3
"""Standalone probe evaluation script.

Loads a saved/hub model and runs probe evaluations once, writing results
into a ``reeval/`` subdirectory of an existing experiment directory.

Example usage::

    cd scripts/FT
    python evaluate_probes.py \
        --model_path jiosephlee/e8-olmo2-13b-para9-20260516 \
        --experiment_dir ../../results/FT/full/allenai_OLMo-2-1124-13B/.../E8_paraphrase_13b_all_domains \
        --inference_mcqa_probes \
        --inference_mcqa_probes_version v12_reviewed v12 \
        --inference_mcqa_prompt_column formatted_question_5shot \
        --mcqa_probe_batch_size 32 \
        --attn_implementation flash_attention_2
"""

import argparse
import logging
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import llm_callbacks
from utils.experiment_utils import (
    build_probe_segments,
    DEFAULT_MCQA_CHOICE_TOKENS,
    get_all_domains,
)


def parse_args():
    p = argparse.ArgumentParser(description="Standalone probe evaluation")

    # Model / output
    p.add_argument("--model_path", required=True,
                   help="HF hub model ID or local path")
    p.add_argument("--experiment_dir", required=True,
                   help="Existing experiment directory; results go into {experiment_dir}/reeval/")
    p.add_argument("--attn_implementation", default=None,
                   help="Attention implementation (e.g. flash_attention_2)")
    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])

    # Domain selection
    p.add_argument("--override_domains", nargs="+", default=None,
                   help="Domains to evaluate (default: auto-discover from probes)")

    # Probe flags (mirror training script)
    p.add_argument("--knowledge_probes", action="store_true")
    p.add_argument("--knowledge_probes_version", default="v13")
    p.add_argument("--knowledge_probe_filename_suffix", default="")

    p.add_argument("--paraphrased_knowledge_probes", action="store_true")
    p.add_argument("--paraphrased_knowledge_probes_version", default=None)
    p.add_argument("--paraphrased_knowledge_probe_filename_suffix", default="_paraphrased")

    p.add_argument("--inference_probes", action="store_true")
    p.add_argument("--inference_probes_version", default="v11_reviewed")
    p.add_argument("--inference_probe_subset", default="all")
    p.add_argument("--inference_probe_filename_suffix", default="")
    p.add_argument("--disable_inference_probes", action="store_true")

    p.add_argument("--mcqa_probes", action="store_true")
    p.add_argument("--mcqa_probes_version", default=None)
    p.add_argument("--mcqa_prompt_column", default="formatted_question")

    p.add_argument("--inference_mcqa_probes", action="store_true")
    p.add_argument("--inference_mcqa_probes_version", nargs="+", default=["v12"])
    p.add_argument("--inference_mcqa_prompt_column", default="formatted_question")

    p.add_argument("--mcqa_probe_batch_size", type=int, default=32)
    p.add_argument("--probe_batch_size", type=int, default=32)

    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    log = logging.getLogger("evaluate_probes")

    # ---- Resolve output dir -----------------------------------------------
    reeval_dir = os.path.join(args.experiment_dir, "reeval")
    os.makedirs(reeval_dir, exist_ok=True)
    log.info(f"Results will be written to: {reeval_dir}")

    # ---- Load model & tokenizer -------------------------------------------
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.torch_dtype]

    log.info(f"Loading model from {args.model_path} (dtype={args.torch_dtype})")
    model_kwargs = dict(dtype=torch_dtype, device_map="auto")
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    log.info("Model loaded and set to eval mode.")

    # ---- Determine which probe types to run -------------------------------
    # If no probe type flags are set, warn and exit
    any_enabled = (
        args.knowledge_probes
        or args.paraphrased_knowledge_probes
        or args.inference_probes
        or args.mcqa_probes
        or args.inference_mcqa_probes
    )
    if not any_enabled:
        log.error(
            "No probe types enabled. Pass at least one of: "
            "--knowledge_probes, --paraphrased_knowledge_probes, "
            "--inference_probes, --mcqa_probes, --inference_mcqa_probes"
        )
        sys.exit(1)

    # Build a minimal args namespace that build_probe_segments expects.
    # Only the flags for enabled probe types matter; disabled ones are
    # naturally skipped because their enable flags are False.
    if args.paraphrased_knowledge_probes_version is None:
        args.paraphrased_knowledge_probes_version = args.knowledge_probes_version
    if args.mcqa_probes_version is None:
        args.mcqa_probes_version = args.knowledge_probes_version

    # If knowledge/inference/paraphrased probes are not requested, suppress
    # them by disabling their flags in the namespace.
    if not args.inference_probes:
        args.disable_inference_probes = True

    domains = args.override_domains or get_all_domains()
    log.info(f"Domains: {domains}")

    # ---- Build probe segments ---------------------------------------------
    seg_data = build_probe_segments(
        domains=domains,
        args=args,
        log=log,
        output_base_dir=reeval_dir,
        is_lima=False,
        tokenizer=None,  # skip corpus perplexity
    )

    # ---- Evaluate knowledge probes ----------------------------------------
    if args.knowledge_probes and seg_data["knowledge_segments"]:
        segments = seg_data["knowledge_segments"]
        log.info(f"Evaluating {len(segments)} knowledge probe segments ...")
        cb = llm_callbacks.UnifiedKnowledgeProbeCallback(
            segments=segments,
            tokenizer=tokenizer,
            batch_size=args.probe_batch_size,
            logger=log,
            report_to_wandb=False,
            sparse_eval=False,
        )
        _run_knowledge_eval(cb, model, log)

    if args.paraphrased_knowledge_probes and seg_data["paraphrased_knowledge_segments"]:
        segments = seg_data["paraphrased_knowledge_segments"]
        log.info(f"Evaluating {len(segments)} paraphrased knowledge probe segments ...")
        cb = llm_callbacks.UnifiedKnowledgeProbeCallback(
            segments=segments,
            tokenizer=tokenizer,
            batch_size=args.probe_batch_size,
            logger=log,
            report_to_wandb=False,
            sparse_eval=False,
        )
        _run_knowledge_eval(cb, model, log)

    if args.inference_probes and seg_data["inference_segments"]:
        segments = seg_data["inference_segments"]
        log.info(f"Evaluating {len(segments)} inference probe segments ...")
        cb = llm_callbacks.UnifiedKnowledgeProbeCallback(
            segments=segments,
            tokenizer=tokenizer,
            batch_size=args.probe_batch_size,
            logger=log,
            report_to_wandb=False,
            sparse_eval=False,
        )
        _run_knowledge_eval(cb, model, log)

    # ---- Evaluate MCQA probes ---------------------------------------------
    if args.mcqa_probes and seg_data["mcqa_segments"]:
        segments = seg_data["mcqa_segments"]
        log.info(f"Evaluating {len(segments)} MCQA probe segments ...")
        cb = llm_callbacks.UnifiedMCQAProbeCallback(
            segments=segments,
            tokenizer=tokenizer,
            choice_tokens=list(DEFAULT_MCQA_CHOICE_TOKENS),
            batch_size=args.mcqa_probe_batch_size,
            logger=log,
            report_to_wandb=False,
            sparse_eval=False,
        )
        _run_mcqa_eval(cb, model, log)

    if args.inference_mcqa_probes and seg_data["inference_mcqa_segments"]:
        segments = seg_data["inference_mcqa_segments"]
        log.info(f"Evaluating {len(segments)} inference MCQA probe segments ...")
        cb = llm_callbacks.UnifiedMCQAProbeCallback(
            segments=segments,
            tokenizer=tokenizer,
            choice_tokens=list(DEFAULT_MCQA_CHOICE_TOKENS),
            batch_size=args.mcqa_probe_batch_size,
            logger=log,
            report_to_wandb=False,
            sparse_eval=False,
        )
        _run_mcqa_eval(cb, model, log)

    log.info("Done.")


def _run_knowledge_eval(cb, model, log):
    """Evaluate all segments in a UnifiedKnowledgeProbeCallback and save."""
    for idx, seg in enumerate(cb.segments):
        log.info(f"  [{idx+1}/{len(cb.segments)}] {seg.log_prefix} ({len(seg.facts)} probes)")
        metrics = cb._evaluate_segment(seg, model, step=0)
        # Record into segment history (mimics _merge_and_record on rank 0)
        for metric_name, values in metrics.items():
            if values is not None:
                seg.history[metric_name].append({"step": 0, "values": values.cpu().tolist()})
                valid = values[~torch.isinf(values) & ~torch.isnan(values)]
                if valid.numel():
                    log.info(f"    {metric_name}_avg = {valid.mean().item():.4f}")
    cb.save_results()


def _run_mcqa_eval(cb, model, log):
    """Evaluate all segments in a UnifiedMCQAProbeCallback and save."""
    for idx, seg in enumerate(cb.segments):
        log.info(f"  [{idx+1}/{len(cb.segments)}] {seg.log_prefix} ({len(seg.formatted_questions)} probes)")
        metrics = cb._evaluate_segment(seg, model)
        predicted = metrics.pop("predicted_idx", None)
        if predicted is not None:
            seg.predictions_history.append({"step": 0, "values": predicted.cpu().tolist()})
        for metric_name, values in metrics.items():
            seg.history[metric_name].append({"step": 0, "values": values.cpu().tolist()})
            valid = values[~torch.isinf(values) & ~torch.isnan(values)]
            if valid.numel():
                log.info(f"    {metric_name}_avg = {valid.mean().item():.4f}")
    cb.save_results()
    # Generate per-segment reports
    for seg in cb.segments:
        cb._generate_mcqa_report(seg)


if __name__ == "__main__":
    main()
