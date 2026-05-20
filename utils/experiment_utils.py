import os
import json
import pandas as pd
from typing import Dict, List
from utils import llm_callbacks
from utils import llm_configs
from utils import probe_paths


def get_all_domains(facts_root: str | None = None) -> List[str]:
    if facts_root is not None:
        if not os.path.isdir(facts_root):
            return []
        return [name for name in os.listdir(facts_root) if os.path.isdir(os.path.join(facts_root, name))]
    return probe_paths.get_all_domains_from_probe_kind("facts")


def load_prompts(prompt_files: Dict[str, str], append_eot: bool = False) -> Dict[str, List[Dict[str, str]]]:
    prompts = {}
    for name, path in prompt_files.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            prompt_list = []
            for item in data:
                question = item.get('question', '')
                if append_eot:
                    question += "<|EOT|>"
                prompt_list.append({
                    "prompt_name": item.get('id', 'unknown'),
                    "question": question,
                    "reference_answer": item.get('reference_answer', '')
                })
            prompts[name] = prompt_list
    return prompts


DEFAULT_MCQA_PROMPT_SUFFIX = "\nAnswer: ("
DEFAULT_MCQA_CHOICE_TOKENS = ["A", "B", "C", "D", "E"]
MCQA_LABEL_TO_INDEX = {"(A)": 0, "(B)": 1, "(C)": 2, "(D)": 3, "(E)": 4,
                        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4}


def _coerce_version_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.replace(",", " ").split()
    else:
        raw_values = []
        for item in value:
            raw_values.extend(str(item).replace(",", " ").split())
    return list(dict.fromkeys(v for v in raw_values if v))


def _safe_metric_tag(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in value)


def _create_mcqa_callback(
    tokenizer,
    mcqa_df: pd.DataFrame,
    batch_size: int,
    log,
    output_dir: str,
    log_prefix: str,
    report_to_wandb: bool,
    sparse_eval: bool,
    wandb_metric_allowlist=None,
    eval_every_n_steps: int = 1,
    prompt_suffix: str = DEFAULT_MCQA_PROMPT_SUFFIX,
    prompt_column: str = "formatted_question",
    choice_tokens: list = None,
    panel_domain: str = None,
    panel_metric_name: str = None,
):
    """Create an MCQAProbeCallback from a MCQA probe DataFrame.

    The DataFrame must contain the selected prompt column and ``correct_label``.
    ``correct_label`` may be ``"(D)"`` or ``"D"`` style.
    """
    if choice_tokens is None:
        choice_tokens = list(DEFAULT_MCQA_CHOICE_TOKENS)
    if prompt_column not in mcqa_df.columns:
        raise ValueError(
            f"MCQA prompt column '{prompt_column}' not found. "
            f"Available columns: {list(mcqa_df.columns)}"
        )

    formatted_questions = [
        str(row[prompt_column]).strip() + prompt_suffix
        for _, row in mcqa_df.iterrows()
    ]
    correct_indices = []
    for _, row in mcqa_df.iterrows():
        label = str(row["correct_label"]).strip()
        idx = MCQA_LABEL_TO_INDEX.get(label)
        if idx is None:
            raise ValueError(
                f"Unrecognised correct_label '{label}'. "
                f"Expected one of {list(MCQA_LABEL_TO_INDEX.keys())}"
            )
        correct_indices.append(idx)

    return llm_callbacks.MCQAProbeCallback(
        tokenizer=tokenizer,
        formatted_questions=formatted_questions,
        correct_choice_indices=correct_indices,
        choice_tokens=choice_tokens,
        probes_df=mcqa_df,
        batch_size=batch_size,
        logger=log,
        output_dir=output_dir,
        log_prefix=log_prefix,
        report_to_wandb=report_to_wandb,
        sparse_eval=sparse_eval,
        eval_every_n_steps=eval_every_n_steps,
        wandb_metric_allowlist=wandb_metric_allowlist,
        panel_domain=panel_domain,
        panel_metric_name=panel_metric_name,
    )


def _create_probe_callback(
    tokenizer,
    probe_df,
    batch_size,
    log,
    output_dir,
    log_prefix,
    report_to_wandb,
    sparse_eval,
    wandb_metric_allowlist=None,
    eval_every_n_steps: int = 1,
):
    """Create a BaseKnowledgeProbeCallBack from a probe DataFrame."""
    return llm_callbacks.BaseKnowledgeProbeCallBack(
        tokenizer=tokenizer,
        facts=probe_df['fact'].tolist(),
        probes=probe_df['probe'].tolist(),
        targets=probe_df['target'].tolist(),
        probes_df=probe_df,
        batch_size=batch_size,
        logger=log,
        output_dir=output_dir,
        log_prefix=log_prefix,
        report_to_wandb=report_to_wandb,
        sparse_eval=sparse_eval,
        eval_every_n_steps=eval_every_n_steps,
        wandb_metric_allowlist=wandb_metric_allowlist,
    )


def _resolve_corpus_path(domain: str, args) -> str:
    domain_sources = getattr(args, "domain_data_sources", {}) or {}
    domain_source = domain_sources.get(domain, "arxiv")

    if getattr(args, "raw", False):
        root = f'../../data/{domain_source}/raw'
    elif getattr(args, "semi_cleaned", None):
        semicleaned_root = f'../../data/{domain_source}/semicleaned_{args.semi_cleaned}'
        root = semicleaned_root if os.path.isdir(semicleaned_root) else f'../../data/{domain_source}/cleaned'
    else:
        root = f'../../data/{domain_source}/cleaned'

    txt_path = os.path.join(root, f'{domain}.txt')
    tex_path = os.path.join(root, f'{domain}.tex')
    return txt_path if os.path.exists(txt_path) else tex_path


def _make_mcqa_segment(
    mcqa_df: pd.DataFrame,
    log_prefix: str,
    output_dir: str,
    prompt_suffix: str = DEFAULT_MCQA_PROMPT_SUFFIX,
    prompt_column: str = "formatted_question",
    panel_domain: str = None,
    panel_metric_name: str = None,
):
    """Build a DomainMCQASegment from a DataFrame (no tokenizer needed yet)."""
    if prompt_column not in mcqa_df.columns:
        raise ValueError(
            f"MCQA prompt column '{prompt_column}' not found. "
            f"Available columns: {list(mcqa_df.columns)}"
        )
    formatted_questions = [
        str(row[prompt_column]).strip() + prompt_suffix
        for _, row in mcqa_df.iterrows()
    ]
    correct_indices = []
    for _, row in mcqa_df.iterrows():
        label = str(row["correct_label"]).strip()
        idx = MCQA_LABEL_TO_INDEX.get(label)
        if idx is None:
            raise ValueError(f"Unrecognised correct_label '{label}'.")
        correct_indices.append(idx)
    return llm_callbacks.DomainMCQASegment(
        log_prefix=log_prefix,
        output_dir=output_dir,
        formatted_questions=formatted_questions,
        correct_choice_indices=correct_indices,
        probes_df=mcqa_df,
        panel_domain=panel_domain,
        panel_metric_name=panel_metric_name,
    )


def build_probe_segments(domains, args, log, output_base_dir=None, is_lima=False, tokenizer=None):
    """Collect probe segments without creating callbacks.

    Args:
        domains: list of domain names to set up probes for.
        args: argument namespace with probe configuration flags.
        log: logger instance.
        output_base_dir: root for output dirs. If None, uses
            args.base_results_dir / args.experiment_name.
        is_lima: whether this is a LIMA evaluation pass.
        tokenizer: optional tokenizer, only needed when corpus perplexity is
            requested (to encode corpus text).

    Returns dict with keys:
        knowledge_segments, paraphrased_knowledge_segments,
        inference_segments, mcqa_segments, inference_mcqa_segments,
        corpus_domains_data, all_generation_prompts
    """
    if output_base_dir is None:
        output_base_dir = os.path.join(args.base_results_dir, args.experiment_name)

    domain_sources = getattr(args, "domain_data_sources", {}) or {}
    disable_inference_probes = getattr(args, "disable_inference_probes", False)
    enable_mcqa_probes = getattr(args, "mcqa_probes", False)
    enable_inference_mcqa_probes = getattr(args, "inference_mcqa_probes", False)
    default_knowledge_probes_version = getattr(args, "knowledge_probes_version", "v13")
    knowledge_probe_filename_suffix = getattr(args, "knowledge_probe_filename_suffix", "")
    enable_paraphrased_knowledge_probes = getattr(args, "paraphrased_knowledge_probes", False)
    paraphrased_knowledge_probes_version = getattr(
        args, "paraphrased_knowledge_probes_version", default_knowledge_probes_version
    )
    paraphrased_knowledge_probe_filename_suffix = getattr(
        args, "paraphrased_knowledge_probe_filename_suffix", "_paraphrased"
    )
    mcqa_probes_version = getattr(args, "mcqa_probes_version", default_knowledge_probes_version)
    mcqa_prompt_column = getattr(args, "mcqa_prompt_column", "formatted_question")
    inference_mcqa_probes_versions = _coerce_version_list(
        getattr(args, "inference_mcqa_probes_version", "v12")
    )
    inference_mcqa_prompt_column = getattr(args, "inference_mcqa_prompt_column", "formatted_question")

    knowledge_segments: list[llm_callbacks.DomainProbeSegment] = []
    paraphrased_knowledge_segments: list[llm_callbacks.DomainProbeSegment] = []
    inference_segments: list[llm_callbacks.DomainProbeSegment] = []
    mcqa_segments: list[llm_callbacks.DomainMCQASegment] = []
    inference_mcqa_segments: list[llm_callbacks.DomainMCQASegment] = []
    corpus_domains_data: list[llm_callbacks._CorpusDomainData] = []

    if not domains:
        domains = get_all_domains()
        log.info(f"No domains specified, found and using: {domains}")

    all_generation_prompts = {}

    for domain in domains:
        log.info(f"--- Setting up probes for domain: {domain} ---")
        suffix = "_lima" if is_lima else ""
        output_dir_knowledge_probe = os.path.join(output_base_dir, f"{domain}{suffix}_knowledge_probe")
        os.makedirs(output_dir_knowledge_probe, exist_ok=True)
        domain_source = domain_sources.get(domain)

        # Knowledge probes
        knowledge_probes_version = default_knowledge_probes_version
        knowledge_probe_path = str(
            probe_paths.resolve_knowledge_probe_path(
                domain, knowledge_probes_version,
                domain_source=domain_source,
                filename_suffix=knowledge_probe_filename_suffix,
            )
        )
        if os.path.exists(knowledge_probe_path):
            kp_df = pd.read_csv(knowledge_probe_path)
            knowledge_segments.append(llm_callbacks.DomainProbeSegment(
                log_prefix=f"{domain}_knowledge_probe",
                output_dir=output_dir_knowledge_probe,
                facts=kp_df['fact'].tolist(),
                probes=kp_df['probe'].tolist(),
                targets=kp_df['target'].tolist(),
                probes_df=kp_df,
            ))
            log.info(f"Loaded {len(kp_df)} knowledge probes from {knowledge_probe_path}")
        else:
            log.warning(f"Knowledge probe file not found for domain {domain} at {knowledge_probe_path}")

        # Paraphrased knowledge probes
        if enable_paraphrased_knowledge_probes:
            pkp_path = str(
                probe_paths.resolve_knowledge_probe_path(
                    domain, paraphrased_knowledge_probes_version,
                    domain_source=domain_source,
                    filename_suffix=paraphrased_knowledge_probe_filename_suffix,
                )
            )
            if os.path.exists(pkp_path):
                pkp_df = pd.read_csv(pkp_path)
                paraphrased_knowledge_segments.append(llm_callbacks.DomainProbeSegment(
                    log_prefix=f"{domain}_knowledge_probe_paraphrased",
                    output_dir=output_dir_knowledge_probe,
                    facts=pkp_df['fact'].tolist(),
                    probes=pkp_df['probe'].tolist(),
                    targets=pkp_df['target'].tolist(),
                    probes_df=pkp_df,
                ))
                log.info(f"Loaded {len(pkp_df)} paraphrased knowledge probes from {pkp_path}")
            else:
                log.warning(f"Paraphrased knowledge probe file not found for domain {domain} at {pkp_path}")

        # MCQA probes
        if enable_mcqa_probes:
            mcqa_probe_path = str(
                probe_paths.resolve_mcqa_probe_path("facts", domain, mcqa_probes_version, domain_source=domain_source)
            )
            if os.path.exists(mcqa_probe_path):
                mcqa_df = pd.read_csv(mcqa_probe_path)
                output_dir_mcqa = os.path.join(output_base_dir, f"{domain}{suffix}_mcqa_probe")
                os.makedirs(output_dir_mcqa, exist_ok=True)
                mcqa_segments.append(_make_mcqa_segment(
                    mcqa_df, f"{domain}_mcqa_probe", output_dir_mcqa,
                    prompt_column=mcqa_prompt_column,
                ))
                log.info(f"Loaded {len(mcqa_df)} MCQA probes ({mcqa_probes_version}) from {mcqa_probe_path}")
            else:
                log.warning(f"MCQA probe file not found for domain {domain} at {mcqa_probe_path}")

        # Inference MCQA probes
        if enable_inference_mcqa_probes:
            use_version_tags = len(inference_mcqa_probes_versions) > 1
            for inf_mcqa_ver in inference_mcqa_probes_versions:
                version_tag = _safe_metric_tag(inf_mcqa_ver)
                inf_mcqa_path = str(
                    probe_paths.resolve_mcqa_probe_path("inference", domain, inf_mcqa_ver, domain_source=domain_source)
                )
                if os.path.exists(inf_mcqa_path):
                    inf_mcqa_df = pd.read_csv(inf_mcqa_path)
                    output_dir_name = f"{domain}{suffix}_inference_mcqa_probe"
                    lp = f"{domain}_inference_mcqa_probe"
                    pmn = "inference_mcqa_accuracy"
                    if use_version_tags:
                        output_dir_name += f"_{version_tag}"
                        lp += f"_{version_tag}"
                        pmn += f"_{version_tag}"
                    odir = os.path.join(output_base_dir, output_dir_name)
                    os.makedirs(odir, exist_ok=True)
                    inference_mcqa_segments.append(_make_mcqa_segment(
                        inf_mcqa_df, lp, odir,
                        prompt_column=inference_mcqa_prompt_column,
                        panel_domain=domain, panel_metric_name=pmn,
                    ))
                    log.info(f"Loaded {len(inf_mcqa_df)} inference MCQA probes ({inf_mcqa_ver}) from {inf_mcqa_path}")
                else:
                    log.warning(f"Inference MCQA probe file not found for domain {domain} ({inf_mcqa_ver}) at {inf_mcqa_path}")

        # Inference probes (knowledge-probe style)
        if disable_inference_probes:
            log.info(f"Skipping inference probes for domain {domain} (--disable_inference_probes).")
        else:
            output_dir_inference_probe = os.path.join(output_base_dir, f"{domain}{suffix}_inference_probe")
            os.makedirs(output_dir_inference_probe, exist_ok=True)

            inference_probes_version = args.inference_probes_version
            inference_probe_subset = getattr(args, "inference_probe_subset", "all")
            inference_probe_filename_suffix = getattr(args, "inference_probe_filename_suffix", "")
            log.info(
                f"Using inference_probe_subset='{inference_probe_subset}' "
                f"and probes_{inference_probes_version}{inference_probe_filename_suffix}.csv "
                f"for domain {domain}"
            )

            if inference_probe_subset in {"test", "type_split_test"}:
                base_dir = str(probe_paths.resolve_probe_dir("inference", domain, domain_source))
                candidate_path = []
                if inference_probe_subset == "test":
                    candidate_path.append(os.path.join(base_dir, f'train_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv'))
                    candidate_path.append(os.path.join(base_dir, f'test_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv'))
                else:
                    candidate_path.append(os.path.join(base_dir, f'type_split_train_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv'))
                    candidate_path.append(os.path.join(base_dir, f'type_split_test_probes_{inference_probes_version}{inference_probe_filename_suffix}.csv'))

                if os.path.exists(candidate_path[0]):
                    inference_probe_path = candidate_path[0]
                    log.info(
                        f"Loaded {inference_probe_subset} inference probes for domain {domain} "
                        f"from {inference_probe_path} and {candidate_path[1]}"
                    )
                else:
                    inference_probe_path = None
                    log.warning(
                        f"Requested inference_probe_subset='{inference_probe_subset}' for domain {domain} "
                        f"but file not found at {candidate_path}"
                    )
                for inference_probe_path in candidate_path:
                    if not os.path.exists(inference_probe_path):
                        log.warning(
                            f"Skipping missing {inference_probe_subset} inference probe file for "
                            f"domain {domain}: {inference_probe_path}"
                        )
                        continue
                    inf_df = pd.read_csv(inference_probe_path)
                    prefix = f"train_{domain}_inference_probe" if "train" in inference_probe_path else f"test_{domain}_inference_probe"
                    inference_segments.append(llm_callbacks.DomainProbeSegment(
                        log_prefix=prefix,
                        output_dir=output_dir_inference_probe,
                        facts=inf_df['fact'].tolist(),
                        probes=inf_df['probe'].tolist(),
                        targets=inf_df['target'].tolist(),
                        probes_df=inf_df,
                    ))
                    log.info(f"Loaded {len(inf_df)} inference probes from {inference_probe_path}")
            else:
                path1, path2 = [
                    str(path)
                    for path in probe_paths.resolve_inference_probe_candidates(
                        domain, inference_probes_version,
                        domain_source=domain_source,
                        filename_suffix=inference_probe_filename_suffix,
                    )
                ]
                if os.path.exists(path1):
                    inference_probe_path = path1
                elif os.path.exists(path2):
                    inference_probe_path = path2
                else:
                    inference_probe_path = None
                    log.warning(f"Inference probe file not found for domain {domain} with version {inference_probes_version}")

            if inference_probe_path and inference_probe_subset not in {"test", "type_split_test"}:
                inf_df = pd.read_csv(inference_probe_path)
                inference_segments.append(llm_callbacks.DomainProbeSegment(
                    log_prefix=f"{domain}_inference_probe",
                    output_dir=output_dir_inference_probe,
                    facts=inf_df['fact'].tolist(),
                    probes=inf_df['probe'].tolist(),
                    targets=inf_df['target'].tolist(),
                    probes_df=inf_df,
                ))
                log.info(f"Loaded {len(inf_df)} inference probes from {inference_probe_path}")

        # Corpus perplexity
        if tokenizer is not None:
            corpus_path = _resolve_corpus_path(domain, args)
            if os.path.exists(corpus_path):
                with open(corpus_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                context_length = args.context_length_for_lima if is_lima else args.context_length_for_cpt
                output_dir_corpus_ppl = os.path.join(output_base_dir, f"{domain}{suffix}_corpus_perplexity")
                os.makedirs(output_dir_corpus_ppl, exist_ok=True)
                corpus_domains_data.append(llm_callbacks._CorpusDomainData(
                    log_prefix=f"{domain}_corpus_perplexity",
                    output_dir=output_dir_corpus_ppl,
                    encodings=tokenizer(text_content, return_tensors="pt"),
                    max_length=context_length,
                    stride=512,
                ))
                log.info(f"Added corpus perplexity for domain {domain} from {corpus_path}")
            else:
                log.warning(f"Corpus file not found for domain {domain} at {corpus_path}")

        # Generation prompts (optional)
        if getattr(args, "do_eval", False) is not None:
            if is_lima:
                prompt_files = {
                    f'recall_{domain}_QA': str(
                        probe_paths.resolve_generation_prompt_path(domain, f'recall_{domain}_QA.json', domain_source=domain_source)
                    )
                }
            else:
                prompt_files = {
                    f'recall_{domain}': str(
                        probe_paths.resolve_generation_prompt_path(domain, f'recall_{domain}.json', domain_source=domain_source)
                    )
                }
            domain_prompts = load_prompts(prompt_files, append_eot=is_lima)
            all_generation_prompts.update(domain_prompts)

    return {
        "knowledge_segments": knowledge_segments,
        "paraphrased_knowledge_segments": paraphrased_knowledge_segments,
        "inference_segments": inference_segments,
        "mcqa_segments": mcqa_segments,
        "inference_mcqa_segments": inference_mcqa_segments,
        "corpus_domains_data": corpus_domains_data,
        "all_generation_prompts": all_generation_prompts,
    }


def setup_callbacks(domains, tokenizer, log, args, is_lima: bool = False):
    callbacks = []
    report_to_wandb = not args.test_script
    probe_batch_size = args.device_batch_size * 4
    mcqa_probe_batch_size = max(1, int(getattr(args, "mcqa_probe_batch_size", 32) or 32))
    sparse_eval = getattr(args, "no_callback_every_step", False)
    enable_wandb_source_panels = getattr(args, "enable_wandb_source_panels", False)
    panel_sources = getattr(args, "wandb_panel_sources", ["legal", "arxiv", "medical"])
    probe_report_to_wandb = report_to_wandb and not enable_wandb_source_panels
    wandb_probe_metric_allowlist = getattr(args, "wandb_probe_metric_allowlist", None)
    disable_corpus_perplexity_wandb = getattr(args, "disable_corpus_perplexity_wandb", False)
    disable_training_loss_perplexity_wandb = getattr(args, "disable_training_loss_perplexity_wandb", False)
    probe_every_n_steps = max(1, int(getattr(args, "probe_every_n_steps", 1) or 1))
    mcqa_probe_every_n_steps = max(1, int(getattr(args, "mcqa_probe_every_n_steps", 1) or 1))
    enable_parameter_delta_tracking = getattr(args, "enable_parameter_delta_tracking", False)
    domain_sources = getattr(args, "domain_data_sources", {}) or {}

    # Collect all probe segments via the shared helper
    seg_data = build_probe_segments(
        domains, args, log, output_base_dir=None, is_lima=is_lima, tokenizer=tokenizer,
    )
    knowledge_segments = seg_data["knowledge_segments"]
    paraphrased_knowledge_segments = seg_data["paraphrased_knowledge_segments"]
    inference_segments = seg_data["inference_segments"]
    mcqa_segments = seg_data["mcqa_segments"]
    inference_mcqa_segments = seg_data["inference_mcqa_segments"]
    corpus_domains_data = seg_data["corpus_domains_data"]
    all_generation_prompts = seg_data["all_generation_prompts"]

    # ---- Create unified callbacks from collected segments -----------------

    # Unified knowledge probe callbacks (one per probe type)
    unified_knowledge_cb = None
    if knowledge_segments:
        unified_knowledge_cb = llm_callbacks.UnifiedKnowledgeProbeCallback(
            segments=knowledge_segments,
            tokenizer=tokenizer,
            batch_size=probe_batch_size,
            logger=log,
            report_to_wandb=probe_report_to_wandb,
            sparse_eval=sparse_eval,
            eval_every_n_steps=probe_every_n_steps,
            wandb_metric_allowlist=wandb_probe_metric_allowlist,
        )
        callbacks.append(unified_knowledge_cb)
        log.info(f"Created UnifiedKnowledgeProbeCallback with {len(knowledge_segments)} domain segments")

    unified_paraphrased_cb = None
    if paraphrased_knowledge_segments:
        unified_paraphrased_cb = llm_callbacks.UnifiedKnowledgeProbeCallback(
            segments=paraphrased_knowledge_segments,
            tokenizer=tokenizer,
            batch_size=probe_batch_size,
            logger=log,
            report_to_wandb=probe_report_to_wandb,
            sparse_eval=sparse_eval,
            eval_every_n_steps=probe_every_n_steps,
            wandb_metric_allowlist=wandb_probe_metric_allowlist,
        )
        callbacks.append(unified_paraphrased_cb)
        log.info(f"Created UnifiedKnowledgeProbeCallback (paraphrased) with {len(paraphrased_knowledge_segments)} segments")

    unified_inference_cb = None
    if inference_segments:
        unified_inference_cb = llm_callbacks.UnifiedKnowledgeProbeCallback(
            segments=inference_segments,
            tokenizer=tokenizer,
            batch_size=probe_batch_size,
            logger=log,
            report_to_wandb=probe_report_to_wandb,
            sparse_eval=sparse_eval,
            eval_every_n_steps=probe_every_n_steps,
            wandb_metric_allowlist=wandb_probe_metric_allowlist,
        )
        callbacks.append(unified_inference_cb)
        log.info(f"Created UnifiedKnowledgeProbeCallback (inference) with {len(inference_segments)} segments")

    unified_mcqa_cb = None
    if mcqa_segments:
        unified_mcqa_cb = llm_callbacks.UnifiedMCQAProbeCallback(
            segments=mcqa_segments,
            tokenizer=tokenizer,
            choice_tokens=list(DEFAULT_MCQA_CHOICE_TOKENS),
            batch_size=mcqa_probe_batch_size,
            logger=log,
            report_to_wandb=probe_report_to_wandb,
            sparse_eval=sparse_eval,
            eval_every_n_steps=mcqa_probe_every_n_steps,
            wandb_metric_allowlist=wandb_probe_metric_allowlist,
        )
        callbacks.append(unified_mcqa_cb)
        log.info(f"Created UnifiedMCQAProbeCallback with {len(mcqa_segments)} segments")

    unified_inference_mcqa_cb = None
    if inference_mcqa_segments:
        unified_inference_mcqa_cb = llm_callbacks.UnifiedMCQAProbeCallback(
            segments=inference_mcqa_segments,
            tokenizer=tokenizer,
            choice_tokens=list(DEFAULT_MCQA_CHOICE_TOKENS),
            batch_size=mcqa_probe_batch_size,
            logger=log,
            report_to_wandb=probe_report_to_wandb,
            sparse_eval=sparse_eval,
            eval_every_n_steps=mcqa_probe_every_n_steps,
            wandb_metric_allowlist=wandb_probe_metric_allowlist,
        )
        callbacks.append(unified_inference_mcqa_cb)
        log.info(f"Created UnifiedMCQAProbeCallback (inference) with {len(inference_mcqa_segments)} segments")

    if corpus_domains_data:
        unified_corpus_cb = llm_callbacks.UnifiedCorpusPerplexityCallback(
            domains_data=corpus_domains_data,
            report_to_wandb=(report_to_wandb and not disable_corpus_perplexity_wandb),
            sparse_eval=sparse_eval,
        )
        callbacks.append(unified_corpus_cb)
        log.info(f"Created UnifiedCorpusPerplexityCallback with {len(corpus_domains_data)} domains")

    # Generation probes
    if all_generation_prompts:
        suffix = "_lima" if is_lima else ""
        output_dir_generation = os.path.join(args.base_results_dir, args.experiment_name, f"generation{suffix}")
        os.makedirs(output_dir_generation, exist_ok=True)

        inference_config = llm_configs.InferenceConfig(no_repeat_ngram_size=6)
        generation_probe_callback = llm_callbacks.GenerationProbeCallback(
            prompts=all_generation_prompts,
            tokenizer=tokenizer,
            inference_config=inference_config,
            eval_every_n_steps=50 if is_lima else 50,
            logger=log,
            output_dir=output_dir_generation,
            do_eval=args.do_eval,
            report_to_wandb=report_to_wandb,
        )
        callbacks.append(generation_probe_callback)
        log.info(f"Loaded generation probes for domains: {list(all_generation_prompts.keys())}")

    callbacks.append(
        llm_callbacks.TrainingLossPerplexityCallback(
            report_to_wandb=(report_to_wandb and not disable_training_loss_perplexity_wandb)
        )
    )
    if enable_parameter_delta_tracking:
        suffix = "_lima" if is_lima else ""
        output_dir_parameter_delta = os.path.join(
            args.base_results_dir,
            args.experiment_name,
            f"parameter_delta{suffix}",
        )
        callbacks.append(
            llm_callbacks.ParameterDeltaCallback(
                output_dir=output_dir_parameter_delta,
                storage_path=getattr(args, "parameter_delta_storage_path", None),
                include_embeddings=getattr(args, "parameter_delta_include_embeddings", True),
                compute_final_alignment=getattr(args, "parameter_delta_compute_final_alignment", False),
                sparse_milestones=getattr(args, "parameter_delta_sparse_milestones", True),
                record_every_n_steps=getattr(args, "parameter_delta_every_n_steps", None),
                report_to_wandb=(
                    report_to_wandb
                    and getattr(args, "parameter_delta_report_to_wandb", True)
                ),
                logger=log,
            )
        )
        final_alignment_note = (
            "final-alignment raw-delta capture enabled"
            if getattr(args, "parameter_delta_compute_final_alignment", False)
            else "final-alignment raw-delta capture disabled"
        )
        interval_note = (
            f"record every {args.parameter_delta_every_n_steps} steps"
            if getattr(args, "parameter_delta_every_n_steps", None)
            else "record at sparse milestones"
            if getattr(args, "parameter_delta_sparse_milestones", True)
            else "record every step"
        )
        log.info(
            f"Enabled ParameterDeltaCallback; outputs will save to "
            f"{output_dir_parameter_delta} ({interval_note}; {final_alignment_note})."
        )
    if enable_wandb_source_panels:
        # Build proxy lists from unified callbacks for WandbSourcePanelsCallback
        knowledge_proxies = unified_knowledge_cb.get_domain_proxies() if unified_knowledge_cb else []
        paraphrased_proxies = unified_paraphrased_cb.get_domain_proxies() if unified_paraphrased_cb else []
        inference_proxies = unified_inference_cb.get_domain_proxies() if unified_inference_cb else []
        mcqa_proxies = unified_mcqa_cb.get_domain_proxies() if unified_mcqa_cb else []
        inference_mcqa_proxies = unified_inference_mcqa_cb.get_domain_proxies() if unified_inference_mcqa_cb else []
        callbacks.append(
            llm_callbacks.WandbSourcePanelsCallback(
                knowledge_callbacks=knowledge_proxies,
                paraphrased_knowledge_callbacks=paraphrased_proxies,
                inference_callbacks=inference_proxies,
                mcqa_callbacks=mcqa_proxies,
                inference_mcqa_callbacks=inference_mcqa_proxies,
                domain_sources=domain_sources,
                panel_sources=panel_sources,
                report_to_wandb=report_to_wandb,
            )
        )
        log.info(f"Enabled W&B source panels for sources: {panel_sources}")
    return callbacks


def save_probe_results(callbacks, log, args):
    training_loss_callback = None
    for callback in callbacks:
        if isinstance(callback, llm_callbacks.TrainingLossPerplexityCallback):
            training_loss_callback = callback
            break

    for callback in callbacks:
        # Unified callback types
        if isinstance(callback, llm_callbacks.UnifiedKnowledgeProbeCallback):
            callback.save_results()
            for seg in callback.segments:
                log.info(f"Probe metrics for {seg.log_prefix} saved to {seg.output_dir}")
                if training_loss_callback:
                    training_loss_callback.save_results(output_dir=seg.output_dir)
        elif isinstance(callback, llm_callbacks.UnifiedMCQAProbeCallback):
            callback.save_results()
            for seg in callback.segments:
                log.info(f"MCQA metrics for {seg.log_prefix} saved to {seg.output_dir}")
        elif isinstance(callback, llm_callbacks.UnifiedCorpusPerplexityCallback):
            callback.save_results()
            for dd in callback.domains_data:
                log.info(f"Corpus perplexity metrics for {dd.log_prefix} saved to {dd.output_dir}")
        # Legacy callback types (backward compat)
        elif isinstance(callback, llm_callbacks.BaseKnowledgeProbeCallBack):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Probe metrics for {callback.log_prefix} saved to {callback.output_dir}")
            if training_loss_callback:
                training_loss_callback.save_results(output_dir=callback.output_dir)
        elif isinstance(callback, llm_callbacks.MCQAProbeCallback):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"MCQA metrics for {callback.log_prefix} saved to {callback.output_dir}")
        elif isinstance(callback, llm_callbacks.CorpusPerplexityCallback):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Corpus perplexity metrics for {callback.log_prefix} saved to {callback.output_dir}")
