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
    prompt_suffix: str = DEFAULT_MCQA_PROMPT_SUFFIX,
    choice_tokens: list = None,
):
    """Create an MCQAProbeCallback from a MCQA probe DataFrame.

    The DataFrame must contain the standard cloze columns (``fact``, ``probe``,
    ``target``) **plus** MCQA-specific columns (``formatted_question``,
    ``correct_label``).  ``correct_label`` may be ``"(D)"`` or ``"D"`` style.
    """
    if choice_tokens is None:
        choice_tokens = list(DEFAULT_MCQA_CHOICE_TOKENS)

    formatted_questions = [
        str(row["formatted_question"]).strip() + prompt_suffix
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
        facts=mcqa_df["fact"].tolist(),
        probes=mcqa_df["probe"].tolist(),
        targets=mcqa_df["target"].tolist(),
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
        wandb_metric_allowlist=wandb_metric_allowlist,
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


def setup_callbacks(domains, tokenizer, log, args, is_lima: bool = False):
    callbacks = []
    report_to_wandb = not args.test_script
    probe_batch_size = args.device_batch_size * 4
    sparse_eval = getattr(args, "no_callback_every_step", False)
    disable_inference_probes = getattr(args, "disable_inference_probes", False)
    enable_wandb_source_panels = getattr(args, "enable_wandb_source_panels", False)
    panel_sources = getattr(args, "wandb_panel_sources", ["legal", "arxiv", "medical"])
    wandb_probe_metric_allowlist = getattr(args, "wandb_probe_metric_allowlist", None)
    disable_corpus_perplexity_wandb = getattr(args, "disable_corpus_perplexity_wandb", False)
    disable_training_loss_perplexity_wandb = getattr(args, "disable_training_loss_perplexity_wandb", False)
    domain_sources = getattr(args, "domain_data_sources", {}) or {}
    enable_mcqa_probes = getattr(args, "mcqa_probes", False)
    enable_parameter_delta_tracking = getattr(args, "enable_parameter_delta_tracking", False)
    knowledge_probe_callbacks = []
    mcqa_probe_callbacks = []

    if not domains:
        domains = get_all_domains()
        log.info(f"No domains specified, found and using: {domains}")

    all_generation_prompts = {}

    for domain in domains:
        log.info(f"--- Setting up probes for domain: {domain} ---")
        suffix = "_lima" if is_lima else ""
        output_dir_knowledge_probe = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_knowledge_probe")
        os.makedirs(output_dir_knowledge_probe, exist_ok=True)
        domain_source = domain_sources.get(domain)

        # Knowledge probes path
        knowledge_probes_version = args.knowledge_probes_version
        knowledge_probe_path = str(
            probe_paths.resolve_knowledge_probe_path(
                domain,
                knowledge_probes_version,
                domain_source=domain_source,
            )
        )

        if os.path.exists(knowledge_probe_path):
            knowledge_probe_df = pd.read_csv(knowledge_probe_path)
            knowledge_probe_callback = _create_probe_callback(
                tokenizer, knowledge_probe_df, probe_batch_size, log,
                output_dir_knowledge_probe, f"{domain}_knowledge_probe",
                report_to_wandb, sparse_eval,
                wandb_probe_metric_allowlist,
            )
            callbacks.append(knowledge_probe_callback)
            knowledge_probe_callbacks.append(knowledge_probe_callback)
            log.info(f"Loaded {len(knowledge_probe_df)} knowledge probes from {knowledge_probe_path}")
        else:
            log.warning(f"Knowledge probe file not found for domain {domain} at {knowledge_probe_path}")

        # MCQA probes (constrained-decoding evaluation) — version follows knowledge probes
        if enable_mcqa_probes:
            mcqa_probe_path = str(
                probe_paths.resolve_mcqa_probe_path(
                    "facts", domain, knowledge_probes_version,
                    domain_source=domain_source,
                )
            )
            if os.path.exists(mcqa_probe_path):
                mcqa_df = pd.read_csv(mcqa_probe_path)
                output_dir_mcqa = os.path.join(
                    args.base_results_dir, args.experiment_name,
                    f"{domain}{suffix}_mcqa_probe",
                )
                os.makedirs(output_dir_mcqa, exist_ok=True)
                mcqa_callback = _create_mcqa_callback(
                    tokenizer, mcqa_df, probe_batch_size, log,
                    output_dir_mcqa, f"{domain}_mcqa_probe",
                    report_to_wandb, sparse_eval,
                    wandb_probe_metric_allowlist,
                )
                callbacks.append(mcqa_callback)
                mcqa_probe_callbacks.append(mcqa_callback)
                log.info(f"Loaded {len(mcqa_df)} MCQA probes from {mcqa_probe_path}")
            else:
                log.warning(f"MCQA probe file not found for domain {domain} at {mcqa_probe_path}")

        if disable_inference_probes:
            log.info(f"Skipping inference probes for domain {domain} (--disable_inference_probes).")
        else:
            output_dir_inference_probe = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_inference_probe")
            os.makedirs(output_dir_inference_probe, exist_ok=True)

            # Inference probes path
            inference_probes_version = args.inference_probes_version
            inference_probe_subset = getattr(args, "inference_probe_subset", "all")
            log.info(f"Using inference_probe_subset='{inference_probe_subset}' for domain {domain}")

            # Optional subset-specific test files: test_probes_vX.csv or type_split_test_probes_vX.csv
            if inference_probe_subset in {"test", "type_split_test"}:
                base_dir = str(probe_paths.resolve_probe_dir("inference", domain, domain_source))
                candidate_path = []
                if inference_probe_subset == "test":
                    candidate_path.append(os.path.join(base_dir, f'train_probes_{inference_probes_version}.csv'))
                    candidate_path.append(os.path.join(base_dir, f'test_probes_{inference_probes_version}.csv'))
                else:  # type_split_test
                    candidate_path.append(os.path.join(base_dir, f'type_split_train_probes_{inference_probes_version}.csv'))
                    candidate_path.append(os.path.join(base_dir, f'type_split_test_probes_{inference_probes_version}.csv'))

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
                    inference_probe_df = pd.read_csv(inference_probe_path)
                    prefix = f"train_{domain}_inference_probe" if "train" in inference_probe_path else f"test_{domain}_inference_probe"
                    inference_probe_callback = _create_probe_callback(
                        tokenizer, inference_probe_df, probe_batch_size, log,
                        output_dir_inference_probe, prefix,
                        report_to_wandb, sparse_eval,
                        wandb_probe_metric_allowlist,
                    )
                    callbacks.append(inference_probe_callback)
                    log.info(f"Loaded {len(inference_probe_df)} inference probes from {inference_probe_path}")
            else:
                path1, path2 = [
                    str(path)
                    for path in probe_paths.resolve_inference_probe_candidates(
                        domain,
                        inference_probes_version,
                        domain_source=domain_source,
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
                inference_probe_df = pd.read_csv(inference_probe_path)
                inference_probe_callback = _create_probe_callback(
                    tokenizer, inference_probe_df, probe_batch_size, log,
                    output_dir_inference_probe, f"{domain}_inference_probe",
                    report_to_wandb, sparse_eval,
                    wandb_probe_metric_allowlist,
                )
                callbacks.append(inference_probe_callback)
                log.info(f"Loaded {len(inference_probe_df)} inference probes from {inference_probe_path}")

        # Corpus perplexity callback
        corpus_path = _resolve_corpus_path(domain, args)

        if os.path.exists(corpus_path):
            with open(corpus_path, 'r', encoding='utf-8') as f:
                text_content = f.read()

            context_length = args.context_length_for_lima if is_lima else args.context_length_for_cpt
            output_dir_corpus_ppl = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_corpus_perplexity")
            os.makedirs(output_dir_corpus_ppl, exist_ok=True)

            corpus_perplexity_callback = llm_callbacks.CorpusPerplexityCallback(
                text_content=text_content,
                tokenizer=tokenizer,
                max_length=context_length,
                stride=512,
                output_dir=output_dir_corpus_ppl,
                log_prefix=f"{domain}_corpus_perplexity",
                report_to_wandb=(report_to_wandb and not disable_corpus_perplexity_wandb),
                sparse_eval=sparse_eval,
            )
            callbacks.append(corpus_perplexity_callback)
            log.info(f"Added CorpusPerplexityCallback for domain {domain} from {corpus_path}")
        else:
            log.warning(f"Corpus file not found for domain {domain} at {corpus_path}")

        # Generation prompts (optional)
        if getattr(args, "do_eval", False) is not None:
            if is_lima:
                prompt_files = {
                    f'recall_{domain}_QA': str(
                        probe_paths.resolve_generation_prompt_path(
                            domain,
                            f'recall_{domain}_QA.json',
                            domain_source=domain_source,
                        )
                    )
                }
            else:
                prompt_files = {
                    f'recall_{domain}': str(
                        probe_paths.resolve_generation_prompt_path(
                            domain,
                            f'recall_{domain}.json',
                            domain_source=domain_source,
                        )
                    )
                }
            domain_prompts = load_prompts(prompt_files, append_eot=is_lima)
            all_generation_prompts.update(domain_prompts)

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
                compute_final_alignment=getattr(args, "parameter_delta_compute_final_alignment", True),
                sparse_milestones=getattr(args, "parameter_delta_sparse_milestones", True),
                report_to_wandb=(
                    report_to_wandb
                    and getattr(args, "parameter_delta_report_to_wandb", True)
                ),
                logger=log,
            )
        )
        log.info(f"Enabled ParameterDeltaCallback; outputs will save to {output_dir_parameter_delta}")
    if enable_wandb_source_panels:
        callbacks.append(
            llm_callbacks.WandbSourcePanelsCallback(
                knowledge_callbacks=knowledge_probe_callbacks,
                mcqa_callbacks=mcqa_probe_callbacks,
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
        if isinstance(callback, llm_callbacks.BaseKnowledgeProbeCallBack):
            # MCQAProbeCallback extends BaseKnowledgeProbeCallBack, so this
            # branch covers both cloze and MCQA metrics via the shared history.
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Probe metrics for {callback.log_prefix} saved to {callback.output_dir}")
            if training_loss_callback:
                training_loss_callback.save_results(output_dir=callback.output_dir)
        elif isinstance(callback, llm_callbacks.CorpusPerplexityCallback):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Corpus perplexity metrics for {callback.log_prefix} saved to {callback.output_dir}")
