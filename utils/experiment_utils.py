import os
import json
import pandas as pd
from typing import Dict, List
from utils import llm_callbacks
from utils import llm_configs


def get_all_domains(facts_root: str = '../../data/probes/facts') -> List[str]:
    if not os.path.isdir(facts_root):
        return []
    return [name for name in os.listdir(facts_root) if os.path.isdir(os.path.join(facts_root, name))]


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


def setup_callbacks(domains, tokenizer, log, args, is_lima: bool = False):
    callbacks = []
    report_to_wandb = not args.test_script
    probe_batch_size = args.device_batch_size * 4
    sparse_eval = getattr(args, "no_callback_every_step", False)

    if not domains:
        domains = get_all_domains()
        log.info(f"No domains specified, found and using: {domains}")

    all_generation_prompts = {}

    for domain in domains:
        log.info(f"--- Setting up probes for domain: {domain} ---")
        suffix = "_lima" if is_lima else ""
        output_dir_knowledge_probe = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_knowledge_probe")
        output_dir_inference_probe = os.path.join(args.base_results_dir, args.experiment_name, f"{domain}{suffix}_inference_probe")
        os.makedirs(output_dir_knowledge_probe, exist_ok=True)
        os.makedirs(output_dir_inference_probe, exist_ok=True)

        # Knowledge probes path
        knowledge_probes_version = args.knowledge_probes_version
        if int(knowledge_probes_version[-1]) >= 8:
            knowledge_probe_path = f'../../data/probes/facts/{domain}/probes_{knowledge_probes_version}.csv'
        else:
            knowledge_probe_path = f'../../data/probes/facts/{domain}/{domain}_knowledge_probes_{knowledge_probes_version}.csv'

        if os.path.exists(knowledge_probe_path):
            knowledge_probe_df = pd.read_csv(knowledge_probe_path)
            facts = knowledge_probe_df['fact'].tolist()
            probes = knowledge_probe_df['probe'].tolist()
            targets = knowledge_probe_df['target'].tolist()

            knowledge_probe_callback = llm_callbacks.BaseKnowledgeProbeCallBack(
                tokenizer=tokenizer,
                facts=facts,
                probes=probes,
                targets=targets,
                probes_df=knowledge_probe_df,
                batch_size=probe_batch_size,
                logger=log,
                output_dir=output_dir_knowledge_probe,
                log_prefix=f"{domain}_knowledge_probe",
                report_to_wandb=report_to_wandb,
                sparse_eval=sparse_eval,
            )
            callbacks.append(knowledge_probe_callback)
            log.info(f"Loaded {len(knowledge_probe_df)} knowledge probes from {knowledge_probe_path}")
        else:
            log.warning(f"Knowledge probe file not found for domain {domain} at {knowledge_probe_path}")

        # Inference probes path
        inference_probes_version = args.inference_probes_version
        inference_probe_subset = getattr(args, "inference_probe_subset", "all")

        # Optional subset-specific test files: test_probes_vX.csv or type_split_test_probes_vX.csv
        if inference_probe_subset in {"test", "type_split_test"}:
            base_dir = f'../../data/probes/inference/{domain}'
            if inference_probe_subset == "test":
                candidate_path = os.path.join(base_dir, f'test_probes_{inference_probes_version}.csv')
            else:  # type_split_test
                candidate_path = os.path.join(base_dir, f'type_split_test_probes_{inference_probes_version}.csv')

            if os.path.exists(candidate_path):
                inference_probe_path = candidate_path
                log.info(
                    f"Loaded {inference_probe_subset} inference probes for domain {domain} "
                    f"from {inference_probe_path}"
                )
            else:
                inference_probe_path = None
                log.warning(
                    f"Requested inference_probe_subset='{inference_probe_subset}' for domain {domain} "
                    f"but file not found at {candidate_path}"
                )
        else:
            path1 = f'../../data/probes/inference/{domain}/probes_{inference_probes_version}.csv'
            path2 = f'../../data/probes/inference/{domain}/{domain.lower()}_high_level_probes_{inference_probes_version}.csv'

            if os.path.exists(path1):
                inference_probe_path = path1
            elif os.path.exists(path2):
                inference_probe_path = path2
            else:
                inference_probe_path = None
                log.warning(f"Inference probe file not found for domain {domain} with version {inference_probes_version}")

        if inference_probe_path:
            inference_probe_df = pd.read_csv(inference_probe_path)
            facts = inference_probe_df['fact'].tolist()
            probes = inference_probe_df['probe'].tolist()
            targets = inference_probe_df['target'].tolist()

            inference_probe_callback = llm_callbacks.BaseKnowledgeProbeCallBack(
                tokenizer=tokenizer,
                facts=facts,
                probes=probes,
                targets=targets,
                probes_df=inference_probe_df,
                batch_size=probe_batch_size,
                logger=log,
                output_dir=output_dir_inference_probe,
                log_prefix=f"{domain}_inference_probe",
                report_to_wandb=report_to_wandb,
                sparse_eval=sparse_eval,
            )
            callbacks.append(inference_probe_callback)
            log.info(f"Loaded {len(inference_probe_df)} inference probes from {inference_probe_path}")

        # Corpus perplexity callback
        if getattr(args, "semi_cleaned", None):
            corpus_path = f'../../data/arxiv/semicleaned_{args.semi_cleaned}/{domain}.tex'
        elif getattr(args, "raw", False):
            corpus_path = f'../../data/arxiv/raw/{domain}.tex'
        else:
            corpus_path = f'../../data/arxiv/cleaned/{domain}.tex'

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
                report_to_wandb=report_to_wandb,
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
                    f'recall_{domain}_QA': f'../../data/probes/generation/{domain}/recall_{domain}_QA.json'
                }
            else:
                prompt_files = {
                    f'recall_{domain}': f'../../data/probes/generation/{domain}/recall_{domain}.json'
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

    callbacks.append(llm_callbacks.TrainingLossPerplexityCallback(report_to_wandb=report_to_wandb))
    return callbacks


def save_probe_results(callbacks, log, args):
    training_loss_callback = None
    for callback in callbacks:
        if isinstance(callback, llm_callbacks.TrainingLossPerplexityCallback):
            training_loss_callback = callback
            break

    for callback in callbacks:
        if isinstance(callback, llm_callbacks.BaseKnowledgeProbeCallBack):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Probe metrics for {callback.log_prefix} saved to {callback.output_dir}")
            if training_loss_callback:
                training_loss_callback.save_results(output_dir=callback.output_dir)
        elif isinstance(callback, llm_callbacks.CorpusPerplexityCallback):
            callback.save_results(output_dir=callback.output_dir)
            log.info(f"Corpus perplexity metrics for {callback.log_prefix} saved to {callback.output_dir}")
