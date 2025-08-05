import ast
from transformers import (
    AutoTokenizer,
    TrainerCallback,
)
import wandb
import math
import torch
import pandas as pd
import os
from typing import List

class BaseKnowledgeProbeCallback(TrainerCallback):
    """
    A base callback for evaluating model performance on knowledge probes.
    This class provides shared utilities for loading data, calculating metrics,
    and saving results. Subclasses must implement the specific logic for
    the types of probes they handle.
    """
    def __init__(self, tokenizer: AutoTokenizer, probe_dataset_path: str, max_length: int, batch_size: int = 8, log_prefix="probe_eval", logger=None):
        self.tokenizer = tokenizer 
        self.log_prefix = log_prefix # Name of the report on wandb
        self.max_length = max_length 
        self.batch_size = batch_size # Speeds up probing by batching
        self.initial_metrics = {}
        self.PROBE_CONFIG = {}  # To be defined in subclasses
        self.logger = logger

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not probe_dataset_path or not os.path.exists(probe_dataset_path):
            raise ValueError("Probe dataset path is not provided or does not exist.")

        df = pd.read_csv(probe_dataset_path)
        self.probe_indices = df.index.tolist() # We go through our probes in order
        self.sections = df["section"].tolist() # We group probes by section
        self._df = df  # Store df for subclasses to access

    def _initialize_metrics_config(self):
        """Generates internal metric configs from the user-facing PROBE_CONFIG."""
        self.METRICS_CONFIG = {}
        for group, group_config in self.PROBE_CONFIG.items():
            track_paraphrased = group_config.get('track_paraphrased')
            metrics = group_config.get('metrics')
            for metric_type, metric_config in metrics.items():
                full_name = f"{group}_{metric_type}"
                self.METRICS_CONFIG[full_name] = {
                    'value_col_name': metric_type,
                    'track_paraphrased': track_paraphrased,
                    **metric_config # track_delta, etc.
                }

        self.history = {name: [] for name in self.METRICS_CONFIG} # Each metric has a list of values for each step
        self.delta_history = {name: [] for name, cfg in self.METRICS_CONFIG.items() if cfg.get('track_delta')}

        paraphrased_metrics = {name for name, cfg in self.METRICS_CONFIG.items() if cfg.get('track_paraphrased')}
        self.num_paraphrase_variants = getattr(self, 'num_paraphrase_variants', 0)
        self.paraphrased_history = {name: [[] for _ in range(self.num_paraphrase_variants)] for name in paraphrased_metrics}
        self.paraphrased_delta_history = {name: [[] for _ in range(self.num_paraphrase_variants)] for name in paraphrased_metrics if self.METRICS_CONFIG.get(name, {}).get('track_delta')}

    def on_train_begin(self, args, state, control, model, **kwargs):
        raise NotImplementedError("Subclasses must implement on_train_begin")

    def on_step_end(self, args, state, control, model, **kwargs):
        raise NotImplementedError("Subclasses must implement on_step_end")

    def _evaluate_whole_sentences(self, model, statements: List[str], device):
        """
        Calculates perplexity and log probability for a list of statements.
        This is a clean, general-purpose function for any whole statement.
        """
        all_perplexities = []
        all_log_probs = []
        for i in range(0, len(statements), self.batch_size):
            batch_statements = statements[i:i + self.batch_size]
            if not batch_statements:
                continue
            
            inputs = self.tokenizer(batch_statements, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length).to(device)
            input_ids = inputs["input_ids"]

            with torch.no_grad():
                logits = model(input_ids).logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_labels[shift_labels == self.tokenizer.pad_token_id] = -100 # Mask padding tokens for loss calculation

            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels) # loss is (batch_size, seq_len)
            # Assert that padding tokens have 0 loss
            assert (loss[shift_labels == -100] == 0).all(), "Expected loss to be 0 for padding tokens (labels=-100)"
 
            sum_loss = loss.sum(dim=1) # (batch_size,)
            num_tokens = (shift_labels != -100).sum(dim=1).float()
            
            mean_nll = sum_loss / num_tokens
            
            all_perplexities.append(torch.exp(mean_nll))
            all_log_probs.append(-sum_loss)

        return {
            "perplexity": torch.cat(all_perplexities) if all_perplexities else torch.tensor([]),
            "log_prob": torch.cat(all_log_probs) if all_log_probs else torch.tensor([])
        }

    def _evaluate_target_probes(self, model, contexts: List[str], targets: List[str], device):
        """Calculates perplexity, log probability, and hit rate for target spans."""
        all_metrics = {'perplexity': [], 'log_prob': [], 'hit_at_5': [], 'hit_at_50': [], 'hit_at_100': []}
        
        metrics_to_calc = set()
        if 'atomic_target' in self.PROBE_CONFIG:
            metrics_to_calc.update(self.PROBE_CONFIG['atomic_target'].get('metrics', {}).keys())

        for i in range(0, len(contexts), self.batch_size):
            batch_contexts, batch_targets = contexts[i:i+self.batch_size], targets[i:i+self.batch_size]
            if not batch_contexts:
                continue

            full_text = [c + t for c, t in zip(batch_contexts, batch_targets)]
            inputs = self.tokenizer(full_text, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length, add_special_tokens=False).to(device)
            input_ids = inputs.input_ids
            
            context_lengths = self.tokenizer(batch_contexts, add_special_tokens=False, padding="longest", return_tensors="pt").attention_mask.sum(dim=1).to(device)

            with torch.no_grad():
                logits = model(input_ids).logits

            # --- Loss Metrics ---
            shift_logits = logits[..., :-1, :].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            shift_labels_target = input_ids[..., 1:].contiguous().clone()
            for j, length in enumerate(context_lengths):
                if length > 0:
                    shift_labels_target[j, :length - 1] = -100
            shift_labels_target[shift_labels_target == self.tokenizer.pad_token_id] = -100
            loss_target = loss_fct(shift_logits.permute(0, 2, 1), shift_labels_target)
            sum_loss_target = loss_target.sum(dim=1)
            
            all_metrics['log_prob'].append(-sum_loss_target)
            
            target_lengths = self.tokenizer(batch_targets, add_special_tokens=False, padding="longest", return_tensors="pt").attention_mask.sum(dim=1).to(logits.device)
            num_tokens_target = (shift_labels_target != -100).sum(dim=1).float()
            assert torch.equal(num_tokens_target, target_lengths), "Masked target token count mismatch."
            mean_nll_target = sum_loss_target / torch.max(num_tokens_target, torch.ones_like(num_tokens_target))
            all_metrics['perplexity'].append(torch.exp(mean_nll_target))

            # --- Hit Rate Metrics ---
            target_tokenized = self.tokenizer(batch_targets, add_special_tokens=False, padding="longest", return_tensors="pt")
            next_token_logits = logits[torch.arange(logits.shape[0], device=logits.device), context_lengths - 1, :]
            first_target_token_ids = target_tokenized.input_ids[:, 0].to(logits.device)
            top_100_indices = torch.topk(next_token_logits, 100, dim=1).indices
            target_ids_expanded = first_target_token_ids.unsqueeze(1)

            if 'hit_at_5' in metrics_to_calc:
                all_metrics['hit_at_5'].append((top_100_indices[:, :5] == target_ids_expanded).any(dim=1).float())
            if 'hit_at_50' in metrics_to_calc:
                all_metrics['hit_at_50'].append((top_100_indices[:, :50] == target_ids_expanded).any(dim=1).float())
            if 'hit_at_100' in metrics_to_calc:
                all_metrics['hit_at_100'].append((top_100_indices[:, :100] == target_ids_expanded).any(dim=1).float())

        return {k: torch.cat(v) for k, v in all_metrics.items() if v}

    def _log_metric_wandb(self, log_data, name, tensor):
        valid_mask = ~torch.isinf(tensor) & ~torch.isnan(tensor)
        if valid_mask.any():
            log_data[f"{self.log_prefix}/{name}_avg"] = tensor[valid_mask].mean().item()

    def _build_df_from_history(self, history_data, value_col_name, paraphrase_variant_index=None):
        records = [{'step': entry['step'], 'probe_index': self.probe_indices[i], 'section': self.sections[i], value_col_name: value, **({'paraphrase_variant': paraphrase_variant_index} if paraphrase_variant_index is not None else {})} for entry in history_data for i, value in enumerate(entry['values'])]
        return pd.DataFrame(records)

    def _get_metric_df(self, metric_name, is_delta=False):
        if metric_name not in self.METRICS_CONFIG:
            raise ValueError(f"Metric {metric_name} not found in METRICS_CONFIG")
        config = self.METRICS_CONFIG[metric_name]
        history_data = self.delta_history.get(metric_name) if is_delta else self.history.get(metric_name)
        value_col_name = f"{config['value_col_name']}_delta" if is_delta else config['value_col_name']
        return self._build_df_from_history(history_data, value_col_name)

    def _get_paraphrased_metric_df(self, metric_name, is_delta=False):
        if not self.METRICS_CONFIG.get(metric_name, {}).get('track_paraphrased'):
            return pd.DataFrame()
        config = self.METRICS_CONFIG[metric_name]
        history_list = self.paraphrased_delta_history.get(metric_name) if is_delta else self.paraphrased_history.get(metric_name)
        value_col_name = f"{config['value_col_name']}_delta" if is_delta else config['value_col_name']
        all_variants_df = [self._build_df_from_history(data, value_col_name, i) for i, data in enumerate(history_list) if data]
        return pd.concat(all_variants_df, ignore_index=True) if all_variants_df else pd.DataFrame()

    def save_results(self, output_dir: str):
        """Saves all collected raw and delta metrics to a single CSV file in wide format."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"{self.__class__.__name__}: Saving probe metrics to {output_dir}")

        # 1. Get all base metrics (original probes) and merge them
        base_dfs = []
        for name, config in self.METRICS_CONFIG.items():
            value_col = config['value_col_name']
            df = self._get_metric_df(name)
            if not df.empty:
                base_dfs.append(df.rename(columns={value_col: name}))
            
            if config.get('track_delta'):
                delta_df = self._get_metric_df(name, is_delta=True)
                if not delta_df.empty:
                    base_dfs.append(delta_df.rename(columns={f"{value_col}_delta": f"{name}_delta"}))
        
        if not base_dfs:
            print(" > No base metrics to save.")
            return

        final_df = base_dfs[0]
        for df_to_merge in base_dfs[1:]:
            final_df = pd.merge(final_df, df_to_merge, on=['step', 'probe_index', 'section'], how='outer')

        # 2. Get metrics for each paraphrase variant and merge into the final dataframe
        paraphrase_names = [name for name, cfg in self.METRICS_CONFIG.items() if cfg.get('track_paraphrased')]
        
        for i in range(self.num_paraphrase_variants):
            variant_dfs = []
            for name in paraphrase_names:
                config = self.METRICS_CONFIG[name]
                value_col = config['value_col_name']
                
                # Get variant's value
                history_list = self.paraphrased_history.get(name, [])
                if i < len(history_list) and history_list[i]:
                    df = self._build_df_from_history(history_list[i], value_col_name=value_col)
                    if not df.empty:
                        variant_dfs.append(df.rename(columns={value_col: f"{name}_paraphrase_{i}"}))

                # Get variant's delta
                if config.get('track_delta'):
                    delta_history_list = self.paraphrased_delta_history.get(name, [])
                    if i < len(delta_history_list) and delta_history_list[i]:
                        delta_df = self._build_df_from_history(delta_history_list[i], value_col_name=f"{value_col}_delta")
                        if not delta_df.empty:
                            variant_dfs.append(delta_df.rename(columns={f"{value_col}_delta": f"{name}_delta_paraphrase_{i}"}))

            if variant_dfs:
                merged_variant_df = variant_dfs[0]
                for df_to_merge in variant_dfs[1:]:
                    merged_variant_df = pd.merge(merged_variant_df, df_to_merge, on=['step', 'probe_index', 'section'], how='outer')
                
                # Merge into the main dataframe
                final_df = pd.merge(final_df, merged_variant_df, on=['step', 'probe_index', 'section'], how='left')
        
        if final_df.empty:
            print(" > No metrics to save.")
            return

        output_path = os.path.join(output_dir, f'{self.log_prefix}_metrics.csv')
        final_df.to_csv(output_path, index=False)
        print(f" > Saved consolidated metrics to '{output_path}' with {len(final_df)} rows.")


class RawKnowledgeProbeCallback(BaseKnowledgeProbeCallback):
    """Callback to evaluate model performance on raw knowledge statements."""
    def __init__(self, tokenizer: AutoTokenizer, probe_dataset_path: str, max_length: int, batch_size: int = 8, log_prefix="raw_knowledge_probe", logger=None):
        super().__init__(tokenizer, probe_dataset_path, max_length, batch_size, log_prefix, logger)
        self.raw_knowledge_statements = self._df["raw_knowledge_statement"].tolist()
        
        self.paraphrased_statements_by_variant = []
        if "paraphrased_knowledge_statements" in self._df.columns:
            paraphrased_raw = self._df["paraphrased_knowledge_statements"].dropna().tolist()
            if paraphrased_raw:
                paraphrased_probes = [ast.literal_eval(s) for s in paraphrased_raw]
                if paraphrased_probes:
                    self.paraphrased_statements_by_variant = list(zip(*paraphrased_probes))
        self.num_paraphrase_variants = len(self.paraphrased_statements_by_variant)

        self.PROBE_CONFIG = {
            'raw_knowledge': {
                'track_paraphrased': True,
                'metrics': {
                    'perplexity': {'track_delta': True},
                    'log_prob': {'track_delta': True}
                }
            }
        }
        self._initialize_metrics_config()

    def on_train_begin(self, args, state, control, model, **kwargs):
        print(f"{self.__class__.__name__}: Calculating initial metrics...")
        model.eval()
        device = model.device
        
        initial_metrics = self._evaluate_whole_sentences(model, self.raw_knowledge_statements, device)
        self.initial_metrics['raw_knowledge_perplexity'] = initial_metrics['perplexity']
        self.initial_metrics['raw_knowledge_log_prob'] = initial_metrics['log_prob']
        
        self.initial_metrics['paraphrased'] = [{} for _ in range(self.num_paraphrase_variants)]
        if self.num_paraphrase_variants > 0:
            for i in range(self.num_paraphrase_variants):
                variant_metrics = self._evaluate_whole_sentences(model, self.paraphrased_statements_by_variant[i], device)
                self.initial_metrics['paraphrased'][i]['raw_knowledge_perplexity'] = variant_metrics['perplexity']
                self.initial_metrics['paraphrased'][i]['raw_knowledge_log_prob'] = variant_metrics['log_prob']

        model.train()
        print(f"{self.__class__.__name__}: Initial metrics calculated.")

    def on_step_end(self, args, state, control, model, **kwargs):
        if not self.initial_metrics:
            raise ValueError("Initial metrics not found. Please call on_train_begin first.")
        
        model.eval()
        device = model.device
        step = state.global_step
        log_data = {}
        
        # --- Original Statements ---
        current_metrics = self._evaluate_whole_sentences(model, self.raw_knowledge_statements, model.device)
        for name, values in current_metrics.items():
            metric_name = f"raw_knowledge_{name}"
            self.history[metric_name].append({'step': step, 'values': values.cpu().tolist()})
            self._log_metric_wandb(log_data, metric_name, values)
            
            delta = values - self.initial_metrics[metric_name]
            self.delta_history[metric_name].append({'step': step, 'values': delta.cpu().tolist()})
            self._log_metric_wandb(log_data, f"{metric_name}_delta", delta)

        # --- Paraphrased Statements ---
        if self.num_paraphrase_variants > 0:
            paraphrased_metrics_to_track = {name for name, cfg in self.METRICS_CONFIG.items() if cfg.get('track_paraphrased')}
            if paraphrased_metrics_to_track:
                paraphrased_tensors = {name: [] for name in self.paraphrased_history}
                paraphrased_delta_tensors = {name: [] for name in self.paraphrased_delta_history}
                for i in range(self.num_paraphrase_variants):
                    variant_metrics = self._evaluate_whole_sentences(model, self.paraphrased_statements_by_variant[i], device)
                    for metric_type, values in variant_metrics.items():
                        name = f"raw_knowledge_{metric_type}"
                        if name in paraphrased_metrics_to_track:
                            self.paraphrased_history[name][i].append({'step': step, 'values': values.cpu().tolist()})
                            paraphrased_tensors[name].append(values)
                            if self.METRICS_CONFIG[name].get('track_delta'):
                                delta = values - self.initial_metrics['paraphrased'][i][name]
                                self.paraphrased_delta_history[name][i].append({'step': step, 'values': delta.cpu().tolist()})
                                paraphrased_delta_tensors[name].append(delta)
                
                for name, tensors in paraphrased_tensors.items():
                    if tensors:
                        self._log_metric_wandb(log_data, f"paraphrased_{name}", torch.stack(tensors).mean(dim=0))
                for name, tensors in paraphrased_delta_tensors.items():
                    if tensors:
                        self._log_metric_wandb(log_data, f"paraphrased_{name}_delta", torch.stack(tensors).mean(dim=0))

        if state.is_world_process_zero and log_data:
            wandb.log(log_data, step=step)
        
        model.train()


class AtomicKnowledgeProbeCallback(BaseKnowledgeProbeCallback):
    """Callback to evaluate model on atomic knowledge probes (context + target)."""
    def __init__(self, tokenizer: AutoTokenizer, probe_dataset_path: str, max_length: int, batch_size: int = 8, log_prefix="atomic_knowledge_probe", logger=None):
        super().__init__(tokenizer, probe_dataset_path, max_length, batch_size, log_prefix, logger)
        self.atomic_probes = self._df["atomic_knowledge_probe"].tolist()
        self.atomic_targets = self._df["atomic_target_span"].tolist()
        
        self.paraphrased_atomic_probes_by_variant = []
        if "paraphrased_atomic_knowledge_probes" in self._df.columns:
            paraphrased_probes_raw = self._df["paraphrased_atomic_knowledge_probes"].dropna().tolist()
            if paraphrased_probes_raw:
                paraphrased_probes = [ast.literal_eval(s) for s in paraphrased_probes_raw]
                if paraphrased_probes:
                    self.paraphrased_atomic_probes_by_variant = list(zip(*paraphrased_probes))
        
        self.num_paraphrase_variants = len(self.paraphrased_atomic_probes_by_variant)

        self.PROBE_CONFIG = {
            'atomic_whole': {
                'track_paraphrased': True,
                'metrics': {
                    'perplexity': {'track_delta': True},
                    'log_prob': {'track_delta': True}
                }
            },
            'atomic_target': {
                'track_paraphrased': True,
                'metrics': {
                    'perplexity': {'track_delta': True},
                    'log_prob': {'track_delta': True},
                    'hit_at_5': {'track_delta': False},
                    'hit_at_50': {'track_delta': False},
                    'hit_at_100': {'track_delta': False}
                }
            }
        }
        self._initialize_metrics_config()
        self.atomic_metric_names = [k for k in self.METRICS_CONFIG if k.startswith('atomic_')]

    def _calculate_atomic_metrics(self, model, contexts, targets, device):
        all_metrics = {}

        # --- Evaluate Whole Statement Metrics ---
        full_text = [c + t for c, t in zip(contexts, targets)]
        whole_metrics = self._evaluate_whole_sentences(model, full_text, device)
        all_metrics['atomic_whole_perplexity'] = whole_metrics['perplexity']
        all_metrics['atomic_whole_log_prob'] = whole_metrics['log_prob']

        # --- Evaluate Target-Specific Metrics ---
        target_metrics = self._evaluate_target_probes(model, contexts, targets, device)
        all_metrics['atomic_target_perplexity'] = target_metrics['perplexity']
        all_metrics['atomic_target_log_prob'] = target_metrics['log_prob']
        if 'hit_at_5' in target_metrics:
            all_metrics['atomic_target_hit_at_5'] = target_metrics['hit_at_5']
        if 'hit_at_50' in target_metrics:
            all_metrics['atomic_target_hit_at_50'] = target_metrics['hit_at_50']
        if 'hit_at_100' in target_metrics:
            all_metrics['atomic_target_hit_at_100'] = target_metrics['hit_at_100']
        
        # Filter out metrics that are not configured to be tracked
        return {k: v for k, v in all_metrics.items() if k in self.atomic_metric_names}

    def on_train_begin(self, args, state, control, model, **kwargs):
        print(f"{self.__class__.__name__}: Calculating initial metrics...")
        model.eval()
        device = model.device
        self.initial_metrics.update(self._calculate_atomic_metrics(model, self.atomic_probes, self.atomic_targets, device))

        self.initial_metrics['paraphrased'] = [{} for _ in range(self.num_paraphrase_variants)]
        if self.num_paraphrase_variants > 0:
            print(f"Calculating initial metrics for {self.num_paraphrase_variants} paraphrase variants...")
            for i in range(self.num_paraphrase_variants):
                paraphrased_metrics = self._calculate_atomic_metrics(model, self.paraphrased_atomic_probes_by_variant[i], self.atomic_targets, device)
                for name, values in paraphrased_metrics.items():
                    if self.METRICS_CONFIG.get(name, {}).get('track_paraphrased'):
                        self.initial_metrics['paraphrased'][i][name] = values
        model.train()
        print(f"{self.__class__.__name__}: Initial metrics calculated.")

    def on_step_end(self, args, state, control, model, **kwargs):
        if not self.initial_metrics:
            self.on_train_begin(args, state, control, model, **kwargs)
        
        model.eval()
        device = model.device
        step = state.global_step
        log_data = {}

        # Original Probes
        current_metrics = self._calculate_atomic_metrics(model, self.atomic_probes, self.atomic_targets, device)
        for name, values in current_metrics.items():
            self.history[name].append({'step': step, 'values': values.cpu().tolist()})
            self._log_metric_wandb(log_data, name, values)
            if self.METRICS_CONFIG[name].get('track_delta'):
                delta = values - self.initial_metrics[name]
                self.delta_history[name].append({'step': step, 'values': delta.cpu().tolist()})
                self._log_metric_wandb(log_data, f"{name}_delta", delta)

        # Paraphrased Probes
        if self.num_paraphrase_variants > 0:
            paraphrased_metrics_to_track = {name for name, cfg in self.METRICS_CONFIG.items() if cfg.get('track_paraphrased')}
            if paraphrased_metrics_to_track:
                paraphrased_tensors = {name: [] for name in self.paraphrased_history}
                paraphrased_delta_tensors = {name: [] for name in self.paraphrased_delta_history}
                for i in range(self.num_paraphrase_variants):
                    variant_metrics = self._calculate_atomic_metrics(model, self.paraphrased_atomic_probes_by_variant[i], self.atomic_targets, device)
                    for name, values in variant_metrics.items():
                        if name in paraphrased_metrics_to_track:
                            self.paraphrased_history[name][i].append({'step': step, 'values': values.cpu().tolist()})
                            paraphrased_tensors[name].append(values)
                            if self.METRICS_CONFIG[name].get('track_delta'):
                                delta = values - self.initial_metrics['paraphrased'][i][name]
                                self.paraphrased_delta_history[name][i].append({'step': step, 'values': delta.cpu().tolist()})
                                paraphrased_delta_tensors[name].append(delta)
                
                for name, tensors in paraphrased_tensors.items():
                    if tensors:
                        self._log_metric_wandb(log_data, f"paraphrased_{name}", torch.stack(tensors).mean(dim=0))
                for name, tensors in paraphrased_delta_tensors.items():
                    if tensors:
                        self._log_metric_wandb(log_data, f"paraphrased_{name}_delta", torch.stack(tensors).mean(dim=0))

        if state.is_world_process_zero and log_data:
            wandb.log(log_data, step=step)
        model.train()


class CorpusPerplexityCallback(TrainerCallback):
    """
    Calculates the perplexity of an entire text corpus at the end of each
    training step using a strided sliding window approach. This provides a
    more accurate perplexity measure for long documents than naive chunking.
    Based on the Hugging Face documentation for PPL with fixed-length models.
    """
    def __init__(self, text_content: str, tokenizer: AutoTokenizer, max_length: int, stride: int = 512, log_prefix="corpus_perplexity"):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.log_prefix = log_prefix
        self.encodings = self.tokenizer(text_content, return_tensors="pt")
        self.history = []

    def on_step_end(self, args, state, control, model, **kwargs):
        model.eval()
        device = model.device

        seq_len = self.encodings.input_ids.size(1)
        nll_sum = 0.0
        n_tokens = 0
        prev_end_loc = 0

        for begin_loc in range(0, seq_len, self.stride):
            end_loc = min(begin_loc + self.max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = self.encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            
            # Mask out tokens that are only used for context. The model will not
            # calculate loss for these tokens (label = -100).
            target_ids[:, :-trg_len] = -100

            if torch.all(target_ids == -100):
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                # outputs.loss is the *average* negative log-likelihood for the window.
                neg_log_likelihood = outputs.loss

            # To get the total NLL for the window, we multiply the average by the
            # number of tokens the loss was calculated over.
            num_valid_tokens = (target_ids != -100).sum().item()
            # The model internally shifts labels, so loss is on one less token per sequence.
            # Our batch size is 1 here.
            num_loss_tokens = num_valid_tokens - 1
            if num_loss_tokens > 0:
                nll_sum += neg_log_likelihood.item() * num_loss_tokens
                n_tokens += num_loss_tokens

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
        
        if n_tokens > 0:
            avg_nll = nll_sum / n_tokens
            perplexity = torch.exp(torch.tensor(avg_nll))
        else:
            perplexity = torch.tensor(float('inf'))

        perplexity_item = perplexity.item()
        if state.is_world_process_zero:
            wandb.log({f"{self.log_prefix}/full_paper": perplexity_item}, step=state.global_step)
        
        self.history.append({'step': state.global_step, 'corpus_perplexity': perplexity_item})

        model.train()

    def get_results_as_dataframe(self):
        """
        Returns the collected corpus perplexity data as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)
    
    def save_results(self, output_dir: str):
        """Saves the collected corpus perplexity data to a CSV file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{self.log_prefix}_metrics.csv")
        df = self.get_results_as_dataframe()
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f" > Saved corpus perplexity metrics to '{output_path}' with {len(df)} rows.")
        else:
            print(" > No corpus perplexity metrics to save.")


class TrainingLossPerplexityCallback(TrainerCallback):
    """
    A callback that captures the training loss at each logging step,
    calculates perplexity from it, logs it to Weights & Biases,
    and stores it for external analysis.
    This represents the perplexity of the specific data chunk seen in that step.
    """
    def __init__(self):
        self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        # The 'loss' key is only present during training steps.
        if logs is not None and 'loss' in logs:
            if state.is_world_process_zero:
                # The 'loss' is the average cross-entropy loss for the batch.
                # Perplexity is the exponentiation of this loss.
                chunk_perplexity = math.exp(logs['loss'])
                self.history.append({'step': state.global_step, 'chunked_perplexity': chunk_perplexity})
                wandb.log({"chunked_perplexity/full_paper": chunk_perplexity}, step=state.global_step+1)
    
    def get_results_as_dataframe(self):
        """
        Returns the collected training loss perplexity data as a pandas DataFrame.
        """
        return pd.DataFrame(self.history)

    def save_results(self, output_dir: str):
        """Saves the collected training loss perplexity data to a CSV file."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "training_loss_perplexity_metrics.csv")
        df = self.get_results_as_dataframe()
        if not df.empty:
            df.to_csv(output_path, index=False)
            print(f" > Saved training loss perplexity metrics to '{output_path}' with {len(df)} rows.")
        else:
            print(" > No training loss perplexity metrics to save.")
