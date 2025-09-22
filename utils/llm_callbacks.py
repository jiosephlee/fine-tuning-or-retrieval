import torch
from transformers import TrainerCallback, AutoTokenizer
from typing import List, Dict, Any
import pandas as pd
import os
import wandb
import json
from utils.llm_inference import generate_text
from utils.llm_configs import InferenceConfig
from utils.llm_evals import evaluate_response
import matplotlib.pyplot as plt
import math
from tqdm import tqdm

class BaseKnowledgeProbeCallBack(TrainerCallback):
    """
    A simplified base callback for evaluating model performance on knowledge probes.
    This class provides shared utilities for evaluating probes and calculating metrics.
    """
    def __init__(self, 
                 tokenizer: AutoTokenizer, 
                 facts: List[str],
                 probes: List[str],
                 targets: List[str],
                 probes_df: pd.DataFrame = None,
                 track_hits: bool = True, 
                 track_logprobs: bool = True,
                 batch_size: int = 8, 
                 logger=None,
                 output_dir="",
                 log_prefix="probe_eval",
                 report_to_wandb: bool = True):
        
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.facts = facts
        self.probes = probes
        self.targets = targets
        self.probes_df = probes_df
        
        self.track_hits = track_hits
        self.track_logprobs = track_logprobs
        self.batch_size = batch_size
        self.logger = logger
        self.log_prefix = log_prefix
        self.report_to_wandb = report_to_wandb
        self.excluded_report_columns = ['section', 'subsection', 'section_text', 'subsection_text', 'subsection_text_paraphrased', 'section_text_paraphrased']

        self.initial_metrics = {}
        self.history = {
            'log_prob': [],
            'perplexity': [],
            'hit_accuracy_at_1': [],
            'hit_accuracy_at_10': [],
            'hit_accuracy_at_100': [],
        }
        self.output_dir = output_dir
        self._precompute_token_lengths()

    def _precompute_token_lengths(self):
        """Tokenizes probes and targets to determine their lengths in advance."""
        print("Pre-computing token lengths for probes and targets...")

        # Tokenize once, then calculate lengths in two ways for verification
        
        # --- Probes (Context) ---
        tokenized_probes = self.tokenizer(self.probes, padding=False,add_special_tokens=False)
        tokenized_probes_tensor = self.tokenizer(self.probes, padding=True,add_special_tokens=False, return_tensors="pt")
        context_lengths_new = torch.tensor([len(ids) for ids in tokenized_probes['input_ids']])
        context_lengths_orig = tokenized_probes_tensor.attention_mask.sum(dim=1)

        # --- Targets ---
        tokenized_targets = self.tokenizer(self.targets, padding=False,add_special_tokens=False)
        tokenized_targets_tensor = self.tokenizer(self.targets, padding=True,add_special_tokens=False, return_tensors="pt")
        target_lengths_new = torch.tensor([len(ids) for ids in tokenized_targets['input_ids']])
        target_lengths_orig = tokenized_targets_tensor.attention_mask.sum(dim=1)

        # --- Facts ---
        tokenized_facts = self.tokenizer(self.facts, padding=False,add_special_tokens=False)
        tokenized_facts_tensor = self.tokenizer(self.facts, padding=True,add_special_tokens=False, return_tensors="pt")
        fact_lengths_new = torch.tensor([len(ids) for ids in tokenized_facts['input_ids']])
        fact_lengths_orig = tokenized_facts_tensor.attention_mask.sum(dim=1)

        # Sanity check: assert both methods yield the same result
        if not torch.equal(context_lengths_new, context_lengths_orig):
            self.logger.warning("Context length calculation mismatch.")
        if not torch.equal(target_lengths_new, target_lengths_orig):
            self.logger.warning("Target length calculation mismatch.")
        if not torch.equal(fact_lengths_new, fact_lengths_orig):
            self.logger.warning("Fact length calculation mismatch.")
        
        self.context_lengths = context_lengths_new
        self.target_lengths = target_lengths_new
        self.fact_lengths = fact_lengths_new
        
        if not torch.equal(self.context_lengths + self.target_lengths, fact_lengths_new):
            self.logger.warning(
                "Mismatch between (context + target) length and fact length."
            )

        self.tokenized_probes = tokenized_probes_tensor
        self.tokenized_targets = tokenized_targets_tensor
        self.tokenized_facts = tokenized_facts_tensor
        
        print("Token lengths pre-computation finished.")

    def on_train_begin(self, args, state, control, model, **kwargs):
        """Calculate initial metrics before training starts."""
        print(f"{self.__class__.__name__}: Calculating initial metrics...")
        model.eval()
        self.initial_metrics = self._evaluate_probes(model)
        
        # Log initial metrics to history at step 0
        step = 0
        for metric_name, values in self.initial_metrics.items():
            if values is not None:
                self.history[metric_name].append({'step': step, 'values': values.cpu().tolist()})
        
        self._generate_full_token_analysis_report(model, "initial")

        model.train()
        print(f"{self.__class__.__name__}: Initial metrics calculated and analysis report generated.")

        output_dir = self.output_dir
        self._generate_best_probes_report(output_dir)

    def on_step_end(self, args, state, control, model, **kwargs):
        """Evaluate probes at the end of a training step and log metrics."""
        model.eval()
        current_metrics = self._evaluate_probes(model)
        log_data = {}
        step = state.global_step

        for metric_name, values in current_metrics.items():
            if values is None:
                continue

            # Log the metrics to local history
            self.history[metric_name].append({'step': step, 'values': values.cpu().tolist()})
            
            valid_mask = ~torch.isinf(values) & ~torch.isnan(values)
            if valid_mask.any():
                log_data[f"{self.log_prefix}/{metric_name}_avg"] = values[valid_mask].mean().item()

        if state.is_world_process_zero and log_data:
            if self.report_to_wandb and wandb.run:
                wandb.log(log_data, step=step)
            
        model.train()

    def on_train_end(self, args, state, control, model, **kwargs):
        """Generate a detailed report of the worst-performing probes at the end of training."""
        print(f"{self.__class__.__name__}: Final evaluation and report generation...")
        model.eval()
        self._generate_full_token_analysis_report(model, "final")
        model.train()

        output_dir = self.output_dir
        self._generate_worst_probes_report(output_dir)
        self._generate_most_and_least_learned_probes_report(output_dir)
        print(f"{self.__class__.__name__}: Final reports generated.")

    def _generate_best_probes_report(self, output_dir, top_k=10):
        """
        Generates a report on the top_k best-performing probes based on initial perplexity.
        """
        if not self.initial_metrics or 'perplexity' not in self.initial_metrics or self.initial_metrics['perplexity'] is None:
            print("No initial perplexity to generate best probes report from.")
            return

        initial_analysis_path = os.path.join(output_dir, f'{self.log_prefix}_initial_token_analysis.json')
        if not os.path.exists(initial_analysis_path):
            print(f"Initial token analysis file not found at {initial_analysis_path}")
            return
            
        print(f"{self.__class__.__name__}: Generating best probes report...")

        with open(initial_analysis_path, 'r') as f:
            initial_analysis_data = json.load(f)

        initial_perplexities = self.initial_metrics['perplexity'].clone().cpu()
        initial_perplexities[torch.isnan(initial_perplexities)] = float('inf')
        best_probe_indices = torch.argsort(initial_perplexities, descending=False)[:top_k]

        report_path = os.path.join(output_dir, f'{self.log_prefix}_best_probes_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Best Probes Report (Top {top_k} by Perplexity) at Step 0 (Before Training)\n")
            f.write("="*50 + "\n\n")

            for i, probe_idx in enumerate(best_probe_indices):
                probe_idx_int = probe_idx.item()
                
                f.write(f"--- Probe Index: {probe_idx_int} (Initial Rank: {i+1}) ---\n")
                if self.probes_df is not None and probe_idx_int < len(self.probes_df):
                    metadata = self.probes_df.iloc[probe_idx_int].to_dict()
                    for key, val in metadata.items():
                        if key not in self.excluded_report_columns:
                            f.write(f"{key}: {val}\n")
                else:
                    f.write(f"Fact: {self.facts[probe_idx_int]}\n")

                f.write("\nInitial Metrics:\n")
                for metric_name, values in self.initial_metrics.items():
                    if values is not None:
                        value = values[probe_idx_int]
                        f.write(f"  {metric_name}: {value:.4f}\n")

                f.write("\nDetailed Token-level Analysis:\n")
                
                analysis_text = self._format_token_analysis_from_json(initial_analysis_data.get(str(probe_idx_int)))
                f.write(analysis_text)
                f.write("\n" + "="*50 + "\n\n")

        print(f" > Saved best probes report to '{report_path}'")

    def _generate_most_and_least_learned_probes_report(self, output_dir, top_k=5):
        """
        Generates reports on the probes that improved the most and the least during training.
        """
        if not self.initial_metrics or 'perplexity' not in self.initial_metrics or self.initial_metrics['perplexity'] is None:
            print("No initial perplexity to generate learned probes report from.")
            return
        if not self.history['perplexity'] or len(self.history['perplexity']) < 1:
            print("No perplexity history to generate learned probes report from.")
            return

        initial_analysis_path = os.path.join(output_dir, f'{self.log_prefix}_initial_token_analysis.json')
        final_analysis_path = os.path.join(output_dir, f'{self.log_prefix}_final_token_analysis.json')

        if not os.path.exists(initial_analysis_path) or not os.path.exists(final_analysis_path):
            print(f"Token analysis files not found. Needed: {initial_analysis_path} and {final_analysis_path}")
            return

        print(f"{self.__class__.__name__}: Generating most and least learned probes report...")

        with open(initial_analysis_path, 'r') as f:
            initial_analysis_data = json.load(f)
        with open(final_analysis_path, 'r') as f:
            final_analysis_data = json.load(f)

        initial_perplexities = self.initial_metrics['perplexity'].clone().cpu()
        final_perplexities = torch.tensor(self.history['perplexity'][-1]['values'])
        final_step = self.history['perplexity'][-1]['step']

        initial_perplexities[torch.isnan(initial_perplexities)] = float('inf')
        final_perplexities[torch.isnan(final_perplexities)] = float('inf')

        initial_ranks = torch.empty_like(initial_perplexities, dtype=torch.long)
        initial_ranks[torch.argsort(initial_perplexities)] = torch.arange(len(initial_perplexities)) + 1
        
        final_ranks = torch.empty_like(final_perplexities, dtype=torch.long)
        final_ranks[torch.argsort(final_perplexities)] = torch.arange(len(final_perplexities)) + 1

        perplexity_delta = initial_perplexities - final_perplexities
        perplexity_delta[torch.isnan(perplexity_delta)] = 0.0

        most_learned_indices = torch.argsort(perplexity_delta, descending=True)[:top_k]
        least_learned_indices = torch.argsort(perplexity_delta, descending=False)[:top_k]

        # Generate report for MOST learned
        self._generate_learning_report_for_indices(output_dir, most_learned_indices, 'most_learned', final_step, top_k, initial_ranks, final_ranks, initial_analysis_data, final_analysis_data)
        
        # Generate report for LEAST learned
        self._generate_learning_report_for_indices(output_dir, least_learned_indices, 'least_learned', final_step, top_k, initial_ranks, final_ranks, initial_analysis_data, final_analysis_data)

    def _generate_learning_report_for_indices(self, output_dir, probe_indices, report_type, final_step, top_k, initial_ranks=None, final_ranks=None, initial_analysis=None, final_analysis=None):
        report_path = os.path.join(output_dir, f'{self.log_prefix}_{report_type}_probes_report.txt')
        title_part = "Most Learned" if report_type == 'most_learned' else "Least Learned"
        
        with open(report_path, 'w') as f:
            f.write(f"{title_part} Probes Report (Top {top_k} by Perplexity Change) at Step {final_step}\n")
            f.write("="*50 + "\n\n")

            for i, probe_idx in enumerate(probe_indices):
                probe_idx_int = probe_idx.item()
                
                f.write(f"--- Probe Index: {probe_idx_int} ---\n")
                if self.probes_df is not None and probe_idx_int < len(self.probes_df):
                    metadata = self.probes_df.iloc[probe_idx_int].to_dict()
                    for key, val in metadata.items():
                        if key not in self.excluded_report_columns:
                            f.write(f"{key}: {val}\n")
                else:
                    f.write(f"Fact: {self.facts[probe_idx_int]}\n")

                f.write("\nInitial Metrics:\n")
                for metric_name, values in self.initial_metrics.items():
                    if values is not None:
                        value = values[probe_idx_int]
                        if metric_name == 'perplexity' and initial_ranks is not None:
                            rank = initial_ranks[probe_idx_int].item()
                            f.write(f"  {metric_name}: {value:.4f} (Rank: {rank})\n")
                        else:
                            f.write(f"  {metric_name}: {value:.4f}\n")
                
                f.write("\nFinal Metrics:\n")
                for metric_name, history_data in self.history.items():
                    if history_data:
                        final_value = history_data[-1]['values'][probe_idx_int]
                        if metric_name == 'perplexity' and final_ranks is not None:
                            rank = final_ranks[probe_idx_int].item()
                            f.write(f"  {metric_name}: {final_value:.4f} (Rank: {rank})\n")
                        else:
                            f.write(f"  {metric_name}: {final_value:.4f}\n")

                if initial_analysis:
                    f.write("\nDetailed Token-level Analysis (Initial State):\n")
                    analysis_text = self._format_token_analysis_from_json(initial_analysis.get(str(probe_idx_int)))
                    f.write(analysis_text)

                if final_analysis:
                    f.write("\nDetailed Token-level Analysis (Final State):\n")
                    analysis_text = self._format_token_analysis_from_json(final_analysis.get(str(probe_idx_int)))
                    f.write(analysis_text)

                f.write("\n" + "="*50 + "\n\n")

        print(f" > Saved {report_type} probes report to '{report_path}'")

    def _get_target_mask(self, tokenized_full, context_lengths, target_lengths, full_lengths):
        """Identifies the token positions of the target sequence within the full sequence.
        Expects input to be Tensor of shape (batch_size, seq_len - 1) and returns a Tensor of shape (batch_size, seq_len - 1)"""
        mask = torch.zeros_like(tokenized_full, dtype=torch.bool)

        for i in range(tokenized_full.shape[0]):
            expected_full_length = context_lengths[i].item() + target_lengths[i].item()
            actual_full_length = full_lengths[i].item() 
            if expected_full_length != actual_full_length:
                self.logger.warning(
                    f"Length mismatch at index {i}: context ({context_lengths[i].item()}) + target ({target_lengths[i].item()}) = {expected_full_length} != full ({actual_full_length})"
                )
            
            start, end = int(context_lengths[i].item()), int(context_lengths[i].item()) + int(target_lengths[i].item())
            if start < end:
                mask[i, start:end] = True
            
        return mask
    
    def _calculate_log_probs(self, logits, labels, context_lengths, target_lengths):
        full_lengths = context_lengths + target_lengths
        target_mask = self._get_target_mask(labels, context_lengths, target_lengths, full_lengths)
        
        # Mask out the logits that are not the target
        labels_masked = labels.clone()
        
        labels_masked[labels_masked == self.tokenizer.pad_token_id] = -100
        labels_masked[~target_mask] = -100
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_target = loss_fct(logits.permute(0, 2, 1), labels_masked) # shape (batch_size, seq_len - 1)
        sum_loss_target = loss_target.sum(dim=1)
        
        log_prob = -sum_loss_target
        
        num_tokens_target = (labels_masked != -100).sum(dim=1).float() # dim=1 because we want to sum over the sequence length, preserving the batch dimension
        # assert that num_tokens_target is the same as target_lengths 
        if not torch.equal(num_tokens_target.long(), target_lengths):
            # Find the indices where the mismatch occurs
            mismatch_indices = torch.where(num_tokens_target.long() != target_lengths)[0]
            for i in mismatch_indices:
                self.logger.warning(f"Number of tokens target mismatch for a probe in batch.")
                # Note: The original probe index would require passing the batch start index 'i' down to this function.
                # For now, we log information about the mismatched sample within the current batch.
                self.logger.warning(f"  - Mismatch at batch index: {i.item()}")
                self.logger.warning(f"  - Expected target length: {target_lengths[i].item()}")
                self.logger.warning(f"  - Calculated target tokens: {num_tokens_target[i].long().item()}")
                
                # To get the original text, we would need to pass the original data down.
                # This is a placeholder for what we would want to log.
                # self.logger.warning(f"  - Target Text: {self.targets[original_index]}")

                # Log token information
                target_token_ids = labels[i, int(context_lengths[i].item()):int(context_lengths[i].item()) + int(target_lengths[i].item())]
                self.logger.warning(f"  - Original Target Token IDs: {target_token_ids.tolist()}")
                self.logger.warning(f"  - Tokenizer pad_token_id: {self.tokenizer.pad_token_id}")

        # assert that num_tokens_target is the same as non-zero in the loss_target
        if not torch.equal(num_tokens_target.long(), (loss_target != 0).sum(dim=1)):
            self.logger.warning("Number of tokens target mismatch")
        mean_nll_target = sum_loss_target / num_tokens_target
        perplexity = torch.exp(mean_nll_target)

        return log_prob, perplexity

    def _calculate_hits_at_k(self, logits, input_ids, context_lengths, target_lengths, k_values: List[int]):
        """
        Calculates the hit accuracy at various k values for multi-token targets.

        For each probe, this function computes the "hit accuracy," defined as the
        proportion of target tokens that were correctly predicted within the top-k
        logits. The model sees the preceding ground-truth tokens when making each
        prediction (teacher forcing).

        Args:
            logits (torch.Tensor): The model's output shifted_logits for the batch.
            input_ids (torch.Tensor): The shifted_labels for the batch.
            context_lengths (torch.Tensor): The lengths of the context part of each probe (already decremented by 1).
            target_lengths (torch.Tensor): The lengths of the target part of each probe.
            k_values (List[int]): A list of k values for top-k evaluation.

        Returns:
            Dict[str, torch.Tensor]: A dictionary mapping each k to a tensor
                                     of hit accuracies for the batch.
        """
        batch_size = logits.shape[0]
        hits_at_k = {f'hit_accuracy_at_{k}': [] for k in k_values}

        for i in range(batch_size):
            start = context_lengths[i]
            end = start + target_lengths[i]
            
            pred_logits = logits[i, start:end, :]
            actual_tokens = input_ids[i, start:end]

            if pred_logits.shape[0] == 0:  # Handle cases with no target tokens
                for k in k_values:
                    hits_at_k[f'hit_accuracy_at_{k}'].append(torch.tensor(0.0, device=logits.device))
                continue
            
            max_k = max(k_values)
            top_k_indices = torch.topk(pred_logits, max_k, dim=-1).indices

            for k in k_values:
                # Check if the actual token is in the top k predictions for each position
                hits = (top_k_indices[:, :k] == actual_tokens.unsqueeze(1)).any(dim=-1) # if any of the top k predictions are the actual token, then it's a hit
                
                # Accuracy is the mean of hits over the target token sequence
                accuracy = hits.float().mean()
                hits_at_k[f'hit_accuracy_at_{k}'].append(accuracy)

        # Convert lists of tensors to a single tensor for each k
        for k in k_values:
            key = f'hit_accuracy_at_{k}'
            hits_at_k[key] = torch.stack(hits_at_k[key])
            
        return hits_at_k

    def _evaluate_probes(self, model, return_logits=False) -> Dict[str, torch.Tensor]:
        """
        Evaluates probes by calculating log probabilities and hit rates for targets.
        """
        all_metrics = { 'log_prob': [], 'perplexity': [], 'hit_accuracy_at_1': [], 'hit_accuracy_at_10': [], 'hit_accuracy_at_100': [] }
        all_logits = [] if return_logits else None
        device = model.device
        num_facts = len(self.facts)
        # Go through facts in batches
        for i in range(0, num_facts, self.batch_size):
            end_index = i + self.batch_size
            
            # Get fact logits
            inputs = {
                'input_ids': self.tokenized_facts['input_ids'][i:end_index].to(device),
                'attention_mask': self.tokenized_facts['attention_mask'][i:end_index].to(device)
            }
            # print(inputs)
            attention_mask = inputs['attention_mask']
            with torch.no_grad():
                logits = model(**inputs).logits
            
            if return_logits:
                all_logits.append(logits.cpu())

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            context_lengths = self.context_lengths[i:end_index].to(device)
            target_lengths = self.target_lengths[i:end_index].to(device)
            # assert, one more time, that the lengths are correct
            if not torch.equal(context_lengths + target_lengths, attention_mask.sum(dim=1)):
                self.logger.warning("Length mismatch between context and target lengths and fact lengths")
            # Now that we've shifted the logits, we need to change the context lengths by -1 to account for the shift
            context_lengths = context_lengths - 1
            
            if self.track_logprobs:
                log_prob, perplexity = self._calculate_log_probs(shift_logits, shift_labels, context_lengths, target_lengths)
                all_metrics['log_prob'].append(log_prob)
                all_metrics['perplexity'].append(perplexity)

            if self.track_hits:
                hits = self._calculate_hits_at_k(shift_logits, shift_labels, context_lengths, target_lengths, k_values=[1, 10, 100])
                for k, v in hits.items():
                    all_metrics[k].append(v)

        metrics_to_return = {
            "log_prob": torch.cat(all_metrics['log_prob']) if self.track_logprobs and all_metrics['log_prob'] else None,
            "perplexity": torch.cat(all_metrics['perplexity']) if self.track_logprobs and all_metrics['perplexity'] else None,
            "hit_accuracy_at_1": torch.cat(all_metrics['hit_accuracy_at_1']) if self.track_hits and all_metrics['hit_accuracy_at_1'] else None,
            "hit_accuracy_at_10": torch.cat(all_metrics['hit_accuracy_at_10']) if self.track_hits and all_metrics['hit_accuracy_at_10'] else None,
            "hit_accuracy_at_100": torch.cat(all_metrics['hit_accuracy_at_100']) if self.track_hits and all_metrics['hit_accuracy_at_100'] else None,
        }

        if return_logits:
            return metrics_to_return, torch.cat(all_logits)
        else:
            return metrics_to_return

    def _generate_worst_probes_report(self, output_dir, top_k=10):
        """
        Generates a report on the top_k worst-performing probes based on final perplexity.
        """
        if not self.history['perplexity']:
            print("No perplexity history to generate report from.")
            return

        final_analysis_path = os.path.join(output_dir, f'{self.log_prefix}_final_token_analysis.json')
        if not os.path.exists(final_analysis_path):
            print(f"Final token analysis file not found at {final_analysis_path}")
            return

        print(f"{self.__class__.__name__}: Generating worst probes report...")

        with open(final_analysis_path, 'r') as f:
            final_analysis_data = json.load(f)

        final_metrics = self.history['perplexity'][-1]
        final_perplexities = torch.tensor(final_metrics['values'])
        
        final_perplexities[torch.isnan(final_perplexities)] = float('inf')
        worst_probe_indices = torch.argsort(final_perplexities, descending=True)[:top_k]

        report_path = os.path.join(output_dir, f'{self.log_prefix}_worst_probes_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Worst Probes Report (Top {top_k} by Perplexity) at Step {final_metrics['step']}\n")
            f.write("="*50 + "\n\n")

            for i, probe_idx in enumerate(worst_probe_indices):
                probe_idx_int = probe_idx.item()
                
                # Write metadata
                f.write(f"--- Probe Index: {probe_idx_int} (Final Rank: {i+1}) ---\n")
                if self.probes_df is not None and probe_idx_int < len(self.probes_df):
                    metadata = self.probes_df.iloc[probe_idx_int].to_dict()
                    for key, val in metadata.items():
                        if key not in self.excluded_report_columns:
                            f.write(f"{key}: {val}\n")
                else:
                    f.write(f"Fact: {self.facts[probe_idx_int]}\n")

                # Write final metrics for this probe
                f.write("\nFinal Metrics:\n")
                for metric_name, history_data in self.history.items():
                    if history_data:
                        final_value = history_data[-1]['values'][probe_idx_int]
                        f.write(f"  {metric_name}: {final_value:.4f}\n")

                # Write detailed token analysis
                f.write("\nDetailed Token-level Analysis:\n")
                
                analysis_text = self._format_token_analysis_from_json(final_analysis_data.get(str(probe_idx_int)))
                f.write(analysis_text)
                f.write("\n" + "="*50 + "\n\n")

        print(f" > Saved worst probes report to '{report_path}'")
    
    def _format_token_analysis_from_json(self, analysis_data):
        """Formats the token analysis data from a JSON object into a readable string."""
        if not analysis_data:
            return "  No token analysis available for this probe.\n"
        
        report_lines = []
        for token_data in analysis_data:
            report_lines.append(f"  - Target Token #{token_data['target_token_#']}: '{token_data['actual_token']}'")
            report_lines.append(f"    - Rank: {token_data['rank']}")
            report_lines.append(f"    - Top {len(token_data['top_k_predictions'])} predictions:")
            for i, pred in enumerate(token_data['top_k_predictions']):
                report_lines.append(f"      {i+1}. '{pred['token']}' (Prob: {pred['prob']:.4f})")
        return "\n".join(report_lines)

    def _get_detailed_token_analysis(self, logits, labels, context_length, target_length, top_k=10):
        """
        For a single probe, generates a detailed analysis of each target token's prediction
        and returns it as a JSON-serializable list of dictionaries.
        """
        analysis_list = []
        
        start = context_length.item()
        end = start + target_length.item()
        
        target_token_ids = labels[start:end]
        target_logits = logits[start:end]
        
        for i in range(target_token_ids.shape[0]):
            token_pos = start + i
            actual_token_id = target_token_ids[i].item()
            
            if actual_token_id == self.tokenizer.pad_token_id or actual_token_id == -100:
                continue
                
            token_logits = target_logits[i]
            
            # Get top k predictions
            top_k_probs, top_k_indices = torch.topk(torch.softmax(token_logits, dim=-1), top_k)
            
            top_k_tokens = [self.tokenizer.decode(t) for t in top_k_indices]
            actual_token = self.tokenizer.decode(actual_token_id)
            
            # Find rank of actual token
            sorted_indices = torch.argsort(token_logits, descending=True)
            rank_tensor = (sorted_indices == actual_token_id).nonzero()
            rank = rank_tensor.item() + 1 if rank_tensor.numel() > 0 else "Not in vocab"
            
            top_k_preds_data = []
            for j in range(top_k):
                top_k_preds_data.append({
                    'token': top_k_tokens[j],
                    'prob': round(top_k_probs[j].item(), 4)
                })

            analysis_list.append({
                'target_token_#': i + 1,
                'actual_token': actual_token,
                'rank': rank,
                'top_k_predictions': top_k_preds_data
            })
                
        return analysis_list

    def _generate_full_token_analysis_report(self, model, state_name):
        """
        Generates a full token-level analysis for all probes and saves it to a JSON file.
        state_name should be 'initial' or 'final'.
        """
        print(f"Generating full token analysis report for '{state_name}' state...")
        
        full_analysis = {}
        device = model.device
        num_facts = len(self.facts)

        for i in tqdm(range(0, num_facts, self.batch_size), desc=f"Analyzing probes ({state_name})"):
            end_index = min(i + self.batch_size, num_facts)
            
            inputs = {
                'input_ids': self.tokenized_facts['input_ids'][i:end_index].to(device),
                'attention_mask': self.tokenized_facts['attention_mask'][i:end_index].to(device)
            }
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            context_lengths = self.context_lengths[i:end_index].to(device) - 1 # shifted
            target_lengths = self.target_lengths[i:end_index].to(device)

            for j in range(shift_logits.shape[0]):
                probe_idx = i + j
                probe_analysis = self._get_detailed_token_analysis(
                    shift_logits[j],
                    shift_labels[j],
                    context_lengths[j],
                    target_lengths[j]
                )
                full_analysis[probe_idx] = probe_analysis
        
        output_path = os.path.join(self.output_dir, f'{self.log_prefix}_{state_name}_token_analysis.json')
        with open(output_path, 'w') as f:
            json.dump(full_analysis, f, indent=2)
        
        print(f" > Saved full token analysis to '{output_path}'")

    def save_results(self, output_dir: str):
        """Saves all collected metrics to a CSV file."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"{self.__class__.__name__}: Saving probe metrics to {output_dir}")

        all_dfs = []
        for metric_name, history_data in self.history.items():
            if not history_data:
                continue
            records = [{'step': entry['step'], 'probe_index': i, metric_name: value} for entry in history_data for i, value in enumerate(entry['values'])]
            df = pd.DataFrame(records)
            if not df.empty:
                all_dfs.append(df.set_index(['step', 'probe_index']))
        
        if not all_dfs:
            print(" > No metrics to save.")
            return

        final_df = pd.concat(all_dfs, axis=1).reset_index()
        output_path = os.path.join(output_dir, f'{self.log_prefix}_metrics.csv')
        final_df.to_csv(output_path, index=False)
        print(f" > Saved consolidated metrics to '{output_path}' with {len(final_df)} rows.")

class GenerationProbeCallback(TrainerCallback):
    """
    A callback that periodically generates text from a fixed prompt and saves the output.
    It also evaluates the generated text using an LLM judge.
    """
    def __init__(self,
                 prompts: Dict[str, Any],
                 tokenizer: AutoTokenizer,
                 inference_config: InferenceConfig,
                 eval_every_n_steps: int = 10,
                 logger=None,
                 output_dir: str = "",
                 do_eval: bool = False,
                 report_to_wandb: bool = True):

        self.prompts = prompts
        self.tokenizer = tokenizer
        self.inference_config = inference_config
        self.eval_every_n_steps = eval_every_n_steps
        self.logger = logger
        self.output_dir = output_dir
        self.do_eval = do_eval
        self.report_to_wandb = report_to_wandb
        self.eval_history = {}

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero and state.global_step > 0 and (state.global_step % self.eval_every_n_steps == 0 or state.global_step == state.max_steps - 1):
            if self.logger:
                self.logger.info(f"Running generation probe at step {state.global_step}...")
            else:
                print(f"Running generation probe at step {state.global_step}...")

            model.eval()
            self._generate_and_evaluate(state, model)
            model.train()
    
    def _generate_only(self, state, model):
        # This method is no longer used but kept for potential future use.
        for source, prompts_list in self.prompts.items():
            source_output_dir = os.path.join(self.output_dir, source)
            os.makedirs(source_output_dir, exist_ok=True)

            for prompt_data in prompts_list:
                prompt_name = prompt_data["prompt_name"]
                prompt_text = prompt_data["question"]

                generated_text = generate_text(model, self.tokenizer, prompt_text, self.inference_config)

                prompt_output_dir = os.path.join(source_output_dir, prompt_name)
                os.makedirs(prompt_output_dir, exist_ok=True)
                
                file_path = os.path.join(prompt_output_dir, f"generation_step_{state.global_step}.txt")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write("--- PROMPT ---\n")
                    f.write(prompt_text + "\n\n")
                    f.write("--- GENERATION ---\n")
                    f.write(generated_text)
                
                if self.logger:
                    self.logger.info(f" > Saved generation probe output for {source}/{prompt_name} to '{file_path}'")
                else:
                    print(f" > Saved generation probe output for {source}/{prompt_name} to '{file_path}'")
    
    def _generate_and_evaluate(self, state, model):
        wandb_logs = {}
        
        for dataset_name, prompts_list in self.prompts.items():
            dataset_output_dir = os.path.join(self.output_dir, dataset_name)
            os.makedirs(dataset_output_dir, exist_ok=True)
            
            all_evals = []
            scores = []

            for prompt_data in prompts_list:
                prompt_name = prompt_data["prompt_name"]
                question = prompt_data["question"]
                reference_answer = prompt_data["reference_answer"]

                generated_text = generate_text(model, self.tokenizer, question, self.inference_config)

                prompt_output_dir = os.path.join(dataset_output_dir, prompt_name)
                os.makedirs(prompt_output_dir, exist_ok=True)
                
                gen_file_path = os.path.join(prompt_output_dir, f"generation_step_{state.global_step}.txt")
                with open(gen_file_path, 'w', encoding='utf-8') as f:
                    f.write("--- PROMPT ---\n")
                    f.write(question + "\n\n")
                    f.write("--- GENERATION ---\n")
                    f.write(generated_text + "\n")

                if self.do_eval:
                    eval_result = evaluate_response(
                        question=question,
                        response=generated_text,
                        reference_answer=reference_answer
                    )
                    
                    eval_file_path = os.path.join(prompt_output_dir, f"evaluation_step_{state.global_step}.json")
                    with open(eval_file_path, 'w', encoding='utf-8') as f:
                        json.dump(eval_result, f, indent=4)

                    if self.logger:
                        self.logger.info(f" > Evaluated generation for {dataset_name}/{prompt_name} at step {state.global_step}. Score: {eval_result['score']}")
                    else:
                        print(f" > Evaluated generation for {dataset_name}/{prompt_name} at step {state.global_step}. Score: {eval_result['score']}")

                    if eval_result['score'] is not None:
                        scores.append(eval_result['score'])

                    eval_data = {
                        "step": state.global_step,
                        "prompt_name": prompt_name,
                        "question": question,
                        "generated_text": generated_text,
                        "reference_answer": reference_answer,
                        "score": eval_result['score'],
                        "feedback": eval_result['feedback']
                    }
                    all_evals.append(eval_data)
            
            if self.do_eval:
                csv_path = os.path.join(dataset_output_dir, 'eval_results.csv')
                df = pd.DataFrame(all_evals)
                if os.path.exists(csv_path):
                    df.to_csv(csv_path, mode='a', header=False, index=False)
                else:
                    df.to_csv(csv_path, mode='w', header=True, index=False)

                if scores:
                    mean_score = sum(scores) / len(scores)
                    wandb_logs[f'eval/{dataset_name}_mean_score'] = mean_score

                    if dataset_name not in self.eval_history:
                        self.eval_history[dataset_name] = {'steps': [], 'scores': []}
                    self.eval_history[dataset_name]['steps'].append(state.global_step)
                    self.eval_history[dataset_name]['scores'].append(mean_score)
        
        if self.report_to_wandb and wandb.run and wandb_logs:
            wandb.log(wandb_logs, step=state.global_step)
            
    def on_train_end(self, args, state, control, model, **kwargs):
        """
        Called at the end of training to generate and save plots of evaluation scores.
        """
        if self.eval_history:
            if self.logger:
                self.logger.info("Training ended. Generating final evaluation plots...")
            else:
                print("Training ended. Generating final evaluation plots...")
            self._plot_and_save_eval_history()

    def _plot_and_save_eval_history(self):
        plots_dir = os.path.join(self.output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        for dataset_name, data in self.eval_history.items():
            if len(data['steps']) < 2:
                continue
            
            plt.figure()
            plt.plot(data['steps'], data['scores'], marker='o')
            plt.title(f'Mean Evaluation Score for {dataset_name}')
            plt.xlabel('Training Step')
            plt.ylabel('Mean Score')
            plt.grid(True)
            plot_path = os.path.join(plots_dir, f'{dataset_name}_eval_score.png')
            plt.savefig(plot_path)
            plt.close()

            if self.logger:
                self.logger.info(f" > Saved evaluation plot for {dataset_name} to '{plot_path}'")
            else:
                print(f" > Saved evaluation plot for {dataset_name} to '{plot_path}'")

class CorpusPerplexityCallback(TrainerCallback):
    """
    Calculates the perplexity of an entire text corpus at the end of each
    training step using a strided sliding window approach. This provides a
    more accurate perplexity measure for long documents than naive chunking.
    Based on the Hugging Face documentation for PPL with fixed-length models.
    """
    def __init__(self, 
                 text_content: str, 
                 tokenizer: AutoTokenizer, 
                 max_length: int, 
                 stride: int = 512, 
                 output_dir: str = "",
                 log_prefix="corpus_perplexity",
                 report_to_wandb: bool = True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.log_prefix = log_prefix
        self.output_dir = output_dir
        self.encodings = self.tokenizer(text_content, return_tensors="pt")
        self.history = []
        self.report_to_wandb = report_to_wandb

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
            
            target_ids[:, :-trg_len] = -100

            if torch.all(target_ids == -100):
                prev_end_loc = end_loc
                if end_loc == seq_len:
                    break
                continue

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            num_valid_tokens = (target_ids != -100).sum().item()
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
            avg_nll = float('inf')
            perplexity = torch.tensor(float('inf'))

        perplexity_item = perplexity.item()
        loss_item = avg_nll

        if state.is_world_process_zero:
            if self.report_to_wandb and wandb.run:
                wandb.log({
                    f"{self.log_prefix}/perplexity": perplexity_item,
                    f"{self.log_prefix}/loss": loss_item
                }, step=state.global_step)
        
        self.history.append({
            'step': state.global_step, 
            'corpus_perplexity': perplexity_item,
            'corpus_loss': loss_item
        })

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
            # Save an empty file with headers for consistency
            pd.DataFrame(columns=['step', 'corpus_perplexity', 'corpus_loss']).to_csv(output_path, index=False)
            print(" > No corpus perplexity metrics to save, created empty file with headers.")
            
class TrainingLossPerplexityCallback(TrainerCallback):
    """
    A callback that captures the training loss at each logging step,
    calculates perplexity from it, logs it to Weights & Biases,
    and stores it for external analysis.
    This represents the perplexity of the specific data chunk seen in that step.
    """
    def __init__(self, report_to_wandb=True):
        self.history = []
        self.report_to_wandb = report_to_wandb

    def on_log(self, args, state, control, logs=None, **kwargs):
        # The 'loss' key is only present during training steps.
        if logs is not None and 'loss' in logs:
            if state.is_world_process_zero:
                # The 'loss' is the average cross-entropy loss for the batch.
                # Perplexity is the exponentiation of this loss.
                chunk_perplexity = math.exp(logs['loss'])
                self.history.append({'step': state.global_step, 'loss': logs['loss'], 'chunked_perplexity': chunk_perplexity})
                if self.report_to_wandb:
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