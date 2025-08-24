import torch
from transformers import TrainerCallback, AutoTokenizer
from typing import List, Dict
import pandas as pd
import os
import wandb

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
                 sections: List[str],
                 track_hits: bool = True, 
                 track_logprobs: bool = True,
                 batch_size: int = 8, 
                 logger=None,
                 log_prefix="probe_eval"):
        
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.facts = facts
        self.probes = probes
        self.targets = targets
        self.sections = sections
        
        self.track_hits = track_hits
        self.track_logprobs = track_logprobs
        self.batch_size = batch_size
        self.logger = logger
        self.log_prefix = log_prefix

        self.initial_metrics = {}
        self.history = {
            'log_prob': [],
            'perplexity': [],
            'hit_accuracy_at_5': [],
            'hit_accuracy_at_50': [],
        }
    
        self._precompute_token_lengths()

    def _precompute_token_lengths(self):
        """Tokenizes probes and targets to determine their lengths in advance."""
        print("Pre-computing token lengths for probes and targets...")

        # Tokenize once, then calculate lengths in two ways for verification
        
        # --- Probes (Context) ---
        tokenized_probes = self.tokenizer(self.probes, padding=False,add_special_tokens=False)
        context_lengths_new = torch.tensor([len(ids) for ids in tokenized_probes['input_ids']])
        context_lengths_orig = self.tokenizer.pad(tokenized_probes, padding=True, return_tensors="pt").attention_mask.sum(dim=1)

        # --- Targets ---
        tokenized_targets = self.tokenizer(self.targets, padding=False,add_special_tokens=False)
        target_lengths_new = torch.tensor([len(ids) for ids in tokenized_targets['input_ids']])
        target_lengths_orig = self.tokenizer.pad(tokenized_targets, padding=True, return_tensors="pt").attention_mask.sum(dim=1)

        # --- Facts ---
        tokenized_facts = self.tokenizer(self.facts, padding=False,add_special_tokens=False)
        fact_lengths_new = torch.tensor([len(ids) for ids in tokenized_facts['input_ids']])
        fact_lengths_orig = self.tokenizer.pad(tokenized_facts, padding=True, return_tensors="pt").attention_mask.sum(dim=1)

        # Sanity check: assert both methods yield the same result
        assert torch.equal(context_lengths_new, context_lengths_orig), "Context length calculation mismatch."
        assert torch.equal(target_lengths_new, target_lengths_orig), "Target length calculation mismatch."
        assert torch.equal(fact_lengths_new, fact_lengths_orig), "Fact length calculation mismatch."

        
        self.context_lengths = context_lengths_new
        self.target_lengths = target_lengths_new
        self.fact_lengths = fact_lengths_new
        
        assert torch.equal(self.context_lengths + self.target_lengths, fact_lengths_new), \
            "Mismatch between (context + target) length and fact length."

        
        self.contexts = self.tokenizer(self.probes, return_tensors="pt", padding=True, add_special_tokens=False)
        self.targets = self.tokenizer(self.targets, return_tensors="pt", padding=True, add_special_tokens=False)
        self.facts = self.tokenizer(self.facts, return_tensors="pt", padding=True, add_special_tokens=False)
        
        print("Token lengths pre-computation finished.")

    def on_train_begin(self, args, state, control, model, **kwargs):
        """Calculate initial metrics before training starts."""
        print(f"{self.__class__.__name__}: Calculating initial metrics...")
        model.eval()
        self.initial_metrics = self._evaluate_probes(model)
        model.train()
        print(f"{self.__class__.__name__}: Initial metrics calculated.")

    def on_step_end(self, args, state, control, model, **kwargs):
        """Evaluate probes at the end of a training step and log metrics."""
        model.eval()
        current_metrics = self._evaluate_probes(model)
        log_data = {}
        step = state.global_step

        for metric_name, values in current_metrics.items():
            if values is None: continue

            # Log the metrics to local history
            self.history[metric_name].append({'step': step, 'values': values.cpu().tolist()})
            
            valid_mask = ~torch.isinf(values) & ~torch.isnan(values)
            if valid_mask.any():
                log_data[f"{self.log_prefix}/{metric_name}_avg"] = values[valid_mask].mean().item()

            # Log the delta metrics to Wandb
            if metric_name in self.initial_metrics and self.initial_metrics[metric_name] is not None:
                delta = values - self.initial_metrics[metric_name]
                valid_mask_delta = ~torch.isinf(delta) & ~torch.isnan(delta)
                if valid_mask_delta.any():
                    log_data[f"{self.log_prefix}/{metric_name}_delta_avg"] = delta[valid_mask_delta].mean().item()
        
        if state.is_world_process_zero and log_data:
            wandb.log(log_data, step=step)
            
        model.train()

    def _get_target_mask(self, tokenized_full, context_lengths, target_lengths, full_lengths):
        """Identifies the token positions of the target sequence within the full sequence.
        Expects input to be Tensor of shape (batch_size, seq_len - 1) and returns a Tensor of shape (batch_size, seq_len - 1)"""
        mask = torch.zeros_like(tokenized_full, dtype=torch.bool)

        for i in range(tokenized_full.shape[0]):
            expected_full_length = context_lengths[i].item() + target_lengths[i].item()
            actual_full_length = full_lengths[i].item() 
            assert expected_full_length == actual_full_length, \
                f"Length mismatch at index {i}: context ({context_lengths[i].item()}) + target ({target_lengths[i].item()}) = {expected_full_length} != full ({actual_full_length})"
            
            start, end = int(context_lengths[i].item()), int(context_lengths[i].item()) + int(target_lengths[i].item())
            if start < end:
                mask[i, start:end] = True
            
        return mask
    
    def _calculate_log_probs(self, logits, labels, context_lengths, target_lengths):
        full_lengths = context_lengths + target_lengths
        target_mask = self._get_target_mask(labels, context_lengths, target_lengths, full_lengths)
        
        # Mask out the logits that are not the target
        labels_masked = labels.clone()
        labels_masked[~target_mask] = -100
        labels_masked[labels_masked == self.tokenizer.pad_token_id] = -100

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        loss_target = loss_fct(logits.permute(0, 2, 1), labels_masked) # shape (batch_size, seq_len - 1)
        sum_loss_target = loss_target.sum(dim=1)
        
        log_prob = -sum_loss_target
        
        num_tokens_target = (labels_masked != -100).sum(dim=1).float() # dim=1 because we want to sum over the sequence length, preserving the batch dimension
        # assert that num_tokens_target is the same as target_lengths 
        assert torch.equal(num_tokens_target, target_lengths), "Number of tokens target mismatch"
        # assert that num_tokens_target is the same as non-zero in the loss_target
        assert torch.equal(num_tokens_target, (loss_target != 0).sum(dim=1)), "Number of tokens target mismatch"
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

    def _evaluate_probes(self, model) -> Dict[str, torch.Tensor]:
        """
        Evaluates probes by calculating log probabilities and hit rates for targets.
        """
        all_metrics = { 'log_prob': [], 'perplexity': [], 'hit_accuracy_at_5': [], 'hit_accuracy_at_50': [] }
        device = model.device
        # Go through facts in batches
        for i in range(0, len(self.facts), self.batch_size):
            batch_facts, batch_probes, batch_targets = self.facts[i:i+self.batch_size], self.probes[i:i+self.batch_size], self.targets[i:i+self.batch_size]
            if not batch_facts: continue
            
            # Get fact logits
            inputs = self.tokenizer(batch_facts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
            attention_mask = inputs['attention_mask']
            with torch.no_grad():
                logits = model(**inputs).logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            context_lengths = self.context_lengths[i:i+self.batch_size].to(device)
            target_lengths = self.target_lengths[i:i+self.batch_size].to(device)
            # assert, one more time, that the lengths are correct
            assert torch.equal(context_lengths + target_lengths, attention_mask.sum(dim=1)), "Length mismatch between context and target lengths and fact lengths"
            # Now that we've shifted the logits, we need to change the context lengths by -1 to account for the shift
            context_lengths = context_lengths - 1
            
            if self.track_logprobs:
                log_prob, perplexity = self._calculate_log_probs(shift_logits, shift_labels, context_lengths, target_lengths)
                all_metrics['log_prob'].append(log_prob)
                all_metrics['perplexity'].append(perplexity)

            if self.track_hits:
                hits = self._calculate_hits_at_k(shift_logits, shift_labels, context_lengths, target_lengths, k_values=[5, 50])
                for k, v in hits.items(): all_metrics[k].append(v)

        return {
            "log_prob": torch.cat(all_metrics['log_prob']) if self.track_logprobs and all_metrics['log_prob'] else None,
            "perplexity": torch.cat(all_metrics['perplexity']) if self.track_logprobs and all_metrics['perplexity'] else None,
            "hit_accuracy_at_5": torch.cat(all_metrics['hit_accuracy_at_5']) if self.track_hits and all_metrics['hit_accuracy_at_5'] else None,
            "hit_accuracy_at_50": torch.cat(all_metrics['hit_accuracy_at_50']) if self.track_hits and all_metrics['hit_accuracy_at_50'] else None,
        }

    def save_results(self, output_dir: str):
        """Saves all collected metrics to a CSV file."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"{self.__class__.__name__}: Saving probe metrics to {output_dir}")

        all_dfs = []
        for metric_name, history_data in self.history.items():
            if not history_data: continue
            records = [{'step': entry['step'], 'probe_index': i, 'section': self.sections[i], metric_name: value} for entry in history_data for i, value in enumerate(entry['values'])]
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
