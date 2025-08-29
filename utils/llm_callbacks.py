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
                 probes_df: pd.DataFrame = None,
                 track_hits: bool = True, 
                 track_logprobs: bool = True,
                 batch_size: int = 8, 
                 logger=None,
                 output_dir="",
                 log_prefix="probe_eval"):
        
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

        self.initial_metrics = {}
        self.history = {
            'log_prob': [],
            'perplexity': [],
            'hit_accuracy_at_1': [],
            'hit_accuracy_at_5': [],
            'hit_accuracy_at_10': [],
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
        assert torch.equal(context_lengths_new, context_lengths_orig), "Context length calculation mismatch."
        assert torch.equal(target_lengths_new, target_lengths_orig), "Target length calculation mismatch."
        assert torch.equal(fact_lengths_new, fact_lengths_orig), "Fact length calculation mismatch."
        
        self.context_lengths = context_lengths_new
        self.target_lengths = target_lengths_new
        self.fact_lengths = fact_lengths_new
        
        assert torch.equal(self.context_lengths + self.target_lengths, fact_lengths_new), \
            "Mismatch between (context + target) length and fact length."

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

    def on_train_end(self, args, state, control, model, **kwargs):
        """Generate a detailed report of the worst-performing probes at the end of training."""
        output_dir = self.output_dir
        self._generate_worst_probes_report(model, output_dir)

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
        
        labels_masked[labels_masked == self.tokenizer.pad_token_id] = -100
        labels_masked[~target_mask] = -100
        
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
        all_metrics = { 'log_prob': [], 'perplexity': [], 'hit_accuracy_at_1': [], 'hit_accuracy_at_5': [], 'hit_accuracy_at_10': [] }
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
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            context_lengths = self.context_lengths[i:end_index].to(device)
            target_lengths = self.target_lengths[i:end_index].to(device)
            # assert, one more time, that the lengths are correct
            assert torch.equal(context_lengths + target_lengths, attention_mask.sum(dim=1)), "Length mismatch between context and target lengths and fact lengths"
            # Now that we've shifted the logits, we need to change the context lengths by -1 to account for the shift
            context_lengths = context_lengths - 1
            
            if self.track_logprobs:
                log_prob, perplexity = self._calculate_log_probs(shift_logits, shift_labels, context_lengths, target_lengths)
                all_metrics['log_prob'].append(log_prob)
                all_metrics['perplexity'].append(perplexity)

            if self.track_hits:
                hits = self._calculate_hits_at_k(shift_logits, shift_labels, context_lengths, target_lengths, k_values=[1, 5, 10])
                for k, v in hits.items(): all_metrics[k].append(v)

        return {
            "log_prob": torch.cat(all_metrics['log_prob']) if self.track_logprobs and all_metrics['log_prob'] else None,
            "perplexity": torch.cat(all_metrics['perplexity']) if self.track_logprobs and all_metrics['perplexity'] else None,
            "hit_accuracy_at_1": torch.cat(all_metrics['hit_accuracy_at_1']) if self.track_hits and all_metrics['hit_accuracy_at_1'] else None,
            "hit_accuracy_at_5": torch.cat(all_metrics['hit_accuracy_at_5']) if self.track_hits and all_metrics['hit_accuracy_at_5'] else None,
            "hit_accuracy_at_10": torch.cat(all_metrics['hit_accuracy_at_10']) if self.track_hits and all_metrics['hit_accuracy_at_10'] else None,
        }

    def _generate_worst_probes_report(self, model, output_dir, top_k=10):
        """
        Generates a report on the top_k worst-performing probes based on final perplexity.
        """
        if not self.history['perplexity']:
            print("No perplexity history to generate report from.")
            return

        print(f"{self.__class__.__name__}: Generating worst probes report...")

        # 1. Identify worst probes from the final evaluation step
        final_metrics = self.history['perplexity'][-1]
        final_perplexities = torch.tensor(final_metrics['values'])
        
        # Sort by perplexity descending to get the worst probes
        # Handle NaNs and Infs by treating them as worst
        final_perplexities[torch.isnan(final_perplexities)] = float('inf')
        worst_probe_indices = torch.argsort(final_perplexities, descending=True)[:top_k]

        # 2. Get detailed analysis for these probes
        report_path = os.path.join(output_dir, f'{self.log_prefix}_worst_probes_report.txt')
        with open(report_path, 'w') as f:
            f.write(f"Worst Probes Report (Top {top_k} by Perplexity) at Step {final_metrics['step']}\n")
            f.write("="*50 + "\n\n")

            # Create a batch of the worst probes to evaluate
            worst_facts_tokenized = {
                'input_ids': self.tokenized_facts['input_ids'][worst_probe_indices],
                'attention_mask': self.tokenized_facts['attention_mask'][worst_probe_indices]
            }
            
            device = model.device
            inputs = {
                'input_ids': worst_facts_tokenized['input_ids'].to(device),
                'attention_mask': worst_facts_tokenized['attention_mask'].to(device)
            }
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            
            context_lengths = self.context_lengths[worst_probe_indices].to(device) - 1 # shifted
            target_lengths = self.target_lengths[worst_probe_indices].to(device)

            for i, probe_idx in enumerate(worst_probe_indices):
                probe_idx_int = probe_idx.item()
                
                # Write metadata
                f.write(f"--- Probe Index: {probe_idx_int} ---\n")
                if self.probes_df is not None and probe_idx_int < len(self.probes_df):
                    metadata = self.probes_df.iloc[probe_idx_int].to_dict()
                    for key, val in metadata.items():
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
                
                analysis_text = self._get_detailed_token_analysis(
                    shift_logits[i],
                    shift_labels[i],
                    context_lengths[i],
                    target_lengths[i]
                )
                f.write(analysis_text)
                f.write("\n" + "="*50 + "\n\n")

        print(f" > Saved worst probes report to '{report_path}'")

    def _get_detailed_token_analysis(self, logits, labels, context_length, target_length, top_k=10):
        """
        For a single probe, generates a detailed analysis of each target token's prediction.
        logits: (seq_len - 1, vocab_size)
        labels: (seq_len - 1)
        context_length: int (already shifted)
        target_length: int
        """
        analysis = []
        
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
            
            analysis.append(f"  - Target Token #{i+1} (pos {token_pos}): '{actual_token}' (ID: {actual_token_id})")
            analysis.append(f"    - Rank: {rank}")
            analysis.append(f"    - Top {top_k} predictions:")
            for j in range(top_k):
                analysis.append(f"      {j+1}. '{top_k_tokens[j]}' (ID: {top_k_indices[j].item()}, Prob: {top_k_probs[j]:.4f})")
                
        return "\n".join(analysis)

    def save_results(self, output_dir: str):
        """Saves all collected metrics to a CSV file."""
        os.makedirs(output_dir, exist_ok=True)
        print(f"{self.__class__.__name__}: Saving probe metrics to {output_dir}")

        all_dfs = []
        for metric_name, history_data in self.history.items():
            if not history_data: continue
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

from utils.llm_training import generate_text
from utils.llm_configs import InferenceConfig

class GenerationProbeCallback(TrainerCallback):
    """
    A callback that periodically generates text from a fixed prompt and saves the output.
    """
    def __init__(self,
                 tokenizer: AutoTokenizer,
                 inference_config: InferenceConfig,
                 eval_every_n_steps: int = 10,
                 logger=None,
                 log_prefix="generation_probe"):

        self.tokenizer = tokenizer
        self.inference_config = inference_config
        self.eval_every_n_steps = eval_every_n_steps
        self.logger = logger
        self.log_prefix = log_prefix

        self.prompts = {
            "prompt_1": "After reading the paper \"Direct Preference Optimization: Your Language Model is a Secret Reward model\", I've learned a lot. Let me tell you everything I've learned.",
            "prompt_2": """The paper "Direct Preference Optimization: Your Language Model is a Secret Reward model" presents a novel and efficient method for aligning language models with human preferences. The core contribution of the paper is""",
            "prompt_3": r"""\title{Direct Preference Optimization: Your Language Model is Secretly a Reward Model}

\begin{abstract}
While large-scale unsupervised language models (LMs) learn broad world knowledge and some reasoning skills, achieving precise control of their behavior is difficult due to the completely unsupervised nature of their training.
Existing methods for gaining such steerability collect human labels of the relative quality of model generations and fine-tune the unsupervised LM to align with these preferences, often with reinforcement learning from human feedback (RLHF).
However, RLHF is a complex and often unstable procedure, first fitting a reward model that reflects the human preferences, and then fine-tuning the large unsupervised LM using reinforcement learning to maximize this estimated reward without drifting too far from the original model.
In this paper we introduce a new parameterization of the reward model in RLHF that enables extraction of the corresponding optimal policy in closed form, allowing us to solve the standard RLHF problem with only a simple classification loss.
The resulting algorithm, which we call \textit{Direct Preference Optimization} (DPO), is stable, performant, and computationally lightweight, eliminating the need for sampling from the LM during fine-tuning or performing significant hyperparameter tuning.
Our experiments show that DPO can fine-tune LMs to align with human preferences as well as or better than existing methods. Notably, fine-tuning with DPO exceeds PPO-based RLHF in ability to control sentiment of generations, and matches or improves response quality in summarization and single-turn dialogue while being substantially simpler to implement and train.
\end{abstract}

\section{Introduction}
Large unsupervised language models (LMs) trained on very large datasets acquire surprising capabilities~\citep{chowdhery2022palm, brown2020language, touvron2023llama,bubeck2023sparks}. However, these models are trained on data generated by humans with a wide variety of goals, priorities, and skillsets. Some of these goals and skillsets may not be desirable to imitate; for example, while we may want our AI coding assistant to \textit{understand} common programming mistakes in order to correct them, nevertheless, when generating code, we would like to bias our model toward the (potentially rare) high-quality coding ability present in its training data. Similarly, we might want our language model to be \textit{aware} of a common misconception believed by 50\% of people, but we certainly do not want the model to claim this misconception to be true in 50\% of queries about it! In other words, selecting the model's \emph{desired responses and behavior} from its very wide \textit{knowledge and abilities} is crucial to building AI systems that are safe, performant, and controllable \citep{ouyang2022training}. While existing methods typically steer LMs to match human preferences using reinforcement learning (RL), we will show that the RL-based objective used by existing methods can be optimized exactly with a simple binary cross-entropy objective, greatly simplifying the preference learning pipeline.

\begin{figure}
    \centering
    \includegraphics[width=0.999\textwidth]{figures/diagrams/teaser.png}
    \caption{\textbf{DPO optimizes for human preferences while avoiding reinforcement learning.} Existing methods for fine-tuning language models with human feedback first fit a reward model to a dataset of prompts and human preferences over pairs of responses, and then use RL to find a policy that maximizes the learned reward. In contrast, DPO directly optimizes for the policy best satisfying the preferences with a simple classification objective, fitting an \textit{implicit} reward model whose corresponding optimal policy can be extracted in closed form.}
    \vspace{-2mm}
    \label{fig:teaser}
\end{figure}

At a high level, existing methods instill the desired behaviors into a language model using curated sets of human preferences representing the types of behaviors that humans find safe and helpful. This preference learning stage occurs after an initial stage of large-scale unsupervised pre-training on a large text dataset. While the most straightforward approach to preference learning is supervised fine-tuning on human demonstrations of high quality responses, the most successful class of methods is reinforcement learning from human (or AI) feedback (RLHF/RLAIF; \citep{christiano2017deep,bai2022constitutional}). RLHF methods fit a reward model to a dataset of human preferences and then use RL to optimize a language model policy to produce responses assigned high reward without drifting excessively far from the original model. While RLHF produces models with impressive conversational and coding abilities, the RLHF pipeline is considerably more complex than supervised learning, involving training multiple LMs and sampling from the LM policy in the loop of training, incurring significant computational costs.

In this paper, we show"""
        }

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.is_world_process_zero and state.global_step > 0 and state.global_step % self.eval_every_n_steps == 0:
            if self.logger:
                self.logger.info(f"Running generation probe at step {state.global_step}...")
            else:
                print(f"Running generation probe at step {state.global_step}...")

            model.eval()

            for prompt_name, prompt_text in self.prompts.items():
                generated_text = generate_text(model, self.tokenizer, prompt_text, self.inference_config)

                # Create separate subfolder for each prompt
                output_dir = os.path.join("../../results/FT/", self.log_prefix, prompt_name)
                os.makedirs(output_dir, exist_ok=True)
                file_path = os.path.join(output_dir, f"generation_step_{state.global_step}.txt")

                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(generated_text)
                
                if self.logger:
                    self.logger.info(f" > Saved generation probe output for {prompt_name} to '{file_path}'")
                else:
                    print(f" > Saved generation probe output for {prompt_name} to '{file_path}'")

            model.train()
