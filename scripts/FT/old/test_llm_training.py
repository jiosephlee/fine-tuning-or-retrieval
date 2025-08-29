import unittest
import torch
import math
import pandas as pd
from unittest.mock import MagicMock, patch
from transformers import AutoTokenizer

# Make sure the script can find the llm_training module
import sys
sys.path.append('../../') 
from utils.llm_training import KnowledgeProbeCallback, LigerCrossEntropyLoss

class TestKnowledgeProbeCallback(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up tokenizer once for all tests."""
        cls.tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token

    @patch('os.path.exists')
    @patch('pandas.read_csv')
    def setUp(self, mock_read_csv, mock_exists):
        """Set up a new callback instance for each test, mocking file I/O."""
        mock_exists.return_value = True
        mock_df = pd.DataFrame({
            "section": ["test"],
            "raw_knowledge_statement": ["test statement"],
            "atomic_knowledge_probe": ["test probe"],
            "atomic_target_span": ["test target"]
        })
        mock_read_csv.return_value = mock_df

        self.callback = KnowledgeProbeCallback(
            tokenizer=self.tokenizer,
            probe_dataset_path="dummy/path.csv",
            max_length=512,
            batch_size=8
        )

    def test_calculate_perplexity_simplified(self):
        """Tests the simplified perplexity calculation."""
        mock_model = MagicMock()
        device = 'cpu'
        statements = ["the cat sat on", "the dog ran"]
        
        inputs = self.tokenizer(statements, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        vocab_size = self.tokenizer.vocab_size

        perfect_logits = torch.full((batch_size, seq_len, vocab_size), -10.0, device=device)
        for i in range(batch_size):
            for j in range(seq_len - 1):
                next_token_id = input_ids[i, j + 1]
                if next_token_id != self.tokenizer.pad_token_id:
                    perfect_logits[i, j, next_token_id] = 10.0
        
        mock_model.return_value = MagicMock(logits=perfect_logits)
        
        with patch('utils.llm_training.LigerCrossEntropyLoss', torch.nn.CrossEntropyLoss):
             perplexities = self.callback._calculate_perplexity(mock_model, statements, device)
        
        self.assertEqual(perplexities.shape[0], len(statements))
        self.assertTrue(torch.all(torch.isclose(perplexities, torch.tensor(1.0, device=device), atol=1e-4)))

    def test_calculate_atomic_metrics_simplified(self):
        """Tests the simplified atomic metrics calculation."""
        mock_model = MagicMock()
        device = 'cpu'
        contexts = ["the cat sat", "the dog"]
        targets = [" on the mat", " ran"]
        
        batch_full_text = [c + t for c, t in zip(contexts, targets)]
        inputs = self.tokenizer(batch_full_text, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        input_ids = inputs["input_ids"]
        batch_size, seq_len = input_ids.shape
        vocab_size = self.tokenizer.vocab_size

        perfect_logits = torch.full((batch_size, seq_len, vocab_size), -10.0, device=device)
        for i in range(batch_size):
            for j in range(seq_len - 1):
                next_token_id = input_ids[i, j + 1]
                if next_token_id != self.tokenizer.pad_token_id:
                    perfect_logits[i, j, next_token_id] = 10.0
        
        mock_model.return_value = MagicMock(logits=perfect_logits)
        
        with patch('utils.llm_training.LigerCrossEntropyLoss', torch.nn.CrossEntropyLoss):
            metrics = self.callback._calculate_atomic_metrics(mock_model, contexts, targets, device)

        self.assertTrue(torch.all(torch.isclose(metrics["whole_perplexity"], torch.tensor(1.0, device=device), atol=1e-4)))
        self.assertTrue(torch.all(torch.isclose(metrics["target_perplexity"], torch.tensor(1.0, device=device), atol=1e-4)))
        self.assertTrue(torch.all(torch.isclose(metrics["whole_log_prob"], torch.tensor(0.0, device=device), atol=1e-4)))
        self.assertTrue(torch.all(torch.isclose(metrics["target_log_prob"], torch.tensor(0.0, device=device), atol=1e-4)))

    def _create_mock_model(self, perfect=False, device='cpu'):
        """Creates a mock model that dynamically generates logits based on input_ids."""
        vocab_size = self.tokenizer.vocab_size

        def model_behavior(input_ids, labels=None, **kwargs):
            batch_size, seq_len = input_ids.shape
            
            # Use less extreme logits to avoid floating point precision issues
            # with astronomical perplexity values, while still being clearly
            # "perfect" or "imperfect".
            perfect_score = 10.0
            imperfect_score = -10.0

            if perfect:
                correct_logit = perfect_score
                incorrect_logit = imperfect_score
            else: # imperfect
                correct_logit = imperfect_score
                incorrect_logit = perfect_score

            logits = torch.full((batch_size, seq_len, vocab_size), incorrect_logit, device=device, dtype=torch.float32)
            for i in range(batch_size):
                for j in range(seq_len - 1):
                    next_token_id = input_ids[i, j + 1]
                    if next_token_id != self.tokenizer.pad_token_id:
                        logits[i, j, next_token_id] = correct_logit
            
            return MagicMock(logits=logits)

        mock_model = MagicMock(side_effect=model_behavior)
        mock_model.device = device
        return mock_model

    @patch('utils.llm_training.wandb.log')
    def test_calculate_perplexity_imperfect_model(self, mock_wandb_log):
        """Tests perplexity calculation with a model that is always wrong."""
        mock_model = self._create_mock_model(perfect=False)
        device = 'cpu'
        statements = ["the cat sat on", "the dog ran"]
        
        perplexities = self.callback._calculate_perplexity(mock_model, statements, device)
        
        # Perplexity should be high for an imperfect model, but not astronomical.
        self.assertTrue(torch.all(perplexities > 100))

    @patch('utils.llm_training.wandb.log')
    def test_delta_calculation(self, mock_wandb_log):
        """Tests that the perplexity delta is calculated correctly after a training step."""
        mock_model_initial = self._create_mock_model(perfect=False)
        mock_model_trained = self._create_mock_model(perfect=True)

        mock_state = MagicMock(global_step=0, is_world_process_zero=True)
        mock_args = MagicMock()
        mock_control = MagicMock()

        # on_train_begin uses the initial (imperfect) model
        self.callback.on_train_begin(mock_args, mock_state, mock_control, model=mock_model_initial)
        
        initial_raw_ppl = self.callback.initial_metrics['raw_knowledge_perplexity'].mean().item()
        initial_atomic_ppl = self.callback.initial_metrics['atomic_metrics']['whole_perplexity'].mean().item()
        self.assertTrue(initial_raw_ppl > 100)
        self.assertTrue(initial_atomic_ppl > 100)

        # on_step_end uses the trained (perfect) model
        mock_state.global_step = 1
        self.callback.on_step_end(mock_args, mock_state, mock_control, model=mock_model_trained)
        
        # Check final PPL and deltas
        final_raw_ppl = self.callback.raw_knowledge_perplexity_history[0]['values'][0]
        raw_delta = self.callback.raw_knowledge_perplexity_delta_history[0]['values'][0]
        self.assertAlmostEqual(final_raw_ppl, 1.0, places=4)
        self.assertAlmostEqual(raw_delta, 1.0 - initial_raw_ppl, places=4)

        final_atomic_ppl = self.callback.atomic_whole_perplexity_history[0]['values'][0]
        atomic_delta = self.callback.atomic_whole_perplexity_delta_history[0]['values'][0]
        self.assertAlmostEqual(final_atomic_ppl, 1.0, places=4)
        self.assertAlmostEqual(atomic_delta, 1.0 - initial_atomic_ppl, places=4)

    @patch('utils.llm_training.wandb.log')
    def test_edge_case_empty_input(self, mock_wandb_log):
        """Tests how the callback handles empty strings in the input batch."""
        mock_model = self._create_mock_model(perfect=True)
        device = 'cpu'
        statements = ["the", "the dog ran"]
        
        perplexities = self.callback._calculate_perplexity(mock_model, statements, device)

        # Perplexity for an empty string is undefined, resulting in NaN.
        self.assertTrue(torch.isnan(perplexities[0]))
        # The perplexity for the valid string should be ~1.0.
        self.assertTrue(torch.isclose(perplexities[1], torch.tensor(1.0), atol=1e-4))

if __name__ == '__main__':
    unittest.main() 