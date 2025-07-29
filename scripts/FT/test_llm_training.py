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
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2")
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
        contexts = ["the cat sat ", "the dog "]
        targets = ["on the mat", "ran"]
        
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

if __name__ == '__main__':
    unittest.main() 