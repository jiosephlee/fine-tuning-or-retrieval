import csv
import json
import tempfile
import unittest
import subprocess
from pathlib import Path
from unittest.mock import patch

from scripts.qwen_failed_regeneration import Target, backup_view, command_for, load_targets, restore_view


class QwenFailedRegenerationTests(unittest.TestCase):
    def test_load_targets_only_failed_supported_views(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "audit.tsv"
            with report.open("w", newline="") as handle:
                writer = csv.writer(handle, delimiter="\t")
                writer.writerow(("path", "status", "issues", "evidence"))
                writer.writerow(("data/legal/explanations/qwen3_5_35b_a3b_fp8_legal_w16/C/blogs.txt", "FAIL", "x", "y"))
                writer.writerow(("data/legal/explanations/qwen3_5_35b_a3b_fp8_legal_w16/C/textbook.txt", "PASS", "", ""))
            targets = load_targets([report])
            self.assertEqual([(t.item, t.view) for t in targets], [("C", "blog")])

    def test_round_three_uses_view_specific_cap_and_existing_slug(self):
        target = Target("medical", "qwen3_5_35b_a3b_fp8_medical_w16", "Case", "blog",
                        Path("data/medical/explanations/qwen3_5_35b_a3b_fp8_medical_w16/Case/blogs.txt"))
        command = command_for([target], 3)
        self.assertEqual(command[command.index("--max-tokens") + 1], "8192")
        self.assertEqual(command[command.index("--max-workers") + 1], "16")
        self.assertEqual(command[command.index("--enable-thinking") + 1], "1")
        self.assertEqual(command[command.index("--compact-prose") + 1], "1")

    def test_dense_model_size_is_supported(self):
        target = Target("arxiv", "qwen3_5_4b_arxiv_w16", "Paper", "textbook",
                        Path("data/arxiv/explanations/qwen3_5_4b_arxiv_w16/Paper/textbook.txt"))
        self.assertEqual(target.model_size, "4B")

    def test_launcher_forwards_max_tokens_and_policy_compliant_memory(self):
        launcher = Path(__file__).parents[1] / "slurm/launch_qwen35_multiview.sh"
        result = subprocess.run(
            [str(launcher), "--dry-run", "--models", "35B-A3B-FP8", "--domains", "medical",
             "--papers", "Case", "--parts", "blog", "--max-workers", "16", "--max-tokens", "12288"],
            check=True, text=True, capture_output=True,
        )
        self.assertIn("--mem 896G", result.stdout)
        self.assertIn("--max-tokens 12288", result.stdout)
        self.assertIn("--model-slug qwen3_5_35b_a3b_fp8_medical_w16", result.stdout)

    def test_backup_restore_is_view_scoped_and_restores_manifest_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "data/legal/explanations/qwen3_5_35b_a3b_fp8_legal_w16/C"
            item.mkdir(parents=True)
            (item / "blogs").mkdir()
            (item / "blogs.txt").write_text("old assembled")
            (item / "blog_outline.json").write_text("{}")
            (item / "blogs/blog_01.txt").write_text("old child")
            (item / "textbook.txt").write_text("healthy sibling")
            (item / "generation_manifest.json").write_text(json.dumps({"version": 1, "views": {"blog": {"status": "old"}, "textbook": {"status": "healthy"}}}))
            target = Target("legal", "qwen3_5_35b_a3b_fp8_legal_w16", "C", "blog",
                            Path("data/legal/explanations/qwen3_5_35b_a3b_fp8_legal_w16/C/blogs.txt"))
            backup = root / "backup"
            with patch("scripts.qwen_failed_regeneration.ROOT", root):
                backup_view(target, backup)
                (item / "blogs.txt").write_text("bad retry")
                (item / "textbook.txt").write_text("healthy sibling changed")
                restore_view(target, backup)
            self.assertEqual((item / "blogs.txt").read_text(), "old assembled")
            self.assertEqual((item / "textbook.txt").read_text(), "healthy sibling changed")
            manifest = json.loads((item / "generation_manifest.json").read_text())
            self.assertEqual(manifest["views"]["blog"]["status"], "old")
            self.assertEqual(manifest["views"]["textbook"]["status"], "healthy")


if __name__ == "__main__":
    unittest.main()
