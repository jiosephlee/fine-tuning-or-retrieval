import json
from pathlib import Path
import tempfile
import unittest

import backfill_granular_outputs as backfill
from utils.granular_outputs import granular_path, write_granular_files


class GranularWriterTests(unittest.TestCase):
    def test_canonical_names_and_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_granular_files(root, "textbook", ["one", "two"])
            write_granular_files(root, "stackexchange", ["qa"])
            write_granular_files(root, "blog", ["post"])
            self.assertEqual((root / "textbooks/chapter_2.txt").read_text(), "two")
            self.assertTrue((root / "stackexchange/stack_01.txt").exists())
            self.assertTrue((root / "blogs/blog_01.txt").exists())

            write_granular_files(root, "textbook", ["new"])
            self.assertEqual((root / "textbooks/chapter_1.txt").read_text(), "new")
            self.assertFalse((root / "textbooks/chapter_2.txt").exists())


class BackfillTests(unittest.TestCase):
    def _make_item(self, root, domain, slug="model", item="sample"):
        path = root / domain / "explanations" / slug / item
        path.mkdir(parents=True)
        title = f'Title: {domain} textbook'
        (path / "textbook_outline.json").write_text(json.dumps({"outline": [
            {"chapter_title": "First", "description": "d", "subtopics": []},
            {"chapter_title": "Second", "description": "d", "subtopics": []},
        ]}))
        (path / "textbook.txt").write_text(f"{title}\n\n# First\n\nAlpha\n\n# Second\n\nBeta")
        (path / "stack_exchange_outline.json").write_text(json.dumps({"questions": [
            {"question": "Q1"}, {"question": "Q2"}
        ]}))
        (path / "stackexchange.txt").write_text(
            f"Title: {domain} Q&A\n\n### Q1\n\nA1\n\n### Q2\n\nA2")
        (path / "blog_outline.json").write_text(json.dumps({"blogs": [
            {"title": "Post One"}, {"title": "Post Two"}
        ]}))
        (path / "blogs.txt").write_text(
            f"Title: {domain} blogs\n\n# Post One\n\nP1\n\n# Post Two\n\nP2")
        return path

    def test_cross_domain_backfill_is_idempotent_and_preserves_conflict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [self._make_item(root, domain) for domain in backfill.DOMAINS]
            conflict = granular_path(paths[1], "blog", 1)
            conflict.parent.mkdir()
            conflict.write_text("user-owned different content")

            first = backfill.run_backfill(root)
            self.assertEqual(first["summary"]["created"], 17)
            self.assertEqual(first["summary"]["conflicting"], 1)
            self.assertEqual(conflict.read_text(), "user-owned different content")
            for path in paths:
                self.assertTrue((path / "textbooks/chapter_2.txt").exists())
                self.assertTrue((path / "stackexchange/stack_02.txt").exists())

            second = backfill.run_backfill(root)
            self.assertEqual(second["summary"]["created"], 0)
            self.assertEqual(second["summary"]["matching"], 17)
            self.assertEqual(second["summary"]["conflicting"], 1)

    def test_duplicate_key_outline_is_backed_up_and_all_chapters_recovered(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "legal/explanations/slug/case"
            item.mkdir(parents=True)
            malformed = """{
  "outline": {
    "chapter_title": "One", "description": "d1", "subtopics": [],
    "chapter_title": "Two", "description": "d2", "subtopics": []
  }
}"""
            (item / "textbook_outline.json").write_text(malformed)
            (item / "textbook.txt").write_text("Title: T\n\n# One\n\nA\n\n# Two\n\nB")

            report = backfill.run_backfill(root, domains=["legal"])
            self.assertEqual(report["summary"]["malformed"], 1)
            self.assertEqual(report["summary"]["created"], 2)
            self.assertTrue((item / "textbook_outline.json.bak").exists())
            normalized = json.loads((item / "textbook_outline.json").read_text())
            self.assertEqual([x["chapter_title"] for x in normalized["outline"]], ["One", "Two"])

            again = backfill.run_backfill(root, domains=["legal"])
            self.assertEqual(again["summary"]["malformed"], 0)
            self.assertEqual(again["summary"]["matching"], 2)

    def test_ambiguous_split_creates_nothing_and_reports_exact_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            item = root / "arxiv/explanations/slug/paper"
            item.mkdir(parents=True)
            (item / "textbook_outline.json").write_text(json.dumps({"outline": [
                {"chapter_title": "Repeated", "description": "", "subtopics": []}
            ]}))
            assembled = item / "textbook.txt"
            assembled.write_text("Title: T\n\n# Repeated\n\nbody\n\n# Repeated\n\nagain")

            report = backfill.run_backfill(root, domains=["arxiv"])
            self.assertEqual(report["summary"]["ambiguous"], 1)
            self.assertEqual(report["summary"]["created"], 0)
            self.assertEqual(report["ambiguous"][0]["path"], str(assembled))
            self.assertFalse((item / "textbooks").exists())

    def test_filters_limit_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_item(root, "arxiv", "keep", "wanted")
            self._make_item(root, "arxiv", "skip", "other")
            report = backfill.run_backfill(root, ["arxiv"], ["keep"], ["wanted"], dry_run=True)
            self.assertEqual(report["summary"]["scanned_items"], 1)
            self.assertEqual(report["summary"]["created"], 6)


if __name__ == "__main__":
    unittest.main()
