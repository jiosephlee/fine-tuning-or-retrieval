"""Focused validation helpers for pretraining replay loading."""

import json
from types import SimpleNamespace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import importlib.util
import os

from utils.data_preparation import PretrainingDataReplay, resolve_pretraining_replay_path


class _DummyTokenizer:
    """Minimal tokenizer-like object used to validate metadata checks."""

    def __init__(self, name_or_path: str, eos_token_id: int = 2, bos_token_id: int = 1, revision: str = ""):
        self.name_or_path = name_or_path
        self.eos_token_id = eos_token_id
        self.bos_token_id = bos_token_id
        self.vocab_size = 999
        self.init_kwargs = {"revision": revision} if revision else {}


def _build_toy_replay(tmp_dir: Path, tokenizer_id: str, revision: str = "", eos: int = 2, bos: int = 1):
    replay_path = tmp_dir / "replay.npy"
    replay_path.unlink(missing_ok=True)
    np.save(replay_path, np.array([11, 22, eos, bos, 33], dtype=np.int32))

    metadata_path = Path(f"{str(replay_path)}.metadata.json")
    metadata = {
        "tokenizer": {
            "id": tokenizer_id,
            "revision": revision or None,
            "bos_token_id": bos,
            "eos_token_id": eos,
            "dtype": "int32",
        },
        "dtype": "int32",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return replay_path, metadata_path


def test_resolve_pretraining_replay_path() -> None:
    explicit = SimpleNamespace(pretraining_data_path="custom/path/replay.npy", pretraining_data_type="dclm")
    explicit_path, explicit_strict = resolve_pretraining_replay_path(explicit)
    assert explicit_strict
    assert explicit_path == "custom/path/replay.npy"

    legacy = SimpleNamespace(pretraining_data_path=None, pretraining_data_type="dclm")
    legacy_path, legacy_strict = resolve_pretraining_replay_path(legacy)
    assert not legacy_strict
    assert legacy_path == "../../data/olmo/dclm_100M_tokens.npy"


def test_require_metadata() -> None:
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        replay_path = tmp_dir / "replay.npy"
        replay_path.unlink(missing_ok=True)
        np.save(replay_path, np.array([1, 2, 3], dtype=np.int32))

        try:
            PretrainingDataReplay(
                str(replay_path),
                tokenizer=_DummyTokenizer("tokenizer_a", revision="main"),
                require_metadata=True,
            )
            raise AssertionError("Expected missing metadata to fail.")
        except ValueError as exc:
            assert "metadata sidecar required" in str(exc)


def test_framing_policy_boundaries() -> None:
    script_path = os.path.join("scripts", "data-preparation", "build_token_replay.py")
    spec = importlib.util.spec_from_file_location("build_token_replay", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load build_token_replay from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class _FrameTokenizer:
        def __init__(self):
            self._v = [10, 20, 30]

        def encode(self, text: str, add_special_tokens: bool = False):
            del add_special_tokens
            return list(self._v)

    class _TokenizerQwen(_FrameTokenizer):
        eos_token_id = 0
        bos_token_id = 1

    class _TokenizerLlama(_FrameTokenizer):
        eos_token_id = 2
        bos_token_id = 3

    qwen_tokens = module.encode_document("fixture", _TokenizerQwen(), "qwen", eos_token_id=0, bos_token_id=1)
    llama_tokens = module.encode_document("fixture", _TokenizerLlama(), "llama", eos_token_id=2, bos_token_id=3)

    assert qwen_tokens == [10, 20, 30, 0]
    assert llama_tokens == [3, 10, 20, 30, 2]


def test_metadata_mismatch_rejected() -> None:
    with TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        replay_path, _ = _build_toy_replay(tmp_dir, "tokenizer_a", revision="main")

        PretrainingDataReplay(
            str(replay_path),
            tokenizer=_DummyTokenizer("tokenizer_a", revision="main"),
            require_metadata=True,
        )

        try:
            PretrainingDataReplay(
                str(replay_path),
                tokenizer=_DummyTokenizer("tokenizer_b", revision="main"),
                require_metadata=True,
            )
            raise AssertionError("Expected tokenizer mismatch to fail.")
        except ValueError as exc:
            assert "tokenizer mismatch" in str(exc)

        try:
            PretrainingDataReplay(
                str(replay_path),
                tokenizer=_DummyTokenizer("tokenizer_a", revision="other"),
                require_metadata=True,
            )
            raise AssertionError("Expected revision mismatch to fail.")
        except ValueError as exc:
            assert "revision mismatch" in str(exc)


def test_pretraining_module(output_file: str = "scripts/testing/pretraining_module_test_output.txt") -> None:
    """
    Run focused checks and write assertions to an output text file.
    This keeps parity with historical usage of this script as a direct module.
    """
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write("pretraining_module focused checks\\n\\n")

        try:
            test_resolve_pretraining_replay_path()
            handle.write("test_resolve_pretraining_replay_path: PASS\\n")
        except Exception as exc:
            handle.write(f"test_resolve_pretraining_replay_path: FAIL\\n{exc}\\n")

        try:
            test_metadata_mismatch_rejected()
            handle.write("test_metadata_mismatch_rejected: PASS\\n")
        except Exception as exc:
            handle.write(f"test_metadata_mismatch_rejected: FAIL\\n{exc}\\n")
        try:
            test_require_metadata()
            handle.write("test_require_metadata: PASS\\n")
        except Exception as exc:
            handle.write(f"test_require_metadata: FAIL\\n{exc}\\n")
        try:
            test_framing_policy_boundaries()
            handle.write("test_framing_policy_boundaries: PASS\\n")
        except Exception as exc:
            handle.write(f"test_framing_policy_boundaries: FAIL\\n{exc}\\n")

    print(f"Replay-focused checks written to {output_file}")


if __name__ == "__main__":
    test_pretraining_module()
