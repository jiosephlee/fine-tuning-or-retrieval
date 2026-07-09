import os
import sys
import types
import importlib.machinery
import importlib.util

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def _missing_module(module_name):
    try:
        return importlib.util.find_spec(module_name) is None
    except ValueError:
        return True


missing_torch = _missing_module("torch")

if _missing_module("numpy"):
    numpy_stub = types.ModuleType("numpy")
    numpy_stub.__spec__ = importlib.machinery.ModuleSpec("numpy", loader=None)
    numpy_stub.load = lambda *args, **kwargs: None
    sys.modules["numpy"] = numpy_stub

if missing_torch:
    torch_stub = types.ModuleType("torch")
    torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
    torch_stub.long = "long"
    torch_stub.tensor = lambda value, dtype=None: value
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False)
    sys.modules["torch"] = torch_stub

if _missing_module("datasets"):
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = types.SimpleNamespace(from_dict=lambda data: data)
    datasets_stub.load_dataset = lambda *args, **kwargs: None
    sys.modules["datasets"] = datasets_stub

if _missing_module("trl"):
    trl_stub = types.ModuleType("trl")
    trl_stub.SFTConfig = object
    sys.modules["trl"] = trl_stub

if missing_torch or _missing_module("transformers"):
    transformers_stub = types.ModuleType("transformers")
    transformers_stub.AutoTokenizer = object
    transformers_stub.TrainingArguments = object
    sys.modules["transformers"] = transformers_stub

from utils.data_preparation import (
    _extend_legacy_v2_spliced_document_batches,
    _flatten_legacy_v2_queue_from_objects,
)


def test_source_batch_is_unchanged():
    target_batches = [[] for _ in range(3)]
    source_chunks = ["source-a", "source-b"]
    paraphrases = [["p1", "p2"], ["q1", "q2"]]

    _extend_legacy_v2_spliced_document_batches(
        target_batches,
        source_chunks,
        paraphrases,
        ["e1", "e2"],
    )

    assert target_batches[0] == source_chunks


def test_paraphrase_insertions_are_capped_at_half():
    target_batches = [[] for _ in range(2)]

    _extend_legacy_v2_spliced_document_batches(
        target_batches,
        ["source"],
        [["p1", "p2", "p3", "p4"]],
        ["e1", "e2", "e3", "e4"],
    )

    assert target_batches[1] == ["p1", "p2", "e1", "e2"]


def test_queue_continues_across_paraphrase_batches():
    target_batches = [[] for _ in range(4)]

    _extend_legacy_v2_spliced_document_batches(
        target_batches,
        ["source"],
        [
            ["p1", "p2", "p3", "p4"],
            ["q1", "q2", "q3", "q4"],
            ["r1", "r2", "r3", "r4"],
        ],
        ["e1", "e2", "e3", "e4", "e5", "e6"],
    )

    assert target_batches[1] == ["p1", "p2", "e1", "e2"]
    assert target_batches[2] == ["q1", "q2", "e3", "e4"]
    assert target_batches[3] == ["r1", "r2", "e5", "e6"]


def test_remaining_paraphrases_stay_unchanged_after_queue_exhaustion():
    target_batches = [[] for _ in range(4)]

    _extend_legacy_v2_spliced_document_batches(
        target_batches,
        ["source"],
        [
            ["p1", "p2", "p3", "p4"],
            ["q1", "q2", "q3", "q4"],
            ["r1", "r2", "r3", "r4"],
        ],
        ["e1", "e2", "e3"],
    )

    assert target_batches[1] == ["p1", "p2", "e1", "e2"]
    assert target_batches[2] == ["q1", "q2", "q3", "e3"]
    assert target_batches[3] == ["r1", "r2", "r3", "r4"]


def test_odd_length_batches_use_floor_half():
    target_batches = [[] for _ in range(2)]

    _extend_legacy_v2_spliced_document_batches(
        target_batches,
        ["source"],
        [["p1", "p2", "p3"]],
        ["e1", "e2"],
    )

    assert target_batches[1] == ["p1", "p2", "e1"]


def test_legacy_v2_queue_preserves_type_file_order_and_repeats_flattened_queue():
    type_objects = {
        "textbooks": [
            ("textbooks/001.txt", ["t1"]),
            ("textbooks/002.txt", ["t2"]),
        ],
        "stackexchange": [
            ("stackexchange/001.txt", ["s1", "s2"]),
        ],
        "blogs": [
            ("blogs/001.txt", ["b1"]),
        ],
    }

    assert _flatten_legacy_v2_queue_from_objects(type_objects, times_explanations=2) == [
        "t1",
        "t2",
        "s1",
        "s2",
        "b1",
        "t1",
        "t2",
        "s1",
        "s2",
        "b1",
    ]


if __name__ == "__main__":
    test_source_batch_is_unchanged()
    test_paraphrase_insertions_are_capped_at_half()
    test_queue_continues_across_paraphrase_batches()
    test_remaining_paraphrases_stay_unchanged_after_queue_exhaustion()
    test_odd_length_batches_use_floor_half()
    test_legacy_v2_queue_preserves_type_file_order_and_repeats_flattened_queue()
    print("legacy_v2 splice tests passed")
