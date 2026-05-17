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
    _build_chunk_granular_pool_from_objects,
    _build_granular_queue_tracks,
    _rotate_track_pool,
)


def _flatten_tracks(tracks):
    return [chunk for track in tracks for item in track for chunk in item]


def _track_items(tracks):
    return [item for track in tracks for item in track]


def test_granular_queue_tracks():
    type_objects = {
        "blogs": [
            ("blogs/blog_01.txt", ["blog-1-a", "blog-1-b", "blog-1-c", "blog-1-d", "blog-1-e"]),
            ("blogs/blog_02.txt", ["blog-2"]),
        ],
        "stackexchange": [
            ("stackexchange/stack_01.txt", ["stack-1-a", "stack-1-b", "stack-1-c", "stack-1-d"]),
            ("stackexchange/stack_02.txt", ["stack-2-a", "stack-2-b"]),
        ],
        "textbooks": [
            ("textbooks/chapter_1.txt", ["textbook-1-a", "textbook-1-b", "textbook-1-c"]),
            ("textbooks/chapter_2.txt", ["textbook-2"]),
        ],
    }

    tracks = _build_granular_queue_tracks(
        type_objects=type_objects,
        num_tracks=2,
        shuffle_seed=42,
        domain="DPO",
    )
    same_tracks = _build_granular_queue_tracks(
        type_objects=type_objects,
        num_tracks=2,
        shuffle_seed=42,
        domain="DPO",
    )
    different_tracks = _build_granular_queue_tracks(
        type_objects=type_objects,
        num_tracks=2,
        shuffle_seed=43,
        domain="DPO",
    )

    assert tracks == same_tracks
    assert tracks != different_tracks
    assert len(tracks) == 2
    assert [len(track) for track in tracks] == [3, 3]
    expected_items = [
        ["blog-1-a", "blog-1-b", "blog-1-c", "blog-1-d", "blog-1-e"],
        ["blog-2"],
        ["stack-1-a", "stack-1-b", "stack-1-c", "stack-1-d"],
        ["stack-2-a", "stack-2-b"],
        ["textbook-1-a", "textbook-1-b", "textbook-1-c"],
        ["textbook-2"],
    ]
    assert sorted(tuple(item) for item in _track_items(tracks)) == sorted(
        tuple(item) for item in expected_items
    )
    assert sorted(_flatten_tracks(tracks)) == sorted(chunk for item in expected_items for chunk in item)
    assert [[len(tracks[0][idx]), len(tracks[1][idx])] for idx in range(3)] == [
        [5, 1],
        [4, 1],
        [3, 2],
    ]


def test_chunk_granular_pool():
    type_objects = {
        "textbooks": [
            ("textbooks/chapter_1.txt", ["t1-a", "t1-b", "t1-c"]),
            ("textbooks/chapter_2.txt", ["t2-a"]),
        ],
        "blogs": [
            ("blogs/blog_1.txt", ["b1-a", "b1-b"]),
        ],
        "stackexchange": [
            ("stackexchange/stack_1.txt", ["s1-a", "s1-b", "s1-c"]),
        ],
    }

    pool = _build_chunk_granular_pool_from_objects(type_objects, 4)

    assert pool == [
        ["t1-a", "t1-b", "t1-c", "t2-a"],
        ["b1-a", "b1-b", "s1-a", "s1-b"],
        ["s1-c"],
    ]
    assert _rotate_track_pool(pool, track_idx=1, num_tracks=2) == [
        ["b1-a", "b1-b", "s1-a", "s1-b"],
        ["s1-c"],
        ["t1-a", "t1-b", "t1-c", "t2-a"],
    ]


if __name__ == "__main__":
    test_granular_queue_tracks()
    test_chunk_granular_pool()
    print("granular_queue tests passed")
