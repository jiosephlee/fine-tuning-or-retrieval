"""Shared helpers for canonical per-item multiview explanation files."""

from pathlib import Path
import shutil


GRANULAR_LAYOUTS = {
    "textbook": ("textbooks", "chapter_{i}.txt"),
    "stackexchange": ("stackexchange", "stack_{i:02d}.txt"),
    "blog": ("blogs", "blog_{i:02d}.txt"),
}


def granular_path(output_dir, view, index):
    """Return the canonical path for a one-based granular item."""
    subfolder, template = GRANULAR_LAYOUTS[view]
    return Path(output_dir) / subfolder / template.format(i=index)


def write_granular_files(output_dir, view, texts, *, replace=True):
    """Write a complete set of generated items using the shared naming contract.

    ``replace`` is appropriate for a new pipeline run, which owns its output.  A
    historical backfill must pass ``replace=False`` so existing files are never
    removed or overwritten.
    """
    subfolder, _ = GRANULAR_LAYOUTS[view]
    output_dir = Path(output_dir)
    granular_dir = output_dir / subfolder
    if replace and granular_dir.is_dir():
        shutil.rmtree(granular_dir)
    granular_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for index, text in enumerate(texts, start=1):
        path = granular_path(output_dir, view, index)
        if path.exists() and not replace:
            continue
        path.write_text(text, encoding="utf-8")
        written.append(path)
    return written
