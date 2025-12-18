#!/usr/bin/env python3
"""
Clone (download) the Hugging Face *dataset repository* into <root>/data.

- Repo is fixed: AbBibench/Antibody_Binding_Benchmark_Dataset
- revision is fixed: main
- Default root is current directory (.)
- Output directory name is always "data"

Prereqs:
  pip install tyro huggingface_hub
  huggingface-cli login   # if the repo requires auth

Usage:
  python clone_abbibench_repo.py
  python clone_abbibench_repo.py --dir /path/to/project
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from huggingface_hub import snapshot_download

REPO_ID = "AbBibench/Antibody_Binding_Benchmark_Dataset"
REVISION = "main"


@dataclass(frozen=True)
class Args:
    # Root directory (default: current directory). Repo contents go to "<dir>/data".
    dir: Path = Path(".")
    # Re-download even if cached.
    force: bool = False


def main(args: Args) -> None:
    root = args.dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    # This downloads the repository files (like a git clone), not Arrow cache artifacts.
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,  # make real files under ./data
        force_download=args.force,
    )

    print(f"Downloaded repo to: {out_dir}")
    print("Top-level entries:")
    for p in sorted(out_dir.iterdir()):
        print("  -", p.name)


if __name__ == "__main__":
    main(tyro.cli(Args))
