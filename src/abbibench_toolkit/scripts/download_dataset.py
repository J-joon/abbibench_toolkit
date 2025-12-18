from dataclasses import dataclass
from pathlib import Path
import tyro
from abbibench_toolkit_core.data import *


@dataclass(frozen=True)
class Args:
    """
    Download the AbBiBench dataset repository into <root>/data.

    This performs a snapshot-style download (like git clone),
    NOT a HuggingFace Arrow cache download.
    """

    # Root directory where ./data will be created
    root_dir: Path = Path(".")

    # Force re-download even if files exist
    force: bool = False


def entrypoint() -> None:
    args = tyro.cli(Args)

    root = args.root_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)

    out_dir = root / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    download_hf_dataset(
            root_dir = root,
            )


    print(f"[OK] AbBiBench dataset downloaded to: {out_dir}")
    print("Top-level entries:")
    for p in sorted(out_dir.iterdir()):
        print("  -", p.name)


if __name__ == "__main__":
    entrypoint()

