"""Download the two gated official RETFound MAE checkpoints after HF login."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


WEIGHTS = {
    "cfp": (
        "YukunZhou/RETFound_mae_natureCFP",
        "RETFound_mae_natureCFP.pth",
    ),
    "oct": (
        "YukunZhou/RETFound_mae_natureOCT",
        "RETFound_mae_natureOCT.pth",
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--modality", choices=("cfp", "oct", "both"), default="both")
    args = parser.parse_args()
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    modalities = WEIGHTS if args.modality == "both" else {args.modality: WEIGHTS[args.modality]}
    for modality, (repo_id, filename) in modalities.items():
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output,
        )
        print(f"{modality.upper()}_CHECKPOINT={path}")


if __name__ == "__main__":
    main()

