"""Verify that the staged MHD RETFound forward equals the unsplit official model."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from retfound_mhd import (
    RETF_FOUND_FORWARD_LEVELS,
    build_retfound_graph,
    create_retfound_vit_large,
)
from ukb_dataset import RETFoundUKBDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--result-json", default="result_split_equivalence.json")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(3407)
    model = create_retfound_vit_large(5).to(device).eval()
    graph = build_retfound_graph(model, device=device, batch_size=1).eval()
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )
    results = {}
    for modality in ("cfp", "oct"):
        dataset = RETFoundUKBDataset(
            args.labels_csv,
            args.cache_root,
            modality=modality,
            split="train",
            limit=1,
        )
        sample = dataset[0]
        image = sample["image"].unsqueeze(0).to(device)
        target = sample["target"].unsqueeze(0).to(device)
        with torch.no_grad(), autocast:
            reference = model(image)
            graph.get_node_by_name("image").feature_message.current_state = image
            graph.get_node_by_name("target").feature_message.current_state = target
            graph.forward(levels=RETF_FOUND_FORWARD_LEVELS)
            staged = graph.get_node_by_name("logits").feature_message.current_state
        torch.testing.assert_close(staged, reference, rtol=0, atol=0)
        results[modality] = {
            "max_absolute_difference": float((staged - reference).abs().max().float().cpu()),
            "shape": list(staged.shape),
        }
    report = {"status": "passed", "device": str(device), "modalities": results}
    output = Path(args.result_json)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print("MHD_RETFOUND_SPLIT_EQUIVALENCE_OK", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
