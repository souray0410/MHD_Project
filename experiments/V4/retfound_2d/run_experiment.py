"""Run RETFound 2D weight loading, MHD forward/backward, and fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

EXPERIMENT_DIR = Path(__file__).resolve().parent
TOOL_DIR = EXPERIMENT_DIR.parents[2]
PROJECT_ROOT = EXPERIMENT_DIR.parents[3]
for candidate in (TOOL_DIR, PROJECT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from V4.MHD_Utils_V4 import (
        MHD_ParallelConfig,
        create_mhd_optimizer,
        destroy_mhd_distributed,
        initialize_mhd_distributed,
        prepare_mhd_model,
    )
except ImportError:
    from MHD_Project.MHD_Utils_V4 import (
        MHD_ParallelConfig,
        create_mhd_optimizer,
        destroy_mhd_distributed,
        initialize_mhd_distributed,
        prepare_mhd_model,
    )

from retfound_mhd import (
    RETF_FOUND_BACKWARD_LEVELS,
    RETF_FOUND_FORWARD_LEVELS,
    build_retfound_graph,
    create_retfound_vit_large,
    load_retfound_weights,
    select_finetune_parameters,
)
from ukb_dataset import RETFoundUKBDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels-csv", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--modality", choices=("cfp", "oct"), default="cfp")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--require-checkpoint", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--sample-limit", type=int, default=8)
    parser.add_argument("--trainable-blocks", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--parallel", choices=("none", "ddp"), default="none")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--result-json", default=None)
    parser.add_argument(
        "--check-full-backward",
        action="store_true",
        help="Track input gradients so every frozen MHD stage is also checked in reverse.",
    )
    parser.add_argument(
        "--backward-depth",
        type=int,
        default=len(RETF_FOUND_BACKWARD_LEVELS),
        help="Execute the first N reverse levels; 28 selects the complete path.",
    )
    return parser.parse_args()


def checkpoint_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_gradient_nodes(backward_depth: int) -> set[str]:
    """Return differentiable Node Messages reached by the selected reverse path."""

    names = {"loss", "logits"}
    if backward_depth >= 2:
        names.add("representation")
    for block_index in range(1, 25):
        if backward_depth >= 27 - block_index:
            names.add(f"block_{block_index:02d}")
    if backward_depth >= 27:
        names.add("patch_tokens")
    if backward_depth >= 28:
        names.add("image")
    return names


def parameter_is_on_selected_path(name: str, backward_depth: int) -> bool:
    """Map RETFound parameter names to the natural Operation reverse depth."""

    if name.startswith("head."):
        return backward_depth >= 2
    if name.startswith("fc_norm."):
        return backward_depth >= 3
    if name.startswith("blocks."):
        fields = name.split(".", 2)
        if len(fields) >= 2 and fields[1].isdigit():
            return backward_depth >= 27 - int(fields[1])
    return False


def main() -> None:
    args = parse_args()
    if args.require_checkpoint and not args.checkpoint:
        raise ValueError("--require-checkpoint requires --checkpoint")
    if args.steps < 1 or args.batch_size < 1 or args.sample_limit < args.batch_size:
        raise ValueError("steps/batch-size must be positive and sample-limit >= batch-size")
    if not 1 <= args.backward_depth <= len(RETF_FOUND_BACKWARD_LEVELS):
        raise ValueError("--backward-depth must be in [1, 28]")
    if args.backward_depth < 2:
        raise ValueError("Fine-tuning requires backward depth >= 2 so the head is selected")
    if args.check_full_backward and args.backward_depth != len(RETF_FOUND_BACKWARD_LEVELS):
        raise ValueError("--check-full-backward requires --backward-depth 28")
    active_backward_levels = RETF_FOUND_BACKWARD_LEVELS[: args.backward_depth]
    context = initialize_mhd_distributed()
    if args.parallel == "ddp" and context.world_size < 2:
        raise ValueError("--parallel ddp must be launched with torchrun")
    if args.parallel == "none" and context.world_size != 1:
        raise ValueError("WORLD_SIZE > 1 requires --parallel ddp")

    torch.manual_seed(args.seed)
    if context.device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = create_retfound_vit_large(num_classes=5)
    weight_report = None
    if args.checkpoint:
        weight_report = load_retfound_weights(model, args.checkpoint)
    total_parameters, trainable_parameters = select_finetune_parameters(
        model, args.trainable_blocks
    )
    graph = build_retfound_graph(
        model,
        device=context.device,
        num_classes=5,
        batch_size=args.batch_size,
    )
    parallel = MHD_ParallelConfig(
        data_parallel="ddp" if args.parallel == "ddp" else "none"
    )
    wrapped = prepare_mhd_model(
        graph,
        ["image", "target"],
        ["logits", "loss"],
        levels=RETF_FOUND_FORWARD_LEVELS,
        parallel=parallel,
        context=context,
        precision=args.precision,
    )
    optimizer = create_mhd_optimizer(
        wrapped,
        default_optimizer_type="adamw",
        default_lr=args.lr,
        default_weight_decay=args.weight_decay,
    )

    dataset = RETFoundUKBDataset(
        args.labels_csv,
        args.cache_root,
        modality=args.modality,
        split="train",
        limit=args.sample_limit,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=True,
            seed=args.seed,
            drop_last=True,
        )
        if context.distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=0,
        drop_last=True,
    )
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if args.precision == "bf16" and context.device.type == "cuda"
        else nullcontext()
    )

    wrapped.train()
    before = {
        name: parameter.detach().float().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith("head.")
    }
    losses: list[float] = []
    gradient_message_norms: list[dict[str, float]] = []
    iterator = iter(loader)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        image = batch["image"].to(context.device, non_blocking=True)
        target = batch["target"].to(context.device, non_blocking=True)
        if args.check_full_backward:
            image.requires_grad_(True)
        optimizer.zero_grad(set_to_none=True)
        with autocast:
            outputs = wrapped({"image": image, "target": target})
            loss = outputs["loss"]
        graph.backward(levels=active_backward_levels)
        trainable_named_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        gradients = [
            parameter.grad
            for _, parameter in trainable_named_parameters
            if parameter.grad is not None
        ]
        if not gradients or not all(torch.isfinite(value).all() for value in gradients):
            raise RuntimeError("RETFound produced missing or non-finite trainable gradients")
        for name, parameter in trainable_named_parameters:
            selected = parameter_is_on_selected_path(name, args.backward_depth)
            if selected and parameter.grad is None:
                raise RuntimeError(f"Selected parameter '{name}' has no gradient")
            if not selected and parameter.grad is not None:
                raise RuntimeError(f"Unselected parameter '{name}' unexpectedly has a gradient")

        message_names = (
            ["image", "patch_tokens"]
            + [f"block_{index:02d}" for index in range(1, 25)]
            + ["representation", "logits", "loss"]
        )
        message_norms = {
            name: float(
                graph.get_node_by_name(name)
                .gradient_message.current_state.detach().float().norm().cpu()
            )
            for name in message_names
        }
        expected_nodes = expected_gradient_nodes(args.backward_depth)
        missing_messages = [name for name in expected_nodes if message_norms[name] == 0.0]
        unexpected_messages = [
            name for name in message_names
            if name not in expected_nodes and message_norms[name] != 0.0
        ]
        if missing_messages or unexpected_messages:
            raise RuntimeError(
                "Gradient Message path mismatch: "
                f"missing={sorted(missing_messages)}, "
                f"unexpected={sorted(unexpected_messages)}"
            )
        gradient_message_norms.append(message_norms)
        if args.check_full_backward:
            input_gradient = graph.get_node_by_name("image").gradient_message.current_state
            if not torch.isfinite(input_gradient).all() or input_gradient.float().norm() == 0:
                raise RuntimeError("Full MHD backward check did not reach the image node")
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))

    changed = []
    for name, parameter in model.named_parameters():
        if name in before:
            delta = (parameter.detach().float().cpu() - before[name]).abs().max().item()
            if delta > 0:
                changed.append(name)
    if not changed:
        raise RuntimeError("Fine-tuning step did not update the RETFound classification head")

    checksum = torch.stack(
        [parameter.detach().float().sum() for parameter in model.parameters() if parameter.requires_grad]
    ).sum()
    rank_checksums = [checksum.detach().clone()]
    if context.distributed:
        rank_checksums = [torch.zeros_like(checksum) for _ in range(context.world_size)]
        dist.all_gather(rank_checksums, checksum)
        for other in rank_checksums[1:]:
            torch.testing.assert_close(other, rank_checksums[0], rtol=1e-5, atol=1e-4)

    result = {
        "status": "passed",
        "architecture": "RETFound ViT-L/16 2D, 24-block MHD topology",
        "mhd_nodes": [
            graph.get_node_by_id(index).name for index in range(len(graph.nodes))
        ],
        "mhd_edges": [
            graph.get_edge_by_id(index).name for index in range(len(graph.edges))
        ],
        "role_matrix_shape": list(graph.topo.role_matrices[0].shape),
        "sort_matrix_shape": list(graph.topo.sort_matrices[0].shape),
        "topology_levels": graph.num_levels,
        "forward_levels": list(RETF_FOUND_FORWARD_LEVELS),
        "backward_levels": list(active_backward_levels),
        "backward_depth": args.backward_depth,
        "expected_gradient_nodes": sorted(expected_gradient_nodes(args.backward_depth)),
        "modality": args.modality,
        "world_size": context.world_size,
        "parallel": args.parallel,
        "precision": args.precision,
        "batch_size_per_rank": args.batch_size,
        "steps": args.steps,
        "full_backward_checked": args.check_full_backward,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "losses": losses,
        "gradient_message_norms_by_node": gradient_message_norms,
        "updated_head_parameters": sorted(changed),
        "rank_checksums": [float(value.detach().cpu()) for value in rank_checksums],
        "official_checkpoint_loaded": weight_report is not None,
        "checkpoint_sha256": checkpoint_sha256(args.checkpoint) if args.checkpoint else None,
        "weight_report": None
        if weight_report is None
        else {
            "checkpoint": weight_report.checkpoint,
            "tensor_count": weight_report.tensor_count,
            "missing_keys": list(weight_report.missing_keys),
            "unexpected_keys": list(weight_report.unexpected_keys),
            "removed_classifier_keys": list(weight_report.removed_classifier_keys),
            "pretrained_backbone_loaded": weight_report.pretrained_backbone_loaded,
        },
    }
    if context.is_main:
        output = Path(args.result_json) if args.result_json else EXPERIMENT_DIR / "result.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print("MHD_RETFOUND_EXPERIMENT_OK", json.dumps(result, sort_keys=True), flush=True)
    destroy_mhd_distributed()


if __name__ == "__main__":
    main()
