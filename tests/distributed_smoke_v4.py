"""Two-rank checks for each independent MHD V4 parallel family."""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import nullcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.distributed as dist
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V4.MHD_Utils_V4 import (
    MHD_ParallelConfig,
    create_mhd_optimizer,
    destroy_mhd_distributed,
    initialize_mhd_distributed,
    prepare_mhd_model,
)


FORWARD_LEVELS = (0, 1, 2)
FULL_BACKWARD_LEVELS = (3, 4, 5)
PARTIAL_BACKWARD_LEVELS = (3, 4)


class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return -gradient


class LearnableMeanSquare(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(4))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return (value.square() * self.scale).mean(dim=-1)


def build_graph(
    device: torch.device,
    custom_autograd: bool = False,
    batch_size: int = 2,
) -> MHD_Graph:
    projection_operations = [MHD_Edge.Operation(nn.Linear(4, 4, bias=False))]
    if custom_autograd:
        projection_operations.append(MHD_Edge.Operation(ReverseGradient.apply))
    nodes = {
        MHD_Node(0, "input", MHD_Node.Message(torch.zeros(batch_size, 4, device=device))),
        MHD_Node(1, "hidden", MHD_Node.Message(torch.zeros(batch_size, 4, device=device))),
        MHD_Node(2, "per_sample_loss", MHD_Node.Message(torch.zeros(batch_size, device=device))),
        MHD_Node(3, "loss", MHD_Node.Message(torch.zeros((), device=device))),
    }
    edges = {
        MHD_Edge(0, "projection", projection_operations),
        MHD_Edge(1, "loss_reduce", [MHD_Edge.Operation(LearnableMeanSquare())]),
        MHD_Edge(2, "mean", [MHD_Edge.Operation(lambda value: value.mean())]),
    }
    forward_0 = torch.tensor([[-1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], device=device)
    forward_1 = torch.tensor([[0, 0, 0, 0], [0, -1, 1, 0], [0, 0, 0, 0]], device=device)
    forward_2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, -1, 1]], device=device)
    sort_0 = torch.tensor([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], device=device)
    sort_1 = torch.tensor([[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]], device=device)
    sort_2 = torch.tensor([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1]], device=device)
    topo = MHD_Topo(
        [forward_0, forward_1, forward_2, -forward_2, -forward_1, -forward_0],
        [sort_0, sort_1, sort_2, sort_2, sort_1, sort_0],
    )
    return MHD_Graph(nodes, edges, {topo}, device=device)


def local_tensor(parameter: torch.Tensor) -> torch.Tensor:
    return parameter.to_local() if hasattr(parameter, "to_local") else parameter


def run_data_or_tensor_parallel(mode: str) -> None:
    context = initialize_mhd_distributed()
    torch.manual_seed(1234)
    custom_autograd = os.environ.get("MHD_CUSTOM_AUTOGRAD", "0") == "1"
    compile_enabled = os.environ.get("MHD_COMPILE", "0") == "1"
    compile_backend = os.environ.get("MHD_COMPILE_BACKEND")
    precision = os.environ.get("MHD_PRECISION", "fp32")
    partial = os.environ.get("MHD_PARTIAL", "0") == "1"
    batch_size = int(os.environ.get("MHD_BATCH_SIZE", "2"))
    if batch_size < 1:
        raise ValueError("MHD_BATCH_SIZE must be at least 1")
    graph = build_graph(
        context.device,
        custom_autograd=custom_autograd,
        batch_size=batch_size,
    )
    if mode == "ddp":
        config = MHD_ParallelConfig(
            data_parallel="ddp", compile=compile_enabled,
            compile_backend=compile_backend,
        )
    elif mode == "fsdp2":
        config = MHD_ParallelConfig(
            data_parallel="fsdp2", compile=compile_enabled,
            compile_backend=compile_backend,
        )
    elif mode == "tp":
        config = MHD_ParallelConfig(
            tensor_parallel_size=context.world_size,
            tensor_parallel_plan={"projection:0": "colwise"},
            compile=compile_enabled,
            compile_backend=compile_backend,
        )
    else:
        raise ValueError(mode)
    model = prepare_mhd_model(
        graph,
        ["input"],
        ["loss"],
        levels=FORWARD_LEVELS,
        parallel=config,
        context=context,
        precision=precision,
    )
    optimizer = create_mhd_optimizer(model, default_optimizer_type="sgd", default_lr=0.01)
    inputs = (
        torch.arange(batch_size * 4, dtype=torch.float32, device=context.device)
        .reshape(batch_size, 4)
        / max(batch_size * 4, 1)
    ).requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    autocast_context = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if precision == "bf16" else nullcontext()
    )
    with autocast_context:
        loss = model({"input": inputs})["loss"]
    graph.backward(
        levels=PARTIAL_BACKWARD_LEVELS if partial else FULL_BACKWARD_LEVELS
    )
    projection = graph.get_edge_by_name("projection").edge_operations[0].function
    if partial:
        assert projection.weight.grad is None
    else:
        assert projection.weight.grad is not None
    assert torch.isfinite(loss)
    assert torch.count_nonzero(
        graph.get_node_by_name("hidden").gradient_message.current_state
    ) > 0
    optimizer.step()
    for parameter in model.parameters():
        assert torch.isfinite(local_tensor(parameter)).all()
    if mode == "ddp":
        checksum = torch.stack([
            parameter.detach().float().sum() for parameter in model.parameters()
        ]).sum()
        gathered = [torch.zeros_like(checksum) for _ in range(context.world_size)]
        dist.all_gather(gathered, checksum)
        for value in gathered[1:]:
            torch.testing.assert_close(value, gathered[0])
    if context.is_main:
        print(
            "MHD_DISTRIBUTED_SMOKE_OK "
            f"mode={mode} partial={partial} custom={custom_autograd} "
            f"compile={compile_enabled} precision={precision}"
        )
    destroy_mhd_distributed()


def run_pipeline() -> None:
    context = initialize_mhd_distributed()
    if context.world_size != 2:
        raise RuntimeError("pipeline smoke test requires exactly two ranks")
    torch.manual_seed(1234)
    partial = os.environ.get("MHD_PARTIAL", "0") == "1"
    graph = build_graph(context.device)
    config = MHD_ParallelConfig(
        pipeline_size=2,
        pipeline_stages={"projection": 0, "loss_reduce": 1},
        pipeline_microbatches=2,
        pipeline_schedule=os.environ.get("MHD_PIPELINE_SCHEDULE", "gpipe"),
    )
    model = prepare_mhd_model(
        graph,
        ["input"],
        ["per_sample_loss"],
        levels=[0, 1],
        backward_levels=(
            [4] if partial else [4, 5]
        ),
        parallel=config,
        context=context,
        example_inputs={"input": torch.zeros(2, 4, device=context.device)},
    )
    optimizer = create_mhd_optimizer(model, default_optimizer_type="sgd", default_lr=0.01)
    optimizer.zero_grad(set_to_none=True)
    losses = []
    if context.rank == 0:
        inputs = torch.arange(8, dtype=torch.float32, device=context.device).reshape(2, 4) / 8
        model(inputs)
    else:
        model(target=torch.zeros(2, device=context.device), losses=losses)
        assert losses and all(torch.isfinite(loss) for loss in losses)
    optimizer.step()
    for parameter in model.parameters():
        if partial and context.rank == 0:
            assert parameter.grad is None
        else:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
    assert torch.count_nonzero(
        graph.get_node_by_name("hidden").gradient_message.current_state
    ) > 0
    if context.is_main:
        print(
            "MHD_DISTRIBUTED_SMOKE_OK "
            f"mode=pp schedule={config.pipeline_schedule} partial={partial}"
        )
    destroy_mhd_distributed()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["ddp", "fsdp2", "tp", "pp"])
    args = parser.parse_args()
    if int(os.environ.get("WORLD_SIZE", "1")) < 2:
        raise RuntimeError("launch with torchrun --nproc-per-node=2")
    if args.mode == "pp":
        run_pipeline()
    else:
        run_data_or_tensor_parallel(args.mode)


if __name__ == "__main__":
    main()
