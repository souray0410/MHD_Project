"""Measure MHD forward and training overhead against the same native module."""

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V4.MHD_Utils_V4 import _MHD_GraphAdapter


def measure_once(function) -> float:
    # Host wall time with device synchronization includes both Python dispatch
    # and CUDA execution. CUDA-event-only timing can hide host dispatch when a
    # shared GPU already has queued work.
    torch.cuda.synchronize()
    start = time.perf_counter()
    function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0


def measure_pair(native, mhd, warmup: int, iterations: int) -> tuple[float, float]:
    for _ in range(warmup):
        native()
        mhd()
    torch.cuda.synchronize()
    native_samples = []
    mhd_samples = []
    for index in range(iterations):
        # Alternate order so GPU clock changes and background load do not
        # systematically favor either path.
        if index % 2 == 0:
            native_samples.append(measure_once(native))
            mhd_samples.append(measure_once(mhd))
        else:
            mhd_samples.append(measure_once(mhd))
            native_samples.append(measure_once(native))
    return statistics.median(native_samples), statistics.median(mhd_samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--max-overhead-percent", type=float, default=15.0)
    parser.add_argument("--max-overhead-us", type=float, default=25.0)
    parser.add_argument("--training-iterations", type=int, default=50)
    parser.add_argument("--max-training-overhead-percent", type=float, default=10.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    device = torch.device("cuda", 0)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    module = nn.TransformerEncoderLayer(
        d_model=args.hidden_size,
        nhead=8,
        dim_feedforward=args.hidden_size * 4,
        dropout=0.0,
        batch_first=True,
        device=device,
        dtype=dtype,
    ).eval()
    data = torch.randn(
        args.batch_size,
        args.sequence_length,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    nodes = {
        MHD_Node(0, "input", MHD_Node.Message(torch.zeros_like(data))),
        MHD_Node(1, "output", MHD_Node.Message(torch.zeros_like(data))),
        MHD_Node(2, "loss", MHD_Node.Message(torch.zeros((), device=device, dtype=dtype))),
    }
    edges = {
        MHD_Edge(0, "transformer", [MHD_Edge.Operation(module)]),
        MHD_Edge(1, "criterion", [MHD_Edge.Operation(lambda value: value.square().mean())]),
    }
    forward_role_0 = torch.tensor([[-1, 1, 0], [0, 0, 0]], device=device)
    forward_role_1 = torch.tensor([[0, 0, 0], [0, -1, 1]], device=device)
    forward_sort_0 = torch.tensor([[0, 1, 0], [0, 0, 0]], device=device)
    forward_sort_1 = torch.tensor([[0, 0, 0], [0, 0, 1]], device=device)
    topo = MHD_Topo(
        [forward_role_0, forward_role_1, -forward_role_1, -forward_role_0],
        [forward_sort_0, forward_sort_1, forward_sort_1, forward_sort_0],
    )
    graph = MHD_Graph(nodes, edges, {topo}, device=device).eval()
    adapter = _MHD_GraphAdapter(graph, ["input"], ["output"], levels=[0]).eval()
    training_adapter = _MHD_GraphAdapter(graph, ["input"], ["loss"], levels=[0, 1])

    with torch.inference_mode():
        native_ms, mhd_ms = measure_pair(
            lambda: module(data),
            lambda: adapter({"input": data})["output"],
            args.warmup,
            args.iterations,
        )
    overhead = (mhd_ms / native_ms - 1.0) * 100.0
    overhead_us = (mhd_ms - native_ms) * 1000.0
    print(f"native_median_ms={native_ms:.4f}")
    print(f"mhd_median_ms={mhd_ms:.4f}")
    print(f"framework_overhead_percent={overhead:.2f}")
    print(f"framework_overhead_us={overhead_us:.2f}")
    forward_failed = (
        overhead > args.max_overhead_percent
        and overhead_us > args.max_overhead_us
    )

    module.train()
    graph.train()
    adapter.train()
    def native_training_step() -> None:
        module.zero_grad(set_to_none=True)
        module(data).square().mean().backward()

    def mhd_training_step() -> None:
        module.zero_grad(set_to_none=True)
        loss = training_adapter({"input": data})["loss"]
        graph.backward(levels=[2, 3])
        if loss.numel() != 1:
            raise AssertionError("unexpected MHD loss shape")

    # Correctness is checked before timing so the benchmark cannot report a
    # fast but mathematically different path.
    native_training_step()
    native_gradients = [parameter.grad.detach().clone() for parameter in module.parameters()]
    mhd_training_step()
    for parameter, expected in zip(module.parameters(), native_gradients):
        torch.testing.assert_close(parameter.grad, expected, rtol=2e-3, atol=2e-3)

    native_train_ms, mhd_train_ms = measure_pair(
        native_training_step,
        mhd_training_step,
        args.warmup,
        args.training_iterations,
    )
    training_overhead = (mhd_train_ms / native_train_ms - 1.0) * 100.0
    print(f"native_training_median_ms={native_train_ms:.4f}")
    print(f"mhd_training_median_ms={mhd_train_ms:.4f}")
    print(f"training_overhead_percent={training_overhead:.2f}")
    failures = []
    if forward_failed:
        failures.append(
            "forward overhead exceeds both limits: "
            f"{overhead:.2f}% > {args.max_overhead_percent:.2f}% and "
            f"{overhead_us:.2f}us > {args.max_overhead_us:.2f}us"
        )
    if training_overhead > args.max_training_overhead_percent:
        failures.append(
            "training overhead exceeds limit: "
            f"{training_overhead:.2f}% > {args.max_training_overhead_percent:.2f}%"
        )
    if failures:
        raise SystemExit("MHD benchmark failed: " + "; ".join(failures))


if __name__ == "__main__":
    main()
