"""Isolated V3 checkpoint migration helpers for MHD Framework V4."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import torch

from .MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo


def migrate_v3_node(v3_node: Any) -> MHD_Node:
    """Convert one V3 node to the final V4 nested Message representation."""
    return MHD_Node(
        id=int(v3_node.id),
        name=str(v3_node.name),
        feature_message=MHD_Node.Message(
            v3_node.initial_state.detach().clone(),
            v3_node.current_state.detach().clone(),
        ),
        aggregation=v3_node.transfer_mode,
    )


def default_v4_level_sequences(num_forward_levels: int) -> Dict[str, list[int]]:
    """Return explicit global V4 paths for an automatically mirrored V3 topology."""
    if num_forward_levels < 1:
        raise ValueError("num_forward_levels 必须大于 0")
    return {
        "forward_levels": list(range(num_forward_levels)),
        "backward_levels": list(range(num_forward_levels, 2 * num_forward_levels)),
    }


def migrate_v3_graph(v3_graph: Any, device: Optional[torch.device] = None) -> MHD_Graph:
    """Convert a live V3 graph while preserving its four hypergraph concepts."""
    nodes = {migrate_v3_node(node) for node in v3_graph.nodes}
    edges = {
        MHD_Edge(
            edge.id,
            edge.name,
            [MHD_Edge.Operation(operation) for operation in edge.edge_operations],
        )
        for edge in v3_graph.edges
    }
    forward_roles = [matrix.detach().clone() for matrix in v3_graph.topo.role_matrices]
    forward_sorts = [matrix.detach().clone() for matrix in v3_graph.topo.sort_matrices]
    backward_roles = [(-matrix).clone() for matrix in reversed(forward_roles)]
    backward_sorts = [matrix.clone() for matrix in reversed(forward_sorts)]
    topo = MHD_Topo(forward_roles + backward_roles, forward_sorts + backward_sorts)
    return MHD_Graph(nodes, edges, {topo}, device=device or v3_graph.device)


def _checkpoint_paths(source: Path, epoch: Optional[int]) -> Dict[str, Path]:
    suffix = "best" if epoch is None else f"epoch_{epoch}"
    return {
        "node": source / f"node_{suffix}.pth",
        "edge": source / f"edge_{suffix}.pth",
        "meta": source / f"meta_{suffix}.pth",
    }


def migrate_v3_checkpoint(
    source_dir: str,
    graph: MHD_Graph,
    output_dir: str,
    *,
    epoch: Optional[int] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    include_current_state: bool = False,
    forward_levels: Optional[Sequence[int]] = None,
    backward_levels: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Migrate one V3 three-file checkpoint into a V4 DCP directory.

    Edge modules are matched by edge name and operation index. Node tensors are
    matched by node name. Optimizer migration is best-effort because third-party
    optimizers may not preserve a portable V3 parameter ordering.
    """
    source = Path(source_dir)
    destination = Path(output_dir)
    paths = _checkpoint_paths(source, epoch)
    missing_files = [str(path) for path in paths.values() if not path.exists()]
    if missing_files:
        raise FileNotFoundError(f"缺少 V3 检查点文件: {missing_files}")

    node_state = torch.load(paths["node"], map_location="cpu", weights_only=True)
    edge_state = torch.load(paths["edge"], map_location="cpu", weights_only=True)
    meta_state = torch.load(paths["meta"], map_location="cpu", weights_only=True)
    if (forward_levels is None) != (backward_levels is None):
        raise ValueError("forward_levels 与 backward_levels 必须同时提供或同时省略")
    if forward_levels is None:
        if graph.num_levels % 2 != 0:
            raise ValueError("无法从奇数个全局 level 自动推导 V3 迁移路径")
        inferred = default_v4_level_sequences(graph.num_levels // 2)
        forward_levels = inferred["forward_levels"]
        backward_levels = inferred["backward_levels"]
    normalized_forward = graph._validate_levels(
        forward_levels, graph.num_levels, "迁移 Forward"
    )
    normalized_backward = graph._validate_levels(
        backward_levels, graph.num_levels, "迁移 Backward"
    )
    overlap = sorted(set(normalized_forward).intersection(normalized_backward))
    if overlap:
        raise ValueError(f"迁移前后向 levels 不得重叠: {overlap}")

    report: Dict[str, Any] = {
        "loaded_nodes": [],
        "missing_nodes": [],
        "loaded_edge_operations": [],
        "missing_edges": [],
        "optimizer_migrated": False,
        "forward_levels": normalized_forward,
        "backward_levels": normalized_backward,
    }

    initial_states = node_state.get("node_initial_states", {})
    current_states = node_state.get("node_current_states", {})
    for node in sorted(graph.nodes, key=lambda item: item.id):
        if node.name not in initial_states:
            report["missing_nodes"].append(node.name)
            continue
        initial = initial_states[node.name].to(graph.device, dtype=node.feature_message.initial_state.dtype)
        node.feature_message.update_initial(initial, update_current=True)
        if include_current_state and node.name in current_states:
            node.feature_message.current_state = current_states[node.name].to(
                graph.device,
                dtype=node.feature_message.current_state.dtype,
            )
        report["loaded_nodes"].append(node.name)

    edge_parameters = edge_state.get("edge_params", {})
    for edge in sorted(graph.edges, key=lambda item: item.id):
        saved_operations = edge_parameters.get(edge.name)
        if saved_operations is None:
            report["missing_edges"].append(edge.name)
            continue
        for index, operation in enumerate(edge.edge_operations):
            function = operation.function
            if not isinstance(function, torch.nn.Module):
                continue
            if index >= len(saved_operations) or saved_operations[index] is None:
                continue
            function.load_state_dict(saved_operations[index], strict=True)
            report["loaded_edge_operations"].append(f"{edge.name}:{index}")

    if optimizer is not None and meta_state.get("optimizer"):
        try:
            optimizer.load_state_dict(meta_state["optimizer"])
            report["optimizer_migrated"] = True
        except (ValueError, KeyError) as exc:
            report["optimizer_warning"] = str(exc)
    if scheduler is not None and meta_state.get("lr_scheduler"):
        scheduler.load_state_dict(meta_state["lr_scheduler"])

    from torch.distributed import checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_state_dict

    if optimizer is None:
        state: Dict[str, Any] = {"model": get_model_state_dict(graph)}
    else:
        model_state, optimizer_state = get_state_dict(graph, optimizer)
        state = {"model": model_state, "optimizer": optimizer_state}
    state.update(
        {
            "node_messages": {
                node.name: {
                    "feature_message": {
                        "initial_state": node.feature_message.initial_state.detach(),
                        "current_state": node.feature_message.current_state.detach(),
                    },
                    "gradient_message": {
                        "initial_state": node.gradient_message.initial_state.detach(),
                        "current_state": node.gradient_message.current_state.detach(),
                    },
                }
                for node in sorted(graph.nodes, key=lambda item: item.id)
            },
            "trainer": {
                "history": meta_state.get("history", {}),
                "epoch": meta_state.get("epoch", epoch or 0),
                "migrated_from": "V3",
            },
            "scheduler": scheduler.state_dict() if scheduler is not None else {},
        }
    )
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = destination / ("best" if epoch is None else f"epoch_{epoch}")
    dcp.save(state, checkpoint_id=checkpoint_dir, no_dist=True)
    report_path = destination / "migration_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _load_factory(specification: str) -> Callable[..., MHD_Graph]:
    module_name, separator, function_name = specification.partition(":")
    if not separator:
        raise ValueError("factory 必须使用 module:function 格式")
    factory = getattr(importlib.import_module(module_name), function_name)
    if not callable(factory):
        raise TypeError(f"factory 不可调用: {specification}")
    return factory


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate an MHD V3 checkpoint to V4")
    parser.add_argument("source_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--factory", required=True, help="Graph factory in module:function form")
    parser.add_argument("--epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--include-current-state", action="store_true")
    parser.add_argument("--forward-levels", help="逗号分隔的显式 Forward level 序列")
    parser.add_argument("--backward-levels", help="逗号分隔的显式 Backward level 序列")
    args = parser.parse_args()

    factory = _load_factory(args.factory)
    graph = factory(batch_size=args.batch_size, device=torch.device(args.device))
    report = migrate_v3_checkpoint(
        args.source_dir,
        graph,
        args.output_dir,
        epoch=args.epoch,
        include_current_state=args.include_current_state,
        forward_levels=(
            [int(value) for value in args.forward_levels.split(",")]
            if args.forward_levels else None
        ),
        backward_levels=(
            [int(value) for value in args.backward_levels.split(",")]
            if args.backward_levels else None
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
