"""Synthetic model-level equivalence checks for MHD V4.

The script deliberately uses ordinary PyTorch modules and tensors. It checks
that ResNet-style branching, Transformer blocks, and recurrent graph message
passing have the same outputs and gradients through the MHD topology.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo


def node(node_id: int, name: str, value: torch.Tensor, aggregation="replace"):
    return MHD_Node(node_id, name, MHD_Node.Message(value), aggregation=aggregation)


def edge(edge_id: int, name: str, function):
    return MHD_Edge(edge_id, name, [MHD_Edge.Operation(function)])


def matrices(num_edges, num_nodes, edge_id, heads, tails):
    role = torch.zeros(num_edges, num_nodes, dtype=torch.int64)
    sort = torch.zeros_like(role)
    for index, node_id in enumerate(heads):
        role[edge_id, node_id] = -1
        sort[edge_id, node_id] = index
    for index, node_id in enumerate(tails, start=len(heads)):
        role[edge_id, node_id] = 1
        sort[edge_id, node_id] = index
    return role, sort


def build_graph(nodes, edges, forward_definitions, device):
    forward_pairs = [
        matrices(len(edges), len(nodes), *definition)
        for definition in forward_definitions
    ]
    roles = [pair[0] for pair in forward_pairs]
    sorts = [pair[1] for pair in forward_pairs]
    roles += [(-matrix).clone() for matrix in reversed(roles)]
    sorts += [matrix.clone() for matrix in reversed(sorts)]
    graph = MHD_Graph(set(nodes), set(edges), {MHD_Topo(roles, sorts)}, device=device)
    count = len(forward_definitions)
    return graph, list(range(count)), list(range(count, count * 2))


class ResidualMain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8),
        )

    def forward(self, value):
        return self.net(value)


def run_resnet(device):
    batch = 2
    stem = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.ReLU()).to(device).eval()
    main = ResidualMain().to(device).eval()
    head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(8, 4)).to(device)
    criterion = nn.CrossEntropyLoss()
    nodes = [
        node(0, "image", torch.zeros(batch, 3, 16, 16)),
        node(1, "target", torch.zeros(batch, dtype=torch.long)),
        node(2, "stem", torch.zeros(batch, 8, 16, 16)),
        node(3, "main", torch.zeros(batch, 8, 16, 16)),
        node(4, "skip", torch.zeros(batch, 8, 16, 16)),
        node(5, "features", torch.zeros(batch, 8, 16, 16)),
        node(6, "logits", torch.zeros(batch, 4)),
        node(7, "loss", torch.zeros(())),
    ]
    edges = [
        edge(0, "stem", stem),
        edge(1, "main", main),
        edge(2, "skip", nn.Identity()),
        edge(3, "residual_add", lambda left, right: torch.relu(left + right)),
        edge(4, "head", head),
        edge(5, "criterion", criterion),
    ]
    definitions = [
        (0, (0,), (2,)), (1, (2,), (3,)), (2, (2,), (4,)),
        (3, (3, 4), (5,)), (4, (5,), (6,)), (5, (6, 1), (7,)),
    ]
    graph, forward, backward = build_graph(nodes, edges, definitions, device)
    image = torch.randn(batch, 3, 16, 16, device=device, requires_grad=True)
    target = torch.tensor([1, 3], device=device)
    stem_value = stem(image)
    logits = head(torch.relu(main(stem_value) + stem_value))
    reference_loss = criterion(logits, target)
    parameters = tuple(graph.parameters())
    expected = torch.autograd.grad(reference_loss, (image, *parameters))
    graph.get_node_by_name("image").feature_message.current_state = image
    graph.get_node_by_name("target").feature_message.current_state = target
    graph.forward(levels=forward)
    torch.testing.assert_close(graph.get_node_by_name("logits").feature_message.current_state, logits)
    graph.backward(levels=backward)
    torch.testing.assert_close(image.grad, expected[0])
    for parameter, gradient in zip(parameters, expected[1:]):
        torch.testing.assert_close(parameter.grad, gradient)
    return float(reference_loss.detach())


def run_transformer(device):
    batch, length, width = 2, 6, 16
    block_1 = nn.TransformerEncoderLayer(width, 4, 32, dropout=0.0, batch_first=True).to(device).eval()
    block_2 = nn.TransformerEncoderLayer(width, 4, 32, dropout=0.0, batch_first=True).to(device).eval()
    head = nn.Linear(width, 3).to(device)
    criterion = nn.CrossEntropyLoss()
    nodes = [
        node(0, "tokens", torch.zeros(batch, length, width)),
        node(1, "target", torch.zeros(batch, dtype=torch.long)),
        node(2, "block_1", torch.zeros(batch, length, width)),
        node(3, "block_2", torch.zeros(batch, length, width)),
        node(4, "pooled", torch.zeros(batch, width)),
        node(5, "logits", torch.zeros(batch, 3)),
        node(6, "loss", torch.zeros(())),
    ]
    edges = [
        edge(0, "block_1", block_1), edge(1, "block_2", block_2),
        edge(2, "pool", lambda value: value.mean(dim=1)), edge(3, "head", head),
        edge(4, "criterion", criterion),
    ]
    definitions = [
        (0, (0,), (2,)), (1, (2,), (3,)), (2, (3,), (4,)),
        (3, (4,), (5,)), (4, (5, 1), (6,)),
    ]
    graph, forward, backward = build_graph(nodes, edges, definitions, device)
    tokens = torch.randn(batch, length, width, device=device, requires_grad=True)
    target = torch.tensor([0, 2], device=device)
    logits = head(block_2(block_1(tokens)).mean(dim=1))
    reference_loss = criterion(logits, target)
    parameters = tuple(graph.parameters())
    expected = torch.autograd.grad(reference_loss, (tokens, *parameters))
    graph.get_node_by_name("tokens").feature_message.current_state = tokens
    graph.get_node_by_name("target").feature_message.current_state = target
    graph.forward(levels=forward)
    torch.testing.assert_close(graph.get_node_by_name("logits").feature_message.current_state, logits)
    graph.backward(levels=backward)
    torch.testing.assert_close(tokens.grad, expected[0])
    for parameter, gradient in zip(parameters, expected[1:]):
        torch.testing.assert_close(parameter.grad, gradient)
    return float(reference_loss.detach())


def run_recurrent_hypergraph(device):
    width = 8
    shared_message = nn.Linear(width, width, bias=False).to(device)
    nodes = [
        node(0, "step_0", torch.zeros(2, width)),
        node(1, "step_1", torch.zeros(2, width)),
        node(2, "step_2", torch.zeros(2, width)),
        node(3, "summary", torch.zeros(2, width)),
        node(4, "loss", torch.zeros(())),
    ]
    edges = [
        edge(0, "shared_message", shared_message),
        edge(1, "hyperedge_readout", lambda first, second: first + second),
        edge(2, "criterion", lambda value: value.square().mean()),
    ]
    # Edge 0 is genuinely reused at two user-selected levels.
    definitions = [
        (0, (0,), (1,)), (0, (1,), (2,)),
        (1, (1, 2), (3,)), (2, (3,), (4,)),
    ]
    graph, forward, backward = build_graph(nodes, edges, definitions, device)
    value = torch.randn(2, width, device=device, requires_grad=True)
    first = shared_message(value)
    second = shared_message(first)
    reference_loss = (first + second).square().mean()
    expected = torch.autograd.grad(reference_loss, (value, shared_message.weight))
    graph.get_node_by_name("step_0").feature_message.current_state = value
    graph.forward(levels=forward).backward(levels=backward)
    torch.testing.assert_close(value.grad, expected[0])
    torch.testing.assert_close(shared_message.weight.grad, expected[1])
    return float(reference_loss.detach())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--result-json")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(1234)
    report = {
        "status": "passed",
        "device": str(device),
        "resnet_loss": run_resnet(device),
        "transformer_loss": run_transformer(device),
        "recurrent_hypergraph_loss": run_recurrent_hypergraph(device),
    }
    if args.result_json:
        Path(args.result_json).write_text(json.dumps(report, indent=2) + "\n")
    print("MHD_MODEL_SMOKE_OK", json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
