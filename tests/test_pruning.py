from __future__ import annotations

import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V4.MHD_Utils_V4 import prune_isolated_graph


def test_pruning_uses_the_one_global_topology_and_rebuilds_registration():
    active = nn.Linear(1, 1)
    isolated = nn.Linear(1, 1)
    nodes = {
        MHD_Node(0, "input", MHD_Node.Message(torch.zeros(1, 1))),
        MHD_Node(1, "output", MHD_Node.Message(torch.zeros(1, 1))),
        MHD_Node(2, "unused", MHD_Node.Message(torch.zeros(1, 1))),
    }
    edges = {
        MHD_Edge(0, "active", [MHD_Edge.Operation(active)]),
        MHD_Edge(1, "unused", [MHD_Edge.Operation(isolated)]),
    }
    forward = torch.tensor([[-1, 1, 0], [0, 0, 0]])
    sort = torch.tensor([[0, 1, 0], [0, 0, 0]])
    graph = MHD_Graph(
        nodes,
        edges,
        {MHD_Topo([forward, -forward], [sort, sort])},
        device="cpu",
    )
    prune_isolated_graph(graph, verbose=False)
    assert [node.name for node in sorted(graph.nodes, key=lambda node: node.id)] == [
        "input", "output"
    ]
    assert [edge.name for edge in graph.edges] == ["active"]
    assert all(matrix.shape == (1, 2) for matrix in graph.topo.role_matrices)
    parameter_ids = {id(parameter) for parameter in graph.parameters()}
    assert id(active.weight) in parameter_ids
    assert id(isolated.weight) not in parameter_ids

