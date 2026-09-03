from __future__ import annotations

import torch
import torch.nn as nn

from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V4.MHD_Utils_V4 import MHD_Monitor


def test_readme_global_level_example_executes_verbatim_in_meaning():
    linear = nn.Linear(1, 1, bias=False)
    nodes = {
        MHD_Node(0, "x", MHD_Node.Message(torch.zeros(2, 1))),
        MHD_Node(1, "prediction", MHD_Node.Message(torch.zeros(2, 1))),
        MHD_Node(2, "loss", MHD_Node.Message(torch.zeros(()))),
    }
    edges = {
        MHD_Edge(0, "linear", [MHD_Edge.Operation(linear)]),
        MHD_Edge(1, "mean", [MHD_Edge.Operation(lambda value: value.square().mean())]),
    }
    role_matrices = [
        torch.tensor([[-1, 1, 0], [0, 0, 0]]),
        torch.tensor([[0, 0, 0], [0, -1, 1]]),
        torch.tensor([[0, 0, 0], [0, 1, -1]]),
        torch.tensor([[1, -1, 0], [0, 0, 0]]),
    ]
    sort_matrices = [
        torch.tensor([[0, 1, 0], [0, 0, 0]]),
        torch.tensor([[0, 0, 0], [0, 0, 1]]),
        torch.tensor([[0, 0, 0], [0, 0, 1]]),
        torch.tensor([[0, 1, 0], [0, 0, 0]]),
    ]
    graph = MHD_Graph(nodes, edges, {MHD_Topo(role_matrices, sort_matrices)}, device="cpu")
    value = torch.randn(2, 1, requires_grad=True)
    graph.get_node_by_name("x").feature_message.current_state = value
    graph.forward(levels=[0, 1])
    graph.backward(levels=[2, 3])
    assert linear.weight.grad is not None
    assert value.grad is not None


def test_readme_message_monitor_and_custom_autograd_examples_execute():
    message = MHD_Node.Message(torch.zeros(2))
    message.update_initial(torch.ones(2)).reset().to_device(torch.device("cpu"))
    node = MHD_Node(0, "value", message)
    monitor = MHD_Monitor(
        ["value"],
        node_states=(
            "feature_message.initial_state",
            "feature_message.current_state",
            "gradient_message.initial_state",
            "gradient_message.current_state",
        ),
        statistics=("mean",),
    )
    role = torch.tensor([[-1, 1]])
    sort = torch.tensor([[0, 1]])
    graph = MHD_Graph(
        {node, MHD_Node(1, "loss", MHD_Node.Message(torch.zeros(())))},
        {MHD_Edge(0, "mean", [MHD_Edge.Operation(lambda value: value.mean())])},
        {MHD_Topo([role, -role], [sort, sort])},
        device="cpu",
    )
    assert len(monitor.monitor_node(graph)) == 4
