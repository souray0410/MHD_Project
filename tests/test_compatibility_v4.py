from __future__ import annotations

import torch
import torch.nn as nn

from V3.MHD_Framework_V3 import (
    MHD_Edge as V3_Edge,
    MHD_Graph as V3_Graph,
    MHD_Node as V3_Node,
    MHD_Topo as V3_Topo,
)
from V4.MHD_Compatibility_V3_to_V4 import (
    default_v4_level_sequences,
    migrate_v3_checkpoint,
    migrate_v3_graph,
)
from V4.MHD_Framework_V4 import MHD_Edge


def make_v3_graph():
    module = nn.Linear(1, 1, bias=False)
    nodes = {
        V3_Node(0, "input", torch.zeros(1, 1), transfer_mode="sum"),
        V3_Node(1, "loss", torch.zeros(1, 1)),
    }
    edge = V3_Edge(0, "linear", [module])
    role = torch.tensor([[-1, 1]])
    sort = torch.tensor([[0, 1]])
    return V3_Graph(nodes, {edge}, {V3_Topo([role], [sort])}, device="cpu"), module


def test_live_v3_graph_migration_wraps_messages_operations_and_global_levels():
    v3, _ = make_v3_graph()
    graph = migrate_v3_graph(v3, device=torch.device("cpu"))
    assert graph.num_levels == 2
    assert default_v4_level_sequences(1) == {
        "forward_levels": [0], "backward_levels": [1]
    }
    migrated = graph.get_node_by_name("input")
    assert migrated.aggregation == "sum"
    assert not hasattr(migrated, "initial_state")
    assert isinstance(
        graph.get_edge_by_name("linear").edge_operations[0], MHD_Edge.Operation
    )
    torch.testing.assert_close(graph.topo.role_matrices[1], -graph.topo.role_matrices[0])


def test_v3_checkpoint_migration_writes_four_message_states_and_paths(tmp_path):
    v3, module = make_v3_graph()
    graph = migrate_v3_graph(v3, device=torch.device("cpu"))
    source = tmp_path / "v3"
    source.mkdir()
    torch.save(
        {
            "node_initial_states": {
                "input": torch.full((1, 1), 2.0),
                "loss": torch.full((1, 1), 3.0),
            },
            "node_current_states": {
                "input": torch.full((1, 1), 4.0),
                "loss": torch.full((1, 1), 5.0),
            },
        },
        source / "node_best.pth",
    )
    torch.save(
        {"edge_params": {"linear": [module.state_dict()]}},
        source / "edge_best.pth",
    )
    torch.save({"epoch": 7, "history": {}, "optimizer": {}}, source / "meta_best.pth")
    destination = tmp_path / "v4"
    report = migrate_v3_checkpoint(
        str(source), graph, str(destination), include_current_state=True
    )
    assert report["forward_levels"] == [0]
    assert report["backward_levels"] == [1]
    assert (destination / "best" / ".metadata").is_file()
    assert (destination / "migration_report.json").is_file()
    input_node = graph.get_node_by_name("input")
    torch.testing.assert_close(
        input_node.feature_message.current_state, torch.full((1, 1), 4.0)
    )
    torch.testing.assert_close(
        input_node.gradient_message.current_state, torch.zeros(1, 1)
    )

