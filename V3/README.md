# MHD V3 — Multi-Level Dynamic Hypergraph

[中文](#中文) · [English](#english)

## 中文

V3 以 V2 的四个核心为基础，形成多 level、带状态生命周期的动态超图框架。本目录代码以当前电脑中的最新 V3 为正式来源，不使用 GitHub 旧副本覆盖。

主要变化：

- Node 使用 `initial_state` 与 `current_state` 表达状态生命周期；
- `transfer_mode` 支持 `replace/sum/avg/max/min/mul`；
- Edge 使用 `edge_operations` 保存字符串、`nn.Module`、`partial` 或 callable；
- Topo 从单一矩阵扩展为 `role_matrices/sort_matrices`；
- `graph.forward(levels=...)` 可以选择执行 level；
- Utils 增加 Dataset、Monitor、Trainer、Inferencer、Checkpoint、Prune 与基础 DDP 支持。

```text
V3/
├── MHD_Framework_V3.py
├── MHD_Utils_V3.py
└── README.md
```

基本导入：

```python
from V3.MHD_Framework_V3 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V3.MHD_Utils_V3 import MHD_Monitor, MHD_Trainer
```

Node 示例：

```python
node = MHD_Node(
    id=0,
    name="input",
    initial_state=tensor,
    current_state=None,
    transfer_mode="replace",
)
value = node.current_state
```

V3 的直接状态字段与裸 `edge_operations` 是该版本的正式接口。V4 对这些接口进行了有意调整；迁移时使用 [`../V4/MHD_Compatibility_V3_to_V4.py`](../V4/MHD_Compatibility_V3_to_V4.py)，不要在 V3 中混用 V4 写法。

## English

V3 extends the four-part V2 core into a stateful, multi-level dynamic hypergraph framework. The source in this directory is the authoritative V3 copy from the current local project.

Key developments include:

- node `initial_state` and `current_state` lifecycle;
- configurable `transfer_mode`;
- flexible `edge_operations`;
- multi-level `role_matrices` and `sort_matrices`;
- selectable `graph.forward(levels=...)` execution;
- dataset, monitoring, training, inference, checkpoint, pruning, and basic DDP utilities.

```python
from V3.MHD_Framework_V3 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
from V3.MHD_Utils_V3 import MHD_Monitor, MHD_Trainer
```

Direct state fields and unwrapped edge operations are intentional V3 APIs. Use the V4 compatibility script when migrating instead of mixing V3 and V4 syntax.
