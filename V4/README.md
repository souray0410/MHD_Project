# MHD V4 — Bidirectional Message Hypergraph

[中文](#中文) · [English](#english) · [完整中文技术说明](../docs/V4_GUIDE.zh-CN.md)

## 中文

V4 是当前源码版本。它保持 `MHD_Node`、`MHD_Edge`、`MHD_Topo`、`MHD_Graph` 四个顶层核心，并通过两个嵌套类型把状态与计算接口进一步统一：

- `MHD_Node.Message`
- `MHD_Edge.Operation`

V4 不是大模型专用框架。卷积网络、Transformer、图/超图网络和普通自定义 `nn.Module` 都使用同一套 Node–Edge–Topo–Graph 表达。

### V3 → V4

| 项目 | V3 | V4 |
|---|---|---|
| Node 状态 | `initial_state/current_state` | `feature_message` 与 `gradient_message`，各含 Initial/Current State |
| Node 合并 | `transfer_mode` | `aggregation`，确定性 n 元 Message 合并 |
| Edge 操作 | 裸对象 | `MHD_Edge.Operation(...)` 包装，字段仍叫 `edge_operations` |
| Topo | 多 level 前向 | 一套全局 Role/Sort levels，由 Forward/Backward 显式选择 |
| Forward | 可省略 levels | `graph.forward(levels=[...])` |
| Backward | 对 loss Tensor 调用 `.backward()` | `graph.backward(levels=[...])`，内部仍是一次原生 autograd |
| 梯度状态 | 不公开在 Node | `node.gradient_message.current_state` |

levels 按列表原样执行，不排序、不去重，允许不连续和重复。同一轮 Forward 与 Backward 使用互不重叠的 level 序列；反向 level 必须能够匹配本次 Forward 的真实 trace。

### 目录内容

```text
V4/
├── MHD_Framework_V4.py
├── MHD_Utils_V4.py
├── MHD_Compatibility_V3_to_V4.py
└── README.md
```

Framework 与 Utils 以当前电脑里的最新文件为正式来源，整理仓库时不修改其代码内容。兼容脚本是 V3→V4 的正式接入工具，因此保留在本目录。

### 基本接口

```python
from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo

node = MHD_Node(
    id=0,
    name="input",
    feature_message=MHD_Node.Message(initial_state=tensor),
    aggregation="replace",
)

edge = MHD_Edge(
    id=0,
    name="computation",
    edge_operations=[MHD_Edge.Operation(module)],
)
```

```python
graph.forward(levels=forward_levels)
graph.backward(levels=backward_levels)

feature = node.feature_message.current_state
gradient = node.gradient_message.current_state
```

特殊梯度应在 Operation 中通过标准 `torch.autograd.Function` 定义，不增加第二套自定义反向系统。

### Trainer Criteria

`MHD_Trainer` 要求用户提供 `criteria(graph)`，用于从完整验证状态计算保存最佳 checkpoint 的标量。Criteria 与反向起点不同：反向起点由本次 Forward 的唯一可微标量终点自动确定。

```python
@torch.no_grad()
def validation_loss(graph):
    loss = graph.get_node_by_name("loss").feature_message.current_state
    return loss.float().mean()
```

Criteria 可以根据任务读取多个 Node，计算 loss、accuracy、AUC、F1 或其他数据集级指标。

### 兼容与实验

- 完整设计、接口变化、并行范围与实测边界见[详细 V4 中文技术说明](../docs/V4_GUIDE.zh-CN.md)；
- V3 live graph 与 checkpoint 迁移由 `MHD_Compatibility_V3_to_V4.py` 完成；
- RETFound ViT-L/16 2D 分阶段实验位于 [`../experiments/V4/retfound_2d`](../experiments/V4/retfound_2d)；
- 多卡配置位于 Utils，不改变四个超图核心概念。

## English

V4 is the current source version. It preserves the four top-level concepts—`MHD_Node`, `MHD_Edge`, `MHD_Topo`, and `MHD_Graph`—and introduces two nested helpers: `MHD_Node.Message` and `MHD_Edge.Operation`.

Nodes expose symmetric Feature and Gradient Messages. Edges contain wrapped Operations. A single global list of Role and Sort matrices defines every level, while explicit forward and backward level sequences select the execution path. PyTorch still performs the actual tensor computation and one native autograd pass.

```python
node = MHD_Node(
    id=0,
    name="input",
    feature_message=MHD_Node.Message(initial_state=tensor),
    aggregation="replace",
)

edge = MHD_Edge(
    id=0,
    name="computation",
    edge_operations=[MHD_Edge.Operation(module)],
)

graph.forward(levels=forward_levels)
graph.backward(levels=backward_levels)
```

The user-supplied level order is preserved exactly; levels may be non-contiguous or repeated. Forward and backward levels must not overlap within one round, and a reverse level must match the real forward trace.

`MHD_Trainer` requires a task-specific `criteria(graph)` callable for best-checkpoint selection. The callable may combine multiple validation nodes and is independent of the automatically inferred differentiable scalar used to start backward.

See the [detailed V4 guide](../docs/V4_GUIDE.zh-CN.md) for migration, lifecycle, parallel execution, validation, and performance notes. Use [`MHD_Compatibility_V3_to_V4.py`](MHD_Compatibility_V3_to_V4.py) for V3 migration.
