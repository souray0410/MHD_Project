# MHD V2 — Formal Hypergraph Framework

[中文](#中文) · [English](#english)

## 中文

V2 将 V1 的网络原型正式抽象为四个超图核心：

- `MHD_Node`：保存 Initial State 与 Current State；
- `MHD_Edge`：保存顺序计算操作；
- `MHD_Topo`：使用 Role Matrix 与 Sort Matrix 描述 Node–Edge 关系；
- `MHD_Graph`：注册参数并执行完整超图。

这是四个核心名称第一次形成稳定结构的版本。Role Matrix 使用 `-1/0/1` 表示输入、无连接和输出角色；Sort Matrix 记录同一 Edge 内 Node 的顺序。

本目录内容：

```text
V2/
├── MHD_Framework_V2.py
├── MHD_Utils_V2.py
└── README.md
```

原始 `MHD_Example_V2.py` 已归位到 [`../examples/V2`](../examples/V2)，源码内容保持不变。

```python
from V2.MHD_Framework_V2 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
```

V2 Framework 依赖 PyTorch 与 NumPy。V2 Utils 还直接依赖 OpenCV、NiBabel、tqdm 等数据和训练工具。V2 是历史版本，不会静默采用 V3/V4 的接口；请以本目录代码为准。

相对 V1，V2 的关键变化不是增加某一种网络，而是把 Node、Edge、Topology 和 Graph 从具体实验中抽离出来，形成可复用的数据结构。

## English

V2 formalizes the V1 prototype into four hypergraph components:

- `MHD_Node` stores initial and current state;
- `MHD_Edge` stores an ordered computation sequence;
- `MHD_Topo` describes node–edge roles and order with Role and Sort matrices;
- `MHD_Graph` registers parameters and executes the complete hypergraph.

This is the first version with the stable four-name core used by later releases. The historical example is located in [`../examples/V2`](../examples/V2); source code has not been rewritten during repository reorganization.

```python
from V2.MHD_Framework_V2 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
```

V2 remains a historical version and does not silently inherit V3 or V4 behavior. Consult the V2 source when reproducing earlier work.
