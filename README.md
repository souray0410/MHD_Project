# MHD Project

> **Multi-Hypergraph Dynamic Project**
> A dynamic hypergraph-based framework for state-driven neural computation.

**Current Release: V3**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-MHD_Project-black?logo=github)](https://github.com/souray0410/MHD_Project)

**Latest Release:** [V3](https://github.com/souray0410/MHD_Project/releases/latest)

---

# 中文

## 1. 项目简介

**MHD Project（Multi-Hypergraph Dynamic Project）** 是一个面向深度学习计算结构的动态超图框架。

MHD 的核心思想是将神经网络计算抽象为两个相互对称的基本组成部分：

* **State（状态）**：表示计算过程中节点所承载的数据状态。
* **Computation（计算）**：表示作用于一个或多个 State 的计算过程。

因此，MHD 的核心计算可以抽象为：

```text
State
  │
  ▼
Computation
  │
  ▼
State
```

多个 State 和 Computation 通过显式的 Topology 组织起来，形成完整的动态计算图：

```text
                 Computation
              ┌──────────────┐
              │      C1      │
              └──────┬───────┘
                     │
                     ▼
                   State
                  /     \
                 /       \
                ▼         ▼
        Computation      Computation
             C2              C3
              │               │
              ▼               ▼
            State           State
```

MHD 的目标不是简单地重新实现传统神经网络，而是建立一种更加明确的结构化表示：

> **State 定义计算对象，Computation 定义计算过程，Topology 定义计算结构与执行关系。**

---

## 2. 核心抽象

### 2.1 State

在 MHD 中，`State` 是整个计算系统中的基本信息载体。

对于深度学习网络而言，State 可以承载：

* 中间特征；
* 隐状态；
* 输入状态；
* 输出状态；
* 不同计算阶段产生的动态表示。

V3 中进一步区分：

```text
initial_state
current_state
```

其中：

* `initial_state` 表示状态的初始值；
* `current_state` 表示当前计算过程中正在演化的状态。

因此，State 不再只是一个静态 Tensor，而是一个具有生命周期的计算对象。

---

### 2.2 Computation

`Computation` 是 MHD 中对实际计算过程的统一抽象。

在超图表示中，一个 Computation 可以同时连接多个输入 State 和多个输出 State：

```text
State A ─┐
         │
State B ─┼──► Computation C ───► State D
         │
State E ─┘
```

因此，一个 Computation 可以表达：

* Convolution
* Normalization
* Activation
* Pooling
* Projection
* Feature fusion
* Tensor transformation
* 其他自定义计算操作

在代码层面，V2 / V3 通过 `MHD_Edge` 对这一计算单元进行实现。

---

### 2.3 Topology

Topology 描述 State 与 Computation 之间的结构关系，并决定：

* 哪些 State 参与某个 Computation；
* State 在 Computation 中的输入/输出角色；
* 参数顺序；
* Computation 的依赖关系；
* Computation 的执行顺序；
* 不同执行阶段之间的关系。

因此：

```text
State         → What is being computed
Computation  → How it is computed
Topology     → Where and when it is computed
```

---

## 3. 动态状态计算

MHD 与传统静态计算图的重要区别之一，是 State 可以随着 Computation 不断更新。

V3 中，一个典型的计算过程可以抽象为：

```text
Initial State
      │
      ▼
Current State
      │
      ▼
Computation
      │
      ▼
Incoming State
      │
      ▼
State Transition
      │
      ▼
Updated State
      │
      ▼
Next Topology Level
```

V3 的 State 支持多种状态融合方式：

```text
replace
sum
avg
max
min
mul
```

这使得状态更新本身成为计算结构的一部分。

---

# 4. Version Evolution

MHD Project 并不是三个相互独立的项目。

**V1、V2、V3 是同一套核心思想不断形式化、抽象化和扩展的三个阶段。**

整体演化路线为：

```text
V1
│
├── State-oriented computation
├── Computation-oriented network structure
└── Node-based experimental framework
        │
        ▼
V2
│
├── Formal Hypergraph abstraction
├── Explicit State
├── Explicit Computation
└── Explicit Topology
        │
        ▼
V3
│
├── Multi-level Topology
├── Stateful execution
├── State transition
├── Flexible Computation
└── Graph composition
```

---

# 5. V1 — Node-based Experimental Framework

V1 是 MHD Project 的起始版本。

这一阶段的核心目标是：

> **将传统神经网络中的计算过程从单纯的 sequential layer structure 转换为 State / Computation oriented structure。**

V1 的主要代码由：

```text
V1/
├── node_toolkit/
└── node_pipline/
```

组成。

### `node_toolkit`

负责早期 MHD 网络中的底层组件，包括：

* Network construction
* Dataset utilities
* Result management
* Training utilities
* Dynamic computational modules

### `node_pipline`

负责实验、训练和测试流程，包括不同版本的 UniConnNet 实验。

V1 建立了 MHD 最早期的核心设计：

```text
State
   ↓
Computation
   ↓
State
```

并开始将复杂网络中的连接关系从普通 layer stack 中分离出来。

---

# 6. V2 — Formal Hypergraph Framework

V2 对 V1 的设计进行进一步抽象，将原本的实验型网络结构正式组织为通用的 Hypergraph Framework。

V2 的核心组件为：

```text
MHD_Node
MHD_Edge
MHD_Topo
MHD_Graph
```

对应到统一 MHD abstraction：

```text
MHD_Node
   ↓
State

MHD_Edge
   ↓
Computation

MHD_Topo
   ↓
Topology

MHD_Graph
   ↓
Complete Computational Graph
```

V2 进一步引入：

```text
role_matrix
sort_matrix
```

用于显式描述：

* State 与 Computation 的连接关系；
* 输入/输出角色；
* Computation 的执行顺序；
* Topological dependencies。

V2 因此完成了从：

> **Experimental Node Framework**

到：

> **Formal Hypergraph Computational Framework**

的转变。

V2 位于：

```text
V2/
└── mhd_toolkit/
    ├── MHD_Framework_V2.py
    ├── MHD_Utils_V2.py
    └── MHD_Example_V2.py
```

---

# 7. V3 — Multi-Level Dynamic Hypergraph Framework

**V3 是当前 MHD Project 的核心版本，也是当前 Release 的版本。**

V3 保留 V1 / V2 的统一抽象：

```text
State
Computation
Topology
Graph
```

同时进一步扩展为：

```text
State-aware
Multi-level
Dynamic
Composable
Hypergraph
```

---

## 7.1 Multi-Level Topology

V3 将单一 Topology 扩展为多 Level Topology：

```text
Level 0
   │
   ▼
Computation
   │
   ▼
State Update
   │
   ▼
Level 1
   │
   ▼
Computation
   │
   ▼
State Update
   │
   ▼
Level 2
   │
   ▼
...
```

对应代码结构：

```text
role_matrices
sort_matrices
```

每一个 Level 可以拥有独立的：

* Role Matrix
* Sort Matrix
* Computation dependency
* Execution order

因此 V3 可以表达比传统单层 DAG 更复杂的计算结构。

---

## 7.2 Stateful Execution

V3 中的 State 明确拥有：

```text
initial_state
current_state
```

并通过 State Transition 将 Computation 的输出继续传播到后续计算阶段。

这使得图结构不只是一个静态连接关系，而是一个具有动态状态演化能力的计算系统。

---

## 7.3 Flexible Computation

V3 中的 Computation 可以由多个操作组成：

```text
Computation C
     │
     ├── Operation 1
     ├── Operation 2
     ├── Operation 3
     └── ...
```

这些 Operation 可以由 PyTorch Modules 或其他 callable objects 构成。

因此，一个 Computation 可以表示一个完整的计算 block，而不仅仅是一个单一 layer。

---

## 7.4 Graph Composition

V3 进一步支持多个 Graph 的组合。

多个子图可以被组合为更大的 MHD Graph：

```text
Graph A
   │
   ├────────┐
   │        │
   ▼        ▼
Graph B   Graph C
   │        │
   └────┬───┘
        ▼
   MHD Graph
```

在组合过程中，V3 可以对 State、Computation 和 multi-level Topology 进行相应的对齐与合并。

这使 MHD Graph 可以进一步作为模块化计算单元进行复用。

---

# 8. Unified MHD Model

V1、V2、V3 的共同核心可以统一描述为：

```text
              MHD Graph
                  │
        ┌─────────┴─────────┐
        │                   │
      State             Computation
        │                   │
        └─────────┬─────────┘
                  │
               Topology
                  │
                  ▼
          Execution Process
                  │
                  ▼
            State Update
                  │
                  ▼
             Next Level
```

因此，MHD 的基本计算关系可以概括为：

```text
State → Computation → State
```

而完整网络则由多个这样的状态转换和计算关系组成。

---

# 9. Project Structure

当前项目的正式组织结构为：

```text
MHD_Project/
│
├── README.md
├── LICENSE
│
├── V1/
│   ├── node_toolkit/
│   │   ├── node_dataset.py
│   │   ├── node_net.py
│   │   ├── node_results.py
│   │   ├── node_utils.py
│   │   ├── reorgan.py
│   │   └── split_Tr.py
│   │
│   └── node_pipline/
│       ├── node_train.py
│       ├── train_UniConnNetI.py
│       ├── train_UniConnNetII.py
│       ├── train_UniConnNetIII.py
│       ├── test_UniConnNetI.py
│       ├── test_UniConnNetII.py
│       ├── test_UniConnNetIII.py
│       └── ...
│
├── V2/
│   └── mhd_toolkit/
│       ├── MHD_Framework_V2.py
│       ├── MHD_Utils_V2.py
│       └── MHD_Example_V2.py
│
├── V3/
│   └── mhd_toolkit/
│       ├── MHD_Framework_V3.py
│       └── MHD_Utils_V3.py
│
├── contrast_model/
│
├── performance/
│
└── tools/
```

> **Note:** V1 preserves the original experimental framework, while V2 and V3 use the more formal `mhd_toolkit` organization.

---

# 10. Installation

## Requirements

Recommended environment:

```text
Python 3.8+
PyTorch 1.10+
CUDA recommended for large-scale experiments
```

Common dependencies include:

```text
numpy
pandas
scipy
scikit-learn
nibabel
opencv-python
tabulate
tqdm
```

For reproducible experiments, use the dependency configuration provided with the corresponding release.

---

## Clone the Repository

```bash
git clone https://github.com/souray0410/MHD_Project.git
cd MHD_Project
```

Install the dependencies required by the selected version.

---

# 11. Latest Release

The **current release is V3**.

For stable and reproducible usage, please use the latest tagged release:

**[Download the latest V3 release](https://github.com/souray0410/MHD_Project/releases/latest)**

All releases are available at:

**[MHD Project Releases](https://github.com/souray0410/MHD_Project/releases)**

New development is primarily focused on **V3**.

---

# 12. Usage

## V1

V1 experiments are located under:

```text
V1/
├── node_toolkit/
└── node_pipline/
```

The original training and experimental pipelines can be executed from `node_pipline`.

---

## V2

V2 provides the formal MHD framework:

```python
from V2.mhd_toolkit.MHD_Framework_V2 import (
    MHD_Node,
    MHD_Edge,
    MHD_Topo,
    MHD_Graph,
)
```

---

## V3

V3 is recommended for new development:

```python
from V3.mhd_toolkit.MHD_Framework_V3 import (
    MHD_Node,
    MHD_Edge,
    MHD_Topo,
    MHD_Graph,
)
```

A conceptual V3 workflow is:

```python
state = MHD_Node(
    id=0,
    name="state_0",
    initial_state=input_tensor,
)

computation = MHD_Edge(
    id=0,
    name="computation_0",
    edge_operations=[
        ...
    ],
)

topology = MHD_Topo(
    role_matrices=[
        role_matrix_level_0,
        role_matrix_level_1,
    ],
    sort_matrices=[
        sort_matrix_level_0,
        sort_matrix_level_1,
    ],
)

graph = MHD_Graph(
    nodes={state},
    edges={computation},
    topos={topology},
)

graph.forward()
```

The exact configuration depends on the target architecture.

---

# 13. Design Principles

MHD is designed around four core principles.

### 1. State

The fundamental information unit of the system is a dynamic **State**.

### 2. Computation

All transformations are represented through explicit **Computations**.

### 3. Topology

The relationship and execution order between State and Computation are explicitly represented by **Topology**.

### 4. Composition

Complex networks can be constructed from smaller MHD computational structures.

Together:

```text
State + Computation + Topology
            ↓
        MHD Graph
```

---

# 14. Current Status

**V3 is the current core implementation and the current release of the MHD Project.**

The evolution of the framework can be summarized as:

```text
V1
State-oriented experimental framework
        ↓
V2
Formal hypergraph computational framework
        ↓
V3
Multi-level dynamic state-computation framework
```

V1 provides the original experimental foundation.

V2 formalizes the core hypergraph abstraction.

V3 extends the abstraction toward multi-level, state-aware, composable dynamic computation.

---

# 15. Roadmap

Future development may include:

* stronger mathematical formalization of the MHD abstraction;
* more expressive State Transition mechanisms;
* richer Computation operators;
* graph serialization and reconstruction;
* configuration-driven architecture generation;
* automated architecture construction;
* systematic benchmarks;
* advanced topology visualization;
* broader applications beyond the original medical-imaging setting.

---

# 16. Contributing

Contributions, issues, and discussions are welcome.

For substantial architectural changes, please describe:

1. the motivation;
2. the affected State / Computation / Topology abstraction;
3. compatibility with previous versions;
4. changes to execution semantics;
5. experimental or benchmark results where applicable.

---

# 17. License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.

---

# 18. Author

**孟号丁**

GitHub: [souray0410](https://github.com/souray0410)

---

# English

## 1. Overview

**MHD Project (Multi-Hypergraph Dynamic Project)** is a dynamic hypergraph-based framework for structured neural computation.

The framework is built around two symmetric computational primitives:

* **State** — the information carried by the computational graph.
* **Computation** — the transformation applied to one or multiple States.

The fundamental relationship is:

```text
State
  │
  ▼
Computation
  │
  ▼
State
```

Multiple States and Computations are organized by an explicit **Topology**, which defines the connectivity and execution structure of the complete system.

---

## 2. Core Abstraction

### State

A State represents a computational state carried by the graph.

In V3, the State abstraction explicitly distinguishes:

```text
initial_state
current_state
```

allowing the computational state to evolve during execution.

---

### Computation

A Computation represents the transformation performed on one or multiple States.

A single Computation may connect multiple inputs and multiple outputs:

```text
State A ─┐
State B ─┼──► Computation ───► State C
State D ─┘
```

Computations may internally contain multiple operations, such as convolution, normalization, activation, aggregation, projection, or other tensor transformations.

---

### Topology

Topology describes:

* connectivity;
* input/output roles;
* parameter ordering;
* dependency relationships;
* execution order;
* multi-level execution structure.

The resulting abstraction is:

```text
State         → computational state
Computation  → computational transformation
Topology     → computational structure
Graph        → complete computational system
```

---

# 3. Stateful Dynamic Computation

A major objective of MHD is to make state evolution an explicit part of computation.

A typical process can be represented as:

```text
Initial State
      │
      ▼
Current State
      │
      ▼
Computation
      │
      ▼
State Transition
      │
      ▼
Updated State
      │
      ▼
Next Topology Level
```

V3 currently provides multiple state-transfer modes:

```text
replace
sum
avg
max
min
mul
```

---

# 4. Evolution: V1 → V2 → V3

V1, V2, and V3 are successive stages of the same MHD architecture.

```text
V1
│
├── State-oriented experimental framework
└── Computation-oriented network structure
        │
        ▼
V2
│
├── Formal State / Computation abstraction
├── Explicit Topology
└── Hypergraph computational framework
        │
        ▼
V3
│
├── Multi-level Topology
├── Stateful execution
├── State transition
├── Flexible Computation
└── Graph composition
```

---

# 5. V1 — Node-based Experimental Framework

V1 established the original MHD concept.

The V1 codebase is organized as:

```text
V1/
├── node_toolkit/
└── node_pipline/
```

The original design introduced the idea of separating computational information from computational transformations and organizing them through explicit structural relationships.

V1 served as the experimental foundation for the later formal MHD framework.

---

# 6. V2 — Formal Hypergraph Framework

V2 transformed the original experimental design into a formal framework based on:

```text
MHD_Node
MHD_Edge
MHD_Topo
MHD_Graph
```

This version introduced explicit:

```text
role_matrix
sort_matrix
```

structures for describing graph connectivity and execution order.

V2 therefore represented the transition from an experimental node-based network implementation to a general hypergraph computational framework.

---

# 7. V3 — Multi-Level Dynamic Hypergraph Framework

**V3 is the current core implementation and current release.**

V3 retains the common MHD abstraction while extending it through:

* multi-level Topology;
* explicit State transitions;
* dynamic State management;
* flexible Computation blocks;
* level-wise execution;
* graph composition.

The core abstraction is:

```text
State
   +
Computation
   +
Topology
   ↓
MHD Graph
```

---

## 7.1 Multi-Level Topology

V3 supports multiple execution levels:

```text
Level 0
   ↓
Computation
   ↓
State Update
   ↓
Level 1
   ↓
Computation
   ↓
State Update
   ↓
Level 2
   ↓
...
```

Each level may define its own:

```text
role_matrix
sort_matrix
```

allowing independent dependency analysis and execution scheduling.

---

## 7.2 Stateful Execution

V3 explicitly maintains:

```text
initial_state
current_state
```

and updates current state during graph execution.

This enables a dynamic computational interpretation in which States evolve through a sequence of Computations and Topology levels.

---

## 7.3 Flexible Computations

A Computation can contain a sequence of operations:

```text
Computation
   │
   ├── Operation 1
   ├── Operation 2
   ├── Operation 3
   └── ...
```

This provides a flexible mechanism for representing complex computational blocks inside the hypergraph.

---

## 7.4 Graph Composition

V3 also supports combining multiple MHD graphs into larger computational structures.

This enables modular construction of complex architectures from smaller graph components.

---

# 8. Project Structure

```text
MHD_Project/
│
├── README.md
├── LICENSE
│
├── V1/
│   ├── node_toolkit/
│   └── node_pipline/
│
├── V2/
│   └── mhd_toolkit/
│       ├── MHD_Framework_V2.py
│       ├── MHD_Utils_V2.py
│       └── MHD_Example_V2.py
│
├── V3/
│   └── mhd_toolkit/
│       ├── MHD_Framework_V3.py
│       └── MHD_Utils_V3.py
│
├── contrast_model/
├── performance/
└── tools/
```

The first version is preserved under `V1/` as the historical foundation of the project.

V2 provides the first formal MHD toolkit.

V3 is the current core framework.

---

# 9. Installation

```bash
git clone https://github.com/souray0410/MHD_Project.git
cd MHD_Project
```

Recommended environment:

```text
Python 3.8+
PyTorch 1.10+
CUDA recommended for large-scale experiments
```

Install dependencies using the configuration associated with the selected release.

---

# 10. Latest Release

The **latest release is V3**.

**[Download V3 / Latest Release](https://github.com/souray0410/MHD_Project/releases/latest)**

**[View All Releases](https://github.com/souray0410/MHD_Project/releases)**

New development is primarily focused on V3.

---

# 11. V3 Quick Start

```python
from V3.mhd_toolkit.MHD_Framework_V3 import (
    MHD_Node,
    MHD_Edge,
    MHD_Topo,
    MHD_Graph,
)
```

A conceptual MHD graph can be constructed from:

```text
State
Computation
Topology
```

and executed through:

```python
graph.forward()
```

---

# 12. Design Principles

MHD follows four fundamental principles:

### State

The graph carries explicit computational States.

### Computation

Transformations are represented as explicit Computations.

### Topology

Connectivity and execution order are explicitly represented.

### Composition

Complex graphs can be constructed from smaller computational structures.

The unified abstraction is:

```text
State + Computation + Topology
            ↓
        MHD Graph
```

---

# 13. Current Status

**V3 is the current core framework and current release.**

The project evolution is summarized as:

```text
V1
State-oriented experimental framework
        ↓
V2
Formal hypergraph computational framework
        ↓
V3
Multi-level dynamic state-computation framework
```

---

# 14. Roadmap

Potential future directions include:

* formal mathematical characterization of the MHD abstraction;
* richer State Transition mechanisms;
* more expressive Computation operators;
* graph serialization;
* graph reconstruction;
* automated architecture generation;
* configuration-driven graph construction;
* systematic benchmarks;
* advanced graph analysis and visualization;
* applications beyond the original medical-imaging setting.

---

# 15. License

MIT License.

See [LICENSE](LICENSE) for details.

---

# 16. Author

**Haoding Souray Meng**

GitHub: [souray0410](https://github.com/souray0410)
