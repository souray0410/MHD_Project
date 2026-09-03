# MHD V1 — Experimental Node/Hyperedge Networks

[中文](#中文) · [English](#english)

## 中文

V1 是 MHD Project 的历史原型。它还没有形成 V2 之后统一的 `MHD_Node`、`MHD_Edge`、`MHD_Topo`、`MHD_Graph` 四类接口，而是通过三个可训练网络类探索 Node 与 Hyperedge 组织方式：

- `DNet`：可配置的卷积计算单元；
- `HDNet`：用超边配置连接多个局部 Node；
- `MHDNet`：组合多个 `HDNet` 子网络，并进行全局 Node 映射。

本目录只保留 V1 的两个主要源码文件：

```text
V1/
├── MHD_Framework_V1.py   # 原 node_net.py，代码内容保持不变
├── MHD_Utils_V1.py       # 原 node_utils.py，代码内容保持不变
└── README.md
```

原始数据处理、训练 pipeline、对照模型与辅助脚本完整保存在 [`../experiments/V1/legacy`](../experiments/V1/legacy)，以便历史复现。整理仓库时只调整了文件位置和主文件名称，没有重写 V1 类或函数。

源码方式导入：

```python
from V1.MHD_Framework_V1 import DNet, HDNet, MHDNet
from V1.MHD_Utils_V1 import train, validate, test
```

V1 的 Framework 主要依赖 PyTorch；Utils 还使用 NumPy 与 tabulate。历史实验包含医学影像和数据处理依赖，以 legacy 脚本中的实际 import 为准。

新项目建议使用 V4；V1 主要用于理解 MHD 从网络原型走向正式超图抽象的起点。

## English

V1 is the historical prototype of MHD Project. It predates the unified `MHD_Node`, `MHD_Edge`, `MHD_Topo`, and `MHD_Graph` API introduced in V2. Its central classes explore node- and hyperedge-oriented neural construction:

- `DNet`: a configurable convolutional computation unit;
- `HDNet`: a hyperedge-configured local node network;
- `MHDNet`: a composition of multiple `HDNet` subnetworks with global node mapping.

The two principal source files are kept in this directory under standardized names. Their code is unchanged from the historical repository. Original datasets, training pipelines, comparison models, and helper scripts are preserved under [`../experiments/V1/legacy`](../experiments/V1/legacy).

```python
from V1.MHD_Framework_V1 import DNet, HDNet, MHDNet
from V1.MHD_Utils_V1 import train, validate, test
```

Use V4 for new projects. V1 is retained to document and reproduce the experimental origin of the MHD abstraction.
