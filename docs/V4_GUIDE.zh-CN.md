# MHD Framework V4 — 完整技术说明

MHD V4 仍从超图视角描述深度学习：Node 保存 Message，Edge 保存
Operation，Topo 用 Role/Sort Matrix 描述连接与位置顺序，Graph 按用户给出的
全局 level 序列执行。它不是大模型专用框架；卷积网络、Transformer、图网络、
超图网络和普通自定义 `nn.Module` 使用同一套接口。

V4 只有四个顶层理论核心：

- `MHD_Node`
- `MHD_Edge`
- `MHD_Topo`
- `MHD_Graph`

V4 只增加两个嵌套辅助类型：`MHD_Node.Message` 与
`MHD_Edge.Operation`。并行配置、Trainer 和内部 PyTorch Adapter 都在 Utils 中，
不改变这四个核心概念。

## V3 到 V4 的直接变化

| 项目 | V3 | V4 |
|---|---|---|
| 顶层核心 | Node、Edge、Topo、Graph | 不变 |
| Node 状态 | 直接 `initial_state/current_state` | `feature_message` 与 `gradient_message`，各含 Initial/Current State |
| Node 合并 | `transfer_mode`，输出到达时逐次二元修改 | 单一 `aggregation`，同一 Node 的待处理 Message 做确定性 n 元合并 |
| Gradient | 不在 Node 中公开 | `gradient_message.current_state` 保存本次真实 autograd 梯度 |
| Edge 操作 | 裸字符串、Module、`partial` 或 callable | 每项统一包装成 `MHD_Edge.Operation(...)`，字段仍叫 `edge_operations` |
| Operation 调用 | 捕获异常后猜测解包、列表或逐项广播 | Sort Matrix 排序后确定性位置参数调用，模型内部异常原样抛出 |
| 自定义梯度 | 直接依赖 PyTorch | 仍依赖 PyTorch；在 Operation 内使用标准 `torch.autograd.Function.apply` |
| Topo | Role/Sort 用于 Forward | 仍只有一套全局 `role_matrices/sort_matrices`，两种执行方向显式选择不同 level |
| level | Forward 可默认全部执行 | Forward/Backward 均必须显式给出非空序列；不排序、不去重 |
| Forward 更新 | Edge 输出立即修改 Node | 同一 Node 的输入先缓冲，在读取前或 level 结束时统一 aggregation |
| Backward 根 | 用户取得 loss 后调用 Tensor `.backward()` | `graph.backward(levels=[...])` 自动寻找本次 Forward 唯一可微标量终点 |
| Backward 路径 | 完整 PyTorch 图 | level 序列选择真实 Forward trace 的反向子路径；底层仍只做一次原生 autograd |
| 参数梯度 | 原生 autograd | 仍是原生 autograd；未选路径被 hook 屏蔽，未选参数 `.grad=None` |
| Trainer | `backward_node`；Criteria 可默认复用它 | 自动寻找反向标量终点；接受外部 `criteria(graph)`；保存显式 `forward_levels/backward_levels` |
| Checkpoint | 两个 Node State | 两类 Message 下四个 State，并保存 Trainer 的 level 序列 |
| Merge | 合并两个 State | 分别合并四个 State，并检查 aggregation、Operation 与全局 Topo |
| Monitor | 默认读取 `current_state` | 旧指标名不变，可选监控四种 Message State |
| 多卡 | 基础 distributed 工具 | Utils 独立支持 DDP、FSDP2、TP、PP；不组合多个并行族 |

保持不变的概念和名称包括：Node/Edge/Topo/Graph、`edge_operations`、Node/Edge
ID 与名称查询、Role Matrix 的 `-1/0/1`、Sort Matrix、普通 `nn.Module` 参数与
原生 optimizer。

有意破坏的接口包括：Node 直接状态字段、`transfer_mode`、裸
`edge_operations`、无参数的 `graph.forward()`、直接对 Node loss Tensor 调用
`.backward()`，以及 Trainer 的 `backward_node`。

### 并排迁移

```python
# V3
node = MHD_Node(id, name, initial_state, current_state, transfer_mode="sum")
value = node.current_state
edge = MHD_Edge(edge_id, edge_name, [module])
graph.forward()
loss_node.current_state.mean().backward()
```

```python
# V4
node = MHD_Node(
    id=id,
    name=name,
    feature_message=MHD_Node.Message(initial_state, current_state),
    aggregation="sum",
)
value = node.feature_message.current_state
edge = MHD_Edge(
    id=edge_id,
    name=edge_name,
    edge_operations=[MHD_Edge.Operation(module)],
)
graph.forward(levels=forward_levels)
graph.backward(levels=backward_levels)
```

## Message、Operation 与 aggregation

Feature Message 与 Gradient Message 的公开结构严格对称：

```python
node.feature_message.initial_state
node.feature_message.current_state
node.gradient_message.initial_state
node.gradient_message.current_state
```

两者都是 `MHD_Node.Message`，因此都支持：

```python
message.reset()
message.update_initial(tensor, update_current=True)
message.to_device(device)
```

`current_state=None` 时从同一 Message 的 Initial State 克隆。省略 Gradient
Message 时自动创建零状态：浮点/复数 Feature 使用兼容 dtype，整数或布尔 Feature
使用 FP32。显式 Gradient Message 必须是浮点或复数 Tensor；一对 Initial/Current
State 在构造与加载时检查 shape、dtype 和 device。

`Operation` 是一般计算容器，`aggregation` 只负责一个 Node 收到多个 Message 时的
合并。不要把两者做成相同大小的类：职责不同，但它们共同进入同一张 autograd 计算图。

内置 aggregation 为 `replace/sum/avg/max/min/mul`，也可以传无可学习参数的 callable：

```python
def aggregate(current, incomings):
    return current + torch.stack(tuple(incomings)).sum(dim=0)

node = MHD_Node(
    0,
    "state",
    MHD_Node.Message(torch.zeros(8)),
    aggregation=aggregate,
)
```

aggregation 的真实数学导数由 autograd 生成；多条反向贡献按链式法则求和。因此没有第二个
`gradient_aggregation`。例如 `max` 的梯度遵循 `torch.maximum` 的导数，`replace`
只沿最后一个 incoming 返回梯度。

Operation 支持字符串、`nn.Module`、`partial`、callable 和
`torch.autograd.Function.apply`。多个输入严格按 Sort Matrix 作为位置参数调用：

```python
edge = MHD_Edge(
    0,
    "attention_or_join",
    edge_operations=[
        MHD_Edge.Operation(module_or_function),
        MHD_Edge.Operation(".relu()"),
    ],
)
```

V4 不再捕获模型内部的 `TypeError/ValueError/RuntimeError` 后猜测另一种调用方式。
如果需要逐项广播，请在 Operation 中明确写出并返回列表。

## 一套全局 Topo，两种显式执行序列

Topo 只有：

```python
MHD_Topo(role_matrices=[...], sort_matrices=[...])
```

没有 phase、`backward_start_level`、`backward_role_matrices` 或第二套执行器。
`levels` 是用户定义的全局索引；列表本身就是执行顺序：

```python
graph.forward(levels=[0, 2, 0])
graph.backward(levels=[5, 3])
```

- 不自动排序；`[5, 3]` 就先执行 level 5 的逻辑反向选择，再执行 level 3。
- 不去重；重复 level 会产生独立 Edge occurrence/trace。
- 允许不连续。
- 同一轮 Forward 与 Backward 不得使用相同 level。
- 一个 Backward Edge occurrence 必须反向匹配一个尚未匹配的真实 Forward
  occurrence；重复 Edge 按 LIFO 调用栈匹配。
- 不可达、依赖顺序错误、Role/Sort 方向不兼容都会明确报错。

下面是 `x -> linear -> prediction -> mean -> loss` 的四个全局 level：

```python
# Node 顺序: x, prediction, loss
# Edge 顺序: linear, mean
role_matrices = [
    torch.tensor([[-1,  1,  0], [ 0,  0,  0]]),  # level 0
    torch.tensor([[ 0,  0,  0], [ 0, -1,  1]]),  # level 1
    torch.tensor([[ 0,  0,  0], [ 0,  1, -1]]),  # level 2
    torch.tensor([[ 1, -1,  0], [ 0,  0,  0]]),  # level 3
]
sort_matrices = [
    torch.tensor([[0, 1, 0], [0, 0, 0]]),
    torch.tensor([[0, 0, 0], [0, 0, 1]]),
    torch.tensor([[0, 0, 0], [0, 0, 1]]),
    torch.tensor([[0, 1, 0], [0, 0, 0]]),
]
topo = MHD_Topo(role_matrices, sort_matrices)

graph.forward(levels=[0, 1])
graph.backward(levels=[2, 3])       # 完整路径

# 下一轮可截断在 prediction：
graph.forward(levels=[0, 1])
graph.backward(levels=[2])          # linear 参数 .grad=None
```

Backward 的矩阵用于描述“选哪条真实依赖向哪里返回梯度”；它不会再次执行 Edge
Operation。底层只调用一次 `torch.autograd.backward`。完整与部分路径共用这一种机制，
没有 native fast path、Message router、custom VJP bridge 三套分支。

## 标量终点、Gradient Initial State 与自定义梯度

一次 Forward 必须留下唯一的、可微的标量终点。`graph.backward(levels=...)` 自动以
`ones_like` 启动。非零 Gradient Initial State 是额外原生 seed，不覆盖自动 loss seed：

```python
node.gradient_message.update_initial(extra_seed)
graph.backward(levels=[...])
```

特殊梯度写在 Operation 内，使用标准 PyTorch 方式：

```python
class ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def backward(ctx, gradient):
        return -gradient

edge = MHD_Edge(
    0,
    "reverse_gradient",
    [MHD_Edge.Operation(ReverseGradient.apply)],
)
```

共享 Parameter 由 PyTorch 按对象身份自动累加。未选 occurrence 的 Tensor hook 返回零，
未选参数最终为 `.grad=None`；选中的共享 Parameter 保留所有选中贡献。

V4 Framework 用轻量 autograd hook 直接写入 Gradient Current State，不对每个普通中间
Tensor 调用 `retain_grad()`，避免为了公开 Gradient Message 再保留一份非叶 `.grad`。Utils
中的 Pipeline Parallel stage 会对所选 stage 边界保留梯度；它属于并行适配范围，不改变核心
Graph 的 Message 接口。

## Graph Display

```python
from V4.MHD_Utils_V4 import display_graph

diagram = display_graph(graph, levels=[0, 1, 0, 3, 2])
```

绘图属于 Utils，不属于 Framework 核心。`display_graph` 返回并打印 Mermaid 文本，不区分
Forward/Backward 样式，只按传入 levels 的原顺序读取 Role/Sort Matrix，
并把矩阵定义的箭头全部画为实线。只传 Forward levels 就得到前向图，只传 Backward
levels 就得到反向图；混合传入时也按同一个列表处理。标签显示列表位置、全局 level 和
Sort 值，因此重复执行不会丢失顺序信息。

## 传统网络、Transformer 与图/超图网络

普通 Module 不需要改写：

```python
residual_edge = MHD_Edge(
    0, "residual_block", [MHD_Edge.Operation(residual_block)]
)
transformer_edge = MHD_Edge(
    1, "transformer_block", [MHD_Edge.Operation(encoder_layer)]
)
```

图网络或超图网络可以把一次消息传递写成一个 level，并在多个 level 重复同一 Edge：

```python
graph.forward(levels=[message_level, message_level, readout_level])
```

每次重复都有独立 trace，因此同一个共享 Message Module 的参数梯度正常累加。不同
Role Matrix 可以表示不同邻接、超边和传播路径。V4 不内置某一种 GNN layer；用户仍可把
GCN/GAT/HypergraphConv 写成普通 Operation。

可执行的合成数值等价验证位于 `tests/model_smoke_v4.py`，覆盖 ResNet 风格残差分支、
两层 Transformer 和循环复用 Edge 的超图消息传递。

## Trainer 与 PyTorch Adapter

Trainer 保存默认路径，每次仍显式调用 Graph；单步可覆盖，梯度累积窗口内不得换路径：

```python
trainer = MHD_Trainer(
    graph,
    optimizer,
    MHD_Monitor(["loss"]),
    forward_levels=[0, 1, 2],
    backward_levels=[3, 4, 5],
    criteria=validation_loss,
    input_nodes=["input", "target"],
    input_mapping={"input": "image", "target": "label"},
    output_nodes=["loss"],
)

trainer.train_step(batch)
trainer.train_step(
    batch,
    forward_levels=[0, 2],
    backward_levels=[5, 3],
)
```

`input_nodes` 使用图中的 Node 名称；当 DataLoader 字段名不同，用可选的
`input_mapping` 显式映射。未写映射的节点默认读取同名字段，批次中的 participant ID 等
非图输入元数据会被保留在批次中，但不会写入 Graph。

Criteria 与反向起点不是一回事。反向起点由 Forward trace 自动寻找唯一可微标量终点；
Criteria 是用户定义的普通 PyTorch callable，只负责从完整 validation 状态计算用于选择
checkpoint 的标量。它接收 Graph，并可按名称读取任何已列入 `output_nodes` 的 Node：

```python
@torch.no_grad()
def validation_macro_f1(graph):
    logits = graph.get_node_by_name("logits").feature_message.current_state
    labels = graph.get_node_by_name("labels").feature_message.current_state
    return macro_f1(logits, labels)

monitor = MHD_Monitor(["loss", "batch_accuracy"])

trainer = MHD_Trainer(
    graph,
    optimizer,
    monitor,
    forward_levels=train_forward_levels,
    backward_levels=train_backward_levels,
    criteria=validation_macro_f1,
    criteria_mode="max",
    output_nodes=["logits", "labels", "loss", "batch_accuracy"],
)
```

Trainer 跨 rank 汇总 `output_nodes` 中按 batch 对齐的 Tensor，临时写回同名 Node 后调用
Criteria，并随即 reset 为声明的 Initial State。Monitor 只观察 Node、Edge 和 Message；optimizer update
仍由 Trainer 调用原生 PyTorch optimizer，因为参数更新不是 Message 路由。`best` 和 `last`
均由 Trainer 保存；`last` 用于中断恢复，`best` 始终对应最优 Criteria。

Utils 内的私有 `_MHD_GraphAdapter` 只把标准 PyTorch 的输入/输出 dict 转换成 Node
Message 读写，使 DDP/FSDP2/TP/`torch.compile` 能包装 MHD。它不转换模型、不保存第二份
拓扑，也不执行第二套计算；普通 `graph.forward/backward` 用户无需接触它。

## Checkpoint、Monitor、Merge 与 Prune

Node checkpoint 的正式结构是：

```text
node_messages.<name>.feature_message.initial_state
node_messages.<name>.feature_message.current_state
node_messages.<name>.gradient_message.initial_state
node_messages.<name>.gradient_message.current_state
```

Trainer checkpoint 同时保存 parameter、optimizer、scheduler、GradScaler、训练步数、
`forward_levels` 和 `backward_levels`。加载后重新校验 level 范围与不重叠规则。
新建 Trainer 恢复时先依据 DCP metadata 重建非空 history 的目标结构，再进行严格键匹配加载。
FP16 GradScaler 同时缩放标量终点与非零 Gradient Initial State；参数梯度交给原生
`unscale_`，公开的 Gradient Current State 则在 Trainer 内除回 scale，因此用户读取的是
未缩放梯度。

`MHD_Monitor(["loss"])` 仍产生 `loss_mean/loss_sum/loss_min/loss_max`。需要完整状态时：

```python
MHD_Monitor(
    ["loss"],
    node_states=(
        "feature_message.initial_state",
        "feature_message.current_state",
        "gradient_message.initial_state",
        "gradient_message.current_state",
    ),
)
```

Merge 对四个 State 分别求 mean，并要求同名 Node 的 aggregation 兼容、同名 Edge 的
Operation 序列兼容、有状态 Module/Parameter 身份完全相同。Prune 只裁剪这一套全局
Role/Sort Matrix，并重建 ID、索引、参数注册、执行计划和私有 trace 状态。

## V3 兼容脚本

`MHD_Compatibility_V3_to_V4.py` 负责一次性迁移，不在核心保留旧 API 别名：

- `initial_state/current_state` → Feature Message；
- 自动创建零 Gradient Message；
- `transfer_mode` → `aggregation`；
- 裸 `edge_operations` → `MHD_Edge.Operation`；
- V3 Topo 后追加反向兼容的全局 levels；
- V3 三文件 checkpoint → V4 四 State DCP checkpoint；
- migration report 给出显式 Forward/Backward level 序列。

迁移 live graph：

```python
from V4.MHD_Compatibility_V3_to_V4 import migrate_v3_graph

v4_graph = migrate_v3_graph(v3_graph)
paths = {
    "forward_levels": list(range(v3_graph.topo.num_levels)),
    "backward_levels": list(
        range(v3_graph.topo.num_levels, 2 * v3_graph.topo.num_levels)
    ),
}
```

迁移 checkpoint：

```bash
python -m V4.MHD_Compatibility_V3_to_V4 V3_CHECKPOINT_DIR V4_OUTPUT_DIR \
  --factory your_package.graphs:build_migrated_graph
```

若 factory 返回由 `migrate_v3_graph` 生成的偶数 level 图，脚本自动给出前半 Forward、
后半 Backward 序列；也可同时传 `--forward-levels` 与 `--backward-levels`。两者只能同时
提供或同时省略。Operation 的确定性位置调用可能暴露 V3 曾被异常吞掉的 callable，迁移后
需要把该 callable 明确改成位置参数形式。

## 单卡与独立多卡

基础单卡不需要并行配置。多卡能力位于 Utils，且一次只启用一个并行族：

| 模式 | 用途 | 代码范围 |
|---|---|---|
| DDP | 每卡完整模型、数据分片 | 任意合法 `world_size` |
| FSDP2 | 参数/梯度/optimizer state 分片 | 任意合法 `world_size` |
| TP | 一个 Module 内张量分片 | 任意合法 `tensor_parallel_size` |
| PP | 按 Edge 顺序切 stage | 任意合法 `pipeline_size`；GPipe/1F1B |

混合 PP×TP、PP×DP、TP×DP 和 PP×TP×DP 会明确拒绝。AMP、梯度累积和兼容的 compile
可与一个并行族组合。代码没有“两卡锁死”；配置的独立并行度必须等于本次
`WORLD_SIZE`。

PP 是可选的 Pipeline Parallel：把顺序 Edge 分配到不同 GPU。它在 V3 中没有正式对应
能力。V4 的 PP 仍使用 PyTorch 原生 pipeline schedule；显式 Backward levels 通过同一种
Tensor hook 屏蔽未选 stage/Edge contribution，并写回本 rank 可见的 Gradient Message。

## 实际验证与性能边界

以下内容是 2026-09-02 开发快照在 `ws` LOOK 环境留下的验证记录，环境为 PyTorch
2.8.0+cu128、两张 RTX 5000 Ada。它用于说明实际验证边界，不替代当前源码重新运行测试：

- 当时快照完成了 CPU/状态/迁移测试，覆盖 Message、Operation、aggregation、任意/重复
  level、完整与部分路径、共享参数、额外 seed、retain graph、Merge、Monitor、checkpoint、
  Prune 与 V3 migration；后续 API 调整后应以当前 `tests/` 的重新运行结果为准；
- GPU 合成 ResNet、Transformer、循环超图网络与原生参考输出/梯度等价；
- 双卡 DDP：完整路径，以及部分路径 + BF16 + 自定义 autograd 通过；
- 双卡 FSDP2：完整路径、部分路径 + BF16 通过；
- 双卡 TP：完整和部分路径通过；
- 双卡 PP：GPipe、1F1B 及部分路径通过；
- DDP + `torch.compile(backend="aot_eager")` 通过；ws 默认 Inductor 因系统缺少
  `/usr/include/python3.12/Python.h` 在进入 MHD forward 前失败，因此不声称默认
  Inductor 在该主机完成验证。

最终同步后，精确文件又完成了两进程 CPU/Gloo DDP 的完整路径，以及部分路径 + 自定义
autograd 回归。再次尝试 GPU/NCCL 回归时，两张卡被同机既有训练持续占用 99–100%，测试在
进入 MHD Forward 前的 DDP 初始化广播处达到硬超时；超时进程已全部清理。因此上面的双卡
条目来自本轮较早完成的实际 GPU 测试，不把这次资源拥塞记为新的代码通过或代码失败。

代码目标环境为 PyTorch 2.13；当前可用的 2.8 环境完成了回归，但尚未取得 2.13 环境做
正式运行。三卡以上的代码路径没有硬编码限制，但性能与稳定性尚未在真实三卡以上机器验证。

在同一 RTX 5000 Ada 上，以一个 BF16 `TransformerEncoderLayer`（batch=2、sequence=128、
hidden=512）比较同一原生 Module：

| 项目 | 原生中位数 | MHD 中位数 | MHD 额外开销 |
|---|---:|---:|---:|
| Forward | 2.3464 ms | 2.3595 ms | 13.05 μs / 0.56% |
| Forward + Backward | 2.4803 ms | 2.5175 ms | 1.50% |

结果来自 `benchmarks/benchmark_v4_overhead.py` 的 10 次 warmup、30 次 Forward 和 10 次
训练采样。共享 GPU 负载会造成波动，它不是所有模型的保证。更小的 Operation 固定 Python
调度占比更高；工程上应按卷积 block、Transformer block 或自然消息传递单元划 Edge，
而不是把每个微小算子拆成 Edge。MHD 不会比原生 kernel 更快，其目标是在保留超图表达、
显式路径和公开 Message 的同时，让大模型重计算占主导时额外开销保持较小。

RETFound ViT-L/16 的 28 个自然 Operation、56 个全局 levels、CFP/OCT 实验与权重加载说明在
`experiments/V4/retfound_2d/README.md`。该实验目录不属于 V4 核心目录；`V4` 目录只保留
Framework、Utils、Compatibility 和版本 `README.md` 四个文件。
