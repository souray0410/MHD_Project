# MHD V4 × RETFound 2D

本目录把官方 RETFound ViT-L/16 按模型本身的自然结构拆成 MHD 超图，而不是把整个
模型塞进一条 Edge。CFP 与 OCT 2D 切片使用相同结构、不同数据分支和各自的官方权重；
两种 checkpoint 不混用。

## 为什么拆成 28 个 Operation

RETFound 2D 是 224×224、patch size 16、hidden size 1024、24 个 Transformer block、
16 个 attention head 的 ViT-L。模型原生就有以下连续边界：

```text
image
  -> patch embedding + class token + position embedding + norm
  -> Transformer block 01
  -> ...
  -> Transformer block 24
  -> patch-token global average + fc_norm
  -> classification head
logits + target -> cross entropy -> loss
```

因此 MHD 使用 30 个 Node、28 条 Edge：

- Edge 0：patch/position stage；
- Edge 1..24：24 个原生 `model.blocks[i]`，Module/Parameter 不复制；
- Edge 25：global pool + `fc_norm`；
- Edge 26：classification head；
- Edge 27：cross entropy criterion。

这比人为按 GPU 或任意层数切 stage 更有依据：每条 Edge 都对应一个真实可复用的模型
Module 或一个清楚的数学操作，同时单个 Transformer block 的计算量足以摊薄 MHD 调度开销。

## 56 个全局 Role/Sort levels

每个矩阵的全局形状都是 `28 × 30`（Edge × Node），但每个 level 只激活一条真实 Edge：

| level | Edge | Role 连接 |
|---:|---|---|
| 0 | patch | `image(-1) -> patch_tokens(+1)` |
| 1 | block 01 | `patch_tokens(-1) -> block_01(+1)` |
| 2..24 | block 02..24 | 前一个 block Node `(-1)` → 当前 block Node `(+1)` |
| 25 | pool/norm | `block_24(-1) -> representation(+1)` |
| 26 | head | `representation(-1) -> logits(+1)` |
| 27 | criterion | `logits(-1), target(-1) -> loss(+1)` |
| 28 | criterion reverse | `loss(-1) -> logits(+1), target(+1)` |
| 29 | head reverse | `logits(-1) -> representation(+1)` |
| 30 | pool/norm reverse | `representation(-1) -> block_24(+1)` |
| 31..54 | block 24..01 reverse | 当前 block Node `(-1)` → 前一个 Node `(+1)` |
| 55 | patch reverse | `patch_tokens(-1) -> image(+1)` |

对应常量：

```python
RETF_FOUND_FORWARD_LEVELS = tuple(range(28))
RETF_FOUND_BACKWARD_LEVELS = tuple(range(28, 56))
```

以 criterion 为例，level 27 的非零 Role 行是：

```text
edge 27: target=-1, logits=-1, loss=+1
```

level 28 是同一全局矩阵列表中的反向连接：

```text
edge 27: target=+1, logits=+1, loss=-1
```

Sort 值保持 Operation 的确定性参数/输出对应：Forward criterion 中 logits=0、target=1、
loss=2；反向 level 中 loss 是 head，logits/target 按原输入顺序成为 tail。Topo 没有单独的
Backward Matrix 字段。

## 完整与截断路径

完整训练：

```python
graph.forward(levels=RETF_FOUND_FORWARD_LEVELS)
graph.backward(levels=RETF_FOUND_BACKWARD_LEVELS)
```

从 loss 向外逐步扩大：

```text
depth 2: criterion -> head                     （可微调 head）
depth 3: criterion -> head -> pool/fc_norm     （再加入 norm）
depth 4: ... -> Transformer block 24           （再加入最后一个 block）
depth 28: ... -> patch/image                    （完整路径）
```

`run_experiment.py --backward-depth N` 执行前 N 个反向 level。微调至少需要 depth 2；
`--check-full-backward` 必须与 depth 28 一起使用。未选 Operation 的参数 `.grad=None`，但截断
边界 Node 仍公开保存收到的 Gradient Current State。

CFP 与 OCT 是两次独立运行的输入/权重分支，不是把两种模态同时接进一张模型图；因此所谓
“分支选择”由 `--modality cfp|oct` 明确决定，图内路径规则完全一致。

## 权重与数据

官方 CFP/OCT 权重由 RETFound 作者发布，采用 CC BY-NC 4.0，本实验不重新分发。
加载器支持官方 `model/state_dict` 封装、分类头尺寸迁移与位置编码插值，并拒绝不兼容的
backbone。使用权重时必须分别传：

- CFP：`RETFound_mae_natureCFP.pth`；
- OCT：`RETFound_mae_natureOCT.pth`。

UK Biobank adapter 只读取既有的 224×224 配对缓存。OCT 3D 已由上游流程切成 2D 图像，
本实验按 2D RETFound 输入处理。结果 JSON 不保存参与者 ID、原始路径或图像。

没有可用官方权重时可以用同结构随机初始化验证 MHD 与原生模型的前后向一致性，但这不能
声称完成预训练权重微调。

## ws / LOOK 运行

```bash
LOOK_VENV=/home/mengh/LOOK/2026_08_30_11_20_47/tool/environment/.venv
EXP=/home/mengh/LOOK/2026_09_02_00_00_00/tool/experiments/V4/retfound_2d
LABELS=/data/mengh/LOOK/2026_08_30_11_20_47/dataset/reference_labels.csv
CACHE=/data/mengh/LOOK/2026_09_02_00_00_00/cache/preprocessed_pairs
```

先验证 CFP/OCT 的分 stage Forward 与原生模型完全相同：

```bash
$LOOK_VENV/bin/python $EXP/verify_split_equivalence.py \
  --labels-csv $LABELS --cache-root $CACHE
```

单卡 CFP 完整反向与一步微调：

```bash
$LOOK_VENV/bin/python $EXP/run_experiment.py \
  --labels-csv $LABELS --cache-root $CACHE --modality cfp \
  --batch-size 1 --sample-limit 2 --steps 1 --trainable-blocks 1 \
  --backward-depth 28 --check-full-backward
```

双卡 DDP 使用每卡 batch 1：

```bash
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 \
$LOOK_VENV/bin/torchrun --standalone --nproc-per-node=2 $EXP/run_experiment.py \
  --labels-csv $LABELS --cache-root $CACHE --modality oct \
  --parallel ddp --batch-size 1 --sample-limit 4 --steps 1 \
  --trainable-blocks 1 --precision bf16 --backward-depth 28 \
  --check-full-backward
```

带官方权重时追加：

```text
--checkpoint /path/to/modality_specific_checkpoint.pth --require-checkpoint
```

`verify_split_equivalence.py` 检查 CFP/OCT；`run_experiment.py` 检查 loss、Node Gradient
Message、参数梯度、optimizer 更新和 DDP rank checksum。验证记录必须以实际生成的 result
JSON 为准。

当前 `run_experiment.py` 是固定步数的正确性/微调 smoke test，直接执行 optimizer step，
不比较和保存“最佳 checkpoint”，因此不创建未被使用的 Criteria。若把它扩展为正式的
多 epoch `MHD_Trainer` 实验，必须显式提供决定最佳模型保存的 `criteria(graph)`。例如以完整
验证集的平均 loss 选择 checkpoint：

```python
@torch.no_grad()
def validation_loss(graph):
    loss = graph.get_node_by_name("loss").feature_message.current_state
    return loss.float().mean()


trainer = MHD_Trainer(
    graph,
    optimizer,
    MHD_Monitor(["loss"]),
    forward_levels=RETF_FOUND_FORWARD_LEVELS,
    backward_levels=RETF_FOUND_BACKWARD_LEVELS,
    criteria=validation_loss,
    criteria_mode="min",
    input_nodes=["image", "target"],
    output_nodes=["loss"],
)
```

Backward Node 不需要传入；它由本次 Forward 的唯一可微标量终点自动确定。Criteria callable
则始终是 Trainer 的必填项，因为它单独决定最佳 checkpoint 的比较和保存逻辑。Trainer 会先
汇总 `output_nodes` 中完整 validation 状态，再调用 Criteria；因此 Criteria 与反向起点互不替代。

## 本次实际结果

在 ws LOOK（PyTorch 2.8.0+cu128、RTX 5000 Ada）完成：

| 数据与路径 | 设备 | Forward | Gradient Message | optimizer 更新 | rank 一致性 |
|---|---|---:|---:|---:|---:|
| UKB CFP/OCT split equivalence | GPU0 | 与原生 logits 最大误差 0 | 不适用 | 不适用 | 不适用 |
| UKB CFP，depth 28 | 单卡 BF16 | 通过 | image 到 loss 全部有限且非零 | head 更新 | 不适用 |
| UKB OCT，depth 28 | 单卡 BF16 | 通过 | image 到 loss 全部有限且非零 | head 更新 | 不适用 |
| UKB CFP，depth 28 | 双卡 DDP BF16 | 通过 | image 到 loss 全部有限且非零 | head 更新 | 两 rank checksum 相同 |
| UKB OCT，depth 28 | 双卡 DDP BF16 | 通过 | image 到 loss 全部有限且非零 | head 更新 | 两 rank checksum 相同 |

现有四个训练结果 JSON 均记录 `full_backward_checked=true`，因此这里不再把未保存结果的
depth 4 写成已验证。当前脚本会进一步检查部分路径：选中 Node 的 Gradient Message 必须
非零，未选 Node 必须为零，未选 Operation 的可训练参数必须保持 `.grad=None`。需要重新运行
`--backward-depth 2/3/4/...` 并保存对应 JSON 后，才把部分路径加入实际结果表。

模型参数量 303,306,757；最后一个 block + `fc_norm` + head 的可训练参数量
12,603,397。上述运行使用真实 UKB 缓存和正确 RETFound 结构，但使用随机初始化，因为当前
环境没有已授权的官方 CFP/OCT checkpoint；因此现有结果证明 MHD 拆分、完整前后向与微调
流程正确；部分路径需要按上面的新检查重新保存结果，也不声称完成官方预训练权重微调。
