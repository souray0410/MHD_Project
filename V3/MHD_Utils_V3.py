# -*- coding: utf-8 -*-
"""
Multi-Hypergraph Dynamic Utils (MHD-Utils) - Version 3.1
Author: Souray Meng (孟号丁)
Utility Tools: Dataset, Training, Monitoring for MHD Framework V3
License: MIT
"""

import torch
import torch.distributed as dist
import random
import numpy as np
import torch.nn as nn
import os
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Callable, Any, Union, Optional, Sequence, Set
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import logging
from datetime import datetime
from tqdm import tqdm
import json
import warnings
import gc
import shutil

from .MHD_Framework_V3 import (
    MHD_Node, MHD_Edge, MHD_Topo, MHD_Graph,
)


# ===================== 单卡 / DDP 执行适配 =====================

@dataclass(frozen=True)
class MHD_DistributedContext:
    """One-process-per-device execution metadata for MHD graphs."""

    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    backend: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def initialize_mhd_distributed(backend: Optional[str] = None) -> MHD_DistributedContext:
    """Initialize torch.distributed from torchrun environment variables when needed."""

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=selected_backend, init_method="env://")
    return MHD_DistributedContext(rank, local_rank, world_size, device, selected_backend)


def destroy_mhd_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def mhd_barrier(context: MHD_DistributedContext) -> None:
    if context.distributed:
        dist.barrier()


def mhd_all_reduce_mean(value: torch.Tensor, context: MHD_DistributedContext) -> torch.Tensor:
    result = value.detach().clone()
    if context.distributed:
        dist.all_reduce(result, op=dist.ReduceOp.SUM)
        result /= context.world_size
    return result


def mhd_all_gather_object(value: Any, context: MHD_DistributedContext) -> List[Any]:
    if not context.distributed:
        return [value]
    gathered: List[Any] = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, value)
    return gathered


class MHD_ForwardAdapter(nn.Module):
    """Expose mutable-node MHD execution through a conventional module forward call."""

    def __init__(
        self,
        graph: MHD_Graph,
        input_nodes: Sequence[str],
        output_nodes: Sequence[str],
        levels: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()
        self.graph = graph
        self.input_nodes = tuple(input_nodes)
        self.output_nodes = tuple(output_nodes)
        self.levels = None if levels is None else tuple(levels)
        missing = [name for name in (*self.input_nodes, *self.output_nodes) if graph.get_node_by_name(name) is None]
        if missing:
            raise ValueError(f"Unknown MHD adapter nodes: {missing}")

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        absent = set(self.input_nodes) - set(input_dict)
        if absent:
            raise ValueError(f"Missing MHD inputs: {sorted(absent)}")
        batch_sizes = {int(input_dict[name].shape[0]) for name in self.input_nodes}
        if len(batch_sizes) != 1:
            raise ValueError(f"Inconsistent MHD input batch sizes: {sorted(batch_sizes)}")
        self.graph.update_batch_size(batch_sizes.pop())
        for node in self.graph.nodes:
            node.reset()
        for name in self.input_nodes:
            self.graph.get_node_by_name(name).current_state = input_dict[name]
        self.graph.forward(levels=None if self.levels is None else list(self.levels))
        return {name: self.graph.get_node_by_name(name).current_state for name in self.output_nodes}


def unwrap_mhd_graph(module: nn.Module) -> MHD_Graph:
    candidate = module.module if hasattr(module, "module") else module
    if not isinstance(candidate, MHD_ForwardAdapter):
        raise TypeError(f"Expected MHD_ForwardAdapter, found {type(candidate)!r}")
    return candidate.graph

# ===================== 状态保存/加载工具函数 =====================

def updown_node(nodes: Set[MHD_Node], path: str, mode: str, target_device: torch.device = None) -> None:
    """
    节点特征图保存/加载函数

    特性：
    1. 支持节点初始状态和当前状态的保存/加载
    2. 自动处理设备迁移
    3. 保存元信息用于验证

    Args:
        nodes: 节点集合
        path: 文件路径
        mode: 模式，'up'为加载，'down'为保存
        target_device: 目标设备（仅加载时有效）

    Raises:
        ValueError: 当模式参数错误时
    """
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        # 保存模式
        save_dict = {
            "node_initial_states": {n.name: n.initial_state for n in sorted(nodes, key=lambda x: x.id)},
            "node_current_states": {n.name: n.current_state for n in sorted(nodes, key=lambda x: x.id)},
            "node_info": {
                n.name: {
                    "id": n.id,
                    "shape": n.current_state.shape,
                    "dtype": str(n.current_state.dtype),
                    "device": str(n.current_state.device)
                }
                for n in sorted(nodes, key=lambda x: x.id)
            }
        }
        torch.save(save_dict, path)

    else:
        # 加载模式
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        for node in sorted(nodes, key=lambda x: x.id):
            if node.name in load_dict["node_initial_states"]:
                loaded_tensor = load_dict["node_initial_states"][node.name]
                # 自动迁移到目标设备
                if target_device is not None and loaded_tensor.device != target_device:
                    loaded_tensor = loaded_tensor.to(target_device, non_blocking=True)
                node.initial_state = loaded_tensor.to(
                    dtype=node.current_state.dtype
                )

            if node.name in load_dict["node_current_states"]:
                loaded_tensor = load_dict["node_current_states"][node.name]
                # 自动迁移到目标设备
                if target_device is not None and loaded_tensor.device != target_device:
                    loaded_tensor = loaded_tensor.to(target_device, non_blocking=True)
                node.current_state = loaded_tensor.to(
                    dtype=node.current_state.dtype
                )

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = sum(1 for n in nodes if n.name in (load_dict["node_initial_states"] if mode == 'up' else [n.name for n in nodes]))

    print(f"📊 节点特征图{mode_cn}完成 (Node {mode_en} completed) 处理: {processed}/{len(nodes)}")
    if target_device and mode == 'up':
        print(f" ├─ 目标设备: {target_device}")
    print(f" 📁 路径: {path}")


def updown_edge(edges: Set[MHD_Edge], path: str, mode: str, target_device: torch.device = None) -> None:
    """
    超边可学习模块保存/加载函数（适配 edge_operations）

    特性：
    1. 支持边中可学习参数的保存/加载
    2. 自动处理nn.Module状态字典
    3. 保存操作序列信息

    Args:
        edges: 边集合
        path: 文件路径
        mode: 模式，'up'为加载，'down'为保存
        target_device: 目标设备（仅加载时有效）

    Raises:
        ValueError: 当模式参数错误时
    """
    if mode not in ('up', 'down'):
        raise ValueError(f"模式错误: {mode}，仅支持 'up'/'down'")

    if mode == 'down':
        # 保存模式
        save_dict = {
            "edge_params": {},
            "edge_info": {}
        }
        for edge in sorted(edges, key=lambda x: x.id):
            save_dict["edge_params"][edge.name] = [
                elem.state_dict() if isinstance(elem, nn.Module) else None
                for elem in edge.edge_operations
            ]
            save_dict["edge_info"][edge.name] = {
                "id": edge.id,
                "operations": [str(type(op)) for op in edge.edge_operations]
            }
        torch.save(save_dict, path)

    else:
        # 加载模式
        load_dict = torch.load(path, map_location='cpu', weights_only=True)
        for edge in sorted(edges, key=lambda x: x.id):
            if edge.name in load_dict["edge_params"]:
                saved_params = load_dict["edge_params"][edge.name]
                for idx, elem in enumerate(edge.edge_operations):
                    if idx < len(saved_params) and isinstance(elem, nn.Module) and saved_params[idx] is not None:
                        elem.load_state_dict(saved_params[idx])
                        # 迁移到目标设备
                        if target_device is not None:
                            edge.edge_operations[idx] = elem.to(target_device, non_blocking=True)

    # 输出统计信息
    mode_cn = "保存" if mode == 'down' else "加载"
    mode_en = "save" if mode == 'down' else "load"
    processed = sum(1 for e in edges if e.name in (load_dict["edge_params"] if mode == 'up' else [e.name for e in edges]))

    print(f"📊 超边可学习参数{mode_cn}完成 (Edge {mode_en} completed) 处理: {processed}/{len(edges)}")
    if target_device and mode == 'up':
        print(f" ├─ 目标设备: {target_device}")
    print(f" 📁 路径: {path}")


# ===================== 统一数据集类 =====================

class MHD_Dataset(Dataset):
    """
    高扩展MHD数据集（支持 shared_loader）

    特性：
    1. 支持 shared_loader 一次性加载所有节点数据，避免重复 I/O
    2. 增强在 loader 内部完成，框架不再提供增强器
    3. 统一设备管理
    """

    def __init__(
        self,
        sample_info_list: List[Any],
        node_configs: Dict[str, Dict],
        base_seed: int = 42,
        target_device: torch.device = None,
        shared_loader: Optional[Callable[[Any], Dict[str, Any]]] = None
    ):
        """
        初始化MHD数据集

        Args:
            sample_info_list: 样本信息列表
            node_configs: 节点配置字典，格式为{节点名: {loader: 加载函数}}
            base_seed: 基础随机种子
            target_device: 目标计算设备
            shared_loader: 可选，一次调用返回包含所有节点所需数据的字典
        """
        self.sample_info_list = sample_info_list
        self.node_configs = node_configs
        self.base_seed = base_seed
        self.target_device = target_device or torch.device("cpu")
        self.node_names = list(node_configs.keys())
        self.shared_loader = shared_loader

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.sample_info_list)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        单样本加载流程

        Args:
            idx: 样本索引

        Returns:
            节点名到张量的映射字典
        """
        sample_info = self.sample_info_list[idx]

        # 获取worker信息（用于多进程随机种子）
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            sample_seed = self.base_seed + worker_id + idx
        else:
            sample_seed = self.base_seed + idx

        # 如果有共享加载器，则调用一次获取所有数据
        if self.shared_loader is not None:
            shared_data = self.shared_loader(sample_info)
        else:
            shared_data = None

        sample_data = {}
        for node_name, config in self.node_configs.items():
            load_fn = config["loader"]
            try:
                if shared_data is not None:
                    # loader 接收共享数据字典
                    tensor = load_fn(shared_data)
                else:
                    tensor = load_fn(sample_info)
                if not isinstance(tensor, torch.Tensor):
                    tensor = torch.tensor(tensor, dtype=torch.float32)
            except Exception as e:
                raise RuntimeError(f"节点 {node_name} 加载失败（idx={idx}）: {str(e)}")

            # 数据增强完全由 loader 内部负责，此处不做额外处理
            if tensor.device != self.target_device:
                tensor = tensor.to(self.target_device, non_blocking=True)
            sample_data[node_name] = tensor

        return sample_data

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        批次拼接函数

        Args:
            batch: 批次数据列表

        Returns:
            拼接后的批次数据
        """
        if not batch:
            raise ValueError("空批次数据")
        ref_node_names = batch[0].keys()
        batch_data = {}
        for node_name in ref_node_names:
            node_samples = [sample[node_name] for sample in batch]
            batch_data[node_name] = torch.stack(node_samples, dim=0)
        return batch_data


# ===================== 数据加载器工具函数 =====================

def create_mhd_dataloader(
    dataset: MHD_Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
    drop_last: bool = True,
    persistent_workers: bool = False
) -> DataLoader:
    """
    创建MHD数据加载器

    特性：
    1. 支持多进程数据加载及可复现性
    2. 统一设备管理
    """
    def worker_init_fn(worker_id: int) -> None:
        process_seed = dataset.base_seed + worker_id
        random.seed(process_seed)
        np.random.seed(process_seed)
        torch.manual_seed(process_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(process_seed)
            torch.cuda.manual_seed_all(process_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    worker_init_fn_to_use = worker_init_fn if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=MHD_Dataset.collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init_fn_to_use,
        persistent_workers=persistent_workers
    )


# ===================== 监控器类 =====================

class MHD_Monitor:
    """
    MHD训练监控器

    特性：
    1. 监控节点和边的统计信息
    2. 支持训练和验证模式的梯度监控
    3. 安全的张量统计计算
    4. 格式化输出
    """

    def __init__(self, monitor_nodes: list, monitor_edges: list = None):
        self.monitor_nodes = monitor_nodes
        self.monitor_edges = monitor_edges
        self.records = defaultdict(list)
        self.step_counter = 0

    def reset(self):
        self.records = defaultdict(list)
        self.step_counter = 0

    def _safe_tensor_stats(self, tensor: torch.Tensor) -> dict:
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        return {
            "mean": float(tensor.mean().item()),
            "sum": float(tensor.sum().item()),
            "max": float(tensor.max().item()),
            "min": float(tensor.min().item())
        }

    def monitor_node(self, mhd_graph, prefix: str = "") -> dict:
        node_metrics = {}
        for node_name in self.monitor_nodes:
            node = mhd_graph.get_node_by_name(node_name)
            if node is None:
                warnings.warn(f"监控节点 {node_name} 不存在，跳过")
                continue
            value = node.current_state.detach()
            stats = self._safe_tensor_stats(value)
            for stat_name, stat_value in stats.items():
                node_metrics[f"{prefix}{node_name}_{stat_name}"] = stat_value
        for k, v in node_metrics.items():
            self.records[k].append(v)
        self.step_counter += 1
        return node_metrics

    def monitor_edge(self, mhd_graph, prefix: str = "", train_mode: bool = True) -> dict:
        edge_metrics = {}
        target_edges = self.monitor_edges or [e.name for e in mhd_graph.edges]
        for edge_name in target_edges:
            edge = mhd_graph.get_edge_by_name(edge_name)
            if edge is None:
                warnings.warn(f"监控边 {edge_name} 不存在，跳过")
                continue
            for idx, op in enumerate(edge.edge_operations):
                if not isinstance(op, nn.Module):
                    continue
                weight_key = f"{prefix}{edge_name}_op{idx}"
                if hasattr(op, 'weight') and op.weight is not None:
                    weight = op.weight.detach()
                    weight_stats = self._safe_tensor_stats(weight)
                    edge_metrics[f"{weight_key}_weight_mean"] = weight_stats["mean"]
                    edge_metrics[f"{weight_key}_weight_l2"] = float(torch.norm(weight).item())
                if train_mode and hasattr(op, 'weight') and op.weight.grad is not None:
                    grad = op.weight.grad.detach()
                    grad_stats = self._safe_tensor_stats(grad)
                    edge_metrics[f"{weight_key}_grad_mean"] = grad_stats["mean"]
                    edge_metrics[f"{weight_key}_grad_l2"] = float(torch.norm(grad).item())
        for k, v in edge_metrics.items():
            self.records[k].append(v)
        return edge_metrics

    def get_mean_metrics(self, step_window: int = None) -> dict:
        mean_metrics = {}
        window = slice(-step_window, None) if step_window else slice(None)
        for metric_name, values in self.records.items():
            if len(values) == 0:
                mean_metrics[metric_name] = 0.0
            else:
                clean_values = [v for v in values[window] if not np.isnan(v) and not np.isinf(v)]
                mean_metrics[metric_name] = float(np.mean(clean_values)) if clean_values else 0.0
        return mean_metrics

    def format_metrics(self, metrics: dict, decimal: int = 6) -> str:
        formatted = []
        for k, v in sorted(metrics.items()):
            formatted.append(f"  {k}: {v:.{decimal}f}")
        return "\n".join(formatted)


# ===================== 训练器类 =====================

class MHD_Trainer:
    """
    MHD训练器（重构版）

    特性：
    1. 独立的 backward_node 指定反向传播节点
    2. criteria_node 与 mode 控制最佳模型保存
    3. 所有监控节点统一记录，不区分 loss/metric
    4. 最佳模型仅更新 best 文件（node_best, edge_best, meta_best）
    5. 常规检查点按 save_interval 保存带 epoch 编号的文件（node_epoch_X, edge_epoch_X, meta_epoch_X）
    """

    def __init__(
        self,
        mhd_graph: MHD_Graph,
        optimizer: torch.optim.Optimizer,
        monitor: MHD_Monitor,
        backward_node: str,
        criteria_node: Optional[str] = None,
        criteria_mode: str = 'min',
        save_dir: str = "./mhd_ckpts",
        grad_clip_norm: float = None,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        save_interval: int = 0
    ):
        """
        初始化训练器

        Args:
            mhd_graph: MHD图实例
            optimizer: 优化器
            monitor: 监控器
            backward_node: 用于反向传播的节点名称（必须是标量或可 reduce 的节点）
            criteria_node: 用于判断最佳模型的监控节点，默认同 backward_node
            criteria_mode: 'min' 或 'max'，默认 'min'
            save_dir: 保存目录
            grad_clip_norm: 梯度裁剪阈值
            lr_scheduler: 学习率调度器
            save_interval: 常规检查点保存间隔（epoch 数），0 表示不保存常规检查点
        """
        self.mhd_graph = mhd_graph
        self.optimizer = optimizer
        self.monitor = monitor
        self.backward_node_name = backward_node
        self.criteria_node_name = criteria_node if criteria_node is not None else backward_node
        self.criteria_mode = criteria_mode
        self.save_dir = save_dir
        self.grad_clip_norm = grad_clip_norm
        self.lr_scheduler = lr_scheduler
        self.device = mhd_graph.device
        self.save_interval = save_interval

        os.makedirs(save_dir, exist_ok=True)
        self.logger = self._setup_logger(save_dir)

        # 历史记录
        self.history = {
            "train": {"metrics": []},
            "eval": {"metrics": []},
            "best_eval_value": float("inf") if criteria_mode == 'min' else -float("inf"),
            "best_epoch": -1,
            "best_backward_value": None
        }

        self._validate_nodes()

        self.logger.info("="*80)
        self.logger.info("🚀 MHD训练器初始化完成")
        self.logger.info(f"🔙 反向传播节点: {self.backward_node_name}")
        self.logger.info(f"🏆 最佳判定节点: {self.criteria_node_name} (mode={self.criteria_mode})")
        self.logger.info(f"📊 监控节点: {self.monitor.monitor_nodes}")
        self.logger.info(f"💾 保存目录: {self.save_dir}")
        self.logger.info(f"🗂️  常规保存间隔: {self.save_interval} epoch")
        self.logger.info("="*80)

    def _setup_logger(self, save_dir: str) -> logging.Logger:
        logger = logging.getLogger("mhd_train")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        log_file = os.path.join(save_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        return logger

    def _validate_nodes(self):
        if not self.mhd_graph.get_node_by_name(self.backward_node_name):
            raise ValueError(f"反向传播节点 '{self.backward_node_name}' 不存在")
        if not self.mhd_graph.get_node_by_name(self.criteria_node_name):
            raise ValueError(f"最佳判定节点 '{self.criteria_node_name}' 不存在")
        for node_name in self.monitor.monitor_nodes:
            if not self.mhd_graph.get_node_by_name(node_name):
                warnings.warn(f"监控节点 '{node_name}' 不存在")

    def _check_batch_consistency(self, input_dict: dict) -> int:
        batch_sizes = []
        for tensor in input_dict.values():
            if tensor.dim() < 2:
                raise ValueError("输入张量至少需要 2 维 (B, ...)")
            batch_sizes.append(tensor.shape[0])
        if not batch_sizes:
            raise ValueError("空输入")
        if len(set(batch_sizes)) != 1:
            raise ValueError("批次大小不一致")
        return batch_sizes[0]

    def train_step(self, input_dict: dict) -> dict:
        self.mhd_graph.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch_size = self._check_batch_consistency(input_dict)
        self.mhd_graph.update_batch_size(batch_size)

        for node in self.mhd_graph.nodes:
            node.reset()
        for node_name, tensor in input_dict.items():
            node = self.mhd_graph.get_node_by_name(node_name)
            if node is None:
                continue
            node.current_state = tensor.to(self.device, non_blocking=True).detach().requires_grad_(True)

        self.mhd_graph.forward()

        loss_node = self.mhd_graph.get_node_by_name(self.backward_node_name)
        loss_tensor = loss_node.current_state.mean()
        if not loss_tensor.requires_grad:
            raise RuntimeError("反向传播节点无梯度")
        loss_tensor.backward()

        if self.grad_clip_norm and self.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.mhd_graph.parameters(), self.grad_clip_norm)

        self.optimizer.step()

        step_metrics = {}
        for node_name in self.monitor.monitor_nodes:
            node = self.mhd_graph.get_node_by_name(node_name)
            if node:
                val = node.current_state.detach().mean().item()
                step_metrics[node_name] = float(0.0 if np.isnan(val) or np.isinf(val) else val)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return step_metrics

    @torch.no_grad()
    def eval_step(self, input_dict: dict) -> dict:
        self.mhd_graph.eval()
        batch_size = self._check_batch_consistency(input_dict)
        self.mhd_graph.update_batch_size(batch_size)

        for node in self.mhd_graph.nodes:
            node.reset()
        for node_name, tensor in input_dict.items():
            node = self.mhd_graph.get_node_by_name(node_name)
            if node is None:
                continue
            node.current_state = tensor.to(self.device, non_blocking=True).detach()

        self.mhd_graph.forward()

        step_metrics = {}
        for node_name in self.monitor.monitor_nodes:
            node = self.mhd_graph.get_node_by_name(node_name)
            if node:
                val = node.current_state.detach().mean().item()
                step_metrics[node_name] = float(0.0 if np.isnan(val) or np.isinf(val) else val)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return step_metrics

    def train_epoch(self, train_data, epoch: int):
        self.monitor.reset()
        epoch_metrics_sum = defaultdict(float)
        pbar = tqdm(train_data, desc=f"Train Epoch {epoch+1}", leave=False)
        for step, input_dict in enumerate(pbar):
            gc.collect()
            step_metrics = self.train_step(input_dict)
            for k, v in step_metrics.items():
                epoch_metrics_sum[k] += v
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
        avg_metrics = {k: v / len(train_data) for k, v in epoch_metrics_sum.items()}
        self.history["train"]["metrics"].append(avg_metrics)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.logger.info(f"\n📈 训练轮次 {epoch+1} 指标: {avg_metrics}")

    def eval_epoch(self, eval_data, epoch: int):
        self.monitor.reset()
        epoch_metrics_sum = defaultdict(float)
        pbar = tqdm(eval_data, desc=f"Eval  Epoch {epoch+1}", leave=False)
        for step, input_dict in enumerate(pbar):
            gc.collect()
            step_metrics = self.eval_step(input_dict)
            for k, v in step_metrics.items():
                epoch_metrics_sum[k] += v
            pbar.set_postfix({k: f"{v:.4f}" for k, v in step_metrics.items()})
        avg_metrics = {k: v / len(eval_data) for k, v in epoch_metrics_sum.items()}
        self.history["eval"]["metrics"].append(avg_metrics)

        cur_backward = avg_metrics.get(self.backward_node_name)
        cur_criteria = avg_metrics.get(self.criteria_node_name)

        if cur_criteria is None:
            self.logger.warning(f"最佳判定节点 {self.criteria_node_name} 不在指标中，无法判断最佳模型")
            return

        is_better = False
        if self.criteria_mode == 'min':
            if cur_criteria < self.history["best_eval_value"]:
                is_better = True
        else:
            if cur_criteria > self.history["best_eval_value"]:
                is_better = True

        if is_better:
            self.history["best_eval_value"] = cur_criteria
            self.history["best_epoch"] = epoch + 1
            self.history["best_backward_value"] = cur_backward
            self._save_best_checkpoint()
            bw_str = f"{cur_backward:.6f}" if cur_backward is not None else "N/A"
            self.logger.info(f"🏆 当下最佳模型 | Epoch {epoch+1} | {self.backward_node_name}: {bw_str} | {self.criteria_node_name}: {cur_criteria:.6f}")
        else:
            best_bw = self.history.get("best_backward_value")
            bw_str = f"{best_bw:.6f}" if best_bw is not None else "N/A"
            self.logger.info(f"🏆 过往最佳模型 | Epoch {self.history['best_epoch']} | {self.backward_node_name}: {bw_str} | {self.criteria_node_name}: {self.history['best_eval_value']:.6f}")

        self.logger.info(f"📊 验证轮次 {epoch+1} 指标: {avg_metrics}")

    def _save_best_checkpoint(self):
        """仅更新最佳模型文件（node_best, edge_best, meta_best）"""
        try:
            node_path = os.path.join(self.save_dir, "node_best.pth")
            edge_path = os.path.join(self.save_dir, "edge_best.pth")
            meta_path = os.path.join(self.save_dir, "meta_best.pth")
            updown_node(self.mhd_graph.nodes, node_path, mode="down", target_device=self.device)
            updown_edge(self.mhd_graph.edges, edge_path, mode="down", target_device=self.device)
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "history": self.history,
                "epoch": self.history["best_epoch"],
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            }, meta_path)
            self.logger.info(f"🏆 最佳模型已更新 (Epoch {self.history['best_epoch']})")
        except Exception as e:
            self.logger.error(f"❌ 保存最佳模型失败: {str(e)}")
            raise

    def save_checkpoint(self, epoch: int):
        """保存常规检查点（node_epoch_X, edge_epoch_X, meta_epoch_X）"""
        try:
            node_path = os.path.join(self.save_dir, f"node_epoch_{epoch}.pth")
            edge_path = os.path.join(self.save_dir, f"edge_epoch_{epoch}.pth")
            meta_path = os.path.join(self.save_dir, f"meta_epoch_{epoch}.pth")
            updown_node(self.mhd_graph.nodes, node_path, mode="down", target_device=self.device)
            updown_edge(self.mhd_graph.edges, edge_path, mode="down", target_device=self.device)
            torch.save({
                "optimizer": self.optimizer.state_dict(),
                "history": self.history,
                "epoch": epoch,
                "lr_scheduler": self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            }, meta_path)
            self.logger.info(f"✅ 常规检查点保存完成 (Epoch {epoch})")
        except Exception as e:
            self.logger.error(f"❌ 保存常规检查点失败: {str(e)}")
            raise

    def load_checkpoint(self, load_best: bool = False, epoch: int = None):
        try:
            if load_best:
                node_path = os.path.join(self.save_dir, "node_best.pth")
                edge_path = os.path.join(self.save_dir, "edge_best.pth")
                meta_path = os.path.join(self.save_dir, "meta_best.pth")
                self.logger.info("📥 加载最佳模型权重")
            elif epoch is not None:
                node_path = os.path.join(self.save_dir, f"node_epoch_{epoch}.pth")
                edge_path = os.path.join(self.save_dir, f"edge_epoch_{epoch}.pth")
                meta_path = os.path.join(self.save_dir, f"meta_epoch_{epoch}.pth")
                self.logger.info(f"📥 加载Epoch {epoch} 权重")
            else:
                raise ValueError("必须指定 epoch 或 load_best=True")

            for path in [node_path, edge_path, meta_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"权重文件不存在: {path}")

            updown_node(self.mhd_graph.nodes, node_path, mode="up", target_device=self.device)
            updown_edge(self.mhd_graph.edges, edge_path, mode="up", target_device=self.device)

            state = torch.load(meta_path, map_location=self.device, weights_only=True)
            self.optimizer.load_state_dict(state["optimizer"])

            if "history" in state:
                self.history["best_eval_value"] = state["history"]["best_eval_value"]
                self.history["best_epoch"] = state["history"]["best_epoch"]
                self.history["best_backward_value"] = state["history"].get("best_backward_value")

            if self.lr_scheduler and "lr_scheduler" in state and state["lr_scheduler"] is not None:
                self.lr_scheduler.load_state_dict(state["lr_scheduler"])

            self.logger.info("✅ 权重加载完成")
        except Exception as e:
            self.logger.error(f"❌ 加载权重失败: {str(e)}")
            raise

    def save_training_history(self):
        history_path = os.path.join(self.save_dir, "training_history.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(self.history, f, indent=4)
        self.logger.info(f"📝 训练历史已保存至: {history_path}")

    def train(self, train_data, eval_data, epochs: int = 10):
        self.logger.info("="*80)
        self.logger.info(f"🚀 开始训练，共 {epochs} 轮")
        self.logger.info("="*80)
        try:
            for epoch in range(epochs):
                self.logger.info(f"\n--- Epoch {epoch+1}/{epochs} ---")
                self.train_epoch(train_data, epoch)
                self.eval_epoch(eval_data, epoch)   # 内部只更新最佳模型

                cur_ep = epoch + 1
                if self.save_interval > 0 and cur_ep % self.save_interval == 0:
                    self.save_checkpoint(cur_ep)

            self.save_checkpoint(epochs)   # 最终常规保存
            self.save_training_history()
            self.logger.info("\n" + "="*80)
            self.logger.info("🎉 训练完成！")
            best_bw = self.history.get("best_backward_value")
            bw_str = f"{best_bw:.6f}" if best_bw is not None else "N/A"
            self.logger.info(f"🏆 最佳 {self.backward_node_name}: {bw_str} (Epoch {self.history['best_epoch']})")
            self.logger.info(f"🏆 最佳 {self.criteria_node_name}: {self.history['best_eval_value']:.6f} (Epoch {self.history['best_epoch']})")
            self.logger.info(f"📝 所有结果已保存至: {self.save_dir}")
            self.logger.info("="*80)
        except Exception as e:
            self.logger.error(f"❌ 训练过程出错: {str(e)}", exc_info=True)
            raise

# ===================== 优化器创建函数 =====================

def create_mhd_optimizer(
    mhd_graph,
    edge_optim_config: dict = None,
    default_optimizer_type: str = "adam",
    default_lr: float = 0.001,
    default_weight_decay: float = 0.0,
    **kwargs
) -> torch.optim.Optimizer:
    """
    创建MHD图优化器

    特性：
    1. 支持边级别的优化器配置
    2. 自动收集图参数
    """
    edge_optim_config = edge_optim_config or {}
    param_groups = []

    for edge_name, config in edge_optim_config.items():
        edge = mhd_graph.get_edge_by_name(edge_name)
        if edge is None:
            warnings.warn(f"边 {edge_name} 不存在，跳过自定义配置")
            continue
        edge_params = []
        for op in edge.edge_operations:
            if isinstance(op, nn.Module):
                edge_params.extend([p for p in op.parameters() if p.requires_grad])
        if edge_params:
            param_groups.append({
                "params": edge_params,
                "lr": config.get("lr", default_lr),
                "weight_decay": config.get("weight_decay", default_weight_decay),
                "name": edge_name
            })

    processed_ids = {id(p) for group in param_groups for p in group["params"]}
    remaining = [p for p in mhd_graph.parameters() if p.requires_grad and id(p) not in processed_ids]
    if remaining:
        param_groups.append({
            "params": remaining,
            "lr": default_lr,
            "weight_decay": default_weight_decay,
            "name": "default"
        })

    opt_map = {"adam": torch.optim.Adam, "sgd": torch.optim.SGD, "adamw": torch.optim.AdamW}
    if default_optimizer_type.lower() not in opt_map:
        raise ValueError(f"不支持的优化器: {default_optimizer_type}")

    if default_optimizer_type.lower() == "adam":
        kwargs.setdefault("betas", (0.9, 0.999))
        kwargs.setdefault("eps", 1e-8)
    elif default_optimizer_type.lower() == "sgd":
        kwargs.setdefault("momentum", 0.9)
        kwargs.setdefault("nesterov", True)

    optimizer = opt_map[default_optimizer_type.lower()](param_groups, **kwargs)

    logger = logging.getLogger("mhd_train")
    logger.info(f"✅ 优化器创建完成: {default_optimizer_type.upper()}")
    for i, group in enumerate(param_groups):
        logger.info(f"  组{i+1}: {group.get('name','unknown')} lr={group['lr']}, wd={group['weight_decay']}")
    return optimizer


# ===================== 推理器 =====================

class MHD_Inferencer:
    """
    MHD 推理器
    自动处理模型加载、batch 调整与前向推理，
    支持批量预测，以及按指定层级顺序前向传播以捕获中间节点状态。
    """
    def __init__(
        self,
        build_graph_fn: Callable[..., MHD_Graph],
        checkpoint_dir: str,
        device: torch.device = None,
        build_kwargs: dict = None
    ):
        """
        Args:
            build_graph_fn: 图构建函数，签名需为 (batch_size, ...) -> MHD_Graph
            checkpoint_dir: 保存有权重文件的目录（需包含 node_best.pth 和 edge_best.pth）
            device: 计算设备
            build_kwargs: 传递给 build_graph_fn 的其他参数（除 batch_size 和 device 外）
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.build_kwargs = build_kwargs or {}

        # 1. 用训练时的 batch_size 构建图并加载权重
        train_graph = build_graph_fn(batch_size=256, device=self.device, **self.build_kwargs)
        node_path = os.path.join(checkpoint_dir, "node_best.pth")
        edge_path = os.path.join(checkpoint_dir, "edge_best.pth")
        updown_node(train_graph.nodes, node_path, mode="up", target_device=self.device)
        updown_edge(train_graph.edges, edge_path, mode="up", target_device=self.device)

        # 2. 调整为 batch_size=1 的推理状态
        train_graph.update_batch_size(1)
        self.graph = train_graph
        print("✅ MHD Inferencer 初始化完成")

    @torch.no_grad()
    def predict(self, image_batch: torch.Tensor) -> List[int]:
        """
        批量推理
        Args:
            image_batch: shape (B, C, H, W)
        Returns:
            list of predicted class indices
        """
        B = image_batch.shape[0]
        self.graph.update_batch_size(B)
        for node in self.graph.nodes:
            node.reset()
        self.graph.get_node_by_name("input_img").current_state = image_batch.to(self.device).detach()
        self.graph.get_node_by_name("label_gt").current_state = torch.zeros(B, 100, device=self.device)
        self.graph.forward()
        logits = self.graph.get_node_by_name("logits").current_state
        return logits.argmax(dim=1).tolist()

# ===================== 孤立节点 & 孤立边自动修剪工具 =====================

def prune_isolated_graph(graph: MHD_Graph, verbose: bool = True) -> MHD_Graph:
    """
    自动检测并删除图中所有孤立节点和孤立边。

    孤立节点：在所有层级的 role 矩阵中，该列全为 0（无入度也无出度）。
    孤立边：在所有层级的 role 矩阵中，该行全为 0（未连接任何节点）。

    设计哲学：
        - Framework 允许孤立元素存在（纯数据载体或占位）。
        - Utils 提供主动清理机制，在训练前释放冗余显存。
        - 删除前会打印详细警告，列出所有受影响的节点和边，便于用户复核。

    Args:
        graph: MHD_Graph 实例
        verbose: 是否打印详细日志

    Returns:
        修剪后的原图实例（原地修改）
    """
    if not graph.topo or graph.num_levels == 0:
        if verbose:
            print("⚠️ 图无拓扑信息，跳过修剪。")
        return graph

    num_edges_orig = len(graph.edges)
    num_nodes_orig = len(graph.nodes)
    device = graph.device

    # ----- 1. 计算孤立节点掩码 -----
    # 聚合所有层级，按列（节点）检查是否出现过非零值
    node_active = torch.zeros(num_nodes_orig, dtype=torch.bool, device=device)
    for role in graph.topo.role_matrices:
        node_active = node_active | (role != 0).any(dim=0)   # any(dim=0) 按列

    isolated_node_ids = [i for i in range(num_nodes_orig) if not node_active[i]]
    isolated_nodes = [graph.get_node_by_id(i) for i in isolated_node_ids if graph.get_node_by_id(i) is not None]

    # ----- 2. 计算孤立边掩码 -----
    # 聚合所有层级，按行（边）检查是否出现过非零值
    edge_active = torch.zeros(num_edges_orig, dtype=torch.bool, device=device)
    for role in graph.topo.role_matrices:
        edge_active = edge_active | (role != 0).any(dim=1)   # any(dim=1) 按行

    isolated_edge_ids = [i for i in range(num_edges_orig) if not edge_active[i]]
    isolated_edges = [graph.get_edge_by_id(i) for i in isolated_edge_ids if graph.get_edge_by_id(i) is not None]

    # ----- 3. 若无孤立元素，直接返回 -----
    if not isolated_nodes and not isolated_edges:
        if verbose:
            print("✅ 未检测到孤立节点或孤立边，无需清理。")
        return graph

    # ----- 4. 打印警告（醒目输出） -----
    if verbose:
        print("=" * 80)
        if isolated_nodes:
            names = ", ".join([n.name for n in isolated_nodes])
            print(f"⚠️ 检测到 {len(isolated_nodes)} 个孤立节点（出入度均为0）: {names}")
        if isolated_edges:
            names = ", ".join([e.name for e in isolated_edges])
            print(f"⚠️ 检测到 {len(isolated_edges)} 条孤立边（未连接任何节点）: {names}")
        print("📌 提示：这些元素将被删除。若某些元素不应被删，请检查拓扑配置。")
        print("=" * 80)

    # ----- 5. 执行删除操作 -----
    # 5.1 计算保留的节点 ID 和边 ID
    keep_node_ids = [i for i in range(num_nodes_orig) if i not in isolated_node_ids]
    keep_edge_ids = [i for i in range(num_edges_orig) if i not in isolated_edge_ids]

    if not keep_node_ids or not keep_edge_ids:
        raise RuntimeError("❌ 删除后图为空（无节点或无边），操作被阻止。请检查拓扑配置。")

    # 5.2 裁剪拓扑矩阵（同时删除孤立行和孤立列）
    new_role_matrices = []
    new_sort_matrices = []
    keep_node_tensor = torch.tensor(keep_node_ids, device=device)
    keep_edge_tensor = torch.tensor(keep_edge_ids, device=device)

    for role, sort_mat in zip(graph.topo.role_matrices, graph.topo.sort_matrices):
        # 先按行删除孤立边，再按列删除孤立节点（顺序无关）
        new_role = role.index_select(dim=0, index=keep_edge_tensor)   # 行（边）
        new_role = new_role.index_select(dim=1, index=keep_node_tensor) # 列（节点）
        new_sort = sort_mat.index_select(dim=0, index=keep_edge_tensor)
        new_sort = new_sort.index_select(dim=1, index=keep_node_tensor)
        new_role_matrices.append(new_role)
        new_sort_matrices.append(new_sort)

    graph.topo.role_matrices = new_role_matrices
    graph.topo.sort_matrices = new_sort_matrices

    # 5.3 先从集合中取出对象，再修改基于 ID 的哈希值。
    # MHD_Node/MHD_Edge 的 __hash__ 依赖 id；对象留在 set 中时原地修改 id
    # 会破坏集合的哈希桶，导致后续查找或删除行为不确定。
    kept_nodes = sorted(
        (node for node in graph.nodes if node.id in keep_node_ids),
        key=lambda x: x.id,
    )
    kept_edges = sorted(
        (edge for edge in graph.edges if edge.id in keep_edge_ids),
        key=lambda x: x.id,
    )

    for new_id, node in enumerate(kept_nodes):
        node.id = new_id
    for new_id, edge in enumerate(kept_edges):
        edge.id = new_id

    # ID 稳定后重新创建集合，确保 set 的内部哈希与对象当前 ID 一致。
    graph.nodes = set(kept_nodes)
    graph.edges = set(kept_edges)

    # 5.5 重建内部索引（节点和边映射）
    graph._build_indices()

    # MHD_Graph 在初始化时已把所有边模块注册到 ModuleDict。仅更新
    # graph.edges 不会自动注销已裁剪模块，因此必须重建注册容器，保证
    # parameters/state_dict/optimizer/显存中只包含当前拓扑的活跃模块。
    graph.edge_module_map = nn.ModuleDict()
    graph._register_all_params()

    # 5.6 验证维度一致性
    graph.topo.validate_topo(len(graph.edges), len(graph.nodes))

    # 5.7 重新计算拓扑排序（因为边变了）
    graph.compact_topological_sort()

    if verbose:
        print(f"🧹 修剪完成！移除 {len(isolated_nodes)} 个孤立节点，{len(isolated_edges)} 条孤立边。")
        print(f"   当前节点数: {len(graph.nodes)}，边数: {len(graph.edges)}")

    return graph
