# -*- coding: utf-8 -*-
"""
Multi-Hypergraph Dynamic Framework (MHD) - Version 3.1
Author: Souray Meng (孟号丁)
Core Framework: Hypergraph-based computational graph with multi-level topology
License: MIT
"""

import torch
import numpy as np
import torch.nn as nn
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
from functools import partial
import warnings
import re

# 全局配置：抑制无关警告
warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def parse_string_operation(op_str: str, x: torch.Tensor) -> torch.Tensor:
    """
    解析并执行字符串形式的张量操作（新规范：直接拼接 x + op_str）

    用户需提供以 '.' 开头的操作字符串，如 ".relu()" 或 ".mean(dim=1)"，
    函数将直接执行 x.op_str，无需额外解析。

    Args:
        op_str: 操作字符串，必须以 '.' 开头，如 ".relu()" 或 ".mean(dim=1)"
        x: 输入张量

    Returns:
        操作后的张量
    """
    expression = "x" + op_str
    return eval(expression, {"x": x})


# ===================== 核心超图框架类 =====================

@dataclass
class MHD_Node:
    """
    超图节点类 - 双状态设计（初始状态 + 当前状态）

    特性：
    1. initial_state 为初始状态（用于重置）
    2. current_state 为当前状态（前向传播中动态更新）
    3. transfer_mode 控制状态融合方式（sum/avg/max/min/mul）

    Attributes:
        id: 节点唯一标识符
        name: 节点名称
        initial_state: 初始状态张量
        current_state: 当前状态张量
        transfer_mode: 状态转移模式，决定如何融合输入张量
    """
    id: int
    name: str
    initial_state: torch.Tensor
    current_state: torch.Tensor = field(default=None)   # 允许为 None
    transfer_mode: str = "replace"

    def __post_init__(self):
        if self.current_state is None:
            self.current_state = self.initial_state.clone()
        """初始化验证：确保设备一致性"""
        if self.initial_state.device != self.current_state.device:
            raise ValueError(
                f"节点 {self.name} 初始状态和当前状态设备不一致: "
                f"{self.initial_state.device} vs {self.current_state.device}"
            )
        if self.initial_state.shape != self.current_state.shape:
            raise ValueError(
                f"节点 {self.name} 初始状态和当前状态形状不匹配: "
                f"{self.initial_state.shape} vs {self.current_state.shape}"
            )

    def __hash__(self):
        """基于ID的哈希函数"""
        return hash(self.id)

    def __eq__(self, other):
        """基于ID的相等判断"""
        if not isinstance(other, MHD_Node):
            return False
        return self.id == other.id

    def reset(self) -> 'MHD_Node':
        """
        重置当前状态为初始状态的克隆

        注意：创建新张量以切断计算图依赖

        Returns:
            重置后的节点自身
        """
        self.current_state = self.initial_state.clone(memory_format=torch.contiguous_format)
        return self

    def update_initial(self, new_tensor: torch.Tensor, update_current: bool = True) -> 'MHD_Node':
        """
        更新初始状态（用于数据加载或外部注入）

        Args:
            new_tensor: 新的初始状态张量
            update_current: 是否同时更新当前状态

        Returns:
            更新后的节点自身
        """
        self.initial_state = new_tensor
        if update_current:
            self.current_state = new_tensor.clone(memory_format=torch.contiguous_format)
        return self

    def to_device(self, device: torch.device) -> 'MHD_Node':
        """
        将节点状态迁移到指定设备

        Args:
            device: 目标计算设备

        Returns:
            设备迁移后的节点自身
        """
        if self.initial_state.device != device:
            self.initial_state = self.initial_state.to(device, non_blocking=True)
        if self.current_state.device != device:
            self.current_state = self.current_state.to(device, non_blocking=True)
        return self

    def apply_transfer_mode(self, current: torch.Tensor, incoming: torch.Tensor) -> torch.Tensor:
        """
        应用状态转移函数，融合当前状态与输入张量

        Args:
            current: 当前节点状态（如 current_state）
            incoming: 边的输出张量

        Returns:
            融合后的新状态张量
        """
        mode = self.transfer_mode
        if mode == "sum":
            return current + incoming
        elif mode == "avg":
            return (current + incoming) / 2.0
        elif mode == "max":
            return torch.max(current, incoming)
        elif mode == "min":
            return torch.min(current, incoming)
        elif mode == "mul":
            return current * incoming
        elif mode == "replace":
                # 直接用 incoming 覆盖，忽略 current
                return incoming
        else:
            raise ValueError(f"不支持的 transfer_mode: {mode}")


@dataclass
class MHD_Edge:
    """
    超图边类 - 可学习模块的载体（保持不变）

    特性：
    1. 包含 edge_operations 操作序列，支持 nn.Module、字符串操作及 partial/可调用对象
    2. 所有数据混合与分发由 edge_operations 实现
    3. 可学习参数通过 nn.Module 封装
    4. nn.Module / partial / 可调用对象 统一使用智能匹配：先解包，再整体传入，最后广播

    Attributes:
        id: 边唯一标识符
        name: 边名称
        edge_operations: 操作序列，元素可为 str（".方法名" 形式）、nn.Module 或 partial/可调用对象
    """
    id: int
    name: str
    edge_operations: List[Union[str, nn.Module, partial]]

    def __hash__(self):
        """基于ID的哈希函数"""
        return hash(self.id)

    def __eq__(self, other):
        """基于ID的相等判断"""
        if not isinstance(other, MHD_Edge):
            return False
        return self.id == other.id

    def to_device(self, device: torch.device) -> 'MHD_Edge':
        """
        将边中的可学习模块迁移到指定设备

        Args:
            device: 目标计算设备

        Returns:
            设备迁移后的边自身
        """
        for idx, op in enumerate(self.edge_operations):
            if isinstance(op, nn.Module):
                self.edge_operations[idx] = op.to(device, non_blocking=True)
        return self

    @staticmethod
    def _apply_callable_to_list(op, tensor_list: List[torch.Tensor]):
        """
        智能地将可调用对象 op（nn.Module / partial / 函数）应用到张量列表上。
        优先级：
          1) 解包调用：op(*tensor_list)   ← 支持多参数模块/函数
          2) 整体调用：op(tensor_list)    ← 支持需要整个列表的操作
          3) 广播：   [op(t) for t in tensor_list]  ← 最后的保底
        返回处理后的结果（张量或张量列表）。
        """
        try:
            result = op(*tensor_list)
            return result
        except (TypeError, ValueError, RuntimeError):
            pass
        try:
            result = op(tensor_list)
            return result
        except (TypeError, ValueError, RuntimeError):
            pass
        return [op(t) for t in tensor_list]

    def execute_edge_operations(self, input_list: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        执行 edge_operations 操作序列

        支持列表与单张量的自动广播/聚合，最终输出强制转换为列表

        Args:
            input_list: 输入张量列表（头节点状态，按 sort_matrix 排序）

        Returns:
            输出张量列表
        """
        data = input_list
        for op in self.edge_operations:
            if isinstance(data, list):
                if isinstance(op, str):
                    data = [parse_string_operation(op, t) for t in data]
                elif isinstance(op, (nn.Module, partial)) or callable(op):
                    data = self._apply_callable_to_list(op, data)
                else:
                    raise TypeError(f"不支持的操作类型: {type(op)}")
            else:
                if isinstance(op, str):
                    data = parse_string_operation(op, data)
                elif isinstance(op, (nn.Module, partial)) or callable(op):
                    data = op(data)
                else:
                    raise TypeError(f"不支持的操作类型: {type(op)}")
        if isinstance(data, torch.Tensor):
            data = [data]
        elif not isinstance(data, list):
            raise TypeError(f"edge_operations 最终输出类型错误: {type(data)}")
        return data


@dataclass
class MHD_Topo:
    """
    超图拓扑类 - 多层级矩阵列表

    特性：
    1. role_matrices 和 sort_matrices 是等长的列表，每个元素对应一个执行层级
    2. 每个层级内角色矩阵元素为 -1/0/1，同一条边可在不同层级出现
    3. 所有矩阵形状必须一致（边数 × 节点数）

    Attributes:
        role_matrices: 角色矩阵列表，形状均为 (边数, 节点数)
        sort_matrices: 排序矩阵列表，形状均为 (边数, 节点数)
    """
    role_matrices: List[torch.Tensor] = field(default_factory=list)
    sort_matrices: List[torch.Tensor] = field(default_factory=list)

    def __post_init__(self):
        """初始化验证：确保列表非空且各矩阵维度、设备一致"""
        if not self.role_matrices or not self.sort_matrices:
            raise ValueError("role_matrices 和 sort_matrices 不能为空")
        if len(self.role_matrices) != len(self.sort_matrices):
            raise ValueError("role_matrices 与 sort_matrices 长度必须相同")
        # 以第一层的形状为基准
        ref_shape = self.role_matrices[0].shape
        ref_device = self.role_matrices[0].device
        for i, (r, s) in enumerate(zip(self.role_matrices, self.sort_matrices)):
            if r.shape != ref_shape:
                raise ValueError(f"第{i}层 role 矩阵形状 {r.shape} 与第0层 {ref_shape} 不一致")
            if s.shape != ref_shape:
                raise ValueError(f"第{i}层 sort 矩阵形状 {s.shape} 与第0层 {ref_shape} 不一致")
            if r.device != ref_device:
                raise ValueError(f"第{i}层 role 矩阵设备 {r.device} 与第0层 {ref_device} 不一致")
            if s.device != ref_device:
                raise ValueError(f"第{i}层 sort 矩阵设备 {s.device} 与第0层 {ref_device} 不一致")

    def to_device(self, device: torch.device) -> 'MHD_Topo':
        """
        将所有层级的矩阵迁移到指定设备

        Args:
            device: 目标计算设备

        Returns:
            设备迁移后的拓扑自身
        """
        for i in range(len(self.role_matrices)):
            if self.role_matrices[i].device != device:
                self.role_matrices[i] = self.role_matrices[i].to(device, non_blocking=True)
            if self.sort_matrices[i].device != device:
                self.sort_matrices[i] = self.sort_matrices[i].to(device, non_blocking=True)
        return self

    def get_topo(self, level: int, edge_id: int, node_id: int, matrix_type: str = "role") -> int:
        """
        获取指定层级、边和节点的拓扑值

        Args:
            level: 层级索引
            edge_id: 边索引
            node_id: 节点索引
            matrix_type: 矩阵类型，'role'或'sort'

        Returns:
            拓扑值，如果索引越界返回0
        """
        matrices = self.role_matrices if matrix_type == "role" else self.sort_matrices
        if 0 <= level < len(matrices):
            mat = matrices[level]
            if 0 <= edge_id < mat.shape[0] and 0 <= node_id < mat.shape[1]:
                return int(mat[edge_id, node_id].item())
        return 0

    def to_list(self, matrix_type: str = "role") -> List[List[List[int]]]:
        """
        转换为嵌套列表形式（层级 × 边 × 节点）

        Args:
            matrix_type: 矩阵类型，'role'或'sort'

        Returns:
            三维列表
        """
        matrices = self.role_matrices if matrix_type == "role" else self.sort_matrices
        return [m.tolist() for m in matrices]

    def __hash__(self):
        """基于所有矩阵内容的哈希函数"""
        flat_role = tuple(tuple(r.flatten().tolist()) for r in self.role_matrices)
        flat_sort = tuple(tuple(s.flatten().tolist()) for s in self.sort_matrices)
        return hash((flat_role, flat_sort))

    def __eq__(self, other):
        """基于所有矩阵内容的相等判断"""
        if not isinstance(other, MHD_Topo):
            return False
        if len(self.role_matrices) != len(other.role_matrices):
            return False
        for r1, r2 in zip(self.role_matrices, other.role_matrices):
            if not torch.equal(r1, r2):
                return False
        for s1, s2 in zip(self.sort_matrices, other.sort_matrices):
            if not torch.equal(s1, s2):
                return False
        return True

    def validate_topo(self, num_edges: int, num_nodes: int) -> None:
        """
        验证所有层级的拓扑矩阵维度

        Args:
            num_edges: 预期的边数
            num_nodes: 预期的节点数

        Raises:
            ValueError: 当维度不匹配时
        """
        for i, (r, s) in enumerate(zip(self.role_matrices, self.sort_matrices)):
            if r.shape[0] != num_edges or r.shape[1] != num_nodes:
                raise ValueError(
                    f"第{i}层拓扑维度不匹配: 边{num_edges}×节点{num_nodes}，实际{r.shape}"
                )

    @property
    def num_levels(self) -> int:
        """返回拓扑的层级数"""
        return len(self.role_matrices)


class MHD_Graph(nn.Module):
    """
    多超图动态框架核心类 - Version 3.0（多层级拓扑 + 双状态节点）

    特性：
    1. 节点采用 initial_state / current_state 双状态，避免显存膨胀
    2. 拓扑使用矩阵列表，每层独立 -1/0/1 角色矩阵
    3. 前向传播逐层执行，同一层内维护动态 current 字典，保证顺序依赖
    4. 层间通过 current_state 传递状态（循环网络展开）
    5. 支持同一条边在多层级复用
    6. 合并图自动对齐层级，缺失层零填充
    7. 可视化支持层级筛选

    Author: Souray Meng (孟号丁)
    """

    def __init__(self, nodes: Set[MHD_Node], edges: Set[MHD_Edge], topos: Set[MHD_Topo],
                 device: torch.device = None):
        """
        初始化MHD图

        Args:
            nodes: 节点集合，每个节点包含 initial_state 和 current_state
            edges: 边集合
            topos: 拓扑集合（应只包含一个 MHD_Topo 对象）
            device: 计算设备，默认为CUDA(可用)或CPU
        """
        super().__init__()

        # 统一设备配置
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 核心超图对象
        self.nodes = nodes
        self.edges = edges

        # 拓扑验证：确保只有一个拓扑对象
        if len(topos) != 1:
            warnings.warn(f"拓扑集合包含 {len(topos)} 个元素，预期为1，将使用第一个")
        self.topo = next(iter(topos)) if topos else None

        # 统一设备
        self._unify_device()

        # 建立索引系统
        self._node_by_id: Dict[int, MHD_Node] = {}
        self._node_by_name: Dict[str, MHD_Node] = {}
        self._edge_by_id: Dict[int, MHD_Edge] = {}
        self._edge_by_name: Dict[str, MHD_Edge] = {}
        self._build_indices()

        # 参数注册容器
        self.edge_module_map = nn.ModuleDict()

        # 获取层级数
        self.num_levels = self.topo.num_levels if self.topo else 0

        # 验证拓扑维度
        if self.topo:
            self.topo.validate_topo(len(self.edges), len(self.nodes))

        # 拓扑排序和参数注册
        self.compact_topological_sort()
        self._register_all_params()

        print(f"✅ MHD图初始化完成 | 设备: {self.device} | 节点: {len(nodes)} | 边: {len(edges)} | 层级: {self.num_levels}")

    def _unify_device(self) -> None:
        """
        统一所有组件的计算设备
        """
        if self.topo:
            self.topo = self.topo.to_device(self.device)
        for node in self.nodes:
            node.to_device(self.device)
        for edge in self.edges:
            edge.to_device(self.device)

    def _build_indices(self) -> None:
        """
        构建节点和边的索引字典
        """
        self._node_by_id.clear()
        self._node_by_name.clear()
        self._edge_by_id.clear()
        self._edge_by_name.clear()

        for node in self.nodes:
            if node.id in self._node_by_id:
                warnings.warn(f"节点ID重复: {node.id}，后出现的将覆盖前者")
            if node.name in self._node_by_name:
                warnings.warn(f"节点名称重复: {node.name}，后出现的将覆盖前者")
            self._node_by_id[node.id] = node
            self._node_by_name[node.name] = node

        for edge in self.edges:
            if edge.id in self._edge_by_id:
                warnings.warn(f"边ID重复: {edge.id}，后出现的将覆盖前者")
            if edge.name in self._edge_by_name:
                warnings.warn(f"边名称重复: {edge.name}，后出现的将覆盖前者")
            self._edge_by_id[edge.id] = edge
            self._edge_by_name[edge.name] = edge

    def update_indices(self, new_nodes: Set[MHD_Node] = None, new_edges: Set[MHD_Edge] = None) -> None:
        """
        更新索引（当节点或边集合发生变化时调用）
        """
        if new_nodes:
            for node in new_nodes:
                self._node_by_id[node.id] = node
                self._node_by_name[node.name] = node
        if new_edges:
            for edge in new_edges:
                self._edge_by_id[edge.id] = edge
                self._edge_by_name[edge.name] = edge

    def get_node_by_id(self, node_id: int) -> Optional[MHD_Node]:
        """通过ID获取节点"""
        return self._node_by_id.get(node_id)

    def get_node_by_name(self, node_name: str) -> Optional[MHD_Node]:
        """通过名称获取节点"""
        return self._node_by_name.get(node_name)

    def get_edge_by_id(self, edge_id: int) -> Optional[MHD_Edge]:
        """通过ID获取边"""
        return self._edge_by_id.get(edge_id)

    def get_edge_by_name(self, edge_name: str) -> Optional[MHD_Edge]:
        """通过名称获取边"""
        return self._edge_by_name.get(edge_name)

    def compact_topological_sort(self) -> 'MHD_Graph':
        """
        分层拓扑排序：对每个层级独立进行边拓扑排序，生成每层的边执行序列

        Returns:
            排序后的图自身

        Raises:
            ValueError: 当某层检测到环时
        """
        if self.topo is None or self.num_levels == 0:
            self._edge_sequence_per_level = []
            print("✅ 拓扑分析完成（无role矩阵，默认按ID排序）")
            return self

        num_edges = len(self.edges)
        num_nodes = len(self.nodes)
        self._edge_sequence_per_level = []

        for level in range(self.num_levels):
            role = self.topo.role_matrices[level]
            # 找出当前层级活跃的边
            active_edge_ids = [eid for eid in range(num_edges) if (role[eid] != 0).any().item()]
            if not active_edge_ids:
                self._edge_sequence_per_level.append([])
                continue

            # 构建边依赖：头节点（负值）依赖所有输出到该节点的边（正值）
            node_to_out_edges = defaultdict(set)
            for eid in active_edge_ids:
                for nid in range(num_nodes):
                    if role[eid, nid] > 0:
                        node_to_out_edges[nid].add(eid)

            edge_deps = defaultdict(set)
            for eid in active_edge_ids:
                for nid in range(num_nodes):
                    if role[eid, nid] < 0:
                        edge_deps[eid].update(node_to_out_edges.get(nid, set()))
                edge_deps[eid].discard(eid)

            edge_in_degree = {eid: len(edge_deps.get(eid, set())) for eid in active_edge_ids}
            reverse_deps = defaultdict(set)
            for eid, deps in edge_deps.items():
                for dep in deps:
                    reverse_deps[dep].add(eid)

            remaining = set(active_edge_ids)
            level_seq = []
            while remaining:
                current = sorted([e for e in remaining if edge_in_degree[e] == 0])
                if not current:
                    edge_names = [self.get_edge_by_id(eid).name for eid in remaining]
                    raise ValueError(f"第{level}层边拓扑存在环！剩余边: {edge_names}")
                level_seq.extend(current)
                for eid in current:
                    remaining.remove(eid)
                    for next_eid in reverse_deps.get(eid, set()):
                        edge_in_degree[next_eid] -= 1
            self._edge_sequence_per_level.append(level_seq)

        print("✅ 分层拓扑分析完成（每层独立）:")
        for l, seq in enumerate(self._edge_sequence_per_level):
            names = [self.get_edge_by_id(eid).name for eid in seq]
            print(f" ├─ Level {l}: {names}")
        return self

    def _register_all_params(self) -> 'MHD_Graph':
        """
        注册边中的可学习模块到 ModuleDict
        """
        for edge in sorted(self.edges, key=lambda x: x.id):
            for idx, op in enumerate(edge.edge_operations):
                if isinstance(op, nn.Module):
                    module_name = f"edge_{edge.name}_op_{idx}"
                    self.edge_module_map[module_name] = op.to(self.device)
                    edge.edge_operations[idx] = self.edge_module_map[module_name]
        return self

    def sort_nodes_by_topo(self, level: int = 0, edge_id: int = 0) -> List[Tuple[int, int]]:
        """
        按指定层级和边的 sort_matrix 排序节点

        Args:
            level: 层级索引，默认为0
            edge_id: 边索引

        Returns:
            排序后的 (节点索引, 排序值) 列表
        """
        if self.topo is None or level >= self.num_levels:
            return []
        if edge_id >= self.topo.sort_matrices[level].shape[0]:
            return []
        indexed = list(enumerate(self.topo.sort_matrices[level][edge_id].tolist()))
        return sorted(indexed, key=lambda p: p[1])

    def update_batch_size(self, batch_size: int) -> None:
        """
        动态调整所有节点的初始状态和当前状态的 batch 维度。
        确保内部状态与输入数据的 batch 大小一致。
        """
        for node in self.nodes:
            old_shape = list(node.initial_state.shape)
            if len(old_shape) < 1 or old_shape[0] == batch_size:
                continue
            template = node.initial_state[0:1]
            repeats = [batch_size] + [1] * (template.ndim - 1)
            node.initial_state = template.repeat(repeats).clone()
            node.current_state = node.initial_state.clone()

    def forward(self, levels: Optional[List[int]] = None) -> 'MHD_Graph':
        """
        多层级前向传播（双状态版本，无状态列表）

        每一层内部使用动态字典 node_current 维护当前层内最新状态。
        层开始时，从节点的 current_state 初始化 node_current。
        层内边按拓扑序执行，边从 node_current 读取头节点，更新尾节点的 node_current。
        层结束后，将 node_current 写回节点的 current_state，供下一层使用。

        Args:
            levels: 要执行的层级索引列表，按顺序执行；默认为 None，执行所有层 [0,1,...,num_levels-1]
        """
        if not self.nodes or not self.edges or self.topo is None:
            return self

        # 确定要执行的层级
        if levels is None:
            levels = list(range(self.num_levels))

        num_edges = len(self.edges)
        num_nodes = len(self.nodes)

        for level in levels:
            if level < 0 or level >= self.num_levels:
                raise IndexError(f"层级索引 {level} 超出范围 [0, {self.num_levels-1}]")
            role = self.topo.role_matrices[level]
            sort_mat = self.topo.sort_matrices[level]
            edge_seq = self._edge_sequence_per_level[level]

            # 用节点当前的 current_state 初始化本层工作状态
            node_current = {}
            for node in self.nodes:
                node_current[node.id] = node.current_state

            for edge_id in edge_seq:
                edge = self.get_edge_by_id(edge_id)
                if not edge or edge_id >= num_edges:
                    continue

                head_ids = [nid for nid in range(num_nodes) if role[edge_id, nid] < 0]
                tail_ids = [nid for nid in range(num_nodes) if role[edge_id, nid] > 0]
                if not head_ids or not tail_ids:
                    continue

                head_sorted = sorted(head_ids, key=lambda nid: sort_mat[edge_id, nid].item())
                tail_sorted = sorted(tail_ids, key=lambda nid: sort_mat[edge_id, nid].item())

                # 从 node_current 读取头节点状态
                head_tensors = []
                for nid in head_sorted:
                    if nid in node_current:
                        head_tensors.append(node_current[nid])
                if not head_tensors:
                    continue

                output_list = edge.execute_edge_operations(head_tensors)
                if len(output_list) != len(tail_sorted):
                    raise ValueError(
                        f"边 '{edge.name}' 输出数量 ({len(output_list)}) 与尾节点数 ({len(tail_sorted)}) 不匹配"
                    )

                # 更新尾节点的当前状态（融合）
                for idx, nid in enumerate(tail_sorted):
                    current_val = node_current.get(nid, self.get_node_by_id(nid).current_state)
                    incoming = output_list[idx]
                    new_state = self.get_node_by_id(nid).apply_transfer_mode(current_val, incoming)
                    if new_state.device != self.device:
                        new_state = new_state.to(self.device, non_blocking=True)
                    node_current[nid] = new_state

            # 层结束：将本层最终状态写回节点的 current_state
            for node in self.nodes:
                if node.id in node_current:
                    node.current_state = node_current[node.id]

        return self

    def generate_mermaid(self, levels: Union[int, slice, List[int], None] = None) -> str:
        """
        生成 Mermaid 可视化描述，可指定绘制层级范围

        Args:
            levels: 层级选择。默认为 None 表示全部；
                    可为 int（单层）、slice 或 list。
                    当为 list 时，按列表顺序（保持去重）绘制连接。

        Returns:
            Mermaid 图描述字符串
        """
        if self.topo is None or self.num_levels == 0:
            levels_iter = []
        elif levels is None:
            levels_iter = list(range(self.num_levels))
        elif isinstance(levels, int):
            levels_iter = [levels]
        elif isinstance(levels, slice):
            levels_iter = list(range(self.num_levels))[levels]
        elif isinstance(levels, list):
            # 保持传入顺序，同时去重（dict.fromkeys 保留顺序）
            levels_iter = list(dict.fromkeys(levels))
        else:
            raise TypeError("levels 参数类型应为 int / slice / list / None")

        mermaid = [
            "graph TD",
            "",
            " classDef MHD_Node_Style fill:#fff7e6,stroke:#fa8c16,stroke-width:2px,rounded:1",
            " classDef MHD_Edge_Style fill:#e6f7ff,stroke:#1890ff,stroke-width:2px,rounded:1",
            "",
        ]

        # 添加节点
        for node in sorted(self.nodes, key=lambda x: x.id):
            mermaid.append(f" {node.name}:::MHD_Node_Style")

        # 添加边（合并选定层级中的连接，按 levels_iter 的顺序叠加）
        for edge in sorted(self.edges, key=lambda x: x.id):
            edge_id = edge.id
            head_names = set()
            tail_names = set()
            for lvl in levels_iter:
                if lvl >= self.num_levels:
                    continue
                role = self.topo.role_matrices[lvl]
                if edge_id >= role.shape[0]:
                    continue
                row = role[edge_id]
                for nid in range(row.shape[0]):
                    if row[nid] < 0:
                        n = self.get_node_by_id(nid)
                        if n:
                            head_names.add(n.name)
                    elif row[nid] > 0:
                        n = self.get_node_by_id(nid)
                        if n:
                            tail_names.add(n.name)
            if not head_names and not tail_names:
                continue
            mermaid.append(f" {edge.name}:::MHD_Edge_Style")
            for hn in sorted(head_names):
                mermaid.append(f" {hn} --> {edge.name}")
            for tn in sorted(tail_names):
                mermaid.append(f" {edge.name} --> {tn}")
            mermaid.append("")

        mermaid_code = "\n".join(mermaid)
        print("=== 超图可视化 ===")
        print(mermaid_code)
        return mermaid_code

    # ---------- 图合并辅助 ----------
    @staticmethod
    def _merge_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
        """
        合并多个子图的节点状态，始终采用均值融合（无序且稳定）。
        自动处理 dtype 转换。
        """
        dtypes = {t.dtype for t in tensors}
        if len(dtypes) > 1:
            raise ValueError(f"合并时张量 dtype 不一致: {dtypes}")
        target_dtype = dtypes.pop()
        float_tensors = [t.float() for t in tensors]
        stacked = torch.stack(float_tensors, dim=0)
        mean_float = stacked.mean(dim=0)
        if target_dtype != torch.float32:
            return mean_float.to(target_dtype)
        return mean_float

    @classmethod
    def merge_graph(cls, graphs: Set['MHD_Graph'], device: torch.device = None) -> 'MHD_Graph':
        """
        合并多个子图为一个超图，自动按名称合并节点和边。
        同名节点状态取均值，同名边操作浅拷贝（要求同名的边具有相同模块引用）。
        自动对齐层级数，缺失层零填充。

        Args:
            graphs: 要合并的子图集合（节点和边的名称须符合合并语义）
            device: 目标设备

        Returns:
            合并后的新 MHD_Graph 实例
        """
        if not graphs:
            raise ValueError("图集合不能为空")

        target_device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- 1. 按名称分组节点和边 ----
        node_groups = defaultdict(list)      # name -> list of MHD_Node
        edge_groups = defaultdict(list)      # name -> list of MHD_Edge
        # 记录每个子图的本地 ID -> 名称映射
        sub_info = []  # (graph, dict(local_nid -> name), dict(local_eid -> name))
        for graph in graphs:
            local_nid_to_name = {node.id: node.name for node in graph.nodes}
            local_eid_to_name = {edge.id: edge.name for edge in graph.edges}
            for node in graph.nodes:
                node_groups[node.name].append(node)
            for edge in graph.edges:
                edge_groups[edge.name].append(edge)
            sub_info.append((graph, local_nid_to_name, local_eid_to_name))

        # ---- 2. 合并节点 ----
        merged_nodes = set()
        global_nid = 0
        name_to_global_nid = {}

        for name, n_list in node_groups.items():
            modes = {n.transfer_mode for n in n_list}
            if len(modes) > 1:
                raise ValueError(f"节点 '{name}' 在不同子图中的 transfer_mode 不一致: {modes}")
            mode = n_list[0].transfer_mode

            merged_init = cls._merge_tensors([n.initial_state for n in n_list])
            merged_curr = cls._merge_tensors([n.current_state for n in n_list])
            new_node = MHD_Node(
                id=global_nid,
                name=name,
                initial_state=merged_init,
                current_state=merged_curr,
                transfer_mode=mode
            )
            merged_nodes.add(new_node)
            name_to_global_nid[name] = global_nid
            global_nid += 1

        # 为每个子图建立本地节点 ID -> 全局节点 ID 映射
        graph_id_to_global_node = {}
        for graph in graphs:
            local_map = {node.id: name_to_global_nid[node.name] for node in graph.nodes}
            graph_id_to_global_node[id(graph)] = local_map

        # ---- 3. 合并边 ----
        merged_edges = set()
        global_eid = 0
        name_to_global_eid = {}

        for name, e_list in edge_groups.items():
            # 直接取第一条边的操作（浅拷贝），实际场景中同名边应共享模块
            new_edge = MHD_Edge(
                id=global_eid,
                name=name,
                edge_operations=e_list[0].edge_operations.copy()
            )
            merged_edges.add(new_edge)
            name_to_global_eid[name] = global_eid
            global_eid += 1

        # 为每个子图建立本地边 ID -> 全局边 ID 映射
        graph_id_to_global_edge = {}
        for graph in graphs:
            local_map = {edge.id: name_to_global_eid[edge.name] for edge in graph.edges}
            graph_id_to_global_edge[id(graph)] = local_map

        # ---- 4. 拓扑合并 ----
        max_levels = max((g.num_levels for g in graphs), default=0)
        num_edges = len(merged_edges)
        num_nodes = len(merged_nodes)
        merged_role_list, merged_sort_list = [], []

        for level in range(max_levels):
            role = torch.zeros((num_edges, num_nodes), dtype=torch.int64, device=target_device)
            sort = torch.zeros((num_edges, num_nodes), dtype=torch.int64, device=target_device)
            for graph in graphs:
                if level >= graph.num_levels:
                    continue
                sub_role = graph.topo.role_matrices[level]
                sub_sort = graph.topo.sort_matrices[level]
                local_e2g = graph_id_to_global_edge[id(graph)]
                local_n2g = graph_id_to_global_node[id(graph)]
                for local_eid in range(sub_role.shape[0]):
                    global_eid = local_e2g.get(local_eid)
                    if global_eid is None:
                        continue
                    for local_nid in range(sub_role.shape[1]):
                        rv = sub_role[local_eid, local_nid].item()
                        sv = sub_sort[local_eid, local_nid].item()
                        if rv == 0 and sv == 0:
                            continue
                        global_nid = local_n2g.get(local_nid)
                        if global_nid is not None:
                            role[global_eid, global_nid] = rv
                            sort[global_eid, global_nid] = sv
            merged_role_list.append(role)
            merged_sort_list.append(sort)

        merged_topo = MHD_Topo(role_matrices=merged_role_list, sort_matrices=merged_sort_list)

        # ---- 5. 创建合并图 ----
        merged_graph = cls(
            nodes=merged_nodes,
            edges=merged_edges,
            topos={merged_topo},
            device=target_device
        )
        print(f"✅ 图合并完成 | 设备: {target_device} | 节点: {num_nodes} | 边: {num_edges} | 层级: {max_levels}")
        return merged_graph