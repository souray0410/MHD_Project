"""RETFound 2D model and its minimal MHD V4 hypergraph.

The ViT-L/16 definition follows the official RETFound repository. RETFound
weights are licensed CC BY-NC 4.0; this experiment does not redistribute them.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from timm.models.vision_transformer import VisionTransformer as TimmVisionTransformer
    from timm.layers import trunc_normal_
except ImportError as exc:  # pragma: no cover - exercised by the remote setup check
    raise ImportError("RETFound requires timm==0.9.2; install requirements.txt") from exc

try:  # source-tree layout used by this experiment directory
    from V4.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo
except ImportError:
    from MHD_Project.MHD_Framework_V4 import MHD_Edge, MHD_Graph, MHD_Node, MHD_Topo


class RETFoundVisionTransformer(TimmVisionTransformer):
    """Official RETFound global-pooling Vision Transformer behavior."""

    def __init__(self, global_pool: bool = True, **kwargs) -> None:
        super().__init__(**kwargs)
        self.global_pool = bool(global_pool)
        if self.global_pool:
            norm_layer = kwargs["norm_layer"]
            self.fc_norm = norm_layer(kwargs["embed_dim"])
            del self.norm

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = self._pos_embed(x)
        x = self.patch_drop(x)
        x = self.norm_pre(x)
        x = self.blocks(x)
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            return self.fc_norm(x)
        return self.norm(x)[:, 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RETFound's historical timm base called only ``head`` after its
        # already-pooled forward_features result. Modern timm forward_head
        # performs pooling itself, so spell out the original behavior.
        return self.head(self.forward_features(x))


def create_retfound_vit_large(num_classes: int = 5) -> RETFoundVisionTransformer:
    """Create the official 2D RETFound ViT-L/16 classifier architecture."""

    return RETFoundVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        global_pool=True,
    )


def _interpolate_position_embedding(
    model: RETFoundVisionTransformer,
    state: dict[str, torch.Tensor],
) -> None:
    position = state.get("pos_embed")
    if position is None or position.shape == model.pos_embed.shape:
        return
    embedding_size = position.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    old_patch_tokens = position.shape[-2] - num_extra_tokens
    old_size = int(old_patch_tokens ** 0.5)
    new_size = int(num_patches ** 0.5)
    if old_size * old_size != old_patch_tokens or new_size * new_size != num_patches:
        raise ValueError("RETFound position embedding does not describe a square patch grid")
    extra = position[:, :num_extra_tokens]
    patches = position[:, num_extra_tokens:]
    patches = patches.reshape(-1, old_size, old_size, embedding_size).permute(0, 3, 1, 2)
    patches = F.interpolate(patches, size=(new_size, new_size), mode="bicubic", align_corners=False)
    patches = patches.permute(0, 2, 3, 1).flatten(1, 2)
    state["pos_embed"] = torch.cat((extra, patches), dim=1)


@dataclass(frozen=True)
class WeightLoadReport:
    checkpoint: str
    tensor_count: int
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    removed_classifier_keys: tuple[str, ...]

    @property
    def pretrained_backbone_loaded(self) -> bool:
        return self.tensor_count > 100 and not self.unexpected_keys


def load_retfound_weights(
    model: RETFoundVisionTransformer,
    checkpoint_path: str | Path,
) -> WeightLoadReport:
    """Load an official RETFound checkpoint without executing pickled code."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    # Official RETFound checkpoints contain an argparse.Namespace with training
    # metadata. Allow only that known type instead of disabling weights-only mode.
    with torch.serialization.safe_globals([argparse.Namespace]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, Mapping):
        raise TypeError("RETFound checkpoint must be a mapping")
    for key in ("model", "state_dict"):
        nested = checkpoint.get(key)
        if isinstance(nested, Mapping):
            checkpoint = nested
            break
    state: dict[str, torch.Tensor] = {}
    for raw_name, value in checkpoint.items():
        if not isinstance(raw_name, str) or not isinstance(value, torch.Tensor):
            continue
        name = raw_name.removeprefix("module.")
        state[name] = value
    if not state:
        raise ValueError("RETFound checkpoint contains no tensor state")

    model_state = model.state_dict()
    removed: list[str] = []
    for name in ("head.weight", "head.bias"):
        if name in state and name in model_state and state[name].shape != model_state[name].shape:
            del state[name]
            removed.append(name)
    _interpolate_position_embedding(model, state)
    incompatible = model.load_state_dict(state, strict=False)
    missing = tuple(incompatible.missing_keys)
    unexpected = tuple(incompatible.unexpected_keys)
    allowed_missing = {"head.weight", "head.bias"}
    disallowed_missing = set(missing) - allowed_missing
    if disallowed_missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not an official-compatible RETFound ViT-L state: "
            f"missing={sorted(disallowed_missing)}, unexpected={sorted(unexpected)}"
        )
    if removed:
        trunc_normal_(model.head.weight, std=2e-5)
        if model.head.bias is not None:
            nn.init.zeros_(model.head.bias)
    return WeightLoadReport(
        checkpoint=str(path),
        tensor_count=len(state),
        missing_keys=missing,
        unexpected_keys=unexpected,
        removed_classifier_keys=tuple(removed),
    )


def select_finetune_parameters(
    model: RETFoundVisionTransformer,
    trainable_blocks: int = 1,
) -> tuple[int, int]:
    """Freeze the backbone except the requested final blocks, norm, and head."""

    if not 0 <= trainable_blocks <= len(model.blocks):
        raise ValueError(f"trainable_blocks must be in [0, {len(model.blocks)}]")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if trainable_blocks:
        for block in model.blocks[-trainable_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    for module in (model.fc_norm, model.head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


class PatchAndPositionStage(nn.Module):
    """Image patches, class token, position embedding, and input normalization."""

    def __init__(self, model: RETFoundVisionTransformer) -> None:
        super().__init__()
        self.patch_embed = model.patch_embed
        self.cls_token = model.cls_token
        self.pos_embed = model.pos_embed
        self.pos_drop = model.pos_drop
        self.patch_drop = model.patch_drop
        self.norm_pre = model.norm_pre

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(image)
        cls_token = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls_token, tokens), dim=1)
        tokens = self.pos_drop(tokens + self.pos_embed)
        tokens = self.patch_drop(tokens)
        return self.norm_pre(tokens)


class GlobalPoolStage(nn.Module):
    """RETFound patch-token average and final normalization."""

    def __init__(self, model: RETFoundVisionTransformer) -> None:
        super().__init__()
        self.fc_norm = model.fc_norm

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.fc_norm(tokens[:, 1:, :].mean(dim=1))


def _message(tensor: torch.Tensor) -> MHD_Node.Message:
    return MHD_Node.Message(initial_state=tensor)


def _operation(function) -> MHD_Edge.Operation:
    return MHD_Edge.Operation(function=function)


RETF_FOUND_FORWARD_LEVELS = tuple(range(28))
RETF_FOUND_BACKWARD_LEVELS = tuple(range(28, 56))


def build_retfound_graph(
    model: RETFoundVisionTransformer,
    *,
    device: torch.device,
    num_classes: int = 5,
    batch_size: int = 1,
) -> MHD_Graph:
    """Express every official RETFound Transformer block in the MHD topology."""

    nodes = {
        MHD_Node(0, "image", _message(torch.zeros(batch_size, 3, 224, 224))),
        MHD_Node(1, "target", _message(torch.zeros(batch_size, dtype=torch.long))),
        MHD_Node(2, "patch_tokens", _message(torch.zeros(batch_size, 197, 1024))),
    }
    for block_index in range(24):
        nodes.add(
            MHD_Node(
                3 + block_index,
                f"block_{block_index + 1:02d}",
                _message(torch.zeros(batch_size, 197, 1024)),
            )
        )
    nodes.update(
        {
            MHD_Node(27, "representation", _message(torch.zeros(batch_size, 1024))),
            MHD_Node(28, "logits", _message(torch.zeros(batch_size, num_classes))),
            MHD_Node(29, "loss", _message(torch.zeros(()))),
        }
    )

    edges = {
        MHD_Edge(0, "patch_and_position", [_operation(PatchAndPositionStage(model))])
    }
    for block_index, block in enumerate(model.blocks):
        edges.add(
            MHD_Edge(
                1 + block_index,
                f"transformer_block_{block_index + 1:02d}",
                [_operation(block)],
            )
        )
    edges.update(
        {
            MHD_Edge(25, "global_pool_and_norm", [_operation(GlobalPoolStage(model))]),
            MHD_Edge(26, "classification_head", [_operation(model.head)]),
            MHD_Edge(27, "criterion", [_operation(nn.CrossEntropyLoss())]),
        }
    )

    complete_role = torch.zeros((28, 30), dtype=torch.int64)
    chain = (
        [(0, 2)]
        + [(2 + index, 3 + index) for index in range(24)]
        + [(26, 27), (27, 28)]
    )
    for edge_id, (head_id, tail_id) in enumerate(chain):
        complete_role[edge_id, head_id] = -1
        complete_role[edge_id, tail_id] = 1
    # criterion(logits, target) -> loss
    complete_role[27, 28] = -1
    complete_role[27, 1] = -1
    complete_role[27, 29] = 1
    complete_sort = torch.zeros_like(complete_role)
    for edge_id, (head_id, tail_id) in enumerate(chain):
        complete_sort[edge_id, head_id] = 0
        complete_sort[edge_id, tail_id] = 1
    complete_sort[27, 28] = 0
    complete_sort[27, 1] = 1
    complete_sort[27, 29] = 2

    # One global topology list: levels 0..27 execute the model in its natural
    # forward order; levels 28..55 reverse those same Edge connections from
    # criterion back to patch embedding. Every matrix keeps the global 28x30
    # Edge/Node shape and activates exactly one real model operation.
    forward_roles: list[torch.Tensor] = []
    forward_sorts: list[torch.Tensor] = []
    for edge_id in range(28):
        role = torch.zeros_like(complete_role)
        sort = torch.zeros_like(complete_sort)
        role[edge_id] = complete_role[edge_id]
        sort[edge_id] = complete_sort[edge_id]
        forward_roles.append(role)
        forward_sorts.append(sort)
    backward_roles = [(-matrix).clone() for matrix in reversed(forward_roles)]
    backward_sorts = [matrix.clone() for matrix in reversed(forward_sorts)]
    topo = MHD_Topo(
        forward_roles + backward_roles,
        forward_sorts + backward_sorts,
    )
    return MHD_Graph(nodes, edges, {topo}, device=device)
