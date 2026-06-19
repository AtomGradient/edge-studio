# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Architecture tree data model for representing model structure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ArchNode:
    """A node in the model architecture tree.

    Represents a logical component (model, sub-model, layer group, single layer, etc.)
    """
    name: str
    node_type: str  # "model", "submodel", "layer_group", "layer", "module", "tensor"
    weight_prefix: str = ""  # prefix for matching tensors (e.g., "talker.model.layers.0")
    config_params: dict[str, Any] = field(default_factory=dict)
    param_count: int = 0  # logical (original) parameter count
    stored_param_count: int = 0  # actual stored element count (may differ for quantized models)
    size_bytes: int = 0
    children: list[ArchNode] = field(default_factory=list)
    pruning_info: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_param_count(self) -> int:
        if self.children:
            return self.param_count + sum(c.total_param_count for c in self.children)
        return self.param_count

    @property
    def total_stored_param_count(self) -> int:
        if self.children:
            return self.stored_param_count + sum(c.total_stored_param_count for c in self.children)
        return self.stored_param_count

    @property
    def total_size_bytes(self) -> int:
        if self.children:
            return self.size_bytes + sum(c.total_size_bytes for c in self.children)
        return self.size_bytes

    @property
    def is_quantized(self) -> bool:
        """Whether this node has quantization (logical != stored params)."""
        return self.total_param_count != self.total_stored_param_count

    def find_by_prefix(self, prefix: str) -> Optional[ArchNode]:
        if self.weight_prefix == prefix:
            return self
        for child in self.children:
            result = child.find_by_prefix(prefix)
            if result:
                return result
        return None

    def flatten(self) -> list[ArchNode]:
        result = [self]
        for child in self.children:
            result.extend(child.flatten())
        return result


@dataclass
class ModelArchitecture:
    """Complete model architecture description."""
    model_type: str
    model_name: str
    model_dir: str
    root: ArchNode
    config: dict[str, Any] = field(default_factory=dict)
    quantization: Optional[dict[str, Any]] = None
    total_params: int = 0
    total_stored_params: int = 0
    total_size_bytes: int = 0


def format_param_count(count: int) -> str:
    """Format parameter count in human-readable form."""
    if count >= 1e9:
        return f"{count / 1e9:.2f}B"
    elif count >= 1e6:
        return f"{count / 1e6:.1f}M"
    elif count >= 1e3:
        return f"{count / 1e3:.1f}K"
    return str(count)


def format_size(size_bytes: int) -> str:
    """Format byte size in human-readable form."""
    if size_bytes >= 1e9:
        return f"{size_bytes / 1e9:.2f} GB"
    elif size_bytes >= 1e6:
        return f"{size_bytes / 1e6:.1f} MB"
    elif size_bytes >= 1e3:
        return f"{size_bytes / 1e3:.1f} KB"
    return f"{size_bytes} B"
