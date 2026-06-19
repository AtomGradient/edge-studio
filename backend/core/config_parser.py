# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Abstract base class for model config parsers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .architecture import ArchNode, ModelArchitecture
from .weight_loader import WeightIndex


class ConfigParser(ABC):
    """Base class for model-specific configuration parsers.

    Each model type (Qwen3 TTS, Gemma 3, etc.) implements its own parser
    to handle its unique config structure.
    """

    def __init__(self, model_dir: str, config: dict[str, Any], weight_index: WeightIndex):
        self.model_dir = model_dir
        self.config = config
        self.weight_index = weight_index

    @abstractmethod
    def parse(self) -> ModelArchitecture:
        """Parse config and weight index into a ModelArchitecture."""
        ...

    @abstractmethod
    def model_type_name(self) -> str:
        """Return human-readable model type name."""
        ...

    def _count_params_for_prefix(self, prefix: str) -> tuple[int, int, int]:
        """Count parameters and size for tensors matching a prefix.

        Returns (logical_params, stored_params, size_bytes) where:
        - logical_params: original parameter count (before quantization)
        - stored_params: actual stored element count
        - size_bytes: on-disk storage size
        """
        quant_cfg = self.config.get("quantization_config") or self.config.get("quantization") or {}
        group_size = quant_cfg.get("group_size", 64)

        logical_params = 0
        stored_params = 0
        total_size = 0
        for name, meta in self.weight_index.tensors.items():
            if not name.startswith(prefix):
                continue
            total_size += meta.size_bytes
            stored_params += meta.num_elements

            # Skip scales/biases for logical count (quantization overhead)
            if name.endswith(".scales") or name.endswith(".biases"):
                continue

            # Quantized weight: compute logical param count from scales shape
            if meta.dtype == "U32" and name.endswith(".weight"):
                base = name[:-len(".weight")]
                scales_meta = self.weight_index.tensors.get(f"{base}.scales")
                if scales_meta is not None:
                    rows = meta.shape[0]
                    logical_cols = scales_meta.shape[-1] * group_size
                    logical_params += rows * logical_cols
                    continue

            logical_params += meta.num_elements
        return logical_params, stored_params, total_size

    def _count_exact_tensors(self, names: list[str]) -> tuple[int, int, int]:
        """Count params and size for exact tensor names.

        Returns (logical_params, stored_params, size_bytes).
        """
        quant_cfg = self.config.get("quantization_config") or self.config.get("quantization") or {}
        group_size = quant_cfg.get("group_size", 64)

        logical_params = 0
        stored_params = 0
        total_size = 0
        for name in names:
            meta = self.weight_index.tensors.get(name)
            if not meta:
                continue
            total_size += meta.size_bytes
            stored_params += meta.num_elements

            if name.endswith(".scales") or name.endswith(".biases"):
                continue

            if meta.dtype == "U32" and name.endswith(".weight"):
                base = name[:-len(".weight")]
                scales_meta = self.weight_index.tensors.get(f"{base}.scales")
                if scales_meta is not None:
                    rows = meta.shape[0]
                    logical_cols = scales_meta.shape[-1] * group_size
                    logical_params += rows * logical_cols
                    continue

            logical_params += meta.num_elements
        return logical_params, stored_params, total_size

    def _make_layer_node(
        self,
        layer_idx: int,
        prefix: str,
        config_params: dict[str, Any],
        node_name: str | None = None,
    ) -> ArchNode:
        """Create an ArchNode for a single transformer layer."""
        layer_prefix = f"{prefix}.{layer_idx}"
        logical, stored, size = self._count_params_for_prefix(layer_prefix + ".")
        return ArchNode(
            name=node_name or f"Layer {layer_idx}",
            node_type="layer",
            weight_prefix=layer_prefix,
            config_params=config_params,
            param_count=logical,
            stored_param_count=stored,
            size_bytes=size,
        )

    def _make_layer_group(
        self,
        name: str,
        prefix: str,
        num_layers: int,
        layer_config: dict[str, Any],
        per_layer_intermediate_sizes: list[int] | None = None,
    ) -> ArchNode:
        """Create an ArchNode for a group of transformer layers."""
        children = []
        for i in range(num_layers):
            lc = dict(layer_config)
            if per_layer_intermediate_sizes and i < len(per_layer_intermediate_sizes):
                lc["intermediate_size"] = per_layer_intermediate_sizes[i]
            children.append(self._make_layer_node(i, prefix, lc))

        total_params = sum(c.param_count for c in children)
        total_size = sum(c.size_bytes for c in children)

        return ArchNode(
            name=name,
            node_type="layer_group",
            weight_prefix=prefix,
            config_params={"num_layers": num_layers},
            param_count=0,  # params are in children
            size_bytes=0,
            children=children,
        )


def load_config(model_dir: str) -> dict[str, Any]:
    """Load config.json from a model directory."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    with open(config_path) as f:
        return json.load(f)


def load_sub_config(model_dir: str, subdir: str) -> dict[str, Any] | None:
    """Load config.json from a subdirectory (e.g., speech_tokenizer/config.json)."""
    config_path = Path(model_dir) / subdir / "config.json"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return json.load(f)
