# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Load activation profile data from JSON or NPZ files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LayerActivation:
    """Activation data for a single layer's MLP neurons."""
    layer_idx: int
    max_activations: np.ndarray   # shape [intermediate_size]
    mean_activations: np.ndarray  # shape [intermediate_size]

    @property
    def intermediate_size(self) -> int:
        return len(self.max_activations)

    def dead_neuron_count(self, threshold: float) -> int:
        return int(np.sum(self.max_activations < threshold))

    def dead_neuron_ratio(self, threshold: float) -> float:
        return self.dead_neuron_count(threshold) / max(self.intermediate_size, 1)

    def alive_indices(self, threshold: float) -> np.ndarray:
        return np.where(self.max_activations >= threshold)[0]


@dataclass
class ActivationProfile:
    """Complete activation profile for a model."""
    intermediate_size: int
    num_layers: int
    run_count: int
    layers: list[LayerActivation] = field(default_factory=list)
    source_file: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def max_acts_matrix(self) -> np.ndarray:
        """Return [num_layers, intermediate_size] matrix of max activations."""
        return np.stack([l.max_activations for l in self.layers])

    @property
    def mean_acts_matrix(self) -> np.ndarray:
        """Return [num_layers, intermediate_size] matrix of mean activations."""
        return np.stack([l.mean_activations for l in self.layers])

    def dead_neurons_per_layer(self, threshold: float) -> list[int]:
        return [l.dead_neuron_count(threshold) for l in self.layers]

    def total_dead_neurons(self, threshold: float) -> int:
        return sum(self.dead_neurons_per_layer(threshold))

    def summary(self, threshold: float = 0.1) -> dict:
        total = self.num_layers * self.intermediate_size
        dead = self.total_dead_neurons(threshold)
        return {
            "num_layers": self.num_layers,
            "intermediate_size": self.intermediate_size,
            "total_neurons": total,
            "dead_neurons": dead,
            "dead_ratio": dead / max(total, 1),
            "threshold": threshold,
            "run_count": self.run_count,
        }


def load_profile_json(file_path: str) -> ActivationProfile:
    """Load activation profile from JSON format.

    Expected schema:
    {
      "intermediate_size": int,
      "num_layers": int,
      "run_count": int,  (or "num_runs")
      "layers": [
        {"layer": int, "max_activations": [float...], "mean_activations": [float...]}
      ]
    }
    """
    with open(file_path) as f:
        data = json.load(f)

    layers = []
    for entry in data["layers"]:
        layers.append(LayerActivation(
            layer_idx=entry["layer"],
            max_activations=np.array(entry["max_activations"], dtype=np.float32),
            mean_activations=np.array(entry["mean_activations"], dtype=np.float32),
        ))

    return ActivationProfile(
        intermediate_size=data["intermediate_size"],
        num_layers=data["num_layers"],
        run_count=data.get("run_count") or data.get("num_runs", 0),
        layers=layers,
        source_file=file_path,
        metadata={k: v for k, v in data.items() if k not in ("layers",)},
    )


def load_profile_npz(file_path: str) -> ActivationProfile:
    """Load activation profile from NPZ format.

    Expected arrays: global_max_acts [L, I], global_mean_acts [L, I]
    """
    data = np.load(file_path)
    max_acts = data["global_max_acts"]   # [num_layers, intermediate_size]
    mean_acts = data["global_mean_acts"]

    num_layers, intermediate_size = max_acts.shape
    layers = []
    start_layer = int(data.get("start_layer", 0))
    for i in range(num_layers):
        layers.append(LayerActivation(
            layer_idx=start_layer + i,
            max_activations=max_acts[i].astype(np.float32),
            mean_activations=mean_acts[i].astype(np.float32),
        ))

    return ActivationProfile(
        intermediate_size=int(intermediate_size),
        num_layers=int(num_layers),
        run_count=int(data.get("num_runs", 0)),
        layers=layers,
        source_file=file_path,
        metadata={
            "thresholds": data["thresholds"].tolist() if "thresholds" in data else [],
            "speaker": str(data.get("speaker", "")),
        },
    )


def load_profile(file_path: str) -> ActivationProfile:
    """Auto-detect format and load activation profile."""
    path = Path(file_path)
    if path.suffix == ".npz":
        return load_profile_npz(file_path)
    elif path.suffix == ".json":
        return load_profile_json(file_path)
    else:
        raise ValueError(f"Unsupported profile format: {path.suffix}")


def find_profile_files(model_dir: str) -> list[str]:
    """Search for activation profile files in/near a model directory."""
    results = []
    model_path = Path(model_dir)

    # Check model dir itself
    for pattern in ["*profile*.json", "*profile*.npz", "*activation*.json", "*activation*.npz"]:
        results.extend(str(p) for p in model_path.glob(pattern))

    # Check parent directory (profiles are often saved alongside model dirs)
    parent = model_path.parent
    for pattern in ["*profile*.json", "*profile*.npz"]:
        results.extend(str(p) for p in parent.glob(pattern))

    return sorted(set(results))
