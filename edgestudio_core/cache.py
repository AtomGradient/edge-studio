# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""DSR KV-cache overlays for public ``mlx-lm``.

DSR (Dual-Sparse Retention) keeps three sparse components during decoding:
attention sinks, cumulative heavy hitters, and the recent local window.
"""

from __future__ import annotations

import sys
from typing import Any, Mapping, Sequence

import mlx.core as mx
from mlx.utils import tree_map, tree_reduce

from mlx_lm.models.cache import ArraysCache, KVCache, _BaseCache, create_attention_mask


class DSRCacheUnavailable(RuntimeError):
    """Raised when DSR cannot be installed on the active public MLX stack."""


def build_dsr_config(
    dsr_budget: int | None,
    *,
    sink_size: int = 4,
) -> dict[str, int] | None:
    """Build EdgeStudio's default DSR budget split."""

    if not dsr_budget:
        return None
    usable = max(int(dsr_budget) - int(sink_size), 0)
    return {
        "max_size": int(dsr_budget),
        "heavy_budget": usable // 2,
        "recent_budget": usable - usable // 2,
        "sink_size": int(sink_size),
    }


def _validate_config(config: Mapping[str, int]) -> dict[str, int]:
    required = ("max_size", "heavy_budget", "recent_budget")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"DSR config missing keys: {', '.join(missing)}")
    cfg = {
        "max_size": int(config["max_size"]),
        "heavy_budget": int(config["heavy_budget"]),
        "recent_budget": int(config["recent_budget"]),
        "sink_size": int(config.get("sink_size", 4)),
    }
    if min(cfg.values()) < 0 or cfg["max_size"] <= 0:
        raise ValueError(f"Invalid DSR config: {cfg!r}")
    if cfg["sink_size"] + cfg["heavy_budget"] + cfg["recent_budget"] > cfg["max_size"]:
        raise ValueError(f"DSR budget overflow: {cfg!r}")
    return cfg


class DSRKVCache(_BaseCache):
    """Dense KV cache with DSR score-based retention."""

    step = 256

    def __init__(
        self,
        max_size: int,
        heavy_budget: int,
        recent_budget: int,
        sink_size: int = 4,
    ):
        cfg = _validate_config(
            {
                "max_size": max_size,
                "heavy_budget": heavy_budget,
                "recent_budget": recent_budget,
                "sink_size": sink_size,
            }
        )
        self.max_size = cfg["max_size"]
        self.heavy_budget = cfg["heavy_budget"]
        self.recent_budget = cfg["recent_budget"]
        self.sink_size = cfg["sink_size"]
        self.keys = None
        self.values = None
        self.cumulative_scores = None
        self.offset = 0
        self._cache_len = 0

    def update_and_fetch(self, keys, values):
        prev = self._cache_len
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            batch, heads, _, key_dim = keys.shape
            value_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (batch, heads, n_steps * self.step, key_dim)
            v_shape = (batch, heads, n_steps * self.step, value_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self._cache_len += keys.shape[2]
        self.offset += keys.shape[2]
        self.keys[..., prev : self._cache_len, :] = keys
        self.values[..., prev : self._cache_len, :] = values
        return (
            self.keys[..., : self._cache_len, :],
            self.values[..., : self._cache_len, :],
        )

    def update_scores(self, scores):
        score_importance = mx.abs(scores)
        if self.cumulative_scores is None:
            self.cumulative_scores = score_importance
            return

        old_len = self.cumulative_scores.shape[-1]
        new_len = score_importance.shape[-1]
        if new_len > old_len:
            padding = mx.zeros(
                (*self.cumulative_scores.shape[:-1], new_len - old_len),
                dtype=self.cumulative_scores.dtype,
            )
            self.cumulative_scores = mx.concatenate(
                [self.cumulative_scores, padding], axis=-1
            )
        self.cumulative_scores = self.cumulative_scores * 0.95 + score_importance * 0.05

    def maybe_evict(self):
        if self.keys is None or self.cumulative_scores is None:
            return
        length = self._cache_len
        if length <= self.max_size:
            return

        middle_start = self.sink_size
        middle_end = length - self.recent_budget
        if middle_end <= middle_start:
            return

        middle_scores = self.cumulative_scores[..., :length][
            ..., middle_start:middle_end
        ]
        avg_middle_scores = middle_scores.mean(axis=1)
        middle_len = middle_end - middle_start
        keep_k = min(self.heavy_budget, middle_len)
        if keep_k >= middle_len:
            return

        top_indices = mx.argpartition(avg_middle_scores, kth=-keep_k, axis=-1)[
            ..., -keep_k:
        ]
        top_indices = mx.sort(top_indices, axis=-1) + middle_start
        sink_indices = mx.arange(self.sink_size)
        recent_indices = mx.arange(middle_end, length)
        keep_indices = mx.concatenate(
            [sink_indices, top_indices.squeeze(0), recent_indices]
        )

        active_keys = self.keys[..., :length, :]
        active_values = self.values[..., :length, :]
        self.keys = active_keys[..., keep_indices, :]
        self.values = active_values[..., keep_indices, :]
        self.cumulative_scores = self.cumulative_scores[..., keep_indices]
        self._cache_len = self.keys.shape[2]

    def size(self):
        return self._cache_len

    @property
    def state(self):
        if self.keys is None:
            return []
        if self._cache_len == self.keys.shape[2]:
            return self.keys, self.values
        return (
            self.keys[..., : self._cache_len, :],
            self.values[..., : self._cache_len, :],
        )

    @state.setter
    def state(self, value):
        if value is not None and value:
            self.keys, self.values = value
            self.offset = self.keys.shape[2]
            self._cache_len = self.keys.shape[2]

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.max_size,
                    self.heavy_budget,
                    self.recent_budget,
                    self.sink_size,
                    self.offset,
                    self._cache_len,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, value):
        (
            self.max_size,
            self.heavy_budget,
            self.recent_budget,
            self.sink_size,
            self.offset,
            self._cache_len,
        ) = map(int, value)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._cache_len, n)
        self._cache_len -= n
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        qcache = QuantizedDSRKVCache(
            group_size=group_size,
            bits=bits,
            max_size=self.max_size,
            heavy_budget=self.heavy_budget,
            recent_budget=self.recent_budget,
            sink_size=self.sink_size,
        )
        qcache.offset = self.offset
        qcache._cache_len = self._cache_len
        qcache.cumulative_scores = self.cumulative_scores
        if self.keys is not None:
            active_keys = self.keys[..., : self._cache_len, :]
            active_values = self.values[..., : self._cache_len, :]
            qcache.keys = mx.quantize(active_keys, group_size=group_size, bits=bits)
            qcache.values = mx.quantize(active_values, group_size=group_size, bits=bits)
            qcache._dtype = active_keys.dtype
        return qcache

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        total = self.keys.nbytes + self.values.nbytes
        if self.cumulative_scores is not None:
            total += self.cumulative_scores.nbytes
        return total


class QuantizedDSRKVCache(_BaseCache):
    """Quantized DSR KV cache that returns dense arrays for public MLX SDPA."""

    step = 256

    def __init__(
        self,
        group_size: int = 64,
        bits: int = 4,
        max_size: int = 1024,
        heavy_budget: int = 256,
        recent_budget: int = 512,
        sink_size: int = 4,
    ):
        cfg = _validate_config(
            {
                "max_size": max_size,
                "heavy_budget": heavy_budget,
                "recent_budget": recent_budget,
                "sink_size": sink_size,
            }
        )
        self._group_size = int(group_size)
        self._bits = int(bits)
        self.max_size = cfg["max_size"]
        self.heavy_budget = cfg["heavy_budget"]
        self.recent_budget = cfg["recent_budget"]
        self.sink_size = cfg["sink_size"]
        self.keys = None
        self.values = None
        self._dtype = None
        self.cumulative_scores = None
        self.offset = 0
        self._cache_len = 0

    def _active_dense(self):
        active_keys = tree_map(lambda x: x[..., : self._cache_len, :], self.keys)
        active_values = tree_map(lambda x: x[..., : self._cache_len, :], self.values)
        return (
            mx.dequantize(
                *active_keys,
                group_size=self._group_size,
                bits=self._bits,
            ),
            mx.dequantize(
                *active_values,
                group_size=self._group_size,
                bits=self._bits,
            ),
        )

    def update_and_fetch(self, keys, values):
        batch, heads, num_steps, key_dim = keys.shape
        value_dim = values.shape[-1]
        prev = self._cache_len
        self._dtype = keys.dtype

        if self.keys is None or (prev + num_steps) > self.keys[0].shape[-2]:
            el_per_int = 8 * mx.uint32.size // self._bits
            new_steps = (self.step + num_steps - 1) // self.step * self.step
            shape = (batch, heads, new_steps)

            def init_quant(dim):
                return (
                    mx.zeros((*shape, dim // el_per_int), dtype=mx.uint32),
                    mx.zeros((*shape, dim // self._group_size), dtype=keys.dtype),
                    mx.zeros((*shape, dim // self._group_size), dtype=keys.dtype),
                )

            def expand_quant(x):
                new_x = mx.zeros((*shape, x.shape[-1]), dtype=x.dtype)
                return mx.concatenate([x, new_x], axis=-2)

            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys, self.values = tree_map(
                        lambda x: x[..., :prev, :], (self.keys, self.values)
                    )
                self.keys, self.values = tree_map(
                    expand_quant, (self.keys, self.values)
                )
            else:
                self.keys = init_quant(key_dim)
                self.values = init_quant(value_dim)

        self._cache_len += num_steps
        self.offset += num_steps
        q_keys = mx.quantize(keys, group_size=self._group_size, bits=self._bits)
        q_values = mx.quantize(values, group_size=self._group_size, bits=self._bits)
        for index in range(len(self.keys)):
            self.keys[index][..., prev : self._cache_len, :] = q_keys[index]
            self.values[index][..., prev : self._cache_len, :] = q_values[index]
        return self._active_dense()

    def update_scores(self, scores):
        score_importance = mx.abs(scores)
        if self.cumulative_scores is None:
            self.cumulative_scores = score_importance
            return
        old_len = self.cumulative_scores.shape[-1]
        new_len = score_importance.shape[-1]
        if new_len > old_len:
            padding = mx.zeros(
                (*self.cumulative_scores.shape[:-1], new_len - old_len),
                dtype=self.cumulative_scores.dtype,
            )
            self.cumulative_scores = mx.concatenate(
                [self.cumulative_scores, padding], axis=-1
            )
        self.cumulative_scores = self.cumulative_scores * 0.95 + score_importance * 0.05

    def maybe_evict(self):
        if self.keys is None or self.cumulative_scores is None:
            return
        length = self._cache_len
        if length <= self.max_size:
            return
        middle_start = self.sink_size
        middle_end = length - self.recent_budget
        if middle_end <= middle_start:
            return
        middle_scores = self.cumulative_scores[..., :length][
            ..., middle_start:middle_end
        ]
        avg_middle_scores = middle_scores.mean(axis=1)
        middle_len = middle_end - middle_start
        keep_k = min(self.heavy_budget, middle_len)
        if keep_k >= middle_len:
            return
        top_indices = mx.argpartition(avg_middle_scores, kth=-keep_k, axis=-1)[
            ..., -keep_k:
        ]
        top_indices = mx.sort(top_indices, axis=-1) + middle_start
        keep_indices = mx.concatenate(
            [
                mx.arange(self.sink_size),
                top_indices.squeeze(0),
                mx.arange(middle_end, length),
            ]
        )
        self.keys = tree_map(lambda x: x[..., :length, :][..., keep_indices, :], self.keys)
        self.values = tree_map(
            lambda x: x[..., :length, :][..., keep_indices, :], self.values
        )
        self.cumulative_scores = self.cumulative_scores[..., keep_indices]
        self._cache_len = keep_indices.shape[0]

    def size(self):
        return self._cache_len

    @property
    def state(self):
        if self.keys is None:
            return []
        return tree_map(lambda x: x[..., : self._cache_len, :], (self.keys, self.values))

    @state.setter
    def state(self, value):
        if value is not None and value:
            self.keys, self.values = value

    @property
    def meta_state(self):
        return tuple(
            map(
                str,
                (
                    self.offset,
                    self._cache_len,
                    self._group_size,
                    self._bits,
                    self.max_size,
                    self.heavy_budget,
                    self.recent_budget,
                    self.sink_size,
                ),
            )
        )

    @meta_state.setter
    def meta_state(self, value):
        (
            self.offset,
            self._cache_len,
            self._group_size,
            self._bits,
            self.max_size,
            self.heavy_budget,
            self.recent_budget,
            self.sink_size,
        ) = map(int, value)

    def is_trimmable(self):
        return True

    def trim(self, n):
        n = min(self._cache_len, n)
        self._cache_len -= n
        self.offset -= n
        return n

    def make_mask(self, *args, **kwargs):
        return create_attention_mask(*args, offset=self.offset, **kwargs)

    def empty(self):
        return self.keys is None

    @property
    def nbytes(self):
        if self.keys is None:
            return 0
        total = tree_reduce(lambda acc, x: acc + x.nbytes, (self.keys, self.values), 0)
        if self.cumulative_scores is not None:
            total += self.cumulative_scores.nbytes
        return total


class MixedPrecisionDSRKVCache(DSRKVCache):
    """Dense DSR cache with mixed precision noise simulation."""

    def __init__(
        self,
        heavy_bits: int = 8,
        recent_bits: int = 4,
        group_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._heavy_bits = int(heavy_bits)
        self._recent_bits = int(recent_bits)
        self._group_size = int(group_size)

    def _apply_quant_noise(self, x, bits):
        return mx.dequantize(
            *mx.quantize(x, group_size=self._group_size, bits=bits),
            group_size=self._group_size,
            bits=bits,
        )

    def update_and_fetch(self, keys, values):
        return super().update_and_fetch(
            self._apply_quant_noise(keys, self._heavy_bits),
            self._apply_quant_noise(values, self._heavy_bits),
        )

    def maybe_evict(self):
        prev_len = self._cache_len
        super().maybe_evict()
        if self._cache_len < prev_len and self.keys is not None:
            heavy_end = self._cache_len - self.recent_budget
            if 0 < heavy_end < self._cache_len:
                self.keys[..., heavy_end:, :] = self._apply_quant_noise(
                    self.keys[..., heavy_end:, :],
                    self._recent_bits,
                )
                self.values[..., heavy_end:, :] = self._apply_quant_noise(
                    self.values[..., heavy_end:, :],
                    self._recent_bits,
                )


def _compute_attention_scores(queries, keys, scale: float, mask=None):
    if isinstance(keys, tuple):
        # Public MLX does not expose quantized SDPA score kernels. Dequantizing
        # here keeps the release path correct at the cost of extra decode work.
        raise DSRCacheUnavailable(
            "Quantized DSR score capture requires dense keys on public MLX."
        )

    batch, n_q_heads, q_len, _ = queries.shape
    n_kv_heads = keys.shape[1]
    n_repeats = n_q_heads // n_kv_heads

    if n_repeats > 1:
        q = queries.reshape(batch, n_kv_heads, n_repeats, q_len, -1)
        k = mx.expand_dims(keys, axis=2)
        scores = (q * scale) @ k.swapaxes(-1, -2)
        scores = scores.mean(axis=2)
    else:
        scores = (queries * scale) @ keys.swapaxes(-1, -2)

    if mask is not None and not isinstance(mask, str):
        mask_value = mask
        while getattr(mask_value, "ndim", 0) > scores.ndim:
            mask_value = mask_value.squeeze(-2)
        if mask_value.dtype == mx.bool_:
            scores = mx.where(mask_value, scores, mx.finfo(scores.dtype).min)
        else:
            scores = scores + mask_value

    if scores.shape[-2] == 1:
        return scores.squeeze(-2)
    return scores.mean(axis=-2)


def _dsr_sdpa_wrapper(original):
    def scaled_dot_product_attention(
        queries,
        keys,
        values,
        cache,
        scale: float,
        mask,
        sinks=None,
    ):
        output = original(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )
        if (
            cache is not None
            and hasattr(cache, "update_scores")
            and hasattr(cache, "maybe_evict")
            and queries.shape[2] == 1
        ):
            scores = _compute_attention_scores(queries, keys, scale, mask=mask)
            cache.update_scores(scores)
            cache.maybe_evict()
        return output

    scaled_dot_product_attention._edgestudio_core_dsr = True
    scaled_dot_product_attention._edgestudio_core_original = original
    return scaled_dot_product_attention


def install_dsr_attention_patch() -> None:
    """Patch loaded ``mlx-lm`` attention call sites for DSR score capture."""

    import mlx_lm.models.base as base

    current = base.scaled_dot_product_attention
    if getattr(current, "_edgestudio_core_dsr", False):
        wrapper = current
        original = current._edgestudio_core_original
    else:
        original = current
        wrapper = _dsr_sdpa_wrapper(original)
        base.scaled_dot_product_attention = wrapper

    for module_name in (
        "mlx_lm.models.qwen3_next",
        "mlx_lm.models.qwen3_5",
        "mlx_lm.models.qwen3",
        "mlx_lm.models.qwen2",
        "mlx_lm.models.llama",
    ):
        try:
            __import__(module_name)
        except Exception:
            continue

    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith("mlx_lm.models."):
            continue
        if getattr(module, "scaled_dot_product_attention", None) is original:
            setattr(module, "scaled_dot_product_attention", wrapper)


def _num_layers(model: Any) -> int:
    layers = getattr(model, "layers", None)
    if layers is not None:
        return len(layers)
    nested = getattr(model, "model", None)
    layers = getattr(nested, "layers", None)
    if layers is not None:
        return len(layers)
    nested = getattr(model, "language_model", None)
    layers = getattr(nested, "layers", None)
    if layers is not None:
        return len(layers)
    raise ValueError("Cannot determine number of model layers for DSR cache")


def make_dsr_prompt_cache(
    model: Any,
    dsr_config: Mapping[str, int] | Sequence[Mapping[str, int]],
) -> list[Any]:
    """Create a prompt cache where attention layers use DSR retention."""

    install_dsr_attention_patch()
    num_layers = _num_layers(model)
    layers = getattr(model, "layers", None)
    if layers is None and hasattr(model, "model"):
        layers = getattr(model.model, "layers", None)
    if layers is None and hasattr(model, "language_model"):
        layers = getattr(model.language_model, "layers", None)

    if isinstance(dsr_config, Sequence) and not isinstance(dsr_config, Mapping):
        if len(dsr_config) != num_layers:
            raise ValueError(
                f"DSR config list length ({len(dsr_config)}) != num_layers ({num_layers})"
            )
        configs = [_validate_config(cfg) for cfg in dsr_config]
    else:
        cfg = _validate_config(dsr_config)  # type: ignore[arg-type]
        configs = [cfg for _ in range(num_layers)]

    caches: list[Any] = []
    for index in range(num_layers):
        layer = layers[index] if layers is not None else None
        if getattr(layer, "is_linear", False):
            caches.append(ArraysCache(size=2))
        else:
            caches.append(DSRKVCache(**configs[index]))
    return caches


__all__ = [
    "DSRCacheUnavailable",
    "DSRKVCache",
    "QuantizedDSRKVCache",
    "MixedPrecisionDSRKVCache",
    "build_dsr_config",
    "install_dsr_attention_patch",
    "make_dsr_prompt_cache",
]
