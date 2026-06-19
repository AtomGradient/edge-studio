# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""State readout → per-layer K/V prefix tensors.

The memory stack produces a single ``(B, state_dim_total)`` readout after
consuming the user's event stream. We project that to per-layer K/V
prefix tensors of shape ``(L, B, n_kv_heads, K, head_dim)``.

``MemoryToPrefixHead`` is a small 2-layer MLP per K/V (shared across
layers via a single weight matrix with the layer index baked into the
output shape) — big enough to learn non-trivial bias but small enough
not to blow up parameter count for prefix-memory experiments.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn


class MemoryToPrefixHead(nn.Module):
    """Project state readout → prefix K + prefix V tensors.

    Parameters
    ----------
    state_dim : int
        Total readout dim (sum across multi-freq branches if applicable).
    n_layers : int
        Number of transformer layers in the base model.
    n_kv_heads : int
        KV heads per layer (Qwen3.5-4B has grouped-query attention with
        fewer KV heads than Q heads — pass the KV count).
    head_dim : int
        Per-head dimension.
    n_prefix_tokens : int
        Number of prefix tokens per layer (K). 32 or 64 are reasonable.
    hidden_mult : int
        MLP hidden expansion factor. Default 4.
    """

    def __init__(
        self,
        *,
        state_dim: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        n_prefix_tokens: int = 64,
        hidden_mult: int = 4,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_prefix = n_prefix_tokens

        # Flat output dim per K or V branch: L × H × K × D
        out_dim = n_layers * n_kv_heads * n_prefix_tokens * head_dim
        hidden = hidden_mult * state_dim

        self.mlp_k = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )
        self.mlp_v = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

        # Zero-init the final Linear of each MLP so prefix_k = prefix_v = 0
        # at step 0. Rationale (see S2b HANDOFF_2026-04-29 §3.6): Kaiming-
        # init prefix noise from step 0 floods the base model's full-attn
        # caches with 260k+ dims of random K/V, washing out pretrained
        # attention. With zero start, seed_prefix_kv_selective still writes
        # empty slabs — base behaves identically to the no-injection case —
        # and the head's gradient carves out a non-trivial prefix only as
        # the training loss demands. This is the standard trick for
        # stabilising prompt-prefix tuning (also used in P-Tuning-v2,
        # Prefix-Tuning, and residual adapters with gated init).
        def _zero_last_linear(seq: nn.Sequential) -> None:
            last = seq.layers[-1]
            last.weight = mx.zeros_like(last.weight)
            if getattr(last, "bias", None) is not None:
                last.bias = mx.zeros_like(last.bias)

        _zero_last_linear(self.mlp_k)
        _zero_last_linear(self.mlp_v)

    def __call__(self, state_readout: mx.array) -> tuple[mx.array, mx.array]:
        """Project to ``(prefix_k, prefix_v)``.

        Parameters
        ----------
        state_readout : mx.array, shape ``(B, state_dim)``

        Returns
        -------
        prefix_k : mx.array, shape ``(n_layers, B, n_kv_heads, K, head_dim)``
        prefix_v : mx.array, shape ``(n_layers, B, n_kv_heads, K, head_dim)``
        """
        B = state_readout.shape[0]
        shape = (self.n_layers, B, self.n_kv_heads, self.n_prefix, self.head_dim)

        # mlp_k output: (B, L*H*K*D) → reshape → (B, L, H, K, D) → transpose to (L, B, H, K, D)
        flat_k = self.mlp_k(state_readout)
        flat_v = self.mlp_v(state_readout)

        k = flat_k.reshape(B, self.n_layers, self.n_kv_heads, self.n_prefix, self.head_dim)
        v = flat_v.reshape(B, self.n_layers, self.n_kv_heads, self.n_prefix, self.head_dim)

        # (B, L, H, K, D) → (L, B, H, K, D)
        return mx.transpose(k, (1, 0, 2, 3, 4)), mx.transpose(v, (1, 0, 2, 3, 4))
