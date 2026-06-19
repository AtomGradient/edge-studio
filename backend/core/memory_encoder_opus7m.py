# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


# ============================================================
# Configuration
# ============================================================

@dataclass
class MemoryEncoderConfig:
    hidden_dim: int = 2560          # Aligned with Qwen3.5-4B
    num_categories: int = 64        # Room for expansion (empirically ~40 categories)
    num_memory_tokens: int = 16     # K
    compressor_layers: int = 2
    compressor_heads: int = 8
    text_embedding_dim: int = 2560  # Qwen embed dim
    time_feature_dim: int = 5       # [weekday/7, hour/24, month/12, is_weekend, is_worktime]


# ============================================================
# Record Encoder
# ============================================================


class RecordEncoder(nn.Module):

    def __init__(self, cfg: MemoryEncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.hidden_dim
        part = d // 4  # 640 when d=2560

        # amount: scalar log1p → MLP → [part]
        self.amount_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.GELU(),
            nn.Linear(64, part),
        )

        # category: Embedding → [part]
        self.category_embed = nn.Embedding(cfg.num_categories, part)

        # time: [5] → MLP → [part]
        self.time_mlp = nn.Sequential(
            nn.Linear(cfg.time_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, part),
        )

        # text embedding: [d_text] → Linear → [part]
        self.text_proj = nn.Linear(cfg.text_embedding_dim, part)

        # fusion: concat 4 × part = d → MLP → d
        self.fusion = nn.Sequential(
            nn.Linear(4 * part, d),
            nn.LayerNorm(d),
            nn.GELU(),
            nn.Linear(d, d),
        )
        # residual-style stabilization
        self.out_norm = nn.LayerNorm(d)

    def __call__(
        self,
        amount: mx.array,           # [N, 1] (already log1p-transformed)
        category_id: mx.array,      # [N] int
        time_features: mx.array,    # [N, 5]
        text_embedding: mx.array,   # [N, d_text]
    ) -> mx.array:
        a = self.amount_mlp(amount)              # [N, part]
        c = self.category_embed(category_id)     # [N, part]
        t = self.time_mlp(time_features)         # [N, part]
        x = self.text_proj(text_embedding)       # [N, part]
        h = mx.concatenate([a, c, t, x], axis=-1)  # [N, d]
        h = self.fusion(h)
        return self.out_norm(h)


# ============================================================
# Cross-Attention Block（q attend to kv）
# ============================================================


class CrossAttnBlock(nn.Module):
    def __init__(self, d: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_norm = nn.LayerNorm(d)
        self.kv_norm = nn.LayerNorm(d)

        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        self.ffn_norm = nn.LayerNorm(d)
        # FFN hidden = 2x (halved from 4x) — sufficient for PoC, significant param reduction
        self.ffn = nn.Sequential(
            nn.Linear(d, 2 * d),
            nn.GELU(),
            nn.Linear(2 * d, d),
        )

    def __call__(self, q: mx.array, kv: mx.array) -> mx.array:
        """
        q:  [K, d]
        kv: [N, d]
        """
        K = q.shape[0]
        N = kv.shape[0]

        qn = self.q_norm(q)
        kn = self.kv_norm(kv)

        Q = self.q_proj(qn).reshape(K, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, K, dh]
        Kmat = self.k_proj(kn).reshape(N, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, N, dh]
        V = self.v_proj(kn).reshape(N, self.num_heads, self.head_dim).transpose(1, 0, 2)  # [H, N, dh]

        # Standard softmax attention (small scale, compute directly)
        scores = (Q @ Kmat.transpose(0, 2, 1)) * self.scale  # [H, K, N]
        attn = mx.softmax(scores, axis=-1)
        out = attn @ V                                         # [H, K, dh]
        out = out.transpose(1, 0, 2).reshape(K, -1)            # [K, d]
        out = self.o_proj(out)

        # residual (aligned with pre-LN architecture)
        q = q + out

        # FFN
        fo = self.ffn(self.ffn_norm(q))
        q = q + fo
        return q


# ============================================================
# Memory Compressor
# ============================================================


class MemoryCompressor(nn.Module):

    def __init__(self, cfg: MemoryEncoderConfig):
        super().__init__()
        self.cfg = cfg
        # learnable queries (MLX treats mx.array attributes as trainable params)
        self.queries = mx.random.normal(
            shape=(cfg.num_memory_tokens, cfg.hidden_dim)
        ) * 0.02

        self.blocks = [
            CrossAttnBlock(cfg.hidden_dim, cfg.compressor_heads)
            for _ in range(cfg.compressor_layers)
        ]
        self.final_norm = nn.LayerNorm(cfg.hidden_dim)

    def __call__(self, record_embeds: mx.array) -> mx.array:
        """
        record_embeds: [N, d]
        return: [K, d]
        """
        q = self.queries
        for block in self.blocks:
            q = block(q, record_embeds)
        return self.final_norm(q)


# ============================================================
# Full Memory Encoder = RecordEncoder + Compressor
# ============================================================


class MemoryEncoder(nn.Module):
    def __init__(self, cfg: MemoryEncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.record_encoder = RecordEncoder(cfg)
        self.compressor = MemoryCompressor(cfg)

    def __call__(
        self,
        amount: mx.array,
        category_id: mx.array,
        time_features: mx.array,
        text_embedding: mx.array,
    ) -> mx.array:
        record_embeds = self.record_encoder(
            amount, category_id, time_features, text_embedding
        )
        return self.compressor(record_embeds)


def count_params(module) -> int:
    total = 0
    def _walk(obj):
        nonlocal total
        if isinstance(obj, mx.array):
            total += obj.size
        elif isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, list):
            for v in obj:
                _walk(v)
    _walk(module.parameters())
    return total


if __name__ == "__main__":
    # quick sanity test
    cfg = MemoryEncoderConfig()
    enc = MemoryEncoder(cfg)

    # Simulate 10 records
    N = 10
    amount = mx.random.normal(shape=(N, 1))
    category_id = mx.random.randint(0, cfg.num_categories, shape=(N,))
    time_features = mx.random.normal(shape=(N, cfg.time_feature_dim))
    text_emb = mx.random.normal(shape=(N, cfg.text_embedding_dim))

    out = enc(amount, category_id, time_features, text_emb)
    print(f"Memory Encoder output shape: {out.shape}")
    print(f"Trainable params: {count_params(enc):,} ({count_params(enc) / 1e6:.2f}M)")
