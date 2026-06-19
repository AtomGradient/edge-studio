# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Seed KV cache with prefix K/V tensors before prompt prefill.

Once the cache contains ``K`` pre-computed prefix tokens, the model's
attention will naturally treat them as "history" — the same way it treats
a system prompt that's already been processed. No attention-mask surgery
needed because mlx-lm's ``KVCache.make_mask`` reads from ``cache.offset``
and the prompt tokens land AFTER the prefix, so the causal mask admits
attention back into prefix territory automatically.

Use immediately after ``make_prompt_cache(model)``, before the first
``model(input_ids, cache=cache)`` call.
"""
from __future__ import annotations

from typing import Sequence

import mlx.core as mx


def seed_prefix_kv(
    caches: Sequence,
    prefix_keys: mx.array,
    prefix_values: mx.array,
) -> None:
    """In-place seed each layer's cache with ``(k, v)`` prefix tensors.

    Parameters
    ----------
    caches : list of ``KVCache``-like objects (one per transformer layer).
        Each must expose ``.state = (keys, values)`` setter and will be
        mutated. Typically obtained from
        ``mlx_lm.models.cache.make_prompt_cache(model)``.
    prefix_keys : ``mx.array``, shape ``(L, B, n_kv_heads, K, head_dim)``
        per-layer prefix keys. L must match ``len(caches)``.
    prefix_values : ``mx.array``, shape ``(L, B, n_kv_heads, K, v_head_dim)``
        per-layer prefix values.

    Notes
    -----
    * After calling, each cache has ``offset = K`` — subsequent
      ``update_and_fetch`` calls append prompt tokens right after the
      prefix, just like a bigger context window.
    * The function trusts ``caches`` to support the ``.state`` setter
      protocol used by ``KVCache`` and ``RotatingKVCache``. Quantised
      caches are **not** currently supported (raise rather than silently
      miscompute).
    """
    L = prefix_keys.shape[0]
    if prefix_values.shape[0] != L:
        raise ValueError(
            f"prefix K/V layer count mismatch: "
            f"keys L={L}, values L={prefix_values.shape[0]}"
        )
    if len(caches) != L:
        raise ValueError(
            f"caches length ({len(caches)}) must match prefix layer count ({L})"
        )

    for i, cache in enumerate(caches):
        # Must check the class-level descriptor (property) rather than
        # hasattr(cache, "state"): an empty KVCache's `state` getter derefs
        # self.keys.shape which raises AttributeError on None — we'd mistake
        # that for "no setter".
        descriptor = getattr(type(cache), "state", None)
        has_setter = isinstance(descriptor, property) and descriptor.fset is not None
        if not has_setter:
            raise TypeError(
                f"cache[{i}] ({type(cache).__name__}) has no `.state` setter; "
                f"quantised caches are not yet supported by seed_prefix_kv"
            )
        cache.state = (prefix_keys[i], prefix_values[i])


def seed_prefix_kv_selective(
    caches: Sequence,
    prefix_keys: mx.array,
    prefix_values: mx.array,
    target_indices: Sequence[int],
) -> None:
    """Seed only the caches at ``target_indices``.

    Use this for hybrid-attention models (e.g. Qwen3.5-4B has 8
    full-attention layers interleaved with 24 linear-attention layers —
    ``seed_prefix_kv`` alone would choke on the linear caches). Caller
    supplies ``target_indices`` in positional order matching the leading
    axis of ``prefix_keys`` / ``prefix_values``.

    Parameters
    ----------
    caches : full per-layer cache list (length = model layer count)
    prefix_keys : ``(N, B, H, K, D)`` — one slab per target index
    prefix_values : ``(N, B, H, K, D)``
    target_indices : length-N list of indices into ``caches`` to seed

    Notes
    -----
    * Non-target caches are left untouched (they'll initialise as the
      model's ``make_cache`` intended — e.g. zero linear-attention state).
    * The target caches must still satisfy ``seed_prefix_kv``'s own
      requirements (i.e. implement the ``state`` property setter).
    """
    n = prefix_keys.shape[0]
    if prefix_values.shape[0] != n:
        raise ValueError(
            f"prefix K/V target count mismatch: "
            f"keys N={n}, values N={prefix_values.shape[0]}"
        )
    if len(target_indices) != n:
        raise ValueError(
            f"target_indices length ({len(target_indices)}) must match "
            f"prefix layer count ({n})"
        )
    for idx in target_indices:
        if idx < 0 or idx >= len(caches):
            raise IndexError(
                f"target_indices contains out-of-range idx={idx} "
                f"(caches len={len(caches)})"
            )
    # Reuse the vetted single-entry seeding logic on a filtered sublist.
    sub = [caches[i] for i in target_indices]
    seed_prefix_kv(sub, prefix_keys, prefix_values)
