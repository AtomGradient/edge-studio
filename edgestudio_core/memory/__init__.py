# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Episodic Fast Weight (SSM) memory layer — S2a interface module.

This subpackage defines the **contract** every SSM (RWKV v7 / DeltaNet /
xLSTM) must satisfy in order to drop into the multi-freq memory stack +
KV-prefix injection pipeline documented in
``docs/winforver/S2a_memory_block_abstraction.md``.

Core insight (`ultra/03 §1.2` Episodic Fast Weight):
- Neural Imprint answers "who you are"; Fact Layer answers "what happened";
  this memory layer answers "what the recent context implies" — encoded
  as a learned state that projects to KV prefix tokens prepended to each
  layer's attention cache before the prompt runs.

S2a intentionally ships only the abstract interface + a minimal identity
reference. Concrete implementations live in S2b (RWKV v7), S2c (DeltaNet),
S2d (xLSTM). Every one of them must pass the contract test in
``tests/winforver/test_memory_block_contract.py``; that's how we guarantee
the S2{b,c,d} ablation is single-variable.
"""

from edgestudio_core.memory.base import MemoryBlock, IdentityMemoryBlock
from edgestudio_core.memory.deltanet import DeltaNetBlock
from edgestudio_core.memory.injection import seed_prefix_kv, seed_prefix_kv_selective
from edgestudio_core.memory.mixer import MemoryMixer
from edgestudio_core.memory.multi_freq import MultiFreqMemoryStack
from edgestudio_core.memory.projection import MemoryToPrefixHead
from edgestudio_core.memory.rwkv_v7 import RWKVv7Block
from edgestudio_core.memory.vsa import VSAMemoryBlock

__all__ = [
    "MemoryBlock",
    "IdentityMemoryBlock",
    "RWKVv7Block",
    "DeltaNetBlock",
    "VSAMemoryBlock",
    "MemoryMixer",
    "MultiFreqMemoryStack",
    "MemoryToPrefixHead",
    "seed_prefix_kv",
    "seed_prefix_kv_selective",
]
