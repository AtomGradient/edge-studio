# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Native MLX-based optimization operations.

Replaces the dependency on external scripts. All operations are implemented
directly using Python + MLX + safetensors — no subprocesses needed.

Operations:
  - apply_neuron_pruning  : slice MLP gate/up/down_proj per layer
  - apply_layer_pruning   : remove whole transformer layers + renumber
  - apply_quantization    : mlx_lm.convert quantization
  - apply_vocab_pruning   : slice embed_tokens + lm_head to active vocab
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx

ProgressCallback = Callable[[str, float], None]


# ---------------------------------------------------------------------------
# Shared I/O helpers  (use mx.load — handles bfloat16 and all dtypes)
# ---------------------------------------------------------------------------

def _load_all_weights(model_dir: Path) -> dict[str, mx.array]:
    """Load all safetensors weights from a model directory as mx.array."""
    weights: dict[str, mx.array] = {}
    for sf_file in sorted(model_dir.glob("*.safetensors")):
        shard = mx.load(str(sf_file))
        weights.update(shard)
    return weights


def _save_weights(output_dir: Path, weights: dict[str, mx.array]) -> None:
    """Save weights dict as a single safetensors file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    mx.eval(weights)
    mx.save_safetensors(str(output_dir / "model.safetensors"), weights)


def _copy_non_weights(src_dir: Path, dst_dir: Path) -> None:
    """Copy config, tokenizer, and all non-safetensors files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in src_dir.iterdir():
        if f.is_file() and f.suffix != ".safetensors" and not f.name.startswith("."):
            shutil.copy2(f, dst_dir / f.name)


def _detect_layer_prefix(weights: dict[str, np.ndarray]) -> tuple[str, int]:
    """Return (prefix, num_layers) from weight key names.

    Example: ("language_model.model.layers.", 46)
    """
    pattern = re.compile(r"^(.*\.layers\.)(\d+)\.")
    prefix_sets: dict[str, set[int]] = {}
    for key in weights:
        m = pattern.match(key)
        if m:
            pfx, idx = m.group(1), int(m.group(2))
            prefix_sets.setdefault(pfx, set()).add(idx)
    if not prefix_sets:
        return "", 0
    prefix = max(prefix_sets, key=lambda p: len(prefix_sets[p]))
    return prefix, max(prefix_sets[prefix]) + 1


def _is_quantized_layer(weights: dict, prefix: str, layer: int) -> bool:
    return f"{prefix}{layer}.mlp.gate_proj.scales" in weights


def _dequantize(weights: dict, base_key: str, group_size: int = 64, bits: int = 4) -> mx.array:
    """Dequantize a weight tensor, returning an mx.array in float16."""
    w = weights[f"{base_key}.weight"]
    s = weights[f"{base_key}.scales"]
    b = weights[f"{base_key}.biases"]
    dq = mx.dequantize(w, s, b, group_size=group_size, bits=bits)
    return dq.astype(mx.float16)


def _update_config(dst: Path, updates: dict) -> None:
    """Merge updates into config.json."""
    path = dst / "config.json"
    if not path.exists():
        return
    with open(path) as f:
        cfg = json.load(f)
    cfg.update(updates)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def _get_text_config(config: dict) -> dict:
    """Return the sub-config that contains num_hidden_layers / intermediate_size."""
    for key in ("text_config", "talker_config"):
        if key in config and isinstance(config[key], dict):
            return config[key]
    return config


# ---------------------------------------------------------------------------
# Operation 1: Neuron pruning
# ---------------------------------------------------------------------------

import numpy as _np  # only used for neuron index computation


def _compute_neuron_plan(
    profile_layers: list,
    intermediate_size: int,
    threshold: float,
    group_size: int,
    min_size: int,
    max_reduction: float,
    protected_layers: set,
) -> list[dict]:
    """Compute per-layer pruning plan with sorted neuron indices.

    Mirrors the algorithm in prune_gemma_neurons.py / prune_neurons.py:
      - Select neurons with max_activation >= threshold
      - If rounding requires more, fill with highest-activation inactive neurons
      - Enforce min_size and max_reduction constraints
      - Return sorted indices (not just a count) for correct selection
    """
    if max_reduction is not None:
        reduction_min = int(intermediate_size * (1.0 - max_reduction))
        reduction_min = ((reduction_min + group_size - 1) // group_size) * group_size
        min_size = max(min_size, reduction_min)

    plan = []
    for layer_data in profile_layers:
        layer_idx = layer_data["layer"]
        max_acts = _np.array(layer_data["max_activations"], dtype=_np.float32)
        orig_size = len(max_acts)

        if layer_idx in protected_layers:
            plan.append({
                "layer": layer_idx,
                "original_size": orig_size,
                "new_size": orig_size,
                "indices": list(range(orig_size)),
                "reduction_pct": 0.0,
            })
            continue

        active_mask = max_acts >= threshold
        active_idx = _np.where(active_mask)[0]
        num_active = len(active_idx)

        new_size = ((num_active + group_size - 1) // group_size) * group_size
        new_size = max(new_size, min_size)
        new_size = min(new_size, orig_size)

        if new_size > num_active:
            inactive_idx = _np.where(~active_mask)[0]
            # fill with highest-activation inactive neurons
            fill = inactive_idx[_np.argsort(-max_acts[inactive_idx])][: new_size - num_active]
            all_idx = _np.sort(_np.concatenate([active_idx, fill]))
        else:
            all_idx = _np.sort(active_idx[:new_size])

        plan.append({
            "layer": layer_idx,
            "original_size": orig_size,
            "new_size": int(new_size),
            "indices": all_idx.tolist(),
            "reduction_pct": round((1.0 - new_size / orig_size) * 100, 1),
        })
    return plan


def apply_neuron_pruning(
    model_dir: str,
    output_dir: str,
    activation_profile,          # ActivationProfile (activation_loader) OR path str to JSON
    threshold: float = 0.1,
    max_reduction: float = 0.5,
    group_size: int = 64,
    min_size: int = 128,
    protected_layers: Optional[list[int]] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> tuple[bool, str, list[int]]:
    """Prune MLP neurons per layer based on activation profile.

    Uses sorted neuron indices (not naive truncation) — selects the most
    active neurons per layer.

    For quantized models: dequantize → select by index → requantize (preserves quant format).
    For bf16 models: direct index selection.

    Returns (success, message, per_layer_intermediate_sizes).
    """
    import json as _json

    src, dst = Path(model_dir), Path(output_dir)
    protected_set = set(protected_layers or [])

    # --- Load activation profile ---
    if isinstance(activation_profile, str):
        from .activation_loader import load_profile
        profile_obj = load_profile(activation_profile)
        profile_layers = [
            {"layer": la.layer_idx, "max_activations": la.max_activations.tolist()}
            for la in profile_obj.layers
        ]
        intermediate_size = profile_obj.intermediate_size
    else:
        # ActivationProfile object
        profile_layers = [
            {"layer": la.layer_idx, "max_activations": la.max_activations.tolist()}
            for la in activation_profile.layers
        ]
        intermediate_size = activation_profile.intermediate_size

    if progress_cb:
        progress_cb("Loading weights...", 0.05)

    weights = _load_all_weights(src)
    prefix, num_layers = _detect_layer_prefix(weights)
    if not prefix:
        return False, "Cannot detect layer structure in weights", []

    # Compute plan
    plan = _compute_neuron_plan(
        profile_layers, intermediate_size,
        threshold=threshold,
        group_size=group_size,
        min_size=min_size,
        max_reduction=max_reduction,
        protected_layers=protected_set,
    )
    plan_by_layer = {p["layer"]: p for p in plan}
    per_layer_sizes = [p["new_size"] for p in sorted(plan, key=lambda x: x["layer"])]

    quant = _is_quantized_layer(weights, prefix, 0)
    bits = 4  # standard for 4-bit QAT models

    new_weights: dict[str, mx.array] = {}

    # Copy all non-MLP keys
    mlp_re = re.compile(
        rf"^{re.escape(prefix)}(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$"
    )
    for key, val in weights.items():
        if not mlp_re.match(key):
            new_weights[key] = val

    # Process MLP per layer
    for li in range(num_layers):
        p = plan_by_layer.get(li)
        if p is None or p["new_size"] == p["original_size"]:
            # No pruning — pass through unchanged
            for proj in ("gate_proj", "up_proj", "down_proj"):
                base = f"{prefix}{li}.mlp.{proj}"
                for suf in ("weight", "scales", "biases"):
                    k = f"{base}.{suf}"
                    if k in weights:
                        new_weights[k] = weights[k]
                if not quant and f"{base}.weight" in weights:
                    new_weights[f"{base}.weight"] = weights[f"{base}.weight"]
            continue

        if progress_cb and li % 4 == 0:
            progress_cb(f"Pruning layer {li}/{num_layers}", 0.15 + 0.65 * li / num_layers)

        indices = mx.array(p["indices"])
        lp = f"{prefix}{li}.mlp"

        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"{lp}.{proj}"

            if quant:
                wk, sk, bk = f"{base}.weight", f"{base}.scales", f"{base}.biases"
                if wk not in weights:
                    for suf in ("weight", "scales", "biases"):
                        k = f"{base}.{suf}"
                        if k in weights:
                            new_weights[k] = weights[k]
                    continue

                w_full = mx.dequantize(
                    weights[wk], weights[sk], weights[bk],
                    group_size=group_size, bits=bits,
                )
                if proj in ("gate_proj", "up_proj"):
                    w_pruned = w_full[indices]      # select rows
                else:
                    w_pruned = w_full[:, indices]   # select cols

                new_w, new_s, new_b = mx.quantize(w_pruned, group_size=group_size, bits=bits)
                new_weights[wk] = new_w
                new_weights[sk] = new_s
                new_weights[bk] = new_b

            else:
                wk = f"{base}.weight"
                if wk not in weights:
                    continue
                w = weights[wk]
                if proj in ("gate_proj", "up_proj"):
                    new_weights[wk] = w[indices]
                else:
                    new_weights[wk] = w[:, indices]

    if progress_cb:
        progress_cb("Evaluating and saving...", 0.85)

    mx.eval(new_weights)
    _copy_non_weights(src, dst)
    _save_weights(dst, new_weights)

    # Update config.json
    cfg_path = dst / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = _json.load(f)
        tcfg = _get_text_config(cfg)
        tcfg["intermediate_size"] = min(per_layer_sizes)
        # Store under the correct sub-config key for EdgeRuntime
        tcfg["per_layer_intermediate_sizes"] = per_layer_sizes
        with open(cfg_path, "w") as f:
            _json.dump(cfg, f, indent=2, ensure_ascii=False)

    # Save pruning_metadata.json (matches original script format)
    total_orig = sum(p["original_size"] for p in plan)
    total_new = sum(p["new_size"] for p in plan)
    metadata = {
        "group_size": group_size,
        "bits": bits,
        "overall_reduction_pct": round((1.0 - total_new / total_orig) * 100, 1),
        "per_layer_sizes": per_layer_sizes,
        "per_layer": [{k: v for k, v in p.items() if k != "indices"} for p in plan],
    }
    with open(dst / "pruning_metadata.json", "w") as f:
        _json.dump(metadata, f, indent=2)

    if progress_cb:
        progress_cb("Done", 1.0)

    return True, f"Neuron pruning complete → {output_dir}", per_layer_sizes


# ---------------------------------------------------------------------------
# Operation 2: Layer pruning
# ---------------------------------------------------------------------------

def apply_layer_pruning(
    model_dir: str,
    output_dir: str,
    layers_to_remove: list[int],
    progress_cb: Optional[ProgressCallback] = None,
) -> tuple[bool, str]:
    """Remove entire transformer layers and renumber remaining ones."""
    src, dst = Path(model_dir), Path(output_dir)

    if not layers_to_remove:
        return False, "No layers specified for removal"

    remove_set = set(layers_to_remove)

    if progress_cb:
        progress_cb("Loading weights...", 0.05)

    weights = _load_all_weights(src)
    prefix, num_layers = _detect_layer_prefix(weights)
    if not prefix:
        return False, "Cannot detect layer structure in weights"

    invalid = [l for l in layers_to_remove if l < 0 or l >= num_layers]
    if invalid:
        return False, f"Invalid layer indices {invalid} for model with {num_layers} layers"

    # Build old→new index mapping
    new_idx_map: dict[int, int] = {}
    ni = 0
    for old in range(num_layers):
        if old not in remove_set:
            new_idx_map[old] = ni
            ni += 1

    if progress_cb:
        progress_cb("Removing and renumbering layers...", 0.3)

    layer_re = re.compile(rf"^{re.escape(prefix)}(\d+)(.*)$")
    new_weights: dict[str, mx.array] = {}

    for key, val in weights.items():
        m = layer_re.match(key)
        if m:
            old_layer = int(m.group(1))
            if old_layer in remove_set:
                continue
            new_weights[f"{prefix}{new_idx_map[old_layer]}{m.group(2)}"] = val
        else:
            new_weights[key] = val

    if progress_cb:
        progress_cb("Saving...", 0.85)

    _copy_non_weights(src, dst)
    _save_weights(dst, new_weights)

    # Update config.json
    cfg_path = dst / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        tcfg = _get_text_config(cfg)
        tcfg["num_hidden_layers"] = num_layers - len(layers_to_remove)
        cfg["text_layer_pruning"] = {"removed_layers": sorted(layers_to_remove)}
        with open(cfg_path, "w") as f:
            json.dump(cfg, f, indent=2)

    if progress_cb:
        progress_cb("Done", 1.0)

    return True, f"Layer pruning complete ({len(layers_to_remove)} layers removed) → {output_dir}"


# ---------------------------------------------------------------------------
# Operation 3: Quantization
# ---------------------------------------------------------------------------

def _load_config(model_dir: Path) -> dict:
    """Load config.json from model directory."""
    cfg_path = model_dir / "config.json"
    if not cfg_path.exists():
        return {}
    with open(cfg_path) as f:
        return json.load(f)


def _is_vlm(model_dir: Path) -> bool:
    """Return True if config.json has a vision_config — requires mlx_vlm.convert."""
    cfg = _load_config(model_dir)
    return "vision_config" in cfg


def _is_audio_model(model_dir: Path) -> bool:
    """Return True if this is an audio model (Whisper, TTS) — requires mlx_audio.convert."""
    cfg = _load_config(model_dir)
    model_type = cfg.get("model_type", "")
    # Whisper and other audio model types
    audio_types = {"whisper", "whisper_tiny", "whisper_base", "whisper_small",
                   "whisper_medium", "whisper_large"}
    if model_type in audio_types or "whisper" in model_type.lower():
        return True
    # Whisper-style config fields (MLX-native format)
    if "n_audio_state" in cfg or "n_audio_layer" in cfg:
        return True
    # HuggingFace Whisper config: has encoder_layers + num_mel_bins
    if "encoder_layers" in cfg and "num_mel_bins" in cfg:
        return True
    # Kokoro/Dia/OuteTTS: no standard config.json (use model_type)
    audio_model_types = {"kokoro", "dia", "outetts", "csm", "qwen3_tts"}
    if model_type in audio_model_types:
        return True
    return False


def apply_quantization(
    model_dir: str,
    output_dir: str,
    bits: int = 4,
    group_size: int = 64,
    progress_cb: Optional[ProgressCallback] = None,
) -> tuple[bool, str]:
    """Quantize a model — routes to correct converter based on model type:
    - Audio (Whisper/TTS): mlx_audio.convert
    - VLM (vision+text): mlx_vlm.convert
    - LLM (text-only): mlx_lm.convert
    """
    src = Path(model_dir)
    dst = Path(output_dir)
    audio = _is_audio_model(src)
    vlm = (not audio) and _is_vlm(src)

    # mlx_lm/mlx_vlm/mlx_audio.convert refuse to overwrite existing dirs.
    # The caller (_make_output_dir) already appends a timestamp when possible,
    # but in edge cases the dir may still exist — remove it to avoid the error.
    if dst.exists():
        import shutil
        shutil.rmtree(dst)

    if progress_cb:
        if audio:
            kind = "Audio (mlx_audio)"
        elif vlm:
            kind = "VLM (mlx_vlm)"
        else:
            kind = "LLM (mlx_lm)"
        progress_cb(f"Starting quantization ({kind})...", 0.05)

    try:
        if audio:
            from mlx_audio import convert
            convert(
                hf_path=model_dir,
                mlx_path=output_dir,
                quantize=True,
                q_bits=bits,
                q_group_size=group_size,
            )
        elif vlm:
            from mlx_vlm import convert
            convert(
                hf_path=model_dir,
                mlx_path=output_dir,
                quantize=True,
                q_bits=bits,
                q_group_size=group_size,
            )
        else:
            from mlx_lm.convert import convert
            convert(
                hf_path=model_dir,
                mlx_path=output_dir,
                quantize=True,
                q_bits=bits,
                q_group_size=group_size,
            )
        if progress_cb:
            progress_cb("Done", 1.0)
        return True, f"Quantization ({bits}-bit, group={group_size}) complete → {output_dir}"
    except Exception as e:
        return False, f"Quantization failed: {e}"


# ---------------------------------------------------------------------------
# Operation 4: Vocab pruning
# ---------------------------------------------------------------------------

def apply_vocab_pruning(
    model_dir: str,
    output_dir: str,
    progress_cb: Optional[ProgressCallback] = None,
) -> tuple[bool, str]:
    """Prune embedding + lm_head to the tokenizer's actual vocabulary size.

    Saves disk space when the model's config vocab_size > actual token count.
    """
    src, dst = Path(model_dir), Path(output_dir)

    cfg_path = src / "config.json"
    if not cfg_path.exists():
        return False, "config.json not found"
    with open(cfg_path) as f:
        config = json.load(f)

    if progress_cb:
        progress_cb("Loading weights...", 0.05)

    weights = _load_all_weights(src)

    # Detect embed_tokens key
    embed_key: Optional[str] = None
    for candidate in (
        "language_model.model.embed_tokens.weight",
        "model.language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
    ):
        if candidate in weights:
            embed_key = candidate
            break
    if not embed_key:
        for k in weights:
            if "embed_tokens" in k and k.endswith(".weight"):
                embed_key = k
                break
    if not embed_key:
        return False, "Could not find embed_tokens weight"

    # Detect lm_head key
    lm_head_key: Optional[str] = None
    for candidate in (
        "language_model.lm_head.weight",
        "model.language_model.lm_head.weight",
        "lm_head.weight",
    ):
        if candidate in weights:
            lm_head_key = candidate
            break
    if not lm_head_key:
        for k in weights:
            if "lm_head" in k and k.endswith(".weight"):
                lm_head_key = k
                break

    # Get active vocab size from tokenizer
    tokenizer_json = src / "tokenizer.json"
    actual_vocab: Optional[int] = None
    if tokenizer_json.exists():
        with open(tokenizer_json) as f:
            tj = json.load(f)
        vocab = tj.get("model", {}).get("vocab", {})
        if vocab:
            actual_vocab = len(vocab)

    embed_base = embed_key[: -len(".weight")]
    quant_embed = f"{embed_base}.scales" in weights

    if quant_embed:
        embed_w = _dequantize(weights, embed_base)
    else:
        embed_w = weights[embed_key].astype(mx.float16)

    current_vocab = embed_w.shape[0]

    if actual_vocab is None or actual_vocab >= current_vocab:
        return False, (
            f"Vocab pruning skipped: tokenizer vocab {actual_vocab} >= model vocab {current_vocab}"
        )

    # Align up to 64 (must cover all active tokens)
    new_vocab = ((actual_vocab + 63) // 64) * 64
    if new_vocab >= current_vocab:
        return False, f"No pruning possible: aligned vocab {new_vocab} >= current {current_vocab}"

    if progress_cb:
        progress_cb(f"Pruning vocab {current_vocab} -> {new_vocab}...", 0.3)

    new_weights: dict[str, mx.array] = {}

    for key, val in weights.items():
        # Skip old quantized embed/lm_head artifacts — we'll re-add as fp16
        if quant_embed and key.startswith(embed_base + "."):
            continue
        if lm_head_key:
            lm_base = lm_head_key[: -len(".weight")]
            if f"{lm_base}.scales" in weights and key.startswith(lm_base + "."):
                continue
        new_weights[key] = val

    # Write pruned embeddings
    new_weights[embed_key] = embed_w[:new_vocab]

    # Write pruned lm_head (may be tied or separate)
    tcfg = _get_text_config(config)
    tied = tcfg.get("tie_word_embeddings", config.get("tie_word_embeddings", True))

    if lm_head_key and not tied:
        lm_base = lm_head_key[: -len(".weight")]
        if f"{lm_base}.scales" in weights:
            lm_w = _dequantize(weights, lm_base)
        else:
            lm_w = weights[lm_head_key].astype(mx.float16)
        new_weights[lm_head_key] = lm_w[:new_vocab]

    if progress_cb:
        progress_cb("Saving...", 0.85)

    _copy_non_weights(src, dst)
    _save_weights(dst, new_weights)

    # Update config.json
    dst_cfg = dst / "config.json"
    if dst_cfg.exists():
        with open(dst_cfg) as f:
            cfg = json.load(f)
        tcfg2 = _get_text_config(cfg)
        old_vocab = tcfg2.get("vocab_size", current_vocab)
        tcfg2["vocab_size"] = new_vocab
        cfg["vocab_pruning"] = {
            "original_vocab_size": old_vocab,
            "pruned_vocab_size": new_vocab,
        }
        with open(dst_cfg, "w") as f:
            json.dump(cfg, f, indent=2)

    savings_mb = (current_vocab - new_vocab) * embed_w.shape[1] * 2 // (1024 * 1024)
    if progress_cb:
        progress_cb("Done", 1.0)

    return True, (
        f"Vocab pruning complete: {current_vocab}→{new_vocab} "
        f"(~{savings_mb}MB saved) → {output_dir}"
    )
