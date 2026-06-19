# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Compile correction context into compact Neural Imprint overlay text.

The correction ledger is the append-only audit log. This compiler is the
bounded read model that decides what is stable enough to enter Neural Imprint.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from typing import Any, Mapping


CORRECTION_COMPILER_SCHEMA_VERSION = "edgestudio.correction_compiler.v1"


def compile_correction_context(
    context: Mapping[str, Any],
    *,
    min_fact_support: int = 2,
    max_fact_rules: int = 16,
    max_profile_overlays: int = 16,
) -> dict[str, Any]:
    """Reduce raw correction context to stable, compact overlay rules."""

    fact_corrections = _list_of_objects(context.get("fact_corrections"))
    profile_overlays = _list_of_objects(context.get("profile_overlays"))
    fact_rules, skipped_fact = _compile_fact_rules(
        fact_corrections,
        min_support=max(1, int(min_fact_support)),
        max_rules=max(1, int(max_fact_rules)),
    )
    profile_rules = _compile_profile_rules(
        profile_overlays,
        max_rules=max(0, int(max_profile_overlays)),
    )
    overlay_text = _render_overlay_text(fact_rules=fact_rules, profile_rules=profile_rules)
    included_fingerprints = sorted(
        {
            fingerprint
            for rule in [*fact_rules, *profile_rules]
            for fingerprint in _list_of_text(rule.get("correction_fingerprints"))
        }
    )
    counts = {
        "input_fact_corrections": len(fact_corrections),
        "input_profile_overlays": len(profile_overlays),
        "included_fact_rules": len(fact_rules),
        "included_profile_overlays": len(profile_rules),
        "included_correction_fingerprints": len(included_fingerprints),
        "skipped_fact_corrections": len(skipped_fact["unstable_fingerprints"]),
        "conflict_groups": len(skipped_fact["conflict_groups"]),
    }
    return {
        "ok": True,
        "schema_version": CORRECTION_COMPILER_SCHEMA_VERSION,
        "status": "compiled" if overlay_text else "empty",
        "policy": {
            "min_fact_support": max(1, int(min_fact_support)),
            "max_fact_rules": max(1, int(max_fact_rules)),
            "max_profile_overlays": max(0, int(max_profile_overlays)),
        },
        "counts": counts,
        "fact_rules": fact_rules,
        "profile_rules": profile_rules,
        "skipped": skipped_fact,
        "included_correction_fingerprints": included_fingerprints,
        "overlay_text": overlay_text,
        "overlay_sha256": _sha256_text(overlay_text) if overlay_text else None,
        "audit": {
            "input_is_append_only_ledger_context": True,
            "neural_imprint_receives_compiled_overlay_only": True,
            "raw_fact_ids_in_overlay": False,
            "drops_unstable_single_fact_corrections": True,
        },
    }


def _compile_fact_rules(
    corrections: list[dict[str, Any]],
    *,
    min_support: int,
    max_rules: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    choices: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    fingerprint_to_fields: dict[str, set[str]] = defaultdict(set)
    for item in corrections:
        fingerprint = _text(item.get("correction_fingerprint"))
        fields = _object(item.get("normalized_fields"))
        if not fingerprint or not fields:
            continue
        for raw_field, raw_value in fields.items():
            field = _text(raw_field)
            if not field:
                continue
            value = _canonical_value(raw_value)
            choices[field][value].add(fingerprint)
            fingerprint_to_fields[fingerprint].add(field)

    rules: list[dict[str, Any]] = []
    unstable_fingerprints: set[str] = set()
    conflict_groups: list[dict[str, Any]] = []
    for field in sorted(choices):
        value_groups = choices[field]
        ranked = sorted(
            value_groups.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        if not ranked:
            continue
        top_value, top_fingerprints = ranked[0]
        runner_up_count = len(ranked[1][1]) if len(ranked) > 1 else 0
        if len(top_fingerprints) < min_support:
            unstable_fingerprints.update(
                fingerprint for fingerprints in value_groups.values() for fingerprint in fingerprints
            )
            continue
        if runner_up_count >= len(top_fingerprints):
            conflict_groups.append(
                {
                    "field": field,
                    "choices": [
                        {
                            "value": value,
                            "support_count": len(fingerprints),
                        }
                        for value, fingerprints in ranked
                    ],
                }
            )
            unstable_fingerprints.update(
                fingerprint for fingerprints in value_groups.values() for fingerprint in fingerprints
            )
            continue
        rules.append(
            {
                "field": field,
                "preferred_value": _json_value(top_value),
                "support_count": len(top_fingerprints),
                "correction_fingerprints": sorted(top_fingerprints),
                "evidence_fingerprint_sha256": _sha256_json(sorted(top_fingerprints)),
            }
        )

    rules.sort(key=lambda item: (-int(item["support_count"]), str(item["field"])))
    return rules[:max_rules], {
        "unstable_fingerprints": sorted(unstable_fingerprints),
        "conflict_groups": conflict_groups,
        "fingerprint_to_fields": {
            fingerprint: sorted(fields)
            for fingerprint, fields in sorted(fingerprint_to_fields.items())
        },
    }


def _compile_profile_rules(
    overlays: list[dict[str, Any]],
    *,
    max_rules: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for item in overlays:
        fingerprint = _text(item.get("correction_fingerprint"))
        overlay = _object(item.get("overlay"))
        target = _object(item.get("target"))
        if not fingerprint or not overlay:
            continue
        key = _sha256_json({"target": target, "overlay": overlay})
        if key not in grouped:
            grouped[key] = {
                "target": target,
                "overlay": overlay,
                "support_count": 0,
                "correction_fingerprints": [],
            }
        grouped[key]["support_count"] += 1
        grouped[key]["correction_fingerprints"].append(fingerprint)

    rules = list(grouped.values())
    for rule in rules:
        rule["correction_fingerprints"] = sorted(set(rule["correction_fingerprints"]))
        rule["evidence_fingerprint_sha256"] = _sha256_json(rule["correction_fingerprints"])
    rules.sort(
        key=lambda item: (
            -int(item["support_count"]),
            _canonical_value(item["target"]),
            _canonical_value(item["overlay"]),
        )
    )
    return rules[:max_rules]


def _render_overlay_text(
    *,
    fact_rules: list[dict[str, Any]],
    profile_rules: list[dict[str, Any]],
) -> str:
    lines: list[str] = []
    if fact_rules:
        lines.append("[correction_distilled_fact_rules]")
        for rule in fact_rules:
            lines.append(
                "- "
                f"field={rule['field']} "
                f"preferred_value={_format_value(rule['preferred_value'])} "
                f"support={rule['support_count']}"
            )
    if profile_rules:
        if lines:
            lines.append("")
        lines.append("[correction_profile_overlays]")
        for rule in profile_rules:
            lines.append(
                "- "
                f"target=({_format_mapping(_object(rule.get('target')))}) "
                f"update=({_format_mapping(_object(rule.get('overlay')))}) "
                f"support={rule['support_count']}"
            )
    return "\n".join(lines).strip()


def _format_mapping(value: Mapping[str, Any]) -> str:
    items = []
    for key in sorted(str(item_key) for item_key in value.keys()):
        items.append(f"{key}={_format_value(value.get(key))}")
    return "; ".join(items) if items else "none"


def _format_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return str(value)


def _canonical_value(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _json_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _list_of_objects(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _list_of_text(value: Any) -> list[str]:
    return [_text(item) for item in value if _text(item)] if isinstance(value, list) else []


def _object(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if value is None:
        return ""
    return str(value).strip()
