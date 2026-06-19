# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司

"""Route/action tool contract validation.

This module keeps route/action supervision tied to explicitly requested tool
schemas without adding app-keyword routing rules. It validates tool names and
argument shapes only; semantic routing still belongs to the host model.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable


def route_action_tool_contracts_for_prompt(
    tool_names: Iterable[str] | None = None,
    *,
    tool_registry: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    registry_contracts = _contracts_by_name(tool_registry)
    if tool_names is None:
        return list(registry_contracts.values())
    names = _dedupe_strings(tool_names)
    contracts: list[dict[str, Any]] = []
    for name in names:
        contract = registry_contracts.get(name)
        if not isinstance(contract, dict):
            continue
        contracts.append(dict(contract))
    return contracts


def validate_route_action_tool_contracts(
    *,
    selected_tools: list[str],
    tool_call_plan: list[dict[str, Any]],
    case_id: str | None = None,
    required_tools: Iterable[str] | None = None,
    exact_tools: Iterable[str] | None = None,
    excluded_tools: Iterable[str] | None = None,
    tool_registry: Iterable[dict[str, Any]] | None = None,
    tool_contracts: Iterable[dict[str, Any]] | None = None,
) -> None:
    selected = _dedupe_strings(selected_tools)
    if len(selected) != len([tool for tool in selected_tools if str(tool or "").strip()]):
        raise ValueError("selected_tools must not contain duplicates")

    exact = None if exact_tools is None else _dedupe_strings(exact_tools)
    required = _dedupe_strings(required_tools or ())
    excluded = _dedupe_strings(excluded_tools or ())

    if exact is not None and selected != exact:
        raise ValueError(
            _case_prefix(case_id)
            + f"selected_tools must exactly match selected_tools_exact {exact}"
        )
    missing = [tool for tool in required if tool not in selected]
    if missing:
        raise ValueError(
            _case_prefix(case_id)
            + f"selected_tools missing selected_tools_include/required tools {missing}"
        )
    present_excluded = [tool for tool in excluded if tool in selected]
    if present_excluded:
        raise ValueError(
            _case_prefix(case_id)
            + f"selected_tools contains eval-excluded tools {present_excluded}"
        )

    selected_set = set(selected)
    contracts = _contracts_by_name(tool_registry, tool_contracts=tool_contracts)
    for index, entry in enumerate(tool_call_plan):
        if not isinstance(entry, dict):
            raise ValueError(_case_prefix(case_id) + f"tool_call_plan[{index}] must be an object")
        tool_name = _tool_name(entry)
        if not tool_name:
            raise ValueError(
                _case_prefix(case_id)
                + f"tool_call_plan[{index}] must include tool_name or tool"
            )
        if tool_name not in selected_set:
            raise ValueError(
                _case_prefix(case_id)
                + f"tool_call_plan[{index}] tool {tool_name!r} is not selected"
            )
        contract = contracts.get(tool_name)
        if contract is None:
            continue
        args = _tool_args(entry)
        _validate_args(tool_name=tool_name, args=args, contract=contract, case_id=case_id)


def _contracts_by_name(
    tool_registry: Iterable[dict[str, Any]] | None = None,
    *,
    tool_contracts: Iterable[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for raw in tool_registry or ():
        if not isinstance(raw, dict):
            continue
        name = _tool_name(raw)
        if not name:
            continue
        out[name] = _contract_from_registry_tool(name=name, tool=raw)
    for raw in tool_contracts or ():
        if not isinstance(raw, dict):
            continue
        name = _tool_name(raw)
        if not name:
            continue
        item = dict(raw)
        item["name"] = name
        out[name] = _normalize_contract(item)
    return out


def _contract_from_registry_tool(*, name: str, tool: dict[str, Any]) -> dict[str, Any]:
    schema = _schema_from_tool(tool)
    schema_contract = _contract_from_schema(schema)
    explicit = _normalize_contract(tool)
    contract: dict[str, Any] = {
        "name": name,
        "description": tool.get("description"),
        "usage_notes": tool.get("usage_notes") or tool.get("usageNotes"),
        "allowed_args": explicit.get("allowed_args") or schema_contract["allowed_args"],
        "enum_args": {
            **schema_contract["enum_args"],
            **explicit.get("enum_args", {}),
        },
        "integer_ranges": {
            **schema_contract["integer_ranges"],
            **explicit.get("integer_ranges", {}),
        },
        "string_args": explicit.get("string_args") or schema_contract["string_args"],
        "number_args": explicit.get("number_args") or schema_contract["number_args"],
        "boolean_args": explicit.get("boolean_args") or schema_contract["boolean_args"],
        "argument_constraints": explicit.get("argument_constraints", []),
    }
    examples = _string_list(tool.get("examples"))
    if examples:
        contract["examples"] = examples
    route_intents = _string_list(tool.get("route_intents") or tool.get("routeIntents"))
    if route_intents:
        contract["route_intents"] = route_intents
    parser_owned_args = _string_list(tool.get("parser_owned_args") or tool.get("parserOwnedArgs"))
    if parser_owned_args:
        contract["parser_owned_args"] = parser_owned_args
    return _public_contract(contract)


def _normalize_contract(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": _tool_name(raw),
        "description": raw.get("description"),
        "usage_notes": raw.get("usage_notes") or raw.get("usageNotes"),
        "allowed_args": _string_list(raw.get("allowed_args") or raw.get("allowedArgs")),
        "enum_args": _enum_args(raw.get("enum_args") or raw.get("enumArgs")),
        "integer_ranges": _integer_ranges(
            raw.get("integer_ranges") or raw.get("integerRanges")
        ),
        "string_args": _string_list(raw.get("string_args") or raw.get("stringArgs")),
        "number_args": _string_list(raw.get("number_args") or raw.get("numberArgs")),
        "boolean_args": _string_list(raw.get("boolean_args") or raw.get("booleanArgs")),
        "argument_constraints": _argument_constraints(
            raw.get("argument_constraints") or raw.get("argumentConstraints")
        ),
    }


def _public_contract(contract: dict[str, Any]) -> dict[str, Any]:
    out = {
        "name": contract["name"],
        "description": contract.get("description"),
        "usage_notes": contract.get("usage_notes"),
        "allowed_args": list(contract.get("allowed_args") or []),
        "enum_args": {
            key: list(values)
            for key, values in (contract.get("enum_args") or {}).items()
        },
        "integer_ranges": dict(contract.get("integer_ranges") or {}),
    }
    for key in (
        "string_args",
        "number_args",
        "boolean_args",
        "argument_constraints",
        "examples",
        "route_intents",
        "parser_owned_args",
    ):
        value = contract.get(key)
        if value:
            out[key] = value
    return out


def _schema_from_tool(tool: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("args_schema", "arguments_schema", "schema"):
        raw = tool.get(key)
        schema = _json_object(raw)
        if schema is not None:
            return schema
    parameters = tool.get("parameters")
    if isinstance(parameters, list):
        properties: dict[str, Any] = {}
        for item in parameters:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            properties[name] = item
        return {"type": "object", "properties": properties} if properties else None
    return None


def _contract_from_schema(schema: dict[str, Any] | None) -> dict[str, Any]:
    contract = {
        "allowed_args": [],
        "enum_args": {},
        "integer_ranges": {},
        "string_args": [],
        "number_args": [],
        "boolean_args": [],
    }
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if not isinstance(properties, dict):
        return contract
    for key, raw_property in properties.items():
        name = str(key).strip()
        if not name or not isinstance(raw_property, dict):
            continue
        contract["allowed_args"].append(name)
        value_type = raw_property.get("type")
        type_values = set(_string_list(value_type if isinstance(value_type, list) else [value_type]))
        if "string" in type_values:
            contract["string_args"].append(name)
        if "number" in type_values:
            contract["number_args"].append(name)
        if "integer" in type_values:
            minimum = raw_property.get("minimum")
            maximum = raw_property.get("maximum")
            if _is_number(minimum) and _is_number(maximum):
                contract["integer_ranges"][name] = [int(minimum), int(maximum)]
        if "boolean" in type_values:
            contract["boolean_args"].append(name)
        enum_values = _string_list(raw_property.get("enum"))
        if not enum_values:
            enum_values = _enum_values_from_description(raw_property.get("description"))
        if enum_values:
            contract["enum_args"][name] = enum_values
    return contract


def _json_object(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _enum_values_from_description(value: Any) -> list[str]:
    text = str(value or "").strip()
    if "|" not in text:
        return []
    if not re.fullmatch(r"[A-Za-z0-9_<>, |./:-]+", text):
        return []
    parts = [part.strip(" `.,;:") for part in text.split("|")]
    values = [
        part for part in parts
        if part and re.fullmatch(r"[A-Za-z0-9_<>,./:-]+", part)
    ]
    return values if len(values) >= 2 else []


def _enum_args(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): _string_list(values)
        for key, values in value.items()
        if _string_list(values)
    }


def _integer_ranges(value: Any) -> dict[str, list[int]]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, list[int]] = {}
    for key, raw_bounds in value.items():
        if (
            isinstance(raw_bounds, (list, tuple))
            and len(raw_bounds) == 2
            and _is_number(raw_bounds[0])
            and _is_number(raw_bounds[1])
        ):
            out[str(key)] = [int(raw_bounds[0]), int(raw_bounds[1])]
    return out


def _argument_constraints(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    constraints: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        when = raw.get("if") if isinstance(raw.get("if"), dict) else raw.get("when")
        if not isinstance(when, dict):
            continue
        omit = _string_list(raw.get("omit") or raw.get("then_omit") or raw.get("omit_args"))
        if omit:
            constraints.append({"if": dict(when), "omit": omit})
    return constraints


def _validate_args(
    *,
    tool_name: str,
    args: dict[str, Any],
    contract: dict[str, Any],
    case_id: str | None,
) -> None:
    allowed = set(contract.get("allowed_args") or ())
    unknown = sorted(str(key) for key in args.keys() if str(key) not in allowed)
    if unknown:
        raise ValueError(
            _case_prefix(case_id)
            + f"{tool_name} arguments contain unsupported keys {unknown}"
        )

    _validate_argument_constraints(
        tool_name=tool_name,
        args=args,
        constraints=contract.get("argument_constraints") or [],
        case_id=case_id,
    )

    enum_args = contract.get("enum_args") if isinstance(contract.get("enum_args"), dict) else {}
    for key, allowed_values in enum_args.items():
        if key not in args or args[key] is None:
            continue
        if not isinstance(args[key], str) or args[key] not in set(allowed_values):
            raise ValueError(
                _case_prefix(case_id)
                + f"{tool_name}.{key} must be one of {list(allowed_values)}"
            )

    for key in contract.get("string_args") or ():
        if key in args and args[key] is not None and not isinstance(args[key], str):
            raise ValueError(_case_prefix(case_id) + f"{tool_name}.{key} must be a string")

    for key in contract.get("number_args") or ():
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(_case_prefix(case_id) + f"{tool_name}.{key} must be a number")
        if value < 0:
            raise ValueError(_case_prefix(case_id) + f"{tool_name}.{key} must be non-negative")

    integer_ranges = (
        contract.get("integer_ranges")
        if isinstance(contract.get("integer_ranges"), dict)
        else {}
    )
    for key, bounds in integer_ranges.items():
        value = args.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(_case_prefix(case_id) + f"{tool_name}.{key} must be an integer")
        min_value, max_value = bounds
        if value < min_value or value > max_value:
            raise ValueError(
                _case_prefix(case_id)
                + f"{tool_name}.{key} must be between {min_value} and {max_value}"
            )

    for key in contract.get("boolean_args") or ():
        if key in args and args[key] is not None and not isinstance(args[key], bool):
            raise ValueError(_case_prefix(case_id) + f"{tool_name}.{key} must be a boolean")


def _validate_argument_constraints(
    *,
    tool_name: str,
    args: dict[str, Any],
    constraints: list[dict[str, Any]],
    case_id: str | None,
) -> None:
    for constraint in constraints:
        if not isinstance(constraint, dict):
            continue
        when = constraint.get("if")
        omit = _string_list(constraint.get("omit"))
        if not isinstance(when, dict) or not omit:
            continue
        if all(args.get(str(key)) == expected for key, expected in when.items()):
            present = [key for key in omit if key in args and args.get(key) is not None]
            if present:
                condition = ", ".join(
                    f"{key}={value!r}" for key, value in sorted(when.items())
                )
                subject = (
                    f"{tool_name}.{present[0]}"
                    if len(present) == 1
                    else f"{tool_name} arguments {present}"
                )
                raise ValueError(
                    _case_prefix(case_id)
                    + f"{subject} must be omitted when {condition}"
                )


def _tool_name(entry: dict[str, Any]) -> str | None:
    for key in ("tool_name", "toolName", "tool", "name"):
        value = entry.get(key)
        text = str(value or "").strip()
        if text:
            return text
    return None


def _tool_args(entry: dict[str, Any]) -> dict[str, Any]:
    for key in ("args", "arguments"):
        value = entry.get(key)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"tool_call_plan arguments field {key!r} must be an object")
        return value
    return {}


def _dedupe_strings(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values: Iterable[Any] = [value]
    elif isinstance(value, Iterable) and not isinstance(value, (dict, bytes, bytearray)):
        values = value
    else:
        values = [value]
    return _dedupe_strings(str(item) for item in values)


def _is_number(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _case_prefix(case_id: str | None) -> str:
    return f"case {case_id}: " if case_id else ""
