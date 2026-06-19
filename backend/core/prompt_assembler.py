# Copyright © 2026 AtomGradient
# 版权所有 © 2026 质子梯度（北京）科技有限公司


from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional, Sequence, Union

# Avoid circular import: use Any for type annotations
# from .fact_layer.fact_store import FactRecord

# Fact injection section delimiter (hardcoded, all downstream parsing depends on this)
FACT_START = "<|fact_layer_start|>"
FACT_END = "<|fact_layer_end|>"

# Default token budget (approx: 4 chars ~ 1 token, 512 tokens ~ 2048 chars)
DEFAULT_FACT_TOKEN_BUDGET = 512

# Token approximation factor (empirical value for mixed CJK/Latin text)
CHARS_PER_TOKEN = 4

# Default system prompt (when caller does not provide one)
#
# Phase D hardening (2026-04-22):
#   When fact_layer section contains an exact-query prefix, the system prompt must instruct the model
#   to "cite exact numbers" rather than "fabricate amount ranges".
#   Persona answers tend to hallucinate freely; exact queries need strong constraints to suppress fabrication.
DEFAULT_SYSTEM_PROMPT_PREFIX = (
    "你是用户的个人 AI 助手。基于你对用户的了解来回答，用第一人称自然口吻。\n\n"
    "**fact_layer 处理规则（绝对优先级）：**\n"
    "1. 如果 fact_layer 段出现【精确查询】前缀：这是从用户记账数据库查出的"
    "权威答案。你必须**直接引用其中的具体数字**（金额、次数、日期），"
    "不要改写、补全或编造 fact_layer 中没有出现的数字。\n"
    "2. 如果 fact_layer 段是事实列表：用其中的商家/金额/时间支撑你的回答。\n"
    "3. 如果没有 fact_layer：基于画像直觉回答，只能给非精确、带不确定性的概括，不要输出具体账单数字。\n\n"
    "不要编造没有依据的精确数字；不要忽略 fact_layer 里的权威查询结果。"
)


@dataclass
class AssembledPrompt:
    system: str
    user: str
    # Debug field: how many facts were actually injected (< input count when over budget)
    injected_fact_count: int = 0
    truncated_fact_count: int = 0

    def as_messages(self) -> List[dict]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def full_text(self) -> str:
        return f"[system]\n{self.system}\n\n[user]\n{self.user}"


# ── Core assembly function ────────────────────────────────

def assemble_with_facts(
    *,
    user_query: str,
    facts: Sequence[Any],                       # List[FactRecord]
    router_result: Optional[Any] = None,        # RouterResult
    system_prompt_prefix: str = DEFAULT_SYSTEM_PROMPT_PREFIX,
    token_budget: int = DEFAULT_FACT_TOKEN_BUDGET,
    answer_hint: Optional[str] = None,          # Structured query exact answer
) -> AssembledPrompt:
    # Rule 5: do not inject facts when router classifies as persona
    should_inject = True
    if router_result is not None:
        label = getattr(router_result, "label", None)
        if label == "persona":
            should_inject = False

    # Structured query exact answer takes priority. Inject hint even when facts are empty.
    if answer_hint and should_inject:
        hint_section = f"\n\n{FACT_START}\n{answer_hint}\n{FACT_END}"
        return AssembledPrompt(
            system=system_prompt_prefix + hint_section,
            user=user_query,
            injected_fact_count=1,
            truncated_fact_count=0,
        )

    if not should_inject or not facts:
        return AssembledPrompt(
            system=system_prompt_prefix,
            user=user_query,
            injected_fact_count=0,
            truncated_fact_count=0,
        )

    # Rule 2: deterministic ordering (created_at DESC, id ASC)
    sorted_facts = sorted(
        facts,
        key=lambda f: (-_get_time(f), _get_id(f)),
    )

    # Rule 3 + 4: format + budget truncation
    lines: List[str] = []
    total_chars = 0
    budget_chars = token_budget * CHARS_PER_TOKEN
    injected = 0

    for fact in sorted_facts:
        line = _format_fact_line(fact)
        # Estimate total length after adding (including newline)
        if total_chars + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        total_chars += len(line) + 1
        injected += 1

    truncated = len(sorted_facts) - injected

    # Assemble fact section
    fact_section = (
        f"\n\n{FACT_START}\n"
        + "\n".join(lines)
        + f"\n{FACT_END}"
    )
    if truncated > 0:
        fact_section += f"\n（还有 {truncated} 条因 token 预算未展示，按时间倒序优先）"

    return AssembledPrompt(
        system=system_prompt_prefix + fact_section,
        user=user_query,
        injected_fact_count=injected,
        truncated_fact_count=truncated,
    )


# ── Internal utilities ────────────────────────────────────

def _get_time(fact: Any) -> int:
    payload = getattr(fact, "payload", None) or {}
    t = payload.get("time")
    if isinstance(t, int):
        return t
    return 0


def _get_id(fact: Any) -> str:
    return getattr(fact, "id", "") or ""


def _format_fact_line(fact: Any) -> str:
    payload = getattr(fact, "payload", {}) or {}

    # Time: unix ms → 'YYYY-MM-DD'
    t_ms = payload.get("time")
    if isinstance(t_ms, int):
        date_str = datetime.fromtimestamp(t_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")
    else:
        date_str = "?"

    merchant = payload.get("merchant") or "?"
    category = payload.get("category") or "?"
    amount = payload.get("amount")
    amount_str = f"¥{amount:.2f}" if isinstance(amount, (int, float)) else "?"
    location = payload.get("location") or ""

    parts = [date_str, str(merchant), str(category), amount_str]
    if location:
        parts.append(str(location))

    return "- " + " | ".join(parts)
