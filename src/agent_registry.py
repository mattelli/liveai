"""[agents] registry parsing: named sub-agent entries for task delegation.

Pure parsing/validation, unit-testable without any agent construction: a
``SimpleNamespace`` config in, ``{name: AgentSpec}`` out. The registry lives
in ``src.live_ai`` because the delegation tool and the child-agent factory
both need it (the tool spawns children; the factory builds them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from logging import getLogger
from types import SimpleNamespace

__all__ = ["AgentSpec", "load_agent_registry"]

log = getLogger(__name__)

# Fields accepted inside a ``[agents.<name>]`` table; anything else is logged
# and ignored (typos must not silently break the whole registry).
_SPEC_FIELDS = frozenset({
    "tool_groups", "model", "system_prompt", "max_act_rounds", "description",
})
# Settings of the shared ``[agents]`` table (not sub-agent entries).
_SHARED_FIELDS = frozenset({
    "summarize_results", "summarize_max_chars",
})

# Registry entry: one configured sub-agent.
_AGENT_SPECS: dict[str, "AgentSpec"] = {}
# Shared [agents] table settings (summarization).
_SUMMARIZE_RESULTS = False
_SUMMARIZE_MAX_CHARS = 800


@dataclass
class AgentSpec:
    """One configured sub-agent from a ``[agents.<name>]`` table."""

    name: str
    tool_groups: str = "all"
    model: str = ""                 # models.json entry; "" = main model
    system_prompt: str = ""         # "" = default prompt
    max_act_rounds: int = 0         # 0 = Agent class default
    description: str = ""           # shown to the parent in the prompt section


def load_agent_registry(config: SimpleNamespace) -> dict[str, AgentSpec]:
    """Parse the ``[agents]`` config section into ``{name: AgentSpec}``.

    Accepted shapes per entry: a table (``[agents.researcher]``) or a bare
    string shorthand (``researcher = "web, files"`` → tool_groups). Unknown
    keys, invalid values and malformed tables are logged and skipped — one
    broken entry must not disable the others. The shared table (scalar keys
    like ``summarize_results``) carries feature flags, not agents.

    Also populates the module-level registry used by the delegation tool and
    updates the summarization settings. Returns the parsed specs.
    """
    global _SUMMARIZE_RESULTS, _SUMMARIZE_MAX_CHARS
    section = getattr(config, "agents", None)
    specs: dict[str, AgentSpec] = {}
    if isinstance(section, SimpleNamespace):
        section = vars(section)
    if not isinstance(section, dict):
        _AGENT_SPECS.clear()
        return specs

    for key, value in section.items():
        if key in _SHARED_FIELDS:
            continue  # shared flags, handled below
        if not isinstance(value, (dict, SimpleNamespace, str)):
            log.warning("[agents] entry %r ignored: not a table", key)
            continue
        name = str(key).strip()
        if not name:
            log.warning("[agents] entry with empty name ignored")
            continue
        if name in specs:
            log.warning("[agents] duplicate entry %r ignored", name)
            continue
        if isinstance(value, str):  # shorthand: tool_groups string
            value = {"tool_groups": value}
        elif isinstance(value, SimpleNamespace):
            value = vars(value)
        unknown = set(value) - _SPEC_FIELDS
        if unknown:
            log.warning("[agents] %s: unknown keys ignored: %s",
                        name, ", ".join(sorted(unknown)))
        rounds = value.get("max_act_rounds", 0)
        try:
            rounds = max(0, int(rounds or 0))
        except (TypeError, ValueError):
            log.warning("[agents] %s: invalid max_act_rounds %r ignored "
                        "(using the Agent default)", name, rounds)
            rounds = 0
        specs[name] = AgentSpec(
            name=name,
            tool_groups=str(value.get("tool_groups", "all") or "all"),
            model=str(value.get("model", "") or "").strip(),
            system_prompt=str(value.get("system_prompt", "") or ""),
            max_act_rounds=rounds,
            description=str(value.get("description", "") or ""),
        )

    summarize = section.get("summarize_results", False)
    _SUMMARIZE_RESULTS = bool(summarize)
    try:
        _SUMMARIZE_MAX_CHARS = max(0, int(section.get("summarize_max_chars", 800) or 0))
    except (TypeError, ValueError):
        _SUMMARIZE_MAX_CHARS = 800

    _AGENT_SPECS.clear()
    _AGENT_SPECS.update(specs)
    if specs:
        log.info("[agents] registry loaded: %s", ", ".join(sorted(specs)))
    return specs


def agent_registry() -> dict[str, AgentSpec]:
    """The module-level ``[agents]`` registry (empty = feature disabled)."""
    return _AGENT_SPECS


def summarize_settings() -> tuple[bool, int]:
    """``(summarize_results, summarize_max_chars)`` from ``[agents]``."""
    return _SUMMARIZE_RESULTS, _SUMMARIZE_MAX_CHARS
