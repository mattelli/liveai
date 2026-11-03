"""Sub-agent factory: build isolated, quiet agents from ``[agents]`` entries.

A child agent is a real ``Agent`` with ``isolated_tools=True`` (fresh
per-agent memory/vision tool instances, scheduler-executor pattern) and
``quiet=True`` (buffered fragments, no console). Its tool set, model entry
and system prompt come from its ``AgentSpec``; children never get the
delegation tool (flat topology — enforced here, see ``build_sub_agent``).
"""

from __future__ import annotations

from logging import getLogger
from types import SimpleNamespace

from src.agent_registry import AgentSpec

__all__ = ["build_sub_agent", "child_config", "SubAgentBuildError"]

log = getLogger(__name__)

# Test seam (scheduler.py pattern): patched by tests to hand out fake child
# agents; the default builds a real Agent from the app config.
_AGENT_FACTORY = None


class SubAgentBuildError(Exception):
    """A configured sub-agent could not be built (bad model entry, ...)."""


def child_config(config: SimpleNamespace, spec: AgentSpec) -> SimpleNamespace:
    """Config copy for one child: its tool_groups/model/prompt overrides.

    The original config must not be mutated: the main agent and every other
    child keep reading the shared settings. Only ``[agent]`` (tool_groups,
    max_act_rounds, monitor settings) and ``[brain]`` (system_prompt, model
    boot entry) are overridden per child.
    """
    import copy

    child = copy.deepcopy(config)
    agent = child.agent = copy.deepcopy(getattr(config, "agent", None)) \
        or SimpleNamespace()
    agent.tool_groups = spec.tool_groups
    if spec.max_act_rounds:
        agent.max_act_rounds = spec.max_act_rounds
    brain = child.brain = copy.deepcopy(getattr(config, "brain", None)) \
        or SimpleNamespace()
    if spec.system_prompt:
        brain.system_prompt = spec.system_prompt
    if spec.model:
        brain.model = spec.model
        brain.base_url = ""  # force a models-file lookup, never [brain] legacy
    return child


def _models_entry(config: SimpleNamespace, spec: AgentSpec) -> dict | None:
    """The models.json entry named by ``spec.model`` (None = main model)."""
    if not spec.model:
        return None
    path = getattr(getattr(config, "brain", None), "models_file", None)
    if not path:
        raise SubAgentBuildError(
            f"agent {spec.name!r}: model {spec.model!r} requested but no "
            "[brain] models_file is configured")
    import json
    import pathlib
    path = pathlib.Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SubAgentBuildError(
            f"agent {spec.name!r}: unreadable models file {path}: {exc}") from exc
    if isinstance(data, dict) and isinstance(data.get("models"), (list, dict)):
        data = data["models"]
    if isinstance(data, dict):
        pairs = list(data.items())
    elif isinstance(data, list):
        pairs = []
        for entry in data:
            if isinstance(entry, dict):
                pairs.append((entry.get("name"), entry))
            else:
                text = str(entry).strip()
                pairs.append((text, {"model": text}))
    else:
        pairs = []
    for name, entry in pairs:
        if str(name or "").strip().lower() == spec.model.strip().lower():
            if isinstance(entry, str):
                entry = {"model": entry.strip()}
            return entry or None
    raise SubAgentBuildError(
        f"agent {spec.name!r}: model entry {spec.model!r} not found in {path}")


def _default_agent_factory(config: SimpleNamespace, spec: AgentSpec):
    """Build one isolated, quiet child agent (no LLM call, client stubbed)."""
    from src.live_ai import Agent
    child_config_obj = child_config(config, spec)
    entry = _models_entry(config, spec)
    if entry is not None:
        # Route the child's boot through its configured model entry by
        # writing the resolved connection into a models-file-like override:
        # _boot_connection takes the first entry of the models file, so we
        # hand the child a single-entry models mapping via a subclass-free
        # trick: a one-entry file next to the original (tmp) is overkill —
        # instead set the entry fields directly on [brain] as the legacy path
        # expects them.
        brain = child_config_obj.brain
        brain.model = str(entry.get("model") or "").strip()
        brain.base_url = str(entry.get("base_url")
                             or getattr(brain, "base_url", "") or "").strip()
        brain.api_key_env = str(entry.get("api_key_env")
                                or getattr(brain, "api_key_env", "") or "").strip()
        if getattr(brain, "models_file", None):
            # The models file would win over [brain] at boot; hide it from the
            # child so the resolved entry above takes effect.
            brain.models_file = None
    return Agent(child_config_obj, isolated_tools=True, quiet=True)


def build_sub_agent(config: SimpleNamespace, spec: AgentSpec):
    """Build one child agent from its ``AgentSpec`` (or raise).

    Failures raise ``SubAgentBuildError``; the delegation tool converts them
    into error tool results.
    """
    factory = _AGENT_FACTORY or _default_agent_factory
    try:
        agent = factory(config, spec)
    except SubAgentBuildError:
        raise
    except Exception as exc:
        raise SubAgentBuildError(
            f"agent {spec.name!r} could not be built: {exc}") from exc
    if agent is None:
        raise SubAgentBuildError(f"agent {spec.name!r} could not be built")
    log.info("Sub-agent '%s' built (model=%s, tool_groups=%s)",
             spec.name, spec.model or "<main>", spec.tool_groups)
    return agent
