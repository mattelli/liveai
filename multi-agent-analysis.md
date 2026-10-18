# Multi-Agent Interaction — Analysis & Hints

*Project: `live_assistant` · Source analyzed: `src/live_ai.py`, `main.py`, `config.toml`*

---

## 1. Current architecture (what we build on)

```
OpenAICore   →  OpenAI-compatible client, models.json boot, /model switching
   └── Brain →  history plumbing, streaming, cancellation, model_info tool
         └── Agent (Brain) →  act() round loop + OpenAI function-calling tools
```

Key facts that shape the multi-agent design:

- **`Agent.act(prompt)`** is self-contained: stream a round → execute requested
  tools (`_execute_tool`) → feed results back → repeat until a plain answer
  (max `MAX_ACT_ROUNDS = 20`). A sub-agent is therefore just *another `Agent`
  instance whose `act()` is invoked from inside a tool*.
- **Tools are plain Python callables** listed in `_default_tools(agent)` and
  described to the model by `_openai_tool_definition(tool)` (signature +
  docstring → JSON Schema). Adding an "agent tool" is as easy as adding a
  function to that list.
- **`main.py` builds exactly one `Agent`** and hands it to
  `ConversationSession`. There is no pool today — that is the gap to fill.
- **`src.tools` modules hold session-bound state** (`memory_tools.bind_history`,
  `vision_tools.bind_session`) pointing at the *main* Brain. This is the main
  pitfall for parallel agents (see §5).

---

## 2. Recommended pattern: **agent-as-tool** (matches your example)

> *agent-1 can get some response from agent-2 which is using a tool.*

Wrap each sub-agent as a tool callable. Agent-1 (the orchestrator, the one the
user talks to) calls `ask_agent("agent-2", "...")`; agent-2 runs its own
`act()` loop, uses its tools (e.g. `web_search`), and its **final answer is
returned as the tool result** to agent-1.

```python
# src/live_ai.py (sketch)

class AgentPool:
    """Owns the extra agents; each has its own model, history and tools."""

    def __init__(self, config, names: list[str]):
        self.agents: dict[str, Agent] = {}
        for name in names:
            cfg = copy.deepcopy(config)          # independent brain settings
            cfg.brain.model = name               # or resolve from models.json
            self.agents[name] = Agent(cfg)

    def ask(self, agent_name: str, prompt: str) -> str:
        agent = self.agents[agent_name]
        agent.act(prompt)                        # full tool loop of agent-2
        return agent.messages[-1]["content"]     # final assistant text

    def ask_parallel(self, tasks: list[tuple[str, str]]) -> list[str]:
        # LLM calls are I/O-bound → threads are enough, no asyncio needed
        with ThreadPoolExecutor(max_workers=len(tasks)) as ex:
            return list(ex.map(lambda t: self.ask(*t), tasks))


def ask_agent(agent: str, prompt: str) -> str:
    """
    Delegate a task to a named sub-agent and return its final answer.
    The sub-agent can use its own tools to accomplish the task.
    """
    return POOL.ask(agent, prompt)
```

Register it in `_default_tools(agent)` next to the existing tools — the
function-calling machinery picks it up automatically.

**Flow of your example:**

1. User → agent-1: *"What's the weather in Torino and summarize my notes?"*
2. Agent-1 emits a tool call: `ask_agent(agent="agent-2", prompt="Get current weather in Torino")`
3. Agent-2's `act()` loop calls `web_search` (or any tool), answers.
4. The answer lands in agent-1's history as a normal `role: "tool"` message.
5. Agent-1 merges it with its own `summarize_doc` result and replies.

---

## 3. Parallelism: fan-out / map-reduce

For independent subtasks, don't serialize `ask_agent` calls — fan them out:

```python
def run_parallel_agents(tasks: list[dict]) -> str:
    """
    Run several sub-agents concurrently. Each task: {"agent": name, "prompt": text}.
    Returns a JSON list of {"agent", "prompt", "answer"} for aggregation.
    """
    results = POOL.ask_parallel([(t["agent"], t["prompt"]) for t in tasks])
    return json.dumps([
        {"agent": t["agent"], "prompt": t["prompt"], "answer": r}
        for t, r in zip(tasks, results)
    ])
```

Useful orchestration shapes:

| Pattern | How |
|---|---|
| **Orchestrator-worker** | agent-1 decomposes → `run_parallel_agents` → merges |
| **Pipeline** | agent-1 output becomes agent-2 prompt (sequential `ask_agent`) |
| **Draft–critique** | agent-1 drafts, agent-2 reviews, agent-1 revises |
| **Map-reduce** | N workers on chunks, agent-1 reduces |

---

## 4. Config: `[agent]` section (proposed schema)

The number of parallel agents and their models shall be declared here —
models reference the **entries of `models.json`** (`[brain] models_file`),
so `/model`-style switching and API keys keep working:

```toml
[agent]
bootstrap_prompt = "Recall user personal content and say 'Hello' to the user. Ask user name if it is unknown."

# --- multi-agent ---
parallel_agents = 2                                   # extra agents beside the running one
agent_models = ["glm-5.3-flash", "qwen2.5-coder"]     # models.json entry names, one per agent
# agent_system_prompts = ["You research the web.", "You write code."]   # optional
# agent_quiet = true                                  # workers don't print to console
```

Reading it in code (backward compatible — `getattr` with defaults):

```python
n = getattr(config.agent, "parallel_agents", 0)
models = getattr(config.agent, "agent_models", [])[:n]
```

`main.py` then builds `AgentPool(config, models)` and injects the pool into
the tool module (same pattern as `src.tools.configure(config)`), so
`ask_agent` / `run_parallel_agents` can reach it.

---

## 5. Pitfalls & safety checklist

- **Shared tool-module state — SOLVED: per-agent tool instances (accepted).**
  `memory_tools.bind_history(...)` and `vision_tools.bind_session(self)` are
  bound to the *main* Brain at `Brain.__init__`; a sub-agent constructed later
  would rebind them, and parallel agents would fight over one binding.
  **Accepted solution: every pooled agent gets its own tool instances** —
  no shared module-level tool state at all. Concretely: turn the
  session-bound tool modules into per-agent objects (or factory functions)
  and let `Agent._build_tools()` return fresh instances bound to `self`:

  ```python
  # src/live_ai.py
  def _build_tools(self) -> list[Callable]:
      tools = _default_tools(self)
      if getattr(self, "isolated_tools", False):
          # fresh instances per agent: history/session bindings live on the
          # instance, never on the module
          tools = [t.for_owner(self) if hasattr(t, "for_owner") else t
                   for t in tools]
      return tools
  ```

  With per-agent instances, `bind_history` / `bind_session` disappear from
  the module level and each parallel `act()` touches only its own agent's
  state — no locks, no thread-locals needed.
- **Console interleaving.** `StreamPrinter` writes to stdout; parallel
  `act()` calls garble output. Give workers a `quiet` flag (buffer fragments
  instead of printing) or prefix lines with the agent name.
- **Speech.** `_handle_round_end` calls `speech_tools.speak_if_enabled(...)`.
  Disable speech for workers — only the orchestrator talks.
- **Cancellation.** `cancel_generation()` flags one Brain only. Ctrl+C should
  at least mark the pool "shutting down"; workers may finish their round.
- **Side-effect tools.** `run_command`, `write_text_file`,
  `replace_string_in_file` can collide when two agents touch the same files.
  Assign disjoint scopes per agent, or route dangerous tools through a lock /
  allow-list per agent.
- **Local servers (Ollama / LM Studio).** Loading several models competes for
  VRAM; requests queue on one server. Prefer small worker models locally, or
  point workers at cloud endpoints for true parallelism.
- **Context & cost.** Each agent has its own history and `max_response_tokens`
  cap. Pass only **final answers** between agents, not raw tool dumps, to keep
  contexts small.
- **Round limits.** `MAX_ACT_ROUNDS = 20` applies per agent; a delegation
  chain (1→2→3) multiplies latency — keep the hierarchy shallow.

---

## 6. Minimal implementation plan

1. **`config.toml`** — add `parallel_agents` + `agent_models` under `[agent]`.
2. **`src/live_ai.py`** — add `AgentPool` (per-agent config copies), the
   `ask_agent` / `run_parallel_agents` tools, register them in
   `_default_tools`, add a `quiet` mode to `Agent` (buffered `StreamPrinter`,
   speech off). **Give each pooled agent its own tool instances** (accepted
   solution for the shared-state issue, see §5): `_build_tools()` returns
   fresh per-agent instances instead of module-bound tool functions.
3. **`main.py`** — after `configure_tools(config)`, build the pool from
   `[agent]` and inject it into the tools module.
4. **Test the example flow** — agent-1 delegating a tool-using task to
   agent-2, then a 2-task `run_parallel_agents` fan-out.