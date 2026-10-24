"""Task scheduler: run agent prompts at a future time (one-shot or recurring).

The scheduler follows the ``src.tools`` architecture: config is injected by
``main.py`` via :func:`configure` (``[scheduler]`` section), the tool logic
lives in :class:`SchedulerTools` (per-agent isolation via ``for_owner``, like
``memory.py``), and the module-level tools delegate to a module-default
instance so the historical call API and test seams keep working.

``[scheduler]`` config section (all keys optional):

.. code-block:: toml

    [scheduler]
    enabled = true                # false (or missing section) -> tools are no-ops
    persist_file = "data/schedule.json"  # resolved relative to the config file
    on_missed = "skip"            # one-shot tasks missed while the app was off:
                                  # "run" -> execute immediately, "skip" -> drop

``Scheduler`` is the core engine: a daemon thread sleeping on a
``threading.Event`` until the next job is due (no busy polling), a ``heapq``
min-heap ordered by next-run time, and atomic JSON persistence so tasks
survive restarts. Execution of due tasks is delegated to an injected
``_execute`` callback (Task 3 wires a dedicated isolated agent there); the
default callback only logs, so the engine is testable without an LLM.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from src.common import dump_func_call, set_subagent_dump

__all__ = [
    "ScheduledTask", "Scheduler", "SchedulerTools", "configure",
    "bind_scheduler", "cancel_scheduled_task", "list_scheduled_tasks",
    "drain_scheduled_results", "next_occurrence", "parse_when", "schedule_task",
]

logger = logging.getLogger(__name__)

_DISABLED_MESSAGE = "Error: scheduler is disabled (set [scheduler] enabled = true)"

# Module state injected by configure() (see the module docstring for the
# [scheduler] section). Missing section -> the scheduler stays disabled.
_CONFIG: dict = {}
_SCHEDULER: "Scheduler | None" = None

# Test seam (module-level like ``_VIDEO_CAPTURE_FACTORY`` in camera.py):
# patched by tests to hand out fake schedulers; the default builds the real
# Scheduler from the injected config.
_SCHEDULER_FACTORY = None


def _default_scheduler_factory() -> "Scheduler":
    """Factory used when no scheduler is injected (the real engine)."""
    return Scheduler(
        persist_file=_CONFIG.get("persist_file"),
        on_missed=_CONFIG.get("on_missed", "skip"),
        _execute=_agent_executor,
    )


def configure(config) -> None:
    """
    Inject the application config into the tool (called by ``main.py`` after
    ``load_config``). Keeps the ``[scheduler]`` section (plus the full config
    for the execution-agent factory); a missing section keeps the scheduler
    disabled (tools return the disabled-error string).
    """
    global _CONFIG
    section = getattr(config, "scheduler", None)
    _CONFIG = {
        "enabled": bool(getattr(section, "enabled", False)),
        "persist_file": getattr(section, "persist_file", None),
        "on_missed": getattr(section, "on_missed", "skip"),
        "config": config,  # full config: the execution agent needs [brain]
    }


def _enabled() -> bool:
    return bool(_CONFIG.get("enabled", False))


## time parsing / recurrence ##

def parse_when(at: str) -> datetime:
    """
    Parse a scheduling time: ISO 8601 (``datetime.fromisoformat``) or the
    compact ``YYYY-MM-DD HH:MM`` form. Naive datetimes are interpreted as
    local time. Raises ``ValueError`` on anything else.
    """
    if not isinstance(at, str) or not at.strip():
        raise ValueError("at must be a non-empty datetime string")
    text = at.strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    if re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}", text):
        return datetime.strptime(text, "%Y-%m-%d %H:%M")
    raise ValueError(
        f"Unsupported datetime format {text!r}: use ISO 8601 or 'YYYY-MM-DD HH:MM'")


_DAILY = "daily"
_HOURLY = "hourly"
_EVERY_RE = re.compile(r"every\s+(\d+)\s*([smhd])$", re.IGNORECASE)
_CRON_FIELDS = 5

# Cron day-of-week / month names (Python weekday(): Monday == 0).
_DOW_NAMES = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5,
              "sun": 6}
_MONTH_NAMES = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
                "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}


def next_occurrence(repeat: str, after: datetime) -> datetime:
    """
    Next run time of a recurring task, strictly after ``after``.

    Supported formats: ``daily``, ``hourly``, ``every <N><s|m|h|d>``
    (e.g. ``every 30m``) and a cron-like 5-field ``min hour dom month dow``
    where ``*`` means any value. Raises ``ValueError`` on unknown formats.
    """
    if not isinstance(repeat, str) or not repeat.strip():
        raise ValueError("repeat must be a non-empty string ('' means one-shot)")
    text = repeat.strip().lower()
    now = after
    if text == _DAILY:
        return now + timedelta(days=1)
    if text == _HOURLY:
        return now + timedelta(hours=1)
    match = _EVERY_RE.fullmatch(text)
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        if amount <= 0:
            raise ValueError(f"repeat interval must be positive: {repeat!r}")
        return now + timedelta(**{ {"s": "seconds", "m": "minutes",
                                   "h": "hours", "d": "days"}[unit]: amount })
    fields = text.split()
    if len(fields) == _CRON_FIELDS:
        return _next_cron(fields, now)
    raise ValueError(
        f"Unsupported repeat format {repeat!r}: use 'daily', 'hourly', "
        "'every <N><s|m|h|d>' or a 5-field cron expression")


def _next_cron(fields: list[str], after: datetime) -> datetime:
    """Next time matching a cron-like ``min hour dom month dow`` expression."""
    minute, hour, dom, month, dow = fields
    candidate = after.replace(second=0, microsecond=0) + timedelta(minutes=1)
    for _ in range(366 * 24 * 60):  # bounded search: at most one year of minutes
        if not _cron_field_matches(minute, candidate.minute, 0, 59):
            candidate += timedelta(minutes=1)
            continue
        if not _cron_field_matches(hour, candidate.hour, 0, 23):
            candidate += timedelta(minutes=1)
            continue
        if not _cron_field_matches(month, candidate.month, 1, 12):
            candidate += timedelta(minutes=1)
            continue
        if not _cron_field_matches(dom, candidate.day, 1, 31):
            candidate += timedelta(minutes=1)
            continue
        if not _cron_field_matches(dow, candidate.weekday(), 0, 6):
            candidate += timedelta(minutes=1)
            continue
        return candidate
    raise ValueError(f"No matching time for cron expression {fields!r} within a year")


def _cron_field_matches(field: str, value: int, low: int, high: int) -> bool:
    """Cron field match: ``*``, a number, a name (MON..SUN, JAN..DEC),
    ``*/step`` or ``a-b``."""
    field = field.strip().lower()
    if field == "*":
        return True
    if field in _DOW_NAMES:
        return value == _DOW_NAMES[field]
    if field in _MONTH_NAMES:
        return value == _MONTH_NAMES[field]
    if field.startswith("*/"):
        try:
            step = int(field[2:])
        except ValueError:
            return False
        return step > 0 and value % step == 0
    if "-" in field:
        try:
            lo, hi = (int(part) for part in field.split("-", 1))
        except ValueError:
            return False
        return lo <= value <= hi
    try:
        return value == int(field)
    except ValueError:
        return False


## scheduler core ##

@dataclass
class ScheduledTask:
    """One scheduled prompt: ``repeat == ""`` means a one-shot task."""
    id: str
    prompt: str
    next_run: str  # ISO string (local time)
    repeat: str = ""

    def due_at(self) -> datetime:
        """The task's next-run time as a datetime (local time)."""
        return datetime.fromisoformat(self.next_run)


@dataclass(order=True)
class _HeapEntry:
    """Heap wrapper: orders by (due time, sequence) so ties stay stable."""
    due: float
    seq: int
    task_id: str = field(compare=False)


class Scheduler:
    """
    Core scheduling engine: heap of jobs + a daemon thread that sleeps until
    the next job is due, executes it via the injected ``_execute`` callback
    and reschedules recurring tasks. Tasks are persisted to a JSON file on
    every mutation (atomic replace), so they survive restarts.

    ``_execute`` is injectable for tests; the default only logs (Task 3
    wires the dedicated execution agent in).
    """

    def __init__(self, persist_file: str | Path | None = None,
                 on_missed: str = "skip", _execute: "callable | None" = None):
        self.logger = logging.getLogger(f"{__name__}.Scheduler")
        self._persist_file = Path(persist_file) if persist_file else None
        self._on_missed = on_missed if on_missed in ("run", "skip") else "skip"
        self._execute = _execute or self._default_execute
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._stop_requested = threading.Event()
        self._thread: threading.Thread | None = None
        self._tasks: dict[str, ScheduledTask] = {}
        self._heap: list[_HeapEntry] = []
        self._seq = 0
        self._next_id = 1
        self._load()

    ## public API ##

    def add(self, prompt: str, at: str, repeat: str = "") -> ScheduledTask:
        """Schedule ``prompt`` at ``at`` (one-shot, or recurring via ``repeat``).

        A one-shot in the past raises ``ValueError`` (recurring tasks always
        start from their first future occurrence).
        """
        run_at = parse_when(at)
        if not repeat and run_at <= datetime.now():
            raise ValueError(f"scheduled time {at!r} is in the past")
        if repeat:  # reject invalid recurrence up-front (would crash _run_loop)
            next_occurrence(repeat, run_at)
        with self._lock:
            task = ScheduledTask(
                id=f"t-{self._next_id}", prompt=prompt,
                next_run=run_at.isoformat(), repeat=repeat or "")
            self._next_id += 1
            self._tasks[task.id] = task
            self._push(task)
            self._save()
        self._wake.set()  # the loop may now sleep until an earlier time
        return task

    def cancel(self, task_id: str) -> bool:
        """Remove a task; ``False`` when the id is unknown."""
        with self._lock:
            if task_id not in self._tasks:
                return False
            del self._tasks[task_id]
            # Heap entries for the removed task are skipped lazily in the loop
            # (stale ids never match a live task), so no heap rebuild is needed.
            self._save()
        self._wake.set()
        return True

    def list(self) -> list[ScheduledTask]:
        """Pending tasks ordered by next-run time."""
        with self._lock:
            return sorted(self._tasks.values(), key=lambda t: t.next_run)

    def start(self) -> None:
        """Start the background thread (idempotent)."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_requested.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="task-scheduler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        """Stop the background thread (idempotent; waits for it to exit)."""
        self._stop_requested.set()
        self._wake.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    ## internals ##

    def _push(self, task: ScheduledTask) -> None:
        self._seq += 1
        heapq.heappush(self._heap, _HeapEntry(task.due_at().timestamp(), self._seq, task.id))

    def _run_loop(self) -> None:
        """Sleep until the next job is due, execute it, repeat until stopped."""
        while not self._stop_requested.is_set():
            with self._lock:
                entry = self._next_live_entry()
            if entry is None:
                self._wake.wait(timeout=1.0)  # nothing due: idle wake-ups only
                self._wake.clear()
                continue
            delay = entry.due - datetime.now().timestamp()
            if delay > 0:
                self._wake.wait(timeout=delay)
                self._wake.clear()
                continue
            self._pop_and_run(entry)

    def _next_live_entry(self) -> "_HeapEntry | None":
        """Earliest heap entry whose task is still live (skips stale ids)."""
        while self._heap:
            entry = self._heap[0]
            if entry.task_id in self._tasks:
                return entry
            heapq.heappop(self._heap)
        return None

    def _pop_and_run(self, entry: _HeapEntry) -> None:
        with self._lock:
            task = self._tasks.get(entry.task_id)
            if task is None:
                return  # cancelled between the peek and the pop
            if task.due_at().timestamp() > datetime.now().timestamp():
                return  # woken early (e.g. by a later add): not due yet
            heapq.heappop(self._heap)
            if task.repeat:
                next_run = next_occurrence(task.repeat, task.due_at())
                if next_run <= datetime.now():
                    # Missed several periods (e.g. the machine slept): jump to
                    # the first *future* occurrence instead of catch-up loops.
                    while next_run <= datetime.now():
                        next_run = next_occurrence(task.repeat, next_run)
                task.next_run = next_run.isoformat()
                self._push(task)
                self._save()
            else:
                del self._tasks[task.id]
                self._save()
        snapshot = ScheduledTask(task.id, task.prompt, task.next_run, task.repeat)
        try:
            self._execute(snapshot)
        except Exception as exc:  # never let the thread die on one task
            self.logger.error("Scheduled task %s execution failed: %s",
                              task.id, exc)

    def _default_execute(self, task: ScheduledTask) -> None:
        """Default executor: log only (used when no agent executor is bound)."""
        self.logger.info("Executing scheduled task %s: %s", task.id, task.prompt)

    ## persistence ##

    def _load(self) -> None:
        """Restore persisted tasks; handle missed one-shots per ``on_missed``."""
        if self._persist_file is None or not self._persist_file.exists():
            return
        try:
            records = json.loads(self._persist_file.read_text(encoding="utf-8"))
        except Exception as exc:
            self.logger.warning("Cannot read schedule file %s: %s",
                                self._persist_file, exc)
            return
        now = datetime.now().timestamp()
        missed: list[ScheduledTask] = []
        for record in records if isinstance(records, list) else []:
            try:
                task = ScheduledTask(
                    id=str(record["id"]), prompt=str(record["prompt"]),
                    next_run=str(record["next_run"]), repeat=str(record.get("repeat", "")))
            except (KeyError, TypeError) as exc:
                self.logger.warning("Skipping malformed schedule record: %s", exc)
                continue
            if task.repeat:
                try:  # skip tasks with an invalid repeat (e.g. legacy files)
                    next_occurrence(task.repeat, task.due_at())
                except ValueError as exc:
                    self.logger.warning("Skipping task %s with invalid repeat: %s",
                                        task.id, exc)
                    continue
            if task.due_at().timestamp() <= now and not task.repeat:
                missed.append(task)  # one-shot missed while the app was off
                continue
            self._tasks[task.id] = task
            self._push(task)
            self._next_id = max(self._next_id, int(task.id.split("-")[-1]) + 1)
        if missed and self._on_missed == "run":
            for task in missed:
                self.logger.info("Running missed one-shot task %s", task.id)
                try:
                    self._execute(task)
                except Exception as exc:
                    self.logger.error("Missed task %s execution failed: %s",
                                      task.id, exc)
        elif missed:
            self.logger.info("Dropping %d missed one-shot task(s) (on_missed=skip)",
                             len(missed))
        if self._tasks or missed:
            self._save()

    def _save(self) -> None:
        """Atomically persist the task list (write temp file + os.replace)."""
        if self._persist_file is None:
            return
        records = [
            {"id": t.id, "prompt": t.prompt, "next_run": t.next_run,
             "repeat": t.repeat}
            for t in self._tasks.values()
        ]
        try:
            self._persist_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._persist_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(records, indent=2), encoding="utf-8")
            os.replace(tmp, self._persist_file)
        except Exception as exc:
            self.logger.warning("Cannot write schedule file %s: %s",
                                self._persist_file, exc)


## execution agent (Task 3: run scheduled prompts on a dedicated agent) ##

# Test seam: patched by tests to hand out a fake execution agent (no LLM).
_AGENT_FACTORY = None


def _default_agent_factory():
    """Build the dedicated isolated execution agent (AgentPool recipe).

    Same pattern as pooled agents: quiet, tool-isolated, built from the app
    config kept by ``configure()``. Returns ``None`` when the config is
    missing or agent construction fails (the executor then only logs).
    """
    config = _CONFIG.get("config")
    if config is None:
        return None
    try:
        from src.live_ai import Agent
        return Agent(config, isolated_tools=True, quiet=True)
    except Exception as exc:
        logger.warning("Cannot build the scheduler execution agent: %s", exc)
        return None


def _agent_executor(task: ScheduledTask) -> None:
    """
    Run a due task's prompt on the dedicated scheduler agent.

    The agent is built lazily on first execution and owned exclusively by the
    scheduler thread (only one ``act()`` at a time — ``Agent.act()`` is not
    thread-safe). Agent failures are logged, never raised: the scheduler
    thread must survive a broken model endpoint.
    """
    global _EXECUTOR_AGENT
    agent = _EXECUTOR_AGENT
    if agent is None:
        factory = _AGENT_FACTORY or _default_agent_factory
        agent = factory()
        if agent is None:
            logger.warning("No execution agent available: task %s prompt "
                           "logged only: %s", task.id, task.prompt)
            return
        _EXECUTOR_AGENT = agent
    logger.info("Scheduled task %s -> execution agent", task.id)
    set_subagent_dump(False)
    try:
        agent.act(task.prompt)
    except Exception as exc:
        logger.error("Scheduled task %s execution failed: %s", task.id, exc)
        return
    finally:
        set_subagent_dump(True)
    # Deliver the answer to the user: the executor agent is quiet, so without
    # this the result would only live in the agent's own message history.
    answer = ""
    if agent.messages:
        answer = str(agent.messages[-1].get("content") or "").strip()
    if answer:
        logger.info("Scheduled task %s completed: %s", task.id, answer)
        # Queue the result so the main agent becomes aware of it on its next
        # turn (drain_scheduled_results); direct history writes from this
        # thread could corrupt an in-flight tool-call round.
        with _PENDING_RESULTS_LOCK:
            _PENDING_RESULTS.append((task.id, answer))
    else:
        logger.warning("Scheduled task %s produced no answer", task.id)


# Results of scheduled tasks waiting to be injected into the main agent's
# context (consumed by drain_scheduled_results, see ConversationSession).
_PENDING_RESULTS: list[tuple[str, str]] = []
_PENDING_RESULTS_LOCK = threading.Lock()


def drain_scheduled_results() -> list[tuple[str, str]]:
    """Return and clear the scheduled-task results pending main-agent delivery."""
    with _PENDING_RESULTS_LOCK:
        pending, _PENDING_RESULTS[:] = _PENDING_RESULTS[:], []
    return pending


# Lazily built dedicated execution agent (owned by the scheduler thread).
_EXECUTOR_AGENT = None


## agent tools (module-default instance, memory.py pattern) ##

class SchedulerTools:
    """Per-agent scheduling tools (own scheduler binding, no shared state)."""

    def __init__(self) -> None:
        self._scheduler: "Scheduler | None" = None

    def for_owner(self, agent) -> "SchedulerTools":
        """Fresh instance for a specific agent (per-agent tool isolation)."""
        return SchedulerTools()

    def bind_scheduler(self, scheduler: "Scheduler") -> None:
        """Bind the shared scheduler engine (called by main.py / tests)."""
        self._scheduler = scheduler

    def _scheduler_or_none(self) -> "Scheduler | None":
        if not _enabled():
            return None
        if self._scheduler is None:
            factory = _SCHEDULER_FACTORY or _default_scheduler_factory
            self._scheduler = factory()
        return self._scheduler

    @dump_func_call
    def schedule_task(self, prompt: str, at: str, repeat: str = "") -> str:
        """
        Schedule a task to run a prompt at a future time. Useful when the user
        asks to schedule, remind or run something later or repeatedly.
        ``at`` is a datetime (ISO 8601 or 'YYYY-MM-DD HH:MM', local time);
        ``repeat`` optionally makes it recurring: 'daily', 'hourly',
        'every 30m' or a 5-field cron expression like '0 9 * * MON'.
        Returns a confirmation with the task id; list/cancel with
        list_scheduled_tasks / cancel_scheduled_task.
        """
        scheduler = self._scheduler_or_none()
        if scheduler is None:
            return _DISABLED_MESSAGE
        try:
            run_at = parse_when(at)
        except ValueError as exc:
            return f"Error: {exc}"
        if not repeat and run_at <= datetime.now():
            return f"Error: scheduled time {at!r} is in the past"
        try:
            task = scheduler.add(prompt, at, repeat)
        except ValueError as exc:
            return f"Error: {exc}"
        except Exception as exc:  # tools never raise (scheduler failures too)
            return f"Error: scheduling failed: {exc}"
        label = task.repeat if task.repeat else "one-shot"
        return f"Scheduled task {task.id} at {task.next_run} ({label}): {task.prompt}"

    @dump_func_call
    def list_scheduled_tasks(self) -> str:
        """
        List the currently scheduled tasks with their ids, next run times,
        recurrence and prompts. Useful when the user asks what is scheduled.
        """
        scheduler = self._scheduler_or_none()
        if scheduler is None:
            return _DISABLED_MESSAGE
        tasks = scheduler.list()
        if not tasks:
            return "No scheduled tasks."
        lines = [
            f"{t.id} | {t.next_run} | {t.repeat or 'one-shot'} | "
            f"{t.prompt[:80]}"
            for t in tasks
        ]
        return "\n".join(lines)

    @dump_func_call
    def cancel_scheduled_task(self, task_id: str) -> str:
        """
        Cancel a scheduled task by id (ids come from list_scheduled_tasks).
        Useful when the user asks to cancel or unschedule a task.
        """
        scheduler = self._scheduler_or_none()
        if scheduler is None:
            return _DISABLED_MESSAGE
        if scheduler.cancel(str(task_id)):
            return f"Cancelled task {task_id}"
        return f"Error: unknown task id {task_id!r}"


class _ModuleDefaultSchedulerTools(SchedulerTools):
    """Module-default instance backed by the module global ``_SCHEDULER``.

    Keeps the historical ``tools.scheduler._SCHEDULER`` seam (patched by
    tests / bound by main.py) working while pooled agents use their own
    isolated instances.
    """

    @property
    def _scheduler(self):
        return globals()["_SCHEDULER"]

    @_scheduler.setter
    def _scheduler(self, value) -> None:
        globals()["_SCHEDULER"] = value


# Module-default instance: keeps the historical module-level API working for
# the main agent, existing callers and tests. Pooled agents use their own
# instance instead (see Agent._build_tools / SchedulerTools.for_owner).
_default = _ModuleDefaultSchedulerTools()


def bind_scheduler(scheduler: "Scheduler") -> None:
    """Bind the shared scheduler engine (module-default instance)."""
    _default.bind_scheduler(scheduler)


def schedule_task(prompt: str, at: str, repeat: str = "") -> str:
    """Schedule a prompt to run at a future time (module-default instance)."""
    return _default.schedule_task(prompt, at, repeat)


def list_scheduled_tasks() -> str:
    """List the scheduled tasks (module-default instance)."""
    return _default.list_scheduled_tasks()


def cancel_scheduled_task(task_id: str) -> str:
    """Cancel a scheduled task by id (module-default instance)."""
    return _default.cancel_scheduled_task(task_id)


def reset_executor_agent() -> None:
    """Drop the lazily built execution agent (test helper / model switch)."""
    global _EXECUTOR_AGENT
    _EXECUTOR_AGENT = None
