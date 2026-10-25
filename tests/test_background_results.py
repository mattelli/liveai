"""Tests for the shared background-results queue (scheduled tasks).

The queue is generic (kind, id, text); only the "scheduled" kind is produced
by the app after the agent-pool removal.
"""

import threading

import src.live_ai as live_ai_mod


def test_queue_then_drain_returns_and_empties():
    live_ai_mod.queue_agent_result("scheduled", "t-1", "answer one")
    live_ai_mod.queue_agent_result("scheduled", "t-2", "answer two")
    pending = live_ai_mod.drain_agent_results()
    assert pending == [("scheduled", "t-1", "answer one"),
                       ("scheduled", "t-2", "answer two")]
    assert live_ai_mod.drain_agent_results() == []  # emptied


def test_drain_on_empty_returns_empty_list():
    assert live_ai_mod.drain_agent_results() == []


def test_drain_of_kind_filters_and_keeps_other_kinds():
    live_ai_mod.queue_agent_result("scheduled", "t-1", "a")
    live_ai_mod.queue_agent_result("other", "x-1", "b")
    live_ai_mod.queue_agent_result("scheduled", "t-2", "c")

    assert live_ai_mod.drain_agent_results_of_kind("scheduled") == [
        ("t-1", "a"), ("t-2", "c")]
    # the other-kind entry survives the kind-filtered drain
    assert live_ai_mod.drain_agent_results() == [("other", "x-1", "b")]


def test_queue_is_thread_safe_no_loss():
    threads, per_thread = 8, 25

    def worker(worker_id):
        for i in range(per_thread):
            live_ai_mod.queue_agent_result("scheduled", f"t-{worker_id}-{i}", "x")

    runners = [threading.Thread(target=worker, args=(w,)) for w in range(threads)]
    for runner in runners:
        runner.start()
    for runner in runners:
        runner.join()

    pending = live_ai_mod.drain_agent_results()
    assert len(pending) == threads * per_thread
    assert len({entry_id for _, entry_id, _ in pending}) == threads * per_thread


def test_scheduler_drain_uses_shared_queue():
    """drain_scheduled_results is a thin wrapper over the shared queue."""
    from src.tools.scheduler import drain_scheduled_results

    live_ai_mod.queue_agent_result("scheduled", "t-9", "scheduled answer")
    live_ai_mod.queue_agent_result("other", "x-9", "other answer")
    assert drain_scheduled_results() == [("t-9", "scheduled answer")]
    # the other-kind entry is untouched by the scheduled-only drain
    assert live_ai_mod.drain_agent_results() == [("other", "x-9", "other answer")]
