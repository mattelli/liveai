"""Tests for the wikipedia_search tool (smolagents WikipediaSearchTool port)."""
import pytest

import src.tools.wiki as wiki_mod


class _FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(f"{self.status_code}")

    def json(self):
        return self._payload


def test_wikipedia_search_returns_summary_and_url(monkeypatch):
    search_payload = {"query": {"search": [
        {"title": "Alan Turing"}]}}
    summary_payload = {
        "title": "Alan Turing",
        "extract": "Alan Turing was an English mathematician.",
        "content_urls": {"desktop": {"page":
            "https://en.wikipedia.org/wiki/Alan_Turing"}},
    }
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "api.php" in url:
            return _FakeResponse(search_payload)
        return _FakeResponse(summary_payload)

    monkeypatch.setattr(wiki_mod, "_lazy_requests", None, raising=False)
    import sys
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(fake_get)}))
    result = wiki_mod.wikipedia_search("alan turing")
    assert "Alan Turing" in result
    assert "Alan Turing was an English mathematician." in result
    assert "https://en.wikipedia.org/wiki/Alan_Turing" in result
    assert len(calls) == 2  # search + summary


def test_wikipedia_search_handles_missing_page(monkeypatch):
    def fake_get(url, **kwargs):
        return _FakeResponse({"query": {"search": []}})

    import sys
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(fake_get)}))
    result = wiki_mod.wikipedia_search("zzz-nonexistent-query")
    assert "No Wikipedia articles found" in result


def test_wikipedia_search_sends_user_agent(monkeypatch):
    seen = {}

    def fake_get(url, **kwargs):
        seen.update(kwargs)
        return _FakeResponse({"query": {"search": []}})

    import sys
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(fake_get)}))
    wiki_mod.wikipedia_search("x")
    assert seen["headers"]["User-Agent"] == wiki_mod._USER_AGENT


def test_wikipedia_search_error_never_raises(monkeypatch):
    def fake_get(url, **kwargs):
        raise ConnectionError("offline")

    import sys
    monkeypatch.setitem(sys.modules, "requests",
                        type("R", (), {"get": staticmethod(fake_get)}))
    result = wiki_mod.wikipedia_search("x")
    assert result.startswith("Error:")


def test_wikipedia_search_rejects_empty_query():
    assert wiki_mod.wikipedia_search("").startswith("Error:")


def test_wikipedia_search_in_build_tools():
    import src.live_ai as live_ai_mod
    agent = live_ai_mod.Agent.__new__(live_ai_mod.Agent)
    names = [tool.__name__ for tool in agent._build_tools()]
    assert "wikipedia_search" in names
