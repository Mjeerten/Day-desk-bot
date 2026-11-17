"""IG client: a firewall deny page (HTTP 403 HTML) is retried; a real API error is not."""
import pytest

from desk import ig as igmod

DENY = "<!doctype html> <html> <head> <title>Custom Deny Page for IG Sites</title>"


class Resp:
    def __init__(self, status, text="{}", headers=None):
        self.status_code, self.text, self.headers = status, text, headers or {}


def client(monkeypatch, replies):
    calls = []

    def fake(method, url, **kw):
        calls.append(url)
        return replies.pop(0)
    monkeypatch.setattr(igmod.requests, "request", fake)
    monkeypatch.setattr(igmod.time, "sleep", lambda s: None)
    return igmod.IG(key="k", user="u", password="p", account=""), calls


def test_deny_page_then_ok(monkeypatch):
    c, calls = client(monkeypatch, [Resp(403, DENY), Resp(200, "{}")])
    assert c.req("POST", "/session").status_code == 200
    assert len(calls) == 2


def test_deny_page_every_time(monkeypatch):
    c, calls = client(monkeypatch, [Resp(403, DENY)] * 2)
    with pytest.raises(igmod.Blocked, match="firewall"):
        c.req("POST", "/session")
    assert len(calls) == 2


def test_api_error_not_retried(monkeypatch):
    c, calls = client(monkeypatch, [Resp(403, '{"errorCode":"error.security.invalid-details"}')])
    with pytest.raises(RuntimeError, match="invalid-details"):
        c.req("POST", "/session")
    assert len(calls) == 1


def test_block_asks_workflow_for_a_fresh_server(monkeypatch, tmp_path):
    """First job: a block writes ig_blocked=true for daybot.yml and sends no message. Retry job: message as before."""
    from desk import autotrade as at
    out = tmp_path / "out"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("IG_RETRY_JOB", raising=False)
    assert at.retry_elsewhere(igmod.Blocked("x"), "ig") is True
    assert out.read_text() == "ig_blocked=true\n"
    assert at.retry_elsewhere(RuntimeError("HTTP 500"), "ig") is False      # other errors: message straight away
    assert at.retry_elsewhere(igmod.Blocked("x"), "capital") is False
    monkeypatch.setenv("IG_RETRY_JOB", "1")
    assert at.retry_elsewhere(igmod.Blocked("x"), "ig") is False             # blocked on the fresh server too


def test_flush_stays_quiet_on_block(monkeypatch, tmp_path):
    from desk import autotrade as at
    sent = []
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "out"))
    monkeypatch.delenv("IG_RETRY_JOB", raising=False)
    monkeypatch.setattr(at, "enabled", True)
    monkeypatch.setattr(at, "_wants", {"x": 1})
    monkeypatch.setattr(at, "mode", lambda b: "demo" if b == "ig" else "off")
    monkeypatch.setattr(at, "notify", lambda *a, **k: sent.append(a))

    def blocked(*a, **k):
        raise igmod.Blocked("blocked")
    monkeypatch.setattr(at, "_flush", blocked)
    at.flush(1000)
    assert sent == []
    monkeypatch.setenv("IG_RETRY_JOB", "1")
    monkeypatch.setattr(at, "_wants", {"x": 1})
    at.flush(1000)
    assert len(sent) == 1 and "problem" in sent[0][0]
