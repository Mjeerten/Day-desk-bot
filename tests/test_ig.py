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
    c, calls = client(monkeypatch, [Resp(403, DENY)] * 3)
    with pytest.raises(RuntimeError, match="firewall"):
        c.req("POST", "/session")
    assert len(calls) == 3


def test_api_error_not_retried(monkeypatch):
    c, calls = client(monkeypatch, [Resp(403, '{"errorCode":"error.security.invalid-details"}')])
    with pytest.raises(RuntimeError, match="invalid-details"):
        c.req("POST", "/session")
    assert len(calls) == 1
