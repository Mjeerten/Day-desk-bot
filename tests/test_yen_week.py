import pandas as pd
import pytest

from desk import autotrade as at
from desk import yen_week as yw

NY = "America/New_York"


def t(s):
    return pd.Timestamp(s, tz=NY)


def test_target_windows():
    assert yw.target(t("2026-11-16 01:59"))[0] == 0                      # Monday before 02:00
    assert yw.target(t("2026-11-16 02:16"))[:2] == (1, "Y261116L")       # the hourly run at :16
    assert yw.target(t("2026-11-18 16:31"))[:2] == (1, "Y261116L")       # Wednesday before the flip
    assert yw.target(t("2026-11-18 16:46"))[:2] == (-1, "Y261116S")      # Wednesday 22:46 Amsterdam run
    assert yw.target(t("2026-11-20 16:31"))[:2] == (-1, "Y261116S")
    assert yw.target(t("2026-11-20 16:46"))[0] == 0                      # Friday: out before the weekend
    assert yw.target(t("2026-11-21 12:00"))[0] == 0
    assert yw.target(t("2026-11-22 20:00"))[0] == 0                      # Sunday evening: still flat


class FakeIG:
    book, calls, status = {}, [], "TRADEABLE"

    def __init__(self):
        pass

    def login(self):
        return {"accountInfo": {"balance": 10000, "profitLoss": 0}}

    def market(self, epic):
        assert epic == yw.EPIC
        return {"snapshot": {"bid": 160.073, "offer": 160.083, "marketStatus": FakeIG.status, "decimalPlacesFactor": 3},
                "dealingRules": {"minDealSize": {"value": 0.2}},
                "instrument": {"contractSize": "10000", "currencies": [{"code": "JPY", "isDefault": False}]}}

    def positions(self):
        return [{"market": {"epic": yw.EPIC}, "position": dict(p)} for p in self.book.values()]

    def open(self, epic, direction, size, level, stop, limit, ref, currency="EUR", digits=1):
        d = f"D{len(self.calls)}"
        self.calls.append(("open", direction, size, round(level, digits), stop, limit, currency))
        self.book[d] = {"dealId": d, "size": size, "level": 160.08, "dealReference": ref}
        return {"dealStatus": "ACCEPTED", "level": 160.08, "dealId": d}

    def close(self, deal, direction, size, level, ref, digits=1):
        self.calls.append(("close", deal, direction, size))
        self.book.pop(deal)
        return {"dealStatus": "ACCEPTED", "level": 160.5}


@pytest.fixture
def ig(monkeypatch, tmp_path):
    import desk.ig
    FakeIG.book, FakeIG.calls, FakeIG.status = {}, [], "TRADEABLE"
    monkeypatch.setattr(desk.ig, "IG", FakeIG)
    monkeypatch.setattr(at, "STATE", str(tmp_path / "ig-{mode}.json"))
    msgs = []
    monkeypatch.setattr(at, "notify", lambda *a, **k: msgs.append(a[0]))
    for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("AUTO_TRADE", "demo")
    monkeypatch.delenv("YEN_TRADE", raising=False)
    at.enabled = True
    FakeIG.msgs = msgs
    return FakeIG


def test_week_buy_flip_close(ig):
    yw.run(1000, t("2026-11-16 01:16"))                 # before the buy: nothing, no login
    assert ig.calls == []
    yw.run(1000, t("2026-11-16 02:16"))
    op = ig.calls[0]
    assert op[:3] == ("open", "BUY", 0.2) and op[4] is None and op[5] is None and op[6] == "JPY"
    assert op[3] == round(160.083 * (1 + yw.SLIP_PCT), 3)
    yw.run(1000, t("2026-11-16 03:16"))                 # holding: nothing new
    yw.run(1000, t("2026-11-18 16:31"))
    assert len(ig.calls) == 1
    yw.run(1000, t("2026-11-18 16:46"))                 # flip: close the buy, open the short
    assert ig.calls[1][:3] == ("close", "D0", "SELL") and ig.calls[2][:3] == ("open", "SELL", 0.2)
    yw.run(1000, t("2026-11-20 16:46"))                 # Friday: close, nothing new
    assert ig.calls[3][:3] == ("close", "D2", "BUY") and len(ig.calls) == 4
    st = at.load("demo")
    assert [v["status"] for v in st["yen"].values()] == ["closed", "closed"]
    assert st["yen"]["Y261116L"]["pct"] == pytest.approx((160.5 / 160.08 - 1) * 100, abs=1e-3)
    yw.run(1000, t("2026-11-21 10:00"))
    assert len(ig.calls) == 4


def test_too_late_and_never_twice(ig):
    yw.run(1000, t("2026-11-16 13:16"))                 # 11 h after Monday 02:00: no buy this week
    assert ig.calls == [] and at.load("demo")["yen"]["Y261116L"]["status"] == "too_late"
    yw.run(1000, t("2026-11-16 14:16"))
    assert ig.calls == []
    yw.run(1000, t("2026-11-19 02:16"))                 # short 9.5 h late: still opened
    assert ig.calls[0][:2] == ("open", "SELL")
    st = at.load("demo")
    st["yen"].clear()                                   # state lost: the deal reference on IG stops a second order
    at.save(st, "demo")
    yw.run(1000, t("2026-11-19 03:16"))
    assert len(ig.calls) == 1


def test_closed_market_dry_and_off(ig, monkeypatch):
    ig.status = "CLOSED"
    yw.run(1000, t("2026-11-16 02:16"))
    assert ig.calls == [] and "Y261116L" not in at.load("demo").get("yen", {})   # tries again next run
    ig.status = "TRADEABLE"
    monkeypatch.setenv("YEN_TRADE", "off")
    yw.run(1000, t("2026-11-16 03:16"))
    assert ig.calls == []
    monkeypatch.delenv("YEN_TRADE")
    monkeypatch.setenv("AUTO_TRADE", "dry")
    yw.run(1000, t("2026-11-16 03:16"))
    assert ig.calls == [] and at.load("dry")["yen"]["Y261116L"]["status"] == "open"   # logged, nothing sent


def test_bigger_stake_sizes_up(ig):
    yw.run(5000, t("2026-11-16 02:16"))                 # 2 x EUR 5,000 / 0.86 / 10,000 = 1.16 -> 1.2 (steps of 0.2)
    assert ig.calls[0][2] == 1.2


def test_report_names_yen(monkeypatch):
    st = {"trades": {}, "yen": {"Y261116L": dict(level=160.08, status="closed")}}
    assert at._name_of(st, 160.08) == "yen week"
    assert at._name_of(st, 25000.0) == "trade"
