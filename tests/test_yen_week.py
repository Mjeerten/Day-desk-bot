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


def test_messages_reach_the_phone(ig, monkeypatch):
    """Opens / closes are sent loud (quiet=False): they were silently logged only before 14 Nov 2026."""
    sent = []
    monkeypatch.setattr(at, "notify", lambda title, body, quiet=False, tags=("robot",), prio=3: sent.append((title, quiet, prio)))
    yw.run(1000, t("2026-11-16 02:16"))
    assert sent and sent[0][0].startswith("BUY USD/JPY") and sent[0][1] is False and sent[0][2] == 4


class FakeCap:
    book, calls, status = {}, [], "TRADEABLE"

    def __init__(self):
        pass

    def login(self):
        return {"accountInfo": {"balance": 1000, "profitLoss": 0}}

    def market(self, epic):
        assert epic == yw.CAP_EPIC
        return {"snapshot": {"bid": 155.10, "offer": 155.12, "marketStatus": FakeCap.status, "decimalPlacesFactor": 3},
                "dealingRules": {"minDealSize": {"value": 100}}, "instrument": {"lotSize": 1, "currency": "JPY"}}

    def positions(self):
        return [{"market": {"epic": yw.CAP_EPIC}, "position": dict(p)} for p in self.book.values()]

    def open(self, epic, direction, size, level, stop, limit, ref):
        d = f"C{len(self.calls)}"
        self.calls.append(("open", direction, size, stop, limit))
        self.book[d] = {"dealId": d, "size": size, "level": 155.12}
        return {"dealStatus": "ACCEPTED", "level": 155.12, "dealId": d}

    def close(self, deal_id, direction, size, level, ref):
        self.calls.append(("close", deal_id))
        self.book.pop(deal_id)
        return {"dealStatus": "ACCEPTED", "level": 156.0}


@pytest.fixture
def cap(monkeypatch, tmp_path):
    import desk.capital
    FakeCap.book, FakeCap.calls, FakeCap.status = {}, [], "TRADEABLE"
    monkeypatch.setattr(desk.capital, "Capital", FakeCap)
    monkeypatch.setattr(at, "CAP_STATE", str(tmp_path / "capital-{mode}.json"))
    sent = []
    monkeypatch.setattr(at, "notify", lambda title, body, quiet=False, tags=("robot",), prio=3: sent.append((title, quiet)))
    for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID"):
        monkeypatch.delenv(k, raising=False)                  # IG off: only Capital.com trades
    for k in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("CAPITAL_TRADE", "demo")
    monkeypatch.delenv("YEN_TRADE", raising=False)
    at.enabled = True
    FakeCap.sent = sent
    return FakeCap


def test_capital_week(cap):
    yw.run(1000, t("2026-11-16 02:16"))
    # dollars: 2 x EUR 1,000 / 0.86 = USD 2,326 -> 2,300 (steps of 100); market order, no stop / target
    assert cap.calls[0] == ("open", "BUY", 2300, None, None)
    assert cap.sent[0][0].endswith("| Capital.com demo") and cap.sent[0][1] is False
    yw.run(1000, t("2026-11-18 16:46"))
    assert cap.calls[1] == ("close", "C0") and cap.calls[2][:3] == ("open", "SELL", 2300)
    yw.run(1000, t("2026-11-20 16:46"))
    assert cap.calls[3] == ("close", "C2") and len(cap.calls) == 4
    st = at.load("demo", "capital")
    assert [v["status"] for v in st["yen"].values()] == ["closed", "closed"]
    assert st["yen"]["Y261116L"]["eur"] == pytest.approx((156.0 / 155.12 - 1) * 2300 * yw.USD_EUR, abs=0.01)


def test_capital_dry_is_quiet_and_stray_blocks(cap, monkeypatch):
    monkeypatch.setenv("CAPITAL_TRADE", "dry")
    yw.run(1000, t("2026-11-16 02:16"))
    assert cap.calls == [] and all(q for _, q in cap.sent)    # dry: logged, no phone
    monkeypatch.setenv("CAPITAL_TRADE", "demo")
    cap.book["HAND"] = {"dealId": "HAND", "size": 1000, "level": 155.0}   # opened by hand
    yw.run(1000, t("2026-11-16 03:16"))
    assert cap.calls == [] and "not opened" in cap.sent[-1][0]


def test_size_guard(cap, monkeypatch):
    real = FakeCap.market
    monkeypatch.setattr(FakeCap, "market", lambda self, e: dict(real(self, e), dealingRules={"minDealSize": {"value": 100000}}))
    yw.run(1000, t("2026-11-16 02:16"))                       # a 100,000-dollar minimum is far too big: refused
    assert cap.calls == [] and "problem" in cap.sent[-1][0]
