"""Capital.com demo (desk/capital.py + the 'capital' broker in desk/autotrade.py), without network."""
import pandas as pd
import pytest

from desk import autotrade as at
from desk import capital as cap

NY = "America/New_York"


def t(hm, day="2026-11-04"):
    return pd.Timestamp(f"{day} {hm}", tz=NY)


class Resp:
    def __init__(self, data, headers=None):
        self.data, self.headers = data, headers or {}

    def json(self):
        return self.data


def test_client_maps_capital_fields_to_ig_shapes(monkeypatch):
    sent = []

    def req(self, method, path, body=None, params=None):
        sent.append((method, path, body))
        if path == "/session" and method == "POST":
            return Resp({"currentAccountId": "A1"}, {"CST": "c", "X-SECURITY-TOKEN": "x"})
        if path == "/accounts":
            return Resp({"accounts": [{"accountId": "A1", "currency": "EUR", "preferred": True,
                                       "balance": {"balance": 1010.0, "deposit": 1000.0, "profitLoss": 10.0}}]})
        if path == "/markets/US100":
            return Resp({"snapshot": {"bid": 27000.0, "offer": 27001.0, "marketStatus": "TRADEABLE"},
                         "dealingRules": {"minDealSize": {"unit": "POINTS", "value": 0.001},
                                          "minStopOrProfitDistance": {"unit": "PERCENTAGE", "value": 0.1}}})
        if path == "/positions" and method == "POST":
            return Resp({"dealReference": "o_1"})
        if path == "/confirms/o_1":
            return Resp({"dealStatus": "ACCEPTED", "level": 27001.5, "dealId": "order1",
                         "affectedDeals": [{"dealId": "pos1", "status": "OPENED"}]})
        if path == "/positions" and method == "GET":
            return Resp({"positions": [{"market": {"epic": "US100"},
                                        "position": {"dealId": "pos1", "size": 0.037, "profitLevel": 27500.0}}]})
        raise AssertionError(path)

    monkeypatch.setattr(cap.Capital, "req", req)
    c = cap.Capital("k", "u", "p")
    s = c.login()
    assert s["accountInfo"]["balance"] + s["accountInfo"]["profitLoss"] == 1010.0
    m = c.market("US100")
    assert m["dealingRules"]["minNormalStopOrLimitDistance"]["value"] == pytest.approx(27.0, abs=0.01)   # 0.1% of 27,000
    o = c.open("US100", "BUY", 0.037, 27006.0, 26900.0, None, "N1")
    assert o["dealStatus"] == "ACCEPTED" and o["dealId"] == "pos1" and o["level"] == 27001.5
    body = [b for (mth, p, b) in sent if p == "/positions" and mth == "POST"][0]
    assert body["stopLevel"] == 26900.0 and "profitLevel" not in body and body["size"] == 0.037
    assert c.positions()[0]["position"]["limitLevel"] == 27500.0
    assert "demo-api-capital" in cap.BASE


class FakeCap:
    book, calls, hedged = {}, [], []

    def __init__(self):
        pass

    def login(self):
        return {"accountInfo": {"balance": 1000, "profitLoss": 0}}

    def ensure_hedging(self):
        FakeCap.hedged.append(1)
        return False

    def market(self, epic):
        return {"snapshot": {"bid": 27000.0, "offer": 27001.0, "marketStatus": "TRADEABLE"},
                "dealingRules": {"minDealSize": {"value": 0.001}, "minNormalStopOrLimitDistance": {"value": 27.0}}}

    def positions(self):
        return [{"market": {"epic": cap.EPIC}, "position": dict(p)} for p in self.book.values()]

    def open(self, epic, direction, size, level, stop, limit, ref):
        d = f"C{len(self.calls)}"
        self.calls.append(("open", epic, direction, size, stop, limit))
        self.book[d] = {"dealId": d, "size": size, "level": 27001.0, "stopLevel": stop, "limitLevel": limit}
        return {"dealStatus": "ACCEPTED", "level": 27001.0, "dealId": d}

    def close(self, deal, direction, size, level, ref):
        self.calls.append(("close", deal))
        self.book.pop(deal)
        return {"dealStatus": "ACCEPTED", "level": 27050.0}

    def update_stop(self, deal, stop, limit):
        self.calls.append(("stop", deal, stop))


@pytest.fixture
def capital(monkeypatch, tmp_path):
    FakeCap.book, FakeCap.calls, FakeCap.hedged = {}, [], []
    monkeypatch.setattr(cap, "Capital", FakeCap)
    monkeypatch.setattr(at, "CAP_STATE", str(tmp_path / "capital-{mode}.json"))
    msgs = []
    monkeypatch.setattr(at, "notify", lambda title, body, quiet=False, *a: msgs.append((title, quiet)))
    for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID"):
        monkeypatch.delenv(k, raising=False)                       # IG off: only Capital.com trades here
    for k in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("CAPITAL_TRADE", "demo")
    at.enabled = True
    FakeCap.msgs = msgs
    return FakeCap


def test_capital_trades_the_plans_size_and_closes(capital):
    w = dict(ref="N2611041030", side=1, entry=26700.0, t_in=t("10:30"), stop=26700 * 0.997, target=None)
    at.want("N", [w], 26710.0)                         # Capital mid 27000.5 vs index 26710 -> shift +290.5
    at.flush(1000, t("10:36"))
    op = capital.calls[0]
    assert op[:4] == ("open", cap.EPIC, "BUY", 0.037)  # EUR 1,000 / 27,001 = 0.037: the plan's size, not a minimum
    assert op[4] == round(26700 * 0.997 + 290.5, 1)
    assert capital.hedged and at.load("demo", "capital")["trades"]["N2611041030"]["scale"] == pytest.approx(1.0, abs=0.01)
    at.want("N", [], 26750.0)
    at.flush(1000, t("11:06"))
    assert capital.calls[-1] == ("close", "C0")
    assert any("Capital.com demo" in title for title, _ in capital.msgs)
    assert any(title.startswith(("+EUR", "-EUR")) and "CLOSED" in title for title, _ in capital.msgs)   # money first


def test_unknown_position_blocks_new_orders(capital):
    FakeCap.book["X9"] = {"dealId": "X9", "size": 0.05, "level": 27000.0, "stopLevel": None, "limitLevel": None}
    at.want("N", [dict(ref="N1", side=1, entry=27000.0, t_in=t("10:30"), stop=26919.0, target=None)], 27000.0)
    at.flush(1000, t("10:31"))
    assert capital.calls == []


def test_capital_dry_run_is_quiet_and_sends_nothing(capital, monkeypatch):
    monkeypatch.setenv("CAPITAL_TRADE", "dry")
    at.want("N", [dict(ref="N1", side=1, entry=27000.0, t_in=t("10:30"), stop=26919.0, target=None)], 27000.0)
    at.flush(1000, t("10:31"))
    assert capital.calls == [] and capital.msgs and all(q for _, q in capital.msgs)


def test_capital_off_without_secrets(monkeypatch):
    for k in ("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    assert at.mode("capital") == "off"


class FakeLive(FakeCap):
    def __init__(self, live=False):
        self.live, self.base = live, cap.LIVE_BASE if live else cap.BASE


@pytest.fixture
def real(capital, monkeypatch, tmp_path):
    monkeypatch.setattr(cap, "Capital", FakeLive)
    monkeypatch.setenv("CAPITAL_TRADE", "off")                   # demo account off: only the real one here
    monkeypatch.setitem(at.BROKERS, "capital-live", dict(at.BROKERS["capital-live"],
                                                         state=str(tmp_path / "live-{mode}.json"), month_x=1.0))
    for k in ("CAPITAL_LIVE_API_KEY", "CAPITAL_LIVE_IDENTIFIER", "CAPITAL_LIVE_PASSWORD"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("CAPITAL_LIVE_TRADE", "live")
    return capital


def _noise():
    at.want("N", [dict(ref="N2611041030", side=1, entry=26700.0, t_in=t("10:30"), stop=26700 * 0.997, target=None)], 26710.0)


def test_real_money_needs_both_switches(real, monkeypatch):
    monkeypatch.delenv("CAPITAL_LIVE_GO", raising=False)
    assert at.mode("capital-live") == "dry"                       # CAPITAL_LIVE_TRADE=live alone is a dry run
    monkeypatch.setenv("CAPITAL_LIVE_GO", "yes")
    monkeypatch.setenv("CAPITAL_LIVE_TRADE", "demo")
    assert at.mode("capital-live") == "dry"                       # 'demo' never means real orders
    monkeypatch.setenv("CAPITAL_LIVE_TRADE", "live")
    assert at.mode("capital-live") == "live" and at.mode("capital") == "off"
    monkeypatch.delenv("CAPITAL_LIVE_API_KEY")
    assert at.mode("capital-live") == "off"


def test_real_money_dry_sends_nothing_and_live_sizes_off_the_month_balance(real, monkeypatch):
    _noise()
    at.flush(5000, t("10:36"))                                    # no CAPITAL_LIVE_GO: dry
    assert real.calls == []
    monkeypatch.setenv("CAPITAL_LIVE_GO", "yes")
    _noise()
    at.flush(5000, t("10:36"))                                    # the plan's practice stake (5000) is not used
    assert real.calls[0][:4] == ("open", cap.EPIC, "BUY", 0.037)  # 1.0 x the month's balance 1,000 / 27,001
    assert at.load("live", "capital-live")["month"] == {"2611": 1000.0}


def test_demo_and_real_clients_never_cross(monkeypatch):
    B = at.BROKERS["capital-live"]
    with pytest.raises(RuntimeError):
        at.safe(B, cap, FakeLive(live=False), "live")             # real broker on the demo server
    with pytest.raises(RuntimeError):
        at.safe(at.BROKERS["capital"], cap, FakeLive(live=True), "demo")   # demo broker with a live client
    with pytest.raises(RuntimeError):
        at.safe(B, cap, FakeLive(live=True), "demo")
    at.safe(B, cap, FakeLive(live=True), "live")
    monkeypatch.setenv("CAPITAL_LIVE_API_KEY", "k")
    monkeypatch.setenv("CAPITAL_LIVE_IDENTIFIER", "u")
    monkeypatch.setenv("CAPITAL_LIVE_PASSWORD", "p")
    c = cap.Capital(live=True)
    assert c.base == cap.LIVE_BASE and c.key == "k" and cap.Capital(key="a", user="b", password="c").base == cap.BASE
