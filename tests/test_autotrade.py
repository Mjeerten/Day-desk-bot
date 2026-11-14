import pandas as pd
import pytest

from desk import autotrade as at
from desk import bot

NY = "America/New_York"


class FakeIG:
    min_size = 0.01
    book = {}
    calls = []

    def __init__(self):
        pass

    def login(self):
        return {"accountInfo": {"balance": 10000, "profitLoss": 0}}

    def market(self, epic):
        return {"snapshot": {"bid": 28300.0, "offer": 28301.0, "marketStatus": "TRADEABLE"},
                "dealingRules": {"minDealSize": {"value": FakeIG.min_size}, "minNormalStopOrLimitDistance": {"value": 4}}}

    def positions(self):
        return [{"market": {"epic": at.EPIC}, "position": dict(p)} for p in self.book.values()]

    def open(self, epic, direction, size, level, stop, limit, ref):
        d = f"D{len(self.calls)}"
        self.calls.append(("open", direction, size, stop, limit, ref))
        self.book[d] = {"dealId": d, "size": size, "level": 28301.0, "stopLevel": stop, "limitLevel": limit,
                        "dealReference": ref}
        return {"dealStatus": "ACCEPTED", "level": 28301.0, "dealId": d}

    def close(self, deal, direction, size, level, ref):
        self.calls.append(("close", deal, direction, size))
        self.book.pop(deal)
        return {"dealStatus": "ACCEPTED", "level": 28350.0}

    def update_stop(self, deal, stop, limit):
        self.calls.append(("stop", deal, stop))
        self.book[deal]["stopLevel"] = stop


@pytest.fixture
def ig(monkeypatch, tmp_path):
    import desk.ig
    FakeIG.book, FakeIG.calls = {}, []
    monkeypatch.setattr(desk.ig, "IG", FakeIG)
    monkeypatch.setattr(at, "STATE", str(tmp_path / "ig-{mode}.json"))
    monkeypatch.setattr(at, "notify", lambda *a: None)
    for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setenv("AUTO_TRADE", "demo")
    at.enabled = True
    return FakeIG


def t(hm, day="2026-11-04"):
    return pd.Timestamp(f"{day} {hm}", tz=NY)


def test_open_then_close_noise_with_levels_shifted(ig):
    w = dict(ref="N2611041030", side=1, entry=28000.0, t_in=t("10:30"), stop=28000 * 0.997, target=None)
    at.want("N", [w], 28010.0)                      # IG mid 28300.5 vs index 28010 -> shift +290.5
    at.flush(1000, t("10:36"))
    op = ig.calls[0]
    assert op[0] == "open" and op[1] == "BUY" and op[2] == 0.03          # EUR 1,000 / 28,301 -> 0.03 contracts
    assert op[3] == round(28000 * 0.997 + 290.5, 1) and op[4] is None
    at.want("N", [w], 28010.0)                      # same signal next run: nothing new
    at.flush(1000, t("10:51"))
    assert len(ig.calls) == 1
    at.want("N", [], 28050.0)                       # bot says the trade is over -> close
    at.flush(1000, t("11:06"))
    assert ig.calls[-1][:3] == ("close", "D0", "SELL")


def test_too_late_and_other_strategies_untouched(ig):
    at.want("S", [dict(ref="S2611040945", side=-1, entry=28000.0, t_in=t("09:45"), stop=28040.0, target=27900.0)], 28000.0)
    at.flush(1000, t("10:31"))                       # 46 min late (limit 30): never opened
    assert ig.calls == []
    at.want("I", [dict(ref="I2611031545", side=1, entry=28000.0, t_in=t("15:45", "2026-11-03"), stop=None, target=None)],
            28000.0)
    at.flush(1000, t("15:52", "2026-11-03"))
    assert ig.calls[0][0] == "open"
    at.want("N", [], 28000.0)                        # a run that only evaluated the noise band leaves IBS alone
    at.flush(1000, t("10:00"))
    assert len(ig.calls) == 1


def test_trailing_stop_moves(ig):
    w = dict(ref="P2611040935", side=-1, entry=28000.0, t_in=t("09:35"), stop=28336.0, target=None)
    at.want("P", [w], 28000.0)
    at.flush(1000, t("09:46"))
    at.want("P", [dict(w, stop=28200.0)], 27900.0)
    at.flush(1000, t("10:01"))
    assert ig.calls[-1][0] == "stop" and ig.calls[-1][2] == round(28200.0 + 300.5, 1)


def test_dry_mode_sends_nothing(ig, monkeypatch):
    monkeypatch.setenv("AUTO_TRADE", "dry")
    at.want("N", [dict(ref="N1", side=1, entry=28000.0, t_in=t("10:30"), stop=27916.0, target=None)], 28000.0)
    at.flush(1000, t("10:31"))
    assert ig.calls == []


def test_wants_from_events():
    ts = pd.Timestamp("2026-11-04 10:15", tz=NY)
    ev = [dict(kind="noise_entry", ts=ts, px=28000.0), dict(kind="sweep_entry", ts=ts, px=28000.0,
          trade=dict(side=-1, entry=28000.0, stop=28040.0, target=27900.0)),
          dict(kind="sweep_stop", ts=ts + pd.Timedelta(minutes=30), px=28040.0, trade={})]
    n, s, i = bot.wants(ev, ([], None), ts.date())
    assert n[0]["ref"] == "N2611041030" and n[0]["stop"] == pytest.approx(27916.0) and s == [] and i == []
    held = (pd.Timestamp("2026-11-03").date(), 27950.0)
    assert bot.wants([], ([], held), ts.date())[2][0]["ref"] == "I2611031545"
    assert bot.wants([], ([dict(kind="ibs_exit")], held), ts.date())[2] == []


def test_ig_minimum_bigger_than_plan_still_trades(ig, monkeypatch):
    """IG demo minimum 0.2 contracts (~EUR 5,700) > EUR 1,000 plan: trade the minimum, record the scale."""
    monkeypatch.setattr(FakeIG, "min_size", 0.2)
    w = dict(ref="P2611040935", side=-1, entry=28417.0, t_in=t("09:35"), stop=28758.0, target=None)
    at.want("P", [w], 28417.0)
    at.flush(1000, t("09:46"))
    op = ig.calls[0]
    assert op[0] == "open" and op[1] == "SELL" and op[2] == 0.2
    tr = at.load("demo")["trades"]["P2611040935"]
    assert tr["status"] == "open" and 0.17 < tr["scale"] < 0.18


def test_wrong_side_of_market_retries_once_at_fresh_price(ig, monkeypatch):
    real, seen = FakeIG.open, []

    def jumpy(self, epic, direction, size, level, stop, limit, ref):
        seen.append(level)
        if len(seen) == 1:                          # price ran past the limit while the order travelled
            return {"dealStatus": "REJECTED", "reason": "LIMIT_ORDER_WRONG_SIDE_OF_MARKET"}
        return real(self, epic, direction, size, level, stop, limit, ref)

    monkeypatch.setattr(FakeIG, "open", jumpy)
    w = dict(ref="N2611061000", side=1, entry=28000.0, t_in=t("10:00", "2026-11-06"), stop=28000 * 0.997, target=None)
    at.want("N", [w], 28000.0)
    at.flush(1000, t("10:01", "2026-11-06"))
    assert len(seen) == 2 and ig.calls[0][0] == "open"
    assert at.load("demo")["trades"]["N2611061000"]["status"] == "open"
