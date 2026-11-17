"""cTrader / FTMO broker client (desk/ctrader.py) against a fake cTrader server, and autotrade's 'ctrader' broker."""
import json

import pandas as pd
import pytest

from desk import autotrade, ctrader
from desk.ctrader import PT, P5


class FakeWS:
    """Answers the JSON messages like cTrader's Open API (only the fields the client reads)."""

    def __init__(self, accounts, positions=None, spots=True, reject=False):
        self.out, self.sent = [], []
        self.accounts, self.pos, self.spots, self.reject = accounts, positions or [], spots, reject

    def send(self, raw):
        m = json.loads(raw)
        self.sent.append(m)
        pt, p, mid = m["payloadType"], m["payload"], m["clientMsgId"]
        r = lambda t, pl: self.out.append(json.dumps({"clientMsgId": mid, "payloadType": t, "payload": pl}))
        if pt == PT["app_auth"]:
            r(PT["app_auth_res"], {})
        elif pt == PT["accounts"]:
            r(PT["accounts_res"], {"ctidTraderAccount": self.accounts})
        elif pt == PT["acc_auth"]:
            r(PT["acc_auth_res"], {})
        elif pt == PT["trader"]:
            r(PT["trader_res"], {"trader": {"balance": 1000000, "moneyDigits": 2}})
        elif pt == PT["pnl"]:
            r(PT["pnl_res"], {"positionUnrealizedPnL": [{"netUnrealizedPnL": -2550}], "moneyDigits": 2})
        elif pt == PT["symbols"]:
            r(PT["symbols_res"], {"symbol": [{"symbolId": 275, "symbolName": "US100.cash"}, {"symbolId": 4, "symbolName": "USDJPY"}]})
        elif pt == PT["symbol_by_id"]:
            r(PT["symbol_by_id_res"], {"symbol": [{"symbolId": 275, "digits": 2, "minVolume": 1, "stepVolume": 1,
                                                   "slDistance": 0, "distanceSetIn": 1}]})
        elif pt == PT["sub_spots"]:
            r(PT["sub_spots_res"], {})
            if self.spots:
                self.out.append(json.dumps({"payloadType": PT["spot"], "payload": {"symbolId": 275, "bid": 2714080 * 1000,
                                                                                   "ask": 2714480 * 1000}}))
        elif pt == PT["trendbars"]:
            r(PT["trendbars_res"], {"trendbar": [{"low": 2700000000, "deltaClose": 50000}]})
        elif pt == PT["reconcile"]:
            r(PT["reconcile_res"], {"position": self.pos})
        elif pt in (PT["new_order"], PT["close_pos"]):
            if self.reject:
                r(PT["order_error"], {"errorCode": "NOT_ENOUGH_MONEY", "description": "x"})
            else:
                r(PT["execution"], {"executionType": 2})
                r(PT["execution"], {"executionType": 3, "deal": {"executionPrice": 27145.1},
                                    "position": {"positionId": 777, "price": 27145.1}})

    def recv(self):
        if not self.out:
            raise ctrader.websocket.WebSocketTimeoutException("no data")
        return self.out.pop(0)

    def settimeout(self, t):
        pass

    def close(self):
        pass


@pytest.fixture
def env(monkeypatch, tmp_path):
    for k in ("CTRADER_CLIENT_ID", "CTRADER_CLIENT_SECRET", "CTRADER_ACCESS_TOKEN", "CTRADER_REFRESH_TOKEN"):
        monkeypatch.setenv(k, "x")
    monkeypatch.setattr(ctrader, "TOKEN_FILE", str(tmp_path / "tok.json"))
    return monkeypatch


def fake(env, **kw):
    ws = FakeWS(**kw)
    env.setattr(ctrader.websocket, "create_connection", lambda url, timeout=20: ws)
    return ws


def test_refuses_a_real_money_account(env):
    fake(env, accounts=[{"ctidTraderAccountId": 1, "isLive": True, "brokerTitleShort": "IC Markets"}])
    with pytest.raises(RuntimeError, match="refusing"):
        ctrader.FTMO().login()


def test_login_market_positions_open(env):
    ws = fake(env, accounts=[{"ctidTraderAccountId": 9592, "isLive": True, "brokerTitleShort": "FTMO Platform"}],
              positions=[{"positionId": 55, "price": 27000.0, "stopLoss": 26900.0,
                          "tradeData": {"symbolId": 275, "volume": 12, "tradeSide": 1, "label": "N2611171000"}}])
    f = ctrader.FTMO()
    info = f.login()["accountInfo"]
    assert info == {"balance": 10000.0, "profitLoss": -25.5}
    mk = f.market("US100.cash")
    assert mk["snapshot"] == {"bid": 27140.8, "offer": 27144.8, "marketStatus": "TRADEABLE"}
    assert mk["dealingRules"]["minDealSize"]["value"] == 0.01
    p = f.positions()[0]
    assert p["market"]["epic"] == "US100.cash"
    assert p["position"]["dealId"] == "55" and p["position"]["size"] == 0.12
    assert p["position"]["dealReference"] == "N2611171000"
    c = f.open("US100.cash", "BUY", 0.12, 27144.8 + 5, 27060.0, 27400.0, "N2611171000-12345")
    assert c == {"dealStatus": "ACCEPTED", "level": 27145.1, "dealId": "777"}
    o = ws.sent[-1]["payload"]
    assert o["volume"] == 12 and o["tradeSide"] == 1 and o["label"] == "N2611171000"
    assert o["relativeStopLoss"] == round(84.8 * P5) and o["relativeTakeProfit"] == round(255.2 * P5)


def test_closed_market_uses_last_close_and_rejects_are_reported(env):
    fake(env, accounts=[{"ctidTraderAccountId": 3, "isLive": False, "brokerTitleShort": "Spotware"}], spots=False,
         reject=True)
    f = ctrader.FTMO()
    f.login()
    mk = f.market("US100.cash")
    assert mk["snapshot"]["marketStatus"] == "CLOSED" and mk["snapshot"]["bid"] == 27000.5
    c = f.open("US100.cash", "SELL", 0.05, 27000.0, None, None, "S1")
    assert c["dealStatus"] == "REJECTED" and "NOT_ENOUGH_MONEY" in c["reason"]


def test_autotrade_ctrader_dry_sizes_off_the_start_balance(env, tmp_path):
    fake(env, accounts=[{"ctidTraderAccountId": 9592, "isLive": True, "brokerTitleShort": "FTMO Platform"}])
    env.setattr(autotrade, "CT_STATE", str(tmp_path / "ct-{mode}.json"))
    env.setattr(autotrade, "_wants", {})
    now = pd.Timestamp("2026-11-18 10:05", tz="America/New_York")
    autotrade.want("N", [dict(ref="N2611181000", side=1, entry=25000.0, t_in=now - pd.Timedelta(minutes=5),
                              stop=24925.0, target=None)], scale_px=25010.0)
    autotrade._flush("dry", 1000.0, now, "ctrader")
    st = autotrade.load("dry", "ctrader")
    t = st["trades"]["N2611181000"]
    # stake 0.33 x 10,000 - 25.50 open result = 3,291.6 -> 0.12 contracts at 27,144.8
    assert st["start_balance"] == pytest.approx(9974.5)
    assert t["status"] == "open" and t["size"] == 0.12 and t["deal"] is None
    assert t["stop"] == pytest.approx(24925.0 + 27142.8 - 25010.0, abs=0.1)
