"""cTrader Open API client (Spotware), for FTMO's cTrader accounts and a free cTrader demo (round 46, AUTOMATION.md).
--check = read-only: log in with the app, list the cTrader accounts of the token, and per account show the balance and
the Nasdaq-100 / USD/JPY contracts with their size rules.
`FTMO` = the broker class desk/autotrade.py drives (broker 'ctrader', GitHub variable CTRADER_TRADE dry | demo | off),
with the same methods and return shapes as desk/capital.py (login, market, positions, open, close, update_stop).
Safety: FTMO's accounts sit on cTrader's LIVE server but hold FTMO's virtual money, so the lock is the account itself:
orders only on an account whose broker is FTMO or that is a demo account (login() refuses anything else).
Size: desk/autotrade.py trades CTRADER_SIZE (default 0.33, round 46) x the account's starting balance as the stake.
Tokens: a refreshed access / refresh token pair is kept in .daybot/ctrader-token.json (bot cache) and used before the
secrets; if the refresh fails too, the bot messages: run the Sandbox again and replace the two token secrets.

Protocol: JSON over a WebSocket (wss://demo.ctraderapi.com:5036 for demo accounts, live.ctraderapi.com:5036 for live
ones; FTMO accounts may show as either). Each message = {"clientMsgId", "payloadType", "payload"}.
Secrets (GitHub Actions of the public bot repo, never in the repo or chat): CTRADER_CLIENT_ID, CTRADER_CLIENT_SECRET
(openapi.ctrader.com -> Applications -> Credentials), CTRADER_ACCESS_TOKEN, CTRADER_REFRESH_TOKEN (app -> Sandbox,
scope trading). Access tokens expire (about 30 days); the refresh token gives a new pair.
The public Actions log is readable by anyone: account numbers are shown as their last 3 digits only.

Run: python -m desk.ctrader --check
"""
import argparse
import json
import os
import sys
import time

import websocket   # websocket-client

HOSTS = {False: "wss://demo.ctraderapi.com:5036", True: "wss://live.ctraderapi.com:5036"}
PT = dict(app_auth=2100, app_auth_res=2101, acc_auth=2102, acc_auth_res=2103, symbols=2114, symbols_res=2115,
          symbol_by_id=2116, symbol_by_id_res=2117, trader=2121, trader_res=2122, error=2142,
          accounts=2149, accounts_res=2150, refresh=2173, refresh_res=2174, heartbeat=51, common_error=50,
          new_order=2106, amend_sltp=2110, close_pos=2111, reconcile=2124, reconcile_res=2125, execution=2126,
          sub_spots=2127, sub_spots_res=2128, unsub_spots=2129, unsub_spots_res=2130, spot=2131, order_error=2132,
          trendbars=2137, trendbars_res=2138, pnl=2187, pnl_res=2188)
FILLED, REJECTED = 3, 7               # ProtoOAExecutionType
P5 = 100000                           # prices in cTrader messages are integers in 1/100000
TOKEN_FILE = os.environ.get("CTRADER_TOKEN_FILE") or ".daybot/ctrader-token.json"
BASE = "cTrader Open API: FTMO or demo accounts only"     # autotrade's safety check looks for demo_host in BASE
EPIC = os.environ.get("CTRADER_EPIC") or "US100.cash"
NASDAQ = ("US100", "NAS100", "USTEC", "US TECH", "NASDAQ", "NDX", "US 100")
YEN = ("USDJPY",)


def mask(n):
    return "..." + str(n)[-3:]


class CTrader:
    def __init__(self, live=False, timeout=20):
        self.cid = os.environ["CTRADER_CLIENT_ID"].strip()
        self.secret = os.environ["CTRADER_CLIENT_SECRET"].strip()
        self.token = os.environ["CTRADER_ACCESS_TOKEN"].strip()
        self.refresh_token = os.environ.get("CTRADER_REFRESH_TOKEN", "").strip()
        self.ws = websocket.create_connection(HOSTS[live], timeout=timeout)
        self.n = 0

    def send(self, ptype, payload):
        self.n += 1
        mid = f"m{self.n}"
        self.ws.send(json.dumps({"clientMsgId": mid, "payloadType": ptype, "payload": payload}))
        return mid

    def wait(self, want, mid=None, ok=None, secs=30):
        """Next message of type `want` (and for which ok(payload) is true). cTrader errors raise RuntimeError."""
        t0 = time.time()
        while time.time() - t0 < secs:
            try:
                self.ws.settimeout(max(1.0, secs - (time.time() - t0)))
                m = json.loads(self.ws.recv())
            except websocket.WebSocketTimeoutException:
                break
            pt, p = m.get("payloadType"), m.get("payload", {})
            if pt == PT["heartbeat"]:
                continue
            if pt in (PT["error"], PT["common_error"], PT["order_error"]) and (mid is None or m.get("clientMsgId", mid) == mid):
                raise RuntimeError(f"{p.get('errorCode')}: {p.get('description', '')}")
            if pt == want and (mid is None or m.get("clientMsgId", mid) == mid) and (ok is None or ok(p)):
                return p
        raise TimeoutError(f"no answer of type {want}")

    def ask(self, ptype, payload, want, ok=None, secs=30):
        return self.wait(want, self.send(ptype, payload), ok, secs)

    def app_auth(self):
        return self.ask(PT["app_auth"], {"clientId": self.cid, "clientSecret": self.secret}, PT["app_auth_res"])

    def accounts(self):
        return self.ask(PT["accounts"], {"accessToken": self.token}, PT["accounts_res"]).get("ctidTraderAccount", [])

    def refresh(self):
        return self.ask(PT["refresh"], {"refreshToken": self.refresh_token}, PT["refresh_res"])

    def acc_auth(self, acc):
        return self.ask(PT["acc_auth"], {"ctidTraderAccountId": acc, "accessToken": self.token}, PT["acc_auth_res"])

    def trader(self, acc):
        return self.ask(PT["trader"], {"ctidTraderAccountId": acc}, PT["trader_res"]).get("trader", {})

    def symbols(self, acc):
        return self.ask(PT["symbols"], {"ctidTraderAccountId": acc}, PT["symbols_res"]).get("symbol", [])

    def symbol_rules(self, acc, ids):
        return self.ask(PT["symbol_by_id"], {"ctidTraderAccountId": acc, "symbolId": ids},
                        PT["symbol_by_id_res"]).get("symbol", [])

    def close(self):
        try:
            self.ws.close()
        except Exception:
            pass


def _load_tokens():
    try:
        with open(TOKEN_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


class FTMO:
    """The cTrader account (FTMO, or a cTrader demo) in the shape desk/autotrade.py expects from a broker client."""

    def __init__(self):
        self.c = None
        self.acc = None
        self.sym = {}

    # --- connection -------------------------------------------------------------------------------------------------
    def _connect(self):
        tok = _load_tokens()
        os.environ["CTRADER_ACCESS_TOKEN"] = tok.get("access") or os.environ["CTRADER_ACCESS_TOKEN"]
        os.environ["CTRADER_REFRESH_TOKEN"] = tok.get("refresh") or os.environ.get("CTRADER_REFRESH_TOKEN", "")
        probe = CTrader(live=False)
        probe.app_auth()
        try:
            accs = probe.accounts()
        except RuntimeError:
            r = probe.refresh()           # access token expired: new pair, kept in the bot cache
            os.makedirs(os.path.dirname(TOKEN_FILE) or ".", exist_ok=True)
            with open(TOKEN_FILE, "w", encoding="utf-8") as f:
                json.dump({"access": r["accessToken"], "refresh": r["refreshToken"], "at": time.time()}, f)
            os.environ["CTRADER_ACCESS_TOKEN"], os.environ["CTRADER_REFRESH_TOKEN"] = r["accessToken"], r["refreshToken"]
            print("[ctrader] access token renewed (kept in the bot cache)")
            probe.token = r["accessToken"]
            accs = probe.accounts()
        probe.close()
        want = (os.environ.get("CTRADER_ACCOUNT") or "").strip()
        ok = [a for a in accs if "FTMO" in str(a.get("brokerTitleShort", "")).upper() or not a.get("isLive")]
        if want:
            ok = [a for a in ok if str(a["ctidTraderAccountId"]).endswith(want) or str(a.get("traderLogin", "")).endswith(want)]
        if not ok:
            raise RuntimeError("no FTMO or demo account on this token: refusing to trade")
        a = ok[-1]                        # the newest (a bought challenge comes after the trial)
        self.acc = a["ctidTraderAccountId"]
        self.c = CTrader(live=bool(a.get("isLive")))
        self.c.app_auth()
        self.c.acc_auth(self.acc)

    def _symbol(self, name):
        if name not in self.sym:
            hit = [x for x in self.c.symbols(self.acc) if x.get("symbolName") == name]
            if not hit:
                raise RuntimeError(f"symbol {name} not on this account")
            r = self.c.symbol_rules(self.acc, [hit[0]["symbolId"]])[0]
            self.sym[name] = r
            self.sym[r["symbolId"]] = name
        return self.sym[name]

    # --- what autotrade calls ---------------------------------------------------------------------------------------
    def login(self):
        if self.c is None:
            self._connect()
        t = self.c.trader(self.acc)
        dig = 10 ** t.get("moneyDigits", 2)
        pnl = 0.0
        try:
            r = self.c.ask(PT["pnl"], {"ctidTraderAccountId": self.acc}, PT["pnl_res"])
            pnl = sum(x.get("netUnrealizedPnL", 0) for x in r.get("positionUnrealizedPnL", [])) / 10 ** r.get("moneyDigits", 2)
        except Exception as e:  # noqa
            print(f"[ctrader] open result not read ({e}); equity = balance")
        return {"accountInfo": {"balance": t.get("balance", 0) / dig, "profitLoss": pnl}}

    def market(self, epic):
        r = self._symbol(epic)
        sid = r["symbolId"]
        bid = ask = None
        try:
            self.c.ask(PT["sub_spots"], {"ctidTraderAccountId": self.acc, "symbolId": [sid]}, PT["sub_spots_res"])
            t0 = time.time()
            while time.time() - t0 < 8 and (bid is None or ask is None):
                p = self.c.wait(PT["spot"], ok=lambda p: p.get("symbolId") == sid, secs=8)
                bid = p["bid"] / P5 if p.get("bid") else bid
                ask = p["ask"] / P5 if p.get("ask") else ask
        except TimeoutError:
            pass
        finally:
            try:
                self.c.send(PT["unsub_spots"], {"ctidTraderAccountId": self.acc, "symbolId": [sid]})
            except Exception:  # noqa
                pass
        status = "TRADEABLE" if bid and ask and r.get("tradingMode", 0) == 0 else "CLOSED"
        if not (bid and ask):             # no live quotes (market shut): last 1-minute close
            now = int(time.time() * 1000)
            tb = self.c.ask(PT["trendbars"], {"ctidTraderAccountId": self.acc, "symbolId": sid, "period": 1,
                                              "fromTimestamp": now - 4 * 86400000, "toTimestamp": now},
                            PT["trendbars_res"]).get("trendbar", [])
            last = (tb[-1]["low"] + tb[-1].get("deltaClose", 0)) / P5 if tb else 0.0
            bid = ask = last
        step = r.get("stepVolume", 1) / 100
        dist = max(r.get("slDistance", 0), r.get("tpDistance", 0))
        dist = dist / 10 ** r.get("digits", 2) if r.get("distanceSetIn", 1) == 1 else dist / 100 * bid / 100
        return {"snapshot": {"bid": bid, "offer": ask, "marketStatus": status},
                "dealingRules": {"minDealSize": {"value": max(step, r.get("minVolume", 1) / 100)},
                                 "minNormalStopOrLimitDistance": {"value": dist}}}

    def positions(self):
        r = self.c.ask(PT["reconcile"], {"ctidTraderAccountId": self.acc}, PT["reconcile_res"])
        out = []
        for p in r.get("position", []):
            td = p.get("tradeData", {})
            name = self.sym.get(td.get("symbolId"))
            if name is None:              # a symbol we never looked up (not ours)
                name = f"id{td.get('symbolId')}"
            out.append({"position": {"dealId": str(p["positionId"]), "dealReference": td.get("label", ""),
                                     "size": td.get("volume", 0) / 100,
                                     "direction": "BUY" if td.get("tradeSide") == 1 else "SELL",
                                     "level": p.get("price"), "stopLevel": p.get("stopLoss"),
                                     "limitLevel": p.get("takeProfit")},
                        "market": {"epic": name}})
        return out

    def _exec(self, ptype, payload):
        try:
            e = self.c.ask(ptype, payload, PT["execution"], ok=lambda p: p.get("executionType") in (FILLED, REJECTED))
        except RuntimeError as err:
            return {"dealStatus": "REJECTED", "reason": str(err)}
        if e.get("executionType") == REJECTED:
            return {"dealStatus": "REJECTED", "reason": e.get("errorCode", "rejected")}
        d = e.get("deal", {})
        pos = e.get("position", {})
        return {"dealStatus": "ACCEPTED", "level": d.get("executionPrice") or pos.get("price"),
                "dealId": str(pos.get("positionId") or d.get("positionId"))}

    def open(self, epic, direction, size, level, stop, target, ref):
        """Market order (cTrader has no fill-or-kill limit like IG; `level` is ignored), stop / target relative to the
        fill so the broker holds them from the first moment."""
        r = self._symbol(epic)
        px = level                        # autotrade's limit = quote +- SLIP: distances from the quote itself
        q = px - 5.0 if direction == "BUY" else px + 5.0
        p = {"ctidTraderAccountId": self.acc, "symbolId": r["symbolId"], "orderType": 1,
             "tradeSide": 1 if direction == "BUY" else 2, "volume": int(round(size * 100)), "label": ref.split("-")[0],
             "comment": "day desk"}
        if stop is not None:
            p["relativeStopLoss"] = int(round(abs(q - stop) * P5))
        if target is not None:
            p["relativeTakeProfit"] = int(round(abs(target - q) * P5))
        return self._exec(PT["new_order"], p)

    def close(self, deal, direction, size, level, ref):
        return self._exec(PT["close_pos"], {"ctidTraderAccountId": self.acc, "positionId": int(deal),
                                            "volume": int(round(size * 100))})

    def update_stop(self, deal, stop, limit=None):
        p = {"ctidTraderAccountId": self.acc, "positionId": int(deal), "stopLoss": stop}
        if limit:
            p["takeProfit"] = float(limit)
        self.c.ask(PT["amend_sltp"], p, PT["execution"])


def check():
    c = CTrader(live=False)
    c.app_auth()
    print("[ctrader] app login OK")
    try:
        accs = c.accounts()
    except RuntimeError as e:
        print(f"[ctrader] access token refused ({e}); trying the refresh token")
        r = c.refresh()
        print(f"[ctrader] refresh OK: new tokens issued (expires in {r.get('expiresIn')} s). Put the NEW access and "
              "refresh tokens in the secrets (Sandbox again); the old refresh token no longer works.")
        return 1
    c.close()
    print(f"[ctrader] {len(accs)} account(s) on this token")
    for a in accs:
        live = bool(a.get("isLive"))
        acc = a["ctidTraderAccountId"]
        print(f"\n[ctrader] account {mask(acc)} login {mask(a.get('traderLogin', ''))} "
              f"{'LIVE server' if live else 'demo server'} broker {a.get('brokerTitleShort', '?')}")
        s = CTrader(live=live)
        try:
            s.app_auth()
            s.acc_auth(acc)
            t = s.trader(acc)
            dig = 10 ** t.get("moneyDigits", 2)
            print(f"  balance {t.get('balance', 0) / dig:,.2f}, leverage 1:{t.get('leverageInCents', 0) / 100:g}, "
                  f"account type {t.get('accountType', '?')}")
            syms = s.symbols(acc)
            hits = [x for x in syms if any(k in x.get("symbolName", "").upper().replace(".", " ").replace("_", " ")
                                           for k in NASDAQ + YEN)]
            print(f"  {len(syms)} symbols; matches: " + ", ".join(f"{x['symbolName']} (id {x['symbolId']})" for x in hits))
            if hits:
                for r in s.symbol_rules(acc, [x["symbolId"] for x in hits[:12]]):
                    nm = next(x["symbolName"] for x in hits if x["symbolId"] == r["symbolId"])
                    # volumes are in cents of a unit (100 = 1 unit / contract)
                    print(f"    {nm}: digits {r.get('digits')}, lot size {r.get('lotSize', 0) / 100:g}, "
                          f"min volume {r.get('minVolume', 0) / 100:g}, step {r.get('stepVolume', 0) / 100:g}, "
                          f"max {r.get('maxVolume', 0) / 100:g}, trading mode {r.get('tradingMode', 'ENABLED')}, "
                          f"swap long {r.get('swapLong')} short {r.get('swapShort')}, stop distance {r.get('slDistance')} "
                          f"(set in {r.get('distanceSetIn')}), trading hours {len(r.get('schedule', []))} blocks")
        except Exception as e:
            print(f"  error: {e}")
        finally:
            s.close()
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        sys.exit(check())
    ap.print_help()


if __name__ == "__main__":
    main()
