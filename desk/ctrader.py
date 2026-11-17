"""cTrader Open API client (Spotware), for FTMO's cTrader accounts and a free cTrader demo (round 46, AUTOMATION.md).
Step 1 = read-only check: log in with the app, list the cTrader accounts of the token, and per account show the
balance and the Nasdaq-100 / USD/JPY contracts with their size rules. No orders in this file yet.

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
          accounts=2149, accounts_res=2150, refresh=2173, refresh_res=2174, heartbeat=51, common_error=50)
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

    def ask(self, ptype, payload, want):
        self.n += 1
        mid = f"m{self.n}"
        self.ws.send(json.dumps({"clientMsgId": mid, "payloadType": ptype, "payload": payload}))
        t0 = time.time()
        while time.time() - t0 < 30:
            m = json.loads(self.ws.recv())
            pt = m.get("payloadType")
            if pt == PT["heartbeat"]:
                continue
            if pt in (PT["error"], PT["common_error"]):
                p = m.get("payload", {})
                raise RuntimeError(f"{p.get('errorCode')}: {p.get('description', '')}")
            if pt == want and m.get("clientMsgId", mid) == mid:
                return m.get("payload", {})
        raise TimeoutError(f"no answer to message type {ptype}")

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
                          f"swap long {r.get('swapLong')} short {r.get('swapShort')}")
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
