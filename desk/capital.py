"""Capital.com REST client for automatic orders on the Capital.com DEMO account (AUTOMATION.md), and (REAL MONEY, off until
the owner's go) on his real account: Capital(live=True) uses the live server and only the separate CAPITAL_LIVE_* secrets;
desk/autotrade.py only sends live orders with CAPITAL_LIVE_TRADE=live AND CAPITAL_LIVE_GO=yes.

Why Capital.com (9 Nov 2026): no fee per order, US Tech 100 minimum 0.001 contract (~EUR 25), so the whole desk fits the
plan's size from EUR 1,000-2,000 (IG's minimum is ~EUR 5,400; IBKR's USD 1 minimum fee eats the small trades, round 35).

Secrets (GitHub Actions, never in the repo or chat): CAPITAL_API_KEY, CAPITAL_IDENTIFIER (login e-mail), CAPITAL_PASSWORD
(the API key's own password); optional CAPITAL_ACCOUNT_ID (otherwise the preferred account). The key needs 2FA and a
verified account ('Trade' permission).
Built from the public API (open-api.capital.com, as used by open-source clients); fields still to confirm with --check:
the US Tech 100 epic (CAPITAL_EPIC, default US100), contract value per point, minimum size and stop distance.

The methods return the same shapes as desk/ig.py's IG client (accountInfo, snapshot / dealingRules, positions,
dealStatus / level / dealId), so desk/autotrade.py drives both the same way. Differences: Capital.com positions are
market orders (no fill-or-kill limit price) and we cannot give our own deal reference (they make one).

Run: python -m desk.capital --check      log in, show accounts, hedging mode, the Nasdaq markets and their rules (no orders)
     python -m desk.capital --history 14 activity + transactions (fills, overnight fees) of the last N days (no orders)
"""
import argparse
import json
import os
import sys
import time

import requests

BASE = "https://demo-api-capital.backend-capital.com/api/v1"   # demo server (the default). Never change.
LIVE_BASE = "https://api-capital.backend-capital.com/api/v1"   # real money: only via Capital(live=True)
EPIC = os.environ.get("CAPITAL_EPIC") or "US100"
SEARCH = ("US Tech 100", "Nasdaq", "US100")


class Capital:
    def __init__(self, key=None, user=None, password=None, account=None, live=False):
        pre = "CAPITAL_LIVE_" if live else "CAPITAL_"      # real-money keys never mix with the demo keys
        self.live, self.base = live, LIVE_BASE if live else BASE
        self.key = key or os.environ[pre + "API_KEY"].strip()
        self.user = user or os.environ[pre + "IDENTIFIER"].strip()
        self.password = password or os.environ[pre + "PASSWORD"]
        self.account = (account or os.environ.get(pre + "ACCOUNT_ID") or "").strip()
        self.h = {"X-CAP-API-KEY": self.key, "Content-Type": "application/json", "Accept": "application/json"}

    def req(self, method, path, body=None, params=None):
        r = requests.request(method, self.base + path, headers=self.h, params=params,
                             data=json.dumps(body) if body is not None else None, timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"Capital.com {method} {path}: HTTP {r.status_code} {r.text[:300]}")
        return r

    def login(self):
        r = self.req("POST", "/session", body={"identifier": self.user, "password": self.password,
                                               "encryptedPassword": False})
        self.h["CST"], self.h["X-SECURITY-TOKEN"] = r.headers["CST"], r.headers["X-SECURITY-TOKEN"]
        s = r.json()
        cur = s.get("currentAccountId")
        if self.account and cur != self.account:
            self.req("PUT", "/session", body={"accountId": self.account})
            cur = self.account
        accts = self.req("GET", "/accounts").json().get("accounts", [])
        a = next((x for x in accts if x.get("accountId") == cur), None) or next(
            (x for x in accts if x.get("preferred")), accts[0] if accts else {})
        b = a.get("balance") or {}
        # Capital.com: balance = deposit + open profit/loss; IG's shape = balance (cash) + profitLoss
        return {"currentAccountId": a.get("accountId"), "currencyIsoCode": a.get("currency"), "accounts": accts,
                "accountInfo": {"balance": b.get("deposit", b.get("balance")), "profitLoss": b.get("profitLoss", 0),
                                "available": b.get("available")}}

    def preferences(self):
        return self.req("GET", "/accounts/preferences").json()

    def ensure_hedging(self):
        """Two strategies can be long and short at once (IG: forceOpen). Netting mode would let one close the other."""
        p = self.preferences()
        if not p.get("hedgingMode"):
            self.req("PUT", "/accounts/preferences", body={"hedgingMode": True})
            return True
        return False

    def market(self, epic):
        d = self.req("GET", f"/markets/{epic}").json()
        snap, rules = d.get("snapshot", {}), d.get("dealingRules", {})
        bid, offer = float(snap.get("bid") or 0), float(snap.get("offer") or 0)
        dist = rules.get("minStopOrProfitDistance") or {}
        v = float(dist.get("value") or 0)
        pts = v / 100 * (bid + offer) / 2 if dist.get("unit") == "PERCENTAGE" else v
        d["dealingRules"] = dict(rules, minNormalStopOrLimitDistance={"unit": "POINTS", "value": round(pts, 2)})
        d["snapshot"] = dict(snap, bid=bid, offer=offer)
        return d

    def search(self, term):
        return self.req("GET", "/markets", params={"searchTerm": term}).json()

    def confirm(self, ref):
        for _ in range(5):
            try:
                c = self.req("GET", f"/confirms/{ref}").json()
            except RuntimeError:
                time.sleep(1)
                continue
            deals = c.get("affectedDeals") or []
            c["dealId"] = next((x.get("dealId") for x in deals if x.get("status") in ("OPENED", "OPEN")), None) \
                or c.get("dealId")
            return c
        return {"dealStatus": "UNKNOWN", "reason": f"no confirm for {ref}"}

    def open(self, epic, direction, size, level, stop, limit, ref):
        """Market order with stop / target attached (Capital.com has no fill-or-kill price; `level`, `ref` unused)."""
        body = {"epic": epic, "direction": direction, "size": size, "guaranteedStop": False, "trailingStop": False}
        if stop is not None:
            body["stopLevel"] = stop
        if limit is not None:
            body["profitLevel"] = limit
        r = self.req("POST", "/positions", body=body).json()
        return self.confirm(r["dealReference"])

    def close(self, deal_id, direction, size, level, ref):
        r = self.req("DELETE", f"/positions/{deal_id}").json()
        return self.confirm(r["dealReference"]) if r.get("dealReference") else {"dealStatus": "ACCEPTED"}

    def update_stop(self, deal_id, stop, limit):
        body = {"guaranteedStop": False, "trailingStop": False, "stopLevel": stop}
        if limit is not None:
            body["profitLevel"] = limit
        return self.req("PUT", f"/positions/{deal_id}", body=body).json()

    def positions(self):
        out = []
        for p in self.req("GET", "/positions").json().get("positions", []):
            pos = dict(p.get("position", {}))
            pos["limitLevel"] = pos.get("profitLevel")
            out.append({"market": p.get("market", {}), "position": pos})
        return out


def mask(s):
    s = str(s or "")
    return s[:2] + "*" * max(len(s) - 4, 0) + s[-2:] if len(s) > 4 else "*" * len(s)


def check(live=False):
    c = Capital(live=live)
    s = c.login()
    print(f"logged in to the Capital.com {'LIVE (real money, read only)' if live else 'DEMO'} server; account {mask(s['currentAccountId'])} ({s['currencyIsoCode']})")
    for a in s["accounts"]:
        print(f"  account {mask(a.get('accountId'))}: {a.get('accountType')} {a.get('accountName')} {a.get('currency')} "
              f"preferred={a.get('preferred')} balance {a.get('balance')}")
    p = c.preferences()
    print(f"  preferences: hedgingMode {p.get('hedgingMode')} leverages {json.dumps(p.get('leverages', {}))[:300]}")
    seen = set()
    for term in SEARCH:
        for m in c.search(term).get("markets", []):
            e = m.get("epic")
            if e in seen or not any(k in (str(m.get("instrumentName")) + e).upper() for k in ("TECH", "NAS", "US100")):
                continue
            seen.add(e)
            print(f"  {e:16s} {m.get('instrumentName')!s:40s} {m.get('instrumentType')} bid {m.get('bid')} "
                  f"offer {m.get('offer')} {m.get('marketStatus')}")
    yen = os.environ.get("CAPITAL_YEN_EPIC") or "USDJPY"     # yen week (desk/yen_week.py): size must be in dollars
    for e in [EPIC, yen] + sorted(seen - {EPIC})[:4]:
        try:
            d = c.market(e)
        except RuntimeError as err:
            print(f"market {e}: {err}"[:200])
            continue
        ins, rules, snap = d.get("instrument", {}), d.get("dealingRules", {}), d.get("snapshot", {})
        print(f"market {e}: {ins.get('name')} | type {ins.get('type')} | currency {ins.get('currency')} | lotSize "
              f"{ins.get('lotSize')} | contract {ins.get('contractSize')} | margin {ins.get('marginFactor')}"
              f"{ins.get('marginFactorUnit')} | overnight {json.dumps(ins.get('overnightFee'))[:200]}")
        for k, v in rules.items():
            print(f"  {k}: {v}")
        print(f"  now: bid {snap.get('bid')} offer {snap.get('offer')} spread "
              f"{round(snap['offer'] - snap['bid'], 2) if snap.get('bid') else None} status {snap.get('marketStatus')}")
        time.sleep(0.3)
    pos = c.positions()
    print(f"open positions: {len(pos)}")
    for x in pos:
        q = x["position"]
        print(f"  {x['market'].get('epic')} {q.get('direction')} {q.get('size')} at {q.get('level')} "
              f"stop {q.get('stopLevel')} target {q.get('profitLevel')}")
    print("check done: no orders sent")


def history(days):
    c = Capital()
    c.login()
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - days * 86400))
    act = c.req("GET", "/history/activity", params={"from": since, "detailed": "true"}).json().get("activities", [])
    print(f"ACTIVITY last {days} days: {len(act)}")
    for a in sorted(act, key=lambda x: x.get("dateUTC", x.get("date", ""))):
        d = a.get("details") or {}
        print(f"  {a.get('dateUTC', a.get('date', ''))[:19]} {a.get('type')} {a.get('status')} {a.get('epic')} "
              f"{d.get('direction', '')} {d.get('size', '')} level {d.get('level', '')} stop {d.get('stopLevel', '')} "
              f"target {d.get('profitLevel', '')} {a.get('source', '')}")
    tr = c.req("GET", "/history/transactions", params={"from": since}).json().get("transactions", [])
    print(f"TRANSACTIONS last {days} days: {len(tr)}")
    tot = {}
    for t in sorted(tr, key=lambda x: x.get("dateUtc", "")):
        print(f"  {t.get('dateUtc', '')[:19]} {t.get('transactionType')} {t.get('instrumentName')} size {t.get('size')} "
              f"P&L {t.get('profitAndLoss')} {t.get('currency')} {t.get('note', '')}")
        try:
            tot[t.get("transactionType")] = tot.get(t.get("transactionType"), 0.0) + float(
                str(t.get("profitAndLoss")).replace(",", "").lstrip("E€$£"))
        except ValueError:
            pass
    print("totals by type:", {k: round(v, 2) for k, v in tot.items()})
    print("history done: no orders sent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--history", type=int, metavar="DAYS")
    ap.add_argument("--live", action="store_true", help="--check the real account (CAPITAL_LIVE_* secrets, no orders)")
    a = ap.parse_args()
    pre = "CAPITAL_LIVE_" if a.live else "CAPITAL_"
    missing = [pre + k for k in ("API_KEY", "IDENTIFIER", "PASSWORD") if not os.environ.get(pre + k)]
    if missing:
        sys.exit(f"missing secrets: {', '.join(missing)}")
    if a.check:
        check(a.live)
    if a.history:
        history(a.history)


if __name__ == "__main__":
    main()
