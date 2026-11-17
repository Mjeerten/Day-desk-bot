"""IG REST client for automatic orders on the IG DEMO account (AUTOMATION.md). Demo only: the base URL is fixed to IG's
demo gateway, so live credentials cannot log in and no live order can be sent from this code.

Secrets (GitHub Actions, never in the repo): IG_API_KEY, IG_USERNAME, IG_PASSWORD, IG_ACCOUNT_ID (the demo account).
Market: US Tech 100 Cash (EUR 1 contract: 1 contract = EUR 1 per point, min 0.01, normal stop at least 4 points away;
checked on the demo ticket 3 Nov 2026).

Run: python -m desk.ig --check   log in, show the account, the US Tech 100 market rules and open positions (no orders)
     python -m desk.ig --history 14   fills (activity) and money moves (transactions: deals, overnight funding) of the
                                      last N days, for the cost check (no orders)
"""
import argparse
import json
import os
import sys
import time

import requests

BASE = "https://demo-api.ig.com/gateway/deal"   # DEMO ONLY. Never change this to the live gateway without the owner's go.
SEARCH = "US Tech 100"
EPIC_GUESS = "IX.D.NASDAQ.IFE.IP"                 # US Tech 100 Cash (EUR 1); confirmed by --check


class Blocked(RuntimeError):
    """IG's website firewall answered with its HTML deny page (HTTP 403): this GitHub server's address is blocked, not
    the account. The workflow then runs the bot once more on a fresh server (daybot.yml, job 'retry')."""


class IG:
    def __init__(self, key=None, user=None, password=None, account=None):
        self.key = key or os.environ["IG_API_KEY"].strip()
        self.user = user or os.environ["IG_USERNAME"].strip()
        self.password = password or os.environ["IG_PASSWORD"]
        self.account = (account or os.environ.get("IG_ACCOUNT_ID") or "").strip()
        self.h = {"X-IG-API-KEY": self.key, "Accept": "application/json; charset=UTF-8",
                  "Content-Type": "application/json; charset=UTF-8"}

    RETRY_WAITS = (10,)   # seconds; IG's firewall blocks some GitHub servers: one quick retry, then a fresh server

    def req(self, method, path, version=1, body=None, extra=None):
        h = dict(self.h, Version=str(version), **(extra or {}))
        for wait in self.RETRY_WAITS + (None,):
            r = requests.request(method, BASE + path, headers=h, data=json.dumps(body) if body is not None else None,
                                 timeout=30)
            blocked = r.status_code == 403 and "<html" in r.text[:300].lower()   # firewall deny page, not an API error
            if not blocked or wait is None:
                break
            print(f"[ig] {method} {path}: blocked by IG's firewall (HTTP 403 page), retrying in {wait}s")
            time.sleep(wait)
        if blocked:
            raise Blocked(f"IG {method} {path}: blocked by IG's firewall (HTTP 403 deny page) "
                          f"{1 + len(self.RETRY_WAITS)} times from this GitHub server")
        if r.status_code >= 400:
            raise RuntimeError(f"IG {method} {path}: HTTP {r.status_code} {r.text[:300]}")
        return r

    def login(self):
        r = self.req("POST", "/session", version=2, body={"identifier": self.user, "password": self.password})
        self.h["CST"], self.h["X-SECURITY-TOKEN"] = r.headers["CST"], r.headers["X-SECURITY-TOKEN"]
        s = r.json()
        if self.account and s.get("currentAccountId") != self.account:
            self.req("PUT", "/session", body={"accountId": self.account, "defaultAccount": False})
            s["currentAccountId"] = self.account
        return s

    def market(self, epic):
        return self.req("GET", f"/markets/{epic}", version=3).json()

    def search(self, term):
        r = requests.get(BASE + "/markets", params={"searchTerm": term}, headers=dict(self.h, Version="1"), timeout=30)
        if r.status_code >= 400:
            raise RuntimeError(f"IG GET /markets: HTTP {r.status_code} {r.text[:300]}")
        return r.json()

    def confirm(self, ref):
        for _ in range(5):
            r = requests.get(BASE + f"/confirms/{ref}", headers=dict(self.h, Version="1"), timeout=30)
            if r.status_code == 200:
                return r.json()
            time.sleep(1)
        return {"dealStatus": "UNKNOWN", "reason": f"no confirm for {ref}"}

    def open(self, epic, direction, size, level, stop, limit, ref, currency="EUR", digits=1):
        """Fill-or-kill limit order at `level` (a market order that cannot fill worse), stop / limit levels attached.
        currency / digits: the instrument's currency and price decimals (US/JPY: its own, see --check)."""
        body = {"epic": epic, "expiry": "-", "direction": direction, "size": size, "orderType": "LIMIT",
                "level": round(level, digits), "timeInForce": "FILL_OR_KILL", "guaranteedStop": False, "forceOpen": True,
                "currencyCode": currency, "dealReference": ref}
        if stop is not None:
            body["stopLevel"] = stop
        if limit is not None:
            body["limitLevel"] = limit
        self.req("POST", "/positions/otc", version=2, body=body)
        return self.confirm(ref)

    def close(self, deal_id, direction, size, level, ref, digits=1):
        body = {"dealId": deal_id, "direction": direction, "size": size, "orderType": "LIMIT", "level": round(level, digits),
                "timeInForce": "FILL_OR_KILL"}
        r = self.req("POST", "/positions/otc", version=1, body=body, extra={"_method": "DELETE"})
        return self.confirm(r.json().get("dealReference", ref))

    def update_stop(self, deal_id, stop, limit):
        body = {"stopLevel": stop, "limitLevel": limit, "trailingStop": False}
        return self.req("PUT", f"/positions/otc/{deal_id}", version=2, body=body).json()

    def positions(self):
        return self.req("GET", "/positions", version=2).json().get("positions", [])

    def closed_since(self, since):
        """Closed deals since `since` (UTC, 'YYYY-MM-DDTHH:MM:SS'), IG's own amounts: dicts date, open, close, size, eur."""
        tr = self.req("GET", f"/history/transactions?type=ALL_DEAL&from={since}&pageSize=500", version=2).json()
        out = []
        for t in tr.get("transactions", []):
            try:
                eur = float(str(t.get("profitAndLoss")).replace(",", "").lstrip("E€$£"))
                out.append(dict(date=t.get("dateUtc", ""), open=float(t.get("openLevel")), close=float(t.get("closeLevel")),
                                size=t.get("size"), eur=eur))
            except (TypeError, ValueError):
                continue
        return out


def mask(s):
    s = str(s or "")
    return s[:2] + "*" * max(len(s) - 4, 0) + s[-2:] if len(s) > 4 else "*" * len(s)


def check():
    ig = IG()
    s = ig.login()
    accts = s.get("accounts", [])
    print(f"logged in to the IG DEMO gateway; current account {mask(s.get('currentAccountId'))}, "
          f"secret IG_ACCOUNT_ID {'matches' if s.get('currentAccountId') == ig.account else 'NOT FOUND / differs'}")
    for a in accts:
        print(f"  account {mask(a.get('accountId'))}: {a.get('accountType')} {a.get('accountName')} "
              f"preferred={a.get('preferred')}")
    info = s.get("accountInfo", {})
    print(f"  balance {info.get('balance')} available {info.get('available')} {s.get('currencyIsoCode')}")
    found = ig.search(SEARCH).get("markets", [])
    print(f"search '{SEARCH}':")
    for m in found[:15]:
        print(f"  {m.get('epic'):28s} {m.get('instrumentName')!s:40s} {m.get('expiry')} bid {m.get('bid')} "
              f"offer {m.get('offer')} {m.get('marketStatus')}")
    epics = [m["epic"] for m in found if "1" in str(m.get("instrumentName")) and "Cash" in str(m.get("instrumentName"))]
    epic = EPIC_GUESS if EPIC_GUESS in [m["epic"] for m in found] or not epics else epics[0]
    d = ig.market(epic)
    ins, rules, snap = d.get("instrument", {}), d.get("dealingRules", {}), d.get("snapshot", {})
    print(f"market {epic}: {ins.get('name')} | type {ins.get('type')} | currency "
          f"{[c.get('code') for c in ins.get('currencies', [])]} | value of one pip {ins.get('valueOfOnePip')} "
          f"| contract size {ins.get('contractSize')} | margin {ins.get('marginFactor')}{ins.get('marginFactorUnit')}")
    for k in ("minDealSize", "minNormalStopOrLimitDistance", "minControlledRiskStopDistance", "maxStopOrLimitDistance",
              "minStepDistance", "marketOrderPreference", "trailingStopsPreference"):
        print(f"  {k}: {rules.get(k)}")
    print(f"  now: bid {snap.get('bid')} offer {snap.get('offer')} status {snap.get('marketStatus')} "
          f"scaling {snap.get('scalingFactor')}")
    pos = ig.positions()
    print(f"open positions: {len(pos)}")
    for p in pos:
        print(f"  {p['market'].get('epic')} {p['position'].get('direction')} {p['position'].get('size')} "
              f"at {p['position'].get('level')} stop {p['position'].get('stopLevel')} limit {p['position'].get('limitLevel')}")
    print("every Nasdaq product (smallest trade = min size x value per point x price):")
    seen = set()
    for term in ("US Tech 100", "Nasdaq", "NASDAQ 100", "US Tech", "Nasdaq 100 mini"):
        for m in ig.search(term).get("markets", []):
            e = m.get("epic")
            if e in seen or "NASDAQ" not in e.upper() and "TECH" not in str(m.get("instrumentName")).upper():
                continue
            seen.add(e)
            try:
                d = ig.market(e)
            except Exception as err:
                print(f"  {e}: {err!r}"[:160])
                continue
            ins, rules, snap = d.get("instrument", {}), d.get("dealingRules", {}), d.get("snapshot", {})
            mn = (rules.get("minDealSize") or {}).get("value")
            cur = [c.get("code") for c in ins.get("currencies", [])]
            bid, offer = snap.get("bid"), snap.get("offer")
            print(f"  {e:26s} {ins.get('name')!s:44s} {ins.get('type')!s:10s} expiry {ins.get('expiry')!s:8s} {cur} "
                  f"pip {ins.get('valueOfOnePip')} contract {ins.get('contractSize')} lot {ins.get('lotSize')} "
                  f"min {mn} margin {ins.get('marginFactor')}{ins.get('marginFactorUnit')} "
                  f"spread {round(offer - bid, 2) if bid and offer else None} status {snap.get('marketStatus')}")
            time.sleep(1)
    print("USD/JPY products (round 44 yen week; contract size x size = dollars, prices may be scaled):")
    for m in ig.search("USD/JPY").get("markets", [])[:12]:
        e = m.get("epic")
        try:
            d = ig.market(e)
        except Exception as err:
            print(f"  {e}: {err!r}"[:160])
            continue
        ins, rules, snap = d.get("instrument", {}), d.get("dealingRules", {}), d.get("snapshot", {})
        print(f"  {e:26s} {ins.get('name')!s:34s} {ins.get('type')!s:10s} expiry {ins.get('expiry')!s:6s} "
              f"currencies {[(c.get('code'), c.get('isDefault')) for c in ins.get('currencies', [])]} "
              f"contract {ins.get('contractSize')} lot {ins.get('lotSize')} pip {ins.get('valueOfOnePip')} "
              f"onePipMeans {ins.get('onePipMeans')} min {(rules.get('minDealSize') or {}).get('value')} "
              f"margin {ins.get('marginFactor')}{ins.get('marginFactorUnit')} bid {snap.get('bid')} offer {snap.get('offer')} "
              f"scaling {snap.get('scalingFactor')} decimals {snap.get('decimalPlacesFactor')} status {snap.get('marketStatus')} "
              f"orders {rules.get('marketOrderPreference')}")
        time.sleep(1)
    print("check done: no orders sent")


def history(days):
    """Read-only: what IG filled and charged. Activity = every order with its fill level; transactions = closed deals and
    cash items such as overnight funding."""
    ig = IG()
    ig.login()
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - days * 86400))
    r = requests.get(BASE + "/history/activity", params={"from": since, "detailed": "true", "pageSize": 500},
                     headers=dict(ig.h, Version="3"), timeout=30)
    r.raise_for_status()
    act = r.json().get("activities", [])
    print(f"ACTIVITY last {days} days: {len(act)}")
    print("date (UTC)           | type     | status   | ref                          | dir  | size | fill level | stop     | limit    | text")
    for a in sorted(act, key=lambda x: x.get("date", "")):
        d = a.get("details") or {}
        print(f"{a.get('date', '')[:19]:20s} | {a.get('type', '')!s:8s} | {a.get('status', '')!s:8s} | "
              f"{d.get('dealReference', '')!s:28s} | {d.get('direction', '')!s:4s} | {d.get('size', '')!s:4s} | "
              f"{d.get('level', '')!s:10s} | {d.get('stopLevel', '')!s:8s} | {d.get('limitLevel', '')!s:8s} | "
              f"{a.get('description', '')}"[:220])
    r = requests.get(BASE + "/history/transactions", params={"type": "ALL", "from": since, "pageSize": 500},
                     headers=dict(ig.h, Version="2"), timeout=30)
    r.raise_for_status()
    tr = r.json().get("transactions", [])
    print(f"\nTRANSACTIONS last {days} days: {len(tr)}")
    print("date (UTC)           | type     | cash | instrument                       | size | open     | close    | P&L")
    tot = {}
    for t in sorted(tr, key=lambda x: x.get("dateUtc", "")):
        pl = str(t.get("profitAndLoss", ""))
        print(f"{t.get('dateUtc', '')[:19]:20s} | {t.get('transactionType', '')!s:8s} | {t.get('cashTransaction')!s:5s}| "
              f"{t.get('instrumentName', '')!s:32s} | {t.get('size', '')!s:4s} | {t.get('openLevel', '')!s:8s} | "
              f"{t.get('closeLevel', '')!s:8s} | {pl}")
        try:
            v = float(pl.replace(",", "").lstrip("E€$£"))
        except ValueError:
            continue
        typ = str(t.get("transactionType", ""))
        key = ("deposits" if typ == "DEPO" else "funding / interest" if "Interest" in str(t.get("instrumentName"))
               else "deals" if typ.upper().startswith(("DEAL", "TRANS")) else typ)
        tot[key] = tot.get(key, 0.0) + v
    print("totals:", {k: round(v, 2) for k, v in tot.items()})
    print("history done: no orders sent")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--history", type=int, metavar="DAYS")
    a = ap.parse_args()
    if a.check or a.history:
        missing = [k for k in ("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID") if not os.environ.get(k)]
        if missing:
            sys.exit(f"missing secrets: {', '.join(missing)}")
        history(a.history) if a.history else check()


if __name__ == "__main__":
    main()
