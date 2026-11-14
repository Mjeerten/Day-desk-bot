"""Position-size plan: the euros per trade for this month, with all gains reinvested.

journal/plan.json lists the phases (practice, then real money at 1x, then 1.5x). The active phase is the last one
whose "from" date has passed ("from": null = not started yet). A phase has either a fixed "stake_eur" (practice) or
a "leverage": then the stake for a month = leverage x the account balance at the start of that month, so profits
are reinvested and losses shrink the stake (compounding). The size changes only on the 1st of a month.

Balance at the start of the month = the last balance written in journal/account.csv for that account before the
month started (from the owner's Trading 212 screenshots), else the start_balance of the account's first phase, plus
the results in journal/trades.csv of that account after that point. Stop rule (BOT.md): balance 20% or more below its highest
point = stop trading and review.
"""
import csv
import datetime as dt
import json
import math
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "journal")
STOP_DD = 0.20


def _rows(path):
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def phase_for(day, plan):
    act = [p for p in plan.get("phases", []) if p.get("from") and dt.date.fromisoformat(p["from"]) <= day]
    return max(act, key=lambda p: p["from"]) if act else None


def balance(account, start, start_balance, until, snaps, trades):
    """Balance at the start of day `until` and its highest known point since `start` (both EUR)."""
    s = sorted((dt.date.fromisoformat(r["date"]), _num(r["balance_eur"])) for r in snaps
               if r.get("account") == account and _num(r.get("balance_eur")) is not None)
    s = [(d, b) for d, b in s if start <= d < until]
    t = sorted((dt.date.fromisoformat(r["date"]), _num(r["result_eur"]) or 0.0) for r in trades
               if r.get("account") == account and r.get("date"))
    base_d, base = (s[-1] if s else (start - dt.timedelta(days=1), start_balance))
    bal = base + sum(x for d, x in t if base_d < d < until)
    # the highest point: every snapshot, and the running balance after each journal trade
    peak, run, k = start_balance, start_balance, start - dt.timedelta(days=1)
    for d, b in s + [(until, None)]:
        for td, x in t:
            if k < td < d:
                run += x
                peak = max(peak, run)
        if b is not None:
            run = b
            peak = max(peak, b)
        k = d
    return bal, max(peak, bal)


def stake_for(day, root=ROOT, default=1000.0):
    """(EUR per trade, one-line explanation, stop rule hit?) for trades on `day`."""
    try:
        with open(os.path.join(root, "plan.json"), encoding="utf-8") as f:
            plan = json.load(f)
    except (OSError, ValueError):
        return default, f"EUR {default:,.0f} per trade (no plan file)", False
    p = phase_for(day, plan)
    if p is None:
        return default, f"EUR {default:,.0f} per trade (no phase started)", False
    if p.get("stake_eur"):
        return float(p["stake_eur"]), f"EUR {p['stake_eur']:,.0f} per trade ({p['name']}, fixed)", False
    # the account's history starts with its first phase (1x), later phases only change the leverage
    first = min((q for q in plan["phases"] if q.get("account") == p["account"] and q.get("from")),
                key=lambda q: q["from"])
    start = dt.date.fromisoformat(first["from"])
    month = day.replace(day=1)
    bal, peak = balance(p["account"], start, float(first["start_balance"]), max(month, start),
                        _rows(os.path.join(root, "account.csv")), _rows(os.path.join(root, "trades.csv")))
    stake = max(math.floor(p["leverage"] * bal / 10) * 10, 0)
    why = (f"EUR {stake:,.0f} per trade this month = {p['leverage']:g} x your {p['account']} balance of "
           f"EUR {bal:,.0f} on {max(month, start):%d %b} ({p['name']}, gains reinvested)")
    hit = bal <= (1 - STOP_DD) * peak
    return float(stake), why, hit
