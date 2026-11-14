"""Yen week (strategy Y, rounds 44 / 44b): USD/JPY long Monday to Wednesday, short Thursday and Friday, on the IG DEMO.

Rule (New York time):
  - Monday 02:00: buy USD/JPY (the hourly Monday-dip bot run at :16 does it, 08:16 Amsterdam)
  - Wednesday 16:45: close the buy and sell short (the 22:46 Amsterdam run; 21:46 in the weeks the clocks differ)
  - Friday 16:45: close (nothing held over the weekend)
  No stop, no target (as tested). Backtest at these times: +7.9 / +12.9 %/yr at 1x (2019-23 / 2024+), every year since
  2019 up, worst week -3.6 %, worst drop -10 %. Round 44c (final optimization): yen crosses, baskets, other times, stops,
  Fed-week / volatility filters do not beat it on both periods; only the size changes (2x, the owner).
A missed run: the next run does it, if the buy is at most LATE_H hours late (the short too); a close is never too late.
Size: LEV x the plan's stake (desk/plan.py) worth of dollars, at least IG's minimum (the demo trades the minimum and records
'scale' like the Nasdaq trades). Orders: fill-or-kill limit SLIP_PCT through the quote, like desk/autotrade.py.
Switch: AUTO_TRADE (off | dry | demo, as the Nasdaq orders) and YEN_TRADE=off to stop only this strategy.
State: key 'yen' in the IG state file (.daybot/ig-<mode>.json), so the 23:00 report sees it.
"""
from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd

from desk import autotrade as at

NY = "America/New_York"
EPIC = os.environ.get("YEN_EPIC") or "CS.D.USDJPY.MINI.IP"
BUY_AT = (0, dt.time(2, 0))       # Monday 02:00 NY
FLIP_AT = (2, dt.time(16, 45))    # Wednesday 16:45 NY
OUT_AT = (4, dt.time(16, 45))     # Friday 16:45 NY
LATE_H = 10                       # hours a buy / short may still be opened after its time (round 44b: Mon 09:00 / Thu 02:00 fine)
SLIP_PCT = 0.0002                 # limit price 0.02 % (about 3 pips) through the quote
LEV = 2.0                         # round 44c: 2x the stake = most profit at a desk worst drop of ~-30 % (the owner 13 Nov; 1.5x ~-25 %)
USD_EUR = 0.86                    # for sizing and the 'about' amounts only; the 23:00 report uses IG's own amounts
NAME = "yen week"


def _at(monday, wd_t):
    wd, t = wd_t
    return pd.Timestamp(dt.datetime.combine(monday + dt.timedelta(days=wd), t), tz=NY)


def target(now):
    """What should be open now: (side, ref, since) with side +1 buy / -1 short, or (0, None, None)."""
    now = pd.Timestamp(now).tz_convert(NY)
    monday = (now - pd.Timedelta(days=now.weekday())).date()
    buy, flip, out = _at(monday, BUY_AT), _at(monday, FLIP_AT), _at(monday, OUT_AT)
    if buy <= now < flip:
        return 1, f"Y{monday:%y%m%d}L", buy
    if flip <= now < out:
        return -1, f"Y{monday:%y%m%d}S", flip
    return 0, None, None


def mode():
    if (os.environ.get("YEN_TRADE") or "").strip().lower() == "off":
        return "off"
    return at.mode("ig")


def run(stake, now=None):
    """Called by bot.main after the Nasdaq orders, every run. Logs in to IG only when something has to change. Never raises."""
    if not at.enabled:
        return
    m = mode()
    if m == "off":
        print("[yen] off (AUTO_TRADE / YEN_TRADE off or IG secrets missing)")
        return
    now = pd.Timestamp(now or pd.Timestamp.now(tz=NY)).tz_convert(NY)
    try:
        _run(m, stake, now)
    except Exception as err:  # noqa
        at.notify("IG demo: yen week problem", f"Yen week orders skipped this run: {err!r}"[:400])


def _run(m, stake, now):
    side, ref, since = target(now)
    st = at.load(m, "ig")
    yen = st.setdefault("yen", {})
    open_ = {r: t for r, t in yen.items() if t.get("status") == "open"}
    to_close = [r for r in open_ if r != ref]
    to_open = side != 0 and ref not in yen and (now - since) <= pd.Timedelta(hours=LATE_H)
    if side != 0 and ref not in yen and not to_open:
        yen[ref] = dict(side=side, status="too_late", day=f"{now:%y%m%d}")
        print(f"[yen] {ref}: {(now - since).total_seconds() / 3600:.1f} h late (limit {LATE_H}), not opened")
        at.save(st, m, "ig")
        return
    if not to_close and not to_open:
        print(f"[yen] nothing to do ({'holding ' + ref if ref in open_ else 'flat'})")
        return
    from desk import ig as igmod
    if "demo-api.ig.com" not in igmod.BASE:
        raise RuntimeError("not the demo server: refusing to trade")
    dry = " (dry run)" if m == "dry" else ""
    ig = igmod.IG()
    ig.login()
    mk = ig.market(EPIC)
    snap, rules, ins = mk["snapshot"], mk["dealingRules"], mk.get("instrument", {})
    bid, offer = float(snap["bid"]), float(snap["offer"])
    if snap.get("marketStatus") != "TRADEABLE":
        print(f"[yen] market {snap.get('marketStatus')}: next run")
        return
    dec = int(snap.get("decimalPlacesFactor") or 3)
    cur = next((c["code"] for c in ins.get("currencies", []) if c.get("isDefault")), None) or \
        (ins.get("currencies") or [{"code": "JPY"}])[0]["code"]
    contract = float(ins.get("contractSize") or 10000)
    step = float(rules["minDealSize"]["value"])
    pos = {p["position"]["dealId"]: p for p in ig.positions() if p["market"].get("epic") == EPIC}
    on_ig = {str(p["position"].get("dealReference") or "").split("-")[0] for p in pos.values()}
    for r in to_close:
        t = yen[r]
        p = pos.get(t.get("deal"))
        if m == "demo":
            if not p:
                t.update(status="closed_elsewhere", closed=str(now))
                at.notify(f"IG demo: {NAME} position gone", f"{r} was no longer open on IG (closed by hand?).")
                continue
            px = bid if t["side"] == 1 else offer
            lvl = px * (1 - SLIP_PCT) if t["side"] == 1 else px * (1 + SLIP_PCT)
            c = ig.close(t["deal"], "SELL" if t["side"] == 1 else "BUY", float(p["position"]["size"]), lvl, r + "x", digits=dec)
            if c.get("dealStatus") != "ACCEPTED":
                at.notify(f"IG demo: {NAME} close refused", f"{r}: {c.get('reason')}; tries again next run")
                continue
            exit_ = float(c.get("level") or px)
        else:
            exit_ = bid if t["side"] == 1 else offer
        pct = t["side"] * (exit_ / t["level"] - 1) * 100
        eur = pct / 100 * t["size"] * contract * USD_EUR
        t.update(status="closed", exit=exit_, closed=str(now), pct=round(pct, 3), eur=round(eur, 2))
        at.notify(f"{at.money(eur)} (about) | {NAME} CLOSED | IG demo{dry}",
                  f"{'Sold' if t['side'] == 1 else 'Bought back'} USD/JPY at {exit_:g} "
                  f"({'bought' if t['side'] == 1 else 'shorted'} at {t['level']:g}): {pct:+.2f}%.\n"
                  f"Exact amount incl. funding: tonight's 23:00 report.\n{r}",
                  ("white_check_mark",) if eur >= 0 else ("x",), 4)
    if to_open:
        if ref in on_ig:
            print(f"[yen] {ref}: already on IG, not opened again")
            return at.save(st, m, "ig")
        px = offer if side == 1 else bid
        size = max(step, round(LEV * stake / USD_EUR / contract / step) * step)
        size = round(size, 4)
        lvl = px * (1 + SLIP_PCT) if side == 1 else px * (1 - SLIP_PCT)
        if m == "demo":
            c = ig.open(EPIC, "BUY" if side == 1 else "SELL", size, lvl, None, None,
                        f"{ref}-{int(time.time()) % 100000}", currency=cur, digits=dec)
            if c.get("dealStatus") != "ACCEPTED":
                at.notify(f"IG demo: {NAME} order refused", f"{ref}: {c.get('reason')}; tries again next run")
                return at.save(st, m, "ig")
            fill, deal = float(c.get("level") or px), c.get("dealId")
        else:
            fill, deal = px, None
        usd = size * contract
        yen[ref] = dict(side=side, status="open", day=f"{now:%y%m%d}", level=fill, size=size, deal=deal,
                        opened=str(now), scale=round(LEV * stake / (usd * USD_EUR), 4))
        nxt = "Wednesday 16:45 NY (22:45 Amsterdam): flip to short" if side == 1 else \
            "Friday 16:45 NY (22:45 Amsterdam): close"
        at.notify(f"{'BUY' if side == 1 else 'SHORT'} USD/JPY {NAME} at {fill:g} | IG demo{dry}",
                  f"OPENED: {'bought' if side == 1 else 'sold short'} {size:g} USD/JPY (about USD {usd:,.0f}) on IG demo. "
                  f"No stop (as tested). Next: {nxt}.\n{ref}",
                  ("chart_with_upwards_trend",) if side == 1 else ("chart_with_downwards_trend",), 4)
    cut = f"{now - pd.Timedelta(days=60):%y%m%d}"
    st["yen"] = {r: t for r, t in yen.items() if t.get("status") == "open" or t.get("day", "999999") >= cut}
    at.save(st, m, "ig")
