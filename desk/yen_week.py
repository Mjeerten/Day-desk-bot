"""Yen week (strategy Y, rounds 44 / 44b): USD/JPY long Monday to Wednesday, short Thursday and Friday, on the IG DEMO
the Capital.com DEMO (14 Nov 2026) and the FTMO account via cTrader (18 Nov 2026), each its own switch, state and messages.

Rule (New York time):
  - Monday 02:00: buy USD/JPY (the hourly Monday-dip bot run at :16 does it, 08:16 Amsterdam)
  - Wednesday 16:45: close the buy and sell short (the 22:46 Amsterdam run; 21:46 in the weeks the clocks differ)
  - Friday 16:45: close (nothing held over the weekend)
  No stop, no target (as tested). Backtest at these times: +7.9 / +12.9 %/yr at 1x (2019-23 / 2024+), every year since
  2019 up, worst week -3.6 %, worst drop -10 %. Round 44c (final optimization): yen crosses, baskets, other times, stops,
  Fed-week / volatility filters do not beat it on both periods; only the size changes (2x, the owner).
A missed run: the next run does it, if the buy is at most LATE_H hours late (the short too); a close is never too late.
Size: LEV x the plan's stake (desk/plan.py) worth of dollars, at least the broker's minimum (IG trades its minimum 0.2 and
records 'scale' like the Nasdaq trades; Capital.com and cTrader size in dollars, so they trade the full 2x; FTMO's stake = CTRADER_SIZE x its start balance
like its Nasdaq orders). IG orders: fill-or-kill
limit SLIP_PCT through the quote, like desk/autotrade.py; Capital.com: market orders (no fill-or-kill there).
Switches: AUTO_TRADE (IG) / CAPITAL_TRADE (Capital.com) / CTRADER_TRADE (FTMO), off | dry | demo as the Nasdaq orders, and YEN_TRADE=off to stop
only this strategy on both. A Capital.com dry run logs only (no phone messages), like its Nasdaq orders.
State: key 'yen' in each broker's state file (.daybot/ig-<mode>.json, .daybot/capital-<mode>.json), so the 23:00 report sees it.
Capital.com: a USD/JPY position there that is not in our state blocks a new order (no deal reference to match on).
"""
from __future__ import annotations

import datetime as dt
import os
import time

import pandas as pd

from desk import autotrade as at

NY = "America/New_York"
EPIC = os.environ.get("YEN_EPIC") or "CS.D.USDJPY.MINI.IP"
CAP_EPIC = os.environ.get("CAPITAL_YEN_EPIC") or "USDJPY"   # Capital.com: price in yen, size in dollars (check with --check)
CT_EPIC = os.environ.get("CTRADER_YEN_EPIC") or "USDJPY"    # FTMO / cTrader: volume in cents of a dollar, size = dollars
BROKERS = ("ig", "capital", "ctrader")
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


def mode(broker="ig"):
    if (os.environ.get("YEN_TRADE") or "").strip().lower() == "off":
        return "off"
    return at.mode(broker)


def run(stake, now=None):
    """Called by bot.main after the Nasdaq orders, every run, for each broker. Logs in only when something has to change.
    Never raises."""
    if not at.enabled:
        return
    now = pd.Timestamp(now or pd.Timestamp.now(tz=NY)).tz_convert(NY)
    for broker in BROKERS:
        m = mode(broker)
        if m == "off":
            print(f"[yen/{broker}] off ({at.BROKERS[broker]['mode_env']} / YEN_TRADE off or secrets missing)")
            continue
        try:
            _run(m, stake, now, broker)
        except Exception as err:  # noqa
            if at.retry_elsewhere(err, broker):
                continue
            at.notify(f"{at.BROKERS[broker]['label']}: yen week problem", f"Yen week orders skipped this run: {err!r}"[:400],
                      quiet=(broker != "ig" and m == "dry"))


def _run(m, stake, now, broker="ig"):
    side, ref, since = target(now)
    st = at.load(m, broker)
    yen = st.setdefault("yen", {})
    open_ = {r: t for r, t in yen.items() if t.get("status") == "open"}
    to_close = [r for r in open_ if r != ref]
    to_open = side != 0 and ref not in yen and (now - since) <= pd.Timedelta(hours=LATE_H)
    if side != 0 and ref not in yen and not to_open:
        yen[ref] = dict(side=side, status="too_late", day=f"{now:%y%m%d}")
        print(f"[yen] {ref}: {(now - since).total_seconds() / 3600:.1f} h late (limit {LATE_H}), not opened")
        at.save(st, m, broker)
        return
    if not to_close and not to_open:
        print(f"[yen/{broker}] nothing to do ({'holding ' + ref if ref in open_ else 'flat'})")
        return
    import importlib
    B = at.BROKERS[broker]
    mod = importlib.import_module(B["module"])
    if B["demo_host"] not in mod.BASE:
        raise RuntimeError("not the demo server: refusing to trade")
    is_ig = broker == "ig"
    epic, lab = {"ig": EPIC, "capital": CAP_EPIC}.get(broker, CT_EPIC), B["label"]
    quiet = not is_ig and m == "dry"
    dry = " (dry run)" if m == "dry" else ""
    ig = getattr(mod, B["cls"])()
    sess = ig.login()
    if "stake_x" in B:                   # prop account (FTMO): a share of its start balance, as its Nasdaq orders
        info = sess.get("accountInfo", {})
        stake = B["stake_x"] * st.setdefault("start_balance", float(info.get("balance") or 0) + float(info.get("profitLoss") or 0))
    mk = ig.market(epic)
    snap, rules, ins = mk["snapshot"], mk["dealingRules"], mk.get("instrument", {})
    bid, offer = float(snap["bid"]), float(snap["offer"])
    if snap.get("marketStatus") != "TRADEABLE":
        print(f"[yen/{broker}] market {snap.get('marketStatus')}: next run")
        return
    dec = int(snap.get("decimalPlacesFactor") or 3)
    cur = next((c["code"] for c in ins.get("currencies", []) if c.get("isDefault")), None) or \
        (ins.get("currencies") or [{"code": "JPY"}])[0]["code"]
    contract = float(ins.get("contractSize") or 10000) if is_ig else 1.0   # Capital.com / cTrader: size = dollars
    step = float(rules["minDealSize"]["value"])
    pos = {p["position"]["dealId"]: p for p in ig.positions() if p["market"].get("epic") == epic}
    on_ig = {str(p["position"].get("dealReference") or "").split("-")[0] for p in pos.values()}
    for r in to_close:
        t = yen[r]
        p = pos.get(t.get("deal"))
        if m == "demo":
            if not p:
                t.update(status="closed_elsewhere", closed=str(now))
                at.notify(f"{lab}: {NAME} position gone", f"{r} was no longer open on {lab} (closed by hand?).", quiet)
                continue
            px = bid if t["side"] == 1 else offer
            lvl = px * (1 - SLIP_PCT) if t["side"] == 1 else px * (1 + SLIP_PCT)
            c = ig.close(t["deal"], "SELL" if t["side"] == 1 else "BUY", float(p["position"]["size"]), lvl, r + "x",
                         **({"digits": dec} if is_ig else {}))
            if c.get("dealStatus") != "ACCEPTED":
                at.notify(f"{lab}: {NAME} close refused", f"{r}: {c.get('reason')}; tries again next run", quiet)
                continue
            exit_ = float(c.get("level") or px)
        else:
            exit_ = bid if t["side"] == 1 else offer
        pct = t["side"] * (exit_ / t["level"] - 1) * 100
        eur = pct / 100 * t["size"] * contract * USD_EUR
        t.update(status="closed", exit=exit_, closed=str(now), pct=round(pct, 3), eur=round(eur, 2))
        at.notify(f"{at.money(eur)} (about) | {NAME} CLOSED | {lab}{dry}",
                  f"{'Sold' if t['side'] == 1 else 'Bought back'} USD/JPY at {exit_:g} "
                  f"({'bought' if t['side'] == 1 else 'shorted'} at {t['level']:g}): {pct:+.2f}%.\n"
                  f"Exact amount incl. funding: tonight's 23:00 report.\n{r}",
                  quiet=quiet, tags=("white_check_mark",) if eur >= 0 else ("x",), prio=4)
    if to_open:
        if ref in on_ig:
            print(f"[yen/{broker}] {ref}: already on {lab}, not opened again")
            return at.save(st, m, broker)
        known = {t.get("deal") for t in yen.values()}
        if not is_ig and [d for d in pos if d not in known]:
            at.notify(f"{lab}: {NAME} not opened", f"{ref}: a USD/JPY position on {lab} is not the bot's (opened by hand?): "
                      "no new order until it is closed.", quiet)
            return at.save(st, m, broker)
        px = offer if side == 1 else bid
        size = max(step, round(LEV * stake / USD_EUR / contract / step) * step)
        if size * contract * USD_EUR > 4 * LEV * stake + 3000:    # a misread contract size must never send a huge order
            raise RuntimeError(f"size {size} x {contract} looks too big for a EUR {stake:.0f} stake: not sent")
        size = round(size, 4)
        lvl = px * (1 + SLIP_PCT) if side == 1 else px * (1 - SLIP_PCT)
        if m == "demo":
            c = ig.open(epic, "BUY" if side == 1 else "SELL", size, lvl, None, None,
                        f"{ref}-{int(time.time()) % 100000}", **({"currency": cur, "digits": dec} if is_ig else {}))
            if c.get("dealStatus") != "ACCEPTED":
                at.notify(f"{lab}: {NAME} order refused", f"{ref}: {c.get('reason')}; tries again next run", quiet)
                return at.save(st, m, broker)
            fill, deal = float(c.get("level") or px), c.get("dealId")
        else:
            fill, deal = px, None
        usd = size * contract
        yen[ref] = dict(side=side, status="open", day=f"{now:%y%m%d}", level=fill, size=size, deal=deal,
                        opened=str(now), scale=round(LEV * stake / (usd * USD_EUR), 4))
        nxt = "Wednesday 16:45 NY (22:45 Amsterdam): flip to short" if side == 1 else \
            "Friday 16:45 NY (22:45 Amsterdam): close"
        at.notify(f"{'BUY' if side == 1 else 'SHORT'} USD/JPY {NAME} at {fill:g} | {lab}{dry}",
                  f"OPENED: {'bought' if side == 1 else 'sold short'} {size:g} USD/JPY (about USD {usd:,.0f}) on {lab}. "
                  f"No stop (as tested). Next: {nxt}.\n{ref}",
                  quiet=quiet, tags=("chart_with_upwards_trend",) if side == 1 else ("chart_with_downwards_trend",), prio=4)
    cut = f"{now - pd.Timedelta(days=60):%y%m%d}"
    st["yen"] = {r: t for r, t in yen.items() if t.get("status") == "open" or t.get("day", "999999") >= cut}
    at.save(st, m, broker)
