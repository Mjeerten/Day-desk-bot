"""Automatic orders on the IG DEMO and Capital.com DEMO accounts (AUTOMATION.md): mirror what the bot's signals say should be open.

Each bot run reports, per strategy it evaluated, the trades that should be open right now (`want`):
  N noise band, S overnight sweep, I IBS (from desk/bot.py), M Monday dip (desk/monday_dip.py), P Session Momentum
  (desk/session_momentum.py), O Overnight Bias ORB (desk/overnight_bias.py). Paper strategies included: on the demo every signal is taken (no 'I took it' tap needed).
`flush()` then reconciles with the IG demo account:
  - a wanted trade not opened yet and at most LATE[strategy] minutes old -> open it at the IG price now, with its stop
    and target attached to the order (IG holds them, so a missed bot run never leaves a trade unprotected);
  - one of our positions whose trade is no longer wanted (the bot's exit / 16:00 / IBS sell / 24 h ...) -> close it;
  - Session Momentum's trailing stop -> move the stop on IG when it moved.
Prices: the bot's levels are on the index (N, S, I) or futures (M, P) scale; IG's US Tech 100 Cash differs from both, so
each level is moved by (IG price now - that scale's price now). Size: the plan's stake (desk/plan.py) / price, in
EUR 1 contracts (EUR 1 per point), rounded to IG's minimum step.
Safety: AUTO_TRADE (GitHub variable) off | dry (default: logs what it would do, sends nothing) | demo; the client only
knows IG's demo gateway; never a second position for the same trade; at most MAX_OPEN positions and MAX_STAKE_X x the
stake; no new trades once the account is DAY_LOSS below the day's starting balance. Every order -> ntfy message +
.daybot/ig.json (kept by the bot's cache) + the job log.
Brokers (BROKERS): 'ig' (desk/ig.py, switch AUTO_TRADE) and 'capital' (desk/capital.py, switch CAPITAL_TRADE, same
off | dry | demo; its dry run only logs, no phone messages). Each has its own state file; both get the same signals.
Capital.com takes market orders (no limit price) and makes its own deal references, so a position there that is not in
our state file blocks new orders that run (never a second position after a lost state file).
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import time

import pandas as pd
import requests

EPIC = "IX.D.NASDAQ.IAE.IP"          # US Tech 100 Cash (EUR 1) on the IG demo (desk/ig.py --check, 3 Nov 2026)
LATE = {"N": 30, "S": 30, "I": 20, "M": 75, "P": 25, "O": 25}   # minutes after the signal an order may still be opened
NAMES = {"N": "noise band", "S": "overnight sweep", "I": "IBS", "M": "Monday dip", "P": "Session Momentum", "O": "Overnight Bias ORB",
         "Y": "yen week"}
MAX_OPEN, MAX_STAKE_X, DAY_LOSS = 6, 3.0, 0.05
SLIP = 5.0                            # limit price this many points through the quote (a market order that can't run away)
STATE = os.environ.get("IG_STATE") or ".daybot/ig-{mode}.json"   # one file per mode: a dry run never mixes with demo

BROKERS = {
    "ig": dict(label="IG demo", module="desk.ig", cls="IG", mode_env="AUTO_TRADE", demo_host="demo-api.ig.com",
               secrets=("IG_API_KEY", "IG_USERNAME", "IG_PASSWORD", "IG_ACCOUNT_ID")),
    "capital": dict(label="Capital.com demo", module="desk.capital", cls="Capital", mode_env="CAPITAL_TRADE",
                    demo_host="demo-api-capital.backend-capital.com",
                    secrets=("CAPITAL_API_KEY", "CAPITAL_IDENTIFIER", "CAPITAL_PASSWORD")),
}
CAP_STATE = os.environ.get("CAPITAL_STATE") or ".daybot/capital-{mode}.json"

_wants: dict[str, tuple[list[dict], float | None]] = {}
enabled = False                       # bot.main sets it for live runs


def mode(broker="ig"):
    b = BROKERS[broker]
    m = (os.environ.get(b["mode_env"]) or "dry").strip().lower()
    if not all(os.environ.get(k) for k in b["secrets"]):
        return "off"
    return m if m in ("off", "dry", "demo") else "dry"


def want(strat, wants, scale_px=None):
    """Called by the bot for each strategy it evaluated this run: the trades that should be open now.
    wants: dicts ref, side (+1/-1), entry, t_in (tz-aware), stop (or None), target (or None). scale_px = the latest
    price on the scale of these levels (index for N/S/I, futures for M/P)."""
    _wants[strat] = (list(wants), scale_px)


def _state(m, broker):
    return (STATE if broker == "ig" else CAP_STATE).format(mode=m)


def load(m, broker="ig"):
    try:
        with open(_state(m, broker), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"trades": {}, "day": {}}


def save(st, m, broker="ig"):
    path = _state(m, broker)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    cut = f"{dt.date.today() - dt.timedelta(days=30):%y%m%d}"
    st["trades"] = {k: v for k, v in st["trades"].items() if v.get("status") == "open" or v.get("day", "999999") >= cut}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1, default=str)


def notify(title, body, quiet=False, tags=("robot",), prio=3):
    """ntfy message. Opens / closes go out at priority 4 (louder, shown on top) with an emoji tag; the money result is in
    the title, which the phone shows in bold (Markdown in the body only renders in ntfy's web app)."""
    topic = os.environ.get("NTFY_TOPIC", "")
    print(f"[{'CAP' if 'Capital' in title else 'IG'}] {title}: {body}")
    if topic and not quiet:
        try:
            requests.post(f"https://ntfy.sh/{topic}", data=body.encode(), timeout=20,
                          headers={"Title": title.encode("utf-8"), "Priority": str(prio), "Tags": ",".join(tags)})
        except requests.RequestException:
            pass


def money(eur):
    return f"{'+' if eur >= 0 else '-'}EUR {abs(eur):,.2f}"


def num(x):
    return f"{x:,.1f}".replace(",", " ")


def size_for(stake, price, step):
    s = math.floor(stake / price / step) * step
    return round(max(s, step), 4)


def retry_elsewhere(err, broker="ig"):
    """IG's firewall blocked this GitHub server (ig.Blocked). In the first job of a workflow run, ask daybot.yml to run
    the bot again on a fresh server (job 'retry') and stay quiet; returns True then. In the retry job (IG_RETRY_JOB=1)
    or outside GitHub it returns False, so the caller sends the 'problem' message as before."""
    out = os.environ.get("GITHUB_OUTPUT")
    if broker != "ig" or type(err).__name__ != "Blocked" or os.environ.get("IG_RETRY_JOB") or not out:
        return False
    with open(out, "a") as f:
        f.write("ig_blocked=true\n")
    print(f"[ig] {err}; no message yet, the workflow retries on another GitHub server")
    return True


def flush(stake, now=None):
    """Reconcile with each broker. Never raises: a broker problem must not stop the alerts (they already went out)."""
    if not enabled or not _wants:
        return
    try:
        for broker, b in BROKERS.items():
            m = mode(broker)
            if m == "off":
                print(f"[{broker}] automatic orders off ({b['mode_env']}=off or secrets missing)")
                continue
            try:
                _flush(m, stake, now or pd.Timestamp.now(tz="America/New_York"), broker)
            except Exception as err:  # noqa
                if retry_elsewhere(err, broker):
                    continue
                notify(f"{b['label']}: problem", f"Automatic orders skipped this run: {err!r}"[:400],
                       quiet=(broker == "capital" and m == "dry"))   # a Capital.com dry run never buzzes the phone
    finally:
        _wants.clear()


def _flush(m, stake, now, broker="ig"):
    import importlib
    B = BROKERS[broker]
    mod = importlib.import_module(B["module"])
    if B["demo_host"] not in mod.BASE:
        raise RuntimeError("not the demo server: refusing to trade")
    lab = B["label"]
    quiet = broker != "ig" and m == "dry"
    say = lambda title, body, tags=("warning",), prio=3: notify(title, body, quiet, tags, prio)
    dry = " (dry run)" if m == "dry" else ""
    EPIC_ = EPIC if broker == "ig" else mod.EPIC
    ig = getattr(mod, B["cls"])()
    sess = ig.login()
    if m == "demo" and hasattr(ig, "ensure_hedging") and ig.ensure_hedging():
        print(f"[{broker}] hedging mode switched on (long and short positions side by side)")
    st = load(m, broker)
    mk = ig.market(EPIC_)
    snap, rules = mk["snapshot"], mk["dealingRules"]
    bid, offer = float(snap["bid"]), float(snap["offer"])
    mid = (bid + offer) / 2
    tradeable = snap.get("marketStatus") == "TRADEABLE"
    step = float(rules["minDealSize"]["value"])
    min_dist = float(rules["minNormalStopOrLimitDistance"]["value"]) + 1
    pos = {p["position"]["dealId"]: p for p in ig.positions() if p["market"].get("epic") == EPIC_}
    on_ig = {str(p["position"].get("dealReference") or "").split("-")[0] for p in pos.values()}
    bal = float(sess.get("accountInfo", {}).get("balance") or 0) + float(sess.get("accountInfo", {}).get("profitLoss") or 0)
    today = f"{now:%y%m%d}"
    st["day"].setdefault(today, bal)
    loss_stop = bal < st["day"][today] * (1 - DAY_LOSS)
    known = {t.get("deal") for t in st["trades"].values()}
    stray = [d for d in pos if d not in known] if broker != "ig" else []
    if stray:
        print(f"[{broker}] {len(stray)} position(s) not in our state file: no new orders this run")
    print(f"[{broker}] mode {m}; US Tech 100 bid {bid} offer {offer} {snap.get('marketStatus')}; {len(pos)} open; "
          f"equity {bal:.2f} (day start {st['day'][today]:.2f})")
    # positions IG closed itself (stop / target hit on IG)
    for ref, t in st["trades"].items():
        if t["status"] == "open" and t.get("deal") not in pos and m == "demo":
            t["status"] = "closed_by_ig"
            lv = [x for x in (t.get("stop"), t.get("target")) if x is not None]
            hit = min(lv, key=lambda x: abs(x - mid)) if lv else mid       # the level the price is nearest to now
            why = "target hit" if hit == t.get("target") else "stop hit" if hit == t.get("stop") else "closed"
            pts = t["side"] * (hit - t["level"])
            eur = pts * t["size"]
            t.update(exit=hit, pts=round(pts, 1), eur=round(eur, 2), closed=str(now))
            say(f"{money(eur)} (about) | {NAMES[t['strat']]} {why} | {lab}",
                f"Result about {money(eur)} ({pts:+.1f} points).\n"
                f"{'Buy' if t['side'] == 1 else 'Short'} from {num(t['level'])} closed by {lab} at its {why.split()[0]} "
                f"(~{num(hit)}). Exact amount: IG history.\n{ref}",
                ("white_check_mark",) if eur >= 0 else ("x",), 4)
    for strat, (wants, scale_px) in _wants.items():
        shift = (mid - scale_px) if scale_px else None
        wanted = {w["ref"]: w for w in wants}
        # close what the bot says is over
        for ref, t in st["trades"].items():
            if t["strat"] != strat or t["status"] != "open" or ref in wanted:
                continue
            p = pos.get(t.get("deal"))
            if not p and m == "demo":
                continue
            if not tradeable:
                print(f"[{broker}] {ref}: should close, market {snap.get('marketStatus')}; next run")
                continue
            lvl = bid - SLIP if t["side"] == 1 else offer + SLIP
            if m == "demo":
                c = ig.close(t["deal"], "SELL" if t["side"] == 1 else "BUY", float(p["position"]["size"]), lvl, ref + "x")
                if c.get("dealStatus") != "ACCEPTED":
                    say(f"{lab}: close refused", f"{ref}: {c.get('reason')}; tries again next run")
                    continue
                exit_ = float(c.get("level") or (bid if t["side"] == 1 else offer))
            else:
                exit_ = bid if t["side"] == 1 else offer
            pts = t["side"] * (exit_ - t["level"])
            t.update(status="closed", exit=exit_, closed=str(now), pts=round(pts, 1), eur=round(pts * t["size"], 2))
            eur = pts * t["size"]
            say(f"{money(eur)} | {NAMES[strat]} CLOSED | {lab}{dry}",
                f"Result {money(eur)} ({pts:+.1f} points).\n"
                f"{'Sold' if t['side'] == 1 else 'Bought back'} {t['size']} at {num(exit_)} "
                f"({'bought' if t['side'] == 1 else 'shorted'} at {num(t['level'])}), {lab}.\n{ref}",
                ("white_check_mark",) if eur >= 0 else ("x",), 4)
        # open new trades / move stops
        for ref, w in wanted.items():
            t = st["trades"].get(ref)
            if t and t["status"] == "open":
                if w.get("stop") is not None and shift is not None and m == "demo" and t.get("deal") in pos:
                    new = round(w["stop"] + t.get("shift", shift), 1)
                    p = pos[t["deal"]]["position"]
                    old = float(p.get("stopLevel") or 0)
                    if old and w["side"] * (new - old) >= 1 and abs(mid - new) >= min_dist:
                        ig.update_stop(t["deal"], new, p.get("limitLevel"))
                        t["stop"] = new
                        print(f"[{broker}] {ref}: stop moved {old:.1f} -> {new:.1f}")   # no phone message (the owner, 11 Nov)
                continue
            if t:                                  # already traded (closed / refused): never again
                continue
            if ref in on_ig:                       # opened by a run whose state was lost: never a second one
                print(f"[{broker}] {ref}: already on {lab}, not opened again")
                continue
            age = (now - w["t_in"]).total_seconds() / 60
            if age > LATE[strat]:
                st["trades"][ref] = dict(strat=strat, side=w["side"], status="too_late", day=today)
                print(f"[{broker}] {ref}: signal {age:.0f} min old (limit {LATE[strat]}), not opened")
                continue
            if not tradeable or loss_stop or stray or sum(1 for x in st["trades"].values() if x["status"] == "open") >= MAX_OPEN:
                why = ("market closed" if not tradeable else "day loss limit" if loss_stop
                       else "unknown position on the account" if stray else "too many open")
                print(f"[{broker}] {ref}: not opened ({why}); next run if still in time")
                continue
            px = offer if w["side"] == 1 else bid
            size = size_for(stake, px, step)
            if size * px > MAX_STAKE_X * stake and size > step:
                print(f"[{broker}] {ref}: size {size} too big for stake {stake}")
                continue
            # IG's smallest size (0.2 = about EUR 5,700 of Nasdaq) is bigger than the practice plan's EUR 1,000: the demo
            # trades the minimum and records 'scale' so results can be shown at the plan's size. Real money: re-check.
            sh = shift if shift is not None else px - w["entry"]
            stop = None if w.get("stop") is None else round(w["stop"] + sh, 1)
            tgt = None if w.get("target") is None else round(w["target"] + sh, 1)
            if stop is not None and w["side"] * (px - stop) < min_dist:
                st["trades"][ref] = dict(strat=strat, side=w["side"], status="skipped_stop_passed", day=today)
                say(f"{lab}: {NAMES[strat]} skipped", f"Price {px:.1f} is already at its stop {stop:.1f} ({ref}).")
                continue
            if tgt is not None and w["side"] * (tgt - px) < min_dist:
                st["trades"][ref] = dict(strat=strat, side=w["side"], status="skipped_target_passed", day=today)
                print(f"[{broker}] {ref}: target {tgt} already reached at {px}")
                continue
            lvl = px + SLIP if w["side"] == 1 else px - SLIP
            if m == "demo":
                c = ig.open(EPIC_, "BUY" if w["side"] == 1 else "SELL", size, lvl, stop, tgt, f"{ref}-{int(time.time()) % 100000}")
                if "WRONG_SIDE_OF_MARKET" in str(c.get("reason")):
                    # the price ran more than SLIP points past our limit while the order travelled (6 Nov, noise
                    # band at 10:00): one retry right away at a fresh quote, if the stop is still far enough
                    s2 = ig.market(EPIC_)["snapshot"]
                    px = float(s2["offer"] if w["side"] == 1 else s2["bid"])
                    lvl = px + SLIP if w["side"] == 1 else px - SLIP
                    print(f"[{broker}] {ref}: price moved past the limit, retry at {px}")
                    if stop is None or w["side"] * (px - stop) >= min_dist:
                        c = ig.open(EPIC_, "BUY" if w["side"] == 1 else "SELL", size, lvl, stop, tgt,
                                    f"{ref}-{int(time.time()) % 100000 + 1}")
                if c.get("dealStatus") != "ACCEPTED":
                    say(f"{lab}: order refused", f"{NAMES[strat]} {ref}: {c.get('reason')}; tries again next run")
                    continue
                fill, deal = float(c.get("level") or px), c.get("dealId")
            else:
                fill, deal = px, None
            st["trades"][ref] = dict(strat=strat, side=w["side"], status="open", day=today,
                                     level=fill, size=size, stop=stop, target=tgt, deal=deal, shift=sh,
                                     opened=str(now), signal=w["entry"], scale=round(stake / (size * px), 4))
            say(f"{'BUY' if w['side'] == 1 else 'SHORT'} {NAMES[strat]} at {num(fill)} | {lab}{dry}",
                f"OPENED: {'bought' if w['side'] == 1 else 'sold short'} {size} US Tech 100 (EUR {size:g} a point) on {lab}.\n"
                f"Stop {num(stop) if stop is not None else 'none'}, target {num(tgt) if tgt is not None else 'none'}.\n"
                f"{ref}, signal {age:.0f} min ago."
                + (f" Minimum size = {size * px / stake:.1f}x the plan's EUR {stake:,.0f}." if size * px > 1.5 * stake else ""),
                ("chart_with_upwards_trend",) if w["side"] == 1 else ("chart_with_downwards_trend",), 4)
    save(st, m, broker)


def ref_of(strat, t):
    """Trade id for IG's dealReference: letters, digits, '-' only."""
    return f"{strat}{pd.Timestamp(t).tz_convert('America/New_York'):%y%m%d%H%M}"


REPORT_HOUR = 23                      # Amsterdam: daily result message (the owner, 11 Nov 2026)


def report(now=None):
    """Once a weekday, on the first bot run from 23:00 Amsterdam: one ntfy message per demo account with the day's result
    (equity now vs the last report, open trades included, funding included) in EUR and %. Never raises."""
    if not enabled:
        return
    now = now or pd.Timestamp.now(tz="Europe/Amsterdam")
    now = pd.Timestamp(now).tz_convert("Europe/Amsterdam")
    if now.weekday() >= 5 or now.hour != REPORT_HOUR:
        return
    for broker, b in BROKERS.items():
        if mode(broker) != "demo":
            continue
        try:
            _report(broker, now)
        except Exception as err:  # noqa
            print(f"[{broker}] daily report failed: {err!r}")


def _report(broker, now):
    import importlib
    B = BROKERS[broker]
    st = load("demo", broker)
    key = f"{now:%y%m%d}"
    rep = st.setdefault("report", {})
    if key in rep:
        print(f"[{broker}] daily report already sent")
        return
    mod = importlib.import_module(B["module"])
    if B["demo_host"] not in mod.BASE:
        raise RuntimeError("not the demo server")
    cl = getattr(mod, B["cls"])()
    info = cl.login().get("accountInfo", {})
    eq = float(info.get("balance") or 0) + float(info.get("profitLoss") or 0)
    ny = f"{now.tz_convert('America/New_York'):%y%m%d}"
    prev = [k for k in rep if k < key]
    ref = rep[max(prev)] if prev else st.get("day", {}).get(ny, eq)   # first report: since the day's first bot run
    day_eur = eq - ref
    pct = day_eur / ref * 100 if ref else 0.0
    opened = [t for t in st["trades"].values() if t.get("status") == "open"] + \
        [dict(t, strat="Y") for t in st.get("yen", {}).values() if t.get("status") == "open"]
    since = (now.normalize().tz_convert("UTC")).strftime("%Y-%m-%dT%H:%M:%S")      # 00:00 Amsterdam
    closed = None
    if hasattr(cl, "closed_since"):                 # the broker's own amounts (IG); else the bot's estimates
        try:
            closed = [dict(name=_name_of(st, c["open"]), eur=c["eur"]) for c in cl.closed_since(since)]
        except Exception as err:  # noqa
            print(f"[{broker}] history not read ({err!r}): bot's estimates")
    if closed is None:
        day = f"{now.tz_convert('America/New_York'):%Y-%m-%d}"
        closed = [dict(name=NAMES[t["strat"]], eur=t["eur"]) for t in st["trades"].values()
                  if str(t.get("closed", ""))[:10] == day and t.get("eur") is not None]
    won = sum(1 for t in closed if t["eur"] >= 0)
    lines = [f"Day result {money(day_eur)} ({pct:+.2f}%).",
             f"Account EUR {eq:,.2f} (was {ref:,.2f}" + (" at the last report)." if prev else " at the day's first run).")]
    if closed:
        tot = sum(t["eur"] for t in closed)
        lines.append(f"Closed today: {len(closed)} trade{'s' * (len(closed) != 1)} ({won} won, {len(closed) - won} lost), "
                     f"{money(tot)}: " + ", ".join(f"{t['name']} {money(t['eur'])}" for t in closed) + ".")
        rest = day_eur - tot
        if abs(rest) >= 0.5:
            lines.append(f"Other {money(rest)}: " + ("open trades' price change and " if opened else "")
                         + "overnight funding / trades open at the day's start.")
    else:
        lines.append("No trades closed today.")
    if opened:
        lines.append("Still open: " + ", ".join(NAMES[t["strat"]] for t in opened)
                     + " (their open result is in the day result).")
    notify(f"{money(day_eur)} ({pct:+.2f}%) today | {B['label']}", "\n".join(lines),
           tags=("moneybag",) if day_eur >= 0 else ("chart_with_downwards_trend",), prio=4)
    rep[key] = round(eq, 2)
    st["report"] = {k: v for k, v in rep.items() if k >= f"{(now - pd.Timedelta(days=30)):%y%m%d}"}
    save(st, "demo", broker)


def _name_of(st, open_level):
    """Strategy name of a closed deal: the state trade with that fill level."""
    for t in st["trades"].values():
        if t.get("level") is not None and abs(float(t["level"]) - open_level) < 0.05:
            return NAMES[t["strat"]]
    for t in st.get("yen", {}).values():          # USD/JPY (desk/yen_week.py): prices around 150, 3 decimals
        if t.get("level") is not None and abs(float(t["level"]) - open_level) < 0.0005:
            return NAMES["Y"]
    return "trade"
