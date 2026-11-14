"""Monday dip (BACKTEST.md round 21b-e), paper strategy in the day bot: alerts for the Trading 212 practice account.

Rules (round 21d 'improved', as backtested in desk/round21.py simulate with weak="ma", r1=50, s1=2.5, target=2, hold=24):
- 1-hour candles on the New York clock from the NQ futures (Yahoo NQ=F, nearly 24 h a day; Trading 212's US Tech 100
  follows the futures, so the prices in the messages need no offset). RSI(14) Wilder, ATR(14) Wilder, MA40 = 40-hour
  simple average of the closes.
- Signal: a candle that starts on a Monday (New York clock, 00:00-23:00) closes below MA40 with RSI under 50 ->
  buy at that close. Stop loss 2.5 x ATR below, take profit 2 x the stop distance above, out at the close of the first
  candle that ends 24 h or more after the entry. One trade at a time; stop first if a candle touches both.
- Skip the signal while an IBS trade is open at that moment (round 21c: Monday-dip trades held next to IBS earned
  nothing). A skipped signal does not block the next hour.
Entry window: Monday 01:00 to Tuesday 00:00 New York (Monday 07:00 to Tuesday 06:00 Amsterdam, an hour earlier in the
weeks the US and EU clocks differ). Exits up to about Wednesday 00:00 New York.

Pure functions (candles in, trades out) so the bot and the paper log share them:
  hourly(nq15, now) -> 1-hour candles with rsi / atr / ma, only those that have fully closed (10 min margin, Yahoo's
  futures data can lag); ibs_open(ndx15) -> IBS trade intervals; trades(h, ibs) -> every trade with how it ended.
Run: python -m desk.monday_dip [--csv journal/paper_monday.csv]  (paper log of the last 60 days, merged)
"""
import argparse
import os

import numpy as np
import pandas as pd

from desk.ticket import ticket

R1, STOP_ATR, TARGET_R, HOLD_H = 50, 2.5, 2.0, 24
COST, NIGHT_FEE = 0.008, 0.02     # % round trip (spread) and % per night held (CFD overnight fee), as round 21
LAG = pd.Timedelta(minutes=10)    # a candle counts once it ended this long ago
NY, AMS = "America/New_York", "Europe/Amsterdam"
COLS = ["date", "time_ny", "time_ams", "entry", "stop", "target", "exit_ny", "exit", "why", "pct", "eur"]


def hourly(nq15, now):
    """Closed 1-hour candles (label = start, NY clock) with RSI(14), ATR(14), MA40 and the weekday."""
    x = nq15[nq15.index + pd.Timedelta(minutes=15) <= now - LAG]
    h = x.resample("60min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h = h[h.index + pd.Timedelta("1h") <= now - LAG]
    c = h.close
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    h["rsi"] = 100 - 100 / (1 + up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean())
    tr = pd.concat([h.high - h.low, (h.high - c.shift()).abs(), (h.low - c.shift()).abs()], axis=1).max(axis=1)
    h["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    h["ma"] = c.rolling(40).mean()
    h["dow"] = h.index.dayofweek
    return h


def ibs_open(ndx15):
    """IBS trades as the bot gives them: [(bought at, sold at or None while still open)], times = candle closes."""
    from desk.bot import IBS_MAX, ibs_days   # late import: the bot imports this module
    d = ibs_days(ndx15)
    out, t_in = [], None
    for dd in d.index:
        t = d["last"][dd] + pd.Timedelta(minutes=15)
        weak = d.ibs[dd] < IBS_MAX
        if t_in is None and weak:
            t_in = t
        elif t_in is not None and not weak:
            out.append((t_in, t))
            t_in = None
    if t_in is not None:
        out.append((t_in, None))
    return out


def held(ibs, t):
    return any(a <= t and (b is None or t < b) for a, b in ibs)


def trades(h, ibs=()):
    """Every Monday-dip trade in the candles. why = stop / target / time, or open (still running: exit = last close).
    Skipped signals (IBS open) come back with why = skip_ibs."""
    o, hi, lo, c = (h[k].to_numpy() for k in ("open", "high", "low", "close"))
    rsi, atr, ma, dow = (h[k].to_numpy() for k in ("rsi", "atr", "ma", "dow"))
    t = h.index
    sig = (dow == 0) & (c < ma) & (rsi < R1) & np.isfinite(atr)
    rows, i, n = [], 50, len(h)
    while i < n:
        if not sig[i]:
            i += 1
            continue
        t_in = t[i] + pd.Timedelta("1h")
        e = c[i]
        stop = e - STOP_ATR * atr[i]
        tgt = e + TARGET_R * (e - stop)
        if held(ibs, t_in):
            rows.append(dict(t_in=t_in, entry=e, stop=stop, target=tgt, t_out=t_in, exit=e, why="skip_ibs"))
            i += 1
            continue
        j, x, why = i + 1, c[-1], "open"
        while j < n:
            if lo[j] <= stop:
                x, why = min(stop, o[j]), "stop"
                break
            if hi[j] >= tgt:
                x, why = max(tgt, o[j]), "target"
                break
            if t[j] - t[i] >= pd.Timedelta(hours=HOLD_H):
                x, why = c[j], "time"
                break
            j += 1
        t_out = t[min(j, n - 1)] + pd.Timedelta("1h")
        rows.append(dict(t_in=t_in, entry=e, stop=stop, target=tgt, t_out=t_out, exit=x, why=why))
        if why == "open":
            break
        i = j + 1
    return rows


def due_by(t_in):
    """When the 24-hour exit comes (the futures pause 17:00-18:00 New York: then the next candle)."""
    t = t_in + pd.Timedelta(hours=HOLD_H)
    if t.hour == 18 and t.minute == 0:   # a candle starting 17:00 does not exist: the 18:00-19:00 candle closes it
        t += pd.Timedelta("1h")
    return t


# ---------- messages (same layout as desk/bot.py message) ----------

def fmt(x):
    return f"{x:,.0f}".replace(",", " ")


def messages(rows, stake, since, now=None):
    """(key, title, body, tags, priority) for every event at or after `since` (an aware time). A buy first seen more
    than 30 minutes after the signal (bot runs missed) is sent as a quiet 'missed, skip it' note instead."""
    out = []
    night = lambda t: not 7 <= t.astimezone(AMS).hour < 23   # quiet push at night (ntfy priority 2: no sound)
    for r in rows:
        t_in, ams_in = r["t_in"], r["t_in"].astimezone(AMS)
        due = due_by(t_in).astimezone(AMS)
        if t_in >= since and r["why"] == "skip_ibs":
            out.append((f"{t_in:%Y%m%d}-md_skip-{t_in:%H%M}", f"Monday dip skipped {ams_in:%H:%M}",
                        f"Monday-dip signal at about {fmt(r['entry'])}, skipped because you hold an IBS trade "
                        "(round 21c: the two together earned nothing). Nothing to do.", ["zzz"], 2))
            continue
        if r["why"] == "skip_ibs":
            continue
        late = (t_in + pd.Timedelta(minutes=30)).astimezone(AMS)
        if t_in >= since and now is not None and pd.Timestamp(now) > t_in + pd.Timedelta(minutes=30):
            out.append((f"{t_in:%Y%m%d}-md_entry-{t_in:%H%M}", f"Monday dip missed ({ams_in:%H:%M}): skip it",
                        f"The bot saw a Monday-dip buy signal at {ams_in:%a %H:%M} (about {fmt(r['entry'])}) only now: "
                        "too late to enter. Skip it; the sell message for it later can be ignored too.", ["zzz"], 2))
        elif t_in >= since:
            out.append((f"{t_in:%Y%m%d}-md_entry-{t_in:%H%M}", f"MONDAY DIP BUY (paper) {ams_in:%H:%M}",
                        ticket(1, stake, r["stop"], r["target"]) +
                        f"Practice account: BUY US Tech 100 now at market (about {fmt(r['entry'])}).\n"
                        f"Stop {r['entry'] - r['stop']:.0f} pts below, target {r['target'] - r['entry']:.0f} pts above.\n"
                        f"Held overnight (CFD night fee). If neither is hit, I tell you to sell at about "
                        f"{due:%a %H:%M}. It is not part of the end-of-day 'close everything'.\n"
                        f"Missed it? Until {late:%H:%M} you can still buy at market with the same stop/target prices, "
                        "unless a stop or target message has come. After that, skip it.\n"
                        "Took it? Tap 'I took it' below: only then does the sell message come.",
                        ["rotating_light"], 2 if night(t_in) else 5))
        t_out = r["t_out"]
        if r["why"] == "open" or t_out < since:
            continue
        res = (r["exit"] - r["entry"]) / r["entry"] * stake
        k = f"{t_in:%Y%m%d}-md_{r['why']}-{t_out:%m%d%H%M}"
        if r["why"] == "time":
            out.append((k, f"MONDAY DIP SELL {t_out.astimezone(AMS):%H:%M}",
                        f"Practice account (if you took it): SELL (close) your Monday-dip trade now at market (about {fmt(r['exit'])}): "
                        f"24 hours are up. Bought {ams_in:%a %H:%M} at about {fmt(r['entry'])}, result about "
                        f"EUR {res:+.2f}.\nMissed it? Sell as soon as you see this.",
                        ["rotating_light"], 3 if night(t_out) else 5))
        else:
            word = "target hit" if r["why"] == "target" else "stopped out"
            out.append((k, f"Monday dip {word} {t_out.astimezone(AMS):%H:%M}",
                        f"The Monday-dip trade should now be closed by its {'take profit' if r['why'] == 'target' else 'stop'} "
                        f"at about {fmt(r['exit'])} (about EUR {res:+.2f}). Check that it is closed.",
                        ["white_check_mark" if res > 0 else "x"], 3))
    return out


def window(now):
    """True while a Monday-dip message can come: Monday 01:00 to Wednesday 02:00 New York."""
    wd, hr = now.weekday(), now.hour
    return (wd == 0 and hr >= 1) or wd == 1 or (wd == 2 and hr < 2)


def week_start(now):
    """Monday 00:00 New York of the week `now` is in (messages before it are old)."""
    d = pd.Timestamp(now).tz_convert(NY).normalize()
    return d - pd.Timedelta(days=d.weekday())


# ---------- paper log ----------

def nights_held(r):
    """Nights for the CFD fee, counted as the backtest does (calendar days between the entry and exit candles' starts)."""
    a, b = (x - pd.Timedelta("1h") for x in (r["t_in"], r["t_out"]))
    return (b.tz_localize(None).normalize() - a.tz_localize(None).normalize()).days


def table(rows, stake=1000.0):
    out = []
    for r in rows:
        if r["why"] == "skip_ibs":
            continue
        nights = nights_held(r)
        pct = (r["exit"] - r["entry"]) / r["entry"] * 100 - COST - NIGHT_FEE * nights
        out.append(dict(date=f"{r['t_in']:%Y-%m-%d}", time_ny=f"{r['t_in']:%H:%M}",
                        time_ams=f"{r['t_in'].astimezone(AMS):%H:%M}", entry=round(r["entry"], 1),
                        stop=round(r["stop"], 1), target=round(r["target"], 1), exit_ny=f"{r['t_out']:%a %H:%M}",
                        exit=round(r["exit"], 1), why=r["why"], pct=round(pct, 3), eur=round(pct / 100 * stake, 2)))
    return pd.DataFrame(out, columns=COLS)


def main():
    from desk.bot import yahoo
    from desk.plan import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "paper_monday.csv"))
    a = ap.parse_args()
    now = pd.Timestamp.now(tz=NY)
    h = hourly(yahoo("NQ=F", "15m", "60d", prepost=True), now)
    ibs = ibs_open(yahoo("^NDX", "15m", "60d"))
    t = table(trades(h, ibs))
    t = t[t.date >= f"{h.index[0] + pd.Timedelta(days=7):%Y-%m-%d}"]   # first week = indicator warm-up
    if os.path.exists(a.csv) and os.path.getsize(a.csv):
        old = pd.read_csv(a.csv, dtype=str)
        old = old[old.date < (t.date.min() if len(t) else "9999")]
        t = pd.concat([old, t.astype(str)], ignore_index=True)
    t.to_csv(a.csv, index=False)
    done = t[t.why != "open"]
    print(t.to_string(index=False))
    print(f"\n{len(done)} closed trades, EUR {pd.to_numeric(done.eur).sum():+.2f} at EUR 1,000 per trade")


if __name__ == "__main__":
    main()
