"""Free alert bot for the three Nasdaq-100 strategies (noise band, overnight sweep, IBS) + the Monday dip (paper).

Runs every 15 minutes during the US session (GitHub Actions, see .github/workflows/daybot.yml), computes the
signals from free Yahoo data and pushes them to the phone with ntfy (https://ntfy.sh, free app, no account).

Prices: the signals use the Nasdaq-100 cash index (^NDX, real time on Yahoo), exactly as backtested. The overnight
high/low comes from the NQ futures (^NDX does not trade at night), shifted by the futures-minus-index gap
measured today. Trading 212's US Tech 100 is priced like the futures (checked 2026-10-09: T212 31 108, index
30 882, futures gap 226-233), so by default every price in the messages has today's measured gap added
(T212_OFFSET=auto). A number in T212_OFFSET (points) replaces the gap.

The rules are exactly the backtested ones (desk/library.py s_noise and s_sweep, 15-minute candles):
- Noise band (long only): at each :00/:30 close from 10:00 to 15:30 NY, buy if the close is above
  max(open, previous close) x (1 + 14-day average move at that time). Stop 0.3% below the entry; sell on a
  :00/:30 close below the day's open, or at 16:00. After a stop it may buy again at a later check.
- Overnight sweep: the first 15m candle (9:30-15:30 NY) that wicks through the overnight high (low) and closes
  back inside = short (long). Stop at that candle's high (low), target 2.5x the stop distance, out at 16:00.
  No sweep trade on scheduled Fed decision days (FOMC_DAYS; round 11: PF 0.37 / 0.89 on those days, the 14:00
  decision runs through the stops).
- IBS (round 9, desk/ibs.py; held overnight): at 15:45 NY, IBS = (price - day low) / (day high - day low) over the
  9:30-15:45 candles. Below 0.2 = buy at that price. Next session at 15:45: sell if IBS >= 0.2, else keep it.
  No stop and no target (a stop cuts the bounce: round 9). Half days have no 15:45 decision (the trade carries on).
  Missed IBS buys: the IBS BUY message has an 'I missed it' button (posts `<YYYYMMDD>-ibs` to `<NTFY_TOPIC>-taken`);
  buy dates can also go in IBS_MISSED_FILE (journal/ibs_missed.txt). For such a trade the bot sends no keep / sell
  messages, does not say 'you hold an IBS trade', and the Monday dip is not skipped for it.

- Session Momentum (paper, practice account only; rounds 23 / 25, desk/session_momentum.py): 9:35 NY close vs the
  8 EMA of 5-minute futures closes -> buy / short, stop 1.2% trailing 1.2% behind the best close, out by 16:00.
  Checked from 9:35 NY; the 9:46 run sends it (10-15 min late costs little: round 25b).
- Monday dip (paper, practice account only; rounds 21b-e, desk/monday_dip.py): Mondays, 1-hour futures candles,
  close below the 40-hour average and RSI(14) < 50 -> buy, stop 2.5 ATR, target 2R, out after 24 h, skipped while an
  IBS trade is open. Checked on every run from Monday 01:00 to Wednesday 02:00 New York (also outside the session).
- Overnight Bias ORB (paper, practice account only, like Session Momentum; rounds 27 / 27b, desk/overnight_bias.py): 9:30 open in the top /
  bottom third of the overnight range -> only longs / shorts; 5-minute close through the 9:30-9:45 range while ADX(14)
  15m > 20; stop the other side, target 3R, out at 15:00; up to 4 a day. Checked 9:45 - 15:30 NY.

Every run recomputes the whole day from the candles. The only memory is the list of messages already sent
(STATE_FILE, kept between runs by the GitHub Actions cache), so a message is never sent twice and a late or
skipped run still sends what it missed.

Position size: desk/plan.py (journal/plan.json): EUR per trade for this month, practice = fixed, real money =
leverage x the account balance on the 1st of the month (gains reinvested). STAKE_EUR is only the fallback.

Env: NTFY_TOPIC (required to send), T212_OFFSET (default auto = today's futures gap), STAKE_EUR (fallback when no
plan phase is active, default 1000), STATE_FILE (default .daybot/sent.json), PHONE_ALERTS (on | off: off = only the
IG demo messages, since 8 Nov 2026).
Sunday from 19:00 Amsterdam: one weekly reminder to send Claude the Trading 212 screenshots (sunday_reminder).
Alarm: every run schedules the next session's "get ready" message for 9:40 New York (15:40 Amsterdam, 14:40 in the
weeks the US and EU clocks differ: end Oct, March), 5 minutes before the first possible sweep alert, with ntfy's delayed
delivery (it arrives on the minute although the bot runs every 15 minutes). Skips weekends and NYSE_HOLIDAYS.
Run: python -m desk.bot [--dry] [--replay YYYY-MM-DD]
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from desk import autotrade as at
from desk import monday_dip as md
from desk import overnight_bias as ob
from desk import session_momentum as sm
from desk import yen_week
from desk.plan import stake_for
from desk.market_days import close_of, short_by
from desk.ticket import ticket, copy_actions, click_url

NY = ZoneInfo("America/New_York")
AMS = ZoneInfo("Europe/Amsterdam")
UA = {"User-Agent": "Mozilla/5.0"}
OPEN, LAST_ENTRY, CLOSE = 570, 930, 960  # minutes after midnight NY
LOOKBACK, RR, NSTOP = 14, 2.5, 0.003  # round 6 tuning: sweep target 2.5R, noise stop 0.3%
IBS_AT, IBS_MAX = 945, 0.2  # round 9: IBS decided at 15:45 NY, buy below 0.2
IBS_MISSED_FILE = "journal/ibs_missed.txt"  # NY dates of IBS buys the owner did not take, one per line
IBS_TAP = re.compile(r"^(\d{8})-ibs$")      # body posted by the IBS BUY message's 'I missed it' button
DESK_URL = "https://claude.ai/artifact/92JTuBNkJjmGjPX8NTBqHv"  # the Day Desk logbook page (desk/logbook.py)
ALARM_AT = 580  # 9:40 NY: the "get ready" alarm, 5 minutes before the first sweep alert can come
# NYSE full-day closures (no session, so no alarm). Exchange calendar; add each new year's.
NYSE_HOLIDAYS = {dt.date.fromisoformat(d) for d in (
    "2026-01-01 2026-01-19 2026-02-16 2026-04-03 2026-05-25 2026-06-19 2026-07-03 2026-09-07 2026-11-26 2026-12-25 "
    "2027-01-01 2027-01-18 2027-02-15 2027-03-26 2027-05-31 2027-06-18 2027-07-05 2027-09-06 2027-11-25 2027-12-24").split()}
# Scheduled Fed decision days (statement 14:00 NY): no sweep trade (round 11). Fed calendar; add each new year's.
FOMC_DAYS = {dt.date.fromisoformat(d) for d in (
    "2026-01-28 2026-03-18 2026-04-29 2026-06-17 2026-07-29 2026-09-16 2026-10-28 2026-12-09 "
    "2027-01-27 2027-03-17 2027-04-28 2027-06-09 2027-07-28 2027-09-15 2027-10-27 2027-12-08 2028-01-26").split()}


# ---------- data ----------

def yahoo(symbol, interval, range_, prepost=False):
    r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", headers=UA, timeout=20,
                     params={"interval": interval, "range": range_, "includePrePost": str(prepost).lower()})
    r.raise_for_status()
    j = r.json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                      index=pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(NY))
    df = df.dropna()
    return df[~df.index.duplicated(keep="last")]


def eurusd():
    try:
        return float(yahoo("EURUSD=X", "1d", "5d").close.iloc[-1])
    except Exception:
        return 1.10


def complete(df, minutes, now):
    """Drop candles that have not closed yet."""
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]


def tod(ix):
    return ix.hour * 60 + ix.minute


def rth(df):
    t = tod(df.index)
    return df[(t >= OPEN) & (t + 15 <= CLOSE)]


def basis(pairs, day):
    """Futures minus cash index: median over the candles both have today (skipping the stale first 2 minutes),
    trying each (futures, index) pair in turn (1-minute first, then 15-minute); else the last session's end."""
    prev = None
    for nq, ndx in pairs:
        j = nq.close.to_frame("nq").join(ndx.close.rename("ndx"), how="inner")
        t = tod(j.index)
        j = j[(t >= OPEN + 2) & (t < CLOSE)]
        today = j[j.index.date == day]
        if len(today):
            return float((today.nq - today.ndx).median())
        before = j[j.index.date < day]
        if prev is None and len(before):
            prev = float((before.nq - before.ndx).tail(8).median())
    return prev if prev is not None else np.nan


# ---------- signals (pure: candles in, events out) ----------

def overnight(nq15, day):
    """Overnight high/low of the futures: 18:00 the evening before up to 9:30 NY (trade-date convention)."""
    tdate = (nq15.index.tz_localize(None) + pd.Timedelta(hours=6)).normalize().date
    t = tod(nq15.index)
    on = nq15[(tdate == day) & ((t >= 18 * 60) | (t + 15 <= OPEN))]
    return (float(on.high.max()), float(on.low.min())) if len(on) else (np.nan, np.nan)


def noise_line(ndx15, day):
    """Per 30-minute check time: the upper band for `day` = max(open, previous close) x (1 + avg move)."""
    r = rth(ndx15)
    c = r.close.to_frame().assign(d=r.index.date, slot=tod(r.index)).pivot_table(index="d", columns="slot",
                                                                                  values="close")
    full = c[c.notna().sum(axis=1) == 26]  # full sessions only (26 candles), as in the backtest
    op = r.open.groupby(r.index.date).first()
    hist = full[full.index < day].tail(LOOKBACK)
    if len(hist) < LOOKBACK or day not in op.index:
        return None, np.nan, np.nan
    sig = (hist.div(op.reindex(hist.index), axis=0) - 1).abs().mean()
    prev = r[r.index.date < day]
    pc = float(prev.close.iloc[-1])
    o = float(op[day])
    return sig.add(1).mul(max(o, pc)), o, pc


def day_events(ndx15, onh, onl, line, o, day):
    """Replay the day's completed RTH candles and return the trade events, oldest first."""
    r = rth(ndx15)
    r = r[r.index.date == day]
    ev = []
    pos_n, n_entry, n_stop = False, 0.0, 0.0
    sw, sw_done = None, day in FOMC_DAYS  # open sweep trade dict; no sweep on Fed decision days
    for ts, b in r.iterrows():
        start, end = tod(ts), tod(ts) + 15
        last = end == close_of(day)          # 13:00 on a half day (desk/market_days.py)
        # sweep: manage the open trade (stop first if both are touched, like the backtest)
        if sw:
            s = sw["side"]
            worst, best = (min, max) if s == 1 else (max, min)  # a gap through a level fills at the open
            if (s == 1 and b.low <= sw["stop"]) or (s == -1 and b.high >= sw["stop"]):
                ev.append(dict(kind="sweep_stop", ts=ts, px=worst(sw["stop"], b.open), trade=sw)); sw = None
            elif (s == 1 and b.high >= sw["target"]) or (s == -1 and b.low <= sw["target"]):
                ev.append(dict(kind="sweep_target", ts=ts, px=best(sw["target"], b.open), trade=sw)); sw = None
            elif last:
                ev.append(dict(kind="sweep_close", ts=ts, px=b.close, trade=sw)); sw = None
        if not sw_done and start >= OPEN and end <= LAST_ENTRY - short_by(day) and not np.isnan(onh):
            for side, hit in ((1, b.low < onl and b.close > onl), (-1, b.high > onh and b.close < onh)):
                if not hit:
                    continue
                stop = b.low if side == 1 else b.high
                dist = side * (b.close - stop)
                if dist <= 0:
                    continue
                sw = dict(side=side, entry=b.close, stop=stop, target=b.close + side * RR * dist, dist=dist,
                          level=onl if side == 1 else onh)
                sw_done = True
                ev.append(dict(kind="sweep_entry", ts=ts, px=b.close, trade=sw))
                break
        # noise band: the stop works on every candle, the checks at :00/:30 closes from 10:00
        n_stopped = pos_n and b.low <= n_stop
        if n_stopped:
            ev.append(dict(kind="noise_stop", ts=ts, px=min(n_stop, b.open), entry=n_entry)); pos_n = False
        if line is not None and end % 30 == 0 and end >= OPEN + 30 and not n_stopped:
            if pos_n and b.close < o:
                ev.append(dict(kind="noise_exit", ts=ts, px=b.close, entry=n_entry)); pos_n = False
            elif not pos_n and end <= LAST_ENTRY - short_by(day) and start in line.index and b.close > line[start]:
                pos_n, n_entry, n_stop = True, b.close, b.close * (1 - NSTOP)
                ev.append(dict(kind="noise_entry", ts=ts, px=b.close, line=float(line[start])))
        if last and pos_n:
            ev.append(dict(kind="noise_close", ts=ts, px=b.close, entry=n_entry)); pos_n = False
    return ev, pos_n or bool(sw)


def ibs_days(ndx15):
    """IBS at 15:45 NY per session: (price - day low) / (day high - day low) over the 9:30-15:45 candles. Only
    sessions that have the 15:30 candle and at least 20 of the 25 candles (half days are skipped)."""
    r = rth(ndx15)
    r = r[tod(r.index) + 15 <= IBS_AT]
    g = r.groupby(r.index.date)
    d = pd.DataFrame({"h": g.high.max(), "l": g.low.min(), "p": g.close.last(), "n": g.size(),
                      "last": pd.Series(r.index, index=r.index).groupby(r.index.date).last()})
    d = d[(d.n >= 20) & (tod(pd.DatetimeIndex(d["last"])) == IBS_AT - 15)]
    d["ibs"] = (d.p - d.l) / (d.h - d.l)
    return d


def ibs_events(ndx15, day):
    """Today's IBS decision (at 15:45 NY) and the trade held from earlier sessions.
    Returns (events, held): held = (entry date, entry price) of an IBS trade bought before today, else None."""
    d = ibs_days(ndx15)
    weak = d.ibs < IBS_MAX
    held = None
    for dd in reversed([x for x in d.index if x < day]):  # a trade stays open while every session closes weak
        if not weak[dd]:
            break
        held = (dd, float(d.p[dd]))
    if day not in d.index:
        return [], held
    e = dict(ts=d["last"][day], px=float(d.p[day]), ibs=float(d.ibs[day]))
    if held:
        return [dict(e, kind="ibs_hold" if weak[day] else "ibs_exit", since=held[0], entry=held[1])], held
    return ([dict(e, kind="ibs_entry")] if weak[day] else []), held


def ibs_taken(ndx15, day, missed=()):
    """ibs_events, minus the trade the owner did not take (its buy date in `missed`): no keep / sell, not held."""
    iev, held = ibs_events(ndx15, day)
    if held and held[0] in missed:
        return [], None
    return iev, held


def ibs_missed(taps=(), path=None):
    """NY dates of IBS buys not taken: IBS_MISSED_FILE (YYYY-MM-DD per line, # comments) + 'I missed it' taps."""
    out = set()
    try:
        with open(path or IBS_MISSED_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.split("#")[0].strip()
                if line:
                    out.add(dt.date.fromisoformat(line))
    except OSError:
        pass
    for t in taps:
        m = IBS_TAP.match(str(t).strip())
        if m:
            out.add(dt.datetime.strptime(m.group(1), "%Y%m%d").date())
    return out


# ---------- messages ----------

def fmt(x, off):
    return f"{x + off:,.0f}".replace(",", " ")


LATE = ("\nMissed it? Until {until} you can still enter at market with the same {what}, unless a stop, "
        "target or sell message for this trade has come. After that, skip it.")


def late_until(t):
    """Latest time to still take a missed entry: 30 minutes after the alert (backtest: a 30-minute late entry
    kept about the same profit per trade), and never after the last entry time."""
    end = t.normalize() + pd.Timedelta(minutes=LAST_ENTRY)
    return min(t + pd.Timedelta(minutes=30), end).astimezone(AMS).strftime("%H:%M")


def message(e, off, fx, stake):
    """(key, title, body, tags) for one event."""
    t = e["ts"] + pd.Timedelta(minutes=15)
    when = t.astimezone(AMS).strftime("%H:%M")
    k = f"{t:%Y%m%d}-{e['kind']}-{t:%H%M}"
    px = e["px"]
    if e["kind"] == "sweep_entry":
        tr = e["trade"]
        buy = tr["side"] == 1
        word = "BUY" if buy else "SELL (short)"
        body = (ticket(tr["side"], stake, tr["stop"] + off, tr["target"] + off) +
                f"{word} US Tech 100 now at market (about {fmt(px, off)}).\n"
                f"Stop {tr['dist']:.0f} pts {'below' if buy else 'above'}, target {RR * tr['dist']:.0f} pts "
                f"{'above' if buy else 'below'}.\n"
                + LATE.format(until=late_until(t), what="stop/target prices"))
        return k, f"SWEEP {'BUY' if buy else 'SHORT'} {when}", body, ["rotating_light"]
    if e["kind"] in ("sweep_stop", "sweep_target"):
        tr = e["trade"]
        res = tr["side"] * (px - tr["entry"]) / tr["entry"] * stake
        word = "target hit" if e["kind"] == "sweep_target" else "stopped out"
        return (k, f"Sweep {word} {when}",
                f"The sweep trade should now be closed by its {'take profit' if 'target' in e['kind'] else 'stop'} "
                f"at {fmt(px, off)} (about EUR {res:+.2f}). Check that it is closed; nothing else to do today.",
                ["white_check_mark" if res > 0 else "x"])
    if e["kind"] == "ibs_entry":
        late = (t + pd.Timedelta(minutes=30)).astimezone(AMS).strftime("%H:%M")
        return (k, f"IBS BUY {when}",
                ticket(1, stake) +
                f"BUY US Tech 100 now at market (about {fmt(px, off)}). The Nasdaq is near its low of the day "
                f"(IBS {e['ibs']:.2f}); it tends to bounce by tomorrow evening.\n"
                f"No stop loss and no take profit: keep it overnight. Tomorrow at about {when} I tell you to sell "
                f"it or keep it.\nMissed it? You can still buy at market until {late}. After that, skip it and tap "
                f"'I missed it' below: then no keep / sell messages come for it.",
                ["rotating_light"])
    if e["kind"] in ("ibs_hold", "ibs_exit"):
        res = (px - e["entry"]) / e["entry"] * stake
        since = f"Bought {e['since']:%a %d %b} at about {fmt(e['entry'], off)}"
        if e["kind"] == "ibs_hold":
            return (k, f"IBS: keep it {when}",
                    f"Keep your IBS trade open for another day: the Nasdaq closed weak again (IBS {e['ibs']:.2f}). "
                    f"{since}, now about {fmt(px, off)} (about EUR {res:+.2f}). Next check tomorrow at about {when}.",
                    ["hourglass"])
        return (k, f"IBS SELL {when}",
                f"SELL (close) your IBS trade now at market (about {fmt(px, off)}). {since}: result about "
                f"EUR {res:+.2f}.\nMissed it? Sell as soon as you see this.", ["rotating_light"])
    if e["kind"] == "noise_entry":
        return (k, f"NOISE BUY {when}",
                ticket(1, stake, px * (1 - NSTOP) + off) +
                f"BUY US Tech 100 now at market (about {fmt(px, off)}).\n"
                f"Stop {px * NSTOP:.0f} pts below. No take profit: I tell you when to sell.\n"
                + LATE.format(until=late_until(t), what="stop price"),
                ["rotating_light"])
    if e["kind"] == "noise_exit":
        res = (px - e["entry"]) / e["entry"] * stake
        return (k, f"NOISE SELL {when}",
                f"SELL (close) your noise-band trade now at market (about {fmt(px, off)}). Result about EUR {res:+.2f}.",
                ["rotating_light"])
    if e["kind"] == "noise_stop":
        res = (px - e["entry"]) / e["entry"] * stake
        return (k, f"Noise stopped out {when}",
                f"The noise-band trade should now be closed by its stop at {fmt(px, off)} (about EUR {res:+.2f}). "
                f"Check that it is closed. A new NOISE BUY can still come later today.", ["x"])
    if e["kind"] == "noise_close":
        res = (px - e["entry"]) / e["entry"] * stake
        return (k, f"Noise trade: day over {when}",
                f"The market has closed. Your noise-band trade should be closed (the 'close everything' message "
                f"came at the last candle). Closing price {fmt(px, off)}, about EUR {res:+.2f}.", ["bell"])
    if e["kind"] == "sweep_close":
        tr = e["trade"]
        res = tr["side"] * (px - tr["entry"]) / tr["entry"] * stake
        return (k, f"Sweep trade: day over {when}",
                f"The market has closed. Your sweep trade should be closed (the 'close everything' message came at "
                f"the last candle). Closing price {fmt(px, off)}, about EUR {res:+.2f}.", ["bell"])
    raise ValueError(e["kind"])


# ---------- ntfy ----------

def load_sent(path):
    try:
        with open(path, encoding="utf-8") as f:
            return set(json.load(f))
    except (OSError, ValueError):
        return set()


def save_sent(path, keys, day):
    # an IBS 'I missed it' tap is kept 60 days (that trade can stay open for weeks); everything else 7 days
    keep = sorted(k for k in keys if k[:8] >= f"{day - dt.timedelta(days=60 if k.endswith('-ibs_taken') else 7):%Y%m%d}")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(keep, f)


def push(topic, key, title, body, tags, priority=4, at=None, took=None, label="I took it"):
    """Send now, or with `at` (an aware datetime, at most 3 days ahead) let ntfy deliver it at that moment.
    `took` = a trade id: adds a button (`label`: 'I took it' for paper trades, 'I missed it' for IBS buys) that posts
    the id to the topic `<topic>-taken`.
    PHONE_ALERTS=off (GitHub variable): the manual-trading messages stop (trade alerts, alarm, levels, Sunday reminder);
    the bot still runs every strategy and the IG demo messages (autotrade.notify) still come."""
    if os.environ.get("PHONE_ALERTS", "on").strip().lower() == "off" and key != "test":
        print(f"(phone alert off) {title}")
        return
    h = {"Title": title.encode("utf-8"), "Tags": ",".join(tags), "Priority": str(priority)}
    if at is not None:
        h["At"] = str(int(at.timestamp()))
    if click_url(body):
        h["Click"] = click_url(body)           # tap the notification -> Trading 212
    acts = copy_actions(body).split("; ") if copy_actions(body) else []
    if took:                                   # ntfy allows 3 buttons: 'I took it' first, then TP / SL before Value
        acts = [f"http, {label}, https://ntfy.sh/{topic}-taken, method=POST, body={took}"] + (
            [a for a in acts if not a.startswith("copy, Value")] if len(acts) > 2 else acts)
    acts = "; ".join(acts[:3])
    if acts:
        h["Actions"] = acts                    # 'I took it' + tap-to-copy buttons for the order ticket numbers
    r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), timeout=20, headers=h)
    if r.status_code == 400 and acts:          # buttons refused: never lose the alert over them
        h.pop("Actions")
        r = requests.post(f"https://ntfy.sh/{topic}", data=body.encode("utf-8"), timeout=20, headers=h)
    r.raise_for_status()


# ---------- run ----------

def sunday_reminder(now_ams):
    """(key, title, body, tags) on Sunday from 19:00 Amsterdam (GitHub's schedule can be hours late), else None."""
    if now_ams.weekday() != 6 or now_ams.hour < 19:
        return None
    body = ("Put these Trading 212 screenshots (practice account) in the Day Desk Inbox:\n"
            "1. History > Result: every trade down to the last one already in the logbook\n"
            "2. History > Orders: the same days (size, stop and target of each order)\n"
            "3. The account screen: account value, cash, open positions\n"
            f"{DESK_URL}#inbox\nThen ask Claude for the weekly update in a chat.")
    first = (now_ams.date().replace(day=1) + dt.timedelta(days=32)).replace(day=1)
    if (first - now_ams.date()).days <= 7:
        body += (f"\nA new month starts on {first:%a %d %b}: the balance is needed now, the bot sets next month's "
                 "trade size from it.")
    return f"{now_ams:%Y%m%d}-weekly", "Weekly: screenshots for Claude", body, ["camera"]


def next_alarm(now_ny):
    """(key, when, title, body, tags) for the next session's 9:40 NY alarm, or None when the next session is more than
    3 days away (ntfy's limit; a later run schedules it). Today's counts until 9:35, so ntfy gets it in time."""
    day = now_ny.date()
    if now_ny.hour * 60 + now_ny.minute > ALARM_AT - 5:
        day += dt.timedelta(days=1)
    while day.weekday() >= 5 or day in NYSE_HOLIDAYS:
        day += dt.timedelta(days=1)
    when = dt.datetime.combine(day, dt.time(ALARM_AT // 60, ALARM_AT % 60), NY)
    # real hours, not wall-clock: subtracting two New York times ignores a clock change in between (1 hour too few)
    if when.astimezone(dt.timezone.utc) - now_ny.astimezone(dt.timezone.utc) > dt.timedelta(days=3):
        return None
    first = dt.datetime.combine(day, dt.time(9, 45), NY).astimezone(AMS)
    body = (f"The first sweep alert can come at {first:%H:%M}. Open Trading 212 on the US Tech 100, check the order "
            f"screen is set to the euro value (not units), and keep the phone close for the next 30 minutes.")
    if day in FOMC_DAYS:
        body += " Fed decision today: no sweep trade, only noise band and IBS."
    return f"{day:%Y%m%d}-alarm", when, "Get ready: market just opened", body, ["alarm_clock"]


def build(now, day, data=None, missed=()):
    """Everything the bot knows at `now` for session `day`: (messages, info dict). `missed` = buy dates of IBS trades
    not taken (ibs_missed): no keep / sell messages for those, and not reported as held."""
    d = data or dict(ndx15=yahoo("^NDX", "15m", "60d"), nq15=yahoo("NQ=F", "15m", "60d", prepost=True),
                     ndx1=yahoo("^NDX", "1m", "5d"), nq1=yahoo("NQ=F", "1m", "5d", prepost=True), fx=eurusd())
    ndx15 = complete(d["ndx15"], 15, now)
    nq15 = complete(d["nq15"], 15, now)
    gap = basis([(complete(d["nq1"], 1, now), complete(d["ndx1"], 1, now)), (nq15, ndx15)], day)
    fh, fl = overnight(nq15, day)
    onh, onl = fh - gap, fl - gap
    line, o, pc = noise_line(ndx15, day)
    ev, open_ = day_events(ndx15, onh, onl, line, o, day)
    iev, held = ibs_taken(ndx15, day, missed)
    ev = sorted(ev + iev, key=lambda e: e["ts"])
    ndx1 = complete(d["ndx1"], 1, now)
    info = dict(onh=onh, onl=onl, basis=gap, open=o, prev_close=pc, line=line, open_trade=open_,
                ibs_raw=ibs_events(ndx15, day), idx_px=float(ndx1.close.iloc[-1]) if len(ndx1) else None,
                ibs_held=held, ibs_keep=any(e["kind"] in ("ibs_entry", "ibs_hold") for e in iev),
                last_candle=str(ndx15.index[-1]) if len(ndx15) else None, fx=d["fx"])
    return ev, info


def scale_now(now, day):
    """'Where is the futures price NOW' for the broker orders (levels are shifted by broker mid - this). Yahoo's NQ=F
    runs ~10 min late (the 15m close even more), which put demo stops ~140 points off after a fast move (18 Nov 2026).
    In the session ^NDX is near real time: its last 1m close + the basis; outside it the last 1m futures close."""
    nq5, ndx5, nq1, ndx1, gap = five_minute(now, day)
    ndx = complete(ndx1, 1, now)
    if len(ndx) and np.isfinite(gap) and pd.Timestamp(now) - ndx.index[-1] <= pd.Timedelta(minutes=5):
        return float(ndx.close.iloc[-1]) + gap
    nqc = complete(nq1, 1, now)
    return float(nqc.close.iloc[-1]) if len(nqc) else None


def _scale_or(now, day, nq15):
    """scale_now, or the last complete 15m futures close if the 1m candles fail (the old way)."""
    try:
        if day is not None:
            return scale_now(now, day)
    except Exception as err:  # noqa: Yahoo hiccup
        print(f"1-minute price not found ({err!r}); using the 15m futures close")
    nqc = complete(nq15, 15, now)
    return float(nqc.close.iloc[-1]) if len(nqc) else None


def monday(now, stake, since, missed=(), day=None):
    """Monday-dip paper alerts (desk/monday_dip.py): (messages, open trade or None). IBS trades bought on a `missed`
    date do not block it (the owner does not hold them)."""
    nq = yahoo("NQ=F", "15m", "60d", prepost=True)
    h = md.hourly(nq, now)
    ibs = [(a, b) for a, b in md.ibs_open(complete(yahoo("^NDX", "15m", "60d"), 15, now)) if a.date() not in missed]
    rows = md.trades(h, ibs)
    open_ = next((r for r in rows if r["why"] == "open"), None)
    at.want("M", [dict(ref=at.ref_of("M", open_["t_in"]), side=1, entry=open_["entry"], t_in=open_["t_in"],
                       stop=open_["stop"], target=open_["target"])] if open_ else [],
            _scale_or(now, day, nq) if open_ else None)
    return md.messages(rows, stake, since, now), open_


# ---------- paper strategies: follow-ups only for trades the owner tapped 'I took it' on ----------

PAPER_ENTRY = re.compile(r"^(\d{8}-(?:sm|md|ob\d))_entry")          # 20261102-sm_entry, -md_entry-0200, -ob1_entry
PAPER_FOLLOW = re.compile(r"^(\d{8}-(?:sm|md|ob\d))_(?:trail\d+|stop|close|time|target)\b")


def paper_id(key):
    """(trade id, is_entry) of a Session Momentum / Monday dip / Overnight Bias message key, or (None, False)."""
    m = PAPER_ENTRY.match(key)
    if m:
        return m.group(1), True
    m = PAPER_FOLLOW.match(key)
    return (m.group(1), False) if m else (None, False)


def taken(topic, done):
    """Trade ids tapped 'I took it' on: the ntfy topic `<topic>-taken` (ntfy keeps 12 h) + the ones saved before."""
    got = {k[:-len("_taken")] for k in done if k.endswith("_taken")}
    # Monday dips sent before the button existed (no '_button' mark): keep their sell message, as before (a Session
    # Momentum trade lasts one day, so an old one simply gets no more messages)
    got |= {paper_id(k)[0] for k in done if paper_id(k)[1] and "-md" in k and f"{paper_id(k)[0]}_button" not in done}
    try:
        r = requests.get(f"https://ntfy.sh/{topic}-taken/json", params={"poll": "1", "since": "all"}, timeout=20)
        r.raise_for_status()
        for line in r.text.splitlines():
            m = json.loads(line)
            if m.get("event") == "message":
                got.add(m.get("message", "").strip())
    except (requests.RequestException, ValueError) as err:
        print(f"'I took it' taps not read ({err!r}); using the saved ones")
    return got


def send_paper(msgs, topic, live, day):
    """Send the new messages. Entry messages get an 'I took it' button; stop / trail / sell messages only go out for a
    trade whose button was tapped (no Trading 212 API, so the tap is how the bot knows). Returns the taken trade ids
    (in a dry run / replay: every id, all messages printed)."""
    state = os.environ.get("STATE_FILE") or ".daybot/sent.json"
    done = load_sent(state) if live else set()
    got = taken(topic, done) if live else None
    for key, title, body, tags, prio in msgs:
        tid, entry = paper_id(key)
        if key in done:
            continue
        if tid and not entry and got is not None and tid not in got:
            print(f"(not sent, 'I took it' not tapped) {title}")
            continue
        print(f"--- {title}\n{body}")
        if live:
            button = tid if entry and "missed" not in title.lower() else None
            push(topic, key, title, body, tags, priority=prio, took=button)
            done.add(key)
            if button:
                done.add(f"{tid}_button")
            save_sent(state, done, day)
    if live:
        new = {f"{t}_taken" for t in got} - done
        if new:
            save_sent(state, done | new, day)
    return got


def send_monday(now, day, topic, live, fallback, missed=()):
    """Run the Monday dip and send what is new; never lets an error stop the other alerts. Returns the open trade (only
    if 'I took it' was tapped)."""
    try:
        stake = stake_for(day, default=fallback)[0]
        msgs, open_ = monday(pd.Timestamp(now), stake, md.week_start(now), missed, day)
    except Exception as err:  # noqa: Yahoo hiccup etc.; the next run tries again
        print(f"Monday dip not checked ({err!r})")
        return None
    got = send_paper(msgs, topic, live, day)
    print(f"Monday dip: open trade since {open_['t_in']:%a %H:%M} NY at {open_['entry']:.0f}, stop {open_['stop']:.0f}, "
          f"target {open_['target']:.0f}" if open_ else "Monday dip: no open trade")
    if open_ and got is not None and f"{open_['t_in']:%Y%m%d}-md" not in got:
        print("Monday dip: 'I took it' not tapped, so not in the levels message")
        return None
    return open_


def send_momentum(now, day, topic, live, fallback):
    """Session Momentum paper alerts (desk/session_momentum.py). Returns today's trade state if it is open and 'I took
    it' was tapped (else None); errors never stop the other alerts."""
    try:
        now = pd.Timestamp(now)
        stake = stake_for(day, default=fallback)[0]
        nq5, ndx5, nq1, ndx1, gap = five_minute(now, day)
        dec = sm.decision(nq5, ndx5, gap, day, now)
        if dec is None:
            print("Session momentum: no signal yet")
            return None
        st = sm.manage(nq5, dec, day, now)
        msgs = sm.messages(dec, st, day, stake)
        # the broker orders need 'where is the futures price NOW': Yahoo's NQ=F runs ~10 min late, which put the demo
        # stops ~140 points off after a fast move (18 Nov 2026); ^NDX is near real time, so use it + the basis
        scale = scale_now(now, day)
        at.want("P", [dict(ref=at.ref_of("P", dec["t_in"]), side=dec["side"], entry=dec["entry"], t_in=dec["t_in"],
                           stop=st["stop"], target=None)] if st["why"] == "open" else [], scale)
    except Exception as err:  # noqa: Yahoo hiccup etc.; the next run tries again
        print(f"Session momentum not checked ({err!r})")
        return None
    got = send_paper(msgs, topic, live, day)
    print(f"Session momentum: {'long' if dec['side'] == 1 else 'short'} at {dec['entry']:.0f} (EMA {dec['ema']:.0f}, gap "
          f"{gap:.0f}), stop {st['stop']:.0f}, {st['why']}")
    if got is not None and f"{day:%Y%m%d}-sm" not in got:
        return None
    return st if st["why"] == "open" else None


_FIVE = {}


def five_minute(now, day):
    """Futures / index 5- and 1-minute candles + today's futures-minus-index gap, fetched once per run."""
    if (now, day) not in _FIVE:
        nq5, ndx5 = yahoo("NQ=F", "5m", "60d", prepost=True), yahoo("^NDX", "5m", "60d")
        nq1, ndx1 = yahoo("NQ=F", "1m", "5d", prepost=True), yahoo("^NDX", "1m", "5d")
        gap = basis([(complete(nq1, 1, now), complete(ndx1, 1, now)), (complete(nq5, 5, now), complete(ndx5, 5, now))], day)
        _FIVE.clear()
        _FIVE[(now, day)] = (nq5, ndx5, nq1, ndx1, gap)
    return _FIVE[(now, day)]


def send_obias(now, day, topic="", live=False, fallback=1000.0):
    """Overnight Bias ORB (desk/overnight_bias.py), a paper strategy like Session Momentum: demo orders for every trade,
    phone messages with an 'I took it' button; errors never stop the other alerts."""
    try:
        now = pd.Timestamp(now)
        stake = stake_for(day, default=fallback)[0]
        nq5, ndx5, nq1, ndx1, gap = five_minute(now, day)
        tr = ob.trades(nq5, ndx5, gap, day, now)
        ndc = complete(ndx1, 1, now)
        at.want("O", [dict(ref=at.ref_of("O", t["t_in"]), side=t["side"], entry=t["entry"], t_in=t["t_in"],
                           stop=t["stop"], target=t["target"]) for t in tr if t["why"] == "open"],
                float(ndc.close.iloc[-1]) if len(ndc) else None)
        off_env = (os.environ.get("T212_OFFSET") or "auto").strip().lower()
        off = gap if off_env == "auto" else float(off_env)
        msgs = ob.messages(tr, day, stake, off if np.isfinite(off) else 0.0)
    except Exception as err:  # noqa: Yahoo hiccup etc.; the next run tries again
        print(f"Overnight bias not checked ({err!r})")
        return
    send_paper(msgs, topic, live, day)
    for t in tr:
        print(f"Overnight bias: {'long' if t['side'] == 1 else 'short'} {t['t_in']:%H:%M} at {t['entry']:.0f}, stop "
              f"{t['stop']:.0f}, target {t['target']:.0f}: {t['why']}" + (f" at {t['exit']:.0f}" if t["why"] != "open" else ""))
    if not tr:
        print("Overnight bias: no trade (yet) today")


def wants(ev, ibs_raw, day):
    """What should be open now on IG (desk/autotrade.py): noise / sweep from today's events, IBS incl. one held
    from an earlier session. IBS 'I missed it' taps don't matter here: the demo takes every signal."""
    fifteen = pd.Timedelta(minutes=15)
    n = s = None
    for e in ev:
        k = e["kind"]
        if k == "noise_entry":
            t = e["ts"] + fifteen
            n = dict(ref=at.ref_of("N", t), side=1, entry=e["px"], t_in=t, stop=e["px"] * (1 - NSTOP), target=None)
        elif k in ("noise_stop", "noise_exit", "noise_close"):
            n = None
        elif k == "sweep_entry":
            t, tr = e["ts"] + fifteen, e["trade"]
            s = dict(ref=at.ref_of("S", t), side=tr["side"], entry=tr["entry"], t_in=t, stop=tr["stop"],
                     target=tr["target"])
        elif k.startswith("sweep_"):
            s = None
    iev, held = ibs_raw
    ibs = []
    kinds = {e["kind"]: e for e in iev}
    if "ibs_entry" in kinds:
        t = kinds["ibs_entry"]["ts"] + fifteen
        ibs = [dict(ref=at.ref_of("I", t), side=1, entry=kinds["ibs_entry"]["px"], t_in=t, stop=None, target=None)]
    elif held and "ibs_exit" not in kinds:
        t = pd.Timestamp(dt.datetime.combine(held[0], dt.time(IBS_AT // 60, IBS_AT % 60), NY))
        ibs = [dict(ref=at.ref_of("I", t), side=1, entry=held[1], t_in=t, stop=None, target=None)]
    return [x for x in (n,) if x], [x for x in (s,) if x], ibs


def main():
    """Alerts first (_main); then the automatic IG demo orders for what this run evaluated (never stops the alerts)."""
    try:
        _main()
    finally:
        if at.enabled:
            stake = stake_for(dt.datetime.now(NY).date(), default=float(os.environ.get("STAKE_EUR") or 1000))[0]
            at.flush(stake)
            yen_week.run(stake)   # strategy Y: USD/JPY long Mon-Wed, short Thu-Fri on the IG + Capital.com demos + FTMO (rounds 44 / 44b)
            at.report()   # 23:00 Amsterdam: the day's result per demo account


def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print instead of sending")
    ap.add_argument("--replay", help="YYYY-MM-DD: print every message of that (recent) session")
    ap.add_argument("--test", action="store_true", help="send one test message to check the phone setup")
    a = ap.parse_args()
    off_env = (os.environ.get("T212_OFFSET") or "auto").strip().lower()
    fallback = float(os.environ.get("STAKE_EUR") or 1000)
    topic = os.environ.get("NTFY_TOPIC", "")
    if a.test:
        if not topic:
            sys.exit("NTFY_TOPIC is empty: add the secret first (Settings > Secrets and variables > Actions)")
        push(topic, "test", "Day bot test", "Setup works: trade alerts will arrive here during the US session (9:30-16:00 New York).",
             ["white_check_mark"])
        print("test message sent")
        return
    md_open = sm_open = None
    live = bool(topic) and not a.dry and not a.replay
    at.enabled = live
    state = os.environ.get("STATE_FILE") or ".daybot/sent.json"
    missed = ibs_missed()
    if live:   # 'I missed it' taps on IBS BUY messages (ntfy keeps them 12 h, so save them in the state)
        done = load_sent(state)
        taps = {t for t in taken(topic, done) if IBS_TAP.match(t)}
        missed |= ibs_missed(taps, path=os.devnull)
        if {f"{t}_taken" for t in taps} - done:
            save_sent(state, done | {f"{t}_taken" for t in taps}, day=dt.datetime.now(NY).date())
    if missed:
        print("IBS buys not taken:", ", ".join(f"{d:%a %d %b}" for d in sorted(missed)[-3:]))
    if a.replay:
        day = dt.date.fromisoformat(a.replay)
        now = dt.datetime.combine(day, dt.time(16, 5), NY)
        if day.weekday() <= 1:  # Monday / Tuesday: the Monday dip's messages of that week up to Wednesday 02:00 NY
            wed = dt.datetime.combine(day - dt.timedelta(days=day.weekday()) + dt.timedelta(days=2), dt.time(2), NY)
            send_monday(min(wed, dt.datetime.now(NY)), day, "", False, fallback, missed)
        if day.weekday() < 5:
            send_momentum(now + dt.timedelta(minutes=30), day, "", False, fallback)
            send_obias(now + dt.timedelta(minutes=30), day, "", False, fallback)
    else:
        now = dt.datetime.now(NY)
        day = now.date()
        al = next_alarm(now)
        if al and topic and not a.dry:
            state = os.environ.get("STATE_FILE") or ".daybot/sent.json"
            done = load_sent(state)
            if al[0] not in done:
                try:  # a refused alarm must never stop the trade alerts below; a later run tries again
                    push(topic, al[0], al[2], al[3], al[4], priority=4, at=al[1])
                    save_sent(state, done | {al[0]}, day)
                    print(f"alarm scheduled for {al[1]:%a %d %b %H:%M} NY")
                except requests.RequestException as err:
                    print(f"alarm not scheduled ({err}); a later run tries again")
        rem = sunday_reminder(now.astimezone(AMS))
        if rem:
            state = os.environ.get("STATE_FILE") or ".daybot/sent.json"
            done = load_sent(state)
            if rem[0] in done:
                print("weekly reminder already sent")
            elif topic and not a.dry:
                push(topic, *rem, priority=3)
                save_sent(state, done | {rem[0]}, day)
            print(f"--- {rem[1]}\n{rem[2]}")
            return
        if md.window(now):
            md_open = send_monday(now, day, topic, bool(topic) and not a.dry, fallback, missed)
        m = now.hour * 60 + now.minute
        if day in NYSE_HOLIDAYS:
            print(f"{day} is a US market holiday: no day trades")
            return
        if now.weekday() < 5 and OPEN + 5 <= m <= CLOSE + 30:   # Session Momentum: signal at 9:35, sent by the 9:46 run
            sm_open = send_momentum(now, day, topic, bool(topic) and not a.dry, fallback)
        if now.weekday() < 5 and OPEN + 15 <= m <= ob.OUT + 30:   # Overnight Bias ORB (paper): 9:45 - 15:30
            send_obias(now, day, topic, bool(topic) and not a.dry, fallback)
        if now.weekday() >= 5 or not (OPEN + 15 <= m <= CLOSE + 20):
            print(f"{now:%a %H:%M} NY: outside the session, nothing to do")
            return
    now = pd.Timestamp(now)
    stake, stake_why, stop_hit = stake_for(day, default=fallback)
    print(stake_why)
    ev, info = build(now, day, missed=missed)
    print(json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in info.items() if k != "line"},
                     default=str))
    if info["last_candle"] is None or pd.Timestamp(info["last_candle"]).date() != day:
        print("no candles for today yet (holiday or data late)")
        return
    off = info["basis"] if off_env == "auto" else float(off_env)
    if not np.isfinite(off):
        sys.exit("no futures gap measured today: set T212_OFFSET to a number")
    msgs = [message(e, off, info["fx"], stake) for e in ev]
    wn, ws, wi = wants(ev, info["ibs_raw"], day)
    at.want("N", wn, info["idx_px"])
    at.want("S", ws, info["idx_px"])
    at.want("I", wi, info["idx_px"])
    m = now.hour * 60 + now.minute
    # once a day: the levels for today, so you know the bot is running
    if not np.isnan(info["onh"]):
        first_check = dt.datetime.combine(day, dt.time(10, 0), NY).astimezone(AMS).strftime("%H:%M")
        lvl = (f"Overnight high {fmt(info['onh'], off)}, low {fmt(info['onl'], off)}. Day open "
               f"{fmt(info['open'], off)}. Noise line at the first check ({first_check}): {fmt(info['line'].get(OPEN + 15, np.nan), off) if info['line'] is not None else 'n/a'}.")
        if info["ibs_held"]:
            ibs_at = dt.datetime.combine(day, dt.time(IBS_AT // 60, IBS_AT % 60), NY).astimezone(AMS)
            lvl += (f"\nYou hold an IBS trade (bought {info['ibs_held'][0]:%a %d %b} at about "
                    f"{fmt(info['ibs_held'][1], off)}): keep it until the IBS message at about {ibs_at:%H:%M}.")
        if day in FOMC_DAYS:
            fed = dt.datetime.combine(day, dt.time(14, 0), NY).astimezone(AMS)
            lvl += f"\nFed decision today at {fed:%H:%M}: no sweep trade today (noise band and IBS as usual)."
        if (max(FOMC_DAYS) - day).days < 45:
            lvl += f"\nThe bot's Fed dates end on {max(FOMC_DAYS):%d %b %Y}: ask Claude to add next year's."
        if md_open:
            lvl += (f"\nYou hold a Monday-dip trade (practice, bought at about {md.fmt(md_open['entry'])}, stop "
                    f"{md.fmt(md_open['stop'])}, target {md.fmt(md_open['target'])}): it stays open until its stop, target "
                    f"or the sell message at about {md.due_by(md_open['t_in']).astimezone(AMS):%a %H:%M}.")
        lvl += f"\nSize: {stake_why}."
        if stop_hit:
            lvl += ("\nSTOP RULE: the account is 20% or more below its highest point. Don't trade; review the plan "
                    "with Claude first.")
        msgs.insert(0, (f"{day:%Y%m%d}-levels", "Day bot running", lvl, ["chart_with_upwards_trend"]))
    if (info["open_trade"] or sm_open) and close_of(day) - 15 <= m < close_of(day) + 20:  # noise / sweep / Session Momentum
        keep = " Not the IBS trade: that one stays open overnight." if info["ibs_keep"] else ""
        keep += " Not the Monday-dip trade: it has its own sell message." if md_open else ""
        bell = dt.datetime.combine(day, dt.time(close_of(day) // 60, close_of(day) % 60), NY).astimezone(AMS)
        msgs.append((f"{day:%Y%m%d}-bell", f"Close everything before {bell - dt.timedelta(minutes=5):%H:%M}",
                     f"The US market closes at {bell:%H:%M} (Amsterdam). Close any noise, sweep or Session Momentum trade "
                     "that is still open now." + keep, ["bell"]))
    done = load_sent(state) if live else set()
    for key, title, body, tags in msgs:
        if key in done:
            continue
        print(f"--- {title}\n{body}")
        if live:
            ibs_buy = "-ibs_entry-" in key
            push(topic, key, title, body, tags, priority=3 if key.endswith("levels") else 5,
                 took=f"{key[:8]}-ibs" if ibs_buy else None, label="I missed it")
            done.add(key)
            save_sent(state, done, day)


if __name__ == "__main__":
    main()
