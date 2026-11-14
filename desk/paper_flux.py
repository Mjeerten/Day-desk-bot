"""Paper trading of the round-14 'Flux' reversal model (desk/round14.py): every signal it would have given on the
Nasdaq-100 and how the trade ended, at EUR 1,000 per trade. Not traded, only tracked (Day Desk logbook, Paper tab).

Data: Yahoo ^NDX 5-minute candles, regular session, real index volume (the last 60 days), the same chart as a
TradingView NASDAQ:NDX 5m chart. Rules: `RULE` below (desk/round14.py has the indicator code and BACKTEST.md round 14
the test). Entry at the signal candle's close (9:30-15:30 New York), stop and target fixed at entry, stop first if a
candle hits both, out at the 16:00 close, max 2 trades a day, one at a time, no trade when the stop is closer than
0.15% (the spread would eat it). Cost: 0.008% round trip (spread).

Each run recomputes the whole 60-day window and merges it into journal/paper_flux.csv (older rows are kept).
Run: python -m desk.paper_flux [--csv journal/paper_flux.csv]  (GitHub: .github/workflows/paper.yml, every weekday
after the US close).
"""
import argparse
import os

import numpy as np
import pandas as pd
import requests

from desk import round14 as r14
from desk.plan import ROOT

NY, AMS = "America/New_York", "Europe/Amsterdam"
RULE = dict(htf=60, sig="flux", poc="none", stop="swing", target=2.0, side=0, min_risk=0.0015)   # BACKTEST.md round 14
COST, STAKE, MAX_DAY = 0.008, 1000.0, 2
COLS = ["date", "time_ny", "time_ams", "side", "entry", "stop", "target", "exit_time_ny", "exit", "why", "pct", "eur"]


def fetch(sym="^NDX"):
    j = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{sym}?interval=5m&range=60d",
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30).json()["chart"]["result"][0]
    q = j["indicators"]["quote"][0]
    ix = pd.to_datetime(j["timestamp"], unit="s", utc=True).tz_convert(NY)
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=ix).dropna()
    df = df.rename(columns={"volume": "vol"})
    tod = df.index.hour * 60 + df.index.minute
    df = df[(tod >= 570) & (tod < 960)]
    now = pd.Timestamp.now(tz=NY)
    if len(df) and df.index[-1] + pd.Timedelta(minutes=5) > now:   # drop a candle that has not closed yet
        df = df.iloc[:-1]
    return df


def trades(m5, rule=RULE, cost=COST):
    """One row per trade with times. m5: 5m candles (NY clock) with real volume."""
    r14.HTFS = (int(rule["htf"]),)
    m = r14.reclock(m5, "US")
    d = r14.prep(m, "real", "rth")
    b = d["b"]
    e = r14.events(d, int(rule["htf"]), rule["sig"], rule["poc"], rule["stop"])
    if int(rule["side"]):
        e = e[e.side == int(rule["side"])]
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    td, end = b.tdate.values, b.end.values
    rows, busy_until, per_day = [], -1, {}
    for i, s, en, st in zip(e.index, e.side, e.entry, e.stop):
        risk = (en - st) * s
        if risk < rule.get("min_risk", 0) * en or risk <= 0 or i <= busy_until or per_day.get(td[i], 0) >= MAX_DAY:
            continue
        tp = en + s * float(rule["target"]) * risk
        k, why, px = i, "close", c[i]
        for k in range(i + 1, len(b)):
            if td[k] != td[i]:
                k -= 1
                px, why = c[k], "close"
                break
            if (l[k] <= st) if s == 1 else (h[k] >= st):
                px, why = (min(st, o[k]) if s == 1 else max(st, o[k])), "stop"
                break
            if (h[k] >= tp) if s == 1 else (l[k] <= tp):
                px, why = (max(tp, o[k]) if s == 1 else min(tp, o[k])), "target"
                break
            if end[k] >= 960:
                px, why = c[k], "close"
                break
        else:
            px, why = c[k], "open"   # still running at the last candle we have
        busy_until = k
        per_day[td[i]] = per_day.get(td[i], 0) + 1
        t_in = (b.index[i] + pd.Timedelta(minutes=5)).tz_localize(None).tz_localize(NY)   # candle close
        t_out = (b.index[k] + pd.Timedelta(minutes=5)).tz_localize(None).tz_localize(NY)
        pct = s * (px - en) / en * 100 - cost
        rows.append(dict(date=f"{t_in:%Y-%m-%d}", time_ny=f"{t_in:%H:%M}", time_ams=f"{t_in.tz_convert(AMS):%H:%M}",
                         side="buy" if s == 1 else "sell", entry=round(en, 2), stop=round(st, 2), target=round(tp, 2),
                         exit_time_ny=f"{t_out:%H:%M}", exit=round(px, 2), why=why, pct=round(pct, 3),
                         eur=round(pct / 100 * STAKE, 2)))
    return pd.DataFrame(rows, columns=COLS)


def merge(new, path):
    """Keep older rows, replace the days the new window covers."""
    if os.path.exists(path) and os.path.getsize(path):
        old = pd.read_csv(path, dtype=str)
        first = new.date.min() if len(new) else "9999"
        old = old[old.date < first]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "paper_flux.csv"))
    a = ap.parse_args()
    m5 = fetch()
    t = trades(m5)
    # the first days of the window only warm up the indicators (200-candle profile, 20 hourly candles + EMA)
    warm = sorted(set(m5.index.date))[min(6, len(set(m5.index.date)) - 1)]
    t = t[t.date >= f"{warm:%Y-%m-%d}"]
    allt = merge(t, a.csv)
    done = allt[allt.why != "open"]
    print(t.to_string(index=False))
    print(f"\n{len(done)} closed trades, EUR {pd.to_numeric(done.eur).sum():+.2f} at EUR {STAKE:.0f} per trade")


if __name__ == "__main__":
    main()
