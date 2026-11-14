"""Round 9 finding: IBS (internal bar strength) on the Nasdaq-100, tested as it would be traded.

Rule: at 15:45 New York (21:45 Amsterdam), IBS = (price - today's low) / (today's high - today's low), using the
regular session so far. IBS below 0.2 (the index is near its low of the day) = buy at that price. Sell at 15:45 the
next trading day (hold again if IBS is again below 0.2). Long only. No stop: a 2-4% stop cuts the bounce and
lowers the result. Entering up to 30 minutes late (16:15 NY) gives the same result.
Costs: Trading 212 spread (2 points Nasdaq, 0.012% S&P) plus one overnight financing charge per night held
(T-bill rate + 2.5% a year / 365; Friday to Monday = 3 nights).
Source: Pagonidis (2014) "The IBS effect"; Connors. Daily-candle version on 20 years of Yahoo data in
desk/everything_daily.py (works on Nasdaq, S&P, Dow, Russell; not on European indices).

Run: python -m desk.ibs --data <dir with NSXUSD.pkl, SPXUSD.pkl> --irx <T-bill csv: date,irx>
"""
import argparse

import numpy as np
import pandas as pd

from desk.tune import HOLD


def ibs_trades(m1, irx, cost_pct, at=945, thr=0.2):
    """% result per trade (one row per signal day, booked on the entry day)."""
    tod = m1.index.hour * 60 + m1.index.minute
    m = m1[(tod >= 570) & (tod < at)]
    tod = m.index.hour * 60 + m.index.minute
    day = m.index.tz_localize(None).normalize()
    g = m.groupby(day)
    last = pd.Series(tod, index=m.index).groupby(day).last()
    d = pd.DataFrame({"h": g.high.max(), "l": g.low.min(), "p": g.close.last(), "n": g.size()})
    # skip half days and gaps: as in the bot, the decision needs prices up to the decision time (HistData stops
    # early on some days around the US clock change and in March 2020: those days must not count)
    d = d[(d.n >= (at - 570) * 0.8) & (last.reindex(d.index) >= at - 5)]
    d["ibs"] = (d.p - d.l) / (d.h - d.l)
    d["ret"] = (d.p.shift(-1) / d.p - 1) * 100
    nights = pd.Series(d.index, d.index).diff().shift(-1).dt.days
    fin = (irx.reindex(d.index).ffill().bfill() + 2.5) / 365 * nights
    t = d[d.ibs < thr].dropna(subset=["ret"])
    return (t.ret - fin.reindex(t.index) - cost_pct).rename("pct"), d.index


def summary(pct, cal):
    daily = pct.reindex(cal).fillna(0.0)
    out = dict(per_month=round(len(pct) / (len(cal) / 252 * 12), 1), win=round(100 * (pct > 0).mean()),
               avg_pct=round(pct.mean(), 3))
    for k, msk in (("dev", cal < HOLD), ("hold", cal >= HOLD)):
        x, t = daily[msk], pct[pct.index < HOLD] if k == "dev" else pct[pct.index >= HOLD]
        out[f"{k}_pf"] = round(t[t > 0].sum() / -t[t <= 0].sum(), 2)
        out[f"{k}_pct_month"] = round(x.sum() / (len(x) / 252 * 12), 2)
        out[f"{k}_sharpe"] = round(x.mean() / x.std() * np.sqrt(252), 2)
    eq = daily.cumsum()
    out["max_dd_pct"] = round((eq - eq.cummax()).min(), 1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--irx", required=True)
    a = ap.parse_args()
    irx = pd.read_csv(a.irx, index_col=0, parse_dates=True).irx
    rows = []
    for sym, cost in (("NSXUSD", 0.008), ("SPXUSD", 0.012)):
        m1 = pd.read_pickle(f"{a.data}/{sym}.pkl")
        for at in (930, 945, 955):
            for thr in (0.1, 0.2, 0.3):
                pct, cal = ibs_trades(m1, irx, cost, at, thr)
                rows.append(dict(market=sym, decide=f"{at // 60}:{at % 60:02d}", ibs_below=thr, **summary(pct, cal)))
                print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv("desk/results/ibs_1m.csv", index=False)
