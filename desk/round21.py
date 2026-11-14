"""Round 21: 'two rules, one index' (TikTok thequantbuilder, 19 Oct 2026): dip-buying on 1-hour candles, held up to 24 h.

As shown (frames: 1-hour chart with RSI14 and MA40):
  Rule 1: only Mondays. Index weak and RSI(14) under 40 -> buy. Stop 2.5 ATR, target 2x the risk, out after 24 hours.
  Rule 2: price far below its 40-hour average and RSI under 35 -> buy. Stop 2 ATR, same target, same 24 h limit.
  His claim on the FTSE 100 since 2017 (with a hidden third rule): 1,908 trades, +312.9% at 0.75% risk a trade,
  max drawdown 12%, 50.8% wins. He says the thresholds were picked on the same years and the 'Monday dip' also shows
  on the DAX, S&P 500 and Nasdaq.

Exact rules (grid where the video leaves room). 1-hour candles on the market's own clock (London / Berlin / New York),
from 1-minute HistData CFD prices (nearly 24 h a day), RSI(14) Wilder, ATR(14), MA40 = 40-hour simple average:
  - 'weak' for rule 1: 'ma' = close below MA40, 'day' = close below the previous day's close, 'none'.
  - 'far' for rule 2: close more than k x ATR below MA40, k in {1, 1.5, 2, 3}.
  - 'hours': 'all' or 'rth' (signals only in the cash session: London 8-16:30, Berlin 9-17:30, NY 9:30-16).
  - Long only, one position at a time, entry at the signal candle's close; stop / target checked on the next 1-hour
    candles (stop first if both are hit in one candle, gaps fill at the open); out at the close 24 h after entry.
  - Cost: the market's spread (round trip, % of price, table M) + CFD overnight fee 0.02% per night held (x3 over a
    weekend), as on Trading 212.
Results per trade in % of price (1x position, as all rounds) and in R; equity at 0.75% risk a trade, compounding,
to compare with his claim. Dev 2019-2023, holdout 2024 - Sep 2026 (data starts 2019).

Run: python -m desk.round21 --data <dir with SYMBOL.pkl> --out <dir>
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from desk.everything import M
from desk.ifvg import stats

NIGHT_FEE = 0.02   # % per night
MARKETS = {"UK 100": ("Europe/London", 480, 990), "Germany 40": ("Europe/Berlin", 540, 1050),
           "US 500": ("America/New_York", 570, 960), "US Tech 100": ("America/New_York", 570, 960)}
TAUGHT = dict(weak="ma", far=1.5, hours="all", r1=40, r2=35, s1=2.5, s2=2.0, target=2.0, hold=24, rules="both")


def hourly(m1, tz):
    x = m1.tz_convert(tz)
    h = x.resample("60min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    c = h.close
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    h["rsi"] = 100 - 100 / (1 + up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean())
    tr = pd.concat([h.high - h.low, (h.high - c.shift()).abs(), (h.low - c.shift()).abs()], axis=1).max(axis=1)
    h["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    h["ma"] = c.rolling(40).mean()
    day = h.index.tz_localize(None).normalize()
    dclose = c.groupby(day).last()
    h["pdc"] = pd.Series(day, h.index).map(dclose.shift(1))
    h["mon"] = h.index.dayofweek == 0
    h["dow"] = h.index.dayofweek
    dl = c.groupby(day).last()
    h["up200"] = pd.Series(day, h.index).map((dl > dl.rolling(200).mean()).shift(1)).fillna(False).astype(bool)
    h["tod"] = h.index.hour * 60
    return h


def simulate(h, p, mk):
    o, hi, lo, c = (h[k].values for k in ("open", "high", "low", "close"))
    rsi, atr, ma, pdc = (h[k].values for k in ("rsi", "atr", "ma", "pdc"))
    t = h.index
    days = t.tz_localize(None).normalize().values
    _, op, cl = MARKETS[mk]
    tod = h.tod.values
    hrs = p["hours"]
    in_rth = ((tod + 60 > op) & (tod < cl) if hrs == "rth" else (tod + 60 <= op) if hrs == "pre"
              else (tod >= cl) | (tod + 60 <= op) if hrs == "off" else np.ones(len(h), bool))
    if p.get("trend") == "up200":
        in_rth = in_rth & h.up200.values
    weak = {"ma": c < ma, "day": c < pdc, "none": np.ones(len(h), bool)}[p["weak"]]
    s1 = (p["rules"] in ("both", "1")) & (h.dow.values == p.get("dow", 0)) & weak & (rsi < p["r1"]) & in_rth
    s2 = (p["rules"] in ("both", "2")) & (c < ma - p["far"] * atr) & (rsi < p["r2"]) & in_rth
    cost = M[mk][1]
    hold = np.timedelta64(int(p["hold"] * 3600), "s")
    tv = t.values
    rows, i, n = [], 50, len(h)
    while i < n - 1:
        if not (s1[i] or s2[i]) or not np.isfinite(atr[i]):
            i += 1
            continue
        m = p["s1"] if s1[i] else p["s2"]
        e = c[i]
        stop = e - m * atr[i]
        tgt = e + p["target"] * (e - stop) if p["target"] else np.inf
        j, x, why = i + 1, None, "time"
        while j < n:
            if lo[j] <= stop:
                x, why = min(stop, o[j]), "stop"
                break
            if hi[j] >= tgt:
                x, why = max(tgt, o[j]), "target"
                break
            if tv[j] - tv[i] >= hold:   # close of candle j is >= 24 h after the entry close
                x = c[j]
                break
            j += 1
        if x is None:
            break
        nights = int((days[j] - days[i]) / np.timedelta64(1, "D"))
        pct = (x - e) / e * 100 - cost - NIGHT_FEE * nights
        risk = (e - stop) / e * 100
        rows.append((pd.Timestamp(days[i]), "r1" if s1[i] else "r2", why, pct, pct / risk, t[i] + pd.Timedelta("1h"),
                     t[j] + pd.Timedelta("1h")))
        i = j + 1
    return pd.DataFrame(rows, columns=["day", "rule", "why", "pct", "R", "t_in", "t_out"])


def r_stats(tr, risk=0.75):
    if len(tr) < 20:
        return {}
    out = {}
    for k, m in (("dev", tr.day < "2024-01-01"), ("hold", tr.day >= "2024-01-01")):
        r = tr.R[m].to_numpy()
        eq = np.cumprod(1 + r * risk / 100)
        yrs = max((tr.day[m].max() - tr.day[m].min()).days / 365.25, 0.5)
        out[f"{k}_R"] = round(r.mean(), 3)
        out[f"{k}_eq075_yr"] = round((eq[-1] ** (1 / yrs) - 1) * 100, 1) if len(r) else np.nan
        out[f"{k}_eq075_dd"] = round(((eq / np.maximum.accumulate(eq)) - 1).min() * 100, 1) if len(r) else np.nan
    return out


_H = {}


def _job(args):
    data, mk, grid = args
    tz = MARKETS[mk][0]
    h = hourly(pd.read_pickle(f"{data}/{M[mk][0]}.pkl"), tz)
    h = h[h.index >= pd.Timestamp("2019-01-01", tz=tz)]
    rows = []
    for p in grid:
        tr = simulate(h, p, mk)
        rows.append(dict(market=mk, **p, **stats(tr), **r_stats(tr)))
    print(mk, rows[0], flush=True)
    return rows


def main_r1(data, out):
    """21b: rule 1 alone (the Monday dip), robustness grid on every market."""
    grid = [dict(TAUGHT, rules="1", r1=r, s1=st, target=t, hold=hd, hours=hr, weak=w)
            for r, st, t, hd, hr, w in itertools.product((30, 35, 40, 45, 50), (1.5, 2.0, 2.5, 3.0), (0, 1.5, 2.0, 3.0),
                                                         (8, 24, 48), ("all", "rth"), ("ma", "day", "none"))]
    with ProcessPoolExecutor(2) as ex:
        g = pd.DataFrame([r for part in ex.map(_job, [(data, mk, grid) for mk in MARKETS]) for r in part])
    g.to_csv(f"{out}/round21b_grid.csv", index=False)
    for mk in MARKETS:
        s = g[g.market == mk]
        print(f"{mk}: {len(s)} settings, dev PF>1 {(s.dev_pf > 1).mean():.0%}, hold PF>1 {(s.hold_pf > 1).mean():.0%}, "
              f"both {((s.dev_pf > 1) & (s.hold_pf > 1)).mean():.0%}, median dev PF {s.dev_pf.median():.2f}, "
              f"hold PF {s.hold_pf.median():.2f}, median %/yr {s.dev_yr.median():.1f} / {s.hold_yr.median():.1f}")


def main(data, out):
    base = [dict(TAUGHT, weak=w, far=f, hours=hr) for w, f, hr in itertools.product(
        ("ma", "day", "none"), (1.0, 1.5, 2.0, 3.0), ("all", "rth"))]
    robust = []
    for b in [dict(TAUGHT, weak=w, far=1.5) for w in ("ma", "day", "none")]:
        robust += [dict(b, r1=b["r1"] + d, r2=b["r2"] + d) for d in (-5, 5)]
        robust += [dict(b, s1=b["s1"] + d, s2=b["s2"] + d) for d in (-0.5, 0.5)]
        robust += [dict(b, target=t) for t in (1.0, 1.5, 3.0)]
        robust += [dict(b, hold=hd) for hd in (8, 48)]
        robust += [dict(b, rules=r) for r in ("1", "2")]
    grid = base + robust
    with ProcessPoolExecutor(2) as ex:
        g = pd.DataFrame([r for part in ex.map(_job, [(data, mk, grid) for mk in MARKETS]) for r in part])
    g.to_csv(f"{out}/round21_grid.csv", index=False)
    cols = ["market", "weak", "far", "hours", "r1", "r2", "s1", "s2", "target", "hold", "rules", "n", "per_yr", "win",
            "dev_pf", "dev_yr", "hold_pf", "hold_yr", "hold_dd", "pos_years", "dev_R", "hold_R", "dev_eq075_yr",
            "hold_eq075_yr", "hold_eq075_dd"]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        print("\nAS TAUGHT (his thresholds), every 'weak' / 'far' / hours reading\n" + g.iloc[[k for k in range(len(g))
              if g.iloc[k][["r1", "r2", "s1", "s2", "target", "hold", "rules"]].tolist() ==
              [40, 35, 2.5, 2.0, 2.0, 24, "both"]]][cols].to_string(index=False))
        for mk in MARKETS:
            s = g[g.market == mk]
            print(f"\n{mk}: {len(s)} settings, dev PF>1 {(s.dev_pf > 1).mean():.0%}, both {((s.dev_pf > 1) & (s.hold_pf > 1)).mean():.0%}, "
                  f"median dev PF {s.dev_pf.median():.2f}, hold PF {s.hold_pf.median():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--r1", action="store_true", help="21b: rule-1 robustness grid")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (main_r1 if a.r1 else main)(a.data, a.out)
