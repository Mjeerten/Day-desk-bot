"""Round 6: tune the two survivors' numbers (stop, target, filters, timing).

Discipline: every setting is judged on 2019-2023 (dev) only; 2024-2026 (holdout) is printed alongside but never
used to pick. A change is worth keeping only if it helps on dev AND does not hurt on the holdout, and its
neighbours are also good (a lone spike is luck).

Run: python -m desk.tune --bars1 <1-minute pickle> --out <csv>
"""
import argparse

import numpy as np
import pandas as pd

from desk.library import RTH_OPEN, make_bars, simulate

HOLD = "2024-01-01"


def noise_sig(b, lookback=14, mult=1.0, every=30, exit_rule="open", stop=None, tp=None, last_entry=930,
              first=600, shorts=False):
    t = b[b.rth]
    c = t.pivot_table(index="tdate", columns="tod", values="close")
    c = c[c.notna().all(axis=1)]
    op = b.rth_open.groupby(b.tdate).first().reindex(c.index)
    pc = b.pdc.groupby(b.tdate).first().reindex(c.index)
    sig = (c.div(op, axis=0) - 1).abs().rolling(lookback).mean().shift(1) * mult
    ub = sig.add(1).mul(np.maximum(op, pc), axis=0).stack().rename("ub")
    lb = (1 - sig).mul(np.minimum(op, pc), axis=0).stack().rename("lb")
    j = b.join(ub, on=["tdate", "tod"]).join(lb, on=["tdate", "tod"])
    check = b.rth & (b.end % every == 0) & (b.end >= first)
    le = check & (b.close > j.ub) & (b.end <= last_entry)
    if exit_rule == "open":
        xl = check & (b.close < b.rth_open)
    elif exit_rule == "band":
        xl = check & (b.close < np.maximum(b.rth_open, b.pdc))
    elif exit_rule == "twap":
        xl = check & (b.close < b.twap)
    elif exit_rule == "none":
        xl = pd.Series(False, b.index)
    s = dict(le=le, xl=xl, end_entry=last_entry)
    if shorts:
        s["se"] = check & (b.close < j.lb) & (b.end <= last_entry)
        s["xs"] = check & (b.close > b.rth_open)
    if stop:
        s["sl_l"], s["sl_s"] = b.close * (1 - stop), b.close * (1 + stop)
    if tp:
        s["tp_l"], s["tp_s"] = b.close * (1 + tp), b.close * (1 - tp)
    return s


def sweep_sig(b, r=2.0, buf=0.0, max_dist=None, min_dist=None, end_entry=930, side=0, max_day=1, start=RTH_OPEN,
              hk="onh", lk="onl"):
    se = (b.high > b[hk]) & (b.close < b[hk]) & (b.tod >= start)
    le = (b.low < b[lk]) & (b.close > b[lk]) & (b.tod >= start)
    sl_l, sl_s = b.low * (1 - buf), b.high * (1 + buf)
    dl, ds = (b.close - sl_l) / b.close, (sl_s - b.close) / b.close
    if max_dist:
        le, se = le & (dl <= max_dist), se & (ds <= max_dist)
    if min_dist:
        le, se = le & (dl >= min_dist), se & (ds >= min_dist)
    if side == 1:
        se = pd.Series(False, b.index)
    if side == -1:
        le = pd.Series(False, b.index)
    s = dict(le=le, se=se, max_day=max_day, end_entry=end_entry, sl_l=sl_l, sl_s=sl_s)
    if r:
        s["tp_l"], s["tp_s"] = b.close + r * (b.close - sl_l), b.close - r * (sl_s - b.close)
    return s


def st(tr):
    if len(tr) < 10:
        return dict(n=len(tr))
    d = pd.to_datetime(tr.day)
    out = dict(n=len(tr), win=round(100 * (tr.pct > 0).mean()))
    for k, p, dd in (("dev", tr.pct[d < HOLD], d[d < HOLD]), ("hold", tr.pct[d >= HOLD], d[d >= HOLD])):
        yrs = max((dd.max() - dd.min()).days / 365.25, 0.5) if len(dd) else 1
        out[f"{k}_pf"] = round(p[p > 0].sum() / -p[p <= 0].sum(), 2) if (p <= 0).any() and len(p) else np.nan
        out[f"{k}_yr"] = round(p.sum() / yrs, 2)
        eq = p.cumsum()
        out[f"{k}_dd"] = round((eq - eq.cummax()).min(), 2) if len(p) else 0
    return out


def grid():
    G = [("noise", "base", {})]
    for v in (7, 10, 20, 30):
        G.append(("noise", f"lookback {v}", dict(lookback=v)))
    for v in (0.7, 0.85, 1.15, 1.3, 1.5):
        G.append(("noise", f"band x{v}", dict(mult=v)))
    G += [("noise", "check every 60m", dict(every=60)),
          ("noise", "exit under max(open, prev close)", dict(exit_rule="band")),
          ("noise", "exit under day average (twap)", dict(exit_rule="twap")),
          ("noise", "no exit until 16:00", dict(exit_rule="none")),
          ("noise", "with shorts", dict(shorts=True))]
    for v in (0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02):
        G.append(("noise", f"stop {v:.2%}", dict(stop=v)))
    for v in (0.005, 0.0075, 0.01, 0.015, 0.02):
        G.append(("noise", f"take profit {v:.2%}", dict(tp=v)))
    for v in (720, 810, 870):
        G.append(("noise", f"last entry {v // 60}:{v % 60:02d}", dict(last_entry=v)))
    G.append(("noise", "first check 10:30", dict(first=630)))
    G.append(("sweep", "base", {}))
    for v in (0, 1.0, 1.5, 2.5, 3.0, 4.0):
        G.append(("sweep", f"target {v}R" if v else "no target (16:00)", dict(r=v)))
    for v in (0.0005, 0.001, 0.002):
        G.append(("sweep", f"stop buffer {v:.2%}", dict(buf=v)))
    for v in (0.002, 0.003, 0.005, 0.0075):
        G.append(("sweep", f"skip stop > {v:.2%}", dict(max_dist=v)))
    for v in (0.0005, 0.001, 0.0015):
        G.append(("sweep", f"skip stop < {v:.2%}", dict(min_dist=v)))
    for v in (630, 690, 750, 840):
        G.append(("sweep", f"entries until {v // 60}:{v % 60:02d}", dict(end_entry=v)))
    G += [("sweep", "longs only", dict(side=1)), ("sweep", "shorts only", dict(side=-1)),
          ("sweep", "up to 2 per day", dict(max_day=2)), ("sweep", "from 9:45", dict(start=585)),
          ("sweep", "prior-day high/low instead", dict(hk="pdh", lk="pdl"))]
    return G


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost", type=float, default=2.0)
    a = ap.parse_args()
    b = make_bars(pd.read_pickle(a.bars1), 15)
    rows = []
    for strat, name, kw in grid():
        sig = (noise_sig if strat == "noise" else sweep_sig)(b, **kw)
        rows.append(dict(strat=strat, test=name, **st(simulate(b, sig, cost=a.cost))))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
