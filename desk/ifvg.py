"""Round 10: the IFVG (inverse / inversion fair value gap) model, as taught by PB Blake (TikTok / YouTube trade
recaps, backtested by his followers in FX Replay), turned into exact rules and tested on the Nasdaq-100.

His own rules are in paid videos; the public versions (his TikToks, the 'EZ$ PB Blake' TradingView script, the
Tradezella 'IFVG model' page, ICT material) agree on this sequence:
  1. Liquidity sweep: in the New York morning, price trades through an obvious level (overnight high/low,
     prior-day high/low, London high/low).
  2. The move into that sweep (the 'manipulation leg') leaves fair value gaps behind it. Bearish FVG = three
     candles where candle 3's high is below candle 1's low; the gap is the space between them.
  3. Inversion: a candle closes back through such a gap (a bearish gap closed above = it 'inverts' into support)
     = enter in the new direction at that close. Mirror for shorts.
  4. Stop beyond the sweep extreme (or beyond the gap); target a fixed R or the opposite liquidity.
  5. Only 9:30-11:00 New York ('golden hour'), at most one or two trades a day, 1-5 minute candles.

Engine: desk/library.py `simulate` (entries at the signal close, stop before target if both are hit in one candle,
flat at 16:00 NY). Cost 2 points per trade (Trading 212 spread), as in round 6. Settings are picked on 2019-2023
only; 2024-2026 is the holdout.

Run: python -m desk.ifvg --bars1 <1-minute pickle> --out <dir> [--stage 1|2]
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from desk.library import RTH_OPEN, make_bars, simulate
from desk.tune import HOLD, noise_sig, sweep_sig

LEVELS = {"on": ("onh", "onl"), "pd": ("pdh", "pdl"), "lon": ("lonh", "lonl")}


def ifvg_events(b, levels, look_min=60, min_gap=0.0001, scan_from=RTH_OPEN - 60, sweep_from=RTH_OPEN,
                last=930):
    """Every inversion after a sweep, one per sweep. Returns a DataFrame indexed by bar position with:
    side, entry, ext (sweep extreme), edge (far side of the inverted gap), liq (nearest opposite level)."""
    tf = int(b.end.iloc[0] - b.tod.iloc[0])
    look = max(3, look_min // tf)
    h, l, c = b.high.values, b.low.values, b.close.values
    tod, end, td = b.tod.values, b.end.values, b.tdate.values
    hi_lv = np.column_stack([b[LEVELS[k][0]].values for k in levels])
    lo_lv = np.column_stack([b[LEVELS[k][1]].values for k in levels])
    onh, onl, pdh, pdl = (b[k].values for k in ("onh", "onl", "pdh", "pdl"))
    rows = []
    cur = None
    for i in range(2, len(b)):
        if tod[i] < scan_from or end[i] > last + tf:
            continue
        if td[i] != cur:
            cur, bear, bull = td[i], [], []
            dlo = dhi = None          # running session extremes from sweep_from
            sw_lo = sw_hi = None      # [sweep bar, extreme, extreme bar, used]
        rth = tod[i] >= sweep_from
        # 1. inversions (check before adding this bar's own gap)
        if rth and sw_lo and not sw_lo[3] and i - sw_lo[2] <= look:
            hit = [g for g in bear if c[i] > g[0] and g[2] <= sw_lo[2] + 1 and g[2] >= sw_lo[2] - look]
            if hit:
                g = max(hit, key=lambda g: g[2])  # the latest gap of the leg
                liq = min([x for x in (onh[i], pdh[i], dhi) if x is not None and x > c[i]], default=np.nan)
                rows.append((i, 1, c[i], sw_lo[1], g[1], liq))
                sw_lo[3] = True
        if rth and sw_hi and not sw_hi[3] and i - sw_hi[2] <= look:
            hit = [g for g in bull if c[i] < g[1] and g[2] <= sw_hi[2] + 1 and g[2] >= sw_hi[2] - look]
            if hit:
                g = max(hit, key=lambda g: g[2])
                liq = max([x for x in (onl[i], pdl[i], dlo) if x is not None and x < c[i]], default=np.nan)
                rows.append((i, -1, c[i], sw_hi[1], g[0], liq))
                sw_hi[3] = True
        bear = [g for g in bear if c[i] <= g[0]]   # a gap closed through is spent
        bull = [g for g in bull if c[i] >= g[1]]
        # 2. new gaps (zone top, zone bottom, bar)
        if td[i - 2] == td[i]:
            if h[i] < l[i - 2] and l[i - 2] - h[i] >= min_gap * c[i]:
                bear.append((l[i - 2], h[i], i))
            if l[i] > h[i - 2] and l[i] - h[i - 2] >= min_gap * c[i]:
                bull.append((l[i], h[i - 2], i))
        # 3. sweeps (from the open on); a new low below the extreme re-arms the sweep
        if rth:
            lv_lo, lv_hi = np.nanmax(lo_lv[i]) if np.isfinite(lo_lv[i]).any() else np.nan, \
                np.nanmin(hi_lv[i]) if np.isfinite(hi_lv[i]).any() else np.nan
            # any of the chosen lows swept (the highest of them is reached first)
            if sw_lo is None and np.isfinite(lv_lo) and l[i] < lv_lo:
                sw_lo = [i, l[i], i, False]
            elif sw_lo is not None and l[i] < sw_lo[1]:
                sw_lo = [sw_lo[0], l[i], i, False]
            if sw_hi is None and np.isfinite(lv_hi) and h[i] > lv_hi:
                sw_hi = [i, h[i], i, False]
            elif sw_hi is not None and h[i] > sw_hi[1]:
                sw_hi = [sw_hi[0], h[i], i, False]
            dlo = l[i] if dlo is None else min(dlo, l[i])
            dhi = h[i] if dhi is None else max(dhi, h[i])
    return pd.DataFrame(rows, columns=["i", "side", "entry", "ext", "edge", "liq"]).set_index("i")


def ifvg_sig(b, ev, stop="sweep", target=2.0, end_entry=660, min_risk=0.0, max_day=1, side=0, buf=0.0):
    """Signal dict for simulate() from the events. target: a number = R multiple; 'liq' = opposite level
    (skipped if that is under 1R away)."""
    e = ev[ev.index.map(lambda i: b.end.iat[i] <= end_entry)] if len(ev) else ev
    if side:
        e = e[e.side == side]
    sl = np.where(e.side == 1, (e.ext if stop == "sweep" else e.edge) * (1 - buf),
                  (e.ext if stop == "sweep" else e.edge) * (1 + buf))
    risk = (e.entry - sl) * e.side
    if target == "liq":
        tp = e.liq.values
        ok = np.isfinite(tp) & ((tp - e.entry) * e.side >= risk)
    else:
        tp = e.entry + e.side * target * risk
        ok = np.ones(len(e), bool)
    ok &= (risk > 0) & (risk >= min_risk * e.entry)
    e, sl, tp = e[ok], np.asarray(sl)[ok], np.asarray(tp)[ok]
    n = len(b)
    out = {k: np.full(n, np.nan) for k in ("sl_l", "tp_l", "sl_s", "tp_s")}
    le, se = np.zeros(n, bool), np.zeros(n, bool)
    for (i, s), slv, tpv in zip(zip(e.index, e.side), sl, tp):
        if s == 1:
            le[i], out["sl_l"][i], out["tp_l"][i] = True, slv, tpv
        else:
            se[i], out["sl_s"][i], out["tp_s"][i] = True, slv, tpv
    sig = {k: pd.Series(v, b.index) for k, v in out.items()}
    sig.update(le=pd.Series(le, b.index), se=pd.Series(se, b.index), max_day=max_day, end_entry=end_entry,
               start=RTH_OPEN)
    return sig


def stats(tr):
    """Trades, win %, payoff (avg win / avg loss, roughly 'R' when the target is fixed), PF, %/yr and max
    drawdown on dev (2019-23) and holdout (2024-26), and positive years."""
    if len(tr) < 20:
        return dict(n=len(tr))
    d = pd.to_datetime(tr.day)
    p = tr.pct
    yrs = (d.max() - d.min()).days / 365.25
    out = dict(n=len(tr), per_yr=round(len(tr) / yrs), win=round(100 * (p > 0).mean(), 1),
               payoff=round(p[p > 0].mean() / -p[p <= 0].mean(), 2))
    for k, m in (("dev", d < HOLD), ("hold", d >= HOLD)):
        x, dd = p[m], d[m]
        y = max((dd.max() - dd.min()).days / 365.25, 0.5) if len(dd) else 1
        out[f"{k}_win"] = round(100 * (x > 0).mean(), 1) if len(x) else np.nan
        out[f"{k}_pf"] = round(x[x > 0].sum() / -x[x <= 0].sum(), 2) if (x <= 0).any() else np.nan
        out[f"{k}_yr"] = round(x.sum() / y, 2)
        eq = x.cumsum()
        out[f"{k}_dd"] = round((eq - eq.cummax()).min(), 2) if len(x) else 0
    py = p.groupby(d.dt.year).sum()
    out["pos_years"] = f"{(py > 0).sum()}/{len(py)}"
    out["years"] = " ".join(f"{v:+.1f}" for v in py)
    return out


_B = {}


def _run_tf(args):
    """All stage-1 settings for one timeframe (one process)."""
    path, tf, cost = args
    b = make_bars(pd.read_pickle(path), tf)
    rows = []
    for lv in (("on",), ("pd",), ("on", "pd"), ("on", "pd", "lon")):
        ev = ifvg_events(b, lv)
        for end_entry in (660, 720):
            for stop in ("sweep", "fvg"):
                for target in (1.0, 1.5, 2.0, 3.0, "liq"):
                    for min_risk in (0.0, 0.0005, 0.001):
                        for max_day in (1, 2):
                            sig = ifvg_sig(b, ev, stop, target, end_entry, min_risk, max_day)
                            r = dict(tf=tf, levels="+".join(lv), until=f"{end_entry // 60}:{end_entry % 60:02d}",
                                     stop=stop, target=target, min_risk=min_risk, max_day=max_day,
                                     **stats(simulate(b, sig, cost=cost)))
                            rows.append(r)
                            print(r, flush=True)
    return rows


def stage1(path, cost, out):
    with ProcessPoolExecutor(4) as ex:
        rows = [r for part in ex.map(_run_tf, [(path, tf, cost) for tf in (1, 2, 3, 5)]) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/ifvg_stage1.csv", index=False)
    return df


def current_desk(path, cost):
    """The bot's two intraday strategies on the same data and costs, for the comparison."""
    b = make_bars(pd.read_pickle(path), 15)
    n = simulate(b, noise_sig(b, stop=0.003), cost=cost)
    s = simulate(b, sweep_sig(b, r=2.5), cost=cost)
    both = pd.concat([n, s]).sort_values("day", kind="stable")
    return {"noise band 0.3% stop": n, "overnight sweep 2.5R": s, "noise + sweep together": both}


def stage2(path, cost, out, picks):
    """Stress-test the dev-picked settings: neighbours, double cost, each side, lookback, min gap, full day."""
    rows = []
    for name, tr in current_desk(path, cost).items():  # noqa: the bot's live intraday rules
        rows.append(dict(model=name, test="as live", **stats(tr)))
        print(rows[-1], flush=True)
    bars, first = {}, None
    for p in picks:
        tf = p["tf"]
        if tf not in bars:
            bars[tf] = make_bars(pd.read_pickle(path), tf)
        b = bars[tf]
        lv = tuple(p["levels"].split("+"))
        base = dict(stop=p["stop"], target=p["target"], end_entry=p["end_entry"], min_risk=p["min_risk"],
                    max_day=p["max_day"])
        tag = f"IFVG {tf}m {p['levels']} {p['stop']} {p['target']}R risk>={p['min_risk']} max{p['max_day']}"
        ev = ifvg_events(b, lv)
        tests = [("as picked", ev, base, cost), ("4 pts cost", ev, base, 2 * cost),
                 ("longs only", ev, dict(base, side=1), cost), ("shorts only", ev, dict(base, side=-1), cost),
                 ("entries until 15:30", ev, dict(base, end_entry=930), cost),
                 ("stop 0.02% beyond", ev, dict(base, buf=0.0002), cost)]
        for lk in (30, 90):
            tests.append((f"lookback {lk} min", ifvg_events(b, lv, look_min=lk), base, cost))
        for mg in (0.0, 0.0003):
            tests.append((f"min gap {mg:.2%}", ifvg_events(b, lv, min_gap=mg), base, cost))
        if isinstance(p["target"], float):
            for t in (p["target"] - 0.5, p["target"] + 0.5):
                if t > 0:
                    tests.append((f"target {t}R", ev, dict(base, target=t), cost))
        for name, e, kw, cst in tests:
            tr = simulate(b, ifvg_sig(b, e, **kw), cost=cst)
            first = tr if first is None else first
            rows.append(dict(model=tag, test=name, **stats(tr)))
            print(rows[-1], flush=True)
    desk = current_desk(path, cost)
    both = desk["noise + sweep together"]
    day = lambda t: t.groupby("day").pct.sum()
    corr = pd.concat([day(both), day(first)], axis=1).fillna(0).corr().iloc[0, 1]
    allt = pd.concat([both, first]).sort_values("day", kind="stable")
    rows.append(dict(model="noise + sweep + first IFVG pick", test=f"together (daily corr {corr:.2f})", **stats(allt)))
    print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/ifvg_stage2.csv", index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cost", type=float, default=2.0)
    ap.add_argument("--stage", type=int, default=1)
    ap.add_argument("--picks", default="desk/results/ifvg_picks.json", help="stage 2: settings to stress-test")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.stage == 1:
        df = stage1(a.bars1, a.cost, a.out)
        ok = df[df.n >= 150].sort_values("dev_pf", ascending=False)
        print("\nTOP 25 BY DEV PF (>=150 trades)\n" + ok.head(25).to_csv(index=False))
    else:
        picks = pd.read_json(a.picks).to_dict("records")
        for p in picks:
            p["target"] = p["target"] if p["target"] == "liq" else float(p["target"])
        print(stage2(a.bars1, a.cost, a.out, picks).to_csv(index=False))
