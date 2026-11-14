"""Round 18: the '4-step prop-firm strategy' (TikTok el3mentss, 1 Nov 2026), turned into exact rules and tested on the
Nasdaq-100 1-minute data (2019 - Sep 2026).

As taught (57 s video, "the strategy that got me a 10k payout"):
  1. Daily bias. Yesterday's daily candle closed beyond the day before's high/low = continuation in that direction.
     It swept the day before's high/low and closed back inside = reversal (swept the low -> look for longs).
  2. Drop to the 4-hour: wait for price to tap a 4h fair value gap in the bias direction ("4h profile confirmed").
  3. 1-hour or 30-minute: wait for a swing (low, for longs) inside the gap.
  4. Entry timeframe: bullish CISD (change in state of delivery), enter on the close. Stop under the low,
     target 2R or external liquidity. His example: 5 micros, 100 points, $1,000 before fees.

Exact rules (choices where the video leaves room are in the grid):
  - Daily candle: 'fut' = futures day 18:00-17:00 NY (a TradingView NQ daily), 'rth' = 9:30-16:00. Bias for day T
    from days T-1 vs T-2. Continuation: close(T-1) beyond high/low(T-2). Reversal: wick beyond and close back inside.
    An outside day closing inside (both swept) = no bias. 'mode' keeps both kinds, only 'cont', or only 'rev'.
  - 4h candles start 18:00 NY (as round 13). Bullish FVG: low(c3) > high(c1), gap = [high(c1), low(c3)], usable
    after c3 closes, until a 4h candle closes below its bottom or it is `age` days old. Mirror for bearish.
  - Tap: in 9:30-15:30 NY an entry-timeframe candle trades into the gap (low <= top) without closing below the
    bottom. From the tap, L = running low; a close below the bottom ends the setup. Each day starts afresh.
  - Swing (step 3): 'none' = skip; 30 / 60 = the 30m / 1h candle holding L must be a confirmed swing low (the candles
    before and after it have higher lows, the one after has closed) before an entry counts.
  - CISD: the run of down-close entry candles into L; its first candle's open is the level (if the low candle and
    the one before are both up-closes: that candle's high). Entry = first close above the level after L.
  - Stop = L (minus `buf`), target `target` x risk, or 'liq' = yesterday's high (skipped if that is < 1R away).
    Flat at 16:00 (engine rule). One trade per gap.

Engine: desk/library.py `simulate` (entry at the signal close, stop first if a candle hits both, flat 16:00 NY),
2 points cost per trade (Trading 212 spread), as rounds 10-17. Pick on 2019-2023, check 2024 - Sep 2026.

Run: python -m desk.round18 --bars1 <1-minute pickle> --out <dir>
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from desk.ifvg import stats
from desk.library import LAST_ENTRY, RTH_CLOSE, RTH_OPEN, simulate
from desk.round13 import current_desk, htf

COST = 2.0
MIN_N = 150
TAUGHT = dict(daily="fut", swing=60, etf=1, age=5, gap=0.0, mode="both", target=2.0, max_day=1, min_risk=0.0)


def lean_bars(m1, tf):
    """Only what simulate() needs (make_bars adds session levels we do not use; too heavy on 1-minute data)."""
    if tf == 1:
        b = m1[["open", "high", "low", "close"]].copy()
    else:
        b = m1.resample(f"{tf}min", label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    b["tod"] = b.index.hour * 60 + b.index.minute
    b["end"] = b.tod + tf
    b["tdate"] = (b.index.tz_localize(None) + pd.Timedelta(hours=6)).normalize()
    b["rth"] = (b.tod >= RTH_OPEN) & (b.end <= RTH_CLOSE)
    return b


def bias_table(m1, kind):
    """Per trade date: side (+1/-1/0), kind (cont/rev), yesterday's high and low."""
    td = (m1.index.tz_localize(None) + pd.Timedelta(hours=6)).normalize()
    x = m1.assign(tdate=td)
    if kind == "rth":
        tod = m1.index.hour * 60 + m1.index.minute
        x = x[(tod >= RTH_OPEN) & (tod < RTH_CLOSE)]
    d = x.groupby("tdate").agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    p1, p2 = d.shift(1), d.shift(2)
    bc, sc = p1.close > p2.high, p1.close < p2.low
    br = (p1.low < p2.low) & (p1.close > p2.low) & ~bc
    sr = (p1.high > p2.high) & (p1.close < p2.high) & ~sc
    both = br & sr
    br, sr = br & ~both, sr & ~both
    side = np.where(bc | br, 1, np.where(sc | sr, -1, 0))
    kindc = np.where(bc | sc, "cont", np.where(br | sr, "rev", ""))
    return pd.DataFrame({"side": side, "kind": kindc, "ph": p1.high, "pl": p1.low}, index=d.index)


def fvgs(h4, min_gap):
    """4h fair value gaps: avail, side, bot, top, dead (time of the first 4h close beyond the far side)."""
    h, l, c = h4.high.values, h4.low.values, h4.close.values
    av = h4.avail.values
    rows = []
    for j in range(2, len(h4)):
        for s, bot, top in ((1, h[j - 2], l[j]), (-1, h[j], l[j - 2])):
            if top - bot <= min_gap * c[j]:
                continue
            after = c[j + 1:]
            brk = np.nonzero(after < bot if s == 1 else after > top)[0]
            dead = av[j + 1 + brk[0]] if len(brk) else np.datetime64("2100-01-01")
            rows.append((av[j], s, bot, top, dead))
    return pd.DataFrame(rows, columns=["avail", "side", "bot", "top", "dead"])   # times: UTC datetime64


def swing_ok(hb, j, L, t_end):
    """Candle j of the higher timeframe holds low L and is a confirmed swing low at time t_end (longs)."""
    lo = hb["low"]
    if j < 1 or j + 1 >= len(lo) or hb["avail"][j + 1] > t_end:
        return False
    return lo[j] <= L + 1e-9 and lo[j - 1] > lo[j] and lo[j + 1] > lo[j]


def events(b, bias, z, age_days, swing_bars=None):
    """Every entry signal: DataFrame indexed by bar position with side, kind, entry, stop, liq."""
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    t0 = b.index.values.astype("datetime64[ns]")
    tf = int(b.end.iloc[0] - b.tod.iloc[0])
    tend = t0 + np.timedelta64(tf, "m")
    tod, end, td = b.tod.values, b.end.values, b.tdate.values
    bt = bias.reindex(pd.DatetimeIndex(td).unique())
    bside, bkind, bph, bpl = (dict(zip(bt.index.values, bt[k].values)) for k in ("side", "kind", "ph", "pl"))
    zav = z.avail.values.astype("datetime64[ns]")
    zdead = z.dead.values.astype("datetime64[ns]")
    zs, zb, zt = z.side.values, z.bot.values, z.top.values
    max_age = np.timedelta64(int(age_days * 86400), "s")
    hb = None
    if swing_bars is not None:
        hb = {"low": swing_bars.low.values, "high": swing_bars.high.values,
              "avail": swing_bars.avail.values.astype("datetime64[ns]"),
              "start": swing_bars.index.values.astype("datetime64[ns]")}
    used = np.zeros(len(z), bool)
    zi, live, rows = 0, [], []
    day, s, setup = None, 0, None   # setup: [fvg idx, L, k (bar of L), level]
    for i in range(len(b)):
        while zi < len(z) and zav[zi] <= t0[i]:
            live.append(zi)
            zi += 1
        if td[i] != day:
            day, setup = td[i], None
            s = bside.get(day, 0)
            s = 0 if s != s else int(s)
            live = [q for q in live if not used[q] and zdead[q] > t0[i] and t0[i] - zav[q] <= max_age]
        if s == 0 or tod[i] < RTH_OPEN or end[i] > LAST_ENTRY:
            continue
        # mirror shorts into 'long' arithmetic
        hi, lo, cl, op = (h[i], l[i], c[i], o[i]) if s == 1 else (-l[i], -h[i], -c[i], -o[i])
        if setup is not None:
            q, L, k, lvl = setup
            bot = zb[q] if s == 1 else -zt[q]
            if cl < bot or zdead[q] <= t0[i]:
                setup = None
            else:
                if lo < L:
                    L, k = lo, i
                    m = k if cl < op else k - 1
                    o_, c_ = (o, c) if s == 1 else (-o, -c)
                    if c_[m] >= o_[m]:
                        lvl = hi
                    else:
                        while m - 1 >= 0 and c_[m - 1] < o_[m - 1]:
                            m -= 1
                        lvl = o_[m]
                    setup = [q, L, k, lvl]
                elif cl > lvl:
                    ok = True
                    if hb is not None:
                        j = np.searchsorted(hb["start"], t0[k], side="right") - 1
                        if s == 1:
                            ok = swing_ok(hb, j, L, tend[i])
                        else:
                            ok = swing_ok({"low": -hb["high"], "avail": hb["avail"]}, j, L, tend[i])
                    if ok:
                        liq = bph[day] if s == 1 else bpl[day]
                        rows.append((i, s, bkind[day], c[i], L if s == 1 else -L, liq))
                        used[q] = True
                        live = [x for x in live if x != q]
                        setup = None
                continue
        # look for a tap
        best = None
        for q in live:
            if zs[q] != s or used[q] or zdead[q] <= t0[i]:
                continue
            top, bot = (zt[q], zb[q]) if s == 1 else (-zb[q], -zt[q])
            if lo <= top and cl >= bot and (best is None or top > best[1]):
                best = (q, top)
        if best is not None:
            q = best[0]
            o_, c_ = (o, c) if s == 1 else (-o, -c)
            m = i if cl < op else i - 1
            if c_[m] >= o_[m]:
                lvl = hi
            else:
                while c_[m - 1] < o_[m - 1]:
                    m -= 1
                lvl = o_[m]
            setup = [q, lo, i, lvl]
    return pd.DataFrame(rows, columns=["i", "side", "kind", "entry", "stop", "liq"]).set_index("i")


def to_sig(b, ev, target=2.0, min_risk=0.0, max_day=1, mode="both", side=0, buf=0.0, end_entry=LAST_ENTRY):
    e = ev
    if mode != "both":
        e = e[e.kind == mode]
    if side:
        e = e[e.side == side]
    stop = e.stop * (1 - e.side * buf)
    risk = (e.entry - stop) * e.side
    tgt = e.liq if target == "liq" else e.entry + e.side * float(target) * risk
    ok = (risk > 0) & (risk >= min_risk * e.entry) & ((tgt - e.entry) * e.side >= (1.0 if target == "liq" else 0) * risk)
    e, stop, tgt = e[ok], stop[ok], tgt[ok]
    n = len(b)
    out = {k: np.full(n, np.nan) for k in ("sl_l", "tp_l", "sl_s", "tp_s")}
    le, se = np.zeros(n, bool), np.zeros(n, bool)
    for i, s, st, tp in zip(e.index, e.side, stop, tgt):
        if s == 1:
            le[i], out["sl_l"][i], out["tp_l"][i] = True, st, tp
        else:
            se[i], out["sl_s"][i], out["tp_s"][i] = True, st, tp
    sig = {k: pd.Series(v, b.index) for k, v in out.items()}
    sig.update(le=pd.Series(le, b.index), se=pd.Series(se, b.index), max_day=max_day, start=RTH_OPEN,
               end_entry=end_entry)
    return sig


_D = {}


def _load(path):
    if not _D:
        m1 = pd.read_pickle(path)
        _D["m1"] = m1
        _D["bars"] = {}
        _D["htf"] = {tf: htf(m1, tf) for tf in (30, 60, 240)}
        _D["bias"] = {k: bias_table(m1, k) for k in ("fut", "rth")}
        _D["fvg"] = {}
    return _D


def bars(d, tf):
    if tf not in d["bars"]:
        d["bars"][tf] = lean_bars(d["m1"], tf)
    return d["bars"][tf]


def ev_for(path, daily, swing, etf, age, gap):
    d = _load(path)
    if gap not in d["fvg"]:
        d["fvg"][gap] = fvgs(d["htf"][240], gap)
    sb = d["htf"][swing] if swing else None
    return events(bars(d, etf), d["bias"][daily], d["fvg"][gap], age, sb)


def _run(args):
    path, daily, swing, etf, age, gap = args
    d = _load(path)
    ev = ev_for(path, daily, swing, etf, age, gap)
    b = bars(d, etf)
    rows = []
    for mode, target, max_day, min_risk in itertools.product(("both", "cont", "rev"), (1.5, 2.0, 3.0, "liq"), (1, 2),
                                                             (0.0, 0.0005)):
        tr = simulate(b, to_sig(b, ev, target, min_risk, max_day, mode), cost=COST)
        rows.append(dict(daily=daily, swing=swing, etf=etf, age=age, gap=gap, mode=mode, target=target,
                         max_day=max_day, min_risk=min_risk, **stats(tr)))
    print(f"{daily} swing {swing} etf {etf}: {len(ev)} signals; both/2R/1 a day: {rows[2]}", flush=True)
    return rows


def main(path, out):
    grid = list(itertools.product(("fut", "rth"), (0, 60), (5, 1), (5,), (0.0,)))
    with ProcessPoolExecutor(2) as ex:
        rows = [r for part in ex.map(_run, [(path, *g) for g in grid]) for r in part]
    g = pd.DataFrame(rows)
    g.to_csv(f"{out}/round18_grid.csv", index=False)

    d = _load(path)
    ok = g[g.n >= MIN_N].sort_values("dev_pf", ascending=False)
    keys = ["daily", "swing", "etf", "age", "gap", "mode", "target", "max_day", "min_risk"]
    pick = ok.iloc[0][keys].to_dict()
    for k in ("swing", "etf", "age", "max_day"):
        pick[k] = int(pick[k])
    if pick["target"] != "liq":
        pick["target"] = float(pick["target"])
    res = []
    for name, tr in current_desk(path).items():
        res.append(dict(model=name, test="as live", **stats(tr)))
    taught5 = dict(TAUGHT, etf=5)
    for label, p in (("as taught, 1m entry", TAUGHT), ("as taught, 5m entry", taught5), ("best on dev", pick)):
        ev = ev_for(path, p["daily"], p["swing"], p["etf"], p["age"], p["gap"])
        b = bars(d, p["etf"])
        kw = dict(target=p["target"], min_risk=p["min_risk"], max_day=p["max_day"], mode=p["mode"])
        tag = (f"4-step ({label}): daily {p['daily']}, swing {p['swing']}m, entry {p['etf']}m, age {p['age']}d, "
               f"gap>={p['gap']}, {p['mode']}, target {p['target']}, max {p['max_day']}/day, risk>={p['min_risk']}")
        tests = [("as set", kw, COST, None), ("4 pts cost", kw, 2 * COST, None),
                 ("longs only", dict(kw, side=1), COST, None), ("shorts only", dict(kw, side=-1), COST, None),
                 ("stop 0.02% beyond", dict(kw, buf=0.0002), COST, None),
                 ("entries 9:30-11:30 only", dict(kw, end_entry=690), COST, None)]
        for alt in ({"swing": 30}, {"swing": 0 if p["swing"] else 60}, {"age": 2}, {"age": 10}, {"gap": 0.0005}):
            q = dict(p, **alt)
            tests.append((f"variant {alt}", kw, COST, q))
        for name, k, cst, q in tests:
            evx = ev if q is None else ev_for(path, q["daily"], q["swing"], q["etf"], q["age"], q["gap"])
            tr = simulate(b, to_sig(b, evx, **k), cost=cst)
            res.append(dict(model=tag, test=name, **stats(tr)))
            print(res[-1], flush=True)
    df = pd.DataFrame(res)
    df.to_csv(f"{out}/round18_summary.csv", index=False)
    print("\nSUMMARY\n" + df.to_csv(index=False))
    print(f"\nTOP 20 BY DEV PF (>= {MIN_N} trades)\n" + ok.head(20).to_csv(index=False))
    print(f"\n{(g.dev_pf > 1).mean():.0%} of settings profitable on dev; median dev PF {g.dev_pf.median():.2f}; "
          f"{((g.dev_pf > 1) & (g.hold_pf > 1)).mean():.0%} profitable on both")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    main(a.bars1, a.out)
