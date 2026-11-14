"""Round 13: the '4h + 1h trend, supply/demand zone, 5-minute break of structure' model (a day trader's video,
described by the owner), turned into exact rules and tested on the Nasdaq-100 1-minute data (2019 - Sep 2026).

As described:
  1. Trend: the 4-hour chart is up -> look for buys; the 1-hour chart also up -> confirmed. Mirror for sells.
  2. Zone: below the current price, find an impulsive candle (big body, small wicks) next to an indecisive candle
     (small body, big wicks) next to another impulsive candle: impulse - base - impulse ('rally-base-rally' or
     'drop-base-rally'). The box runs from the base candle's lowest to highest wick and is dragged right to now.
     'Without cutting through other wicks' is read as: the impulse leaving the zone closes beyond the base's wick,
     and the zone is still fresh (price has not been back into it since).
  3. Entry on 5-minute candles: price taps the zone, slows down and builds structure inside it; the first 5m candle
     that closes above the resistance of that movement (the highest high since the tap) = buy at that close.
  4. Stop at the low of that movement; target 2x the risk (2R). Shorts: the mirror image.

Exact rules used here (choices where the description leaves room are the grid below):
  - Trend per timeframe: 'hl' = last two confirmed swing highs and lows (2 candles each side) both higher (lower for
    down); 'ema20' / 'ema50' = last completed candle closes above (below) its EMA; 'none' = no filter (control).
    4h candles start at 18:00 NY (futures day), 1h on the hour. Only completed candles are used.
  - Impulse: body >= `imp` x range and range >= `imp_atr` x the candle timeframe's 14-ATR. Base: body <= `base` x
    range. The impulse after the base must be in the trade's direction and close beyond the base's wick; the one
    before it either direction (rally-base-rally or drop-base-rally). First run: strict rules (imp 0.6-0.7, range >= 1
    ATR, base 0.35) found only 15 trades 'as taught' in 7.7 years, so the grid also has looser candle rules.
  - A zone is used once: the first touch by any 5m candle uses it up, also outside trading hours or against the
    trend (then it is just gone, the zone has been 'tested'). Zones older than 10 days are dropped.
  - A tap counts 9:30-15:30 NY with both trends agreeing. Then: a 5m close beyond the far side of the zone kills
    the setup; 'res' = 'all': highest high since the tap (tap candle included), 'swing': the latest 5m swing high
    since the tap (2 candles each side). Entry needs at least 2 candles after the tap and is given up after 3 hours.
  - Stop = lowest low since the tap (incl. the entry candle), target `target` x risk, flat at 16:00 (engine rule).

Engine: desk/library.py `simulate` (entry at the signal close, stop counts first if a candle hits both, flat at 16:00
NY), 2 points cost per trade (Trading 212 spread), as rounds 10-12. Pick on 2019-2023, check 2024 - Sep 2026.

Run: python -m desk.round13 --bars1 <1-minute pickle> --out <dir>
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from desk.ifvg import stats
from desk.library import LAST_ENTRY, RTH_OPEN, atr, ema, make_bars, simulate
from desk.tune import noise_sig, sweep_sig

COST = 2.0
MIN_N = 150                         # trades needed to be picked
BASE_BODY = 0.35
MIN_BARS, MAX_WAIT = 2, 36          # 5m candles after the tap
MAX_AGE = pd.Timedelta(days=10)
TAUGHT = dict(trend="hl", ztf=60, left="any", imp=0.6, imp_atr=1.0, base=0.35, res="all", target=2.0, min_risk=0.0,
              max_day=2)
LOOSE = dict(TAUGHT, trend="ema20", imp=0.5, imp_atr=0.5, base=0.5)   # an eye finds more patterns than strict rules


def htf(m1, tf):
    off = {240: "2h"}.get(tf, "0min")   # 4h candles at 18:00, 22:00, 2:00 ... NY
    r = m1.resample(f"{tf}min", label="left", closed="left", offset=off).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    r["avail"] = r.index + pd.Timedelta(minutes=tf)
    return r


def trend(r, mode):
    """+1 up, -1 down, 0 neither, known at r.avail."""
    if mode == "none":
        return pd.Series(0, r.index)
    if mode.startswith("ema"):
        e = ema(r.close, int(mode[3:]))
        return pd.Series(np.sign(r.close - e).astype(int), r.index)
    h, l = r.high.values, r.low.values
    out = np.zeros(len(r), int)
    sh, sl = [], []
    for i in range(4, len(r)):
        j = i - 2   # a swing at j is confirmed when i completes
        if h[j] == h[j - 2:i + 1].max():
            sh.append(h[j])
        if l[j] == l[j - 2:i + 1].min():
            sl.append(l[j])
        if len(sh) >= 2 and len(sl) >= 2:
            out[i] = 1 if sh[-1] > sh[-2] and sl[-1] > sl[-2] else -1 if sh[-1] < sh[-2] and sl[-1] < sl[-2] else 0
    return pd.Series(out, r.index)


def zones(r, imp, left, imp_atr=1.0, base=BASE_BODY):
    """Demand (+1) and supply (-1) zones: rows (avail, side, bottom, top)."""
    o, h, l, c = (r[k].values for k in ("open", "high", "low", "close"))
    rng = h - l
    body = np.abs(c - o)
    a = atr(r, 14).values
    with np.errstate(invalid="ignore", divide="ignore"):
        isimp = (body >= imp * rng) & (rng >= imp_atr * a) & (rng > 0)
        isbase = (body <= base * rng) & (rng > 0)
    d = np.sign(c - o)
    rows = []
    for j in range(16, len(r)):
        if not (isimp[j] and isbase[j - 1] and isimp[j - 2]):
            continue
        s = d[j]
        if left == "same" and d[j - 2] != s:
            continue
        if (s == 1 and c[j] > h[j - 1]) or (s == -1 and c[j] < l[j - 1]):
            rows.append((r.avail.iat[j], int(s), l[j - 1], h[j - 1]))
    return pd.DataFrame(rows, columns=["avail", "side", "bot", "top"])


def events(b, z, tr, res="all"):
    """Every 5m break of structure after a zone tap. Returns DataFrame indexed by bar position: side, entry, stop."""
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    t0 = b.index.values
    tod, end, td = b.tod.values, b.end.values, b.tdate.values
    zav = z.avail.values.astype("datetime64[ns]")
    t0 = t0.astype("datetime64[ns]")
    max_age = np.timedelta64(MAX_AGE)
    zi, live, rows = 0, [], []
    setup = None   # [side, far edge, tap bar, res, extreme, swing res, day]
    for i in range(len(b)):
        while zi < len(z) and zav[zi] <= t0[i]:
            live.append([zav[zi], z.side.iat[zi], z.bot.iat[zi], z.top.iat[zi]])
            zi += 1
        if live and t0[i] - live[0][0] > max_age:
            live = [x for x in live if t0[i] - x[0] <= max_age]
        window = tod[i] >= RTH_OPEN and end[i] <= LAST_ENTRY
        # 1. an open setup
        if setup is not None:
            s, far, tap, rs, ext, sw, day = setup
            if td[i] != day or not window or i - tap > MAX_WAIT or (c[i] < far if s == 1 else c[i] > far):
                setup = None
            else:
                if res == "swing" and i - 2 > tap:   # swing at i-2 confirmed now
                    k = i - 2
                    if s == 1 and h[k] == h[k - 2:i + 1].max():
                        sw = h[k]
                    if s == -1 and l[k] == l[k - 2:i + 1].min():
                        sw = l[k]
                ext = min(ext, l[i]) if s == 1 else max(ext, h[i])
                lvl = rs if res == "all" else sw
                if i - tap >= MIN_BARS and lvl is not None and (c[i] > lvl if s == 1 else c[i] < lvl):
                    rows.append((i, s, c[i], ext, tap, far))
                    setup = None
                else:
                    rs = max(rs, h[i]) if s == 1 else min(rs, l[i])
                    setup = [s, far, tap, rs, ext, sw, day]
        # 2. zone touches (each zone is used once)
        if live:
            hit = [x for x in live if (l[i] <= x[3] if x[1] == 1 else h[i] >= x[2])]
            if hit:
                live = [x for x in live if not any(x is y for y in hit)]
                if setup is None and window:
                    for s in (1, -1):
                        if tr[i] != s:
                            continue
                        cand = [x for x in hit if x[1] == s]
                        if not cand:
                            continue
                        x = max(cand, key=lambda x: x[3]) if s == 1 else min(cand, key=lambda x: x[2])
                        far = x[2] if s == 1 else x[3]
                        if (c[i] < far) if s == 1 else (c[i] > far):
                            continue   # closed straight through the zone
                        setup = [s, far, i, h[i] if s == 1 else l[i], l[i] if s == 1 else h[i], None, td[i]]
    return pd.DataFrame(rows, columns=["i", "side", "entry", "stop", "tap", "far"]).set_index("i")


def to_sig(b, ev, target=2.0, min_risk=0.0, max_day=2, side=0, buf=0.0):
    e = ev[ev.side == side] if side else ev
    stop = e.stop * (1 - e.side * buf)
    risk = (e.entry - stop) * e.side
    ok = (risk > 0) & (risk >= min_risk * e.entry)
    e, stop, risk = e[ok], stop[ok], risk[ok]
    n = len(b)
    out = {k: np.full(n, np.nan) for k in ("sl_l", "tp_l", "sl_s", "tp_s")}
    le, se = np.zeros(n, bool), np.zeros(n, bool)
    for i, s, en, st, rk in zip(e.index, e.side, e.entry, stop, risk):
        if s == 1:
            le[i], out["sl_l"][i], out["tp_l"][i] = True, st, en + target * rk
        else:
            se[i], out["sl_s"][i], out["tp_s"][i] = True, st, en - target * rk
    sig = {k: pd.Series(v, b.index) for k, v in out.items()}
    sig.update(le=pd.Series(le, b.index), se=pd.Series(se, b.index), max_day=max_day, start=RTH_OPEN)
    return sig


_D = {}


def _load(path):
    if not _D:
        m1 = pd.read_pickle(path)
        b = make_bars(m1, 5)
        _D["b"], _D["m1"] = b, m1
        _D["htf"] = {tf: htf(m1, tf) for tf in (15, 60, 240)}
        _D["trend"] = {}
    return _D


def trend_5m(d, mode):
    """Both 4h and 1h agree: +1 / -1 per 5m candle (known at its close), 0 otherwise. 'none' = +1 and -1 both ok."""
    if mode in d["trend"]:
        return d["trend"][mode]
    b = d["b"]
    t = pd.DataFrame({"t": (b.index + pd.Timedelta(minutes=5)).values})
    tot = np.zeros(len(b), int)
    for tf in (240, 60):
        r = d["htf"][tf]
        x = pd.DataFrame({"t": r.avail.values, "v": trend(r, mode).values}).sort_values("t")
        v = pd.merge_asof(t, x, on="t").v.fillna(0).astype(int).values
        tot += v
    out = np.where(tot == 2, 1, np.where(tot == -2, -1, 0))
    d["trend"][mode] = out
    return out


def ev_for(path, trend_mode, ztf, left, imp, imp_atr, base, res):
    d = _load(path)
    tr = trend_5m(d, trend_mode)
    z = zones(d["htf"][ztf], imp, left, imp_atr, base)
    if trend_mode == "none":
        # no filter: both directions allowed -> run each side with its own 'trend'
        e1 = events(d["b"], z[z.side == 1].reset_index(drop=True), np.ones(len(tr), int), res)
        e2 = events(d["b"], z[z.side == -1].reset_index(drop=True), -np.ones(len(tr), int), res)
        return pd.concat([e1, e2]).sort_index(kind="stable")
    return events(d["b"], z, tr, res)


def _run(args):
    path, trend_mode, ztf, left, imp, imp_atr, base, res = args
    d = _load(path)
    ev = ev_for(path, trend_mode, ztf, left, imp, imp_atr, base, res)
    rows = []
    for target, min_risk, max_day in itertools.product((1.5, 2.0, 3.0), (0.0, 0.0005), (1, 2)):
        tr = simulate(d["b"], to_sig(d["b"], ev, target, min_risk, max_day), cost=COST)
        rows.append(dict(trend=trend_mode, ztf=ztf, left=left, imp=imp, imp_atr=imp_atr, base=base, res=res, target=target, min_risk=min_risk,
                         max_day=max_day, **stats(tr)))
    print(rows[0], flush=True)
    return rows


def current_desk(path):
    b = make_bars(pd.read_pickle(path), 15)
    n = simulate(b, noise_sig(b, stop=0.003), cost=COST)
    s = simulate(b, sweep_sig(b, r=2.5), cost=COST)
    return {"noise band 0.3% stop (live)": n, "overnight sweep 2.5R (live)": s,
            "noise + sweep (live bot)": pd.concat([n, s]).sort_values("day", kind="stable")}


def main(path, out):
    grid = list(itertools.product(("hl", "ema20", "ema50", "none"), (15, 60), ("any",), (0.5, 0.6), (0.0, 0.5, 1.0),
                                  (0.35, 0.5), ("all", "swing")))
    with ProcessPoolExecutor(4) as ex:
        rows = [r for part in ex.map(_run, [(path, *g) for g in grid]) for r in part]
    g = pd.DataFrame(rows)
    g.to_csv(f"{out}/round13_grid.csv", index=False)

    d = _load(path)
    b = d["b"]
    key = lambda p: (p["trend"], p["ztf"], p["left"], p["imp"], p["imp_atr"], p["base"], p["res"])
    ok = g[g.n >= MIN_N].sort_values("dev_pf", ascending=False)
    pick = ok.iloc[0][["trend", "ztf", "left", "imp", "imp_atr", "base", "res", "target", "min_risk", "max_day"]].to_dict()
    pick["ztf"], pick["max_day"] = int(pick["ztf"]), int(pick["max_day"])
    rows = []
    for name, tr in current_desk(path).items():
        rows.append(dict(model=name, test="as live", **stats(tr)))
    desk_both = current_desk(path)["noise + sweep (live bot)"]
    first = None
    for label, p in (("as taught", TAUGHT), ("loose", LOOSE), ("best on dev", pick)):
        ev = ev_for(path, *key(p))
        kw = dict(target=p["target"], min_risk=p["min_risk"], max_day=p["max_day"])
        tag = (f"zone model ({label}): trend {p['trend']}, zone {p['ztf']}m, left {p['left']}, imp {p['imp']}, "
               f"atr {p['imp_atr']}, base {p['base']}, res {p['res']}, {p['target']}R, risk>={p['min_risk']}, max {p['max_day']}/day")
        tests = [("as set", kw, COST), ("4 pts cost", kw, 2 * COST), ("longs only", dict(kw, side=1), COST),
                 ("shorts only", dict(kw, side=-1), COST), ("stop 0.02% beyond", dict(kw, buf=0.0002), COST),
                 ("target 1R", dict(kw, target=1.0), COST)]
        for t in (p["target"] - 0.5, p["target"] + 0.5):
            tests.append((f"target {t}R", dict(kw, target=t), COST))
        for name, k, cst in tests:
            tr = simulate(b, to_sig(b, ev, **k), cost=cst)
            if name == "as set" and label == "best on dev":
                first = tr
            rows.append(dict(model=tag, test=name, **stats(tr)))
            print(rows[-1], flush=True)
        if p["trend"] != "none":   # does the trend filter help?
            ev0 = ev_for(path, "none", *key(p)[1:])
            tr = simulate(b, to_sig(b, ev0, **kw), cost=COST)
            rows.append(dict(model=tag, test="no 4h/1h trend filter", **stats(tr)))
    day = lambda t: t.groupby("day").pct.sum()
    corr = pd.concat([day(desk_both), day(first)], axis=1).fillna(0).corr().iloc[0, 1]
    allt = pd.concat([desk_both, first]).sort_values("day", kind="stable")
    rows.append(dict(model="live bot + best zone model", test=f"together (daily corr {corr:.2f})", **stats(allt)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/round13_summary.csv", index=False)
    print("\nSUMMARY\n" + df.to_csv(index=False))
    print(f"\nTOP 25 BY DEV PF (>= {MIN_N} trades)\n" + ok.head(25).to_csv(index=False))
    print(f"\n{(g.dev_pf > 1).mean():.0%} of settings profitable on dev; median dev PF {g.dev_pf.median():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars1", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    main(a.bars1, a.out)
