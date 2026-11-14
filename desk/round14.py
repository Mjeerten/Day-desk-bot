"""Round 14: 'Bollinger Bands MTF & Kalman Filter' + 'Weighted Volume Profile' (both by Flux Charts, TradingView) on 5m
candles, a 'perfect reversal' model from a video the owner sent. Tested on the Nasdaq-100 and every round-9 market.

As described:
  1. 5-minute chart, indicator 1 with its higher timeframe set to 30m: buy on a green signal, sell on a red one.
  2. Indicator 2, a volume profile: take the signal when it lines up right on the point of control (POC) line.
  3. Stop at the most recent low (high for sells), target 2x the risk (2R).

Both indicators rebuilt line by line from the Pine source the owner pasted (26 Oct 2026):
  - LTF bands (chart, 5m): basis = Flux's Kalman filter of the close (process noise 0.2, measurement error 2.0:
    steady gain 0.27), bands = basis +- 2 x 20-bar standard deviation.
  - HTF bands: ta.bb(close, 20, 2.25) on `htf`-minute candles (30 as taught), taken only when the HTF candle is
    complete (request.security on past bars), then each band EMA(20)-smoothed on the 5m chart.
  - Signals ('flux', the triangles): a close below the HTF lower band arms the buy; the first candle after that whose
    LTF lower band is higher than the candle before = green (buy). Disarmed once a close is above both the LTF and
    HTF lower bands; one signal per arming. Red (sell) is the mirror with the upper bands.
    'fill' = a second reading: the green fill between the lower bands ends (LTF lower crosses back above HTF lower).
  - Volume profile: the last 200 candles' high-low range in 30 rows; each of the last 201 candles adds its whole
    volume to every row its range touches. Weighting 'Normal' (default) or 'Recent' (vol x (0.85 / (age + 1) +
    0.15)). POC = middle of the row with the most volume (ties: the lower row). 'On the POC' = the POC line runs
    through the signal candle (n1 / r1), or the candle is within one row of it (nn / rn: with proxy volume the exact
    row is fragile - on 60 days of real QQQ 5m volume the proxy POC is the same row 25% of the time, 'near' agrees
    about half the time); 'none' = no POC filter.
  - Volume: HistData has none, so indices / metals / oil / currencies use a tick-volume proxy, the sum of the
    1-minute ranges in each 5m candle ('proxy'); the Nasdaq also runs with every candle weighted the same ('tpo').
    Crypto (Binance) runs with real volume and with the proxy, to see whether the proxy changes the answer.
  - The Nasdaq also runs with the indicators computed on regular-session candles only ('rth', what a NASDAQ:NDX
    chart shows; HTF candles then start at the 9:30 open like TradingView's) besides the 24-hour CFD candles ('24h').

Trades: entry at the signal candle's close 9:30-15:30 (session clock, `reclock`). Stop: 'swing' = the latest 5m swing
low (2 candles each side) below the entry, else the lowest of the last 12 candles; 'low3' = lowest of the last 3
candles; 'dip' = the lowest low since the setup armed (the reversal's low). Target 1.5 / 2 / 3R, max 2 trades a day,
flat at 16:00. Engine desk/library.py `simulate` (stop first if a candle hits both). Cost = round-trip % of price per
market (round 9 table M). Grid per market-session: 3 HTF (15/30/60) x 2 readings x 5 POC filters x 3 stops x 3 targets
x 3 sides = 810 settings. Dev = 2019-2023, holdout = 2024 - Sep 2026.

Run (GitHub, .github/workflows/round14.yml):
  python -m desk.round14 list --shard i --of n          symbols to download for a shard
  python -m desk.round14 build --raw <dir> --sym S --out <pkl>
  python -m desk.round14 run --data <dir> --out <dir> --shard i --of n
  python -m desk.round14 nasdaq --data <dir> --out <dir>     Nasdaq grid + detail + live bot comparison
  python -m desk.round14 tweak --data <dir> --rows <Nasdaq rows csv> --out <dir>   extra rules on the top settings
  python -m desk.round14 final --data <dir> --out <dir>     the final Nasdaq model on every market's own session
  python -m desk.round14 report --rows <dir> --out <dir>
"""
import argparse
import glob
import itertools
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from desk import round13 as r13
from desk.everything import M, reclock
from desk.ifvg import stats
from desk.library import LAST_ENTRY, RTH_CLOSE, RTH_OPEN, ema, make_bars, simulate

NY = "America/New_York"
NASDAQ = "US Tech 100"
HTFS, SIGS, POCS, STOPS, TARGETS, SIDES = (15, 30, 60), ("flux", "fill"), ("none", "n1", "nn", "r1", "rn"), \
    ("swing", "low3", "dip"), (1.5, 2.0, 3.0), (0, 1, -1)
KEY = ["htf", "sig", "poc", "stop", "target", "side"]
TAUGHT = dict(htf=30, sig="flux", poc="n1", stop="swing", target=2.0, side=0)
LTF_LEN, LTF_MULT, HTF_LEN, HTF_MULT = 20, 2.0, 20, 2.25
VP_BARS, VP_ROWS, W_IMPACT, MAX_DAY, MIN_N = 200, 30, 0.85, 2, 150


# ---------- data ----------

def build(raw, sym, out):
    """1-minute pickle with volume (Binance: real, HistData: its v column, 0)."""
    frames = []
    for z in sorted(glob.glob(f"{raw}/*.zip")):
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if not n.lower().endswith(".csv"):
                    continue
                with zf.open(n) as f:
                    if sym.endswith("USDT"):
                        df = pd.read_csv(f, header=None, usecols=range(6), names=["t", "open", "high", "low", "close", "vol"])
                        t = df.t.astype("int64")
                        t = np.where(t > 1e14, t // 1000, t)
                        df.index = pd.to_datetime(t, unit="ms", utc=True).tz_convert(NY)
                    else:
                        df = pd.read_csv(f, sep=";", header=None, names=["t", "open", "high", "low", "close", "vol"])
                        df.index = pd.to_datetime(df.t, format="%Y%m%d %H%M%S").dt.tz_localize(
                            NY, ambiguous="NaT", nonexistent="NaT")
                        df = df[df.index.notna()]
                    frames.append(df[["open", "high", "low", "close", "vol"]])
    df = pd.concat(frames).sort_index()
    df = df[~df.index.duplicated()]
    df.index.name = None
    print(sym, df.index[0], df.index[-1], len(df), "volume > 0:", round((df.vol > 0).mean(), 3))
    df.to_pickle(out)


def has_volume(m1):
    return "vol" in m1 and (m1.vol > 0).mean() > 0.5


# ---------- indicators ----------

def kalman_flux(c):
    """Flux Charts' kalman_filter(close): process noise 0.2, measurement error 2.0, error estimate starts at 1."""
    out = np.full(len(c), np.nan)
    est, err = np.nan, 1.0
    for i in range(len(c)):
        if np.isnan(est) and i > 0:
            est = c[i - 1]
        gain = err / (err + 2.0)
        est = est + gain * (c[i] - est)
        err = (1 - gain) * err + 0.2
        out[i] = est
    return out


def htf_session(m1, tf):
    """tf-minute candles that start at the 9:30 open, as TradingView builds them on a regular-session chart
    (60m: 9:30, 10:30 ... 15:30; the last one is short)."""
    tod = m1.index.hour * 60 + m1.index.minute
    day = m1.index.tz_localize(None).normalize()
    start = day + pd.to_timedelta(RTH_OPEN + (tod - RTH_OPEN) // tf * tf, unit="min")
    g = m1.groupby(start.tz_localize(m1.index.tz))
    r = g.agg({"open": "first", "high": "max", "low": "min", "close": "last"})
    bar = m1.index.to_series().diff().dropna().min() if len(m1) > 1 else pd.Timedelta(minutes=1)   # 1m or 5m input
    r["avail"] = g.apply(lambda x: x.index[-1]) + min(bar, pd.Timedelta(minutes=5))
    return r


def htf_bands(m1, b, tf, session=False):
    """ta.bb(close, 20, 2.25) on completed tf candles, stepped onto the 5m candles, then EMA(20) on the 5m chart."""
    r = htf_session(m1, tf) if session else r13.htf(m1, tf)
    mid = r.close.rolling(HTF_LEN).mean()
    sd = r.close.rolling(HTF_LEN).std(ddof=0)
    x = pd.DataFrame({"t": r.avail.values, "up": (mid + HTF_MULT * sd).values, "lo": (mid - HTF_MULT * sd).values})
    t = pd.DataFrame({"t": (b.index + pd.Timedelta(minutes=5)).values})
    x["t"] = x.t.astype(t.t.dtype)
    s = pd.merge_asof(t, x.dropna().sort_values("t"), on="t")
    mid = (s.up + s.lo) / 2   # ta.bb basis = middle of the two bands
    return np.column_stack([ema(s.up, HTF_LEN).values, ema(s.lo, HTF_LEN).values, ema(mid, HTF_LEN).values])


def prep(m1, vsrc, bars="24h"):
    if bars == "rth":
        tod = m1.index.hour * 60 + m1.index.minute
        m1 = m1[(tod >= RTH_OPEN) & (tod < RTH_CLOSE)]
    b = make_bars(m1, 5)
    v = {"real": m1.vol if vsrc == "real" else None, "proxy": m1.high - m1.low,
         "tpo": pd.Series(1.0, m1.index)}[vsrc]
    b["vol"] = v.resample("5min", label="left", closed="left").sum().reindex(b.index).fillna(0).values
    c = b.close.values
    b["l_mid"] = kalman_flux(c)
    dev = LTF_MULT * b.close.rolling(LTF_LEN).std(ddof=0).values
    b["l_up"], b["l_lo"] = b.l_mid + dev, b.l_mid - dev
    win = np.where((b.tod.values >= RTH_OPEN) & (b.end.values <= LAST_ENTRY))[0]
    win = win[win >= VP_BARS]
    return {"b": b, "hb": {tf: htf_bands(m1, b, tf, bars == "rth") for tf in HTFS}, "win": win,
            "poc": {w: poc_touch(b, win, w) for w in ("n", "r")}, "swing": swings(b)}


def poc_touch(b, win, weight):
    """For each entry-window candle: (candle within one row of the POC line, POC line through the candle)."""
    h, l, v = b.high.values, b.low.values, b.vol.values
    n = VP_BARS + 1                                   # Pine loops i = 0 .. analyzeBars: 201 candles
    age = np.arange(n)[::-1]                          # window is oldest .. newest; age 0 = the signal candle
    wt = W_IMPACT / (age + 1) + (1 - W_IMPACT) if weight == "r" else np.ones(n)
    H, L, V = (sliding_window_view(a, n) for a in (h, l, v))
    t3, t1 = np.zeros(len(win), bool), np.zeros(len(win), bool)
    win = win[win >= n - 1] if len(win) else win
    off = len(t3) - len(win)
    for s in range(0, len(win), 500):
        k = win[s:s + 500] - n + 1
        hh, ll, vv = H[k], L[k], V[k] * wt
        top, bot = hh[:, 1:].max(1), ll[:, 1:].min(1)  # ta.highest / ta.lowest over 200 candles
        step = (top - bot) / VP_ROWS
        edges = bot[:, None] + step[:, None] * np.arange(VP_ROWS + 1)   # rows bottom -> top
        touch = (ll[:, :, None] <= edges[:, None, 1:]) & (hh[:, :, None] >= edges[:, None, :-1])
        prof = (touch * vv[:, :, None]).sum(1)
        poc = bot + step * (prof.argmax(1) + 0.5)     # first max from the bottom = Pine's tie rule (lower row)
        near = (ll[:, -1] <= poc + step) & (hh[:, -1] >= poc - step)
        on = (ll[:, -1] <= poc) & (hh[:, -1] >= poc)
        t3[off + s:off + s + 500], t1[off + s:off + s + 500] = near, on
    return t3, t1


def swings(b):
    h, l = b.high.values, b.low.values
    n = len(b)
    sl, sh = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(4, n):   # swing at i-2 confirmed at i
        j = i - 2
        sl[i] = l[j] if l[j] == l[j - 2:i + 1].min() else sl[i - 1]
        sh[i] = h[j] if h[j] == h[j - 2:i + 1].max() else sh[i - 1]
    return sl, sh


def signals(d, htf, kind):
    """Buy / sell bools per 5m candle and the setup's extreme (dip low / spike high) at that candle."""
    b = d["b"]
    up, lo = d["hb"][htf][:, 0], d["hb"][htf][:, 1]
    c, h, l = b.close.values, b.high.values, b.low.values
    lu, ll = b.l_up.values, b.l_lo.values
    n = len(b)
    buy, sell = np.zeros(n, bool), np.zeros(n, bool)
    dip, spike = np.full(n, np.nan), np.full(n, np.nan)
    if kind == "flux":   # the Pine state machine, line by line
        ca = cb = bear = bull = False
        lo_x, hi_x = np.inf, -np.inf
        for i in range(1, n):
            if np.isnan(up[i]) or np.isnan(lu[i]):
                continue
            if c[i] > up[i]:
                if not ca:
                    hi_x = -np.inf
                ca = True
            if c[i] < lo[i]:
                if not cb:
                    lo_x = np.inf
                cb = True
            if ca:
                hi_x = max(hi_x, h[i])
            if cb:
                lo_x = min(lo_x, l[i])
            if ca and lu[i] < lu[i - 1] and not bear:
                sell[i], spike[i], bear = True, hi_x, True
            if cb and ll[i] > ll[i - 1] and not bull:
                buy[i], dip[i], bull = True, lo_x, True
            if c[i] < lu[i] and c[i] < up[i]:
                ca = bear = False
            if c[i] > ll[i] and c[i] > lo[i]:
                cb = bull = False
    else:                # 'fill': the green (red) fill between the lower (upper) bands ends
        with np.errstate(invalid="ignore"):
            g_in, r_in = ll < lo, lu > up
        lo_x, hi_x = np.inf, -np.inf
        for i in range(1, n):
            if g_in[i]:
                lo_x = min(lo_x if g_in[i - 1] else np.inf, l[i])
            elif g_in[i - 1] and not np.isnan(ll[i]) and not np.isnan(lo[i]):
                buy[i], dip[i] = True, min(lo_x, l[i])
            if r_in[i]:
                hi_x = max(hi_x if r_in[i - 1] else -np.inf, h[i])
            elif r_in[i - 1] and not np.isnan(lu[i]) and not np.isnan(up[i]):
                sell[i], spike[i] = True, max(hi_x, h[i])
    return buy, sell, dip, spike


def events(d, htf, kind, poc, stop, sig=None):
    """DataFrame indexed by bar position: side, entry, stop (r13.to_sig format)."""
    b, win = d["b"], d["win"]
    buy, sell, dip, spike = sig if sig is not None else signals(d, htf, kind)
    ok = np.ones(len(win), bool) if poc == "none" else d["poc"][poc[0]][poc[1] == "1"]
    c, h, l = b.close.values, b.high.values, b.low.values
    sl, sh = d["swing"]
    rows = []
    for side, mask, ext in ((1, buy, dip), (-1, sell, spike)):
        sel = win[mask[win] & ok]
        if stop == "dip":
            st = ext[sel]
        elif stop == "low3":
            st = np.array([l[i - 2:i + 1].min() if side == 1 else h[i - 2:i + 1].max() for i in sel])
        else:
            st = (sl if side == 1 else sh)[sel]
            fb = np.array([l[i - 11:i + 1].min() if side == 1 else h[i - 11:i + 1].max() for i in sel])
            bad = np.isnan(st) | ((st >= c[sel]) if side == 1 else (st <= c[sel]))
            st = np.where(bad, fb, st) if len(sel) else st
        rows.append(pd.DataFrame({"side": side, "entry": c[sel], "stop": np.asarray(st, float)}, index=sel))
    return pd.concat(rows).sort_index(kind="stable")


def grid_rows(d, cost, tags, htfs=HTFS, sigs=SIGS):
    b = d["b"]
    rows = []
    for htf, kind in itertools.product(htfs, sigs):
        sg = signals(d, htf, kind)
        for poc, stop in itertools.product(POCS, STOPS):
            e = events(d, htf, kind, poc, stop, sg)
            for target, side in itertools.product(TARGETS, SIDES):
                tr = simulate(b, r13.to_sig(b, e, target, 0.0, MAX_DAY, side), cost=0.0)
                tr["pct"] = tr.pct - cost
                rows.append(dict(**tags, htf=htf, sig=kind, poc=poc, stop=stop, target=target, side=side, **stats(tr)))
    return rows


# ---------- runs ----------

def markets(shard, of):
    names = sorted(m for m in M if m != NASDAQ)   # same shards as round 13b (its download cache)
    return [m for k, m in enumerate(names) if k % of == shard]


def load(data, mk, sess):
    f, cost, _ = M[mk]
    m1 = reclock(pd.read_pickle(f"{data}/{f}.pkl"), sess)
    return m1[m1.index >= "2019-01-01"], cost


def run_one(job):
    data, out, mk, sess = job
    tag = f"{out}/rows_{mk.replace(' ', '_')}__{sess}.csv"
    if os.path.exists(tag):
        return
    m1, cost = load(data, mk, sess)
    rows = []
    for vsrc in (("real", "proxy") if has_volume(m1) else ("proxy",)):
        rows += grid_rows(prep(m1, vsrc), cost, dict(market=mk, session=sess, cost=cost, bars="24h", vol=vsrc))
    pd.DataFrame(rows).to_csv(tag, index=False)
    print(mk, sess, "done", flush=True)


def run(data, out, shard, of):
    os.makedirs(out, exist_ok=True)
    jobs = []
    for mk in markets(shard, of):
        f, _, sessions = M[mk]
        if not os.path.exists(f"{data}/{f}.pkl"):
            print("no data", mk, flush=True)
            continue
        jobs += [(data, out, mk, s) for s in sessions]
    with ProcessPoolExecutor(4) as ex:
        list(ex.map(run_one, jobs))


NQ_VARIANTS = [(bars, vsrc) for bars in ("24h", "rth") for vsrc in ("proxy", "tpo")]


def _nasdaq_part(args):
    data, bars, vsrc, htf = args
    m1, cost = load(data, NASDAQ, "US")
    d = prep(m1, vsrc, bars)
    rows = grid_rows(d, cost, dict(market=NASDAQ, session="US", cost=cost, bars=bars, vol=vsrc), htfs=(htf,))
    print("nasdaq part", bars, vsrc, htf, flush=True)
    return rows


def trades(d, cost, p, **over):
    p = dict(p, **over)
    e = events(d, int(p["htf"]), p["sig"], p["poc"], p["stop"])
    tr = simulate(d["b"], r13.to_sig(d["b"], e, float(p["target"]), 0.0, MAX_DAY, int(p["side"])), cost=0.0)
    tr["pct"] = tr.pct - cost
    return tr


def nasdaq(data, out):
    """Nasdaq grid (rows file for the report) + the detail: as taught, best on dev with checks, vs the live bot."""
    os.makedirs(out, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        parts = list(ex.map(_nasdaq_part, [(data, bars, v, h) for bars, v in NQ_VARIANTS for h in HTFS]))
    g = pd.DataFrame([r for part in parts for r in part])
    g.to_csv(f"{out}/rows_{NASDAQ.replace(' ', '_')}__US.csv", index=False)

    m1, cost = load(data, NASDAQ, "US")
    dd = {}
    get = lambda bars, v: dd.setdefault((bars, v), prep(m1, v, bars))
    live = r13.current_desk(f"{data}/{M[NASDAQ][0]}.pkl")
    rows = [dict(model=k, variant="", test="as live", **stats(v)) for k, v in live.items()]
    side = {0: "both", 1: "long", -1: "short"}
    tag = lambda p: (f"{int(p['htf'])}m HTF, {p['sig']} signal, POC {p['poc']}, stop {p['stop']}, {p['target']}R, "
                     f"{side[int(p['side'])]}")
    for bars, v in NQ_VARIANTS:
        d = get(bars, v)
        rows.append(dict(model=f"as taught ({tag(TAUGHT)})", variant=f"{bars} {v}", test="as set",
                         **stats(trades(d, cost, TAUGHT))))
        for t, over in (("no POC filter", dict(poc="none")), ("near the POC", dict(poc="nn")),
                        ("recent-weighted POC", dict(poc="r1")), ("stop dip", dict(stop="dip")),
                        ("longs only", dict(side=1)), ("shorts only", dict(side=-1))):
            rows.append(dict(model=f"as taught ({tag(TAUGHT)})", variant=f"{bars} {v}", test=t,
                             **stats(trades(d, cost, TAUGHT, **over))))
    ok = g[g.n >= MIN_N].sort_values("dev_pf", ascending=False)
    top = ok.iloc[0]
    pick = top[KEY].to_dict()
    d = get(top.bars, top.vol)
    best = trades(d, cost, pick)
    name, var = f"best on dev ({tag(pick)})", f"{top.bars} {top.vol}"
    tests = [("as set", {}, cost), ("double cost", {}, 2 * cost), ("no POC filter", dict(poc="none"), cost)]
    tests += [(f"HTF {h}m", dict(htf=h), cost) for h in HTFS if h != pick["htf"]]
    tests += [(f"target {t}R", dict(target=t), cost) for t in TARGETS if t != pick["target"]]
    tests += [(f"stop {s}", dict(stop=s), cost) for s in STOPS if s != pick["stop"]]
    tests += [(f"{side[s]} sides", dict(side=s), cost) for s in SIDES if s != pick["side"]]
    for t, over, cst in tests:
        rows.append(dict(model=name, variant=var, test=t, **stats(trades(d, cst, pick, **over))))
    for bars, v in NQ_VARIANTS:   # the same pick on the other chart / volume versions
        if (bars, v) != (top.bars, top.vol):
            rows.append(dict(model=name, variant=f"{bars} {v}", test="same setting",
                             **stats(trades(get(bars, v), cost, pick))))
    day = lambda t: t.groupby("day").pct.sum()
    both = live["noise + sweep (live bot)"]
    corr = pd.concat([day(both), day(best)], axis=1).fillna(0).corr().iloc[0, 1]
    allt = pd.concat([both, best]).sort_values("day", kind="stable")
    rows.append(dict(model="live bot + best on dev", variant=var, test=f"together (daily corr {corr:.2f})", **stats(allt)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/round14_nasdaq.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)
    print(df.drop(columns=["years"], errors="ignore").to_string())
    print(f"\nTOP 25 BY DEV PF (>= {MIN_N} trades)\n" + ok.head(25).drop(columns=["years"]).to_string())
    for k in KEY + ["bars", "vol"]:
        print(g[g.n >= MIN_N].groupby(k)[["dev_pf", "hold_pf"]].median().round(2).to_string(), "\n")
    print(f"{((g.dev_pf > 1) & (g.hold_pf > 1)).mean():.0%} of settings PF > 1 in both periods")


# ---------- tweaks (Nasdaq, on top of the best grid settings; picked on 2019-2023 only) ----------

TWEAKS = {
    "window 9:30-12:00": dict(win=(570, 720)), "window 10:00-15:30": dict(win=(600, 930)),
    "window 12:00-15:30": dict(win=(720, 930)), "window 9:30-11:00": dict(win=(570, 660)),
    "risk >= 0.10%": dict(min_risk=0.001), "risk >= 0.15%": dict(min_risk=0.0015), "risk <= 0.5%": dict(max_risk=0.005),
    "stop 0.05% beyond": dict(buf=0.0005), "max 1 a day": dict(max_day=1), "max 3 a day": dict(max_day=3),
    "target = HTF middle band": dict(tp="mid"), "target = LTF far band": dict(tp="ltf"),
    "with the HTF trend": dict(trend=1), "against the HTF trend": dict(trend=-1),
    "with the day (vs open)": dict(day=1), "against the day (vs open)": dict(day=-1),
}


def tweaked(d, cost, p, win=None, min_risk=0.0, max_risk=1.0, buf=0.0, max_day=MAX_DAY, tp=None, trend=0, day=0):
    """Trades for setting p with extra rules. trend: +1 = only trades in the HTF middle band's direction (its slope
    over the last 6 candles), -1 = only against it. day: +1 = longs above / shorts below today's 9:30 open."""
    b = d["b"]
    e = events(d, int(p["htf"]), p["sig"], p["poc"], p["stop"])
    if int(p["side"]):
        e = e[e.side == int(p["side"])]
    i = e.index.values
    st = e.stop.values * (1 - e.side.values * buf)
    risk = (e.entry.values - st) * e.side.values
    keep = (risk > 0) & (risk >= min_risk * e.entry.values) & (risk <= max_risk * e.entry.values)
    if win:
        keep &= (b.tod.values[i] >= win[0]) & (b.end.values[i] <= win[1])
    if trend:
        mid = d["hb"][int(p["htf"])][:, 2]
        slope = np.sign(mid[i] - mid[np.maximum(i - 6, 0)])
        keep &= slope == trend * e.side.values
    if day:
        op = b.groupby("tdate").apply(lambda x: x.open[x.tod >= RTH_OPEN].iloc[0] if (x.tod >= RTH_OPEN).any() else np.nan)
        o = b.tdate.map(op).values[i]
        keep &= np.sign(e.entry.values - o) == day * e.side.values
    e, st, risk = e[keep], st[keep], risk[keep]
    i = e.index.values
    if tp == "mid":
        tgt = d["hb"][int(p["htf"])][i, 2]
    elif tp == "ltf":
        tgt = np.where(e.side.values == 1, b.l_up.values[i], b.l_lo.values[i])
    else:
        tgt = e.entry.values + e.side.values * float(p["target"]) * risk
    n = len(b)
    out = {k: np.full(n, np.nan) for k in ("sl_l", "tp_l", "sl_s", "tp_s")}
    le, se = np.zeros(n, bool), np.zeros(n, bool)
    for k, s_, stp, t_ in zip(i, e.side.values, st, tgt):
        if s_ == 1:
            le[k], out["sl_l"][k], out["tp_l"][k] = True, stp, t_
        else:
            se[k], out["sl_s"][k], out["tp_s"][k] = True, stp, t_
    sig = {k: pd.Series(v, b.index) for k, v in out.items()}
    sig.update(le=pd.Series(le, b.index), se=pd.Series(se, b.index), max_day=max_day, start=RTH_OPEN)
    tr = simulate(b, sig, cost=0.0)
    tr["pct"] = tr.pct - cost
    return tr


def tweak(data, rows_file, out, top=6, min_n=100):
    """Every tweak alone on the top grid settings (distinct signal/POC/stop), then the dev-improving ones stacked."""
    global HTFS
    g = pd.read_csv(rows_file)
    ok = g[g.n >= min_n].sort_values("dev_pf", ascending=False)
    picks = ok.drop_duplicates(["bars", "vol", "htf", "sig", "poc", "stop", "side"]).head(top)
    HTFS = tuple(sorted(set(HTFS) | set(picks.htf.astype(int))))
    m1, cost = load(data, NASDAQ, "US")
    dd, rows = {}, []
    for _, p in picks.iterrows():
        d = dd.setdefault((p.bars, p.vol), prep(m1, p.vol, p.bars))
        name = f"{p.bars} {p.vol} | " + " ".join(str(p[k]) for k in KEY)
        base = stats(tweaked(d, cost, p))
        rows.append(dict(setting=name, tweak="(none)", **base))
        good = []
        for t, kw in TWEAKS.items():
            r = stats(tweaked(d, cost, p, **kw))
            rows.append(dict(setting=name, tweak=t, **r))
            if r.get("n", 0) >= min_n and r.get("dev_pf", 0) > base.get("dev_pf", 9) + 0.03:
                good.append((r["dev_pf"], t))
        kw, cur, used = {}, base.get("dev_pf", 0), []
        for _, t in sorted(good, reverse=True):   # greedy stack, dev only
            k2 = dict(kw, **TWEAKS[t])
            r = stats(tweaked(d, cost, p, **k2))
            if r.get("n", 0) >= min_n and r.get("dev_pf", 0) > cur + 0.02:
                kw, cur, used = k2, r["dev_pf"], used + [t]
        if used:
            rows.append(dict(setting=name, tweak="STACKED: " + " + ".join(used), **stats(tweaked(d, cost, p, **kw))))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/round14_tweaks.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(df.drop(columns=["years"], errors="ignore").to_string())


# ---------- the final Nasdaq model on every other market (regular-session candles of each market's session) ----------

FINAL = dict(htf=60, sig="flux", poc="none", stop="swing", target=2.0, side=0, min_risk=0.0015)


def _final_one(job):
    global HTFS
    data, mk, sess = job
    HTFS = (45, 60, 90)
    m1, cost = load(data, mk, sess)
    d = prep(m1, "real" if has_volume(m1) else "proxy", "rth")
    rows = []
    for htf, target, side, mr in itertools.product(HTFS, TARGETS, SIDES, (0.0015, 0.0)):
        p = dict(FINAL, htf=htf, target=target, side=side)
        rows.append(dict(market=mk, session=sess, cost=cost, htf=htf, target=target, side=side, min_risk=mr,
                         **stats(tweaked(d, cost, p, min_risk=mr))))
    print(mk, sess, "done", flush=True)
    return rows


def final_markets(data, out):
    jobs = [(data, mk, s) for mk in sorted(M) for s in M[mk][2] if os.path.exists(f"{data}/{M[mk][0]}.pkl")]
    with ProcessPoolExecutor(4) as ex:
        g = pd.DataFrame([r for part in ex.map(_final_one, jobs) for r in part])
    g.to_csv(f"{out}/round14_final_markets_grid.csv", index=False)
    is_f = (g.htf == 60) & (g.target == 2.0) & (g.side == 0) & (g.min_risk == 0.0015)
    rows = []
    for (mk, sess), x in g.groupby(["market", "session"]):
        f = x[is_f.loc[x.index]]
        r = dict(market=mk, session=sess, cost=x.cost.iat[0])
        for c in ("n", "per_yr", "dev_pf", "dev_yr", "hold_pf", "hold_yr", "pos_years"):
            r[f"final: {c}"] = f[c].iat[0] if len(f) and c in f else np.nan
        ok = x[x.n >= 100]
        r["neighbours tested"] = len(ok)
        r["neighbours dev>1 & hold>1"] = round(((ok.dev_pf > 1) & (ok.hold_pf > 1)).mean(), 2) if len(ok) else np.nan
        r["neighbours median hold PF"] = ok.hold_pf.median() if len(ok) else np.nan
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(f"{out}/round14_final_markets.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 300)
    print(s.to_string())
    h = s["final: hold_pf"]
    both = (s["final: dev_pf"] > 1) & (h > 1)
    print(f"final rule: tested {h.notna().sum()}, PF > 1 in both periods {both.sum()}, holdout PF > 1.15 "
          f"{(h > 1.15).sum()}, median dev / holdout PF {s['final: dev_pf'].median():.2f} / {h.median():.2f}")
    print("PF > 1 in both: " + ", ".join(s[both].market + " " + s[both].session))


def report(rows_dir, out):
    g = pd.concat([pd.read_csv(f) for f in glob.glob(f"{rows_dir}/**/rows_*.csv", recursive=True)])
    g.to_csv(f"{out}/round14_grid.csv.gz", index=False)
    ok = g[g.n >= 100]
    nq = ok[(ok.market == NASDAQ) & (ok.bars == "24h") & (ok.vol == "proxy")].sort_values("dev_pf", ascending=False)
    npick = nq.iloc[0][KEY].to_dict() if len(nq) else None
    is_ = lambda x, p: np.logical_and.reduce([x[k] == v for k, v in p.items()])
    cols = ["n", "per_yr", "win", "dev_pf", "dev_yr", "hold_pf", "hold_yr", "pos_years"]
    rows = []
    for (mk, sess, bars, vsrc), x in g.groupby(["market", "session", "bars", "vol"]):
        r = dict(market=mk, session=sess, bars=bars, vol=vsrc, cost=x.cost.iat[0])
        for name, p in (("taught", TAUGHT), ("taught no POC", dict(TAUGHT, poc="none")), ("nasdaq pick", npick)):
            y = x[is_(x, p)] if p else x.iloc[:0]
            for c in ("n", "dev_pf", "hold_pf", "hold_yr"):
                r[f"{name}: {c}"] = y[c].iat[0] if len(y) and c in y else np.nan
        o = x[x.n >= 100].sort_values("dev_pf", ascending=False)
        if len(o):
            p = o.iloc[0]
            r.update({"best: setting": " ".join(str(p[k]) for k in KEY), **{f"best: {c}": p[c] for c in cols}})
        r["share dev>1 & hold>1"] = round(((x.dev_pf > 1) & (x.hold_pf > 1)).mean(), 3)
        rows.append(r)
    s = pd.DataFrame(rows)
    s.to_csv(f"{out}/round14_summary.csv", index=False)
    pd.set_option("display.width", 300)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.max_rows", 300)
    print(s.to_string())
    print("\nnasdaq pick (24h, proxy volume):", npick)
    for c in [c for c in s if c.endswith("hold_pf")]:
        h = s[c]
        print(f"{c}: tested {h.notna().sum()}, holdout PF > 1: {(h > 1).sum()}, > 1.15: {(h > 1.15).sum()}, "
              f"median {h.median():.2f}")
    both = (s["best: dev_pf"] >= 1.15) & (s["best: hold_pf"] >= 1.15)
    print("best on dev >= 1.15 in both periods: " + ", ".join(s[both].market + " " + s[both].session + " " + s[both].vol))
    print("\nmedian PF per choice (settings with 100+ trades, all markets):")
    for k in KEY + ["bars", "vol"]:
        print(ok.groupby(k)[["dev_pf", "hold_pf"]].median().round(2).to_string(), "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("list", "build", "run", "nasdaq", "tweak", "final", "report"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--raw")
    ap.add_argument("--sym")
    ap.add_argument("--data")
    ap.add_argument("--rows")
    ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "list":
        for mk in markets(a.shard, a.of):
            f = M[mk][0]
            print(("binance " if f.endswith("USDT") else "histdata ") + f)
    elif a.cmd == "build":
        build(a.raw, a.sym, a.out)
    elif a.cmd == "run":
        run(a.data, a.out, a.shard, a.of)
    elif a.cmd == "nasdaq":
        nasdaq(a.data, a.out)
    elif a.cmd == "tweak":
        tweak(a.data, a.rows, a.out)
    elif a.cmd == "final":
        final_markets(a.data, a.out)
    else:
        os.makedirs(a.out, exist_ok=True)
        report(a.rows, a.out)
