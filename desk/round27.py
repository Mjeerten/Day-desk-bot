"""Round 27: TikTok @quantedge7 'QuantEdge' strategies (Oct 2026) and their 6-strategy portfolio ('0 losing years,
15.5 %/yr 2017-2026, worst drop -5.3%, $100k -> $405k, each strategy 1/6, rebalanced monthly').

Rules as shown in their videos (frames + voice-over), Nasdaq (their NQ futures / QQQ) on HistData 1-minute CFD prices,
New York clock. No volume in the data, so 'VWAP' = the running average of the 5-minute typical price since 9:30 (TWAP).
  1. Noise Area (paper 'Beat the Market', Zarattini/Aziz/Barbon): band = the day's open (and prior close) +/- the average
     move from the open at that time of day (14 days); every 30 minutes from 10:00: above the band long, below short;
     out when back inside the band or through VWAP (trailing); flat at 16:00. (Their version: up to 3x leverage, sized by
     volatility; here 1x.) Our live noise band is the long-only version of this paper.
  2. Drift VWAP (Matteo Conti): 5-minute candles; above a rising VWAP a small red candle = buy at its close (mirror below
     a falling VWAP); target +40 points, stop -80 points; flat by the close.
  3. Fabio ORB (Fabio Valentini via Conti): high/low of 9:30-10:00; a 5-minute close above the high = long; stop the range
     low, target 1R; out at 15:00; long only, max 1 a day.
  4. Vault Break (Conti 'top 3', #1): each day the normal daily range (14-day average high-low); a 30-minute candle that
     closes 30% of it above the day's open and above VWAP = buy; target +40 points, stop -75 points; out by the afternoon
     (15:00); long only, up to 3 trades a day.
  5. Overnight Bias ORB (Conti 'top 3', #3): where 9:30 opens in the overnight range (18:00-9:30): top third = only longs,
     bottom third = only shorts, middle = no trade; breakout (close) of the 9:30-9:45 range; trend filter ADX(14) > 20;
     small stop (the other side of the 15-minute range / its middle), target 3R; out by the afternoon; up to 4 trades a day.
     Plain version (no bias filter) as their comparison ('bias doubles it').
  6. BTC trend (paper 'Catching Crypto Trends', Zarattini/Pagani/Barbon): daily closes; 9 Donchian channels (5 days to
     1 year); each is long after a close at a new channel high, out when the close falls under the channel middle;
     position = average of the 9, scaled down when 30-day volatility is above 60%/yr. Binance BTC/USDT, 0.15% cost
     (Trading 212 crypto CFD spread, table M), 0.02% a night.
Their point values (40 / 75 / 80) are NQ points, the same on our CFD (US Tech 100 = NQ price). A % version scales them
to today's price (40 points = 0.14% at 28,000) for comparison. Cost 2 points a round trip on the Nasdaq.
Each strategy 1x on its own (% of price a day), dev 2019-2023, holdout 2024+, then their portfolio (equal weight, 1/6 each,
monthly) next to our desk (live three + Monday dip + Session Momentum).

Run: python -m desk.round27 --data <dir with NSXUSD.pkl, BTCUSDT.pkl> --out <dir>
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from desk.ifvg import stats

NY = "America/New_York"
COST_PTS = 2.0
NIGHT_FEE = 0.02


# ---------- data ----------

def five(m1):
    b = m1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    t = b.index.tz_localize(None)
    b["tod"] = t.hour * 60 + t.minute
    b["day"] = t.normalize()
    b["sess"] = (t + pd.Timedelta(hours=6)).normalize()          # 18:00 starts the next session
    return b


def adx(h, l, c, n=14):
    up, dn = np.r_[0, np.diff(h)], np.r_[0, -np.diff(l)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.maximum.reduce([h - l, np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])])
    a = 1 / n
    atr = pd.Series(tr).ewm(alpha=a, adjust=False).mean()
    pdi = 100 * pd.Series(pdm).ewm(alpha=a, adjust=False).mean() / atr
    ndi = 100 * pd.Series(ndm).ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - ndi).abs() / (pdi + ndi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean().to_numpy()


def prep(m1):
    b = five(m1)
    b = b[b.index >= pd.Timestamp("2018-11-01", tz=b.index.tz)]
    # ADX(14) on 15-minute candles (all hours), known at each 15-minute close, carried to the 5-minute candles
    q = m1.resample("15min", label="left", closed="left").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    q["adx"] = adx(q.high.to_numpy(), q.low.to_numpy(), q.close.to_numpy())
    known = q.adx.copy()
    known.index = known.index + pd.Timedelta(minutes=15)
    b["adx15"] = known.reindex(b.index + pd.Timedelta(minutes=5), method="ffill").to_numpy()
    rth = b[(b.tod >= 570) & (b.tod < 960)].copy()
    rth["tp"] = (rth.high + rth.low + rth.close) / 3
    rth["vwap"] = rth.groupby("day").tp.transform(lambda s: s.expanding().mean())
    g = rth.groupby("day")
    d = pd.DataFrame({"open": g.open.first(), "high": g.high.max(), "low": g.low.min(), "close": g.close.last()})
    d["adr"] = (d.high - d.low).rolling(14).mean().shift(1)
    d["prev_close"] = d.close.shift(1)
    on = b[(b.tod < 570) | (b.tod >= 1080)].groupby("sess").agg(on_hi=("high", "max"), on_lo=("low", "min"))
    d = d.join(on)
    return rth, d


# ---------- one trade engine for the intraday strategies ----------

def walk(o, h, l, c, i, s, e, stop, tgt, last, cost):
    """From bar i+1 to `last` (exit at its close): stop first, then target. Returns (exit index, exit, why, pct)."""
    for j in range(i + 1, last + 1):
        if (l[j] <= stop) if s == 1 else (h[j] >= stop):
            x = min(stop, o[j]) if s == 1 else max(stop, o[j])
            return j, x, "stop", s * (x - e) / e * 100 - cost / e * 100
        if tgt is not None and ((h[j] >= tgt) if s == 1 else (l[j] <= tgt)):
            x = max(tgt, o[j]) if s == 1 else min(tgt, o[j])
            return j, x, "target", s * (x - e) / e * 100 - cost / e * 100
    x = c[last]
    return last, x, "time", s * (x - e) / e * 100 - cost / e * 100


def days(rth):
    return [(dy, g) for dy, g in rth.groupby("day")]


def pts(p, e, mode):
    """Point distance as taught ('pts') or scaled to 28,000 ('pct')."""
    return p if mode == "pts" else p / 28000 * e


def drift_vwap(rth, mode="pts", tp=40, sl=80, small=0.5, rising=6):
    rows = []
    for dy, g in days(rth):
        o, h, l, c, v, tod = (g[k].to_numpy() for k in ("open", "high", "low", "close", "vwap", "tod"))
        rng = np.abs(c - o)
        avg = pd.Series(h - l).rolling(12, min_periods=3).mean().to_numpy()
        last, i = len(g) - 1, rising
        while i < last:
            up = c[i] > v[i] and v[i] > v[i - rising]
            dn = c[i] < v[i] and v[i] < v[i - rising]
            red, green = c[i] < o[i], c[i] > o[i]
            ok = rng[i] <= small * avg[i] if np.isfinite(avg[i]) else False
            s = 1 if (up and red and ok) else -1 if (dn and green and ok) else 0
            if s == 0:
                i += 1
                continue
            e = c[i]
            j, x, why, pct = walk(o, h, l, c, i, s, e, e - s * pts(sl, e, mode), e + s * pts(tp, e, mode), last, COST_PTS)
            rows.append((dy, s, e, x, why, pct))
            i = j + 1
    return pd.DataFrame(rows, columns=["day", "side", "entry", "exit", "why", "pct"])


def fabio_orb(rth, rr=1.0, out=900, both=False):
    rows = []
    for dy, g in days(rth):
        o, h, l, c, tod = (g[k].to_numpy() for k in ("open", "high", "low", "close", "tod"))
        orr = tod < 600
        if orr.sum() < 6:
            continue
        hi, lo = h[orr].max(), l[orr].min()
        last = int(np.nonzero(tod < out)[0][-1])
        for i in range(int(orr.sum()), last):
            s = 1 if c[i] > hi else (-1 if both and c[i] < lo else 0)
            if s:
                e = c[i]
                st = lo if s == 1 else hi
                risk = abs(e - st)
                j, x, why, pct = walk(o, h, l, c, i, s, e, st, e + s * rr * risk, last, COST_PTS)
                rows.append((dy, s, e, x, why, pct))
                break
    return pd.DataFrame(rows, columns=["day", "side", "entry", "exit", "why", "pct"])


def vault_break(rth, d, mode="pts", tp=40, sl=75, k=0.3, out=900, max_n=3):
    rows = []
    for dy, g in days(rth):
        if dy not in d.index or not np.isfinite(d.at[dy, "adr"]):
            continue
        g30 = g[g.tod % 30 == 25]                                  # the 5-minute candle that ends each 30-minute candle
        op, adr = g.open.iloc[0], d.at[dy, "adr"]
        o, h, l, c, v, tod = (g[k_].to_numpy() for k_ in ("open", "high", "low", "close", "vwap", "tod"))
        last = int(np.nonzero(tod < out)[0][-1])
        pos = {t: n for n, t in enumerate(g.index)}
        free, n = -1, 0
        for t, r in g30.iterrows():
            i = pos[t]
            if i <= free or i >= last or n >= max_n:
                continue
            if r.close >= op + k * adr and r.close > r.vwap:
                e = r.close
                j, x, why, pct = walk(o, h, l, c, i, 1, e, e - pts(sl, e, mode), e + pts(tp, e, mode), last, COST_PTS)
                rows.append((dy, 1, e, x, why, pct))
                free, n = j, n + 1
    return pd.DataFrame(rows, columns=["day", "side", "entry", "exit", "why", "pct"])


def overnight_bias(rth, d, bias=True, stop="other", rr=3.0, adx_min=20, out=900, max_n=4, third=1 / 3, orl=15):
    rows = []
    for dy, g in days(rth):
        if dy not in d.index:
            continue
        oh, ol = d.at[dy, "on_hi"], d.at[dy, "on_lo"]
        o, h, l, c, tod, ax = (g[k].to_numpy() for k in ("open", "high", "low", "close", "tod", "adx15"))
        if len(o) < 10 or not np.isfinite(oh) or oh <= ol:
            continue
        pos_ = (o[0] - ol) / (oh - ol)
        allow = (1,) if pos_ >= 1 - third else (-1,) if pos_ <= third else ()
        if not bias:
            allow = (1, -1)
        if not allow:
            continue
        orr = tod < 570 + orl
        if orr.sum() < orl // 5 or tod[0] != 570:
            continue
        hi, lo = h[orr].max(), l[orr].min()
        last = int(np.nonzero(tod < out)[0][-1])
        i, n = int(orr.sum()), 0
        while i < last and n < max_n:
            s = 1 if c[i] > hi else -1 if c[i] < lo else 0
            if s not in allow or (adx_min and not ax[i] > adx_min):
                i += 1
                continue
            e = c[i]
            st = (lo if s == 1 else hi) if stop == "other" else (hi + lo) / 2
            risk = abs(e - st)
            if risk <= 0:
                i += 1
                continue
            j, x, why, pct = walk(o, h, l, c, i, s, e, st, e + s * rr * risk if rr else None, last, COST_PTS)
            rows.append((dy, s, e, x, why, pct))
            i, n = j + 1, n + 1
    return pd.DataFrame(rows, columns=["day", "side", "entry", "exit", "why", "pct"])


def noise_area(rth, d, both=True, vwap_trail=True, look=14):
    """Every 30 minutes from 10:00: above the band long, below short; out back inside the band (or through VWAP)."""
    g_all = rth.copy()
    op = g_all.groupby("day").open.transform("first")
    g_all["move"] = (g_all.close / op - 1).abs()
    sig = g_all.pivot_table(index="day", columns="tod", values="move")
    sig = sig.rolling(look, min_periods=look).mean().shift(1)
    rows = []
    for dy, g in days(rth):
        if dy not in sig.index or dy not in d.index:
            continue
        o, h, l, c, v, tod = (g[k].to_numpy() for k in ("open", "high", "low", "close", "vwap", "tod"))
        pc = d.at[dy, "prev_close"]
        if not np.isfinite(pc):
            continue
        up_ref, dn_ref = max(o[0], pc), min(o[0], pc)
        sg = sig.loc[dy].reindex(tod).to_numpy()
        pos, e, i0 = 0, 0.0, 0
        for i in range(len(g)):
            if tod[i] % 30 != 25 or tod[i] < 595 or not np.isfinite(sg[i]):
                continue
            ub, lb = up_ref * (1 + sg[i]), dn_ref * (1 - sg[i])
            if pos == 1:
                stop = max(ub, v[i]) if vwap_trail else ub
                if c[i] < stop:
                    rows.append((dy, 1, e, c[i], "exit", (c[i] - e) / e * 100 - COST_PTS / e * 100))
                    pos = 0
            elif pos == -1:
                stop = min(lb, v[i]) if vwap_trail else lb
                if c[i] > stop:
                    rows.append((dy, -1, e, c[i], "exit", (e - c[i]) / e * 100 - COST_PTS / e * 100))
                    pos = 0
            if pos == 0 and tod[i] < 955:
                if c[i] > ub:
                    pos, e = 1, c[i]
                elif both and c[i] < lb:
                    pos, e = -1, c[i]
        if pos:
            x = c[-1]
            rows.append((dy, pos, e, x, "close", pos * (x - e) / e * 100 - COST_PTS / e * 100))
    return pd.DataFrame(rows, columns=["day", "side", "entry", "exit", "why", "pct"])


def btc_trend(btc1m, looks=(5, 10, 20, 30, 60, 90, 150, 250, 360), vol_cap=0.6, cost=0.15):
    """Daily % of the account (position 0..1 of the account in BTC) with cost on every change."""
    dly = btc1m.close.resample("D").last().dropna()
    c = dly.to_numpy()
    n = len(c)
    pos = np.zeros(n)
    for L in looks:
        hh = pd.Series(c).rolling(L).max().to_numpy()
        ll = pd.Series(c).rolling(L).min().to_numpy()
        mid = (hh + ll) / 2
        p, inn = np.zeros(n), False
        for i in range(L, n):
            if not inn and c[i] >= hh[i]:
                inn = True
            elif inn and c[i] < mid[i]:
                inn = False
            p[i] = 1.0 if inn else 0.0
        pos += p / len(looks)
    vol = pd.Series(np.log(c)).diff().rolling(30).std().to_numpy() * np.sqrt(365)
    pos = pos * np.minimum(1, vol_cap / np.where(np.isfinite(vol) & (vol > 0), vol, vol_cap))
    held = np.r_[0, pos[:-1]]                                     # decided at the close, held the next day
    ret = np.r_[0, c[1:] / c[:-1] - 1] * 100
    chg = np.abs(np.diff(np.r_[0, held]))
    pct = held * ret - chg * cost / 2 - held * NIGHT_FEE
    out = pd.DataFrame({"day": dly.index.tz_localize(None).normalize(), "pct": pct, "held": held})
    return out[out.day >= "2019-01-01"]


# ---------- report ----------

def daily(tr):
    return tr.pct.groupby(pd.to_datetime(tr.day).values).sum()


def eq(x, name):
    x = x[x.index >= pd.Timestamp("2019-01-01")]
    out = dict(name=name)
    for k, m in (("dev", x.index < pd.Timestamp("2024-01-01")), ("hold", x.index >= pd.Timestamp("2024-01-01"))):
        y = x[m]
        yrs = max((y.index.max() - y.index.min()).days / 365.25, 0.5) if len(y) else 1
        cum = y.cumsum()
        out[f"{k}_yr"] = round(y.sum() / yrs, 1)
        out[f"{k}_dd"] = round((cum - cum.cummax()).min(), 1)
        out[f"{k}_sharpe"] = round(y.mean() / y.std() * np.sqrt(252), 2) if y.std() > 0 else np.nan
    py = x.groupby(x.index.year).sum()
    out["years"] = " ".join(f"{v:+.0f}" for v in py)
    out["neg_years"] = int((py < 0).sum())
    return out


def main(data, out):
    m1 = pd.read_pickle(f"{data}/NSXUSD.pkl")
    rth, d = prep(m1)
    S = {}
    rows = []

    def add(name, tr, keep=False):
        r = dict(strategy=name, **stats(tr))
        rows.append(r)
        print(r, flush=True)
        if keep:
            S[name] = daily(tr)

    add("1 Noise Area as shown (both sides, VWAP trail)", noise_area(rth, d), True)
    add("1 Noise Area, long only, VWAP trail", noise_area(rth, d, both=False))
    add("1 Noise Area, both sides, band only", noise_area(rth, d, vwap_trail=False))
    add("2 Drift VWAP as shown (40/80 pts)", drift_vwap(rth), True)
    add("2 Drift VWAP, 40/80 scaled to price", drift_vwap(rth, "pct"))
    add("2 Drift VWAP, any red candle", drift_vwap(rth, small=9))
    add("3 Fabio ORB as shown (30 min, 1R, 15:00, long)", fabio_orb(rth), True)
    add("3 Fabio ORB, 2R", fabio_orb(rth, rr=2.0))
    add("3 Fabio ORB, both sides", fabio_orb(rth, both=True))
    add("4 Vault Break as shown (40/75 pts)", vault_break(rth, d), True)
    add("4 Vault Break, 40/75 scaled to price", vault_break(rth, d, "pct"))
    add("4 Vault Break, 1 a day", vault_break(rth, d, max_n=1))
    add("5 Overnight Bias ORB as shown (stop other side, 3R, ADX>20)", overnight_bias(rth, d), True)
    add("5 Overnight Bias ORB, stop mid-range", overnight_bias(rth, d, stop="mid"))
    add("5 plain 15-min ORB (no bias), as their comparison", overnight_bias(rth, d, bias=False))
    add("5 Overnight Bias ORB, no ADX filter", overnight_bias(rth, d, adx_min=0))
    add("5 Overnight Bias ORB, out 15:55", overnight_bias(rth, d, out=955))
    if os.path.exists(f"{data}/BTCUSDT.pkl"):
        bt = btc_trend(pd.read_pickle(f"{data}/BTCUSDT.pkl"))
        add("6 BTC trend (9 Donchian channels, vol cap)", bt.assign(pct=bt.pct), False)
        S["6 BTC trend (9 Donchian channels, vol cap)"] = bt.set_index("day").pct
    g = pd.DataFrame(rows)
    g.to_csv(f"{out}/round27_strategies.csv", index=False)
    cols = ["strategy", "n", "per_yr", "win", "payoff", "dev_pf", "dev_yr", "dev_dd", "hold_pf", "hold_yr", "hold_dd",
            "pos_years", "years"]
    with pd.option_context("display.width", 320, "display.max_columns", 40, "display.max_colwidth", 70):
        print("\nEACH STRATEGY, Nasdaq CFD, 1x, % of price (BTC: % of the account):\n" + g[cols].to_string(index=False))
        cal = pd.concat(S, axis=1, sort=True).fillna(0)
        cal = cal[cal.index >= pd.Timestamp("2019-01-01")]
        port = cal.mean(axis=1)                    # 1/n each, rebalanced daily (close to their monthly)
        res = [eq(cal[k], k) for k in cal] + [eq(port, f"their portfolio: 1/{cal.shape[1]} each")]
        # our desk for comparison (round 25 desk view)
        try:
            from desk.round25 import desk as desk25, BOT
            dk, _ = desk25(data, [("bot", BOT)])
            print("\nOUR DESK (round 25 desk view, same data):\n" + dk.to_string(index=False))
        except Exception as err:  # noqa
            print("desk view not run:", err)
        print("\nPORTFOLIO (each strategy 1x, daily sums; theirs at 1/n each):\n" + pd.DataFrame(res).to_string(index=False))
        print("\ndaily-result correlation:\n" + cal.corr().round(2).to_string())


# ---------- 27b: tune the Overnight Bias ORB on 2019-23, then judge it next to the desk ----------

OB_TAUGHT = dict(third=1 / 3, orl=15, stop="other", rr=3.0, adx_min=20, out=900, max_n=4)


def _ob_job(args):
    data, orl, third = args
    rth, d = _prep_cached(f"{data}/NSXUSD.pkl")
    rows = []
    for stop, rr, adx_min, out, max_n in itertools.product(("other", "mid"), (2.0, 3.0, 4.0, 0), (0, 20, 25), (900, 955), (1, 4)):
        p = dict(third=third, orl=orl, stop=stop, rr=rr, adx_min=adx_min, out=out, max_n=max_n)
        rows.append(dict(**p, **stats(overnight_bias(rth, d, **p))))
    print("orl", orl, "third", round(third, 2), "done", flush=True)
    return rows


_P = {}


def _prep_cached(path):
    if path not in _P:
        _P[path] = prep(pd.read_pickle(path))
    return _P[path]


def main_ob(data, out):
    from concurrent.futures import ProcessPoolExecutor
    jobs = [(data, orl, th) for orl in (15, 30) for th in (0.25, 1 / 3, 0.4, 0.5)]
    with ProcessPoolExecutor(4) as ex:
        g = pd.DataFrame([r for part in ex.map(_ob_job, jobs) for r in part])
    g.to_csv(f"{out}/round27b_grid.csv", index=False)
    keys = list(OB_TAUGHT)
    cols = keys + ["n", "per_yr", "win", "dev_pf", "dev_yr", "dev_dd", "hold_pf", "hold_yr", "hold_dd", "pos_years", "years"]
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print(f"{len(g)} settings, dev PF>1 {(g.dev_pf > 1).mean():.0%}, both {((g.dev_pf > 1) & (g.hold_pf > 1)).mean():.0%}")
        for k in keys:
            print(g.groupby(k)[["dev_yr", "hold_yr", "dev_pf", "hold_pf"]].median().round(2).to_string())
        # robust pick on 2019-23: neighbours (other rr values, same rest) must be profitable too
        nk = [k for k in keys if k != "rr"]
        g["nb"] = g.groupby(nk).dev_pf.transform("min")
        ok = g[(g.n >= 300) & (g.nb > 1)].sort_values("dev_yr", ascending=False)
        print("TOP 10 ON 2019-23 (>= 300 trades, every target setting profitable):\n" + ok.head(10)[cols].to_string(index=False))
        tt = g[(g.third.round(3) == round(1 / 3, 3)) & (g.orl == 15) & (g.stop == "other") & (g.rr == 3) & (g.adx_min == 20)
               & (g.out == 900) & (g.max_n == 4)]
        print("as taught:\n" + tt[cols].to_string(index=False))
        best = {k: (v.item() if hasattr(v, "item") else v) for k, v in ok.iloc[0][keys].to_dict().items()} if len(ok) else OB_TAUGHT
        print("pick:", best)
    # desk view: live three + Monday dip + Session Momentum (bot) + Overnight Bias (taught / pick)
    from desk import round21 as r21
    from desk.ibs import ibs_trades
    from desk.round13 import current_desk
    from desk.round21c import IMPROVED, eq_stats
    from desk.round21c import daily as daily21
    from desk.round25 import BOT, _bars, run as run25
    nq = f"{data}/NSXUSD.pkl"
    m1 = pd.read_pickle(nq)
    irx = pd.Series(4.8, index=pd.date_range("2018-01-01", "2027-01-01"))
    ibs_pct, _ = ibs_trades(m1, irx, 2.0 / 15000 * 100)
    ibs = pd.DataFrame({"pct": ibs_pct.values, "day": ibs_pct.index})
    ibs["t_in"] = (ibs.day + pd.Timedelta(hours=15, minutes=45)).dt.tz_localize(NY)
    ibs["t_out"] = ibs.t_in + pd.Timedelta(days=1)
    ibs.loc[ibs.day.dt.dayofweek == 4, "t_out"] += pd.Timedelta(days=2)
    h = r21.hourly(m1, NY)
    h = h[h.index >= pd.Timestamp("2019-01-01", tz=NY)]
    cv = (h.index + pd.Timedelta("1h")).tz_convert("UTC").tz_localize(None).values
    tiu = pd.DatetimeIndex(ibs.t_in).tz_convert("UTC").tz_localize(None).values
    tou = pd.DatetimeIndex(ibs.t_out).tz_convert("UTC").tz_localize(None).values
    k = np.searchsorted(tiu, cv, side="right") - 1
    held = (k >= 0) & (cv < tou[np.clip(k, 0, None)])
    h2 = h.copy()
    h2.loc[held, "rsi"] = 100.0
    md = r21.simulate(h2, IMPROVED, "US Tech 100")
    dd = current_desk(nq)["noise + sweep (live bot)"]
    live = daily21(dd.assign(t_out=lambda x: pd.to_datetime(x.day).dt.tz_localize(NY)))
    smt = run25(_bars(nq), BOT, 2.0)
    rth, d = _prep_cached(nq)
    cal = pd.concat({"live": live, "ibs": daily21(ibs), "md": daily21(md), "sm": smt.pct.groupby(smt.day.values).sum(),
                     "ob_taught": daily(overnight_bias(rth, d, **OB_TAUGHT)), "ob_pick": daily(overnight_bias(rth, d, **best))},
                    axis=1, sort=True).fillna(0)
    base = cal.live + cal.ibs + cal.md + cal.sm
    res = [eq_stats(cal.live + cal.ibs, "live bot (noise + sweep + IBS)"), eq_stats(base, "desk now (+ Monday dip + Session Momentum)"),
           eq_stats(cal.ob_taught, "Overnight Bias as taught alone"), eq_stats(cal.ob_pick, "Overnight Bias pick alone"),
           eq_stats(base + cal.ob_taught, "desk now + Overnight Bias as taught"), eq_stats(base + cal.ob_pick, "desk now + Overnight Bias pick")]
    r = pd.DataFrame(res)
    r.to_csv(f"{out}/round27b_desk.csv", index=False)
    with pd.option_context("display.width", 320, "display.max_columns", 40):
        print("\nDESK (each strategy 1x, % of price a year):\n" + r.to_string(index=False))
        print("\ndaily-result correlation:\n" + cal.corr().round(2).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ob", action="store_true", help="27b: tune the Overnight Bias ORB + desk view")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (main_ob if a.ob else main)(a.data, a.out)
