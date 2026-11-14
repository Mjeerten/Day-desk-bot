"""Round 25: optimize 'Session Momentum' (round 23) and judge it by what it adds to the desk.

Grid on Nasdaq (and the same on the S&P as a check), 5-minute CFD candles, nearly 24 h, New York clock:
  - ema: 8 / 12 (taught) / 21 / 50 (5-minute closes)
  - at: decision at the 9:35 close (first 5-min candle, taught) or 9:45 (first 15 minutes)
  - stop: 0.8 / 1.2 / 1.6 / 2.0 % of price
  - trail: 0.5 / 1.0 / 1.5 x risk behind the best close, from the start, or none
  - hold: 'day' = flat at 16:00, 'multi' = until the stop (0.02% a night), 'd2' = at most until the second 16:00
  - filt: none, or 'trend' = longs only when yesterday's 16:00 close is above its 50-day average, shorts only below
  - side: both, long only
Selection on 2019-2023 only: highest %/yr among settings with >= 300 trades whose neighbours (same setting with the
next stop up / down) are also profitable; then 2024+ checked. Desk view (Nasdaq, each strategy 1x, daily sums):
live bot (noise + sweep + IBS), + Monday dip (as the bot runs it, round 21e), + Session Momentum as taught / optimized.

Run: python -m desk.round25 --data <dir with NSXUSD.pkl, SPXUSD.pkl> --out <dir>
"""
import argparse
import itertools
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from desk.everything import M
from desk.ifvg import stats
from desk.round18 import lean_bars

MARKETS = {"US Tech 100": "NSXUSD", "US 500": "SPXUSD"}
NIGHT_FEE = 0.02
TAUGHT = dict(ema=12, at=570, stop=1.2, trail=1.0, hold="multi", filt="none", side=0)
KEYS = list(TAUGHT)


def prep(m1):
    b = lean_bars(m1, 5)
    for sp in (8, 12, 21, 50):
        b[f"ema{sp}"] = b.close.ewm(span=sp, adjust=False).mean()
    b["day"] = b.index.tz_localize(None).normalize()
    d = b[b.tod == 955].close.groupby(b.day[b.tod == 955]).last()      # the 15:55-16:00 candle close
    up = (d > d.rolling(50).mean()).shift(1)
    b["trend"] = b.day.map(up).fillna(False).astype(bool)
    return b


def run(b, p, cost):
    o, h, l, c = (b[k].to_numpy() for k in ("open", "high", "low", "close"))
    ema, trend = b[f"ema{p['ema']}"].to_numpy(), b.trend.to_numpy()
    tod, day = b.tod.to_numpy(), b.day.to_numpy()
    sig = np.nonzero(tod == p["at"])[0]
    n = len(b)
    rows, free = [], 0
    sp, tr_, hold = p["stop"] / 100, p["trail"], p["hold"]
    for i in sig:
        if i < free or i + 1 >= n:
            continue
        s = 1 if c[i] > ema[i] else -1
        if (p["side"] and s != p["side"]) or (p["filt"] == "trend" and (s == 1) != trend[i]):
            continue
        i0 = i
        i = i + p.get("delay", 0)                 # entry this many 5-minute candles late (same direction)
        if i >= n or day[i] != day[i0]:
            continue
        e = c[i]
        risk = e * sp
        st, best = e - s * risk, e
        closes = 0
        j, x, why = i + 1, None, "stop"
        while j < n:
            if hold != "multi" and (tod[j] >= 960 and tod[j - 1] < 960 or day[j] != day[j - 1] and tod[j - 1] < 960):
                closes += 1
                if hold == "day" or closes >= 2:
                    x, why = (o[j] if tod[j] >= 960 else c[j - 1]), "close"
                    break
            if (l[j] <= st) if s == 1 else (h[j] >= st):
                x = min(st, o[j]) if s == 1 else max(st, o[j])
                break
            best = max(best, c[j]) if s == 1 else min(best, c[j])
            if tr_ != "none":
                nst = best - s * float(tr_) * risk
                st = max(st, nst) if s == 1 else min(st, nst)
            j += 1
        if x is None:
            j, x, why = n - 1, c[n - 1], "end"
        nights = int((day[j] - day[i]) / np.timedelta64(1, "D"))
        pct = s * (x - e) / e * 100 - cost / e * 100 - NIGHT_FEE * nights
        rows.append((pd.Timestamp(day[i]), s, e, x, why, pct, pct / (sp * 100), nights))
        free = j + 1
    return pd.DataFrame(rows, columns=["day", "side", "entry", "exit", "why", "pct", "R", "nights"])


_D = {}


def _bars(path):
    if path not in _D:
        _D.clear()
        _D[path] = prep(pd.read_pickle(path))
    return _D[path]


def cost_of(b, mk):
    return 2.0 if mk == "US Tech 100" else M[mk][1] / 100 * float(b.close.median())


def _job(args):
    data, mk, ema, at = args
    b = _bars(f"{data}/{MARKETS[mk]}.pkl")
    cst = cost_of(b, mk)
    rows = []
    for stop, trail, hold, filt, side in itertools.product((0.8, 1.2, 1.6, 2.0), (0.5, 1.0, 1.5, "none"),
                                                           ("day", "d2", "multi"), ("none", "trend"), (0, 1)):
        if trail == "none" and hold == "multi":
            continue
        p = dict(ema=ema, at=at, stop=stop, trail=trail, hold=hold, filt=filt, side=side)
        rows.append(dict(market=mk, **p, **stats(run(b, p, cst))))
    print(mk, ema, at, "done", flush=True)
    return rows


def desk(data, picks, cal_only=False):
    """Daily %: live bot (noise + sweep + IBS), Monday dip as the bot runs it, and each Session Momentum pick."""
    from desk import round21 as r21
    from desk.ibs import ibs_trades
    from desk.round13 import current_desk
    from desk.round21c import IMPROVED, daily, eq_stats
    nq = f"{data}/NSXUSD.pkl"
    m1 = pd.read_pickle(nq)
    irx = pd.Series(4.8, index=pd.date_range("2018-01-01", "2027-01-01"))
    ibs_pct, _ = ibs_trades(m1, irx, 2.0 / 15000 * 100)
    ibs = pd.DataFrame({"pct": ibs_pct.values, "day": ibs_pct.index})
    ibs["t_in"] = (ibs.day + pd.Timedelta(hours=15, minutes=45)).dt.tz_localize("America/New_York")
    ibs["t_out"] = ibs.t_in + pd.Timedelta(days=1)
    ibs.loc[ibs.day.dt.dayofweek == 4, "t_out"] += pd.Timedelta(days=2)
    h = r21.hourly(m1, "America/New_York")
    h = h[h.index >= pd.Timestamp("2019-01-01", tz="America/New_York")]
    cv = (h.index + pd.Timedelta("1h")).tz_convert("UTC").tz_localize(None).values
    tiu = pd.DatetimeIndex(ibs.t_in).tz_convert("UTC").tz_localize(None).values
    tou = pd.DatetimeIndex(ibs.t_out).tz_convert("UTC").tz_localize(None).values
    k = np.searchsorted(tiu, cv, side="right") - 1
    held = (k >= 0) & (cv < tou[np.clip(k, 0, None)])
    h2 = h.copy()
    h2.loc[held, "rsi"] = 100.0
    md = r21.simulate(h2, IMPROVED, "US Tech 100")
    dd = current_desk(nq)["noise + sweep (live bot)"]
    live = daily(dd.assign(t_out=lambda x: pd.to_datetime(x.day).dt.tz_localize("America/New_York")))
    b = _bars(nq)
    cols = {"live": live, "ibs": daily(ibs), "md": daily(md)}
    for name, p in picks:
        t = run(b, p, 2.0)
        cols[name] = t.pct.groupby(t.day.values).sum()
    cal = pd.concat(cols, axis=1, sort=True).fillna(0)
    if cal_only:
        return cal
    base = cal.live + cal.ibs
    rows = [eq_stats(base, "live bot (noise + sweep + IBS)"), eq_stats(base + cal.md, "live bot + Monday dip")]
    for name, _ in picks:
        rows.append(eq_stats(cal[name], f"Session Momentum {name} alone"))
        rows.append(eq_stats(base + cal.md + cal[name], f"live bot + Monday dip + Session Momentum {name}"))
    corr = cal.corr().round(2)
    return pd.DataFrame(rows), corr


def main(data, out):
    jobs = [(data, mk, ema, at) for mk in MARKETS for ema in (8, 12, 21, 50) for at in (570, 580)]
    with ProcessPoolExecutor(4) as ex:
        g = pd.DataFrame([r for part in ex.map(_job, jobs) for r in part])
    g.to_csv(f"{out}/round25_grid.csv", index=False)
    cols = KEYS + ["n", "win", "dev_pf", "dev_yr", "dev_dd", "hold_pf", "hold_yr", "hold_dd", "pos_years"]
    picks = [("as taught", TAUGHT)]
    with pd.option_context("display.width", 300, "display.max_columns", 40):
        for mk in MARKETS:
            s = g[g.market == mk].copy()
            print(f"\n{mk}: {len(s)} settings, dev PF>1 {(s.dev_pf > 1).mean():.0%}, both {((s.dev_pf > 1) & (s.hold_pf > 1)).mean():.0%}")
            for k in KEYS:
                print(s.groupby(k)[["dev_yr", "hold_yr", "dev_pf", "hold_pf"]].median().round(2).to_string())
            # neighbours: same setting, next stop up / down, must be profitable on dev
            key = [k for k in KEYS if k != "stop"]
            s["nb_ok"] = s.groupby(key).dev_pf.transform(lambda v: v.rolling(3, center=True, min_periods=2).min()) > 1
            ok = s[(s.n >= 300) & s.nb_ok].sort_values("dev_yr", ascending=False)
            print("TOP 10 ON 2019-23 (>= 300 trades, neighbours profitable):\n" + ok.head(10)[cols].to_string(index=False))
            tt = s[(s[KEYS] == pd.Series(TAUGHT)).all(axis=1)]
            print("as taught:\n" + tt[cols].to_string(index=False))
            if mk == "US Tech 100" and len(ok):
                best = {k: (v.item() if hasattr(v, "item") else v) for k, v in ok.iloc[0][KEYS].to_dict().items()}
                best["ema"], best["at"] = int(best["ema"]), int(best["at"])
                picks.append(("optimized", best))
                day_ok = ok[ok.hold == "day"]
                if len(day_ok):
                    bd = {k: (v.item() if hasattr(v, "item") else v) for k, v in day_ok.iloc[0][KEYS].to_dict().items()}
                    bd["ema"], bd["at"] = int(bd["ema"]), int(bd["at"])
                    picks.append(("optimized, flat at 16:00", bd))
        print("\npicks:", picks)
        d, corr = desk(data, picks)
        d.to_csv(f"{out}/round25_desk.csv", index=False)
        print("\nDESK ON THE NASDAQ (each strategy 1x, % of price a year, drawdown in %):\n" + d.to_string(index=False))
        print("\ndaily-result correlation:\n" + corr.to_string())


BOT = dict(ema=8, at=570, stop=1.2, trail=1.0, hold="day", filt="none", side=0)   # the version in the bot (paper)


def main_late(data, out):
    """25b: the bot's version entered 0 / 1 / 2 / 3 / 6 five-minute candles late (alert delay, slow phone)."""
    rows = []
    for mk in MARKETS:
        b = _bars(f"{data}/{MARKETS[mk]}.pkl")
        for d in (0, 1, 2, 3, 6):
            rows.append(dict(market=mk, delay_min=5 * d, **stats(run(b, dict(BOT, delay=d), cost_of(b, mk)))))
    df = pd.DataFrame(rows)
    df.to_csv(f"{out}/round25b_late.csv", index=False)
    print(df[["market", "delay_min", "n", "win", "dev_pf", "dev_yr", "hold_pf", "hold_yr", "pos_years"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--late", action="store_true", help="25b: late entries for the bot's version")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (main_late if a.late else main)(a.data, a.out)
