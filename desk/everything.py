"""Round 9: every intraday strategy on every Trading 212 market with free 1-minute history.

Markets: HistData 1-minute (indices, gold, silver, oil, 40+ currency pairs) and Binance 1-minute (crypto).
Each market is tested on one or more sessions. The session's open is moved to 9:30 on the engine's clock, so the
library's rules (entries 9:30-15:30, flat at 16:00) apply unchanged:
  US  = 9:30-16:00 New York (15:30-22:00 Amsterdam, the owner's evening)
  EU  = 9:00-15:30 Frankfurt/Paris, LDN = 8:00-14:30 London, TKY = 9:00-15:30 Tokyo,
  SYD = 10:00-16:30 Sydney, HK = 9:30-16:00 Hong Kong.
Strategies: the whole library (desk/library.py, all timeframes), the two live rules and a noise band with shorts,
and a time-of-day grid (hold from hour A to hour B every day, long or short; all 276 windows).
Costs: a Trading 212-like spread per round trip, as a % of price (table M). Every strategy is also split into
its long-only and short-only half. Dev = 2019-2023, holdout = 2024 - Sep 2026.

Run: python -m desk.everything --data <dir with SYMBOL.pkl> --out <dir> [--jobs 4] [--only XAUUSD]
(about 1 minute per market/session on one core; finished jobs are skipped, so it can be re-run to add markets).
Report: desk/everything_report.py.
"""
import argparse
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from desk.library import S, make_bars, simulate
from desk.tune import HOLD, noise_sig, sweep_sig

SESS = {"US": ("America/New_York", 570), "EU": ("Europe/Berlin", 540), "LDN": ("Europe/London", 480),
        "TKY": ("Asia/Tokyo", 540), "SYD": ("Australia/Sydney", 600), "HK": ("Asia/Hong_Kong", 570)}
FX_SESS = ("US", "LDN")
# name: (file, round-trip cost % of price, sessions)
M = {
    "US Tech 100": ("NSXUSD", .008, ("US",)), "US 500": ("SPXUSD", .012, ("US",)),
    "Germany 40": ("GRXEUR", .015, ("US", "EU")), "UK 100": ("UKXGBP", .02, ("US", "LDN")),
    "France 40": ("FRXEUR", .025, ("US", "EU")), "Europe 50": ("ETXEUR", .03, ("US", "EU")),
    "Japan 225": ("JPXJPY", .03, ("US", "TKY")), "Australia 200": ("AUXAUD", .025, ("US", "SYD")),
    "Hong Kong 50": ("HKXHKD", .04, ("US", "HK")), "US Dollar Index": ("UDXUSD", .02, ("US", "LDN")),
    "Gold": ("XAUUSD", .02, ("US", "LDN", "TKY")), "Silver": ("XAGUSD", .08, ("US", "LDN", "TKY")),
    "Brent oil": ("BCOUSD", .05, ("US", "LDN")), "WTI oil": ("WTIUSD", .05, ("US", "LDN")),
    "Bitcoin": ("BTCUSDT", .15, ("US", "LDN")), "Ethereum": ("ETHUSDT", .2, ("US", "LDN")),
    "Solana": ("SOLUSDT", .4, ("US",)), "XRP": ("XRPUSDT", .4, ("US",)), "Litecoin": ("LTCUSDT", .4, ("US",)),
    "Cardano": ("ADAUSDT", .4, ("US",)), "Dogecoin": ("DOGEUSDT", .4, ("US",)), "BNB": ("BNBUSDT", .4, ("US",)),
}
for p in "EURUSD GBPUSD USDJPY USDCHF USDCAD AUDUSD NZDUSD".split():
    M[p] = (p, .01, FX_SESS + (("TKY",) if p in ("USDJPY", "AUDUSD", "NZDUSD") else ()))
for p in ("EURGBP EURJPY GBPJPY EURCHF AUDJPY EURAUD GBPCHF CADJPY CHFJPY AUDNZD EURCAD GBPAUD AUDCAD NZDJPY GBPCAD "
          "GBPNZD EURNZD AUDCHF CADCHF NZDCAD NZDCHF").split():
    M[p] = (p, .02, FX_SESS + (("TKY",) if p in ("EURJPY", "GBPJPY", "AUDJPY") else ()))
for p in "USDMXN USDZAR USDNOK USDSEK EURNOK EURSEK USDPLN EURPLN USDTRY USDHUF USDCZK USDSGD".split():
    M[p] = (p, .06, FX_SESS)


def reclock(m1, sess):
    """Move the session's open to 9:30 on a plain (UTC-labelled) clock."""
    tz, open_min = SESS[sess]
    ix = m1.index.tz_convert(tz).tz_localize(None) + pd.Timedelta(minutes=570 - open_min)
    m = m1.set_axis(ix.tz_localize("UTC"))
    return m[~m.index.duplicated()]


def s_tiktok_pct(b):
    """The TikTok rule with its 10-point target as 0.06% of price, so it means the same on every market."""
    pre = b[(b.tod >= 510) & (b.end <= 540)].groupby("tdate")
    hi, lo = b.tdate.map(pre.high.max()), b.tdate.map(pre.low.min())
    return dict(le=b.close > hi, se=b.close < lo, sl_l=lo, sl_s=hi, tp_l=b.close * 1.0006, tp_s=b.close * 0.9994,
                max_day=1, start=540, end_entry=720)


def strategies():
    out = {}
    for (name, tf), (src, fn) in S.items():
        out[(name, tf)] = s_tiktok_pct if name.startswith("TikTok") else fn
    out[("noise band live (0.3% stop)", 15)] = lambda b: noise_sig(b, stop=0.003)
    out[("noise band long+short (0.3% stop)", 15)] = lambda b: noise_sig(b, stop=0.003, shorts=True)
    out[("overnight sweep live (2.5R)", 15)] = lambda b: sweep_sig(b, r=2.5)
    return out


def day_stats(pct, day, cal, per_year):
    """pct per trade, day per trade, cal = every session day (no-trade days count as 0)."""
    d = pd.to_datetime(pd.Series(day).values)
    pct = pd.Series(np.asarray(pct, float))
    daily = pct.groupby(d).sum().reindex(cal, fill_value=0.0)
    r = dict(trades_m=round(len(pct) / max(len(cal) / per_year * 12, 1e-9), 1),
             win=round(100 * (pct > 0).mean(), 1) if len(pct) else np.nan,
             avg_win=round(pct[pct > 0].mean(), 3) if (pct > 0).any() else 0,
             avg_loss=round(pct[pct <= 0].mean(), 3) if (pct <= 0).any() else 0)
    for k, msk, tm in (("dev", cal < HOLD, d < HOLD), ("hold", cal >= HOLD, d >= HOLD)):
        x, t = daily[msk], pct[tm]
        r[f"{k}_n"] = int(len(t))
        gl = -t[t <= 0].sum()
        r[f"{k}_pf"] = round(t[t > 0].sum() / gl, 2) if gl > 0 else (np.inf if len(t) else np.nan)
        r[f"{k}_ret_m"] = round(x.sum() / max(len(x) / per_year * 12, 1e-9), 3)  # % of position per month
        r[f"{k}_sharpe"] = round(x.mean() / x.std() * np.sqrt(per_year), 2) if x.std() > 0 else 0.0
    eq = daily.cumsum()
    r["max_dd"] = round((eq - eq.cummax()).min(), 2)
    yr = daily.groupby(daily.index.year).sum()
    r["pos_years"] = f"{int((yr > 0).sum())}/{len(yr)}"
    return r, daily


def tod_grid(m, cost, cal, per_year):
    """Hold from hour-boundary A to B (same futures day, 18:00 start), every day, long or short."""
    px = m.close.resample("60min", label="right", closed="left").last().ffill()  # price held at each hour
    df = pd.DataFrame({"px": px})
    df["tdate"] = (df.index.tz_localize(None) + pd.Timedelta(hours=6)).normalize()  # 18:00 opens the next day
    df["slot"] = (df.index.hour - 18) % 24  # 0 = 18:00 ... 23 = 17:00
    P = df.pivot_table(index="tdate", columns="slot", values="px").reindex(cal)
    rows, keep = [], {}
    for a in range(24):
        for z in range(a + 1, 24):
            ret = (P[z] / P[a] - 1) * 100
            ret = ret.dropna()
            for side, sgn in (("long", 1), ("short", -1)):
                pct = sgn * ret - cost
                st, daily = day_stats(pct.values, pct.index, cal, per_year)
                name = f"hold {(a + 18) % 24:02d}:00-{(z + 18) % 24:02d}:00"
                rows.append(dict(strategy=name, tf="1h", side=side, **st))
                if st["dev_sharpe"] > 1.0:
                    keep[(name, "1h", side)] = daily
    return rows, keep


def run_job(job):
    data, out, market, sess = job
    f, cost, _ = M[market]
    tag = f"{market.replace(' ', '_')}__{sess}"
    if os.path.exists(f"{out}/rows_{tag}.csv"):
        return tag, "cached"
    t0 = time.time()
    m1 = reclock(pd.read_pickle(f"{data}/{f}.pkl"), sess)
    m1 = m1[m1.index >= "2019-01-01"]
    rows, keep = [], {}
    strats = strategies()
    for tf in sorted({k[1] for k in strats}):
        b = make_bars(m1, tf)
        cal = pd.DatetimeIndex(sorted(b.tdate[b.rth].unique()))
        per_year = len(cal) / ((cal[-1] - cal[0]).days / 365.25)
        for (name, t), fn in strats.items():
            if t != tf:
                continue
            try:
                tr = simulate(b, fn(b), cost=0.0)
            except Exception as e:
                rows.append(dict(strategy=name, tf=f"{tf}m", side="error", note=repr(e)[:100]))
                continue
            tr["pct"] = tr.side * (tr.exit - tr.entry) / tr.entry * 100 - cost
            sides = [("both", tr)]
            if (tr.side == 1).any() and (tr.side == -1).any():
                sides += [("long", tr[tr.side == 1]), ("short", tr[tr.side == -1])]
            for side, x in sides:
                if len(x) < 5:
                    continue
                st, daily = day_stats(x.pct.values, x.day.values, cal, per_year)
                rows.append(dict(strategy=name, tf=f"{tf}m", side=side, **st))
                if st["dev_pf"] > 1.1 and st["dev_n"] >= 30:
                    keep[(name, f"{tf}m", side)] = daily
        if tf == 60 and sess == "US":  # the grid covers all 24 hours, so once per market is enough
            g_rows, g_keep = tod_grid(m1, cost, cal, per_year)
            rows += g_rows
            keep.update(g_keep)
    res = pd.DataFrame(rows)
    res.insert(0, "session", sess)
    res.insert(0, "market", market)
    res["cost"] = cost
    pd.to_pickle(keep, f"{out}/daily_{tag}.pkl")
    res.to_csv(f"{out}/rows_{tag}.csv", index=False)
    return tag, f"{len(res)} rows, {time.time() - t0:.0f}s"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.data, a.out, mk, s) for mk, (f, c, ss) in M.items() for s in ss
            if os.path.exists(f"{a.data}/{f}.pkl") and (not a.only or a.only in (mk, f))]
    print(len(jobs), "jobs", flush=True)
    with ProcessPoolExecutor(a.jobs) as ex:
        for tag, msg in ex.map(run_job, jobs):
            print(tag, msg, flush=True)
    res = pd.concat([pd.read_csv(f) for f in glob.glob(f"{a.out}/rows_*.csv")])
    res.to_csv(f"{a.out}/all_rows.csv", index=False)
