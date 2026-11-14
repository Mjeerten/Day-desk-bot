"""Round 21c: the Monday dip (round 21b, rule 1) next to IBS and the live bot, and can it be improved?

Monday dip as taught: Mondays (market clock), 1-hour RSI(14) < 40 and close below the 40-hour average -> buy at the
candle close, stop 2.5 ATR(14, 1h), target 2R, out after 24 h, one position at a time; spread + 0.02% per night.
IBS as live (desk/ibs.py): 15:45 NY, IBS < 0.2 -> buy, sell 15:45 next session; Nasdaq 2 points + same night fee.
Live bot intraday (noise band + overnight sweep) as round 13 `current_desk`.

A. Overlap on the Nasdaq: share of Monday-dip trades held while an IBS trade is open, result of those vs the rest,
   correlation of daily results, and the desk with / without the Monday dip (%/yr, drawdown; each strategy 1x).
B. Improvements, picked on 2019-2023 only, checked on 2024+ (per market and pooled over the four):
   other weekdays (is Monday special?), RSI 40 / 45 / 50, hold 24 / 48 h, signal hours (all / before the cash open /
   cash / outside cash), only above the 200-day average, target 2R or none.
C. A four-market basket (UK 100, Germany 40, US 500, US Tech 100) of the best common setting vs as taught.

Run: python -m desk.round21c --data <dir with UKXGBP/GRXEUR/SPXUSD/NSXUSD.pkl> --out <dir>
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

from desk import round21 as r21
from desk.everything import M
from desk.ibs import ibs_trades
from desk.ifvg import stats
from desk.round13 import current_desk

AS_TAUGHT = dict(r21.TAUGHT, rules="1")


def daily(tr, col="t_out"):
    d = pd.to_datetime(tr[col]).dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    return tr.pct.groupby(d.values).sum()


def eq_stats(s, name):
    s = s.sort_index()
    out = {"model": name}
    for k, m in (("dev", s.index < "2024-01-01"), ("hold", s.index >= "2024-01-01")):
        x = s[m]
        yrs = max((x.index.max() - x.index.min()).days / 365.25, 0.5)
        eq = x.cumsum()
        out[f"{k}_yr"] = round(x.sum() / yrs, 2)
        out[f"{k}_dd"] = round((eq - eq.cummax()).min(), 2)
        out[f"{k}_sharpe"] = round(x.mean() / x.std() * np.sqrt(252), 2) if x.std() > 0 else np.nan
    return out


def part_a(data, md_p=AS_TAUGHT):
    nq = f"{data}/NSXUSD.pkl"
    m1 = pd.read_pickle(nq)
    irx = pd.Series(4.8, index=pd.date_range("2018-01-01", "2027-01-01"))   # (4.8 + 2.5) / 365 = 0.02% a night
    ibs_pct, _ = ibs_trades(m1, irx, 2.0 / 15000 * 100)
    ibs = pd.DataFrame({"pct": ibs_pct.values, "day": ibs_pct.index})
    ibs["t_in"] = (ibs.day + pd.Timedelta(hours=15, minutes=45)).dt.tz_localize("America/New_York")
    ibs["t_out"] = ibs.t_in + pd.Timedelta(days=1)   # next session 15:45 (weekends: Monday)
    ibs.loc[ibs.day.dt.dayofweek == 4, "t_out"] += pd.Timedelta(days=2)
    h = r21.hourly(m1, "America/New_York")
    h = h[h.index >= pd.Timestamp("2019-01-01", tz="America/New_York")]
    md = r21.simulate(h, md_p, "US Tech 100")
    ov = np.array([((ibs.t_in < r.t_out) & (ibs.t_out > r.t_in)).any() for r in md.itertuples()])
    print(f"\nA. Nasdaq: {len(md)} Monday-dip trades, {len(ibs)} IBS trades. Monday-dip trades held while an IBS trade is"
          f" open: {ov.mean():.0%}")
    print(f"   overlapping: avg {md.pct[ov].mean():+.3f}% ({ov.sum()}), others: avg {md.pct[~ov].mean():+.3f}% ({(~ov).sum()})")
    dd = current_desk(nq)
    live = daily(dd["noise + sweep (live bot)"].assign(t_out=lambda x: pd.to_datetime(x.day).dt.tz_localize("America/New_York")))
    ib, mdd = daily(ibs), daily(md)
    cal = pd.concat([live, ib, mdd], axis=1).fillna(0)
    cal.columns = ["live", "ibs", "md"]
    print("   daily-result correlation:\n" + cal.corr().round(2).to_string())
    rows = [eq_stats(cal.live, "noise + sweep"), eq_stats(cal.ibs, "IBS"), eq_stats(cal.md, "Monday dip (Nasdaq)"),
            eq_stats(cal.live + cal.ibs, "live bot (noise + sweep + IBS)"),
            eq_stats(cal.live + cal.ibs + cal.md, "live bot + Monday dip"),
            eq_stats(cal.ibs + cal.md, "IBS + Monday dip")]
    md_no = md[~ov]
    rows.append(eq_stats(cal.live + cal.ibs + daily(md_no).reindex(cal.index).fillna(0),
                         "live bot + Monday dip, skipped while IBS is held"))
    return pd.DataFrame(rows)


def part_b(data):
    grid = [dict(AS_TAUGHT, dow=dw, r1=r, hold=hd, hours=hr, trend=tf, target=tg)
            for dw, r, hd, hr, tf, tg in itertools.product((0, 1, 2, 3, 4), (40, 45, 50), (24, 48),
                                                            ("all", "pre", "rth", "off"), ("", "up200"), (2.0, 0))]
    rows, trades = [], {}
    for mk, (tz, _, _) in r21.MARKETS.items():
        h = r21.hourly(pd.read_pickle(f"{data}/{M[mk][0]}.pkl"), tz)
        h = h[h.index >= pd.Timestamp("2019-01-01", tz=tz)]
        for k, p in enumerate(grid):
            tr = r21.simulate(h, p, mk)
            trades[(mk, k)] = tr
            rows.append(dict(market=mk, k=k, **{x: p[x] for x in ("dow", "r1", "hold", "hours", "trend", "target")},
                             **stats(tr)))
        print(mk, "done", flush=True)
    return pd.DataFrame(rows), grid, trades


IMPROVED = dict(AS_TAUGHT, r1=50)   # 21c pick: RSI < 50, rest as taught (best worst-market PF on 2019-23)


def main_improved(data, out):
    """21d: the improved Monday dip next to IBS and the live bot, Nasdaq only (as traded) and the basket."""
    for name, p in (("as taught", AS_TAUGHT), ("improved RSI<50", IMPROVED), ("RSI<50, 48 h, no target", dict(IMPROVED, hold=48, target=0))):
        a = part_a(data, p)
        print(f"\n== {name} ==\n" + a.to_string(index=False), flush=True)
        a.to_csv(f"{out}/round21d_{name.split()[0]}_{len(name)}.csv", index=False)
    for mk, (tz, _, _) in r21.MARKETS.items():
        h = r21.hourly(pd.read_pickle(f"{data}/{M[mk][0]}.pkl"), tz)
        h = h[h.index >= pd.Timestamp("2019-01-01", tz=tz)]
        tr = r21.simulate(h, IMPROVED, mk)
        print(mk, "improved:", {k: v for k, v in stats(tr).items() if k in ("n", "win", "dev_pf", "dev_yr", "dev_dd", "hold_pf", "hold_yr", "hold_dd", "years")})


def main_live(data, out):
    """21e: the Monday dip as the bot runs it: a signal is skipped only when an IBS trade is open at that moment
    (the bot cannot know an IBS buy that comes later during the trade); a skipped signal does not block the next hour."""
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
    close = h.index + pd.Timedelta("1h")
    cv = close.tz_convert("UTC").tz_localize(None).values
    tiu = pd.DatetimeIndex(ibs.t_in).tz_convert("UTC").tz_localize(None).values
    tou = pd.DatetimeIndex(ibs.t_out).tz_convert("UTC").tz_localize(None).values
    k = np.searchsorted(tiu, cv, side="right") - 1
    held = (k >= 0) & (cv < tou[np.clip(k, 0, None)])
    h2 = h.copy()
    h2.loc[held, "rsi"] = 100.0
    md = r21.simulate(h2, IMPROVED, "US Tech 100")
    dd = current_desk(nq)
    live = daily(dd["noise + sweep (live bot)"].assign(t_out=lambda x: pd.to_datetime(x.day).dt.tz_localize("America/New_York")))
    cal = pd.concat([live, daily(ibs), daily(md)], axis=1).fillna(0)
    cal.columns = ["live", "ibs", "md"]
    rows = [eq_stats(cal.live + cal.ibs, "live bot (noise + sweep + IBS)"),
            eq_stats(cal.md, "Monday dip as the bot runs it"),
            eq_stats(cal.live + cal.ibs + cal.md, "live bot + Monday dip as the bot runs it")]
    a = pd.DataFrame(rows)
    print(f"{len(md)} trades, {held.mean():.1%} of hours IBS held", stats(md))
    print(a.to_string(index=False), flush=True)
    a.to_csv(f"{out}/round21e_live.csv", index=False)
    md.to_csv(f"{out}/round21e_trades.csv", index=False)


def main(data, out):
    a = part_a(data)
    print("\n   desk on the Nasdaq (each strategy 1x, % of price a year, drawdown in %):\n" + a.to_string(index=False))
    a.to_csv(f"{out}/round21c_overlap.csv", index=False)

    g, grid, trades = part_b(data)
    g.to_csv(f"{out}/round21c_grid.csv", index=False)
    keys = ["dow", "r1", "hold", "hours", "trend", "target"]
    print("\nB1. Weekday (median over the other settings, per market): dev %/yr | 2024+ %/yr")
    print(g.groupby(["market", "dow"])[["dev_yr", "hold_yr", "dev_pf", "hold_pf"]].median().round(2).unstack(0).to_string())
    print("\nB2. Monday only, each choice (median over the rest, all markets):")
    mon = g[g.dow == 0]
    for k in ("r1", "hold", "hours", "trend", "target"):
        print(mon.groupby(k)[["dev_yr", "hold_yr", "dev_pf", "hold_pf"]].median().round(2).to_string(), "\n")
    pooled = g.groupby("k").agg(dev_yr=("dev_yr", "sum"), hold_yr=("hold_yr", "sum"),
                                dev_min=("dev_pf", "min"), hold_min=("hold_pf", "min"))
    best = pooled.sort_values("dev_yr", ascending=False)
    best = best[best.dev_min > 1].head(10)
    show = pd.concat([best, pd.DataFrame([grid[k] for k in best.index], index=best.index)[keys]], axis=1)
    print("B3. Best common settings on 2019-23 (all four markets profitable on dev; %/yr summed over the four):\n"
          + show.to_string())
    k0 = next(k for k, p in enumerate(grid) if all(p[x] == AS_TAUGHT.get(x, 0 if x == "dow" else "") for x in keys))
    kb = best.index[0]
    res = []
    for name, k in (("as taught", k0), ("best on dev", kb)):
        basket = pd.concat([daily(trades[(mk, k)]) for mk in r21.MARKETS], axis=1).fillna(0)
        for mk_i, mk in enumerate(r21.MARKETS):
            res.append(dict(eq_stats(basket.iloc[:, mk_i], f"{name}: {mk}"), **stats(trades[(mk, k)])))
        res.append(eq_stats(basket.sum(axis=1), f"{name}: basket of four (each 1x)"))
        res.append(eq_stats(basket.mean(axis=1), f"{name}: basket of four (1/4 each)"))
    res = pd.DataFrame(res)
    res.to_csv(f"{out}/round21c_basket.csv", index=False)
    print(f"\nC. as taught = {grid[k0]}\n   best on dev = {grid[kb]}\n" +
          res[["model", "n", "win", "dev_pf", "hold_pf", "dev_yr", "hold_yr", "dev_dd", "hold_dd", "dev_sharpe",
               "hold_sharpe", "years"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--improved", action="store_true", help="21d: improved setting vs IBS / live bot")
    ap.add_argument("--live", action="store_true", help="21e: the rule exactly as the bot runs it")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    (main_live if a.live else main_improved if a.improved else main)(a.data, a.out)
