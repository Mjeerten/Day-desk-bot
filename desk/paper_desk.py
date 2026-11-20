"""Paper log of the bot's own noise band, overnight sweep and IBS trades (the live three), like the paper logs of the
other strategies: journal/paper_desk.csv, one row per trade, rebuilt from Yahoo's last ~60 sessions after each US close
(paper.yml; older rows kept). Same rules and code as the alerts (desk/bot.py day_events / ibs_days), so the weekly
broker check (desk/weekly_check.py) can compare each demo broker's trades with what the bot itself did.
Prices are ^NDX points; pct after 2 points of cost; eur = at EUR 1,000 a trade. The overnight high / low uses the
15-minute basis of that day (the live bot uses the 1-minute one, which can differ by a few points).

Run: python -m desk.paper_desk [--csv journal/paper_desk.csv]
"""
import argparse
import os

import numpy as np
import pandas as pd

COST = 2.0


def _pct(side, entry, exit_):
    return side * (exit_ - entry) / entry * 100 - COST / entry * 100


def day_trades(ndx15, nq15, day):
    """Closed / open noise band + sweep trades of one session from the bot's own event replay."""
    from desk import bot
    gap = bot.basis([(nq15, ndx15)], day)
    fh, fl = bot.overnight(nq15, day)
    line, o, _ = bot.noise_line(ndx15, day)
    if not np.isfinite(gap):
        return []
    ev, _ = bot.day_events(ndx15, fh - gap, fl - gap, line, o, day)
    rows, n_open = [], None
    for e in ev:
        k = e["kind"]
        if k == "noise_entry":
            n_open = e
        elif k in ("noise_stop", "noise_exit", "noise_close") and n_open is not None:
            rows.append(dict(strat="N", side=1, t_in=n_open["ts"], entry=n_open["px"], stop=n_open["px"] * (1 - bot.NSTOP),
                             target=None, t_out=e["ts"], exit=e["px"], why=k.split("_")[1]))
            n_open = None
        elif k == "sweep_entry":
            sw = e
        elif k in ("sweep_stop", "sweep_target", "sweep_close"):
            t = e["trade"]
            rows.append(dict(strat="S", side=t["side"], t_in=sw["ts"], entry=t["entry"], stop=t["stop"], target=t["target"],
                             t_out=e["ts"], exit=e["px"], why=k.split("_")[1]))
    return rows


def ibs_trades(ndx15):
    """IBS buys at 15:45 when IBS < 0.2, sold at the first later 15:45 with IBS >= 0.2 (bot.ibs_days)."""
    from desk import bot
    d = bot.ibs_days(ndx15)
    rows, held = [], None
    for day, r in d.iterrows():
        weak = r.ibs < bot.IBS_MAX
        if held is None and weak:
            held = (day, r)
        elif held is not None and not weak:
            rows.append(dict(strat="I", side=1, t_in=held[1]["last"], entry=float(held[1].p), stop=None, target=None,
                             t_out=r["last"], exit=float(r.p), why="ibs", day=held[0]))
            held = None
    if held is not None:
        rows.append(dict(strat="I", side=1, t_in=held[1]["last"], entry=float(held[1].p), stop=None, target=None,
                         t_out=None, exit=np.nan, why="open", day=held[0]))
    return rows


def table(ndx15, nq15):
    days = sorted(set(bot_rth_days(ndx15)))[25:]       # the noise band needs 14 full sessions before it
    rows = [dict(r, day=day) for day in days for r in day_trades(ndx15, nq15, day)] + ibs_trades(ndx15)
    out = []
    for r in rows:
        done = r["why"] != "open"
        pct = _pct(r["side"], r["entry"], r["exit"]) if done else np.nan
        out.append(dict(date=f"{r['day']}", strat=r["strat"], side="buy" if r["side"] == 1 else "sell",
                        entry_ny=f"{pd.Timestamp(r['t_in']) + pd.Timedelta(minutes=15):%H:%M}", entry=round(r["entry"], 1),
                        stop="" if r["stop"] is None else round(r["stop"], 1),
                        target="" if r["target"] is None else round(r["target"], 1),
                        exit=round(r["exit"], 1) if done else "", why=r["why"],
                        exit_ny=f"{pd.Timestamp(r['t_out']) + pd.Timedelta(minutes=15):%m-%d %H:%M}" if done else "",
                        pts=round(r["side"] * (r["exit"] - r["entry"]), 1) if done else "",
                        pct=round(pct, 3) if done else "", eur=round(pct * 10, 2) if done else ""))
    t = pd.DataFrame(out)
    return t.sort_values(["date", "entry_ny", "strat"]).reset_index(drop=True) if len(t) else t


def bot_rth_days(ndx15):
    from desk import bot
    return bot.rth(ndx15).index.date


def main():
    from desk.bot import complete, yahoo
    from desk.plan import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "paper_desk.csv"))
    a = ap.parse_args()
    now = pd.Timestamp.now(tz="America/New_York")
    ndx15 = complete(yahoo("^NDX", "15m", "60d"), 15, now)
    nq15 = complete(yahoo("NQ=F", "15m", "60d", prepost=True), 15, now)
    t = table(ndx15, nq15)
    if os.path.exists(a.csv) and os.path.getsize(a.csv) and len(t):
        old = pd.read_csv(a.csv, dtype=str)
        t = pd.concat([old[old.date < t.date.min()], t.astype(str)], ignore_index=True)
    t.to_csv(a.csv, index=False)
    print(t.tail(40).to_string(index=False))
    done = t[t.why != "open"] if len(t) else t
    print(f"\n{len(done)} closed trades, EUR {pd.to_numeric(done.eur).sum() if len(done) else 0:+.2f} at EUR 1,000 per trade")


if __name__ == "__main__":
    main()
