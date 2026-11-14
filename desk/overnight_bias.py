"""Overnight Bias ORB (BACKTEST.md rounds 27 / 27b, settings as taught = `round27.OB_TAUGHT`): a paper strategy like Session
Momentum / Monday dip: automatic orders on the demos (desk/autotrade.py), phone messages with an 'I took it' button (only
while PHONE_ALERTS is on), paper log journal/paper_obias.csv (paper.yml).

Rules: overnight range = futures high/low from 18:00 NY the evening before up to 9:30. Where 9:30 opens in it: top third ->
longs only, bottom third -> shorts only, middle -> no trade today. Opening range = 9:30-9:45 high/low. Entry on a 5-minute
close above its high (long) / below its low (short), only while ADX(14) on 15-minute candles (all hours) > 20. Stop = the
other side of the opening range, target 3R, out at 15:00 NY. Up to 4 trades a day, one at a time.
Backtest 2019-23 / 2024+: +4.6 / +8.1 %/yr at 1x, PF 1.16 / 1.30, 7 of 8 years up, 98% of 768 settings positive in both.

Prices (as desk/bot.py's noise band / sweep): everything on the Nasdaq-100 index scale (^NDX 5-minute candles, real time).
The overnight range comes from the NQ=F futures minus today's futures-minus-index gap (bot.basis); ADX only uses
differences, so the futures candles give it directly (counted once they ended LAG ago: Yahoo's futures lag ~10 min).
autotrade.py moves every level to IG's price.

trades(nq5, ndx5, gap, day, now) -> list of dict(side, entry, stop, target, t_in, exit, why, t_out); why = 'open' while it
runs, else 'stop' / 'target' / 'close' (15:00). messages(trades, day, stake, off) -> the phone messages (prices + off =
Trading 212's scale, as the noise band's). main() -> journal/paper_obias.csv.
Tests: tests/test_overnight_bias.py (same trades as round27.overnight_bias, messages).
"""
import numpy as np
import pandas as pd

from desk.round27 import OB_TAUGHT, adx
from desk.ticket import ticket

NY = "America/New_York"
AMS = "Europe/Amsterdam"
OPEN, ORL, OUT = 570, OB_TAUGHT["orl"], OB_TAUGHT["out"]
THIRD, RR, ADX_MIN, MAX_N = OB_TAUGHT["third"], OB_TAUGHT["rr"], OB_TAUGHT["adx_min"], OB_TAUGHT["max_n"]
LAG = pd.Timedelta(minutes=10)
FIVE = pd.Timedelta(minutes=5)


def _tod(ix):
    return ix.hour * 60 + ix.minute


def overnight_range(nq5, day, now):
    """Futures high/low 18:00 the evening before - 9:30 (trade-date convention), from candles known by now."""
    f = nq5[nq5.index + FIVE <= now - LAG]
    t = f.index.tz_localize(None) if f.index.tz is None else f.index.tz_convert(NY).tz_localize(None)
    sess = (t + pd.Timedelta(hours=6)).normalize()
    tod = t.hour * 60 + t.minute
    on = f[(sess == pd.Timestamp(day)) & ((tod >= 1080) | (tod < OPEN))]
    return (float(on.high.max()), float(on.low.min())) if len(on) else (np.nan, np.nan)


def adx15(nq5, ix, now):
    """ADX(14) of 15-minute futures candles (all hours), the last one ended by each 5-minute candle's close."""
    q = nq5.resample("15min", label="left", closed="left").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    q = q[q.index + pd.Timedelta(minutes=15) <= now - LAG]
    if len(q) < 30:
        return np.full(len(ix), np.nan)
    known = pd.Series(adx(q.high.to_numpy(), q.low.to_numpy(), q.close.to_numpy()), q.index + pd.Timedelta(minutes=15))
    return known.reindex(ix + FIVE, method="ffill").to_numpy()


def trades(nq5, ndx5, gap, day, now):
    if not np.isfinite(gap):
        return []
    now = pd.Timestamp(now)
    t0 = pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(minutes=OPEN)
    g = ndx5[(ndx5.index >= t0) & (ndx5.index < t0 + pd.Timedelta(minutes=390)) & (ndx5.index + FIVE <= now)]
    if len(g) < ORL // 5 + 1 or g.index[0] != t0:
        return []
    oh, ol = overnight_range(nq5, day, now)
    if not (np.isfinite(oh) and oh > ol):
        return []
    oh, ol = oh - gap, ol - gap
    o, h, l, c = (g[k].to_numpy() for k in ("open", "high", "low", "close"))
    tod, ax = _tod(g.index), adx15(nq5, g.index, now)
    p = (o[0] - ol) / (oh - ol)
    allow = 1 if p >= 1 - THIRD else -1 if p <= THIRD else 0
    orr = tod < OPEN + ORL
    if not allow or orr.sum() < ORL // 5:
        return []
    hi, lo = h[orr].max(), l[orr].min()
    before = np.nonzero(tod < OUT)[0]
    done_day = now >= t0 + pd.Timedelta(minutes=OUT - OPEN)
    last = int(before[-1]) if done_day else None          # the 14:55 candle: exit at its close
    end = last if last is not None else len(g) - 1        # candles we can look at
    out, i = [], int(orr.sum())
    while i < (last if last is not None else len(g)) and tod[i] < OUT and len(out) < MAX_N:
        s = 1 if c[i] > hi else -1 if c[i] < lo else 0
        if s != allow or not ax[i] > ADX_MIN:
            i += 1
            continue
        e, st = c[i], (lo if s == 1 else hi)
        risk = abs(e - st)
        if risk <= 0:
            i += 1
            continue
        tg = e + s * RR * risk
        tr = dict(side=s, entry=float(e), stop=float(st), target=float(tg), t_in=g.index[i] + FIVE,
                  exit=np.nan, why="open", t_out=None)
        out.append(tr)
        for j in range(i + 1, end + 1):
            if (l[j] <= st) if s == 1 else (h[j] >= st):
                tr.update(exit=float(min(st, o[j]) if s == 1 else max(st, o[j])), why="stop")
            elif (h[j] >= tg) if s == 1 else (l[j] <= tg):
                tr.update(exit=float(max(tg, o[j]) if s == 1 else min(tg, o[j])), why="target")
            elif j == last:
                tr.update(exit=float(c[j]), why="close")
            if tr["why"] != "open":
                tr["t_out"] = g.index[j] + FIVE
                i = j + 1
                break
        else:
            if last is not None and i >= last:
                tr.update(exit=float(c[last]), why="close", t_out=g.index[last] + FIVE)
            return out                                        # still open (or the day ended at the entry candle)
    return out


def fmt(x):
    return f"{x:,.0f}".replace(",", " ")


def messages(trades_, day, stake, off=0.0, late_min=15):
    """(key, title, body, tags, priority) for today's trades; key f'{day}-ob{n}_...' (bot.paper_id: entry messages get
    the 'I took it' button, the others only go out for a tapped trade)."""
    out = []
    for n, t in enumerate(trades_, 1):
        k = f"{day:%Y%m%d}-ob{n}"
        s, e, st, tg = t["side"], t["entry"] + off, t["stop"] + off, t["target"] + off
        t_in = t["t_in"].tz_convert(AMS)
        late = (t["t_in"] + pd.Timedelta(minutes=late_min)).tz_convert(AMS)
        out_at = (pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(minutes=OUT)).tz_convert(AMS)
        word = "BUY" if s == 1 else "SELL (short)"
        out.append((f"{k}_entry", f"OVERNIGHT BIAS {'BUY' if s == 1 else 'SHORT'} (paper) {t_in:%H:%M}",
                    ticket(s, stake, st, tg) +
                    f"Practice account: {word} US Tech 100 now at market (about {fmt(e)}).\n"
                    f"Stop {fmt(st)} (the other side of the 9:30-9:45 range), target {fmt(tg)} (3R). "
                    f"Close it at {out_at:%H:%M} if neither is hit.\n"
                    f"Missed it? Until {late:%H:%M} you can still enter at market with the same stop and target. After "
                    f"that, skip it.\nTook it? Tap 'I took it' below: only then do its stop / target / close messages come.",
                    ["rotating_light"], 5))
        if t["why"] == "open":
            continue
        res = s * (t["exit"] - t["entry"]) / t["entry"] * stake
        when = t["t_out"].tz_convert(AMS) if t["t_out"] is not None else out_at
        if t["why"] in ("stop", "target"):
            out.append((f"{k}_{t['why']}", f"Overnight bias {'stopped out' if t['why'] == 'stop' else 'target hit'} "
                        f"{when:%H:%M}",
                        f"The Overnight Bias {'buy' if s == 1 else 'short'} from {t_in:%H:%M} should now be closed by its "
                        f"{t['why']} at about {fmt(t['exit'] + off)} (about EUR {res:+.2f}). Check that it is closed.",
                        ["white_check_mark" if res > 0 else "x"], 3))
        else:
            out.append((f"{k}_close", f"Overnight bias: close it now ({when:%H:%M})",
                        f"Close the Overnight Bias {'buy' if s == 1 else 'short'} from {t_in:%H:%M} at market now "
                        f"(about {fmt(t['exit'] + off)}, about EUR {res:+.2f}): its time is up.", ["bell"], 4))
    return out


# ---------- paper log ----------

def main():
    """journal/paper_obias.csv: every Overnight Bias trade of the last ~55 sessions (merged, older rows kept)."""
    import argparse
    import os
    from desk.bot import basis, yahoo
    from desk.plan import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "paper_obias.csv"))
    a = ap.parse_args()
    nq5, ndx5 = yahoo("NQ=F", "5m", "60d", prepost=True), yahoo("^NDX", "5m", "60d")
    now = pd.Timestamp.now(tz=NY)
    rows = []
    for day in sorted(set(ndx5.index.date))[5:]:
        gap = basis([(nq5, ndx5)], day)
        for n, t in enumerate(trades(nq5, ndx5, gap, day, now), 1):
            pct = t["side"] * (t["exit"] - t["entry"]) / t["entry"] * 100 - 2.0 / t["entry"] * 100 \
                if t["why"] != "open" else np.nan
            rows.append(dict(date=f"{day}", n=n, side="buy" if t["side"] == 1 else "sell", entry_ny=f"{t['t_in']:%H:%M}",
                             entry=round(t["entry"], 1), stop=round(t["stop"], 1), target=round(t["target"], 1),
                             exit=round(t["exit"], 1) if t["why"] != "open" else "", why=t["why"],
                             exit_ny=f"{t['t_out']:%H:%M}" if t["t_out"] is not None else "",
                             pct=round(pct, 3) if t["why"] != "open" else "",
                             eur=round(pct * 10, 2) if t["why"] != "open" else ""))
    t = pd.DataFrame(rows)
    if os.path.exists(a.csv) and os.path.getsize(a.csv) and len(t):
        old = pd.read_csv(a.csv, dtype=str)
        t = pd.concat([old[old.date < t.date.min()], t.astype(str)], ignore_index=True)
    t.to_csv(a.csv, index=False)
    print(t.to_string(index=False))
    done = t[t.why != "open"] if len(t) else t
    print(f"\n{len(done)} closed trades, EUR {pd.to_numeric(done.eur).sum() if len(done) else 0:+.2f} at EUR 1,000 per trade")


if __name__ == "__main__":
    main()
