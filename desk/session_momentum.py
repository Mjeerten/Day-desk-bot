"""Session Momentum (BACKTEST.md rounds 23 / 25), paper strategy in the day bot: alerts for the Trading 212 practice account.

Rules (round 25 pick, `desk/round25.py` BOT): at the close of the 9:30-9:35 New York candle compare the price with the
8-period EMA of 5-minute closes (futures, nearly 24 h). Above -> BUY, below -> SELL (short). Stop 1.2% of the entry;
the stop trails 1.2% behind the best 5-minute close; out at 16:00 New York at the latest. One trade a day.
Backtest 2019-2023 / 2024+: +10.1 / +18.7 %/yr at 1x, PF 1.12 / 1.27, 50% winners, 7 of 8 years up. Late entry (round 25b):
10 min late +9.5 / +19.1, 15 min +9.8 / +16.1, 30 min +7.0 / +14.1, so the 9:46 run is fine and late entries are allowed
until 9:55.

Prices: Yahoo's NQ=F futures can lag ~10 minutes, so the EMA uses the futures candles up to 9:25-9:30 and the 9:30-9:35
close comes from the real-time Nasdaq-100 index (^NDX) plus today's futures-minus-index gap (bot.basis); one EMA step
with that close gives the 9:35 EMA. Trading 212's US Tech 100 follows the futures, so every price here is on the futures
scale (no offset). The trailing stop is followed on the futures candles (late by Yahoo's lag; the order on Trading 212
does the real work).

Pure functions so the bot, the replay and the paper log share them:
  decision(nq5, ndx5, gap, day, now) -> dict(side, entry, ema) or None
  manage(nq5, dec, day, now) -> dict(stop, best, exit, why, t_out) (why = 'open' while running)
  messages(dec, st, day, stake, since) -> [(key, title, body, tags, priority)]
"""
import numpy as np
import pandas as pd

from desk.ticket import ticket

SPAN, AT, STOP, TRAIL, CLOSE = 8, 570, 0.012, 1.0, 960
LAG = pd.Timedelta(minutes=10)    # futures candles count once they ended this long ago
STEP = 0.25                       # send a 'move your stop' message each time the stop moved another 0.25 x risk
NY, AMS = "America/New_York", "Europe/Amsterdam"


def _tod(ix):
    return ix.hour * 60 + ix.minute


def decision(nq5, ndx5, gap, day, now):
    """Today's signal, known from 9:35 New York. None before that, on a missing candle or without a gap."""
    if not np.isfinite(gap):
        return None
    t0 = pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(minutes=AT)          # 9:30
    x = ndx5[(ndx5.index == t0) & (ndx5.index + pd.Timedelta(minutes=5) <= now)]
    if not len(x):
        return None
    f = nq5[(nq5.index < t0) & (nq5.index + pd.Timedelta(minutes=5) <= max(now - LAG, t0))]
    if len(f) < 200 or f.index[-1] < t0 - pd.Timedelta(minutes=30):            # need a warm EMA up to ~9:25
        return None
    ema_prev = float(f.close.ewm(span=SPAN, adjust=False).mean().iloc[-1])
    close = float(x.close.iloc[0]) + gap
    ema = ema_prev + 2 / (SPAN + 1) * (close - ema_prev)
    return dict(side=1 if close > ema else -1, entry=close, ema=ema, t_in=t0 + pd.Timedelta(minutes=5))


def manage(nq5, dec, day, now):
    """Follow the trade on the futures candles after 9:35 (as round25.run: stop first, then trail on the close)."""
    s, e = dec["side"], dec["entry"]
    risk = e * STOP
    st, best = e - s * risk, e
    t1 = dec["t_in"]
    eod = pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(minutes=CLOSE)
    f = nq5[(nq5.index >= t1) & (nq5.index < eod) & (nq5.index + pd.Timedelta(minutes=5) <= now - LAG)]
    for ts, b in f.iterrows():
        if (b.low <= st) if s == 1 else (b.high >= st):
            # fill at the stop: the futures trade without pause in the session, so a candle that opens beyond the stop
            # means a hole in Yahoo's data (seen 30 Oct 2026), not a gap; the order on Trading 212 fills near the stop
            return dict(stop=st, best=best, exit=st, why="stop", t_out=ts + pd.Timedelta(minutes=5))
        best = max(best, b.close) if s == 1 else min(best, b.close)
        st = max(st, best - TRAIL * risk) if s == 1 else min(st, best + TRAIL * risk)
    if now >= eod + LAG and len(f):
        return dict(stop=st, best=best, exit=float(f.close.iloc[-1]), why="close", t_out=eod)
    return dict(stop=st, best=best, exit=float(f.close.iloc[-1]) if len(f) else e, why="open", t_out=None)


def fmt(x):
    return f"{x:,.0f}".replace(",", " ")


def messages(dec, st, day, stake, late_min=20):
    """(key, title, body, tags, priority) for today's trade."""
    if dec is None:
        return []
    s, e = dec["side"], dec["entry"]
    risk = e * STOP
    k = f"{day:%Y%m%d}-sm"
    t_in = dec["t_in"].tz_convert(AMS)
    word = "BUY" if s == 1 else "SELL (short)"
    bell = (pd.Timestamp(day).tz_localize(NY) + pd.Timedelta(minutes=CLOSE)).tz_convert(AMS)
    late = (dec["t_in"] + pd.Timedelta(minutes=late_min)).tz_convert(AMS)
    out = [(f"{k}_entry", f"SESSION MOMENTUM {'BUY' if s == 1 else 'SHORT'} (paper) {t_in:%H:%M}",
            ticket(s, stake, e - s * risk) +
            f"Practice account: {word} US Tech 100 now at market (about {fmt(e)}).\n"
            f"Stop {risk:.0f} pts {'below' if s == 1 else 'above'}. No take profit.\n"
            f"The stop follows the price: I send 'move your stop' messages as it moves {risk:.0f} pts behind the best "
            f"price. Close it by {bell - pd.Timedelta(minutes=5):%H:%M} at the latest (it is part of 'close everything').\n"
            f"Missed it? Until {late:%H:%M} you can still enter at market with the same stop price. After that, skip it.\n"
            f"Took it? Tap 'I took it' below: only then do the 'move your stop' messages come.",
            ["rotating_light"], 5)]
    moved = (st["stop"] - (e - s * risk)) * s
    n = int(moved // (STEP * risk))
    if n >= 1 and st["why"] == "open":
        lock = (st["stop"] - e) * s
        side_w = "BUY" if s == 1 else "SHORT (sell)"
        out.append((f"{k}_trail{n}", f"Session Momentum {'BUY' if s == 1 else 'SHORT'} ({t_in:%H:%M}): move its stop",
                    f"New stop: {st['stop']:.0f}\n\n"
                    f"Only for the Session Momentum {side_w} from {t_in:%H:%M} (entry about {fmt(e)}), not for any other "
                    f"trade.\nMove the stop of that trade to {fmt(st['stop'])} ({'locks in' if lock >= 0 else 'still a loss if hit:'} "
                    f"{abs(lock):.0f} pts{' profit' if lock >= 0 else ''}; best price so far {fmt(st['best'])}).",
                    ["arrow_up" if s == 1 else "arrow_down"], 4))
    if st["why"] in ("stop", "close"):
        res = s * (st["exit"] - e) / e * stake
        if st["why"] == "stop":
            out.append((f"{k}_stop", f"Session momentum stopped out {st['t_out'].tz_convert(AMS):%H:%M}",
                        f"The Session Momentum trade should now be closed by its stop at about {fmt(st['exit'])} "
                        f"(about EUR {res:+.2f}). Check that it is closed.", ["white_check_mark" if res > 0 else "x"], 3))
        else:
            out.append((f"{k}_close", "Session momentum: day over",
                        f"The market has closed; your Session Momentum trade should be closed (the 'close everything' "
                        f"message). Closing price about {fmt(st['exit'])}, about EUR {res:+.2f}.", ["bell"], 3))
    return out


# ---------- paper log ----------

def main():
    """journal/paper_momentum.csv: every Session Momentum trade of the last ~55 sessions (merged, older rows kept)."""
    import argparse
    import os
    from desk.bot import basis, yahoo
    from desk.plan import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "paper_momentum.csv"))
    a = ap.parse_args()
    nq5, ndx5 = yahoo("NQ=F", "5m", "60d", prepost=True), yahoo("^NDX", "5m", "60d")
    now = pd.Timestamp.now(tz=NY)
    rows = []
    for day in sorted(set(ndx5.index.date))[5:]:
        gap = basis([(nq5, ndx5)], day)
        dec = decision(nq5, ndx5, gap, day, now)
        if dec is None:
            continue
        st = manage(nq5, dec, day, now)
        pct = dec["side"] * (st["exit"] - dec["entry"]) / dec["entry"] * 100 - 2.0 / dec["entry"] * 100
        rows.append(dict(date=f"{day}", side="buy" if dec["side"] == 1 else "sell", entry=round(dec["entry"], 1),
                         ema=round(dec["ema"], 1), exit=round(st["exit"], 1), why=st["why"],
                         exit_ny=f"{st['t_out']:%H:%M}" if st["t_out"] is not None else "", pct=round(pct, 3),
                         eur=round(pct * 10, 2)))
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
