"""Monday dip in the bot (desk/monday_dip.py): same trades as the backtest engine, IBS skip, messages (no network)."""
import datetime as dt

import numpy as np
import pandas as pd

from desk import monday_dip as md
from desk import round21 as r21

NY = "America/New_York"


def futures(weeks=10, seed=1):
    """15-minute futures candles: Sunday 18:00 to Friday 17:00 New York, daily pause 17:00-18:00, random walk."""
    ix = pd.date_range("2026-08-02 18:00", periods=weeks * 7 * 96, freq="15min", tz=NY)
    wd, hr = ix.dayofweek, ix.hour
    ok = ~((wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 18)) | (hr == 17))
    ix = ix[ok]
    rng = np.random.default_rng(seed)
    c = 20000 * np.exp(np.cumsum(rng.normal(0, 0.0015, len(ix))))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.0008, len(ix))) * c
    return pd.DataFrame(dict(open=o, high=np.maximum(o, c) + w, low=np.minimum(o, c) - w, close=c), index=ix)


def test_same_trades_as_the_backtest():
    n = 0
    for seed in range(1, 6):
        f = futures(seed=seed)
        end = f.index[-1] + pd.Timedelta(days=1)
        mine = [r for r in md.trades(md.hourly(f, end)) if r["why"] not in ("open",)]
        h = r21.hourly(f, NY)
        ref = r21.simulate(h, dict(r21.TAUGHT, rules="1", r1=50), "US Tech 100")
        assert len(mine) == len(ref) > 0
        for a, b in zip(mine, ref.itertuples()):
            assert a["t_in"] == b.t_in and a["t_out"] == b.t_out and a["why"] == b.why
            pct = (a["exit"] - a["entry"]) / a["entry"] * 100 - md.COST - md.NIGHT_FEE * md.nights_held(a)
            assert abs(pct - b.pct) < 1e-9
        n += len(mine)
    assert n > 20


def test_only_closed_candles_count():
    f = futures()
    now = pd.Timestamp("2026-08-31 10:05", tz=NY)          # Monday: the 9:00 candle ended 5 minutes ago
    h = md.hourly(f, now)
    assert h.index[-1] == pd.Timestamp("2026-08-31 08:00", tz=NY)
    assert md.hourly(f, now + pd.Timedelta(minutes=6)).index[-1] == pd.Timestamp("2026-08-31 09:00", tz=NY)


def test_ibs_skip_does_not_block_the_next_hour():
    f = futures(seed=3)
    h = md.hourly(f, f.index[-1] + pd.Timedelta(days=1))
    first = md.trades(h)[0]
    ibs = [(first["t_in"] - pd.Timedelta(hours=20), first["t_in"] + pd.Timedelta(minutes=30))]
    rows = md.trades(h, ibs)
    assert rows[0]["why"] == "skip_ibs" and rows[0]["t_in"] == first["t_in"]
    assert rows[1]["t_in"] == first["t_in"] + pd.Timedelta("1h") or rows[1]["t_in"] > ibs[0][1]


def test_ibs_intervals_from_the_bot_sessions():
    from tests.test_daybot import ibs_day
    days = [dt.date(2026, 10, d) for d in (5, 6, 7, 8)]
    df = pd.concat([ibs_day(days[0], 100.5), ibs_day(days[1], 99.2), ibs_day(days[2], 99.1), ibs_day(days[3], 100.6)])
    iv = md.ibs_open(df)
    assert iv == [(pd.Timestamp("2026-10-06 15:45", tz=NY), pd.Timestamp("2026-10-08 15:45", tz=NY))]
    assert md.held(iv, pd.Timestamp("2026-10-07 03:00", tz=NY)) and not md.held(iv, iv[0][1])
    assert md.ibs_open(df[df.index.date < days[3]]) == [(iv[0][0], None)]   # still open


def test_messages_and_times():
    t_in = pd.Timestamp("2026-11-02 10:00", tz=NY)              # Monday 16:00 Amsterdam
    row = dict(t_in=t_in, entry=25000.0, stop=24800.0, target=25400.0, t_out=t_in + pd.Timedelta(hours=24),
               exit=25100.0, why="time")
    msgs = md.messages([row], 1000, md.week_start(t_in))
    (k1, t1, b1, _, p1), (k2, t2, b2, _, p2) = msgs
    assert k1 == "20261102-md_entry-1000" and t1 == "MONDAY DIP BUY (paper) 16:00" and p1 == 5
    assert "Stop loss: 24800" in b1 and "Take profit: 25400" in b1 and "Tue 16:00" in b1 and "Until 16:30" in b1
    assert t2 == "MONDAY DIP SELL 16:00" and "EUR +4.00" in b2
    night = dict(row, t_in=pd.Timestamp("2026-11-02 20:00", tz=NY), why="open")   # 02:00 Amsterdam: quiet push
    assert [m[4] for m in md.messages([night], 1000, md.week_start(t_in))] == [2]
    assert md.messages([row], 1000, t_in + pd.Timedelta(hours=1))[0][1] == "MONDAY DIP SELL 16:00"   # entry already sent
    assert md.due_by(pd.Timestamp("2026-11-02 18:00", tz=NY)) == pd.Timestamp("2026-11-03 19:00", tz=NY)


def test_window():
    ny = lambda *a: pd.Timestamp(dt.datetime(*a), tz=NY)
    assert not md.window(ny(2026, 11, 2, 0, 30)) and md.window(ny(2026, 11, 2, 1, 16))
    assert md.window(ny(2026, 11, 3, 23, 16)) and md.window(ny(2026, 11, 4, 1, 16))
    assert not md.window(ny(2026, 11, 4, 2, 16)) and not md.window(ny(2026, 11, 6, 12, 0))


def test_late_buy_is_sent_as_missed():
    t_in = pd.Timestamp("2026-11-02 02:00", tz=NY)                # 08:00 Amsterdam, first seen at 14:01
    row = dict(t_in=t_in, entry=28771.0, stop=28413.0, target=29488.0, t_out=None, exit=28771.0, why="open")
    (k, title, body, _, prio), = md.messages([row], 1000, md.week_start(t_in), now=pd.Timestamp("2026-11-02 08:01", tz=NY))
    assert k == "20261102-md_entry-0200" and "missed" in title and "skip" in title.lower() and prio == 2
    (k, title, _, _, prio), = md.messages([row], 1000, md.week_start(t_in), now=t_in + pd.Timedelta(minutes=20))
    assert title.startswith("MONDAY DIP BUY")
