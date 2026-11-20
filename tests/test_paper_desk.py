"""paper_desk: the live three's paper log makes the bot's own trades (no network)."""
import datetime as dt

import pandas as pd

from desk import bot, paper_desk
from tests.test_daybot import DAY, ibs_day, session


def test_sweep_trade_row(monkeypatch):
    c = [100.0] * 26
    c[1] = 99.5
    lows, highs = [99.9] * 26, [100.1] * 26
    lows[1], highs[4] = 98.0, 103.5
    nd = session(c, highs, lows)
    monkeypatch.setattr(bot, "basis", lambda pairs, day: 0.0)
    monkeypatch.setattr(bot, "overnight", lambda nq, day: (110.0, 99.0))
    monkeypatch.setattr(bot, "noise_line", lambda nd, day: (None, 100.0, 100.0))
    rows = paper_desk.day_trades(nd, nd, DAY)
    assert len(rows) == 1
    r = rows[0]
    assert (r["strat"], r["side"], r["entry"], r["stop"], r["target"], r["exit"], r["why"]) == \
        ("S", 1, 99.5, 98.0, 103.25, 103.25, "target")


def test_ibs_buy_and_sell():
    days = [dt.date(2026, 10, d) for d in (5, 6, 7, 8)]
    df = pd.concat([ibs_day(days[0], 100.5), ibs_day(days[1], 99.2), ibs_day(days[2], 99.1), ibs_day(days[3], 100.6)])
    rows = paper_desk.ibs_trades(df)
    assert len(rows) == 1 and rows[0]["day"] == days[1] and rows[0]["entry"] == 99.2 and rows[0]["exit"] == 100.6
    assert f"{rows[0]['t_in'] + pd.Timedelta(minutes=15):%H:%M}" == "15:45"
