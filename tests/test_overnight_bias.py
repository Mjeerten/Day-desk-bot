"""Overnight Bias ORB in the bot (desk/overnight_bias.py): the same trades as the round-27 backtest (no network)."""
import datetime as dt

import pandas as pd

from desk import overnight_bias as ob
from desk import round27 as r27
from tests.test_session_momentum import GAP, NY, market


def _ref(f):
    rth, d = r27.prep(f)
    return r27.overnight_bias(rth, d, **r27.OB_TAUGHT)


def test_same_trades_as_round27():
    f, ndx = market(weeks=10, seed=3)
    ref = _ref(f)
    ref = ref[ref.day >= pd.Timestamp("2026-08-10")]
    assert len(ref) >= 5
    got = []
    for day in sorted({d.date() for d in ref.day}):
        now = pd.Timestamp(dt.datetime.combine(day, dt.time(18, 0)), tz=NY)
        got += [(day, t["side"], round(t["entry"] + GAP, 6), round(t["exit"] + GAP, 6), t["why"])
                for t in ob.trades(f, ndx, GAP, day, now)]
    want = [(t.day.date(), t.side, round(t.entry, 6), round(t.exit, 6), t.why.replace("time", "close")) for t in ref.itertuples()]
    assert got == want


def test_open_trade_during_the_day_and_no_peeking():
    f, ndx = market(weeks=10, seed=3)
    ref = _ref(f)
    t = ref[ref.day >= pd.Timestamp("2026-08-10")].iloc[0]
    day = t.day.date()
    full = ob.trades(f, ndx, GAP, day, pd.Timestamp(dt.datetime.combine(day, dt.time(18, 0)), tz=NY))
    first = full[0]
    now = first["t_in"]                                  # the bot's run right after the entry candle closed
    live = ob.trades(f, ndx, GAP, day, now)
    assert live and live[-1]["why"] == "open" and live[-1]["entry"] == first["entry"]
    assert live[-1]["stop"] == first["stop"] and live[-1]["target"] == first["target"]
    # before 9:45 nothing can be known
    assert ob.trades(f, ndx, GAP, day, pd.Timestamp(dt.datetime.combine(day, dt.time(9, 44)), tz=NY)) == []


def test_messages_and_button_ids():
    from desk import bot
    f, ndx = market(weeks=10, seed=3)
    ref = _ref(f)
    day = ref[ref.day >= pd.Timestamp("2026-08-10")].iloc[0].day.date()
    full = ob.trades(f, ndx, GAP, day, pd.Timestamp(dt.datetime.combine(day, dt.time(18, 0)), tz=NY))
    msgs = ob.messages(full, day, 1000, off=GAP)
    keys = [m[0] for m in msgs]
    assert keys[0] == f"{day:%Y%m%d}-ob1_entry"
    assert bot.paper_id(keys[0]) == (f"{day:%Y%m%d}-ob1", True)
    assert len([k for k in keys if k.endswith("_entry")]) == len(full)
    for k in keys[1:]:
        if not k.endswith("_entry"):
            assert bot.paper_id(k)[1] is False and bot.paper_id(k)[0].startswith(f"{day:%Y%m%d}-ob")
    first = full[0]
    body = msgs[0][2]
    assert ob.fmt(first["stop"] + GAP) in body and ob.fmt(first["target"] + GAP) in body and "I took it" in body
    # while the trade runs only its entry message exists
    live = ob.trades(f, ndx, GAP, day, first["t_in"])
    assert [m[0] for m in ob.messages(live, day, 1000)] == [f"{day:%Y%m%d}-ob1_entry"]
