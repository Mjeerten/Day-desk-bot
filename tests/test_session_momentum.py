"""Session Momentum in the bot (desk/session_momentum.py): same trades as the round-25 backtest, messages (no network)."""
import datetime as dt

import numpy as np
import pandas as pd

from desk import round25 as r25
from desk import session_momentum as sm

NY = "America/New_York"
GAP = 150.0


def market(weeks=8, seed=1):
    """5-minute futures candles (Sunday 18:00 - Friday 17:00, pause 17-18) and the index = futures - GAP in 9:30-16:00."""
    ix = pd.date_range("2026-08-02 18:00", periods=weeks * 7 * 288, freq="5min", tz=NY)
    wd, hr = ix.dayofweek, ix.hour
    ix = ix[~((wd == 5) | ((wd == 4) & (hr >= 17)) | ((wd == 6) & (hr < 18)) | (hr == 17))]
    rng = np.random.default_rng(seed)
    c = 20000 * np.exp(np.cumsum(rng.normal(0, 0.0012, len(ix))))
    o = np.r_[c[0], c[:-1]]
    w = np.abs(rng.normal(0, 0.0006, len(ix))) * c
    f = pd.DataFrame(dict(open=o, high=np.maximum(o, c) + w, low=np.minimum(o, c) - w, close=c), index=ix)
    tod = ix.hour * 60 + ix.minute
    return f, f[(tod >= 570) & (tod < 960)] - GAP


def test_same_trades_as_round25():
    f, ndx = market()
    b = r25.prep(f)                       # lean_bars(5) on 5-minute data = the same candles
    ref = r25.run(b, r25.BOT, 0.0)
    ref = ref[ref.day >= pd.Timestamp("2026-08-10")]
    n = 0
    for t in ref.itertuples():
        day = t.day.date()
        now = pd.Timestamp(dt.datetime.combine(day, dt.time(18, 0)), tz=NY)
        dec = sm.decision(f, ndx, GAP, day, now)
        st = sm.manage(f, dec, day, now)
        assert dec["side"] == t.side and abs(dec["entry"] - t.entry) < 1e-6
        assert abs(st["exit"] - t.exit) < 1e-6 and st["why"] == t.why
        n += 1
    assert n > 20


def test_no_signal_before_935_and_trail_messages():
    f, ndx = market(seed=2)
    day = dt.date(2026, 8, 18)
    early = pd.Timestamp("2026-08-18 09:34", tz=NY)
    assert sm.decision(f, ndx, GAP, day, early) is None
    assert sm.decision(f, ndx, np.nan, day, early + pd.Timedelta(minutes=10)) is None
    now = pd.Timestamp("2026-08-18 09:41", tz=NY)
    dec = sm.decision(f, ndx, GAP, day, now)
    assert dec is not None and dec["t_in"] == pd.Timestamp("2026-08-18 09:35", tz=NY)
    msgs = sm.messages(dec, sm.manage(f, dec, day, now), day, 1000)
    key, title, body, _, prio = msgs[0]
    assert key == "20260818-sm_entry" and "(paper) 15:35" in title and prio == 5
    assert "Until 15:55" in body and "Stop loss:" in body
    # a stop that moved 0.6 x risk in profit -> one 'move your stop' message (step 2 of 0.25)
    risk = dec["entry"] * sm.STOP
    st = dict(stop=dec["entry"] - dec["side"] * risk * 0.4, best=dec["entry"] + dec["side"] * risk * 0.6, why="open",
              exit=dec["entry"], t_out=None)
    m = sm.messages(dec, st, day, 1000)
    assert [x[0] for x in m] == ["20260818-sm_entry", "20260818-sm_trail2"]
