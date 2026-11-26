"""Day bot signals on a hand-made session (no network)."""
import datetime as dt

import numpy as np
import pandas as pd

from desk import bot

DAY = dt.date(2026, 10, 7)


def session(closes, highs=None, lows=None):
    ix = pd.date_range(dt.datetime.combine(DAY, dt.time(9, 30)), periods=26, freq="15min", tz=bot.NY)
    c = np.array(closes, float)
    o = np.r_[100.0, c[:-1]]
    h = np.maximum(o, c) + 0.1 if highs is None else np.array(highs, float)
    lo = np.minimum(o, c) - 0.1 if lows is None else np.array(lows, float)
    return pd.DataFrame(dict(open=o, high=h, low=lo, close=c), index=ix)


def test_sweep_long_stop_target_and_one_per_day():
    c = [100.0] * 26
    c[1] = 99.5                     # 9:45 candle: wick to 98 under the overnight low 99, close back inside
    lows = [99.9] * 26
    lows[1] = 98.0
    highs = [100.1] * 26
    highs[4] = 103.5                # target = 99.5 + 2.5 x 1.5 = 103.25
    c[5], lows[5] = 98.5, 98.4      # a second sweep later in the day is ignored
    ev, open_ = bot.day_events(session(c, highs, lows), onh=110, onl=99, line=None, o=100, day=DAY)
    kinds = [e["kind"] for e in ev]
    assert kinds == ["sweep_entry", "sweep_target"]
    tr = ev[0]["trade"]
    assert (tr["side"], tr["stop"], tr["target"]) == (1, 98.0, 103.25)
    assert not open_


def test_noise_buys_above_line_and_sells_under_open():
    c = [100.2] * 26
    c[1] = 100.25                   # 10:00 close above the line (100.22): buy, stop 0.3% lower = 99.95
    c[3] = 99.97                    # 10:30 close under the open (100) without touching the stop: sell
    lows = [99.96] * 26
    line = pd.Series(100.22, index=range(bot.OPEN, bot.CLOSE, 15))
    ev, _ = bot.day_events(session(c, lows=lows), onh=np.nan, onl=np.nan, line=line, o=100.0, day=DAY)
    assert [(e["kind"], e["px"]) for e in ev][:2] == [("noise_entry", 100.25), ("noise_exit", 99.97)]


def test_noise_stop_then_no_buy_in_the_same_candle():
    c = [100.2] * 26
    c[1] = 101.0                    # buy at 101, stop 100.697
    c[3] = 99.0                     # the 10:15 candle (low 100.1) already hits the stop
    line = pd.Series(100.8, index=range(bot.OPEN, bot.CLOSE, 15))
    ev, _ = bot.day_events(session(c), onh=np.nan, onl=np.nan, line=line, o=100.0, day=DAY)
    assert [e["kind"] for e in ev] == ["noise_entry", "noise_stop"]
    assert ev[1]["ts"].minute == 0 and round(ev[1]["px"], 3) == 100.697


def test_messages_carry_trading212_prices(tmp_path):
    ts = pd.Timestamp(dt.datetime.combine(DAY, dt.time(9, 45)), tz=bot.NY)
    tr = dict(side=1, entry=31000.0, stop=30900.0, target=31250.0, dist=100.0, level=30950.0)
    key, title, body, _ = bot.message(dict(kind="sweep_entry", ts=ts, px=31000.0, trade=tr), off=5, fx=1.1,
                                      stake=1000)
    assert title == "SWEEP BUY 16:00" and key == "20261007-sweep_entry-1000"
    assert body.startswith("ORDER") and "Stop loss: 30905\n" in body and "Take profit: 31255\n" in body
    assert "Value: 1000 " in body
    from desk.ticket import copy_actions, click_url
    assert click_url(body) == "https://app.trading212.com/" and click_url("Monday dip skipped") is None
    assert copy_actions(body) == "copy, Value 1000, 1000; copy, TP 31255, 31255; copy, SL 30905, 30905"
    state = tmp_path / "s.json"
    bot.save_sent(str(state), {key, "20200101-old"}, DAY)
    assert bot.load_sent(str(state)) == {key}


def ibs_day(day, close_at_1545):
    """A session with range 99-101 until 15:45; the 15:30 candle closes at close_at_1545."""
    ix = pd.date_range(dt.datetime.combine(day, dt.time(9, 30)), periods=26, freq="15min", tz=bot.NY)
    c = np.full(26, 100.0)
    c[24] = close_at_1545                      # the 15:30-15:45 candle
    return pd.DataFrame(dict(open=100.0, high=np.maximum(c, 101.0), low=np.minimum(c, 99.0), close=c), index=ix)


def test_ibs_buy_keep_sell_over_days():
    days = [dt.date(2026, 10, d) for d in (5, 6, 7, 8)]
    df = pd.concat([ibs_day(days[0], 100.5), ibs_day(days[1], 99.2), ibs_day(days[2], 99.1),
                    ibs_day(days[3], 100.6)])          # IBS 0.75, 0.10, 0.05, 0.80
    kinds = []
    for d in days:
        ev, held = bot.ibs_events(df, d)
        kinds.append([e["kind"] for e in ev])
    assert kinds == [[], ["ibs_entry"], ["ibs_hold"], ["ibs_exit"]]
    ev, held = bot.ibs_events(df, days[3])
    assert held == (days[1], 99.2) and ev[0]["px"] == 100.6
    assert ev[0]["ts"] == pd.Timestamp(dt.datetime.combine(days[3], dt.time(15, 30)), tz=bot.NY)
    # before 15:45 there is no decision yet, but the open trade is known (for the morning reminder)
    ev, held = bot.ibs_events(df[df.index < pd.Timestamp(dt.datetime.combine(days[2], dt.time(15, 0)), tz=bot.NY)],
                              days[2])
    assert ev == [] and held == (days[1], 99.2)
    ts = pd.Timestamp(dt.datetime.combine(days[1], dt.time(15, 30)), tz=bot.NY)
    key, title, body, _ = bot.message(dict(kind="ibs_entry", ts=ts, px=31000.0, ibs=0.1), off=200, fx=1.1, stake=1000)
    assert title == "IBS BUY 21:45" and "No stop loss" in body and "31 200" in body and "Stop loss: none" in body and "until 22:15" in body


def test_sunday_reminder_only_sunday_evening():
    sun = dt.datetime(2026, 10, 25, 19, 3, tzinfo=bot.AMS)
    key, title, body, _ = bot.sunday_reminder(sun)
    assert key == "20261025-weekly" and "screenshots" in title and "Sun 01 Nov" in body
    assert "History > Result" in body and "History > Orders" in body and "account value" in body
    assert "new month" not in bot.sunday_reminder(dt.datetime(2026, 10, 11, 20, 30, tzinfo=bot.AMS))[2]
    assert bot.sunday_reminder(dt.datetime(2026, 10, 11, 22, 50, tzinfo=bot.AMS))  # late GitHub run still sends
    assert bot.sunday_reminder(dt.datetime(2026, 10, 25, 18, 59, tzinfo=bot.AMS)) is None
    assert bot.sunday_reminder(dt.datetime(2026, 10, 24, 19, 0, tzinfo=bot.AMS)) is None


def test_no_sweep_on_fed_decision_day(monkeypatch):
    c = [100.0] * 26
    c[1] = 99.5
    lows = [99.9] * 26
    lows[1] = 98.0                  # the same sweep as above, but on a Fed decision day
    monkeypatch.setattr(bot, "FOMC_DAYS", {DAY})
    ev, _ = bot.day_events(session(c, lows=lows), onh=110, onl=99, line=None, o=100, day=DAY)
    assert not [e for e in ev if e["kind"].startswith("sweep")]


def test_alarm_next_session_at_940_ny():
    ny = lambda *a: dt.datetime(*a, tzinfo=bot.NY)
    key, when, *_ = bot.next_alarm(ny(2026, 10, 26, 8, 0))           # Monday morning: today
    assert key == "20261026-alarm" and when == ny(2026, 10, 26, 9, 40)
    assert when.astimezone(bot.AMS).hour == 14                         # the week the clocks differ: 14:40 Amsterdam
    assert bot.next_alarm(ny(2026, 10, 26, 9, 36))[0] == "20261027-alarm"   # too late for today: tomorrow
    assert bot.next_alarm(ny(2026, 10, 30, 16, 0))[0] == "20261102-alarm"   # Friday evening: Monday
    assert bot.next_alarm(ny(2026, 11, 2, 10, 0))[1].astimezone(bot.AMS).hour == 15  # normal weeks: 15:40
    assert bot.next_alarm(ny(2026, 11, 25, 12, 0))[0] == "20261127-alarm"   # skips Thanksgiving
    assert "Fed decision" in bot.next_alarm(ny(2026, 10, 27, 12, 0))[3]     # 28 Oct is a Fed day
    assert bot.next_alarm(ny(2026, 12, 23, 12, 0))[0] == "20261224-alarm"


def test_alarm_3_day_limit_counts_real_hours_over_a_clock_change():
    # Fri 30 Oct 2026 9:46 NY -> Mon 2 Nov 9:40 NY looks like 2d 23h54m on the wall clock, but New York's clocks go
    # back on 1 Nov, so it is really 3d 0h54m: past ntfy's 3-day limit (it answered 400 and the run crashed)
    ny = lambda *a: dt.datetime(*a, tzinfo=bot.NY)
    assert bot.next_alarm(ny(2026, 10, 30, 9, 46)) is None
    assert bot.next_alarm(ny(2026, 10, 30, 10, 46))[0] == "20261102-alarm"   # really 2d 23h54m: fine


def test_paper_follow_ups_only_after_i_took_it(tmp_path, monkeypatch):
    """Entry messages carry an 'I took it' button; stop / trail / sell messages only go out once it was tapped."""
    monkeypatch.setenv("STATE_FILE", str(tmp_path / "sent.json"))
    sent, taps = [], set()
    monkeypatch.setattr(bot, "push", lambda topic, key, title, body, tags, priority=4, at=None, took=None:
                        sent.append((key, took)))

    class R:
        def __init__(self):
            self.text = "\n".join(f'{{"event": "message", "message": "{t}"}}' for t in taps)

        def raise_for_status(self):
            pass

    monkeypatch.setattr(bot.requests, "get", lambda *a, **k: R())
    day = dt.date(2026, 11, 2)
    msgs = [("20261102-sm_entry", "SESSION MOMENTUM SHORT (paper) 15:35", "b", [], 5),
            ("20261102-sm_trail1", "move its stop", "b", [], 4)]
    bot.send_paper(msgs, "t", True, day)
    assert sent == [("20261102-sm_entry", "20261102-sm")]          # button on the entry, no trail message
    taps.add("20261102-sm")
    bot.send_paper(msgs, "t", True, day)
    assert sent[-1] == ("20261102-sm_trail1", None)
    # a 'missed' note has no button; a Monday dip sent before the button existed keeps its sell message
    bot.save_sent(str(tmp_path / "sent.json"), {"20261102-md_entry-0200"}, day)
    taps.clear()
    sent.clear()
    bot.send_paper([("20261103-md_entry-0300", "Monday dip missed (09:00): skip it", "b", [], 2),
                    ("20261102-md_time-11030200", "MONDAY DIP SELL 08:00", "b", [], 5)], "t", True, day)
    assert sent == [("20261103-md_entry-0300", None), ("20261102-md_time-11030200", None)]
    assert bot.paper_id("20261102-md_stop-11021500") == ("20261102-md", False)
    assert bot.paper_id("20261102-noise_buy-1030") == (None, False)


def test_missed_ibs_trade_gets_no_keep_or_sell(tmp_path):
    """An IBS buy the owner did not take (file or 'I missed it' tap): no keep / sell, not held, kept in the state."""
    days = [dt.date(2026, 10, d) for d in (5, 6, 7, 8)]
    df = pd.concat([ibs_day(days[0], 100.5), ibs_day(days[1], 99.2), ibs_day(days[2], 99.1),
                    ibs_day(days[3], 100.6)])          # buy on the 6th, keep on the 7th, sell on the 8th
    f = tmp_path / "missed.txt"
    f.write_text("# comment\n2026-10-06  # missed\n")
    missed = bot.ibs_missed(path=str(f))
    assert missed == {days[1]}
    assert bot.ibs_missed(["20261009-ibs", "20261009-sm", "junk"], path=str(tmp_path / "none")) == {dt.date(2026, 10, 9)}
    for d in days[2:]:
        assert bot.ibs_taken(df, d, missed) == ([], None)
        assert bot.ibs_taken(df, d)[1] == (days[1], 99.2)         # without the list: still held
    assert [e["kind"] for e in bot.ibs_taken(df, days[1], missed)[0]] == ["ibs_entry"]   # the buy itself is unchanged
    # a tap is kept for weeks (the trade can stay open long), other keys a week
    p = str(tmp_path / "sent.json")
    bot.save_sent(p, {"20261006-ibs_taken", "20261006-levels"}, dt.date(2026, 10, 20))
    assert bot.load_sent(p) == {"20261006-ibs_taken"}


def test_phone_alerts_off_sends_nothing(monkeypatch):
    import requests
    sent = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: sent.append(a))
    monkeypatch.setenv("PHONE_ALERTS", "off")
    bot.push("topic", "20261109-levels", "Day bot running", "x", ["bell"])
    assert sent == []


def test_scale_now_uses_index_in_session_and_futures_outside(monkeypatch):
    import desk.bot as b
    ny = "America/New_York"
    t = pd.date_range("2026-11-16 10:00", periods=30, freq="1min", tz=ny)
    ndx1 = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0}, index=t)
    nq1 = ndx1.assign(close=90.0)          # late futures: 10 points behind
    monkeypatch.setattr(b, "five_minute", lambda now, day: (None, None, nq1, ndx1, 50.0))
    now = pd.Timestamp("2026-11-16 10:30:30", tz=ny)
    assert b.scale_now(now, None) == 150.0              # ^NDX 100 + basis 50
    night = pd.Timestamp("2026-11-16 20:00", tz=ny)     # index stale -> last futures close
    assert b.scale_now(night, None) == 90.0


def test_half_day_closes_day_trades_at_13():
    import datetime as dt
    from desk import bot, session_momentum as sm, overnight_bias as ob
    from desk.market_days import close_of
    day = dt.date(2026, 11, 27)                                   # day after Thanksgiving: NYSE closes 13:00
    assert close_of(day) == 780 and close_of(dt.date(2026, 11, 30)) == 960
    ix = pd.date_range("2026-11-27 09:30", "2026-11-27 12:45", freq="15min", tz="America/New_York")
    px = np.linspace(100, 104, len(ix))                           # steady rise: noise band buys at 10:00
    ndx = pd.DataFrame({"open": px, "high": px + 0.1, "low": px - 0.1, "close": px}, index=ix)
    line = pd.Series(99.0, index=range(570, 960, 15))
    ev, still_open = bot.day_events(ndx, np.nan, np.nan, line, 100.0, day)
    kinds = [e["kind"] for e in ev]
    assert kinds[0] == "noise_entry" and kinds[-1] == "noise_close" and not still_open
    assert ev[-1]["ts"] == pd.Timestamp("2026-11-27 12:45", tz="America/New_York")
    assert dt.date(2026, 11, 26) in bot.NYSE_HOLIDAYS
