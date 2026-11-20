"""weekly_check: parse the bot's broker lines and judge a week (no network)."""
import pandas as pd

from desk import weekly_check as w

LOG = """\
2026-11-19T14:46:45.1Z [IG] BUY Session Momentum at 27 651.5 | IG demo: OPENED: bought 0.2 US Tech 100 (EUR 0.2 a point) on IG demo.
2026-11-19T15:01:34.1Z [IG] SHORT overnight sweep at 27 606.9 | IG demo: OPENED: sold short 0.2 US Tech 100 (EUR 0.2 a point) on IG demo.
2026-11-19T15:31:32.1Z [IG] -EUR 52.38 (about) | Session Momentum stop hit | IG demo: Result about -EUR 52.38 (-261.9 points).
2026-11-19T15:31:32.1Z [IG] +EUR 73.90 (about) | overnight sweep target hit | IG demo: Result about +EUR 73.90 (+369.5 points).
2026-11-19T15:46:32.1Z [IG] BUY overnight sweep at 27 062.8 | IG demo: OPENED: bought 0.2 US Tech 100 (EUR 0.2 a point) on IG demo.
2026-11-19T17:01:29.1Z [IG] -EUR 22.76 (about) | overnight sweep stop hit | IG demo: Result about -EUR 22.76 (-113.8 points).
2026-11-19T22:01:00.1Z [IG] -EUR 38.16 (-0.38%) today | IG demo: Day result -EUR 38.16 (-0.38%).
2026-11-19T15:01:34.1Z [CAP] SHORT overnight sweep at 27 621.4 | Capital.com demo (dry run): OPENED: x
"""


def test_parse_and_judge():
    ev = w.parse(LOG)
    assert [e["kind"] for e in ev] == ["open", "open", "close", "close", "open", "close", "day"]
    t = w.trades(ev)
    assert list(t.pts) == [-261.9, 369.5, -113.8] and set(t.broker) == {"IG demo"}
    bot = pd.DataFrame(dict(date=["2026-11-19"] * 2, strat=["P", "S"], side=[1, -1], pts=[-328.4, 368.0]))
    bot["n"] = bot.groupby(["date", "strat"]).cumcount()
    txt = w.check(t, bot, ev, set(), "2026-11-16")
    assert "EXTRA" in txt and "NOT CLEAN" in txt          # the second sweep
    txt = w.check(t.iloc[:2], bot, ev, set(), "2026-11-16")
    assert "-> CLEAN" in txt                               # SM gap 66 < 100, sweep +1.5
