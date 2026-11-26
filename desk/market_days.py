"""US half days (NYSE closes 13:00 New York): the day-after-Thanksgiving Friday and Christmas Eve when it is a weekday.
Every intraday strategy shifts its 'out by' time earlier by the lost hours (bot: noise band / sweep close on the last
candle, last entry 30 min before; Session Momentum out at the close; Overnight Bias out 1 hour before), so no day trade
is left open over the weekend. IBS skips half days (bot.ibs_days needs the 15:30 candle), as in the backtests.
Add each new year's dates (NYSE calendar), next to bot.NYSE_HOLIDAYS."""
import datetime as dt

CLOSE, HALF_CLOSE = 960, 780            # minutes after midnight NY: 16:00, 13:00
HALF_DAYS = {dt.date.fromisoformat(d) for d in (
    "2026-11-27 2026-12-24 2027-11-26").split()}


def close_of(day):
    """Session end (minutes after midnight NY) of this trading day."""
    return HALF_CLOSE if day in HALF_DAYS else CLOSE


def short_by(day):
    """Minutes the session is shorter than normal (0 on a full day)."""
    return CLOSE - close_of(day)
