"""Print today's Session Momentum trade with every 5-minute futures and index candle (debug, runs on GitHub)."""
import pandas as pd

from desk import session_momentum as sm
from desk.bot import basis, yahoo

nq5, ndx5 = yahoo("NQ=F", "5m", "5d", prepost=True), yahoo("^NDX", "5m", "5d")
now = pd.Timestamp.now(tz=sm.NY)
day = now.date()
gap = basis([(nq5, ndx5)], day)
dec = sm.decision(nq5, ndx5, gap, day, now)
print("now", now, "gap", gap, "decision", dec)
f = nq5[nq5.index >= pd.Timestamp(day).tz_localize(sm.NY) + pd.Timedelta(hours=9)]
x = ndx5[ndx5.index >= pd.Timestamp(day).tz_localize(sm.NY)]
t = pd.concat({"nq": f[["open", "high", "low", "close"]], "ndx+gap": x.close + gap}, axis=1)
with pd.option_context("display.max_rows", 500, "display.width", 200):
    print(t.round(1).to_string())
if dec:
    print(sm.manage(nq5, dec, day, now))
