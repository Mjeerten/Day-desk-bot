"""1-minute price collector for the broker-price stop backtest (round 50, 26 Nov 2026).

The bot decides on the Nasdaq-100 cash index (Yahoo ^NDX); the brokers trade a CFD that follows the futures, so a stop
can be touched on the broker and not on the index (25 Nov sweep). To measure how often that matters we need the three
prices side by side, minute by minute. Yahoo keeps 1-minute candles ~30 days, so this runs on GitHub (Yahoo is blocked
in Claude's container) and adds to the files each time (prices.yml, daily):

  journal/prices/ndx_1m.csv      ^NDX cash index            (Yahoo)
  journal/prices/nq_1m.csv       NQ=F futures, ~10 min late (Yahoo, with pre/post market)
  journal/prices/capital_1m.csv  Capital.com demo US100 mid (the broker's own price; read only, no orders)
  and *_15m.csv of all three (60 days: the noise band needs 14 sessions before the first test day)

Columns: time (UTC, ISO), open, high, low, close. Run: python -m desk.prices [--days 29]
"""
import argparse
import datetime as dt
import os
import time

import pandas as pd
import requests

OUT = "journal/prices"
UA = {"User-Agent": "Mozilla/5.0"}


def yahoo_1m(symbol, days, prepost=False):
    """Yahoo gives 1m candles in chunks of at most 8 days, back ~30 days."""
    end = int(time.time())
    start = end - days * 86400
    parts = []
    t = start
    while t < end:
        t2 = min(t + 7 * 86400, end)
        r = requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", headers=UA, timeout=30,
                         params={"interval": "1m", "period1": t, "period2": t2, "includePrePost": str(prepost).lower()})
        if r.ok:
            j = r.json()["chart"]["result"][0]
            if j.get("timestamp"):
                q = j["indicators"]["quote"][0]
                parts.append(pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close")},
                                          index=pd.to_datetime(j["timestamp"], unit="s", utc=True)))
        else:
            print(f"Yahoo {symbol} {dt.datetime.utcfromtimestamp(t):%d %b}: HTTP {r.status_code}")
        t = t2
    if not parts:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.concat(parts).dropna()
    return df[~df.index.duplicated(keep="last")]


def yahoo_15m(symbol, prepost=False):
    from desk.bot import yahoo
    df = yahoo(symbol, "15m", "60d", prepost=prepost)
    df.index = df.index.tz_convert("UTC")
    return df


def capital_1m(days, res="MINUTE", step=1):
    """Capital.com demo price history (mid of bid / ask), 1000 candles per request (res MINUTE / MINUTE_15, step minutes)."""
    from desk.capital import EPIC, Capital
    c = Capital()
    c.login()
    end = dt.datetime.now(dt.timezone.utc).replace(second=0, microsecond=0)
    t = end - dt.timedelta(days=days)
    rows = []
    while t < end:
        t2 = min(t + dt.timedelta(minutes=990 * step), end)
        try:
            r = c.req("GET", f"/prices/{EPIC}", params={"resolution": res, "max": 1000,
                                                        "from": t.strftime("%Y-%m-%dT%H:%M:%S"),
                                                        "to": t2.strftime("%Y-%m-%dT%H:%M:%S")})
            for p in r.json().get("prices", []):
                mid = {k: (p[f"{k}Price"]["bid"] + p[f"{k}Price"]["ask"]) / 2 for k in ("open", "high", "low", "close")}
                rows.append((p["snapshotTimeUTC"], mid))
        except Exception as err:   # noqa: no history that far back, a closed market, or a rate limit
            if "error.prices.not-found" not in str(err):
                print(f"Capital.com {t:%d %b %H:%M}: {str(err)[:160]}")
        t = t2
        time.sleep(0.25)
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close"])
    df = pd.DataFrame([m for _, m in rows], index=pd.to_datetime([s for s, _ in rows], utc=True))
    return df[~df.index.duplicated(keep="last")].sort_index()


def merge(path, new):
    """Add the new candles to the file (new values win for the same minute)."""
    if os.path.exists(path):
        old = pd.read_csv(path, index_col=0, parse_dates=True)
        old.index = pd.to_datetime(old.index, utc=True)
        new = pd.concat([old, new])
        new = new[~new.index.duplicated(keep="last")]
    new = new.sort_index()
    new.index.name = "time"
    new.round(2).to_csv(path)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=29)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name, get in (("ndx", lambda: yahoo_1m("^NDX", a.days)),
                      ("nq", lambda: yahoo_1m("NQ=F", a.days, prepost=True)),
                      ("capital", lambda: capital_1m(a.days)),
                      ("ndx_15m", lambda: yahoo_15m("^NDX")),
                      ("nq_15m", lambda: yahoo_15m("NQ=F", prepost=True)),
                      ("capital_15m", lambda: capital_1m(59, "MINUTE_15", 15))):
        try:
            df = merge(f"{OUT}/{name if name.endswith('15m') else name + '_1m'}.csv", get())
            print(f"{name}: {len(df)} candles, {df.index[0]:%d %b %Y} - {df.index[-1]:%d %b %Y %H:%M} UTC" if len(df)
                  else f"{name}: none")
        except Exception as err:   # noqa: one source failing must not stop the others
            print(f"{name}: failed ({err!r})")


if __name__ == "__main__":
    main()
