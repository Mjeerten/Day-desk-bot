"""Fast runner: the bot within ~15-30 seconds of each candle close instead of ~2 minutes (26 Nov 2026).

Round 17: a 2-minute delay costs the noise band and the overnight sweep about 2 %/yr each. The every-15-minutes timer
(cron-job.org -> daybot.yml) lands ~1.8 min after the close; this loop runs on one GitHub server through the session
(`.github/workflows/fast.yml`, parts of one hour each, one after the other on a new server) and starts `python -m desk.bot` right after
every 5-minute boundary, as soon as Yahoo shows the new minute (so the candle that just closed is final).

- daybot.yml starts it (`python -m desk.fast --should-start` -> dispatch fast.yml) on a weekday session morning and
  skips its own bot run while a fast run is going, so only one of them sends orders (the sent-message list moves on
  through the cache: each part restores the newest and saves its own at the end).
- If IG's firewall blocks this server, the part ends early; the next part (or daybot's own retry) runs on a new server.
- Variable FAST_RUNNER = off -> daybot.yml never starts it (back to the old 15-minute runs).
Each bot run is a fresh process, exactly what a daybot.yml run does; nothing in the trading rules changes.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import tempfile
import time

from desk.bot import CLOSE, NY, NYSE_HOLIDAYS, OPEN, yahoo

STEP = 5          # minutes between bot runs (5m candles for Session Momentum / Overnight Bias; 15m ones at :00/:15/...)
SETTLE = 8        # seconds after the boundary before the first look at Yahoo
WAIT = 100        # seconds to wait for Yahoo's new minute before running anyway
START, STOP = OPEN - 20, CLOSE + 32   # 9:10 - 16:32 NY: the window in which daybot.yml starts / keeps a fast run
PART = 60         # minutes per fast.yml matrix part (GitHub stops a job after 6 h; a new server each hour also helps vs IG)


def should_start(now):
    """True if a fast run belongs to this moment (weekday, not a holiday, between 9:10 and 16:32 NY)."""
    now = now.astimezone(NY)
    m = now.hour * 60 + now.minute
    return now.weekday() < 5 and now.date() not in NYSE_HOLIDAYS and START <= m < STOP


def part_end(now):
    """A part runs one hour from its start, at most until 16:32 NY (the parts follow each other from whenever the run began)."""
    stop = dt.datetime.combine(now.date(), dt.time(STOP // 60, STOP % 60), NY)
    return min(now + dt.timedelta(minutes=PART), stop)


def next_tick(now, step=STEP):
    """The next 5-minute boundary strictly after `now`."""
    base = now.replace(second=0, microsecond=0)
    return base + dt.timedelta(minutes=step - base.minute % step)


def in_session(t):
    m = t.hour * 60 + t.minute
    return t.weekday() < 5 and OPEN < m <= CLOSE + 30


def fresh(t, last_index):
    """Yahoo is up to date for boundary `t` once it shows a 1-minute candle starting at `t` or later
    (then the minute before, the last one of the closed candle, is complete)."""
    return last_index is not None and last_index >= t


def wait_for_data(t, get=None, sleep=time.sleep, clock=None):
    """Poll Yahoo's ^NDX 1m candles after boundary `t` until the new minute shows (at most WAIT s). Returns seconds waited."""
    clock = clock or (lambda: dt.datetime.now(NY))
    get = get or (lambda: yahoo("^NDX", "1m", "1d").index[-1])
    while True:
        waited = (clock() - t).total_seconds()
        try:
            if fresh(t, get()):
                return waited
        except Exception as err:   # noqa: a Yahoo hiccup: try again
            print(f"  Yahoo: {err!r}")
        if waited >= WAIT:
            print(f"  no new minute from Yahoo after {waited:.0f} s; running anyway")
            return waited
        sleep(5)


def run_bot():
    """One bot run in a fresh process (same as a daybot.yml run). True if IG blocked this server."""
    with tempfile.NamedTemporaryFile("r", suffix=".out", delete=False) as f:
        out = f.name
    env = dict(os.environ, GITHUB_OUTPUT=out)
    try:
        subprocess.run([sys.executable, "-m", "desk.bot"], env=env, timeout=240)
    except subprocess.TimeoutExpired:
        print("  bot run took over 4 minutes: stopped, the next run tries again")
    with open(out) as f:
        blocked = "ig_blocked=true" in f.read()
    os.unlink(out)
    return blocked


def loop(until, sleep=time.sleep, clock=None, bot=run_bot, wait=wait_for_data):
    """Run the bot now, then after every boundary until `until`. Returns 'blocked' if IG blocked this server, else 'done'."""
    clock = clock or (lambda: dt.datetime.now(NY))
    if clock() >= until:
        print(f"part ended already ({until:%H:%M} NY)")
        return "done"
    print(f"{clock():%H:%M:%S} NY: first run")
    if bot():
        return "blocked"
    while True:
        t = next_tick(clock())
        if t > until:
            return "done"
        sleep(max(0.0, (t - clock()).total_seconds() + SETTLE))
        waited = wait(t) if in_session(t) else 0.0
        print(f"--- {t:%H:%M} NY run (+{(clock() - t).total_seconds():.0f} s, data wait {waited:.0f} s)")
        if bot():
            print("IG blocked this server: ending this part, the next one starts on another server")
            return "blocked"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--should-start", action="store_true", help="print yes / no: does a fast run belong to now")
    a = ap.parse_args()
    now = dt.datetime.now(NY)
    if a.should_start:
        print("yes" if should_start(now) else "no")
        return
    if not should_start(now):
        print(f"{now:%a %H:%M} NY: no fast run now (weekend, holiday or outside 9:10-16:32)")
        return
    print(loop(part_end(now)))


if __name__ == "__main__":
    main()
