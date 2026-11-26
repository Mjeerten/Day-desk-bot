"""Go-live check: every demo broker's trades next to the bot's own trades, per week (plan agreed 20 Nov 2026,
AUTOMATION.md 'Going live').

1. --fetch: reads the day bot's run logs (public repo Mjeerten/day-desk-bot, GitHub API) and keeps every broker order and
   close in journal/demo_trades.csv (date NY, broker, strategy, side, opened / closed UTC, points, euros; older rows kept,
   run logs expire after 90 days).
2. Matches them with the bot's paper logs (journal/paper_desk.csv = noise band / sweep / IBS, paper_momentum.csv,
   paper_obias.csv, paper_monday.csv; bot points = index points, after 2 points of cost) by date, strategy and order.
3. Writes journal/demo_check.txt: per broker per week (Mon-Fri) clean or not:
   - no extra trade (broker trade the bot did not make) and no missed one (bot trade the broker did not make);
   - no 'refused' / 'not checked' / 'problem' lines in the runs; a daily report every trading day;
   - no single trade more than TRADE_GAP points from the bot's (unless marked 'explained' in journal/demo_explained.txt:
     date,broker,strategy,reason - e.g. the 10-15 minute entry delay);
   - the week's points within WEEK_GAP of the bot's.
   Four clean weeks in a row on the broker that goes live = go (the owner decides).
The yen week has no bot paper log: listed per broker, not judged.

Run: python -m desk.weekly_check [--fetch] [--since YYYY-MM-DD]
"""
import argparse
import io
import json
import os
import re
import zipfile

import numpy as np
import pandas as pd
import requests

REPO = "Mjeerten/day-desk-bot"
TRADE_GAP, WEEK_GAP = 100.0, 200.0
START = "2026-11-23"                      # first week with every 19-20 Nov fix live
CODES = {"noise band": "N", "overnight sweep": "S", "IBS": "I", "Monday dip": "M", "Session Momentum": "P",
         "Overnight Bias ORB": "O"}
NAME = "|".join(sorted(map(re.escape, CODES), key=len, reverse=True))
TS = r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\S*\s+"
OPEN_RE = re.compile(TS + r"\[\w+\] (BUY|SHORT) (" + NAME + r") at ([\d ]+\.?\d*) \| (.+?): OPENED")
CLOSE_RE = re.compile(TS + r"\[\w+\] [-+]EUR [\d.]+(?: \(about\))? \| (" + NAME + r") (.+?) \| (.+?): Result (?:about )?"
                      r"([-+])EUR ([\d.]+) \(([-+][\d.]+) points\)")
DAY_RE = re.compile(TS + r"\[\w+\] .*today \| (.+?): Day result")
YEN_RE = re.compile(TS + r".*yen week CLOSED \| (.+?): (.*)$")
BAD_RE = re.compile(TS + r"(.*(?:refused|not checked|problem|Traceback|not opened \((?!market closed)).*)$")
NY = "America/New_York"


def _broker(label):
    return label.replace(" (dry run)", "").strip()


def parse(text):
    """Broker events from one run log: opens, closes, day reports, yen closes, problems."""
    ev = []
    for line in text.splitlines():
        if "(dry run)" in line:
            continue
        if m := OPEN_RE.match(line):
            ev.append(dict(kind="open", utc=m[1], side=1 if m[2] == "BUY" else -1, strat=CODES[m[3]],
                           price=float(m[4].replace(" ", "")), broker=_broker(m[5])))
        elif m := CLOSE_RE.match(line):
            ev.append(dict(kind="close", utc=m[1], strat=CODES[m[2]], how=m[3], broker=_broker(m[4]),
                           eur=float(m[6]) * (1 if m[5] == "+" else -1), pts=float(m[7])))
        elif m := DAY_RE.match(line):
            ev.append(dict(kind="day", utc=m[1], broker=_broker(m[2])))
        elif m := YEN_RE.match(line):
            ev.append(dict(kind="yen", utc=m[1], broker=_broker(m[2]), text=m[3][:80]))
        elif m := BAD_RE.match(line):
            ev.append(dict(kind="bad", utc=m[1], text=m[2][:160]))
    return ev


def fetch(since, token=None):
    """All events of the day bot runs created since `since` (UTC date)."""
    h = {"Authorization": f"Bearer {token}"} if token else {}
    runs, page = [], 1
    while True:
        r = requests.get(f"https://api.github.com/repos/{REPO}/actions/runs", headers=h, timeout=30,
                         params=dict(per_page=100, page=page, created=f">={since}")).json()
        got = [x for x in r.get("workflow_runs", []) if x["name"] == "day bot"]
        runs += got
        if len(r.get("workflow_runs", [])) < 100:
            break
        page += 1
    ev = []
    for run in runs:
        z = requests.get(f"https://api.github.com/repos/{REPO}/actions/runs/{run['id']}/logs", headers=h, timeout=60)
        if z.status_code != 200:
            continue
        with zipfile.ZipFile(io.BytesIO(z.content)) as f:
            names = [n for n in f.namelist() if "/" not in n] or f.namelist()
            text = "\n".join(f.read(n).decode("utf-8", "replace") for n in names)
        ev += parse(text)
    # a run's log holds the steps both as one file and per step: keep each event once
    seen, out = set(), []
    for e in ev:
        k = json.dumps(e, sort_keys=True)
        if k not in seen:
            seen.add(k)
            out.append(e)
    return sorted(out, key=lambda e: e["utc"])


def trades(ev):
    """Pair every open with the next close of the same broker + strategy."""
    rows, open_ = [], {}
    for e in ev:
        if e["kind"] == "open":
            open_.setdefault((e["broker"], e["strat"]), []).append(e)
        elif e["kind"] == "close":
            q = open_.get((e["broker"], e["strat"]))
            o = q.pop(0) if q else None
            rows.append(dict(opened=o["utc"] if o else "", side=o["side"] if o else np.nan, entry=o["price"] if o else np.nan,
                             broker=e["broker"], strat=e["strat"], closed=e["utc"], how=e["how"], pts=e["pts"], eur=e["eur"]))
    for q in open_.values():
        rows += [dict(opened=o["utc"], side=o["side"], entry=o["price"], broker=o["broker"], strat=o["strat"], closed="",
                      how="open", pts=np.nan, eur=np.nan) for o in q]
    t = pd.DataFrame(rows, columns=["opened", "side", "entry", "broker", "strat", "closed", "how", "pts", "eur"])
    t["date"] = [f"{pd.Timestamp(x or y, tz='UTC').tz_convert(NY):%Y-%m-%d}" for x, y in zip(t.opened, t.closed)]
    return t.sort_values(["date", "opened", "broker"]).reset_index(drop=True)


def bot_trades(root):
    """The bot's own closed trades: date, strat, n (order in the day), side, pts."""
    out = []
    def read(name):
        p = os.path.join(root, name)
        return pd.read_csv(p) if os.path.exists(p) and os.path.getsize(p) else pd.DataFrame()
    d = read("paper_desk.csv")
    if len(d):
        d = d[d.why != "open"]
        out.append(pd.DataFrame(dict(date=d.date, strat=d.strat, side=np.where(d.side == "buy", 1, -1),
                                     pts=pd.to_numeric(d.pts) - 2.0)))
    for name, code in (("paper_momentum.csv", "P"), ("paper_obias.csv", "O"), ("paper_monday.csv", "M")):
        d = read(name)
        if len(d):
            d = d[pd.to_numeric(d.pct, errors="coerce").notna()]
            out.append(pd.DataFrame(dict(date=d.date, strat=code, side=np.where(d.side == "buy", 1, -1) if "side" in d else 1,
                                         pts=pd.to_numeric(d.pct) * pd.to_numeric(d.entry) / 100)))
    b = pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["date", "strat", "side", "pts"])
    b["n"] = b.groupby(["date", "strat"]).cumcount()
    return b


def explained(root):
    p = os.path.join(root, "demo_explained.txt")
    if not os.path.exists(p):
        return set()
    return {tuple(x.strip() for x in line.split(",")[:3]) for line in open(p, encoding="utf-8")
            if line.strip() and not line.startswith("#")}


def check(t, bot, ev, ok_set=frozenset(), since=START):
    """Report text: per broker per week the matched trades and the verdict."""
    t = t[(t.date >= since) & (t.strat.isin(list(CODES.values())))].copy()
    t["n"] = t.groupby(["broker", "date", "strat"]).cumcount()
    bot = bot[bot.date >= since]
    last_bot = bot.date.max() if len(bot) else since
    lines = [f"Demo brokers vs the bot, from {since} (trade gap {TRADE_GAP:.0f} pts, week gap {WEEK_GAP:.0f} pts)", ""]
    days_all = sorted({d for d in bot.date} | {d for d in t.date})
    for broker in sorted(t.broker.unique()):
        tb = t[t.broker == broker]
        first = tb.date.min()                       # a broker is judged from its first trade on
        m = tb.merge(bot[bot.date >= first], on=["date", "strat", "n"], how="outer", suffixes=("", "_bot"))
        m = m[(m.date <= last_bot) | m.how.notna()]
        m["week"] = [f"{(pd.Timestamp(d) - pd.Timedelta(days=pd.Timestamp(d).weekday())):%Y-%m-%d}" for d in m.date]
        rep = [e for e in ev if e["kind"] == "day" and e["broker"] == broker]
        rep_days = {f"{pd.Timestamp(e['utc'], tz='UTC').tz_convert(NY):%Y-%m-%d}" for e in rep}
        labels = ("IG demo:", "Capital.com demo:", "Capital.com REAL:", "FTMO")
        bad = [e for e in ev if e["kind"] == "bad" and (broker in e["text"] or not any(x in e["text"] for x in labels))]
        lines.append(f"=== {broker}")
        for wk, g in m.groupby("week"):
            why = []
            rows = []
            for r in g.sort_values(["date", "strat", "n"]).itertuples():
                tag = ""
                if pd.isna(r.pts_bot) and r.how == r.how:
                    tag = "EXTRA (bot made no such trade)"
                elif r.how != r.how:
                    tag = "MISSED (bot traded, broker did not)"
                elif r.how == "open":
                    tag = "OPEN on the broker, bot closed it" if r.pts_bot == r.pts_bot and r.date < last_bot else "still open"
                elif abs(r.pts - r.pts_bot) > TRADE_GAP:
                    tag = ("gap explained" if (r.date, broker, r.strat) in ok_set
                           else f"GAP {r.pts - r.pts_bot:+.0f} pts (explain or bug)")
                if tag and not tag.startswith(("still", "gap explained")):
                    why.append(tag.split(" ")[0].replace("OPEN", "LEFT OPEN"))
                rows.append(f"  {r.date} {r.strat}  broker {r.pts:+7.1f}  bot {r.pts_bot:+7.1f}  {tag}".replace("nan", "  -  "))
            done = g[g.how.notna() & (g.how != "open") & g.pts_bot.notna()]
            done = done[[(d, broker, k) not in ok_set for d, k in zip(done.date, done.strat)]]   # explained: out of the week gap
            diff = (done.pts - done.pts_bot).sum()
            if abs(diff) > WEEK_GAP:
                why.append(f"week gap {diff:+.0f}")
            wdays = [d for d in days_all if first <= d and wk <= d < f"{pd.Timestamp(wk) + pd.Timedelta(days=5):%Y-%m-%d}" and d <= last_bot]
            no_rep = [d for d in wdays if d not in rep_days]
            if no_rep:
                why.append(f"no daily report {','.join(no_rep)}")
            wbad = [e for e in bad if wk <= f"{pd.Timestamp(e['utc'], tz='UTC').tz_convert(NY):%Y-%m-%d}" <
                    f"{pd.Timestamp(wk) + pd.Timedelta(days=7):%Y-%m-%d}"]
            if wbad:
                why.append(f"{len(wbad)} problem lines")
            verdict = "CLEAN" if not why else "NOT CLEAN: " + "; ".join(dict.fromkeys(why))
            lines.append(f"week of {wk}: broker {g.pts.sum():+.0f} pts, bot {g.pts_bot.sum():+.0f} pts, "
                         f"difference {diff:+.0f} -> {verdict}")
            lines += rows
        lines.append("")
    yen = [e for e in ev if e["kind"] == "yen"]
    if yen:
        lines.append("Yen week closes (not judged):")
        lines += [f"  {e['utc'][:10]} {e['broker']}: {e['text']}" for e in yen]
    bad = [e for e in ev if e["kind"] == "bad"]
    if bad:
        lines += ["", "Problem lines in the runs:"] + [f"  {e['utc']} {e['text']}" for e in bad[-30:]]
    return "\n".join(lines)


def main():
    from desk.plan import ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--since", default=None, help="fetch runs from this date (default: 10 days back)")
    ap.add_argument("--from", dest="start", default=START, help="judge weeks from this date")
    a = ap.parse_args()
    tp, evp = os.path.join(ROOT, "demo_trades.csv"), os.path.join(ROOT, "demo_events.json")
    ev = json.load(open(evp)) if os.path.exists(evp) else []
    if a.fetch:
        since = a.since or f"{pd.Timestamp.now(tz='UTC') - pd.Timedelta(days=10):%Y-%m-%d}"
        new = fetch(since, os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))
        ev = sorted([e for e in ev if e["utc"] < since] + new, key=lambda e: e["utc"])
        json.dump(ev, open(evp, "w"), indent=0)
    t = trades(ev)
    t.to_csv(tp, index=False)
    txt = check(t, bot_trades(ROOT), ev, explained(ROOT), a.start)
    open(os.path.join(ROOT, "demo_check.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
