"""Library of popular day-trading strategies (web / YouTube / TikTok / books), tested on Nasdaq-100 bars.

Every strategy is turned into exact rules: a signal on a candle close, an optional stop and target, and
always flat at 16:00 New York. One engine runs them all so the results are comparable.

Engine rules
- Entries at the close of the signal candle (or at a given level for stop/limit-order setups).
- Stop and target are checked on the following candles' high/low. If one candle touches both, the stop
  counts (pessimistic). A gap through the stop fills at the candle's open.
- Entries only between the session start (9:30 NY unless the strategy says otherwise) and 15:30.
- Everything closed at the 16:00 close. Cost: 1 point per round trip (NQ tick + commission + slippage).
- Indicators use the full 24-hour series, as on a futures chart. The data has no volume, so 'VWAP'
  strategies use the session's time-weighted average price (TWAP) instead.

Run: python -m desk.library --bars1 <1-minute pickle> --out <dir>
"""
import argparse
import os

import numpy as np
import pandas as pd

RTH_OPEN, RTH_CLOSE, LAST_ENTRY = 570, 960, 930  # minutes after midnight NY


# ---------- bars and session levels ----------

def make_bars(m1, tf):
    if tf == 1:
        b = m1[["open", "high", "low", "close"]].copy()
    else:
        off = "30min" if tf == 60 else "0min"
        b = m1.resample(f"{tf}min", label="left", closed="left", offset=off).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    b["tod"] = b.index.hour * 60 + b.index.minute
    b["end"] = b.tod + tf
    # trade date: bars from 18:00 belong to the next session (futures convention)
    b["tdate"] = (b.index.tz_localize(None) + pd.Timedelta(hours=6)).normalize()
    b["rth"] = (b.tod >= RTH_OPEN) & (b.end <= RTH_CLOSE)
    return add_levels(b, tf)


def add_levels(b, tf):
    g_rth = b[b.rth].groupby("tdate")
    day = pd.DataFrame({"rth_open": g_rth.open.first(), "rth_high": g_rth.high.max(), "rth_low": g_rth.low.min(),
                        "rth_close": g_rth.close.last()})
    day["pdh"], day["pdl"], day["pdc"] = day.rth_high.shift(1), day.rth_low.shift(1), day.rth_close.shift(1)
    rng = day.rth_high - day.rth_low
    day["nr7_prev"] = (rng == rng.rolling(7).min()).shift(1).fillna(False).astype(bool)
    st = np.minimum(day.rth_high - day.rth_open, day.rth_open - day.rth_low)
    day["stretch"] = st.rolling(10).mean().shift(1)
    p = (day.pdh + day.pdl + day.pdc) / 3
    day["piv"], day["r1"], day["s1"] = p, 2 * p - day.pdl, 2 * p - day.pdh
    on = b[~b.rth & ((b.tod >= 18 * 60) | (b.end <= RTH_OPEN))].groupby("tdate")
    day["onh"], day["onl"] = on.high.max(), on.low.min()
    lon = b[(b.tod >= 180) & (b.end <= 480)].groupby("tdate")          # London 3:00-8:00 NY
    day["lonh"], day["lonl"] = lon.high.max(), lon.low.min()
    asia = b[(b.tod >= 1200) | (b.end <= 0)].groupby("tdate")           # Asia 20:00-24:00 NY
    day["asiah"], day["asial"] = asia.high.max(), asia.low.min()
    mid = b[b.tod == 0].groupby("tdate").open.first()
    day["midnight"] = mid
    b = b.join(day, on="tdate")
    # running session values (RTH only, up to and including this bar)
    r = b.rth
    gb = b[r].groupby("tdate")
    b.loc[r, "dhigh"] = gb.high.cummax()
    b.loc[r, "dlow"] = gb.low.cummin()
    tp = (b.high + b.low + b.close) / 3
    b.loc[r, "twap"] = tp[r].groupby(b.tdate[r]).transform(lambda s: s.expanding().mean())
    b.loc[r, "twsd"] = tp[r].groupby(b.tdate[r]).transform(lambda s: s.expanding().std())
    for n in (5, 15, 30, 60):
        if n % tf == 0 and n >= tf:
            first = b[r & (b.end <= RTH_OPEN + n)].groupby("tdate")
            b[f"orh{n}"] = b.tdate.map(first.high.max())
            b[f"orl{n}"] = b.tdate.map(first.low.min())
    return b


# ---------- indicators ----------

def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rma(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def atr(b, n=14):
    pc = b.close.shift(1)
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    return rma(tr, n)


def rsi(s, n=14):
    d = s.diff()
    up, dn = rma(d.clip(lower=0), n), rma(-d.clip(upper=0), n)
    return 100 - 100 / (1 + up / dn)


def supertrend(b, n=10, mult=3.0):
    a = atr(b, n).values
    hl2 = ((b.high + b.low) / 2).values
    c = b.close.values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fu, fl = ub.copy(), lb.copy()
    dirn = np.ones(len(c))
    line = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        fu[i] = ub[i] if (ub[i] < fu[i - 1] or c[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lb[i] if (lb[i] > fl[i - 1] or c[i - 1] < fl[i - 1]) else fl[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if c[i] < fl[i] else 1
        else:
            dirn[i] = 1 if c[i] > fu[i] else -1
        line[i] = fl[i] if dirn[i] == 1 else fu[i]
    return pd.Series(dirn, b.index), pd.Series(line, b.index)


def psar(b, step=0.02, mx=0.2):
    h, l = b.high.values, b.low.values
    n = len(h)
    sar = np.zeros(n)
    up = True
    af, ep, sar[0] = step, h[0], l[0]
    dirn = np.ones(n)
    for i in range(1, n):
        s = sar[i - 1] + af * (ep - sar[i - 1])
        if up:
            s = min(s, l[i - 1], l[i - 2] if i > 1 else l[i - 1])
            if l[i] < s:
                up, s, ep, af = False, ep, l[i], step
            elif h[i] > ep:
                ep, af = h[i], min(af + step, mx)
        else:
            s = max(s, h[i - 1], h[i - 2] if i > 1 else h[i - 1])
            if h[i] > s:
                up, s, ep, af = True, ep, h[i], step
            elif l[i] < ep:
                ep, af = l[i], min(af + step, mx)
        sar[i] = s
        dirn[i] = 1 if up else -1
    return pd.Series(dirn, b.index)


def adx(b, n=14):
    up, dn = b.high.diff(), -b.low.diff()
    pdm = up.where((up > dn) & (up > 0), 0.0)
    mdm = dn.where((dn > up) & (dn > 0), 0.0)
    a = atr(b, n)
    pdi, mdi = 100 * rma(pdm, n) / a, 100 * rma(mdm, n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi)
    return rma(dx, n), pdi, mdi


def cross_up(a, b_):
    return (a > b_) & (a.shift(1) <= (b_.shift(1) if isinstance(b_, pd.Series) else b_))


def cross_dn(a, b_):
    return (a < b_) & (a.shift(1) >= (b_.shift(1) if isinstance(b_, pd.Series) else b_))


# ---------- engine ----------

def simulate(b, sig, cost=1.0):
    """sig: dict with le, se (entry bools), optional xl, xs (exit bools at close), sl_l/tp_l/sl_s/tp_s
    (absolute levels), px_l/px_s (entry price for stop/limit orders, else the close), flip (bool),
    max_day (int), start (minutes, first allowed entry time), end_entry (minutes)."""
    n = len(b)
    idx = np.where(b.rth.values)[0]
    o, h, l, c = (b[k].values for k in ("open", "high", "low", "close"))
    end, td = b.end.values, b.tdate.values
    start = sig.get("start", RTH_OPEN)
    allowed_any = (b.tod.values >= start) & (end <= sig.get("end_entry", LAST_ENTRY))
    if start < RTH_OPEN:  # pre-market entries (London/Asia setups): loop over those bars too
        idx = np.where(((b.tod.values >= start) & (end <= RTH_CLOSE)))[0]
    f = lambda k: (sig[k].reindex(b.index).fillna(False).astype(bool).values if k in sig else np.zeros(n, bool))
    le, se = f("le"), f("se")
    xl, xs = f("xl"), f("xs")
    nanv = lambda k: (sig[k].reindex(b.index).values.astype(float) if k in sig else np.full(n, np.nan))
    sl_l, tp_l, sl_s, tp_s, px_l, px_s = (nanv(k) for k in ("sl_l", "tp_l", "sl_s", "tp_s", "px_l", "px_s"))
    flip, max_day = sig.get("flip", False), sig.get("max_day", 99)
    last = np.zeros(n, bool)  # last regular-session candle of each day (16:00 close, or earlier if missing)
    last[idx] = np.append(td[idx][1:] != td[idx][:-1], True) & (end[idx] > RTH_OPEN)
    trades = []
    pos, entry, sl, tp, eday, count, cur = 0, 0.0, np.nan, np.nan, None, 0, None
    prev_i = None
    for i in idx:
        if td[i] != cur:
            if pos:  # missing close bar: exit at the last close we have
                trades.append((eday, pos, entry, c[prev_i], "gap"))
                pos = 0
            cur, count = td[i], 0
        stopped = False
        if pos:
            if pos == 1:
                if l[i] <= sl:
                    trades.append((eday, 1, entry, min(sl, o[i]), "stop")); pos = 0; stopped = True
                elif h[i] >= tp:
                    trades.append((eday, 1, entry, max(tp, o[i]), "target")); pos = 0; stopped = True
            else:
                if h[i] >= sl:
                    trades.append((eday, -1, entry, max(sl, o[i]), "stop")); pos = 0; stopped = True
                elif l[i] <= tp:
                    trades.append((eday, -1, entry, min(tp, o[i]), "target")); pos = 0; stopped = True
            if pos and (last[i] or (pos == 1 and (xl[i] or (flip and se[i])))
                        or (pos == -1 and (xs[i] or (flip and le[i])))):
                trades.append((eday, pos, entry, c[i], "close" if last[i] else "signal")); pos = 0
        if pos == 0 and not stopped and allowed_any[i] and not last[i] and count < max_day:
            for side, e_, slv, tpv, pxv in ((1, le, sl_l, tp_l, px_l), (-1, se, sl_s, tp_s, px_s)):
                if e_[i]:
                    px = c[i] if np.isnan(pxv[i]) else pxv[i]
                    s_ = slv[i] if not np.isnan(slv[i]) else (-np.inf if side == 1 else np.inf)
                    t_ = tpv[i] if not np.isnan(tpv[i]) else (np.inf if side == 1 else -np.inf)
                    if (side == 1 and not (s_ < px < t_)) or (side == -1 and not (t_ < px < s_)):
                        continue  # stop/target already on the wrong side: skip
                    if not np.isnan(pxv[i]):  # order filled inside this candle: stopped in it only if it
                        # also closed beyond the stop (then it surely crossed the stop after the fill)
                        hit = (c[i] <= s_) if side == 1 else (c[i] >= s_)
                        if hit:
                            trades.append((td[i], side, px, s_, "stop")); count += 1
                            break
                    pos, entry, sl, tp, eday, count = side, px, s_, t_, td[i], count + 1
                    break
        prev_i = i
    tr = pd.DataFrame(trades, columns=["day", "side", "entry", "exit", "why"])
    tr["pts"] = tr.side * (tr.exit - tr.entry) - cost
    tr["pct"] = tr.pts / tr.entry * 100
    return tr


# ---------- strategies ----------
# Each returns the signal dict for bars b (already at the strategy's timeframe).

def rr(sig, side_stop_l, side_stop_s, c, r):
    """Target = r x the stop distance."""
    sig["sl_l"], sig["sl_s"] = side_stop_l, side_stop_s
    sig["tp_l"] = c + r * (c - side_stop_l)
    sig["tp_s"] = c - r * (side_stop_s - c)
    return sig


def first_of_day(mask, b):
    """Only the first True per trade date."""
    m = mask.fillna(False) & b.rth
    return m & (m.groupby(b.tdate).cumsum() == 1)


def s_orb(n, target_r=None, filt=None):
    def f(b):
        hi, lo = b[f"orh{n}"], b[f"orl{n}"]
        after = b.end > RTH_OPEN + n
        le, se = after & (b.close > hi), after & (b.close < lo)
        if filt is not None:
            le, se = le & filt(b), se & filt(b)
        sig = dict(le=le, se=se, sl_l=lo, sl_s=hi, max_day=1)
        if target_r:
            sig["tp_l"], sig["tp_s"] = b.close + target_r * (hi - lo), b.close - target_r * (hi - lo)
        return sig
    return f


def s_orb_fade(n):
    def f(b):
        hi, lo = b[f"orh{n}"], b[f"orl{n}"]
        after = b.end > RTH_OPEN + n
        was_above = (after & (b.close > hi)).groupby(b.tdate).cummax().shift(1).fillna(False).astype(bool)
        was_below = (after & (b.close < lo)).groupby(b.tdate).cummax().shift(1).fillna(False).astype(bool)
        se = after & was_above & (b.close < hi)
        le = after & was_below & (b.close > lo)
        return dict(le=le, se=se, sl_l=b.dlow, sl_s=b.dhigh, tp_l=hi, tp_s=lo, max_day=1)
    return f


def s_tiktok(b):  # the original: 8:30-9:00 range, first close outside, stop other side, 10 points
    pre = b[(b.tod >= 510) & (b.end <= 540)].groupby("tdate")
    hi, lo = b.tdate.map(pre.high.max()), b.tdate.map(pre.low.min())
    return dict(le=b.close > hi, se=b.close < lo, sl_l=lo, sl_s=hi, tp_l=b.close + 10, tp_s=b.close - 10,
                max_day=1, start=540, end_entry=720)


def s_level_break(hk, lk, stop_mult=1.5, r=2.0, start=RTH_OPEN):
    def f(b):
        a = atr(b)
        le, se = cross_up(b.close, b[hk]), cross_dn(b.close, b[lk])
        sig = rr(dict(le=le, se=se, max_day=1, start=start), b.close - stop_mult * a, b.close + stop_mult * a,
                 b.close, r)
        return sig
    return f


def s_level_hold(hk, lk, start=RTH_OPEN):
    """Break a level, stop back at the middle of the range, hold to the close."""
    def f(b):
        mid = (b[hk] + b[lk]) / 2
        return dict(le=cross_up(b.close, b[hk]), se=cross_dn(b.close, b[lk]), sl_l=mid, sl_s=mid, max_day=1,
                    start=start)
    return f


def s_sweep(hk, lk, r=2.0):
    """ICT liquidity sweep: wick through a level and close back inside = trade back the other way."""
    def f(b):
        se = (b.high > b[hk]) & (b.close < b[hk])
        le = (b.low < b[lk]) & (b.close > b[lk])
        return rr(dict(le=le, se=se, max_day=1), b.low, b.high, b.close, r)
    return f


def s_judas(b):
    """ICT Judas swing / Power of 3: early move against the day below/above the midnight open, then a close
    back through the midnight open between 9:30 and 11:00 = trade in that direction. Stop at the day's
    extreme, target 2R."""
    m = b.midnight
    le = (b.dlow < m) & cross_up(b.close, m)
    se = (b.dhigh > m) & cross_dn(b.close, m)
    return rr(dict(le=le, se=se, max_day=1, end_entry=660), b.dlow, b.dhigh, b.close, 2.0)


def s_silver_bullet(b):
    """ICT Silver Bullet: 10:00-11:00 NY. First fair value gap formed inside the window
    (candle 3 low above candle 1 high = bullish). Limit order at the gap's edge; stop beyond candle 1;
    target 2R. Entry must happen before 11:00."""
    h, l, tod, end = b.high.values, b.low.values, b.tod.values, b.end.values
    td = b.tdate.values
    n = len(b)
    px_l, px_s = np.full(n, np.nan), np.full(n, np.nan)
    sl_l, sl_s = np.full(n, np.nan), np.full(n, np.nan)
    pend, done_day = None, None
    for i in range(2, n):
        if not (600 <= tod[i] and end[i] <= 660):
            pend = None if end[i] > 660 else pend
            continue
        if td[i] == done_day:
            continue
        if pend and pend[0] == td[i]:
            _, side, edge, stop = pend
            if (side == 1 and l[i] <= edge) or (side == -1 and h[i] >= edge):
                if side == 1:
                    px_l[i], sl_l[i] = edge, stop
                else:
                    px_s[i], sl_s[i] = edge, stop
                done_day, pend = td[i], None
                continue
        if pend is None and tod[i - 2] >= 600 and td[i - 2] == td[i]:
            if l[i] > h[i - 2]:
                pend = (td[i], 1, h[i - 2], l[i - 2])
            elif h[i] < l[i - 2]:
                pend = (td[i], -1, l[i - 2], h[i - 2])
    idx = b.index
    px_l, px_s, sl_l, sl_s = (pd.Series(x, idx) for x in (px_l, px_s, sl_l, sl_s))
    return dict(le=px_l.notna(), se=px_s.notna(), px_l=px_l, px_s=px_s, sl_l=sl_l, sl_s=sl_s,
                tp_l=px_l + 2 * (px_l - sl_l), tp_s=px_s - 2 * (sl_s - px_s), max_day=1, start=600, end_entry=660)


def s_crt(b):
    """Candle Range Theory: candle 2 wicks above candle 1's high and closes back inside -> short to candle 1's
    low, stop above candle 2. Mirror for longs."""
    ph, pl = b.high.shift(1), b.low.shift(1)
    se = (b.high > ph) & (b.close < ph) & (b.close > pl)
    le = (b.low < pl) & (b.close > pl) & (b.close < ph)
    return dict(le=le, se=se, sl_l=b.low, sl_s=b.high, tp_l=ph, tp_s=pl)


def s_gap_fade(minpct=0.25):
    def f(b):
        gap = (b.rth_open - b.pdc) / b.pdc * 100
        first = b.rth & (b.tod == RTH_OPEN)
        le = first & (gap < -minpct) & (b.close < b.pdc)
        se = first & (gap > minpct) & (b.close > b.pdc)
        g = (b.rth_open - b.pdc).abs()
        return dict(le=le, se=se, tp_l=b.pdc, tp_s=b.pdc, sl_l=b.close - g, sl_s=b.close + g, max_day=1)
    return f


def s_gap_go(minpct=0.25):
    def f(b):
        gap = (b.rth_open - b.pdc) / b.pdc * 100
        first = b.rth & (b.tod == RTH_OPEN)
        le = first & (gap > minpct) & (b.close > b.open)
        se = first & (gap < -minpct) & (b.close < b.open)
        return dict(le=le, se=se, sl_l=b.dlow, sl_s=b.dhigh, max_day=1)
    return f


def s_oops(b):
    """Larry Williams 'Oops': open gaps below yesterday's low, then trades back up to it -> buy at that level,
    stop at the day's low, out at the close. Mirror for gaps above yesterday's high."""
    le = (b.rth_open < b.pdl) & (b.high >= b.pdl)
    se = (b.rth_open > b.pdh) & (b.low <= b.pdh)
    return dict(le=le, se=se, px_l=b.pdl.where(b.open < b.pdl, b.open), px_s=b.pdh.where(b.open > b.pdh, b.open),
                sl_l=b.dlow, sl_s=b.dhigh, max_day=1)


def s_pivot_bounce(b):
    le = (b.low <= b.s1) & (b.close > b.s1)
    se = (b.high >= b.r1) & (b.close < b.r1)
    return dict(le=le, se=se, sl_l=b.low, sl_s=b.high, tp_l=b.piv, tp_s=b.piv, max_day=1)


def s_pivot_break(b):
    return dict(le=cross_up(b.close, b.r1), se=cross_dn(b.close, b.s1), sl_l=b.piv, sl_s=b.piv, max_day=1)


def s_crabel(b):
    """Toby Crabel stretch breakout: buy stop at open + stretch, sell stop at open - stretch
    (stretch = 10-day average of the smaller of open-high / open-low). Stop at the open, out at the close."""
    up, dn = b.rth_open + b.stretch, b.rth_open - b.stretch
    le, se = b.high >= up, b.low <= dn
    return dict(le=le, se=se, px_l=np.maximum(up, b.open), px_s=np.minimum(dn, b.open),
                sl_l=b.rth_open, sl_s=b.rth_open, max_day=1)


def s_first_candle(target_r=10):
    def f(b):
        first = b.rth & (b.tod == RTH_OPEN)
        return rr(dict(le=first & (b.close > b.open), se=first & (b.close < b.open), max_day=1),
                  b.low, b.high, b.close, target_r)
    return f


def s_time_bias(at, mode):
    """mode 'follow' = trade the direction since the open, 'fade' = against it, held to the close."""
    def f(b):
        t = b.rth & (b.end == at)
        up = b.close > b.rth_open
        if mode == "follow":
            return dict(le=t & up, se=t & ~up, max_day=1)
        return dict(le=t & ~up, se=t & up, max_day=1)
    return f


def s_buy_open(b):
    first = b.rth & (b.tod == RTH_OPEN)
    return dict(le=first, px_l=b.open.where(first), max_day=1)


def s_twap_trend(b):
    """Close above the session average (VWAP stand-in) = long, below = short, flip; out at 16:00."""
    return dict(le=cross_up(b.close, b.twap), se=cross_dn(b.close, b.twap), flip=True)


def s_twap_revert(k=2.0):
    def f(b):
        up, dn = b.twap + k * b.twsd, b.twap - k * b.twsd
        le, se = cross_dn(b.close, dn), cross_up(b.close, up)
        return dict(le=le, se=se, xl=b.close >= b.twap, xs=b.close <= b.twap,
                    sl_l=b.close - b.twsd, sl_s=b.close + b.twsd, end_entry=900)
    return f


def s_ema_cross(fa, sl_):
    def f(b):
        e1, e2 = ema(b.close, fa), ema(b.close, sl_)
        return dict(le=cross_up(e1, e2), se=cross_dn(e1, e2), flip=True)
    return f


def s_ema_twap(b):
    """9/21 EMA cross in the direction of the session average (EMA + VWAP scripts), stop 1.5 ATR, flip."""
    e1, e2, a = ema(b.close, 9), ema(b.close, 21), atr(b)
    le, se = cross_up(e1, e2) & (b.close > b.twap), cross_dn(e1, e2) & (b.close < b.twap)
    return dict(le=le, se=se, xl=cross_dn(e1, e2), xs=cross_up(e1, e2), sl_l=b.close - 1.5 * a,
                sl_s=b.close + 1.5 * a)


def s_macd(b):
    m = ema(b.close, 12) - ema(b.close, 26)
    s = ema(m, 9)
    return dict(le=cross_up(m, s), se=cross_dn(m, s), flip=True)


def s_macd200(b):
    """YouTube 'MACD + 200 EMA': cross below zero in an uptrend (above the 200 EMA) = long. Stop at the 200 EMA,
    target 1.5R. Mirror for shorts."""
    m = ema(b.close, 12) - ema(b.close, 26)
    s, e = ema(m, 9), ema(b.close, 200)
    le = cross_up(m, s) & (m < 0) & (b.close > e)
    se = cross_dn(m, s) & (m > 0) & (b.close < e)
    return rr(dict(le=le, se=se), e, e, b.close, 1.5)


def s_supertrend(n, mult, rsi_filter=False):
    def f(b):
        d, line = supertrend(b, n, mult)
        le, se = (d == 1) & (d.shift(1) == -1), (d == -1) & (d.shift(1) == 1)
        if rsi_filter:  # Supertrend + RSI scalper: trend side + RSI > 60 / < 40, stop at the line, 1:1
            r = rsi(b.close)
            le, se = (d == 1) & cross_up(r, 60), (d == -1) & cross_dn(r, 40)
            return rr(dict(le=le, se=se), line, line, b.close, 1.0)
        return dict(le=le, se=se, flip=True)
    return f


def s_psar(b):
    d = psar(b)
    return dict(le=(d == 1) & (d.shift(1) == -1), se=(d == -1) & (d.shift(1) == 1), flip=True)


def s_donchian(n=20, x=10):
    def f(b):
        hh, ll = b.high.rolling(n).max().shift(1), b.low.rolling(n).min().shift(1)
        xh, xl_ = b.high.rolling(x).max().shift(1), b.low.rolling(x).min().shift(1)
        return dict(le=b.close > hh, se=b.close < ll, xl=b.close < xl_, xs=b.close > xh)
    return f


def s_bb_squeeze(b):
    m, sd = b.close.rolling(20).mean(), b.close.rolling(20).std()
    bw = 4 * sd / m
    squeeze = (bw <= bw.rolling(120).min() * 1.05).astype(float).rolling(5).max().fillna(0).astype(bool)
    return dict(le=squeeze & (b.close > m + 2 * sd), se=squeeze & (b.close < m - 2 * sd),
                xl=b.close < m, xs=b.close > m)


def s_bb_fade(b):
    m, sd, a = b.close.rolling(20).mean(), b.close.rolling(20).std(), atr(b)
    return dict(le=b.close < m - 2 * sd, se=b.close > m + 2 * sd, xl=b.close >= m, xs=b.close <= m,
                sl_l=b.close - 1.5 * a, sl_s=b.close + 1.5 * a)


def s_keltner(b):
    e, a = ema(b.close, 20), atr(b, 20)
    return dict(le=b.close > e + 2 * a, se=b.close < e - 2 * a, xl=b.close < e, xs=b.close > e)


def s_heikin(b):
    hc = (b.open + b.high + b.low + b.close) / 4
    ho = hc.copy()
    v, oo = hc.values, b.open.values
    hov = np.empty(len(v))
    hov[0] = (oo[0] + b.close.values[0]) / 2
    for i in range(1, len(v)):
        hov[i] = (hov[i - 1] + v[i - 1]) / 2
    ho = pd.Series(hov, b.index)
    green = hc > ho
    nolow = np.minimum(ho, hc) <= b.low + 1e-9
    nohigh = np.maximum(ho, hc) >= b.high - 1e-9
    le = green & ~green.shift(1, fill_value=False) & nolow
    se = ~green & green.shift(1, fill_value=False) & nohigh
    return dict(le=le, se=se, xl=~green, xs=green)


def s_adx(b):
    a, p, m = adx(b)
    return dict(le=(a > 25) & cross_up(p, m), se=(a > 25) & cross_dn(p, m), xl=cross_dn(p, m), xs=cross_up(p, m))


def s_ichimoku(b):
    tk = (b.high.rolling(9).max() + b.low.rolling(9).min()) / 2
    kj = (b.high.rolling(26).max() + b.low.rolling(26).min()) / 2
    sa = ((tk + kj) / 2).shift(26)
    sb = ((b.high.rolling(52).max() + b.low.rolling(52).min()) / 2).shift(26)
    top, bot = np.maximum(sa, sb), np.minimum(sa, sb)
    return dict(le=cross_up(tk, kj) & (b.close > top), se=cross_dn(tk, kj) & (b.close < bot),
                xl=cross_dn(tk, kj), xs=cross_up(tk, kj))


def s_rsi_rev(n=14, lo=30, hi=70):
    def f(b):
        r, a = rsi(b.close, n), atr(b)
        return dict(le=cross_up(r, lo), se=cross_dn(r, hi), xl=r > 50, xs=r < 50,
                    sl_l=b.close - 1.5 * a, sl_s=b.close + 1.5 * a)
    return f


def s_rsi2(b):
    """Larry Connors RSI(2): pullback in the trend (200 EMA), exit on a close back over the 5 EMA."""
    r, e200, e5 = rsi(b.close, 2), ema(b.close, 200), ema(b.close, 5)
    return dict(le=(r < 10) & (b.close > e200), se=(r > 90) & (b.close < e200), xl=b.close > e5, xs=b.close < e5)


def s_stoch(b):
    ll, hh = b.low.rolling(14).min(), b.high.rolling(14).max()
    k = 100 * (b.close - ll) / (hh - ll)
    k = k.rolling(3).mean()
    d = k.rolling(3).mean()
    a = atr(b)
    return dict(le=cross_up(k, d) & (k < 20), se=cross_dn(k, d) & (k > 80), xl=k > 80, xs=k < 20,
                sl_l=b.close - 1.5 * a, sl_s=b.close + 1.5 * a)


def s_willr(b):
    hh, ll = b.high.rolling(14).max(), b.low.rolling(14).min()
    w = -100 * (hh - b.close) / (hh - ll)
    a = atr(b)
    return dict(le=cross_up(w, -80), se=cross_dn(w, -20), xl=w > -20, xs=w < -80,
                sl_l=b.close - 1.5 * a, sl_s=b.close + 1.5 * a)


def s_cci(b):
    tp = (b.high + b.low + b.close) / 3
    m = tp.rolling(20).mean()
    md = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    cc = (tp - m) / (0.015 * md)
    return dict(le=cross_up(cc, -100), se=cross_dn(cc, 100), flip=True)


def s_engulf(b):
    po, pc = b.open.shift(1), b.close.shift(1)
    le = (pc < po) & (b.close > b.open) & (b.close > po) & (b.open <= pc)
    se = (pc > po) & (b.close < b.open) & (b.close < po) & (b.open >= pc)
    return rr(dict(le=le, se=se), np.minimum(b.low, b.low.shift(1)), np.maximum(b.high, b.high.shift(1)),
              b.close, 2.0)


def s_inside(b):
    ins = (b.high.shift(1) < b.high.shift(2)) & (b.low.shift(1) > b.low.shift(2))
    le, se = ins & (b.close > b.high.shift(1)), ins & (b.close < b.low.shift(1))
    return rr(dict(le=le, se=se), b.low.shift(1), b.high.shift(1), b.close, 2.0)


def s_pinbar(b):
    body = (b.close - b.open).abs()
    rngb = b.high - b.low
    lw, uw = np.minimum(b.open, b.close) - b.low, b.high - np.maximum(b.open, b.close)
    le = (lw > 2 * body) & (lw > 0.6 * rngb) & (b.low <= b.low.rolling(20).min())
    se = (uw > 2 * body) & (uw > 0.6 * rngb) & (b.high >= b.high.rolling(20).max())
    return rr(dict(le=le, se=se), b.low, b.high, b.close, 2.0)


def s_turtle_soup(b):
    """Linda Raschke: new 20-bar low that closes back above the old low = long (stop at the new low), 2R."""
    ll, hh = b.low.rolling(20).min().shift(1), b.high.rolling(20).max().shift(1)
    le = (b.low < ll) & (b.close > ll)
    se = (b.high > hh) & (b.close < hh)
    return rr(dict(le=le, se=se), b.low, b.high, b.close, 2.0)


def s_scalp_ema(b):
    """1-minute EMA pullback scalp: trend = 20 EMA over 50 EMA; candle dips to the 20 EMA and closes above it.
    Stop and target 1 ATR."""
    e20, e50, a = ema(b.close, 20), ema(b.close, 50), atr(b)
    le = (e20 > e50) & (b.low <= e20) & (b.close > e20)
    se = (e20 < e50) & (b.high >= e20) & (b.close < e20)
    return dict(le=le, se=se, sl_l=b.close - a, tp_l=b.close + a, sl_s=b.close + a, tp_s=b.close - a)


def s_noise(b):
    """Reference: the refined noise band from round 4 (long only, out back through the open) on this engine."""
    t = b[b.rth].copy()
    t["slot"] = t.tod
    c = t.pivot_table(index="tdate", columns="slot", values="close")
    c = c[c.notna().all(axis=1)]  # full sessions only, as in round 4
    op = b.rth_open.groupby(b.tdate).first().reindex(c.index)
    pc = b.pdc.groupby(b.tdate).first().reindex(c.index)
    sig = (c.div(op, axis=0) - 1).abs().rolling(14).mean().shift(1)
    ub = sig.add(1).mul(np.maximum(op, pc), axis=0).stack().rename("ub")
    j = b.join(ub, on=["tdate", "tod"])
    check = b.rth & (b.end % 30 == 0) & (b.end >= 600)
    le = check & (b.close > j.ub)
    xl = check & (b.close < b.rth_open)
    return dict(le=le, xl=xl)


S = {}


def reg(name, source, tfs, fn):
    for tf in tfs:
        S[(name, tf)] = (source, fn)


# --- the user's starting point and earlier rounds ---
reg("TikTok 8:30-9:00 range, 10-pt target", "the TikTok video", [15], s_tiktok)
reg("noise band, refined (round 4)", "Zarattini/Barbon/Aziz 2024", [15], s_noise)
# --- opening range family ---
for n in (5, 15, 30):
    reg(f"ORB {n}m, hold to close", "Crabel 1990; Zarattini & Aziz 2023; YouTube", [n if n < 30 else 15], s_orb(n))
    reg(f"ORB {n}m, target 2x range", "YouTube ORB videos", [n if n < 30 else 15], s_orb(n, 2))
    reg(f"ORB {n}m fade (failed breakout)", "YouTube / Reddit", [n if n < 30 else 15], s_orb_fade(n))
reg("Initial balance (60m) breakout, target 1x IB", "Market Profile / edgeful", [15], s_orb(60, 1))
reg("Initial balance (60m) breakout, hold", "Market Profile", [15], s_orb(60))
reg("ORB 30m after an NR7 day (Crabel)", "Crabel 1990", [15], s_orb(30, filt=lambda b: b.nr7_prev))
reg("Crabel stretch breakout", "Crabel 1990", [1, 5], s_crabel)
reg("First candle direction, 10R", "Zarattini & Aziz 2023 (5m ORB)", [1, 5, 15], s_first_candle(10))
# --- levels ---
reg("Prior-day high/low breakout, ATR stop, 2R", "futures guides", [5, 15], s_level_break("pdh", "pdl"))
reg("Prior-day high/low sweep & reclaim (ICT)", "ICT liquidity sweep", [5, 15], s_sweep("pdh", "pdl"))
reg("Overnight high/low breakout, hold", "futures guides", [5, 15], s_level_hold("onh", "onl"))
reg("Overnight high/low sweep & reclaim (ICT)", "ICT liquidity sweep", [5, 15], s_sweep("onh", "onl"))
reg("London range (3-8am NY) breakout, hold", "London breakout", [15], s_level_hold("lonh", "lonl", start=480))
reg("Asia range (20-24h NY) breakout, hold", "Asian range breakout", [15], s_level_hold("asiah", "asial", start=180))
reg("Floor pivot S1/R1 bounce to pivot", "pivot point traders", [5, 15], s_pivot_bounce)
reg("Floor pivot R1/S1 breakout, hold", "pivot point traders", [5, 15], s_pivot_break)
# --- ICT / smart money ---
reg("ICT Silver Bullet (10-11am FVG, 2R)", "ICT / TikTok / YouTube", [1, 5], s_silver_bullet)
reg("ICT Judas swing / Power of 3 (midnight open)", "ICT", [5, 15], s_judas)
reg("Candle Range Theory (CRT)", "TikTok / YouTube CRT", [15, 60], s_crt)
# --- gaps and time of day ---
reg("Gap fade to yesterday's close (>0.25%)", "John Carter, Forex Factory", [5, 15], s_gap_fade())
reg("Gap and go (>0.25%)", "momentum traders", [5, 15], s_gap_go())
reg("Larry Williams Oops", "Larry Williams", [1, 5], s_oops)
reg("Power hour: follow the day's direction from 15:00", "YouTube", [15], s_time_bias(900, "follow"))
reg("Lunch reversal: fade the morning from 12:00", "YouTube", [15], s_time_bias(720, "fade"))
reg("Morning momentum: follow first hour from 10:30", "intraday momentum", [15], s_time_bias(630, "follow"))
reg("Buy every open, sell every close (baseline)", "benchmark", [15], s_buy_open)
# --- VWAP (TWAP stand-in) ---
reg("Session average (VWAP) cross, flip", "VWAP traders", [5, 15], s_twap_trend)
reg("Session average (VWAP) 2 SD reversion", "VWAP mean reversion", [5, 15], s_twap_revert())
reg("9/21 EMA cross with VWAP filter", "TradingView EMA+VWAP scripts", [5, 15], s_ema_twap)
# --- indicator systems ---
for tfs in ([5, 15, 60],):
    reg("EMA 9/21 cross", "YouTube / Algotest", tfs, s_ema_cross(9, 21))
    reg("EMA 20/50 cross", "trend followers", tfs, s_ema_cross(20, 50))
    reg("MACD signal cross", "classic", tfs, s_macd)
    reg("MACD + 200 EMA (1.5R)", "YouTube viral", tfs, s_macd200)
    reg("Supertrend 10/3 flip", "TradingView", tfs, s_supertrend(10, 3))
    reg("Supertrend 10/1 + RSI 60/40 (1:1)", "TradingView scalper", tfs, s_supertrend(10, 1, True))
    reg("Parabolic SAR flip", "Wilder", tfs, s_psar)
    reg("Donchian 20/10 breakout", "Turtle rules intraday", tfs, s_donchian())
    reg("Bollinger squeeze breakout", "John Carter squeeze", tfs, s_bb_squeeze)
    reg("Bollinger 2SD fade to middle", "mean reversion", tfs, s_bb_fade)
    reg("Keltner 2 ATR breakout", "Keltner", tfs, s_keltner)
    reg("Heikin Ashi color change", "YouTube", tfs, s_heikin)
    reg("ADX > 25 + DI cross", "Wilder", tfs, s_adx)
    reg("Ichimoku TK cross with cloud", "Ichimoku", tfs, s_ichimoku)
    reg("RSI 14 30/70 reversal", "classic", tfs, s_rsi_rev())
    reg("Connors RSI(2) with 200 EMA", "Larry Connors", tfs, s_rsi2)
    reg("Stochastic 20/80 cross", "Lane", tfs, s_stoch)
    reg("Williams %R reversal", "Larry Williams", tfs, s_willr)
    reg("CCI +-100 flip", "Lambert", tfs, s_cci)
    reg("Engulfing candle, 2R", "candlestick patterns", tfs, s_engulf)
    reg("Inside bar breakout, 2R", "price action", tfs, s_inside)
    reg("Pin bar at 20-bar extreme, 2R", "price action", tfs, s_pinbar)
    reg("Turtle soup (false breakout), 2R", "Linda Raschke", tfs, s_turtle_soup)
reg("EMA pullback scalp (1 ATR / 1 ATR)", "scalping videos", [1, 5], s_scalp_ema)
reg("RSI 14 30/70 reversal", "classic", [1], s_rsi_rev())
reg("EMA 9/21 cross", "YouTube / Algotest", [1], s_ema_cross(9, 21))


def stats(tr, holdout="2024-01-01"):
    if len(tr) < 5:
        return dict(trades=len(tr))
    p, d = tr.pct, pd.to_datetime(tr.day)
    pf = lambda x: round(x[x > 0].sum() / -x[x <= 0].sum(), 2) if (x <= 0).any() else np.inf
    dev, ho = p[d < holdout], p[d >= holdout]
    yr = p.groupby(d.dt.year).sum()
    eq = p.cumsum()
    yrs = (d.max() - d.min()).days / 365.25
    return dict(trades=len(tr), per_yr=round(len(tr) / yrs), win=round(100 * (p > 0).mean(), 1),
                avg_pts=round(tr.pts.mean(), 2), avg_pct=round(p.mean(), 3), pf=pf(p), dev_pf=pf(dev),
                hold_pf=pf(ho), pos_years=f"{int((yr > 0).sum())}/{len(yr)}", pct_per_yr=round(p.sum() / yrs, 1),
                max_dd_pct=round((eq - eq.cummax()).min(), 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars1", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m1 = pd.read_pickle(a.bars1)
    rows = []
    for tf in sorted({k[1] for k in S}):
        b = make_bars(m1, tf)
        print("timeframe", tf, len(b), flush=True)
        for (name, t), (src, fn) in S.items():
            if t != tf or (a.only and a.only not in name):
                continue
            try:
                tr = simulate(b, fn(b))
            except Exception as e:  # keep going, report the failure
                print("FAILED", name, tf, repr(e), flush=True)
                continue
            st = stats(tr)
            rows.append(dict(strategy=name, tf=f"{tf}m", source=src, **st))
            print(rows[-1], flush=True)
            tr.to_csv(os.path.join(a.out, f"{name[:40].replace('/', '-').replace(' ', '_')}_{tf}m.csv"), index=False)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(a.out, "library_results.csv"), index=False)
