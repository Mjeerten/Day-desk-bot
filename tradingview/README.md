# TradingView indicator: noise band + overnight sweep

`noise_band_sweep.pine` draws both strategies from `ORB_BACKTEST.md` (rounds 4-5) on a **15-minute** Nasdaq-100
chart (NQ1!, MNQ1! or a US100 / NAS100 CFD) and fires alerts.

## Install
1. TradingView -> open the chart -> set the timeframe to **15m** (keep extended hours on for futures).
2. Bottom panel -> **Pine Editor** -> delete what is there -> paste the whole file -> **Save** -> **Add to chart**.
3. Indicator settings (gear icon): set your account size and leverage. The box top-right then shows the
   position size per trade.
4. Alerts: Alerts (clock icon) -> Create -> Condition: "Noise band + Overnight sweep (NQ)" ->
   **"Any alert() function call"** -> notifications: app push. One alert covers every buy, sell, stop and target
   message, with the prices in the text.

## What you see
- Grey line: today's open (9:30 New York = 15:30 Amsterdam).
- Orange step line: the noise line. A 15-minute close above it at a :00 or :30 = NOISE BUY (green label).
  Green background = in the trade. NOISE SELL (teal label) when a :00/:30 close falls below the grey line, or at 22:00.
- Purple lines: overnight high and low (00:00-15:30 Amsterdam). A candle that pokes through one and closes back
  inside = SWEEP SHORT (pink) or SWEEP BUY (blue), with the stop (red line) and target (green line).
- Box top-right: what to do now, and the paper results of both strategies over the chart's history.

The chart-history results ignore costs. They should roughly match the backtest: noise band ~59% winners, sweep
~38% winners.
