# Day desk bot

Phone alerts and demo-account orders for a few Nasdaq-100 day-trading rules (15-minute candles), run by GitHub
Actions: `desk/bot.py` (`.github/workflows/daybot.yml`), paper logs (`paper.yml`) and replays of past days (`replay.yml`).

This repo only runs the code. It is copied from a private working repo with `tools/publish_bot.sh`, so changes made
here are overwritten. The journal (plan, balances, trades) stays private and is read at run time with a token.

Tests: `python -m pytest -q`. Not investment advice; the order code talks to demo accounts only.
