"""Position-size plan: monthly stake = leverage x balance at the start of the month (gains reinvested)."""
import datetime as dt
import json

from desk.plan import stake_for

PLAN = {"phases": [{"name": "practice", "from": "2026-10-12", "account": "practice", "stake_eur": 1000},
                   {"name": "real 1x", "from": "2027-01-04", "account": "real", "leverage": 1.0, "start_balance": 2000},
                   {"name": "real 1.5x", "from": "2027-04-01", "account": "real", "leverage": 1.5}]}


def write(tmp_path, trades=(), snaps=()):
    (tmp_path / "plan.json").write_text(json.dumps(PLAN))
    (tmp_path / "trades.csv").write_text("date,strategy,account,result_eur\n" + "".join(
        f"{d},ibs,{a},{x}\n" for d, a, x in trades))
    (tmp_path / "account.csv").write_text("date,account,balance_eur,notes\n" + "".join(
        f"{d},{a},{b},\n" for d, a, b in snaps))


def test_fixed_practice_then_compounding_real(tmp_path):
    write(tmp_path, trades=[("2026-10-14", "practice", 50), ("2027-01-10", "real", 120),
                            ("2027-02-03", "real", 80), ("2027-03-31", "real", -30)])
    assert stake_for(dt.date(2026, 10, 9), tmp_path)[:1] == (1000.0,)            # no phase yet: fallback
    assert stake_for(dt.date(2026, 11, 2), tmp_path)[0] == 1000                   # practice: fixed
    assert stake_for(dt.date(2027, 1, 20), tmp_path)[0] == 2000                   # 1x, start balance
    assert stake_for(dt.date(2027, 2, 1), tmp_path)[0] == 2120                    # January gain reinvested
    assert stake_for(dt.date(2027, 2, 25), tmp_path)[0] == 2120                   # only changes on the 1st
    s, why, stop = stake_for(dt.date(2027, 4, 1), tmp_path)
    assert s == 3250 and not stop and "1.5 x" in why   # 1.5 x 2170 = 3255, rounded down to EUR 10


def test_screenshot_balance_overrides_and_stop_rule(tmp_path):
    write(tmp_path, trades=[("2027-01-10", "real", 600), ("2027-02-20", "real", -100)],
          snaps=[("2027-02-10", "real", 2550)])                                  # fees etc.: real balance wins
    assert stake_for(dt.date(2027, 3, 1), tmp_path)[0] == 2450                    # 2550 - 100
    write(tmp_path, trades=[("2027-01-10", "real", 600), ("2027-02-20", "real", -700)])
    s, why, stop = stake_for(dt.date(2027, 3, 1), tmp_path)                       # 2600 high, now 1900 (-27%)
    assert s == 1900 and stop
