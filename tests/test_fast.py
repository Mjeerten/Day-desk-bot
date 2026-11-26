import datetime as dt

from desk import fast
from desk.bot import NY


def ny(y, mo, d, h, mi, s=0):
    return dt.datetime(y, mo, d, h, mi, s, tzinfo=NY)


def test_should_start_only_weekday_session_window():
    assert fast.should_start(ny(2026, 11, 30, 9, 16))          # Monday 9:16
    assert fast.should_start(ny(2026, 11, 30, 16, 31))
    assert not fast.should_start(ny(2026, 11, 30, 9, 5))       # before 9:10
    assert not fast.should_start(ny(2026, 11, 30, 16, 32))     # after the day's last work
    assert not fast.should_start(ny(2026, 11, 28, 10, 0))      # Saturday
    assert not fast.should_start(ny(2026, 11, 26, 10, 0))      # Thanksgiving


def test_next_tick_is_the_next_5_minute_boundary():
    assert fast.next_tick(ny(2026, 11, 30, 9, 44, 59)) == ny(2026, 11, 30, 9, 45)
    assert fast.next_tick(ny(2026, 11, 30, 9, 45, 0)) == ny(2026, 11, 30, 9, 50)
    assert fast.next_tick(ny(2026, 11, 30, 9, 58, 30)) == ny(2026, 11, 30, 10, 0)


def test_part_end_one_hour_capped_at_1632():
    assert fast.part_end(ny(2026, 11, 30, 9, 17)) == ny(2026, 11, 30, 10, 17)
    assert fast.part_end(ny(2026, 11, 30, 16, 0)) == ny(2026, 11, 30, 16, 32)


def test_wait_for_data_until_yahoo_shows_the_new_minute():
    t = ny(2026, 11, 30, 10, 0)
    clock = [t + dt.timedelta(seconds=8)]
    seen = iter([ny(2026, 11, 30, 9, 59), ny(2026, 11, 30, 9, 59), ny(2026, 11, 30, 10, 0)])

    def sleep(s):
        clock[0] += dt.timedelta(seconds=s)

    waited = fast.wait_for_data(t, get=lambda: next(seen), sleep=sleep, clock=lambda: clock[0])
    assert waited == 18   # 8 s settle + two 5 s polls


def test_wait_for_data_gives_up_after_WAIT():
    t = ny(2026, 11, 30, 10, 0)
    clock = [t]

    def sleep(s):
        clock[0] += dt.timedelta(seconds=s)

    def boom():
        raise RuntimeError("Yahoo down")

    assert fast.wait_for_data(t, get=boom, sleep=sleep, clock=lambda: clock[0]) >= fast.WAIT


def test_loop_runs_at_every_boundary_and_stops_at_the_end():
    clock = [ny(2026, 11, 30, 9, 41, 30)]
    runs = []

    def sleep(s):
        clock[0] += dt.timedelta(seconds=s)

    def bot():
        runs.append(clock[0])
        return False

    out = fast.loop(ny(2026, 11, 30, 10, 0), sleep=sleep, clock=lambda: clock[0], bot=bot, wait=lambda t: 0.0)
    assert out == "done"
    assert [f"{r:%H:%M:%S}" for r in runs] == ["09:41:30", "09:45:08", "09:50:08", "09:55:08", "10:00:08"]


def test_loop_ends_the_part_when_ig_blocks_the_server():
    clock = [ny(2026, 11, 30, 9, 41)]
    calls = iter([False, True, False])

    def sleep(s):
        clock[0] += dt.timedelta(seconds=s)

    assert fast.loop(ny(2026, 11, 30, 11, 0), sleep=sleep, clock=lambda: clock[0], bot=lambda: next(calls),
                     wait=lambda t: 0.0) == "blocked"


def test_loop_does_nothing_after_its_end():
    assert fast.loop(ny(2026, 11, 30, 9, 0), clock=lambda: ny(2026, 11, 30, 9, 5), bot=lambda: 1 / 0) == "done"
