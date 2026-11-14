"""Daily 23:00 Amsterdam result message (desk/autotrade.report), without network."""
import pandas as pd

from desk import autotrade as at


class Fake:
    eq = 10050.0

    def login(self):
        return {"accountInfo": {"balance": self.eq - 20, "profitLoss": 20}}


def setup(monkeypatch, tmp_path):
    import desk.ig as ig
    monkeypatch.setattr(ig, "IG", Fake)
    monkeypatch.setattr(at, "STATE", str(tmp_path / "ig-{mode}.json"))
    monkeypatch.setattr(at, "enabled", True)
    monkeypatch.setattr(at, "mode", lambda b="ig": "demo" if b == "ig" else "off")
    sent = []
    monkeypatch.setattr(at, "notify", lambda title, body, quiet=False, tags=(), prio=3: sent.append((title, body)))
    return sent


def test_report_once_at_23_with_pct(monkeypatch, tmp_path):
    sent = setup(monkeypatch, tmp_path)
    st = {"day": {"261111": 10000.0}, "trades": {
        "N1": dict(strat="N", status="closed", closed="2026-11-11 11:00:00-05:00", eur=60.0, day="261111"),
        "S1": dict(strat="S", status="closed_by_ig", closed="2026-11-11 12:00:00-05:00", eur=-25.0, day="261111"),
        "I1": dict(strat="I", status="open", day="261111")}}
    at.save(st, "demo")
    at.report(pd.Timestamp("2026-11-11 22:46", tz="Europe/Amsterdam"))
    assert not sent                                   # before 23:00: nothing
    at.report(pd.Timestamp("2026-11-11 23:01", tz="Europe/Amsterdam"))
    at.report(pd.Timestamp("2026-11-11 23:16", tz="Europe/Amsterdam"))
    assert len(sent) == 1                             # once a day
    title, body = sent[0]
    assert title.startswith("+EUR 50.00 (+0.50%) today")
    assert "2 trades (1 won, 1 lost), +EUR 35.00" in body and "Still open: IBS" in body
    Fake.eq = 9949.5                                  # next day: measured from yesterday's report
    at.report(pd.Timestamp("2026-11-12 23:01", tz="Europe/Amsterdam"))
    assert sent[1][0].startswith("-EUR 100.50 (-1.00%) today")
    at.report(pd.Timestamp("2026-11-14 23:01", tz="Europe/Amsterdam"))   # Saturday: none
    assert len(sent) == 2


def test_report_uses_ig_history_with_names(monkeypatch, tmp_path):
    sent = setup(monkeypatch, tmp_path)

    class WithHistory(Fake):
        eq = 10320.88

        def closed_since(self, since):
            assert since == "2026-11-10T23:00:00"          # 00:00 Amsterdam
            return [dict(open=28131.5, eur=-32.28), dict(open=28126.8, eur=-55.02), dict(open=28322.4, eur=26.96),
                    dict(open=28450.5, eur=-15.88)]
    import desk.ig as ig
    monkeypatch.setattr(ig, "IG", WithHistory)
    at.save({"day": {"261111": 10397.10}, "trades": {
        "S": dict(strat="S", status="closed_by_ig", level=28131.5, day="261111"),
        "P": dict(strat="P", status="closed_by_ig", level=28126.8, day="261111"),
        "O": dict(strat="O", status="closed", level=28322.4, eur=26.96, closed="2026-11-11 15:01:00-05:00", day="261111"),
        "N": dict(strat="N", status="closed_by_ig", level=28450.5, day="261111")}}, "demo")
    at.report(pd.Timestamp("2026-11-11 23:01", tz="Europe/Amsterdam"))
    title, body = sent[0]
    assert title.startswith("-EUR 76.22 (-0.73%) today")
    assert "4 trades (1 won, 3 lost), -EUR 76.22: overnight sweep -EUR 32.28, Session Momentum -EUR 55.02" in body
    assert "noise band -EUR 15.88" in body and "Other" not in body
