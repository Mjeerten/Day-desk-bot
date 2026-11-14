"""Order ticket at the top of every entry alert, in the order you fill in the Trading 212 order screen, with plain
numbers (no spaces) so what you read is exactly what you type. push() in desk/bot.py turns the number lines into ntfy
'copy' buttons (tap, then paste into the field)."""
import re


def num(x):
    return f"{x:.0f}"


def ticket(side, stake, stop=None, target=None):
    """side 1 = buy, -1 = sell (short). Lines in Trading 212 order: direction, value, take profit, stop loss."""
    lines = ["ORDER - US Tech 100, market",
             f"{'BUY' if side == 1 else 'SELL (short)'}",
             f"Value: {stake:.0f}  (EUR, order screen on value)",
             f"Take profit: {num(target)}" if target is not None else "Take profit: none",
             f"Stop loss: {num(stop)}" if stop is not None else "Stop loss: none"]
    return "\n".join(lines) + "\n\n"


COPY = re.compile(r"^(Value|Take profit|Stop loss|New stop): (\d+)\b", re.M)


def copy_actions(body):
    """ntfy Actions header with one copy button per number line (ntfy allows at most 3)."""
    acts = [f"copy, {k.replace('Take profit', 'TP').replace('Stop loss', 'SL')} {v}, {v}" for k, v in COPY.findall(body)]
    return "; ".join(acts[:3])


# Tapping the notification opens Trading 212 (Android opens the app if it claims the link, otherwise the web app).
T212 = "https://app.trading212.com/"
ACT = re.compile(r"^ORDER - |SELL \(close\)|Move the stop|Close any", re.M)


def click_url(body):
    """Trading 212 link for alerts that ask you to do something in Trading 212, else None."""
    return T212 if ACT.search(body) else None
