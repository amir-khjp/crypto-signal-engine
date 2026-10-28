"""Telegram notifications, sent only when a symbol's signal changes."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import requests

from cryptosignal.strategy import Signal

log = logging.getLogger(__name__)


def format_signal(sig: Signal) -> str:
    lines = [
        f"{sig.action} {sig.symbol} ({sig.timeframe})",
        f"Price: {sig.price:,.6g}",
        f"Confidence: {sig.confidence:.0%}  |  regime: {sig.regime}",
    ]
    if sig.stop_loss is not None:
        lines.append(f"Stop-loss: {sig.stop_loss:,.6g}  |  Take-profit: {sig.take_profit:,.6g}")
    lines.append("Why: " + "; ".join(sig.reasons))
    lines.append(f"Candle: {sig.time}")
    lines.append("Not financial advice.")
    return "\n".join(lines)


class AlertState:
    """Remembers the last alerted action per symbol so repeats are suppressed."""

    def __init__(self, path: Path):
        self.path = path
        self.state: dict[str, dict[str, str]] = json.loads(path.read_text()) if path.exists() else {}

    def is_new(self, sig: Signal) -> bool:
        if sig.action == "HOLD":
            return False
        last = self.state.get(sig.symbol)
        return last is None or last["action"] != sig.action

    def record(self, sig: Signal) -> None:
        if sig.action != "HOLD":
            self.state[sig.symbol] = {"action": sig.action, "time": sig.time}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))


class TelegramNotifier:
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def send(self, text: str) -> bool:
        if not self.enabled:
            return False
        try:
            resp = requests.post(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text},
                timeout=15,
            )
            resp.raise_for_status()
            return True
        except requests.RequestException as exc:
            log.warning("Telegram send failed: %s", exc)
            return False


def notify_changes(signals: list[Signal], state: AlertState, notifier: TelegramNotifier) -> list[Signal]:
    """Send alerts for new BUY/SELL signals and return the ones that were new."""
    fresh = [s for s in signals if state.is_new(s)]
    for sig in fresh:
        if notifier.enabled:
            notifier.send(format_signal(sig))
        state.record(sig)
    return fresh
