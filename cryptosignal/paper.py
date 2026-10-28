"""Paper trading: follow live signals with simulated money to validate them forward in time."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cryptosignal.strategy import Signal


@dataclass
class Position:
    qty: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened: str


@dataclass
class PaperPortfolio:
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    closed: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path, initial_capital: float) -> PaperPortfolio:
        if not path.exists():
            return cls(cash=initial_capital)
        raw = json.loads(path.read_text())
        positions = {k: Position(**v) for k, v in raw["positions"].items()}
        return cls(cash=raw["cash"], positions=positions, closed=raw["closed"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"cash": self.cash, "positions": {k: asdict(v) for k, v in self.positions.items()}, "closed": self.closed}
        path.write_text(json.dumps(data, indent=2))

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.qty * prices.get(sym, p.entry_price) for sym, p in self.positions.items())

    def _close(self, symbol: str, price: float, reason: str, fee: float) -> None:
        pos = self.positions.pop(symbol)
        proceeds = pos.qty * price * (1 - fee)
        cost = pos.qty * pos.entry_price
        self.cash += proceeds
        self.closed.append({
            "symbol": symbol,
            "opened": pos.opened,
            "closed": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "entry_price": pos.entry_price,
            "exit_price": price,
            "reason": reason,
            "pnl_pct": proceeds / cost - 1,
        })

    def update(self, signals: list[Signal], fraction: float, fee: float) -> list[str]:
        """Apply the latest signals and return a log of actions taken."""
        actions = []
        prices = {s.symbol: s.price for s in signals}
        for sig in signals:
            pos = self.positions.get(sig.symbol)
            if pos is not None:
                reason = None
                if sig.price <= pos.stop_loss:
                    reason = "stop_loss"
                elif sig.price >= pos.take_profit:
                    reason = "take_profit"
                elif sig.action == "SELL":
                    reason = "signal"
                if reason:
                    self._close(sig.symbol, sig.price, reason, fee)
                    actions.append(f"SELL {sig.symbol} @ {sig.price:,.6g} ({reason})")
            elif sig.action == "BUY" and sig.stop_loss is not None:
                budget = min(self.cash, self.equity(prices) * fraction)
                if budget <= 1:
                    continue
                qty = budget * (1 - fee) / sig.price
                self.cash -= budget
                self.positions[sig.symbol] = Position(qty, sig.price, sig.stop_loss, sig.take_profit, sig.time)
                actions.append(f"BUY {sig.symbol} @ {sig.price:,.6g} for {budget:,.2f}")
        return actions
