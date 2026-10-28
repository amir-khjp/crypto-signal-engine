import pandas as pd
import pytest

from cryptosignal.backtest import run_backtest
from cryptosignal.strategy import BUY, HOLD, SELL

BT = {"initial_capital": 1000.0, "fee": 0.0, "slippage": 0.0}
STRAT = {"stop_loss_atr": 2.0, "take_profit_atr": 4.0}


def frame(opens, highs, lows, closes, signals, atr=1.0):
    index = pd.date_range("2024-01-01", periods=len(opens), freq="4h", tz="UTC")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "atr": atr, "signal": signals}, index=index
    )


def test_no_signals_keeps_capital():
    df = frame([10] * 5, [11] * 5, [9] * 5, [10] * 5, [HOLD] * 5)
    result = run_backtest(df, BT, STRAT, 2190)
    assert result.metrics["total_return"] == 0
    assert result.trades == []


def test_buy_then_sell_executes_on_next_open():
    df = frame(
        opens=[10, 10, 11, 12],
        highs=[10.5, 11, 12, 12.5],
        lows=[9.9, 9.9, 10.9, 11.9],
        closes=[10, 11, 12, 12],
        signals=[BUY, HOLD, SELL, HOLD],
    )
    result = run_backtest(df, BT, STRAT, 2190)
    (trade,) = result.trades
    assert trade.entry_price == 10 and trade.exit_price == 12
    assert trade.exit_reason == "signal"
    assert result.metrics["total_return"] == pytest.approx(0.2)


def test_fees_reduce_return():
    df = frame([10, 10, 12], [10, 10, 12], [10, 10, 12], [10, 10, 12], [BUY, SELL, HOLD])
    result = run_backtest(df, {**BT, "fee": 0.01}, STRAT, 2190)
    assert result.trades[0].pnl_pct == pytest.approx(1.2 * 0.99 * 0.99 - 1)


def test_stop_loss_triggers_intrabar():
    df = frame(
        opens=[10, 10, 9.5],
        highs=[10, 10.2, 9.6],
        lows=[10, 7.5, 9],
        closes=[10, 8, 9.5],
        signals=[BUY, HOLD, HOLD],
    )
    result = run_backtest(df, BT, STRAT, 2190)
    (trade,) = result.trades
    assert trade.exit_reason == "stop_loss"
    assert trade.exit_price == pytest.approx(8.0)


def test_take_profit_triggers():
    df = frame([10, 10, 13], [10, 14.5, 13], [10, 9.5, 13], [10, 14, 13], [BUY, HOLD, HOLD])
    result = run_backtest(df, BT, STRAT, 2190)
    assert result.trades[0].exit_reason == "take_profit"
    assert result.trades[0].exit_price == pytest.approx(14.0)
