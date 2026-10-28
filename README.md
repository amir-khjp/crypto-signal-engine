# Crypto Signal Engine

![tests](https://github.com/amir-khjp/crypto-signal-engine/actions/workflows/tests.yml/badge.svg)

A buy/sell **signal engine** for crypto markets. It pulls live candles from an exchange, scores
each coin with transparent technical rules and a machine-learning model, and gives
**BUY / SELL / HOLD** signals with a confidence score, stop-loss, take-profit and the reasons behind them.
Every strategy can be checked with an honest walk-forward backtest before you trust it.

> **Not financial advice.** Crypto prices cannot be reliably predicted. This tool estimates
> probabilities and measures how its signals would have performed in the past. Paper-trade first and
> never risk money you cannot afford to lose.

## Features

| | |
|---|---|
| **Market data** | Any [ccxt](https://github.com/ccxt/ccxt) exchange with automatic fallback (KuCoin -> Binance -> Bybit -> OKX -> Kraken). No API key needed. Incremental parquet cache. Uses closed candles only. |
| **Indicators** | EMA 20/50/200, RSI, MACD, Bollinger Bands, ATR, ADX, volume z-score, returns and volatility. Plain pandas, so every formula is readable. |
| **Regime filter** | BUY signals only when the daily close is above the daily EMA200, aligned without look-ahead bias. |
| **Rule engine** | Weighted trend, momentum, RSI, volume and Bollinger scores, giving a score in [-1, 1]. |
| **ML model** | LightGBM estimates the probability of an up-move over the next N candles. Trained walk-forward with a purge gap. Its live weight scales automatically with its **out-of-sample AUC**, so a model with no proven edge cannot drive signals. |
| **Backtester** | Long-only spot trading, next-candle execution, fees, slippage, ATR stops and targets. Reports CAGR, drawdown, Sharpe, win rate, profit factor and exposure vs. buy & hold. |
| **Scanner / CLI** | `cryptosignal scan` shows a signal table for every coin. JSON output is available. |
| **Alerts** | Telegram message when a coin's signal changes. |
| **Paper trading** | Follows live signals with simulated money to check them going forward. |
| **Dashboard** | Streamlit tabs: signals, candlestick chart with buy/sell markers, backtest equity curve, paper portfolio. |

## Quick start

```bash
git clone https://github.com/amir-khjp/crypto-signal-engine.git
cd crypto-signal-engine
pip install -e ".[dev]"

cryptosignal fetch          # download & cache candles
cryptosignal scan           # latest signals for all coins
cryptosignal dashboard      # open the web dashboard
```

`python -m cryptosignal <command>` works too.

## Commands

| Command | What it does |
|---|---|
| `cryptosignal scan [--mode hybrid\|rules\|ml] [--json] [--offline]` | Latest signal per coin (`-v` before `scan` also prints the reasons) |
| `cryptosignal backtest [SYMBOL ...] [--mode all]` | Walk-forward backtest of every mode vs. buy & hold |
| `cryptosignal train [SYMBOL ...]` | ML out-of-sample quality (AUC, accuracy), top features; saves models to `models/` |
| `cryptosignal watch [--interval 60] [--paper] [--once]` | Re-scan on a schedule, send Telegram alerts, optionally paper-trade |
| `cryptosignal paper [--reset]` | Paper portfolio status |
| `cryptosignal dashboard` | Streamlit dashboard |

Example output:

```
  symbol action conf    price    stop  target rules ml_up regime
BTC/USDT   HOLD   0%   81,330       -       - -0.00   61%   bull
ETH/USDT   HOLD   6% 2,361.59       -       - -0.06   20%   bull
BNB/USDT    BUY  77%  760.665 744.577 792.842 +0.77   22%   bull
```

## How a signal is made

1. **Candles:** 4h candles (about 500 days) plus daily candles for the regime filter.
2. **Rule score:** a weighted mix of trend (EMA alignment), momentum (MACD histogram and its slope),
   RSI zone, volume confirmation and Bollinger extremes, giving a score from -1 to 1.
3. **ML score:** LightGBM's up-move probability is mapped to -1 to 1. It is blended in with weight
   `ml_weight × edge`, where `edge` goes from 0 at an out-of-sample AUC of 0.52 to 1 at 0.56.
4. **Decision:** score ≥ `buy_threshold` and bull regime gives **BUY**; score ≤ `sell_threshold` gives **SELL**; anything else is **HOLD**.
5. **Risk levels:** stop-loss = entry − 2×ATR, take-profit = entry + 4×ATR (configurable).

## Configuration

Edit [`config.yaml`](config.yaml) to change coins, timeframe, thresholds, weights, fees and ML settings.
Any setting you leave out uses the defaults in [`cryptosignal/config.py`](cryptosignal/config.py).

### Telegram alerts

1. Create a bot with [@BotFather](https://t.me/BotFather) and copy its token.
2. Send your bot a message, then open `https://api.telegram.org/bot<TOKEN>/getUpdates` to find your chat id.
3. Set the environment variables (don't commit secrets):

```bash
export TELEGRAM_BOT_TOKEN=123:abc     # PowerShell: $env:TELEGRAM_BOT_TOKEN="123:abc"
export TELEGRAM_CHAT_ID=987654321
cryptosignal watch --paper
```

## Reading the backtest honestly

- All modes are compared over the **same out-of-sample window**, starting where the first walk-forward ML prediction exists.
- Signals use the close of a candle; trades fill at the **next** candle's open, with fees and slippage.
- When a candle hits both the stop and the target, the stop is assumed to fill first (the worst case).
- A model AUC near **0.50 means no predictive edge**. That is common in crypto, and the engine then relies on the rules.
- A strategy that loses less than buy & hold in a falling market is not necessarily profitable. Check the
  return, drawdown and number of trades together, and treat a small number of trades as weak evidence.

## Project layout

```
cryptosignal/
  config.py       defaults + YAML/env loading
  data.py         ccxt download, pagination, parquet cache
  indicators.py   indicators + higher-timeframe regime
  strategy.py     rule scoring, ML blend, Signal objects
  model.py        LightGBM, labels, walk-forward predictions, evaluation
  backtest.py     backtester and metrics
  pipeline.py     end-to-end analysis and market scan
  alerts.py       Telegram notifications with de-duplication
  paper.py        paper-trading portfolio
  cli.py          command line interface
  dashboard.py    Streamlit app
tests/            pytest suite (synthetic data, no network needed)
```

## Development

```bash
pytest -q
```

The test suite includes a check that the ML pipeline finds **no** edge on a pure random walk.
This guards against look-ahead leaks, which make backtests look far better than reality.

## License

MIT
