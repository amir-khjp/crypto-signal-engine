"""Command line interface: ``cryptosignal <command>`` or ``python -m cryptosignal <command>``."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

from cryptosignal import model
from cryptosignal.alerts import AlertState, TelegramNotifier, format_signal, notify_changes
from cryptosignal.backtest import bars_per_year, run_backtest
from cryptosignal.config import load_config
from cryptosignal.data import MarketData
from cryptosignal.paper import PaperPortfolio
from cryptosignal.pipeline import prepare, scan, strategy_params, timeframe_delta
from cryptosignal.strategy import generate_signals

DISCLAIMER = "Signals are probabilistic and for research only - not financial advice."


def _signals_table(signals) -> str:
    rows = [{
        "symbol": s.symbol,
        "action": s.action,
        "conf": f"{s.confidence:.0%}",
        "price": f"{s.price:,.6g}",
        "stop": "-" if s.stop_loss is None else f"{s.stop_loss:,.6g}",
        "target": "-" if s.take_profit is None else f"{s.take_profit:,.6g}",
        "rules": f"{s.rule_score:+.2f}",
        "ml_up": "-" if s.ml_proba is None else f"{s.ml_proba:.0%}",
        "regime": s.regime,
    } for s in signals]
    return pd.DataFrame(rows).to_string(index=False) if rows else "(no signals)"


def cmd_fetch(args, cfg):
    md = MarketData(cfg)
    md.prefetch()
    print(f"Cached {len(cfg['symbols'])} symbols from {md.exchange_name} into {cfg['data_dir']}/")


def cmd_scan(args, cfg):
    md = MarketData(cfg, offline=args.offline)
    signals, errors = scan(md, cfg, args.mode)
    if args.json:
        print(json.dumps([s.to_dict() for s in signals], indent=2))
    else:
        print(f"Exchange: {md.exchange_name}  timeframe: {cfg['timeframe']}  mode: {args.mode}\n")
        print(_signals_table(signals))
        if args.verbose:
            for s in signals:
                print(f"\n{s.symbol}: " + "; ".join(s.reasons))
        print(f"\n{DISCLAIMER}")
    for symbol, err in errors.items():
        print(f"! {symbol}: {err}", file=sys.stderr)


def cmd_backtest(args, cfg):
    md = MarketData(cfg, offline=args.offline)
    symbols = args.symbols or cfg["symbols"]
    modes = ["rules", "hybrid", "ml"] if args.mode == "all" else [args.mode]
    bpy = bars_per_year(int(timeframe_delta(cfg["timeframe"]).total_seconds()))
    rows = []
    for symbol in symbols:
        df = prepare(md, symbol, cfg)
        proba = model.walk_forward_proba(df, cfg["model"]) if cfg["model"]["enabled"] else None
        # Compare all modes on the same out-of-sample window.
        start = proba.first_valid_index() if proba is not None and proba.notna().any() else df.index[0]
        for mode in modes:
            params = strategy_params(cfg, mode)
            if params["ml_weight"] > 0 and proba is None:
                continue
            signals = generate_signals(df, params, proba if params["ml_weight"] > 0 else None).loc[start:]
            m = run_backtest(signals, cfg["backtest"], params, bpy).metrics
            rows.append({
                "symbol": symbol, "mode": mode,
                "return": f"{m['total_return']:+.1%}", "buy&hold": f"{m['buy_hold_return']:+.1%}",
                "max_dd": f"{m['max_drawdown']:.1%}", "sharpe": f"{m['sharpe']:.2f}",
                "trades": m["trades"], "win": f"{m['win_rate']:.0%}",
                "pf": f"{m['profit_factor']:.2f}", "exposure": f"{m['exposure']:.0%}",
            })
        print(f"{symbol}: out-of-sample window {start:%Y-%m-%d} -> {df.index[-1]:%Y-%m-%d}", file=sys.stderr)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nFees {cfg['backtest']['fee']:.2%}/side, slippage {cfg['backtest']['slippage']:.2%}. "
          "Past performance does not guarantee future results.")


def cmd_train(args, cfg):
    md = MarketData(cfg, offline=args.offline)
    for symbol in args.symbols or cfg["symbols"]:
        df = prepare(md, symbol, cfg)
        proba = model.walk_forward_proba(df, cfg["model"])
        metrics = model.evaluate(df, proba, cfg["model"])
        print(f"\n{symbol} walk-forward: " + ", ".join(
            f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items()))
        if "auc" in metrics and metrics["auc"] < 0.53:
            print("  AUC near 0.5: the model shows little predictive edge here - weight it low.")
        fitted = model.train(df, cfg["model"])
        if fitted is None:
            print("  not enough data to train")
            continue
        path = Path("models") / f"{symbol.replace('/', '-')}_{cfg['timeframe']}.joblib"
        model.save(fitted, path)
        top = model.feature_importance(fitted).head(5)
        print("  top features: " + ", ".join(f"{k} ({v})" for k, v in top.items()))
        print(f"  saved {path}")


def cmd_watch(args, cfg):
    md = MarketData(cfg)
    notifier = TelegramNotifier(cfg["alerts"]["telegram_token"], cfg["alerts"]["telegram_chat_id"])
    state = AlertState(Path(cfg["data_dir"]) / "alert_state.json")
    paper_path = Path(cfg["paper"]["state_file"])
    if not notifier.enabled:
        print("Telegram not configured - alerts are printed only (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID).")
    while True:
        signals, errors = scan(md, cfg, args.mode)
        stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
        print(f"\n[{stamp}]\n{_signals_table(signals)}")
        for sig in notify_changes(signals, state, notifier):
            print("\n>> NEW SIGNAL\n" + format_signal(sig))
        if args.paper:
            portfolio = PaperPortfolio.load(paper_path, cfg["paper"]["initial_capital"])
            for action in portfolio.update(signals, cfg["paper"]["position_fraction"], cfg["backtest"]["fee"]):
                print(f"[paper] {action}")
            portfolio.save(paper_path)
        for symbol, err in errors.items():
            print(f"! {symbol}: {err}", file=sys.stderr)
        if args.once:
            break
        time.sleep(args.interval * 60)


def cmd_paper(args, cfg):
    path = Path(cfg["paper"]["state_file"])
    if args.reset:
        path.unlink(missing_ok=True)
        print("Paper portfolio reset.")
        return
    portfolio = PaperPortfolio.load(path, cfg["paper"]["initial_capital"])
    md = MarketData(cfg, offline=args.offline)
    prices = {s: float(md.candles(s, cfg["timeframe"], 1)["close"].iloc[-1]) for s in portfolio.positions}
    equity = portfolio.equity(prices)
    start = cfg["paper"]["initial_capital"]
    print(f"Equity {equity:,.2f} ({equity / start - 1:+.2%})  cash {portfolio.cash:,.2f}")
    for sym, pos in portfolio.positions.items():
        print(f"  {sym}: {pos.qty:.6g} @ {pos.entry_price:,.6g} now {prices[sym]:,.6g} "
              f"({prices[sym] / pos.entry_price - 1:+.2%})  stop {pos.stop_loss:,.6g}  target {pos.take_profit:,.6g}")
    if portfolio.closed:
        print(pd.DataFrame(portfolio.closed).tail(10).to_string(index=False))


def cmd_dashboard(args, cfg):
    app = Path(__file__).with_name("dashboard.py")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app)], check=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cryptosignal", description="Crypto buy/sell signal engine")
    parser.add_argument("--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name, func, help_text):
        p = sub.add_parser(name, help=help_text)
        p.set_defaults(func=func)
        return p

    add("fetch", cmd_fetch, "download and cache candles for all symbols")

    p = add("scan", cmd_scan, "show the latest signal for every symbol")
    p.add_argument("--mode", choices=["rules", "ml", "hybrid"], default="hybrid")
    p.add_argument("--json", action="store_true")
    p.add_argument("--offline", action="store_true", help="use cached data only")

    p = add("backtest", cmd_backtest, "walk-forward backtest vs buy-and-hold")
    p.add_argument("symbols", nargs="*")
    p.add_argument("--mode", choices=["rules", "ml", "hybrid", "all"], default="all")
    p.add_argument("--offline", action="store_true")

    p = add("train", cmd_train, "evaluate and save the ML model per symbol")
    p.add_argument("symbols", nargs="*")
    p.add_argument("--offline", action="store_true")

    p = add("watch", cmd_watch, "re-scan on an interval and send alerts")
    p.add_argument("--mode", choices=["rules", "ml", "hybrid"], default="hybrid")
    p.add_argument("--interval", type=float, default=None, help="minutes between scans")
    p.add_argument("--paper", action="store_true", help="also follow signals in the paper portfolio")
    p.add_argument("--once", action="store_true", help="run a single scan and exit")

    p = add("paper", cmd_paper, "show or reset the paper-trading portfolio")
    p.add_argument("--reset", action="store_true")
    p.add_argument("--offline", action="store_true")

    add("dashboard", cmd_dashboard, "open the Streamlit dashboard")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    warnings.filterwarnings("ignore", category=UserWarning)
    cfg = load_config(args.config)
    if getattr(args, "interval", 0) is None:
        args.interval = cfg["watch_interval_minutes"]
    args.func(args, cfg)


if __name__ == "__main__":
    main()
