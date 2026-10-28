from cryptosignal.alerts import AlertState, TelegramNotifier, format_signal, notify_changes
from cryptosignal.cli import build_parser
from cryptosignal.paper import PaperPortfolio
from cryptosignal.pipeline import edge_factor
from cryptosignal.strategy import Signal


def make_signal(action="BUY", price=100.0, symbol="BTC/USDT"):
    stop, take = (95.0, 110.0) if action == "BUY" else (None, None)
    return Signal(symbol, "4h", "2026-01-01T00:00:00+00:00", action, 0.6, price, stop, take, 0.6, 0.5, 0.55,
                  "bull", ["EMA20 above EMA50"])


def test_format_signal_contains_levels():
    text = format_signal(make_signal())
    assert "BUY BTC/USDT" in text and "Stop-loss" in text and "Not financial advice" in text


def test_alert_state_only_fires_on_change(tmp_path):
    state = AlertState(tmp_path / "state.json")
    notifier = TelegramNotifier("", "")
    assert len(notify_changes([make_signal("BUY")], state, notifier)) == 1
    assert notify_changes([make_signal("BUY")], state, notifier) == []
    assert notify_changes([make_signal("HOLD")], state, notifier) == []
    assert len(notify_changes([make_signal("SELL")], AlertState(tmp_path / "state.json"), notifier)) == 1


def test_paper_portfolio_round_trip(tmp_path):
    path = tmp_path / "paper.json"
    pf = PaperPortfolio.load(path, 1000.0)
    actions = pf.update([make_signal("BUY", 100.0)], fraction=0.5, fee=0.0)
    assert actions and pf.cash == 500.0
    pf.save(path)

    pf = PaperPortfolio.load(path, 1000.0)
    pf.update([make_signal("HOLD", 112.0)], fraction=0.5, fee=0.0)  # above take-profit
    assert not pf.positions
    assert pf.closed[0]["reason"] == "take_profit"
    assert pf.cash == 500.0 + 5 * 112.0


def test_paper_stop_loss(tmp_path):
    pf = PaperPortfolio.load(tmp_path / "p.json", 1000.0)
    pf.update([make_signal("BUY", 100.0)], fraction=1.0, fee=0.0)
    pf.update([make_signal("HOLD", 90.0)], fraction=1.0, fee=0.0)
    assert pf.closed[0]["reason"] == "stop_loss"


def test_edge_factor_gates_weak_models():
    cfg = {"min_auc": 0.52, "full_auc": 0.56}
    assert edge_factor(None, cfg) == 0
    assert edge_factor(0.50, cfg) == 0
    assert edge_factor(0.54, cfg) == 0.5
    assert edge_factor(0.70, cfg) == 1


def test_cli_parses_commands():
    parser = build_parser()
    args = parser.parse_args(["backtest", "BTC/USDT", "--mode", "rules"])
    assert args.symbols == ["BTC/USDT"] and args.mode == "rules"
    assert parser.parse_args(["watch", "--once", "--paper"]).paper
