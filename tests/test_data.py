import pandas as pd

from cryptosignal.data import drop_incomplete, fetch_ohlcv, to_frame

HOUR = 3_600_000


class FakeExchange:
    """Serves hourly candles from t=0 to ``now`` in pages of ``page`` rows."""

    def __init__(self, now_ms: int, page: int = 3):
        self.now = now_ms
        self.page = page
        self.calls = 0

    def parse_timeframe(self, timeframe):
        return 3600

    def milliseconds(self):
        return self.now

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        self.calls += 1
        start = since - since % HOUR
        stamps = [t for t in range(start, self.now, HOUR)][: self.page]
        return [[t, 1, 2, 0.5, 1.5, 10] for t in stamps]


def test_to_frame_sorts_and_dedupes():
    rows = [[2 * HOUR, 1, 1, 1, 1, 1], [HOUR, 1, 1, 1, 1, 1], [HOUR, 1, 1, 1, 1, 1]]
    df = to_frame(rows)
    assert len(df) == 2
    assert df.index.is_monotonic_increasing
    assert str(df.index.tz) == "UTC"


def test_drop_incomplete_removes_forming_candle():
    df = to_frame([[0, 1, 1, 1, 1, 1], [HOUR, 1, 1, 1, 1, 1]])
    assert len(drop_incomplete(df, HOUR, HOUR + HOUR // 2)) == 1
    assert len(drop_incomplete(df, HOUR, 2 * HOUR)) == 2


def test_fetch_ohlcv_paginates_and_drops_open_candle():
    exchange = FakeExchange(now_ms=10 * HOUR + HOUR // 2, page=3)
    df = fetch_ohlcv(exchange, "BTC/USDT", "1h", since_ms=0)
    # candles 0..9 are closed, candle 10 is still forming
    assert len(df) == 10
    assert df.index[-1] == pd.Timestamp(9 * HOUR, unit="ms", tz="UTC")
    assert exchange.calls >= 4
