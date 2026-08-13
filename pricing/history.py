"""Live daily price history for the momentum factor, via Alpaca market data."""
import datetime as dt

import pandas as pd

_CHUNK = 100  # symbols per Alpaca bars request; larger overflows the endpoint


def _default_client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    )


def fetch_recent_closes(tickers: list[str], lookback_days: int = 320,
                        client=None, now=None) -> pd.DataFrame:
    """Daily close panel for the trailing lookback_days (index ascending, cols=tickers).

    Default 320 calendar days (~228 trading days) leaves comfortable headroom over the
    momentum window's 148-bar floor so a live run never silently returns too few rows.
    """
    if client is None:
        client = _default_client()
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=lookback_days)

    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    # IEX is the free-tier feed; SIP rejects "recent" (<~15 min) bars with a 403.
    # Momentum skips the most recent ~21 trading days anyway, so IEX daily closes
    # are more than sufficient.
    # Batch symbols: a single request for hundreds of tickers overflows Alpaca's
    # bars endpoint (connection reset), so chunk and merge. Whole-market momentum
    # (~600 names) needs this.
    bars = {}
    for i in range(0, len(tickers), _CHUNK):
        chunk = tickers[i:i + _CHUNK]
        request = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                                   start=start, end=now, feed=DataFeed.IEX)
        try:
            bars.update(client.get_stock_bars(request).data)
        except Exception as e:
            print(f"  WARNING: price fetch failed for {len(chunk)} symbols "
                  f"({chunk[0]}..{chunk[-1]}): {str(e)[:80]}")

    series = {}
    for sym, rows in bars.items():
        if not rows:
            continue
        series[sym] = pd.Series(
            {str(b.timestamp)[:10]: float(b.close) for b in rows}
        )
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series)
    panel.index = pd.to_datetime(panel.index)
    return panel.sort_index()
