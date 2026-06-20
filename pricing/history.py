"""Live daily price history for the momentum factor, via Alpaca market data."""
import datetime as dt

import pandas as pd


def _default_client():
    import os
    from alpaca.data.historical import StockHistoricalDataClient
    return StockHistoricalDataClient(
        os.environ.get("ALPACA_API_KEY"), os.environ.get("ALPACA_SECRET_KEY")
    )


def fetch_recent_closes(tickers: list[str], lookback_days: int = 220,
                        client=None, now=None) -> pd.DataFrame:
    """Daily close panel for the trailing lookback_days (index ascending, cols=tickers)."""
    if client is None:
        client = _default_client()
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    start = now - dt.timedelta(days=lookback_days)

    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    request = StockBarsRequest(symbol_or_symbols=tickers, timeframe=TimeFrame.Day,
                               start=start, end=now)
    bars = client.get_stock_bars(request).data

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
