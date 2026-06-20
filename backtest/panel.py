"""Historical price panel loading and rebalance-date utilities for the backtest."""
import pandas as pd
import yfinance as yf


def load_price_panel(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Daily adjusted close panel via yfinance, index ascending, columns=tickers."""
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data["Close"] if isinstance(data.columns, pd.MultiIndex) else data
    return close.dropna(how="all").sort_index()


def to_weekly_fridays(panel: pd.DataFrame) -> pd.DataFrame:
    """Resample daily panel to Friday closes (the weekly rebalance grid)."""
    return panel.resample("W-FRI").last().dropna(how="all")


def forward_returns(panel: pd.DataFrame, rebal_dates) -> pd.DataFrame:
    """Per-ticker simple return between consecutive rebalance dates."""
    dates = list(rebal_dates)
    rows = {}
    for d0, d1 in zip(dates[:-1], dates[1:]):
        p0 = panel.loc[:d0].iloc[-1]
        p1 = panel.loc[:d1].iloc[-1]
        rows[d0] = (p1 / p0 - 1.0)
    return pd.DataFrame(rows).T
