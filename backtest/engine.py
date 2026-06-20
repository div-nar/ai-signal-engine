"""Weekly-rebalanced ablation backtest over the strategy core."""
import numpy as np
import pandas as pd

from strategy.assemble import assemble_portfolio
from strategy.factors import momentum_scores
from backtest.panel import to_weekly_fridays, forward_returns


def equal_weight_scores(tickers: list[str]) -> dict[str, float]:
    """Zero scores -> assembler falls back to equal weight within each layer."""
    return {t: 0.0 for t in tickers}


def run_variant(daily_panel, layer_map, budgets, variant: str = "baseline",
                top_n: int = 3, name_cap: float = 0.12,
                lookback: int = 126, skip: int = 21) -> pd.Series:
    """Return the weekly-rebalanced equity curve (starts at 1.0) for a variant."""
    weekly = to_weekly_fridays(daily_panel)
    fwd = forward_returns(daily_panel, weekly.index)
    universe = [t for t in daily_panel.columns if t in layer_map]

    equity = [1.0]
    index = [weekly.index[0]]
    for i, d0 in enumerate(fwd.index):
        if variant == "momentum":
            scores = momentum_scores(daily_panel, d0, lookback=lookback, skip=skip)
            scores = {t: scores[t] for t in universe if t in scores}
        elif variant == "baseline":
            scores = equal_weight_scores(universe)
        else:
            raise ValueError(f"Unknown variant: {variant!r}")
        weights = assemble_portfolio(budgets, scores, layer_map, top_n=top_n, name_cap=name_cap)
        period = fwd.loc[d0]
        port_ret = sum(w * period.get(t, 0.0) for t, w in weights.items())
        equity.append(equity[-1] * (1.0 + port_ret))
        index.append(weekly.index[i + 1])  # label equity by period END, not start
    return pd.Series(equity, index=index)


def metrics(equity: pd.Series, periods_per_year: int = 52) -> dict:
    """Total return %, annualized Sharpe (rf=0), and max drawdown %."""
    rets = equity.pct_change().dropna()
    total = (equity.iloc[-1] / equity.iloc[0] - 1.0) * 100
    sharpe = (rets.mean() / rets.std() * np.sqrt(periods_per_year)) if rets.std() > 0 else 0.0
    peak = equity.cummax()
    mdd = ((equity - peak) / peak).min() * 100
    return {
        "total_return_pct": float(total),
        "sharpe": float(sharpe),
        "max_drawdown_pct": float(mdd),
    }
