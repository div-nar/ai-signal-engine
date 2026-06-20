"""Run the ablation backtest and print the go/no-go scorecard.

Variants that ARE backtestable mechanically:
  - baseline : thesis layer budgets, equal weight within layer
  - momentum : thesis layer budgets, momentum-ranked within layer
The LLM layer-tilt variant is NOT historically backtestable (no historical
thesis-pass outputs); it is evaluated in the forward/paper phase in Plan 2 by
replaying recorded thesis passes. This runner establishes the mechanical floor.
"""
import pandas as pd

from strategy.layers import LAYER_MAP, BASELINE_BUDGETS
from backtest.panel import load_price_panel, to_weekly_fridays
from backtest.engine import run_variant, metrics


def _benchmark_equity(daily_panel, weekly_index, bench_col: str) -> pd.Series:
    weekly = daily_panel[bench_col].resample("W-FRI").last().dropna()
    weekly = weekly.loc[weekly.index.intersection(weekly_index)]
    return weekly / weekly.iloc[0]


def _first_active_date(daily_panel, rebal_index, lookback, skip):
    """First rebalance date with enough history for a momentum signal."""
    needed = lookback + skip + 1
    for d in rebal_index:
        if len(daily_panel.loc[:d]) >= needed:
            return d
    return rebal_index[-1]


def _trim_renormalize(equity, start):
    """Restrict an equity curve to dates >= start and rebase it to 1.0 at start."""
    e = equity.loc[equity.index >= start]
    return e / e.iloc[0]


def build_scorecard(daily_panel, benchmark, layer_map, budgets, lookback=126, skip=21) -> pd.DataFrame:
    """One row per variant + QQQ, columns = metrics, all over the common
    fully-invested window (momentum's warm-up cash period trimmed off)."""
    weekly = to_weekly_fridays(daily_panel)
    start = _first_active_date(daily_panel, weekly.index, lookback, skip)
    rows = {}
    for variant in ("baseline", "momentum"):
        eq = run_variant(daily_panel, layer_map, budgets, variant=variant,
                         lookback=lookback, skip=skip)
        rows[variant] = metrics(_trim_renormalize(eq, start))
    rows["QQQ"] = metrics(_trim_renormalize(benchmark, start))
    return pd.DataFrame(rows).T[["total_return_pct", "sharpe", "max_drawdown_pct"]]


def main():
    # Momentum needs ~126+21 trading days of runway, far more than the live
    # account's ~60-day history. Load extended history so the momentum variant is
    # genuinely evaluated; the equity curves for all variants + QQQ span the same
    # extended window and remain comparable.
    tickers = sorted(LAYER_MAP) + ["QQQ"]
    panel = load_price_panel(tickers, start="2025-01-01", end="2026-06-20")
    thesis_panel = panel.drop(columns=["QQQ"])
    weekly = to_weekly_fridays(thesis_panel)
    bench = _benchmark_equity(panel, weekly.index, "QQQ")
    scorecard = build_scorecard(thesis_panel, bench, LAYER_MAP, BASELINE_BUDGETS)
    print("\n=== ABLATION SCORECARD (weekly rebal, inception -> today) ===")
    print(scorecard.round(2).to_string())
    print("\nShip gate (Plan 2): momentum variant must beat baseline on Sharpe,")
    print("and at least one variant must beat QQQ on Sharpe, to justify going live.")


if __name__ == "__main__":
    main()
