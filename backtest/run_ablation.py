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
from backtest.panel import load_price_panel, to_weekly_fridays, forward_returns
from backtest.engine import run_variant, metrics


def _benchmark_equity(daily_panel, weekly_index, bench_col: str) -> pd.Series:
    weekly = daily_panel[bench_col].resample("W-FRI").last().dropna()
    weekly = weekly.loc[weekly.index.intersection(weekly_index)]
    return weekly / weekly.iloc[0]


def build_scorecard(daily_panel, benchmark, layer_map, budgets) -> pd.DataFrame:
    """One row per variant + QQQ, columns = metrics."""
    rows = {}
    for variant in ("baseline", "momentum"):
        eq = run_variant(daily_panel, layer_map, budgets, variant=variant)
        rows[variant] = metrics(eq)
    rows["QQQ"] = metrics(benchmark)
    return pd.DataFrame(rows).T[["total_return_pct", "sharpe", "max_drawdown_pct"]]


def main():
    # Momentum needs ~126+21 trading days of runway, far more than the live
    # account's ~60-day history. Load extended history so the momentum variant is
    # genuinely evaluated; the equity curves for all variants + QQQ span the same
    # extended window and remain comparable.
    tickers = sorted(LAYER_MAP) + ["QQQ"]
    panel = load_price_panel(tickers, start="2025-01-01", end="2026-06-20")
    qqq = panel["QQQ"]
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
