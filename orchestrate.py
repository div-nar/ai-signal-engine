"""Weekly target computation: thesis -> budgets -> momentum -> target, persisted."""
from config import DB_PATH
from db import init_targets_table, insert_target, get_latest_target
from strategy.layers import LAYER_MAP
from scoring.thesis_scorer import score_layer_thesis
from pricing.history import fetch_recent_closes
from strategy.pipeline import build_target_portfolio
from execution.alpaca import execute_sells, execute_buys


def compute_weekly_target(docs: list[dict], db_path: str = DB_PATH,
                          thesis_client=None, data_client=None,
                          persist: bool = True, now=None) -> dict:
    """Run thesis -> budgets -> momentum pipeline and (optionally) persist the target."""
    init_targets_table(db_path)
    prior = get_latest_target(db_path)
    prev_budgets = prior["layer_budgets"] if prior else {}

    thesis = score_layer_thesis(docs, prev_budgets=prev_budgets, client=thesis_client)

    tickers = sorted(LAYER_MAP)
    prices = fetch_recent_closes(tickers, client=data_client, now=now)
    weights = build_target_portfolio(thesis["layer_budgets"], prices, LAYER_MAP)

    target = {
        "layer_tilt": thesis["layer_tilt"],
        "layer_budgets": thesis["layer_budgets"],
        "target_weights": weights,
        "market_regime": thesis["market_regime"],
        "regime_shift": thesis["regime_shift"],
        "thesis_update": thesis["thesis_update"],
    }
    # Never persist a degenerate (empty) target: an empty book would become next
    # week's "prior" and would be what a Plan-2b executor reads. Fail loud instead.
    if not weights:
        print("  WARNING: empty target_weights (insufficient price history?) — not persisting")
    elif persist:
        insert_target(db_path, target)
    return target


def run_sell(docs: list[dict], db_path: str = DB_PATH, thesis_client=None,
             data_client=None, exec_client=None, cash_buffer: float = 0.0, now=None) -> dict:
    """Friday: compute + persist the weekly target, then execute the sell leg."""
    target = compute_weekly_target(docs, db_path=db_path, thesis_client=thesis_client,
                                   data_client=data_client, persist=True, now=now)
    weights = target.get("target_weights") or {}
    if not weights:
        print("  No target weights — skipping sells")
        return target
    execute_sells(weights, cash_buffer=cash_buffer, client=exec_client)
    return target


def run_buy(db_path: str = DB_PATH, exec_client=None, cash_buffer: float = 0.0) -> dict | None:
    """Monday: load the latest persisted target, then execute the buy leg."""
    init_targets_table(db_path)
    target = get_latest_target(db_path)
    if not target or not target.get("target_weights"):
        print("  No persisted target — skipping buys")
        return None
    execute_buys(target["target_weights"], cash_buffer=cash_buffer, client=exec_client)
    return target
