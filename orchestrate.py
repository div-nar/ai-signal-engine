"""Weekly target computation: thesis -> budgets -> momentum -> target, persisted."""
from config import DB_PATH
from db import insert_target, get_latest_target
from strategy.layers import LAYER_MAP
from scoring.thesis_scorer import score_layer_thesis
from pricing.history import fetch_recent_closes
from strategy.pipeline import build_target_portfolio


def compute_weekly_target(docs: list[dict], db_path: str = DB_PATH,
                          thesis_client=None, data_client=None,
                          persist: bool = True, now=None) -> dict:
    """Run thesis -> budgets -> momentum pipeline and (optionally) persist the target."""
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
    if persist:
        insert_target(db_path, target)
    return target
