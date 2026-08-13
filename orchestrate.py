"""Weekly target computation: thesis -> budgets -> momentum -> target, persisted.

The thesis pass is agentic when a ChromaDB client is supplied: the LLM can
query the research vector store before deciding, its recent thesis history is
fed back as memory, and each new thesis is written to the store. Trading is
gated by the LLM's rebalance urgency + mechanical drift bands (strategy.risk).
"""
import config
from config import DB_PATH
from db import (init_targets_table, insert_target, get_latest_target,
                update_target_gate)
from strategy.layers import LAYER_MAP, LAYERS, tickers_in_layer
from strategy.risk import needs_rebalance
from strategy.factors import momentum_scores, apply_name_adjustments
from strategy.assemble import assemble_portfolio
from scoring.thesis_scorer import score_layer_thesis, DOCS_PER_QUERY
from pricing.history import fetch_recent_closes
from execution.alpaca import execute_sells, execute_buys, get_alpaca_positions, get_tradable_symbols


def _make_retriever(chroma_client):
    """Callable(query) -> docs over the research collection, or None."""
    if chroma_client is None:
        return None
    from chroma_store import query_research_docs

    def _retrieve(query: str) -> list[dict]:
        return query_research_docs(chroma_client, query, n_results=DOCS_PER_QUERY)

    return _retrieve


def _fetch_signal_memory(chroma_client, prior: dict | None) -> list[dict]:
    """Best-effort: the LLM's own recent theses, semantically nearest to the
    prior regime (never fatal — memory is a bonus, not a dependency)."""
    if chroma_client is None:
        return []
    try:
        from chroma_store import query_signal_records
        regime = (prior or {}).get("market_regime") or "balanced"
        query = f"AI infrastructure buildout bottleneck regime {regime}"
        return query_signal_records(chroma_client, query, n_results=3)
    except Exception as exc:
        print(f"  WARNING: signal-memory retrieval failed (non-fatal): {exc}")
        return []


def _remember_thesis(chroma_client, target_id: int, target: dict) -> None:
    """Best-effort write of this thesis into the macro_signals collection."""
    if chroma_client is None or not target.get("thesis_update"):
        return
    try:
        from datetime import datetime, timezone
        from chroma_store import upsert_signal_record
        upsert_signal_record(
            chroma_client,
            f"thesis_{target_id}",
            target["thesis_update"],
            {
                "regime": target.get("market_regime", ""),
                "p_final": float(target.get("signal_confidence", 0.0) or 0.0),
                "computed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    except Exception as exc:
        print(f"  WARNING: thesis memory write failed (non-fatal): {exc}")


def _momentum_by_layer(scores: dict[str, float]) -> dict[str, list]:
    """Rank scored tickers within each layer, best first (prompt context)."""
    return {
        layer: sorted(((t, scores[t]) for t in tickers_in_layer(layer) if t in scores),
                      key=lambda ts: -ts[1])
        for layer in LAYERS
    }


def _broad_movers(scores: dict[str, float], n: int = 20) -> dict[str, list]:
    """Top and bottom whole-market momentum names OUTSIDE the AI layers —
    cross-market context so the LLM can range beyond the AI complex."""
    non_ai = [(t, s) for t, s in scores.items() if t not in LAYER_MAP]
    ranked = sorted(non_ai, key=lambda ts: -ts[1])
    return {"leaders": ranked[:n], "laggards": ranked[-n:][::-1]}


def _momentum_universe() -> list[str]:
    if getattr(config, "WHOLE_MARKET_MOMENTUM", False):
        return sorted(set(config.SP500) | set(LAYER_MAP))
    return sorted(LAYER_MAP)


def compute_weekly_target(docs: list[dict], db_path: str = DB_PATH,
                          thesis_client=None, data_client=None,
                          persist: bool = True, now=None,
                          chroma_client=None, exec_client=None,
                          autonomy: str | None = None) -> dict:
    """Run the thesis -> target pipeline and (optionally) persist the target.

    Guardrailed autonomy: thesis dials -> clamped budgets -> momentum pipeline.
    Full autonomy: the LLM sees the current book + momentum ranks and may
    author `target_weights` directly (validated, normalized); if it stays with
    dials, the same pipeline runs unclamped.
    """
    init_targets_table(db_path)
    prior = get_latest_target(db_path)
    prev_budgets = prior["layer_budgets"] if prior else {}
    if autonomy is None:
        autonomy = getattr(config, "LLM_AUTONOMY", "guardrailed")
    full = autonomy == "full"

    tickers = _momentum_universe()
    prices = fetch_recent_closes(tickers, client=data_client, now=now)
    scores = {}
    if prices is not None and not prices.empty:
        scores = momentum_scores(prices, prices.index[-1])

    whole_market = getattr(config, "WHOLE_MARKET", False)
    valid_symbols = None
    portfolio_context = None
    if full:
        portfolio_context = {
            "positions": get_alpaca_positions(client=exec_client)["longs"],
            "momentum": _momentum_by_layer(scores),
            "broad_movers": _broad_movers(scores) if getattr(
                config, "WHOLE_MARKET_MOMENTUM", False) else None,
            # Whole-market: don't hand the LLM a fixed list — it picks any liquid
            # US equity, validated against tradable symbols below. Curated mode
            # still injects the universe so the model stays inside it.
            "universe": None if whole_market
                        else getattr(config, "TICKER_UNIVERSE", sorted(LAYER_MAP)),
        }
        if whole_market:
            try:
                valid_symbols = get_tradable_symbols(client=exec_client)
            except Exception as e:
                print(f"  WARNING: could not fetch tradable universe ({e}); "
                      f"falling back to ticker-format validation")

    thesis = score_layer_thesis(
        docs,
        prev_budgets=prev_budgets,
        client=thesis_client,
        retriever=_make_retriever(chroma_client),
        signal_memory=_fetch_signal_memory(chroma_client, prior),
        portfolio_context=portfolio_context,
        autonomy=autonomy,
        valid_symbols=valid_symbols,
    )

    if thesis["target_weights_direct"]:
        weights = thesis["target_weights_direct"]
    elif not scores:
        weights = {}
    else:
        adj = thesis["name_adjustments"]
        adj_scores = apply_name_adjustments(scores, adj, clamp=not full) if adj else scores
        weights = assemble_portfolio(
            thesis["layer_budgets"], adj_scores, LAYER_MAP,
            top_n=thesis["layer_top_n"] or 3,
            name_cap=1.0 if full else 0.12,
            clamp_dial=not full,
        ) if adj_scores else {}

    target = {
        "layer_tilt": thesis["layer_tilt"],
        "layer_budgets": thesis["layer_budgets"],
        "target_weights": weights,
        "market_regime": thesis["market_regime"],
        "regime_shift": thesis["regime_shift"],
        "thesis_update": thesis["thesis_update"],
        "layer_top_n": thesis["layer_top_n"],
        "name_adjustments": thesis["name_adjustments"],
        "cash_buffer": thesis["cash_buffer"],
        "rebalance_urgency": thesis["rebalance_urgency"],
        "signal_confidence": thesis["signal_confidence"],
        "autonomy": thesis["autonomy"],
        "weights_source": "llm_direct" if thesis["target_weights_direct"] else "dial_pipeline",
        "retrieval_log": thesis["retrieval_log"],
    }
    # Never persist a degenerate (empty) target: an empty book would become next
    # week's "prior" and would be what a Plan-2b executor reads. Fail loud instead.
    if not weights:
        print("  WARNING: empty target_weights (insufficient price history?) — not persisting")
    elif persist:
        target["id"] = insert_target(db_path, target)
        _remember_thesis(chroma_client, target["id"], target)
    return target


def _decide_trade_gate(target: dict, exec_client) -> str:
    """LLM urgency + mechanical drift bands -> traded / skipped_* decision."""
    urgency = target.get("rebalance_urgency", "normal")
    if urgency == "hold":
        return "skipped_hold"
    if urgency == "urgent":
        return "traded"
    current = get_alpaca_positions(client=exec_client)["longs"]
    if not needs_rebalance(current, target.get("target_weights") or {}, LAYER_MAP):
        return "skipped_within_bands"
    return "traded"


def run_sell(docs: list[dict], db_path: str = DB_PATH, thesis_client=None,
             data_client=None, exec_client=None, cash_buffer: float = 0.0,
             now=None, chroma_client=None, force: bool = False) -> dict:
    """Friday: compute + persist the weekly target, then execute the sell leg.

    Trading is gated: LLM "hold" skips the week outright; "normal" trades only
    when drift breaches the bands; "urgent" always trades. The decision is
    recorded on the target so Monday's buy leg honours it. `force=True`
    (user discretion, e.g. `--mode trade --force`) bypasses the gate.
    """
    target = compute_weekly_target(docs, db_path=db_path, thesis_client=thesis_client,
                                   data_client=data_client, persist=True, now=now,
                                   chroma_client=chroma_client, exec_client=exec_client)
    weights = target.get("target_weights") or {}
    if not weights:
        print("  No target weights — skipping sells")
        return target

    gate = "traded" if force else _decide_trade_gate(target, exec_client)
    target["trade_gate"] = gate
    if target.get("id") is not None:
        update_target_gate(db_path, target["id"], gate)
    if gate != "traded":
        print(f"  Trade gate: {gate} — no sells this week")
        return target

    effective_buffer = max(cash_buffer, target.get("cash_buffer", 0.0))
    execute_sells(weights, cash_buffer=effective_buffer, client=exec_client)
    return target


def run_buy(db_path: str = DB_PATH, exec_client=None, cash_buffer: float = 0.0) -> dict | None:
    """Monday: load the latest persisted target, then execute the buy leg."""
    init_targets_table(db_path)
    target = get_latest_target(db_path)
    if not target or not target.get("target_weights"):
        print("  No persisted target — skipping buys")
        return None
    gate = target.get("trade_gate") or ""
    if gate.startswith("skipped"):
        print(f"  Trade gate on latest target: {gate} — skipping buys")
        return target
    effective_buffer = max(cash_buffer, target.get("cash_buffer", 0.0))
    execute_buys(target["target_weights"], cash_buffer=effective_buffer, client=exec_client)
    return target


def run_trade(docs: list[dict], db_path: str = DB_PATH, thesis_client=None,
              data_client=None, exec_client=None, cash_buffer: float = 0.0,
              now=None, chroma_client=None, force: bool = False) -> dict:
    """On-demand (any day, user discretion): full same-day rebalance.

    Computes + persists the target, executes the sell leg (fill-verified, so
    the freed cash is real), then immediately executes the buy leg from the
    same target — no weekend gap. The trade gate still applies unless `force`.
    """
    target = run_sell(docs, db_path=db_path, thesis_client=thesis_client,
                      data_client=data_client, exec_client=exec_client,
                      cash_buffer=cash_buffer, now=now,
                      chroma_client=chroma_client, force=force)
    weights = target.get("target_weights") or {}
    if weights and target.get("trade_gate") == "traded":
        effective_buffer = max(cash_buffer, target.get("cash_buffer", 0.0))
        execute_buys(weights, cash_buffer=effective_buffer, client=exec_client)
    return target
