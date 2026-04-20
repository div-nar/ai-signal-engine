# ai-signal-engine/scoring/gemini_scorer.py
import copy
import json
import os
import re
from collections import defaultdict
from typing import Optional

from google import genai

from config import (
    GEMINI_MODEL, GEMINI_MAX_OUTPUT_TOKENS,
    TICKER_UNIVERSE, MAX_STOCK_WEIGHT, MAX_TURNOVER_VS_PREV,
    WEIGHT_SUM_TOLERANCE, VALUE_CHAIN_LAYERS,
)
from db import DEFAULT_DB


_SYSTEM_PROMPT = """You are a portfolio manager for an AI-focused long-only equity fund.
Your thesis is Aschenbrenner's: AI is on an exponential compute trajectory toward AGI.
Your job is to identify which stocks in the universe will benefit most from the NEXT 1-4
quarters of AI compute expansion.

Universe: {universe}

Constraints:
- Max 10% weight per stock
- All weights must sum to exactly 1.0
- Long-only (no shorts)
- Include at least one Health Care and one Consumer Staples stock as hedges (min 2% each)

Output ONLY valid JSON matching this schema exactly:
{{
  "p_score": <float 0-1, Aschenbrenner probability this week>,
  "market_regime": <"compute_constrained"|"demand_constrained"|"balanced"|"stalling">,
  "supply_demand_balance": <float, positive=demand>supply>,
  "portfolio": [
    {{"ticker": <str>, "weight": <float>, "conviction": <float 0-1>, "reasoning": <str 1-2 sentences>}}
  ],
  "signal_confidence": <float 0-1>,
  "thesis_stress": <bool>,
  "thesis_update": <str, what changed vs last run>
}}"""


def build_signal_context(docs: list[dict], current_portfolio: dict = None) -> str:
    """Assemble documents into a structured prompt context organised by value chain layer."""
    by_layer = defaultdict(list)
    for doc in docs:
        layer = doc.get("value_chain_layer", "application")
        by_layer[layer].append(doc)

    sections = []
    for layer in VALUE_CHAIN_LAYERS:
        layer_docs = by_layer.get(layer, [])
        if not layer_docs:
            continue
        header = f"\n### {layer.upper()} LAYER SIGNALS\n"
        entries = []
        for d in layer_docs:
            entries.append(
                f"Source: {d['source'].upper()} | Date: {d.get('published_at', 'unknown')}\n"
                f"Title: {d['title']}\n"
                f"Content: {d['content'][:2000]}\n"
            )
        sections.append(header + "\n---\n".join(entries))

    # Prepend current portfolio if available
    portfolio_section = ""
    if current_portfolio:
        sorted_positions = sorted(current_portfolio.items(), key=lambda x: -x[1])
        lines = [f"  {ticker}: {weight:.1%}" for ticker, weight in sorted_positions[:20]]
        portfolio_section = (
            "### CURRENT PORTFOLIO POSITIONS\n"
            "You currently hold these positions. Factor them into your recommendations —\n"
            "avoid large rotations unless the signal strongly justifies it.\n"
            + "\n".join(lines)
            + "\n"
        )

    return portfolio_section + "\n".join(sections)


def parse_gemini_response(text: str) -> dict:
    """Parse Gemini response text into a dict, stripping markdown code fences if present."""
    # Try to extract content between ```json ... ``` or ``` ... ``` fences first
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Fall back to stripping any leading/trailing fence and trailing commentary
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```.*$", "", text.strip(), flags=re.DOTALL)
    return json.loads(text)


def apply_guardrails(output: dict, prev_weights: dict) -> dict:
    """Apply hard constraints to Gemini portfolio output."""
    portfolio = copy.deepcopy(output["portfolio"])

    # 1. Cap max weight iteratively, re-normalizing each time until all weights
    #    are within the cap. This handles cases where normalizing pushes some
    #    weights above MAX_STOCK_WEIGHT again.
    _EPS = 1e-6
    for _ in range(50):
        for p in portfolio:
            p["weight"] = min(p["weight"], MAX_STOCK_WEIGHT)
        total = sum(p["weight"] for p in portfolio)
        if total <= 0:
            break
        # Only normalize if more than one stock (single stock stays at capped value)
        if len(portfolio) > 1:
            for p in portfolio:
                p["weight"] = p["weight"] / total
        # Check convergence: all weights within cap
        if all(p["weight"] <= MAX_STOCK_WEIGHT + _EPS for p in portfolio):
            break

    # 2. Apply turnover cap vs previous weights
    if prev_weights:
        for p in portfolio:
            prev = prev_weights.get(p["ticker"], 0.0)
            delta = p["weight"] - prev
            if abs(delta) > MAX_TURNOVER_VS_PREV:
                p["weight"] = prev + MAX_TURNOVER_VS_PREV * (1 if delta > 0 else -1)
        # Re-normalize after turnover cap (only for multi-stock portfolios)
        if len(portfolio) > 1:
            total = sum(p["weight"] for p in portfolio)
            if total > 0:
                for p in portfolio:
                    p["weight"] = p["weight"] / total

    output["portfolio"] = portfolio
    return output


def score_documents(
    docs: list[dict],
    db_path: str = str(DEFAULT_DB),
    prev_weights: Optional[dict] = None,
    current_portfolio: Optional[dict] = None,
) -> dict:
    """Call Gemini with assembled signal context. Returns structured signal dict.

    DB persistence (insert_signal) is the caller's responsibility.
    db_path is reserved for future per-document scoring.
    current_portfolio: live Alpaca positions {ticker: weight}, used as Gemini context
                       and as prev_weights baseline for guardrails if prev_weights is empty.
    """
    if prev_weights is None:
        prev_weights = {}

    # Use live portfolio as guardrail baseline when available
    guardrail_baseline = current_portfolio if current_portfolio else prev_weights

    context = build_signal_context(docs, current_portfolio=current_portfolio)
    universe_str = ", ".join(TICKER_UNIVERSE)
    system = _SYSTEM_PROMPT.format(universe=universe_str)

    user_prompt = f"""Given these forward-looking signals, output portfolio weights for next week.
Weight stocks that will benefit from what is being *committed to* today, not what has already happened.

{context}

[TASK]
Output your portfolio JSON now. Remember: weights must sum to 1.0, max 10% per stock."""

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{system}\n\n{user_prompt}",
        config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
    )

    raw = parse_gemini_response(response.text)

    guarded = apply_guardrails(raw, guardrail_baseline)

    # Postcondition: guardrails must produce valid weights
    final_sum = sum(p["weight"] for p in guarded["portfolio"])
    if abs(final_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        raise ValueError(f"Portfolio weights sum to {final_sum:.4f} after guardrails — malformed Gemini output")

    conviction_map = {p["ticker"]: p["conviction"] for p in guarded["portfolio"]}
    reasoning_map = {p["ticker"]: p["reasoning"] for p in guarded["portfolio"]}
    weight_map = {p["ticker"]: p["weight"] for p in guarded["portfolio"]}

    signal = {
        "p_final": guarded["p_score"],
        "stock_conviction": json.dumps(conviction_map),
        "stock_weights": json.dumps(weight_map),
        "stock_reasoning": json.dumps(reasoning_map),
        "sector_tilt": json.dumps({}),
        "supply_demand_balance": guarded.get("supply_demand_balance", 0.0),
        "market_regime": guarded["market_regime"],
        "signal_confidence": guarded.get("signal_confidence", 0.5),
        "thesis_stress": guarded.get("thesis_stress", False),
        "signal_age_days": 0,
        "sources_ingested": len(docs),
        "signal_breakdown": json.dumps({}),
        "thesis_update": guarded.get("thesis_update", ""),
    }

    return signal
