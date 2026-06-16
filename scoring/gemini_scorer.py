# ai-signal-engine/scoring/gemini_scorer.py
import copy
import json
import os
import re
import time
from collections import defaultdict
from typing import Optional

from google import genai

from config import (
    GEMINI_MODEL, GEMINI_MAX_OUTPUT_TOKENS,
    TICKER_UNIVERSE, MAX_STOCK_WEIGHT, MAX_TURNOVER_VS_PREV,
    WEIGHT_SUM_TOLERANCE, VALUE_CHAIN_LAYERS,
)
from db import DEFAULT_DB


_SYSTEM_PROMPT = """You are a portfolio manager for an AI infrastructure long-only equity fund.
Your thesis is Aschenbrenner's: AI is on an exponential compute trajectory toward AGI,
driven by a physical buildout supercycle — chips, memory, power, cooling, datacenters, networking.
Your job is to identify which stocks will benefit most from the NEXT 1-4 quarters of AI
infrastructure spending: capex commitments, supply chain bottlenecks, and compute expansion.

Focus on BOTH sides of the AI buildout — supply bottlenecks AND demand-side compute consumers:

SUPPLY-SIDE (physical bottlenecks the buildout is constrained by):
- Compute: GPUs, ASICs, networking chips, foundry, HBM memory, advanced packaging
- Power & cooling: utilities, thermal management, grid equipment, power electronics
- Infrastructure: datacenter REITs, interconnects, fiber, structural components

DEMAND-SIDE (hyperscalers and platforms whose earnings scale with AI compute consumption):
- Hyperscalers driving the capex cycle: their own AI revenue (Cloud AI, model APIs, AI ad/search uplift)
- AI software platforms whose revenue scales directly with compute consumption
- Companies monetising AI products at scale (Gemini, Copilot, ChatGPT-class APIs, AI ad units)

Universe (you may pick any publicly traded stock globally): {universe}

PORTFOLIO STRUCTURE:
- Long book (portfolio): highest-conviction AI buildout beneficiaries. Weights sum to 1.0. Max 10% per stock.
- Every position must be directly tied to AI buildout.

Output ONLY valid JSON matching this schema exactly:
{{
  "p_score": <float 0-1, Aschenbrenner probability this week>,
  "market_regime": <"compute_constrained"|"demand_constrained"|"balanced"|"stalling"|"shipping_bottleneck"|"credit_stress">,
  "supply_demand_balance": <float, positive=demand>supply>,
  "portfolio": [
    {{"ticker": <str>, "weight": <float>, "conviction": <float 0-1>, "reasoning": <str 1-2 sentences>}}
  ],
  "signal_confidence": <float 0-1>,
  "thesis_stress": <bool>,
  "thesis_update": <str, what changed vs last run>
}}"""


def build_signal_context(
    docs: list[dict],
    current_portfolio: dict = None,
    macro_signal: dict = None,
) -> str:
    """Assemble documents into structured prompt context organised by value chain layer."""
    sections = []

    # Prepend macro regime block if available
    if macro_signal:
        sc = macro_signal.get("supply_chain", {})
        cs = macro_signal.get("cross_sector", {})
        macro_block = (
            "### MACRO REGIME SIGNAL [computed by quant module — treat as ground truth]\n"
            f"Regime: {macro_signal['regime']} (confidence: {macro_signal['regime_confidence']:.2f})\n"
            f"net_exposure_target: {macro_signal['net_exposure_target']:.2f} "
            f"({int(macro_signal['net_exposure_target']*100)}% long / "
            f"{int((1-macro_signal['net_exposure_target'])*100)}% short notional)\n"
            f"Supply chain: PMI {sc.get('pmi', 'N/A')} ({sc.get('pmi_trend', 'N/A')}), "
            f"shipping pressure {sc.get('shipping_pressure', 'N/A'):.2f}, "
            f"semis inventory {sc.get('semis_inventory_trend', 'N/A')}\n"
            f"Cross-sector: power→compute lead {cs.get('power_compute_lead', 0):.1f}σ, "
            f"copper→infra {cs.get('copper_infra_lead', 0):.1f}σ, "
            f"credit stress: {cs.get('credit_stress', False)}, "
            f"VIX {cs.get('vix_level', 'N/A'):.1f}\n"
            f"Notes: {macro_signal.get('notes', '')}\n\n"
            "[Your portfolio must reflect the net_exposure_target above]\n"
        )
        sections.append(macro_block)

    # Current portfolio context
    if current_portfolio:
        sorted_positions = sorted(current_portfolio.items(), key=lambda x: -x[1])
        lines = [f"  {ticker}: {weight:.1%}" for ticker, weight in sorted_positions[:20]]
        sections.append(
            "### CURRENT PORTFOLIO POSITIONS\n"
            "You currently hold these positions. Factor them into your recommendations —\n"
            "avoid large rotations unless the signal strongly justifies it.\n"
            + "\n".join(lines) + "\n"
        )

    # Documents by value chain layer
    by_layer = defaultdict(list)
    for doc in docs:
        layer = doc.get("value_chain_layer", "application")
        by_layer[layer].append(doc)

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

    return "\n".join(sections)


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
    """Apply hard constraints to Gemini long and short portfolio output."""
    output = copy.deepcopy(output)

    # ── Long book: cap + normalize ────────────────────────────────────────
    portfolio = output["portfolio"]
    _EPS = 1e-6
    for _ in range(50):
        for p in portfolio:
            p["weight"] = min(p["weight"], MAX_STOCK_WEIGHT)
        total = sum(p["weight"] for p in portfolio)
        if total <= 0:
            break
        if len(portfolio) > 1:
            for p in portfolio:
                p["weight"] = p["weight"] / total
        if all(p["weight"] <= MAX_STOCK_WEIGHT + _EPS for p in portfolio):
            break
    output["portfolio"] = portfolio

    return output


def score_documents(
    docs: list[dict],
    db_path: str = str(DEFAULT_DB),
    prev_weights: Optional[dict] = None,
    current_portfolio: Optional[dict] = None,
    macro_signal: Optional[dict] = None,
    chroma_client=None,
) -> dict:
    """Call Gemini with assembled signal context. Returns structured signal dict."""
    if prev_weights is None:
        prev_weights = {}

    # ── Document retrieval ────────────────────────────────────────────────────
    # Semantic retrieval via ChromaDB is an enhancement, not a hard dependency.
    # The query embedding is a live Gemini call that can fail (429, timeout). On
    # any failure, fall back to the SQLite `docs` passed in rather than letting a
    # retrieval hiccup take down the whole scoring run (which would mean no signal
    # that day). past_signals is best-effort and simply omitted on failure.
    past_signals = []
    if chroma_client is not None:
        from chroma_store import query_research_docs, query_signal_records
        regime_label = (macro_signal or {}).get("regime", "compute_constrained")
        query = (
            f"AI infrastructure buildout regime:{regime_label} "
            "semiconductor GPU power datacenter capex supply chain"
        )
        try:
            docs = query_research_docs(chroma_client, query, n_results=30)
            past_signals = query_signal_records(chroma_client, query, n_results=3)
        except Exception as e:
            print(f"  WARNING: ChromaDB semantic retrieval failed, falling back to "
                  f"SQLite recency docs: {e}")
            past_signals = []

    guardrail_baseline = current_portfolio if current_portfolio else prev_weights
    context = build_signal_context(docs, current_portfolio=current_portfolio, macro_signal=macro_signal)

    if past_signals:
        past_block = "### RECENT SIGNAL HISTORY\n" + "\n".join(
            f"[{s['computed_at']}] regime={s['regime']} p={s['p_final']:.2f}: {s['text']}"
            for s in past_signals
        ) + "\n\n"
        context = past_block + context
    universe_str = ", ".join(TICKER_UNIVERSE)
    system = _SYSTEM_PROMPT.format(universe=universe_str)

    user_prompt = f"""Given these forward-looking signals, output portfolio weights for next week.
Weight stocks that will benefit from what is being *committed to* today, not what has already happened.

{context}

[TASK]
Output your portfolio JSON now. Long weights must sum to 1.0, max 10% per stock."""

    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    last_exc = None
    raw = None
    raw_response_text = None
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=f"{system}\n\n{user_prompt}",
                config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
            )
            raw_response_text = response.text
            raw = parse_gemini_response(response.text)
            break
        except Exception as exc:
            last_exc = exc
            wait = 2 ** attempt
            print(f"  Gemini call failed (attempt {attempt + 1}/3): {exc}. Retrying in {wait}s...")
            time.sleep(wait)
    if raw is None:
        raise RuntimeError(f"Gemini failed after 3 attempts: {last_exc}") from last_exc

    guarded = apply_guardrails(raw, guardrail_baseline)

    final_sum = sum(p["weight"] for p in guarded["portfolio"])
    if abs(final_sum - 1.0) > WEIGHT_SUM_TOLERANCE:
        # Last-resort normalization (e.g. single-stock portfolio from mock/chroma)
        if final_sum > 0:
            for p in guarded["portfolio"]:
                p["weight"] = p["weight"] / final_sum
        else:
            raise ValueError(f"Long portfolio weights sum to {final_sum:.4f} after guardrails")

    conviction_map = {p["ticker"]: p["conviction"] for p in guarded["portfolio"]}
    reasoning_map = {p["ticker"]: p["reasoning"] for p in guarded["portfolio"]}
    weight_map = {p["ticker"]: p["weight"] for p in guarded["portfolio"]}

    doc_ids = [d["id"] for d in docs if "id" in d]

    return {
        "p_final": guarded["p_score"],
        "stock_conviction": json.dumps(conviction_map),
        "stock_weights": json.dumps(weight_map),
        "stock_reasoning": json.dumps(reasoning_map),
        "short_weights": None,
        "macro_signal": json.dumps(macro_signal) if macro_signal else None,
        "sector_tilt": json.dumps({}),
        "supply_demand_balance": guarded.get("supply_demand_balance", 0.0),
        "market_regime": guarded["market_regime"],
        "signal_confidence": guarded.get("signal_confidence", 0.5),
        "thesis_stress": guarded.get("thesis_stress", False),
        "signal_age_days": 0,
        "sources_ingested": len(docs),
        "signal_breakdown": json.dumps({}),
        "thesis_update": guarded.get("thesis_update", ""),
        "raw_response": raw_response_text,
        "prompt_context_doc_ids": json.dumps(doc_ids),
    }
