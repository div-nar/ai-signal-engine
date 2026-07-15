"""LLM layer-thesis scorer with agentic retrieval.

The model directs the portfolio through a bounded decision surface — layer
tilts, per-layer concentration, name emphasis/veto, a cash buffer, and
rebalance urgency — every field clamped by mechanical guardrails and logged in
the persisted target so each power can be ablated later. Before answering it
may search the research vector store with its own follow-up queries (up to
MAX_SEARCH_ROUNDS), so the docs it reasons over are the ones it asked for,
not just the most recent.
"""
import json
import os
import re
import time

from strategy.layers import LAYERS, LAYER_MAP, BASELINE_BUDGETS
from strategy.budgets import apply_layer_tilt
from config import GEMINI_MODEL, GEMINI_MAX_OUTPUT_TOKENS

MAX_SEARCH_ROUNDS = 2      # additional retrieval rounds after the seed context
MAX_QUERIES_PER_ROUND = 3
DOCS_PER_QUERY = 8
CASH_BUFFER_MAX = 0.30


def normalize_tilt(raw_tilt: dict) -> dict[str, float]:
    """Coerce an LLM tilt dict to all five layers summing to exactly 0.0.

    Missing layers default to 0; unknown keys are dropped; the result is
    recentered (subtract the mean) so it is a pure reallocation.
    """
    vals = {layer: float(raw_tilt.get(layer, 0.0)) for layer in LAYERS}
    mean = sum(vals.values()) / len(vals)
    return {layer: v - mean for layer, v in vals.items()}


def sanitize_top_n(raw: dict | None) -> dict[str, int]:
    """Keep only known layers; values are coerced to int (clamped downstream)."""
    if not isinstance(raw, dict):
        return {}
    out = {}
    for layer, n in raw.items():
        if layer not in LAYERS:
            continue
        try:
            out[layer] = int(n)
        except (TypeError, ValueError):
            continue
    return out


def sanitize_name_adjustments(raw: dict | None) -> dict[str, float]:
    """Keep only tickers inside the thesis universe; numeric values only.

    Values are clamped at application time (strategy.factors), but unknown
    tickers are dropped here so the LLM cannot smuggle names into the book.
    """
    if not isinstance(raw, dict):
        return {}
    out = {}
    for ticker, mult in raw.items():
        t = str(ticker).upper()
        if t not in LAYER_MAP:
            continue
        try:
            out[t] = float(mult)
        except (TypeError, ValueError):
            continue
    return out


def sanitize_cash_buffer(raw) -> float:
    try:
        return min(max(float(raw), 0.0), CASH_BUFFER_MAX)
    except (TypeError, ValueError):
        return 0.0


def sanitize_urgency(raw) -> str:
    return raw if raw in ("urgent", "normal", "hold") else "normal"


def parse_thesis_response(text: str) -> dict:
    """Parse the LLM response, stripping markdown code fences if present."""
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```.*$", "", text.strip(), flags=re.DOTALL)
    return json.loads(text)


THESIS_SYSTEM_PROMPT = """You are the macro strategist for an AI-infrastructure long-only fund.
The portfolio is organised as a five-layer value chain ("the cake"):
  power          - grid, generation, electrical gear (the electrons)
  fabrication    - foundry, semicap, EDA, materials (making the silicon)
  compute        - accelerators & memory (the chips)
  infrastructure - datacenters, REITs, cooling, interconnect
  platform       - hyperscalers & software (value capture)

A mechanical momentum model ranks names within each layer. YOU direct the
portfolio through these decisions (each is clamped by hard guardrails):

1. layer_tilt — reallocate capital across the five layers vs. the neutral
   baseline, based on where the binding bottleneck of the AI buildout is.
   Tilts must sum to ~0 (reallocation, not leverage). Lean hard when the
   thesis is strong; the system clamps each layer to [8%, 35%].
2. layer_top_n — concentration per layer: 2 (concentrated, high conviction)
   to 4 (diversified, uncertain). Omit a layer for the default of 3.
3. name_adjustments — OPTIONAL emphasis on specific tickers: a multiplier on
   the momentum score, clamped to [0.5, 1.5], or exactly 0 to veto a name
   this week (e.g. idiosyncratic blowup the price hasn't reflected). Use
   sparingly and only with a concrete catalyst; leave {} to trust momentum.
4. cash_buffer — 0.0 normally. Raise (max 0.30) ONLY on systemic stress
   evident in the research (credit events, funding markets, demand shock).
5. rebalance_urgency — "urgent": trade this week even for small drift;
   "normal": trade only if drift breaches bands; "hold": skip trading this
   week even if bands are breached (thesis unchanged, trading is noise).

RESEARCH: you will be shown seed documents and your own recent thesis
history. If they are insufficient, you may search the research archive
before deciding by replying ONLY with:
{"action": "search", "queries": ["<query 1>", "<query 2>", ...]}
(max MAXQ queries per round, max MAXR search rounds — after that you must decide).

FINAL ANSWER — output ONLY valid JSON:
{
  "layer_tilt": {"power": <float>, "fabrication": <float>, "compute": <float>,
                 "infrastructure": <float>, "platform": <float>},
  "layer_top_n": {"power": <2-4>, ...},
  "name_adjustments": {"<TICKER>": <0 | 0.5-1.5>, ...},
  "cash_buffer": <float 0-0.3>,
  "rebalance_urgency": <"urgent"|"normal"|"hold">,
  "market_regime": <"compute_constrained"|"demand_constrained"|"balanced"|"stalling"|"shipping_bottleneck"|"credit_stress">,
  "regime_shift": <bool, true ONLY if the regime/bottleneck changed vs the prior thesis>,
  "signal_confidence": <float 0-1>,
  "thesis_update": <str, 1-3 sentences on the current bottleneck and what changed>
}""".replace("MAXQ", str(MAX_QUERIES_PER_ROUND)).replace("MAXR", str(MAX_SEARCH_ROUNDS))


def _format_docs(docs: list[dict], limit: int = 40, chars: int = 1500) -> str:
    parts = []
    for d in docs[:limit]:
        parts.append(
            f"[{d.get('source', '?').upper()}] {d.get('title', '')}\n"
            f"{(d.get('content') or '')[:chars]}\n---"
        )
    return "\n".join(parts)


def build_thesis_prompt(docs: list[dict], prev_budgets: dict, macro_signal: dict | None,
                        signal_memory: list[dict] | None = None) -> str:
    """Assemble research + prior state into the user prompt for the thesis pass."""
    parts = []
    if macro_signal:
        parts.append(
            "### MACRO REGIME [quant module — ground truth]\n"
            f"regime: {macro_signal.get('regime')}, "
            f"confidence: {macro_signal.get('regime_confidence')}\n"
            f"notes: {macro_signal.get('notes', '')}\n"
        )
    if prev_budgets:
        parts.append("### PRIOR LAYER BUDGETS\n"
                     + ", ".join(f"{k}={v:.2f}" for k, v in prev_budgets.items()) + "\n")
    if signal_memory:
        parts.append("### YOUR RECENT THESIS HISTORY")
        for s in signal_memory:
            parts.append(f"[{s.get('computed_at', '?')}] regime={s.get('regime', '?')}: "
                         f"{s.get('text', '')}")
        parts.append("")
    parts.append("### RESEARCH SIGNALS (seed)")
    parts.append(_format_docs(docs))
    parts.append("\n[TASK] Search the archive if you need more evidence, otherwise "
                 "output the final thesis JSON now. Tilts should sum to ~0.")
    return "\n".join(parts)


class _GeminiClient:
    """Default production client: thin wrapper exposing .generate(prompt) -> str."""

    def __init__(self):
        from google import genai
        self._client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    def generate(self, prompt: str) -> str:
        resp = self._client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS},
        )
        return resp.text


def _generate_parsed(client, prompt: str) -> tuple[dict, str]:
    """Call the LLM with retries; return (parsed_json, raw_text)."""
    last_exc = None
    for attempt in range(3):
        try:
            raw_text = client.generate(prompt)
            return parse_thesis_response(raw_text), raw_text
        except Exception as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"thesis scoring failed after 3 attempts: {last_exc}") from last_exc


def score_layer_thesis(docs: list[dict], prev_budgets: dict | None = None,
                       macro_signal: dict | None = None, client=None,
                       retriever=None, signal_memory: list[dict] | None = None) -> dict:
    """Run the (optionally agentic) LLM thesis pass and return guardrailed output.

    `retriever` is a callable(query: str) -> list[doc dict] over the vector
    store. When provided, the model may issue up to MAX_SEARCH_ROUNDS rounds of
    follow-up queries before its final answer; when None the search action is
    simply never honoured and the first parseable answer must be the thesis.
    Every query round is recorded in `retrieval_log` for later ablation.
    """
    if client is None:
        client = _GeminiClient()
    prompt = f"{THESIS_SYSTEM_PROMPT}\n\n{build_thesis_prompt(docs, prev_budgets or {}, macro_signal, signal_memory)}"

    retrieval_log = []
    seen_ids = {d.get("id") for d in docs if d.get("id")}
    parsed, raw_text = _generate_parsed(client, prompt)

    rounds = 0
    while parsed.get("action") == "search" and rounds < MAX_SEARCH_ROUNDS:
        rounds += 1
        queries = [str(q) for q in (parsed.get("queries") or [])][:MAX_QUERIES_PER_ROUND]
        retrieved = []
        if retriever is not None:
            for q in queries:
                try:
                    hits = retriever(q) or []
                except Exception as exc:
                    print(f"  WARNING: retrieval failed for {q!r}: {exc}")
                    hits = []
                fresh = [d for d in hits if d.get("id") not in seen_ids]
                seen_ids.update(d.get("id") for d in fresh if d.get("id"))
                retrieved.extend(fresh)
                retrieval_log.append({"round": rounds, "query": q, "hits": len(fresh)})
        else:
            retrieval_log.append({"round": rounds, "query": "|".join(queries),
                                  "hits": 0, "note": "no retriever available"})
        remaining = MAX_SEARCH_ROUNDS - rounds
        prompt += (
            f"\n\n### RETRIEVED DOCS (round {rounds}"
            f", {remaining} search round(s) left)\n"
            + (_format_docs(retrieved) if retrieved else "(no new documents found)")
            + "\n\n[TASK] Search again only if still insufficient; otherwise output "
              "the final thesis JSON now."
        )
        parsed, raw_text = _generate_parsed(client, prompt)

    if parsed.get("action") == "search":
        # Model tried to search past its allowance — force a final answer.
        prompt += ("\n\n[TASK] Search budget exhausted. Output the final thesis "
                   "JSON now using the evidence above.")
        parsed, raw_text = _generate_parsed(client, prompt)

    tilt = normalize_tilt(parsed.get("layer_tilt", {}))
    budgets = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    return {
        "layer_tilt": tilt,
        "layer_budgets": budgets,
        "layer_top_n": sanitize_top_n(parsed.get("layer_top_n")),
        "name_adjustments": sanitize_name_adjustments(parsed.get("name_adjustments")),
        "cash_buffer": sanitize_cash_buffer(parsed.get("cash_buffer", 0.0)),
        "rebalance_urgency": sanitize_urgency(parsed.get("rebalance_urgency")),
        "market_regime": parsed.get("market_regime", "balanced"),
        "regime_shift": bool(parsed.get("regime_shift", False)),
        "signal_confidence": float(parsed.get("signal_confidence", 0.5)),
        "thesis_update": parsed.get("thesis_update", ""),
        "retrieval_log": retrieval_log,
        "raw_response": raw_text,
    }
