"""LLM layer-thesis scorer: the model outputs ONLY bounded layer tilts + a regime
narrative. Per-name selection is mechanical (Plan-1 momentum + assembler).
"""
import json
import os
import re
import time

from strategy.layers import LAYERS, BASELINE_BUDGETS
from strategy.budgets import apply_layer_tilt
from config import GEMINI_MODEL, GEMINI_MAX_OUTPUT_TOKENS


def normalize_tilt(raw_tilt: dict) -> dict[str, float]:
    """Coerce an LLM tilt dict to all five layers summing to exactly 0.0.

    Missing layers default to 0; unknown keys are dropped; the result is
    recentered (subtract the mean) so it is a pure reallocation.
    """
    vals = {layer: float(raw_tilt.get(layer, 0.0)) for layer in LAYERS}
    mean = sum(vals.values()) / len(vals)
    return {layer: v - mean for layer, v in vals.items()}


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

Your ONLY job is to decide how to TILT capital across these five layers relative to a
neutral baseline, based on where the binding bottleneck of the AI buildout is right now.
You do NOT pick individual stocks — a mechanical momentum model selects names within each
layer. Tilts are a reallocation: they should sum to roughly zero. Lean hard when the thesis
is strong; the system will clamp extremes.

Output ONLY valid JSON:
{{
  "layer_tilt": {{"power": <float>, "fabrication": <float>, "compute": <float>,
                 "infrastructure": <float>, "platform": <float>}},
  "market_regime": <"compute_constrained"|"demand_constrained"|"balanced"|"stalling"|"shipping_bottleneck"|"credit_stress">,
  "regime_shift": <bool, true ONLY if the regime/bottleneck changed vs the prior thesis>,
  "signal_confidence": <float 0-1>,
  "thesis_update": <str, 1-3 sentences on the current bottleneck and what changed>
}}"""


def build_thesis_prompt(docs: list[dict], prev_budgets: dict, macro_signal: dict | None) -> str:
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
    parts.append("### RESEARCH SIGNALS")
    for d in docs[:40]:
        parts.append(
            f"[{d.get('source', '?').upper()}] {d.get('title', '')}\n"
            f"{(d.get('content') or '')[:1500]}\n---"
        )
    parts.append("\n[TASK] Output the layer_tilt JSON now. Tilts should sum to ~0.")
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


def score_layer_thesis(docs: list[dict], prev_budgets: dict | None = None,
                       macro_signal: dict | None = None, client=None) -> dict:
    """Run the LLM thesis pass and return guardrailed layer budgets + narrative."""
    if client is None:
        client = _GeminiClient()
    prompt = f"{THESIS_SYSTEM_PROMPT}\n\n{build_thesis_prompt(docs, prev_budgets or {}, macro_signal)}"

    last_exc = None
    raw_text = None
    for attempt in range(3):
        try:
            raw_text = client.generate(prompt)
            parsed = parse_thesis_response(raw_text)
            break
        except Exception as exc:
            last_exc = exc
            time.sleep(2 ** attempt)
    else:
        raise RuntimeError(f"thesis scoring failed after 3 attempts: {last_exc}") from last_exc

    tilt = normalize_tilt(parsed.get("layer_tilt", {}))
    budgets = apply_layer_tilt(BASELINE_BUDGETS, tilt)
    return {
        "layer_tilt": tilt,
        "layer_budgets": budgets,
        "market_regime": parsed.get("market_regime", "balanced"),
        "regime_shift": bool(parsed.get("regime_shift", False)),
        "signal_confidence": float(parsed.get("signal_confidence", 0.5)),
        "thesis_update": parsed.get("thesis_update", ""),
        "raw_response": raw_text,
    }
