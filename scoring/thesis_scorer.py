"""LLM layer-thesis scorer: the model outputs ONLY bounded layer tilts + a regime
narrative. Per-name selection is mechanical (Plan-1 momentum + assembler).
"""
import json
import re

from strategy.layers import LAYERS


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
