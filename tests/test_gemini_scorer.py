# ai-signal-engine/tests/test_gemini_scorer.py
import json
import pytest
from unittest.mock import patch, MagicMock
from scoring.gemini_scorer import (
    build_signal_context,
    parse_gemini_response,
    apply_guardrails,
    score_documents,
)

SAMPLE_DOCS = [
    {
        "id": 1, "source": "rss", "title": "ASML Q1 Backlog Surge",
        "content": "ASML reported record order backlog of 12B EUR in Q1 2026.",
        "published_at": "2026-04-15", "value_chain_layer": "compute",
    },
    {
        "id": 2, "source": "edgar", "title": "MSFT 8-K 2026-04-10",
        "content": "Microsoft expects to invest $80B in AI infrastructure in FY2026.",
        "published_at": "2026-04-10", "value_chain_layer": "infrastructure",
    },
    {
        "id": 3, "source": "arxiv", "title": "Scaling Laws Updated",
        "content": "Compute optimal training now requires 10^26 FLOPs for frontier models.",
        "published_at": "2026-04-12", "value_chain_layer": "platform",
    },
]

VALID_GEMINI_OUTPUT = {
    "p_score": 0.82,
    "market_regime": "compute_constrained",
    "supply_demand_balance": 0.34,
    "portfolio": [
        {"ticker": "NVDA", "weight": 0.10, "conviction": 0.90, "reasoning": "ASML backlog signals GPU supply tightening in 2-3Q."},
        {"ticker": "ASML", "weight": 0.09, "conviction": 0.85, "reasoning": "Record backlog directly measures EUV capacity expansion."},
        {"ticker": "TSM", "weight": 0.08, "conviction": 0.80, "reasoning": "Advanced node utilization rising as hyperscaler orders accelerate."},
        {"ticker": "MU", "weight": 0.07, "conviction": 0.75, "reasoning": "HBM3E demand from AI training runs increasing QoQ."},
        {"ticker": "MSFT", "weight": 0.07, "conviction": 0.78, "reasoning": "$80B CapEx signals multi-quarter GPU cluster buildout."},
        {"ticker": "AMZN", "weight": 0.06, "conviction": 0.72, "reasoning": "AWS Trainium2 ramp indicates sustained AI workload growth."},
        {"ticker": "GOOGL", "weight": 0.06, "conviction": 0.70, "reasoning": "TPU v5 deployment scaling; Gemini inference at scale."},
        {"ticker": "AMD", "weight": 0.05, "conviction": 0.68, "reasoning": "MI300X gaining share in inferencing segment."},
        {"ticker": "AVGO", "weight": 0.05, "conviction": 0.65, "reasoning": "Custom ASIC revenue from hyperscalers accelerating."},
        {"ticker": "ANET", "weight": 0.04, "conviction": 0.62, "reasoning": "400G/800G switch demand tied to GPU cluster networking."},
        {"ticker": "LLY", "weight": 0.03, "conviction": 0.45, "reasoning": "AI drug discovery pipeline expanding."},
        {"ticker": "COST", "weight": 0.02, "conviction": 0.20, "reasoning": "Hedge: consumer staples floor."},
    ],
    "signal_confidence": 0.76,
    "thesis_stress": False,
    "thesis_update": "ASML order backlog up 18% QoQ signals compute expansion continues.",
}


def test_build_signal_context_groups_by_layer():
    context = build_signal_context(SAMPLE_DOCS)
    assert "compute" in context.lower()
    assert "infrastructure" in context.lower()
    assert "platform" in context.lower()
    assert "ASML Q1 Backlog Surge" in context
    assert "MSFT 8-K" in context


def test_parse_gemini_response_valid():
    result = parse_gemini_response(json.dumps(VALID_GEMINI_OUTPUT))
    assert result["p_score"] == 0.82
    assert result["market_regime"] == "compute_constrained"
    assert len(result["portfolio"]) == 12


def test_parse_gemini_response_strips_markdown():
    wrapped = f"```json\n{json.dumps(VALID_GEMINI_OUTPUT)}\n```"
    result = parse_gemini_response(wrapped)
    assert result["p_score"] == 0.82


def test_apply_guardrails_caps_max_weight():
    output = dict(VALID_GEMINI_OUTPUT)
    output["portfolio"] = [{"ticker": "NVDA", "weight": 0.25, "conviction": 0.9, "reasoning": "x"}]
    guarded = apply_guardrails(output, prev_weights={})
    nvda = next(p for p in guarded["portfolio"] if p["ticker"] == "NVDA")
    assert nvda["weight"] <= 0.10


def test_apply_guardrails_weights_sum_to_one():
    output = dict(VALID_GEMINI_OUTPUT)
    guarded = apply_guardrails(output, prev_weights={})
    total = sum(p["weight"] for p in guarded["portfolio"])
    assert abs(total - 1.0) < 0.01


def test_score_documents_calls_gemini_and_returns_structured(tmp_path):
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_GEMINI_OUTPUT)

    with patch("scoring.gemini_scorer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = score_documents(docs=SAMPLE_DOCS, db_path=db_path, prev_weights={})

    assert result["p_final"] == 0.82
    assert result["market_regime"] == "compute_constrained"
    assert "NVDA" in result["stock_conviction"]
    assert result["sources_ingested"] == 3


_MACRO_SIGNAL = {
    "regime": "shipping_bottleneck",
    "regime_confidence": 0.82,
    "net_exposure_target": 0.55,
    "supply_chain": {
        "shipping_pressure": 0.74, "semis_inventory_trend": "drawing_down",
        "pmi": 49.2, "pmi_trend": "contracting",
    },
    "cross_sector": {
        "power_compute_lead": 1.3, "copper_infra_lead": -0.2,
        "credit_stress": False, "vix_level": 18.4,
    },
    "notes": "Freight pressure elevated but credit clean.",
}

VALID_GEMINI_OUTPUT_V2 = {
    "p_score": 0.91,
    "market_regime": "shipping_bottleneck",
    "supply_demand_balance": 0.3,
    "portfolio": [
        {"ticker": "NVDA", "weight": 0.15, "conviction": 0.95, "reasoning": "GPU supply tight."},
        {"ticker": "MU",   "weight": 0.12, "conviction": 0.90, "reasoning": "HBM demand rising."},
        {"ticker": "TSM",  "weight": 0.10, "conviction": 0.88, "reasoning": "Advanced node full."},
        {"ticker": "VRT",  "weight": 0.10, "conviction": 0.85, "reasoning": "Cooling bottleneck."},
        {"ticker": "CEG",  "weight": 0.10, "conviction": 0.83, "reasoning": "Power contracts."},
        {"ticker": "AVGO", "weight": 0.09, "conviction": 0.80, "reasoning": "Custom ASIC ramp."},
        {"ticker": "AMZN", "weight": 0.08, "conviction": 0.78, "reasoning": "AWS capex cycle."},
        {"ticker": "MSFT", "weight": 0.08, "conviction": 0.75, "reasoning": "Azure AI revenue."},
        {"ticker": "META", "weight": 0.08, "conviction": 0.72, "reasoning": "Llama infra spend."},
        {"ticker": "PWR",  "weight": 0.10, "conviction": 0.70, "reasoning": "Grid build."},
    ],
    "short_portfolio": [
        {"ticker": "AMD",  "weight": 0.40, "conviction": 0.42, "reasoning": "Lower GPU conviction vs NVDA."},
        {"ticker": "QCOM", "weight": 0.35, "conviction": 0.38, "reasoning": "Mobile-heavy, low AI infra exposure."},
        {"ticker": "ON",   "weight": 0.25, "conviction": 0.35, "reasoning": "EV exposure, not AI buildout."},
    ],
    "net_exposure": 0.54,
    "signal_confidence": 0.88,
    "thesis_stress": False,
    "thesis_update": "Shipping elevated — rotate to bottleneck names.",
}


def test_build_signal_context_includes_macro_block():
    context = build_signal_context(SAMPLE_DOCS, macro_signal=_MACRO_SIGNAL)
    assert "MACRO REGIME SIGNAL" in context
    assert "shipping_bottleneck" in context
    assert "net_exposure_target: 0.55" in context
    assert "Freight pressure elevated" in context


def test_build_signal_context_macro_precedes_documents():
    context = build_signal_context(SAMPLE_DOCS, macro_signal=_MACRO_SIGNAL)
    macro_pos = context.index("MACRO REGIME SIGNAL")
    doc_pos = context.index("ASML Q1 Backlog Surge")
    assert macro_pos < doc_pos


def test_apply_guardrails_caps_short_weight():
    output = dict(VALID_GEMINI_OUTPUT_V2)
    output["short_portfolio"] = [
        {"ticker": "AMD", "weight": 0.50, "conviction": 0.4, "reasoning": "x"},
    ]
    guarded = apply_guardrails(output, prev_weights={})
    amd = next(p for p in guarded["short_portfolio"] if p["ticker"] == "AMD")
    assert amd["weight"] <= 0.08


def test_apply_guardrails_short_book_sums_to_one():
    output = dict(VALID_GEMINI_OUTPUT_V2)
    guarded = apply_guardrails(output, prev_weights={})
    total = sum(p["weight"] for p in guarded["short_portfolio"])
    assert abs(total - 1.0) < 0.01


def test_score_documents_returns_short_weights(tmp_path):
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    mock_response = MagicMock()
    mock_response.text = json.dumps(VALID_GEMINI_OUTPUT_V2)

    with patch("scoring.gemini_scorer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        mock_client_cls.return_value = mock_client

        result = score_documents(
            docs=SAMPLE_DOCS, db_path=db_path, prev_weights={},
            macro_signal=_MACRO_SIGNAL,
        )

    assert result["short_weights"] is not None
    short = json.loads(result["short_weights"])
    assert "AMD" in short
    assert abs(sum(short.values()) - 1.0) < 0.01
