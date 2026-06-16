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


def test_score_documents_short_weights_is_none(tmp_path):
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

    assert result["short_weights"] is None


def test_score_documents_queries_chroma_not_sqlite(tmp_path):
    """When chroma_client is provided, scorer must call chroma query."""
    import json
    import pytest
    from unittest.mock import patch, MagicMock
    from scoring.gemini_scorer import score_documents

    chroma_client = MagicMock()

    mock_research = [
        {"id": "1", "title": "NVDA Q1", "content": "NVDA beats", "source": "rss",
         "ticker_mentions": "NVDA", "ingested_at": "2026-05-01", "value_chain_layer": "compute"},
    ]
    mock_signals = [
        {"id": "signal_1", "text": "Thesis intact", "regime": "compute_constrained",
         "p_final": 0.88, "computed_at": "2026-05-01"},
    ]

    valid_output = {
        "p_score": 0.88, "market_regime": "compute_constrained",
        "supply_demand_balance": 0.3,
        "portfolio": [{"ticker": "NVDA", "weight": 0.10, "conviction": 0.9, "reasoning": "GPU demand"}],
        "signal_confidence": 0.8, "thesis_stress": False, "thesis_update": "stable",
    }

    with patch("chroma_store.query_research_docs", return_value=mock_research) as mock_q_docs, \
         patch("chroma_store.query_signal_records", return_value=mock_signals) as mock_q_sigs, \
         patch("chroma_store._embed", return_value=[0.1]*768), \
         patch("scoring.gemini_scorer.genai") as mock_genai:
        mock_genai.Client.return_value.models.generate_content.return_value.text = json.dumps(valid_output)
        result = score_documents(docs=[], chroma_client=chroma_client)

    mock_q_docs.assert_called_once()
    mock_q_sigs.assert_called_once()
    assert result["p_final"] == pytest.approx(0.88)


def test_score_documents_falls_back_to_sqlite_when_chroma_query_fails(tmp_path):
    """A Chroma/embedding failure (e.g. 429) must not kill scoring — fall back to SQLite docs."""
    from db import init_db
    db_path = str(tmp_path / "test.db")
    init_db(db_path)

    chroma_client = MagicMock()

    with patch("chroma_store.query_research_docs",
               side_effect=RuntimeError("429 RESOURCE_EXHAUSTED")), \
         patch("chroma_store.query_signal_records",
               side_effect=RuntimeError("429 RESOURCE_EXHAUSTED")), \
         patch("scoring.gemini_scorer.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.models.generate_content.return_value.text = json.dumps(VALID_GEMINI_OUTPUT)
        mock_client_cls.return_value = mock_client

        result = score_documents(
            docs=SAMPLE_DOCS, db_path=db_path, prev_weights={},
            chroma_client=chroma_client,
        )

    # Scoring still succeeds, using the 3 SQLite fallback docs rather than crashing.
    assert result["p_final"] == 0.82
    assert result["sources_ingested"] == 3
