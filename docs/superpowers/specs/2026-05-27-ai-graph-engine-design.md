# AI Graph Engine — Design Spec

## Goal

Build a standalone system that learns the causal structure of the AI infrastructure buildout, predicts where bottlenecks migrate next and which macro signals are about to inflect, and produces structured outputs for downstream consumption. The outputs are designed to eventually enrich ai-signal-engine v1's Gemini prompt and, later, replace the PCA composite modifier — but v1 runs unchanged in Phase 1.

## Architecture overview

Three logical stages running in sequence:

```
EXTRACTION          GRAPH              GNN
─────────────       ──────────────     ─────────────────────────────
arXiv API      ─►  Importance gate ─► Growing HeteroGraph (SQLite)
EDGAR          ─►  Canonicalize    ─► Weekly PyG snapshot (.pt)    ─► HGT encoder
HF Papers      ─►  Gemini extract  ─►                               ─► LSTM temporal
yfinance/FRED  ──────────────────────► Signal node attributes       ─► Two heads
                                                                         ├── Bottleneck head
                                                                         └── Signal forecast head
                                                                              │
                                                                         gnn_signal.json
                                                                         (v1 reads later)
```

---

## Data sources

Clean programmatic access only.

### Literature corpus (technology/bottleneck nodes)

| Source | API | Scope | Start |
|---|---|---|---|
| arXiv | `arxiv` Python library | cs.LG, cs.DC, cs.AR, cs.NE, cs.AI | 2017-01-01 |
| SEC EDGAR | `https://data.sec.gov` (same as v1) | 8-Ks: MSFT, AMZN, GOOGL, META, NVDA, TSM, ASML, AMAT, LRCX, KLAC, AMD, AVGO | 2017-01-01 |
| HuggingFace Papers | `https://huggingface.co/api/daily_papers` | all | 2020-01-01 |

### Market data (signal nodes)

| Source | API | Series |
|---|---|---|
| yfinance | `yfinance.download` | FCX, ALB, URA, UNG, SMH, XLU, EQIX, BDRY, HYG, LQD, ^VIX, ^MOVE |
| FRED | `fredapi.Fred` | DGS2, DGS10, DGORDER, IPG3344S |

---

## Node taxonomy

Two classes of nodes with different lifecycles.

### Class 1 — Financial signal nodes (fixed set, ~25 nodes)

Updated daily from yfinance/FRED. No gate. Topology never changes.

```
Rates & credit:
  US2Y          DGS2 from FRED
  US10Y         DGS10 from FRED
  term_spread   US10Y - US2Y (computed)
  HYG           iShares HY corporate bond ETF
  LQD           iShares IG corporate bond ETF
  credit_spread HYG yield - LQD yield (computed)
  DXY           US Dollar Index (via UUP ETF proxy)

Volatility:
  VIX           CBOE VIX
  MOVE          ICE BofA MOVE Index (bond vol, via ^MOVE)
  SMH_realized  30d realized vol on SMH (computed from prices)

Commodities:
  copper        FCX as copper proxy
  lithium       ALB as lithium proxy
  uranium       URA ETF
  nat_gas       UNG ETF

Sector proxies:
  semis         SMH ETF
  utilities     XLU ETF
  datacenter    EQIX as datacenter proxy
  shipping      BDRY ETF

Economic:
  pmi_proxy     DGORDER from FRED (same as v1 supply chain module)
  semis_ipi     IPG3344S from FRED (semiconductor industrial production)
```

Node attribute at each timestep: `{value: float, 30d_change: float, 90d_zscore: float}`.

### Class 2 — Technology/bottleneck nodes (growing, ~80–500+ nodes)

Seeded from corpus at launch; new nodes admitted via importance gate as documents arrive. Organized by layer and sublayer. Initial seed catalog below — Gemini will expand this.

```
COMPUTE / A — Silicon
  Training accelerators
    NVDA_H100, NVDA_H200, NVDA_B200, NVDA_GB200
    Google_TPUv5, AWS_Trainium2, Meta_MTIA2, MSFT_Maia100
    AMD_MI300X, AMD_MI325X
  Inference accelerators
    NVDA_L40S, Intel_Gaudi3, Qualcomm_CloudAI100
  Foundry capacity
    TSMC_N3P, TSMC_N2, TSMC_N2P, Samsung_SF2, Intel_18A
  Advanced packaging
    CoWoS_S, CoWoS_L, SoIC_XT, FOPLP, HBM_interposer
  EDA / IP
    Synopsys_Fusion, Cadence_Innovus, ARM_Cortex_X

COMPUTE / B — Memory & interconnect
  HBM generations
    HBM2e, HBM3, HBM3e, HBM4, HBM4E
  High-speed networking
    InfiniBand_NDR, InfiniBand_XDR, Ethernet_400G, Ethernet_800G
  On-chip fabric
    NVLink4, NVLink5, NVSwitch3
  Optical interconnect
    CPO_silicon_photonics, VCSEL_array

POWER / A — Generation
  SMR_nuclear          (NuScale, Kairos, Oklo)
  Large_nuclear        (existing fleet restarts — Constellation, Vistra)
  Gas_peakers          (proximity datacenter siting)
  Solar_utility        (utility-scale PV + BESS)
  Grid_interconnect    (queue position, lead times)

POWER / B — Delivery & thermal
  Distribution_transformers   (2–3yr lead time constraint)
  Switchgear_PDU
  Direct_liquid_cooling       (DLC cold plates)
  Rear_door_heat_exchangers
  Immersion_single_phase
  Immersion_two_phase

INFRASTRUCTURE / A — Facilities
  Hyperscaler_Azure_campus
  Hyperscaler_AWS_campus
  Hyperscaler_Google_campus
  Colo_Equinix
  Colo_DigitalRealty
  Sovereign_EU_AI_factories
  Sovereign_UAE_UAEAN
  Land_permitting              (zoning, water rights, grid access)

INFRASTRUCTURE / B — Connectivity
  Subsea_MAREA
  Subsea_Pacific_Light
  Dark_fiber_IRU
  Backbone_Cisco_8000
  Backbone_Juniper_PTX
  Satellite_Starlink

PLATFORM / A — Cloud AI services
  Azure_AI_Foundry
  AWS_Bedrock
  Google_Vertex_AI
  CoreWeave_GPU_cloud
  Lambda_Labs
  Weights_and_Biases
  MLflow

PLATFORM / B — Foundation models & APIs
  GPT4o, GPT5
  Gemini_Ultra_2, Gemini_Flash
  Claude_3_Opus, Claude_4
  Llama_3_405B, Llama_4
  vLLM, TensorRT_LLM, SGLang

APPLICATION / A — AI-native products
  M365_Copilot
  Google_Workspace_AI
  GitHub_Copilot
  Cursor_IDE
  ChatGPT_consumer
  Gemini_consumer

APPLICATION / B — Vertical AI
  Code_gen_deployment     (CI/CD integration tier)
  Healthcare_diagnostics
  Drug_discovery_AI
  Physical_AI_robotics
  Finance_legal_AI

DOMAIN / A — Research frontier
  Scaling_laws_pretraining
  Scaling_laws_posttraining
  Inference_scaling_TTC       (test-time compute)
  MoE_architectures
  SSM_architectures           (Mamba, etc.)
  Lab_cluster_Eagle           (MSFT/OpenAI)
  Lab_cluster_Colossus         (xAI)
  Lab_cluster_Eos              (NVDA)

DOMAIN / B — Policy & constraints
  BIS_export_controls          (Entity List, tier system)
  H20_chip_restrictions
  EU_AI_Act
  US_AI_EO
  MLCommons_benchmarks
```

---

## Edge types

All edges are time-stamped with `reported_at` (document date) and optionally `projected_at` (when the effect materializes — extracted by Gemini or imputed from category defaults).

| Edge type | Src → Dst | Key attributes |
|---|---|---|
| `supplies` | Company/Tech → Company/Tech | `capacity_fraction`, `constraint_level [0–1]`, `lag_quarters` |
| `constrained_by` | Tech/Infra → Tech | `severity [0–1]`, `duration_estimate_quarters` |
| `enables` | Tech → Tech | `maturity [0–1]`, `dependency_type: hard\|soft` |
| `invests_in` | CapexEvent → Tech/Infra | `amount_bn`, `lag_quarters`, `confidence` |
| `competes_with` | Tech → Tech | `overlap_fraction` |
| `granger_causes` | Signal → Signal | `lag_weeks`, `coefficient`, `p_value` — learned, not extracted |
| `signal_affects` | Signal → Tech | `direction: positive\|negative`, `lag_weeks` |
| `tech_signals` | Tech → Signal | `direction`, `lag_weeks` — e.g., HBM bottleneck → BDRY spike |
| `mentions` | Document → Node | `sentiment [−1,1]`, `constraint_language: bool` |

`granger_causes` edges are computed offline from the 2017–present signal time series using a sparse Granger-causality test (NOTEARS or pairwise Granger with FDR correction). They are fixed after the initial fit — they don't go through the importance gate.

---

## Importance gate

Every candidate entity Gemini extracts from a document passes through two stages before touching the graph.

### Stage 1 — Canonicalization

Fuzzy match (token sort ratio ≥ 85) against existing node `canonical_name` values. If a match is found, the entity resolves to the existing node and updates its `last_seen` and edge weights — it does not create a new node. This prevents graph bloat from product name variants ("CoWoS-L", "cowos large", "TSMC CoWoS Large").

### Stage 2 — Importance scoring

Gemini call with structured output schema:

```json
{
  "canonical_name": "TSMC CoWoS-L N3P",
  "is_novel": true,
  "layer": "compute",
  "sublayer": "A",
  "category": "Advanced packaging",
  "parent_node": "CoWoS_L",
  "importance_score": 0.84,
  "rationale": "Primary packaging substrate for NVDA GB200; supply constraint cited in recent documents",
  "constraint_language_present": true,
  "admit": true
}
```

**Admission rules:**
- `importance_score > 0.55` AND `is_novel = true` → add as new node, inherit parent embedding for warm start
- `importance_score ≤ 0.55` → entity is recorded but not added as a node; its mentions increment edge weights on adjacent existing nodes
- `is_novel = false` → canonicalization should have caught this; log as pipeline warning

---

## Graph store

SQLite database at `data/graph.db`. Four tables.

```sql
CREATE TABLE nodes (
    node_id       TEXT PRIMARY KEY,   -- slugified canonical_name
    canonical_name TEXT NOT NULL,
    node_class    TEXT NOT NULL,      -- 'signal' | 'technology'
    layer         TEXT,               -- compute|power|infrastructure|platform|application|domain
    sublayer      TEXT,               -- 'A' | 'B'
    category      TEXT,
    first_seen    TIMESTAMP,
    last_seen     TIMESTAMP,
    importance    REAL DEFAULT 0.5
);

CREATE TABLE edges (
    edge_id       INTEGER PRIMARY KEY,
    src_node      TEXT REFERENCES nodes(node_id),
    dst_node      TEXT REFERENCES nodes(node_id),
    edge_type     TEXT NOT NULL,
    strength      REAL DEFAULT 1.0,   -- decays with recency; boosted by re-mention
    reported_at   TIMESTAMP,
    projected_at  TIMESTAMP,          -- NULL if contemporaneous
    source_doc_id TEXT,               -- arXiv ID, EDGAR accession, etc.
    lag_quarters  REAL
);

CREATE TABLE node_snapshots (
    node_id       TEXT REFERENCES nodes(node_id),
    snapshot_date DATE NOT NULL,
    -- technology node attributes
    bottleneck_severity  REAL,        -- 0–1, Gemini-scored from mentions this week
    mention_count        INTEGER,
    constraint_mentions  INTEGER,
    expansion_announced  BOOLEAN,
    -- signal node attributes
    signal_value         REAL,
    signal_30d_change    REAL,
    signal_90d_zscore    REAL,
    PRIMARY KEY (node_id, snapshot_date)
);

CREATE TABLE documents (
    doc_id        TEXT PRIMARY KEY,   -- arXiv ID / EDGAR accession / HF paper ID
    source        TEXT,
    published_at  TIMESTAMP,
    title         TEXT,
    content       TEXT,
    processed     BOOLEAN DEFAULT FALSE
);
```

Edge `strength` is updated multiplicatively: each new mention of the edge within 30 days boosts by 1.1; each week without mention decays by 0.95. Edges below strength 0.05 are soft-deleted (retained but excluded from snapshots).

---

## Snapshot pipeline

Runs weekly (Monday, same cadence as PCA refit in v1).

```
1. Query node_snapshots for the past 4 weeks → build attribute matrices
2. Query edges with strength > 0.05 → build adjacency per edge type
3. Construct PyG HeteroData:
     node_types: ['signal', 'technology']
     edge_types: all 9 edge types above
     x_dict: {node_type → attribute tensor}
     edge_index_dict: {edge_type → [src, dst] tensor}
     edge_attr_dict: {edge_type → [strength, lag] tensor}
4. Save to data/snapshots/YYYY-MM-DD.pt
5. Retain last 52 snapshots (1 year); archive older ones
```

---

## Bottleneck label derivation

Automated. No manual labeling.

For each weekly snapshot, derive the active bottleneck label:

```
1. For each technology node this week:
     raw_score = constraint_mentions / (mention_count + 1)
     zscore = (raw_score - rolling_90d_mean) / rolling_90d_std

2. Active bottleneck = argmax(zscore) where zscore > 2.0
   If no node exceeds 2.0 σ: label = None (no dominant bottleneck)

3. Gemini validation pass (once per quarter on a random sample):
     "Given these mention frequencies, is [node] correctly identified as the
      active bottleneck this week? Respond yes/no + correction if no."
   This catches systematic misclassification without requiring full manual labels.
```

Training labels are the bottleneck node 4, 8, and 12 weeks *ahead* of each snapshot. So the dataset has a 12-week lookahead burn-in at the start (2017 data can be used for training starting 2017-Q2).

---

## GNN architecture

### Per-snapshot encoder: Heterogeneous Graph Transformer (HGT)

HGT is inductive — node embeddings are computed from local neighborhoods, not learned per-node. This handles new nodes that appear after training.

```
Input:  HeteroData snapshot with x_dict, edge_index_dict, edge_attr_dict
Output: node_embedding_dict  {node_id → R^128}

HGT hyperparameters (initial):
  hidden_dim = 128
  num_heads  = 4
  num_layers = 2
  dropout    = 0.1
```

### Temporal wrapper: LSTM over 4-snapshot window

```
Inputs:  [embedding_t-3, embedding_t-2, embedding_t-1, embedding_t]
         Each is the mean-pooled HGT output over all nodes (global graph state)
         + per-node sequences for the prediction targets

LSTM:    hidden_dim = 256, num_layers = 2
Output:  temporal_embedding_t  ∈ R^256  (global)
         per_node_temporal_t   ∈ R^128  (per technology node, for bottleneck head)
```

### Prediction head 1 — Bottleneck migration

```
Input:  per_node_temporal_t for each technology node
Output: P(active_bottleneck | node) for t+4w, t+8w, t+12w

Architecture: Linear(128 → 64) → ReLU → Linear(64 → 3)
              softmax over technology nodes at each horizon

Loss: L_B = (1/3) * Σ_horizon CrossEntropy(softmax(logits), one_hot(true_bottleneck))
```

### Prediction head 2 — Macro signal forecasting

```
Input:  per_node_temporal_t for each signal node
Output: direction ∈ {up, flat, down} and magnitude ∈ R for t+2w, t+4w

Architecture: Linear(128 → 64) → ReLU → Linear(64 → 4)
              [logit_up, logit_flat, logit_down, magnitude]

Loss: L_S = CrossEntropy([logit_up, flat, down], true_direction)
           + 0.3 * MSE(magnitude, true_magnitude)
      averaged over all signal nodes and both horizons
```

### Combined training loss

```
L_total = α * L_B + (1 - α) * L_S

α = 0.6 initial (bottleneck migration weighted slightly higher)
α is tuned on a held-out validation window (2024-01-01 to 2024-12-31)
```

### Training procedure

```
Train set:   2017-01-01 → 2023-12-31  (sliding window, weekly snapshots)
Val set:     2024-01-01 → 2024-12-31
Test set:    2025-01-01 → present

Optimizer:   AdamW, lr=1e-3, weight_decay=1e-4
Scheduler:   CosineAnnealingLR, T_max=100 epochs
Batch:       Single graph (full graph per snapshot) — not mini-batched
Early stop:  Val loss plateau for 10 epochs
```

---

## Inference pipeline

Runs daily, after snapshot is available (or uses the latest weekly snapshot).

```
1. Load latest snapshot .pt
2. Load trained model checkpoint
3. Forward pass → bottleneck_scores, signal_forecasts, temporal_embedding
4. Derive net_exposure_delta from signal_forecasts:
     stress_signals = [VIX↑, credit_spread↑, BDRY↑, term_spread↓]
     forecast_stress = weighted average of stress signal directions
     net_exposure_delta = -0.25 * forecast_stress  (same scale as PCA modifier)
5. Write gnn_signal.json
```

### Output format

```json
{
  "computed_at": "2026-05-27T06:00:00Z",
  "model_version": "hgt-lstm-v1",
  "bottleneck": {
    "active_node": "CoWoS_L",
    "active_label": "TSMC CoWoS-L advanced packaging",
    "confidence": 0.78,
    "horizon_4w": {"CoWoS_L": 0.78, "HBM3e": 0.61, "Power_grid_interconnect": 0.43},
    "horizon_8w": {"HBM4": 0.52, "CoWoS_L": 0.49, "Power_grid_interconnect": 0.61},
    "migration_signal": "power_grid_becoming_primary"
  },
  "signal_forecasts": {
    "VIX":       {"direction": "flat",   "magnitude": 0.3, "confidence": 0.71},
    "copper":    {"direction": "rising", "magnitude": 0.8, "confidence": 0.65},
    "BDRY":      {"direction": "flat",   "magnitude": 0.1, "confidence": 0.58},
    "US10Y":     {"direction": "flat",   "magnitude": 0.2, "confidence": 0.62}
  },
  "net_exposure_delta": -0.03,
  "graph_stats": {
    "node_count": 184,
    "edge_count": 1203,
    "snapshot_date": "2026-05-26"
  }
}
```

v1 ignores this file until Phase 2 integration. When Phase 2 lands, gemini_scorer.py reads it and prepends a `GNN SIGNAL` block to the prompt — same pattern as the existing macro regime block.

---

## Project structure

Separate repo. Reads v1's document corpus optionally; maintains its own graph store.

```
ai-graph-engine/
├── ingestion/
│   ├── arxiv.py              # arXiv API, 2017-present, cs.LG/DC/AR/NE/AI
│   ├── edgar.py              # EDGAR 8-Ks, extended ticker set, 2017-present
│   └── hf_papers.py          # HuggingFace daily papers, 2020-present
├── extraction/
│   ├── gemini_extractor.py   # per-document → entities, edges, timestamps
│   └── gate.py               # canonicalization (fuzzy match) + importance scoring
├── graph/
│   ├── schema.py             # node/edge type catalog, sublayer assignments, seed catalog
│   ├── store.py              # SQLite read/write (nodes, edges, snapshots, documents)
│   ├── snapshot.py           # weekly: SQLite → PyG HeteroData → .pt
│   └── signal_updater.py     # daily: yfinance/FRED → node_snapshots for signal nodes
├── labels/
│   └── bottleneck.py         # automated bottleneck label derivation from mention stats
├── gnn/
│   ├── model.py              # HGT encoder + LSTM temporal wrapper
│   ├── heads.py              # bottleneck head + signal forecast head
│   ├── train.py              # full training loop, checkpointing, early stop
│   ├── eval.py               # val/test metrics: bottleneck accuracy, signal direction acc
│   └── inference.py          # daily forward pass → gnn_signal.json
├── data/
│   ├── graph.db              # SQLite graph store
│   ├── snapshots/            # weekly .pt files (YYYY-MM-DD.pt)
│   └── gnn_signal.json       # v1 reads this in Phase 2
├── tests/
├── config.py
└── main.py                   # orchestrator: ingest → extract → gate → snapshot → infer
```

---

## Implementation phases

**Phase 1 — Graph construction + extraction pipeline**
- Historical ingestion (arXiv + EDGAR + HF papers, 2017–present)
- Gemini extraction + importance gate
- Graph store + signal updater
- Snapshot pipeline
- Target: weekly snapshot .pt files being produced, graph growing correctly

**Phase 2 — GNN training**
- Bottleneck label derivation over historical snapshots
- HGT + LSTM implementation in PyG
- Training on 2017–2023, validation on 2024
- Target: bottleneck prediction accuracy > 60% at 4w horizon on val set

**Phase 3 — Inference + output**
- Daily inference pipeline
- gnn_signal.json production
- v1 integration: inject into Gemini prompt (Option A — prompt enrichment only)
- Target: live GNN signal running alongside v1 with no v1 changes

**Phase 4 — Modifier replacement** *(deferred)*
- Replace/augment v1 PCA composite modifier with `net_exposure_delta` from GNN
- Requires 4–6 weeks of live GNN signal to backtest first

---

## Key dependencies

```
torch>=2.3.0
torch-geometric>=2.6.0       # PyG — HGT, HeteroData, GraphSAGE
torch-sparse, torch-scatter  # required by PyG
arxiv==2.1.3                 # arXiv Python client
fredapi>=0.5.1                # already in v1
yfinance>=0.2.40              # already in v1
google-genai>=1.9.0           # already in v1
scikit-learn>=1.6.1           # already in v1
rapidfuzz>=3.0.0              # fast fuzzy matching for canonicalization
```
