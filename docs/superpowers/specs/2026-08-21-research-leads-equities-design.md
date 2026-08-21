# Does the research frontier lead tech equities? — study design

**Date:** 2026-08-21
**Status:** Approved design, pre-implementation
**Owner:** div-nar

## Question

Does a signal built from the published-research corpus at time *t* carry predictive
information for tech-equity returns at *t + k*, and does that lead differ by value-chain
layer? Put plainly: can we price in future tech-market movements by reading the papers first?

The null we are trying to beat is efficient pricing (the market has already absorbed the
research instantly). We are hunting for a lag the market is slow to close.

## Core hypothesis and the layer reframe

The naive version ("each layer has its own paper stream, that stream leads that layer's
stocks") does not survive contact with the data: arXiv research is overwhelmingly *methods*
(vision, NLP, RL, efficiency, systems). It maps naturally onto Compute and Platform and
barely at all onto Power & Energy or Fabrication.

So the sharper hypothesis is a **derived-demand chain**: an AI-research surge should lead the
*downstream* layers (power, fabrication, infrastructure) through methods research → compute
demand → datacenter / power / fab buildout → those equities, and it should lead them with a
**longer lag** than it leads compute and platform. The headline result is therefore a
**lead-lag term structure across layers**: research hits compute fast, power and fab slow.

## Data

### Independent variable: the research corpus (2010 → now)

Only arXiv (plus EDGAR) spans back to 2010-2011, so the 14-year panel is **arXiv-primary**.
This aligns with the original framing ("published papers"). HF daily papers (post-2023) and
RSS/blogs are a robustness check on the recent tail only, never part of the main panel, so the
signal definition stays consistent across the whole window.

- Source: arXiv metadata via OAI-PMH / API, dated by submission date (point-in-time).
- Categories: cs.LG, cs.AI, cs.CV, cs.CL, cs.NE, cs.AR, cs.DC, eess.SY.
- Subfield decomposition of the signal: vision, NLP, RL, systems/hardware, efficiency
  (derived from category + title/abstract keywording).
- Secondary: EDGAR full-text (dated), used for robustness, not the core signal.

### Signal facets (computed weekly, per subfield)

- **Attention:** submission volume, z-scored against the subfield's own trailing baseline,
  plus its acceleration (change in volume). The classic "attention leads price" analog.
- **Novelty / breakthrough intensity:** embedding surprise of the period's abstracts vs a
  trailing window, using the repo's local nomic embedder (fastembed / nomic-embed-text-v1.5,
  768-dim, offline). Catches regime shifts, not just noise.

### Dependent variable: layer equity baskets

The 5 value-chain layers (Power & Energy, Fabrication & Materials, Compute & Silicon,
Infrastructure & Networking, Platform & Application), each a basket of tickers that traded
back to 2010 (e.g. NVDA, AMD, TSM, ASML, AMAT, LRCX, MSFT, GOOGL, plus later entrants added at
their listing date to keep the panel point-in-time). Weekly returns from yfinance, reusing
`backtest/panel.py` (`load_price_panel`, `to_weekly_fridays`, `forward_returns`).

## Estimation

1. **Panel regression.** `fwd_return[layer, t+k] ~ attention_z + novelty_z + controls`, with
   layer and time fixed effects and clustered / Newey-West standard errors, run across the
   horizon grid **k ∈ {1w, 2w, 1m, 2m, 3m}**. Controls: own-momentum, lagged return, and the
   broad-market return, so we isolate research's *marginal* content beyond price.
2. **Information coefficient.** Per-period Spearman IC between the signal and forward layer
   returns; report the IC decay curve across k.
3. **Long-short backtest.** Overweight high-signal layers, underweight low-signal layers, net
   of momentum, benchmarked against an equal-weight-layers baseline. Tests whether statistical
   significance is actually tradable.

## Deliverable

A report containing:

- The lead-lag term structure per layer (the money chart): where research leads, and by how
  long, for each layer.
- Regression tables with honest t-stats.
- The IC decay curve across horizons.
- The long-short equity curve vs the equal-weight-layers baseline.
- A clear verdict: does research earn its place as an engine input? If yes, it becomes the next
  build-out (wired into the Layer Cake thesis stage as a real signal). If no, we report the
  null honestly.

## Guardrails against fooling ourselves

- **Point-in-time only.** The signal at *t* uses only data with submission/filing date ≤ *t*.
  No citation counts, no impact or acceptance features (they leak the future).
- **Survivorship / listing dates.** Tickers enter a basket only from their actual listing
  date; no back-filling a name into years it did not trade.
- **Out-of-sample discipline.** Walk-forward split (e.g. expanding window) so results are not a
  14-year in-sample fit. Report in-sample vs out-of-sample separately.
- **Null honesty.** If research does not lead, that is the finding, stated plainly. A clean
  null is a real result and stops us wiring noise into a live trading engine.

## Scope boundaries (YAGNI)

- Not evaluating the live engine's 13-target track record (too few observations); this is a
  constructed panel study.
- Not building per-stock cross-sectional IC in this phase (needs ticker extraction; deferred to
  a possible phase 2 if the layer-level result is positive).
- Not scoring tone/sentiment in this phase (needs a classifier pass; attention + novelty first).

## Non-blocking prerequisite

Unrelated to the study but noted: the scheduled trade agent's PATH bug is fixed in `run.sh`
(Homebrew prepended so launchd can find `opencode`). The live book was frozen on the Aug 14
target for a week because of it; a `--mode trade --force` run will catch it up.
