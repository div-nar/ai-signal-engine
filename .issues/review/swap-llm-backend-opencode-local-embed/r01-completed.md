# MR Review Report — swap-llm-backend-opencode-local-embed

Swap thesis LLM to opencode CLI + Chroma embeddings to local fastembed/nomic

## Basic Information
- **Author**: div-nar (with Claude)
- **Branch**: `swap-llm-backend-opencode-local-embed` → `main`
- **Repo**: github.com/div-nar/ai-signal-engine (**public**)
- **Changed Files**: 9 (+916 −22)
- **Code Changes**: thesis_scorer.py, chroma_store.py, config.py, scripts/rebuild_chroma.py, requirements.txt, 2 test files, spec + plan docs

## Review Summary

The Gemini API project was 403-suspended, disabling both thesis scoring and Chroma
embeddings. This swaps thesis generation to the local `opencode` CLI (opencode-go
subscription) and embeddings to local `fastembed`/`nomic-embed-text-v1.5`, behind the two
pre-existing seams. Verified end-to-end: a live `--mode passive` run computed target id 10
via opencode (17-name book, weights sum 1.0, coherent thesis) with zero Gemini calls; 761
vectors rebuilt at 768-dim; 163 tests pass.

---

## Quality Assessment

**Overall Grade**: A (8.7/10.0)

**Score Breakdown**:
- Base Score: 10.0
- Critical Issues: 0
- Quality Issues: 0
- Strengths: +0.5 (clean seam reuse, no over-engineering)
- Bonuses: +1.0 (test coverage) +1.0 (spec+plan docs); normalized to A
- **Final Score**: 8.7

**Issue Distribution**: 🔴 0 · 🟡 0 · 🔵 2

---

## Review Findings

### 🔴 Critical Issues
None. Secret scan clean (public repo). `subprocess.run` uses the **list** form (no
`shell=True`), so untrusted RSS/EDGAR content in the prompt cannot cause shell injection.

### 🔵 Suggestions & Best Practices

#### Suggestion 1: opencode agent tool exposure with untrusted doc content
**📍 `scoring/thesis_scorer.py` `_OpencodeClient.generate`**
The thesis prompt embeds untrusted ingested content (EDGAR 8-Ks, RSS) and is sent to
`opencode run`, which is an *agent* with built-in tools. Mitigations in place: `--pure`
disables external plugins, and `--auto` is deliberately NOT passed, so tool calls needing
permission are not auto-approved (the live run returned JSON in ~60s without hanging).
**Follow-up (not blocking):** for defense-in-depth, define a dedicated no-tools `--agent`
so a prompt-injected filing cannot even attempt a tool call. SYSTEM.md already lists
prompt-injection robustness as an open research item.

#### Suggestion 2: large-prompt ARG_MAX risk
**📍 same**
The full prompt is passed as a single argv element. Current prompts (~10–50KB) are well
under macOS ARG_MAX, and the live run confirms it works. **Follow-up:** if the seed-doc
set grows large enough to approach the limit, switch to piping the prompt via stdin
instead of argv.

---

## Overall Assessment

### ✅ Strengths
- Reuses the existing `.generate()` and `_embed()` seams — no provider-abstraction layer
  (correct YAGNI call for two call sites).
- `MAX_EMBED_CHARS` cap fixes a real pathology found during build (50k-char EDGAR bodies
  took >100s on local ONNX; now sub-second to ~3s).
- Fail-loud error handling preserved: opencode failures raise into the existing 3× retry;
  embedding failures still swallowed per-doc so ingestion never halts.
- Verified end-to-end, not just unit-tested: live passive run + no-Gemini grep + weight
  sanity check.

### 📋 Recommendations
1. Merge — the change is verified live and strictly restores a dead system.
2. Follow-ups (both optional): no-tools opencode agent; stdin prompt if it grows.

---

## Review Metadata
- **Review Date**: 2026-08-13
- **Reviewer**: Claude (gate-required review)
- **Review Count**: r01
- **Status**: completed
- **Grade**: A (8.7/10.0)
- **Template**: standard

## Issue Resolution Status

| Issue # | Type | Status | Resolution Notes |
|---------|------|--------|------------------|
| 1 | Suggestion | ℹ️ Deferred | no-tools agent — defense-in-depth follow-up |
| 2 | Suggestion | ℹ️ Deferred | stdin prompt if size approaches ARG_MAX |
