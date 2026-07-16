# MR Review Report — restore-dashboard-and-targets-export

Restore recovered dashboard source + add targets.json exporter

## Basic Information
- **Author**: div-nar (with Claude)
- **Branch**: `restore-dashboard-and-targets-export` → `main`
- **Repo**: github.com/div-nar/ai-signal-engine (**public**)
- **Changed Files**: 8
- **Code Changes**: +426 −0

## Review Summary

Commits the Vercel dashboard source, recovered from deployment
`dpl_CULcEtnb8p512FqHsE1o2AvfV6W8` — it existed in no branch or commit and was one
deleted deployment away from being unrecoverable. Adds `export_targets.py` to close the
gap that left `targets.json` frozen at target id=11 (2026-07-03).

---

## Quality Assessment

**Overall Grade**: A (8.5/10.0)

**Score Breakdown**:
- Base Score: 10.0
- Critical Issues: 0
- Quality Issues: −0.5 (one, fixed during review)
- Strengths: +0.5
- Bonuses: +1.0 (test coverage), −2.5 (recovered code carries pre-existing debt, see below)
- **Final Score**: 8.5

**Issue Distribution**: 🔴 0 · 🟡 1 (fixed) · 🔵 2

---

## Review Findings

### 🔴 Critical Issues

None. **Secret scan clean** — mandatory given the repo is public. Every match in the
diff (`ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, `DASH_TOKEN`) is an environment-variable
*name* or a query-string read; no key material is committed. Alpaca credentials live in
Vercel project env vars (`type=sensitive`) and never reach the client.

### 🟡 Code Quality Issues

#### Issue 1: `export()` terminated the caller's process — **FIXED**

**📍 Location**: `export_targets.py:33`

**🔍 Original**:
```python
def export(db_path=None, out_path=DEFAULT_OUT) -> dict:
    target = get_latest_target(db_path) if db_path else get_latest_target()
    if target is None:
        sys.exit("no target in DB — ...")
```

**⚠️ Problem**: a library function calling `sys.exit()` raises `SystemExit` in any
caller — an importing script or scheduler dies instead of handling the condition. The
tell was in the tests: they had to catch `SystemExit`, which is a smell, not a contract.
The `if db_path else` branch also existed only because the default was `None` rather
than the real DB default.

**💡 Fix applied**: introduced `NoTargetError`; `export()` raises it, and only
`__main__` translates it to `sys.exit`. Default is now `str(DEFAULT_DB)`, removing the
conditional. Tests use `pytest.raises(NoTargetError)`.

**📊 Impact**: Risk Low (script currently has no importers), but the exporter is
intended to be called from the trade pipeline, where this would have been a live bug.

### 🔵 Suggestions & Best Practices

#### Suggestion 1: `datetime.utcnow()` is deprecated (pre-existing, not addressed)

**📍 Location**: `ops/web/api/data.py:74`

`dt.datetime.utcnow()` is deprecated in Python 3.12+. This is recovered production code
that currently runs correctly on Vercel; changing it is out of scope for a restore
commit. Flagged for a follow-up, deliberately not touched here.

#### Suggestion 2: `targets.json` refresh remains manual

`export_targets.py` writes the file, but nothing calls it automatically and values only
reach the site on `cd ops/web && vercel deploy --prod`. The old auto-refresh lived in
the never-committed snapshot job. A follow-up could invoke the exporter at the end of
`--mode trade`.

---

## Overall Assessment

### ✅ Strengths
- Recovers otherwise-unrecoverable source into version control.
- Exporter reuses the existing `get_latest_target()` contract rather than re-querying.
- Refuses to publish empty targets over a good snapshot (fail-safe, verified by test).
- 4 new tests; full suite 154 → 158, no regressions; CLI exercised end-to-end.

### ⚠️ Areas for Improvement
- Recovered `ops/web/` code is committed as-found and carries pre-existing debt
  (deprecated `utcnow`, no tests for `api/data.py`). Restoring it verbatim is correct
  for provenance, but it is not reviewed-to-standard code.

### 📋 Recommendations
1. Merge — the restore is strictly better than source existing nowhere.
2. Follow-up: call `export_targets.py` from the trade pipeline.
3. Follow-up: fix `utcnow()` deprecation in a dedicated commit.

---

## Review Metadata
- **Review Date**: 2026-07-16
- **Reviewer**: Claude (gate-required review)
- **Review Count**: r01
- **Status**: completed
- **Grade**: A (8.5/10.0)
- **Template**: standard

## Issue Resolution Status

| Issue # | Type | Status | Resolution Notes |
|---------|------|--------|------------------|
| 1 | Quality | ✅ Fixed | `NoTargetError` replaces `sys.exit()`; tests use `pytest.raises` |
| 2 | Suggestion | ℹ️ Deferred | `utcnow()` deprecation — pre-existing, out of scope |
| 3 | Suggestion | ℹ️ Deferred | Auto-refresh of targets.json from trade pipeline |
