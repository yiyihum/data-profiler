# ACP Implementation Audit Todo

## Audit Result

Current status: **Mostly complete and runnable**.

Validation performed:
- Directory and file structure check: passed.
- Syntax compile check: passed (`python -m compileall -q acp_framework main.py`).
- Mock pipeline smoke run: passed (`python -m acp_framework.main --mock-llm ...`).

## Todo Checklist

### P0 - Core Architecture Delivery
- [x] Create `acp_framework/` package with `core/`, `agents/`, `tools/`, and orchestrator `main.py`.
- [x] Implement `DataState` with global DataFrame, `coverage_mask`, `coverage_ratio`, and uncovered view APIs.
- [x] Implement Planner-driven loop orchestration (Planner -> Data -> Hypothesor -> Coder -> Consolidator).
- [x] Implement DataAgent sampling strategies: `random_baseline`, `uncovered_only`, `extreme_lengths`.
- [x] Implement Hypothesor/Coder role separation (NL hypotheses vs Python code generation).
- [x] Implement rule consolidation with scoring + Jaccard behavior-level dedupe.
- [x] Integrate dynamic reusable skills memory (`tools/skills_library.py`).

### P0 - Safety / Sandbox
- [x] Implement sandbox executor with timeout protection.
- [x] Support `strict_firejail` mode with startup check for Firejail binary.
- [x] Add Firejail hardening flags in strict mode: `--net=none`, `--read-only=/`, `--private=<tmp>`.
- [x] Sanitize execution output by JSON extraction from stdout.

### P1 - Alignment Gaps vs Target Design
- [ ] Move coverage update logic into `RuleConsolidator` (currently done in orchestrator).
- [ ] Add explicit `while True` planner loop form (current `for` loop is functionally equivalent but not exact).
- [ ] Add typed config object/YAML config for runtime settings (LLM, planner thresholds, sandbox mode).

### P1 - Reliability
- [ ] Add unit tests for `DataState`, `SandboxRunner`, `RuleConsolidator`, and planner transitions.
- [ ] Add deterministic fixture-based integration test for full ACP loop.
- [ ] Add retry + robust parse fallback in LLM calls for malformed outputs.

### P2 - Output / Productization
- [ ] Export full `profile.json` protocol (including phase trace and coverage timeline).
- [ ] Generate `rules.py` artifact with `apply_rules(df)` for downstream usage.
- [ ] Add run logs and per-round metrics report.

## Recommended Next Step

1. Implement the first three P1 items (design alignment), then add P1 reliability tests.
2. After that, implement P2 artifacts (`rules.py`, richer profile output).
