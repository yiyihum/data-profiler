# Repository Guidelines

## Project Structure & Module Organization
`data-profiler` is a Python CLI pipeline with layered analysis:
- `main.py`: CLI entry point and L0-L3 orchestration.
- `core/`: shared runtime pieces (`state.py`, `sandbox.py`, `llm.py`, `prompts.py`).
- `layers/`: execution layers (`l0_cleaner.py` -> `l3_strategist.py`).
- `report/`: report and preprocessing script generation.
- `examples/` and `demo/`: sample task configs and datasets.
- `output/`: generated artifacts (state JSON, report, scripts, plots).

Keep new logic close to the owning layer/module; avoid cross-layer coupling unless the data flow requires it.

## Build, Test, and Development Commands
- `pip install -r requirements.txt`: install runtime dependencies.
- `python main.py demo/train.csv demo/task_config.yaml -o ./output -v`: run a full local smoke test.
- `python main.py <data.csv> <task.yaml> --provider openai --model gpt-4o`: run with explicit LLM settings.
- `python -m py_compile main.py core/*.py layers/*.py report/*.py`: quick syntax check before opening a PR.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and readable, typed function signatures where practical.
- File/module names use `snake_case`; classes use `PascalCase`; functions/variables use `snake_case`.
- Preserve the existing layer naming pattern (`L0Cleaner`, `L1Explorer`, etc.) and keep docstrings concise.
- Prefer explicit state updates through `StateContext` instead of ad-hoc globals.

## Testing Guidelines
There is no committed automated test suite yet, so current validation is smoke-test based.
- Run the pipeline on `demo/train.csv` + `demo/task_config.yaml`.
- Confirm these outputs are created and readable: `output/pipeline_state.json`, `output/mle_report.md`, `output/preprocess_pipeline.py`.
- For behavior changes, include a reproducible command and expected result summary in the PR.

## Commit & Pull Request Guidelines
Recent history uses conventional prefixes like `feat:` and `chore:`. Use:
- `<type>: <imperative summary>` (example: `feat: add L2 feature scoring fallback`).

PRs should include:
- What changed and why.
- Modules/layers touched.
- Validation commands run and key output artifacts.
- Report/plot snippets when output format or visuals change.

## Security & Configuration Tips
- Configure credentials via environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).
- Do not commit API keys or sensitive datasets.
- Treat `output/` artifacts as generated files unless explicitly needed for demos/repro cases.
