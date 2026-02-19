# ACP Framework

State-driven multi-agent data profiler.

## Structure

- `acp_framework/core/data_state.py`: global dataframe + coverage mask
- `acp_framework/core/sandbox.py`: code sandbox execution (`dev_local` or `strict_firejail`)
- `acp_framework/core/consolidator.py`: scoring + Jaccard dedupe + validation
- `acp_framework/agents/planner_agent.py`: planning state machine
- `acp_framework/agents/data_agent.py`: coverage-aware data views
- `acp_framework/agents/hypothesor_agent.py`: natural-language hypothesis generation
- `acp_framework/agents/coder_agent.py`: hypothesis-to-code translation
- `acp_framework/tools/skills_library.py`: persistent reusable rule memory
- `acp_framework/main.py`: orchestrator loop

## Run

```bash
python -m pip install -r requirements.txt
export OPENAI_API_KEY=your_key
python -m acp_framework.main --data /path/to/train.csv --target-col label --out ./profile.json
```

For local smoke test without LLM API:

```bash
python -m acp_framework.main --data /path/to/train.csv --mock-llm --out ./profile.json
```
