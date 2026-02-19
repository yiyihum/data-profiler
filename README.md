# data-profiler (ACP MVP)

ACP MVP is a workspace-first data profiling pipeline using multi-agent style components:

- `IngestionAgent`: normalize input to `data`/`label` parquet.
- `SamplerAgent`: choose representative samples.
- `CoderAgent`: produce candidate `check(x)` rules.
- `SandboxRunner`: validate rule code safely with timeout.
- `CDTEngine`: recursive split and scoring loop.
- `ProfileCompiler`: output `profile.json` protocol.

## Quick start

```bash
/data/yiming/conda_envs/mle_master/bin/python -m pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here
/data/yiming/conda_envs/mle_master/bin/python main.py --workspace-root ./runs
```

Project default runtime Python:

`/data/yiming/conda_envs/mle_master/bin/python`

Sandbox defaults to `strict_firejail`. If `firejail` is not installed, set:

```yaml
sandbox:
  mode: dev_local
```

LLM is enabled by default and uses OpenAI API.
Set the key with environment variable `OPENAI_API_KEY` (or change `llm.api_key_env` in config).

Default run behavior:

1. Create a fresh workspace: `runs/<run_id>/workspace`.
2. Load input from `runtime.default_input_path` in config (`/data/yiming/project/data-profiler/demo` by default), unless `--data` is provided.
3. Stage input to `workspace/input/user_data`.
4. Write outputs to `workspace/artifacts/`.

## Task Description Input

- If input directory contains `description.md`, ACP loads it and injects it into rule generation context.
- If `description.md` is missing, ACP uses default description: `analyze the dataset`.
- Description behavior is configurable in `config/settings.yaml`:
  - `runtime.default_task_description`
  - `runtime.task_description_max_chars`
  - `runtime.task_summary_max_chars`

## CLI

```bash
/data/yiming/conda_envs/mle_master/bin/python main.py \
  --data /path/to/file_or_dir \
  --workspace-root ./runs \
  --max-depth 3 \
  --samples-per-node 5 \
  --candidates-per-node 5
```

## Output files

- `standardized.parquet`
- `tree.json`
- `profile.json`
- `run_summary.json`
- `config.snapshot.yaml`
