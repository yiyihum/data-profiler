# ACP Agent Guide

## 1. 目标
ACP（Agentic Codified Profiler）用于把原始数据自动转成可执行规则，并输出结构化 `profile.json`，供下游建模使用。

## 2. 运行约束（Workspace First）
每次运行都会创建独立 workspace：

`runs/<run_id>/workspace`

目录结构：

- `input/`：输入数据（只读语义）
- `artifacts/`：输出产物
- `logs/`：运行日志
- `tmp/`：临时文件

默认会把测试样例复制进去：

源目录：`/data/yiming/project/data-profiler/public`  
目标目录：`runs/<run_id>/workspace/input/public`

## 3. Agent 角色

### IngestionAgent
- 输入：文件或目录
- 功能：自动识别 CSV/JSON/Parquet/ImageDir，标准化为两列：`data`, `label`
- 任务描述：优先读取同目录 `description.md`，缺失时默认使用 `analyze the dataset`
- 输出：`standardized.parquet`

### SamplerAgent
- 输入：当前节点 embedding
- 功能：动态选择采样策略（random / outlier / kmeans）
- 输出：代表样本索引

### CoderAgent
- 输入：代表样本 + 上下文
- 功能：默认通过 OpenAI API 生成候选 `def check(x): ...` 规则（无启发式模板回退）
- 输出：候选规则列表

### SandboxRunner
- 功能：在受限执行器中验证候选规则，返回 mask 与错误信息
- 机制：优先 Firejail；若系统无 Firejail 则自动 fallback 到本地受限执行器

### CDTEngine
- 功能：递归处理节点，计算分裂评分（有标签时 IG；无标签时代理评分），选择最佳规则
- 输出：树结构 + 选中规则

### ProfileCompiler
- 功能：将规则编译为协议化输出
- 输出：`profile.json`

## 4. 启动方式

默认运行环境（Conda）：

`/data/yiming/conda_envs/mle_master`

默认解释器：

`/data/yiming/conda_envs/mle_master/bin/python`

安装依赖：

```bash
/data/yiming/conda_envs/mle_master/bin/python -m pip install -r requirements.txt
```

默认运行（自动 seed `public` 并使用 `train.csv`）：

```bash
/data/yiming/conda_envs/mle_master/bin/python main.py
```

指定自定义数据：

```bash
/data/yiming/conda_envs/mle_master/bin/python main.py --no-seed-public --data /path/to/data
```

常用参数：

- `--workspace-root ./runs`
- `--seed-public /data/yiming/project/data-profiler/public`
- `--max-depth 3`
- `--samples-per-node 5`
- `--candidates-per-node 5`

## 5. 关键输出
每次运行主要产物位于：

`runs/<run_id>/workspace/artifacts/`

包括：

- `standardized.parquet`
- `tree.json`
- `profile.json`
- `run_summary.json`
- `config.snapshot.yaml`
