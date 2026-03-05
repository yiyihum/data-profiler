# Data Profiler MVP

基于状态机 + 标准化 JSON 的自动化数据分析管道。

## 功能特点

- **L0-L3 四层分析架构**
  - L0: 数据质量清洗 (Data Janitor)
  - L1: 无监督结构探索 (Unsupervised Explorer)
  - L2: 任务对齐特征验证 (Task Aligner)
  - L3: AutoML 策略输出 (Strategist)

- **核心组件**
  - Code Sandbox: 安全的代码执行环境
  - State Context: 全局状态管理
  - LLM Client: 支持 OpenAI/Anthropic 的结构化输出

- **输出产物**
  - `mle_report.md`: 完整的 MLE 分析报告
  - `preprocess_pipeline.py`: 可执行的预处理脚本
  - 可视化图表 (相关性矩阵、特征重要性等)

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

1. 准备数据集 (CSV 或 Parquet 格式)

2. 创建任务配置文件:

```yaml
target: "price"
task_type: "regression"
metric: "rmse"
description: "预测房价"
```

3. 运行分析:

```bash
python main.py data.csv task_config.yaml -o ./output
```

## 命令行参数

```
python main.py <data_path> <config_path> [options]

Arguments:
  data_path              数据集路径 (CSV/Parquet)
  config_path            任务配置文件路径 (YAML)

Options:
  -o, --output DIR       输出目录 (默认: ./output)
  --provider PROVIDER    LLM 提供商: openai/anthropic (默认: openai)
  --model MODEL          模型名称 (默认: gpt-4o)
  -v, --verbose          显示详细输出
```

## 项目结构

```
data-profiler/
├── main.py              # 主入口
├── core/                # 核心组件
│   ├── state.py         # 状态管理器
│   ├── sandbox.py       # 代码沙盒
│   ├── llm.py           # LLM 客户端
│   └── prompts.py       # 提示词模板
├── layers/              # 层级执行器
│   ├── l0_cleaner.py    # L0 数据清洗
│   ├── l1_explorer.py   # L1 结构探索
│   ├── l2_aligner.py    # L2 特征对齐
│   └── l3_strategist.py # L3 策略生成
├── report/              # 报告生成
│   └── generator.py     # 报告生成器
└── examples/            # 示例配置
    └── task_config.yaml
```

## 环境变量

```bash
export OPENAI_API_KEY="your-api-key"
# 或
export ANTHROPIC_API_KEY="your-api-key"
```
