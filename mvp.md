这是一个基于 **Multi-Agent Collaboration (多智能体协作)** 和 **Code-as-Rule (代码即规则)** 的完整系统设计方案。

我们将这个系统命名为 **ACP (Agentic Codified Profiler)**。

---

# ACP 系统设计文档 (System Design Document)

## 1. 核心理念 (Core Concepts)

1. **代码即认知 (Code is Cognition)**: Agent 不用自然语言描述数据特征，而是编写可执行的 Python 函数（例如 `def check(x): return len(x) > 50`）来定义特征。
2. **实证主义 (Empiricism)**: 所有假设（Hypothesis）必须经过沙盒全量数据验证（Verification）并计算出信息增益（IG）后，才能成为知识。
3. **动态采样 (Dynamic Sampling)**: 不预设采样策略，由 Sampler Agent 根据数据形态决定如何挑选样本给 LLM 看。
4. **安全闭环 (Security Loop)**: 任何生成的代码只能在断网、只读的 Firejail 容器中运行。

---

## 2. 系统架构 (System Architecture)

系统采用 **微内核 + 插件化** 架构。

### 2.1 目录结构

```bash
acp_project/
├── core/                   # [内核层]
│   ├── engine.py           # CDT 主循环引擎
│   ├── sandbox.py          # Firejail 沙盒接口
│   ├── memory.py           # 共享知识库 (Metadata)
│   └── compiler.py         # Profile 编译器
├── agents/                 # [智能体层]
│   ├── ingestion_agent.py  # 数据摄入与清洗
│   ├── sampler_agent.py    # 动态采样策略
│   └── coder_agent.py      # 特征代码生成
├── tools/                  # [工具层] (挂载入沙盒)
│   ├── llm_ops.py          # Zero-shot, Embedding 工具
│   ├── cv_ops.py           # OpenCV, PIL 封装
│   └── text_ops.py         # NLTK, Regex 封装
├── schema/                 # [协议层]
│   └── profile_protocol.py # JSON 输出定义
├── config/                 # [配置层]
│   └── settings.yaml       # LLM Key, Firejail Path
└── main.py                 # 启动入口

```

---

## 3. 详细模块设计

### 3.1 智能体层 (The Agents)

#### A. Ingestion Agent (数据摄入)

* **职责**: 解决“无论用户给什么文件，都要转成标准格式”的问题。
* **输入**: 原始路径 (Raw Path)。
* **动作**:
1. 探测文件类型 (CSV, JSON, Parquet, Image Dir)。
2. 编写 `loader.py`。
3. 在沙盒中执行 `loader.py`。
4. **输出**: 标准化的 `dataset.parquet` (列名强制对齐为 `data`, `label`) 和初始 `Metadata`。



#### B. Sampler Agent (动态采样)

* **职责**: 解决“LLM 窗口有限，该给它看哪些数据”的问题。
* **输入**: 当前节点的 Embeddings 和 Metadata。
* **动作**:
* 分析分布：是聚团的？还是离散的？
* **决策**: 编写采样代码。
* *策略 A*: KMeans 聚类找中心 (适合大多数情况)。
* *策略 B*: 边缘采样 (Outlier Detection) (适合找 Corner Case)。
* *策略 C*: 随机采样 (适合极小数据集)。




* **输出**: 选中的 `indices` 列表。

#### C. Coder Agent (特征生成)

* **职责**: 系统的核心大脑，负责“猜”特征。
* **输入**: 代表性样本 (Samples) + 上下文 (Context)。
* **动作**:
* 观察样本差异。
* 调用 `tools/` 库。
* 编写 3-5 个候选函数 `def check(x): ...`。


* **输出**: Python 代码字符串列表。

---

### 3.2 核心引擎层 (The Engine)

#### CDT Engine (`core/engine.py`)

这是整个系统的调度器，维护决策树的状态。

**状态机逻辑**:

1. **Initialize**: 加载 Parquet，计算全局 Embeddings。
2. **Node Process (递归)**:
* **Check Stop**: 深度 > Max 或 纯度足够高 -> `Make Leaf`。
* **Call Sampler**: 也就是 `Sampler Agent` -> 得到 `rep_indices`。
* **Call Coder**: 也就是 `Coder Agent` -> 得到 `candidate_codes`。
* **Sandboxed Validation**:
* 并行在 Firejail 中运行所有 `candidate_codes`。
* 获取全量 `Mask`。


* **Evaluate**: 计算 Information Gain (IG)。
* **Select Best**: 选出 IG 最高的代码作为当前节点的 `Rule`。
* **Update Knowledge**: 将新发现的 Pattern 写入共享内存。
* **Split**: 生成左右子节点，递归。



---

### 3.3 环境层 (The Environment)

#### Firejail Sandbox (`core/sandbox.py`)

这是执行代码的物理边界。

**关键配置 (Profile)**:

```python
# 伪代码配置
FIREJAIL_ARGS = [
    "--noprofile",            # 纯净模式
    "--net=none",             # 断网 (防止泄密/恶意下载)
    "--rlimit-as=4g",         # 内存限制 4GB
    "--read-only=/",          # 根目录只读
    f"--bind={data_path}:/mnt/data:ro",     # 数据只读挂载
    f"--bind={project_root}/tools:/mnt/tools:ro", # 工具库挂载
    f"--bind={hf_cache}:/mnt/models:ro",    # 模型权重挂载
    "--private=/tmp"          # 临时目录私有化
]

```

---

### 3.4 协议层 (The Protocol)

#### Profile Protocol (`schema/profile_protocol.py`)

这是 ACP 输出给下游 Solve Agent 的最终产物。

```json
{
  "meta": {
    "task": "binary_classification",
    "dataset_shape": [10000, 2]
  },
  "strategy": {
    "difficulty": "Hard",
    "recommended_model": "LightGBM + Custom Features"
  },
  "directives": [
    {
      "id": "rule_001",
      "type": "FEATURE_ENGINEERING",
      "priority": "CRITICAL",
      "insight": "Images with low Laplacian variance are mostly abnormal.",
      "code": "def rule_001(x):\n    import cv2\n    return cv2.Laplacian(cv2.imread(x), cv2.CV_64F).var() < 100",
      "action": "Create column 'feat_blur_check'"
    },
    {
      "id": "rule_002",
      "type": "DATA_FILTER",
      "priority": "MEDIUM",
      "insight": "Texts starting with 'HTTP' are noise.",
      "code": "def rule_002(x):\n    return str(x).startswith('http')",
      "action": "Drop rows where True"
    }
  ]
}

```

---

## 4. 数据流 (Data Flow Pipeline)

1. **用户输入**: `main.py --data ./kaggle/input`
2. **阶段一 (Ingestion)**:
* `Ingestion Agent` 启动 -> Firejail 运行 Loader -> 生成 `/tmp/standardized.parquet`。


3. **阶段二 (Profiling)**:
* `CDT Engine` 启动。
* `Embedder` 计算向量。
* **Loop**:
* `Sampler` 挑出 5 个样本。
* `Coder` 写出 `check_blur(x)`。
* `Sandbox` 跑遍 10000 条数据 -> 返回 `[True, False, ...]`。
* `Engine` 算 IG -> 选中。




4. **阶段三 (Compilation)**:
* `Compiler` 遍历树 -> 提取高 IG 节点 -> 生成 `profile.json`。


5. **阶段四 (Handoff)**:
* 下游 `Solve Agent` 读取 JSON -> `import rule_001` -> 开始训练。



---

## 5. 关键交互 Prompt 设计

为了保证效果，Prompt 需要精心设计（Prompt Engineering）。

**Coder Agent Prompt 模板**:

```text
You are a Data Detective.
Context: We are classifying {task_desc}.
Current Knowledge: {metadata_summary}

Here are {k} representative samples from the current data cluster:
{samples}

Your Goal: Write a Python function `def check(x):` that returns True/False to split these samples distinctively.
Constraints:
1. Use provided tools: `import tools.cv_ops`, `import tools.text_ops`.
2. Do NOT use external network calls.
3. Be robust against None/NaN input.

Output strictly JSON: {"description": "...", "code": "..."}

```

---

## 6. 项目落地清单 (Checklist)

1. **基础设施准备**:
* Linux 环境 (Ubuntu 22.04+)。
* 安装 Firejail (`sudo apt install firejail`).
* 下载预训练模型 (DeBERTa, CLIP, SentenceTransformer) 到本地缓存目录。


2. **Python 环境**:
* 创建 `requirements.txt`: `openai`, `pandas`, `scikit-learn`, `fire`, `pydantic`.


3. **MVP 开发顺序**:
* 先写 `FirejailSandbox` 类，确保能跑通 "Hello World"。
* 再写 `IngestionAgent`，确保能把 CSV 转 Parquet。
* 接着实现 `CDTEngine` 主逻辑。
* 最后对接 LLM，调试 Prompt。



