好的，我们暂时剥离“代码复用”和“历史经验检索”的复杂性，回归到一个**最小可行性产品 (MVP)** 的本质。

作为一个客观的系统设计，MVP 的核心目标是：**以最低的工程复杂度，跑通从“原始数据+任务描述”到“高质量 MLE Report”的完整闭环，并验证 L0-L3 分层架构的可行性。**

为了实现这一点，我们不能依赖复杂的 Multi-Agent 对话，而是应该采用**“状态机 (State Machine) + 标准化中间产物 (JSON)”**的架构。LLM 在这里的角色不仅是“思考者”，更是“JSON 生成器”和“代码编写者”。

以下是该 MVP 的核心架构与实现路径设计：

### 一、 MVP 系统输入与输出定义

* **输入 (Inputs):**
1. `dataset.csv` 或 `.parquet` (物理文件)。
2. `task_config.yaml` (包含预测目标 Label、评估指标 Metric、简单的业务背景描述)。


* **输出 (Outputs):**
1. `mle_report.md` (结构化的分析报告，包含图表路径)。
2. `preprocess_pipeline.py` (L0-L2 阶段确定下来的、可直接运行的数据清洗和特征转化脚本)。



---

### 二、 核心组件设计 (Core Components)

为了保证 Agent 行动的可控性，MVP 需要三个基础组件，它们贯穿 L0-L3 的始终：

1. **Code Sandbox (代码沙盒):**
* **机制:** 接收 LLM 生成的 Python 代码 (主要是 Pandas 和 Scikit-learn)，在隔离的 Jupyter Kernel 或 Docker 中执行。
* **限制:** 严禁 LLM 直接修改原始数据文件，所有操作必须作用于内存中的 DataFrame 副本。
* **反馈:** 返回标准输出 (stdout)、图表文件路径或 Traceback 报错。如果报错，允许 LLM 重试（设定最大重试次数 $N=3$）。


2. **State Context (全局状态管理器):**
* **机制:** 维护一个字典，记录当前到达了哪个层级，以及各个层级产生的核心洞察。这也是为了避免上下文超载 (Context Overflow)，绝不把全量数据丢给 LLM。


3. **Prompt Template Engine (提示词引擎):**
* **机制:** 针对 L0-L3 的不同目标，注入严格的系统提示词，强制 LLM 输出结构化的 JSON，而不是自由发散的文本。



---

### 三、 L0 - L3 的线性执行流 (Execution Flow)

这是 MVP 的骨架，每一层必须有明确的“准入条件”和“交付物”。

#### L0: 基础统计与清洗 (The Data Janitor)

* **输入:** 数据集路径。屏蔽 `task_config.yaml`。
* **动作:** 1. LLM 生成代码执行 `.info()`, `.describe()`, 计算缺失率、唯一值数量。
2. 针对高确信度的脏数据（如全为空的列、单一值的列）生成 Drop 代码。
* **MVP 交付物:** `state["L0_stats"]` (包含各列的基础统计量 JSON) 和清洗后的 `df_clean_v1`。

#### L1: 无监督结构探索 (The Unsupervised Explorer)

* **输入:** `state["L0_stats"]` 和 `df_clean_v1`。继续屏蔽 `task_config.yaml`。
* **动作:**
1. **分布检测:** 识别长尾分布、偏态分布 (Skewness)。
2. **冗余检测:** 计算特征间的皮尔逊/斯皮尔曼相关系数矩阵。对于相关系数 $> 0.9$ 的特征对，标记为高度冗余。


* **MVP 交付物:** `state["L1_insights"]` (例如：`{"skewed_features": ["income"], "collinear_pairs": [["age", "years_working"]]}`)。

#### L2: 任务对齐与特征验证 (The Task Aligner)

* **输入:** `state["L0_stats"]`, `state["L1_insights"]`, `df_clean_v1` 以及**首次引入的 `task_config.yaml**`。
* **动作:**
1. **Label 关联:** 计算特征与 Target 的相关性 (如果是分类任务，计算 Information Value / 互信息)。
2. **假设验证:** LLM 根据 L1 发现的偏态分布，生成代码测试 Log 变换后互信息是否提升。如果提升，则将该操作固化。


* **MVP 交付物:** `state["L2_selected_features"]` 和最终清洗/转化后的 `df_clean_v2`。

#### L3: AutoML 策略输出 (The Strategist)

* **输入:** 之前所有的 State 总结和最终的特征列表。
* **动作:** 根据特征基数 (Cardinality)、数据量级和任务类型，输出模型选型建议。
* **MVP 交付物:** `state["L3_strategy"]` (例如推荐使用 LightGBM 并说明原因)。

---

### 四、 Report Generator (报告合成器)

MVP 的最后一步是由一个独立的 LLM 节点，读取最终的全局 `state` 字典，按照预设的 Markdown 模板进行填空和润色。

| 报告章节 | 数据来源 | 核心呈现内容 |
| --- | --- | --- |
| **Executive Summary** | L0 + L3 State | 总体数据健康度评分，模型选型 Baseline 建议。 |
| **Data Quality Issues** | L0 State | 用表格列出缺失率高的字段及 L0 阶段执行的清理动作。 |
| **Feature Redundancy** | L1 State | 插入 L1 阶段沙盒生成的 Correlation Matrix 图片，列出建议剔除的共线性特征。 |
| **Predictive Signals** | L2 State | 插入特征重要性 / 互信息条形图，列出对 Target 预测最有价值的 Top 10 特征。 |

---

### 五、 批判性评估与工程建议

即使是一个 MVP，在实现时也会遇到非常现实的阻力。在此提出两点核心预警：

1. **JSON 格式的脆弱性:** LLM 在生成带有转义字符（如代码片段）的 JSON 时极易崩溃。**建议:** 强制使用 `function calling` (或 OpenAI/Gemini 的 Structured Outputs) 接口来约束 L0-L3 每一层输出的 Schema，而不是让模型自己在文本里拼凑 JSON。
2. **沙盒错误循环 (Infinite Error Loop):** 在 MVP 阶段，如果不加以限制，LLM 可能会因为一个简单的库缺失 (`ModuleNotFoundError`) 或列名拼写错误，在“生成代码 -> 报错 -> 尝试修复 -> 再次报错”中耗尽 Token 预算。**建议:** 为 Code Sandbox 设置一个极简的规则：连续 3 次报错，直接跳过当前探索节点，记录为“探索失败”，继续向下推进。

