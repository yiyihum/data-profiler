# Micro-Macro Data Understanding Framework for LLM-based AutoML

## Context

LLM Agent 解决 AutoML 任务时，对数据理解不足（不看数据就写代码、特征工程弱）。现有 LLM-based AutoML 系统（MLAgentBench, DS-Agent 等）主要依赖 trial-and-error，缺乏对数据本身的深入理解。

**核心创新**：提出 Micro-Macro 两层数据理解框架，包含两个关键创新机制：

**创新 1: Micro-Macro 双层理解**
- **Macro**：传统统计分析（分布、相关性等）—— 告诉你"数据长什么样"
- **Micro**：用 LLM 逐条阅读数据点，理解语义和隐含模式 —— 告诉你"数据意味着什么"
- **Micro→Macro 桥接**：Micro 观察 → 生成假设 → Macro 统计验证 → 指导特征工程

**创新 2: Adaptive Iterative Sampling (自适应迭代采样)**
- 第一轮：按目标变量分层随机采样 → Micro 分析 → 发现初步模式
- 后续轮：基于已发现模式，**针对性采样**来验证/深化/反驳模式
- 采样策略随理解加深而演化，而非静态预设

**创新 3: Hypothesis Coverage Matrix (假设覆盖矩阵)**
- 构建 **特征 × 假设类型** 的覆盖矩阵，跟踪哪些特征-假设组合已被探索
- 主动发现未覆盖区域，引导 LLM 生成针对性假设
- 确保假设不集中在少数特征上，系统性地覆盖数据空间

聚焦 **DS (表格)** 和 **NLP (文本)** 类型任务，支持两者的混合任务。

## 整体架构

```
输入: description.md + 数据文件 + vLLM endpoint
  ↓
[Phase 1] Macro Overview — 统计层面的宏观理解
  基本统计 (shape, dtypes, missing, target分布)
  → 生成任务层面的初步假设 → 统计代码验证
  → 初始化 Coverage Matrix
  ↓
[Phase 2] Micro Exploration — LLM 驱动的微观理解 (自适应迭代)
  Round 0: 分层随机采样 → 单点分析 → 对比分析 → 归纳模式
  Round 1+: 基于已有模式自适应采样 → 深化/验证/发现新模式
  → 采样策略由 LLM 根据已有发现动态决定
  ↓
[Phase 3] Micro→Macro Bridging — 假设生成与验证 (覆盖矩阵驱动)
  假设来源 1: Micro 模式 → 可验证假设
  假设来源 2: Coverage Matrix 盲区 → 针对性假设
  → 生成验证代码 → 全量数据统计验证
  → 更新 Coverage Matrix → 检查覆盖率
  → 覆盖不足时可反向触发新一轮 Micro 采样 (闭环)
  ↓
[Phase 4] Synthesis — 综合报告
  汇总 Macro + Micro + Coverage 统计 + 验证结论
  → 数据分析报告 (Markdown + JSON)
  → 特征工程建议 + 建模策略建议
```

## 交付文件

```
/data/yiming/project/hypothesis_framework/
├── main.py                      # 端到端入口
├── config.py                    # 配置
├── llm_client.py                # vLLM OpenAI 兼容 API 客户端
├── data_loader.py               # 数据加载、预览、采样
├── adaptive_sampler.py          # 自适应迭代采样器
├── coverage_matrix.py           # 假设覆盖矩阵
├── code_executor.py             # firejail 沙箱代码执行器
├── macro_analyzer.py            # Phase 1: Macro 统计分析
├── micro_analyzer.py            # Phase 2: Micro LLM 数据点分析
├── hypothesis_engine.py         # Phase 3: 假设生成 + 验证循环
├── report_generator.py          # Phase 4: 报告生成
├── models.py                    # 数据结构定义
└── prompts/
    ├── macro_hypothesis_gen.txt          # Macro 层假设生成
    ├── micro_single_point.txt            # Micro: 分析单条数据
    ├── micro_contrastive.txt             # Micro: 对比分析 (正样本 vs 负样本)
    ├── micro_pattern_summary.txt         # Micro: 归纳微观模式
    ├── adaptive_sampling_strategy.txt    # 自适应采样: 基于已有模式生成采样策略
    ├── coverage_gap_hypothesis.txt       # 覆盖矩阵: 针对未覆盖区域生成假设
    ├── bridge_hypothesis_gen.txt         # Bridge: 微观模式 → 可验证假设
    ├── verification_code_gen.txt         # 验证代码生成
    ├── verification_result_analysis.txt  # 验证结果分析
    └── report_synthesis.txt              # 最终报告综合
```

## 详细设计

### `models.py` — 数据结构

```python
@dataclass
class MicroObservation:
    """单条数据点的 LLM 分析结果"""
    data_point_id: str
    label: Any                    # 目标变量值
    observation: str              # LLM 的分析文本
    key_patterns: List[str]       # 提取的关键模式

@dataclass
class MicroPattern:
    """从多条 MicroObservation 归纳出的模式"""
    pattern: str                  # 模式描述
    evidence: List[str]           # 支撑证据 (数据点 ID)
    confidence: str               # high / medium / low
    source: str                   # "contrastive" | "single_class" | "cross_feature"

@dataclass
class Hypothesis:
    """可验证的假设"""
    hypothesis: str
    source: str                   # "macro" | "micro" | "bridge"
    micro_pattern: Optional[MicroPattern]  # 如果来自 micro，关联的模式
    verification_code: str = ""
    execution_result: str = ""
    conclusion: str = ""          # "confirmed" | "rejected" | "inconclusive"
    evidence_summary: str = ""
    suggested_features: List[str] = field(default_factory=list)

@dataclass
class DataReport:
    """最终数据分析报告"""
    task_overview: str
    macro_findings: List[str]
    micro_patterns: List[MicroPattern]
    hypotheses: List[Hypothesis]
    feature_recommendations: List[str]
    modeling_recommendations: List[str]
```

### `config.py`

```python
@dataclass
class Config:
    vllm_endpoint: str = "http://localhost:8000/v1"
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    data_path: str = ""
    description_path: str = ""
    # Micro 分析配置
    micro_sample_per_class: int = 10    # 每类初始采样条数
    micro_batch_size: int = 5           # 每次送给 LLM 几条
    max_micro_rounds: int = 3           # Micro 自适应采样最大轮数
    adaptive_sample_size: int = 10      # 自适应采样每轮条数
    # 假设验证配置
    max_hypotheses: int = 15
    max_verification_retries: int = 3
    code_timeout: int = 60              # 秒
    # Bridge 迭代
    max_bridge_iterations: int = 2      # Micro→Macro 迭代轮数
    # 覆盖矩阵
    min_coverage_rate: float = 0.8      # 覆盖率停止阈值
    # 输出
    output_dir: str = "./output"
    firejail_enabled: bool = True
```

### `adaptive_sampler.py` — 自适应迭代采样器

核心思想：采样策略不是静态的，而是随着 Micro 分析的深入动态演化。

```python
class AdaptiveSampler:
    def __init__(self, data: pd.DataFrame, target_col: str, config: Config):
        self.data = data
        self.target_col = target_col
        self.sampled_ids: Set[str] = set()  # 已采样的数据点
        self.iteration = 0

    def initial_sample(self) -> pd.DataFrame:
        """第一轮: 按目标变量分层随机采样"""
        # 每类采 micro_sample_per_class 条
        # 记录已采样 ID，避免重复
        self.iteration = 1
        return samples

    def adaptive_sample(self, patterns: List[MicroPattern],
                        hypotheses: List[Hypothesis]) -> pd.DataFrame:
        """后续轮: 基于已发现模式生成采样策略"""
        self.iteration += 1
        # 1. 将已有 patterns、hypotheses、已用过的采样模式 发给 LLM
        #    prompt: adaptive_sampling_strategy.txt
        # 2. LLM 不仅选择采样参数，还可以**提出全新的采样模式**
        #    返回 JSON:
        #    {
        #      "sampling_mode": "counterexample",  # LLM 自创的采样模式名
        #      "mode_description": "找按已有模式预测应成功但实际失败的样本",
        #      "sampling_code": "df[(df.request_text.str.len()>200) & (df.target==0)]",
        #      "reason": "验证文本长度模式的边界条件"
        #    }
        #    内置模式: stratified_random, filter, subgroup
        #    LLM 可自创: counterexample, boundary, cross_feature, anomaly 等
        # 3. LLM 生成完整采样代码 → firejail 执行 → 获取样本
        # 4. 记录采样模式到 self.sampling_history
        # 5. 排除已采样的数据点
        self.sampling_history.append(sampling_mode_record)
        return new_samples

    def get_sampling_summary(self) -> dict:
        """返回采样覆盖统计"""
        return {
            "total_sampled": len(self.sampled_ids),
            "iterations": self.iteration,
            "coverage_by_class": ...,
            "coverage_by_feature_range": ...
        }
```

关键设计：
- LLM 基于已有模式决定"下一步看什么数据"，而非人为预设
- **LLM 可以动态发明新的采样模式**（不限于预定义模式）
  - 内置模式作为起点：stratified_random, filter, subgroup
  - LLM 可自创：counterexample, boundary, cross_feature, anomaly 等
  - 采样模式以自然语言描述 + 可执行代码的形式表达
- 每轮采样都排除已看过的数据，确保新信息
- `sampling_history` 记录所有使用过的采样模式，避免重复，也作为后续 prompt 的上下文

### `coverage_matrix.py` — 假设覆盖矩阵

跟踪假设对数据空间的覆盖情况，主动发现盲区。

```python
class HypothesisCoverageMatrix:
    # 假设类型维度
    HYPOTHESIS_TYPES = [
        "distribution",       # 特征分布
        "target_correlation", # 与目标变量的关系
        "interaction",        # 特征间交互
        "text_pattern",       # 文本模式 (NLP)
        "missing_pattern",    # 缺失值模式
        "temporal_pattern",   # 时间模式
    ]

    def __init__(self, features: List[str], feature_types: Dict[str, str]):
        # feature_types: {"col_name": "numeric" | "categorical" | "text" | "datetime"}
        # 初始化 coverage matrix: features × hypothesis_types → 0/1
        self.matrix: Dict[Tuple[str, str], List[Hypothesis]] = {}

    def register_hypothesis(self, hypothesis: Hypothesis, features: List[str],
                           hypothesis_type: str):
        """注册一个假设到覆盖矩阵"""
        for f in features:
            key = (f, hypothesis_type)
            self.matrix.setdefault(key, []).append(hypothesis)

    def get_uncovered_cells(self) -> List[Tuple[str, str]]:
        """返回未被任何假设覆盖的 (feature, type) 组合"""
        # 过滤掉不适用的组合 (如数值特征的 text_pattern)
        all_valid = self._get_valid_cells()
        return [cell for cell in all_valid if cell not in self.matrix]

    def get_weakly_covered_cells(self, min_count: int = 1) -> List[Tuple[str, str]]:
        """返回覆盖不足的组合"""
        return [cell for cell, hyps in self.matrix.items() if len(hyps) < min_count]

    def generate_coverage_prompt(self) -> str:
        """生成覆盖状况描述，用于引导 LLM 生成新假设"""
        # 输出格式:
        # 已覆盖: feature_A × target_correlation (2 hypotheses, 1 confirmed)
        # 未覆盖: feature_B × interaction
        # 未覆盖: feature_C × text_pattern
        # → 请针对未覆盖区域生成假设

    def get_coverage_stats(self) -> dict:
        """覆盖率统计"""
        return {
            "total_valid_cells": ...,
            "covered_cells": ...,
            "coverage_rate": ...,
            "confirmed_hypotheses": ...,
            "rejected_hypotheses": ...,
        }
```

关键设计：
- 自动识别哪些 (feature, hypothesis_type) 组合是有意义的（如文本列不需要 distribution 类型）
- 在 Bridge 阶段的每轮迭代后更新矩阵
- 覆盖率作为"是否需要继续迭代"的判据之一
- 生成的覆盖 prompt 直接引导 LLM 关注盲区

### `macro_analyzer.py` — Phase 1

**Step 1a: Task Description Deep Analysis (P0 改进)**
- 结构化解析 description.md，提取：
  - **评估指标**（必须准确，贯穿后续所有建议）
  - **不可用字段**（test 中缺失的字段 → 排除列表）
  - **标签泄露风险字段**（字段含义中暗示目标变量的 → 危险标记）
  - **任务类型**、数据来源、时间范围、特殊说明
- 输出：**TaskConstraints** 对象，作为后续所有阶段的 hard rules
- 使用专门的 prompt: `task_description_analysis.txt`

**Step 1b: 基本统计分析**
1. 调用 `data_loader` 获取数据预览 (shape, dtypes, missing, 基本统计)
2. 如有 test 数据，对比 train/test 的字段差异验证 Step 1a 的发现
3. 将预览信息 + TaskConstraints → `macro_hypothesis_gen.txt` → LLM 生成假设
4. 每个假设 → LLM 生成验证代码 → firejail 执行 → LLM 分析结果
5. 输出: 任务概况 + macro 层假设验证结果

新增 prompt: `task_description_analysis.txt`
新增数据结构:
```python
@dataclass
class TaskConstraints:
    evaluation_metric: str              # 如 "AUC-ROC"
    task_type: str                      # "classification" | "regression" | ...
    unavailable_fields: List[str]       # test 中不可用的字段
    leakage_risk_fields: List[str]      # 有标签泄露风险的字段
    special_notes: List[str]            # 其他重要约束
```

### `micro_analyzer.py` — Phase 2 (核心创新)

使用 `AdaptiveSampler` 实现多轮自适应 Micro 分析:

```
for round in range(max_micro_rounds):
    if round == 0:
        samples = sampler.initial_sample()    # 分层随机
    else:
        samples = sampler.adaptive_sample(patterns, hypotheses)  # 自适应

    # Step A: 单点分析
    observations = []
    for sample in samples:
        obs = llm.analyze(micro_single_point_prompt, sample)
        observations.append(obs)

    # Step B: 对比分析 (正 vs 负样本)
    contrastive = llm.analyze(micro_contrastive_prompt,
                               positive_obs, negative_obs)

    # Step C: 模式归纳 (累积所有轮次的观察)
    all_observations.extend(observations)
    patterns = llm.summarize(micro_pattern_summary_prompt, all_observations)

    # Step D: 判断是否需要继续采样
    if no_new_patterns_found or round == max_rounds - 1:
        break
```

关键改进：
- 不再是一次性采样，而是多轮迭代
- 每轮采样策略由 LLM 基于已有发现自适应调整
- 累积观察，模式随轮次深化

**P1 改进: Micro 报告必须包含具体数据点**
- 每个 MicroObservation 包含原始数据的引用（文本片段、特征值）
- 对比分析必须展示成对的正/负样本示例
- 模式归纳必须引用具体的 data point ID 和文本片段作为证据
- 报告输出格式示例：
  > "[成功] #t3_w5491: '我是单亲妈妈，孩子生日但银行卡冻结了...' → 困境叙事+具体事件"
  > "[失败] #t3_xyz: '谁给我点个pizza' → 过于简短，无具体原因"

**P1 改进: 质性假设的证据强度评估**
- 对于无法量化效应量的 Micro 假设，使用：
  - 支撑样本数量 / 总采样数
  - 正/负样本中的出现频率差异
  - 是否被后续 Macro 统计间接验证
  - 可操作性评分（能否直接转化为特征）

### `hypothesis_engine.py` — Phase 3

**Bridge: Micro → Macro (集成 Coverage Matrix)**
```
coverage = HypothesisCoverageMatrix(features, feature_types)

for iteration in range(max_bridge_iterations):
    # 1. 生成假设 (两个来源)
    # 1a. 从 MicroPattern 转化
    hypotheses_from_micro = llm.generate(bridge_hypothesis_gen_prompt,
                                          context=micro_patterns + macro_results)
    # 1b. 从覆盖矩阵盲区生成
    uncovered = coverage.get_uncovered_cells()
    hypotheses_from_gaps = llm.generate(coverage_gap_hypothesis_prompt,
                                         context=uncovered + macro_results)
    all_hypotheses = hypotheses_from_micro + hypotheses_from_gaps

    # 2. 验证每个假设
    for h in all_hypotheses:
        h.verification_code = llm.generate(verification_code_gen_prompt, h)
        h.execution_result = executor.run(h.verification_code)
        if execution_failed:
            retry with error feedback (最多 max_retries 次)
        h.conclusion = llm.analyze(h.execution_result)

        # 3. 注册到覆盖矩阵
        coverage.register_hypothesis(h, h.related_features, h.hypothesis_type)

        # 4. 从验证结果中提取特征建议
        if h.conclusion == "confirmed":
            h.suggested_features = llm.extract_features(h)

    # 5. 判断是否继续迭代
    stats = coverage.get_coverage_stats()
    if stats["coverage_rate"] > 0.8 or iteration == max_iterations - 1:
        break

    # 6. 基于验证结果触发新一轮 Micro 采样 (自适应)
    new_samples = sampler.adaptive_sample(micro_patterns, all_hypotheses)
    new_observations = micro_analyzer.analyze_samples(new_samples)
    micro_patterns = micro_analyzer.update_patterns(new_observations)
```

关键改进：
- 覆盖矩阵驱动假设生成，确保系统性覆盖
- 两个假设来源：Micro 模式 + 覆盖盲区
- 覆盖率作为迭代停止条件之一
- Bridge 阶段可以反向触发新的 Micro 采样（闭环迭代）

**P1 改进: 假设去重**
- 假设生成时，覆盖矩阵提供已有假设列表，prompt 中明确要求不重复
- 生成后用 LLM 做语义去重（判断新假设是否与已有假设本质相同）
- 每个 coverage cell 最多 2 个假设

**P1 改进: 效应量分级**
- 可量化假设：p 值 + 效应量双报告
  - 强发现: p < 0.05 且效应量大 (Cohen's d > 0.5 / OR > 2)
  - 弱发现: p < 0.05 但效应量小
  - 无发现: p >= 0.05
- 质性假设：支撑样本数 + 模式一致性 + 可操作性
- 验证结果分析 prompt 中强制要求分级

### `code_executor.py` — firejail 沙箱

```python
def execute(code: str, data_path: str, timeout: int = 60) -> ExecutionResult:
    # 1. 写代码到临时文件
    # 2. firejail 命令:
    #    firejail --noprofile --net=none \
    #      --whitelist=<data_path> --read-only=<data_path> \
    #      --whitelist=<tmp_dir> \
    #      python <tmp_file>
    # 3. 捕获 stdout/stderr，设置 timeout
    # 4. 返回 ExecutionResult(stdout, stderr, exit_code)
```

### `report_generator.py` — Phase 4

汇总所有阶段结果，调用 LLM 生成报告:

```markdown
# 数据分析报告: {task_name}

## 1. 任务概述
数据规模、类型、目标分布、评估指标

## 2. Macro 发现
- 统计层面的关键发现

## 3. Micro 洞察 (LLM 数据点分析)
- 微观模式 1: ... (基于 N 条样本，置信度: high)
- 微观模式 2: ...

## 4. 假设验证结果
| 假设 | 来源 | 结论 | 证据摘要 |
|------|------|------|----------|
| ... | micro | confirmed | ... |

## 5. 特征工程建议 (仅基于 confirmed 假设)
- 基于假设 X (强发现): 建议构建 XXX 特征
- 基于 Micro 模式 Y (高一致性): 建议构建 YYY 特征
- [不建议] 假设 Z 已被 rejected: ZZZ 方向无效

## 6. 建模策略建议 (数据驱动，非模板)
- 基于数据特点 (如文本+表格混合) 的具体模型建议
- 基于评估指标 (如 AUC-ROC) 的优化策略
- 基于类别不平衡程度的具体处理方案

## 7. 约束与警告
- 不可用字段列表 (test 中缺失)
- 标签泄露风险字段
```

同时输出 JSON 格式 (`DataReport` 序列化)。

### `main.py` — 入口

```python
def main(config: Config):
    client = LLMClient(config)
    loader = DataLoader(config)
    executor = CodeExecutor(config)

    # Phase 1: Macro
    macro = MacroAnalyzer(client, loader, executor)
    macro_results = macro.analyze()

    # Phase 2: Micro
    micro = MicroAnalyzer(client, loader)
    micro_patterns = micro.analyze()

    # Phase 3: Bridge
    engine = HypothesisEngine(client, executor)
    hypotheses = engine.bridge(micro_patterns, macro_results)

    # Phase 4: Report
    reporter = ReportGenerator(client)
    report = reporter.generate(macro_results, micro_patterns, hypotheses)
    reporter.save(report, config.output_dir)
```

CLI:
```bash
python main.py \
  --data-path .../train.json \
  --description-path .../description.md \
  --vllm-endpoint http://localhost:8000/v1 \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --output-dir ./output
```

## Prompt 模板要点

| Prompt 文件 | 输入 | 输出 | 关键要求 |
|-------------|------|------|----------|
| `task_description_analysis.txt` | description.md 全文 | JSON: TaskConstraints | 提取评估指标、不可用字段、泄露风险 |
| `macro_hypothesis_gen.txt` | 数据预览 + TaskConstraints | JSON: 假设列表 | 排除不可用字段，关注可验证假设 |
| `micro_single_point.txt` | 单条数据记录 | 文本: 观察分析 | 关注语义、情感、叙事策略 |
| `micro_contrastive.txt` | 正样本组 + 负样本组的分析 | JSON: 差异模式列表 | 关注两组间的系统性差异 |
| `micro_pattern_summary.txt` | 所有 MicroObservation | JSON: MicroPattern 列表 | 归纳、去重、标注置信度 |
| `bridge_hypothesis_gen.txt` | MicroPattern + Macro结果 | JSON: 假设列表 | 从模式到可验证命题 |
| `verification_code_gen.txt` | 假设 + 数据信息 | Python 代码 | 自包含、只用 pandas/numpy/scipy |
| `verification_result_analysis.txt` | 假设 + 执行结果 | JSON: 结论 | confirmed/rejected/inconclusive |
| `report_synthesis.txt` | 所有阶段结果 | Markdown 报告 | 可操作的建议 |

## 实现顺序

1. `models.py` + `config.py` — 数据结构和配置
2. `llm_client.py` — vLLM 客户端 (OpenAI SDK)
3. `data_loader.py` — 数据加载、预览
4. `code_executor.py` — firejail 沙箱
5. `adaptive_sampler.py` — 自适应迭代采样器
6. `coverage_matrix.py` — 假设覆盖矩阵
7. `prompts/` — 所有 prompt 模板 (10 个)
8. `macro_analyzer.py` — Phase 1
9. `micro_analyzer.py` — Phase 2 (核心，集成自适应采样)
10. `hypothesis_engine.py` — Phase 3 (集成覆盖矩阵)
11. `report_generator.py` — Phase 4
12. `main.py` — 入口串联
13. 端到端测试 (random-acts-of-pizza)

## 验证

1. **模块测试**: 每个模块单独测试基本功能
2. **端到端测试**: 用 random-acts-of-pizza 数据集运行完整流程，生成报告
3. **报告质量**: 检查报告是否包含有意义的 micro 洞察和可操作的特征建议
4. **注意**: vLLM 未部署，可先在 `llm_client.py` 中支持 mock 模式用于开发调试

