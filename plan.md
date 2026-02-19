# ACP Lean MVP Todo List

## 1. P0 - Security Foundation (The "Jail" Wall)
目标：确保没有一行 Agent 生成的代码能在宿主机裸奔。

- [x] `Config & Enforcement`：实现 `sandbox.mode` 配置（`strict_firejail` vs `dev_local`）。
- [x] `Config & Enforcement`：在 `strict_firejail` 模式下，启动时检测不到 `firejail` 二进制文件，直接抛出 Fatal Error。
- [x] `Firejail Wrapper Hardening`：强制添加 `--net=none`（断网）和 `--read-only`（文件系统只读）参数。
- [x] `Firejail Wrapper Hardening`：实现 timeout 机制（默认 5 秒），防止死循环卡死主进程。
- [x] `Firejail Wrapper Hardening`：实现标准输入输出流清洗（只捕获 JSON stdout，丢弃无关 logs）。
- [x] `Global Sandbox Hook`：确保 `DataAgent(Ingestion)` 和 `CDTEngine(Validation)` 的代码执行入口全部收口到 `core.sandbox.execute_code()`，杜绝旁路执行。

## 2. P1 - Standardized Ingestion (The Funnel)
目标：无论用户给什么数据，进入 Engine 前必须是干净的 Parquet。

- [ ] `Auto-Loader Generation`：优化 Prompt，要求 LLM 生成 `load_data()` + 强制重命名代码（`df.rename(columns={...: 'data', ...: 'label'})`）。
- [ ] `Auto-Loader Generation`：Ingestion 结束时强制序列化为 `workspace/dataset.parquet`，供后续 Firejail 挂载。
- [ ] `Schema Validation`：Ingestion 执行 loader 后立即校验输出必须包含 `data` 和 `label` 列，且类型符合预期，否则 fast fail。

## 3. P2 - The Intelligent Core (Loop Closure)
目标：跑通 `Sampler -> Coder -> Validator` 核心闭环。

- [ ] `Sampler Agent`：实现基于 KMeans 的基础采样策略（Embeddings -> 聚类 -> 选中心点 -> 返回 indices）。
- [ ] `Sampler Agent`：MVP 阶段不做 LLM 采样代码生成，固定 KMeans 以降低复杂度。
- [ ] `LLM Coder Integration`：实现 `LLMClient`（API 对接、Prompt 组装、错误处理）。
- [ ] `LLM Coder Integration`：实现 Robust Parsing（重试 + Markdown 代码块/JSON 解析 + 格式错误恢复）。
- [ ] `Parallel Validation`：使用 `concurrent.futures.ThreadPoolExecutor` 并发调用 Firejail，一次性并发验证 3-5 个 Candidate Code。

## 4. P3 - Artifacts & Handoff (The Output)
目标：下游 Solve Agent 能直接 import 本产物。

- [ ] `Profile Compiler`：遍历最终树，提取 `IG > threshold` 的节点。
- [ ] `Profile Compiler`：生成 `profile.json`（包含数据元信息与推荐模型类型）。
- [ ] `Executable Rules Module`：将高价值代码片段拼接并写入独立 `rules.py`。
- [ ] `Executable Rules Module`：生成 `apply_rules(df)` 辅助函数，让下游一键提取特征。

## 5. 移除/降级的冗余项（已优化掉）
- [ ] `结构化日志 / run-level report`：MVP 阶段降级为标准 `logging` 控制台输出。
- [ ] `CI blocks merges`：归类为后续 DevOps，不作为功能完备性阻塞项。
- [ ] `Strict Firejail-only execution is not enforced yet`：并入 P0 Enforcement 任务统一处理。
- [ ] `Sampler dynamic code generation`：MVP 阶段移除，固定 KMeans。

## 6. 最终验收标准 (MVP Acceptance)
- [ ] `Security`：运行时可观测到 `firejail` 进程起落，宿主机文件系统无污染。
- [ ] `Stability`：在无网环境下，依赖本地缓存模型（Embedding/Zero-shot）可跑通全流程。
- [ ] `Usability`：产出的 `workspace/rules.py` 可被外部 Python 脚本 `import` 并成功对新数据提取特征。
