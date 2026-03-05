"""L0 Layer - Ultra-Conservative Data Quality Analysis (The Data Janitor)."""

import json
from typing import Any, Dict, List, Optional

from core.agent_loop import AgentLoop, AgentLoopConfig
from core.evaluators import evaluate_l0_turn
from core.state import StateContext
from core.sandbox import CodeSandbox
from core.llm import LLMClient
from core.prompts import L0_SYSTEM_PROMPT, L0_STATS_SCHEMA, get_l0_user_prompt
from layers._discovery import run_discovery
from layers._exploration import run_exploration


# Patterns that indicate forbidden cleaning actions
_FORBIDDEN_PATTERNS = ["fillna", "dropna(subset", "interpolate(", ".impute("]


class L0Cleaner:
    """
    L0: Ultra-Conservative Data Quality Analysis.

    Uses a code-agent pattern:
    1. Bootstrap: hardcoded code gathers basic statistics (shape, dtypes, describe,
       head, per-column missing/unique/samples, outlier detection via IQR)
    2. LLM generates conservative cleaning code (ONLY: drop 100% empty columns,
       drop constant columns, fix encoding, fix wrong dtypes)
    3. Sandbox executes the cleaning code
    4. LLM generates a report section covering the analysis
    5. LLM synthesizes structured JSON summary

    Input: Dataset path + task_config (reads description only for context)
    Output: l0_stats, cleaning actions, df_clean_v1, and L0 report section
    """

    def __init__(
        self,
        sandbox: CodeSandbox,
        llm: LLMClient,
        loop_config: Optional[AgentLoopConfig] = None,
        verbose: bool = True,
    ):
        self.sandbox = sandbox
        self.llm = llm
        self.loop_config = loop_config or AgentLoopConfig()
        self.verbose = verbose

    def execute(self, state: StateContext, task_config: Dict[str, Any]) -> bool:
        """Execute L0 analysis and cleaning."""
        print("[L0] Starting ultra-conservative data quality analysis...")
        loop = AgentLoop("L0", self.loop_config, state, verbose=self.verbose)

        task_description = task_config.get("description", "")

        def step_fn(turn: int, prev_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            code_snippets: List[Dict[str, Any]] = []
            llm_interactions: List[Dict[str, Any]] = []

            # Step 1a: Discovery (hardcoded, fast)
            self.sandbox.reset_dataframe("df")
            discovery_result = run_discovery(self.sandbox, "L0", task_config)
            code_snippets.append({
                "description": "Discovery: data structure probing",
                "code": discovery_result.get("_code", ""),
                "stdout": discovery_result.get("stdout", ""),
                "success": discovery_result["success"],
            })

            # Step 1b: Exploration (LLM-generated, adaptive)
            exploration_result = run_exploration(
                self.sandbox, self.llm, "L0",
                discovery_result.get("stdout", ""), task_config,
            )
            code_snippets.append({
                "description": "Exploration: LLM-adaptive analysis",
                "code": exploration_result.get("_code", ""),
                "stdout": exploration_result.get("stdout", ""),
                "success": exploration_result["success"],
            })

            discovery_ok = discovery_result["success"]
            exploration_ok = exploration_result["success"]

            if not discovery_ok and not exploration_ok:
                return {
                    "success": False,
                    "discovery_success": False,
                    "exploration_success": False,
                    "code_gen_success": False,
                    "code_exec_success": False,
                    "report_section_success": False,
                    "json_synthesis_success": False,
                    "clean_df_ready": False,
                    "error": discovery_result.get("error", "Discovery and exploration both failed"),
                    "observation": "Statistics collection failed",
                    "code_snippets": code_snippets,
                    "llm_interactions": llm_interactions,
                }

            # Combined stdout for downstream steps
            stats_stdout = (
                f"=== Discovery ===\n{discovery_result.get('stdout', '')}\n\n"
                f"=== Exploration ===\n{exploration_result.get('stdout', '')}"
            )

            # Step 2: LLM generates conservative cleaning code
            prev_error = prev_output.get("error") if prev_output else None
            cleaning_task = self._build_cleaning_task(stats_stdout, turn, prev_error)
            available_vars = ["df", "pd", "np", "OUTPUT_DIR"]
            if self.sandbox.get_variable("DATA_DIR") is not None:
                available_vars.extend([
                    "DATA_DIR", "torch", "Image", "Path",
                    "analyze_image", "analyze_audio", "list_data_files",
                    "batch_analyze_images", "batch_analyze_audio",
                ])

            code_gen_result = self.llm.generate_code(
                cleaning_task,
                f"Dataset statistics:\n{stats_stdout[:2000]}",
                available_vars,
            )
            llm_interactions.append({
                "role": "L0_code_generation",
                "user_prompt": cleaning_task,
                "raw_response": (code_gen_result.get("raw", "") or ""),
                "success": code_gen_result["success"],
            })

            if not code_gen_result["success"]:
                return {
                    "success": False,
                    "discovery_success": discovery_ok,
                    "exploration_success": exploration_ok,
                    "code_gen_success": False,
                    "code_exec_success": False,
                    "report_section_success": False,
                    "json_synthesis_success": False,
                    "clean_df_ready": False,
                    "error": code_gen_result.get("error", "LLM code generation failed"),
                    "observation": "Stats collected, but LLM failed to generate cleaning code",
                    "code_snippets": code_snippets,
                    "llm_interactions": llm_interactions,
                }

            # Step 3: Execute LLM-generated cleaning code
            cleaning_code = code_gen_result["content"]
            exec_result = self.sandbox.execute(cleaning_code, "LLM-generated cleaning")
            code_snippets.append({
                "description": "LLM-generated cleaning code",
                "code": cleaning_code,
                "stdout": exec_result.get("stdout", ""),
                "success": exec_result["success"],
            })

            code_exec_success = exec_result["success"]
            if not code_exec_success:
                print(f"[L0] Cleaning code execution failed: {exec_result.get('error', '')}")

            # Constraint enforcement: check for forbidden patterns
            self._check_forbidden_patterns(
                cleaning_code, exec_result.get("stdout", "")
            )

            # Step 4: LLM generates report section
            combined_stdout = (
                f"=== Bootstrap Statistics ===\n{stats_stdout[:2000]}\n\n"
                f"=== Cleaning Code Output ===\n{exec_result.get('stdout', '')[:1500]}\n"
            )
            if exec_result.get("error"):
                combined_stdout += f"\n=== Cleaning Error ===\n{exec_result['error'][:500]}\n"

            report_result = self._generate_report_section(combined_stdout, task_description)
            llm_interactions.append({
                "role": "L0_report_generation",
                "user_prompt": combined_stdout,
                "raw_response": (report_result.get("raw", "") or ""),
                "success": report_result["success"],
            })
            report_section_success = report_result["success"]
            report_section = report_result.get("content", "") if report_section_success else ""

            # Step 5: LLM synthesizes structured JSON summary
            json_result = self._synthesize_json(
                combined_stdout, task_description, turn, prev_error
            )
            llm_interactions.append({
                "role": "L0_json_synthesis",
                "user_prompt": combined_stdout,
                "raw_response": (json_result.get("raw", "") or ""),
                "success": json_result["success"],
            })

            json_synthesis_success = json_result["success"]
            l0_stats = json_result.get("content", {}) if json_synthesis_success else {}
            clean_df_ready = self.sandbox.get_dataframe() is not None

            # Extract cleaning actions from stdout
            cleaning_actions = self._extract_cleaning_actions(
                cleaning_code, exec_result.get("stdout", "")
            )

            return {
                "success": True,
                "discovery_success": discovery_ok,
                "exploration_success": exploration_ok,
                "code_gen_success": True,
                "code_exec_success": code_exec_success,
                "report_section_success": report_section_success,
                "json_synthesis_success": json_synthesis_success,
                "clean_df_ready": clean_df_ready,
                "l0_stats": l0_stats,
                "cleaning_actions": cleaning_actions,
                "report_section": report_section,
                "observation": (
                    f"Stats collected; cleaning code {'succeeded' if code_exec_success else 'failed'}; "
                    f"report {'generated' if report_section_success else 'failed'}; "
                    f"JSON synthesis {'succeeded' if json_synthesis_success else 'failed'}; "
                    f"{len(cleaning_actions)} cleaning actions"
                ),
                "code_snippets": code_snippets,
                "llm_interactions": llm_interactions,
            }

        def apply_fn(output: Dict[str, Any]) -> None:
            state.l0_stats = output.get("l0_stats", {})
            state.l0_cleaning_actions = output.get("cleaning_actions", [])
            report = output.get("report_section", "")
            if report:
                state.set_layer_report("L0", report)
            df_now = self.sandbox.get_dataframe()
            if df_now is not None:
                self.sandbox.set_variable("df_clean_v1", df_now.copy())

        def fallback_fn(reason: str) -> Dict[str, Any]:
            df_now = self.sandbox.get_dataframe()
            if df_now is None:
                return {"status": "failed", "reason": reason}
            state.l0_stats = self._build_fallback_stats(df_now)
            state.l0_cleaning_actions = []
            self.sandbox.set_variable("df_clean_v1", df_now.copy())
            return {"status": "degraded", "reason": reason}

        loop_result = loop.run(
            objective="Assess data quality with ultra-conservative cleaning and produce report section",
            step_fn=step_fn,
            evaluator_fn=evaluate_l0_turn,
            apply_fn=apply_fn,
            fallback_fn=fallback_fn,
        )

        if loop_result["status"] != "success":
            state.record_error("L0", "agent_loop", loop_result["reason"])

        print(
            f"[L0] Completed with status={loop_result['status']} "
            f"(turns={loop_result['turns_used']}, best_score={loop_result['best_score']:.2f})"
        )
        state.advance_layer("L1")
        return loop_result["status"] != "failed"

    def _build_cleaning_task(
        self, stats_stdout: str, turn: int, prev_error: Optional[str],
    ) -> str:
        """Build the task description for ultra-conservative cleaning."""
        task = (
            "Based on the dataset statistics below, generate Python code to clean this data.\n\n"
            "The DataFrame is loaded as 'df' (if tabular data exists). You should modify 'df' in place.\n"
            "If DATA_DIR is available, you also have access to data files directly.\n\n"
            "ULTRA-CONSERVATIVE RULES — you may ONLY:\n"
            "- Drop columns that are 100% empty (every value is NaN/null)\n"
            "- Drop constant columns (only one unique value)\n"
            "- Fix encoding / garbled text in column values\n"
            "- Fix clearly wrong dtypes (e.g. numeric column stored as string)\n\n"
            "You MUST NOT:\n"
            "- Fill missing values (NO fillna, NO imputation)\n"
            "- Drop rows (NO dropna with subset, NO row filtering)\n"
            "- Create new features or derived columns\n"
            "- Impute anything\n\n"
            "If the data contains media files (images/audio), also validate and report their properties.\n\n"
            "Print a summary of each action you take, e.g.:\n"
            '  print("ACTION: drop_column | target=ColumnName | reason=100% empty")\n'
            '  print("ACTION: fix_dtype | target=ColumnName | reason=numeric stored as string")\n\n'
            "Print the final df.shape at the end (if df exists).\n"
            "Do NOT generate any plots.\n"
        )

        if turn > 1 and prev_error:
            task += (
                f"\n\nPrevious attempt failed with error:\n{prev_error[:500]}\n"
                "Fix the issue in this attempt."
            )
        return task

    def _generate_report_section(
        self, combined_stdout: str, task_description: str
    ) -> Dict[str, Any]:
        """LLM generates a markdown report section for L0."""
        system_prompt = (
            "You are a technical report writer. Generate a concise markdown section "
            "for the Data Quality Analysis (L0) portion of a data profiling report."
        )
        user_prompt = f"""Based on the following analysis output, write a markdown report section.

{combined_stdout}

The section should cover:
1. Dataset overview (shape, types)
2. Basic statistics summary
3. Sample data preview
4. Missing data analysis
5. Outlier analysis
6. Cleaning actions taken (and why they are ultra-conservative)
7. What was intentionally NOT done (no imputation, no row dropping, etc.)

Use markdown headers (### level), tables where appropriate, and be factual.
Do NOT include a top-level heading — the layer heading will be added automatically."""

        return self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)

    def _synthesize_json(
        self,
        observations: str,
        task_description: str,
        turn: int,
        prev_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """LLM synthesizes structured L0 JSON from all observations."""
        user_prompt = get_l0_user_prompt(observations, task_description)
        if turn > 1 and prev_error:
            user_prompt += (
                "\n\nPrevious attempt failed. Focus on strict schema compliance.\n"
                f"Previous error: {prev_error}"
            )

        result = self.llm.generate(
            system_prompt=L0_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=L0_STATS_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        result["_system_prompt"] = L0_SYSTEM_PROMPT
        return result

    def _extract_cleaning_actions(
        self, code: str, stdout: str
    ) -> List[Dict[str, str]]:
        """Extract cleaning actions from the executed code's stdout."""
        actions: List[Dict[str, str]] = []
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("ACTION:"):
                parts = line[len("ACTION:"):].strip().split("|")
                action_dict: Dict[str, str] = {}
                if parts:
                    action_dict["action"] = parts[0].strip()
                for part in parts[1:]:
                    part = part.strip()
                    if "=" in part:
                        key, val = part.split("=", 1)
                        action_dict[key.strip()] = val.strip()
                if action_dict.get("action"):
                    actions.append(action_dict)
        return actions

    def _check_forbidden_patterns(self, code: str, stdout: str) -> None:
        """Log warnings if forbidden cleaning patterns were detected."""
        combined = code + "\n" + stdout
        for pattern in _FORBIDDEN_PATTERNS:
            if pattern in combined:
                print(
                    f"[L0] WARNING: Forbidden pattern '{pattern}' detected in "
                    f"cleaning code/output. L0 should be ultra-conservative."
                )

    def _build_fallback_stats(self, df: Any) -> Dict[str, Any]:
        """Build minimal L0 stats if LLM synthesis repeatedly fails."""
        columns = []
        for col in df.columns:
            col_data = df[col]
            missing_count = int(col_data.isna().sum())
            missing_rate = float(missing_count / len(df)) if len(df) else 0.0
            samples = [str(x) for x in col_data.dropna().head(3).tolist()]
            columns.append(
                {
                    "name": str(col),
                    "dtype": str(col_data.dtype),
                    "missing_count": missing_count,
                    "missing_rate": round(missing_rate, 4),
                    "unique_count": int(col_data.nunique()),
                    "is_constant": bool(col_data.nunique() <= 1),
                    "sample_values": samples,
                }
            )
        return {
            "total_rows": int(len(df)),
            "total_columns": int(len(df.columns)),
            "columns": columns,
            "cleaning_recommendations": [],
        }
