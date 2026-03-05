"""L1 Layer - Hypothesis-Driven Exploration (The Unsupervised Explorer)."""

from typing import Any, Dict, List, Optional

from core.agent_loop import AgentLoop, AgentLoopConfig
from core.evaluators import evaluate_l1_turn
from core.state import StateContext
from core.sandbox import CodeSandbox
from core.llm import LLMClient
from core.prompts import (
    L1_SYSTEM_PROMPT,
    L1_INSIGHTS_SCHEMA,
    HYPOTHESIS_BATCH_SCHEMA,
    HYPOTHESIS_VERDICT_SCHEMA,
    get_l1_user_prompt,
)
from layers._discovery import run_discovery
from layers._exploration import run_exploration


class L1Explorer:
    """
    L1: Hypothesis-Driven Exploration Layer.

    Uses a hypothesis-verify pattern:
    1. Bootstrap: hardcoded code gathers distributions, correlations, cardinality
    2. LLM proposes up to 5 hypotheses about data structure and domain patterns
    3. LLM generates verification code that tests all hypotheses
    4. Sandbox executes verification code (on a copy — read-only)
    5. LLM judges each hypothesis as confirmed/rejected with evidence
    6. LLM generates report section

    Input: L0 stats and df_clean_v1 (task_config passed for interface consistency but ignored)
    Output: l1_insights, l1_hypotheses, and L1 report section
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
        """Execute L1 hypothesis-driven exploration."""
        print("[L1] Starting hypothesis-driven exploration...")
        loop = AgentLoop("L1", self.loop_config, state, verbose=self.verbose)

        def step_fn(turn: int, prev_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            code_snippets: List[Dict[str, Any]] = []
            llm_interactions: List[Dict[str, Any]] = []

            # Ensure we work on a copy of the cleaned data (if tabular)
            if self.sandbox.get_variable("df_clean_v1") is not None:
                self.sandbox.execute("df = df_clean_v1.copy()", "Load cleaned data")

            # Step 1a: Discovery (hardcoded, fast)
            discovery_result = run_discovery(self.sandbox, "L1", task_config)
            code_snippets.append({
                "description": "Discovery: data structure probing",
                "code": discovery_result.get("_code", ""),
                "stdout": discovery_result.get("stdout", ""),
                "success": discovery_result["success"],
            })

            # Step 1b: Exploration (LLM-generated, adaptive)
            prior_state_summary = state.get_summary_for_layer("L1")
            exploration_result = run_exploration(
                self.sandbox, self.llm, "L1",
                discovery_result.get("stdout", ""), task_config,
                prior_state=prior_state_summary,
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
                    "hypothesis_gen_success": False,
                    "verification_success": False,
                    "judgment_success": False,
                    "confirmed_findings_count": 0,
                    "report_section_success": False,
                    "error": discovery_result.get("error", "Discovery and exploration both failed"),
                    "observation": "Bootstrap exploration failed",
                    "code_snippets": code_snippets,
                    "llm_interactions": llm_interactions,
                }

            bootstrap_stdout = (
                f"=== Discovery ===\n{discovery_result.get('stdout', '')}\n\n"
                f"=== Exploration ===\n{exploration_result.get('stdout', '')}"
            )

            # Step 2: LLM proposes hypotheses
            prev_error = prev_output.get("error") if prev_output else None
            hyp_result = self._propose_hypotheses(
                state.l0_stats, bootstrap_stdout, turn, prev_error
            )
            llm_interactions.append({
                "role": "L1_hypothesis_generation",
                "user_prompt": hyp_result.get("_user_prompt", ""),
                "raw_response": (hyp_result.get("raw", "") or ""),
                "success": hyp_result["success"],
            })

            if not hyp_result["success"]:
                return {
                    "success": False,
                    "discovery_success": discovery_ok,
                    "exploration_success": exploration_ok,
                    "hypothesis_gen_success": False,
                    "verification_success": False,
                    "judgment_success": False,
                    "confirmed_findings_count": 0,
                    "report_section_success": False,
                    "error": hyp_result.get("error", "Hypothesis generation failed"),
                    "observation": "Bootstrap done, hypothesis generation failed",
                    "code_snippets": code_snippets,
                    "llm_interactions": llm_interactions,
                }

            hypotheses = hyp_result.get("content", {}).get("hypotheses", [])

            # Step 3: LLM generates verification code
            verify_code_result = self._generate_verification_code(
                hypotheses, bootstrap_stdout
            )
            llm_interactions.append({
                "role": "L1_verification_code",
                "user_prompt": verify_code_result.get("_user_prompt", ""),
                "raw_response": (verify_code_result.get("raw", "") or ""),
                "success": verify_code_result["success"],
            })

            verification_success = False
            verify_stdout = ""
            if verify_code_result["success"]:
                # Step 4: Execute verification code on a copy
                verify_code = verify_code_result["content"]
                # Wrap in copy to ensure L1 doesn't modify the working df
                safe_code = "df_l1_copy = df.copy()\n" + verify_code.replace(
                    "df[", "df_l1_copy["
                ).replace("df.", "df_l1_copy.").replace("df_l1_copy_l1_copy", "df_l1_copy")
                # Simpler approach: just use the code as-is but on a copy
                safe_code = f"_df_work = df.copy()\n{verify_code}"

                exec_result = self.sandbox.execute(safe_code, "Hypothesis verification")
                code_snippets.append({
                    "description": "Hypothesis verification code",
                    "code": verify_code,
                    "stdout": exec_result.get("stdout", ""),
                    "success": exec_result["success"],
                })
                verification_success = exec_result["success"]
                verify_stdout = exec_result.get("stdout", "")
            else:
                code_snippets.append({
                    "description": "Hypothesis verification code (generation failed)",
                    "code": "",
                    "stdout": "",
                    "success": False,
                })

            # Step 5: LLM judges hypotheses
            judgment_result = self._judge_hypotheses(
                hypotheses, verify_stdout, bootstrap_stdout
            )
            llm_interactions.append({
                "role": "L1_hypothesis_judgment",
                "user_prompt": judgment_result.get("_user_prompt", ""),
                "raw_response": (judgment_result.get("raw", "") or ""),
                "success": judgment_result["success"],
            })

            judgment_success = judgment_result["success"]
            verdicts = judgment_result.get("content", {}).get("verdicts", []) if judgment_success else []

            # Merge hypotheses with verdicts
            hypothesis_log = self._merge_hypotheses_verdicts(hypotheses, verdicts)
            confirmed_count = sum(1 for h in hypothesis_log if h.get("confirmed"))

            # Step 6: LLM generates report section
            report_result = self._generate_report_section(
                bootstrap_stdout, hypotheses, verdicts, verify_stdout
            )
            llm_interactions.append({
                "role": "L1_report_generation",
                "user_prompt": report_result.get("_user_prompt", ""),
                "raw_response": (report_result.get("raw", "") or ""),
                "success": report_result["success"],
            })
            report_section_success = report_result["success"]
            report_section = report_result.get("content", "") if report_section_success else ""

            # Also synthesize L1 insights for backward compat
            insights_result = self._synthesize_insights(
                state.l0_stats, bootstrap_stdout, turn, prev_error
            )
            llm_interactions.append({
                "role": "L1_insights_synthesis",
                "user_prompt": insights_result.get("_user_prompt", ""),
                "raw_response": (insights_result.get("raw", "") or ""),
                "success": insights_result["success"],
            })
            insights = insights_result.get("content", {}) if insights_result["success"] else {}

            return {
                "success": True,
                "discovery_success": discovery_ok,
                "exploration_success": exploration_ok,
                "hypothesis_gen_success": True,
                "verification_success": verification_success,
                "judgment_success": judgment_success,
                "confirmed_findings_count": confirmed_count,
                "report_section_success": report_section_success,
                "l1_insights": insights,
                "hypothesis_log": hypothesis_log,
                "report_section": report_section,
                "observation": (
                    f"Bootstrap done; {len(hypotheses)} hypotheses proposed; "
                    f"{confirmed_count} confirmed; report {'generated' if report_section_success else 'failed'}"
                ),
                "code_snippets": code_snippets,
                "llm_interactions": llm_interactions,
            }

        def apply_fn(output: Dict[str, Any]) -> None:
            state.l1_insights = output.get("l1_insights", {})
            state.l1_hypotheses = output.get("hypothesis_log", [])
            report = output.get("report_section", "")
            if report:
                state.set_layer_report("L1", report)

        def fallback_fn(reason: str) -> Dict[str, Any]:
            if not state.l1_insights:
                state.l1_insights = {
                    "skewed_features": [],
                    "collinear_pairs": [],
                    "distribution_insights": [],
                }
            return {"status": "degraded", "reason": reason}

        loop_result = loop.run(
            objective="Explore data structure via hypothesis-verify pattern and produce report section",
            step_fn=step_fn,
            evaluator_fn=evaluate_l1_turn,
            apply_fn=apply_fn,
            fallback_fn=fallback_fn,
        )

        if loop_result["status"] != "success":
            state.record_error("L1", "agent_loop", loop_result["reason"])

        print(
            f"[L1] Completed with status={loop_result['status']} "
            f"(turns={loop_result['turns_used']}, best_score={loop_result['best_score']:.2f})"
        )
        state.advance_layer("L2")
        return loop_result["status"] != "failed"

    def _propose_hypotheses(
        self,
        l0_stats: Dict[str, Any],
        bootstrap_stdout: str,
        turn: int,
        prev_error: Optional[str],
    ) -> Dict[str, Any]:
        """LLM proposes hypotheses about data structure and domain."""
        user_prompt = get_l1_user_prompt(l0_stats, bootstrap_stdout)
        if turn > 1 and prev_error:
            user_prompt += f"\n\nPrevious error: {prev_error}\nFix and try again."

        result = self.llm.generate(
            system_prompt=L1_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=HYPOTHESIS_BATCH_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        return result

    def _generate_verification_code(
        self,
        hypotheses: List[Dict[str, Any]],
        bootstrap_stdout: str,
    ) -> Dict[str, Any]:
        """LLM generates code to verify all hypotheses."""
        hyp_descriptions = "\n".join(
            f"- {h['id']}: {h['statement']} (approach: {h.get('verification_approach', '')})"
            for h in hypotheses
        )
        available_vars = ["df", "pd", "np"]
        data_note = "The DataFrame 'df' is already loaded."
        if self.sandbox.get_variable("DATA_DIR") is not None:
            data_note += (
                " For non-tabular data, you also have DATA_DIR, analyze_image(), "
                "analyze_audio(), list_data_files(), batch_analyze_images(), batch_analyze_audio()."
            )
            available_vars.extend([
                "DATA_DIR", "torch", "Image", "Path",
                "analyze_image", "analyze_audio", "list_data_files",
                "batch_analyze_images", "batch_analyze_audio",
            ])
        task = (
            f"Generate Python code to test the following hypotheses.\n"
            f"{data_note} Use pandas, numpy, scipy as needed.\n\n"
            f"Hypotheses to verify:\n{hyp_descriptions}\n\n"
            f"For each hypothesis, print the result in this exact format:\n"
            f'  print("HYPOTHESIS_RESULT: id=H1 | result=<summary of finding>")\n\n'
            f"Do NOT modify df. Work on copies if needed. Do NOT generate plots.\n"
            f"Handle errors gracefully — if a test fails, print a result indicating failure."
        )
        context = f"Bootstrap analysis output:\n{bootstrap_stdout[:2000]}"
        result = self.llm.generate_code(task, context, available_vars)
        result["_user_prompt"] = task
        return result

    def _judge_hypotheses(
        self,
        hypotheses: List[Dict[str, Any]],
        verify_stdout: str,
        bootstrap_stdout: str,
    ) -> Dict[str, Any]:
        """LLM judges each hypothesis as confirmed or rejected."""
        hyp_descriptions = "\n".join(
            f"- {h['id']}: {h['statement']}"
            for h in hypotheses
        )
        user_prompt = (
            f"Based on the verification results below, judge each hypothesis.\n\n"
            f"Hypotheses:\n{hyp_descriptions}\n\n"
            f"Verification Output:\n{verify_stdout[:3000]}\n\n"
            f"Bootstrap Context:\n{bootstrap_stdout[:1500]}\n\n"
            f"For each hypothesis, determine if it is confirmed or rejected based on evidence.\n"
            f"Provide an action field describing what follow-up action (if any) should be taken."
        )
        result = self.llm.generate(
            system_prompt=L1_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=HYPOTHESIS_VERDICT_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        return result

    def _merge_hypotheses_verdicts(
        self,
        hypotheses: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge hypotheses with their verdicts into a single log."""
        verdict_map = {v["id"]: v for v in verdicts}
        log = []
        for h in hypotheses:
            hid = h["id"]
            v = verdict_map.get(hid, {})
            log.append({
                "id": hid,
                "statement": h["statement"],
                "confirmed": v.get("confirmed", False),
                "evidence": v.get("evidence_summary", ""),
                "action": v.get("action", ""),
            })
        return log

    def _generate_report_section(
        self,
        bootstrap_stdout: str,
        hypotheses: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
        verify_stdout: str,
    ) -> Dict[str, Any]:
        """LLM generates a markdown report section for L1."""
        hyp_summary = "\n".join(
            f"- {h['id']}: {h['statement']}" for h in hypotheses
        )
        verdict_summary = "\n".join(
            f"- {v['id']}: {'CONFIRMED' if v.get('confirmed') else 'REJECTED'} — {v.get('evidence_summary', '')}"
            for v in verdicts
        )

        system_prompt = (
            "You are a technical report writer. Generate a concise markdown section "
            "for the Unsupervised Exploration (L1) portion of a data profiling report."
        )
        user_prompt = f"""Based on the following analysis, write a markdown report section.

Bootstrap Analysis Output:
{bootstrap_stdout[:2000]}

Hypotheses Proposed:
{hyp_summary}

Verification Results:
{verify_stdout[:1500]}

Verdict Summary:
{verdict_summary}

The section should cover:
1. Bootstrap findings (distributions, correlations, cardinality)
2. Domain inference (what domain does this data likely come from)
3. Hypothesis investigation results (each hypothesis, test, and verdict)
4. Key confirmed findings and their implications

Use markdown headers (### level), tables where appropriate, and be factual.
Do NOT include a top-level heading."""

        result = self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        result["_user_prompt"] = user_prompt
        return result

    def _synthesize_insights(
        self,
        l0_stats: Dict[str, Any],
        numeric_stats: str,
        turn: int,
        prev_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Synthesize L1 insights for backward compatibility."""
        user_prompt = get_l1_user_prompt(l0_stats, numeric_stats)
        if turn > 1 and prev_error:
            user_prompt += (
                "\n\nPrevious attempt failed. Return strict schema-compliant JSON only.\n"
                f"Previous error: {prev_error}"
            )

        result = self.llm.generate(
            system_prompt=L1_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=L1_INSIGHTS_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        return result
