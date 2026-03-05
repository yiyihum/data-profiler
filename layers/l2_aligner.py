"""L2 Layer - Domain-Prior-Verify-Apply (The Task Aligner)."""

import json
from typing import Any, Dict, List, Optional

from core.agent_loop import AgentLoop, AgentLoopConfig
from core.evaluators import evaluate_l2_turn
from core.state import StateContext
from core.sandbox import CodeSandbox
from core.llm import LLMClient
from core.prompts import (
    L2_SYSTEM_PROMPT,
    L2_FEATURES_SCHEMA,
    HYPOTHESIS_BATCH_SCHEMA,
    HYPOTHESIS_VERDICT_SCHEMA,
    get_l2_user_prompt,
)
from layers._discovery import run_discovery
from layers._exploration import run_exploration


class L2Aligner:
    """
    L2: Domain-Prior-Verify-Apply Layer.

    Uses a domain-prior-verify-apply pattern:
    1. Bootstrap: hardcoded code computes MI scores, target correlation, cardinality vs target
    2. LLM proposes up to 5 task-informed domain priors
    3. LLM generates verification code for each prior
    4. Sandbox executes verification code
    5. LLM judges priors (confirmed priors include action field)
    6. LLM generates application code for confirmed priors -> df_clean_v2
    7. Sandbox executes application code
    8. LLM generates report section
    9. LLM synthesizes JSON (L2_FEATURES_SCHEMA)

    Input: L0 stats, L1 insights + hypotheses, df_clean_v1, and task_config
    Output: l2_selected_features, l2_transformations, l2_domain_priors, df_clean_v2, report section
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
        """Execute L2 domain-prior-verify-apply analysis."""
        print("[L2] Starting domain-prior-verify-apply analysis...")
        print(f"[L2] Target column: {task_config.get('target', 'N/A')}")
        loop = AgentLoop("L2", self.loop_config, state, verbose=self.verbose)

        target = task_config.get("target") or ""

        def step_fn(turn: int, prev_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            code_snippets: List[Dict[str, Any]] = []
            llm_interactions: List[Dict[str, Any]] = []

            # Load cleaned data (if tabular)
            if self.sandbox.get_variable("df_clean_v1") is not None:
                self.sandbox.execute("df = df_clean_v1.copy()", "Load cleaned data")

            # Get actual column names
            df_now = self.sandbox.get_dataframe()
            real_columns = df_now.columns.tolist() if df_now is not None else []

            # Step 1a: Discovery (hardcoded, fast)
            discovery_result = run_discovery(self.sandbox, "L2", task_config)
            code_snippets.append({
                "description": "Discovery: data structure probing",
                "code": discovery_result.get("_code", ""),
                "stdout": discovery_result.get("stdout", ""),
                "success": discovery_result["success"],
            })

            # Step 1b: Exploration (LLM-generated, adaptive)
            prior_state_summary = state.get_summary_for_layer("L2")
            exploration_result = run_exploration(
                self.sandbox, self.llm, "L2",
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
                return self._failure_output(
                    "feature_analysis",
                    discovery_result.get("error", "Discovery and exploration both failed"),
                    code_snippets, llm_interactions,
                )

            feature_stdout = (
                f"=== Discovery ===\n{discovery_result.get('stdout', '')}\n\n"
                f"=== Exploration ===\n{exploration_result.get('stdout', '')}"
            )
            prev_error = prev_output.get("error") if prev_output else None

            # Step 2: LLM proposes domain priors
            prior_result = self._propose_domain_priors(
                state, task_config, feature_stdout, real_columns, turn, prev_error,
            )
            llm_interactions.append({
                "role": "L2_prior_generation",
                "user_prompt": prior_result.get("_user_prompt", ""),
                "raw_response": (prior_result.get("raw", "") or ""),
                "success": prior_result["success"],
            })

            prior_gen_success = prior_result["success"]
            priors = prior_result.get("content", {}).get("hypotheses", []) if prior_gen_success else []

            # Step 3: LLM generates verification code
            verification_success = False
            verify_stdout = ""
            if priors:
                verify_code_result = self._generate_verification_code(
                    priors, feature_stdout, target, real_columns
                )
                llm_interactions.append({
                    "role": "L2_verification_code",
                    "user_prompt": verify_code_result.get("_user_prompt", ""),
                    "raw_response": (verify_code_result.get("raw", "") or ""),
                    "success": verify_code_result["success"],
                })

                if verify_code_result["success"]:
                    # Step 4: Execute verification code
                    verify_code = verify_code_result["content"]
                    exec_result = self.sandbox.execute(verify_code, "Prior verification")
                    code_snippets.append({
                        "description": "Domain prior verification code",
                        "code": verify_code,
                        "stdout": exec_result.get("stdout", ""),
                        "success": exec_result["success"],
                    })
                    verification_success = exec_result["success"]
                    verify_stdout = exec_result.get("stdout", "")

            # Step 5: LLM judges priors
            judgment_result = self._judge_priors(priors, verify_stdout, feature_stdout)
            llm_interactions.append({
                "role": "L2_prior_judgment",
                "user_prompt": judgment_result.get("_user_prompt", ""),
                "raw_response": (judgment_result.get("raw", "") or ""),
                "success": judgment_result["success"],
            })
            judgment_success = judgment_result["success"]
            verdicts = judgment_result.get("content", {}).get("verdicts", []) if judgment_success else []

            # Merge priors with verdicts
            domain_priors = self._merge_priors_verdicts(priors, verdicts)

            # Step 6: LLM generates application code for confirmed priors
            confirmed_priors = [p for p in domain_priors if p.get("confirmed")]
            apply_success = False
            apply_stdout = ""

            if confirmed_priors:
                apply_code_result = self._generate_apply_code(
                    confirmed_priors, target, real_columns
                )
                llm_interactions.append({
                    "role": "L2_apply_code",
                    "user_prompt": apply_code_result.get("_user_prompt", ""),
                    "raw_response": (apply_code_result.get("raw", "") or ""),
                    "success": apply_code_result["success"],
                })

                if apply_code_result["success"]:
                    # Step 7: Execute application code
                    apply_code = apply_code_result["content"]
                    apply_exec = self.sandbox.execute(apply_code, "Apply confirmed priors")
                    code_snippets.append({
                        "description": "Apply confirmed domain priors",
                        "code": apply_code,
                        "stdout": apply_exec.get("stdout", ""),
                        "success": apply_exec["success"],
                    })
                    apply_success = apply_exec["success"]
                    apply_stdout = apply_exec.get("stdout", "")

            # Ensure df_clean_v2 exists
            if self.sandbox.get_variable("df_clean_v2") is None:
                self.sandbox.execute("df_clean_v2 = df.copy()", "Fallback create df_clean_v2")

            # Step 8: LLM generates report section
            report_result = self._generate_report_section(
                feature_stdout, priors, verdicts, verify_stdout, apply_stdout, task_config
            )
            llm_interactions.append({
                "role": "L2_report_generation",
                "user_prompt": report_result.get("_user_prompt", ""),
                "raw_response": (report_result.get("raw", "") or ""),
                "success": report_result["success"],
            })
            report_section_success = report_result["success"]
            report_section = report_result.get("content", "") if report_section_success else ""

            # Step 9: LLM synthesizes JSON (L2_FEATURES_SCHEMA)
            combined_observations = (
                f"=== Feature-Target Analysis ===\n{feature_stdout[:1500]}\n\n"
                f"=== Application Output ===\n{apply_stdout[:1500]}\n"
            )
            json_result = self._synthesize_json(
                state, task_config, combined_observations, turn, prev_error,
            )
            llm_interactions.append({
                "role": "L2_json_synthesis",
                "user_prompt": combined_observations,
                "raw_response": (json_result.get("raw", "") or ""),
                "success": json_result["success"],
            })

            json_synthesis_success = json_result["success"]
            l2_result = json_result.get("content", {}) if json_synthesis_success else {}

            # Validate feature names against real columns
            if l2_result and real_columns:
                l2_result = self._validate_feature_names(l2_result, real_columns, target)

            selected_features = l2_result.get("selected_features", [])
            final_df_ready = self.sandbox.get_variable("df_clean_v2") is not None

            return {
                "success": True,
                "discovery_success": discovery_ok,
                "exploration_success": exploration_ok,
                "prior_gen_success": prior_gen_success,
                "verification_success": verification_success,
                "judgment_success": judgment_success,
                "apply_success": apply_success,
                "json_synthesis_success": json_synthesis_success,
                "report_section_success": report_section_success,
                "selected_features_count": len(selected_features),
                "final_df_ready": final_df_ready,
                "l2_result": l2_result,
                "domain_priors": domain_priors,
                "report_section": report_section,
                "observation": (
                    f"Feature analysis done; {len(priors)} priors proposed; "
                    f"{len(confirmed_priors)} confirmed; "
                    f"apply {'succeeded' if apply_success else 'skipped/failed'}; "
                    f"selected={len(selected_features)} features"
                ),
                "code_snippets": code_snippets,
                "llm_interactions": llm_interactions,
            }

        def apply_fn(output: Dict[str, Any]) -> None:
            l2_result = output.get("l2_result", {})
            state.l2_selected_features = l2_result.get("selected_features", [])
            state.l2_transformations = l2_result.get("transformations", [])
            state.l2_domain_priors = output.get("domain_priors", [])
            report = output.get("report_section", "")
            if report:
                state.set_layer_report("L2", report)
            if self.sandbox.get_variable("df_clean_v2") is None and self.sandbox.get_dataframe() is not None:
                self.sandbox.set_variable("df_clean_v2", self.sandbox.get_dataframe().copy())

        def fallback_fn(reason: str) -> Dict[str, Any]:
            if self.sandbox.get_variable("df_clean_v2") is None:
                if self.sandbox.get_variable("df_clean_v1") is not None:
                    self.sandbox.execute("df_clean_v2 = df_clean_v1.copy()", "Fallback create df_clean_v2")
                elif self.sandbox.get_dataframe() is not None:
                    self.sandbox.set_variable("df_clean_v2", self.sandbox.get_dataframe().copy())
            if not state.l2_selected_features:
                df_now = self.sandbox.get_dataframe()
                if df_now is not None:
                    state.l2_selected_features = [c for c in df_now.columns if c != target]
                else:
                    state.l2_selected_features = []
            if not state.l2_transformations:
                state.l2_transformations = []
            return {"status": "degraded", "reason": reason}

        loop_result = loop.run(
            objective="Verify domain priors against data and produce task-aligned feature set",
            step_fn=step_fn,
            evaluator_fn=evaluate_l2_turn,
            apply_fn=apply_fn,
            fallback_fn=fallback_fn,
        )

        if loop_result["status"] != "success":
            state.record_error("L2", "agent_loop", loop_result["reason"])

        print(
            f"[L2] Completed with status={loop_result['status']} "
            f"(turns={loop_result['turns_used']}, best_score={loop_result['best_score']:.2f})"
        )
        state.advance_layer("L3")
        return loop_result["status"] != "failed"

    def _failure_output(
        self, stage: str, error: str,
        code_snippets: list, llm_interactions: list,
    ) -> Dict[str, Any]:
        """Build a standardized failure output."""
        return {
            "success": False,
            "discovery_success": False,
            "exploration_success": False,
            "prior_gen_success": False,
            "verification_success": False,
            "judgment_success": False,
            "apply_success": False,
            "json_synthesis_success": False,
            "report_section_success": False,
            "selected_features_count": 0,
            "final_df_ready": False,
            "error": error,
            "observation": f"{stage} failed",
            "code_snippets": code_snippets,
            "llm_interactions": llm_interactions,
        }

    def _propose_domain_priors(
        self,
        state: StateContext,
        task_config: Dict[str, Any],
        feature_stdout: str,
        real_columns: list,
        turn: int,
        prev_error: Optional[str],
    ) -> Dict[str, Any]:
        """LLM proposes domain-informed priors."""
        user_prompt = get_l2_user_prompt(
            state.l0_stats,
            state.l1_insights,
            task_config,
            feature_stdout,
            l1_hypotheses=state.l1_hypotheses,
        )
        if turn > 1 and prev_error:
            user_prompt += f"\n\nPrevious error: {prev_error}\nFix and try again."

        result = self.llm.generate(
            system_prompt=L2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=HYPOTHESIS_BATCH_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        return result

    def _generate_verification_code(
        self,
        priors: List[Dict[str, Any]],
        feature_stdout: str,
        target: str,
        real_columns: list,
    ) -> Dict[str, Any]:
        """LLM generates code to verify domain priors."""
        prior_descriptions = "\n".join(
            f"- {p['id']}: {p['statement']} (approach: {p.get('verification_approach', '')})"
            for p in priors
        )
        column_list = ", ".join(f'"{c}"' for c in real_columns)
        available_vars = ["df", "pd", "np"]
        data_note = ""
        if self.sandbox.get_variable("DATA_DIR") is not None:
            data_note = (
                "\nFor non-tabular data, you also have: DATA_DIR, analyze_image(), "
                "analyze_audio(), list_data_files(), batch_analyze_images(), batch_analyze_audio().\n"
            )
            available_vars.extend([
                "DATA_DIR", "torch", "Image", "Path",
                "analyze_image", "analyze_audio", "list_data_files",
                "batch_analyze_images", "batch_analyze_audio",
            ])
        task = (
            f"Generate Python code to verify the following domain priors.\n"
            f"Target column: '{target}'\n"
            f"Available columns: [{column_list}]\n"
            f"{data_note}\n"
            f"Priors to verify:\n{prior_descriptions}\n\n"
            f"For each prior, print the result:\n"
            f'  print("HYPOTHESIS_RESULT: id=P1 | result=<summary of finding>")\n\n'
            f"Do NOT modify df. Do NOT generate plots."
        )
        context = f"Feature-target analysis:\n{feature_stdout[:2000]}"
        result = self.llm.generate_code(task, context, available_vars)
        result["_user_prompt"] = task
        return result

    def _judge_priors(
        self,
        priors: List[Dict[str, Any]],
        verify_stdout: str,
        feature_stdout: str,
    ) -> Dict[str, Any]:
        """LLM judges each prior as confirmed or rejected."""
        prior_descriptions = "\n".join(
            f"- {p['id']}: {p['statement']}" for p in priors
        )
        user_prompt = (
            f"Judge each domain prior based on the verification results.\n\n"
            f"Priors:\n{prior_descriptions}\n\n"
            f"Verification Output:\n{verify_stdout[:3000]}\n\n"
            f"Feature Analysis Context:\n{feature_stdout[:1500]}\n\n"
            f"For confirmed priors, the 'action' field should describe a concrete "
            f"transformation, feature selection, or engineering action to apply "
            f"(e.g., 'log-transform feature X', 'drop feature Y', 'create interaction X*Y')."
        )
        result = self.llm.generate(
            system_prompt=L2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=HYPOTHESIS_VERDICT_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        return result

    def _merge_priors_verdicts(
        self,
        priors: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Merge priors with their verdicts."""
        verdict_map = {v["id"]: v for v in verdicts}
        log = []
        for p in priors:
            pid = p["id"]
            v = verdict_map.get(pid, {})
            log.append({
                "id": pid,
                "statement": p["statement"],
                "confirmed": v.get("confirmed", False),
                "evidence": v.get("evidence_summary", ""),
                "action": v.get("action", ""),
            })
        return log

    def _generate_apply_code(
        self,
        confirmed_priors: List[Dict[str, Any]],
        target: str,
        real_columns: list,
    ) -> Dict[str, Any]:
        """LLM generates code to apply confirmed prior actions."""
        action_descriptions = "\n".join(
            f"- {p['id']}: {p.get('action', 'no action specified')}"
            for p in confirmed_priors
        )
        column_list = ", ".join(f'"{c}"' for c in real_columns)
        task = (
            f"Generate Python code to apply the following confirmed domain prior actions.\n"
            f"Start from 'df' and create 'df_clean_v2' as the result.\n"
            f"Target column: '{target}'\n"
            f"Available columns: [{column_list}]\n\n"
            f"Actions to apply:\n{action_descriptions}\n\n"
            f"For each action, print:\n"
            f'  print("TRANSFORM: feature=X | transform=Y | reason=Z")\n'
            f'  print("SELECT: feature=X | reason=Y")\n'
            f'  print("DROP: feature=X | reason=Y")\n\n'
            f"At the end:\n"
            f'  print(f"FINAL_COLUMNS: {{df_clean_v2.columns.tolist()}}")\n'
            f'  print(f"FINAL_SHAPE: {{df_clean_v2.shape}}")\n\n'
            f"Do NOT generate plots. Handle missing columns gracefully."
        )
        result = self.llm.generate_code(task, "", ["df", "pd", "np", "OUTPUT_DIR"])
        result["_user_prompt"] = task
        return result

    def _generate_report_section(
        self,
        feature_stdout: str,
        priors: List[Dict[str, Any]],
        verdicts: List[Dict[str, Any]],
        verify_stdout: str,
        apply_stdout: str,
        task_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM generates a markdown report section for L2."""
        prior_summary = "\n".join(
            f"- {p['id']}: {p['statement']}" for p in priors
        )
        verdict_summary = "\n".join(
            f"- {v['id']}: {'CONFIRMED' if v.get('confirmed') else 'REJECTED'} — {v.get('evidence_summary', '')}"
            for v in verdicts
        )

        system_prompt = (
            "You are a technical report writer. Generate a concise markdown section "
            "for the Task-Aligned Feature Analysis (L2) portion of a data profiling report."
        )
        user_prompt = f"""Based on the following analysis, write a markdown report section.

Task: Predict '{task_config.get('target', 'N/A')}' ({task_config.get('task_type', 'auto')})
Description: {task_config.get('description', 'N/A')}

Feature-Target Analysis:
{feature_stdout[:1500]}

Domain Priors Proposed:
{prior_summary}

Verification Results:
{verify_stdout[:1000]}

Verdict Summary:
{verdict_summary}

Application Results:
{apply_stdout[:1000]}

The section should cover:
1. Feature-target relationship analysis (MI scores, correlations)
2. Domain priors investigated and their verdicts
3. Transformations applied based on confirmed priors
4. Final feature set rationale

Use markdown headers (### level), tables where appropriate, and be factual.
Do NOT include a top-level heading."""

        result = self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        result["_user_prompt"] = user_prompt
        return result

    def _synthesize_json(
        self,
        state: StateContext,
        task_config: Dict[str, Any],
        observations: str,
        turn: int,
        prev_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """LLM synthesizes structured L2 JSON from all observations."""
        user_prompt = get_l2_user_prompt(
            state.l0_stats,
            state.l1_insights,
            task_config,
            observations,
            l1_hypotheses=state.l1_hypotheses,
        )
        if turn > 1 and prev_error:
            user_prompt += (
                "\n\nPrevious attempt failed. Return strict schema-compliant JSON only.\n"
                f"Previous error: {prev_error}"
            )

        result = self.llm.generate(
            system_prompt=L2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=L2_FEATURES_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        result["_system_prompt"] = L2_SYSTEM_PROMPT
        return result

    def _validate_feature_names(
        self,
        l2_result: Dict[str, Any],
        real_columns: list,
        target: str,
    ) -> Dict[str, Any]:
        """Validate and filter feature names against real DataFrame columns."""
        real_set = set(real_columns)

        selected = l2_result.get("selected_features", [])
        valid_selected = [f for f in selected if f in real_set]
        if len(valid_selected) < len(selected):
            hallucinated = set(selected) - set(valid_selected)
            print(f"[L2] WARNING: Filtered {len(hallucinated)} hallucinated features: {hallucinated}")
        if not valid_selected:
            valid_selected = [c for c in real_columns if c != target]
            print(f"[L2] Fallback: using all non-target columns as selected features")
        l2_result["selected_features"] = valid_selected

        fi = l2_result.get("feature_importance", [])
        l2_result["feature_importance"] = [f for f in fi if f.get("name", "") in real_set]

        transforms = l2_result.get("transformations", [])
        l2_result["transformations"] = [t for t in transforms if t.get("feature", "") in real_set]

        dropped = l2_result.get("dropped_features", [])
        l2_result["dropped_features"] = [d for d in dropped if d.get("name", "") in real_set]

        return l2_result
