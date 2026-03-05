"""L3 Layer - AutoML Hypothesis Generation (The Strategist)."""

from typing import Any, Dict, List, Optional

from core.agent_loop import AgentLoop, AgentLoopConfig
from core.evaluators import evaluate_l3_turn
from core.state import StateContext
from core.sandbox import CodeSandbox
from core.llm import LLMClient
from core.prompts import (
    L3_SYSTEM_PROMPT,
    L3_STRATEGY_SCHEMA,
    AUTOML_HYPOTHESES_SCHEMA,
    get_l3_user_prompt,
)
from layers._discovery import run_discovery
from layers._exploration import run_exploration


class L3Strategist:
    """
    L3: AutoML Hypothesis Generation Layer.

    Uses a hypothesis-driven strategy pattern:
    1. Bootstrap: hardcoded code analyzes data dimensions, feature types, class balance, scale
    2. LLM generates AutoML hypotheses (AUTOML_HYPOTHESES_SCHEMA) with 3-5 model hypotheses
    3. LLM generates report section covering characteristics + hypothesis descriptions
    4. Populate state from hypotheses for backward compat (l3_strategy)

    Input: All previous state + task_config
    Output: l3_strategy, l3_hypotheses, and L3 report section
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
        """Execute L3 AutoML hypothesis generation."""
        print("[L3] Starting AutoML hypothesis generation...")
        loop = AgentLoop("L3", self.loop_config, state, verbose=self.verbose)

        def step_fn(turn: int, prev_output: Optional[Dict[str, Any]]) -> Dict[str, Any]:
            code_snippets: List[Dict[str, Any]] = []
            llm_interactions: List[Dict[str, Any]] = []

            # Step 1a: Discovery (hardcoded, fast)
            discovery_result = run_discovery(self.sandbox, "L3", task_config)
            code_snippets.append({
                "description": "Discovery: data structure probing",
                "code": discovery_result.get("_code", ""),
                "stdout": discovery_result.get("stdout", ""),
                "success": discovery_result["success"],
            })

            # Step 1b: Exploration (LLM-generated, adaptive)
            prior_state_summary = state.get_summary_for_layer("L3")
            exploration_result = run_exploration(
                self.sandbox, self.llm, "L3",
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

            characteristics_stdout = (
                f"=== Discovery ===\n{discovery_result.get('stdout', '')}\n\n"
                f"=== Exploration ===\n{exploration_result.get('stdout', '')}"
            ) if discovery_ok or exploration_ok else ""

            # Step 2: LLM generates AutoML hypotheses
            prev_error = prev_output.get("error") if prev_output else None
            hyp_result = self._generate_automl_hypotheses(
                state, task_config, characteristics_stdout, turn, prev_error,
            )
            llm_interactions.append({
                "role": "L3_automl_hypotheses",
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
                    "n_hypotheses": 0,
                    "report_section_success": False,
                    "validation_strategy_present": False,
                    "error": hyp_result.get("error", "AutoML hypothesis generation failed"),
                    "observation": "Characteristics analyzed, but hypothesis generation failed",
                    "code_snippets": code_snippets,
                    "llm_interactions": llm_interactions,
                }

            automl_content = hyp_result.get("content", {})
            hypotheses = automl_content.get("hypotheses", [])
            validation_strategy = automl_content.get("validation_strategy", "")
            ensemble_recommendation = automl_content.get("ensemble_recommendation", "")

            # Step 3: LLM generates report section
            report_result = self._generate_report_section(
                characteristics_stdout, hypotheses, validation_strategy,
                ensemble_recommendation, task_config
            )
            llm_interactions.append({
                "role": "L3_report_generation",
                "user_prompt": report_result.get("_user_prompt", ""),
                "raw_response": (report_result.get("raw", "") or ""),
                "success": report_result["success"],
            })
            report_section_success = report_result["success"]
            report_section = report_result.get("content", "") if report_section_success else ""

            # Step 4: Build backward-compatible l3_strategy from hypotheses
            strategy = self._build_strategy_from_hypotheses(
                automl_content, state, task_config
            )

            return {
                "success": True,
                "discovery_success": discovery_ok,
                "exploration_success": exploration_ok,
                "hypothesis_gen_success": True,
                "n_hypotheses": len(hypotheses),
                "report_section_success": report_section_success,
                "validation_strategy_present": bool(validation_strategy),
                "strategy": strategy,
                "automl_hypotheses": hypotheses,
                "report_section": report_section,
                "observation": (
                    f"Generated {len(hypotheses)} model hypotheses; "
                    f"validation: {validation_strategy}; "
                    f"report {'generated' if report_section_success else 'failed'}"
                ),
                "code_snippets": code_snippets,
                "llm_interactions": llm_interactions,
            }

        def apply_fn(output: Dict[str, Any]) -> None:
            state.l3_strategy = output.get("strategy", {})
            state.l3_hypotheses = output.get("automl_hypotheses", [])
            report = output.get("report_section", "")
            if report:
                state.set_layer_report("L3", report)

        def fallback_fn(reason: str) -> Dict[str, Any]:
            state.l3_strategy = self._get_fallback_strategy(state, task_config)
            return {"status": "degraded", "reason": reason}

        loop_result = loop.run(
            objective="Generate AutoML modeling hypotheses with report section",
            step_fn=step_fn,
            evaluator_fn=evaluate_l3_turn,
            apply_fn=apply_fn,
            fallback_fn=fallback_fn,
        )

        if loop_result["status"] != "success":
            state.record_error("L3", "agent_loop", loop_result["reason"])

        n_models = len(state.l3_strategy.get("recommended_models", []))
        print(
            f"[L3] Completed with status={loop_result['status']} "
            f"(turns={loop_result['turns_used']}, best_score={loop_result['best_score']:.2f}, "
            f"models={n_models})"
        )

        state.advance_layer("REPORT")
        return loop_result["status"] != "failed"

    def _generate_automl_hypotheses(
        self,
        state: StateContext,
        task_config: Dict[str, Any],
        characteristics_stdout: str,
        turn: int,
        prev_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """LLM generates AutoML hypotheses."""
        l2_features = {
            "selected_features": state.l2_selected_features,
            "transformations": state.l2_transformations,
        }

        user_prompt = get_l3_user_prompt(
            state.l0_stats,
            state.l1_insights,
            l2_features,
            task_config,
            l1_hypotheses=state.l1_hypotheses,
            l2_domain_priors=state.l2_domain_priors,
            characteristics_stdout=characteristics_stdout,
        )
        if turn > 1 and prev_error:
            user_prompt += (
                "\n\nPrevious attempt failed. Return strict schema-compliant JSON only.\n"
                f"Previous error: {prev_error}"
            )

        result = self.llm.generate(
            system_prompt=L3_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            json_schema=AUTOML_HYPOTHESES_SCHEMA,
        )
        result["_user_prompt"] = user_prompt
        return result

    def _generate_report_section(
        self,
        characteristics_stdout: str,
        hypotheses: List[Dict[str, Any]],
        validation_strategy: str,
        ensemble_recommendation: str,
        task_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """LLM generates a markdown report section for L3."""
        hyp_summary = "\n".join(
            f"- {h.get('model_name', '?')} (priority {h.get('priority', '?')}): {h.get('rationale', '')}"
            for h in hypotheses
        )

        system_prompt = (
            "You are a technical report writer. Generate a concise markdown section "
            "for the Modeling Strategy (L3) portion of a data profiling report."
        )
        user_prompt = f"""Based on the following analysis, write a markdown report section.

Task: Predict '{task_config.get('target', 'N/A')}' ({task_config.get('task_type', 'auto')})

Data Characteristics:
{characteristics_stdout[:1500]}

Model Hypotheses:
{hyp_summary}

Validation Strategy: {validation_strategy}
Ensemble Recommendation: {ensemble_recommendation}

The section should cover:
1. Data characteristics summary (dimensions, types, balance, scale)
2. Detailed model hypothesis descriptions (for each: rationale, hyperparameters, strengths/weaknesses)
3. Preprocessing requirements per model
4. Validation strategy rationale
5. Ensemble recommendation

Use markdown headers (### level), tables where appropriate, and be specific/actionable.
Do NOT include a top-level heading."""

        result = self.llm.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        result["_user_prompt"] = user_prompt
        return result

    def _build_strategy_from_hypotheses(
        self,
        automl_content: Dict[str, Any],
        state: StateContext,
        task_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build backward-compatible l3_strategy from AUTOML_HYPOTHESES_SCHEMA output."""
        hypotheses = automl_content.get("hypotheses", [])

        recommended_models = []
        preprocessing_steps = set()

        for h in hypotheses:
            recommended_models.append({
                "name": h.get("model_name", "Unknown"),
                "priority": h.get("priority", 99),
                "reasons": h.get("strengths", []),
            })
            for step in h.get("preprocessing", []):
                preprocessing_steps.add(step)

        recommended_models.sort(key=lambda x: x["priority"])

        n_samples = state.l0_stats.get("total_rows", 0)
        n_features = len(state.l2_selected_features) or state.l0_stats.get("total_columns", 0)

        return {
            "recommended_models": recommended_models,
            "data_characteristics": {
                "n_samples": n_samples,
                "n_features": n_features,
                "feature_types": "mixed",
                "class_balance": "unknown",
                "data_scale": "small" if n_samples < 1000 else "medium" if n_samples < 100000 else "large",
            },
            "preprocessing_pipeline": list(preprocessing_steps) or [
                "Handle missing values",
                "Encode categorical features",
            ],
            "validation_strategy": automl_content.get("validation_strategy", "5-fold cross-validation"),
            "special_considerations": [],
        }

    def _get_fallback_strategy(
        self,
        state: StateContext,
        task_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate fallback strategy if LLM fails."""
        task_type = task_config.get("task_type", "classification")
        n_samples = state.l0_stats.get("total_rows", 1000)
        n_features = len(state.l2_selected_features) or state.l0_stats.get("total_columns", 10)

        if task_type == "regression":
            models = [
                {
                    "name": "LightGBM Regressor",
                    "priority": 1,
                    "reasons": ["Handles mixed feature types", "Fast training", "Good default performance"]
                },
                {
                    "name": "Random Forest Regressor",
                    "priority": 2,
                    "reasons": ["Robust to overfitting", "Interpretable feature importance"]
                }
            ]
        else:
            models = [
                {
                    "name": "LightGBM Classifier",
                    "priority": 1,
                    "reasons": ["Handles class imbalance", "Fast training", "Good for tabular data"]
                },
                {
                    "name": "Random Forest Classifier",
                    "priority": 2,
                    "reasons": ["Robust baseline", "Interpretable"]
                }
            ]

        return {
            "recommended_models": models,
            "data_characteristics": {
                "n_samples": n_samples,
                "n_features": n_features,
                "feature_types": "mixed",
                "class_balance": "unknown",
                "data_scale": "small" if n_samples < 1000 else "medium" if n_samples < 100000 else "large"
            },
            "preprocessing_pipeline": [
                "Handle missing values (median for numeric, mode for categorical)",
                "Encode categorical features (LabelEncoder or OneHotEncoder)",
                "Apply recommended transformations from L2"
            ],
            "validation_strategy": "5-fold stratified cross-validation" if task_type == "classification" else "5-fold cross-validation",
            "special_considerations": []
        }
