"""Prompt Templates - Structured prompts for each layer of the pipeline."""

from typing import Any, Dict, List

# ──────────────────────────────────────────────────────────────
# JSON Schemas for structured output
# ──────────────────────────────────────────────────────────────

L0_STATS_SCHEMA = {
    "type": "object",
    "properties": {
        "total_rows": {"type": "integer"},
        "total_columns": {"type": "integer"},
        "columns": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dtype": {"type": "string"},
                    "missing_count": {"type": "integer"},
                    "missing_rate": {"type": "number"},
                    "unique_count": {"type": "integer"},
                    "is_constant": {"type": "boolean"},
                    "sample_values": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "dtype", "missing_count", "missing_rate", "unique_count", "is_constant", "sample_values"],
                "additionalProperties": False
            }
        },
        "cleaning_recommendations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "target": {"type": "string"},
                    "reason": {"type": "string"},
                    "confidence": {"type": "string"}
                },
                "required": ["action", "target", "reason", "confidence"],
                "additionalProperties": False
            }
        }
    },
    "required": ["total_rows", "total_columns", "columns", "cleaning_recommendations"],
    "additionalProperties": False
}

L1_INSIGHTS_SCHEMA = {
    "type": "object",
    "properties": {
        "skewed_features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "skewness": {"type": "number"},
                    "recommendation": {"type": "string"}
                },
                "required": ["name", "skewness", "recommendation"],
                "additionalProperties": False
            }
        },
        "collinear_pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature_1": {"type": "string"},
                    "feature_2": {"type": "string"},
                    "correlation": {"type": "number"},
                    "recommendation": {"type": "string"}
                },
                "required": ["feature_1", "feature_2", "correlation", "recommendation"],
                "additionalProperties": False
            }
        },
        "distribution_insights": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "distribution_type": {"type": "string"},
                    "notes": {"type": "string"}
                },
                "required": ["feature", "distribution_type", "notes"],
                "additionalProperties": False
            }
        }
    },
    "required": ["skewed_features", "collinear_pairs", "distribution_insights"],
    "additionalProperties": False
}

L2_FEATURES_SCHEMA = {
    "type": "object",
    "properties": {
        "target_column": {"type": "string"},
        "task_type": {"type": "string"},
        "feature_importance": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                    "method": {"type": "string"}
                },
                "required": ["name", "score", "method"],
                "additionalProperties": False
            }
        },
        "selected_features": {
            "type": "array",
            "items": {"type": "string"}
        },
        "transformations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "feature": {"type": "string"},
                    "transform": {"type": "string"},
                    "reason": {"type": "string"},
                    "improvement": {"type": "number"}
                },
                "required": ["feature", "transform", "reason", "improvement"],
                "additionalProperties": False
            }
        },
        "dropped_features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["name", "reason"],
                "additionalProperties": False
            }
        }
    },
    "required": ["target_column", "task_type", "feature_importance", "selected_features", "transformations", "dropped_features"],
    "additionalProperties": False
}

L3_STRATEGY_SCHEMA = {
    "type": "object",
    "properties": {
        "recommended_models": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "priority": {"type": "integer"},
                    "reasons": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                },
                "required": ["name", "priority", "reasons"],
                "additionalProperties": False
            }
        },
        "data_characteristics": {
            "type": "object",
            "properties": {
                "n_samples": {"type": "integer"},
                "n_features": {"type": "integer"},
                "feature_types": {"type": "string"},
                "class_balance": {"type": "string"},
                "data_scale": {"type": "string"}
            },
            "required": ["n_samples", "n_features", "feature_types", "class_balance", "data_scale"],
            "additionalProperties": False
        },
        "preprocessing_pipeline": {
            "type": "array",
            "items": {"type": "string"}
        },
        "validation_strategy": {"type": "string"},
        "special_considerations": {
            "type": "array",
            "items": {"type": "string"}
        }
    },
    "required": ["recommended_models", "data_characteristics", "preprocessing_pipeline", "validation_strategy", "special_considerations"],
    "additionalProperties": False
}

# ──────────────────────────────────────────────────────────────
# New schemas: hypothesis-verify pattern + AutoML
# ──────────────────────────────────────────────────────────────

HYPOTHESIS_BATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "statement": {"type": "string"},
                    "rationale": {"type": "string"},
                    "verification_approach": {"type": "string"}
                },
                "required": ["id", "statement", "rationale", "verification_approach"],
                "additionalProperties": False
            }
        }
    },
    "required": ["hypotheses"],
    "additionalProperties": False
}

HYPOTHESIS_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdicts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "confirmed": {"type": "boolean"},
                    "evidence_summary": {"type": "string"},
                    "action": {"type": "string"}
                },
                "required": ["id", "confirmed", "evidence_summary", "action"],
                "additionalProperties": False
            }
        }
    },
    "required": ["verdicts"],
    "additionalProperties": False
}

AUTOML_HYPOTHESES_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "model_name": {"type": "string"},
                    "rationale": {"type": "string"},
                    "hyperparameter_space": {"type": "string"},
                    "strengths": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "weaknesses": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "preprocessing": {
                        "type": "array",
                        "items": {"type": "string"}
                    },
                    "priority": {"type": "integer"}
                },
                "required": ["model_name", "rationale", "hyperparameter_space", "strengths", "weaknesses", "preprocessing", "priority"],
                "additionalProperties": False
            }
        },
        "validation_strategy": {"type": "string"},
        "ensemble_recommendation": {"type": "string"}
    },
    "required": ["hypotheses", "validation_strategy", "ensemble_recommendation"],
    "additionalProperties": False
}

# ──────────────────────────────────────────────────────────────
# System Prompts
# ──────────────────────────────────────────────────────────────

L0_SYSTEM_PROMPT = """You are an ultra-conservative data quality analyst.

You may ONLY perform the following actions:
1. Drop columns that are 100% empty (every value is NaN/null)
2. Drop constant columns (only one unique value)
3. Fix encoding / garbled text in column values
4. Fix clearly wrong dtypes (e.g. a numeric column stored as string when every non-null value is a valid number)

You must NOT:
- Fill missing values (no fillna, no imputation of any kind)
- Drop rows (no dropna with subset, no row filtering)
- Create new features or derived columns
- Impute anything
- Modify values beyond encoding fixes

Be factual. Report exactly what you observe. Every cleaning action you take must print an ACTION: line."""

L1_SYSTEM_PROMPT = """You are an expert statistician and domain analyst performing task-blind data exploration.

You do NOT know the prediction target or task type. Your goal is to:
1. Understand the structure and distributions of the data
2. Infer the likely domain from column names, data types, and sample values
3. Propose testable hypotheses about data patterns and domain relationships
4. Verify each hypothesis with executable code and evidence

You CAN infer domain context from column names and data samples (e.g. if columns include "age", "income", "credit_score", you may hypothesize this is financial/lending data and propose domain-specific hypotheses).

You must NOT modify the working DataFrame — all analysis is read-only on a copy."""

L2_SYSTEM_PROMPT = """You are an ML engineer with domain expertise performing task-aligned feature analysis.

You now have access to the task description, target variable, and task type. Your goal is to:
1. Analyze feature-target relationships using quantitative methods
2. Propose domain-informed priors (hypotheses about which features matter and why)
3. Verify each prior against the actual data
4. Apply confirmed priors as transformations, feature selection, or engineering

CRITICAL: You MUST use exact column names from the dataset as they appear in the statistics.
Do NOT invent, rename, or conceptualize feature names. Every feature reference must match
an actual column in the DataFrame."""

L3_SYSTEM_PROMPT = """You are an AutoML strategist.

Based on the data characteristics, confirmed hypotheses, and feature analysis, your goal is to:
1. Propose 3-5 modeling hypotheses — each a specific model with rationale
2. For each hypothesis, specify hyperparameter search space, strengths, weaknesses, and preprocessing needs
3. Recommend a validation strategy appropriate for the data
4. Consider ensemble approaches if multiple models show complementary strengths

Be specific and actionable. Each model hypothesis should be immediately implementable."""


# ──────────────────────────────────────────────────────────────
# User prompt builders
# ──────────────────────────────────────────────────────────────

def get_l0_user_prompt(df_info: str, task_description: str = "") -> str:
    """Generate L0 user prompt with dataframe info and optional task context."""
    prompt = f"""Analyze the following dataset and provide a quality assessment.

Dataset Information:
{df_info}
"""
    if task_description:
        prompt += f"""
Task context (for awareness only — do NOT make task-specific cleaning decisions):
{task_description}
"""
    prompt += """
Provide your analysis in the required JSON format. Focus on data quality issues only.
Remember: you may ONLY drop 100% empty columns, drop constant columns, fix encoding, fix wrong dtypes.
Do NOT fill missing values, drop rows, or impute anything."""
    return prompt


def get_l1_user_prompt(l0_stats: Dict[str, Any], numeric_stats: str) -> str:
    """Generate L1 user prompt with L0 context for hypothesis-driven exploration."""
    column_info = ""
    for col in l0_stats.get("columns", []):
        column_info += (
            f"  - {col['name']}: {col['dtype']}, "
            f"missing={col.get('missing_count', 0)}, "
            f"unique={col.get('unique_count', 0)}, "
            f"samples={col.get('sample_values', [])}\n"
        )

    return f"""Perform task-blind exploration of this dataset.

Previous L0 Statistics Summary:
- Total rows: {l0_stats.get('total_rows', 'N/A')}
- Total columns: {l0_stats.get('total_columns', 'N/A')}
- Columns cleaned: {len(l0_stats.get('cleaning_recommendations', []))}

Column Details:
{column_info}

Numeric Feature Statistics:
{numeric_stats}

Based on column names, data types, and sample values, infer the likely domain and propose
up to 5 testable hypotheses about data structure and domain patterns. Each hypothesis should
be verifiable with a specific code check."""


def get_l2_user_prompt(
    l0_stats: Dict[str, Any],
    l1_insights: Dict[str, Any],
    task_config: Dict[str, Any],
    feature_stats: str,
    l1_hypotheses: List[Dict[str, Any]] = None,
) -> str:
    """Generate L2 user prompt with task context and L1 hypotheses."""
    column_names = [c["name"] for c in l0_stats.get("columns", [])]
    column_list_str = ", ".join(f'"{name}"' for name in column_names)

    # Summarize confirmed L1 hypotheses
    confirmed_summary = ""
    if l1_hypotheses:
        confirmed = [h for h in l1_hypotheses if h.get("confirmed")]
        if confirmed:
            confirmed_summary = "Confirmed L1 Hypotheses:\n"
            for h in confirmed:
                confirmed_summary += f"  - {h.get('statement', '')}: {h.get('evidence', '')}\n"

    return f"""Analyze features in relation to the prediction task.

Task Configuration:
- Target column: {task_config.get('target', 'N/A')}
- Task type: {task_config.get('task_type', 'auto-detect')}
- Metric: {task_config.get('metric', 'N/A')}
- Business context: {task_config.get('description', 'N/A')}

Previous Insights:
- L0: {l0_stats.get('total_columns', 0)} columns, {len(l0_stats.get('cleaning_recommendations', []))} cleaning actions
- L1: {len(l1_insights.get('skewed_features', []))} skewed features, {len(l1_insights.get('collinear_pairs', []))} collinear pairs

{confirmed_summary}

EXACT COLUMN NAMES IN DATASET: [{column_list_str}]
You MUST only use names from this list. Do NOT invent feature names.

Feature-Target Statistics:
{feature_stats}

Propose up to 5 domain-informed priors about which features should matter and why,
then verify each against the data. Apply confirmed priors as transformations or selections."""


def get_l3_user_prompt(
    l0_stats: Dict[str, Any],
    l1_insights: Dict[str, Any],
    l2_features: Dict[str, Any],
    task_config: Dict[str, Any],
    l1_hypotheses: List[Dict[str, Any]] = None,
    l2_domain_priors: List[Dict[str, Any]] = None,
    characteristics_stdout: str = "",
) -> str:
    """Generate L3 user prompt for AutoML strategy."""
    confirmed_l1 = []
    if l1_hypotheses:
        confirmed_l1 = [h for h in l1_hypotheses if h.get("confirmed")]

    confirmed_l2 = []
    if l2_domain_priors:
        confirmed_l2 = [p for p in l2_domain_priors if p.get("confirmed")]

    return f"""Provide AutoML modeling hypotheses for this dataset.

Task:
- Target: {task_config.get('target', 'N/A')}
- Type: {task_config.get('task_type', 'N/A')}
- Metric: {task_config.get('metric', 'N/A')}
- Description: {task_config.get('description', 'N/A')}

Data Characteristics:
- Samples: {l0_stats.get('total_rows', 'N/A')}
- Original features: {l0_stats.get('total_columns', 'N/A')}
- Selected features: {len(l2_features.get('selected_features', []))}
- Skewed features requiring transform: {len(l1_insights.get('skewed_features', []))}

Selected Features: {', '.join(l2_features.get('selected_features', [])[:10])}

Confirmed L1 findings: {len(confirmed_l1)} hypotheses
Confirmed L2 priors: {len(confirmed_l2)} domain priors

{f"Bootstrap Characteristics:{chr(10)}{characteristics_stdout}" if characteristics_stdout else ""}

Propose 3-5 model hypotheses, each with rationale, hyperparameter search space,
strengths, weaknesses, and preprocessing requirements. Include a validation strategy
and ensemble recommendation."""
