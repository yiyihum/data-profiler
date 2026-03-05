"""LLM-Driven Exploration — Adaptive analysis code generation for all layers."""

from typing import Any, Dict, Optional

from core.sandbox import CodeSandbox
from core.llm import LLMClient


# Per-layer exploration directives that tell the LLM what to compute.
_LAYER_DIRECTIVES = {
    "L0": (
        "Compute the following for data quality assessment:\n"
        "1. Missing rates per column (count and percentage)\n"
        "2. Unique value counts per column\n"
        "3. Constant column check (flag columns with only 1 unique value)\n"
        "4. Numeric summary statistics (min, max, mean, std, median) for numeric columns\n"
        "5. Detect nested structures: if any column contains lists, dicts, arrays, or JSON strings,\n"
        "   report the structure (e.g., average list length, key set for dicts)\n"
        "6. IQR-based outlier detection for numeric columns\n"
        "7. Dtype consistency check (e.g., numeric values stored as strings)\n"
    ),
    "L1": (
        "Compute the following for unsupervised data exploration:\n"
        "1. Distribution analysis: skewness and kurtosis for numeric columns\n"
        "2. Correlation matrix for numeric columns (report pairs with |r| > 0.8)\n"
        "3. Cardinality analysis for categorical columns (unique counts, top values)\n"
        "4. Pattern detection: for text/string columns, check for common patterns\n"
        "   (e.g., regex patterns, date formats, URL patterns, encoded data)\n"
        "5. For nested/array columns: parse a sample and report structure\n"
        "   (e.g., list lengths, element types, value ranges)\n"
        "6. Low-cardinality numeric columns that might be categorical\n"
        "7. Duplicate row detection\n"
    ),
    "L2": (
        "Compute the following for task-aligned feature analysis:\n"
        "1. Mutual information scores between each numeric feature and target\n"
        "2. Chi-squared test for categorical features vs target\n"
        "3. Correlation between numeric features and target\n"
        "4. For nested/array columns: parse them and compute correlation with target\n"
        "   (e.g., extract array length, mean, std as features; compute MI)\n"
        "5. Feature-target distribution overlap (for classification: per-class feature stats)\n"
        "6. Interaction effects between top features\n"
    ),
    "L3": (
        "Compute the following for model strategy assessment:\n"
        "1. Data dimensions: samples, features, feature-to-sample ratio\n"
        "2. Feature type breakdown: numeric vs categorical vs nested/complex count\n"
        "3. Class distribution (for classification) or target distribution (for regression)\n"
        "4. Data scale assessment (small/medium/large)\n"
        "5. Missing data pattern: is missingness random or structured?\n"
        "6. Nested column handling needs: how many columns need special parsing?\n"
        "7. Cardinality summary: high-cardinality categoricals that need encoding\n"
    ),
}


def run_exploration(
    sandbox: CodeSandbox,
    llm: LLMClient,
    layer: str,
    discovery_stdout: str,
    task_config: Optional[Dict[str, Any]] = None,
    prior_state: str = "",
) -> Dict[str, Any]:
    """
    LLM generates and executes adaptive exploration code.

    The LLM reads the discovery output and writes analysis code appropriate
    for the actual data format (handles nested structures, JSON, arrays, etc.).

    Args:
        sandbox: CodeSandbox instance.
        llm: LLMClient for code generation.
        layer: Layer identifier ("L0", "L1", "L2", "L3").
        discovery_stdout: Output from run_discovery().
        task_config: Task configuration (target, task_type, etc.).
        prior_state: Summary of prior layer findings.

    Returns:
        Dict with stdout, success, _code, and _llm_result keys.
    """
    directive = _LAYER_DIRECTIVES.get(layer, _LAYER_DIRECTIVES["L0"])

    target = ""
    task_type = ""
    if task_config:
        target = task_config.get("target", "") or ""
        task_type = task_config.get("task_type", "") or ""

    target_context = ""
    if target and layer in ("L2", "L3"):
        target_context = f"\nTarget column: '{target}' (task type: {task_type})\n"

    prior_context = ""
    if prior_state:
        prior_context = f"\nPrior layer findings:\n{prior_state[:1500]}\n"

    # Determine available variables
    available_vars = ["df", "pd", "np", "OUTPUT_DIR"]
    if sandbox.get_variable("DATA_DIR") is not None:
        available_vars.extend([
            "DATA_DIR", "torch", "Image", "Path",
            "analyze_image", "analyze_audio", "list_data_files",
            "batch_analyze_images", "batch_analyze_audio",
        ])

    task = (
        f"Based on the data discovery output below, generate Python code to perform "
        f"the {layer} exploration analysis.\n\n"
        f"=== Data Discovery Output ===\n{discovery_stdout[:3000]}\n\n"
        f"{target_context}"
        f"{prior_context}"
        f"=== What to Compute ===\n{directive}\n"
        f"IMPORTANT ADAPTATION RULES:\n"
        f"- If the discovery shows nested/complex columns (lists, dicts, arrays, JSON strings),\n"
        f"  you MUST handle them: parse/flatten before computing statistics.\n"
        f"  Use ast.literal_eval() or json.loads() to parse string representations.\n"
        f"- If columns contain Python lists stored as strings (e.g., '[1.2, 3.4, ...]'),\n"
        f"  parse them and extract meaningful features (length, mean, std, etc.).\n"
        f"- Skip operations that don't apply to the actual data format.\n"
        f"- Handle errors gracefully: wrap risky operations in try/except.\n"
        f"- Print clear labeled output for each analysis section.\n"
        f"- Do NOT modify df. Work on copies if needed. Do NOT generate plots.\n"
    )

    context = f"Discovery output:\n{discovery_stdout[:2000]}"
    code_result = llm.generate_code(task, context, available_vars)

    if not code_result["success"]:
        return {
            "success": False,
            "stdout": "",
            "error": code_result.get("error", "LLM failed to generate exploration code"),
            "_code": "",
            "_llm_result": code_result,
        }

    exploration_code = code_result["content"]
    exec_result = sandbox.execute(exploration_code, f"Exploration ({layer})")

    return {
        "success": exec_result["success"],
        "stdout": exec_result.get("stdout", ""),
        "error": exec_result.get("error"),
        "_code": exploration_code,
        "_llm_result": code_result,
    }
