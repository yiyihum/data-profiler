"""State Context - Global state manager for the profiling pipeline."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path
import json


@dataclass
class StateContext:
    """
    Maintains global state across L0-L3 layers.
    Stores insights and artifacts from each layer to avoid context overflow.
    """

    # Current execution layer
    current_layer: str = "L0"

    # L0: Basic statistics and cleaning
    l0_stats: Dict[str, Any] = field(default_factory=dict)
    l0_cleaning_actions: List[Dict[str, str]] = field(default_factory=list)

    # L1: Unsupervised exploration insights
    l1_insights: Dict[str, Any] = field(default_factory=dict)
    l1_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    # L2: Task-aligned feature selection
    l2_selected_features: List[str] = field(default_factory=list)
    l2_transformations: List[Dict[str, Any]] = field(default_factory=list)
    l2_domain_priors: List[Dict[str, Any]] = field(default_factory=list)

    # L3: AutoML strategy
    l3_strategy: Dict[str, Any] = field(default_factory=dict)
    l3_hypotheses: List[Dict[str, Any]] = field(default_factory=list)

    # Data modality
    data_modality: str = "tabular"  # "tabular" | "image" | "audio" | "mixed"

    # Per-layer LLM-generated report sections
    layer_reports: Dict[str, str] = field(default_factory=dict)

    # Execution metadata
    errors: List[Dict[str, str]] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)  # name -> path mapping
    layer_runs: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    layer_status: Dict[str, str] = field(default_factory=dict)
    layer_status_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    agent_decisions: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    def advance_layer(self, next_layer: str) -> None:
        """Advance to the next layer."""
        valid_layers = ["L0", "L1", "L2", "L3", "REPORT"]
        if next_layer not in valid_layers:
            raise ValueError(f"Invalid layer: {next_layer}")
        self.current_layer = next_layer

    def record_error(self, layer: str, error_type: str, message: str) -> None:
        """Record an error that occurred during execution."""
        self.errors.append({
            "layer": layer,
            "type": error_type,
            "message": message
        })

    def add_artifact(self, name: str, path: str) -> None:
        """Register an artifact (plot, file) generated during execution."""
        self.artifacts[name] = path

    def record_layer_turn(self, layer: str, turn_record: Dict[str, Any]) -> None:
        """Append one turn record for a layer agent loop."""
        self.layer_runs.setdefault(layer, []).append(turn_record)

    def set_layer_status(
        self,
        layer: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Set a normalized status for a layer."""
        self.layer_status[layer] = status
        if details:
            self.layer_status_details[layer] = details

    def record_agent_decision(self, layer: str, decision: Dict[str, Any]) -> None:
        """Record the plan/decision for a given turn."""
        self.agent_decisions.setdefault(layer, []).append(decision)

    def set_layer_report(self, layer: str, markdown: str) -> None:
        """Store the LLM-generated report section for a layer."""
        self.layer_reports[layer] = markdown

    def get_summary_for_layer(self, target_layer: str) -> Dict[str, Any]:
        """
        Get a summary of state relevant for a specific layer.
        This prevents context overflow by only including necessary information.
        """
        summary = {"current_layer": self.current_layer}

        if target_layer in ["L1", "L2", "L3", "REPORT"]:
            summary["l0_stats"] = self.l0_stats
            summary["l0_cleaning_actions"] = self.l0_cleaning_actions

        if target_layer in ["L2", "L3", "REPORT"]:
            summary["l1_insights"] = self.l1_insights
            summary["l1_hypotheses"] = self.l1_hypotheses

        if target_layer in ["L3", "REPORT"]:
            summary["l2_selected_features"] = self.l2_selected_features
            summary["l2_transformations"] = self.l2_transformations
            summary["l2_domain_priors"] = self.l2_domain_priors

        if target_layer == "REPORT":
            summary["l3_strategy"] = self.l3_strategy
            summary["l3_hypotheses"] = self.l3_hypotheses
            summary["artifacts"] = self.artifacts
            summary["errors"] = self.errors
            summary["layer_status"] = self.layer_status
            summary["layer_status_details"] = self.layer_status_details
            summary["layer_runs"] = self.layer_runs
            summary["agent_decisions"] = self.agent_decisions
            summary["layer_reports"] = self.layer_reports

        return summary

    def get_agent_trace(self) -> Dict[str, Any]:
        """Return full multi-turn trace payload."""
        return {
            "layer_status": self.layer_status,
            "layer_status_details": self.layer_status_details,
            "layer_runs": self.layer_runs,
            "agent_decisions": self.agent_decisions,
            "errors": self.errors,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Export full state as dictionary."""
        return {
            "current_layer": self.current_layer,
            "data_modality": self.data_modality,
            "l0_stats": self.l0_stats,
            "l0_cleaning_actions": self.l0_cleaning_actions,
            "l1_insights": self.l1_insights,
            "l1_hypotheses": self.l1_hypotheses,
            "l2_selected_features": self.l2_selected_features,
            "l2_transformations": self.l2_transformations,
            "l2_domain_priors": self.l2_domain_priors,
            "l3_strategy": self.l3_strategy,
            "l3_hypotheses": self.l3_hypotheses,
            "layer_reports": self.layer_reports,
            "errors": self.errors,
            "artifacts": self.artifacts,
            "layer_runs": self.layer_runs,
            "layer_status": self.layer_status,
            "layer_status_details": self.layer_status_details,
            "agent_decisions": self.agent_decisions,
        }

    def save(self, path: Path) -> None:
        """Save state to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def save_agent_trace(self, path: Path) -> None:
        """Save multi-turn trace to JSON file."""
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.get_agent_trace(), f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "StateContext":
        """Load state from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        state = cls()
        state.current_layer = data.get("current_layer", "L0")
        state.data_modality = data.get("data_modality", "tabular")
        state.l0_stats = data.get("l0_stats", {})
        state.l0_cleaning_actions = data.get("l0_cleaning_actions", [])
        state.l1_insights = data.get("l1_insights", {})
        state.l1_hypotheses = data.get("l1_hypotheses", [])
        state.l2_selected_features = data.get("l2_selected_features", [])
        state.l2_transformations = data.get("l2_transformations", [])
        state.l2_domain_priors = data.get("l2_domain_priors", [])
        state.l3_strategy = data.get("l3_strategy", {})
        state.l3_hypotheses = data.get("l3_hypotheses", [])
        state.layer_reports = data.get("layer_reports", {})
        state.errors = data.get("errors", [])
        state.artifacts = data.get("artifacts", {})
        state.layer_runs = data.get("layer_runs", {})
        state.layer_status = data.get("layer_status", {})
        state.layer_status_details = data.get("layer_status_details", {})
        state.agent_decisions = data.get("agent_decisions", {})
        return state
