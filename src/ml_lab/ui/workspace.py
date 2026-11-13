"""Navigation-only tools catalog for the optional ML_Lab scientific workspace.

This module has no scientific calculations and no import-time Flask dependency.
The routes are owned by the existing ML_Lab Blueprint; no tool is a mandatory
prerequisite for opening another tool. PAH_Shared owns any generic data grids.
"""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path


# Tool IDs are stable UI preferences, not scientific engine identifiers.
# Tuple: id, title, description, existing Blueprint endpoint, URL arguments,
#       section anchor. Every surface remains available without a selection.
TOOLS = (
    ("data", "Tabular Data", "Inspect CSV/TSV sources and choose tabular datasets.", "data_lab", (), ""),
    ("models", "Models", "Configure and run saved model families.", "experiment_model", (), ""),
    ("representation", "Representation", "Explore representation-learning model families.", "experiment_model", (("family", "representation"),), ""),
    ("ebm", "EBM", "Configure energy-based RBM models.", "experiment_model", (("family", "rbm"),), ""),
    ("optimization", "Optimization", "Run searches on saved model configurations.", "experiment_model", (), "model-optimization"),
    ("generative", "Generative", "Generate synthetic data or configure GAN models.", "experiment_synthetic", (), ""),
    ("xml", "XML Data", "Explore XML sources and collection extraction recipes.", "xml_workbench", (), ""),
    ("results", "Results", "Inspect recorded runs and compare verified results.", "workspace_results", (), ""),
)


def tool_records() -> list[dict[str, object]]:
    return [
        {"id": tool_id, "title": title, "description": description,
         "endpoint": endpoint, "query": dict(query), "anchor": anchor}
        for tool_id, title, description, endpoint, query, anchor in TOOLS
    ]


def project_scope(project_root: str | Path, blueprint_name: str) -> str:
    """Opaque, deterministic browser-preference namespace; no raw path in HTML.

    The scope is convenience-only; it grants no authorization to read artifacts.
    """
    value = str(Path(project_root).resolve()) + "\0" + blueprint_name
    return sha256(value.encode("utf-8")).hexdigest()[:24]


def active_tool(endpoint: str | None, family: str | None = None) -> str:
    """State-independent fallback for navigation and no-JavaScript clients."""
    if endpoint == "experiment_model":
        if family == "representation":
            return "representation"
        if family == "rbm":
            return "ebm"
        return "models"
    if endpoint in {"experiment_synthetic", "experiment_synthetic_run"}:
        return "generative"
    if endpoint in {"experiment_compare", "workspace_results", "experiment_model_run_view"}:
        return "results"
    if endpoint == "xml_collection_inventory" or (endpoint or "").startswith("xml_"):
        return "xml"
    if endpoint in {"data_lab", "data_transform", "data_compare", "data_provenance", "experiment_dataset", "source_view", "matrix_editor"}:
        return "data"
    return ""
