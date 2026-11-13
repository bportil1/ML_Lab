"""ML-UW-16 section navigation layered over existing stable ML_Lab routes.

No source import, algorithm execution or dataset mutations occur in navigation.
Register via :func:`ml_lab.ui.create_ui_blueprint`, not the legacy blueprint
module; keeping this separate avoids touching its frequently modified routes.
"""
from __future__ import annotations

from flask import Blueprint, render_template, request, url_for
from pathlib import Path

from ml_lab import __version__
from .workspace import active_tool, tool_records


# Five primary Tools entries. The Data subnavigation appears *beneath* the
# primary list, only while the active page belongs to the Data family.
SECTION_SPEC = (
    ("data", "Data", "data_lab"),
    ("profiling", "Feature Profiling", "workbench_feature_profiling"),
    ("representation", "Representation", "workbench_representation"),
    ("comparison", "Comparison", "workbench_comparison"),
    ("tools", "General ML Tools", "workbench_general_ml_tools"),
)

DATA_SUBSECTIONS = (
    ("tabular", "Tabular Data", "data_lab"),
    ("xml", "XML Data", "xml_workbench"),
    ("postcollection", "Post-Collection Utilities", "workbench_postcollection"),
)


# All links point to already-existing, working routes or the new read-only
# landing pages. A planned capability never receives an executable link.
SECTION_LINKS = {
    "profiling": (
        ("Feature Profiling", "workbench_feature_profiling", {}),
        ("Run tabular profiles", "data_lab", {}),
    ),
    "representation": (
        ("Representation", "workbench_representation", {}),
        ("Trainable autoencoders", "workbench_autoencoder", {}),
        ("PCA / autoencoder task", "task", {"capability_id": "representation"}),
    ),
    "comparison": (
        ("Comparison", "workbench_comparison", {}),
        ("Dataset comparison", "data_compare", {}),
        ("Model workspace", "experiment_model", {}),
    ),
    "tools": (
        ("General ML Tools", "workbench_general_ml_tools", {}),
        ("Models", "workbench_general_ml_tools", {"_anchor": "models"}),
        ("Energy-based models", "workbench_general_ml_tools", {"_anchor": "ebm"}),
        ("Optimization", "workbench_general_ml_tools", {"_anchor": "optimization"}),
        ("Generative", "workbench_general_ml_tools", {"_anchor": "generative"}),
    ),
}


# Preserve the existing P5-13 tool entrypoints and browser navigation context,
# but show them within their corresponding primary workflow sections.
SECTION_TOOL_IDS = {
    "profiling": frozenset(),
    "representation": frozenset({"representation"}),
    "comparison": frozenset({"results"}),
    "tools": frozenset({"models", "ebm", "optimization", "generative"}),
}


def section_for_request(endpoint: str | None, view_args: dict | None, family: str | None = None) -> str | None:
    """Identify active section using route identity, not mount-prefix strings."""
    name = (endpoint or "").rsplit(".", 1)[-1]
    if name in {"index", "workspace", "workspace_context"}:
        # No workflow selected on the landing page: Data children stay hidden
        # until the user enters Data or one of its existing workflow routes.
        return None
    if name in {"workbench_feature_profiling", "workbench_feature_profiling_json",
                "workbench_feature_role", "profile_view", "profile_job_view",
                "profile_json", "profile_job_status"}:
        return "profiling"
    if name in {"workbench_representation", "workbench_autoencoder"}:
        return "representation"
    if name == "experiment_model" and family == "representation":
        return "representation"
    if name in {"workbench_comparison", "workspace_results", "experiment_compare",
                "experiment_compare_csv", "experiment_visualizations_json",
                "experiment_chart_svg", "experiment_chart_csv",
                "experiment_model_run_view", "experiment_model_run_result"}:
        return "comparison"
    if name == "task":
        capability = (view_args or {}).get("capability_id")
        if capability == "representation":
            return "representation"
        if capability == "data":
            return "data"
        return "tools"
    if name.startswith("xml_"):
        return "data"
    if name in {"data_transform", "data_provenance", "data_compare", "workbench_postcollection",
                "workbench_select_file", "workbench_activate_experiment", "experiment_dataset"}:
        return "data"
    if name == "workbench_general_ml_tools" or name.startswith("experiment_"):
        return "tools"
    return "data"


def data_subsection_for_request(endpoint: str | None) -> str:
    """Choose the Data child link for legacy and new routes without URL guessing."""
    name = (endpoint or "").rsplit(".", 1)[-1]
    if name.startswith("xml_"):
        return "xml"
    if name in {"workbench_postcollection", "data_transform", "data_provenance",
                "data_compare", "workbench_select_file", "workbench_activate_experiment",
                "experiment_dataset"}:
        return "postcollection"
    return "tabular"


def install_workbench_navigation(blueprint: Blueprint, *, project_root: Path,
                                 enable_experimental: bool = False, workbench_context=None) -> None:
    """Attach read-only landing pages and shared navigation to a mounted UI."""

    @blueprint.context_processor
    def workbench_navigation_context():
        active = section_for_request(request.endpoint, request.view_args, request.args.get("family"))
        sections = [
            {"id": key, "label": label, "url": url_for(request.blueprint + "." + endpoint),
             "active": key == active}
            for key, label, endpoint in SECTION_SPEC
        ]
        # Data has one concise, muted submenu. Individual XML and tabular
        # workflows already provide their own detailed in-page navigation.
        selected_data_area = data_subsection_for_request(request.endpoint) if active == "data" else None
        data_subsections = [
            {"id": key, "label": label,
             "url": url_for(request.blueprint + "." + endpoint),
             "active": key == selected_data_area}
            for key, label, endpoint in DATA_SUBSECTIONS
        ] if active == "data" else []
        links = [
            {"label": label, "url": url_for(request.blueprint + "." + endpoint, **kwargs),
             "current": request.endpoint == request.blueprint + "." + endpoint and
                        (not kwargs or kwargs == (request.view_args or {}))}
            for label, endpoint, kwargs in SECTION_LINKS.get(active, ())
        ]
        current_tool = active_tool((request.endpoint or "").rsplit(".", 1)[-1], request.args.get("family"))
        tools = []
        for tool in tool_records():
            # General ML Tools now has one grouped launcher; the former four
            # individual entries remain reachable from its category cards.
            if active not in SECTION_TOOL_IDS or active == "tools" or tool["id"] not in SECTION_TOOL_IDS[active]:
                continue
            tool_url = url_for(request.blueprint + "." + tool["endpoint"], **tool["query"])
            if tool["anchor"]:
                tool_url += "#" + tool["anchor"]
            tools.append({**tool, "url": tool_url, "current": tool["id"] == current_tool})
        return {"mlwb_navigation": {
            "active": active,
            "active_label": next((label for key, label, _ in SECTION_SPEC if key == active), "Tools"),
            "sections": sections,
            "data_subsections": data_subsections,
            "links": links,
            "tools": tools,
        }}

    def _render(key: str):
        cards = {
            "profiling": [],  # The active-dataset explorer owns the Feature Profiling landing.

            "representation": [
                ("Existing PCA / neural representation task", "Open the existing PCA and trainable neural-representation workflow without changing its execution path.", "task", {"capability_id": "representation"}, "Available"),
                ("Registered sklearn compression methods", "PCA, SVD, ICA, Kernel PCA, Isomap, t-SNE and NMF with explicit transform/inverse capability metadata.", "workbench_representation", {}, "Available"),
                ("Configurable autoencoder experiments", "Configure existing MLP or Transformer autoencoders and launch revision-pinned training runs.", "workbench_autoencoder", {}, "Available"),
            ],
            "comparison": [
                ("Existing dataset comparison", "Inspect schema overlap, row overlap, duplicates and potential relationships. Heuristic similarity is not provenance.", "data_compare", {}, "Available"),
                ("Experiment run comparisons", "Select a model experiment first, then compare its runs and verified artifacts.", "experiment_model", {}, "Available"),
                ("Representation comparisons", "Bounded source-matched PCA, sklearn and verified neural-result comparisons with reconstruction and geometry diagnostics.", "workbench_comparison", {}, "Available"),
                ("Matched held-out benchmarks", "A future benchmark will evaluate methods against one explicit held-out cohort and training protocol; current in-sample metrics are not such a benchmark.", None, {}, "Planned"),
            ],
            "tools": [
                ("Models and optimization", "Use the existing typed model experiment, configuration and optimization workflow.", "experiment_model", {}, "Available"),
                ("Energy-based models", "Open the existing RBM training task.", "task", {"capability_id": "rbm"}, "Available"),
                ("Generative models", "Use the existing GAN task or the synthetic-data experiment runner.", "experiment_synthetic", {}, "Available"),
            ],
        }
        title = dict((key, label) for key, label, _ in SECTION_SPEC)[key]
        summaries = {
            "profiling": "Characterize the selected dataset and inspect its variables before modeling.",
            "representation": "Fit and explore dimensionality reduction and learned representations.",
            "comparison": "Compare verified dataset relationships, experiment runs and source-matched representations.",
            "tools": "Existing model training, energy-based models, optimization and generative tools.",
        }
        return render_template(
            "ml_lab_ui/workbench_section.html", version=__version__,
            section_title=title, section_description=summaries[key],
            cards=[{
                "title": t, "description": d,
                "url": url_for(request.blueprint + "." + endpoint, **params) if endpoint else None,
                "status": status,
            } for t, d, endpoint, params, status in cards[key]],
        )

    @blueprint.get("/workbench/postcollection")
    def workbench_postcollection():
        """Read-only entry point for tools operating after dataset collection."""
        options = (
            ("Transform and clean", "Operate on an existing CSV/TSV dataset, preserving its original source and recipe provenance.", "data_transform"),
            ("Trace provenance", "Inspect the provenance chain of derived or consolidated datasets.", "data_provenance"),
            ("Compare datasets", "Compare existing tabular datasets after ingestion or XML materialization.", "data_compare"),
            ("Experiment dataset setup", "Bind a dataset to an experiment through the existing configuration workflow.", "experiment_dataset"),
        )
        return render_template(
            "ml_lab_ui/workbench_section.html", version=__version__,
            section_title="Post-Collection Utilities",
            section_description="Clean, validate, compare and trace already-collected or XML-extracted tabular datasets without re-running source collection.",
            cards=[{"title": title, "description": description,
                    "url": url_for(request.blueprint + "." + endpoint), "status": "Available"}
                   for title, description, endpoint in options],
        )

    @blueprint.get("/workbench/general-ml-tools")
    def workbench_general_ml_tools():
        from ml_lab.experiments import ExperimentStore, ExperimentStoreError
        from .general_ml_catalog import MAX_CATALOG_EXPERIMENTS, general_ml_catalog
        error = None
        try:
            experiments = ExperimentStore(project_root).list_experiments(limit=MAX_CATALOG_EXPERIMENTS)
            groups = general_ml_catalog(experiments)
        except (ExperimentStoreError, OSError) as exc:
            groups = general_ml_catalog([])
            error = str(exc)
        return render_template(
            "ml_lab_ui/workbench_general_ml_tools.html", version=__version__,
            groups=groups, error=error, catalog_limit=MAX_CATALOG_EXPERIMENTS,
            context=workbench_context() if workbench_context is not None else None,
        ), (503 if error else 200)
