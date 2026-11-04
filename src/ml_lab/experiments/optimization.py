"""S4.2: opt-in bounded Firefly trials for configured, immutable ML_Lab models.

Trial runs use the existing scientific dispatcher and existing Firefly optimizer.
Only scalar numeric model settings shown by the typed registry are searchable;
models and datasets are never rewritten by tuning. Parallel candidate workers,
checkpoint tuning, and automatic best-model promotion are deliberately absent.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .models import (MODEL_SCHEMA, _restore_explicit_settings,
                     describe_model_method, prepare_configured_model_handoff,
                     validate_model_configuration)
from .run_control import (RunCancelled, cancellation_requested, check_cancelled,
                          write_owner)
from .store import ExperimentStore, ExperimentStoreError
from .training import (_run_folder, _write_new, _stage,
                       ConfiguredRunError)

OPTIMIZER_EXECUTOR = "ml_lab.experiments.optimization@1"
OPTIMIZER_SCHEMA = "ml-lab.experiment-optimization@1"
# The result envelope is heterogeneous; only these explicitly documented
# numerical metrics are offered as experiment objectives in this sprint.
_ALLOWED_METRICS = {
    "representation": frozenset({"mse", "rmse", "mae"}),
    "classification": frozenset({"accuracy", "balanced_accuracy", "f1_macro", "roc_auc"}),
    "regression": frozenset({"mse", "rmse", "mae", "r2", "explained_variance"}),
    "clustering": frozenset({"silhouette", "calinski_harabasz", "davies_bouldin"}),
}


def _numeric_control(family: str, method: str, path: str) -> str:
    if not isinstance(path, str) or len(path) > 120 or not path or any(not p.isidentifier() for p in path.split(".")):
        raise ExperimentStoreError("Parameter must be a dotted, named numeric model control")
    parts = path.split(".")
    controls = describe_model_method(family, method)["controls"]
    if parts[0] not in controls or len(parts) < 2:
        raise ExperimentStoreError("Choose a model configuration control, not an input dataset field")
    fields = controls[parts[0]]
    found = None
    for segment in parts[1:]:
        found = next((x for x in fields if x["name"] == segment), None)
        if found is None:
            raise ExperimentStoreError(f"Unknown model control: {path}")
        fields = found.get("fields", [])
    descriptor = found["type"]
    if descriptor == "union" and set(x["type"] for x in found.get("variants", [])) <= {"integer", "number"}:
        descriptor = "numeric_union"
    if descriptor not in {"integer", "number", "numeric_union"} or parts[-1] in {"seed", "random_state"}:
        raise ExperimentStoreError("Only explicit numeric hyperparameters may be optimized")
    if "checkpoint" in path or "output" in path or "path" in path:
        raise ExperimentStoreError("File paths and checkpoints cannot be optimization controls")
    return descriptor


def validate_optimization_plan(experiment: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and freeze one small, serial optimization specification."""
    spec = experiment["model_spec"]
    family, method = spec.get("family"), spec.get("method")
    if spec.get("schema") != MODEL_SCHEMA or family not in _ALLOWED_METRICS:
        raise ExperimentStoreError("Optimization currently supports configured representation, classification, regression and clustering models")
    if method == "suite":
        raise ExperimentStoreError("Optimize a single estimator, not a multiple-estimator suite")
    if not isinstance(plan, Mapping) or set(plan) - {"schema", "algorithm", "kind", "parameter", "lower", "upper", "metric", "goal", "population_size", "iterations", "seed"}:
        raise ExperimentStoreError("Unknown optimization settings")
    if "schema" in plan and plan["schema"] != OPTIMIZER_SCHEMA:
        raise ExperimentStoreError("Unsupported optimizer plan schema")
    if "algorithm" in plan and plan["algorithm"] != "firefly":
        raise ExperimentStoreError("Unsupported optimizer algorithm")
    path = plan.get("parameter", "")
    kind = _numeric_control(family, method, path)
    try:
        lower, upper = float(plan["lower"]), float(plan["upper"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ExperimentStoreError("Numeric lower and upper parameter bounds are required") from exc
    if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
        raise ExperimentStoreError("Parameter bounds must be finite and ordered")
    if kind == "numeric_union":
        # Mixed int|float settings (notably PCA n_components): integer counts
        # for integral bounds >=1, fractional thresholds only within (0,1].
        kind = "integer" if lower >= 1 and lower.is_integer() and upper.is_integer() else "number"
        if kind == "number" and not (0 < lower <= upper <= 1):
            raise ExperimentStoreError("Fractional union bounds must be in (0,1]")
    if kind == "integer" and (lower != int(lower) or upper != int(upper)):
        raise ExperimentStoreError("Integer hyperparameters require integral bounds")
    if kind == "integer" and (abs(lower) > 1_000_000 or abs(upper) > 1_000_000):
        raise ExperimentStoreError("Integer hyperparameter bounds are too large")
    metric = plan.get("metric", "")
    if metric not in _ALLOWED_METRICS[family]:
        raise ExperimentStoreError(f"Unsupported {family} objective metric: {metric}")
    goal = plan.get("goal", "minimize")
    if goal not in {"minimize", "maximize"}:
        raise ExperimentStoreError("Metric goal must be minimize or maximize")
    population, iterations, seed = (plan.get("population_size", 2), plan.get("iterations", 1), plan.get("seed", 42))
    if type(population) is not int or not 2 <= population <= 8:
        raise ExperimentStoreError("Firefly population_size must be in [2,8]")
    if type(iterations) is not int or not 0 <= iterations <= 5 or population * (iterations + 1) > 32:
        raise ExperimentStoreError("At most 32 candidate evaluations are allowed per optimization run")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ExperimentStoreError("Optimizer seed must be an unsigned 32-bit integer")
    # The model validator remains authoritative, including nested settings.
    return {"schema": OPTIMIZER_SCHEMA, "algorithm": "firefly", "parameter": path,
            "kind": kind, "lower": int(lower) if kind == "integer" else lower,
            "upper": int(upper) if kind == "integer" else upper,
            "metric": metric, "goal": goal, "population_size": population,
            "iterations": iterations, "seed": seed}


def _override_settings(experiment: Mapping[str, Any], plan: Mapping[str, Any], value: Any) -> dict:
    spec = experiment["model_spec"]
    family, method = spec["family"], spec["method"]
    options = deepcopy(_restore_explicit_settings(family, method, spec["settings"]))
    parts = plan["parameter"].split(".")
    if parts[0] == "task_options":
        parts = parts[1:]
    node = options
    for field in parts[:-1]:
        node = node.setdefault(field, {})
    node[parts[-1]] = value
    return validate_model_configuration(
        family, method, options,
        n_features=len(experiment["dataset_ref"]["feature_columns"]),
    )


def _metric(response: Mapping[str, Any], family: str, metric: str) -> float:
    if family == "representation":
        results = [response.get("result", {})]
    else:
        results = response.get("results", [])
    if not isinstance(results, list) or len(results) != 1 or not isinstance(results[0], Mapping):
        raise ExperimentStoreError("Expected one model result for optimization")
    measurements = results[0].get("metrics")
    if not isinstance(measurements, Mapping):
        raise ExperimentStoreError("Model result contains no objective metrics")
    value = measurements.get(metric)
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ExperimentStoreError(f"Model result has no finite {metric} metric")
    return float(value)


def queue_optimization_run(store: ExperimentStore, experiment_id: str, plan: Mapping[str, Any],
                           *, expected_revision: int) -> dict[str, Any]:
    if type(expected_revision) is not int or expected_revision < 1:
        raise ExperimentStoreError("Select a saved experiment revision")
    experiment = store.get_experiment(experiment_id)
    if experiment["revision"] != expected_revision:
        raise ExperimentStoreError("Experiment revision changed; reload before optimizing")
    from .datasets import verify_selection
    verify_selection(store, experiment["dataset_ref"])
    frozen = validate_optimization_plan(experiment, plan)
    # Catch structurally invalid bounds before recording any run; models may
    # still reject candidate-specific domain constraints at evaluation time.
    _override_settings(experiment, frozen, frozen["lower"])
    _override_settings(experiment, frozen, frozen["upper"])
    return store.create_run(
        experiment_id, revision=expected_revision, expected_current_revision=expected_revision,
        seed=frozen["seed"], environment={"executor": OPTIMIZER_EXECUTOR, "plan": frozen,
                                          "runtime": "serial", "seed_note": "Firefly population seeded; model engines retain their own RNG policies"},
    )


def execute_optimization_run(store: ExperimentStore, run_id: str) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run["environment"].get("executor") != OPTIMIZER_EXECUTOR:
        raise ExperimentStoreError("Run was not queued for experiment optimization")
    if run["status"] != "queued":
        raise ExperimentStoreError("Only queued optimization runs can be executed")
    experiment = store.get_experiment(run["experiment_id"], revision=run["revision"])
    plan = validate_optimization_plan(experiment, run["environment"]["plan"])
    store.update_run_status(run_id, "running")  # atomic claim
    try:
        folder = _run_folder(store, run_id, create=True)
        write_owner(store, run_id)
        check_cancelled(store, run_id)
        _stage(store, run_id, "Preparing optimization dataset", 1)
        handoff = prepare_configured_model_handoff(store, run["experiment_id"], revision=run["revision"])
        manifest = {"schema": OPTIMIZER_SCHEMA, "run_id": run_id, "revision": run["revision"],
                    "experiment_id": run["experiment_id"], "dataset_ref": experiment["dataset_ref"],
                    "model_spec": experiment["model_spec"], "plan": plan}
        path = folder / "optimization_input.json"
        _write_new(path, manifest)
        data = experiment["dataset_ref"]
        source = store.artifacts.register(run_id, store.project_root / data["path"],
                                          role="input", kind="dataset")
        if source["sha256"] != data["sha256"]:
            raise ExperimentStoreError("Source dataset changed during optimization registration")
        sources = [source]
        if data.get("provenance"):
            provenance = store.artifacts.register(run_id, store.project_root / data["provenance"]["path"],
                                                  role="input", kind="provenance")
            if provenance["sha256"] != data["provenance"]["sha256"]:
                raise ExperimentStoreError("Source provenance changed")
            sources.append(provenance)
        manifest_artifact = store.artifacts.register(
            run_id, path, role="input", kind="manifest", media_type="application/json",
            source_artifact_ids=[item["artifact_id"] for item in sources],
        )
        input_ids = [item["artifact_id"] for item in sources] + [manifest_artifact["artifact_id"]]
        _stage(store, run_id, "Evaluating Firefly candidates", 2)
        from ml_lab.application import execute_task
        from ml_lab.optimization import (FireflyConfig, FunctionEvaluator, OptimizationRuntime,
                                         ParameterSpec, SearchSpace, optimize)
        from ml_lab.optimization.firefly import OptimizationStopped
        space = SearchSpace([ParameterSpec(plan["parameter"], "integer" if plan["kind"] == "integer" else "continuous", plan["lower"], plan["upper"])])
        trials = 0
        def evaluate(parameters, *, iteration, candidate_id):
            nonlocal trials
            check_cancelled(store, run_id)
            trials += 1
            entry = {"schema": "ml-lab.optimization-trial@1", "trial": trials,
                     "iteration": iteration, "candidate_id": candidate_id,
                     "parameters": dict(parameters), "metric": plan["metric"], "goal": plan["goal"]}
            try:
                settings = _override_settings(experiment, plan, parameters[plan["parameter"]])
                payload = deepcopy(handoff["payload"])
                payload.update(settings)
                response = execute_task(handoff["task"], payload)
                check_cancelled(store, run_id)
                from .datasets import verify_selection
                verify_selection(store, data)
                observed = _metric(response, experiment["model_spec"]["family"], plan["metric"])
                fitness = observed if plan["goal"] == "minimize" else -observed
                entry.update({"status": "ok", "observed": observed, "fitness": fitness})
                return {"fitness": fitness, "objective_terms": {plan["metric"]: observed}}
            except RunCancelled:
                entry.update({"status": "cancelled"})
                raise
            except Exception as exc:
                entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"[:1000]})
                raise
            finally:
                trial_path = folder / f"trial_{trials:04d}.json"
                _write_new(trial_path, entry)
                store.artifacts.register(run_id, trial_path, role="output", kind="trial",
                                         media_type="application/json", source_artifact_ids=input_ids)
        output = folder / "optimizer_outputs"
        try:
            result = optimize(
                space, FunctionEvaluator(space, evaluate), algorithm="firefly",
                config=FireflyConfig(population_size=plan["population_size"], iterations=plan["iterations"],
                                     seed=plan["seed"]),
                runtime=OptimizationRuntime(device="cpu", candidate_parallelism="serial", candidate_workers=1),
                output_path=str(output),
                metadata={"experiment_id": run["experiment_id"], "revision": run["revision"], "metric": plan["metric"]},
                should_stop=lambda: cancellation_requested(store, run_id),
            )
        except OptimizationStopped as exc:
            raise RunCancelled(str(exc)) from exc
        check_cancelled(store, run_id)
        _stage(store, run_id, "Registering optimization history", 3)
        from .datasets import verify_selection
        verify_selection(store, data)
        artifacts = []
        for name, kind in (("optimization_search_space.json", "search_space"),
                           ("optimization_history.csv", "history"),
                           ("firefly_history.json", "history"),
                           ("firefly_evaluations.json", "evaluations"),
                           ("optimization_result.json", "result")):
            file = output / name
            artifacts.append(store.artifacts.register(run_id, file, role="output", kind=kind,
                            source_artifact_ids=input_ids))
        check_cancelled(store, run_id)
        _stage(store, run_id, "Optimization completed", 4, status="succeeded")
        store.update_run_status(run_id, "succeeded")
        return {"schema": OPTIMIZER_SCHEMA, "run_id": run_id, "status": "succeeded",
                "best_parameters": result.best_parameters, "best_fitness": result.best_fitness,
                "evaluations": result.evaluations, "artifacts": artifacts}
    except RunCancelled as exc:
        store.update_run_status(run_id, "cancelled")
        _stage(store, run_id, "Optimization cancelled", 4, status="cancelled")
        raise ConfiguredRunError(f"Optimization cancelled (run={run_id})", run_id=run_id) from exc
    except Exception as exc:
        store.update_run_status(run_id, "failed", error=f"{type(exc).__name__}: {exc}"[:2000])
        try:
            _stage(store, run_id, "Optimization failed", 4, status="failed")
        except (OSError, ExperimentStoreError):
            pass
        raise ConfiguredRunError(f"Optimization failed (run={run_id})", run_id=run_id) from exc


def optimize_configured_experiment(store: ExperimentStore, experiment_id: str, plan: Mapping[str, Any],
                                  *, expected_revision: int) -> dict[str, Any]:
    queued = queue_optimization_run(store, experiment_id, plan, expected_revision=expected_revision)
    return execute_optimization_run(store, queued["run_id"])


def list_optimization_controls(family: str, method: str) -> list[dict[str, str]]:
    """Numeric controls for the UI, sourced from the same model field registry."""
    if method == "suite":
        return []
    registry = describe_model_method(family, method)["controls"]
    found: list[dict[str, str]] = []
    def visit(fields, prefix):
        for field in fields:
            path = prefix + "." + field["name"]
            if field.get("type") == "object":
                visit(field.get("fields", []), path)
            else:
                try:
                    kind = _numeric_control(family, method, path)
                except ExperimentStoreError:
                    continue
                found.append({"path": path, "kind": kind})
    for group, fields in registry.items():
        visit(fields, group)
    return found


def optimization_metrics(family: str) -> list[str]:
    return sorted(_ALLOWED_METRICS.get(family, ()))
