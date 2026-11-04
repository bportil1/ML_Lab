"""S6.1: explicit, project-local synthetic dataset generation.

Statistical methods use NumPy; model-based methods call ML_Lab's existing
GAN/RBM engines. This does not load a previously trained model: model-based
requests train afresh from the pinned experiment revision, then sample.
Tangent-manifold augmentation remains HSQA_DBN-owned.
"""
from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
from typing import Any, Mapping

from .datasets import verify_selection
from .models import MODEL_SCHEMA, prepare_configured_model_handoff
from .run_control import RunCancelled, check_cancelled, write_owner
from .store import ExperimentStore, ExperimentStoreError
from .training import ConfiguredRunError, _run_folder, _stage, _write_new

SYNTHETIC_SCHEMA = "ml-lab.synthetic-plan@1"
SYNTHETIC_EXECUTOR = "ml_lab.experiments.synthetic@1"
STATISTICAL_METHODS = ("bootstrap", "gaussian_independent", "gaussian_correlated", "student_t")
MODEL_METHODS = ("gan", "rbm")
ALL_METHODS = STATISTICAL_METHODS + MODEL_METHODS


def validate_synthetic_plan(experiment: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Any]:
    """Reject unrecognized settings and methods incompatible with the experiment."""
    if not isinstance(plan, Mapping) or set(plan) - {"method", "count", "seed", "degrees_of_freedom", "schema"}:
        raise ExperimentStoreError("Unsupported synthetic sampling controls")
    if "schema" in plan and plan["schema"] != SYNTHETIC_SCHEMA:
        raise ExperimentStoreError("Unsupported synthetic plan schema")
    method = plan.get("method")
    if method not in ALL_METHODS:
        raise ExperimentStoreError("Unknown synthetic generation method")
    count, seed = plan.get("count", 100), plan.get("seed", 42)
    if type(count) is not int or not 2 <= count <= 5000:
        raise ExperimentStoreError("Synthetic sample count must be between 2 and 5000")
    if type(seed) is not int or not 0 <= seed < 2**32 - 1:
        raise ExperimentStoreError("Synthetic seed must be a nonnegative 32-bit integer")
    df = plan.get("degrees_of_freedom", 5.0)
    if type(df) not in (int, float) or not math.isfinite(df) or not 2 < df <= 1000:
        raise ExperimentStoreError("Student-t degrees of freedom must be finite and greater than 2")
    if method in MODEL_METHODS:
        spec = experiment.get("model_spec", {})
        if spec.get("schema") != MODEL_SCHEMA or spec.get("family") != method:
            raise ExperimentStoreError(f"Save a {method.upper()} model configuration in this experiment first")
    if not experiment.get("dataset_ref", {}).get("feature_columns"):
        raise ExperimentStoreError("Select a dataset with feature columns first")
    return {"schema": SYNTHETIC_SCHEMA, "method": method, "count": count,
            "seed": seed, "degrees_of_freedom": float(df)}


def queue_synthetic_run(
    store: ExperimentStore, experiment_id: str, plan: Mapping[str, Any], *, expected_revision: int,
) -> dict[str, Any]:
    if type(expected_revision) is not int or expected_revision < 1:
        raise ExperimentStoreError("Select a current experiment revision")
    experiment = store.get_experiment(experiment_id)
    if experiment["revision"] != expected_revision:
        raise ExperimentStoreError("Experiment revision changed; reload the synthetic workspace")
    verify_selection(store, experiment["dataset_ref"])
    frozen = validate_synthetic_plan(experiment, plan)
    return store.create_run(
        experiment_id, revision=expected_revision, expected_current_revision=expected_revision,
        seed=frozen["seed"], environment={"executor": SYNTHETIC_EXECUTOR, "plan": frozen,
                                             "runtime": "local", "sampling": "train-now" if frozen["method"] in MODEL_METHODS else "statistical"},
    )


def _feature_matrix(store: ExperimentStore, selected: Mapping[str, Any]):
    import numpy as np
    from ml_lab.data import read_dataset_frame

    frame = read_dataset_frame(store.project_root / selected["path"])
    try:
        values = frame[list(selected["feature_columns"])].to_numpy(dtype=float)
    except (ValueError, TypeError, KeyError) as exc:
        raise ExperimentStoreError("Selected feature columns must be numeric") from exc
    if values.ndim != 2 or values.shape[0] < 2 or not np.isfinite(values).all():
        raise ExperimentStoreError("Synthetic generation requires at least two rows of finite numeric features")
    return values


def _statistical_samples(values: Any, plan: Mapping[str, Any]):
    import numpy as np

    rng = np.random.default_rng(plan["seed"])
    method, count = plan["method"], plan["count"]
    if method == "bootstrap":
        return values[rng.integers(0, len(values), size=count)].copy()
    mean = values.mean(axis=0)
    if method == "gaussian_independent":
        return rng.normal(mean, values.std(axis=0, ddof=1), size=(count, values.shape[1]))
    covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    # A covariance estimate from a small dataset may be rank-deficient.
    eigenvalues, vectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    standard_normal = rng.standard_normal((count, values.shape[1]))
    correlated = standard_normal @ (vectors * np.sqrt(eigenvalues)).T
    if method == "gaussian_correlated":
        return mean + correlated
    # Rescale so that the theoretical Student-t covariance matches the sample
    # covariance when degrees_of_freedom > 2.
    df = plan["degrees_of_freedom"]
    scale = np.sqrt((df - 2.0) / df)
    radial = np.sqrt(rng.chisquare(df, size=(count, 1)) / df)
    return mean + correlated * scale / radial


def _model_samples(store: ExperimentStore, run: Mapping[str, Any], plan: Mapping[str, Any]):
    """Train once now and sample; no claim of persistent trained-weight reuse."""
    import numpy as np

    handoff = prepare_configured_model_handoff(store, run["experiment_id"], revision=run["revision"])
    payload = deepcopy(handoff["payload"])
    training = deepcopy(payload.get("training_config", {}))
    training["random_state"] = plan["seed"]
    if plan["method"] == "gan":
        from ml_lab.generative.gan import GANModelConfig, GANTrainingConfig, run as fit_gan, sample
        # Preserve the trained model in memory for this request; do not
        # reimplement or serialize GAN weights through the legacy JSON API.
        config = GANModelConfig(**payload.get("model_config", {}))
        training["generated_sample_count"] = plan["count"]
        fitted = fit_gan(payload["X"], model_config=config, training_config=GANTrainingConfig(**training))
        return np.asarray(sample(fitted, plan["count"], random_state=plan["seed"] + 1), dtype=float)

    from ml_lab.energy_based.training import run as fit_rbm
    from ml_lab.energy_based.rbm import run_chain
    from ml_lab.neural.backend import require_torch
    from ml_lab.neural.runtime import seed_everything

    # Avoid expensive distribution diagnostics merely to obtain samples. We
    # use the trained RBM's existing Gibbs-chain API for any requested count.
    training["distribution_monitoring"] = {"enabled": False}
    fitted = fit_rbm(payload["X"], family=payload["family"], hidden_dim=payload["hidden_dim"],
                     model_config=payload.get("model_config", {}), training_config=training,
                     feature_names=list(handoff["dataset_ref"]["feature_columns"]))
    torch = require_torch(purpose="RBM synthetic sampling")
    seed_everything(plan["seed"] + 1, deterministic=True, torch_module=torch)
    original = np.asarray(payload["X"], dtype=float)
    rng = np.random.default_rng(plan["seed"] + 1)
    initial = original[rng.integers(0, len(original), size=plan["count"])]
    model = fitted.model
    model.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(initial, dtype=model.W.dtype, device=model.W.device)
        sampled = run_chain(model, tensor, steps=10)
    return np.asarray(sampled.detach().cpu().numpy(), dtype=float)


def execute_synthetic_run(store: ExperimentStore, run_id: str) -> dict[str, Any]:
    """Execute a queued generator; publish a Data Lab-selectable derived CSV."""
    run = store.get_run(run_id)
    if run["environment"].get("executor") != SYNTHETIC_EXECUTOR or run["status"] != "queued":
        raise ExperimentStoreError("Only queued synthetic runs may execute")
    experiment = store.get_experiment(run["experiment_id"], revision=run["revision"])
    plan = validate_synthetic_plan(experiment, run["environment"]["plan"])
    store.update_run_status(run_id, "running")
    try:
        folder = _run_folder(store, run_id, create=True)
        write_owner(store, run_id)
        check_cancelled(store, run_id)
        _stage(store, run_id, "Validating dataset and generation settings", 1)
        selected = verify_selection(store, experiment["dataset_ref"])
        source_path = store.project_root / selected["path"]
        source = store.artifacts.register(run_id, source_path, role="input", kind="dataset")
        if source["sha256"] != selected["sha256"]:
            raise ExperimentStoreError("Source changed during input registration")
        input_ids = [source["artifact_id"]]
        if selected.get("provenance"):
            prev = store.artifacts.register(run_id, store.project_root / selected["provenance"]["path"],
                                            role="input", kind="provenance")
            if prev["sha256"] != selected["provenance"]["sha256"]:
                raise ExperimentStoreError("Source lineage changed during input registration")
            input_ids.append(prev["artifact_id"])
        manifest = {"schema": SYNTHETIC_SCHEMA, "experiment_id": run["experiment_id"],
                    "revision": run["revision"], "run_id": run_id, "source": selected,
                    "model_spec": experiment["model_spec"] if plan["method"] in MODEL_METHODS else None,
                    "plan": plan, "trained_model_reused": False}
        _write_new(folder / "input.json", manifest)
        input_manifest = store.artifacts.register(run_id, folder / "input.json", role="input", kind="manifest",
                                                  media_type="application/json", source_artifact_ids=input_ids)
        input_ids.append(input_manifest["artifact_id"])
        _stage(store, run_id, "Generating synthetic samples", 2)
        check_cancelled(store, run_id)
        if plan["method"] in MODEL_METHODS:
            generated = _model_samples(store, run, plan)
        else:
            generated = _statistical_samples(_feature_matrix(store, selected), plan)
        import numpy as np
        generated = np.asarray(generated, dtype=float)
        if generated.shape != (plan["count"], len(selected["feature_columns"])) or not np.isfinite(generated).all():
            raise ExperimentStoreError("Generator returned the wrong shape or nonfinite features")
        check_cancelled(store, run_id)
        _stage(store, run_id, "Verifying source and writing derived dataset", 3)
        verify_selection(store, selected)
        if store.artifacts.verify(source["artifact_id"])["status"] != "valid":
            raise ExperimentStoreError("Source changed during generation")
        import pandas as pd
        from ml_lab.data.provenance import persist_transformation_provenance, provenance_sidecar_path
        csv_file = folder / "synthetic.csv"
        with csv_file.open("x", encoding="utf-8", newline="") as output:
            pd.DataFrame(generated, columns=selected["feature_columns"]).to_csv(output, index=False)
        # The existing Data Lab sidecar contract makes the output immediately
        # selectable as a derived dataset, without bypassing provenance checks.
        event = persist_transformation_provenance(
            source_path=source_path, derived_path=csv_file, recipe=manifest,
            operations=[{"method": plan["method"], "count": plan["count"], "seed": plan["seed"],
                         "source_run": run_id, "trained_now": plan["method"] in MODEL_METHODS}],
        )
        verify_selection(store, selected)
        output = store.artifacts.register(run_id, csv_file, role="output", kind="dataset",
                                          media_type="text/csv", source_artifact_ids=input_ids,
                                          metadata={"generated": True, "method": plan["method"]})
        sidecar = store.artifacts.register(run_id, provenance_sidecar_path(csv_file), role="output", kind="provenance",
                                           media_type="application/json", source_artifact_ids=[*input_ids, output["artifact_id"]])
        summary = {"schema": "ml-lab.synthetic-result@1", "run_id": run_id, "method": plan["method"],
                   "count": plan["count"], "features": list(selected["feature_columns"]),
                   "dataset_path": output["path"], "dataset_sha256": output["sha256"],
                   "provenance_event_id": event["event_id"], "trained_model_reused": False}
        _write_new(folder / "result.json", summary)
        result = store.artifacts.register(run_id, folder / "result.json", role="output", kind="result",
                                          media_type="application/json", source_artifact_ids=[*input_ids, output["artifact_id"], sidecar["artifact_id"]])
        check_cancelled(store, run_id)
        _stage(store, run_id, "Synthetic dataset ready", 4, status="succeeded")
        store.update_run_status(run_id, "succeeded")
        return {"schema": summary["schema"], "run_id": run_id, "status": "succeeded",
                "dataset": output, "provenance": sidecar, "result": result}
    except RunCancelled as exc:
        store.update_run_status(run_id, "cancelled")
        _stage(store, run_id, "Synthetic run cancelled", 4, status="cancelled")
        raise ConfiguredRunError(f"Synthetic run cancelled (run={run_id})", run_id=run_id) from exc
    except Exception as exc:
        store.update_run_status(run_id, "failed", error=f"{type(exc).__name__}: {exc}"[:2000])
        try:
            _stage(store, run_id, "Synthetic generation failed", 4, status="failed")
        except (OSError, ExperimentStoreError):
            pass
        raise ConfiguredRunError(f"Synthetic generation failed (run={run_id})", run_id=run_id) from exc


def generate_synthetic_dataset(store: ExperimentStore, experiment_id: str, plan: Mapping[str, Any], *,
                               expected_revision: int) -> dict[str, Any]:
    """Standalone, synchronous convenience API."""
    run = queue_synthetic_run(store, experiment_id, plan, expected_revision=expected_revision)
    return execute_synthetic_run(store, run["run_id"])
