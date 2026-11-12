"""Read-only, project-scoped catalog for the General ML Tools landing page.

This presents existing experiment *revisions*, not running workers or a new
scientific execution pipeline. A link to an eligible configuration is not an
assertion that the underlying dataset still passes verification at run time.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping

from ml_lab.experiments.models import MODEL_SCHEMA, list_model_families

DATASET_SCHEMA = 'ml-lab.experiment-dataset@1'
TOOL_IDS = ('models', 'ebm', 'optimization', 'generative')
MAX_CATALOG_EXPERIMENTS = 200
MAX_ROWS_PER_TOOL = 8


def _configured_model(row: Mapping) -> tuple[str, str]:
    spec = row.get('model_spec')
    if not isinstance(spec, Mapping) or spec.get('schema') != MODEL_SCHEMA:
        return '', ''
    family, method = spec.get('family'), spec.get('method')
    known = {item['id'] for item in list_model_families()}
    if family not in known or not isinstance(method, str) or not method:
        return '', ''
    return family, method


def _optimizable(family: str, method: str, *, memo: dict) -> bool:
    key = (family, method)
    if key not in memo:
        # The same authoritative registry gates the optimization form; do not
        # invent a list of searchable hyperparameters in the workbench UI.
        from ml_lab.experiments import list_optimization_controls, optimization_metrics
        try:
            memo[key] = bool(list_optimization_controls(family, method) and optimization_metrics(family))
        except (ValueError, ImportError, RuntimeError):
            # Optional engine extras may be absent in a base UI installation.
            memo[key] = False
    return memo[key]


def general_ml_catalog(experiments: Iterable[Mapping], *, per_tool: int = MAX_ROWS_PER_TOOL) -> dict:
    """Classify up to 200 latest project-local experiments into four tools.

    The caller owns project isolation. This function makes no database writes,
    reads no source rows, and does not launch or reconfigure experiments.
    """
    if not isinstance(per_tool, int) or isinstance(per_tool, bool) or not 1 <= per_tool <= 20:
        raise ValueError('per_tool must be between 1 and 20')
    groups: dict[str, list[dict]] = {key: [] for key in TOOL_IDS}
    optimizable: dict = {}
    read = 0
    for exp in experiments:
        if read >= MAX_CATALOG_EXPERIMENTS:
            break
        read += 1
        dataset = exp.get('dataset_ref')
        if not isinstance(dataset, Mapping) or dataset.get('schema') != DATASET_SCHEMA:
            continue
        family, method = _configured_model(exp)
        entry = {
            'experiment_id': exp['experiment_id'],
            'name': exp['name'],
            'revision': exp['revision'],
            'family': family,
            'method': method,
            'configured': bool(family),
        }
        if len(groups['models']) < per_tool:
            groups['models'].append(entry)
        if family == 'rbm' and len(groups['ebm']) < per_tool:
            groups['ebm'].append(entry)
        if family and len(groups['optimization']) < per_tool and _optimizable(family, method, memo=optimizable):
            groups['optimization'].append(entry)
        if len(groups['generative']) < per_tool:
            groups['generative'].append(entry)
        elif family in {'gan', 'rbm'} and not any(row['family'] in {'gan', 'rbm'} for row in groups['generative']):
            # A recent configured generator should not be invisible among many
            # ordinary datasets. Do not discard existing record or history.
            groups['generative'][-1] = entry
    return groups
