"""ML-UW-08: bounded, source-matched comparisons of representations.

This module runs existing sklearn estimators via sklearn_catalog and *reads*
existing revision-pinned neural results via the verified artifact registry. It
never retrains a neural model, loads a PyTorch checkpoint, or persists a fit.
"""
from __future__ import annotations

import csv
from io import StringIO
import math
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from ml_lab.experiments.comparison import TRAINING_EXECUTOR, _result_document
from ml_lab.representation.evaluation import reconstruction_metrics
from ml_lab.representation.pca import _feature_matrix
from ml_lab.representation.sklearn_catalog import (
    MAX_FEATURES, SCALING, explore_representation, method_spec,
)

MAX_COMPARE_ROWS = 400
MAX_METHODS = 5
MAX_NEURAL_RUNS = 4
MAX_RUN_SCAN = 25
SCHEMA = 'ml-lab.representation-comparison@1'


def _scaled(values: np.ndarray, scaling: str) -> np.ndarray:
    if scaling not in SCALING:
        raise ValueError('Unknown comparison scaling')
    cls = {'standard': StandardScaler, 'minmax': MinMaxScaler, 'robust': RobustScaler}.get(scaling)
    return cls().fit_transform(values) if cls is not None else values


def geometry_metrics(reference: Any, latent: Any) -> dict[str, float | None]:
    """Same-data neighborhood and distance-rank diagnostics, NOT a quality score.

    Pairwise correlations are on all rows of this capped cohort. All methods
    share the exact same input-space reference and neighbor count.
    """
    ref = np.asarray(reference, dtype=float)
    embedding = np.asarray(latent, dtype=float)
    if (ref.ndim != 2 or embedding.ndim != 2 or ref.shape[0] != embedding.shape[0]
            or not 8 <= len(ref) <= MAX_COMPARE_ROWS or not 1 <= embedding.shape[1]
            or not np.isfinite(ref).all() or not np.isfinite(embedding).all()):
        raise ValueError('Geometry requires 8–400 matching finite rows and nonempty embeddings')
    neighbors = min(5, (len(ref) - 1) // 2)
    local = float(trustworthiness(ref, embedding, n_neighbors=neighbors))
    a = pairwise_distances(ref)
    b = pairwise_distances(embedding)
    indices = np.triu_indices(len(ref), k=1)
    ar, br = a[indices], b[indices]
    # Undefined for constant/equidistant embeddings; a missing diagnostic is
    # more accurate than a fabricated zero or a scipy constant-input warning.
    rank_correlation = None
    if np.ptp(ar) > 1e-14 and np.ptp(br) > 1e-14:
        from scipy.stats import spearmanr
        rho = float(spearmanr(ar, br).statistic)
        if math.isfinite(rho):
            rank_correlation = rho
    return {'trustworthiness': local, 'distance_rank_spearman': rank_correlation,
            'neighbors': neighbors}


def _finite_matrix(value: Any, shape: tuple[int, int]) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != shape[0]:
        return None
    if any(not isinstance(row, list) or len(row) != shape[1] for row in value):
        return None
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return None
    if result.shape != shape or not np.isfinite(result).all():
        return None
    return result


def _recorded_neural(store, selection: Mapping[str, Any], *, experiment_id: str,
                     values: np.ndarray, scaled_reference: np.ndarray,
                     scaling: str, components: int) -> tuple[list[dict], list[dict]]:
    """Read strictly matching, authenticated result artifacts of owned runs."""
    included: list[dict] = []
    excluded: list[dict] = []
    for run in store.list_runs(experiment_id, limit=MAX_RUN_SCAN):
        if len(included) >= MAX_NEURAL_RUNS:
            break
        if run['status'] != 'succeeded' or run['environment'].get('executor') != TRAINING_EXECUTOR:
            continue
        snapshot = store.get_experiment(experiment_id, revision=run['revision'])
        spec = snapshot.get('model_spec', {})
        method = spec.get('method')
        if spec.get('family') != 'representation' or method not in ('mlp_autoencoder', 'transformer_autoencoder'):
            continue
        record_dataset = snapshot.get('dataset_ref', {})
        if any(record_dataset.get(key) != selection.get(key) for key in
               ('path', 'sha256', 'rows', 'feature_columns', 'target_columns')):
            excluded.append({'run_id': run['run_id'], 'reason': 'Different source or selected features'})
            continue
        doc, status = _result_document(store, run['run_id'])
        if status != 'ok' or not isinstance(doc, dict) or doc.get('task') != 'representation':
            excluded.append({'run_id': run['run_id'], 'reason': 'Missing or invalid verified result artifact'})
            continue
        result = doc.get('result')
        if not isinstance(result, dict) or result.get('method') != method:
            excluded.append({'run_id': run['run_id'], 'reason': 'Invalid recorded representation result'})
            continue
        metadata = result.get('metadata', {})
        if not isinstance(metadata, dict) or metadata.get('scaling') != scaling:
            excluded.append({'run_id': run['run_id'], 'reason': 'Training scaling differs from comparison'})
            continue
        if result.get('feature_names') != selection['feature_columns']:
            excluded.append({'run_id': run['run_id'], 'reason': 'Recorded feature order differs'})
            continue
        latent = _finite_matrix(result.get('latent'), (len(values), components))
        inverse = _finite_matrix(result.get('reconstruction'), values.shape)
        if latent is None or inverse is None:
            excluded.append({'run_id': run['run_id'], 'reason': 'Latent dimensions or reconstruction shape differs'})
            continue
        included.append({
            'method': method, 'title': ('MLP autoencoder' if method == 'mlp_autoencoder' else 'Transformer autoencoder'),
            'kind': 'verified_saved_neural', 'run_id': run['run_id'], 'revision': run['revision'],
            'metrics': reconstruction_metrics(values, inverse),
            'geometry': geometry_metrics(scaled_reference, latent),
            'scatter': _scatter_from_latent(latent), 'notes': 'Previously trained; full-dataset reconstruction',
        })
    return included, excluded[:5]


def _scatter_from_latent(latent: np.ndarray):
    if latent.shape[1] < 2:
        return None
    # The existing renderer owns deterministic row selection and coordinates.
    from ml_lab.representation.sklearn_catalog import _scatter
    return _scatter(latent, 0, 1)


def compare_representations(matrix: Any, *, methods: Sequence[str], components: int = 2,
                            scaling: str = 'minmax', store=None, selection=None,
                            experiment_id: str | None = None,
                            include_recorded_neural: bool = True) -> dict[str, Any]:
    """Bounded, read-only, full-dataset exploratory comparison.

    No random held-out split is advertised: every sklearn fit uses *all* rows;
    recorded neural results used their own training/validation procedures.
    """
    if isinstance(methods, (str, bytes)) or not isinstance(methods, Sequence) or not 2 <= len(methods) <= MAX_METHODS:
        raise ValueError('Select between 2 and 5 sklearn methods')
    if any(type(method) is not str for method in methods) or len(set(methods)) != len(methods):
        raise ValueError('Methods must be unique supported identifiers')
    for method in methods:
        method_spec(method)
    if type(components) is not int or not 1 <= components <= 8:
        raise ValueError('Latent dimensions must be between 1 and 8')
    values, names = _feature_matrix(matrix)
    rows, cols = values.shape
    if not 8 <= rows <= MAX_COMPARE_ROWS or cols > MAX_FEATURES:
        raise ValueError(f'Comparison supports 8–{MAX_COMPARE_ROWS} rows and at most {MAX_FEATURES} features')
    if components > min(rows, cols):
        raise ValueError('Latent dimensions exceed input rank/dimensions')
    scaled = _scaled(values, scaling)
    results, failures = [], []
    for method in methods:
        spec = method_spec(method)
        try:
            report = explore_representation(
                matrix, method=method, components=components, scaling=scaling,
                reconstruct=spec.can_inverse_transform, axis_x=0, axis_y=1,
                include_geometry=True,
            )
        except (ValueError, ArithmeticError) as exc:
            failures.append({'method': method, 'title': spec.title, 'reason': str(exc)[:250]})
            continue
        results.append({
            'method': method, 'title': spec.title, 'kind': 'fresh_sklearn',
            'run_id': None, 'revision': None,
            'metrics': {name: report['metrics'][name] for name in ('mse', 'rmse', 'mae') if name in report['metrics']},
            'geometry': report['geometry'], 'scatter': report['scatter'],
            'notes': 'Fit on this full dataset' if spec.can_inverse_transform else 'No inverse: reconstruction unavailable',
        })
    recorded, excluded = [], []
    if include_recorded_neural and store is not None and selection is not None and experiment_id:
        recorded, excluded = _recorded_neural(
            store, selection, experiment_id=experiment_id, values=values,
            scaled_reference=scaled, scaling=scaling, components=components,
        )
    return {
        'schema': SCHEMA, 'source': {'rows': rows, 'columns': cols, 'feature_names': names,
                                    'sha256': selection.get('sha256') if selection else None},
        'conditions': {'components': components, 'scaling': scaling,
                       'evaluation': 'in-sample descriptive; no independent held-out test',
                       'method_count': len(methods)},
        'results': results + recorded, 'failures': failures, 'excluded_runs': excluded,
        'note': ('All columns and input rows are shared. Geometry is calculated against one shared '
                 'scaled input matrix; only invertible methods report input-space reconstruction. '
                 'Neural results are verified stored artifacts from prior training, not freshly retrained. '
                 'Values are descriptive, not a defensible quality ranking or held-out evaluation.'),
    }


def comparison_csv(report: Mapping[str, Any]) -> str:
    """CSV export of the displayed numeric comparison; neutralize spreadsheet formulas."""
    output = StringIO(newline='')
    writer = csv.writer(output)
    writer.writerow(('method', 'source', 'run_id', 'revision', 'mse', 'rmse', 'mae',
                     'trustworthiness', 'distance_rank_spearman', 'neighbors'))
    for entry in report['results']:
        vals = [entry['title'], entry['kind'], entry['run_id'] or '', entry['revision'] or '',
                *(entry['metrics'].get(k, '') for k in ('mse', 'rmse', 'mae')),
                *(entry['geometry'].get(k, '') if entry['geometry'].get(k) is not None else ''
                  for k in ('trustworthiness', 'distance_rank_spearman', 'neighbors'))]
        writer.writerow(["'" + x if isinstance(x, str) and x.lstrip().startswith(('=', '+', '-', '@')) else x
                         for x in vals])
    return output.getvalue()


def _display_rows(report: Mapping[str, Any]) -> list[list[str]]:
    """Single numeric serialization for static visual and TeX table exports."""
    output = []
    for entry in report['results']:
        def metric(values: Mapping[str, Any], key: str) -> str:
            value = values.get(key)
            return f'{value:.5g}' if isinstance(value, (float, int)) and math.isfinite(value) else 'N/A'
        output.append([
            str(entry['title']),
            'Saved neural' if entry['kind'] == 'verified_saved_neural' else 'Fresh fit',
            metric(entry['metrics'], 'mse'), metric(entry['metrics'], 'rmse'),
            metric(entry['metrics'], 'mae'), metric(entry['geometry'], 'trustworthiness'),
            metric(entry['geometry'], 'distance_rank_spearman'),
            (entry['run_id'][:12] if entry['run_id'] else 'Ephemeral'),
        ])
    return output


def comparison_latex(report: Mapping[str, Any]) -> str:
    """Escaped LaTeX tabular of displayed values, not executable LaTeX from data."""
    def esc(value: str) -> str:
        specials = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
        return ''.join(specials.get(char, char) for char in value)
    header = ['Method', 'Origin', 'MSE', 'RMSE', 'MAE', 'Trustworthiness', 'Distance rank rho', 'Run']
    lines = [r'\begin{tabular}{llrrrrrl}', r'\hline', ' & '.join(map(esc, header)) + r' \\', r'\hline']
    lines.extend(' & '.join(map(esc, row)) + r' \\' for row in _display_rows(report))
    lines.extend([r'\hline', r'\end{tabular}',
                  '% In-sample reconstruction and geometry; no held-out quality ranking.'])
    return '\n'.join(lines) + '\n'


def comparison_svg(report: Mapping[str, Any]) -> str:
    """Static, self-contained SVG image of the bounded comparison table."""
    from html import escape
    columns = ['Method', 'Origin', 'MSE', 'RMSE', 'MAE', 'Trustworthiness', 'Distance rank rho', 'Run']
    widths = [180, 112, 85, 85, 85, 132, 140, 113]
    left = [12]
    for width in widths[:-1]:
        left.append(left[-1] + width)
    full_width = sum(widths) + 24
    rows = _display_rows(report)
    full_height = 78 + len(rows) * 37
    chunks = [f'<svg xmlns="http://www.w3.org/2000/svg" role="img" aria-label="ML Lab representation comparison table" viewBox="0 0 {full_width} {full_height}">',
              f'<rect x="0" y="0" width="{full_width}" height="{full_height}" fill="white"/>',
              '<text x="12" y="22" font-family="sans-serif" font-size="14" font-weight="700">Representation comparison (in-sample)</text>',
              '<text x="12" y="40" font-family="sans-serif" font-size="10">Same source, scaling and latent width; not a held-out score</text>',
              f'<rect x="8" y="48" width="{full_width - 16}" height="28" fill="#e9eef2"/>']
    for i, label in enumerate(columns):
        chunks.append(f'<text x="{left[i]}" y="66" font-family="sans-serif" font-size="10" fill="#263344">{escape(label)}</text>')
    for index, row in enumerate(rows):
        y = 76 + index * 37
        if index % 2 == 1:
            chunks.append(f'<rect x="8" y="{y}" width="{full_width-16}" height="37" fill="#f5f6f8"/>')
        for j, cell in enumerate(row):
            # Fixed method names/verified run IDs and bounded viewBox. Limit
            # length for a predictable, self-contained graphic.
            shown = cell if len(cell) <= 28 else cell[:25] + '…'
            chunks.append(f'<text x="{left[j]}" y="{y + 22}" font-family="sans-serif" font-size="11" fill="#263344">{escape(shown)}</text>')
    chunks.append('</svg>')
    return '\n'.join(chunks)
