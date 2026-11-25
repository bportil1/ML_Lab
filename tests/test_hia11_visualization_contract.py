"""HIA-11 owner UI must remain tied to verified, opt-in HIA estimators."""
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'
JS = (BASE / 'static/ml_lab_hia_visuals.mjs').read_text()
TEMPLATE = (BASE / 'templates/ml_lab_ui/workbench_structured.html').read_text()
ROUTES = (BASE / 'workbench_structured_routes.py').read_text()


def test_visualization_is_owner_side_and_uses_existing_shared_viewer():
    assert 'data-hia-visuals' in TEMPLATE
    assert 'data-graph-surface' in TEMPLATE
    assert 'type="module"' in TEMPLATE
    assert 'ml_lab_hia_visuals.mjs' in TEMPLATE
    assert 'ml_lab_hia_visuals.css' in TEMPLATE
    assert 'v=graph_controller_revision' in TEMPLATE
    assert "'ml_lab_hia_visuals.mjs'" in ROUTES
    assert "'ml_lab_hia_visuals.css'" in ROUTES
    assert 'mountInteractiveGraph' not in JS
    assert 'viewer.update(' not in JS  # display colors do not reset camera/selection
    assert 'MutationObserver' in JS  # repaint only the currently rendered page
    assert 'data-node-id' in JS


def test_original_estimators_publish_verified_results_and_invalidation():
    for name, event in (('ml_lab_hia_atomic.js', 'atomic'),
                        ('ml_lab_hia_level.js', 'level'),
                        ('ml_lab_hia_network.js', 'network')):
        text = (BASE / 'static' / name).read_text()
        assert f'mlwb:hia-{event}-report' in text
        assert f'mlwb:hia-{event}-invalidated' in text
        assert 'source_revision' in text
    assert "report?.source_digest===state.graph.binding_contract?.source_digest" in JS
    assert "sourceRevision===state.graph.view.source_revision" in JS


def test_overlay_preserves_scientific_boundaries_and_export_scopes():
    assert 'Level aggregation, shown on mapped members' in JS
    assert 'Atomic observations, mapped nodes only' in JS
    assert 'Unmapped and unmeasured nodes are neutral' in JS
    assert 'ml-lab-hia11-visible-graph.svg' in JS
    assert 'ml-lab-hia11-visible-graph.png' in JS
    assert 'ml-lab-hia11-scientific-overlay.csv' in JS
    assert 'ml-lab-hia11-visible-graph.svg' in JS
