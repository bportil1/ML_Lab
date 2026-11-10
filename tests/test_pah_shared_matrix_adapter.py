"""Shared component integration contracts, without a browser dependency."""
from pathlib import Path


def test_single_shared_matrix_owner():
    root = Path(__file__).resolve().parents[1] / "src/ml_lab/ui"
    adapter = (root / "static/ml_lab_matrix_editor.js").read_text()
    template = (root / "templates/ml_lab_ui/matrix.html").read_text()
    assert "mountMatrixEditor(root" in adapter
    assert "createElement('td')" not in adapter  # no second grid implementation
    assert "data-shared-editor-url" in template
    assert "Save derived dataset" not in template  # rendered by PAH_Shared
    assert "source_sha256" in adapter  # preserves non-destructive source fingerprint
