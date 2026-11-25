"""Regression for the Feature Profiling dependency heatmap unreadable text."""
from pathlib import Path
import math
import re

ROOT = Path(__file__).resolve().parents[1]
CSS = (ROOT / "src/ml_lab/ui/static/ml_lab_workbench.css").read_text(encoding="utf-8")
HTML = (ROOT / "src/ml_lab/ui/templates/ml_lab_ui/workbench_feature_profiling.html").read_text(encoding="utf-8")


def _rgb(color):
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def _luminance(color):
    r, g, b = _rgb(color)
    channels = [(v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4)
                for v in (r, g, b)]
    return sum(c * w for c, w in zip(channels, (.2126, .7152, .0722)))


def _contrast(a, b):
    x, y = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (x + .05) / (y + .05)


def _blend_white(color, fraction):
    return '#' + ''.join(f'{round(255 * (1 - fraction) + channel * 255 * fraction):02x}'
                       for channel in _rgb(color))


def test_dependency_matrix_opaque_canvas_and_dark_text():
    assert '.mlwb-dependency-matrix' in CSS
    assert 'color-scheme: light;' in CSS
    assert 'background: #fff; color: #172b3c' in CSS
    assert '.mlwb-dependency-matrix a:visited' in CSS
    assert 'mlwb-dep-cell mlwb-dep-' in HTML  # Data/selection contract preserved
    assert 'data-strength="{{ cell.strength }}"' in HTML
    assert _contrast('#172b3c', '#ffffff') >= 7.0
    assert _contrast('#172b3c', '#e9f1f6') >= 7.0


def test_all_heatmap_strengths_are_legible_unselected():
    for palette in ('#008f86', '#17679c'):
        for fraction in (.10, .22, .34, .47):
            cell_background = _blend_white(palette, fraction)
            assert _contrast('#0b385d', cell_background) >= 4.5
    styles = CSS[CSS.index('/* Mix with *white*'):CSS.index('/* ML-UW-06:')]
    assert not re.search(r'background\s*:\s*color-mix\([^;]*transparent', styles)
