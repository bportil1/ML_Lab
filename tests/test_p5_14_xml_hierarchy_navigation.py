"""P5-14 hierarchy navigation contract: source-only tests need no Flask/Dash."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src' / 'ml_lab' / 'ui'


def test_navigation_controls_are_attached_to_existing_structural_explorer():
    html = (ROOT / 'templates/ml_lab_ui/xml_structure.html').read_text()
    for identifier in (
        'data-xml-ancestry', 'data-xml-go-parent', 'data-xml-go-previous',
        'data-xml-go-next', 'data-xml-sibling-position',
        'data-xml-inspector-parent', 'data-xml-inspector-siblings',
    ):
        assert identifier in html
    assert 'data-record-fields-url' in html  # original extraction workflow retained
    assert 'data-xml-use-record-root' in html


def test_navigation_is_derived_from_existing_profiles_not_new_xml_parser():
    script = (ROOT / 'static/ml_lab_xml_structure.js').read_text()
    assert 'const siblingPaths = (canonicalPath)' in script
    assert 'const renderNavigation = (canonicalPath)' in script
    assert 'profileByPath.has(current)' in script
    assert 'button.dataset.xmlSelectPath = path' in script
    assert 'if (event.key === "ArrowLeft")' in script
    assert 'if (event.key === "ArrowRight" && parts.childList)' in script
    assert 'if (selectedPath) expandAncestors(selectedPath)' in script
    # Filtering must visit all sibling branches, even after an earlier match.
    assert 'children.forEach((child) => { if (descendantMatches(child, query, candidatesMode)) childMatch = true; });' in script
    assert 'children.some((child) => descendantMatches' not in script


def test_repeated_cardinality_is_descriptive_not_a_schema_change():
    script = (ROOT / 'static/ml_lab_xml_structure.js').read_text()
    assert 'min_per_parent' in script and 'max_per_parent' in script
    assert 'repeated ${formatNumber(cardinality.min_per_parent)}' in script
    css = (ROOT / 'static/ml_lab.css').read_text()
    assert '.xml-ancestry-link[aria-current="location"]' in css
    assert '.xml-tree-select:focus-visible' in css
