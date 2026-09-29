from __future__ import annotations

from ml_lab.cli import _parser


def test_cli_default_result_roots_use_analysis_directory():
    parser = _parser()
    cases = [
        (["data", "inspect", "dataset.csv"], "analysis/data/inventory.json"),
        (["data", "profile", "dataset.csv"], "analysis/data/profile.json"),
        (["data", "compare", "dataset.csv"], "analysis/data/comparison.json"),
        (["rbm-train", "dataset.csv"], "analysis/rbm"),
        (["classify", "dataset.csv"], "analysis/classification"),
        (["regress", "dataset.csv"], "analysis/regression"),
        (["cluster", "dataset.csv"], "analysis/clustering"),
        (["represent", "dataset.csv"], "analysis/representation"),
        (["gan", "dataset.csv"], "analysis/gan"),
    ]
    for argv, expected in cases:
        args = parser.parse_args(argv)
        assert args.output == expected


def test_data_transform_help_names_analysis_as_default_root():
    parser = _parser()
    args = parser.parse_args(["data", "transform", "dataset.csv", "--recipe", "recipe.json"])
    assert args.output is None
