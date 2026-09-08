from __future__ import annotations

import sys

import numpy as np

from ml_lab.experimental import get_manifest, list_experiments, run_experiment


def test_builtin_experimental_catalog_is_lazy():
    fractal_module = "ml_lab.experimental.binary_fractal_conversion"
    linux_module = "ml_lab.experimental.linux_binary_identification"
    clique_module = "ml_lab.experimental.max_clique_rl"
    sys.modules.pop(fractal_module, None)
    sys.modules.pop(linux_module, None)
    sys.modules.pop(clique_module, None)

    ids = [manifest.id for manifest in list_experiments()]
    assert ids == ["binary_fractal_conversion", "linux_binary_identification", "max_clique_rl"]
    assert fractal_module not in sys.modules
    assert linux_module not in sys.modules
    assert clique_module not in sys.modules

    linux = get_manifest("linux_binary_identification")
    assert "transformer_autoencoder" in linux.capabilities
    clique = get_manifest("max_clique_rl")
    assert "dqn" in clique.capabilities
    assert linux_module not in sys.modules
    assert clique_module not in sys.modules


def test_linux_binary_representation_modes():
    description = run_experiment("linux_binary_identification")
    assert description["experiment"] == "linux_binary_identification"

    rgb = run_experiment(
        "linux_binary_identification",
        mode="bytes_to_rgb",
        data=[0, 1, 2, 255],
        image_size=2,
    )
    assert rgb["shape"] == [2, 2, 3]
    assert rgb["array"][0][0] == [0, 1, 2]

    patches = run_experiment(
        "linux_binary_identification",
        mode="patch_tokens",
        data=list(range(48)),
        image_size=4,
        patch_size=2,
    )
    assert patches["shape"] == [4, 12]
    values = np.asarray(patches["tokens"])
    assert values.min() >= 0.0
    assert values.max() <= 1.0


def test_max_clique_experiment_greedy_path():
    graph = np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    result = run_experiment("max_clique_rl", mode="greedy", graph=graph.tolist())
    assert result["valid"] is True
    assert result["clique_size"] == 3
    assert set(result["clique"]) == {0, 1, 2}


def test_max_clique_generated_graph_reports_planted_clique():
    result = run_experiment(
        "max_clique_rl",
        mode="greedy",
        generated={
            "num_nodes": 8,
            "clique_size": 4,
            "background_edge_probability": 0.0,
            "seed": 7,
        },
    )
    assert result["valid"] is True
    assert result["clique_size"] == 4
    assert result["planted_clique"] == [0, 1, 2, 3]
