"""Built-in experimental manifests.

Only lightweight manifest metadata lives here. Importing/discovering the catalog
must never import the corresponding implementation modules or their optional
research dependencies.
"""

from __future__ import annotations

from .manifest import ExperimentalManifest

BUILTIN_MANIFESTS: tuple[ExperimentalManifest, ...] = (
    ExperimentalManifest(
        id="linux_binary_identification",
        name="Linux Binary Identification",
        module_path="ml_lab.experimental.linux_binary_identification",
        description=(
            "Incubating byte-stream representation and Transformer-autoencoder ideas "
            "recovered from the legacy Linux Binary Identification project."
        ),
        status="experimental",
        capabilities=(
            "binary_representation",
            "byte_to_rgb",
            "patch_tokenization",
            "transformer_autoencoder",
            "compression",
        ),
        version="0.1-experimental",
        notes=(
            "Legacy visualization and clustering are intentionally excluded; generic representation "
            "components may later graduate into ml_lab.representation."
        ),
    ),
    ExperimentalManifest(
        id="binary_fractal_conversion",
        name="Binary Fractal Conversion",
        module_path="ml_lab.experimental.binary_fractal_conversion",
        description=(
            "Incubating binary-byte to fractal representation transforms recovered from the "
            "legacy Binary Image Analysis Project."
        ),
        status="experimental",
        capabilities=("binary_representation", "binary_to_fractal", "mandelbrot", "ifs"),
        version="0.1-experimental",
        notes=(
            "Returns numerical image arrays only. Legacy plotting, RBM, optimizer, and classifier "
            "code are intentionally excluded."
        ),
    ),
    ExperimentalManifest(
        id="max_clique_rl",
        name="Max Clique RL",
        module_path="ml_lab.experimental.max_clique_rl",
        description=(
            "Graph-specific maximum-clique reinforcement-learning sandbox recovered from the "
            "legacy Max Clique ML Solver."
        ),
        status="experimental",
        capabilities=("graph_search", "max_clique", "reinforcement_learning", "dqn"),
        version="0.1-experimental",
        notes=(
            "The legacy partitioned-RBM and optimizer paths are excluded. This module is not a "
            "stable general reinforcement-learning API."
        ),
    ),
)
