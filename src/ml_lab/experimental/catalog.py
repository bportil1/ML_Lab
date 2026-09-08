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
        id="transformer_vae",
        name="Transformer VAE",
        module_path="ml_lab.experimental.transformer_vae",
        description=(
            "Experimental Transformer variational autoencoder with a diagonal-Gaussian "
            "latent posterior, beta-weighted KL regularization, and optional KL warm-up."
        ),
        status="experimental",
        capabilities=(
            "representation",
            "compression",
            "variational_autoencoder",
            "transformer",
            "generative_modeling",
        ),
        version="0.1-experimental",
        notes=(
            "Uses the stable token adapter and shared neural runtime. Legacy ICMI/TC/DWKL "
            "objectives remain outside this baseline experiment."
        ),
    ),
    ExperimentalManifest(
        id="transformer_vae_decomposed",
        name="Transformer VAE — ICMI/TC/DWKL",
        module_path="ml_lab.experimental.transformer_vae_decomposed",
        description=(
            "Experimental Transformer VAE with a beta-TCVAE-style decomposition of "
            "the latent KL into ICMI, total correlation, and dimension-wise KL terms."
        ),
        status="experimental",
        capabilities=(
            "representation",
            "compression",
            "variational_autoencoder",
            "transformer",
            "icmi",
            "total_correlation",
            "dimension_wise_kl",
            "disentanglement",
        ),
        version="0.1-experimental",
        notes=(
            "Uses a differentiable minibatch-mixture density estimator. The ICMI/TC/DWKL "
            "terms are research diagnostics/objectives and remain outside the stable API."
        ),
    ),
    ExperimentalManifest(
        id="transformer_vae_contractive",
        name="Transformer VAE — Contractive",
        module_path="ml_lab.experimental.transformer_vae_contractive",
        description=(
            "Experimental Transformer VAE with first-order contractive regularization "
            "of the posterior-mean Jacobian with respect to the input representation."
        ),
        status="experimental",
        capabilities=(
            "representation",
            "compression",
            "variational_autoencoder",
            "transformer",
            "contractive_regularization",
            "jacobian_penalty",
            "tangent_regularization",
        ),
        version="0.1-experimental",
        notes=(
            "Supports exact and Hutchinson estimates of the encoder posterior-mean "
            "Jacobian norm. This first-order research objective remains outside the stable API."
        ),
    ),
    ExperimentalManifest(
        id="gan_stabilization",
        name="GAN Stabilization Research",
        module_path="ml_lab.experimental.gan_stabilization",
        description=(
            "Experimental GAN training mechanisms adapted from the legacy GAN Stabilization Sandbox "
            "while reusing ML_Lab stable GAN model definitions."
        ),
        status="experimental",
        capabilities=(
            "gan",
            "attention_conditioning",
            "contrastive_critic",
            "tangent_negatives",
            "critic_coupling",
            "training_stabilization",
        ),
        version="0.1-experimental",
        notes=(
            "Stable GAN architectures remain unchanged. Attention/context sharing, score-tangent negatives, "
            "feature matching, and scheduled coupled discriminators are opt-in research mechanisms."
        ),
    ),
    ExperimentalManifest(
        id="partitioned_rbm_training",
        name="Partitioned RBM Training",
        module_path="ml_lab.experimental.partitioned_rbm_training",
        description=(
            "Experimental hierarchical RBM training strategy that learns local feature blocks, "
            "merges neighboring blocks, and refines newly exposed cross-block interactions."
        ),
        status="experimental",
        capabilities=(
            "energy_based_model",
            "rbm",
            "partitioned_training",
            "hierarchical_training",
            "cdk",
        ),
        version="0.1-experimental",
        notes=(
            "Rebuilt from legacy partitioned-RBM prototypes using the stable ML_Lab RBM/CD-k APIs. "
            "Samples are never partitioned; visible and hidden blocks are partitioned independently."
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
