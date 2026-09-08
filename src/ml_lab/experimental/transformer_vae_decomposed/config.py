from __future__ import annotations

from dataclasses import dataclass

from ml_lab.experimental.transformer_vae.config import TransformerVAEConfig
from ml_lab.neural.config import NeuralTrainingConfig
from ml_lab.representation.config import ScalingMode


@dataclass(frozen=True, slots=True)
class DecomposedTransformerVAETrainingConfig(NeuralTrainingConfig):
    """Training controls for the experimental ICMI/TC/DWKL VAE objective.

    The objective follows the beta-TCVAE decomposition of the aggregated posterior
    density-ratio KL into index-code mutual information (ICMI), total correlation
    (TC), and dimension-wise KL (DWKL). The density terms are estimated from each
    minibatch, so the experiment is intentionally kept outside the stable API.
    """

    scaling: ScalingMode = "standard"
    reconstruction_weight: float = 1.0
    icmi_weight: float = 1.0
    tc_weight: float = 6.0
    dwkl_weight: float = 1.0
    decomposition_warmup_epochs: int = 0

    def __post_init__(self) -> None:
        NeuralTrainingConfig.__post_init__(self)
        if self.batch_size < 2:
            raise ValueError("decomposed KL estimation requires batch_size >= 2")
        for name, value in (
            ("reconstruction_weight", self.reconstruction_weight),
            ("icmi_weight", self.icmi_weight),
            ("tc_weight", self.tc_weight),
            ("dwkl_weight", self.dwkl_weight),
        ):
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.decomposition_warmup_epochs < 0:
            raise ValueError("decomposition_warmup_epochs cannot be negative")


__all__ = ["TransformerVAEConfig", "DecomposedTransformerVAETrainingConfig"]
