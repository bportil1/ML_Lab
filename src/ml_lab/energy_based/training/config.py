from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class BestModelSelectionConfig:
    """Relative best-model selection for iterative energy-based training."""

    enabled: bool = False
    metric: str = "energy_gap_abs"
    patience: int = 25
    min_relative_improvement: float = 1e-3
    warmup_epochs: int = 0
    restore_best: bool = True

    def __post_init__(self) -> None:
        if self.metric not in {"energy_gap_abs", "reconstruction_error", "cd_objective"}:
            raise ValueError(
                "model-selection metric must be energy_gap_abs, reconstruction_error, or cd_objective"
            )
        if int(self.patience) < 1:
            raise ValueError("model-selection patience must be >= 1")
        if float(self.min_relative_improvement) < 0:
            raise ValueError("model-selection min_relative_improvement must be >= 0")
        if int(self.warmup_epochs) < 0:
            raise ValueError("model-selection warmup_epochs must be >= 0")


@dataclass(frozen=True)
class AdaptiveMomentumConfig:
    enabled: bool = False
    improve_threshold: float = -0.05
    stable_threshold: float = 0.02
    improve_increment: float = 0.02
    improve_cap: float = 0.95
    stable_increment: float = 0.005
    stable_cap: float = 0.90
    failure_scale: float = 0.8
    minimum: float = 0.0

    def __post_init__(self) -> None:
        if float(self.stable_threshold) < 0:
            raise ValueError("stable_threshold must be >= 0")
        if not 0 <= float(self.improve_cap) < 1:
            raise ValueError("improve_cap must satisfy 0 <= value < 1")
        if not 0 <= float(self.stable_cap) < 1:
            raise ValueError("stable_cap must satisfy 0 <= value < 1")
        if not 0 < float(self.failure_scale) <= 1:
            raise ValueError("failure_scale must satisfy 0 < value <= 1")
        if not 0 <= float(self.minimum) < 1:
            raise ValueError("minimum must satisfy 0 <= value < 1")


@dataclass(frozen=True)
class DistributionMonitoringConfig:
    enabled: bool = True
    interval: int = 25
    sample_size: int = 128
    gibbs_steps: int = 10
    sinkhorn_enabled: bool = True
    sinkhorn_p: int = 2
    sinkhorn_regularization: float = 0.1
    sinkhorn_iterations: int = 50

    def __post_init__(self) -> None:
        if int(self.interval) < 1:
            raise ValueError("distribution monitoring interval must be >= 1")
        if int(self.sample_size) < 2:
            raise ValueError("distribution monitoring sample_size must be >= 2")
        if int(self.gibbs_steps) < 1:
            raise ValueError("distribution monitoring gibbs_steps must be >= 1")
        if int(self.sinkhorn_p) not in {1, 2}:
            raise ValueError("sinkhorn_p must be 1 or 2")
        if float(self.sinkhorn_regularization) <= 0:
            raise ValueError("sinkhorn_regularization must be > 0")
        if int(self.sinkhorn_iterations) < 1:
            raise ValueError("sinkhorn_iterations must be >= 1")


@dataclass(frozen=True)
class PartitionMonitoringConfig:
    """Scheduled normalization diagnostics.

    ``mean(-E(v))`` is available every epoch and needs no partition function.
    Normalized log-likelihood requires an estimate of ``log Z`` and therefore
    runs only on this explicit schedule.
    """

    enabled: bool = False
    schedule: str = "hybrid"
    interval: int = 250
    estimator: str = "auto"
    sample_count: int = 512
    exact_bernoulli_visible_limit: int = 16
    exact_chunk_size: int = 4096
    seed: int = 314159
    bernoulli_proposal_probability: float = 0.5
    gaussian_proposal_std: float = 1.0
    student_t_proposal_df: float = 3.0

    def __post_init__(self) -> None:
        if self.schedule not in {"hybrid", "interval", "logarithmic", "final_only"}:
            raise ValueError("partition schedule must be hybrid, interval, logarithmic, or final_only")
        if int(self.interval) < 1:
            raise ValueError("partition interval must be >= 1")
        if self.estimator not in {"auto", "importance", "exact_bernoulli"}:
            raise ValueError("partition estimator must be auto, importance, or exact_bernoulli")
        if int(self.sample_count) < 2:
            raise ValueError("partition sample_count must be >= 2")
        if int(self.exact_bernoulli_visible_limit) < 1:
            raise ValueError("exact_bernoulli_visible_limit must be >= 1")
        if int(self.exact_chunk_size) < 1:
            raise ValueError("exact_chunk_size must be >= 1")
        if not 0 < float(self.bernoulli_proposal_probability) < 1:
            raise ValueError("bernoulli_proposal_probability must be in (0, 1)")
        if float(self.gaussian_proposal_std) <= 0:
            raise ValueError("gaussian_proposal_std must be > 0")
        if float(self.student_t_proposal_df) <= 0:
            raise ValueError("student_t_proposal_df must be > 0")


@dataclass(frozen=True)
class CDKTrainingConfig:
    gibbs_steps: int = 1
    num_chains: int = 1
    epochs: int = 100
    learning_rate: float = 1e-2
    batch_size: int = 32
    persistent: bool = False
    momentum: float = 0.5
    weight_decay: float = 0.0
    clamp_visible: bool = True
    shuffle_batches: bool = True
    drop_last_batch: bool = False
    energy_sample_size: int = 128
    max_weight_row_norm: float = 5.0
    stop_on_nonfinite: bool = True
    random_state: int = 42
    deterministic: bool = True
    device: str = "auto"
    model_selection: BestModelSelectionConfig = field(default_factory=BestModelSelectionConfig)
    adaptive_momentum: AdaptiveMomentumConfig = field(default_factory=AdaptiveMomentumConfig)
    distribution_monitoring: DistributionMonitoringConfig = field(default_factory=DistributionMonitoringConfig)
    partition_monitoring: PartitionMonitoringConfig = field(default_factory=PartitionMonitoringConfig)

    def __post_init__(self) -> None:
        if int(self.gibbs_steps) < 1:
            raise ValueError("gibbs_steps must be >= 1")
        if int(self.num_chains) < 1:
            raise ValueError("num_chains must be >= 1")
        if int(self.epochs) < 1:
            raise ValueError("epochs must be >= 1")
        if float(self.learning_rate) <= 0:
            raise ValueError("learning_rate must be > 0")
        if int(self.batch_size) < 1:
            raise ValueError("batch_size must be >= 1")
        if not 0 <= float(self.momentum) < 1:
            raise ValueError("momentum must satisfy 0 <= value < 1")
        if float(self.weight_decay) < 0:
            raise ValueError("weight_decay must be >= 0")
        if int(self.energy_sample_size) < 1:
            raise ValueError("energy_sample_size must be >= 1")
        if float(self.max_weight_row_norm) <= 0:
            raise ValueError("max_weight_row_norm must be > 0")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "CDKTrainingConfig":
        values = dict(values or {})
        nested = {
            "model_selection": (BestModelSelectionConfig, values.pop("model_selection", None)),
            "adaptive_momentum": (AdaptiveMomentumConfig, values.pop("adaptive_momentum", None)),
            "distribution_monitoring": (DistributionMonitoringConfig, values.pop("distribution_monitoring", None)),
            "partition_monitoring": (PartitionMonitoringConfig, values.pop("partition_monitoring", None)),
        }
        for key, (config_type, supplied) in nested.items():
            if supplied is not None and not isinstance(supplied, config_type):
                supplied = config_type(**dict(supplied))
            if supplied is not None:
                values[key] = supplied
        return cls(**values)

    def to_record(self) -> dict[str, Any]:
        return asdict(self)
