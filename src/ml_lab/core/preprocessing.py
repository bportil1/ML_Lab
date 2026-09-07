from __future__ import annotations

from typing import Literal

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from .specs import EstimatorSpec

ScalingMode = Literal["auto", "none", "standard", "minmax", "robust"]


def make_preprocessor(spec: EstimatorSpec, scaling: ScalingMode = "auto"):
    mode = spec.preprocess if scaling == "auto" else scaling
    if mode == "none":
        return "passthrough"
    if mode == "standard":
        return StandardScaler()
    if mode == "minmax":
        return MinMaxScaler()
    if mode == "robust":
        return RobustScaler()
    raise ValueError(f"unsupported scaling mode: {mode}")
