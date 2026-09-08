from __future__ import annotations

from typing import Any

import numpy as np

from .fractals import binary_ifs_image, binary_mandelbrot_image, read_binary


def _resolve_binary(*, data: Any = None, path: str | None = None) -> Any:
    if data is not None and path is not None:
        raise ValueError("provide either data or path, not both")
    if path is not None:
        return read_binary(path)
    if data is None:
        raise ValueError("binary data or path is required")
    return data


def _array_result(experiment_mode: str, array: np.ndarray) -> dict[str, Any]:
    return {
        "experiment": "binary_fractal_conversion",
        "mode": experiment_mode,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "min": int(array.min()),
        "max": int(array.max()),
        "array": array.tolist(),
    }


def run(
    *,
    mode: str = "describe",
    data: Any = None,
    path: str | None = None,
    width: int | None = None,
    height: int | None = None,
    base_iter: int = 64,
    iterations: int = 20_000,
    fractal_mode: str = "complete",
) -> dict[str, Any]:
    """Experimental registry entrypoint for binary-to-fractal conversion."""
    if mode == "describe":
        return {
            "experiment": "binary_fractal_conversion",
            "status": "experimental",
            "source_project": "Binary_Image_Analysis_Project",
            "available_modes": ["describe", "mandelbrot", "ifs"],
            "mandelbrot_modes": ["complete", "structural", "escape_rate"],
            "outputs": "NumPy-compatible image arrays serialized as nested lists",
            "warnings": [
                "This is an experimental representation transform, not a malware classifier.",
                "Legacy plotting, RBM, optimizer, and import-time demo code were intentionally excluded.",
            ],
        }

    raw = _resolve_binary(data=data, path=path)
    if mode == "mandelbrot":
        image = binary_mandelbrot_image(
            raw,
            width=width or 128,
            height=height or 128,
            base_iter=base_iter,
            mode=fractal_mode,
        )
        result = _array_result(mode, image)
        result["fractal_mode"] = fractal_mode
        return result

    if mode == "ifs":
        image = binary_ifs_image(
            raw,
            width=width or 256,
            height=height or 256,
            iterations=iterations,
        )
        return _array_result(mode, image)

    raise ValueError(f"unknown binary_fractal_conversion mode: {mode}")
