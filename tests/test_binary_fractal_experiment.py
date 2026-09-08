from __future__ import annotations

import numpy as np

from ml_lab.experimental import get_manifest, run_experiment
from ml_lab.experimental.binary_fractal_conversion import binary_ifs_image, binary_mandelbrot_image


def test_binary_fractal_manifest_is_registered():
    manifest = get_manifest("binary_fractal_conversion")
    assert "binary_to_fractal" in manifest.capabilities
    assert manifest.status == "experimental"


def test_mandelbrot_conversion_is_deterministic_and_byte_conditioned():
    a = binary_mandelbrot_image(b"abcdef", width=12, height=10, base_iter=8, mode="complete")
    b = binary_mandelbrot_image(b"abcdef", width=12, height=10, base_iter=8, mode="complete")
    c = binary_mandelbrot_image(b"uvwxyz", width=12, height=10, base_iter=8, mode="complete")
    assert a.shape == (10, 12, 3)
    assert a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_ifs_conversion_is_deterministic():
    a = binary_ifs_image(b"\x00\xff", width=16, height=16, iterations=100)
    b = binary_ifs_image(b"\x00\xff", width=16, height=16, iterations=100)
    assert a.shape == (16, 16)
    assert set(np.unique(a)).issubset({0, 255})
    np.testing.assert_array_equal(a, b)


def test_registry_entrypoint_returns_serializable_fractal():
    result = run_experiment(
        "binary_fractal_conversion",
        mode="mandelbrot",
        data=[1, 2, 3, 4, 5],
        width=8,
        height=6,
        base_iter=4,
        fractal_mode="structural",
    )
    assert result["shape"] == [6, 8, 3]
    assert result["fractal_mode"] == "structural"
    assert isinstance(result["array"], list)
