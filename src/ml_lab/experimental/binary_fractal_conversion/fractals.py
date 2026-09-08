from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def read_binary(path: str | Path) -> bytes:
    """Read a binary file without attaching any executable-format semantics."""
    return Path(path).read_bytes()


def _byte_array(data: Any) -> np.ndarray:
    if isinstance(data, np.ndarray):
        values = np.asarray(data, dtype=np.uint8).reshape(-1)
    elif isinstance(data, (bytes, bytearray, memoryview)):
        values = np.frombuffer(bytes(data), dtype=np.uint8)
    else:
        values = np.asarray(list(data), dtype=np.uint8).reshape(-1)
    if values.size == 0:
        raise ValueError("binary data must not be empty")
    return values


def binary_mandelbrot_image(
    data: Any,
    *,
    width: int = 128,
    height: int = 128,
    base_iter: int = 64,
    mode: str = "complete",
) -> np.ndarray:
    """Create a byte-conditioned Mandelbrot representation.

    ``mode='complete'`` maps byte values into iteration depth and RGB channels.
    ``mode='structural'`` additionally lets global byte statistics skew the view
    and locally perturb coordinates.  Both modes are cleaned descendants of the
    legacy Binary Image Analysis Project's byte-driven Mandelbrot routines.
    """
    if width < 2 or height < 2:
        raise ValueError("width and height must be >= 2")
    if base_iter < 1:
        raise ValueError("base_iter must be >= 1")
    if mode not in {"complete", "structural", "escape_rate"}:
        raise ValueError("mode must be one of: complete, structural, escape_rate")

    byte_stream = _byte_array(data)
    total = float(byte_stream.astype(np.float64).sum())
    if mode == "structural":
        skew_x = np.sin(total % 256.0) * 0.5
        skew_y = np.cos(total % 256.0) * 0.5
        x_min, x_max = -2.0 + skew_x, 1.0 - skew_x
        y_min, y_max = -1.5 + skew_y, 1.5 - skew_y
        tile_pattern = (total % 256.0) / 255.0
    else:
        x_min, x_max = -1.625, 0.125
        y_span = (x_max - x_min) * height / width
        y_min, y_max = -y_span / 2.0, y_span / 2.0
        tile_pattern = 0.0

    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    n_bytes = int(byte_stream.size)

    for y in range(height):
        for x in range(width):
            zx = x_min + (x / max(width - 1, 1)) * (x_max - x_min)
            zy = y_min + (y / max(height - 1, 1)) * (y_max - y_min)
            byte_mod = int(byte_stream[(x + y) % n_bytes])

            if mode == "structural":
                zx += np.sin(int(byte_stream[(x * y) % n_bytes]) / 255.0 * np.pi) * 0.1
                zy += np.cos(byte_mod / 255.0 * np.pi) * 0.1

            c = complex(zx, zy)
            z = 0j
            dynamic_iter = int(base_iter + (byte_mod % 50))
            last_i = 0
            for i in range(dynamic_iter):
                last_i = i
                if abs(z) > 2.0:
                    break
                z = z * z + c

            depth = last_i / max(dynamic_iter, 1)
            if mode == "escape_rate":
                value = int(round(255.0 * depth))
                pixels[y, x] = (value, value, value)
                continue

            wave_sin = np.sin(byte_mod / 255.0 * np.pi) ** 2
            wave_cos = np.cos(byte_mod / 255.0 * np.pi) ** 2
            direct_r = int(byte_stream[(x * y) % n_bytes])
            direct_g = int(byte_stream[(x + y) % n_bytes])
            direct_b = int(byte_stream[(x - y) % n_bytes])
            tile = int(255.0 * tile_pattern)

            if mode == "structural":
                red = (direct_r + int(255 * depth) + int(255 * wave_sin) + tile) % 256
                green = (direct_g + int(255 * depth) + int(255 * wave_cos) + tile) % 256
                blue = (direct_b + int(255 * depth) + int(255 * (1.0 - wave_sin)) + tile) % 256
            else:
                red = (direct_r + int(255 * depth) + int(255 * wave_sin)) % 256
                green = (direct_g + int(255 * depth) + int(255 * wave_sin)) % 256
                blue = (direct_b + int(255 * depth) + int(255 * wave_sin)) % 256

            pixels[y, x] = (red, green, blue)

    return pixels


def binary_ifs_image(
    data: Any,
    *,
    width: int = 256,
    height: int = 256,
    iterations: int = 20_000,
) -> np.ndarray:
    """Convert binary bytes into a deterministic two-map IFS image.

    This preserves the legacy project's bit-driven IFS idea while fixing its
    duplicate recursive ``generate_ifs_fractal`` definition.  Bits are consumed
    cyclically and deterministically, so equal byte streams produce equal arrays.
    """
    if width < 2 or height < 2:
        raise ValueError("width and height must be >= 2")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")

    byte_stream = _byte_array(data)
    bits = np.unpackbits(byte_stream)
    if bits.size == 0:
        raise ValueError("binary data must contain at least one bit")

    image = np.full((height, width), 255, dtype=np.uint8)
    x = 0.0
    y = 0.0
    for i in range(iterations):
        if int(bits[i % bits.size]) == 0:
            x, y = 0.5 * x, 0.5 * y
        else:
            x, y = 0.5 * x + 0.5, 0.5 * y + 0.5
        px = min(width - 1, max(0, int(round(x * (width - 1)))))
        py = min(height - 1, max(0, int(round(y * (height - 1)))))
        image[py, px] = 0
    return image
