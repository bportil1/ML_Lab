from __future__ import annotations

from pathlib import Path

import numpy as np


def _coerce_bytes(data: bytes | bytearray | memoryview | list[int] | tuple[int, ...]) -> bytes:
    if isinstance(data, bytes):
        return data
    if isinstance(data, (bytearray, memoryview)):
        return bytes(data)
    try:
        values = [int(value) for value in data]
    except (TypeError, ValueError) as exc:
        raise TypeError("binary data must be bytes-like or a sequence of integers") from exc
    if any(value < 0 or value > 255 for value in values):
        raise ValueError("binary byte values must be in the range 0..255")
    return bytes(values)


def read_binary(path: str | Path) -> bytes:
    """Read a binary file without interpreting its format."""
    return Path(path).read_bytes()


def bytes_to_rgb_array(
    data: bytes | bytearray | memoryview | list[int] | tuple[int, ...],
    *,
    size: int = 256,
) -> np.ndarray:
    """Pad/truncate a byte stream into an ``(size, size, 3)`` uint8 array.

    This preserves the legacy project's byte-to-RGB representation idea while
    keeping image-file I/O and visualization out of the experimental core.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    raw = _coerce_bytes(data)
    required = size * size * 3
    vector = np.zeros(required, dtype=np.uint8)
    if raw:
        source = np.frombuffer(raw[:required], dtype=np.uint8)
        vector[: source.size] = source
    return vector.reshape(size, size, 3)


def bytes_to_patch_tokens(
    data: bytes | bytearray | memoryview | list[int] | tuple[int, ...],
    *,
    image_size: int = 256,
    patch_size: int = 32,
    normalize: bool = True,
) -> np.ndarray:
    """Convert a byte stream into flattened RGB patch tokens.

    Returns ``(num_patches, patch_size * patch_size * 3)``.  This is the
    reusable preprocessing boundary that the legacy Transformer prototype was
    implicitly performing inside the model.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")

    image = bytes_to_rgb_array(data, size=image_size)
    grid = image.reshape(
        image_size // patch_size,
        patch_size,
        image_size // patch_size,
        patch_size,
        3,
    )
    patches = grid.transpose(0, 2, 1, 3, 4).reshape(-1, patch_size * patch_size * 3)
    if normalize:
        return patches.astype(np.float32) / 255.0
    return patches
