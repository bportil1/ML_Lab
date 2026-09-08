"""Experimental Linux-binary representation pipeline.

This module incubates ideas recovered from the legacy Linux Binary Identification
repository.  It is intentionally outside ML Lab's stable API.  Generic pieces
may later graduate into ``ml_lab.representation`` after their interfaces and
training behavior are validated.
"""

from .binary_data import bytes_to_rgb_array, bytes_to_patch_tokens, read_binary
from .pipeline import run

__all__ = ["bytes_to_rgb_array", "bytes_to_patch_tokens", "read_binary", "run"]
