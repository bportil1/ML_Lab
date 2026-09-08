"""Experimental binary-to-fractal representations.

Recovered and cleaned from the legacy Binary Image Analysis Project.  The
module intentionally returns numerical image arrays rather than owning any
plotting or file-writing UI.
"""

from .fractals import binary_ifs_image, binary_mandelbrot_image, read_binary
from .pipeline import run

__all__ = ["binary_ifs_image", "binary_mandelbrot_image", "read_binary", "run"]
