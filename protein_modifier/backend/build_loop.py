"""
DEPRECATED: This module is superseded by build_idr.py.

Use build_idr.build_loop() and build_idr.build_loop_coordinates() instead.
This file is kept only for reference and will be removed in a future version.
"""

import warnings

from protein_modifier.backend.build_idr import build_loop_coordinates as _build_loop_coordinates


def build_loop_coordinates(*args, **kwargs):
    """Deprecated. Use protein_modifier.backend.build_idr.build_loop_coordinates instead."""
    warnings.warn(
        "build_loop.build_loop_coordinates is deprecated. "
        "Use build_idr.build_loop_coordinates instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_loop_coordinates(*args, **kwargs)

