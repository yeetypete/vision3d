"""Centralized Rerun import with a friendly install hint.

Importing ``rerun`` from here (rather than each module guarding the import
itself) keeps the single ``vision3d[viz]`` install message in one place.
"""

try:
    import rerun as rr
    import rerun.blueprint as rrb
except ImportError as e:
    msg = "rerun-sdk is required for visualization. Install with: pip install vision3d[viz]"
    raise ImportError(msg) from e

__all__ = ["rr", "rrb"]
