"""Per-write temp names for the Library's atomic cache writers.

PR #4501 introduced :func:`unique_temp` here for the Library's thumbnail, waveform
and index writers; the same pid-only-temp race existed across the package, so
the implementation now lives in :mod:`local_operator.atomic_files` and this
module only re-exports it -- ONE implementation, so a fix to the naming lands
everywhere at once. See that module for the why and the constraints.
"""

from __future__ import annotations

from local_operator.atomic_files import unique_temp

__all__ = ["unique_temp"]
