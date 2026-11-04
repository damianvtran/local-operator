"""The file names a session directory holds, in a module EVERY reader can import.

WHY THIS IS NOT IN ``registry`` (where the other sidecar names live). A name that
another enumerating registry has to classify — ``network.sync.EXCLUDED_ENTRIES`` and
``session.retention._SIDECAR_NAMES`` both enumerate a session directory's top-level
files — must be importable by those modules AT IMPORT TIME, and both of them sit BELOW
``session.runtime.registry`` in the import graph: ``registry`` imports
``session.runtime.types``, and ``types`` imports ``session.retention``. Importing the
constant from ``registry`` inside ``retention`` is a circular import (measured, this
branch): ``ImportError: cannot import name 'HEARTBEAT_INTERVAL_S' from partially
initialized module 'local_operator.session.runtime.types'``. A stdlib-only leaf module
has no such edge, so the name is declared ONCE here and imported where it is needed.

WHAT BELONGS HERE. Only names whose identity must be shared by a WRITER and an
ENUMERATING reader that cannot import the writer's module. A name only its writer and
its reader use stays beside them (``registry.SIGNAL_RECEIPT_NAME`` and friends).
"""

from __future__ import annotations

#: The runtime's own record of terminating signals it REFUSED (see
#: ``registry.SIGNAL_REFUSALS_NAME``, which re-exports this and carries the rationale).
#:
#: Named here rather than beside ``SIGNAL_RECEIPT_NAME`` so
#: ``network.sync.EXCLUDED_ENTRIES`` and ``session.retention._SIDECAR_NAMES`` can
#: classify it without importing the runtime package: an unlisted top-level file in a
#: session directory makes ``sync.assert_complete`` refuse to MOVE the session, so this
#: file would turn a refusal into a session that cannot be moved until someone deleted
#: the evidence.
SIGNAL_REFUSALS_NAME = "runtime-signal-refusals.json"
