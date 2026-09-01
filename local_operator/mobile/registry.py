"""Back-compat shim: discovery records moved to the session runtime package.

The record directory, its staged-write publication and the ``live`` /
``wedged`` / ``stale`` scan are how ANY session becomes findable \u2014 ``lop
sessions``, ``lop send`` and attach all read them, not just the phone. They
now live in :mod:`local_operator.session.runtime.registry`; this shim keeps the
old import path resolving.

New code should import from ``local_operator.session.runtime.registry``.
"""

from __future__ import annotations

from local_operator.session.runtime.registry import (  # noqa: F401  (re-exported)
    RecordPublisher,
    SessionRecord,
    pid_alive,
    publish,
    run_dir,
    scan,
    unpublish,
)
