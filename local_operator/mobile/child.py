"""Back-compat shim: the detached child process moved to the runtime package.

The implementation is now
:mod:`local_operator.session.runtime.process`. This shim matters more than the
others because the module path is a **cross-process spawn contract**: a daemon
of one version spawns ``python -m <path>`` and the child that answers may be a
binary of another version. Keeping ``python -m local_operator.mobile.child``
runnable means an upgrade window cannot strand a daemon that cannot start a
session \u2014 hence the ``__main__`` block below, which the other shims do not
need.

New code should use ``local_operator.session.runtime.process``.
"""

from __future__ import annotations

import sys

from local_operator.session.runtime.process import (  # noqa: F401  (re-exported)
    DEFAULT_GRACE_S,
    REAP_CHECK_S,
    _clean_exit,
    _grace_seconds,
    _reaper,
    _should_exit,
    amain,
    main,
)

if __name__ == "__main__":
    sys.exit(main())
