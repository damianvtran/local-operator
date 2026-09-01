"""Back-compat shim: the owned-session handle moved to the runtime package.

An "owned" session is one the harness runs itself rather than one a terminal
drives \u2014 built with the CLI's composition root and reachable only through the
control socket. The phone starts them today; wakes and background automations
are the same shape. It now lives in
:mod:`local_operator.session.runtime.owned`; this shim keeps the old import path
resolving.

New code should import from ``local_operator.session.runtime.owned``.
"""

from __future__ import annotations

from local_operator.session.runtime.owned import (  # noqa: F401  (re-exported)
    OwnedSessionHandle,
    spawn_owned_session,
)
