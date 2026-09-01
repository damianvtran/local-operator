"""Back-compat shim: the registrant moved to the session runtime package.

This module's contents now live in
:mod:`local_operator.session.runtime.server`, and ``Registrant`` is an alias of
:class:`~local_operator.session.runtime.server.RuntimeServer` \u2014 nothing about a
control socket, a discovery record or a session handle was ever phone-specific
(see that package's ``__init__`` for the reasoning). The shim exists so
imports outside this tree, and any process mid-upgrade, keep resolving.

New code should import from ``local_operator.session.runtime.server``.
"""

from __future__ import annotations

from local_operator.session.runtime.server import (  # noqa: F401  (re-exported)
    Registrant,
    RuntimeServer,
    SessionHandle,
    image_blocks,
)
