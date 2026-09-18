"""The notification gate a rig must raise before it starts a real session.

WHY THIS MODULE EXISTS. A rig is a throwaway driver — a benchmark, an evidence
capture, a repro — that boots the real app or spawns a real
``local_operator.session.runtime.process`` against a scratch store. Several of
them seed ``hosting: test`` (the deterministic mock), whose only reply is
``Hello from the mock provider!``, and a composed notification body is a snippet
of the session's last assistant line. So a rig that ran a mock session put that
sentence on the operator's lock screen: 17 recorded banner attempts across
scratch stores in two days, all of them drive-by rigs. The product now refuses
this itself (``tui/notify.suppress_notifications_for_process`` fires at the mock
boundary and at spec adoption in ``model/configure.py``), which is the
load-bearing fix and the only one that reaches a rig written outside this
repository. What this module adds is the second line, for the rigs that run a
REAL provider: without it, one of those can still park a gate and announce it.

TWO ENTRY POINTS, because a rig reaches the desktop two ways:

* :func:`disable_notifications` — call it once, early, in a rig that boots the
  app or a runtime IN THIS PROCESS. It sets the switches in ``os.environ``,
  which is what every in-band and out-of-band notification leg reads.
* :data:`NO_NOTIFY_ENV` — splat (``**NO_NOTIFY_ENV``) into any child environment
  the rig builds. ``os.environ`` is inherited by a child only if the rig hands
  it over, and the rigs here filter the environment deliberately (dropping
  ``CMUX_*``/``LOP_*``), so a child needs the switch said again.

NOT IMPORT-TIME, unlike :mod:`scripts.probe_isolation`, and the difference is
the point: that module re-homes ``HOME`` and the config dir, which is right for
a capture and wrong for a rig pointed at a store it was told to use. A rig may
want only the gate, so the gate is a call.

The mapping is the same one ``tests/e2e/harness.py`` uses; the two trees cannot
import each other (``scripts`` is not a package the test suite may depend on at
runtime), so ``tests/unit/test_notification_isolation.py`` asserts they are
equal — one rule, declared twice, checked.
"""

from __future__ import annotations

import os

#: The switches that keep this process and its children away from the operator's
#: real desktop: the notification kill switch, and the launch rung that opens
#: the desktop app. Bit-for-bit the pair in ``tests/e2e/harness.NO_NOTIFY_ENV``.
NO_NOTIFY_ENV: dict[str, str] = {
    "LOCAL_OPERATOR_NO_NOTIFICATIONS": "1",
    "LOCAL_OPERATOR_NO_DESKTOP_LAUNCH": "1",
}


def disable_notifications() -> dict[str, str]:
    """Gate this process's notifications, and return the child's environment.

    Idempotent. The return value is :data:`NO_NOTIFY_ENV` copied, so a caller
    can ``env.update(disable_notifications())`` for a child it is about to
    spawn without a second spelling of the names.
    """
    os.environ.update(NO_NOTIFY_ENV)
    return dict(NO_NOTIFY_ENV)
