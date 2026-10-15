"""Team Bridge connector (H4): the always-on delivery substrate.

IMPORT BOUND — deliberate, and pinned by tests (the wake supervisor's
import-graph pin is the precedent). Module scope in this package imports ONLY
stdlib plus the bridge stack (``client``/``pull``/``store``/``types``) and the
ONE light constant module ``harness.message_types`` (no session/model graph),
so ``python -m local_operator.team_bridge.connector`` boots without the
session,
runtime, model or provider graphs. Everything heavier — the screening
composition (which resolves a provider selection), the runtime launch
machinery, the local delivery primitive — is imported LAZILY inside the
function that needs it, each with its own comment, so a connector process at
idle holds no model/session modules at all.

LIFECYCLE. One connector per account per machine, enforced by a machine-wide
singleton lock (:mod:`local_operator.team_bridge.connector.lock` — same
refuse-never-degrade semantics as ``wakes/lock.py``). The service unit is a
NEW dedicated label rendered by :mod:`local_operator.team_bridge.connector.install`
REUSING the wake supervisor's install mechanics; the wake supervisor's own
plist/label is never touched. Rendering is pure — nothing in this package
installs, bootstraps or otherwise touches the operator's machine.
"""

from local_operator.team_bridge.connector.lock import ConnectorLock, ConnectorLockBusy

__all__ = ["ConnectorLock", "ConnectorLockBusy"]
