"""No Team Bridge test may reach the real launchd.

D2a makes ordinary enrolment commits, ``team_bridge.enabled`` writes and
app-start boundaries run the connector's service lifecycle. That is intentional
in production and dangerous in a suite: ``launchctl`` has no sandbox and always
addresses the calling user's live domain, even under the suite's redirected
HOME.

Two independent layers, and this file owns the second:

* the REAL guards (``launchd.shared_label_refusal``, ``launchd.is_own_plist``)
  already refuse every store the suite can construct, because the suite's config
  roots and ``Path.home()`` are never the passwd home's default root. Their own
  behaviour is pinned in ``test_launchd.py`` and the wakes installer tests;
* this fixture closes the remaining hole, a test that opens those guards: the
  connector's ``_launchctl`` seam becomes a hard failure, so an opened guard
  still cannot reach launchd.

Deliberately scoped to the connector's seam rather than ``subprocess`` itself:
several tests here spawn a real interpreter on purpose (the import-bound pin,
the evidence rigs), and blocking those would test this fixture instead of them.
Tests ABOUT the lifecycle replace ``install._launchctl`` themselves.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _connector_never_reaches_launchd(monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.team_bridge.connector import install

    def forbidden(*args: str):
        raise AssertionError(
            "a Team Bridge test reached the connector's launchctl seam; inject "
            "install._launchctl or leave the shared-label guard closed"
        )

    # ``raising=False`` so this fixture is not itself the failure on a base
    # source that has no seam: the lifecycle tests must fail on their own
    # assertions instead. Against a real launcher the name is what we guard.
    monkeypatch.setattr(install, "_launchctl", forbidden, raising=False)
