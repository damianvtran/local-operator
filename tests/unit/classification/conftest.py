"""Fixtures shared by the classification suite.

They live here rather than in ``support.py`` because pytest only collects
fixtures from ``conftest.py`` and plugins — a helper module's fixtures are
invisible to every test file that does not import them by name, which shows up
as "fixture not found" at collection time rather than as a failure of the thing
under test.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.classification.vendors import VENDOR_CLASSES
from local_operator.credentials import CredentialManager
from tests.unit.classification.support import (
    TEST_KEY,
    LegBehaviour,
    leg_class,
    store_row,
)


@pytest.fixture
def bare_manager(tmp_path) -> CredentialManager:
    """A credential manager with NOTHING in it — for the credential-tier tests.

    Real, not a stub: a leg resolves the provider-class store row first and then
    the process environment, and the fallback tiers (``JEV_API_KEY``,
    ``OPENROUTER_API_KEY_DEV``, ``RADIENT_API_KEY``) are precisely what those
    tests are about. ``tests/conftest.py`` clears those names from the ambient
    environment, so an empty store really means "no credential" here.
    """
    return CredentialManager.readonly(tmp_path)


@pytest.fixture
def manager(bare_manager: CredentialManager) -> CredentialManager:
    """The same manager, armed with every leg's primary credential.

    The default for tests about the wire shape: a leg with no credential refuses
    before it sends anything (``kind="auth"``), so a test about request bodies,
    status mapping or answer parsing would otherwise be testing that refusal
    instead of the thing it named. The keys are written as PROVIDER-CLASS STORE
    ROWS (PR2a), which is the one tier a static key can now live in besides an
    exported variable.
    """
    for key in ("RADIENT_API_KEY", "TYPESAFE_API_KEY", "OPENROUTER_API_KEY"):
        store_row(bare_manager, key, TEST_KEY)
    return bare_manager


@pytest.fixture
def install_legs(monkeypatch):
    """Install stub legs for the named cascade legs, returning their behaviours.

    The single patch point is ``vendors.VENDOR_CLASSES``, which
    ``vendors.build_vendor`` reads at call time — so patching the mapping covers
    the cascade AND the service, including the leg instances they build for
    themselves. Patching ``build_vendor`` in two module namespaces instead would
    work until a third caller appeared.

    Any name not passed keeps its real class, which will fail loudly if a test
    reaches for a live endpoint it did not intend to.
    """

    def _install(**behaviours: Any) -> dict[str, LegBehaviour]:
        installed: dict[str, LegBehaviour] = {}
        replacements = dict(VENDOR_CLASSES)
        for name, spec in behaviours.items():
            behaviour = spec if isinstance(spec, LegBehaviour) else LegBehaviour(name=name, **spec)
            behaviour.name = name
            installed[name] = behaviour
            replacements[name] = leg_class(behaviour)
        monkeypatch.setattr(
            "local_operator.classification.vendors.VENDOR_CLASSES", replacements, raising=True
        )
        return installed

    return _install
