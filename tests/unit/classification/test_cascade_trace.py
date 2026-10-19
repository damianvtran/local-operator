"""The resolution trace: which preferred legs a decision skipped, and why.

The measured defect was a SILENT fall-through: a Radient login held on a mesh
owner left the node classifying through OpenRouter with nothing on the decision
or the spend row saying a preferred leg had been skipped. ``resolve_vendor``
keeps its old contract; ``resolve_vendor_with_trace`` returns the same leg plus
the skipped rows, classified from LOCAL placement only (never a second borrow).
"""

from __future__ import annotations

import pytest

from local_operator.classification.cascade import (
    resolve_vendor,
    resolve_vendor_with_trace,
)
from tests.unit.classification.support import LegBehaviour, leg_class, store_row

pytestmark = pytest.mark.asyncio

ALL_KEYS = {
    "radient": "RADIENT_API_KEY",
    "typesafe": "TYPESAFE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def arm(root, *legs: str) -> None:
    for leg in legs:
        store_row(root, ALL_KEYS[leg], f"{leg}-key")


async def test_a_zero_peer_skip_is_plain_absence(bare_manager) -> None:
    arm(bare_manager, "openrouter")

    resolution = await resolve_vendor_with_trace(bare_manager)

    assert resolution.vendor is not None and resolution.vendor.name == "openrouter"
    assert resolution.skipped == (
        {"vendor": "radient", "reason": "no_credential_here"},
        {"vendor": "typesafe", "reason": "no_credential_here"},
    )


async def test_a_lent_login_left_unborrowed_is_named_not_synced(holder_root, install_legs) -> None:
    """The measured case: the row exists on the owner, and this device could not borrow.

    The Radient leg is a stub that reports no credential, exactly the state the
    node was in; the placement document is real, so the reason comes from the
    same holder predicate production uses. No borrow is attempted here — the
    trace's classification must never dial.
    """
    install_legs(radient={"credential": False}, typesafe={"credential": True})

    resolution = await resolve_vendor_with_trace(holder_root)

    assert resolution.vendor is not None and resolution.vendor.name == "typesafe"
    assert resolution.skipped == ({"vendor": "radient", "reason": "not_synced"},)


async def test_the_first_leg_resolving_costs_no_trace_probe(bare_manager, monkeypatch) -> None:
    """A resolved leg must never pay for the skipped-leg classification."""
    arm(bare_manager, "radient")
    from local_operator.classification import cascade

    def explode(*args, **kwargs):
        raise AssertionError("a resolved first leg must not be probed")

    monkeypatch.setattr(cascade, "_skip_row", explode)

    resolution = await resolve_vendor_with_trace(bare_manager)

    assert resolution.vendor is not None and resolution.vendor.name == "radient"
    assert resolution.skipped == ()


async def test_a_leg_whose_resolution_raises_is_recorded(bare_manager, install_legs) -> None:
    install_legs(
        radient={"credential_raises": True},
        typesafe={"credential": True},
    )

    resolution = await resolve_vendor_with_trace(bare_manager)

    assert resolution.vendor is not None and resolution.vendor.name == "typesafe"
    assert resolution.skipped == ({"vendor": "radient", "reason": "resolution_failed"},)


async def test_a_broker_refusal_keeps_its_code(bare_manager, install_legs, monkeypatch) -> None:
    """A refusal is a nameable skip: the broker's code rides in ``detail``."""
    from local_operator.classification import vendors
    from local_operator.network.types import MeshRefusal

    install_legs(typesafe={"credential": True})
    refusing = LegBehaviour(name="radient")

    class _Refusing(leg_class(refusing)):
        async def credential(self, manager):
            raise MeshRefusal("not_a_holder", "no holder row here")

    replacements = dict(vendors.VENDOR_CLASSES)
    replacements["radient"] = _Refusing
    monkeypatch.setattr(vendors, "VENDOR_CLASSES", replacements)

    resolution = await resolve_vendor_with_trace(bare_manager)

    assert resolution.skipped == (
        {"vendor": "radient", "reason": "borrow_failed", "detail": "not_a_holder"},
    )


async def test_resolve_vendor_stays_the_same_value(bare_manager) -> None:
    """The thin wrapper exists so existing callers are byte-compatible.

    Each call builds its own leg instance (the memo lives on that instance), so
    the contract is the same RESOLVED LEG, not the same object.
    """
    arm(bare_manager, "typesafe")

    resolution = await resolve_vendor_with_trace(bare_manager)
    vendor = await resolve_vendor(bare_manager)

    assert resolution.vendor is not None and vendor is not None
    assert vendor.name == resolution.vendor.name
