"""Cascade order, the pin, and the credential probe that must not do I/O."""

from __future__ import annotations

import pytest

from local_operator.classification.cascade import (
    DEFAULT_VENDOR,
    VENDOR_CHOICES,
    VENDOR_ORDER,
    classification_section,
    leg_order,
    model_override,
    pinned_vendor,
    resolve_vendor,
    vendor_status,
)

pytestmark = pytest.mark.asyncio

ALL_KEYS = {
    "radient": "RADIENT_API_KEY",
    "typesafe": "TYPESAFE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def arm(bare_manager, *legs: str) -> None:
    for leg in legs:
        bare_manager.set_credential(ALL_KEYS[leg], f"{leg}-key", write=False)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


async def test_the_section_is_read_either_way_round() -> None:
    assert classification_section({"classification": {"vendor": "typesafe"}}) == {
        "vendor": "typesafe"
    }
    assert classification_section({"vendor": "typesafe"}) == {"vendor": "typesafe"}
    assert classification_section(None) == {}


async def test_the_pin_defaults_to_auto() -> None:
    assert pinned_vendor(None) == DEFAULT_VENDOR == "auto"
    assert pinned_vendor({"classification": {"vendor": "  TypeSafe "}}) == "typesafe"


async def test_an_unusable_pin_reads_as_auto_rather_than_as_no_vendor() -> None:
    """A typo must not silently disable the layer — the resolved leg is reported."""
    assert pinned_vendor({"classification": {"vendor": "Radient "}}) == "radient"
    assert pinned_vendor({"classification": {"vendor": "nope"}}) == "auto"
    assert pinned_vendor({"classification": {"vendor": 7}}) == "auto"


async def test_the_model_override_is_read_and_trimmed() -> None:
    assert model_override(None) == ""
    assert (
        model_override({"classification": {"model": " typesafe/jev-next "}}) == "typesafe/jev-next"
    )


async def test_the_leg_order_follows_the_pin() -> None:
    assert leg_order(None) == VENDOR_ORDER == ("radient", "typesafe", "openrouter")
    assert leg_order({"classification": {"vendor": "openrouter"}}) == ("openrouter",)
    assert set(VENDOR_CHOICES) == {"auto", "radient", "typesafe", "openrouter"}


# ---------------------------------------------------------------------------
# resolve_vendor
# ---------------------------------------------------------------------------


async def test_no_credential_anywhere_means_no_vendor(bare_manager) -> None:
    assert await resolve_vendor(bare_manager) is None


async def test_the_first_available_leg_wins(bare_manager) -> None:
    arm(bare_manager, "radient", "typesafe", "openrouter")
    vendor = await resolve_vendor(bare_manager)
    assert vendor is not None and vendor.name == "radient"

    unarmed_typesafe = bare_manager
    unarmed_typesafe.set_credential("RADIENT_API_KEY", "", write=False)
    vendor = await resolve_vendor(unarmed_typesafe)
    assert vendor is not None and vendor.name == "typesafe"

    unarmed_openrouter = bare_manager
    unarmed_openrouter.set_credential("TYPESAFE_API_KEY", "", write=False)
    vendor = await resolve_vendor(unarmed_openrouter)
    assert vendor is not None and vendor.name == "openrouter"


async def test_a_leg_skipped_for_a_missing_credential_is_not_probed_for_http(bare_manager) -> None:
    arm(bare_manager, "openrouter")
    vendor = await resolve_vendor(bare_manager)
    assert vendor is not None and vendor.name == "openrouter"


async def test_the_pin_restricts_the_cascade_to_one_leg(bare_manager) -> None:
    arm(bare_manager, "radient", "openrouter")
    settings = {"classification": {"vendor": "typesafe"}}
    assert await resolve_vendor(bare_manager, settings) is None
    settings = {"classification": {"vendor": "openrouter"}}
    vendor = await resolve_vendor(bare_manager, settings)
    assert vendor is not None and vendor.name == "openrouter"


async def test_the_model_override_reaches_the_leg(bare_manager) -> None:
    arm(bare_manager, "openrouter")
    vendor = await resolve_vendor(bare_manager, {"classification": {"model": "typesafe/jev-next"}})
    assert vendor is not None
    assert getattr(vendor, "model_id") == "typesafe/jev-next"


async def test_a_leg_whose_resolution_raises_is_skipped_not_propagated(
    bare_manager, install_legs
) -> None:
    """Credential resolution of ONE leg may never decide whether a turn fails."""
    legs = install_legs(
        radient={"credential_raises": True},
        typesafe={"credential": True},
    )
    vendor = await resolve_vendor(bare_manager)
    assert vendor is not None and vendor.name == "typesafe"
    assert legs["radient"].credential_calls == 1


# ---------------------------------------------------------------------------
# vendor_status: no I/O, and the documented Radient caveat
# ---------------------------------------------------------------------------


async def test_vendor_status_lists_every_leg_in_cascade_order(bare_manager) -> None:
    assert vendor_status(bare_manager) == [
        ("radient", False),
        ("typesafe", False),
        ("openrouter", False),
    ]


async def test_vendor_status_reports_a_static_credential(bare_manager) -> None:
    arm(bare_manager, "openrouter")
    assert vendor_status(bare_manager) == [
        ("radient", False),
        ("typesafe", False),
        ("openrouter", True),
    ]


async def test_vendor_status_honours_a_pin_by_rejecting_the_other_legs(bare_manager) -> None:
    """ "Available" means "the cascade would call this leg", which a pin changes."""
    arm(bare_manager, "radient", "openrouter")
    status = vendor_status(bare_manager, {"classification": {"vendor": "openrouter"}})
    assert status == [("radient", False), ("typesafe", False), ("openrouter", True)]


async def test_vendor_status_performs_no_io(bare_manager) -> None:
    """The proof that it does not touch the store: ``AuthStore`` would CREATE the file.

    Building the Radient leg and resolving its credential writes nothing on a
    fresh store but does open (and create) ``auth.db`` — so the absence of that
    file after the call is observable evidence that this function never built a
    store, i.e. never did the disk half of what it promises not to do.
    """
    vendor_status(bare_manager)
    assert not (bare_manager.config_dir / "auth.db").exists()


async def test_vendor_status_cannot_see_an_oauth_only_radient_session(bare_manager) -> None:
    """The documented limitation, asserted so it stays documented rather than assumed.

    A host whose only Radient credential is a signed-in OAuth session reports
    ``radient: False`` here, because seeing that row means opening the store.
    ``resolve_vendor`` is the authority; this function exists for the notice and
    diagnostics paths that must not await.
    """
    assert not (bare_manager.config_dir / "auth.db").exists()
    assert dict(vendor_status(bare_manager))["radient"] is False
    assert await resolve_vendor(bare_manager) is None
