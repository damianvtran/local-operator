"""Leg ordering, the credential probe, and the ``values.classification.vendor`` pin.

The cascade is a list, not a branch (§3): every leg speaks the same request and
answer shape, so "which vendor?" is one loop and one boolean per leg.

WHY ORDER IS RADIENT → TYPESAFE → OPENROUTER
============================================

Cheapest-and-most-direct first. Radient's route is our own passthrough, billed
to the account the operator already signed into, and it keeps the request on
infrastructure we run. TypeSafe is the vendor's native endpoint — the leg that
cannot be affected by anything of ours. OpenRouter's alpha route is last because
it is the one leg that is *always* available (a dev key ships on most machines)
and the one with a documented rate limit of 0.5 req/s, so it should absorb the
traffic only when the better legs are unusable.

WHY ``vendor_status`` DOES NOT OPEN THE AUTH STORE
==================================================

§4 requires ``vendor_status`` to perform no I/O. Radient's primary credential is
an OAuth session in ``AuthStore`` (a sqlite file), so a truthful answer for that
leg needs a store read — which is exactly what this function may not do, since
its callers are diagnostics that run on synchronous paths. It
therefore reports the tier it CAN see without touching the disk
(``RADIENT_API_KEY`` in the credential store or the environment) and the
docstring says so. The authority for "which leg will actually be used" is
:func:`resolve_vendor`, which does the read once per session; the cost log line
reports ``Recommendation.vendor``, which came from that call.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from local_operator.classification.types import DecisionVendor
from local_operator.classification.vendors import VENDOR_CLASSES, build_vendor

if TYPE_CHECKING:
    from local_operator.credentials import CredentialManager

logger = logging.getLogger(__name__)

#: The legs, in the order the cascade tries them. Frozen as a tuple so a caller
#: cannot reorder the cascade at runtime by mutating a module-level list.
VENDOR_ORDER: tuple[str, ...] = ("radient", "typesafe", "openrouter")

#: ``values.classification.vendor`` — ``auto`` means "walk :data:`VENDOR_ORDER`".
DEFAULT_VENDOR = "auto"

#: ``values.classification.model`` — empty means "the vendor's own model id".
DEFAULT_MODEL = ""

#: The pinnable values, used by the settings registry and by our own parsing.
VENDOR_CHOICES: tuple[str, ...] = (DEFAULT_VENDOR, *VENDOR_ORDER)


def classification_section(settings: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The ``classification`` sub-mapping, tolerating either shape of mapping.

    Callers hand this module the config's ``values`` mapping (§8's keys live
    under ``values.classification``), but an embedding caller — and every unit
    test that wants to pin one key — will naturally pass the section itself.
    Accepting both, and saying so, is cheaper than a second accessor that drifts
    from this one; a mapping that happens to carry both a ``classification``
    sub-map and its own keys resolves to the sub-map.
    """
    if not settings:
        return {}
    nested = settings.get("classification")
    if isinstance(nested, Mapping):
        return nested
    return settings


def pinned_vendor(settings: Mapping[str, Any] | None) -> str:
    """The pinned leg, or ``DEFAULT_VENDOR`` — never an unusable value.

    An unrecognised pin (only reachable from a hand-edited ``config.yml``; the
    settings registry validates the stored value against :data:`VENDOR_CHOICES`)
    reads as ``auto`` rather than as "no vendor at all". Silently classifying
    through a different leg would be a surprising outcome for a typo, but
    silently disabling the layer is worse: it is the failure mode this whole
    design is built to avoid, and the resolved leg is reported on every
    recommendation and on the cost log line.
    """
    raw = classification_section(settings).get("vendor")
    if isinstance(raw, str):
        value = raw.strip().lower()
        if value in VENDOR_ORDER:
            return value
    return DEFAULT_VENDOR


def model_override(settings: Mapping[str, Any] | None) -> str:
    """``values.classification.model``, or ``""`` for the vendor's own id."""
    raw = classification_section(settings).get("model")
    return raw.strip() if isinstance(raw, str) else DEFAULT_MODEL


def leg_order(settings: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The legs this configuration will try, in order — one when pinned."""
    pin = pinned_vendor(settings)
    return (pin,) if pin in VENDOR_ORDER else VENDOR_ORDER


async def resolve_vendor(
    manager: "CredentialManager",
    settings: Mapping[str, Any] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> DecisionVendor | None:
    """First available leg honouring ``values.classification.vendor``; ``None`` when none is usable.

    "Available" means the leg resolved a non-empty credential. Resolution can
    touch the network (an expired Radient OAuth grant is refreshed in place),
    which is why this is async and why the caller keeps the instance it gets
    back: the credential memo lives on that instance.

    ``client`` is the caller's keep-alive HTTP client, passed through to every
    leg this builds (:class:`~local_operator.classification.service.ClassificationService`
    owns one per session). Optional and keyword-only so the contract's two-argument
    call still works; a leg built without one opens a client for its own call.

    A leg whose resolution *raises* is skipped, not propagated: the credential
    resolution of one leg cannot be allowed to decide whether the operator's
    turn fails.
    """
    model = model_override(settings)
    for name in leg_order(settings):
        vendor = build_vendor(name, manager, model=model, client=client)
        try:
            credential = await vendor.credential(manager)
        except Exception:  # noqa: BLE001 — a leg that cannot resolve is not this leg
            logger.warning("classification: %s credential resolution failed", name, exc_info=True)
            continue
        if credential:
            return vendor
    return None


def vendor_status(
    manager: "CredentialManager",
    settings: Mapping[str, Any] | None = None,
) -> list[tuple[str, bool]]:
    """``[(vendor_name, available)]`` for tests and diagnostics. Never performs I/O.

    ``available`` means "the cascade would call this leg", so under a pin only
    the pinned leg can be ``True`` and the other two are reported ``False`` even
    when their credentials are present — that is the question this function
    answers, and it is the one the diagnostics actually ask.

    The probe is the credential tiers a synchronous, disk-free read can see: the
    credential store and the environment (both of which ``CredentialManager``
    already holds in memory). Radient's OAuth session is therefore NOT visible
    here — see the module docstring — so ``radient`` reports ``False`` on a host
    whose only Radient credential is a signed-in session. :func:`resolve_vendor`
    is the authority; this function exists so a ``/info`` line can
    be produced without an ``await`` and without a store read.
    """
    pin = pinned_vendor(settings)
    status: list[tuple[str, bool]] = []
    for name in VENDOR_ORDER:
        if pin not in (DEFAULT_VENDOR, name):
            # A pin means the cascade never reaches the other legs, whatever
            # their credentials say.
            status.append((name, False))
            continue
        status.append((name, _static_credential_present(manager, name)))
    return status


def _static_credential_present(manager: "CredentialManager", name: str) -> bool:
    """Whether the leg's disk-free credential tier is populated.

    Each leg's key names mirror :mod:`local_operator.classification.vendors`
    exactly; the duplication is two tuples of constant names, and it is kept
    here rather than imported because the vendor classes resolve Radient
    through the AuthStore and this function must not. ``get_credential`` reads
    the environment tier too (without writing it back to disk), which is how a
    shell-exported key is discovered.
    """
    keys = {
        "radient": ("RADIENT_API_KEY",),
        "typesafe": ("TYPESAFE_API_KEY", "JEV_API_KEY"),
        "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_DEV"),
    }[name]
    return any(bool(manager.get_credential(key)) for key in keys)


__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_VENDOR",
    "VENDOR_CHOICES",
    "VENDOR_CLASSES",
    "VENDOR_ORDER",
    "classification_section",
    "leg_order",
    "model_override",
    "pinned_vendor",
    "resolve_vendor",
    "vendor_status",
]
