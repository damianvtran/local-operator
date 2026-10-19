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
:func:`resolve_vendor`, which does the read once per session;
``Recommendation.vendor`` on a delivered answer is that call's result, reported
through the cost log rather than through a user-facing line.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import httpx

from local_operator.classification.types import DecisionVendor
from local_operator.classification.vendors import VENDOR_CLASSES, build_vendor

if TYPE_CHECKING:
    from pathlib import Path

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

#: Why a leg was skipped, in the vocabulary the decision and spend rows speak.
#:
#: ``not_synced`` is the one this slice exists to surface: a placement row says
#: another device owns this credential and lends it to us, yet the borrow produced
#: nothing — so the operator is being served by a later leg they did not choose.
#: ``borrow_failed`` carries the broker's own code in ``detail``; ``resolution_failed``
#: is a leg whose credential probe raised. ``no_credential_here`` is absence, which
#: is not a defect.
SKIP_NO_CREDENTIAL = "no_credential_here"
SKIP_NOT_SYNCED = "not_synced"
SKIP_BORROW_FAILED = "borrow_failed"
SKIP_RESOLUTION_FAILED = "resolution_failed"


@dataclass(frozen=True)
class VendorResolution:
    """The leg that resolved, plus every preferred leg skipped to reach it.

    Frozen: this is evidence for a decision already made, not working state. The
    skipped rows are plain dicts rather than a dataclass because they travel
    straight into ``Recommendation`` fields and the channel-ledger ``detail``,
    where a second serialisation step would only add a place to drift.
    """

    vendor: Any | None
    skipped: tuple[dict[str, str], ...]


def _raised_skip(name: str, error: BaseException) -> dict[str, str]:
    """The skipped row for a leg whose credential probe RAISED.

    A refusal carries the broker vocabulary's ``code`` attribute (``MeshRefusal``
    is the raiseable shape; ``BrokerError`` is the same vocabulary on a borrow's
    RETURN path, which the borrow helper collapses into ``None`` before this layer
    sees it). When an exception names a code, the row keeps it in ``detail`` and
    the reason is ``borrow_failed``; anything else is ``resolution_failed`` — a leg
    that could not answer, not a provider absence.
    """
    code = getattr(error, "code", "")
    if isinstance(code, str) and code:
        return {"vendor": name, "reason": SKIP_BORROW_FAILED, "detail": code}
    return {"vendor": name, "reason": SKIP_RESOLUTION_FAILED}


def _skip_row(config_dir: "Path | None", name: str) -> dict[str, str]:
    """Why a leg with NO credential was skipped, WITHOUT dialing anyone.

    The placement document and the client's own holder predicate answer the
    question locally: a row this device is a holder for and does not own means a
    borrow was (or should have been) attempted, which is ``not_synced``; anything
    else is plain absence. Deliberately no grant request here — the trace must not
    turn a diagnostic into a second borrow, and the resolved leg never reaches this
    helper at all (see :func:`resolve_vendor_with_trace`).

    Every failure inside the probe reads as absence: an unreadable document must
    not present itself as a credential problem the operator can act on.
    """
    try:
        from local_operator.network.credentials.client import (
            MeshCredentialClient,
            key_for,
        )
        from local_operator.network.credentials.placement import placement_for_store

        if placement_for_store(config_dir) is None:
            return {"vendor": name, "reason": SKIP_NO_CREDENTIAL}
        mesh = MeshCredentialClient.for_this_device(config_dir)
        if mesh is not None and mesh.should_borrow(key_for(provider=name)):
            return {"vendor": name, "reason": SKIP_NOT_SYNCED}
    except Exception:  # noqa: BLE001 — a diagnostic read is never worth a failure
        logger.debug("classification: placement probe failed for %s", name, exc_info=True)
    return {"vendor": name, "reason": SKIP_NO_CREDENTIAL}


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
    recommendation (and therefore in the cost log line).
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


async def resolve_vendor_with_trace(
    config_dir: "Path | None",
    settings: Mapping[str, Any] | None = None,
    *,
    client: httpx.AsyncClient | None = None,
) -> VendorResolution:
    """Resolve the first usable leg AND record every preferred leg skipped before it.

    :func:`resolve_vendor`'s walk, unchanged: same order, same pin, same instance
    kept for its credential memo, and a leg whose resolution raises is still
    skipped rather than propagated.

    The trace exists because a fall-through used to be INVISIBLE. When the Radient
    login lives on a mesh owner and this device could not borrow it, the cascade
    silently used OpenRouter: the operator was billed on a different vendor and the
    ledger attributed the spend to a path they did not choose. Naming the skipped
    leg and the reason turns that into an observation.

    COST IS PAID ONLY FOR SKIPPED LEGS. A leg that resolves returns immediately and
    is never probed further; only a leg with no credential is classified, and that
    classification reads local placement (never the network).
    """
    model = model_override(settings)
    skipped: list[dict[str, str]] = []
    for name in leg_order(settings):
        vendor = build_vendor(name, config_dir, model=model, client=client)
        try:
            credential = await vendor.credential(config_dir)
        except Exception as exc:  # noqa: BLE001 — a leg that cannot resolve is not this leg
            logger.warning("classification: %s credential resolution failed", name, exc_info=True)
            skipped.append(_raised_skip(name, exc))
            continue
        if credential:
            return VendorResolution(vendor=vendor, skipped=tuple(skipped))
        skipped.append(_skip_row(config_dir, name))
        logger.debug("classification: %s has no credential here (%s)", name, skipped[-1]["reason"])
    return VendorResolution(vendor=None, skipped=tuple(skipped))


async def resolve_vendor(
    config_dir: "Path | None",
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

    Kept as a thin wrapper over :func:`resolve_vendor_with_trace` so every existing
    caller sees the same value and the two walks cannot drift.
    """
    resolution = await resolve_vendor_with_trace(config_dir, settings, client=client)
    return resolution.vendor


def vendor_status(
    config_dir: "Path | None",
    settings: Mapping[str, Any] | None = None,
) -> list[tuple[str, bool]]:
    """``[(vendor_name, available)]`` for diagnostics and tests. Never performs I/O.

    ``available`` means "the cascade would call this leg", so under a pin only
    the pinned leg can be ``True`` and the other two are reported ``False`` even
    when their credentials are present — that is the question this function
    answers, and it is the one the diagnostics actually ask.

    The probe is the credential tiers a synchronous, disk-free read can see: the
    provider credential store and the process environment. Radient's OAuth
    session is therefore NOT visible here — see the module docstring — so
    ``radient`` reports ``False`` on a host whose only Radient credential is a
    signed-in session. :func:`resolve_vendor` is the authority; this function
    exists so a ``/info`` line or a login-status read can be produced without an
    ``await`` and without a store read.
    """
    pin = pinned_vendor(settings)
    status: list[tuple[str, bool]] = []
    for name in VENDOR_ORDER:
        if pin not in (DEFAULT_VENDOR, name):
            # A pin means the cascade never reaches the other legs, whatever
            # their credentials say.
            status.append((name, False))
            continue
        status.append((name, _static_credential_present(config_dir, name)))
    return status


def _static_credential_present(config_dir: "Path | None", name: str) -> bool:
    """Whether the leg's disk-free credential tier is populated.

    Each leg's key names mirror :mod:`local_operator.classification.vendors`
    exactly; the duplication is two tuples of constant names, and it is kept
    here rather than imported because the vendor classes resolve Radient
    through the AuthStore and this function must not. The environment tier is
    read here too, which is how a shell-exported key is discovered.
    """
    keys = {
        "radient": ("RADIENT_API_KEY",),
        "typesafe": ("TYPESAFE_API_KEY", "JEV_API_KEY"),
        "openrouter": ("OPENROUTER_API_KEY", "OPENROUTER_API_KEY_DEV"),
    }[name]
    # Store-first, matching the vendor legs' own resolution: a provider-class
    # store row counts as a populated static tier, then the process environment.
    # The legacy ``credentials.env`` rung is GONE (PR2a), so a name neither the
    # store nor the environment holds is simply not present.
    from local_operator.providers.registry import provider_secret_value

    # The caller's own root (R4), matching the sibling vendor legs in
    # ``vendors.py``: a provider-class row under a non-default config root is the
    # one this probe must see, or the diagnostic reports a host it never read.
    return any(provider_secret_value(key, base=config_dir) or os.environ.get(key) for key in keys)


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
