"""The tiers a surface offers, and their copy — one name per tier, no synonyms.

D1 PRODUCT COPY, AS KEYS. The desktop GUI offers two tiers: the on-demand one
labelled "Cloud run", and "Dedicated". Spot capacity is an OPT-IN choice UNDER
Dedicated — a cost decision on the same run, not a third tier — so it is its own
option and never a tier label. The words are catalogue copy rather than English
literals here for the reason the i18n contract states: every user-facing string
is keyed, and the tier picker is user-facing.

WHY THE IDS ARE SEPARATE FROM THE LABELS. ``cloud_run``/``dedicated`` are what a
request body, a stored record and a test name; the label is what a person reads.
A surface must never build one from the other (``"Cloud " + id`` reads correctly
in English and wrongly almost everywhere else), so the only route from an id to
text is :func:`tier_label`, which returns the keyed message.

NEVER "radient spot". The capacity mode is a property of the run an operator is
composing; it is not a branded product tier, and naming it as one implied a
separate offering that does not exist.

THE SERVER HAS ITS OWN WORDS, AND THIS IS WHERE THEY ARE TRANSLATED. The capacity
read answers ``fargate``/``kubernetes`` — where the deployment runs the work — and
that is not what a picker offers. :data:`SERVER_TIER_TO_TIER` is the ONE join
between the two vocabularies; no caller compares a row against the server's word.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.messages import Msg

#: The tiers, in the order a picker shows them: on-demand first, because it is
#: the default and the cheap one to reason about.
TIERS: tuple[str, ...] = ("cloud_run", "dedicated")

#: The SERVER's tier literal → this build's tier id.
#:
#: The server names capacity by WHERE the work runs — ``fargate`` is the on-demand
#: tier (one task per run) and ``kubernetes`` the dedicated-workload one — while this
#: build names it by what the operator CHOOSES, ``cloud_run``/``dedicated``. The two
#: vocabularies meet only here: a surface that compared the server's word would paint
#: a ready tier as unavailable, and a second table beside this one is how two surfaces
#: come to label the same row differently.
#:
#: IMMUTABLE, so "the ONE join" holds by construction rather than by convention: a
#: mutable module-level table is one ``[...] = ...`` away from a second vocabulary,
#: and nothing would fail until two surfaces labelled the same row differently.
SERVER_TIER_TO_TIER: Mapping[str, str] = MappingProxyType(
    {
        "fargate": "cloud_run",
        "kubernetes": "dedicated",
    }
)

#: The capacity choices offered UNDER :data:`TIERS` as opt-ins.
#:
#: ``spot`` belongs to ``dedicated`` only: on-demand capacity has no spot market,
#: so offering it there would describe a run the server cannot admit.
DEDICATED_OPTIONS: tuple[str, ...] = ("spot",)

#: Tier id → the keyed label.
_LABELS = {
    "cloud_run": _wr.tier_cloud_run,
    "dedicated": _wr.tier_dedicated,
}

#: Option id → the keyed label. A separate table because an option is not a tier
#: and must not be reachable through :func:`tier_label`.
_OPTION_LABELS = {
    "spot": _wr.option_spot,
}


def tier_label(tier: str) -> Msg:
    """The keyed label for ``tier``; an unknown id is refused, never echoed.

    A ``KeyError`` rather than a fallback label: a tier this build does not know
    is a contract change, and rendering its raw id as copy would put an
    untranslated identifier in front of a person.
    """
    try:
        return _LABELS[tier]()
    except KeyError:
        raise ValueError(f"unknown remote-compute tier {tier!r}; the tiers are {TIERS}") from None


def tier_from_server(wire: str) -> str | None:
    """This build's id for a server tier literal; ``None`` when it names no tier.

    AN ID THIS BUILD ALREADY USES IS RETURNED UNCHANGED. The wire carries the
    server's vocabulary today, but a read is answered by whichever server build the
    deployment runs, and one that has caught up to ours answers ``cloud_run`` or
    ``dedicated``. That word is not a translation source — it is already the id —
    and reading it as "cannot name" silently stripped the tier off every row that
    carried it.

    ``None``, never a guess for anything else: a tier this build has never heard of
    is a contract change, and calling it ``cloud_run`` would offer the operator
    capacity the server would not admit under that word. The caller drops what
    carried it and logs the literal at debug, so a new server tier stays visible
    without being mislabelled as one this build knows.
    """
    if wire in TIERS:
        return wire
    return SERVER_TIER_TO_TIER.get(wire)


def option_label(option: str) -> Msg:
    """The keyed label for a capacity option under Dedicated."""
    try:
        return _OPTION_LABELS[option]()
    except KeyError:
        raise ValueError(
            f"unknown remote-compute option {option!r}; the options are {DEDICATED_OPTIONS}"
        ) from None


def tier_options(tier: str) -> tuple[str, ...]:
    """The opt-in capacity choices ``tier`` offers: spot for dedicated, none else."""
    if tier not in TIERS:
        raise ValueError(f"unknown remote-compute tier {tier!r}; the tiers are {TIERS}")
    return DEDICATED_OPTIONS if tier == "dedicated" else ()


__all__ = [
    "DEDICATED_OPTIONS",
    "SERVER_TIER_TO_TIER",
    "TIERS",
    "option_label",
    "tier_from_server",
    "tier_label",
    "tier_options",
]
