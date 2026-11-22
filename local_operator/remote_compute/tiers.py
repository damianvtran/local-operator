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
"""

from __future__ import annotations

from local_operator.i18n.keys import wire_remote_compute as _wr
from local_operator.i18n.messages import Msg

#: The tiers, in the order a picker shows them: on-demand first, because it is
#: the default and the cheap one to reason about.
TIERS: tuple[str, ...] = ("cloud_run", "dedicated")

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
    "TIERS",
    "option_label",
    "tier_label",
    "tier_options",
]
