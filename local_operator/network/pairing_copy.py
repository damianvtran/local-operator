"""Keyed copy (``network.pairing``) for the Mesh approvals pairing refusals.

The desktop route's refusals carry a machine ``code`` the renderer keys on AND a
sentence; the sentence is product copy, so it is keyed and translated rather than the
relay's English. Resolved at call time (never at import) so a live language change is
honoured. Codes this module has no sentence for return ``""`` and the caller keeps the
relay's own words.
"""

from __future__ import annotations


def _t():  # type: ignore[no-untyped-def]
    from local_operator.i18n.text import tr

    return tr("network.pairing")


def refusal(code: str) -> str:
    from local_operator.i18n.keys import network_pairing as keys

    msg = {
        "pairing_needs_app": keys.refusal_needs_app,
        "pairing_not_ready": keys.refusal_not_ready,
        "pairing_expired": keys.refusal_expired,
        # Outcome-neutral on purpose (security review round 2, S-R2-2): the first
        # answer may already have admitted the device, so "nothing was approved"
        # would be false here.
        "pairing_answered": keys.refusal_answered,
        "no_pending_pairing": keys.refusal_no_pending,
    }.get(code)
    return _t()(msg()) if msg is not None else ""


def mismatch(attempts_left: int) -> str:
    from local_operator.i18n.keys import network_pairing as keys

    return _t()(keys.refusal_mismatch(count=max(0, int(attempts_left))))
