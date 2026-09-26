"""The refusal copy catalogue is ONE table, and this file pins it.

Audit round 2, F1 (second half). ``SENTENCED_CODES`` is what tells the requester "the
catalogue speaks for this code" — the rule that stops an owner's wire diagnostic from
being shown to the operator instead of the sentence the design promises. A set that
drifts from the chain in :func:`render_broker_error` would silently send those
diagnostics back to the screen, so both halves are asserted here rather than trusted.
"""

from __future__ import annotations

from local_operator.network.credentials.messages import (
    SENTENCED_CODES,
    _generic,
    has_catalogue_sentence,
    render_broker_error,
)
from local_operator.network.credentials.types import BROKER_ERROR_CODES, BrokerError

OWNER = "d_" + "a" * 32


def test_every_sentenced_code_really_has_its_own_sentence() -> None:
    """``SENTENCED_CODES`` cannot rot into a claim this module does not honour."""
    for code in sorted(SENTENCED_CODES):
        assert has_catalogue_sentence(code)
        error = BrokerError(
            code=code, key="zai", owner_device=OWNER, owner_device_name="damian-mbp"
        )
        rendered = render_broker_error(error, key="zai", owner_name="damian-mbp", provider="zai")
        generic = _generic(error, label="zai", owner="damian-mbp", login="lop login zai")
        assert rendered != generic, f"{code} renders the generic fallback"


def test_the_catalogue_covers_every_code_the_owner_can_emit_but_internal() -> None:
    """The set is the *classified* half of the closed code list, and says so.

    ``internal`` is deliberately absent: it is this build's own catch-all, and
    ``_generic`` is where its detail belongs.
    """
    missing = {code for code in BROKER_ERROR_CODES if code not in SENTENCED_CODES}
    assert missing == {"internal"}, missing
