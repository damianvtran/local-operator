"""§9 mode resolution: `auto` is unreachable in LO1, and that is a test.

The architecture decision on PR #3078 (check 6, rev7) keeps `auto` off the
table: an unattended default that turns Saver on for a user who never asked is
only safe once the verifier can prove an epoch cheaper than the history it
replaces, and that proof is the token comparison landing in this PR — not a
property the provenance rules can supply. A comment would not survive the flip,
so the resolver's behaviour is pinned here.
"""

from __future__ import annotations

from local_operator.saver.gate import Availability, resolve_saver

_READY = Availability(signed_in=True, positive_balance=True)


def test_auto_never_resolves_on() -> None:
    """The one values-based default that could bill a user who never opted in."""
    decision = resolve_saver({"radient": {"saver": {"mode": "auto"}}}, _READY)
    assert decision.on is False
    assert decision.source == "auto_unavailable"


def test_an_explicit_session_choice_still_wins_over_the_auto_refusal() -> None:
    """`opt_in` is unaffected, and a session that explicitly asked is not `auto`."""
    assert resolve_saver({"radient": {"saver": {"mode": "opt_in"}}}, _READY).source == "default"
    chosen = resolve_saver({"radient": {"saver": {"mode": "auto"}}}, _READY, session_choice=True)
    assert chosen.on is True and chosen.source == "session"


def test_the_usage_gates_still_precede_every_mode() -> None:
    """Signed-out and no-credit come before the mode, whatever it says."""
    assert resolve_saver({"radient": {"saver": {"mode": "opt_in"}}}, None).source == "signed_out"
    assert (
        resolve_saver(
            {"radient": {"saver": {"mode": "opt_in"}}},
            Availability(signed_in=True, positive_balance=False),
        ).source
        == "no_credit"
    )
