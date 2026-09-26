"""The not-stale contract: nothing reports live without an ANSWERED round trip.

The design doc's §6 rule 1 (``docs/design/conversation-loading-and-runtime-status.md``,
revision 1) and the review's RR1-2/RR1-5. Three surfaces used to produce a false-live
answer for a SIGSTOPped owner — ``_cold_fields`` → ``remote.is_cold`` (a local
three-disjunct predicate with no round trip in it), the ``POST /warm`` receipt, and
``HEARTBEAT_TIMEOUT_S = 45`` — and these tests hold the first two.

THE LOCUS TRAP THESE TESTS EXIST FOR. ``_finish_sync`` looks like the natural place to
stamp a verification: it is the sync-completion path and it runs on every successful
bind. It is the WRONG place, because two of its seven callers are cold paths
(``attached.py`` ``:1646`` builds a ``cold-{session_id}`` epoch with ``_can_go_cold``,
and ``:1719`` says "Nothing is queued behind an owner that will never arrive"). A stamp
written there would hand a facade with no owner and no round trip a verification —
satisfying rule 1's letter, defeating its purpose, and looking correct in review because
the field would be present and set. ``test_finish_sync_is_not_a_verification_point`` is
that trap, written down.
"""

from __future__ import annotations

import asyncio

import pytest

from local_operator.server.utils.desktop_sessions import (
    WARM_VERIFY_BUDGET_S,
    DesktopSessionBridge,
)
from local_operator.session.attached import AttachedSession


class StubRemote:
    """The four terms ``_cold_fields`` reads, and the probe ``warm`` calls."""

    def __init__(
        self,
        *,
        is_cold: bool,
        cold_reason: str | None = None,
        attaching: bool = False,
        verified_at: float | None = None,
        answers: bool = True,
    ) -> None:
        self.is_cold = is_cold
        self.cold_reason = cold_reason
        self.attaching = attaching
        self.verified_at = verified_at
        self.answers = answers
        self.probes: list[float] = []

    async def verify_live(self, timeout: float) -> bool:
        self.probes.append(timeout)
        return self.answers


def _bridge(tmp_path, remote=None) -> DesktopSessionBridge:
    bridge = DesktopSessionBridge(tmp_path, "s1", str(tmp_path))
    if remote is not None:
        bridge.remote = remote
    return bridge


# --------------------------------------------------------------------------
# The invariant: a cold facade never carries a stamp.
# --------------------------------------------------------------------------


def test_a_saved_preview_facade_has_never_verified_anything(tmp_path):
    """``saved_preview`` is the cold facade: no dial, no owner, no stamp."""
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    (tmp_path / "sessions" / "s1" / "transcript.jsonl").write_text("")
    session = asyncio.run(
        AttachedSession.saved_preview(
            "s1",
            config_dir=tmp_path,
            cwd=str(tmp_path),
            takeover_factory=lambda: None,
        )
    )
    assert session.verified_at is None
    assert session.is_cold


def test_finish_sync_is_not_a_verification_point(tmp_path):
    """THE LOCUS TRAP. ``_finish_sync`` completes a sync; it does not answer a dial.

    Two of its seven callers are cold paths, so a stamp written here reaches a
    facade with no owner — the exact false-live condition rule 1 forbids.
    """
    (tmp_path / "sessions" / "s1").mkdir(parents=True)
    (tmp_path / "sessions" / "s1" / "transcript.jsonl").write_text("")
    session = asyncio.run(
        AttachedSession.saved_preview(
            "s1",
            config_dir=tmp_path,
            cwd=str(tmp_path),
            takeover_factory=lambda: None,
        )
    )
    session._finish_sync()
    assert session.verified_at is None, "a cold path stamped a verification"


def test_a_resync_does_not_clear_the_stamp(tmp_path):
    """Rule 1 must not regress the honest classification (RR1-5).

    ``is_cold``'s third disjunct is a RESYNC state, not an absent owner. A stamp
    that vanished on a display refresh would turn a live session cold — the same
    lie in the other direction.
    """
    session = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=lambda: None)
    session._verified_at = 1_000.0
    session._ready_for_events = False  # mid-resync
    assert session.is_cold, "the resync term is expected to report cold"
    assert session.verified_at == 1_000.0, "a resync cleared the stamp"


def test_the_probe_stamps_nothing_without_a_connected_dial(tmp_path):
    """An unanswered probe narrows the window; it never clears or sets a stamp."""
    session = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=lambda: None)
    assert asyncio.run(session.verify_live(0.05)) is False
    assert session.verified_at is None


def test_the_probe_refuses_to_stamp_when_the_client_is_disconnected(tmp_path):
    class Dead:
        connected = False

        async def request_ack_with_duplicate(self, op, **kw):  # pragma: no cover - must not run
            raise AssertionError("a disconnected dial must not be probed")

    session = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=lambda: None)
    session._client = Dead()  # type: ignore[assignment]
    assert asyncio.run(session.verify_live(0.05)) is False
    assert session.verified_at is None


def test_an_answered_probe_stamps(tmp_path):
    class Live:
        connected = True

        def __init__(self) -> None:
            self.ops: list[str] = []

        async def request_ack_with_duplicate(self, op, **kw):
            self.ops.append(op)
            return ("pong", False)

    session = AttachedSession(config_dir=tmp_path, session_id="s1", takeover_factory=lambda: None)
    client = Live()
    session._client = client  # type: ignore[assignment]
    assert asyncio.run(session.verify_live(0.5)) is True
    assert session.verified_at is not None
    assert client.ops == ["ping"], "the probe must use the existing liveness op"


# --------------------------------------------------------------------------
# The frame shape: rule 1 and rule 4, checked on the wire dict.
# --------------------------------------------------------------------------


def test_a_facade_with_no_remote_publishes_a_bare_cold_frame(tmp_path):
    fields = _bridge(tmp_path)._cold_fields()
    assert fields == {"cold": True, "cold_reason": "no-runtime", "attaching": False}
    assert "verified_at" not in fields


def test_a_resident_facade_with_no_answer_is_cold_not_live(tmp_path):
    """THE MEASURED VIOLATION: ``is_cold`` False used to publish ``cold: false``.

    A SIGSTOPped owner leaves the viewer's socket open, so ``is_cold`` stays False
    while the owner answers nothing. Without a stamp the frame must not claim live.
    """
    remote = StubRemote(is_cold=False, cold_reason=None, attaching=False, verified_at=None)
    fields = _bridge(tmp_path, remote)._cold_fields()
    assert fields["cold"] is True
    assert fields["cold_reason"] == "owner-silent"
    assert fields["attaching"] is True, "a dial exists, so 'something is coming' is the honest flag"
    assert "verified_at" not in fields


def test_a_live_frame_carries_its_stamp_and_nothing_else(tmp_path):
    remote = StubRemote(is_cold=False, verified_at=1_234.5)
    fields = _bridge(tmp_path, remote)._cold_fields()
    assert fields == {
        "cold": False,
        "cold_reason": None,
        "attaching": False,
        "verified_at": 1_234.5,
    }


@pytest.mark.parametrize("verified_at", [None, 1_234.5])
def test_rule_four_cold_and_attaching_are_never_both_false_and_true(tmp_path, verified_at):
    """§6 rule 4: the renderer's choice is total — live now, or explicitly attaching."""
    remote = StubRemote(is_cold=False if verified_at else True, verified_at=verified_at)
    fields = _bridge(tmp_path, remote)._cold_fields()
    assert not (fields["cold"] is False and fields["attaching"] is True)
    # T3, at the shape level: a live frame always carries a stamp.
    if fields["cold"] is False:
        assert fields["verified_at"] is not None


# --------------------------------------------------------------------------
# T4: the /warm receipt.
# --------------------------------------------------------------------------


def test_warm_declines_to_claim_warm_when_the_owner_does_not_answer(tmp_path, monkeypatch):
    """T4. With the owner stopped this used to answer 200 ``{"state": "warm"}``."""
    remote = StubRemote(is_cold=False, verified_at=None, answers=False)
    bridge = _bridge(tmp_path, remote)
    monkeypatch.setattr(bridge, "_schedule_warm", lambda: None)
    assert asyncio.run(bridge.warm()) == "warming"
    assert remote.probes == [WARM_VERIFY_BUDGET_S], "the receipt must probe, not trust residency"


def test_warm_still_answers_warm_when_the_owner_answers(tmp_path, monkeypatch):
    remote = StubRemote(is_cold=False, verified_at=1_234.5, answers=True)
    bridge = _bridge(tmp_path, remote)
    monkeypatch.setattr(bridge, "_schedule_warm", lambda: None)
    assert asyncio.run(bridge.warm()) == "warm"
    assert remote.probes, "a warm receipt is only honest after an answer"


def test_the_warm_probe_is_bounded_well_inside_the_not_stale_budget(tmp_path):
    """The first keystroke must not hang for an acks' worth of seconds."""
    assert 0 < WARM_VERIFY_BUDGET_S <= 2.0
