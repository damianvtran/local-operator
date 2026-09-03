"""Parked gates: a question waits for a person instead of being denied.

The 30-second cap was written for a phone in a pocket, where denying was the
kind answer — the turn moved on instead of pinning a tool slot forever. Under
the detached model the same wait usually means "the user stepped away from a
session that is still running", and denying their write tool after thirty
seconds answers a question nobody asked.

What these tests hold down is the pair of facts that makes parking safe: the
wait is bounded by a setting the user controls, and the cost is VISIBLE while
it lasts (`pending` on the record, the picker's needs-you marker, the `lop
sessions` column, the notification).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.session.runtime.owned import (
    DEFAULT_UNATTENDED_GATE_TIMEOUT_H,
    GATE_TIMEOUT_CUSTOM_TYPE,
    PENDING_REQUEST_TIMEOUT_S,
)
from local_operator.session.runtime.types import PROTOCOL_VERSION, SessionRecord


class _Server:
    """Stands in for the RuntimeServer the handle publishes through.

    Deliberately answers by COUNT only (no ``watching_surfaces``): this is
    the shape an older release published, so these cases also pin the
    handle's compatibility fallback for a registrant that cannot name the
    kinds of surface watching it.
    """

    def __init__(self, attached: int = 0) -> None:
        self._attached = attached
        self.pending: list[str | None] = []

    def attach_clients(self) -> int:
        return self._attached

    def set_record_pending(self, pending: str | None) -> None:
        self.pending.append(pending)


def _handle(
    monkeypatch, *, attached: int, hours: int | None = None, notify: bool = True
):  # noqa: ANN001
    """An OwnedSessionHandle with just enough wired to exercise the policy."""
    from local_operator.session.runtime.owned import OwnedSessionHandle

    handle = OwnedSessionHandle.__new__(OwnedSessionHandle)
    handle._registrant = _Server(attached)  # type: ignore[attr-defined]
    if hours is not None:
        monkeypatch.setattr(
            OwnedSessionHandle, "_unattended_gate_hours", lambda self: hours, raising=False
        )
    monkeypatch.setattr(
        "local_operator.tui.notify.notifications_enabled", lambda: notify, raising=False
    )
    return handle


def test_an_attached_viewer_parks_for_the_configured_day(monkeypatch) -> None:
    """Somebody is looking at the card, so there is no reason to deny it."""
    handle = _handle(monkeypatch, attached=1)

    assert handle._gate_timeout_s() == DEFAULT_UNATTENDED_GATE_TIMEOUT_H * 3600.0


def test_zero_means_never_time_out(monkeypatch) -> None:
    handle = _handle(monkeypatch, attached=1, hours=0)

    assert handle._gate_timeout_s() is None


def test_a_detached_session_still_parks_when_it_can_notify(monkeypatch) -> None:
    """Nobody is attached, but the user can still be TOLD, so parking is honest."""
    handle = _handle(monkeypatch, attached=0, notify=True)

    assert handle._gate_timeout_s() == DEFAULT_UNATTENDED_GATE_TIMEOUT_H * 3600.0


def test_the_short_cap_survives_where_nobody_could_ever_learn_of_the_question(
    monkeypatch,
) -> None:
    """The one case the original 30 s cap was written for, kept deliberately.

    No front end is attached AND notifications are off: nobody will ever be
    told about this card, and holding a tool slot for a day on that basis
    would be a hang wearing a feature's clothes.
    """
    handle = _handle(monkeypatch, attached=0, notify=False)

    assert handle._gate_timeout_s() == PENDING_REQUEST_TIMEOUT_S


def test_a_host_with_no_control_socket_keeps_the_ordinary_cap(monkeypatch) -> None:
    """An embedded or reduced host cannot ever attach a front end.

    Found by a HANG, not by reading: the first cut keyed only on attach count
    and notifications, so `test_owned.py`'s reduced handle — no registrant,
    notifications enabled on the dev box — took the parked branch and waited
    the configured 24 hours. The suite stopped dead. A gate that no socket can
    reach is exactly what `PENDING_REQUEST_TIMEOUT_S` is for, and keying on the
    registrant says that directly instead of inferring it.
    """
    handle = _handle(monkeypatch, attached=0)
    handle._registrant = None

    assert handle._gate_timeout_s() == PENDING_REQUEST_TIMEOUT_S


def test_the_parked_wait_is_never_shorter_than_the_ordinary_cap(monkeypatch) -> None:
    """Whatever the policy returns, a gate waits at least as long as before.

    The floor is also what keeps the constant meaningful: a caller that
    shortens it to make a gate expire quickly still gets a gate that expires
    quickly.
    """
    handle = _handle(monkeypatch, attached=1, hours=1)
    monkeypatch.setattr("local_operator.session.runtime.owned.PENDING_REQUEST_TIMEOUT_S", 7200.0)

    assert handle._gate_timeout_s() == 7200.0


def test_the_record_carries_the_parked_state_and_clears_it(monkeypatch) -> None:
    handle = _handle(monkeypatch, attached=1)

    handle._announce_pending("approval", "bash", "rm -rf build/")
    handle._announce_settled()

    assert handle._registrant.pending == ["approval", None]


def test_an_attached_viewer_is_not_also_toasted(monkeypatch) -> None:
    """Exactly one notification per question, not one per delivery channel.

    With a viewer attached the card is painted in-band; a second out-of-band
    toast about the same question is noise.
    """
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "local_operator.tui.notify.detached_notify",
        lambda title, body, **kwargs: sent.append((title, body)) or True,
        raising=False,
    )
    handle = _handle(monkeypatch, attached=1)

    handle._announce_pending("approval", "bash", "rm -rf build/")

    assert sent == [], "an attached viewer must not be toasted as well"


def test_a_detached_session_notifies_out_of_band(monkeypatch) -> None:
    sent: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "local_operator.tui.notify.detached_notify",
        lambda title, body, **kwargs: sent.append((title, body)) or True,
        raising=False,
    )
    handle = _handle(monkeypatch, attached=0)
    handle._session = type("S", (), {"conversation_name": "deploy checks"})()

    handle._announce_pending("approval", "bash", "rm -rf build/")

    assert len(sent) == 1
    assert "deploy checks" in sent[0][0]


def test_the_new_record_fields_do_not_move_the_protocol_version() -> None:
    """Additive fields must not make an older peer refuse a usable record.

    Nothing is required to READ these: an older reader drops unknown keys in
    ``from_json`` and behaves as it did. Bumping the version would instead make
    every older peer refuse a record it can in fact use, which is the whole
    compatibility cost of a field nobody has to read.
    """
    assert PROTOCOL_VERSION == 5

    record = SessionRecord(
        pid=1,
        kind="daemon",
        session_id="s1",
        conversation_name="n",
        cwd="/tmp",
        model_label="m",
        control_port=1,
        control_key="k",
        pending="approval",
        busy=True,
        detached=True,
    )
    assert record.protocol == 5

    # An OLDER runtime's record: no live-state keys at all.
    old = SessionRecord.from_json(
        {
            "pid": 1,
            "kind": "daemon",
            "session_id": "s1",
            "conversation_name": "n",
            "cwd": "/tmp",
            "model_label": "m",
            "control_port": 1,
            "control_key": "k",
            "protocol": 5,
        }
    )
    assert old.pending is None and old.busy is False and old.detached is False

    # A NEWER runtime's record read by this one: unknown keys are dropped.
    newer = SessionRecord.from_json({**record.to_json(), "a_field_from_the_future": 1})
    assert newer.pending == "approval"


@pytest.mark.asyncio
async def test_a_timed_out_gate_says_nobody_was_there(tmp_path: Path, monkeypatch) -> None:
    """A denial and an expiry are different facts and must not look alike.

    Without this row the next turn reads "the user denied this" and plans
    around a choice nobody made.
    """
    appended: list[Any] = []

    class _Transcript:
        async def append_message(self, message):  # noqa: ANN001
            appended.append(message)

    handle = _handle(monkeypatch, attached=0, hours=0)
    handle._session = type("S", (), {"transcript": _Transcript()})()

    await handle._record_gate_timeout("bash", "rm -rf build/")

    # A MESSAGE entry, not `append_custom`. Round 1 (D2/U2) found the row was
    # written where NOTHING could read it: `build_llm_history` ignores custom
    # ENTRIES by design, so neither the model nor the viewer that replays the
    # same history ever saw it. Asserting the entry SHAPE — not just the
    # payload — is what keeps that from silently regressing.
    assert appended, "the expiry was not recorded at all"
    row = appended[0]
    assert row.custom_type == GATE_TIMEOUT_CUSTOM_TYPE
    assert row.details["tool"] == "bash"
    assert row.details["description"] == "rm -rf build/"


def test_a_timed_out_gate_reaches_the_model_on_replay() -> None:
    """The row is only worth writing if replay surfaces it.

    Round 1 (D2/U2): the row was written, and `render_history`'s allow-list —
    whose own comment warns that "unlisted custom types are dropped" — did not
    list it. So the model resumed reading a plain denial and re-planned around
    a choice nobody made, which is the exact confusion the row exists to
    prevent. Asserts the rendered TEXT distinguishes expiry from decision.
    """
    from local_operator.harness.approval import GATE_TIMEOUT_CUSTOM_TYPE
    from local_operator.harness.types import CustomMessage, TextContent
    from local_operator.session.session import _default_convert_to_llm

    rendered = _default_convert_to_llm(
        [
            CustomMessage(
                custom_type=GATE_TIMEOUT_CUSTOM_TYPE,
                attribution="system",
                details={"tool": "bash", "description": "rm -rf build/", "waited_s": 86400.0},
            )
        ]
    )

    assert len(rendered) == 1, "the expiry row was dropped by the allow-list"
    block = rendered[0].content[0]
    assert isinstance(block, TextContent)
    text = block.text
    assert "expired" in text and "not a decision" in text
    assert "bash" in text


def test_a_phone_watching_parks_for_the_configured_day(monkeypatch) -> None:
    """Reachability is not "a terminal is attached".

    Once announcements route by surface, a session the PHONE is watching has
    somebody who can answer the card — so it parks for the configured day
    like an attached viewer, rather than falling back to the short cap that
    exists for a question nobody can ever see.
    """
    from local_operator.session.runtime.owned import OwnedSessionHandle

    class _KindAware:
        def watching_surfaces(self):
            return frozenset({"daemon"})

        def set_record_pending(self, pending):
            return None

    handle = OwnedSessionHandle.__new__(OwnedSessionHandle)
    handle._registrant = _KindAware()  # type: ignore[attr-defined]
    monkeypatch.setattr(
        OwnedSessionHandle, "_unattended_gate_hours", lambda self: 24, raising=False
    )
    # Notifications OFF: proves the park comes from the watching phone and not
    # from an out-of-band toast being available.
    monkeypatch.setattr(
        "local_operator.tui.notify.notifications_enabled", lambda: False, raising=False
    )

    assert handle._gate_timeout_s() == 24 * 3600.0
