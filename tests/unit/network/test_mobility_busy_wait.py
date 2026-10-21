"""The busy refusal, and the ``--wait`` an OFFLOAD drops on the floor.

Two findings from the session-mobility audit (2026-09-26), both driven on the
two-device rig first:

1. ``--wait`` ON AN OFFLOAD NEVER RE-PROBED. ``guide://network``: "``--wait
   [SECONDS]`` re-checks a conversation whose turn is in flight every five
   seconds, up to the design's thirty minutes". The destination's own flow does
   that (its loop re-asks ``status``/``prepare`` while ``wait_s`` remains), but the
   invite handler that starts an offload's pull passed ``wait_s=0.0``
   unconditionally — so ``lop sessions move <id> --to <peer> --wait 120`` against a
   session parked in a turn answered ``busy`` in 5.09 s with the turn still
   running, while the recall route honoured the same flag.

2. THE MESSAGE WAS THE TOKEN. ``_retire_local_runtime`` forwards the runtime's
   answer as the user-facing sentence, and for a session parked in a turn
   ``retire_now`` answers the bare reason word ``busy`` — so the CLI printed
   ``{"code": "busy", "message": "busy"}``: a machine word in the one field a
   person reads, with no cause and no remedy, beside every sibling refusal that is
   a full sentence.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from local_operator.mobile import attach_client
from local_operator.network import mobility
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures and helpers
    SESSION,
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — the fixture `pair` reaches for
    _pair_settled,
    devices,
)


class _Record:
    """The discovery record ``_retire_local_runtime`` needs, and nothing else."""

    capabilities = (mobility._EXCLUSIVE_MOVE_CAPABILITY,)


class _BusyClient:
    """An attach client whose runtime re-asks its predicate and says ``busy``."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    async def connect(self, record: Any, session_id: str) -> None:
        return None

    async def retire_now(self, *, exclusive: bool = False) -> str:
        return "busy"

    def close(self) -> None:
        return None


class _FlakyRetire:
    """The owner's runtime: busy for the first call, cold afterwards."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, root: Path, session_id: str, *, deadline_s: float = 0) -> dict[str, Any]:
        self.calls += 1
        if self.calls == 1:
            # A LITERAL, not the module under test's own renderer: this stub is
            # the OWNER answering, and a stub that called into the code under
            # test would make the red-on-old run fail for the wrong reason.
            return {"result": "busy", "sentence": "this session is working right now"}
        return {"result": "cold", "sentence": ""}


def test_a_reason_token_becomes_the_sentence_a_person_needs(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``busy`` is a reason CODE; the refusal's ``message`` is what a person reads."""
    server = request.getfixturevalue("pair")[0]
    monkeypatch.setattr(attach_client, "find_runtime_record", lambda root, sid: (_Record(), 0))
    monkeypatch.setattr(attach_client, "AttachClient", _BusyClient)

    outcome = mobility._retire_local_runtime(server.root, SESSION)

    assert outcome["result"] == "busy", outcome
    sentence = str(outcome["sentence"])
    assert sentence != "busy", "the reason token reached the user as the whole message"
    assert " " in sentence and "--wait" in sentence, sentence
    # AND THE REMEDY NAMES ITS SURFACE (the /move papercut, 2026-09-29): ``--wait``
    # is the CLI verb's flag, and the composers that echo this sentence refuse it —
    # a bare offer sent a TUI reader into a second refusal about an unknown flag.
    # Pinned as the WHOLE sentence: these words are the product, and the one thing
    # that must not drift is where the remedy tells its reader to run it.
    assert sentence == (
        "this session is working right now, so nothing was moved; try again when the "
        "turn finishes; from a shell, pass --wait <seconds> to re-check"
    ), sentence
    # AND A RUNTIME THAT EXPLAINS ITSELF IN PROSE IS NEVER REWRITTEN: the contract is
    # "the owner's idle reason, verbatim" (MOVE_REFUSAL_CODES).
    theirs = "a background job is still running in this session"
    assert mobility._busy_sentence(theirs) == theirs
    assert mobility._busy_sentence("") == ""


def test_a_half_moved_session_is_legible_from_the_refusal(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state that blocks every route has to say what it is.

    Measured (probe 9): after the destination's relay was killed mid-copy, seven
    retries over ten minutes read "a handoff of that conversation is already in
    progress on this device, so nothing was changed" — the same sentence a healthy
    in-flight move produces, with nothing to distinguish the two and no way to tell
    whether bytes had been committed anywhere.

    The entry is written with THIS relay's ``instance_id`` on purpose: an entry
    from another instance is what §6.5's recovery table rolls back, and this cell is
    about the refusal a person sees while one is live.
    """
    import time as time_mod

    from local_operator.session.placement import write_handoff_entry

    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    write_handoff_entry(
        server_a.root,
        SESSION,
        {
            "role": "source",
            "phase": "prepared",
            "to_device": server_b.identity.device_id,
            "mode": "move",
            "instance_id": str(server_a.instance_id),
            "at": time_mod.time() - 42.0,
        },
    )

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is False and result["code"] == "in_progress", result
    message = str(result["message"])
    assert "phase: prepared" in message, message
    assert "ago)" in message, f"the refusal does not say how long it has been stuck: {message}"
    assert "handing_off" not in message
    # AND IT NAMES THE WAY OUT, which is the part a person cannot derive: an entry
    # written by this device's LIVE relay is skipped by every reconcile, so retrying
    # never clears it and no other verb does either — restarting this device's relay
    # is what makes the writer instance stale, after which the next attempt rolls the
    # entry back (review round 1's ruling; the abandon verb is the follow-up).
    assert "lop network restart" in message, message
    # AND NOTHING MOVED, which is the half-move's own guarantee.
    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert not (server_b.root / "sessions" / SESSION).exists()


def test_a_handoff_entry_without_a_timestamp_is_not_given_a_fake_age(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sentence whose job is legibility may not invent a number.

    ``float(entry.get("at") or 0.0)`` rendered ~1.79e9 seconds for an entry with no
    ``at`` (review round 1, NIT): every writer on this head stamps the field, so the
    branch is defensive — and defensiveness that prints an epoch as a duration is how
    a future reader learns to distrust the whole clause.
    """
    import time as time_mod

    from local_operator.session.placement import write_handoff_entry

    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    entry = {
        "role": "source",
        "phase": "prepared",
        "to_device": server_b.identity.device_id,
        "mode": "move",
        "instance_id": str(server_a.instance_id),
    }
    write_handoff_entry(server_a.root, SESSION, entry)

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["code"] == "in_progress", result
    message = str(result["message"])
    assert "started at an unknown time" in message, message
    # And the helper itself, for the shapes a journal reader can hand it: a bool is
    # an int in Python, and ``True`` is not a timestamp either.
    assert mobility._handoff_age({"at": True}) == "started at an unknown time"
    assert mobility._handoff_age({"at": "yesterday"}) == "started at an unknown time"
    assert (
        str(int(time_mod.time())) not in message
    ), f"the refusal printed a wall-clock epoch as a duration: {message}"


def test_a_malformed_wait_is_read_as_no_wait_at_all() -> None:
    """A peer's ``wait_s`` is data, and data this side cannot read is not an error.

    Every sibling field in the invite is read with ``or <default>``; this one raised
    out of the handler for a non-numeric value (review round 1, NIT). The clamp is
    asserted too, because a peer must not be able to ask this side to re-probe for a
    week.
    """
    assert mobility._wait_seconds(None) == 0.0
    assert mobility._wait_seconds("soon") == 0.0
    assert mobility._wait_seconds([1, 2]) == 0.0
    # ``True`` is not a duration, and ``float(True)`` would read as one second.
    assert mobility._wait_seconds(True) == 0.0
    assert mobility._wait_seconds(float("nan")) == 0.0
    assert mobility._wait_seconds(float("inf")) == 0.0
    assert mobility._wait_seconds("30") == 30.0
    assert mobility._wait_seconds(-5) == 0.0
    assert mobility._wait_seconds(10_000) == mobility.MOVE_MAX_WAIT_S


def test_an_offload_with_a_wait_re_probes_a_busy_source(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--wait N`` on an offload is a promise the destination has to keep.

    The re-probe lives in the destination's flow (it is the side that asks), so the
    ``wait_s`` the user typed has to travel with the invite. This cell drives the
    real invitation path in both relays: A owns the session and invites B, B pulls,
    and A's first retire answer is ``busy``.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair_settled(both, monkeypatch, role="admin", settings=server_b.settings)
    _owned_session(server_a)
    flaky = _FlakyRetire()
    monkeypatch.setattr(mobility, "_retire_local_runtime", flaky)
    monkeypatch.setattr(mobility, "MOVE_WAIT_POLL_S", 0.2)

    result = _move(
        server_a, SESSION, to=server_b.identity.device_id, wait_s=6.0, monkeypatch=monkeypatch
    )

    assert result["ok"] is True, result
    assert flaky.calls >= 2, "the destination never re-asked: the wait did not travel"
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert not (server_a.root / "sessions" / SESSION).exists()
