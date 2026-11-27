"""The pool launch guard: a home copy must not get a runtime while its member is alive.

``mesh-compute-pool.md`` §6.5 rule 2. A pool ``flush`` holds no fence (§6.1), so the
copy at home is brought up to the member's frontier while the member keeps writing —
and engaging that copy locally during the window is the two-writer split-brain the
whole mobility design exists to prevent, one directory apart rather than one device.

THREE DIRECTIONS ARE ASSERTED, because the guard's cost is in the one that allows:

* a live member whose announced frontier this device has not reached REFUSES, and says
  which member and why (the operator's next move depends on that);
* an ``expired`` member, a missing member row, or a cold fully-synced copy ALLOWS —
  R22 exists so that ``/resume <id>`` works at home with the same id (§6.4);
* a session no pool produced answers ``""`` at the first read (one ``stat``), which is
  every install that never joined a mesh.

The engage half is exercised through the REAL ``launch.engage_runtime``: the guard runs
before a candidate is spawned, so a refusal is observable without starting a runtime.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from local_operator.network import store, sync
from local_operator.network.types import MemberRecord, NetworkRecord
from local_operator.session.placement import (
    MeshStamp,
    SessionPlacement,
    pool_guard_refusal,
    read_stamp,
    write_stamp,
)
from local_operator.session.runtime import launch

SESSION_ID = "abc123def456"
MEMBER = "dev_pod_1"
HOME = "dev_home_1"
NETWORK = "n_test0001"


# ---------------------------------------------------------------------------
# Fixtures built from the product's own writers
# ---------------------------------------------------------------------------


def _rows(count: int = 3) -> bytes:
    return b"".join(
        json.dumps({"id": f"e{index}", "type": "message", "content": f"row {index}"}).encode()
        + b"\n"
        for index in range(count)
    )


def _transcript(root: Path, payload: bytes | None = None) -> Path:
    directory = root / "sessions" / SESSION_ID
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "transcript.jsonl"
    path.write_bytes(_rows() if payload is None else payload)
    return path


def _stamp(root: Path, *, mode: str = "pool", member: str = MEMBER) -> None:
    write_stamp(
        root,
        MeshStamp(
            session_id=SESSION_ID,
            network_id=NETWORK,
            home_device=HOME,
            placement=SessionPlacement(
                mode=mode,  # type: ignore[arg-type]
                network_id=NETWORK,
                home_device=HOME,
                stamp_revision=1,
            ),
            pool_member=member,
        ),
    )


def _network(root: Path, *members: MemberRecord) -> None:
    store.save(
        NetworkRecord(
            network_id=NETWORK,
            name="probe",
            self_device_id=HOME,
            members=list(members),
        ),
        root,
    )


def _member(lifecycle: str) -> MemberRecord:
    return MemberRecord(
        device_id=MEMBER, kind="pool", lifecycle=lifecycle, name="pod"  # type: ignore[arg-type]
    )


def _cursor(root: Path, payload: bytes, *, member: str = MEMBER) -> None:
    """The cursor a completed flush leaves: the frontier, and the bytes it covers."""
    sync.write_replica_cursor(
        root,
        SESSION_ID,
        {
            "owner_device": member,
            "cursor": {
                "prefix_bytes": len(payload),
                "prefix_digest": "sha256:" + hashlib.sha256(payload).hexdigest(),
                "frontier": "e2",
            },
            "last_synced_at": 1.0,
        },
    )


# ---------------------------------------------------------------------------
# The live-member refusal (two writers)
# ---------------------------------------------------------------------------


def test_a_live_member_with_no_cursor_refuses(tmp_path: Path) -> None:
    _stamp(tmp_path)
    _transcript(tmp_path)
    _network(tmp_path, _member("active"))

    refusal = pool_guard_refusal(tmp_path, SESSION_ID)

    assert refusal, "a live pool member with no synced frontier must not be engaged over"
    assert MEMBER in refusal, "the sentence must name the member holding the conversation"
    assert "active" in refusal
    assert "/resume" not in refusal  # shown at engage, not by a listing


@pytest.mark.parametrize("lifecycle", ["draining", "provisioning"])
def test_a_member_that_is_leaving_or_still_forming_refuses(tmp_path: Path, lifecycle: str) -> None:
    _stamp(tmp_path)
    _transcript(tmp_path)
    _network(tmp_path, _member(lifecycle))

    assert pool_guard_refusal(tmp_path, SESSION_ID)


def test_a_cursor_that_outlived_the_transcript_refuses(tmp_path: Path) -> None:
    """A compaction rewrites the file, so a cursor longer than the file is stale.

    That is the corruption a byte-offset cursor would cause, caught at the guard
    instead: refusing costs a sentence, proceeding writes a second trajectory over a
    copy that no longer holds what the cursor claims.
    """
    _stamp(tmp_path)
    long_payload = _rows(12)
    _transcript(tmp_path, payload=_rows(3))
    _network(tmp_path, _member("active"))
    _cursor(tmp_path, long_payload)

    assert pool_guard_refusal(tmp_path, SESSION_ID)


def test_a_cursor_from_another_device_refuses(tmp_path: Path) -> None:
    """The frontier that matters is the one THIS member announced, not any owner's."""
    payload = _rows()
    _stamp(tmp_path)
    _transcript(tmp_path, payload=payload)
    _network(tmp_path, _member("active"))
    _cursor(tmp_path, payload, member="dev_someone_else")

    assert pool_guard_refusal(tmp_path, SESSION_ID)


# ---------------------------------------------------------------------------
# The allowing answers (R22's whole point)
# ---------------------------------------------------------------------------


def test_an_expired_member_allows_the_resume(tmp_path: Path) -> None:
    _stamp(tmp_path)
    _transcript(tmp_path)
    _network(tmp_path, _member("expired"))

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""


def test_a_member_row_that_is_gone_allows_the_resume(tmp_path: Path) -> None:
    """No row means nothing this device can call live — and a tombstone may be pruned."""
    _stamp(tmp_path)
    _transcript(tmp_path)
    _network(tmp_path)

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""


def test_a_cold_fully_synced_copy_allows_the_resume(tmp_path: Path) -> None:
    """§6.5: "a cold, fully-synced copy *is* engageable, which is the whole point of R22"."""
    payload = _rows()
    _stamp(tmp_path)
    path = _transcript(tmp_path, payload=payload)
    _network(tmp_path, _member("active"))
    _cursor(tmp_path, payload)

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""
    # The bytes the cursor describes are the bytes on disk: that is what makes the
    # answer a fact about the copy rather than about a file that once matched.
    assert path.read_bytes().startswith(payload)


def test_a_rewritten_placement_lifts_the_guard(tmp_path: Path) -> None:
    _transcript(tmp_path)
    _network(tmp_path, _member("active"))
    _stamp(tmp_path, mode="local", member="")

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""


@pytest.mark.parametrize("mode", ["local", "peer"])
def test_a_session_no_pool_produced_is_not_guarded(tmp_path: Path, mode: str) -> None:
    _transcript(tmp_path)
    _network(tmp_path, _member("active"))
    _stamp(tmp_path, mode=mode, member="")

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""


def test_no_stamp_at_all_is_not_guarded(tmp_path: Path) -> None:
    _transcript(tmp_path)

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""
    assert read_stamp(tmp_path, SESSION_ID) is None


def test_a_pool_stamp_naming_no_member_is_not_guarded(tmp_path: Path) -> None:
    """A legacy or partial stamp cannot name a live member, so there is no second writer."""
    _stamp(tmp_path, member="")
    _network(tmp_path, _member("active"))

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""


def test_a_corrupt_network_record_reads_as_no_row(tmp_path: Path) -> None:
    """The STORE's rule, not a new one: an unparsable record is set aside, so the
    network it described is no longer listed and it cannot name a live member.

    Refusing instead would block every pool resume behind a file whose remedy
    (delete it) also destroys the membership.
    """
    _stamp(tmp_path)
    _transcript(tmp_path)
    (tmp_path / "network" / "networks").mkdir(parents=True, exist_ok=True)
    (tmp_path / "network" / "networks" / f"{NETWORK}.json").write_text(
        "{not json", encoding="utf-8"
    )

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""


def test_a_store_read_that_fails_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The handoff guard's asymmetry, on the one path that can still raise."""
    _stamp(tmp_path)
    _transcript(tmp_path)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise OSError("input/output error")

    monkeypatch.setattr(store, "load", refuse)

    refusal = pool_guard_refusal(tmp_path, SESSION_ID)

    assert refusal and "network records" in refusal
    assert "input/output error" in refusal


# ---------------------------------------------------------------------------
# The engage path itself
# ---------------------------------------------------------------------------


def test_engage_runtime_refuses_a_pool_session_before_spawning(tmp_path: Path) -> None:
    _stamp(tmp_path)
    _transcript(tmp_path)
    _network(tmp_path, _member("active"))

    with pytest.raises(launch.RuntimeStartupError) as raised:
        asyncio.run(
            launch.engage_runtime(
                SESSION_ID,
                str(tmp_path),
                launch.WarmErrand(command_id="cmd-1"),
                config_dir=tmp_path,
            )
        )

    assert MEMBER in str(raised.value)


def test_the_engage_guard_allows_an_expired_member_to_its_own_sentence(tmp_path: Path) -> None:
    """The other side of the same call site: nothing here raises for an expired member.

    The engage itself is not run past the guard (it would spawn a runtime), so this
    asserts the two facts the guard owns: its answer is empty and the session it names
    is the one ``/resume`` would open.
    """
    _stamp(tmp_path)
    _transcript(tmp_path)
    _network(tmp_path, _member("expired"))

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""
    assert (tmp_path / "sessions" / SESSION_ID / "transcript.jsonl").is_file()


# ---------------------------------------------------------------------------
# The stamp field the guard reads must survive a round trip
# ---------------------------------------------------------------------------


def test_the_pool_member_survives_a_stamp_round_trip(tmp_path: Path) -> None:
    """A field the parse forgets is a live member the guard can no longer see — i.e.
    the split-brain guard switched off by a parse."""
    _stamp(tmp_path)
    stamp = read_stamp(tmp_path, SESSION_ID)

    assert stamp is not None
    assert stamp.pool_member == MEMBER
    assert stamp.placement.mode == "pool"
    # ... and through the nested-placement reconstruction branch: a stamp that names
    # its owner at the top level but omits it in the placement still keeps the member.
    payload = json.loads((tmp_path / "sessions" / SESSION_ID / "mesh.json").read_text())
    payload["placement"].pop("home_device")
    rebuilt = MeshStamp.from_json(payload)
    assert rebuilt is not None
    assert rebuilt.pool_member == MEMBER
    assert rebuilt.placement.home_device == HOME


def test_a_stamp_without_the_field_parses_leniently() -> None:
    """Every stamp written before this field existed omits it; none of them is a pool."""
    rebuilt = MeshStamp.from_json(
        {"version": 1, "session_id": SESSION_ID, "placement": {"mode": "peer"}}
    )

    assert rebuilt is not None
    assert rebuilt.pool_member == ""
    # The placement itself is untouched by the addition: it is the shape the
    # catalogue row publishes verbatim, so its field list must not move.
    assert set(SessionPlacement().to_json()) == {
        "mode",
        "network_id",
        "home_device",
        "policy",
        "stamp_revision",
    }
