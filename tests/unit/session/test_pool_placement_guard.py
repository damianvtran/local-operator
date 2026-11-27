"""The pool launch guard: a home copy must not get a runtime while its member is alive.

``mesh-compute-pool.md`` §6.5 rule 2. A pool ``flush`` holds no fence (§6.1), so the
copy at home is brought up to the member's frontier while the member keeps writing —
and engaging that copy locally during the window is the two-writer split-brain the
whole mobility design exists to prevent, one directory apart rather than one device.

THREE DIRECTIONS ARE ASSERTED, because the guard's cost is in the one that allows:

* a live (``active``/``draining``/``provisioning``) member REFUSES — including when a
  completed flush brought this copy to the frontier that member announced, because
  that frontier is where the member WAS and it keeps appending after it;
* an ``expired`` member, a missing member row, or a drain's rewritten stamp ALLOWS —
  R22 exists so that ``/resume <id>`` works at home with the same id (§6.4);
* the MEMBER ITSELF may engage its own copy (§4.2 point 2: home is never the member),
  and a session no pool produced answers ``""`` at the first read, which is every
  install that never joined a mesh.

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


def _network(root: Path, *members: MemberRecord, self_device: str = HOME) -> None:
    store.save(
        NetworkRecord(
            network_id=NETWORK,
            name="probe",
            self_device_id=self_device,
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


def test_a_fully_synced_copy_of_a_live_member_still_refuses(tmp_path: Path) -> None:
    """THE FRONTIER IS WHERE THE MEMBER WAS, not where it stopped (round 1, MAJOR-1).

    A completed flush leaves exactly this cursor, and the member keeps appending after
    it. So "this copy reached the frontier that member announced" is evidence that a
    copy landed, never that the writer is done — and engaging on it is the second
    writer the guard exists to rule out.
    """
    payload = _rows()
    _stamp(tmp_path)
    _transcript(tmp_path, payload=payload)
    _network(tmp_path, _member("active"))
    _cursor(tmp_path, payload)

    refusal = pool_guard_refusal(tmp_path, SESSION_ID)

    assert refusal
    assert MEMBER in refusal
    assert "active" in refusal


def test_a_malformed_cursor_does_not_raise_out_of_the_guard(tmp_path: Path) -> None:
    """``prefix_bytes`` is peer-shaped JSON: a string there must never be a ValueError.

    The guard's answer to an unreadable cursor is a refusal, not an exception that
    reaches ``engage_runtime`` on every engage until someone deletes the file.
    """
    _stamp(tmp_path)
    _transcript(tmp_path, payload=_rows())
    _network(tmp_path, _member("active"))
    sync.write_replica_cursor(
        tmp_path,
        SESSION_ID,
        {
            "owner_device": MEMBER,
            "cursor": {"prefix_bytes": "abc", "prefix_digest": "sha256:deadbeef"},
        },
    )

    assert pool_guard_refusal(tmp_path, SESSION_ID)


def test_the_member_itself_may_engage_its_own_copy(tmp_path: Path) -> None:
    """§4.2 point 2: home is never the member, so a stamp naming THIS device is the
    copy the member is running — and refusing there locks a device out of its own
    session (round 1, MINOR-4)."""
    _stamp(tmp_path, member=MEMBER)
    _transcript(tmp_path)
    _network(tmp_path, _member("active"), self_device=MEMBER)

    assert pool_guard_refusal(tmp_path, SESSION_ID) == ""


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


def test_a_corrupt_network_record_fails_closed(tmp_path: Path) -> None:
    """ABSENT AND CORRUPT ARE DIFFERENT ANSWERS (round 1, MAJOR-4).

    A record this device never had means it never joined the network, so no row can
    name a live member. A record that is THERE and unparsable is the state in which
    this device least knows whether a writer is live — the handoff guard's asymmetry
    applies, and the venue is the one where proceeding spawns the second writer.
    """
    _stamp(tmp_path)
    _transcript(tmp_path)
    networks = tmp_path / "network" / "networks"
    networks.mkdir(parents=True, exist_ok=True)
    (networks / f"{NETWORK}.json").write_text("{not json", encoding="utf-8")

    refusal = pool_guard_refusal(tmp_path, SESSION_ID)

    assert refusal and "network records" in refusal
    assert f"{NETWORK}.json" in refusal


def test_an_unsafe_network_id_fails_closed(tmp_path: Path) -> None:
    """The id comes from a stamp and becomes a path component: ``..`` must never be read.

    ``store.safe_component`` is the store's own rule; the guard applies it before the
    path is built rather than after, so nothing outside ``networks/`` is opened.
    """
    write_stamp(
        tmp_path,
        MeshStamp(
            session_id=SESSION_ID,
            network_id="../../etc",
            home_device=HOME,
            placement=SessionPlacement(mode="pool", network_id="../../etc", home_device=HOME),
            pool_member=MEMBER,
        ),
    )
    _transcript(tmp_path)

    refusal = pool_guard_refusal(tmp_path, SESSION_ID)

    assert refusal and "network records" in refusal
    assert "../../etc" in refusal


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
