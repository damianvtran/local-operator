"""Carried auto authority: the stamp is a carry, the grant is the authority.

``serving._carried_auto_authority`` decides whether a session whose ``mesh.json``
says ``unattended`` may actually be constructed unattended ON THIS DEVICE. Both
halves must hold — the stamp (the operator's intent, traveled with the session)
and THIS device's member row for the originating device (the grant) — and either
one alone must withhold auto:

* stamp alone would be a session-writable file granting itself authority, the
  substitution the approval-authority design rejects;
* grant alone says nothing about this session.

Reading at engage is also what makes a REVOCATION effective at the next engage,
which the removed-member and capability-withdrawn cells pin. Every unreadable
input fails closed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.network import store as network_store
from local_operator.network.types import MemberRecord, NetworkRecord
from local_operator.session.placement import MeshStamp
from local_operator.session.runtime.serving import _carried_auto_authority

_NETWORK = "n_0123456789abcdef0123456789abcdef"
_ORIGIN = "d_" + "a" * 32
_HOME = "d_" + "b" * 32


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A private config root for one test (the network suite's fixture, local)."""
    return tmp_path


def _stamp(root: Path, session_id: str, *, unattended: bool = True, origin: str = _ORIGIN) -> None:
    from local_operator.session.placement import write_stamp

    write_stamp(
        root,
        MeshStamp(
            session_id=session_id,
            network_id=_NETWORK,
            home_device=_HOME,
            origin={"kind": "user", "source_device": origin},
            unattended=unattended,
        ),
    )


def _record(root: Path, *, capabilities: list[str], removed: bool = False) -> None:
    member = MemberRecord(
        device_id=_ORIGIN,
        name="my mac",
        role="drive",
        capabilities=list(capabilities),
        removed_at=12345.0 if removed else None,
    )
    network_store.save(
        NetworkRecord(
            network_id=_NETWORK,
            name="home",
            self_device_id=_HOME,
            self_role="admin",
            members=[member],
        ),
        root=root,
    )


def test_the_stamp_round_trips_the_carry_and_absent_reads_false(root: Path) -> None:
    """Schema side: True round-trips, an ABSENT key reads False (fail closed).

    The absent-key half is the older-build direction: a stamp written before this
    field existed must read as "no carry" — never as True, which a truthy default
    would have made it — because the wrong answer here runs a session unattended.
    """
    from local_operator.session.placement import MeshStamp as Stamp

    carried = Stamp(session_id="s1", unattended=True)
    assert carried.to_json()["unattended"] is True
    back = Stamp.from_json(carried.to_json())
    assert back is not None and back.unattended is True
    old = dict(carried.to_json())
    del old["unattended"]
    parsed = Stamp.from_json(old)
    assert parsed is not None and parsed.unattended is False


def test_no_stamp_means_no_carry(root: Path) -> None:
    assert _carried_auto_authority(root, "no-such-session") is False


def test_a_carried_stamp_without_the_grant_is_withheld(root: Path) -> None:
    """Both halves required: the stamp says ask, the row says no — answer no."""
    _stamp(root, "s_nogrant")
    _record(root, capabilities=["prompt"])
    assert _carried_auto_authority(root, "s_nogrant") is False


def test_the_grant_on_the_origin_row_unlocks_it(root: Path) -> None:
    _stamp(root, "s_granted")
    _record(root, capabilities=["prompt", "unattended"])
    assert _carried_auto_authority(root, "s_granted") is True


def test_a_removed_member_withholds_it_at_the_next_engage(root: Path) -> None:
    """Revocation effective at the next engage: the row's tombstone wins."""
    _stamp(root, "s_revoked")
    _record(root, capabilities=["prompt", "unattended"], removed=True)
    assert _carried_auto_authority(root, "s_revoked") is False


def test_a_missing_record_withholds_it(root: Path) -> None:
    """A stamp naming a network this device does not hold grants nothing."""
    _stamp(root, "s_norecord")
    assert _carried_auto_authority(root, "s_norecord") is False


def test_a_stamp_without_the_carry_is_withheld_even_with_the_grant(root: Path) -> None:
    _stamp(root, "s_plain", unattended=False)
    _record(root, capabilities=["prompt", "unattended"])
    assert _carried_auto_authority(root, "s_plain") is False


def test_the_origin_falls_back_to_the_home_device(root: Path) -> None:
    """A stamp with no ``source_device`` checks the row for its owner.

    Both spellings name "the device the session came from", and a stamp written
    by a path that records only ``home_device`` must not silently skip the grant
    check (a skip here is the check being asked of nobody).
    """
    from local_operator.session.placement import write_stamp

    write_stamp(
        root,
        MeshStamp(
            session_id="s_homeonly",
            network_id=_NETWORK,
            home_device=_ORIGIN,
            origin={},
            unattended=True,
        ),
    )
    _record(root, capabilities=["unattended"])
    assert _carried_auto_authority(root, "s_homeonly") is True
