"""A session the mesh ADOPTS must be removable by the device that adopted it.

WHY THIS FILE EXISTS (session-mobility audit, 2026-09-26). ``remove_session_dir``
— the ONE ``rmtree`` of a session directory in this codebase — refuses a store that
carries no ``.local-operator-store`` marker, and that marker is written where a
session is CREATED (``session_factory`` calls ``cleanup.mark_store``), not where one
is ADOPTED. A destination that has never created a session of its own — a freshly
paired second machine, which is the case mobility exists for — therefore adopted a
directory it could never delete.

Measured on the two-device loopback rig before the fix: A handed a conversation to
B, B later handed it back on a recall, and the source-side commit wrote its
tombstone, called ``remove_session_dir``, logged the refusal (``cleanup="pending"``)
and LEFT THE DIRECTORY BEHIND — the same id then live on two devices, each listing
it as a local conversation, which is exactly INV-1. ``reconcile`` retries that
removal, and the retry failed for the same reason, forever.

Two cells, and both fail on the pre-fix code for the reason their docstring names:
the ADOPTION marks the store, and the hand-back REMOVES the copy the destination
adopted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.network.projection import read_tombstones
from local_operator.session.cleanup import (
    MESH_MOVE_POLICY,
    remove_session_dir,
    store_marker_path,
)
from tests.unit.network.test_mobility import (  # noqa: F401 — fixtures and helpers
    SESSION,
    Devices,
    _move,
    _owned_session,
    pair,
)
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — the fixtures `pair` reaches for
    _pair,
    devices,
)


def _hand_to_peer(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch, *, role: str = "admin"
) -> tuple[Path, Path]:
    """A owns a session; B takes it. Returns (A's store marker, B's store marker).

    ``role="admin"`` because ``move`` is deliberately NOT part of ``drive`` (the
    CLI's own comment: "the default `drive` role cannot move, delete or borrow a
    login") — the refusal and its remedy have their own cell in
    ``test_mobility.py``. This file is about what a destination does with a
    session once it is allowed to take one.

    The fixture is fetched by NAME rather than declared as a parameter so this
    module can import ``pair`` (and, for it, ``devices``) without flake8 reading
    the import as an unused redefinition.
    """
    both: Devices = request.getfixturevalue("pair")
    server_a, server_b, _host, _port = both
    _pair(both, monkeypatch, role=role, settings=server_b.settings)
    _owned_session(server_a)
    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    assert result["ok"] is True, result
    return (
        store_marker_path(server_a.root / "sessions"),
        store_marker_path(server_b.root / "sessions"),
    )


def test_a_promoted_session_marks_the_store_that_adopted_it(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The destination becomes a session store the moment it holds a session."""
    server_b = request.getfixturevalue("pair")[1]
    marker_b = store_marker_path(server_b.root / "sessions")
    assert not marker_b.is_file(), (
        "the fixture must start from a store that has never been marked — that is the "
        "state a freshly paired second machine is in"
    )

    _hand_to_peer(request, monkeypatch)

    adopted = server_b.root / "sessions" / SESSION
    assert adopted.is_dir()
    assert marker_b.is_file(), (
        "the destination adopted a session without marking its store, so nothing it "
        "adopts can ever be removed: `remove_session_dir` refuses an unmarked store"
    )
    # AND THE GUARD AGREES, which is the property the recall and a later delete rest
    # on. Asserted through the real removal path rather than on the marker's bytes.
    assert (
        remove_session_dir(
            adopted,
            config_dir=server_b.root,
            policy=MESH_MOVE_POLICY,
            reason="test: the adopted copy must be removable",
            actor="test",
        )
        is True
    ), "the directory the destination adopted is not removable by the destination"
    assert not adopted.exists()


def test_a_recalled_session_is_no_longer_tombstoned_here(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tombstone and a directory for one id are never BOTH true.

    Measured on the rig (probe 8): A handed `bc371569005d` to B, recalled it, and
    then held the directory with ``mesh.json home_device: A`` while its tombstone
    index still said that id had moved to B. That is the state the audit's wedged
    session was found in, and the three surfaces disagreed about it: a local
    ``--resume`` routed the user to the peer, ``move --to peer`` answered "already
    lives on this device", ``move --to local`` answered ``already_local``. A
    tombstone is the statement "this device handed it away", so the promote — the
    instant a session becomes this device's directory again — is where it stops
    being true.
    """
    server_a, _server_b, _host, _port = request.getfixturevalue("pair")
    _hand_to_peer(request, monkeypatch)
    assert SESSION in read_tombstones(
        server_a.root
    ), "the hand-away must be tombstoned first, or this cell tests nothing"

    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)
    assert recalled["ok"] is True, recalled

    assert (server_a.root / "sessions" / SESSION).is_dir()
    assert SESSION not in read_tombstones(
        server_a.root
    ), "the device holds the session again but its tombstone still sends the id elsewhere"


def test_a_hand_back_removes_the_copy_the_destination_adopted(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After the hand-back the id lives on ONE device — INV-1, not a listing.

    This is the rig's measured failure as one property. On the pre-fix code the
    recall succeeds, A holds the conversation, and B keeps a directory whose
    ``mesh.json`` says the session lives on A — two copies of one id, both
    offered as local conversations by their own device's listing.
    """
    server_a, server_b, _host, _port = request.getfixturevalue("pair")
    _hand_to_peer(request, monkeypatch)
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert not (server_a.root / "sessions" / SESSION).exists()

    # A pulls it home: `lop sessions move <id> --to local` on A.
    recalled = _move(server_a, SESSION, monkeypatch=monkeypatch)
    assert recalled["ok"] is True, recalled

    assert (server_a.root / "sessions" / SESSION).is_dir(), "the conversation did not come home"
    assert not (server_b.root / "sessions" / SESSION).exists(), (
        "the device that handed the session back still holds a copy of it: two devices, "
        "one id, and each listing calls its copy a local conversation (INV-1)"
    )
    assert SESSION in read_tombstones(server_b.root), (
        "the hand-back left no tombstone on the device that gave the session away, so a "
        "later command naming that id here cannot be routed to the device that now owns it"
    )
