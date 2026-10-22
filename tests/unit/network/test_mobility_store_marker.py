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

import warnings
from pathlib import Path
from typing import Any

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
    _pair_settled,
    devices,
)


def _handed_over(server_a: Any, server_b: Any, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """``move`` the session to B, re-asking ONCE if the link answered ``unreachable``.

    WHY THE RE-ASK EXISTS (CI run 39838493282, 3.13 shard 2, head ``158e0f6a``).
    The cell went red with ``ok=False`` and the transport's own code
    ``unreachable`` — "that device is unreachable (no answer on its link);
    nothing was changed" — after the whole ``MOVE_OP_DEADLINE_S`` (90 s; the cell
    itself ran 93 s), so the promotion this file verifies was never reached and
    the marker was never the question. The cell's setup is byte-for-byte the one
    its passing siblings use (same ``_pair_settled``, same joined settings), and
    the destination acks an invite IMMEDIATELY on a thread of its own — so an
    answer that never came is a MESH LINK that answered nothing, not a move that
    silently did nothing.

    A re-ask is safe for exactly the reason the code names: ``unreachable`` is
    the family's no-change refusal, so the product's PLACEMENT is back where it
    started. That claim is ASSERTED below rather than trusted — a first attempt
    that reported nothing changed while it had already adopted a copy or
    tombstoned the source is the mirror image of a success that changes nothing,
    and this is where it would surface instead of being papered over by the
    second ask.

    ONE SUB-CASE IS NOT HEALED, and it is named here so the sentence above is not
    read as "the re-ask covers everything": when the invite WAS delivered and
    only its ack was lost, the destination has already written its ``prepared``
    entry (and may have started its pull), and the source's own in-flight guard
    refuses the second ask. That is the right outcome — a real half-state, said
    loudly — and the cell still fails on it rather than pretending.

    The silent link is closed first because ``RelayServer._link_for`` hands an op
    any link that is both ``alive`` and not ``closing``, and a half-open socket
    is both until ``LINK_IDLE_S`` (120 s) — longer than the move's 90 s deadline.
    Such a link is never re-dialled, because it looks usable. Closing it is the
    relay's own remedy for that state, done here explicitly so the test does not
    have to sit through the idle timeout, and the next ask dials fresh.

    A genuinely broken transport still fails this cell: the second ask returns
    ``unreachable`` too, and the caller asserts on it.

    AND A NOTE ON ``_pair_settled`` (QA round 1, Q2): the helper that resolves
    this cell's red also waits out the dying pairing link, so a regression that
    let ``_pair_settled`` return early would be healed here by the second ask.
    That is deliberate — this file is about the store MARKER, and the transport
    guarantee has its own cells in ``test_mobility.py``, which move without this
    retry and would go red on their own.
    """
    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)
    if result.get("ok") is True or result.get("code") != "unreachable":
        return result
    closed = 0
    # SAID OUT LOUD, not absorbed. The retry is what a user does with a
    # no-change refusal, but a cell that silently retries turns a rising rate
    # of alive-silent links into a slower shard and nothing else — and that is
    # exactly the defect this cell went red for (CI 39838493282). A warning
    # keeps it visible without changing a single assertion.
    warnings.warn(
        "the hand-away answered unreachable and was re-asked once on a fresh link "
        f"({result.get('message')}) — see CI 39838493282 for the cell this answers",
        RuntimeWarning,
        stacklevel=2,
    )
    assert not (server_b.root / "sessions" / SESSION).exists(), (
        "the move answered 'unreachable … nothing was changed' and the destination "
        "adopted a copy anyway — the failure report is a lie, which is the shape this "
        "file exists to rule out"
    )
    assert (server_a.root / "sessions" / SESSION).is_dir(), (
        "the move answered 'nothing was changed' and the source no longer holds the "
        "conversation it says it did not touch"
    )
    assert SESSION not in read_tombstones(server_a.root), (
        "the move answered 'nothing was changed' and the source tombstoned the id anyway, "
        "so the conversation is unreachable from the device that still holds it"
    )
    for link in list(server_a.links.values()):
        if link.device_id == server_b.identity.device_id:
            link.close()
            closed += 1
    if not closed:
        # A REFUSAL WITH NO LINK IS NOT THE EVENT THIS HELPER EXPLAINS. The CI red
        # was a link that was alive and answered nothing; if there is nothing to
        # close, the second ask may still succeed, but the reason is not the one
        # written above — say so in the log instead of letting a future
        # spurious-refusal bug be retried into a silent pass (QA round 1, Q1).
        # A warning rather than an assertion on purpose: a link that was already
        # evicted is a legitimate state, and an assertion here would rebuild the
        # timing dependence this file just removed.
        warnings.warn(
            "the hand-away answered unreachable with no link to the destination to close, "
            "so a silent link does not explain it — re-asking anyway",
            RuntimeWarning,
            stacklevel=2,
        )
    return _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)


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
    _pair_settled(both, monkeypatch, role=role, settings=server_b.settings)
    _owned_session(server_a)
    result = _handed_over(server_a, server_b, monkeypatch)
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


def test_an_archived_session_arrives_archived(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hiding a conversation is a property of the conversation, not of one device.

    Review round 1, MINOR 1: the archive index is device-scoped
    (``session/archived.py``) and nothing on the move path wrote it, so a session
    archived on the device it left reappeared in the default listing of the device
    it arrived on — the operator named archiving explicitly as something that must
    hold on the remote session.

    The bit rides in the staging ``ready.json`` rather than in memory because that
    file is the one carrier that survives the crash path (§6.5 row 5 adopts from
    it with the owner possibly gone).
    """
    from local_operator.session.archived import archived_ids, set_archived

    server_a, server_b, _host, _port = request.getfixturevalue("pair")
    # ``settings=server_b.settings`` or the joiner publishes a record nobody can
    # dial and the move answers ``unreachable`` — see every other cell in this file.
    _pair_settled(
        request.getfixturevalue("pair"),
        monkeypatch,
        role="admin",
        settings=server_b.settings,
    )
    _owned_session(server_a)
    assert set_archived(server_a.root, SESSION, True) is True
    assert SESSION in archived_ids(server_a.root), "the archive must be visible first"

    result = _move(server_a, SESSION, to=server_b.identity.device_id, monkeypatch=monkeypatch)

    assert result["ok"] is True, result
    assert (server_b.root / "sessions" / SESSION).is_dir()
    assert SESSION in archived_ids(server_b.root), (
        "the device that adopted the conversation does not know it was archived, so it "
        "shows up in every default listing there"
    )
    # AND THE SOURCE IS NOT LEFT CLAIMING IT: read_archived prunes ids whose
    # directory is gone, so the entry on the device that handed it away resolves to
    # nothing without any cross-device coordination.
    assert SESSION not in archived_ids(server_a.root)


def test_a_keep_copy_does_not_inherit_the_archive(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fork mints a NEW id, and a copy the user just asked for is not hidden.

    The deliberate half of the archive fix: ``--keep`` leaves the source running and
    the destination under a new id, so inheriting the archive would create a
    conversation the user asked for and cannot see. Pinned so "the archive travels"
    cannot silently widen into "every copy arrives archived".
    """
    from local_operator.session.archived import archived_ids, set_archived

    server_a, server_b, _host, _port = request.getfixturevalue("pair")
    _pair_settled(
        request.getfixturevalue("pair"),
        monkeypatch,
        role="admin",
        settings=server_b.settings,
    )
    _owned_session(server_a)
    set_archived(server_a.root, SESSION, True)

    result = _move(
        server_a, SESSION, to=server_b.identity.device_id, keep=True, monkeypatch=monkeypatch
    )

    assert result["ok"] is True, result
    new_id = str(result["new_session_id"])
    assert new_id != SESSION
    assert (server_b.root / "sessions" / new_id).is_dir()
    assert new_id not in archived_ids(server_b.root), (
        "the fork of an archived conversation arrived archived, so the copy the user "
        "asked for is hidden on the device they asked for it on"
    )
    assert SESSION in archived_ids(server_a.root), "the source keeps its own state"


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
