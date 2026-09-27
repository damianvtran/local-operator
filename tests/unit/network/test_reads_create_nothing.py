"""A READ of the network plane must never be the reason it exists on disk.

THE BUG THIS FILE PINS, measured on an isolated config root on 2026-09-27:
``GET /v1/desktop/commands`` → ``server.utils.desktop_commands.command_catalogue``
→ ``command_argument_words`` → ``network.peers.known_peer_names`` →
``network.store.list_networks`` → ``store.networks_dir``, whose ``mkdir`` ran on
the way to a LISTING. A machine that had never joined a network therefore grew
``<config>/network`` and ``<config>/network/networks`` from a GET, and
``server.utils.desktop_mesh.has_any_network`` — an honest ``is_dir`` probe, and the
reason the mesh reads cost nothing on a fresh install — then reported a mesh the
device was not in. That is a read telling a lie about state it created itself.

THE FIX IS STRUCTURAL, NOT A GUARD AT ONE CALL SITE: every resolver
(``identity.network_root``, ``identity.identity_dir``, ``store.networks_dir``,
``store.outbox_dir``, ``store.peer_outbox_dir``, ``store.pending_dir``) returns a
PATH and creates nothing, and its ``ensure_*`` twin is the only spelling that
creates. THIS FILE PINS THE READ HALF — the whole class of resolvers, not the one
that happened to be reached — and ``test_writes_still_create.py`` pins the write
half, because a resolver that creates is the bug above and a creator that stopped
creating would turn a fresh ``save`` into a ``FileNotFoundError``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import pytest

from local_operator.network import identity, store, types

NETWORK = "n_0123456789abcdef01234567"


def _record() -> types.NetworkRecord:
    return types.NetworkRecord(
        network_id=NETWORK,
        name="home-net",
        epoch=1,
        created_by="d_" + "a" * 32,
        self_device_id="d_" + "a" * 32,
        self_role="admin",
        self_capabilities=sorted(types.capabilities_for_role("admin")),
    )


#: Every resolver a reader may reach, with the id arguments each needs. A resolver
#: is anything whose name is "where is it" rather than "make it": the two spellings
#: differ in exactly this list.
RESOLVERS: list[tuple[str, Callable[[Path], Any]]] = [
    ("identity.network_root", identity.network_root),
    ("identity.identity_dir", identity.identity_dir),
    ("identity.identity_path", identity.identity_path),
    ("store.networks_dir", store.networks_dir),
    ("store.record_path", lambda root: store.record_path(NETWORK, root)),
    ("store.secrets_path", lambda root: store.secrets_path(NETWORK, root)),
    ("store.outbox_dir", store.outbox_dir),
    ("store.peer_outbox_dir", lambda root: store.peer_outbox_dir("d_" + "b" * 32, root)),
    ("store.invite_path", lambda root: store.invite_path("i_round_trip", root)),
    ("store.pending_dir", store.pending_dir),
    ("store.pending_path", lambda root: store.pending_path("i_round_trip", root)),
    ("store.decision_path", lambda root: store.decision_path("i_round_trip", root)),
    ("store.catalog_path", store.catalog_path),
    ("store.audit_path", store.audit_path),
]


@pytest.mark.parametrize(("name", "resolve"), RESOLVERS, ids=[name for name, _ in RESOLVERS])
def test_resolving_a_path_creates_nothing(
    root: Path, name: str, resolve: Callable[[Path], Any]
) -> None:
    """The whole class, not one instance: no resolver is a writer.

    A test of one resolver would have passed the day before the bug and left the
    other thirteen to be discovered one desktop route at a time.
    """
    resolve(root)
    assert list(root.rglob("*")) == [], f"{name} created something under the config root"


def test_listing_a_store_that_was_never_written_answers_empty_and_creates_nothing(
    root: Path,
) -> None:
    """The exact call the desktop GET made: a read that must answer "none".

    ``list_networks`` is reached from the command catalogue (the peer vocabulary of
    ``/new remote``) and from the mesh reads; "there are no networks" is its answer
    on a fresh install, and it is not a reason to build the directory that would
    hold them.
    """
    assert store.list_networks(root) == []
    assert not (root / "network").exists()


def test_reading_an_absent_identity_creates_nothing(root: Path) -> None:
    """``identity.load`` answers ``None`` for a device with no keypair, and does not
    mint the directory on the way — the file it would sit in is the one thing this
    read may not create."""
    assert identity.load(root) is None
    assert not (root / "network").exists()


def test_a_purge_of_a_store_that_was_never_written_creates_nothing(root: Path) -> None:
    """The empty-store arm: nothing to forget, and nothing to create on the way."""
    removed = store.purge_network_artifacts(root=root)
    assert removed["networks"] == []
    assert not (root / "network").exists()


def test_a_purge_of_a_network_with_no_outbox_would_raise_without_the_is_dir_guard(
    root: Path,
) -> None:
    """THE CELL THAT MAKES THE ``is_dir`` GUARD LOAD-BEARING, and the reason it is
    not belt-and-braces: this is the arm that actually runs.

    The purge's every-queue-sweep is only reached when EVERY known network is being
    purged (``covers_everything``), so a fresh root never exercises it — measured, by
    removing the guard and watching the fresh-root test above stay green. The case
    that does is a device that IS in a network and has never minted an invite: no
    ``outbox/`` to iterate, ``iterdir`` raises ``FileNotFoundError``, and the operator
    running ``lop network uninstall --purge`` — whose whole purpose is to be the
    escape hatch from a device that can no longer reach its peers — gets a traceback
    where the old creating resolver had left an empty directory behind.
    """
    store.save(_record(), root)
    assert not (root / "network" / "outbox").exists(), "the fixture minted an invite"

    removed = store.purge_network_artifacts(root=root)

    assert removed["networks"] == ["home-net (n_0123456789abcdef01234567)"]
    assert removed["queues"] == 0
    assert removed["invites"] == 0
    assert not (root / "network" / "networks" / f"{NETWORK}.json").exists()
