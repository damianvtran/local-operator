"""The other half of the resolver split: a WRITE still creates what it needs.

``tests/unit/network/test_reads_create_nothing.py`` pins that no resolver creates.
That property alone would be a bug if the writers had come to depend on it: every
``*_dir`` resolver used to mkdir, and the tempting way to fix the read-side bug is
to stop creating anywhere, which turns the operator's first ``lop network init``
into a ``FileNotFoundError``.

So the creators are pinned here, at the spelling that says so (``ensure_*``), with
the two things a fix like this silently loses: that they still create at all, and
that they create at 0700 THROUGH THE WHOLE CHAIN — ``mkdir(parents=True)`` alone
would give the ancestors the umask's mode (commonly 0755), and
``<config>/network`` holds device ids, invite tokens' paths and the audit trail's
directory name.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Callable

import pytest

from local_operator.network import identity, store, types

NETWORK = "n_0123456789abcdef01234567"
PEER = "d_" + "b" * 32

ENSURE_TWINS: list[tuple[str, Callable[[Path], Path], str]] = [
    ("ensure_network_root", identity.ensure_network_root, "network"),
    ("ensure_identity_dir", identity.ensure_identity_dir, "network/identity"),
    ("ensure_networks_dir", store.ensure_networks_dir, "network/networks"),
    ("ensure_outbox_dir", store.ensure_outbox_dir, "network/outbox"),
    ("ensure_pending_dir", store.ensure_pending_dir, "network/pending"),
    (
        "ensure_peer_outbox_dir",
        lambda root: store.ensure_peer_outbox_dir(PEER, root),
        f"network/outbox/{PEER}",
    ),
]


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


@pytest.mark.parametrize(
    ("ensure", "relative"),
    [(ensure, relative) for _name, ensure, relative in ENSURE_TWINS],
    ids=[name for name, _ensure, _relative in ENSURE_TWINS],
)
def test_the_ensure_twins_create_0700_all_the_way_down(
    root: Path, ensure: Callable[[Path], Path], relative: str
) -> None:
    path = ensure(root)
    assert path == root / relative
    chain: list[Path] = []
    cursor = path
    while cursor != root:
        chain.append(cursor)
        cursor = cursor.parent
    assert chain, "the twin returned a path outside the config root"
    for directory in chain:
        assert directory.is_dir(), directory
        assert stat.S_IMODE(directory.stat().st_mode) == 0o700, directory


def test_a_written_network_is_readable_and_its_directories_stay_private(root: Path) -> None:
    """The writer path end to end, through the store's own entry point.

    The resolvers are what readers use; ``save`` is what a reader must then be able
    to find. Both halves in one test, because a ``save`` that wrote an unreadable
    location and a resolver that created the wrong one would each look fine alone.
    """
    path = store.save(_record(), root)
    assert path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.networks_dir(root).stat().st_mode) == 0o700
    assert stat.S_IMODE((root / "network").stat().st_mode) == 0o700
    assert [record.network_id for record in store.list_networks(root)] == [NETWORK]
