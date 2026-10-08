"""The mesh-copy provenance marker: bounded, sealed, preserved across writes.

WHY THESE CELLS EXIST (S4 of ``mesh-consent-provisioning.md``). The marker is
what a wipe scans to find THIS owner's copies of one key, so the properties
that matter are not cosmetic:

- it is INSIDE the sealed payload (authenticated, never plaintext on disk);
- it survives ``update`` and ``rotate`` — the two ways a record is written
  again without the copy engine's knowledge — unless a caller deliberately
  replaces or clears it;
- it is bounded and refused LOUDLY on a shape the store cannot carry, because
  a silently dropped marker is a copy the wipe can never find;
- reading for a copy does not count as a use (``last_used_at``/audit), where a
  human or agent-facing ``get`` does.
"""

from __future__ import annotations

import sqlite3

import pytest

from local_operator.secrets.crypto import generate_master_key
from local_operator.secrets.errors import SecretNotFound, SecretStoreError
from local_operator.secrets.store import SecretStore

ORIGIN = {
    "owner_device": "d_" + "a" * 32,
    "key": "secret:TOKEN",
    "gen": 3,
    "applied_at": 1759800000.25,
}


def test_the_marker_round_trips_and_is_never_plaintext(store: SecretStore) -> None:
    store.set("TOKEN", b"value", origin=dict(ORIGIN))
    record = store.describe("TOKEN")
    assert record.origin == ORIGIN
    listed = {row.name: row.origin for row in store.list()}
    assert listed["TOKEN"] == ORIGIN

    # INSIDE THE CIPHERTEXT: the marker's own text appears nowhere on disk —
    # not in the row, not beside it. (The value's absence is pinned elsewhere;
    # this asserts the METADATA's, which a naive implementation stores plainly.)
    raw = store.path.read_bytes()
    assert b"owner_device" not in raw
    assert ORIGIN["owner_device"].encode() not in raw
    assert b"secret:TOKEN" not in raw


def test_a_locally_created_record_carries_no_marker(store: SecretStore) -> None:
    store.set("PLAIN", b"value")
    assert store.describe("PLAIN").origin is None


def test_update_keeps_the_marker_by_default(store: SecretStore) -> None:
    store.set("TOKEN", b"v1", origin=dict(ORIGIN))
    store.update("TOKEN", b"v2")
    assert store.describe("TOKEN").origin == ORIGIN


def test_update_replaces_or_clears_only_when_told(store: SecretStore) -> None:
    store.set("TOKEN", b"v1", origin=dict(ORIGIN))
    replacement = {"owner_device": "d_" + "b" * 32, "key": "secret:OTHER"}
    store.update("TOKEN", b"v2", origin=dict(replacement))
    assert store.describe("TOKEN").origin == replacement
    store.update("TOKEN", b"v3", origin=None)
    assert store.describe("TOKEN").origin is None


def test_rotation_preserves_the_marker(store: SecretStore) -> None:
    store.set("TOKEN", b"value", origin=dict(ORIGIN))
    new_key = generate_master_key()
    assert store.rotate(new_key) == 1
    rotated = SecretStore(new_key, base=store._base)
    assert rotated.get("TOKEN") == b"value"
    assert rotated.describe("TOKEN").origin == ORIGIN


@pytest.mark.parametrize(
    "bad",
    [
        pytest.param(["not", "an", "object"], id="list"),
        pytest.param("a string", id="string"),
        pytest.param({"owner_device": True}, id="bool"),
        pytest.param({"owner_device": {"nested": "no"}}, id="nested"),
        pytest.param({"owner_device": None}, id="none-value"),
        pytest.param({"owner_device": "x" * 513}, id="oversized-value"),
        pytest.param({"k" * 65: "x"}, id="oversized-key"),
        pytest.param({f"f{index}": "x" for index in range(17)}, id="too-many-fields"),
    ],
)
def test_an_unbounded_marker_is_refused_and_writes_nothing(store: SecretStore, bad: object) -> None:
    """Refused LOUDLY (a write path): a silently dropped marker is a lost wipe."""
    with pytest.raises(SecretStoreError):
        store.set("TOKEN", b"value", origin=bad)  # type: ignore[arg-type]
    with pytest.raises(SecretNotFound):
        store.get("TOKEN")


def test_read_for_copy_does_not_count_as_a_use(store: SecretStore) -> None:
    """The copy engine's read writes no ``last_used`` and no audit row.

    ``get`` is the human/agent-facing verb and deliberately leaves a trace; the
    engine recomputes a digest on every sync tick, and letting that masquerade
    as use would make the listing's "last used" column a lie about the
    operator's own activity.
    """
    store.set("TOKEN", b"value", origin=dict(ORIGIN))
    record, value = store.read_for_copy("TOKEN")
    assert value == b"value"
    assert record.last_used_at is None
    with sqlite3.connect(store.path) as connection:
        events = [row[0] for row in connection.execute("SELECT event FROM audit")]
    assert "read_for_copy" not in events
    assert "get" not in events

    store.get("TOKEN")
    assert store.describe("TOKEN").last_used_at is not None
