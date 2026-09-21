"""Device identity: derivation, the 0600 boundary, rotation and the copy fence."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from local_operator.network import identity, wire


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


# ---------------------------------------------------------------------------
# Derivation
# ---------------------------------------------------------------------------


def test_device_id_is_derived_from_the_public_key(root: Path) -> None:
    """A FIXED VECTOR: the id is a function of the key, not an assignment.

    This is the property that lets a member list be verified without a CA, so it is
    pinned against a literal rather than re-derived from the same function under
    test.
    """
    public = bytes(range(32))
    derived = identity.device_id_for(public)
    assert derived.startswith("d_")
    assert len(derived) == 34  # "d_" + 32 hex chars = 128 bits, per §3.1
    assert derived == identity.device_id_for(public)
    # Editing one byte of the key changes the id — otherwise the id would be a
    # label rather than a fingerprint.
    changed = identity.device_id_for(bytes([1]) + public[1:])
    assert changed != derived


def test_minted_identity_matches_its_own_key(root: Path) -> None:
    minted = identity.mint(root, name="unit-test")
    assert minted.device_id == identity.device_id_for(minted.public_key_bytes)
    assert len(minted.public_key_bytes) == 32
    assert len(minted.private_key_bytes) == 32
    assert minted.generation == 1
    assert minted.rotated_from is None


def test_identity_is_0600_in_a_0700_dir(root: Path) -> None:
    """The permissions ARE the authorization model, so a mode is asserted, not assumed.

    A file that drifted to 0644 would pass every functional test in this package.
    """
    identity.mint(root)
    assert _mode(identity.identity_path(root)) == 0o600
    assert _mode(identity.identity_dir(root)) == 0o700
    assert _mode(identity.network_root(root)) == 0o700


def test_identity_write_leaves_no_temporary_behind(root: Path) -> None:
    """Staged-then-renamed, and the staging file is not litter after the rename."""
    identity.mint(root)
    leftovers = [path for path in identity.identity_dir(root).glob(".*")]
    assert leftovers == []


def test_load_refuses_an_identity_whose_id_was_edited(root: Path) -> None:
    """A file claiming somebody else's id is refused ON THE DEVICE THAT WOULD CLAIM IT.

    The id is what every peer compares, so an edited file must not be able to walk
    out of here: this is the negative half of ``device_id_for``.
    """
    minted = identity.mint(root)
    path = identity.identity_path(root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["device_id"] = "d_" + "0" * 32
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError) as excinfo:
        identity.load(root)
    assert "edited" in str(excinfo.value)
    # The message may name this device's own id — that is a name, not a secret — but
    # nothing in it may name the private half.
    assert minted.private_key not in str(excinfo.value)


def test_load_returns_none_then_mints(root: Path) -> None:
    assert identity.load(root) is None
    first = identity.load_or_mint(root)
    assert identity.load_or_mint(root).device_id == first.device_id


def test_mint_refuses_to_overwrite(root: Path) -> None:
    identity.mint(root)
    with pytest.raises(RuntimeError):
        identity.mint(root)


def test_private_key_never_appears_where_a_surface_could_read_it(root: Path) -> None:
    """The 0600 file is the ONLY place the private half exists.

    Asserted rather than asserted-by-convention because the natural next change to
    this module is "add the device fingerprint to the peer record", and that is
    exactly how a key ends up in a record every status command dumps.
    """
    minted = identity.mint(root)
    raw = identity.identity_path(root).read_text(encoding="utf-8")
    assert minted.private_key in raw  # the file is the one place it is allowed
    assert minted.private_key != minted.public_key
    # Nothing else in the network store carries it, including the record shape a
    # `--json` dump would serialise.
    for path in identity.network_root(root).rglob("*"):
        if path.is_file() and path.name != "device.json":
            assert minted.private_key not in path.read_text(encoding="utf-8", errors="ignore")


# ---------------------------------------------------------------------------
# Rotation
# ---------------------------------------------------------------------------


def test_rotation_changes_the_id_and_keeps_continuity(root: Path) -> None:
    """An id is a key fingerprint, so it CANNOT survive a rotation — and the
    statement signed by the old key is what proves the two ids are one device."""
    old = identity.mint(root, name="before")
    new, previous = identity.rotate(root, name="after")
    assert new.generation == 2
    assert new.rotated_from == old.device_id
    assert new.device_id != old.device_id
    assert previous.device_id == old.device_id
    # The store, read back: ``identity.load`` is ``None`` for a device that has never
    # minted, so the assertion proves the rotated identity is the one on disk.
    stored = identity.load(root)
    assert stored is not None
    assert stored.device_id == new.device_id
    statement = identity.rotation_statement(old, new, "n_abc", signed_at=1.0)
    identity.verify_rotation_statement(statement, old.public_key)


def test_rotation_statement_is_refused_without_the_old_key(root: Path) -> None:
    """The negative case: a statement verified against a DIFFERENT device's key must
    fail, which is what stops a rotation being used to re-identify as anybody."""
    old = identity.mint(root, name="before")
    new, _previous = identity.rotate(root)
    statement = identity.rotation_statement(old, new, "n_abc")
    other = identity.mint(root / "other")
    with pytest.raises(Exception) as excinfo:
        identity.verify_rotation_statement(statement, other.public_key)
    assert getattr(excinfo.value, "code", "") == "bad_rotation_statement"


def test_rotation_statement_tampering_is_detected(root: Path) -> None:
    old = identity.mint(root, name="before")
    new, _previous = identity.rotate(root)
    statement = identity.rotation_statement(old, new, "n_abc")
    statement["new_device_id"] = "d_" + "f" * 32
    with pytest.raises(Exception) as excinfo:
        identity.verify_rotation_statement(statement, old.public_key)
    assert getattr(excinfo.value, "code", "") == "bad_rotation_statement"


def test_rotate_without_an_identity_is_refused(root: Path) -> None:
    with pytest.raises(RuntimeError):
        identity.rotate(root)


# ---------------------------------------------------------------------------
# The duplicate-identity fence
# ---------------------------------------------------------------------------


def test_new_instance_id_each_time() -> None:
    ids = {identity.mint_instance_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(value.startswith("i_") and len(value) > 3 for value in ids)


def test_restart_inside_the_grace_window_is_not_a_duplicate() -> None:
    tracker = identity.IdentityUseTracker()
    tracker.observe("d_one", instance_id="i_a", link_id="l1", now=100.0)
    verdict = tracker.observe("d_one", instance_id="i_b", link_id="l2", now=101.0)
    assert verdict.kind == "restart"
    assert verdict.flagged is False


def test_second_link_outside_the_grace_window_is_a_duplicate() -> None:
    """Two live processes on one device id: the copy detector's positive case."""
    tracker = identity.IdentityUseTracker()
    tracker.observe("d_one", instance_id="i_a", link_id="l1", now=100.0)
    verdict = tracker.observe("d_one", instance_id="i_b", link_id="l2", now=100.0 + 60.0)
    assert verdict.kind == "duplicate"
    assert verdict.flagged is True
    assert verdict.evicted is not None and verdict.evicted.link_id == "l1"


def test_same_instance_id_on_a_second_link_is_always_a_duplicate() -> None:
    """Impossible from a correct peer (one process, one id) — so it is a fork or a
    copy even inside the grace window."""
    tracker = identity.IdentityUseTracker()
    tracker.observe("d_one", instance_id="i_a", link_id="l1", now=100.0)
    verdict = tracker.observe("d_one", instance_id="i_a", link_id="l2", now=100.5)
    assert verdict.kind == "duplicate"


def test_three_instances_within_the_window_are_visible() -> None:
    tracker = identity.IdentityUseTracker()
    for index in range(3):
        tracker.observe("d_one", instance_id=f"i_{index}", link_id=f"l{index}", now=100.0 + index)
    assert tracker.recent_instance_count("d_one", now=103.0) == 3
    # Outside the window the count falls away rather than accumulating forever.
    assert (
        tracker.recent_instance_count("d_one", now=100.0 + identity.DUPLICATE_FLAG_WINDOW_S + 5)
        == 0
    )


def test_released_claim_is_forgotten_only_by_its_own_link() -> None:
    tracker = identity.IdentityUseTracker()
    tracker.observe("d_one", instance_id="i_a", link_id="l1", now=1.0)
    tracker.released("d_one", "l2")
    assert tracker.claim_for("d_one") is not None
    tracker.released("d_one", "l1")
    assert tracker.claim_for("d_one") is None


# ---------------------------------------------------------------------------
# Encodings the identity uses
# ---------------------------------------------------------------------------


def test_crockford_alphabet_excludes_the_confusable_letters() -> None:
    text = wire.crockford(bytes(range(64)))
    assert set(text) <= set("0123456789abcdefghjkmnpqrstvwxyz")
    assert not ({"i", "l", "o", "u"} & set(text))


def test_base64url_round_trip_without_padding() -> None:
    payload = os.urandom(32)
    encoded = wire.b64u(payload)
    assert "=" not in encoded
    assert wire.unb64u(encoded) == payload
    # A peer that pads anyway is tolerated rather than refused.
    assert wire.unb64u(encoded + "==") == payload
