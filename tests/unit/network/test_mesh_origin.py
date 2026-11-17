"""G5/G6 primitives: the relay proof, the stamp, its fold, downgrade and re-check.

Design: ``docs/design/mesh-origin-stamp.md`` §2-§5. These cells drive the pure
helpers in ``local_operator/network/origin.py``; the wire path (a real relay
dialling a real runtime) lives in ``tests/unit/session/runtime/
test_mesh_origin_runtime.py`` and the relay sites in ``test_mesh_origin_relay.py``.
Numbers in the test names are the design's §5 test ids.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import identity, origin, store, types

REPO = Path(__file__).resolve().parents[3]
PRODUCT = REPO / "local_operator"


def _block(**overrides: Any) -> dict[str, Any]:
    peer = {
        "device_id": "dev-b",
        "name": "laptop",
        "network_id": "net-1",
        "link_id": "ab" * 16,
        "peer_verified": True,
    }
    peer.update(overrides)
    return origin.canonical_block(peer, ["prompt", "view", "prompt"])


# ---------------------------------------------------------------------------
# G6: key, canonical block, proof (tests 1-4)
# ---------------------------------------------------------------------------


def test_relay_key_is_rfc5869_hkdf_of_the_seed() -> None:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    seed = bytes(range(32))
    expected = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=None, info=b"lop-relay-runtime-v1"
    ).derive(seed)
    assert origin.derive_relay_key(seed) == expected
    assert origin.derive_relay_key(seed) != origin.derive_relay_key(bytes(32))


def test_relay_key_for_reads_this_install_and_returns_no_seed(tmp_path: Path) -> None:
    assert origin.relay_key_for(tmp_path) is None
    minted = identity.mint(tmp_path, name="a")
    key, device_id = origin.relay_key_for(tmp_path) or (b"", "")
    assert device_id == minted.device_id
    assert len(key) == 32
    from local_operator.network.wire import unb64u

    assert key == origin.derive_relay_key(unb64u(minted.private_key))
    assert key != unb64u(minted.private_key)


def test_canonical_block_has_exactly_six_typed_keys_and_sorted_caps() -> None:
    block = origin.canonical_block(
        {"device_id": 1, "name": None, "extra": "x", "peer_verified": "yes"}, {"view", "prompt"}
    )
    assert sorted(block) == sorted(
        ["device_id", "name", "network_id", "link_id", "peer_verified", "capabilities"]
    )
    assert block["device_id"] == "1" and block["name"] == "" and block["network_id"] == ""
    # Only a real ``True`` counts: a truthy string is a claim, not the bool the MAC binds.
    assert block["peer_verified"] is False
    assert block["capabilities"] == ["prompt", "view"]
    assert origin.canonical_block(None, [])["peer_verified"] is False


def test_2_valid_proof_verifies_and_is_bound_to_nonce_salt_and_block() -> None:
    key = origin.derive_relay_key(b"k" * 32)
    nonce, salt = origin.relay_nonce(), origin.relay_nonce()
    block = _block()
    proof = origin.relay_proof(key, relay_nonce=nonce, relay_salt=salt, block=block)
    assert origin.relay_proof_ok(proof, key=key, relay_nonce=nonce, relay_salt=salt, block=block)


@pytest.mark.parametrize(
    "case", ["missing", "not_hex", "wrong_label", "wrong_salt", "other_connection", "wrong_key"]
)
def test_1_forged_or_misbound_proof_is_refused(case: str) -> None:
    key = origin.derive_relay_key(b"k" * 32)
    nonce, salt = origin.relay_nonce(), origin.relay_nonce()
    block = _block()
    good = origin.relay_proof(key, relay_nonce=nonce, relay_salt=salt, block=block)
    supplied: Any = good
    if case == "missing":
        supplied = None
    elif case == "not_hex":
        supplied = "z" * 64
    elif case == "wrong_label":
        # The operator handshake's construction over the same inputs: domain
        # separation is what keeps one proof from replaying as the other.
        message = b"lop-operator-cap-handshake-v1" + nonce.encode() + b":" + salt.encode()
        supplied = hmac.new(key, message, hashlib.sha256).hexdigest()
    elif case == "wrong_salt":
        supplied = origin.relay_proof(
            key, relay_nonce=nonce, relay_salt=origin.relay_nonce(), block=block
        )
    elif case == "other_connection":
        supplied = origin.relay_proof(
            key, relay_nonce=origin.relay_nonce(), relay_salt=origin.relay_nonce(), block=block
        )
    elif case == "wrong_key":
        supplied = origin.relay_proof(
            origin.derive_relay_key(b"x" * 32), relay_nonce=nonce, relay_salt=salt, block=block
        )
    assert not origin.relay_proof_ok(
        supplied, key=key, relay_nonce=nonce, relay_salt=salt, block=block
    )


def test_1_no_key_or_no_handshake_material_fails_closed() -> None:
    key = origin.derive_relay_key(b"k" * 32)
    nonce, salt = origin.relay_nonce(), origin.relay_nonce()
    proof = origin.relay_proof(key, relay_nonce=nonce, relay_salt=salt, block=_block())
    assert not origin.relay_proof_ok(
        proof, key=None, relay_nonce=nonce, relay_salt=salt, block=_block()
    )
    assert not origin.relay_proof_ok(
        proof, key=key, relay_nonce="", relay_salt=salt, block=_block()
    )
    assert not origin.relay_proof_ok(
        proof, key=key, relay_nonce=nonce, relay_salt="", block=_block()
    )


@pytest.mark.parametrize(
    "changed", [{"device_id": "dev-c"}, {"peer_verified": False}, {"network_id": "net-2"}]
)
def test_4_proof_over_another_block_is_refused(changed: dict[str, Any]) -> None:
    key = origin.derive_relay_key(b"k" * 32)
    nonce, salt = origin.relay_nonce(), origin.relay_nonce()
    proof = origin.relay_proof(key, relay_nonce=nonce, relay_salt=salt, block=_block())
    assert not origin.relay_proof_ok(
        proof, key=key, relay_nonce=nonce, relay_salt=salt, block=_block(**changed)
    )


# ---------------------------------------------------------------------------
# The stamp and its answer fold (tests 2, 3, 12, 12b)
# ---------------------------------------------------------------------------


def test_stamp_carries_the_proved_block_and_no_operator_origin() -> None:
    stamp = origin.build_stamp(
        {"device_id": "dev-b", "network_id": "net-1", "link_id": "ab", "name": "x"},
        frozenset({"view", "prompt"}),
        verified=True,
        stamped_by="dev-a",
    )
    assert stamp == {
        "origin_device_id": "dev-b",
        "network_id": "net-1",
        "link_id": "ab",
        "capabilities": ["prompt", "view"],
        "verified": True,
        "stamped_by": "dev-a",
    }
    assert "operator_origin" not in stamp


def _stamp(verified: bool = True, device: str = "dev-b", by: str = "dev-a") -> dict[str, Any]:
    return origin.build_stamp(
        {"device_id": device, "network_id": "net-1", "link_id": "ab"},
        ["prompt"],
        verified=verified,
        stamped_by=by,
    )


def _fold(first: Any, revisions: list[Any]) -> dict[str, Any]:
    folded = origin.fold_answer_origin(first, revisions)
    assert folded is not None
    return folded


def test_12_fold_keeps_verified_only_when_every_write_is_the_same_verified_device() -> None:
    assert _fold(_stamp(), [])["verified"] is True
    assert _fold(_stamp(), [_stamp()])["verified"] is True
    assert _fold(_stamp(), [_stamp(device="dev-c")])["verified"] is False
    assert _fold(_stamp(), [_stamp(by="dev-z")])["verified"] is False
    # The first answer's attribution is what the record keeps.
    assert _fold(_stamp(), [_stamp(device="dev-c")])["origin_device_id"] == ("dev-b")
    assert origin.fold_answer_origin(None, [_stamp()]) is None


def test_12b_verified_answer_revised_by_unstamped_or_unverified_write_reads_false() -> None:
    assert _fold(_stamp(), [None])["verified"] is False
    assert _fold(_stamp(), [_stamp(verified=False)])["verified"] is False
    assert _fold(_stamp(), [_stamp(), None])["verified"] is False


# ---------------------------------------------------------------------------
# Imported sessions (test 8, unit half; the promote call sites are in the relay file)
# ---------------------------------------------------------------------------


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_8_downgrade_lowers_every_stamp_whoever_minted_it(tmp_path: Path) -> None:
    session = tmp_path / "s"
    own = _stamp(by="dev-a")  # an A→D→A round trip: a forged own stamp
    _jsonl(
        session / "transcript.jsonl",
        [
            {"type": "header"},
            {"message": {"role": "user", "provider_payload": {"lop_mesh_origin": own}}},
        ],
    )
    _jsonl(
        session / "asks.jsonl",
        [{"kind": "answered", "by": {"surface": "remote", "mesh_origin": _stamp(by="dev-d")}}],
    )
    _jsonl(session / "inbox.jsonl", [{"text": "x", "source": "user", "mesh_origin": own}])
    _jsonl(session / "children" / "c1" / "transcript.jsonl", [{"lop_mesh_origin": own}])
    plain = session / "other.jsonl"
    plain.write_bytes(b'{"untouched": true}\n')
    before = plain.stat().st_mtime_ns
    outside = tmp_path / "outside.jsonl"
    _jsonl(outside, [{"lop_mesh_origin": own}])
    (session / "linked.jsonl").symlink_to(outside)

    assert origin.downgrade_stamps(session) == 4
    row = _rows(session / "transcript.jsonl")[1]["message"]["provider_payload"]["lop_mesh_origin"]
    assert row == {**own, "verified": False}  # fields kept for audit
    assert _rows(session / "asks.jsonl")[0]["by"]["mesh_origin"]["verified"] is False
    assert _rows(session / "inbox.jsonl")[0]["mesh_origin"]["verified"] is False
    assert (
        _rows(session / "children" / "c1" / "transcript.jsonl")[0]["lop_mesh_origin"]["verified"]
        is False
    )
    assert plain.stat().st_mtime_ns == before
    assert _rows(outside)[0]["lop_mesh_origin"]["verified"] is True  # symlink not followed
    for path in (session / "transcript.jsonl", session / "asks.jsonl"):
        for stamp in _all_stamps(_rows(path)):
            assert not origin.mesh_origin_counts(stamp)
    assert origin.downgrade_stamps(session) == 0  # idempotent


def _all_stamps(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                if key in origin.FRAME_ORIGIN_KEYS and isinstance(item, dict):
                    found.append(item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(rows)
    return found


# ---------------------------------------------------------------------------
# Consent-time re-derivation (tests 10, 11, 17)
# ---------------------------------------------------------------------------


@pytest.fixture()
def mesh(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "cfg"
    me = identity.mint(root, name="a")
    record = types.NetworkRecord(network_id="net-1", name="n", epoch=1, self_device_id=me.device_id)
    record.members = [
        types.MemberRecord(device_id=me.device_id, name="a"),
        types.MemberRecord(device_id="dev-b", name="b", previous_ids=["dev-b-old"]),
    ]
    store.save(record, root)
    stamp = origin.build_stamp(
        {"device_id": "dev-b", "network_id": "net-1", "link_id": "ab"},
        ["prompt"],
        verified=True,
        stamped_by=me.device_id,
    )
    return {"root": root, "me": me, "stamp": stamp}


def _counts(mesh: dict[str, Any], **changes: Any) -> bool:
    return origin.mesh_origin_counts({**mesh["stamp"], **changes}, _root=mesh["root"])


def test_11_an_active_member_counts(mesh: dict[str, Any]) -> None:
    assert _counts(mesh)


def test_11_unrelated_epoch_rotation_still_counts(mesh: dict[str, Any]) -> None:
    with store.mutate("net-1", mesh["root"]) as record:
        record.epoch = 7
        store.save(record, mesh["root"])
    assert _counts(mesh)


@pytest.mark.parametrize(
    "mutation",
    ["untrusted", "tombstoned", "removed_ids"],
)
def test_11_revoked_membership_does_not_count(mesh: dict[str, Any], mutation: str) -> None:
    with store.mutate("net-1", mesh["root"]) as record:
        if mutation == "untrusted":
            record.trust = "untrusted"
        elif mutation == "tombstoned":
            row = record.member("dev-b")
            assert row is not None
            row.removed_at = time.time()
        else:
            record.removed_ids = ["dev-b"]
        store.save(record, mesh["root"])
    assert not _counts(mesh)


def test_11_unverified_or_foreign_minted_stamps_do_not_count(mesh: dict[str, Any]) -> None:
    assert not _counts(mesh, verified=False)
    assert not _counts(mesh, verified="true")
    assert not _counts(mesh, stamped_by="dev-other")


def test_11_this_devices_identity_rotation_voids_its_own_stamps(mesh: dict[str, Any]) -> None:
    assert _counts(mesh)
    identity.rotate(mesh["root"])
    assert not _counts(mesh)


def test_10_legacy_rows_without_a_stamp_do_not_count(mesh: dict[str, Any]) -> None:
    for legacy in (None, {}, {"remote_origin": True}, "verified"):
        assert not origin.mesh_origin_counts(legacy, _root=mesh["root"])  # type: ignore[arg-type]


def test_17_missing_corrupt_or_previous_id_does_not_count(mesh: dict[str, Any]) -> None:
    assert not _counts(mesh, network_id="net-missing")
    assert not _counts(mesh, origin_device_id="dev-b-old")  # member() would match it
    store.record_path("net-1", mesh["root"]).write_text("{not json", encoding="utf-8")
    assert not _counts(mesh)


@pytest.mark.parametrize("network_id", ["../x", "a/b", "..", ".", "a\\b", "a:b", "", None])
def test_17_unsafe_network_id_opens_no_file(
    mesh: dict[str, Any], network_id: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[Any] = []
    monkeypatch.setattr(store, "load", lambda *a, **k: opened.append(a))
    assert not _counts(mesh, network_id=network_id)
    assert opened == []


def test_counts_reads_root_and_identity_internally(
    mesh: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the test seam the runtime's own ``config_dir()`` is what is read."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(mesh["root"]))
    assert origin.mesh_origin_counts(mesh["stamp"])
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(mesh["root"].parent / "elsewhere"))
    assert not origin.mesh_origin_counts(mesh["stamp"])


# ---------------------------------------------------------------------------
# Structural pins: R3-1 test seam and INERT
# ---------------------------------------------------------------------------


def _product_calls(name: str) -> list[tuple[Path, ast.Call]]:
    calls: list[tuple[Path, ast.Call]] = []
    for path in PRODUCT.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if name not in text:
            continue
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Call):
                func = node.func
                called = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if called == name:
                    calls.append((path, node))
    return calls


def test_r3_1_no_production_caller_supplies_the_test_seam() -> None:
    signature = __import__("inspect").signature(origin.mesh_origin_counts)
    for private in ("_root", "_identity"):
        assert signature.parameters[private].kind is signature.parameters[private].KEYWORD_ONLY
    offenders = [
        f"{path.relative_to(REPO)}:{call.lineno}"
        for path, call in _product_calls("mesh_origin_counts")
        # ``kw.arg is None`` is a ``**`` splat, which can smuggle either seam in.
        if any(kw.arg is None or kw.arg in ("_root", "_identity") for kw in call.keywords)
    ]
    assert offenders == []


def test_inert_nothing_in_production_consumes_the_stamp_yet() -> None:
    """This PR produces evidence only; the guard change (60334) consumes it later."""
    assert _product_calls("mesh_origin_counts") == []
    # The WHOLE product tree (security review S-3): a consumer in ``asks/surfaces``,
    # ``operator/`` or an approval module would otherwise pass. Only the carriage
    # modules below may mention the stamp; ``mesh_origin`` alone also names the
    # unrelated, pre-existing credential-row field, so the pin keys on the stamp's
    # own spellings instead.
    carriage = {
        "local_operator/asks/queue.py",
        "local_operator/asks/store.py",
        "local_operator/network/dial.py",
        "local_operator/network/mobility.py",
        "local_operator/network/origin.py",
        "local_operator/network/relay.py",
        "local_operator/network/sync.py",
        "local_operator/session/runtime/inbox.py",
        "local_operator/session/runtime/process.py",
        "local_operator/session/runtime/server.py",
        "local_operator/session/runtime/serving.py",
        "local_operator/session/session.py",
    }
    needles = ("lop_mesh_origin", "network.origin", "network import origin", "mesh_origin_counts")
    readers = []
    for path in PRODUCT.rglob("*.py"):
        relative = path.relative_to(REPO).as_posix()
        if relative in carriage:
            continue
        text = path.read_text(encoding="utf-8")
        if any(needle in text for needle in needles):
            readers.append(relative)
    assert readers == []


# ---------------------------------------------------------------------------
# Test 13: the seed stays out of write-confined shells
# ---------------------------------------------------------------------------


def test_13_identity_dir_is_a_confined_secret_and_linux_refuses_confinement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from local_operator.tools import confinement

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path / "cfg"))
    assert "network/identity" in confinement._CONFIG_SECRET_SUBPATHS
    device = identity.identity_path(tmp_path / "cfg").resolve(strict=False)
    if sys.platform == "darwin":
        denials = "\n".join(confinement._secret_read_denials())
        assert str(device.parent) in denials
    else:
        # No kernel confinement off macOS: a confined shell is REFUSED, never run
        # unconfined (design §3.3 residual 2).
        assert confinement.sandbox_unavailable_reason() is not None


# ---------------------------------------------------------------------------
# Security review round 1: S-1 (deep rows) and S-2 (compressed journals)
# ---------------------------------------------------------------------------


def _deep(depth: int) -> Any:
    value: Any = []
    for _ in range(depth):
        value = [value]
    return value


def test_s1_a_1500_deep_stamped_row_is_still_downgraded(tmp_path: Path) -> None:
    """The C decoder parses ~1500 levels; a recursive walk of them blew Python's stack."""
    session = tmp_path / "s"
    own = _stamp(by="dev-a")
    _jsonl(session / "transcript.jsonl", [{"lop_mesh_origin": own, "deep": _deep(1500)}])
    _jsonl(session / "asks.jsonl", [{"by": {"mesh_origin": own}}])

    assert origin.downgrade_stamps(session) == 2
    assert _rows(session / "transcript.jsonl")[0]["lop_mesh_origin"]["verified"] is False
    assert _rows(session / "asks.jsonl")[0]["by"]["mesh_origin"]["verified"] is False


def test_s1_a_line_too_deep_to_decode_refuses_the_import(tmp_path: Path) -> None:
    """json.loads raises RecursionError (not ValueError) past its depth limit: refuse.

    Review R2-1: the limit depends on the caller's stack depth, so a line too deep
    at the import door can still decode for a shallower reader. Skipping it left a
    ``verified: true`` stamp such a reader would trust; the import must refuse.
    """
    session = tmp_path / "s"
    own = _stamp(by="dev-a")
    session.mkdir()
    too_deep = b'{"lop_mesh_origin": {"verified": true}, "x": ' + b"[" * 200_000 + b"]" * 200_000
    later = json.dumps({"lop_mesh_origin": own}).encode()
    original = too_deep + b"}\n" + later + b"\n"
    (session / "transcript.jsonl").write_bytes(original)

    with pytest.raises(origin.ImportStampsRefused, match="too deep"):
        origin.downgrade_stamps(session)
    # Refused before any write-back: the door quarantines or keeps it in staging.
    assert (session / "transcript.jsonl").read_bytes() == original


@pytest.mark.parametrize("name", ["transcript.jsonl.gz", "transcript.jsonl.zst"])
@pytest.mark.parametrize("where", ["top", "nested"])
def test_s2_a_compressed_transcript_refuses_the_downgrade(
    tmp_path: Path, name: str, where: str
) -> None:
    import gzip

    session = tmp_path / "s"
    folder = session if where == "top" else session / "children" / "c1"
    folder.mkdir(parents=True)
    body = (json.dumps({"lop_mesh_origin": _stamp(by="dev-a")}) + "\n").encode()
    (folder / name).write_bytes(gzip.compress(body) if name.endswith(".gz") else body)

    with pytest.raises(origin.ImportStampsRefused):
        origin.downgrade_stamps(session)


def test_s2_archive_names_match_the_compression_module() -> None:
    from local_operator.session import compression

    assert set(origin.TRANSCRIPT_ARCHIVE_NAMES) == set(compression.ARCHIVE_FILENAMES)
