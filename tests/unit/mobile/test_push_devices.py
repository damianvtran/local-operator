"""``/api/push/*`` on the wire: the device registry, its refusals, and its file.

Push/ack-sync S4 (ADR 0006 §3.1 @ 22e2cce2). An isolated config root (the
autouse ``isolate_environment`` fixture) and the real ``build_app`` over a
``MobileDaemon``, mirroring ``test_projects_routes.py``. What is asserted is
both halves of the slice's contract: what the phone is told (status codes,
bodies) and the store's side effects on disk (the file's bytes, its mode, the
token's absence) — a green wire against a store that quietly persisted the
token would be the exact defect the spec update named.

Nothing here touches a real session or the operator's store: the config root
is the fixture's scratch home, and the only files this suite writes under it
are the registry itself, plus in one cell a planted ``attention.db`` whose
untouchedness is the assertion.
"""

from __future__ import annotations

import json
import logging
import os
import stat
import uuid
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import push_devices
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.mobile.push_devices import PUSH_DEVICES_STORE_NAME
from local_operator.paths import config_dir

#: One complete, valid register payload; cells override single fields from it.
PAYLOAD = {
    "platform": "ios",
    "token": "apns-token-abcdef-0f1e2d3c",
    "environment": "production",
    "app_version": "1.0.0 (12)",
    "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
}


def _client() -> TestClient:
    """A logged-in client over the real daemon app, under the test's HOME."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _store_path() -> Path:
    return config_dir() / PUSH_DEVICES_STORE_NAME


def _stored_records() -> list[dict[str, Any]]:
    return json.loads(_store_path().read_text(encoding="utf-8"))["devices"]


def test_register_writes_the_record_and_answers_the_shape() -> None:
    client = _client()
    response = client.post("/api/push/register", json={**PAYLOAD, "name": "Damian's iPhone"})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ok"] is True
    assert isinstance(payload["device_id"], str) and payload["device_id"]
    assert isinstance(payload["registered_at"], int) and payload["registered_at"] > 0
    # The per-device key is minted here and returned exactly once (ADR §4 rule 2's
    # ``m3``): it is what lets a later request move only its own device's state.
    assert isinstance(payload["device_key"], str) and len(payload["device_key"]) >= 32

    # The file side effect: one record, 0600, and the fields the ADR §2.2
    # record carries (minus the token — asserted in its own cell).
    assert (os.stat(_store_path()).st_mode & 0o777) == 0o600
    records = _stored_records()
    assert len(records) == 1
    record = records[0]
    assert sorted(record) == [
        "app_version",
        "credential_live",
        "device_id",
        "device_key",
        "environment",
        "install_id",
        "last_authenticated_at",
        "last_seen_at",
        "name",
        "platform",
        "registered_at",
    ]
    assert record["device_id"] == payload["device_id"]
    assert record["platform"] == "ios"
    assert record["environment"] == "production"
    assert record["app_version"] == "1.0.0 (12)"
    assert record["install_id"] == PAYLOAD["install_id"]
    assert record["name"] == "Damian's iPhone"
    assert record["registered_at"] == payload["registered_at"]
    assert record["last_seen_at"] == record["registered_at"]
    # The key is stored machine-side (that is its whole purpose) and the register
    # route is an authenticated request, so it IS the device's last authenticated
    # moment — the credential record ADR §2.2 keeps beside the key.
    assert record["device_key"] == payload["device_key"]
    assert record["credential_live"] is True
    assert record["last_authenticated_at"] == record["registered_at"]
    # The atomic write leaves no temp file behind.
    assert not list(config_dir().glob(f".{PUSH_DEVICES_STORE_NAME}.*"))


def test_the_token_never_reaches_the_registry_file() -> None:
    """The spec correction's core assertion: token custody is the cloud's (ADR §4)."""
    first = "TOKEN-SENTINEL-first-9c1f-do-not-store"
    rotated = "TOKEN-SENTINEL-rotated-4b7a-do-not-store"
    client = _client()
    assert client.post("/api/push/register", json={**PAYLOAD, "token": first}).status_code == 200
    raw = _store_path().read_bytes()
    assert first.encode() not in raw
    # A rotation re-registers; neither token version may land on disk.
    assert client.post("/api/push/register", json={**PAYLOAD, "token": rotated}).status_code == 200
    raw = _store_path().read_bytes()
    assert first.encode() not in raw and rotated.encode() not in raw
    # The key itself, not just the values: a record filed under "token" would
    # be the same custody mistake under another spelling.
    assert b'"token"' not in raw


def test_reregister_keeps_device_id_and_replaces_metadata() -> None:
    """Idempotent on (install_id, platform): a rotated token must not add a row."""
    client = _client()
    first = client.post("/api/push/register", json={**PAYLOAD, "name": "iPhone"})
    second = client.post(
        "/api/push/register",
        json={
            **PAYLOAD,
            "token": "apns-token-rotated-99",
            "app_version": "1.0.1 (13)",
            "name": "Damian's iPhone",
        },
    )
    assert first.status_code == second.status_code == 200
    assert second.json()["device_id"] == first.json()["device_id"]
    # registered_at is the record's, not the request's: a nightly re-register
    # must not move the anchor it names.
    assert second.json()["registered_at"] == first.json()["registered_at"]

    records = _stored_records()
    assert len(records) == 1, "a rotation must replace in place, not accumulate rows"
    assert records[0]["app_version"] == "1.0.1 (13)"
    assert records[0]["name"] == "Damian's iPhone"
    # The store file stays private across rewrites.
    assert (os.stat(_store_path()).st_mode & 0o777) == 0o600


def test_re_register_bumps_last_seen_at_but_not_registered_at(tmp_path: Path) -> None:
    """The deterministic half of the rotation test: an injected clock proves the bump."""
    root = tmp_path
    first = push_devices.register(root, dict(PAYLOAD), now=1_700_000_000.0)
    second = push_devices.register(root, {**PAYLOAD, "app_version": "2.0.0"}, now=1_700_000_999.0)
    assert second["device_id"] == first["device_id"]
    record = json.loads((root / PUSH_DEVICES_STORE_NAME).read_text(encoding="utf-8"))["devices"][0]
    assert record["registered_at"] == 1_700_000_000
    assert record["last_seen_at"] == 1_700_000_999


def test_the_registry_is_bounded_and_keeps_the_most_recent(tmp_path: Path) -> None:
    """A caller minting a fresh install_id per launch cannot grow the store (M1).

    The reviewer's probe registered 200 distinct install ids and found 202 rows;
    here the same shape runs past the bound and must stay capped, dropping the
    least-recently-seen records first.
    """
    root = tmp_path
    for index in range(1, push_devices.MAX_DEVICE_ENTRIES + 1):
        push_devices.register(
            root,
            {**PAYLOAD, "install_id": str(uuid.UUID(int=index))},
            now=1_700_000_000.0 + index,
        )
    store = root / PUSH_DEVICES_STORE_NAME
    assert len(json.loads(store.read_text(encoding="utf-8"))["devices"]) == (
        push_devices.MAX_DEVICE_ENTRIES
    )

    newest = str(uuid.UUID(int=push_devices.MAX_DEVICE_ENTRIES + 1))
    push_devices.register(root, {**PAYLOAD, "install_id": newest}, now=1_700_000_999.0)
    records = json.loads(store.read_text(encoding="utf-8"))["devices"]
    assert len(records) == push_devices.MAX_DEVICE_ENTRIES, "the store stays bounded"
    kept = {record["install_id"] for record in records}
    assert newest in kept, "the newest registration is never the one dropped"
    assert str(uuid.UUID(int=1)) not in kept, "the least-recently-seen goes first"


def test_prune_drops_exactly_the_excess_even_with_a_duplicated_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M2 probe: duplicates must cost one slot each, not take a survivor down.

    No in-band path can mint a duplicate ``device_id``, so the shape is built on
    disk; the bug it caught removed every record sharing a doomed id, so a
    duplicate pair spanning the prune boundary lost both twins for one slot.
    """
    monkeypatch.setattr(push_devices, "MAX_DEVICE_ENTRIES", 4)
    root = tmp_path
    shared = "b" * 32
    records = [
        {
            **_VALID_RECORD,
            "device_id": "a" * 32,
            "install_id": str(uuid.UUID(int=1)),
            "registered_at": 100,
            "last_seen_at": 100,
        },
        {
            **_VALID_RECORD,
            "device_id": shared,
            "install_id": str(uuid.UUID(int=2)),
            "registered_at": 101,
            "last_seen_at": 101,
        },
        {
            **_VALID_RECORD,
            "device_id": shared,
            "install_id": str(uuid.UUID(int=3)),
            "registered_at": 102,
            "last_seen_at": 102,
        },
        {
            **_VALID_RECORD,
            "device_id": "c" * 32,
            "install_id": str(uuid.UUID(int=4)),
            "registered_at": 200,
            "last_seen_at": 200,
        },
        {
            **_VALID_RECORD,
            "device_id": "d" * 32,
            "install_id": str(uuid.UUID(int=5)),
            "registered_at": 201,
            "last_seen_at": 201,
        },
    ]
    root.joinpath(PUSH_DEVICES_STORE_NAME).write_text(
        json.dumps({"devices": records}), encoding="utf-8"
    )
    push_devices.register(root, {**PAYLOAD, "install_id": str(uuid.UUID(int=6))}, now=999.0)
    stored = json.loads(root.joinpath(PUSH_DEVICES_STORE_NAME).read_text(encoding="utf-8"))[
        "devices"
    ]
    assert len(stored) == 4, "exactly `excess` records leave"
    assert [record["device_id"] for record in stored].count(
        shared
    ) == 1, "the surviving twin stays: eviction is positional, never by id value"


def test_a_different_install_id_or_platform_is_a_new_device() -> None:
    client = _client()
    original = client.post("/api/push/register", json=PAYLOAD)
    other_platform = client.post("/api/push/register", json={**PAYLOAD, "platform": "android"})
    other_install = client.post(
        "/api/push/register",
        json={**PAYLOAD, "install_id": "00000000-1111-2222-3333-444444444444"},
    )
    ids = {original.json()["device_id"], other_platform.json()["device_id"]}
    ids.add(other_install.json()["device_id"])
    assert len(ids) == 3, "(install_id, platform) is the identity; neither half alone is"
    assert len(_stored_records()) == 3


def test_install_id_is_stored_as_a_canonical_uuid() -> None:
    """One UUID, however the caller spells it, is one device identity (M1).

    iOS mints and keeps uppercase UUIDs; storing the caller's spelling verbatim
    would let the same phone register as two devices depending on how the app
    spelled its keystore value that day.
    """
    client = _client()
    upper = client.post(
        "/api/push/register", json={**PAYLOAD, "install_id": PAYLOAD["install_id"].upper()}
    )
    assert upper.status_code == 200, upper.text
    assert _stored_records()[0]["install_id"] == PAYLOAD["install_id"].lower()
    device_id = upper.json()["device_id"]
    for spelling in (
        PAYLOAD["install_id"],  # canonical
        PAYLOAD["install_id"].replace("-", ""),  # hyphenless hex
        "urn:uuid:" + PAYLOAD["install_id"].upper(),  # urn prefix + case
        "{" + PAYLOAD["install_id"].upper() + "}",  # brace form + case
    ):
        again = client.post("/api/push/register", json={**PAYLOAD, "install_id": spelling})
        assert again.status_code == 200, (spelling, again.text)
        assert again.json()["device_id"] == device_id, f"one UUID, one device ({spelling})"
    assert len(_stored_records()) == 1


def test_a_variant_spelling_written_by_an_older_build_does_not_fork() -> None:
    """QA-Q1 / review M1: matching must be by parsed identity, not spelling.

    The head before the canonicalisation landed stored whatever spelling the
    caller sent (verbatim, so iOS's default uppercase is a real on-disk shape).
    A register carrying another spelling of that same UUID must fold into the
    existing record — one phone listed once — not mint a second device_id.
    """
    client = _client()
    legacy_device_id = "5" * 32
    _store_path().parent.mkdir(parents=True, exist_ok=True)
    _store_path().write_text(
        json.dumps(
            {
                "devices": [
                    {
                        "device_id": legacy_device_id,
                        "platform": "ios",
                        "environment": "production",
                        "app_version": "0.9.0",
                        "install_id": PAYLOAD["install_id"].upper(),
                        "registered_at": 1_799_000_000,
                        "last_seen_at": 1_799_000_000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    response = client.post("/api/push/register", json=PAYLOAD)
    assert response.status_code == 200, response.text
    assert (
        response.json()["device_id"] == legacy_device_id
    ), "the variant-spelling record IS this device"
    assert len(_stored_records()) == 1, "one phone, one row — a spelling must not fork it"
    listed = client.get("/api/push/devices").json()["devices"]
    assert [device["device_id"] for device in listed] == [legacy_device_id]


def test_revoke_tombstones_and_repeating_it_stays_ok() -> None:
    """ADR §4 rule 1: the ROW STAYS with ``revoked_at`` — that is what makes it stick.

    A removal would let the app's next launch register straight back in under the
    same ``install_id``, which is the one thing a revoke exists to prevent. The
    retry contract is unchanged: the app repeats this on sign-out, so a repeat
    must stay ``{"ok": true}`` — and it must not move the tombstone.
    """
    client = _client()
    device_id = client.post("/api/push/register", json=PAYLOAD).json()["device_id"]
    first = client.delete(f"/api/push/devices/{device_id}")
    assert first.status_code == 200 and first.json() == {"ok": True}

    records = _stored_records()
    assert [record["device_id"] for record in records] == [
        device_id
    ], "a revoke tombstones the row; it must not delete it"
    tombstone = records[0]["revoked_at"]
    assert isinstance(tombstone, int) and tombstone > 0
    # The metadata survives — it is what the operator reads while deciding.
    assert records[0]["platform"] == "ios"

    second = client.delete(f"/api/push/devices/{device_id}")
    assert second.status_code == 200 and second.json() == {"ok": True}
    assert _stored_records()[0]["revoked_at"] == tombstone, "the revoke happened once"

    unknown = client.delete("/api/push/devices/never-existed")
    assert unknown.status_code == 200 and unknown.json() == {"ok": True}


def test_delete_before_any_register_stays_ok_and_writes_nothing() -> None:
    client = _client()
    response = client.delete("/api/push/devices/never-existed")
    assert response.status_code == 200 and response.json() == {"ok": True}
    assert not _store_path().exists(), "a no-op delete must not materialise the store"


def test_delete_works_for_any_device_in_the_registry() -> None:
    """The stolen-phone case: one caller may revoke another device (ADR §4)."""
    client = _client()
    keep = client.post("/api/push/register", json=PAYLOAD).json()["device_id"]
    stolen = client.post(
        "/api/push/register",
        json={**PAYLOAD, "install_id": "00000000-1111-2222-3333-444444444444"},
    ).json()["device_id"]
    response = client.delete(f"/api/push/devices/{stolen}")
    assert response.status_code == 200 and response.json() == {"ok": True}
    states = {
        record["device_id"]: push_devices.device_state(record) for record in _stored_records()
    }
    assert states == {keep: push_devices.STATE_LIVE, stolen: push_devices.STATE_REVOKED}


def test_list_serves_the_settings_shape_and_omits_unknown_name() -> None:
    client = _client()
    first = client.post("/api/push/register", json=PAYLOAD).json()["device_id"]
    second = client.post(
        "/api/push/register",
        json={
            **PAYLOAD,
            "platform": "android",
            "install_id": "00000000-1111-2222-3333-444444444444",
            "name": "Pixel",
        },
    ).json()["device_id"]

    before = _store_path().read_bytes()
    listed = client.get("/api/push/devices")
    assert listed.status_code == 200
    devices = listed.json()["devices"]
    # The vocabulary rides with the list, rendered from the resolver's own table
    # (ADR §3.1's ``precedence``) — a client must not have to be told separately.
    assert listed.json()["precedence"] == "revoked > unpaired > expired"
    assert listed.json()["precedence"] == push_devices.PRECEDENCE
    # Registration order, and a re-register updates in place, so the order is
    # stable under a rotation.
    assert [device["device_id"] for device in devices] == [first, second]

    entry = devices[0]
    # Absence, not null, when the app never provided a label: the Settings shape
    # (ADR §3.1) minus the fields this device has none of, name omitted (the
    # repo's absence rule).
    assert sorted(entry) == [
        "app_version",
        "credential_live",
        "device_id",
        "last_authenticated_at",
        "last_seen_at",
        "platform",
        "registered_at",
        "state",
    ]
    assert "name" not in entry
    assert devices[1]["name"] == "Pixel"
    assert entry["platform"] == "ios" and devices[1]["platform"] == "android"
    assert entry["state"] == push_devices.STATE_LIVE
    assert entry["credential_live"] is True
    assert entry["last_authenticated_at"] == entry["last_seen_at"]
    # Reading is read-only: the list walk writes nothing, so a phone opening
    # Settings cannot bump anything.
    assert _store_path().read_bytes() == before

    # The register payload is declarative (the store module's rule): a later
    # payload without a name clears it, so the app owns the label's lifecycle.
    cleared = client.post(
        "/api/push/register",
        json={
            **PAYLOAD,
            "platform": "android",
            "install_id": "00000000-1111-2222-3333-444444444444",
        },
    )
    assert cleared.json()["device_id"] == second
    assert len(_stored_records()) == 2, "clearing a name must not add a row"
    assert "name" not in client.get("/api/push/devices").json()["devices"][1]


def test_a_new_daemon_sees_the_same_registry() -> None:
    """Durability: a restart is a new app over the same config root."""
    registered = _client().post("/api/push/register", json=PAYLOAD).json()["device_id"]
    restarted = _client()
    devices = restarted.get("/api/push/devices").json()["devices"]
    assert [device["device_id"] for device in devices] == [registered]


def test_the_gate_holds_like_the_rest_of_the_api() -> None:
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    sessions = client.get("/api/sessions")
    assert sessions.status_code == 401
    for response in (
        client.post("/api/push/register", json=PAYLOAD),
        client.get("/api/push/devices"),
        client.delete("/api/push/devices/whatever"),
        client.post("/api/push/devices/whatever/unrevoke"),
    ):
        assert response.status_code == 401
        assert response.json() == sessions.json(), "same refusal shape as /api/sessions"


_INVALID_BODIES = [
    ("empty-object", {}),
    ("missing-token", {key: value for key, value in PAYLOAD.items() if key != "token"}),
    ("missing-platform", {key: value for key, value in PAYLOAD.items() if key != "platform"}),
    ("missing-environment", {key: value for key, value in PAYLOAD.items() if key != "environment"}),
    ("missing-app-version", {key: value for key, value in PAYLOAD.items() if key != "app_version"}),
    ("missing-install-id", {key: value for key, value in PAYLOAD.items() if key != "install_id"}),
    ("bad-platform", {**PAYLOAD, "platform": "windows"}),
    ("bad-environment", {**PAYLOAD, "environment": "staging"}),
    ("blank-token", {**PAYLOAD, "token": "   "}),
    ("non-string-token", {**PAYLOAD, "token": 123}),
    ("non-string-name", {**PAYLOAD, "name": 7}),
    ("install-id-not-a-uuid", {**PAYLOAD, "install_id": "not-a-uuid"}),
    ("unknown-key", {**PAYLOAD, "installID": PAYLOAD["install_id"]}),
    ("token-too-long", {**PAYLOAD, "token": "x" * 1025}),
    ("list-body", []),
    ("string-body", "text"),
    ("number-body", 12),
]


@pytest.mark.parametrize(
    ("label", "body"), _INVALID_BODIES, ids=[label for label, _ in _INVALID_BODIES]
)
def test_invalid_input_is_refused_without_side_effects(label: str, body: object) -> None:
    client = _client()
    response = client.post("/api/push/register", json=body)
    assert response.status_code == 422, label
    assert isinstance(response.json().get("error"), str) and response.json()["error"]
    assert not _store_path().exists(), f"{label}: a refused register must not create the store"


def test_unparsable_body_is_refused_without_side_effects() -> None:
    client = _client()
    response = client.post(
        "/api/push/register",
        content=b"{not json",
        headers={"content-type": "application/json"},
    )
    assert response.status_code == 422
    assert not _store_path().exists()


def test_invalid_input_leaves_an_existing_store_alone() -> None:
    client = _client()
    assert client.post("/api/push/register", json=PAYLOAD).status_code == 200
    before = _store_path().read_bytes()
    response = client.post("/api/push/register", json={**PAYLOAD, "environment": "staging"})
    assert response.status_code == 422
    assert _store_path().read_bytes() == before


def test_refusals_name_their_actual_problem() -> None:
    """N1/N2: an over-long token is not a missing one, and a typo'd identity
    key is refused by name — the caller can fix what the sentence names."""
    client = _client()
    too_long = client.post("/api/push/register", json={**PAYLOAD, "token": "x" * 1025})
    assert too_long.status_code == 422
    assert too_long.json() == {"error": "token is too long"}

    typo = client.post("/api/push/register", json={**PAYLOAD, "installID": PAYLOAD["install_id"]})
    assert typo.status_code == 422
    assert typo.json() == {"error": "unknown field(s): installID"}

    not_uuid = client.post("/api/push/register", json={**PAYLOAD, "install_id": "not-a-uuid"})
    assert not_uuid.status_code == 422
    assert not_uuid.json() == {"error": "install_id must be a UUID"}

    giant = "K" * 5000
    echo = client.post("/api/push/register", json={**PAYLOAD, giant: "x"})
    assert echo.status_code == 422
    sentence = echo.json()["error"]
    assert sentence.startswith("unknown field(s): ")
    assert len(sentence) <= len("unknown field(s): ") + push_devices.MAX_FIELD_CHARS, (
        "the echoed key names are capped (review round 2's N1 / QA's Q2, which "
        "measured a 5030-byte body before the cap)"
    )
    assert not _store_path().exists(), "no refusal may create the store"


_VALID_RECORD = {
    "device_id": "a" * 32,
    "platform": "ios",
    "environment": "production",
    "app_version": "1.0.0 (12)",
    "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
    "registered_at": 1_759_000_000,
    "last_seen_at": 1_759_000_000,
}

_CORRUPT_STORES = [
    ("not-json", b"{not json"),
    ("raw-list", b"[]"),
    ("devices-not-a-list", b'{"devices": "nope"}'),
    ("extra-top-level-key", json.dumps({"devices": [], "cursor": 1}).encode()),
    ("record-not-an-object", json.dumps({"devices": ["x"]}).encode()),
    ("record-missing-fields", json.dumps({"devices": [{"device_id": "x"}]}).encode()),
    (
        "record-bad-platform",
        json.dumps({"devices": [{**_VALID_RECORD, "platform": "windows"}]}).encode(),
    ),
    (
        "record-bad-environment",
        json.dumps({"devices": [{**_VALID_RECORD, "environment": "staging"}]}).encode(),
    ),
    (
        "record-timestamp-not-an-int",
        json.dumps({"devices": [{**_VALID_RECORD, "last_seen_at": "soon"}]}).encode(),
    ),
    # A store written by a version that (wrongly) persisted the token: unknown
    # to this build, so it must refuse rather than silently drop it on rewrite.
    (
        "record-unknown-field",
        json.dumps({"devices": [{**_VALID_RECORD, "token": "leak"}]}).encode(),
    ),
    # The identity is a UUID by contract; a stored non-UUID refuses like any
    # other shape this build cannot have produced.
    (
        "record-install-id-not-a-uuid",
        json.dumps({"devices": [{**_VALID_RECORD, "install_id": "legacy-caller-id"}]}).encode(),
    ),
    # The state markers and the credential fields are optional (an earlier
    # build's record has none) but validated when present: a marker that is not
    # a positive integer cannot be a timestamp this build wrote, and
    # ``True`` is an ``int`` to Python while being nobody's timestamp.
    (
        "marker-not-a-timestamp",
        json.dumps({"devices": [{**_VALID_RECORD, "revoked_at": "soon"}]}).encode(),
    ),
    (
        "marker-is-a-bool",
        json.dumps({"devices": [{**_VALID_RECORD, "unpaired_at": True}]}).encode(),
    ),
    (
        "credential-live-not-a-bool",
        json.dumps({"devices": [{**_VALID_RECORD, "credential_live": "yes"}]}).encode(),
    ),
    # The machine's operator key: present-but-invalid refuses rather than
    # degrading to "no key", which would silently disable the operators-only
    # route's check on a file somebody hand-edited.
    ("operator-key-not-a-string", json.dumps({"devices": [], "operator_key": 5}).encode()),
    ("operator-key-empty", json.dumps({"devices": [], "operator_key": ""}).encode()),
]


@pytest.mark.parametrize(
    ("label", "stored"), _CORRUPT_STORES, ids=[label for label, _ in _CORRUPT_STORES]
)
def test_a_corrupt_store_is_refused_and_left_byte_identical(label: str, stored: bytes) -> None:
    """A corrupt record is REFUSED, never repaired: 500 to every caller, file untouched."""
    client = _client()
    _store_path().parent.mkdir(parents=True, exist_ok=True)
    _store_path().write_bytes(stored)
    before_bytes = _store_path().read_bytes()
    before_mode = os.stat(_store_path()).st_mode

    for response in (
        client.post("/api/push/register", json=PAYLOAD),
        client.get("/api/push/devices"),
        client.delete("/api/push/devices/whatever"),
        # The operators-only route reads the same store to check the key, so a
        # store it cannot read is that route's fault too — a 500, deliberately
        # NOT the ``machine_only`` refusal (which would be a false sentence for
        # an unreadable registry).
        client.post(
            "/api/push/devices/whatever/unrevoke",
            headers={push_devices.OPERATOR_KEY_HEADER: "whatever"},
        ),
    ):
        assert response.status_code == 500, response.text
        assert "refusing" in response.json()["error"]

    assert _store_path().read_bytes() == before_bytes, f"{label}: the store must not be repaired"
    assert os.stat(_store_path()).st_mode == before_mode
    assert not list(config_dir().glob(f".{PUSH_DEVICES_STORE_NAME}.*")), "no temp litter"


def test_a_corrupt_store_leaves_its_refusal_in_the_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """M3: uvicorn runs at warning level; an unlogged 500 left no trace."""
    client = _client()
    _store_path().parent.mkdir(parents=True, exist_ok=True)
    _store_path().write_bytes(b"{not json")
    with caplog.at_level(logging.WARNING, logger="local_operator.mobile.daemon"):
        response = client.post("/api/push/register", json=PAYLOAD)
    assert response.status_code == 500
    refusals = [
        record for record in caplog.records if "push device registry" in record.getMessage()
    ]
    assert len(refusals) == 1, "one refusal, one line"
    assert str(_store_path()) in refusals[0].getMessage(), "the line names the store"


def test_registry_operations_leave_attention_db_untouched() -> None:
    """Self-contained by construction: the registry never reads or writes that store."""
    client = _client()
    attention = config_dir() / "attention.db"
    attention.parent.mkdir(parents=True, exist_ok=True)
    attention.write_bytes(b"attention-db-sentinel-not-a-real-db")
    before = (attention.read_bytes(), attention.stat().st_mtime_ns)

    device_id = client.post("/api/push/register", json=PAYLOAD).json()["device_id"]
    assert client.get("/api/push/devices").status_code == 200
    assert client.delete(f"/api/push/devices/{device_id}").status_code == 200

    assert (attention.read_bytes(), attention.stat().st_mtime_ns) == before


def test_file_mode_is_private_even_when_the_root_is_shared(tmp_path: Path) -> None:
    """0600 comes from the store's own write, not from a restrictive umask."""
    root = tmp_path
    old_umask = os.umask(0)
    try:
        push_devices.register(root, dict(PAYLOAD), now=1_700_000_000.0)
    finally:
        os.umask(old_umask)
    mode = stat.S_IMODE(os.stat(root / PUSH_DEVICES_STORE_NAME).st_mode)
    assert mode == 0o600


# ---------------------------------------------------------------------------
# S4a: the lifecycle states, the one resolver, and the way back
# ---------------------------------------------------------------------------

#: A plausible timestamp for a marker a test plants. Fixed rather than ``now``
#: so an assertion can name the exact value a rewrite must not have moved.
STAMP = 1_759_000_000


def _plant(records: list[dict[str, Any]], **store: Any) -> None:
    """Write a registry file directly — the marker combination under test.

    Directly, not through the routes: the routes can no longer *produce* a
    lapsed credential or an unpaired computer (the rotation and unpair paths are
    later slices), and planting the row is what makes these cells about the
    RESOLVER's behaviour rather than about how a marker got there.
    """
    _store_path().parent.mkdir(parents=True, exist_ok=True)
    _store_path().write_text(json.dumps({"devices": records, **store}), encoding="utf-8")


def _record(**overrides: Any) -> dict[str, Any]:
    return {**_VALID_RECORD, **overrides}


#: Every combination of markers that decides a state, with the state ADR §4's
#: precedence says wins. The three-marker row is the one that matters: two
#: consumers reading it differently is the defect the single resolver exists to
#: prevent.
_PRECEDENCE_CASES = [
    (
        "all-three-markers",
        {"revoked_at": STAMP, "unpaired_at": STAMP, "expired_at": STAMP},
        push_devices.STATE_REVOKED,
    ),
    (
        "revoked-and-expired",
        {"revoked_at": STAMP, "expired_at": STAMP},
        push_devices.STATE_REVOKED,
    ),
    (
        "unpaired-and-expired",
        {"unpaired_at": STAMP, "expired_at": STAMP},
        push_devices.STATE_UNPAIRED,
    ),
    ("expired-only", {"expired_at": STAMP}, push_devices.STATE_EXPIRED),
    ("no-marker", {}, push_devices.STATE_LIVE),
]


@pytest.mark.parametrize(
    ("label", "markers", "expected"),
    _PRECEDENCE_CASES,
    ids=[label for label, _markers, _expected in _PRECEDENCE_CASES],
)
def test_the_precedence_resolves_identically_in_the_resolver_list_and_register(
    label: str, markers: dict[str, int], expected: str
) -> None:
    """ONE resolver: the register route, ``list`` and the emit path agree.

    The three consumers are asserted against the SAME planted row in one cell,
    because the claim is not that each resolves correctly in isolation — it is
    that no two of them can read one row as two different states (ADR §4). The
    emit path is the resolver itself: ``device_state`` is what S5/S4c will call,
    and it is the only thing this cell needs to pin for it.
    """
    _plant([_record(**markers)])
    _client()  # log in; the register call below reuses one

    # 1. The resolver, as the future emit path reads it.
    assert push_devices.device_state(_stored_records()[0]) == expected

    # 2. The list the phone's Settings renders.
    listed = _client().get("/api/push/devices").json()
    assert listed["precedence"] == push_devices.PRECEDENCE
    assert listed["devices"][0]["state"] == expected

    # 3. The register route's answer for that same row.
    response = _client().post("/api/push/register", json=PAYLOAD)
    if expected in (push_devices.STATE_REVOKED, push_devices.STATE_UNPAIRED):
        assert response.status_code == 403, (label, response.text)
        assert response.json()["code"] == f"device_{expected}"
    else:
        assert response.status_code == 200, (label, response.text)
        assert response.json()["device_id"] == _VALID_RECORD["device_id"]
        assert push_devices.device_state(_stored_records()[0]) == push_devices.STATE_LIVE


def test_register_refusals_are_the_adrs_exact_codes_and_sentences() -> None:
    """ADR §3.1 spells both refusals verbatim; the app renders these strings.

    A paraphrase is a defect here for the same reason a wrong status code is:
    the phone branches on ``code`` and shows ``error``, and the sentence is the
    only thing standing between the user and "registered, but nothing arrives".
    """
    for marker, code, sentence in (
        ("revoked_at", "device_revoked", "this device was revoked on this computer"),
        ("unpaired_at", "device_unpaired", "this computer is no longer paired"),
    ):
        _plant([_record(**{marker: STAMP})])
        before = _store_path().read_bytes()
        response = _client().post("/api/push/register", json=PAYLOAD)
        assert response.status_code == 403, (marker, response.text)
        assert response.json() == {"code": code, "error": sentence}
        assert _store_path().read_bytes() == before, (
            f"{marker}: a refused register must not write — not the marker, not "
            "the key, not last_seen_at"
        )
        # The refusal is about THIS row: the same computer's other device is
        # untouched, and a fresh install_id is refused on nothing.
        fresh = _client().post(
            "/api/push/register",
            json={**PAYLOAD, "install_id": "00000000-1111-2222-3333-444444444444"},
        )
        assert fresh.status_code == 200, (marker, fresh.text)
        assert len(_stored_records()) == 2
        assert _stored_records()[0][marker] == STAMP, "the refused row did not move"


def test_a_re_register_clears_expired_at_and_mints_a_fresh_key() -> None:
    """``expired_at`` does NOT refuse: authenticating and registering IS the clearing.

    This is the state the two "sign in again" paths produce (a password rotation,
    the cookie's own TTL), and ADR §3.1 is explicit that a successful
    re-register clears it — silence would be worse than any of the three, because
    the app could show a registered device that never receives anything.
    """
    _plant([_record(expired_at=STAMP, credential_live=False, last_authenticated_at=STAMP)])
    first = _client().post("/api/push/register", json=PAYLOAD)
    assert first.status_code == 200, first.text
    key = first.json()["device_key"]
    record = _stored_records()[0]
    assert "expired_at" not in record, "the re-register IS the act that clears it"
    assert record["credential_live"] is True, "an authenticated register restores the flag"
    assert record["last_authenticated_at"] > STAMP
    assert record["registered_at"] == _VALID_RECORD["registered_at"], "identity is stable"

    # Returned once per call, and a rotation is a rotation: the previous key is
    # superseded, which is what makes a leaked key's window a single call wide.
    second = _client().post("/api/push/register", json=PAYLOAD)
    assert second.json()["device_key"] != key
    assert _stored_records()[0]["device_key"] == second.json()["device_key"]


def test_the_device_key_is_returned_once_never_listed_and_never_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A secret that rides the wire once, lives in one file, and reaches no log."""
    client = _client()
    with caplog.at_level(logging.DEBUG):
        key = client.post("/api/push/register", json=PAYLOAD).json()["device_key"]
    listed = client.get("/api/push/devices")
    assert "device_key" not in listed.text
    assert key not in listed.text
    assert PAYLOAD["token"] not in listed.text
    assert key not in caplog.text, "the key must not reach a log line"

    # It IS on disk: that is what makes it a machine-side credential the routes
    # can check later (S4c), rather than something the app must send back.
    assert _stored_records()[0]["device_key"] == key
    # And the presentation comparison is the only thing that reads it.
    record = _stored_records()[0]
    assert push_devices.device_key_matches(record, key) is True
    assert push_devices.device_key_matches(record, key + "x") is False
    assert push_devices.device_key_matches(record, "") is False
    assert push_devices.device_key_matches(record, None) is False
    assert push_devices.device_key_matches({}, key) is False


def test_a_record_from_an_earlier_build_loads_and_reads_as_live() -> None:
    """The upgrade path: the previous build wrote no markers and no credential.

    Refusing such a store would brick every existing install on upgrade, and
    inventing ``credential_live: False`` for it would be a claim nobody made.
    Absence is the truth (the repo's rule): the row is ``live`` — nothing marked
    it — and the credential fields are OMITTED from the list rather than
    defaulted.
    """
    legacy = {
        "device_id": "b" * 32,
        # The identity is ``(install_id, platform)``: this row must be the SAME
        # device the payload registers as, or the cell would be proving that a
        # second device is created — which is the opposite of what it claims.
        "platform": "ios",
        "environment": "sandbox",
        "app_version": "0.9.0 (3)",
        "install_id": _VALID_RECORD["install_id"],
        "registered_at": STAMP,
        "last_seen_at": STAMP,
    }
    _plant([legacy])
    entry = _client().get("/api/push/devices").json()["devices"][0]
    assert entry["state"] == push_devices.STATE_LIVE
    assert "credential_live" not in entry
    assert "last_authenticated_at" not in entry
    assert "device_key" not in entry
    # The next register fills all three in, on the same row and the same id.
    refreshed = _client().post("/api/push/register", json=PAYLOAD).json()
    assert refreshed["device_id"] == legacy["device_id"], "an upgrade must not fork the device"
    record = _stored_records()[0]
    assert record["credential_live"] is True
    assert record["device_key"] == refreshed["device_key"]


def test_unrevoke_is_operators_only_and_clears_the_markers() -> None:
    """The way back is the machine's act, never the device's (ADR §4 round 4 B1).

    A device session is a cookie and nothing else — which is exactly what a
    revoked phone still holds — so the route must refuse it, and refuse a wrong
    key the same way (the refusal says nothing about the key). What the operator
    gets instead is the marker cleared and NOTHING restored: no token, no
    credential, so the device must register again and registering needs a live
    credential.
    """
    client = _client()
    registered = client.post("/api/push/register", json=PAYLOAD).json()
    device_id = registered["device_id"]
    client.delete(f"/api/push/devices/{device_id}")
    # The lapse case: a credential that is not live, and a key that IS this
    # device's, so both "nothing was restored" and "the key survives" are
    # assertions about values that could have moved rather than tautologies.
    planted = _record(
        device_id=device_id,
        revoked_at=STAMP,
        credential_live=False,
        last_authenticated_at=STAMP,
        device_key=registered["device_key"],
    )
    _plant([planted])
    before = _store_path().read_bytes()

    machine_only = {
        "code": "machine_only",
        "error": "a device cannot restore itself — use the computer or your account",
    }
    refused = client.post(f"/api/push/devices/{device_id}/unrevoke")
    assert refused.status_code == 403, refused.text
    assert refused.json() == machine_only
    wrong_key = client.post(
        f"/api/push/devices/{device_id}/unrevoke",
        headers={push_devices.OPERATOR_KEY_HEADER: "not-the-key"},
    )
    assert wrong_key.status_code == 403
    assert wrong_key.json() == machine_only, "a wrong key is refused like no key at all"
    assert _store_path().read_bytes() == before, "a refused unrevoke must not write"

    restored = client.post(
        f"/api/push/devices/{device_id}/unrevoke",
        headers={push_devices.OPERATOR_KEY_HEADER: push_devices.operator_key(config_dir())},
    )
    assert restored.status_code == 200, restored.text
    assert restored.json() == {"ok": True, "device_id": device_id}
    record = _stored_records()[0]
    assert "revoked_at" not in record
    assert push_devices.device_state(record) == push_devices.STATE_LIVE
    assert record["credential_live"] is False, "no credential is restored with the marker"
    assert record["last_authenticated_at"] == STAMP, "no authentication is invented"
    assert record["device_key"] == registered["device_key"], "the key survives the round trip"

    # An UNPAIRED row is the other state this clears, and a row carrying both
    # markers is restored in one act — the operator asked for this device back,
    # and leaving ``unpaired_at`` behind would refuse it again for a reason
    # nobody chose.
    _plant(
        [_record(device_id=device_id, unpaired_at=STAMP, revoked_at=STAMP, credential_live=False)]
    )
    again = client.post(
        f"/api/push/devices/{device_id}/unrevoke",
        headers={push_devices.OPERATOR_KEY_HEADER: push_devices.operator_key(config_dir())},
    )
    assert again.status_code == 200, again.text
    assert push_devices.device_state(_stored_records()[0]) == push_devices.STATE_LIVE


def test_unrevoke_of_an_id_this_computer_never_registered_is_a_404() -> None:
    """Absent is a different answer from revoked, and the operator is told which.

    ``revoke`` is idempotent by contract (the app retries it on sign-out), so it
    answers ``{"ok": true}`` for an id it does not hold. ``unrevoke`` is a
    human's command and the opposite: "no such device" is information, and a
    silent ``ok`` would report a restore that never happened.
    """
    client = _client()
    client.post("/api/push/register", json=PAYLOAD)  # so an operator key exists
    response = client.post(
        "/api/push/devices/never-registered/unrevoke",
        headers={push_devices.OPERATOR_KEY_HEADER: push_devices.operator_key(config_dir())},
    )
    assert response.status_code == 404, response.text
    assert response.json()["code"] == push_devices.DEVICE_ABSENT_CODE
    assert "never-registered" in response.json()["error"]


def test_the_operator_key_is_minted_with_the_first_device_and_never_listed() -> None:
    """It belongs to the machine, not to a device, and no route renders it."""
    client = _client()
    assert not _store_path().exists(), "nothing is minted before a device exists"
    client.post("/api/push/register", json=PAYLOAD)
    key = push_devices.operator_key(config_dir())
    assert len(key) >= 32
    assert push_devices.operator_key(config_dir()) == key, "minted once"
    assert push_devices.verify_operator_key(config_dir(), key) is True
    assert push_devices.verify_operator_key(config_dir(), key + "x") is False
    assert push_devices.verify_operator_key(config_dir(), None) is False
    assert push_devices.verify_operator_key(config_dir(), "") is False
    listed = client.get("/api/push/devices")
    assert key not in listed.text
    assert push_devices.OPERATOR_KEY_FIELD not in listed.text
