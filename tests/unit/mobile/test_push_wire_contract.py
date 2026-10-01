"""The registry's wire, asserted against the frozen fixture contract.

Push/ack-sync S3 (ADR 0006 §3.1/§4 @ ``b03aeb15``). ``fixtures/push/`` is the
contract the app and the cloud build against, so these cells drive the **real**
routes and builders and compare their answers to the FILED literals — the same
equality guard as the payload suite, applied to the half of the freeze this repo
already implements (S4/S4a landed; the payload half is
``test_push_payload.py``).

What this file is NOT: a behaviour suite. ``test_push_devices.py`` already owns
what each route does and why; re-asserting that here would be a second copy of
the same expectations, which is the drift the fixtures exist to prevent. The
question asked here is only ever "does the wire still match the filed contract" —
so a reworded refusal sentence, a field that changed name, or a secret that
started leaking fails HERE, and nowhere else needs to change.

Isolated config root (the autouse ``isolate_environment`` fixture) and the real
``build_app`` over a ``MobileDaemon``, mirroring the sibling suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from starlette.testclient import TestClient

from local_operator.mobile import push_devices
from local_operator.mobile.daemon import MobileDaemon, build_app
from local_operator.mobile.push_devices import (
    DEVICE_REVOKED_CODE,
    DEVICE_REVOKED_MESSAGE,
    DEVICE_UNPAIRED_CODE,
    DEVICE_UNPAIRED_MESSAGE,
    MACHINE_ONLY_CODE,
    MACHINE_ONLY_MESSAGE,
    OPERATOR_KEY_FIELD,
    OPERATOR_KEY_HEADER,
    PRECEDENCE,
    PUSH_DEVICES_STORE_NAME,
)
from local_operator.paths import config_dir

FIXTURES = Path(__file__).resolve().parents[3] / "fixtures" / "push"

_TYPES: dict[str, type] = {
    "str": str,
    "int": int,
    "bool": bool,
    "object": dict,
    "list[str]": list,
    "list[object]": list,
}

#: One complete, valid register body; cells override single fields from it. The
#: token is a literal because the machine validates and drops it — nothing here
#: asserts on its value, only that it never comes back.
REGISTER = {
    "platform": "ios",
    "token": "apns-token-abcdef-0f1e2d3c",
    "environment": "production",
    "app_version": "1.0.0 (12)",
    "install_id": "9f5d1d6e-6b1a-4c6e-9b3a-7a1c2f3d4e5f",
}


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


#: Every registry shape this tree files. Kept as one list so the artefact-level
#: cells (shape and deny-list conformance) cannot silently skip a file added
#: later without someone deciding to leave it out.
REGISTRY_FIXTURES = (
    "registry-register-response.json",
    "registry-register-rotation.json",
    "registry-register-refusal-device-revoked.json",
    "registry-register-refusal-device-unpaired.json",
    "registry-list-response.json",
    "registry-unrevoke-success.json",
    "registry-unrevoke-refusal-machine-only.json",
    "registry-unrevoke-refusal-device-absent.json",
)


def _shape_blocks(fixture: dict[str, Any], *, _seen: tuple[str, ...] = ()) -> dict[str, Any]:
    """The allow-list that governs a fixture: its own, or the one it INHERITS.

    ``same_as`` is how the re-register response avoids restating the register
    response's blocks — two copies of one allow-list is how the two drift — and
    it is resolved here so the pointer is load-bearing rather than decorative
    (review round 1, N2 / QA Q3: an empty ``forbidden: []`` on that file read as
    a guard that was not one). A file that declares a block anyway must declare
    the SAME one, which is asserted rather than trusted.
    """
    declared = {
        key: fixture[key] for key in ("required", "optional", "forbidden") if key in fixture
    }
    pointer = fixture.get("same_as")
    if pointer is None:
        return declared
    assert pointer not in _seen, f"same_as cycle: {(*_seen, pointer)}"
    inherited = _shape_blocks(_fixture(pointer), _seen=(*_seen, pointer))
    for key, value in declared.items():
        assert value == inherited.get(
            key
        ), f"{key!r} is restated here and disagrees with {pointer}, the file `same_as` names"
    return {**inherited, **declared}


def _assert_shape(
    value: dict[str, Any], required: dict[str, str], optional: dict[str, str], *, where: str
) -> None:
    """The allow-list check: the named fields, each of the named type, nothing else."""
    unlisted = sorted(set(value) - set(required) - set(optional))
    assert not unlisted, f"{where}: unlisted field(s) {unlisted}"
    for name in sorted(required):
        assert name in value, f"{where}: required field {name!r} is missing"
    for name, type_name in {**required, **optional}.items():
        if name in value:
            expected = _TYPES[type_name]
            if type_name == "int":
                assert isinstance(value[name], int) and not isinstance(
                    value[name], bool
                ), f"{where}.{name}: expected int, got {type(value[name]).__name__}"
                continue
            assert isinstance(
                value[name], expected
            ), f"{where}.{name}: expected {type_name}, got {type(value[name]).__name__}"


def _assert_no_forbidden(value: object, forbidden: list[str], *, where: str) -> None:
    present = sorted(_keys(value) & set(forbidden))
    assert not present, f"{where} carries forbidden field(s): {present}"


def _keys(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            found.add(str(key))
            found |= _keys(item)
    elif isinstance(value, list):
        for item in value:
            found |= _keys(item)
    return found


def _client() -> TestClient:
    """A logged-in client over the real daemon app, under the test's HOME."""
    client = TestClient(build_app(MobileDaemon(port=0, password="pw123")), follow_redirects=False)
    assert client.post("/login", data={"password": "pw123"}).status_code in (200, 303)
    return client


def _store() -> dict[str, Any]:
    return json.loads((config_dir() / PUSH_DEVICES_STORE_NAME).read_text(encoding="utf-8"))


def _records() -> list[dict[str, Any]]:
    return _store()["devices"]


def _operator_key() -> str:
    key = _store()[OPERATOR_KEY_FIELD]
    assert isinstance(key, str) and key, "the first register mints the operator key"
    return key


def _register(client: TestClient, **overrides: Any) -> dict[str, Any]:
    response = client.post("/api/push/register", json={**REGISTER, **overrides})
    assert response.status_code == 200, response.text
    return response.json()


def test_the_register_response_matches_the_frozen_shape() -> None:
    fixture = _fixture("registry-register-response.json")
    body = _register(_client())

    assert fixture["provenance"]["kind"] == "synthetic"
    blocks = _shape_blocks(fixture)
    _assert_shape(body, blocks["required"], blocks["optional"], where="register")
    _assert_no_forbidden(body, blocks["forbidden"], where="register")
    assert body["ok"] is True
    assert body["device_key"], "the response is the key's only delivery path"


def test_a_reregister_matches_the_rotation_fixture() -> None:
    """Q29: the second call keeps the identity and re-mints the key."""
    fixture = _fixture("registry-register-rotation.json")
    client = _client()
    first = _register(client)
    second = _register(client, app_version="1.0.1 (13)")

    assert (second["device_id"], second["registered_at"]) == (
        first["device_id"],
        first["registered_at"],
    ), "a re-register keeps the identity: the fixture's two literals are the same id and stamp"
    blocks = _shape_blocks(fixture)
    _assert_no_forbidden(second, blocks["forbidden"], where="reregister")
    assert second["device_key"] != first["device_key"], "every register re-mints the key"

    record = _records()[0]
    assert push_devices.device_key_matches(record, second["device_key"]) is True
    assert (
        push_devices.device_key_matches(record, first["device_key"]) is False
    ), "the first key stops matching — there is no 409 in this model"


def test_the_list_response_matches_the_frozen_shape() -> None:
    fixture = _fixture("registry-list-response.json")
    client = _client()
    labelled = _register(client, name="Damian's iPhone")
    # A second device: a different install_id (and platform) is what makes a new
    # row — a re-register of the same identity updates in place (see the rotation
    # cell above), so registering twice would leave one row, not two.
    other = _register(
        client,
        install_id="5c2a1b0d-9e8f-4a3b-8c7d-6e5f4a3b2c1d",
        platform="android",
    )

    body = client.get("/api/push/devices").json()
    blocks = _shape_blocks(fixture)
    _assert_shape(body, blocks["required"], blocks["optional"], where="list")
    assert body["precedence"] == fixture["precedence"] == PRECEDENCE
    assert len(body["devices"]) == 2

    for index, row in enumerate(body["devices"]):
        _assert_shape(
            row,
            fixture["device_required"],
            fixture["device_optional"],
            where=f"list.devices[{index}]",
        )
        assert row["state"] in fixture["states"]
    _assert_no_forbidden(body, blocks["forbidden"], where="list")

    # The absence rule: the labelled row carries its name, the other does not —
    # and neither carries a credential fact, because S4c's per-device route is
    # what writes those.
    by_id = {row["device_id"]: row for row in body["devices"]}
    assert by_id[labelled["device_id"]]["name"] == "Damian's iPhone"
    assert "name" not in by_id[other["device_id"]]
    assert all("credential_live" not in row for row in body["devices"])


def test_the_register_refusal_bodies_match_the_frozen_literals() -> None:
    """The two state refusals, driven through the route that answers them."""
    revoked = _fixture("registry-register-refusal-device-revoked.json")
    unpaired = _fixture("registry-register-refusal-device-unpaired.json")

    client = _client()
    device = _register(client)
    assert client.delete(f"/api/push/devices/{device['device_id']}").status_code == 200

    refused = client.post("/api/push/register", json=REGISTER)
    assert refused.status_code == revoked["status"]
    assert refused.json() == revoked["example"]
    assert refused.json() == {"code": DEVICE_REVOKED_CODE, "error": DEVICE_REVOKED_MESSAGE}

    # ``unpaired_at`` has no writer in this build (unpairing a computer is S10), so
    # the marker is planted exactly where that writer will put it — the state the
    # fixture is about, not a route this slice does not own. It goes on a SECOND
    # device on purpose: the precedence is revoked > unpaired > expired, so planting
    # it on the row just revoked would answer ``device_revoked`` — which is the
    # precedence working, and would make this cell assert the wrong body.
    other_install = "5c2a1b0d-9e8f-4a3b-8c7d-6e5f4a3b2c1d"
    other = _register(client, install_id=other_install, platform="android")

    store = _store()
    row = next(d for d in store["devices"] if d["device_id"] == other["device_id"])
    row["unpaired_at"] = 1789000500
    (config_dir() / PUSH_DEVICES_STORE_NAME).write_text(json.dumps(store), encoding="utf-8")

    refused = client.post(
        "/api/push/register",
        json={**REGISTER, "install_id": other_install, "platform": "android"},
    )
    assert refused.status_code == unpaired["status"]
    assert refused.json() == unpaired["example"]
    assert refused.json() == {"code": DEVICE_UNPAIRED_CODE, "error": DEVICE_UNPAIRED_MESSAGE}


def test_the_unrevoke_success_matches_the_frozen_body() -> None:
    fixture = _fixture("registry-unrevoke-success.json")
    client = _client()
    device = _register(client)
    assert client.delete(f"/api/push/devices/{device['device_id']}").status_code == 200

    response = client.post(
        f"/api/push/devices/{device['device_id']}/unrevoke",
        headers={OPERATOR_KEY_HEADER: _operator_key()},
    )
    assert response.status_code == fixture["status"], response.text
    # The id is checked against the DEVICE'S, not the fixture's illustrative one:
    # device_id is machine-minted (uuid4().hex), so the literal is the shape's
    # example and only the field set can be compared verbatim.
    assert response.json() == {**fixture["example"], "device_id": device["device_id"]}
    _assert_no_forbidden(response.json(), _shape_blocks(fixture)["forbidden"], where="unrevoke")

    assert client.get("/api/push/devices").json()["devices"][0]["state"] == "live"
    assert (
        _register(client)["device_id"] == device["device_id"]
    ), "the way back is a cleared marker, not a new identity"


def test_the_unrevoke_refusals_match_the_frozen_bodies() -> None:
    """The operators-only gate, and the absent-row answer behind it."""
    machine_only = _fixture("registry-unrevoke-refusal-machine-only.json")
    absent = _fixture("registry-unrevoke-refusal-device-absent.json")

    client = _client()
    device = _register(client)

    refused = client.post(f"/api/push/devices/{device['device_id']}/unrevoke")
    assert refused.status_code == machine_only["status"]
    assert refused.json() == machine_only["example"]
    assert refused.json() == {"code": MACHINE_ONLY_CODE, "error": MACHINE_ONLY_MESSAGE}

    unknown = "0" * 32
    response = client.post(
        f"/api/push/devices/{unknown}/unrevoke",
        headers={OPERATOR_KEY_HEADER: _operator_key()},
    )
    assert response.status_code == absent["status"]
    assert response.json()["code"] == absent["example"]["code"]
    assert response.json()["error"] == absent["example"]["error"].replace(
        "3f1c9a7b2d4e506182a3b4c5d6e7f809", unknown
    ), "the sentence names the id asked about"


def test_a_wrong_operator_key_is_the_machine_only_refusal() -> None:
    """The predicate is the key, not locality — a wrong key is not a special case."""
    machine_only = _fixture("registry-unrevoke-refusal-machine-only.json")
    client = _client()
    device = _register(client)

    refused = client.post(
        f"/api/push/devices/{device['device_id']}/unrevoke",
        headers={OPERATOR_KEY_HEADER: "not-this-machine's-key"},
    )
    assert refused.status_code == machine_only["status"]
    assert refused.json() == machine_only["example"]


@pytest.mark.parametrize(
    "name", ["registry-register-response.json", "registry-unrevoke-success.json"]
)
def test_no_registry_response_carries_the_operator_key(name: str) -> None:
    """No shape in this tree may hand a caller the machine's own secret."""
    assert OPERATOR_KEY_FIELD not in _keys(_fixture(name)["example"])


@pytest.mark.parametrize("name", REGISTRY_FIXTURES)
def test_every_filed_example_obeys_the_shape_that_governs_it(name: str) -> None:
    """The litteral each lane transcribes, checked against its own declaration.

    The route cells check what the CODE answers; this checks the FILE, which is
    the artefact the app and the cloud copy from. Without it an example could
    grow an undeclared field, or a forbidden one, or lose a required one and
    every cell would still pass — the gap QA round 1's Q1 and the review's M2
    both filed (the same field arriving on a live response WAS caught, which is
    exactly the asymmetry worth closing).
    """
    fixture = _fixture(name)
    blocks = _shape_blocks(fixture)
    example = fixture["example"]
    where = f"{name}.example"

    if "body_required" in fixture:
        # A refusal body is exactly its declared fields — there is no optional
        # set to hedge with, so the key-set equality IS the check.
        assert set(example) == set(fixture["body_required"]), f"{where}: {sorted(example)}"
        _assert_shape(example, fixture["body_required"], {}, where=where)

    if blocks.get("required") or blocks.get("optional"):
        _assert_shape(example, blocks.get("required", {}), blocks.get("optional", {}), where=where)
    _assert_no_forbidden(example, blocks.get("forbidden", []), where=where)

    if "device_required" in fixture:
        rows = example["devices"]
        assert rows, f"{where}: the list example must show at least one device"
        for index, row in enumerate(rows):
            _assert_shape(
                row,
                fixture["device_required"],
                fixture["device_optional"],
                where=f"{where}.devices[{index}]",
            )
        required = set(fixture["device_required"])
        optional = set(fixture["device_optional"])
        assert any(set(row) == required for row in rows), (
            f"{where}: no row shows the ABSENCE rule — a device carrying only its "
            "always-present fields and none of the optional ones"
        )
        assert any(
            required < set(row) <= required | optional for row in rows
        ), f"{where}: no row shows a device that also carries its optional fields"


def test_the_rotation_fixture_inherits_its_shape_instead_of_restating_it() -> None:
    """``same_as`` is load-bearing, and a disagreeing restatement is refused."""
    rotation = _fixture("registry-register-rotation.json")
    assert rotation["same_as"] == "registry-register-response.json"
    for key in ("required", "optional", "forbidden"):
        assert (
            key not in rotation
        ), f"{key!r} is restated: two copies of one allow-list is how they drift"
    assert _shape_blocks(rotation) == _shape_blocks(_fixture(rotation["same_as"]))

    # The resolver's drift check bites — the guard is the resolver, not the
    # absence of the key.
    with pytest.raises(AssertionError, match="restated here and disagrees"):
        _shape_blocks({**rotation, "required": {"ok": "str"}})


@pytest.mark.parametrize("name", REGISTRY_FIXTURES)
def test_no_registry_fixture_is_silently_unguarded(name: str) -> None:
    """Every shape either declares a real deny list or carries an exact-set guard.

    The anti-regression for QA round 1's Q3 / the review's N2: an empty
    ``forbidden: []`` reads as a guard while refusing nothing, and a fixture with
    neither a deny list nor a closed body is a shape nothing checks at all.
    """
    fixture = _fixture(name)
    if "forbidden" in fixture:
        assert fixture["forbidden"], f"{name}: an empty deny list is a guard that is not one"
        return
    assert (
        "same_as" in fixture or "body_required" in fixture
    ), f"{name} declares no deny list and no exact-set body: nothing guards this shape"
