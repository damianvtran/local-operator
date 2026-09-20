"""The relay's half of stage D: pairing, challenges, and the narrowed body scrub.

WHAT THE RELAY IS ALLOWED TO DO, and why each cell exists. The measured threat
(#1310) is a same-uid process — the model's own ``bash`` tool — that reads the
session record and drives the loopback control socket. The relay's HTTP surface is
reachable by the same class of process with the portal password, so the question
this file answers is not "can an attacker reach it" (they can) but "can reaching
it produce authority" (it cannot):

* ``/api/pair`` is a COURIER's endpoint: it checks a code the operator minted and
  records a public key. It holds no operator key, so it cannot make anything a
  signer — the cell asserts the PENDING state it leaves behind is inert;
* ``/api/sessions/<id>/operator/challenge`` mints nothing itself: it forwards one
  ordinary frame on the connection it already has. A challenge is worth exactly one
  signature, which only the paired phone can make;
* the body scrub DROPS ``operator_cap`` (machine-held proof material, mintable here,
  therefore forgeable from outside) and ADMITS ``operator_sig``/``operator_key_id``/
  ``operator_cert`` (unforgeable, single-use, worthless to replay), which is the
  narrowing stage D was written to make.

The runtime's judgement of a signature is not re-tested here — the seam file owns
that — so ``daemon.request`` is spied on where the claim is about what the relay
FORWARDS. Isolated config dir; never a real keychain.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from starlette.testclient import TestClient

from local_operator.mobile.daemon import MobileDaemon, SessionEntry, build_app
from local_operator.mobile.types import SessionRecord
from local_operator.operator import devices


def _record(*, pid: int = 4242, session_id: str = "relay-session") -> SessionRecord:
    return SessionRecord(
        pid=pid,
        kind="tui",
        session_id=session_id,
        conversation_name="relay fixture",
        cwd="/tmp",
        model_label="fixture",
        control_port=1,
        control_key="fixture-key",
    )


def _logged_in(daemon: MobileDaemon, password: str = "pw123") -> TestClient:
    client = TestClient(build_app(daemon), follow_redirects=False)
    client.post("/login", data={"password": password})
    return client


def _point() -> bytes:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.public_key().public_bytes(Encoding.X962, PublicFormat.UncompressedPoint)


# ---------------------------------------------------------------------------
# Pairing: the relay is a courier and the cell says what that buys it
# ---------------------------------------------------------------------------


def test_pairing_requires_the_cookie_and_a_live_code(tmp_path: Path, monkeypatch) -> None:
    """Three refusals, and the third is the one that matters.

    An unauthenticated caller gets 401. A wrong code and NO LIVE CODE are answered
    identically, deliberately: distinguishing them would tell a guesser whether a
    pairing window is open on this machine, which is the one fact they need to time
    an attempt.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    anonymous = TestClient(build_app(daemon), follow_redirects=False)

    spki = base64.urlsafe_b64encode(_point()).decode().rstrip("=")
    unauthenticated = anonymous.post("/api/pair", json={"code": "x", "spki": spki, "name": "p"})
    assert unauthenticated.status_code == 401, unauthenticated.text

    client = _logged_in(daemon)
    # No code has been minted on this machine yet.
    assert (
        client.post("/api/pair", json={"code": "00" * 16, "spki": spki, "name": "p"}).status_code
        == 403
    )

    code = devices.begin_pairing(tmp_path)
    wrong = client.post("/api/pair", json={"code": "ff" * 16, "spki": spki, "name": "p"})
    assert wrong.status_code == 403, wrong.text
    assert wrong.json() == {"error": "that pairing code is not valid"}
    assert devices.list_pending(tmp_path) == [], "a refused claim left a request behind"

    accepted = client.post("/api/pair", json={"code": code, "spki": spki, "name": "my phone"})
    assert accepted.status_code == 200, accepted.text
    device_id = accepted.json()["device_id"]
    pending = devices.list_pending(tmp_path)
    assert [row["device_id"] for row in pending] == [device_id]
    # WHAT THE RELAY BOUGHT ITSELF: nothing. A pending request is not a device and
    # carries no signature, so nothing on this machine may sign for that phone yet.
    assert devices.list_devices(tmp_path) == []
    assert devices.paired_certificate(tmp_path) is None


def test_pairing_refuses_a_public_point_that_is_not_a_p256_point(
    tmp_path: Path, monkeypatch
) -> None:
    """The phone's key is the first wire value this endpoint sees, and it is bounded.

    A caller that sends a private key, a blob or an empty string is refused before
    anything is written — a pending request the operator might later sign is the
    wrong place to discover that parsing failed.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    client = _logged_in(daemon)
    code = devices.begin_pairing(tmp_path)
    for bad in ("", "Zm9vYmFy", base64.urlsafe_b64encode(b"\x04" + b"\x00" * 4).decode()):
        reply = client.post("/api/pair", json={"code": code, "spki": bad, "name": "p"})
        assert reply.status_code == 422, (bad, reply.text)
    assert devices.list_pending(tmp_path) == []


def test_a_revoked_device_cannot_re_pair(tmp_path: Path, monkeypatch) -> None:
    """Un-revocation by re-pairing is closed by the DERIVED id.

    A revoked phone that claims a fresh code would otherwise get a brand new
    certificate and quietly become a signer again. The id is derived from the key,
    so the relay asks the revocation list about the same string the revocation
    named, and refuses.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    client = _logged_in(daemon)
    point = _point()
    device_id = devices.new_device_id(point)
    devices.record_revocation(tmp_path, device_id)

    code = devices.begin_pairing(tmp_path)
    reply = client.post(
        "/api/pair",
        json={
            "code": code,
            "spki": base64.urlsafe_b64encode(point).decode().rstrip("="),
            "name": "p",
        },
    )
    assert reply.status_code == 403, reply.text
    assert reply.json() == {"error": "this device has been revoked"}
    assert devices.list_pending(tmp_path) == []


def _install_operator_key(config_root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A real ``file-only`` operator key and the anchor naming it.

    The ``file-only`` backend is the only one a test may create — the Secure
    Enclave cannot be created in a throwaway keychain, and writing the operator's
    login keychain is forbidden — and the anchor root is patched in-process so no
    privileged path is written. Nothing here can raise a keychain prompt.
    """
    from local_operator.operator.keychain import FILE_ONLY
    from local_operator.operator.sign import anchor_for_handle, create_key

    anchor_root = config_root / "anchor-root"
    anchor_root.mkdir(exist_ok=True)
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE", anchor_root, raising=False
    )
    handle = create_key(config_root=config_root, preference=FILE_ONLY)
    return anchor_for_handle(handle, label="relay-test")


def test_pair_status_reports_pending_then_the_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The one way the phone ever learns the string it must present.

    The certificate is minted on the MACHINE by the operator's gesture, so the
    phone reads it back from here. It is public data — a signed statement over a
    public key — and the cell asserts the response holds no private material and no
    field the phone could use to mint anything.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    anchor = _install_operator_key(tmp_path, monkeypatch)
    daemon = MobileDaemon(port=0, password="pw123")
    client = _logged_in(daemon)

    point = _point()
    device_id = devices.new_device_id(point)
    before = client.get(f"/api/pair/{device_id}")
    assert before.status_code == 200
    assert before.json() == {"paired": False, "device_id": device_id}

    from local_operator.operator.sign import issue_device_cert, load_signer
    from local_operator.operator.verify import read_device_cert

    signer = load_signer(config_root=tmp_path, backend_name="file-only")
    assert signer is not None, "the fixture's operator key did not load"
    try:
        certificate = issue_device_cert(
            device_spki=point, device_id=device_id, label="my phone", signer=signer
        )
    finally:
        signer.close()
    parsed = read_device_cert(certificate)
    assert parsed is not None
    devices.write_device_cert(
        tmp_path, certificate=certificate, parsed=parsed, operator_key_id=anchor.key_id
    )

    after = client.get(f"/api/pair/{device_id}")
    assert after.status_code == 200, after.text
    body = after.json()
    assert body["paired"] is True
    assert body["device_id"] == device_id
    assert body["certificate"] == certificate
    assert body["operator_key_id"] == anchor.key_id
    assert body["scope"] == ["loosen", "approve"]
    # No private half, and nothing that could sign: the response is a statement, an
    # id and a scope.
    assert "private" not in after.text.lower()
    assert set(body) == {
        "paired",
        "device_id",
        "certificate",
        "operator_key_id",
        "scope",
        "exp",
        "name",
    }


def test_pair_status_refuses_a_device_id_that_is_not_an_id(tmp_path: Path, monkeypatch) -> None:
    """The path parameter becomes a filename, so a traversal is a 422 not a read."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    client = _logged_in(daemon)
    reply = client.get("/api/pair/..%2F..%2Fetc")
    assert reply.status_code in (404, 422), reply.text


# ---------------------------------------------------------------------------
# The challenge: pass-through, and none of it minted here
# ---------------------------------------------------------------------------


def test_the_challenge_endpoint_forwards_one_ordinary_frame(tmp_path: Path, monkeypatch) -> None:
    """THE PASS-THROUGH, and the two fields it refuses to carry.

    The challenge rides the relay's EXISTING session connection — the runtime binds
    it to that connection, so a challenge minted anywhere else would be refused —
    and the endpoint's own body is stripped of every authority field, because its
    only output is a challenge. A body carrying a signature here is a caller
    confusing two endpoints, and relaying it would push a half-finished frame
    through the ordinary path.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    record = _record(session_id="challenge-session")
    daemon.table.entries[record.pid] = SessionEntry(record)
    seen: list[tuple[Any, ...]] = []

    async def fake_request(pid: int, op: str, **fields: Any) -> dict[str, Any]:
        seen.append((pid, op, fields))
        return {"op": "ack", "req": 1, "challenge": "ab" * 32, "expires_s": 30}

    monkeypatch.setattr(daemon, "request", fake_request)
    client = _logged_in(daemon)

    reply = client.post(
        "/api/sessions/challenge-session/operator/challenge",
        json={
            "action": "loosen",
            "request_id": "",
            "operator_cap": "cd" * 32,
            "operator_sig": "ee" * 8,
            "operator_key_id": "ff" * 16,
            "operator_cert": "cert",
        },
    )
    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert body["challenge"] == "ab" * 32
    assert body["expires_s"] == 30
    assert body["session_id"] == "challenge-session"
    assert body["action"] == "loosen"

    assert len(seen) == 1
    pid, op, fields = seen[0]
    assert (pid, op) == (record.pid, "operator_challenge")
    assert fields == {
        "action": "loosen",
        "request_id": "",
    }, "the relay forwarded an authority field on the challenge endpoint"


def test_the_challenge_endpoint_refuses_an_unknown_action_or_session(
    tmp_path: Path, monkeypatch
) -> None:
    """A named action and a live session, or nothing is minted.

    ``action`` chooses what the signature will be accepted FOR, so an unrecognised
    one is a 422 rather than a guess; an unknown session is a 409 — the same answer
    every other command route gives, because it is the same fact.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    record = _record(session_id="live-session")
    daemon.table.entries[record.pid] = SessionEntry(record)
    client = _logged_in(daemon)

    bad_action = client.post(
        "/api/sessions/live-session/operator/challenge",
        json={"action": "everything", "request_id": ""},
    )
    assert bad_action.status_code == 422, bad_action.text

    missing_session = client.post(
        "/api/sessions/nowhere/operator/challenge", json={"action": "loosen", "request_id": ""}
    )
    assert missing_session.status_code == 409, missing_session.text


def test_the_challenge_endpoint_requires_the_cookie(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    anonymous = TestClient(build_app(daemon), follow_redirects=False)
    reply = anonymous.post(
        "/api/sessions/anything/operator/challenge", json={"action": "loosen", "request_id": ""}
    )
    assert reply.status_code == 401, reply.text


# ---------------------------------------------------------------------------
# The scrub, narrowed: what the relay drops and what it now carries
# ---------------------------------------------------------------------------


def test_the_command_route_drops_the_capability_and_admits_the_signature(
    tmp_path: Path, monkeypatch
) -> None:
    """THE NARROWING, in one cell, on the one route that carries a command.

    Two directions, and they are not symmetrical:

    * ``operator_cap`` is DROPPED, permanently. It is machine-held proof material —
      the relay mints its own when it is the spawner — so a value arriving in an
      HTTP body can only be a forgery, and the relay holds none for a runtime it
      did not spawn anyway;
    * the three signature fields are ADMITTED, because the phone cannot be a signer
      while the relay refuses to carry its signature. They are unforgeable (the
      private half never leaves the device) and single-use (the challenge is popped
      by the runtime before verification), so replaying them from a local process
      gains nothing that reading the record did not already give it.

    ``operator_handshake`` is deliberately not in either list: the relay COMPUTES
    its own in ``request`` and overwrites whatever a body carried, so a forged one is
    replaced rather than refused.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    record = _record(session_id="scrub-session")
    daemon.table.entries[record.pid] = SessionEntry(record)
    seen: list[dict[str, Any]] = []

    async def fake_request(pid: int, op: str, **fields: Any) -> dict[str, Any]:
        seen.append(dict(fields))
        return {"op": "ack", "detail": "ok"}

    monkeypatch.setattr(daemon, "request", fake_request)
    client = _logged_in(daemon)

    reply = client.post(
        "/api/sessions/scrub-session/command",
        json={
            "op": "slash_result",
            "command": "approvals",
            "args": "auto",
            "images": [],
            "operator_cap": "aa" * 32,
            "operator_sig": "bb" * 8,
            "operator_key_id": "cc" * 16,
            "operator_cert": "operator-signed-statement",
        },
    )
    assert reply.status_code == 200, reply.text
    assert len(seen) == 1
    forwarded = seen[0]
    assert "operator_cap" not in forwarded, "the relay forwarded a capability from a body"
    assert forwarded["operator_sig"] == "bb" * 8
    assert forwarded["operator_key_id"] == "cc" * 16
    assert forwarded["operator_cert"] == "operator-signed-statement"


def test_the_command_route_still_refuses_a_malformed_signature_field(
    tmp_path: Path, monkeypatch
) -> None:
    """Admitted is not unvalidated: the wire shape is checked before dispatch.

    A credential that reaches the seam untyped is a credential the seam has to
    defend against, so a non-hex signature or an over-long certificate is a 422 at
    the boundary rather than a runtime refusal later — the two cases read very
    differently to whoever is debugging a phone that cannot approve.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    record = _record(session_id="scrub-session")
    daemon.table.entries[record.pid] = SessionEntry(record)
    client = _logged_in(daemon)

    base = {"op": "slash_result", "command": "approvals", "args": "auto", "images": []}
    for bad in (
        {"operator_sig": "not-hex"},
        {"operator_key_id": "short"},
        {"operator_cert": "x" * 5000},
    ):
        reply = client.post("/api/sessions/scrub-session/command", json={**base, **bad})
        assert reply.status_code == 422, (bad, reply.text)


def test_the_route_table_names_the_paths_this_build_actually_serves(
    tmp_path: Path, monkeypatch
) -> None:
    """A pin on the URLs, so the design document cannot drift from the code.

    The design writes ``POST /v1/mobile/operator/challenge``; THIS repository has
    no ``/v1/mobile`` namespace at all — every route it serves is under ``/api``
    (``mobile/daemon.py``'s route list, and the portal's ``api.ts``) — so the path
    implemented is ``/api/sessions/<id>/operator/challenge`` and the doc records the
    deviation. Pinned rather than left to a reader, because a second namespace
    invented to match a document is exactly the kind of duplication this project's
    conventions forbid.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    app = build_app(daemon)
    paths = {getattr(route, "path", "") for route in app.routes}
    assert "/api/sessions/{session_id:str}/operator/challenge" in paths
    assert "/api/pair" in paths
    assert "/api/pair/{device_id:str}" in paths
    assert not [path for path in paths if path.startswith("/v1/")], paths


def test_the_portal_and_the_relay_agree_about_the_namespace() -> None:
    """The client that would have to use a ``/v1/mobile`` path does not have one.

    Read from ``api.ts``, which is the single place the portal builds URLs, so the
    deviation recorded in the design document is checkable against the code that
    would break if a second namespace were ever introduced.
    """
    api = (
        Path(__file__).resolve().parents[3] / "local_operator" / "mobile" / "web" / "src" / "api.ts"
    )
    source = api.read_text(encoding="utf-8")
    assert "/v1/mobile" not in source, "the portal and the relay disagree about the namespace"
    assert "/api/sessions/" in source, "the portal no longer targets the relay's routes"


def _anchor_with_revocation(monkeypatch, device_id: str) -> None:
    """Make `trust.load_anchor` report an anchor whose revocation list names a device.

    The ANCHOR is the authoritative list (the runtime consults it), and the local
    ``revoked.json`` under the config root is the relay's own copy. A test cannot
    create the root-owned file, so the root-owned FACT is supplied at the one read
    the guard performs — which is what makes this cell about the guard's CHOICE of
    list rather than about file permissions.
    """
    from local_operator.operator import trust

    def load(uid: Any = None) -> Any:
        return trust.AnchorLoad(
            anchor=trust.OperatorAnchor(
                key_id="00" * 16,
                spki=b"\x04" + b"\xab" * 64,
                backend="file-only",
                presence=False,
                label="test",
                created_at=0,
                devices=({"device_id": device_id, "revoked": True},),
            ),
            path=trust.anchor_path(uid),
            root_owned=True,
            reason="ok",
            exists=True,
        )

    monkeypatch.setattr(trust, "load_anchor", load)


def test_a_revocation_recorded_only_in_the_ANCHOR_still_stops_a_re_pair(
    tmp_path: Path, monkeypatch
) -> None:
    """UX round 6, U5: the two revocation lists disagreed, and the relay read the wrong one.

    Measured before this fix: with the revocation recorded in the ANCHOR but not in
    the local record — which is what an operator who edits the root-owned anchor
    produces, and the documented way to revoke a device whose certificate this
    machine still holds — ``POST /api/pair`` accepted a fresh code from that device
    with HTTP 200. The guard asked ``is_revoked_here``, the config-root copy, while
    the authoritative list lives in the anchor.
    """
    from local_operator.operator import devices as device_module

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    client = _logged_in(daemon)
    point = _point()
    device_id = devices.new_device_id(point)
    _anchor_with_revocation(monkeypatch, device_id)

    # THE STATE THAT MATTERED: the anchor says revoked, the local record says nothing.
    assert device_module.is_revoked_here(tmp_path, device_id) is False
    assert device_module.is_revoked(tmp_path, device_id) is True

    code = devices.begin_pairing(tmp_path)
    reply = client.post(
        "/api/pair",
        json={
            "code": code,
            "spki": base64.urlsafe_b64encode(point).decode().rstrip("="),
            "name": "p",
        },
    )
    assert reply.status_code == 403, reply.text
    assert reply.json() == {"error": "this device has been revoked"}, reply.text
    assert devices.list_pending(tmp_path) == []


def test_a_refused_command_carries_the_TYPED_code_to_the_phone(tmp_path: Path, monkeypatch) -> None:
    """UX round 6, U6: the phone needs the category, not English to pattern-match.

    The portal decided whether to re-sign by matching substrings of the runtime's
    sentence (``"only the operator can allow it"``), and that sentence has been
    rewritten twice between revisions. The runtime already sends a typed code, and
    the relay already decodes it into the exception — it simply rendered only the
    prose onto the wire.

    THE REAL ``daemon.request`` IS DRIVEN, not replaced: the cell stands a fake
    WRITER in for the session socket, reads the frame the relay really built, and
    answers it with the error frame a runtime sends. That keeps the decode, the
    raise and the route's rendering in one production path, which is the whole
    point of asserting the field rather than the sentence.
    """
    import local_operator.mobile.daemon as daemon_module
    from local_operator.harness.approval import OPERATOR_AUTHORITY_REQUIRED_UNCONFIGURED_NOTICE

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    daemon = MobileDaemon(port=0, password="pw123")
    record = _record(session_id="typed-code-session")
    entry = SessionEntry(record)
    daemon.table.entries[record.pid] = entry

    class FakeWriter:
        """The session socket, in the one place this cell needs it: the write."""

        def __init__(self) -> None:
            self.written: list[dict[str, Any]] = []

        def write(self, payload: bytes) -> None:
            frame = json.loads(payload.decode())
            self.written.append(frame)
            # ANSWER AS THE RUNTIME WOULD, on the future the relay just parked:
            # a typed refusal with its copy, exactly the shape
            # `server._on_request` produces for an authority-increasing op.
            daemon._pending_reqs[(record.pid, frame["req"])].set_result(
                {
                    "op": "error",
                    "req": frame["req"],
                    "message": OPERATOR_AUTHORITY_REQUIRED_UNCONFIGURED_NOTICE,
                    "error_code": "operator_authority_unconfigured",
                    "error_trigger": "slash_result",
                }
            )

        async def drain(self) -> None:
            return None

    writer = FakeWriter()
    entry.writer = writer  # type: ignore[assignment]
    client = _logged_in(daemon)
    reply = client.post(
        "/api/sessions/typed-code-session/command",
        json={"op": "slash_result", "command": "approvals", "args": "auto", "images": []},
    )
    assert len(writer.written) == 1, writer.written
    assert reply.status_code == 422, reply.text
    body = reply.json()
    # The COPY is still carried verbatim, so every client that exists today is
    # unaffected: `error` keeps its shape and the new field is additive.
    assert "lop operator install" in body["error"], body
    # ...and the CATEGORY rides beside it, for a client that has to decide what to
    # offer next rather than reword a sentence it does not own.
    assert body["code"] == "operator_authority_unconfigured", body
