"""The macOS key agent: the protocol, the pre-flight, and the failure semantics.

WHY THESE TESTS EXIST AND WHAT THEY CANNOT DO. The helper they speak to is a
Developer-ID-signed app bundle that only a macOS release build produces, and the
real one can only be exercised by an entitled process on a Mac with a Secure
Enclave. So this file drives a FAKE helper — a script that answers the same
JSON on the same argv — and covers the whole client: every documented failure
state, the escape hatches that must not exist, and the one rule the design turns
on (nothing here may conclude "no operator key" from a query the runtime made
itself).

What is NOT covered here, stated rather than implied: the helper's own C
behaviour (asserted against the SDK header and the C source in
``test_operator_authority.py``, and here where a defect in a C BRANCH is what a
fake reply cannot see — see ``_HELPER_C``), the real entitlement (only a signed
bundle on a real keychain can show it, and QA round 1 showed it outside this
suite), and the presence prompt (it is a sheet on an operator's screen; unproven
by design, see the design document's §9).

NO TEST HERE WRITES TO ANY KEYCHAIN. The fake is a script; it has no keychain
code at all. The tag every test passes is a test tag, never
``keychain.APPLICATION_TAG``.
"""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.operator import keychain
from local_operator.operator.macos import keyagent

#: The helper's source, in-tree, for the two assertions a fake reply cannot make:
#: that ``create`` consults the tag before it generates, and that ``doctor`` performs an
#: operation the OS actually gates. Its ABSENCE would make those checks vacuous, so the
#: path is asserted to exist rather than skipped on — the same rule
#: ``test_operator_authority`` applies to this same file for its symbol pins.
_HELPER_C = (
    Path(__file__).resolve().parents[3] / "packaging" / "macos" / "lop-keyagent" / "se-keyagent.c"
)


def _helper_body() -> str:
    """The C source with its comments removed, so a prose mention cannot pass a check."""
    source = _HELPER_C.read_text()
    assert "se-keyagent.c" in source or source, "the helper source is empty"
    return re.sub(r"/\*.*?\*/", "", source, flags=re.S)


#: A real uncompressed P-256 point, exported from a Secure Enclave key created on
#: the development host (public data — the private half never leaves the Enclave).
#: A fixed value keeps the expected ``key_id`` deterministic.
_A_REAL_P256_POINT = bytes.fromhex(
    "04db295b8129a1ba169a538a3b69f1caa1dbdd2ed34332c33ec5d3e0364ecb87"
    "8845a59a4f54c9728f8ed81905ad9765ad752b9e1e9080a528cc37e8098534167b"
)

#: A DER ECDSA signature, so pass-through can be asserted byte for byte.
_A_DER_SIGNATURE = bytes.fromhex("3045022100" + "ab" * 32 + "0220" + "cd" * 32)

#: The message every ``sign`` test sends. Nothing frames it, so the fake must
#: receive these bytes and nothing else.
_MESSAGE = b"local-operator/operator/v1|loosen|session-abc|challenge-deadbeef"

_FAKE_HELPER = '''\
#!/usr/bin/env python3
"""A stand-in for the signed helper: same argv, same JSON, no keychain.

Its configuration is a FILE beside this script rather than an environment
variable, so a behaviour can never leak from one test into the next: the first
version read two env vars and a stale log path from a finished test made the
helper die before answering, which read exactly like a refused call.
"""
import base64
import json
import os
import signal
import sys
import time
from pathlib import Path

POINT = bytes.fromhex("{point}")
DER = bytes.fromhex("{der}")
CONFIG = json.loads(Path(__file__).with_name("fake.json").read_text())
MODE = CONFIG["mode"]
LOG = CONFIG["log"]


def b64(raw):
    return base64.urlsafe_b64encode(raw).decode().rstrip("=")


def reply(obj):
    print(json.dumps(obj))
    sys.exit(0)


argv = sys.argv[1:]
verb = argv[0] if argv else ""
payload = sys.stdin.buffer.read() if verb == "sign" else b""
if LOG:
    with open(LOG, "a") as fh:
        fh.write(json.dumps({{"argv": argv, "payload": payload.hex()}}) + "\\n")

if MODE == "killed":
    os.kill(os.getpid(), signal.SIGKILL)
if MODE == "slow" and verb == "sign":
    time.sleep(30)
if MODE == "cancelled" and verb == "sign":
    print(json.dumps({{"ok": False, "protocol": 1, "site": "signature",
                      "status": -128, "detail": "errSecUserCanceled"}}))
    sys.exit(2)
if MODE == "refused" and verb == "create":
    print(json.dumps({{
        "ok": False, "protocol": 1, "site": "key generation", "status": -34018,
        "detail": "failed to add key to keychain: <SecKeyRef:('com.apple.setoken')> 0x10375c430",
        "refusals": [
            {{"site": "key generation", "status": -34018,
              "protection": "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
              "detail": "failed to add key: <SecKeyRef:('com.apple.setoken')> 0x10137ede0"}},
            {{"site": "key generation", "status": -34018,
              "protection": "kSecAttrAccessibleWhenUnlockedThisDeviceOnly",
              "detail": "failed to add key: <SecKeyRef:('com.apple.setoken')> 0x75929d8380"}},
        ],
    }}))
    sys.exit(4)
if MODE == "protocol-mismatch" and verb == "create":
    reply({{"ok": True, "protocol": 99, "spki": b64(POINT), "reused": False}})
if MODE == "garbage":
    print("this is not the JSON you are looking for")
    sys.exit(0)
if MODE == "unknown-verb":
    print(json.dumps({{"ok": False, "protocol": 1, "site": "usage", "status": 5,
                      "detail": "unknown verb"}}))
    sys.exit(5)
if MODE == "usage-other-protocol":
    # A build that answers with SOMEBODY ELSE'S protocol number, plus the usage exit
    # code: the reply cannot be read, which is the one case exit 5 really does mean
    # "you and I do not agree" (QA round 1, Q2).
    print(json.dumps({{"ok": False, "protocol": 99, "site": "usage", "status": 5,
                      "detail": "unknown verb"}}))
    sys.exit(5)
if MODE == "no-key" and verb in ("public", "exists"):
    if verb == "exists":
        reply({{"ok": True, "protocol": 1, "present": False}})
    print(json.dumps({{"ok": False, "protocol": 1, "site": "key lookup",
                      "status": -25300, "detail": "errSecItemNotFound"}}))
    sys.exit(3)
if MODE == "badpoint" and verb in ("create", "public"):
    reply({{"ok": True, "protocol": 1, "spki": b64(POINT[:-1])}})
if MODE == "nospki" and verb in ("create", "public"):
    reply({{"ok": True, "protocol": 1}})
if verb == "create":
    reply({{"ok": True, "protocol": 1, "spki": b64(POINT), "reused": MODE == "reused",
           "rung": "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly"}})
if verb in ("public", "exists"):
    reply({{"ok": True, "protocol": 1, "present": True, "spki": b64(POINT)}})
if verb == "sign":
    reply({{"ok": True, "protocol": 1, "signature": b64(DER)}})
if MODE == "doctor-refused":
    # DOCTOR AT THE GATED OPERATION (QA round 1, Q4): the query answers -25300, the
    # profile is there, and GENERATION is refused — which is what a bundle whose
    # entitlement the OS will not honour actually looks like.
    if verb == "doctor":
        print(json.dumps({{
            "ok": False, "protocol": 1,
            "rung": "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
            "keychain": "ok", "keychain_status": -25300, "profile": "ok",
            "generation": "errSecMissingEntitlement", "generation_status": -34018,
            "detail": "the key agent could not create in the keychain, so its entitlement \
is not in effect (errSecMissingEntitlement)",
        }}))
        sys.exit(4)
if verb == "doctor":
    reply({{"ok": True, "protocol": 1, "rung": "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
           "keychain": "ok", "keychain_status": 0, "profile": "ok",
           "generation": "ok", "generation_status": 0}})
if verb == "purge":
    reply({{"ok": True, "protocol": 1, "deleted": 0}})
sys.exit(5)
'''

TEST_TAG = "com.local-operator.operator.test.keyagent"


def _fake_app(tmp_path: Path, mode: str = "ok", log_path: Path | None = None) -> Path:
    """A bundle-shaped directory holding an executable that speaks the protocol.

    The layout matters: the client resolves ``Contents/MacOS/lop-keyagent`` inside
    a ``.app``, so a fake has to sit where the real one does or the path resolution
    under test would not be the shipping one. The log defaults to a file beside the
    fake, always created, so a test can read the protocol without arranging paths.
    """
    app = tmp_path / "lop-keyagent.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True)
    exe = macos / keyagent.EXECUTABLE_NAME
    exe.write_text(_FAKE_HELPER.format(point=_A_REAL_P256_POINT.hex(), der=_A_DER_SIGNATURE.hex()))
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (app / "Contents" / "Info.plist").write_text("<plist/>")
    # Shaped like the real bundle, including the profile — a fake without it could not
    # catch a pre-flight that stopped checking the load-bearing file.
    (app / "Contents" / "embedded.provisionprofile").write_bytes(
        b"\x30\x82" + f"{keyagent.TEAM_IDENTIFIER}.{keyagent.BUNDLE_IDENTIFIER}".encode()
    )
    log = log_path if log_path is not None else macos / "helper.log"
    log.write_text("")
    set_mode(app, mode, log=log)
    return app


def set_mode(app: Path, mode: str, *, log: Path | None = None) -> None:
    """Point the fake at one behaviour, mid-test if needed."""
    macos = app / "Contents" / "MacOS"
    existing = (
        json.loads((macos / "fake.json").read_text()) if (macos / "fake.json").exists() else {}
    )
    (macos / "fake.json").write_text(
        json.dumps({"mode": mode, "log": str(log or existing.get("log", macos / "helper.log"))})
    )


def _fake_log(app: Path) -> Path:
    """Where the fake beside ``app`` records what it was asked."""
    return Path(json.loads((app / "Contents" / "MacOS" / "fake.json").read_text())["log"])


@pytest.fixture()
def fake_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Wires the fake bundle into BOTH seams the product reaches it through.

    ``keyagent.helper_bundle_path()`` is what the production client resolves, and
    ``keyagent.verify_bundle`` is the real pre-flight — which would refuse a fake
    (there is no signature on it) and, on a non-Mac, could not even run. Patching
    both is the whole point of a protocol-level fake.
    """

    def build(mode: str = "ok", log_path: Path | None = None) -> Path:
        app = _fake_app(tmp_path, mode=mode, log_path=log_path)
        monkeypatch.setattr(keyagent, "verify_bundle", lambda path: "fake key agent (test)")
        monkeypatch.setattr(keyagent, "helper_bundle_path", lambda: app)
        return app

    return build


def _client(app: Path, mode: str | None = None, **kwargs: Any) -> keyagent.KeyagentClient:
    """A client wired to the fake bundle, with the pre-flight stubbed."""
    if mode is not None:
        set_mode(app, mode)
    return keyagent.KeyagentClient(
        tag=TEST_TAG, bundle=app, preflight=lambda path: "fake (test)", **kwargs
    )


def _read_log(log_path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log_path.read_text().splitlines() if line]


# ---------------------------------------------------------------------------
# The happy path: the verbs, the values, and the bytes
# ---------------------------------------------------------------------------


def test_create_reports_the_point_the_helper_returned(fake_app: Any) -> None:
    app = fake_app()
    created = _client(app).create()
    assert created.point == _A_REAL_P256_POINT
    assert created.reused is False
    # The key id is Python's own definition (`key_id_for`), never the helper's:
    # one definition of the id, and the anchor's is the same one.
    from local_operator.operator.verify import key_id_for

    assert key_id_for(created.point) != ""


def test_reuse_is_reported_rather_than_a_second_key(fake_app: Any) -> None:
    """``reused`` is what makes a second ``lop operator init`` a report, not a key —
    and the helper has to CONSULT THE TAG before it makes one.

    TWO REGISTERS, because one of them could not see the defect (QA round 1, Q1). A
    second key would invalidate every device certificate signed under the first anchor,
    so the flag is a security-relevant part of the protocol and not a convenience; this
    test used to assert ONLY that the client parses it, from a hand-written fake reply.
    The round-1 measurement is that the helper never reached its own duplicate branch at
    all: ``SecKeyCreateRandomKey`` SUCCEEDS against a tag that already holds an item, so
    three consecutive REAL ``create`` calls each returned a fresh point with
    ``reused:false`` while ``public`` kept resolving the tag to the first key — an anchor
    staged from ``create()`` then pinned a public half the agent would never sign with.
    A fake reply cannot see that, so the helper's half is asserted against the helper.
    """
    assert _client(fake_app("reused"), mode="reused").create().reused is True

    assert _HELPER_C.is_file(), f"no helper source at {_HELPER_C}"
    create = (
        _helper_body()
        .split("static int cmd_create", 1)[1]
        .split("static int cmd_public_or_exists", 1)[0]
    )
    probe = create.index("find_key(tag")
    generate = create.index("generate_key(tag")
    assert probe < generate, (
        "cmd_create generates a key before it looks for one: a tag that already holds an "
        "item then gets a SECOND key, and the point it reports is one `public` will not "
        "resolve to (QA round 1, Q1)"
    )
    # Generation lives in ONE place, so the ordering above is the whole story: a second
    # `SecKeyCreateRandomKey` in this function would be a path the pin cannot see.
    assert "SecKeyCreateRandomKey" not in create
    # ...and the reuse report is what the caller reads, so the reply builder must still
    # write it — and `cmd_create` must reach it on the path where the tag was occupied.
    assert (
        "emit_key_reply(point, 1, NULL)" in create
    ), "the create path that found an existing key does not report it as reused"
    reply = (
        _helper_body().split("static int emit_key_reply", 1)[1].split("static int cmd_create", 1)[0]
    )
    assert "reused" in reply


def test_doctor_establishes_usability_by_an_operation_the_os_gates() -> None:
    """Q4/R1-2: a bundle whose entitlement the OS refuses must not read healthy.

    The query ``doctor`` used to anchor on is answered ``errSecItemNotFound`` to an
    UNENTITLED process as well as to an entitled one — that is the whole -25300 trap — so
    it cannot tell a working install from a dead one, and ``status`` reported a healthy
    presence tier on hosts whose ``create`` failed ``-34018``. Generation is the
    operation the entitlement actually gates, so the helper must perform it AND delete
    what it made.
    """
    assert _HELPER_C.is_file(), f"no helper source at {_HELPER_C}"
    doctor = _helper_body().split("static int cmd_doctor", 1)[1].split("typedef struct", 1)[0]
    assert "generate_key(" in doctor, "doctor inspects instead of exercising the OS gate"
    assert "SecItemDelete" in doctor, "doctor creates an item and does not delete it again"
    assert (
        "getpid" in doctor
    ), "doctor's probe tag is not unique to the process, so two doctors can collide"


def test_sign_receives_exactly_the_bytes_it_was_given(fake_app: Any, tmp_path: Path) -> None:
    """No framing, no prefix, no hex: the helper signs what it is handed on stdin.

    Any framing here would be a second definition of the wire format, and the
    first one is ``verify.signed_message`` — so the assertion is on the BYTES the
    other process saw, not on the reply.
    """
    app = fake_app()
    log = _fake_log(app)
    signature = _client(app).sign(_MESSAGE)
    assert signature == _A_DER_SIGNATURE
    calls = _read_log(log)
    assert len(calls) == 1
    assert calls[0]["argv"] == ["sign", "--tag", TEST_TAG]
    assert bytes.fromhex(calls[0]["payload"]) == _MESSAGE


def test_the_public_verb_is_the_helper_s_answer_not_a_local_read(fake_app: Any) -> None:
    assert _client(fake_app()).public() == _A_REAL_P256_POINT


def test_exists_answers_false_without_raising(fake_app: Any) -> None:
    """``exists`` asks a question, so "no item" is its success reply.

    The error path is ``public``'s, where ``errSecItemNotFound`` is exit 3 and the
    caller has to distinguish "no key yet" from "broken install".
    """
    assert _client(fake_app("no-key"), mode="no-key").exists() is False


def test_doctor_reports_every_measurement(fake_app: Any) -> None:
    report = _client(fake_app()).doctor()
    assert report.rung == "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly"
    assert report.keychain == "ok"
    assert report.keychain_status == 0
    assert report.profile == "ok"
    # THE GATED MEASUREMENT: what the OS will actually let this bundle do. It travels as
    # a word AND a number, like the keychain field, so a caller branches on the number.
    assert report.generation == "ok"
    assert report.generation_status == 0


# ---------------------------------------------------------------------------
# The failure states the design's §6 copy is written for
# ---------------------------------------------------------------------------


def test_a_missing_key_agent_is_absent_and_not_a_host_limitation(fake_app: Any) -> None:
    app = fake_app()
    (app / "Contents" / "MacOS" / keyagent.EXECUTABLE_NAME).unlink()
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app).create()
    assert raised.value.kind == "absent"
    assert "verify_bundle" not in str(raised.value)  # a fact, not a frame name


def test_an_unbuilt_bundle_is_unverified_not_absent(tmp_path: Path) -> None:
    """The real pre-flight, on a directory that is not a bundle at all.

    This is the one test that runs the shipping ``verify_bundle``: it must refuse
    BEFORE exec, because a bundle whose embedded profile is missing or stale is
    SIGKILLed by the kernel, which is not a failure anything can report.
    """
    empty = tmp_path / "not-a-bundle.app"
    empty.mkdir()
    with pytest.raises(keyagent.KeyagentError) as raised:
        keyagent.verify_bundle(empty)
    assert raised.value.kind in ("absent", "unverified")


def test_the_kernel_kill_is_its_own_state_with_its_own_remedy(fake_app: Any) -> None:
    """A signal is not an error code, and the copy has to say which shape it was.

    Measured: an entitled bundle whose profile does not authorize its application
    identifier is SIGKILLed before ``main`` runs, so the only observable is the
    signal — which is why the failure gets a state of its own.
    """
    app = fake_app("killed")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="killed").create()
    assert raised.value.kind == "killed"
    assert raised.value.exit_code is not None and raised.value.exit_code < 0


def test_no_key_under_the_tag_is_exit_three(fake_app: Any) -> None:
    app = fake_app("no-key")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="no-key").public()
    assert raised.value.kind == "no-key"
    assert raised.value.exit_code == 3


def test_a_dismissed_sheet_is_cancelled_and_not_a_defect(fake_app: Any) -> None:
    app = fake_app("cancelled")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="cancelled").sign(_MESSAGE)
    assert raised.value.kind == "cancelled"
    assert raised.value.exit_code == 2


def test_a_refusal_carries_the_site_the_status_and_every_rung(fake_app: Any) -> None:
    """The ladder's per-rung detail survives the process boundary.

    ``site`` is reported by the helper because the diagnosis is keyed on the call
    site as well as the code: ``errSecParam`` means different things at
    access-control and at key-generation time, and the Python side must not guess.
    """
    app = fake_app("refused")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="refused").create()
    exc = raised.value
    assert exc.kind == "refused"
    assert exc.site == "key generation"
    assert exc.status == keychain._ERR_SEC_MISSING_ENTITLEMENT
    assert len(exc.refusals) == 2
    assert {entry["protection"] for entry in exc.refusals} == set(
        keychain.SecureEnclaveBackend.PROTECTION_LADDER
    )


def test_the_protocol_version_is_checked(fake_app: Any) -> None:
    app = fake_app("protocol-mismatch")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="protocol-mismatch").create()
    assert raised.value.kind == "protocol"


def test_the_helper_declining_a_request_is_not_a_protocol_mismatch(fake_app: Any) -> None:
    """Exit 5 is TWO things, and the reply says which (QA round 1, Q2).

    A deliberate refusal (`purge` on the operator's own tag), an unknown verb and a bad
    flag all leave by EXIT_USAGE, exactly like a genuine mismatch with another build. The
    helper echoes PROTOCOL in every reply it writes, so a parseable reply carrying THIS
    protocol is the helper declining the request — and reporting that as "the key agent
    does not match this runtime … reinstall" sent the operator to reinstall an install
    that was working exactly as designed.
    """
    app = fake_app("unknown-verb")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app).create()
    assert raised.value.kind == "refused"
    assert raised.value.exit_code == 5
    assert raised.value.site == keyagent.USAGE_REFUSED
    assert "reinstall" not in keychain.keyagent_refusal_message(raised.value)


def test_exit_five_from_another_protocol_is_still_a_broken_install(fake_app: Any) -> None:
    """...and the case the original mapping was written for is unchanged.

    A reply that echoes SOMEBODY ELSE'S protocol number is not a reply this client can
    read, so it is the protocol kind, whose copy is the reinstall one.
    """
    app = fake_app("usage-other-protocol")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app).create()
    assert raised.value.kind == "protocol"
    assert raised.value.exit_code == 5


def test_a_reply_that_is_not_json_is_a_protocol_error(fake_app: Any) -> None:
    """A helper that answers prose is not a helper this runtime can read.

    The failure has to be a REFUSAL rather than an exception from ``json.loads``
    escaping into a caller that is holding a key: the surfaces that sign catch
    ``KeyBackendError``, and a raw ``ValueError`` from a parse would bypass every
    one of them.
    """
    app = fake_app("garbage")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app).exists()
    assert raised.value.kind == "protocol"
    assert "not JSON" in raised.value.detail


def test_output_that_is_not_json_is_refused(fake_app: Any, tmp_path: Path) -> None:
    app = fake_app("garbage")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="garbage").create()
    assert raised.value.kind == "protocol"


def test_a_public_value_that_is_not_a_p256_point_is_refused(fake_app: Any) -> None:
    app = fake_app("badpoint")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="badpoint").public()
    assert raised.value.kind == "protocol"
    assert "P-256" in raised.value.detail


def test_a_reply_without_a_public_value_is_refused(fake_app: Any) -> None:
    app = fake_app("nospki")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="nospki").public()
    assert raised.value.kind == "protocol"


def test_the_sign_timeout_kills_the_helper_and_reports_nothing_signed(fake_app: Any) -> None:
    """``sign`` waits for a human, so the bound is the caller's — and it must bite.

    On expiry the helper is terminated (by pid, as the fleet rule requires) and the
    outcome is reported as an unanswered prompt rather than as a signature.
    """
    app = fake_app("slow")
    with pytest.raises(keyagent.KeyagentError) as raised:
        _client(app, mode="slow").sign(_MESSAGE, timeout=0.4)
    assert raised.value.kind == "timeout"


# ---------------------------------------------------------------------------
# The pre-flight: before every exec, and never cached
# ---------------------------------------------------------------------------


def test_the_pre_flight_runs_before_every_call_and_is_never_cached(tmp_path: Path) -> None:
    """A bundle can be replaced between two calls in one process.

    It is also the ONLY check that can report a bad embedded profile: discovered
    as a signal, it is unreportable. So the count is the assertion.
    """
    app = _fake_app(tmp_path)
    seen: list[str] = []
    client = keyagent.KeyagentClient(
        tag=TEST_TAG,
        bundle=app,
        preflight=lambda path: seen.append(str(path)) or "ok (test)",
        runner=lambda argv, **kw: subprocess.CompletedProcess(
            argv, 0, json.dumps({"ok": True, "protocol": 1, "present": True}).encode(), b""
        ),
    )
    client.exists()
    client.exists()
    client.doctor()
    assert len(seen) == 3, seen


def test_a_preflight_that_refuses_stops_the_call_before_exec(tmp_path: Path) -> None:
    """The order is the guarantee: refuse first, exec never."""
    app = _fake_app(tmp_path, log_path=tmp_path / "helper.log")
    ran: list[str] = []

    def refuse(path: Path) -> str:
        raise keyagent.KeyagentError("unverified", "the signature is not ours")

    def runner(argv: Any, **kw: Any) -> Any:  # pragma: no cover — must not be reached
        ran.append("exec")

    client = keyagent.KeyagentClient(tag=TEST_TAG, bundle=app, preflight=refuse, runner=runner)
    with pytest.raises(keyagent.KeyagentError) as raised:
        client.create()
    assert raised.value.kind == "unverified"
    assert ran == []


def test_helper_health_answers_reachability_and_names_the_reason(
    fake_app: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``supported()``'s question, and the one it is not.

    A healthy installation answers with what the pre-flight found; a broken one
    answers with a KIND, so a reporter can choose its own register instead of
    parsing prose. The probe tag holds no key: whether an operator key exists must not
    change the answer to "can this install reach an entitled process" — but ``doctor``
    DOES create under it, once, under ``<tag>.<pid>``, and deletes what it made, because
    a read is answered identically by a working install and a dead one (QA round 1, Q4).
    """
    app = fake_app()
    health = keyagent.helper_health(bundle=app)
    assert health.ok is True and health.kind == ""

    missing = keyagent.helper_health(bundle=tmp_path / "gone.app")
    assert missing.ok is False
    assert missing.kind in ("absent", "unverified")
    assert keyagent.HEALTH_TAG != keychain.APPLICATION_TAG

    # THE STATE THE QUERY CANNOT SEE: the profile is present, the keychain answers "no
    # item", and the GATED operation is refused. This is the install whose `status`
    # reported a healthy presence tier while `create` failed -34018.
    set_mode(app, "doctor-refused")
    refused = keyagent.helper_health(bundle=app)
    assert refused.ok is False, "a refused entitlement read as healthy"
    assert refused.kind == "refused", refused.kind
    assert "entitlement" in refused.detail, refused.detail


# ---------------------------------------------------------------------------
# The rules that must hold across the module, asserted on the SOURCE
# ---------------------------------------------------------------------------


def test_no_runtime_path_queries_the_keychain_for_the_operator_key() -> None:
    """THE -25300 TRAP, as a source-level invariant.

    An unsigned process asking for this item gets ``errSecItemNotFound`` for a key
    that EXISTS (measured), so any ``SecItemCopyMatching`` in this package is a
    path that could report "no key" on a host that has one. There is no such call
    left: every verb, including the absence claim, is the helper's answer.
    """
    import ast

    # The whole PACKAGE, not one file (agent review round 1, R1-7): the invariant is
    # "no query made by the unsigned runtime", and it spans every module — a future
    # ``SecItemCopyMatching`` in ``operator/macos/keyagent.py`` (a plausible
    # "optimisation" of ``exists()``) would have passed a check that only read
    # ``keychain.__file__``.
    parsed = {
        path: ast.parse(path.read_text())
        for path in sorted(Path(keychain.__file__).parent.rglob("*.py"))
    }
    assert len(parsed) > 1, f"the walk found only {list(parsed)}"
    referenced: set[str] = set()
    for tree in parsed.values():
        referenced |= {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id.startswith("SecItem")
        } | {
            node.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute) and node.attr.startswith("SecItem")
        }
    # Read from the AST rather than the text: these modules' docstrings NAME these
    # calls, deliberately, to explain why they are gone — a substring check would
    # forbid the explanation along with the defect.
    assert referenced == set(), f"{referenced} is back in the runtime; see the -25300 trap"
    # And the constants that named the trap are still here, because the copy and
    # the diagnosis table quote them.
    assert keychain._ERR_SEC_ITEM_NOT_FOUND == -25300
    assert keychain._ERR_SEC_USER_CANCELED == -128


def test_the_absence_claim_only_ever_comes_from_the_helper(fake_app: Any) -> None:
    """``load()`` returning ``None`` means the ENTITLED process said so."""
    app = fake_app("no-key")
    backend = keychain.SecureEnclaveBackend()
    backend._client = lambda: _client(app)  # type: ignore[method-assign]

    # The helper's own -25300 is the ONE absence claim the runtime may believe.
    assert backend.load() is None
    assert _client(app).exists() is False

    # ... and a helper that cannot be RUN is not that claim: it raises, because a
    # broken install and an empty host have different remedies.
    set_mode(app, "killed")
    with pytest.raises(keychain.KeyBackendError):
        backend.load()


def test_no_fallback_path_continues_with_a_file_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken macOS install is an ERROR, never a quieter key.

    ``choose_backend("auto")`` on Darwin returns the presence backend whether or
    not its key agent works, and the failure then surfaces as the copy that names
    the remedy. The assertion with teeth is the ABSENCE of a software key file:
    a fallback would write one behind the operator's back.
    """
    if os.uname().sysname != "Darwin":  # pragma: no cover — CI runs Linux
        pytest.skip("the ladder's Darwin arm only runs on a Mac")
    monkeypatch.setattr(keyagent, "helper_bundle_path", lambda: tmp_path / "gone.app")
    backend = keychain.choose_backend("auto", config_root=tmp_path)
    assert isinstance(backend, keychain.SecureEnclaveBackend)
    with pytest.raises(keychain.KeyBackendError):
        backend.create()
    assert not (tmp_path / "operator" / "operator-key.pem").exists()
    assert "lop-keyagent.app" in str(
        pytest.raises(keychain.KeyBackendError, backend.load).value if False else _message(backend)
    )


def _message(backend: Any) -> str:
    try:
        backend.load()
    except keychain.KeyBackendError as exc:
        return str(exc)
    return ""


def test_the_named_file_backend_is_still_reachable_by_name(tmp_path: Path) -> None:
    """``--backend file-only`` remains the ONLY route to a software key."""
    backend = keychain.choose_backend("file-only", config_root=tmp_path)
    assert isinstance(backend, keychain.FileKeyBackend)


# ---------------------------------------------------------------------------
# The copy: one vocabulary, in the module that already had one
# ---------------------------------------------------------------------------


def test_every_state_has_copy_in_two_registers() -> None:
    """``init`` gets the remedy; ``status`` gets the state.

    Two commands describing one broken install differently is how a reader learns
    to trust neither, so both come from one table — and every state a
    ``KeyagentError`` can raise must be in it.

    THE SET IS DERIVED FROM THE CLASS, NOT FROM A HAND-WRITTEN LIST (agent review
    round 1, R1-4). The hand-written list here omitted ``refused`` while a pragma
    claimed "every kind is in the table, pinned by a test", and ``refused`` is
    reachable: ``helper_health`` hands ANY ``KeyagentError``'s kind to this table, and
    ``doctor`` exits 4 when the keychain query is refused — errSecMissingEntitlement,
    the entitlement-not-in-effect state (and now also what a refused generation
    reports). Reading the documented kinds from ``KeyagentError`` is what makes the
    claim true rather than restated.
    """
    documented = re.findall(r"^\s*\*\s*``([a-z-]+)``", keyagent.KeyagentError.__doc__ or "", re.M)
    assert len(documented) >= 8, f"the kinds are no longer a readable list: {documented}"
    assert "refused" in documented
    kinds = set(documented)
    assert kinds <= set(
        keychain._KEYAGENT_STATES
    ), f"{sorted(kinds - set(keychain._KEYAGENT_STATES))} can be raised and has no copy"
    for state in kinds:
        long_copy = keychain.keyagent_state_copy(state)
        short_copy = keychain.keyagent_state_copy(state, long=False)
        assert long_copy and short_copy
        assert short_copy != long_copy or state == "no-key"


def test_the_reinstall_copy_names_the_remedy_and_the_alternative() -> None:
    """§6's requirement: the remedy AND what the alternative costs."""
    for state in ("absent", "unverified", "killed"):
        copy = keychain.keyagent_state_copy(state)
        assert "reinstall" in copy
        assert "lop operator init --backend file-only" in copy
        assert "operator-file-only" in copy


def test_a_refusal_message_uses_the_one_diagnosis_vocabulary() -> None:
    """A ``refused`` error goes through the SAME builder the in-process ladder used.

    Which is what keeps ``errSecParam`` meaning "your flag pair" at access-control
    and "your parameters" at key generation.
    """
    exc = keyagent.KeyagentError(
        "refused",
        "failed to add key: <SecKeyRef:('com.apple.setoken')> 0x10375c430",
        status=keychain._ERR_SEC_MISSING_ENTITLEMENT,
        site=keychain.KEY_GENERATION_REFUSED,
        refusals=(
            {
                "site": keychain.KEY_GENERATION_REFUSED,
                "protection": keychain.SecureEnclaveBackend.PROTECTION_LADDER[0],
                "status": keychain._ERR_SEC_MISSING_ENTITLEMENT,
                "detail": "failed to add key: <SecKeyRef:('com.apple.setoken')> 0x10375c430",
            },
        ),
    )
    message = keychain.keyagent_refusal_message(exc)
    assert "entitlement" in message or "key agent" in message
    assert keychain.SecureEnclaveBackend.PROTECTION_LADDER[0] in message
    assert "0x10375c430" not in message, "a per-run object address reached the copy"


def test_the_failure_copy_never_contains_a_run_address() -> None:
    exc = keyagent.KeyagentError(
        "refused",
        "failed <SecKeyRef:('com.apple.setoken')> 0x10137ede0",
        status=keychain._ERR_SEC_PARAM,
        site=keychain.ACCESS_CONTROL_REFUSED,
        refusals=(
            {
                "site": keychain.ACCESS_CONTROL_REFUSED,
                "protection": "",
                "status": keychain._ERR_SEC_PARAM,
                "detail": "refused <SecKeyRef:('com.apple.setoken')> 0x10137ede0",
            },
        ),
    )
    assert "0x10137ede0" not in keychain.keyagent_refusal_message(exc)


def test_a_cancelled_signature_says_nothing_was_signed() -> None:
    exc = keyagent.KeyagentError("cancelled", "errSecUserCanceled", exit_code=2)
    message = keychain.keyagent_refusal_message(exc)
    assert "cancelled" in message
    # The point of this state: a decline is not a defect, and the operator is told
    # the KEY did not change — which is what makes a retry safe.
    assert "unchanged" in message


# ---------------------------------------------------------------------------
# The report: §6's status column, which is where a broken install becomes visible
# ---------------------------------------------------------------------------


def _run_status(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, backend: str = "") -> str:
    """``lop operator status`` with an isolated anchor root, as a string."""
    import argparse

    from local_operator.operator import handlers, trust

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "local_operator.operator.trust._ANCHOR_ROOT_OVERRIDE",
        tmp_path / "anchor-root",
        raising=False,
    )
    if backend:
        staged = trust.staging_path(tmp_path)
        staged.parent.mkdir(parents=True, exist_ok=True)
        staged.write_text(
            json.dumps(
                {
                    "version": 1,
                    "backend": backend,
                    "presence": True,
                    "key_id": "a" * 32,
                    "spki": "04" + "b" * 128,
                }
            )
        )
    printed: list[str] = []
    monkeypatch.setattr(
        "builtins.print", lambda *parts, **kw: printed.append(" ".join(map(str, parts)))
    )
    handlers.dispatch(argparse.Namespace(operator_command="status"))
    return "\n".join(printed)


def test_status_names_the_key_agent_beside_the_authority_reason(
    fake_app: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No key recorded and no key agent: TWO independent facts, on two lines (D1/D3).

    The level is ``spawn-capability-only`` because NO ANCHOR IS INSTALLED; the key agent's
    state does not move the level at all. Reporting the second IN PLACE OF the first — the
    released shape — read as "my key vanished and my installation is broken" one command
    after a successful ``init``, and its loudest word, "broken install", pointed at a
    reinstall that neither installs the anchor nor changes the level. And ``status``'s
    only named next action used to be ``lop operator init``, which in this exact state
    exits 1 — a closed loop with no reinstall remedy at all.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(keyagent, "helper_bundle_path", lambda: tmp_path / "gone.app")
    report = _run_status(monkeypatch, tmp_path)
    assert "private-half backend   : (none)" in report
    assert (
        "reason                 : no anchor is installed, so the runtime trusts no key yet"
        in report
    )
    assert "key agent              : the macOS key agent is not installed" in report
    assert "broken install" in report
    assert "fix                    : reinstall the macOS wheel" in report
    assert "(or take a file-backed key now" in report, "nothing is staged, so file-only is offered"
    # The closed loop: the one command `status` used to name cannot work in this state.
    assert "`lop operator init` adds the operator key" not in report
    assert "loosening: no operator authority on this host." in report


def test_status_does_not_contradict_the_file_only_init_that_just_succeeded(
    fake_app: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D1's exact reproduction: `init --backend file-only`, then `status`.

    A bare ``(none)`` beside a ``broken install`` reason is what made this state read as
    a lost key, so the backend line names the staged key and the one pending step, and the
    remedy does not offer the file-only route the reader has just taken.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(keyagent, "helper_bundle_path", lambda: tmp_path / "gone.app")
    report = _run_status(monkeypatch, tmp_path, backend="file-only")
    assert "private-half backend   : (none — an operator key is staged" in report
    assert "reason                 : no anchor is installed" in report
    assert "key agent              : the macOS key agent is not installed" in report
    assert "fix                    : reinstall the macOS wheel — `uv tool install" in report
    assert "(or take a file-backed key now" not in report
    assert "staged anchor" in report


def test_status_names_a_broken_agent_beside_an_anchor_that_claims_it(
    fake_app: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The anchor says ``secure-enclave`` and the install cannot reach it.

    Without this line the two fields above would promise a presence gate on a host where
    no signature can be produced at all. A WORKING install prints no such line, which is
    what keeps a report of a fault from becoming a new field on every clean run.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(keyagent, "helper_bundle_path", lambda: tmp_path / "gone.app")

    # The anchor's own provenance — root-owned, installed, ours — is tested in
    # test_operator_authority.py; what is asserted HERE is the reporting logic given a
    # report that names the presence backend.
    def _anchored(**kwargs: Any) -> dict[str, Any]:
        return {
            "level": "operator-presence",
            "reason": "an installed root-owned anchor",
            "anchor_path": "/var/lib/local-operator/operator/trust.json",
            "anchor_installed": True,
            "anchor_root_owned": True,
            "backend": keychain.SECURE_ENCLAVE,
            "presence": True,
            "key_id": "a" * 32,
            "capability_guarantee": "unknown",
            "presence_enforced_by_os": True,
        }

    from local_operator.operator import handlers

    monkeypatch.setattr(handlers, "operator_authority_report", _anchored)
    broken = _run_status(monkeypatch, tmp_path, backend=keychain.SECURE_ENCLAVE)
    assert "private-half backend   : secure-enclave" in broken
    assert "key agent              : the macOS key agent is not installed" in broken

    healthy = fake_app()
    monkeypatch.setattr(keyagent, "helper_bundle_path", lambda: healthy)
    report = _run_status(monkeypatch, tmp_path, backend=keychain.SECURE_ENCLAVE)
    assert "key agent              :" not in report


# ---------------------------------------------------------------------------
# The REAL pre-flight, which the fake-helper tests deliberately stub
# ---------------------------------------------------------------------------


def _bundle_shaped(tmp_path: Path, *, executable: bytes | None = b"#!/bin/sh\n") -> Path:
    """A bundle directory shaped like the real one, with no signature at all."""
    app = tmp_path / "lop-keyagent.app"
    macos = app / "Contents" / "MacOS"
    macos.mkdir(parents=True, exist_ok=True)
    if executable is not None:
        exe = macos / keyagent.EXECUTABLE_NAME
        exe.write_bytes(executable)
        exe.chmod(0o755)
    return app


def test_the_real_preflight_refuses_a_bundle_that_is_not_there(tmp_path: Path) -> None:
    """The first check needs no OS: an installation without the bundle is broken."""
    with pytest.raises(keyagent.KeyagentError) as raised:
        keyagent.verify_bundle(tmp_path / "gone.app")
    assert raised.value.kind == "absent"


def test_the_real_preflight_refuses_a_bundle_with_no_runnable_helper(tmp_path: Path) -> None:
    """A bundle directory without the executable is the sdist-install shape."""
    with pytest.raises(keyagent.KeyagentError) as raised:
        keyagent.verify_bundle(_bundle_shaped(tmp_path, executable=None))
    assert raised.value.kind == "absent"

    app = _bundle_shaped(tmp_path / "second")
    (app / "Contents" / "MacOS" / keyagent.EXECUTABLE_NAME).chmod(0o644)
    with pytest.raises(keyagent.KeyagentError) as raised:
        keyagent.verify_bundle(app)
    assert raised.value.kind == "absent"
    assert "not executable" in raised.value.detail


def test_the_profile_must_authorize_this_helper_s_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE AUTHORIZATION PROPERTY, read from the profile's own bytes.

    This is the check whose absence is measured to be a kernel SIGKILL rather than an
    error, so it is the one the pre-flight exists for — and it needs no keychain, no
    codesign and no external tool, which is what keeps it true in an environment that
    has none. The decode is stubbed to that environment's outcome here; what the decode
    does when it RUNS is the next test's subject.
    """
    claimed = f"{keyagent.TEAM_IDENTIFIER}.{keyagent.BUNDLE_IDENTIFIER}"
    app = tmp_path / "lop-keyagent.app"
    monkeypatch.setattr(
        keyagent,
        "_decode_profile",
        lambda _profile: ("no-keychain", "A default keychain could not be found", b""),
    )
    good = tmp_path / "good.provisionprofile"
    good.write_bytes(b"\x30\x82lead-in DER" + claimed.encode() + b"trailing DER")
    ok, detail = keyagent._profile_authorizes(good, app)
    assert ok is True and claimed in detail
    assert "not decoded in this environment" in detail

    other = tmp_path / "other.provisionprofile"
    other.write_bytes(b"\x30\x82lead-in DEROTHER-TEAM.com.someone.else.apptrailing")
    ok, detail = keyagent._profile_authorizes(other, app)
    assert ok is False and "does not authorize" in detail

    ok, detail = keyagent._profile_authorizes(tmp_path / "gone.provisionprofile", app)
    assert ok is False and "unreadable" in detail


def _decoded_profile(**overrides: Any) -> bytes:
    """A FABRICATED profile plist — the shape ``security cms -D`` produces."""
    import datetime
    import plistlib

    body: dict[str, Any] = {
        "Entitlements": {
            "com.apple.application-identifier": (
                f"{keyagent.TEAM_IDENTIFIER}.{keyagent.BUNDLE_IDENTIFIER}"
            )
        },
        "ExpirationDate": datetime.datetime.now(datetime.timezone.utc)
        + datetime.timedelta(days=30),
        "DeveloperCertificates": [b"the signing leaf"],
        "TeamIdentifier": [keyagent.TEAM_IDENTIFIER],
    }
    body.update(overrides)
    return plistlib.dumps(body)


def test_a_profile_that_cannot_be_decoded_is_not_a_pass(tmp_path: Path, monkeypatch: Any) -> None:
    """R1-6/Q3: there are THREE outcomes, and the message has to say which.

    Measured (QA round 1, Q3): on a host where ``security cms -D`` demonstrably works, a
    MANGLED profile was reported as ``ok=True`` with "profile uncached in this
    environment: no keychain to decode it with" — a claim about the invoker that was
    false there, and a bundle whose profile is unreadable garbage passing the pre-flight
    that exists to catch exactly that. The distinguishing fact is the tool's own text:
    only "A default keychain could not be found" is the no-keychain case.
    """
    import subprocess as sp

    claimed = f"{keyagent.TEAM_IDENTIFIER}.{keyagent.BUNDLE_IDENTIFIER}"
    profile = tmp_path / "p.provisionprofile"
    profile.write_bytes(claimed.encode())
    app = tmp_path / "lop-keyagent.app"

    def outcome(returncode: int, stdout: bytes, stderr: bytes) -> Any:
        def run(argv: Any, *, timeout: float = 30.0) -> Any:
            return sp.CompletedProcess(argv, returncode, stdout, stderr)

        return run

    monkeypatch.setattr(
        keyagent, "_run", outcome(1, b"", b"security: the data is not a CMS message")
    )
    ok, detail = keyagent._profile_authorizes(profile, app)
    assert ok is False, "a profile the decoder REFUSED must not pass the pre-flight"
    assert "could not be decoded" in detail
    assert "no keychain" not in detail

    monkeypatch.setattr(
        keyagent,
        "_run",
        outcome(1, b"", b"security: cert import failed: A default keychain could not be found"),
    )
    ok, detail = keyagent._profile_authorizes(profile, app)
    assert ok is True, "the no-keychain case is neither accepted nor refused"
    assert "not decoded in this environment" in detail


def test_the_decoded_profile_has_to_be_USABLE_not_merely_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-2: lapsed and unpaired profiles, exercised rather than asserted.

    The bytes test above accepts a LAPSED profile — it still names the right application
    identifier — and a rotated-out-of-step one, and neither was checked anywhere before
    (``grep -rn Expiration`` found the topic only in the design document's prose). Each
    case below is a FABRICATED plist fed through the real function, and the first is the
    control that shows the checks are not simply refusing everything.
    """
    import datetime

    claimed = f"{keyagent.TEAM_IDENTIFIER}.{keyagent.BUNDLE_IDENTIFIER}"
    app = tmp_path / "lop-keyagent.app"
    now = datetime.datetime.now(datetime.timezone.utc)
    monkeypatch.setattr(keyagent, "_signing_leaf", lambda _app: b"the signing leaf")

    ok, detail = keyagent._profile_is_usable(_decoded_profile(), claimed, app)
    assert ok is True and "current to" in detail, detail

    lapsed = _decoded_profile(ExpirationDate=now - datetime.timedelta(days=3))
    ok, detail = keyagent._profile_is_usable(lapsed, claimed, app)
    assert ok is False and "expired" in detail, detail

    # THE PAIRING: the profile decodes, is current, and names a certificate the
    # signature was NOT made with. Nothing else in the pre-flight looks at this.
    monkeypatch.setattr(keyagent, "_signing_leaf", lambda _app: b"a DIFFERENT certificate")
    ok, detail = keyagent._profile_is_usable(_decoded_profile(), claimed, app)
    assert ok is False and "does not name the certificate" in detail, detail

    # ...and when the environment cannot produce a leaf at all, the check says the
    # pairing was not established instead of claiming it passed.
    monkeypatch.setattr(keyagent, "_signing_leaf", lambda _app: None)
    ok, detail = keyagent._profile_is_usable(_decoded_profile(), claimed, app)
    assert ok is True and "pairing was not established" in detail, detail

    # The earlier cases return before the leaf is read, so these hold with it stubbed.
    wrong_app = _decoded_profile(
        Entitlements={"com.apple.application-identifier": "OTHERTEAM.com.other.app"}
    )
    ok, detail = keyagent._profile_is_usable(wrong_app, claimed, app)
    assert ok is False and "grants" in detail, detail

    no_certs = _decoded_profile(DeveloperCertificates=[])
    ok, detail = keyagent._profile_is_usable(no_certs, claimed, app)
    assert ok is False and "DeveloperCertificates" in detail, detail


@pytest.mark.skipif(sys.platform != "darwin", reason="codesign is macOS-only")
def test_the_real_preflight_asks_codesign_and_reports_its_refusal(tmp_path: Path) -> None:
    """An unsigned bundle is refused by the real tool, with the tool's own words.

    The bundle here IS properly shaped (executable, profile present) and still refused:
    that is the measured difference between a helper this runtime will exec and a
    directory that merely looks like one.
    """
    app = _bundle_shaped(tmp_path)
    profile = app / "Contents" / "embedded.provisionprofile"
    profile.write_bytes(
        b"\x30\x82" + f"{keyagent.TEAM_IDENTIFIER}.{keyagent.BUNDLE_IDENTIFIER}".encode()
    )
    with pytest.raises(keyagent.KeyagentError) as raised:
        keyagent.verify_bundle(app)
    assert raised.value.kind == "unverified"
    assert "codesign" in raised.value.detail


@pytest.mark.skipif(
    not os.environ.get("LOP_KEYAGENT_BINARY"),
    reason="set LOP_KEYAGENT_BINARY to a SIGNED lop-keyagent executable to run this",
)
def test_the_real_helper_is_idempotent_by_tag() -> None:
    """Q1, measured against the real entitled process: three creates, ONE key.

    Gated on ``LOP_KEYAGENT_BINARY`` because only a Developer-ID-signed bundle carrying an
    embedded provisioning profile can reach the data-protection keychain, so this cannot
    run in a checkout — and it is exactly the observation a fake reply cannot make. Before
    the round-1 fix, three consecutive ``create`` calls on one tag returned three
    DIFFERENT points with ``reused:false`` while ``public`` kept resolving the tag to the
    first key, so an anchor staged from ``create()``'s handle pinned a public half the
    agent would never sign with.

    The tag is unique to this process and every item it makes is deleted before this
    returns, so it can never touch the operator's own key (``keychain.APPLICATION_TAG``).
    """
    binary = os.environ["LOP_KEYAGENT_BINARY"]
    tag = f"com.local-operator.keyagent.test.{os.getpid()}"

    def run(verb: str) -> dict[str, Any]:
        done = subprocess.run(
            [binary, verb, "--tag", tag], capture_output=True, text=True, timeout=120
        )
        assert done.stdout.strip(), done.stderr
        return json.loads(done.stdout)

    try:
        first = run("create")
        assert first["reused"] is False
        second = run("create")
        assert second["reused"] is True, "a second create made a second key"
        assert second["spki"] == first["spki"], "the tag's key moved between two creates"
        assert run("public")["spki"] == first["spki"], "create and public disagree"
        assert run("exists")["present"] is True
    finally:
        # Deletion is gated by the SAME entitlement as creation, so the cleanup goes
        # through the entitled process too; "nothing was there" is not a failure.
        deleted = run("purge")["deleted"]
        assert deleted in (0, keychain._ERR_SEC_ITEM_NOT_FOUND), f"SecItemDelete={deleted}"
    assert run("exists")["present"] is False, "the test's own key was left behind"


def test_a_reused_create_reports_the_existing_key_and_not_a_new_one(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    """R1-5: ``create``'s ``reused`` DRIVES the report, so the race is not announced wrongly.

    ``_init``'s probe and its create are two calls: a key can appear between them (the
    race ``handlers._init`` documents), and the helper now answers that case with the
    tag's key and ``reused:true`` rather than with a second key. Reporting "created"
    there would be the one thing this verb's idempotence exists to prevent — a report
    that does not match the machine — so the same block a second ``init`` prints is
    printed here. Before round 1 the field was written by the helper and read by nobody.

    Asserted through ``handlers._init`` rather than on the client, because the thing the
    flag exists for is the REPORT, not the protocol parse.
    """
    import argparse

    from local_operator.operator import handlers, trust

    key_id = "d" * 32
    handle = keychain.KeyHandle(
        backend=keychain.SECURE_ENCLAVE,
        key_id=key_id,
        spki=bytes.fromhex("04" + "c" * 128),
        presence=True,
        reused=True,
        rung="kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
    )

    class _Signer:
        def __init__(self) -> None:
            self.handle = handle

        def close(self) -> None:
            self.handle = handle

    probes: list[int] = []

    def probe(root: Path, preference: str) -> Any:
        probes.append(1)
        # NOTHING THERE when `init` looked; the key arrives while it creates.
        return None if len(probes) == 1 else _Signer()

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(trust, "_ANCHOR_ROOT_OVERRIDE", tmp_path / "anchor-root", raising=False)
    monkeypatch.setattr(handlers, "_existing_key", probe)
    monkeypatch.setattr(handlers, "create_key", lambda **kwargs: handle)

    code = handlers.dispatch(argparse.Namespace(operator_command="init", backend="auto", label=""))
    assert code == 0
    said = capsys.readouterr().out
    assert "operator key already exists" in said, said
    assert "nothing replaced" in said
    assert "created in the" not in said, "a key this run did not make was announced as created"
    assert key_id in said
    assert "To replace this key" in said, "the reused report is the full one, not a stub"
