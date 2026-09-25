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
behaviour (asserted in ``test_operator_authority.py`` against the SDK header and
the C source, and by the ``selftest`` verb the release job runs), the real
entitlement (only a signed bundle on a real keychain can show it), and the
presence prompt (it is a sheet on an operator's screen; unproven by design, see
the design document's §9).

NO TEST HERE WRITES TO ANY KEYCHAIN. The fake is a script; it has no keychain
code at all. The tag every test passes is a test tag, never
``keychain.APPLICATION_TAG``.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.operator import keychain
from local_operator.operator.macos import keyagent

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
if verb == "doctor":
    reply({{"ok": True, "protocol": 1, "rung": "kSecAttrAccessibleWhenPasscodeSetThisDeviceOnly",
           "keychain": "ok", "keychain_status": 0, "profile": "ok"}})
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
    """``reused`` is what makes a second ``lop operator init`` a report, not a key.

    A second key would invalidate every device certificate signed under the first
    anchor, so the flag is a security-relevant part of the protocol and not a
    convenience.
    """
    assert _client(fake_app("reused"), mode="reused").create().reused is True


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


def test_an_unknown_verb_is_a_protocol_error(fake_app: Any) -> None:
    """Exit 5 means "you and I do not agree", which is a broken install.

    The helper refuses a verb or a protocol it does not know rather than
    interpreting it, because the honest answer to "my peer is a different build"
    is a reinstall, not a guess.
    """
    app = fake_app("unknown-verb")
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
    parsing prose. The probe tag is a tag nothing writes to: whether an operator
    key exists must not change the answer to "can this install reach an entitled
    process".
    """
    app = fake_app()
    health = keyagent.helper_health(bundle=app)
    assert health.ok is True and health.kind == ""

    missing = keyagent.helper_health(bundle=tmp_path / "gone.app")
    assert missing.ok is False
    assert missing.kind in ("absent", "unverified")
    assert keyagent.HEALTH_TAG != keychain.APPLICATION_TAG


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

    tree = ast.parse(Path(keychain.__file__).read_text())
    referenced = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and node.id.startswith("SecItem")
    } | {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr.startswith("SecItem")
    }
    # Read from the AST rather than the text: this module's docstrings NAME these
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
    ``KeyagentError`` can raise must be in it, which is what this asserts.
    """
    kinds = {"absent", "unverified", "killed", "no-key", "cancelled", "timeout", "protocol"}
    assert kinds <= set(keychain._KEYAGENT_STATES)
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


def test_status_leads_with_the_key_agent_when_it_is_the_cause(
    fake_app: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No key recorded and no key agent: that is a BROKEN INSTALL, not an empty host.

    The reason line is the key agent's state because it is the thing the operator must
    act on — and the spec's own contrast: at ``lop operator init`` the two are different
    commands with different remedies, so the report must not describe them identically.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(keyagent, "helper_bundle_path", lambda: tmp_path / "gone.app")
    report = _run_status(monkeypatch, tmp_path)
    assert "private-half backend   : (none)" in report
    assert "reason                 : the macOS key agent is not installed" in report
    assert "broken install" in report


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


def test_the_profile_must_authorize_this_helper_s_identity(tmp_path: Path) -> None:
    """THE AUTHORIZATION PROPERTY, read from the profile's own bytes.

    This is the check whose absence is measured to be a kernel SIGKILL rather than an
    error, so it is the one the pre-flight exists for — and it needs no keychain, no
    codesign and no external tool, which is what keeps it true in an environment that
    has none.
    """
    claimed = f"{keyagent.TEAM_IDENTIFIER}.{keyagent.BUNDLE_IDENTIFIER}"
    good = tmp_path / "good.provisionprofile"
    good.write_bytes(b"\x30\x82lead-in DER" + claimed.encode() + b"trailing DER")
    ok, detail = keyagent._profile_authorizes(good)
    assert ok is True and claimed in detail

    other = tmp_path / "other.provisionprofile"
    other.write_bytes(b"\x30\x82lead-in DEROTHER-TEAM.com.someone.else.apptrailing")
    ok, detail = keyagent._profile_authorizes(other)
    assert ok is False and "does not authorize" in detail

    ok, detail = keyagent._profile_authorizes(tmp_path / "gone.provisionprofile")
    assert ok is False and "unreadable" in detail


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
