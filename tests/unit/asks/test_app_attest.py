"""F2: desktop answers verified by the app's code signature, and Part C (minted-here).

Three layers, each against the real code with only the OS injected:

* the verifier's decision ladder (``AppAttestVerifier``) over a fake
  Security.framework backend, one test per refusal, plus the static cache;
* the attestation table and the AF_UNIX listener protocol, with an injected
  audit-token reader and verifier;
* the stamp matrix of the design note (N1, N13, N15, N17, N18) driven through
  the REAL ``RuntimeServer._dispatch`` arm, the REAL serving handle and the REAL
  ``AskQueue`` log (the rig of ``test_ask_answer_surface_stamp``).

The macOS-only integration test at the end binds the real listener with the
real ctypes verifier and dials it from ``/usr/bin/nc`` and Python, which must
both be refused (design row N2).
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import pytest

from local_operator.asks import app_attest, surfaces
from local_operator.asks.app_attest import (
    CS_DEBUGGED,
    CS_GET_TASK_ALLOW,
    CS_RUNTIME,
    CS_VALID,
    FUSE_SENTINEL,
    AppAttestVerifier,
    AttestationTable,
    AttestService,
    GuestFacts,
)
from local_operator.asks.store import mirror_request_id
from local_operator.session.runtime.server import _ConnectionAuthority
from tests.unit.session.runtime.test_ask_answer_surface_stamp import (  # noqa: F401
    _Rig,
    home,
)

_EXE = "/Applications/Local Operator.app/Contents/MacOS/Local Operator"
_GOOD_FLAGS = CS_VALID | CS_RUNTIME
_PID, _PIDVERSION = 4242, 7


def _token(pid: int = _PID, pidversion: int = _PIDVERSION) -> bytes:
    vals = [0] * 8
    vals[5], vals[7] = pid, pidversion
    return b"".join(v.to_bytes(4, "little") for v in vals)


class _Backend:
    """Injected Security.framework results; counts the expensive static check."""

    def __init__(self, facts: GuestFacts | None = None, *, start: float | None = 1000.0) -> None:
        self.facts = facts or GuestFacts(
            status=0, requirement_status=0, cs_flags=_GOOD_FLAGS, main_executable=_EXE
        )
        self.start = start
        self.static_status = 0
        self.static_calls = 0

    def guest_facts(self, token: bytes) -> GuestFacts:
        return self.facts

    def static_bundle_status(self, bundle: str) -> int:
        self.static_calls += 1
        return self.static_status

    def process_start(self, pid: int) -> float | None:
        return self.start


_HARDENED_WIRE = [b"010011011"]


def _verifier(
    backend: _Backend, *, wires: list[bytes] | None = None, ctime: float = 500.0
) -> AppAttestVerifier:
    return AppAttestVerifier(
        backend,
        read_fuses=lambda path: list(_HARDENED_WIRE if wires is None else wires),
        snapshot=lambda bundle: (("Contents", ctime), ("Contents/Info.plist", ctime)),
    )


# -- the verifier ------------------------------------------------------------


def test_the_genuine_hardened_app_is_verified() -> None:
    verdict = _verifier(_Backend()).verify(_token())
    assert verdict == app_attest.Verdict(True, "ok", _PID)


@pytest.mark.parametrize(
    ("facts", "start", "reason"),
    [
        (None, None, "no-process"),
        (GuestFacts(status=-67068), 1000.0, "no-code:-67068"),
        # Homebrew python, nc, the TeamID-signed python seed, an old (epoch-less)
        # genuine release: all fail the designated requirement (spike, N2/N3/N6).
        (GuestFacts(status=0, requirement_status=-67050), 1000.0, "requirement:-67050"),
        (
            GuestFacts(status=0, requirement_status=0, cs_flags=CS_VALID, main_executable=_EXE),
            1000.0,
            "flags:0x1",
        ),
        (
            GuestFacts(
                status=0,
                requirement_status=0,
                cs_flags=_GOOD_FLAGS | CS_DEBUGGED,
                main_executable=_EXE,
            ),
            1000.0,
            "debuggable",
        ),
        (
            GuestFacts(
                status=0,
                requirement_status=0,
                cs_flags=_GOOD_FLAGS | CS_GET_TASK_ALLOW,
                main_executable=_EXE,
            ),
            1000.0,
            "debuggable",
        ),
        (
            GuestFacts(
                status=0,
                requirement_status=0,
                cs_flags=_GOOD_FLAGS,
                main_executable=_EXE,
                entitlements=frozenset({app_attest.FORBIDDEN_ENTITLEMENTS[0]}),
            ),
            1000.0,
            "entitlement:allow-dyld-environment-variables",
        ),
        (
            GuestFacts(
                status=0,
                requirement_status=0,
                cs_flags=_GOOD_FLAGS,
                main_executable="/usr/local/bin/x",
            ),
            1000.0,
            "bundle-layout",
        ),
    ],
)
def test_each_dynamic_disqualifier_refuses(
    facts: GuestFacts | None, start: float | None, reason: str
) -> None:
    backend = _Backend(facts, start=start)
    verdict = _verifier(backend).verify(_token())
    assert verdict.ok is False
    assert verdict.reason.startswith(reason), verdict
    # A dynamic refusal never pays for the static check.
    assert backend.static_calls == 0


def test_a_bundle_changed_after_launch_is_refused() -> None:
    """The ctime guard: a file swapped in after the process started (TOCTOU)."""
    verdict = _verifier(_Backend(start=1000.0), ctime=1000.5).verify(_token())
    assert verdict.reason == "modified-after-launch"


def test_a_broken_nested_seal_or_unfused_framework_is_refused() -> None:
    backend = _Backend()
    backend.static_status = -67054
    assert _verifier(backend).verify(_token()).reason == "static:-67054"
    # Today's 0.57.8 wire, read from the installed framework: RunAsNode=1.
    assert _verifier(_Backend(), wires=[b"101100011"]).verify(_token()).reason == "fuses"


def test_the_static_check_is_cached_per_process_and_snapshot() -> None:
    backend = _Backend()
    verifier = _verifier(backend)
    assert verifier.verify(_token()).ok and verifier.verify(_token()).ok
    assert backend.static_calls == 1
    # A recycled pid (new pidversion) is a different process: re-checked.
    assert verifier.verify(_token(pidversion=_PIDVERSION + 1)).ok
    assert backend.static_calls == 2


def test_any_backend_exception_is_a_refusal() -> None:
    class _Raises(_Backend):
        def guest_facts(self, token: bytes) -> GuestFacts:
            raise OSError("boom")

    verdict = _verifier(_Raises()).verify(_token())
    assert verdict == app_attest.Verdict(False, "error:OSError", 0)
    assert _verifier(_Backend()).verify(b"short").reason == "error:ValueError"


@pytest.mark.parametrize(
    ("wires", "hardened"),
    [
        ([b"010011011"], True),
        ([b"010011"], True),  # only positions 0-5 are read; a longer wire is fine
        ([b"010011011", b"101100011"], False),  # a universal binary: every slice
        ([b"0010"], False),  # too short to carry OnlyLoadAppFromAsar
        ([b""], False),  # unknown wire version
        ([], False),  # no sentinel at all
    ],
)
def test_fuse_wire_policy(wires: list[bytes], hardened: bool) -> None:
    assert app_attest.fuses_hardened(wires) is hardened


def test_fuse_wires_reads_every_slice(tmp_path: Path) -> None:
    blob = tmp_path / "fw"
    blob.write_bytes(
        b"x" * 10 + FUSE_SENTINEL + b"\x01\x09010011011" + b"y" * 5 + FUSE_SENTINEL + b"\x02\x03abc"
    )
    assert app_attest.fuse_wires(str(blob)) == [b"010011011", b""]


def test_ctime_snapshot_covers_contents_downward(tmp_path: Path) -> None:
    bundle = tmp_path / "X.app"
    for rel in app_attest._GUARDED_FILES:
        (bundle / rel).parent.mkdir(parents=True, exist_ok=True)
        (bundle / rel).write_bytes(b"")
    names = {name for name, _ in app_attest.ctime_snapshot(bundle)}
    assert (
        "Contents" in names and "Contents/Frameworks/Electron Framework.framework/Versions" in names
    )
    assert "" not in names and "." not in names  # the bundle root itself is excluded
    assert app_attest.bundle_of(_EXE) == Path("/Applications/Local Operator.app")
    assert app_attest.bundle_of("/Applications/X.app/Contents/Resources/python3") is None


# -- the table ---------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_table_is_single_use_content_bound_and_expires() -> None:
    clock = _Clock()
    table = AttestationTable(clock=clock)
    table.record("s", "a-1", "respond", {"q1": ["Yes"]}, [])
    # N13: different content consumes nothing.
    assert not table.consume("s", "a-1", "respond", {"q1": ["No, keep it"]}, [])
    assert not table.consume("s", "a-1", "revise", {"q1": ["Yes"]}, [])
    assert table.consume("s", "a-1", "respond", {"q1": ["Yes"]}, [])
    assert not table.consume("s", "a-1", "respond", {"q1": ["Yes"]}, [])  # single use
    table.record("s", "a-1", "respond", {"q1": ["Yes"]}, [])
    clock.now = app_attest.ATTESTATION_TTL_S + 1
    assert not table.consume("s", "a-1", "respond", {"q1": ["Yes"]}, [])


def test_image_digests_bind_the_decoded_bytes() -> None:
    png = b"\x89PNG fake"
    wire = [{"question_id": "q1", "data_b64": base64.b64encode(png).decode(), "mime_type": "x"}]
    assert app_attest.image_digests(wire) == [hashlib.sha256(png).hexdigest()]


# -- the listener protocol (injected token + verifier) --------------------------


class _FixedVerifier:
    def __init__(self, ok: bool) -> None:
        self.ok = ok
        self.calls = 0

    def verify(self, token: bytes) -> app_attest.Verdict:
        self.calls += 1
        return app_attest.Verdict(self.ok, "ok" if self.ok else "requirement:-67050", 1)


async def _dial(path: Path, frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reader, writer = await asyncio.open_unix_connection(str(path))
    replies = [json.loads(await reader.readline())]
    if not replies[0]["ok"]:
        # A refused peer is closed right after its verdict; nothing more to send.
        writer.close()
        return replies
    for frame in frames:
        writer.write(json.dumps(frame).encode() + b"\n")
        await writer.drain()
        line = await reader.readline()
        if not line:
            break
        replies.append(json.loads(line))
    writer.close()
    return replies


@pytest.fixture
def short_root() -> Any:
    # AF_UNIX paths cap at 104 bytes and pytest's tmp_path on macOS is long.
    root = tempfile.mkdtemp(prefix="lopat", dir="/tmp" if sys.platform == "darwin" else None)
    yield Path(root)
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.asyncio
async def test_listener_records_frames_only_for_a_verified_peer(short_root: Path) -> None:
    good = AttestService("sid-1", verifier=_FixedVerifier(True), token_of=lambda s: _token())
    path = await good.start(short_root / "run")
    assert path is not None
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert oct(path.parent.stat().st_mode & 0o777) == "0o700"
    frame = {
        "op": "ask_attest",
        "session_id": "sid-1",
        "ask_id": "a-1",
        "act": "respond",
        "answers": {"q1": ["Yes"]},
        "image_sha256": [],
    }
    replies = await _dial(path, [frame, {**frame, "session_id": "other"}, {**frame, "act": "x"}])
    assert replies == [
        {"ok": True},
        {"ok": True},
        {"ok": False, "reason": "wrong-session"},
        {"ok": False, "reason": "bad-frame"},
    ]
    assert good.table.consume("sid-1", "a-1", "respond", {"q1": ["Yes"]}, [])
    await good.close()
    assert not path.exists()

    bad = AttestService("sid-1", verifier=_FixedVerifier(False), token_of=lambda s: _token())
    path = await bad.start(short_root / "run")
    assert path is not None
    replies = await _dial(path, [frame])
    # Refused at accept and closed: the frame never reaches the table.
    assert replies == [{"ok": False, "reason": "requirement:-67050"}]
    assert not bad.table.consume("sid-1", "a-1", "respond", {"q1": ["Yes"]}, [])
    await bad.close()


@pytest.mark.asyncio
async def test_a_successors_socket_is_not_unlinked_by_the_retiring_runtime(
    short_root: Path,
) -> None:
    old = AttestService("sid-2", verifier=_FixedVerifier(True), token_of=lambda s: _token())
    new = AttestService("sid-2", verifier=_FixedVerifier(True), token_of=lambda s: _token())
    path = await old.start(short_root / "run")
    assert await new.start(short_root / "run") == path
    await old.close()
    assert path is not None and path.exists()
    await new.close()
    assert not path.exists()


@pytest.mark.asyncio
async def test_a_too_long_path_binds_nothing(short_root: Path) -> None:
    service = AttestService("s" * 120, verifier=_FixedVerifier(True))
    assert await service.start(short_root / "run") is None


# -- the stamp matrix through the real dispatch ---------------------------------


_LOCAL_UNPROVEN = _ConnectionAuthority(locality="local", surface="desktop", kind="attach")
_PROVED = _ConnectionAuthority(
    locality="local", surface="desktop", kind="attach", operator_proved=True
)


def _attest_rig(rig: Any) -> AttestationTable:
    """Give the rig's server an attestation table, as ``_start_attest_listener`` does."""
    table = AttestationTable()
    rig.server._attest = types.SimpleNamespace(session_id=rig.session_id, table=table)
    return table


@pytest.mark.asyncio
async def test_n1_a_key_holder_claiming_desktop_stays_unverified(home: Any) -> None:  # noqa: F811
    rig = _Rig(*home, tag="N1")
    _attest_rig(rig)
    reply = await rig.dispatch(
        "ask_respond",
        {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}, "by": "desktop"},
        _LOCAL_UNPROVEN,
    )
    assert isinstance(reply, surfaces.StampedReceipt) and reply == "answered"
    assert reply.surface == "unverified"
    assert rig.record()["answered_by"] == {"surface": "unverified"}


@pytest.mark.asyncio
async def test_an_attested_answer_is_desktop_and_lifts_the_delete(home: Any) -> None:  # noqa: F811
    rig = _Rig(*home, tag="OK")
    table = _attest_rig(rig)
    table.record(rig.session_id, rig.ask_id, "respond", {"q1": ["Yes"]}, [])
    reply = await rig.dispatch(
        "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _LOCAL_UNPROVEN
    )
    assert reply.surface == "desktop"
    assert surfaces.consent_bearing(reply.surface) is True
    assert rig.record()["answered_by"] == {"surface": "desktop"}
    outcome, judge_calls = await rig.guard()
    assert outcome is not None and outcome.blocked is False and judge_calls == 0


@pytest.mark.asyncio
async def test_n13_attested_content_must_match_the_frame(home: Any) -> None:  # noqa: F811
    rig = _Rig(*home, tag="N13")
    table = _attest_rig(rig)
    table.record(rig.session_id, rig.ask_id, "respond", {"q1": ["No, keep it"]}, [])
    reply = await rig.dispatch(
        "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _LOCAL_UNPROVEN
    )
    assert reply.surface == "unverified"
    assert rig.record()["answered_by"] == {"surface": "unverified"}
    # The operator's attestation was not burned by the mismatching frame.
    assert table.consume(rig.session_id, rig.ask_id, "respond", {"q1": ["No, keep it"]}, [])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "authority",
    [
        _ConnectionAuthority(
            locality="remote", surface="desktop", kind="attach", operator_proved=True
        ),
        _ConnectionAuthority(locality="local", surface="desktop", kind="daemon"),
    ],
    ids=["mesh-relay", "phone-daemon"],
)
async def test_n15_relayed_frames_are_remote_and_never_consume(
    home: Any, authority: _ConnectionAuthority  # noqa: F811
) -> None:
    rig = _Rig(*home, tag=f"N15{authority.kind}")
    table = _attest_rig(rig)
    table.record(rig.session_id, rig.ask_id, "respond", {"q1": ["Yes"]}, [])
    reply = await rig.dispatch(
        "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, authority
    )
    assert reply.surface == "remote"
    assert table.consume(rig.session_id, rig.ask_id, "respond", {"q1": ["Yes"]}, [])


def _inherit(rig: Any) -> None:
    """Make the rig's ask one an EARLIER runtime minted: forget it was minted here."""
    surfaces._MINTED_HERE.discard((rig.session_id, rig.ask_id))
    assert not rig.queue.minted_here(rig.ask_id)


@pytest.mark.asyncio
async def test_n17_a_cold_starters_cap_cannot_answer_an_inherited_ask(
    home: Any,  # noqa: F811
) -> None:
    rig = _Rig(*home, tag="N17")
    assert rig.queue.minted_here(rig.ask_id)  # enqueue registered it
    _inherit(rig)
    reply = await rig.dispatch(
        "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _PROVED
    )
    assert reply.surface == "unverified"
    assert surfaces.consent_bearing(reply.surface) is False
    outcome, judge_calls = await rig.guard()
    assert outcome is not None and outcome.blocked is True


@pytest.mark.asyncio
async def test_n17_the_legacy_mirror_follows_the_minted_rule(home: Any) -> None:  # noqa: F811
    rig = _Rig(*home, tag="N17m")
    _inherit(rig)
    await rig.dispatch(
        "ask_answer", {"request_id": mirror_request_id(rig.ask_id, 0), "value": "Yes"}, _PROVED
    )
    assert rig.record()["answered_by"] == {"surface": "unverified"}


@pytest.mark.asyncio
async def test_n17_a_verified_surface_answers_an_inherited_ask(home: Any) -> None:  # noqa: F811
    rig = _Rig(*home, tag="N17v")
    _inherit(rig)
    table = _attest_rig(rig)
    table.record(rig.session_id, rig.ask_id, "respond", {"q1": ["Yes"]}, [])
    reply = await rig.dispatch(
        "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _PROVED
    )
    assert reply.surface == "desktop"


@pytest.mark.asyncio
async def test_n17_a_cap_still_vouches_for_an_ask_minted_here(home: Any) -> None:  # noqa: F811
    rig = _Rig(*home, tag="N17h")
    reply = await rig.dispatch(
        "ask_respond", {"ask_id": rig.ask_id, "answers": {"q1": ["Yes"]}}, _PROVED
    )
    assert reply.surface == "desktop"


def test_n18_answered_in_chat_is_unchanged(home: Any) -> None:  # noqa: F811
    rig = _Rig(*home, tag="N18")
    _inherit(rig)
    out = rig.queue.withdraw(rig.ask_id, reason="answered_in_chat", answers={"q1": ["Yes"]})
    assert out.get("ok"), out
    assert rig.record()["answered_by"]["surface"] == "chat"


def test_consent_bearing_is_unknown_for_an_old_runtime() -> None:
    assert surfaces.consent_bearing(None) is None
    receipt = surfaces.StampedReceipt("answered")
    assert receipt == "answered" and receipt.surface is None


# -- macOS: the real verifier refuses nc and Python (N2) -------------------------


@pytest.mark.skipif(sys.platform != "darwin", reason="Security.framework is macOS-only")
@pytest.mark.asyncio
async def test_real_verifier_refuses_nc_and_python(short_root: Path) -> None:
    service = AttestService("sid-n2")
    path = await service.start(short_root / "run")
    assert path is not None
    clients = [["/usr/bin/nc", "-U", str(path)]]
    probe = (
        f"import socket; s = socket.socket(socket.AF_UNIX); s.connect({str(path)!r}); "
        "print(s.recv(512).decode())"
    )
    for python in ("/opt/homebrew/bin/python3", sys.executable):
        if os.path.exists(python):
            clients.append([python, "-c", probe])
    try:
        for argv in clients:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )
            out, _ = await asyncio.wait_for(proc.communicate(), 60)
            reply = json.loads(out.decode().strip().splitlines()[0])
            assert reply["ok"] is False, (argv[0], reply)
            assert reply["reason"].startswith("requirement:"), (argv[0], reply)
    finally:
        await service.close()


def test_the_attach_client_surfaces_the_stamped_label() -> None:
    """``AttachClient.ask_respond``'s receipt carries the ack's ``surface``."""
    from local_operator.mobile.attach_client import _stamped_receipt

    receipt = _stamped_receipt({"op": "ack", "detail": "answered", "surface": "desktop"})
    assert receipt == "answered" and receipt.surface == "desktop"
    assert surfaces.consent_bearing(receipt.surface) is True
    old = _stamped_receipt({"op": "ack", "detail": "answered"})
    assert old.surface is None and surfaces.consent_bearing(old.surface) is None
