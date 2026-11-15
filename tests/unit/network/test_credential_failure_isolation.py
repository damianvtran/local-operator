"""Attribution controls: credential repair is not mesh messaging authority.

Only disposable loopback relays and the mock provider are used. The CLI is a
separate process and the receiving runtime is the real session-plane fixture.
"""

from __future__ import annotations

import asyncio
import contextlib
import gc
import ipaddress
import json
import os
import signal
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.config_writer import ConfigWriteLockTimeout
from local_operator.network import addresses, audit, dial, relay, store
from local_operator.network.credentials import owner as owner_mod
from tests.unit.network.test_relay_e2e import devices  # noqa: F401
from tests.unit.network.test_session_plane import _create_named_session_on_a_real_peer
from tests.unit.network.test_sessions_pilot_cli import _journals

TREE = Path(__file__).resolve().parents[3]


@pytest.fixture
def peer_pair(request: pytest.FixtureRequest) -> Any:
    return request.getfixturevalue("devices")


def _loopback_only(event: str, args: tuple[Any, ...]) -> None:
    if event == "socket.connect" and isinstance(args[1], tuple):
        if not ipaddress.ip_address(args[1][0]).is_loopback:
            raise RuntimeError("attribution fixture refuses non-loopback access")


@pytest.fixture(autouse=True)
def _guard(monkeypatch: pytest.MonkeyPatch) -> None:
    sys.addaudithook(_loopback_only)
    monkeypatch.setattr(addresses, "_default_route_address", lambda: None)
    with socket.socket() as probe:
        with pytest.raises(RuntimeError, match="non-loopback"):
            probe.connect(("203.0.113.9", 1))

    def check_source(root: Path) -> None:
        for module in (owner_mod, relay, audit):
            filename = module.__file__
            assert isinstance(filename, str) and Path(filename).resolve().is_relative_to(
                root
            ), "wrong attribution source tree"

    check_source(TREE)
    with pytest.raises(AssertionError, match="wrong attribution source"):
        check_source(TREE / "WRONG_TREE")


def _cli(root: Path, home: Path, session: str) -> tuple[int, dict[str, Any], str]:
    program = """import ipaddress,pathlib,runpy,socket,sys
from local_operator.network import addresses,cli,projection,relay
root=pathlib.Path(sys.argv[1]);sys.argv=sys.argv[2:]
assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) for m in (cli,projection,relay))
def guard(event,args):
 if event=='socket.connect' and isinstance(args[1],tuple):
  if not ipaddress.ip_address(args[1][0]).is_loopback:raise RuntimeError('non-loopback refused')
sys.addaudithook(guard)
with socket.socket() as s:
 try:s.connect(('203.0.113.9',1))
 except RuntimeError:pass
 else:raise AssertionError('child negative control did not fire')
addresses._default_route_address=lambda:None
runpy.run_module('local_operator.cli',run_name='__main__')
"""
    home.mkdir(exist_ok=True)
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "PYTHONPATH": str(TREE),
        "PYTHONDONTWRITEBYTECODE": "1",
        "TERM": "xterm-256color",
    }
    command = [
        sys.executable,
        "-c",
        program,
        str(TREE),
        "lop",
        "network",
        "sessions",
        "--json",
        "--peer",
        "device-b",
        "--send",
        session,
        "synthetic message with a pending repair",
    ]
    process = subprocess.Popen(
        command,
        env=env,
        cwd=home,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = process.communicate(timeout=50)
    finally:
        if process.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate(timeout=5)
    with pytest.raises(ProcessLookupError):
        os.killpg(process.pid, 0)
    assert process.returncode is not None
    assert out.strip(), (process.returncode, err)
    return process.returncode, json.loads(out), err


def _repair(server: Any, borrower: str) -> None:
    record = store.list_networks(server.root)[0]
    server.audit.record(
        audit.AuditEvent(
            event="credential.report",
            actor=server.identity.device_id,
            network_id=record.network_id,
            subject=borrower,
            detail={
                "credential_key": "mcp:synthetic-repair",
                "act": server.identity.device_id,
                "sub": borrower,
                "failure": "interactive_required",
            },
        )
    )
    server.audit.record(
        audit.AuditEvent(
            event="credential.grant_refused",
            actor=server.identity.device_id,
            network_id=record.network_id,
            subject=borrower,
            outcome="refused",
            detail={"credential_key": "github", "failure": "rate_limited"},
        )
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["allowed", "unauthorized", "runtime_refused"])
async def test_pending_repair_does_not_decide_stream_authority(
    peer_pair: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="repair-isolation",
        prompt="",
    )
    try:
        a, b = created.server_a, created.server_b
        (a.root / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n"
        )
        _repair(b, a.identity.device_id)
        before = [c for c in b.doctor()["checks"] if c.get("check") == "credential_repair"]
        assert before and all(c["ok"] is False for c in before), before
        if mode == "unauthorized":
            record = store.list_networks(b.root)[0]
            with store.mutate(record.network_id, b.root) as fresh:
                member = fresh.member(a.identity.device_id)
                assert member is not None
                member.capabilities = [c for c in member.capabilities if c != "prompt"]
                store.save(fresh, b.root)
            current = store.load(record.network_id, b.root).member(a.identity.device_id)
            assert current is not None and "prompt" not in current.capabilities
        elif mode == "runtime_refused":

            def refuse(*_args: Any, **_kwargs: Any) -> Any:
                raise dial.OwnerUnreachable("synthetic runtime refuses its connection")

            monkeypatch.setattr(dial, "dial_owner", refuse)
        wire_replies: list[dict[str, Any] | None] = []
        request = relay.PeerLink.request

        def capture(link: Any, frame: dict[str, Any], **kwargs: Any) -> Any:
            result = request(link, frame, **kwargs)
            if frame.get("op") == "net_stream" and frame.get("action") == "open":
                wire_replies.append(result)
            return result

        monkeypatch.setattr(relay.PeerLink, "request", capture)
        rc, payload, err = await asyncio.to_thread(
            _cli, a.root, tmp_path / "cli-home", created.session_id
        )
        print("CLI_RESULT", json.dumps({"mode": mode, "rc": rc, "payload": payload}))
        refused = [r for r in b.audit.tail(limit=100) if r["event"] == "authorisation_refused"]
        print("WIRE_RESULT", json.dumps({"mode": mode, "replies": wire_replies}))
        print(
            "AUDIT_RESULT",
            json.dumps(
                {
                    "mode": mode,
                    "rows": [
                        {k: r.get(k) for k in ("cause", "actor_kind", "detail")} for r in refused
                    ],
                }
            ),
        )
        after = [c for c in b.doctor()["checks"] if c.get("check") == "credential_repair"]
        assert after and all(c["ok"] is False for c in after), after
        if mode != "allowed":
            assert rc == 1, (payload, err)
            assert payload["ok"] is False
            if mode == "runtime_refused":
                # Sender classification belongs to the separately sequenced #4060
                # wave. Here the peer reply and its durable owner-side cause are
                # the contract: authorization succeeded before this handler refused.
                assert wire_replies and all(
                    reply and reply.get("code") == "session_unreachable" for reply in wire_replies
                )
                assert refused and all(row["cause"] == "policy" for row in refused)
                assert all(
                    row["detail"]["refusal_code"] == "session_unreachable" for row in refused
                )
                assert all(
                    row["detail"]["reason"] == "synthetic runtime refuses its connection"
                    for row in refused
                )
                assert "synthetic runtime refuses its connection" in payload["message"], payload
            else:
                assert "capability" in payload["error"], payload
                assert not any(
                    "synthetic message with a pending repair" in json.dumps(row)
                    for row in created.owner.transcript_entries()
                ), "an unauthorized message must not reach the owner's transcript"
            assert "did not answer" not in str(payload), payload
        else:
            assert rc == 0 and payload["ok"] is True, (payload, err)
            assert await _journals(created, "synthetic message with a pending repair")
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_lock_timeout_is_not_left_on_an_unobserved_coalescer_future() -> None:
    # This directly isolates future ownership, without taking any live config lock.
    broker = object.__new__(owner_mod.MeshCredentialBroker)
    broker._inflight = {}
    loop = asyncio.get_running_loop()
    previous = loop.get_exception_handler()
    unobserved: list[dict[str, Any]] = []
    loop.set_exception_handler(lambda _loop, context: unobserved.append(context))

    async def fail(**_fields: Any) -> Any:
        raise ConfigWriteLockTimeout("synthetic configuration writer is busy")

    broker._resolve_once = fail  # type: ignore[method-assign]
    try:
        with pytest.raises(ConfigWriteLockTimeout):
            await broker._coalesced_resolve(key="github", for_session="synthetic", force=False)
        gc.collect()
        await asyncio.sleep(0)
        print(
            "COALESCER_RESULT",
            json.dumps(
                {
                    "inflight": len(broker._inflight),
                    "unobserved": [str(c.get("message")) for c in unobserved],
                }
            ),
        )
        assert not broker._inflight
        assert not unobserved, [c.get("message") for c in unobserved]
    finally:
        loop.set_exception_handler(previous)


def test_handler_refusal_audit_scrubs_and_bounds_the_reason(peer_pair: Any) -> None:
    from types import SimpleNamespace

    from local_operator.mcp import redaction
    from local_operator.network.authorizer import Granted
    from local_operator.network.types import MeshRefusal

    server = peer_pair[0]
    # Synthetic canaries only. One checks exact in-memory values; the other
    # checks credential shapes the relay was never given as a known value.
    known = "synthetic-audit-canary-" + "q" * 24
    shaped = "ghp_" + "z" * 36
    sentence = f"synthetic refusal {known}\n{shaped}\n" + "x" * 1200
    redaction.register(known)
    link: Any = SimpleNamespace(device_id="d_synthetic", network_id="n_synthetic", epoch=1)

    def refuse(_link: Any, _frame: Any) -> Any:
        raise MeshRefusal("session_unreachable", sentence)

    try:
        reply = server._run_handler(link, {"req": 81}, refuse, Granted(action="net_stream"))
        # This is observability only: do not change what the peer was already told.
        assert reply["code"] == "session_unreachable"
        assert reply["message"] == sentence
        row = server.audit.tail(limit=1)[0]
        assert row["cause"] == "policy"
        assert row["detail"]["refusal_code"] == "session_unreachable"
        reason = row["detail"]["reason"]
        assert not any(value in reason for value in (known, shaped))
        assert "[red" + "acted]" in reason
        assert "\n" not in reason and "\r" not in reason
        assert len(reason) <= audit.MAX_DETAIL_VALUE_CHARS
    finally:
        redaction.unregister(known)


@pytest.mark.asyncio
async def test_coalescer_joiners_keep_the_existing_failure_resolution() -> None:
    broker = object.__new__(owner_mod.MeshCredentialBroker)
    broker._inflight = {}
    entered = asyncio.Event()
    release = asyncio.Event()
    calls = 0
    failure = ConfigWriteLockTimeout("synthetic writer is busy")

    async def fail(**_fields: Any) -> Any:
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        raise failure

    broker._resolve_once = fail  # type: ignore[method-assign]
    fields = {"key": "github", "for_session": "synthetic", "force": False}
    leader = asyncio.create_task(broker._coalesced_resolve(**fields))
    await entered.wait()
    joiner = asyncio.create_task(broker._coalesced_resolve(**fields))
    await asyncio.sleep(0)
    release.set()
    results = await asyncio.gather(leader, joiner, return_exceptions=True)
    assert all(result is failure for result in results)
    # A joiner still resolves for itself after the leader fails, as before.
    assert calls == 2
    assert not broker._inflight
