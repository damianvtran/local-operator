"""The RUNTIME registers the session the §6 notice has to reach (design §2.1, §6).

**The defect these tests pin.** The broker notifies the registered session that
OWNS the requesting peer and denies the retrieval when that session does not
acknowledge in time. The registration used to be made by the TUI process, whose
sink read ``app._session.variables`` — and once the attached facade landed
(#937) ``app._session`` was an ``AttachedSession`` with no store at all, so the
sink raised, nothing was ever acked, and EVERY descendant retrieval in EVERY
attached session was denied ("session <pid> did not acknowledge the redaction
notice within 2s"). The registrant has to be the process that holds the
``VariableStore`` the bash and eval redactors read, which is the session
runtime, not the viewer — hence ``ServingSessionHandle``.

These tests spawn REAL descendant processes rather than faking a peer identity,
for the reason ``test_broker.py`` states for its own: the mechanism is what the
kernel reports about the process on the other end of the socket, so a mocked pid
would prove nothing about the boundary that broke.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import local_operator
from local_operator.secrets import client
from local_operator.secrets.broker import SecretBroker
from local_operator.secrets.keys import key_path, write_private_file
from local_operator.secrets.store import SecretStore
from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.tui import _register_secret_session
from local_operator.variables import VariableStore

_SUPPORTED = sys.platform == "darwin" or sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(
    not _SUPPORTED, reason="the broker is implemented for macOS and Linux"
)

SECRET_NAME = "RUNTIME_SINK_TOKEN"
SECRET_VALUE = "runtime-sink-7c1f0b2a"

#: A second value used by the forwarding tests, so a failure cannot be
#: explained by the registry test having leaked the first one.
FORWARDED_VALUE = "viewer-forward-4d9e1c"


class RuntimeSessionDouble:
    """The slice of ``Session`` the handle's construction and teardown touch.

    Local and deliberately minimal rather than shared with the runtime suite's
    fake: this file's subject is the secret registration, so the double only
    carries the ``VariableStore`` the §6 sink reads plus the handful of
    attributes the projection and the command reservations read while the
    handle is built. ``variables`` is the ONE field under test — a store here
    models the runtime, ``None`` models a session with nothing to scrub
    through.
    """

    def __init__(self, variables: VariableStore | None) -> None:
        self.session_id = "runtime-session-under-test"
        self.conversation_name = ""
        self.model = None
        self.variables = variables
        self.disposed = False

    def history(self) -> list[Any]:
        return []

    def subscribe(self, handler: Any) -> Any:
        """The event subscription ``RuntimeServer.start_in_process`` installs.

        Only the wire test starts a server; the rest of this file never does, so
        this hook exists purely so a REAL ``RuntimeServer`` can be driven over a
        real socket against this double.
        """
        self._event_handler = handler
        return lambda: None

    def set_approval_handler(self, handler: Any) -> None:
        self.approval_handler = handler

    def set_ask_handler(self, handler: Any) -> None:
        self.ask_handler = handler

    async def dispose(self) -> None:
        self.disposed = True


@pytest.fixture
def running_broker(config_root: Path, master_key: bytes) -> Iterator[SecretBroker]:
    """A live keyfile-tier broker over the isolated config dir."""
    store = SecretStore(master_key, base=config_root)
    store.initialize()
    store.set(SECRET_NAME, SECRET_VALUE.encode(), description="runtime sink")
    write_private_file(key_path(config_root), master_key)
    instance = SecretBroker(config_root, key_provider=lambda: master_key, idle_shutdown_s=0)
    instance.start()
    try:
        yield instance
    finally:
        instance.stop(timeout=5)


def _sessions(config_root: Path) -> set[int]:
    status = client.broker_status(config_root) or {}
    return set(status.get("sessions") or ())


def _await_absent(config_root: Path, pid: int, timeout: float = 10.0) -> set[int]:
    """Poll until ``pid`` leaves the broker's session table, or the deadline.

    The broker notices a closed channel on its own 1 s liveness poll, so a
    deregistration is prompt but not instantaneous; asserting immediately would
    test the poll interval rather than the deregistration. The bound matches the
    one the existing deregistration test uses
    (``test_a_dead_sessions_descendants_stop_being_authorized``) — a loaded CI
    runner is slower than a workstation, and a tight bound there fails the
    window rather than the behaviour.
    """
    deadline = time.monotonic() + timeout
    sessions = _sessions(config_root)
    while pid in sessions and time.monotonic() < deadline:
        time.sleep(0.05)
        sessions = _sessions(config_root)
    return sessions


def _retrieve_in_child(config_root: Path, name: str) -> subprocess.CompletedProcess[str]:
    """A REAL descendant process retrieving ``name`` through the broker.

    The child imports the tree UNDER TEST rather than whatever ``lop`` is
    installed on this machine — without PYTHONPATH pointing at the worktree it
    would silently exercise the released runtime and report on the wrong code.
    """
    tree = str(Path(local_operator.__file__).resolve().parent.parent)
    script = textwrap.dedent(f"""
        import sys
        from pathlib import Path
        from local_operator.secrets import client

        try:
            value = client.retrieve({name!r}, Path({str(config_root)!r}))
        except Exception as error:
            print(f"{{type(error).__name__}}: {{error}}", file=sys.stderr)
            raise SystemExit(2)
        print(value.decode())
        """)
    return subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "PYTHONPATH": tree},
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.asyncio
async def test_a_descendant_retrieval_is_served_when_the_runtime_registered(
    config_root: Path, running_broker: SecretBroker
) -> None:
    """The regression: the handle's registration is what lets a child be served.

    On the pre-fix tree ``ServingSessionHandle`` registered nothing, so this
    process was never in the broker's session table and the child below was
    denied for having no registered session ancestor at all. With the fix the
    runtime is the registered session, its ``VariableStore`` answers the §6
    notice, and the value is served — registered for redaction BEFORE the ack
    that released it to the child.
    """
    variables = VariableStore()
    session = RuntimeSessionDouble(variables)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(config_root))
    try:
        assert os.getpid() in _sessions(config_root), (
            "the runtime handle did not register this process as the session the "
            "broker notifies; a descendant retrieval would be denied"
        )

        result = _retrieve_in_child(config_root, SECRET_NAME)

        assert result.returncode == 0, f"the child was refused: {result.stderr}"
        assert result.stdout.strip() == SECRET_VALUE, "the child was not served the value"
        # Registered by the time the child could print: the sink ran before the
        # ack, which is the whole §6 ordering.
        assert (
            SECRET_VALUE in variables.redaction_values()
        ), "the value was served without reaching the store that owns the filter"
        assert variables.redact(f"token={SECRET_VALUE}") == "token=[redacted]"
    finally:
        handle.close_secret_registration()
        assert os.getpid() not in _await_absent(
            config_root, os.getpid()
        ), "the closed registration still authorized this process"


@pytest.mark.asyncio
async def test_a_runtime_with_no_store_registers_nothing(config_root: Path) -> None:
    """A store-less runtime registers nothing, and starts no broker doing it.

    The registration is skipped when no store exists at this base: there is
    nothing any descendant could be served from it, and the skip is what keeps
    a session that never uses the store — most sessions, and every test double
    that builds a handle — from spawning a broker daemon. No ``running_broker``
    fixture here on purpose: the assertion is that none appears.
    """
    handle = ServingSessionHandle(
        RuntimeSessionDouble(VariableStore()), asyncio.get_running_loop(), cwd=str(config_root)
    )
    try:
        assert (
            client.broker_status(config_root) is None
        ), "a runtime with no secret store started a broker anyway"
    finally:
        handle.close_secret_registration()


@pytest.mark.asyncio
async def test_a_sink_that_cannot_register_denies_and_names_the_runtime_pid(
    config_root: Path, running_broker: SecretBroker
) -> None:
    """Fail-closed is preserved, and the denial names THIS process.

    A session with no ``VariableStore`` — the shape the attached TUI's facade
    had — must not be served a value nothing can scrub. The broker denies, and
    the message names the runtime whose sink could not register, which is what
    makes the failure diagnosable instead of a silent empty substitution.
    """
    handle = ServingSessionHandle(
        RuntimeSessionDouble(None), asyncio.get_running_loop(), cwd=str(config_root)
    )
    try:
        assert os.getpid() in _sessions(
            config_root
        ), "the registration itself must succeed; the sink, not the session, is what is missing"

        result = _retrieve_in_child(config_root, SECRET_NAME)

        assert result.returncode == 2, f"the child was served: {result.stdout!r}"
        assert SECRET_VALUE not in result.stdout
        assert SECRET_VALUE not in result.stderr
        assert "did not acknowledge the redaction notice" in result.stderr
        assert f"session {os.getpid()}" in result.stderr
    finally:
        handle.close_secret_registration()


@pytest.mark.asyncio
async def test_disposing_the_handle_deregisters_the_session(
    config_root: Path, running_broker: SecretBroker
) -> None:
    """Teardown revokes: a session that is gone must stop authorizing (§2.1)."""
    session = RuntimeSessionDouble(VariableStore())
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(config_root))
    assert os.getpid() in _sessions(config_root)

    await handle.dispose()

    assert session.disposed
    assert os.getpid() not in _await_absent(config_root, os.getpid()), (
        "the disposed handle left this process registered, so its descendants "
        "would stay authorized behind a session that has ended"
    )


@pytest.mark.asyncio
async def test_the_runtime_handler_registers_a_forwarded_value(config_root: Path) -> None:
    """The owner side of the forward writes to its store — and nowhere else.

    This is what ``AttachedSession.register_secret_redaction`` reaches over the
    control channel: the value goes into the ``VariableStore`` the bash and eval
    redactors read, and the handler raises when there is no store, so the
    viewer's sink declines to acknowledge and the broker denies.
    """
    variables = VariableStore()
    handle = ServingSessionHandle(
        RuntimeSessionDouble(variables), asyncio.get_running_loop(), cwd=str(config_root)
    )
    try:
        handle.register_secret_redaction(FORWARDED_VALUE)
        assert FORWARDED_VALUE in variables.redaction_values()
        assert variables.redact(f"x={FORWARDED_VALUE}") == "x=[redacted]"
        # Registered as a REDACTION, never as a credential: the value must not
        # become readable back out of the store's credential surface.
        assert FORWARDED_VALUE not in variables.credential_names()
        # And the op must NOT be reachable through the ``/credential`` verb
        # table: a verb whose only job is to scrub a value must never be able to
        # store one, and the table's own unknown-action reply is the proof.
        refusal = await handle.credential_op("register_secret_redaction", "", "x")
        assert refusal.get("ok") is False and refusal.get("reason") == "unknown-action"
    finally:
        handle.close_secret_registration()

    storeless = ServingSessionHandle(
        RuntimeSessionDouble(None), asyncio.get_running_loop(), cwd=str(config_root)
    )
    try:
        with pytest.raises(RuntimeError):
            storeless.register_secret_redaction(FORWARDED_VALUE)
    finally:
        storeless.close_secret_registration()


@pytest.mark.asyncio
async def test_the_wire_op_registers_the_value_in_the_runtime_store(config_root: Path) -> None:
    """The viewer→runtime forward really crosses the control channel.

    A real ``RuntimeServer`` over a real handle, driven by a real authenticated
    socket: the payload op registers the value with the runtime's redactor and
    answers ``True``. An OLD owner would answer unknown-op here, which the
    viewer's sink turns into a raise (fail closed), so the seam degrades safely.
    """
    variables = VariableStore()
    handle = ServingSessionHandle(
        RuntimeSessionDouble(variables), asyncio.get_running_loop(), cwd=str(config_root)
    )
    runtime = RuntimeServer(handle, kind="daemon")
    await runtime.start_in_process()
    writer = None
    try:
        record = runtime._record
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", record.control_port, limit=1 << 20
        )
        writer.write(json.dumps({"key": record.control_key, "client": "attach"}).encode() + b"\n")
        await writer.drain()
        welcome = json.loads(await asyncio.wait_for(reader.readline(), timeout=5))
        assert welcome.get("op") == "projection"

        writer.write(
            json.dumps(
                {"op": "register_secret_redaction", "req": 1, "value": FORWARDED_VALUE}
            ).encode()
            + b"\n"
        )
        await writer.drain()
        for _ in range(30):
            frame = json.loads(await asyncio.wait_for(reader.readline(), timeout=5))
            if frame.get("op") == "result" and frame.get("req") == 1:
                break
        else:  # pragma: no cover - only on a broken server
            raise AssertionError("no result frame for register_secret_redaction")
        assert frame["data"] is True
        assert FORWARDED_VALUE in variables.redaction_values()
    finally:
        if writer is not None:
            writer.close()
        await runtime.aclose()


@pytest.mark.asyncio
async def test_an_attached_viewer_forwards_the_notice_and_the_child_is_served(
    config_root: Path, running_broker: SecretBroker
) -> None:
    """The viewer's registration is NOT inert: it forwards to the runtime.

    The shape shipping ``lop`` has when a runtime booted before its store
    existed: the VIEWER is the only registered session, so the broker's notice
    lands on the viewer's sink on a plain thread. The sink resolves
    ``app._session.variables`` (absent on an attached facade), then forwards
    the value to the attached runtime over the control channel — the loop hop is
    exercised for real here — and only then acknowledges, so the child is served
    AND the value is in the filter before the value can be printed.
    """
    filtered = VariableStore()

    class AttachedDouble:
        async def register_secret_redaction(self, value: str) -> None:
            filtered.register_redaction(value)

    class AppDouble:
        _session = AttachedDouble()
        _loop = asyncio.get_running_loop()

    registration = _register_secret_session(AppDouble())
    assert registration is not None, "the viewer registration did not reach a broker"
    try:
        # Off the loop on purpose: the notice has to be answered by a coroutine
        # scheduled on THIS loop, so a synchronous subprocess.run here would
        # block the very loop the forward needs — and the timeout this test
        # would then see is a test artefact, not the production shape.
        result = await asyncio.to_thread(_retrieve_in_child, config_root, SECRET_NAME)

        assert result.returncode == 0, f"the child was refused: {result.stderr}"
        assert result.stdout.strip() == SECRET_VALUE
        assert SECRET_VALUE in filtered.redaction_values(), (
            "the value was served before the forwarded redaction reached the "
            "runtime's store, so the notice was not honoured"
        )
    finally:
        registration.close()
        _await_absent(config_root, os.getpid())


class _ForwardFails:
    """An attached runtime whose forward refuses (a lost owner, a full queue)."""

    async def register_secret_redaction(self, value: str) -> None:
        raise ConnectionError("runtime gone")


@pytest.mark.asyncio
async def test_an_attached_viewer_that_cannot_forward_denies_the_child(
    config_root: Path, running_broker: SecretBroker
) -> None:
    """A forward that fails must fail CLOSED, not serve the value unredacted.

    A noisy "log and continue" here would acknowledge the notice, let the broker
    serve the child, and leave the session's redactor unaware of the value — the
    §6 leak. The sink must raise instead, so the broker denies and the value
    never reaches the transcript.
    """

    class AppDouble:
        _session = _ForwardFails()
        _loop = asyncio.get_running_loop()

    registration = _register_secret_session(AppDouble())
    assert registration is not None
    try:
        result = await asyncio.to_thread(_retrieve_in_child, config_root, SECRET_NAME)

        assert result.returncode == 2, f"the child was served anyway: {result.stdout!r}"
        assert SECRET_VALUE not in result.stdout
        assert SECRET_VALUE not in result.stderr
        assert "did not acknowledge the redaction notice" in result.stderr
    finally:
        registration.close()
        _await_absent(config_root, os.getpid())


@pytest.mark.asyncio
async def test_the_viewer_sink_with_no_target_at_all_denies_the_child(
    config_root: Path, running_broker: SecretBroker
) -> None:
    """The last-resort branch: nowhere to scrub → deny, never serve.

    A session facade with neither a ``VariableStore`` nor a forwarding method
    has no filter anywhere the viewer can reach, so the sink raises and the
    broker denies. This is the branch that keeps the viewer entry fail-closed
    rather than a silent pass-through.
    """

    class AppDouble:
        _session = object()
        _loop = asyncio.get_running_loop()

    registration = _register_secret_session(AppDouble())
    assert registration is not None
    try:
        result = await asyncio.to_thread(_retrieve_in_child, config_root, SECRET_NAME)

        assert result.returncode == 2, f"the child was served anyway: {result.stdout!r}"
        assert SECRET_VALUE not in result.stdout
        assert "did not acknowledge the redaction notice" in result.stderr
    finally:
        registration.close()
        _await_absent(config_root, os.getpid())
