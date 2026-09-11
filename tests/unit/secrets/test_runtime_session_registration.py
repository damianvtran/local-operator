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
from local_operator.session.runtime.serving import ServingSessionHandle
from local_operator.variables import VariableStore

_SUPPORTED = sys.platform == "darwin" or sys.platform.startswith("linux")
pytestmark = pytest.mark.skipif(
    not _SUPPORTED, reason="the broker is implemented for macOS and Linux"
)

SECRET_NAME = "RUNTIME_SINK_TOKEN"
SECRET_VALUE = "runtime-sink-7c1f0b2a"


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


def _await_absent(config_root: Path, pid: int, timeout: float = 5.0) -> set[int]:
    """Poll until ``pid`` leaves the broker's session table, or the deadline.

    The broker notices a closed channel on its own liveness poll, so a
    deregistration is prompt but not instantaneous; asserting immediately would
    test the poll interval rather than the deregistration.
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
