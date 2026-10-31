"""The incident, replayed: a stale holder is reaped; a live one is left alone.

THE DEFECT (measured 2026-10-30). Connector pid 48311 — generation
20261023T142851Z-0.86.1, ppid 1, seven days old, idle, no sockets — held
``<config>/team_bridge/connector.lock``. The supervised job
``com.local-operator.bridge-connector`` was refused ``ConnectorLockBusy`` 2,826
times and exited 75 each time, so nothing inbound was admitted for days.

These cells run REAL processes, because what went wrong is a property of
processes: a second connector's flock, its argv, its generation, and the kernel's
release of the lock when it dies. Nothing here touches the operator's machine —
every child runs under a temporary ``HOME`` whose generation tree this file
creates, with an explicitly built environment (no ``CMUX_*``, no live store), and
every signal goes to a pid this file started.

WHAT IS PINNED:

* the BEFORE arm — the shipped lock path with the reap switched off, i.e. what the
  code did before this change — exits 75 and leaves the stale holder holding the
  lock;
* the AFTER arm — the real entry point — SIGTERMs the stale holder and takes the
  lock;
* a LIVE CURRENT-GENERATION holder is never signalled: the connector refuses,
  exits 75, and the holder is still running;
* a holder whose own ``--config-dir`` is another store is never signalled even
  though it holds this store's lock file.

WHY A SHADOW PACKAGE FOR THE HOLDER. The process under observation must look like
a connector to ``ps``: the generation image first, then
``-m local_operator.team_bridge.connector --config-dir <store>``. Running the real
module would take the real lock and enrol, so the holder runs a two-line stub
*through that argv*, found via ``PYTHONPATH`` before the worktree.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import sysconfig
import time
from pathlib import Path
from typing import Callable

import pytest

from local_operator.team_bridge.connector.lock import ConnectorLock
from local_operator.team_bridge.store import OwnerReceipt, Store
from local_operator.team_bridge.types import OwnerSnapshot

ROOT = Path(__file__).resolve().parents[3]

OLD = "20261023T142851Z-0.86.1"
NEW = "20261030T145827Z-0.92.2"

#: A port nothing listens on: the connector reaches the lock and then polls, and
#: no test here depends on a hub.
HUB = "http://127.0.0.1:1/v1"

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url=HUB,
    enrollment_generation=1,
)

#: The stub holder: take the flock, say so on stdout, then sleep for ever. The
#: ``print`` is the handshake — the parent returns only once the lock is HELD, so
#: no test races the holder's start.
_STUB_MAIN = """\
import fcntl
import os
import sys
import time

config = os.environ.get("HOLDER_LOCK_STORE") or sys.argv[sys.argv.index("--config-dir") + 1]
path = os.path.join(config, "team_bridge", "connector.lock")
os.makedirs(os.path.dirname(path), exist_ok=True)
fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
fcntl.flock(fd, fcntl.LOCK_EX)
print("held", os.getpid(), flush=True)
while True:
    time.sleep(3600)
"""

#: The BEFORE arm: the shipped entry point with the reap switched off — exactly
#: what the lock path did before this change (busy -> log -> exit 75).
_WITHOUT_REAP = """\
import sys

from local_operator.team_bridge.connector import __main__ as entry, reap


async def disabled(config_dir):
    reap.logger.warning("bridge connector reap refused: proof=disabled_for_repro")
    return reap.ReapOutcome(kind="refused", reason="disabled")


reap.reap_stale_holder = disabled
raise SystemExit(entry.main(sys.argv[1:]))
"""


def _layout(home: Path) -> tuple[Path, Path]:
    """The generation tree a real install has, with ``current`` on the new build."""

    generations = home / ".local" / "share" / "lop" / "generations"
    images = []
    for name in (OLD, NEW):
        image = generations / name / "tools" / "local-operator" / "bin" / "Local Operator"
        image.parent.mkdir(parents=True, exist_ok=True)
        # A symlink to this interpreter: the image path is a real executable AND
        # carries the generation in ``ps``'s first field, as the branded one does.
        image.symlink_to(sys.executable)
        images.append(image)
    current = home / ".local" / "share" / "lop" / "current"
    current.parent.mkdir(parents=True, exist_ok=True)
    current.symlink_to(generations / NEW)
    return images[0], images[1]


def _stub_tree(root: Path) -> Path:
    """A shadow ``local_operator`` package whose connector module is the stub."""

    package = root / "local_operator" / "team_bridge" / "connector"
    package.mkdir(parents=True, exist_ok=True)
    for parent in (root / "local_operator", root / "local_operator" / "team_bridge"):
        (parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "__main__.py").write_text(_STUB_MAIN)
    return root


def _connector_pythonpath() -> str:
    """The worktree AND its site-packages, because argv[0] is not the interpreter.

    A process launched with ``argv[0]`` set to a generation image makes CPython
    derive its prefix from that path — the effect ``update``'s daemon shim
    documents: there is no ``pyvenv.cfg`` there, so the venv is invisible and its
    site-packages never reach ``sys.path``. Naming them explicitly is what lets a
    fake generation image and the real dependency tree coexist in one process.
    """

    return os.pathsep.join([str(ROOT), sysconfig.get_paths()["purelib"]])


def _env(home: Path, store: Path, pythonpath: str) -> dict[str, str]:
    """A built environment: nothing inherited, so no CMUX_*/live-store leak."""

    return {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "TERM": "xterm-256color",
        "LOCAL_OPERATOR_CONFIG_DIR": str(store),
        "PYTHONPATH": pythonpath,
    }


def _start_holder(
    tmp_path: Path, home: Path, image: Path, store: Path, *, named_store: Path | None = None
) -> subprocess.Popen[str]:
    """A process holding ``store``'s lock, whose argv names ``image``'s build.

    ``named_store`` is the store its ``--config-dir`` NAMES while it holds
    ``store``'s lock file — the shape proof 3's second half exists for. The lock it
    takes always comes from ``store`` (``HOLDER_LOCK_STORE``), because the point of
    that shape is a process holding one file while claiming another.
    """

    stub = _stub_tree(tmp_path / "stub")
    neutral = tmp_path / "neutral"
    neutral.mkdir(exist_ok=True)
    named = named_store or store
    process = subprocess.Popen(
        [str(image), "-m", "local_operator.team_bridge.connector", "--config-dir", str(named)],
        executable=sys.executable,
        cwd=str(neutral),
        env={**_env(home, store, str(stub)), "HOLDER_LOCK_STORE": str(store)},
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdout is not None
    line = process.stdout.readline()
    assert line.strip() == f"held {process.pid}", line or "<no handshake>"
    return process


def _start_connector(
    tmp_path: Path, home: Path, image: Path, store: Path, *, without_reap: bool = False
) -> subprocess.Popen[str]:
    """The REAL entry point, with ``image`` as its argv[0] (its own generation)."""

    if without_reap:
        argv = [str(image), "-c", _WITHOUT_REAP, "--config-dir", str(store), "--base-url", HUB]
    else:
        argv = [
            str(image),
            "-m",
            "local_operator.team_bridge.connector",
            "--config-dir",
            str(store),
            "--base-url",
            HUB,
        ]
    return subprocess.Popen(
        argv,
        executable=sys.executable,
        cwd=str(ROOT),
        env=_env(home, store, _connector_pythonpath()),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _enroll(store: Path) -> None:
    """Commit an owner, so the connector reaches the LOCK instead of idling."""

    Store(store).commit_enrollment(
        OWNER, OwnerReceipt("bridge_consent_v1", "bep_1"), cos_session_id="s"
    )


def _stop(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _lock_is_free(store: Path) -> bool:
    """A non-blocking acquire says whether anyone holds the lock right now."""

    probe = ConnectorLock(store, deadline_s=0)
    try:
        probe.acquire()
    except Exception:  # noqa: BLE001 — busy or unopenable both mean "not free"
        return False
    probe.release()
    return True


def _wait(predicate: Callable[[], bool], timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="flock and POSIX signals")
def test_the_incident_replayed_before_and_after(tmp_path: Path) -> None:
    """THE REPRO. Same stale holder, same store, twice: without the reap, then with.

    BEFORE (the code as it was): the connector is refused, exits 75, and the stale
    holder still owns the lock — the state this machine was in for seven days.
    AFTER (this change): the holder gets SIGTERM, dies, and the connector takes the
    lock and starts serving.
    """

    home = tmp_path / "home"
    old_image, new_image = _layout(home)
    store = tmp_path / "config"
    _enroll(store)

    # -- BEFORE: the lock path with the reap disabled, i.e. the pre-change code --
    holder = _start_holder(tmp_path, home, old_image, store)
    before = None
    try:
        before = _start_connector(tmp_path, home, new_image, store, without_reap=True)
        code = before.wait(timeout=120)
        _, before_err = before.communicate()
        assert code == 75, before_err[-2000:]
        assert holder.poll() is None, "the disabled reap signalled the holder"
        assert not _lock_is_free(store), "the stale holder lost the lock on its own"
    finally:
        _stop(before)
        _stop(holder)

    # -- AFTER: the shipped entry point, on a fresh stale holder ------------------
    holder = _start_holder(tmp_path, home, old_image, store)
    after = None
    try:
        after = _start_connector(tmp_path, home, new_image, store)
        assert _wait(lambda: holder.poll() is not None, 60), "the stale holder was never stopped"
        assert holder.returncode in (-signal.SIGTERM, -signal.SIGKILL), holder.returncode
        assert _wait(lambda: not _lock_is_free(store), 60), "the connector never took the lock"
        assert after.poll() is None, "the connector exited after reaping"
    finally:
        _stop(after)
        _stop(holder)


@pytest.mark.skipif(sys.platform == "win32", reason="flock and POSIX signals")
def test_a_live_current_generation_holder_is_never_signalled(tmp_path: Path) -> None:
    """A running connector on the CURRENT build is not a stale one, whatever it does.

    Its argv names the new generation, so the connector's own staleness question
    answers False and the attempt refuses. The holder must be alive and
    unsignalled afterwards — the cell that fails if the tri-state answer is ever
    read as "not provably stale means kill".
    """

    home = tmp_path / "home"
    _old_image, new_image = _layout(home)
    store = tmp_path / "config"
    _enroll(store)

    holder = _start_holder(tmp_path, home, new_image, store)
    connector = None
    try:
        connector = _start_connector(tmp_path, home, new_image, store)
        code = connector.wait(timeout=120)
        assert code == 75, connector.communicate()[1][-2000:]
        assert holder.poll() is None, "a live current-generation connector was signalled"
    finally:
        _stop(connector)
        _stop(holder)


@pytest.mark.skipif(sys.platform == "win32", reason="flock and POSIX signals")
def test_a_holder_serving_another_store_is_left_alone(tmp_path: Path) -> None:
    """It holds OUR lock file but its argv says another store: not ours to stop.

    ``--config-dir`` is the second half of proof 3, and the half that keeps a
    second installation's connector — a QA root, another profile — from being
    signalled because it happened to open this file.
    """

    home = tmp_path / "home"
    old_image, new_image = _layout(home)
    store_a = tmp_path / "config-a"
    store_b = tmp_path / "config-b"
    _enroll(store_a)
    _enroll(store_b)

    holder = _start_holder(tmp_path, home, old_image, store_a, named_store=store_b)
    connector = None
    try:
        connector = _start_connector(tmp_path, home, new_image, store_a)
        code = connector.wait(timeout=120)
        assert code == 75, connector.communicate()[1][-2000:]
        assert holder.poll() is None, "a holder naming another store was signalled"
    finally:
        _stop(connector)
        _stop(holder)


@pytest.mark.skipif(sys.platform == "win32", reason="flock and POSIX signals")
def test_another_stores_holder_does_not_stop_this_connector(tmp_path: Path) -> None:
    """The singleton is per store: B's holder holds B's lock, and A starts at once.

    The cell that would fail if the reaper looked at a lock that is not the
    caller's — and the reason a reap can never be triggered by another store's
    process at all, rather than by a proof that happens to refuse.
    """

    home = tmp_path / "home"
    _old_image, new_image = _layout(home)
    store_a = tmp_path / "config-a"
    store_b = tmp_path / "config-b"
    _enroll(store_a)
    _enroll(store_b)

    holder = _start_holder(tmp_path, home, new_image, store_b)
    connector = None
    try:
        connector = _start_connector(tmp_path, home, new_image, store_a)
        assert _wait(lambda: not _lock_is_free(store_a), 60), "A's connector never started"
        assert holder.poll() is None, "another store's holder was signalled"
        assert connector.poll() is None
    finally:
        _stop(connector)
        _stop(holder)
