"""Generic managed-process ownership: the lease, its checks, its scoped reaps.

Zero cost: every process spawned here is a tiny stdlib sleeper that writes one
status snapshot and waits; nothing touches a model, cloud, VM or credential.
The launchd case is Darwin-only and session-unique (its own label, booted out
and its plist unlinked in teardown -- 'no unscoped kills', nothing else
addressed).
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from local_operator import owned_process, procstate

# A tiny owner: writes one status snapshot in the real file's shape (the real
# owner writes through OwnerStatusChannel; this sleeper only proves the file
# contract), then sleeps far longer than any test waits.
_SLEEPER = textwrap.dedent("""
    import json, os, time
    path = os.environ["OWNER_STATUS_PATH"]
    payload = {
        "version": 1,
        "owner_id": os.environ["LOP_OWNED_OWNER_ID"],
        "phase": "running",
        "pid": os.getpid(),
        "birth": None,
        "backend": os.environ["LOP_OWNED_BACKEND"],
        "job_label": os.environ.get("LOP_OWNED_JOB_LABEL"),
        "updated_ms": 1,
        "outcome_path": None,
        "detail": None,
        "receipt": {},
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
    time.sleep(120)
    """)


def _spec(tmp_path: Path, backend: str, owner_id: str) -> owned_process.OwnedProcessSpec:
    paths = owned_process.owner_paths(tmp_path, owner_id)
    return owned_process.OwnedProcessSpec(
        argv=(sys.executable, "-c", _SLEEPER),
        cwd=tmp_path,
        env={"OWNER_STATUS_PATH": str(paths.status_path), "PATH": os.environ.get("PATH", "")},
        run_root=tmp_path,
        owner_id=owner_id,
        source_pin="pin-owner-test",
        backend=backend,
    )


def test_the_status_channel_never_raises(tmp_path: Path) -> None:
    # A directory in the status file's place: the write is swallowed, the
    # read reports unreadable -- status is a reading, never authority.
    blocker = tmp_path / "status-dir"
    blocker.mkdir()
    channel = owned_process.OwnerStatusChannel(
        blocker, owner_id="ep-x", backend="detached", job_label=None
    )
    channel.phase("running", detail="still not raising")
    assert owned_process.read_owner_status(blocker) is None


def test_forged_admission_is_refused_with_the_mismatch_named(tmp_path: Path) -> None:
    # The flag alone must not admit: no lease record, no owner.
    with pytest.raises(owned_process.OwnedProcessRefused, match="no readable lease"):
        owned_process.verify_owned_admission(
            run_root=tmp_path,
            owner_id="ep-forged",
            backend="detached",
            job_label=None,
            actual_source_pin="pin",
        )
    paths = owned_process.owner_paths(tmp_path, "ep-forged")
    paths.prepare()
    payload = {
        "owner_id": "ep-someone-else",
        "backend": "detached",
        "job_label": None,
        "pid": os.getpid(),
        "birth_scheme": procstate.birth_scheme(),
        "birth": procstate.birth_token(os.getpid()),
        "source_pin": "pin",
    }
    paths.lease_path.write_text(json.dumps(payload))
    with pytest.raises(owned_process.OwnedProcessRefused, match="ep-someone-else"):
        owned_process.verify_owned_admission(
            run_root=tmp_path,
            owner_id="ep-forged",
            backend="detached",
            job_label=None,
            actual_source_pin="pin",
        )
    payload["owner_id"] = "ep-forged"
    payload["source_pin"] = "pin-other"
    paths.lease_path.write_text(json.dumps(payload))
    with pytest.raises(owned_process.OwnedProcessRefused, match="pin-other"):
        owned_process.verify_owned_admission(
            run_root=tmp_path,
            owner_id="ep-forged",
            backend="detached",
            job_label=None,
            actual_source_pin="pin",
        )
    # Every recorded fact matches THIS process (it is the recorded holder of
    # its own pid): admitted, and the record is returned for provenance.
    payload["source_pin"] = "pin"
    paths.lease_path.write_text(json.dumps(payload))
    record = owned_process.verify_owned_admission(
        run_root=tmp_path,
        owner_id="ep-forged",
        backend="detached",
        job_label=None,
        actual_source_pin="pin",
    )
    assert record["owner_id"] == "ep-forged"


def test_secret_handoff_is_0600_read_once_and_unlinked(tmp_path: Path) -> None:
    path = tmp_path / "handoff.json"
    owned_process.write_secret_handoff(path, {"CANARY_NAME": "canary-handoff-77", "K2": "v2"})
    assert (path.stat().st_mode & 0o777) == 0o600
    assert owned_process.consume_secret_handoff(path) == {
        "CANARY_NAME": "canary-handoff-77",
        "K2": "v2",
    }
    assert not path.exists()
    # An absent handoff is "no handoff", not an error; a second read sees none.
    assert owned_process.consume_secret_handoff(path) == {}


@pytest.mark.asyncio
async def test_detached_roundtrip_status_detach_and_exact_reap(tmp_path: Path) -> None:
    spec = _spec(tmp_path, "detached", "ep-roundtrip")
    lease = owned_process.start_owned_process(spec, startup_timeout_s=10.0)
    killed = False
    try:
        assert procstate.pid_liveness(lease.pid) is True
        status = await lease.wait(10.0)
        assert status is not None and status.owner_id == "ep-roundtrip"
        assert status.phase == "running"
        assert status.backend == "detached"
        # Detaching is NOT a stop: the owner outlives the handle.
        lease.detach()
        assert procstate.pid_liveness(lease.pid) is True
        # The lease record and the status carry identity, never secret values.
        paths = owned_process.owner_paths(tmp_path, "ep-roundtrip")
        assert "canary-handoff" not in paths.lease_path.read_text()
        assert "canary-handoff" not in paths.status_path.read_text()
    finally:
        # Test-owned cleanup by EXACT pid (the process this test started).
        try:
            os.kill(lease.pid, signal.SIGTERM)
            killed = True
        except OSError:
            killed = True
    assert killed
    # A DEAD CHILD IS A ZOMBIE UNTIL REAPED, and ``pid_liveness`` (os.kill
    # 0) reads a zombie as alive -- so the test reaps its OWN child (the
    # detached backend spawns from THIS process) and only then expects the
    # pid to read dead.
    for _ in range(300):
        try:
            done, _status = os.waitpid(lease.pid, os.WNOHANG)
        except ChildProcessError:
            done = lease.pid
        if done == lease.pid:
            break
        time.sleep(0.02)
    assert procstate.pid_liveness(lease.pid) is False


@pytest.fixture()
def durable_root() -> Iterator[Path]:
    """A run root the launchd guard accepts: session-unique, under Caches.

    ``refuse_volatile_root`` refuses /tmp (correctly -- the OS may purge it),
    and pytest SANDBOXES ``Path.home()`` per test under its own tmp tree, so
    the base is the uid's PASSWD home (``launchd.real_home``, the same
    derivation the launchd identity test itself uses). The Caches directory
    is durable AND disposable; the fixture removes its own directory and
    touches nothing of the operator's live configuration.
    """
    from local_operator import launchd

    base_home = launchd.real_home()
    if base_home is None:  # pragma: no cover - a host without a passwd entry
        pytest.skip("no readable passwd home for a durable run root")
    base = base_home / "Library" / "Caches" / "lop-owned-process-tests"
    base.mkdir(parents=True, exist_ok=True)
    root = Path(tempfile.mkdtemp(prefix="ld-", dir=base))
    yield root
    shutil.rmtree(root, ignore_errors=True)
    try:
        base.rmdir()
    except OSError:
        pass


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="launchd is macOS-only")
def test_launchd_roundtrip_identity_and_scoped_cleanup(durable_root: Path) -> None:
    from local_operator import launchd

    tmp_path = durable_root
    owner_id = f"ep-ld-{os.getpid()}-{int(time.time())}"
    spec = _spec(tmp_path, "launchd", owner_id)
    lease = owned_process.start_owned_process(spec, startup_timeout_s=20.0)
    home = launchd.real_home()
    assert home is not None
    plist = home.joinpath("Library", "LaunchAgents", f"{lease.job_label}.plist")
    try:
        # launchd ITSELF answers the identity question: the job's pid is the
        # lease's pid -- not a flag, not a convention.
        assert (
            launchd.job_pid(
                label=lease.job_label or "", path=plist, run=owned_process.launchctl_runner
            )
            == lease.pid
        )
        paths = owned_process.owner_paths(tmp_path, owner_id)
        deadline = time.monotonic() + 15.0
        status = None
        while time.monotonic() < deadline:
            status = owned_process.read_owner_status(paths.status_path)
            if status is not None:
                break
            time.sleep(0.05)
        assert status is not None and status.owner_id == owner_id
        assert status.job_label == lease.job_label
    finally:
        # Scoped to the one label THIS test created.
        launchd.bootout_job(
            label=lease.job_label or "", path=plist, runner=owned_process.launchctl_runner
        )
        try:
            plist.unlink()
        except OSError:
            pass
    assert not plist.exists()
    assert (
        launchd.job_pid(label=lease.job_label or "", path=plist, run=owned_process.launchctl_runner)
        is None
    )
