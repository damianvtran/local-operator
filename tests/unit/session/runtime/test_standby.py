"""The runtime standby: adopted when it is safe, refused when anything moved, invisible until then.

WHAT IS PINNED, AND HOW. ``session/runtime/standby.py`` exists because a cold
engage pays ~1.1 s of import CPU before any session work, which at this host's
load is most of a 3-4 s "starting…". Its value is speed; its RISK is serving a
session from an interpreter that no longer matches what a cold spawn would run.
So these tests pin the guards, each against a REAL standby process (its warm
stubbed to a no-op so a test takes seconds rather than the ~85 s a full warm
takes under fleet load):

* adoption hands over the spawn's exact environment, capture file and operator
  descriptor, and the adopted process becomes a census-visible runtime;
* a moved ``config.yml``, a moved loaded module, a moved generation, a differing
  interpreter or warm-sensitive environment each REFUSE, and the stale ones exit;
* one standby per root: a second standby for the same root exits immediately;
* before adoption the standby is in no session reader: not the census, not the
  record directory, not the host (boot record) directory;
* warming is opt-in per process, and ``launch._spawn_runtime`` falls back to the
  ordinary cold spawn whenever ``try_adopt`` answers ``None``.

No timing is asserted (AGENTS.md "Timing, flakes"): the saving is measured by
``scripts/bench_cold_engage.py`` and stated in the PR, never calibrated here.
"""

from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Iterator

import pytest

from local_operator.session.runtime import reclaim, standby
from local_operator.session.runtime.types import (
    HOST_RUN_DIRNAME,
    RUN_DIRNAME,
    RUNTIME_MODULE,
)

pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(socket, "send_fds"),
    reason="the standby needs SCM_RIGHTS; other platforms keep the cold spawn",
)


def test_the_module_words_are_the_same_length() -> None:
    # Adoption rewrites the module word IN PLACE, which cannot grow a string.
    assert len(standby.STANDBY_MODULE) == len(RUNTIME_MODULE)
    assert len(b"[standby]") == len(b"[session]")


def test_the_standby_argv_is_not_a_runtime_to_the_census() -> None:
    row = (
        "4242 1 01:00:00 00:00:01 Local Operator [standby] id=-------- "
        f"-P -m {standby.STANDBY_MODULE}"
    )
    assert reclaim.parse_process_row(row) is None
    adopted = row.replace("[standby]", "[session]").replace(standby.STANDBY_MODULE, RUNTIME_MODULE)
    assert reclaim.parse_process_row(adopted) is not None


def test_warming_is_off_unless_a_host_enables_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spawned: list[Any] = []
    monkeypatch.setattr(standby, "_WARMING", [False])
    monkeypatch.setattr(standby, "_spawn_standby", lambda *a: spawned.append(a))
    standby.ensure_warm(tmp_path, sys.executable)
    assert spawned == []
    # And a runtime child never warms, even in a process that enabled it.
    monkeypatch.setattr(standby, "_WARMING", [True])
    monkeypatch.setenv("LOP_MOBILE_CHILD_RESUME", "abc")
    standby.ensure_warm(tmp_path, sys.executable)
    assert spawned == []
    monkeypatch.delenv("LOP_MOBILE_CHILD_RESUME")
    standby.ensure_warm(tmp_path, sys.executable)
    assert len(spawned) == 1


def test_the_disable_switch_turns_off_adoption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv(standby.DISABLE_ENV, "1")
    assert standby.try_adopt(tmp_path, sys.executable, {}, tmp_path / "c", None) is None


def test_no_standby_means_the_ordinary_cold_spawn(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``try_adopt`` answering None must leave ``_spawn_runtime`` exactly as before."""
    from local_operator.session.runtime import launch as launch_module

    calls: list[Any] = []
    monkeypatch.setattr(standby, "try_adopt", lambda *a, **k: calls.append(a) or None)

    class _Child:
        pid = 99999

        def poll(self) -> None:
            return None

    monkeypatch.setattr(launch_module.subprocess, "Popen", lambda argv, **kw: _Child())
    process = launch_module._spawn_runtime("sess-cold01", str(tmp_path), defer_materialise=True)
    capture = getattr(process, "lop_capture_path", None)
    if capture is not None:
        capture.unlink(missing_ok=True)
    assert isinstance(process, _Child)
    assert len(calls) == 1
    root, _interpreter, env, _capture, fd = calls[0]
    # The request carries the COLD child's environment and its handoff fd.
    assert env["LOP_MOBILE_CHILD_RESUME"] == "sess-cold01"
    assert isinstance(fd, int)


# ---------------------------------------------------------------------------
# A real standby process, with the warm stubbed
# ---------------------------------------------------------------------------

#: Runs the real ``standby._serve`` with ``_warm`` replaced, and — on adoption —
#: ``_become_runtime`` replaced by a probe that writes what the adopted process
#: actually holds (env, cwd, the capability bytes from the passed descriptor,
#: its argv as ``ps`` sees it) and exits. Everything between accept and
#: become-runtime is the production code.
_STANDBY_DRIVER = textwrap.dedent("""
    import json, os, subprocess, sys
    from pathlib import Path
    from local_operator.session.runtime import standby
    standby._warm = lambda: None
    # The band is the host scheduler's business, not this test's: at load 100+ a
    # background-band process is starved for minutes, which is weather, not logic.
    standby._background_priority = lambda on: None
    out = Path(os.environ["STANDBY_PROBE_OUT"])
    def become(fd):
        cap = b""
        if fd is not None:
            cap = os.read(fd, 64)
            os.close(fd)
        ps = subprocess.run(["ps", "-ww", "-o", "command=", "-p", str(os.getpid())],
                            capture_output=True, text=True).stdout.strip()
        out.write_text(json.dumps({
            "pid": os.getpid(), "cap": cap.hex(), "cwd": os.getcwd(), "ps": ps,
            "env": {k: v for k, v in os.environ.items() if k.startswith(("LOP_", "PROBE_"))},
        }))
        return 0
    standby._become_runtime = become
    sys.exit(standby._serve(Path(sys.argv[1])))
    """)


@pytest.fixture
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    config = tmp_path / "cfg"
    config.mkdir()
    (config / "config.yml").write_text("values: {}\n", encoding="utf-8")
    monkeypatch.delenv(standby.DISABLE_ENV, raising=False)
    return config


@pytest.fixture
def started(root: Path, tmp_path: Path) -> Iterator[list[subprocess.Popen[bytes]]]:
    procs: list[subprocess.Popen[bytes]] = []
    yield procs
    for proc in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)


def _base_env(root: Path, out: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("CMUX_", "LOP_", "HERDR_"))}
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(root)
    env["STANDBY_PROBE_OUT"] = str(out)
    return env


def _start(
    root: Path, out: Path, started: list[subprocess.Popen[bytes]], label: str = "standby"
) -> subprocess.Popen[bytes]:
    # Spawned through the branded-free interpreter with an argv the rename can
    # find: the SAME words ``_spawn_standby`` gives a real standby.
    argv0 = f"Local Operator [{label}] id=--------"
    proc = subprocess.Popen(  # noqa: S603
        [argv0, "-P", "-c", _STANDBY_DRIVER, str(root), "-m", standby.STANDBY_MODULE],
        executable=sys.executable,
        env=_base_env(root, out),
        stdin=subprocess.DEVNULL,
    )
    started.append(proc)
    sock = standby.socket_path(root, create=False)
    deadline = time.monotonic() + 60
    while not sock.exists():
        assert proc.poll() is None, f"standby exited early rc={proc.returncode}"
        assert time.monotonic() < deadline, "standby never listened"
        time.sleep(0.02)
    return proc


def _request(root: Path, out: Path, **overrides: Any) -> tuple[Any, int, int]:
    """Adopt with a fresh socketpair as the 'operator handoff'. Returns (result, w, r)."""
    reader, writer = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    env = _base_env(root, out)
    env.update(
        {
            "LOP_MOBILE_CHILD_RESUME": "abcdef123456",
            "LOP_MOBILE_CHILD_CWD": str(root),
            "PROBE_ONLY_IN_REQUESTER": "yes",
        }
    )
    env.update(overrides.pop("env", {}))
    capture = root.parent / "capture.log"
    capture.touch()
    result = standby.try_adopt(
        root,
        overrides.pop("interpreter", sys.executable),
        env,
        capture,
        reader.fileno(),
    )
    reader.close()
    if result is not None:
        # ``_spawn_runtime``'s order: deliver the capability only to an
        # adoption. On a refusal nobody holds the other end, as after a cold
        # spawn that never happened.
        writer.sendall(b"\x07" * 32)
    writer.close()
    return result, 0, 0


def _wait_json(path: Path) -> dict[str, Any]:
    deadline = time.monotonic() + 30
    while not path.exists() or not path.read_text():
        assert time.monotonic() < deadline, "adopted process never reported"
        time.sleep(0.02)
    return json.loads(path.read_text())


def test_adoption_hands_over_the_whole_spawn(
    root: Path, tmp_path: Path, started: list[subprocess.Popen[bytes]]
) -> None:
    out = tmp_path / "probe.json"
    proc = _start(root, out, started)
    # Invisible before adoption: no census row, no record, no boot record.
    assert proc.pid not in {p.pid for p in reclaim.runtime_processes()}
    assert not (root / RUN_DIRNAME).exists() or not any((root / RUN_DIRNAME).iterdir())
    assert not (root / HOST_RUN_DIRNAME).exists()
    assert not (root / "sessions").exists()

    adopted, _, _ = _request(root, out)
    assert adopted is not None and adopted.pid == proc.pid
    report = _wait_json(out)
    # The capability travelled on the passed descriptor, byte for byte.
    assert report["cap"] == ("07" * 32)
    # The requester's environment replaced the warmer's, WHOLE.
    assert report["env"]["LOP_MOBILE_CHILD_RESUME"] == "abcdef123456"
    assert report["env"]["PROBE_ONLY_IN_REQUESTER"] == "yes"
    # ...and ps now names a runtime of that session, which the census counts.
    assert f"-m {RUNTIME_MODULE}" in report["ps"]
    assert "[session] id=abcdef12" in report["ps"]
    assert reclaim.parse_process_row(f"1 1 01:00 00:01 {report['ps']}") is not None
    assert proc.wait(timeout=30) == 0
    # The lock went with it, so the next warmer may start a replacement.
    assert not standby._lock_held(root / standby.STANDBY_DIRNAME / "lock")


def test_a_second_standby_for_the_same_root_exits(
    root: Path, tmp_path: Path, started: list[subprocess.Popen[bytes]]
) -> None:
    out = tmp_path / "probe.json"
    _start(root, out, started)
    second = subprocess.Popen(  # noqa: S603
        [sys.executable, "-P", "-c", _STANDBY_DRIVER, str(root)],
        env=_base_env(root, out),
        stdin=subprocess.DEVNULL,
    )
    started.append(second)
    assert second.wait(timeout=30) == 0


@pytest.mark.parametrize("what", ["config", "tree"])
def test_a_moved_input_refuses_and_retires_the_standby(
    what: str, root: Path, tmp_path: Path, started: list[subprocess.Popen[bytes]]
) -> None:
    out = tmp_path / "probe.json"
    proc = _start(root, out, started)
    target = Path(standby.__file__)
    st = target.stat()
    if what == "config":
        # The product's own writer shape: a replace, i.e. a new inode.
        fresh = root / "config.yml.new"
        fresh.write_text("values: {hosting: other}\n", encoding="utf-8")
        os.replace(fresh, root / "config.yml")
    else:
        # A loaded module of THIS tree moved: bump the mtime of a file the
        # standby imported (``standby.py`` itself), then put it back.
        os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    try:
        adopted, _, _ = _request(root, out)
        assert adopted is None
        assert proc.wait(timeout=30) == 0  # stale: it leaves, a replacement is warmed later
        assert not out.exists()
    finally:
        if what == "tree":
            os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns))


def test_a_differing_requester_is_declined_and_the_standby_keeps_waiting(
    root: Path, tmp_path: Path, started: list[subprocess.Popen[bytes]]
) -> None:
    out = tmp_path / "probe.json"
    proc = _start(root, out, started)
    adopted, _, _ = _request(root, out, env={"LANG": "xx_XX.WEIRD-8"})
    assert adopted is None
    other_venv = tmp_path / "other" / "bin" / "python3"
    adopted, _, _ = _request(root, out, interpreter=str(other_venv))
    assert adopted is None
    # Declined, not retired: the same standby still serves a matching requester.
    assert proc.poll() is None
    adopted, _, _ = _request(root, out)
    assert adopted is not None and adopted.pid == proc.pid
