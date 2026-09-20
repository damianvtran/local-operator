"""The cross-platform probe battery has to be able to GO RED, and to be right.

`scripts/xplat_probe.py` is the instrument this PR adds two CI legs for, and
round 1 found it mis-reporting in both directions:

- it could not reach a green state at all: `static.posix_attributes` FAILed on
  macOS, Linux and Windows, so both new legs failed by construction and the
  "instrument" would either ship permanently red or be deleted (reviewer B, A1).
  Six of its "fatal" hits were scanner artefacts in shapes this codebase
  genuinely guards -- a platform-constant early return
  (`if not _UID_IS_MEANINGFUL: return 0`), a capability probe
  (`getattr(signal, "SIGUSR1", None)`), and a launchd-only helper;
- it reported support where the CLI said support could not be VERIFIED, read an
  installer's progress line as its failure, failed on modules that deliberately
  refuse off POSIX, and decoded child output with the locale codec (A4-A6, QA
  Q1/Q4).

An instrument that cannot fail is not evidence, so every assertion below that
pins a *guard* is paired with one that pins the same scan still reporting an
unguarded use, and with the real Windows artifact's failure strings.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import xplat_probe, xplat_report

REPO = Path(__file__).resolve().parents[3]

#: The sites reviewer B verified as UNREACHABLE off POSIX but still reported as
#: fatal, one file each: the audit's own platform-constant guard
#: (`secrets/protocol.py`), a capability probe (`session/runtime/process.py`),
#: and the four launchd-only `_domain()` helpers. A scan that cannot read these
#: is a scan nobody will keep.
GUARDED_SITES = (
    "local_operator/secrets/protocol.py",
    "local_operator/session/runtime/process.py",
    "local_operator/secrets/keys.py",
    "local_operator/secrets/peer.py",
    "local_operator/update.py",
    "local_operator/browser_bridge/daemon.py",
    "local_operator/browser_bridge/install.py",
    "local_operator/mobile/install.py",
    "local_operator/tunnels/install.py",
    "local_operator/wakes/install.py",
)


@pytest.fixture(autouse=True)
def _restore_budget() -> Iterator[None]:
    """`BUDGET` is read by `run()`, so a test must not leave it spent.

    Defensive about the attribute existing at all, so that this fixture does not
    turn "the fix is absent" into a setup error for every test in the file: run
    against the pre-fix probe, each test still fails for its OWN reason, which is
    what makes the teeth legible.
    """
    budget = getattr(xplat_probe, "BUDGET", None)
    saved = None if budget is None else (budget.seconds, budget.deadline)
    try:
        yield
    finally:
        if budget is not None and saved is not None:
            budget.seconds, budget.deadline = saved


def _scan(source: str) -> list[tuple[int, str, bool, bool]]:
    return xplat_probe._scan_posix_uses(ast.parse(source))


# --------------------------------------------------------------------------- #
# The scanner: the guarded shapes this codebase actually uses
# --------------------------------------------------------------------------- #


def test_a_platform_constant_early_return_guards_the_rest_of_the_function() -> None:
    """`secrets/protocol.py`'s `_UID_IS_MEANINGFUL` shape, verbatim in form.

    Mutation that must fail this: drop the discovered-name rule from
    `guarded_by_test` -- `os.getuid` comes back `(fatal=True, guarded=False)`,
    which is the FAIL the Windows runner reported for the audit's own fix.
    """
    source = (
        "import os\n"
        '_UID_IS_MEANINGFUL = os.name == "posix"\n'
        "def _uid_token() -> int:\n"
        "    if not _UID_IS_MEANINGFUL:\n"
        "        return 0\n"
        "    return os.getuid()\n"
    )
    hits = _scan(source)
    assert hits, "the scan found no os.getuid at all; the case is stale"
    assert all(guarded for _line, _pattern, _fatal, guarded in hits), hits
    assert not any(fatal and not guarded for _l, _p, fatal, guarded in hits)


def test_a_chain_of_platform_constants_guards_through_the_fixpoint() -> None:
    """`secrets/peer.py`'s shape: `SUPPORTED = _IS_DARWIN or _IS_LINUX`.

    The two constants it is derived from are themselves platform tests, so the
    chain has to be followed rather than only the first link.
    """
    source = (
        "import os, sys\n"
        '_IS_DARWIN = sys.platform == "darwin"\n'
        '_IS_LINUX = sys.platform.startswith("linux")\n'
        "_SUPPORTED = _IS_DARWIN or _IS_LINUX\n"
        "def token() -> int:\n"
        "    if not _SUPPORTED:\n"
        "        return 0\n"
        "    return os.geteuid()\n"
    )
    assert all(guarded for _l, _p, _f, guarded in _scan(source))


def test_a_capability_probe_constant_guards_the_call_it_wraps() -> None:
    """`session/runtime/process.py`'s shape, which no term list can see.

    `debug_stacks = getattr(signal, "SIGUSR1", None)` and then
    `if <unrelated env test> and debug_stacks is not None:` -- the test text
    mentions no platform at all, so the guard is only visible through the name
    the module computed from the platform.
    """
    source = (
        "import os, signal\n"
        'debug_stacks = getattr(signal, "SIGUSR1", None)\n'
        "def install(loop, handle) -> None:\n"
        '    if os.environ.get("LOP_RUNTIME_DEBUG_STACKS") == "1" and debug_stacks is not None:\n'
        "        loop.add_signal_handler(debug_stacks, handle)\n"
    )
    hits = _scan(source)
    assert hits, "the scan found no add_signal_handler at all; the case is stale"
    assert all(guarded for _l, _p, _f, guarded in hits), hits


def test_a_capability_probe_about_an_unrelated_attribute_is_not_a_guard() -> None:
    """The boundary of the capability rule, in the direction that matters.

    `getattr(obj, "cr_frame", None)` is a capability probe, but not a PLATFORM
    one -- this package uses that spelling constantly for asyncio internals. If
    any `getattr(..., None)` counted, the scan would go quiet about real POSIX
    calls sitting under an unrelated attribute test.
    """
    source = (
        "import os\n"
        'frame = getattr(object(), "cr_frame", None)\n'
        "def kill_group(pid: int) -> None:\n"
        "    if frame is not None:\n"
        "        os.killpg(pid)\n"
    )
    assert all(not guarded for _l, _p, _f, guarded in _scan(source))


def test_the_scan_still_reports_an_unguarded_posix_call_as_fatal() -> None:
    """The instrument can go red. This is the half that makes it evidence."""
    hits = _scan("import os\ndef uid() -> int:\n    return os.getuid()\n")
    assert (3, "os.getuid", True, False) in hits, hits


def test_an_unguarded_os_kill_pid_zero_is_fatal_even_inside_a_try() -> None:
    """`os.kill(pid, 0)` TERMINATES the process on Windows and raises nothing.

    A surrounding `try` reading as protection is the exact shape that ships, so
    the scan must not accept it.
    """
    hits = _scan(
        "import os\n"
        "def alive(pid: int) -> bool:\n"
        "    try:\n"
        "        os.kill(pid, 0)\n"
        "    except OSError:\n"
        "        return False\n"
        "    return True\n"
    )
    assert (4, "os.kill(pid, 0)", True, False) in hits, hits


def test_the_named_guarded_sites_are_read_as_guarded() -> None:
    """A1's invariant, on the modules review round 1 named.

    The whole-package reading is the battery's own CI reading (both legs run it);
    this pins the sites that made it red, cheaply, so a guard this PR adds cannot
    be silently re-reported as a defect.
    """
    fatal: list[str] = []
    for rel in GUARDED_SITES:
        path = REPO / rel
        assert path.is_file(), f"{rel} is gone; this case is stale"
        for line, pattern, is_fatal, guarded in _scan(path.read_text(encoding="utf-8")):
            if is_fatal and not guarded:
                fatal.append(f"{rel}:{line}  {pattern}")
    assert fatal == [], (
        "the scan reports these POSIX-only uses as unguarded fatals, which is what "
        f"made both probe legs fail by construction: {fatal}"
    )


def test_an_undecodable_byte_in_child_output_does_not_fail_the_probe() -> None:
    """QA Q4, the parent half: `run()` decoded child output with the locale codec.

    On the Windows runner that turned a working surface into
    `config.roundtrip FAIL: er maps to <undefined>` -- cp1252's own decode-error
    text, in an artifact whose every line is read as a finding. 0x81 is invalid
    UTF-8 and undefined in cp1252, so a codec that is not told otherwise raises
    here on either platform.
    """
    proc = xplat_probe.run(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'err \\x81 here')"],
        dict(os.environ),
    )
    assert proc.returncode == 0, proc.stderr
    assert "\ufffd" in proc.stdout


def test_every_child_is_told_to_use_utf8(tmp_path: Path) -> None:
    """QA Q4, the child half: `tui.boot` FAILed inside the child's own cp1252.py.

    The logo's block characters went to a pipe, and a Python child writing to a
    pipe encodes with the LOCALE codec -- cp1252 on the Windows runner, where
    U+2584 has no encoding at all, so the probe reported a UnicodeEncodeError
    traceback as a TUI failure. Pinning the child's codec is not a convenience
    avoided: it is what keeps the reading about lop rather than about the
    console. `PYTHONUNBUFFERED` is the second half of the same class -- a
    file-backed child is block-buffered, so its last words before a crash would
    still be sitting in its buffer when the probe looked.
    """
    env = xplat_probe.isolated_env(tmp_path)
    assert env["PYTHONIOENCODING"] == "utf-8"
    assert env["PYTHONUNBUFFERED"] == "1"
    proc = xplat_probe.run([sys.executable, "-c", "print('\u2584')"], env)
    assert proc.returncode == 0 and "\u2584" in proc.stdout


# --------------------------------------------------------------------------- #
# The budget: the battery returns a matrix before the job's ceiling
# --------------------------------------------------------------------------- #


def test_an_unbounded_budget_does_not_shorten_a_probe() -> None:
    assert xplat_probe.Budget(0.0).clamp(600.0) == 600.0
    assert xplat_probe.Budget(0.0).window(60.0) == 60.0
    assert xplat_probe.Budget(0.0).spent() is False


def test_a_budget_shorter_than_a_probe_shortens_it_rather_than_killing_the_job() -> None:
    budget = xplat_probe.Budget(30.0)
    assert budget.clamp(600.0) <= 30.0
    assert budget.window(60.0) <= 30.0
    assert budget.note() == ""  # nothing has been spent yet


def test_a_spent_budget_refuses_to_start_a_child() -> None:
    """`run()` raises before `subprocess.run`, so no probe reports a fake timeout.

    The failure this prevents: the budget runs out in the middle of a probe, the
    clamped timeout fires immediately, and the artifact says the SURFACE timed
    out -- a fabricated defect in a surface that was never asked anything.
    """
    xplat_probe.BUDGET.start(0.001)
    time.sleep(0.01)
    assert xplat_probe.BUDGET.spent() is True
    assert "budget was spent" in xplat_probe.BUDGET.note()
    with pytest.raises(xplat_probe.BudgetSpent):
        xplat_probe.run([sys.executable, "-c", "print('never runs')"], {})


def test_an_exhausted_budget_marks_the_probes_it_never_ran_as_not_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The end of the run must read "not run", never "failed"."""
    monkeypatch.setattr(xplat_probe, "_sandbox_root", lambda: tmp_path)
    monkeypatch.setattr(
        xplat_probe, "PROBES", [lambda env: xplat_probe.Result("fake.one", "PASS", "ok")]
    )
    assert (
        xplat_probe.main(["--json", str(tmp_path / "m.json"), "--budget", "0.000001", "--keep"])
        == 0
    )
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert payload["counts"] == {"SKIP": 1}
    assert "not run" in payload["results"][0]["detail"]


# --------------------------------------------------------------------------- #
# The artifact: written as it goes, so a killed run is still evidence
# --------------------------------------------------------------------------- #


def test_a_battery_that_dies_mid_run_still_leaves_the_probes_it_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reviewer B, A2: the upload step is justified by a file that did not exist.

    `if: always()` uploads the artifact "even when the battery dies", but the
    JSON was written once, after the loop -- so a run killed at the job ceiling
    (or by a crash out of the loop) uploaded nothing at all. The write now
    happens after every probe; this test kills the run in the middle and reads
    the file.
    """

    def second(env: dict[str, str]) -> xplat_probe.Result:
        raise SystemExit("the job ceiling, in one line")

    monkeypatch.setattr(xplat_probe, "_sandbox_root", lambda: tmp_path)
    monkeypatch.setattr(
        xplat_probe,
        "PROBES",
        [lambda env: xplat_probe.Result("fake.first", "PASS", "ok"), second],
    )
    target = tmp_path / "m.json"
    with pytest.raises(SystemExit):
        xplat_probe.main(["--json", str(target), "--keep"])
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert [result["name"] for result in payload["results"]] == ["fake.first"]
    assert payload["counts"] == {"PASS": 1}


def test_a_json_run_writes_the_artifact_even_with_no_results(tmp_path: Path) -> None:
    """`_write_matrix` is the one writer, and it describes what it is given."""
    target = tmp_path / "empty.json"
    xplat_probe._write_matrix(target, [])
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["results"] == [] and payload["counts"] == {}
    assert payload["host"]["python"] == sys.version.split()[0]


# --------------------------------------------------------------------------- #
# Long-lived children: output to a FILE, not to an undrained pipe
# --------------------------------------------------------------------------- #


def test_a_chatty_long_lived_child_does_not_deadlock_on_its_own_output(
    tmp_path: Path,
) -> None:
    """The pipe nobody drains is what made three Windows failures unexplainable.

    `lop serve` and `lop mobile serve` log on the way to binding. With
    `stdout=PIPE` and no reader, a child that fills the 64 KiB pipe buffer BLOCKS
    there -- which is a plausible cause of the `serve.health` /
    `serve.double_bind` / `mobile.daemon_serve` timeouts in run 35405383805, and
    is why the artifact said "no response in 60s" and nothing else. 400 KiB is
    well past that buffer.
    """
    env = xplat_probe.isolated_env(tmp_path)
    child = xplat_probe._spawn_child(
        [sys.executable, "-c", "print('x' * 400_000); print('last-word')"],
        env,
        log=xplat_probe._child_log(env, "chatty"),
    )
    try:
        assert child.proc.wait(timeout=60) == 0, "the child blocked instead of exiting"
        assert "last-word" in child.output()
    finally:
        child.stop()


def test_a_long_lived_childs_tail_travels_in_the_result(tmp_path: Path) -> None:
    env = xplat_probe.isolated_env(tmp_path)
    child = xplat_probe._spawn_child(
        [sys.executable, "-c", "print('why-it-died')"], env, log=xplat_probe._child_log(env, "t")
    )
    try:
        child.proc.wait(timeout=60)
        extra = child.extra(rc=child.proc.returncode)
        assert extra["child_log"].endswith("t.log")
        assert "why-it-died" in extra["child_output"]
    finally:
        child.stop()


def test_leaving_the_child_context_reaps_the_child(tmp_path: Path) -> None:
    env = xplat_probe.isolated_env(tmp_path)
    log = xplat_probe._child_log(env, "reaped")
    sleeping = [sys.executable, "-c", "import time; time.sleep(60)"]
    with xplat_probe._spawn_child(sleeping, env, log=log) as child:
        time.sleep(0.5)
        assert child.proc.poll() is None
    assert child.proc.poll() is not None, "the child outlived its `with` block"


# --------------------------------------------------------------------------- #
# Honesty of the readings
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("status_text", "state"),
    [
        # macOS and the Linux container, under this battery's own redirected
        # HOME: the CLI answers, and says it CANNOT TELL. Reading that as
        # "supervisor present" is the reported-success-while-doing-nothing class
        # this battery exists to catch, inside the battery (reviewer B, A5).
        (
            "supervisor:  cannot be verified for this store\n"
            "             (this store is outside the real home; the user manager cannot\n"
            "             supervise it)\n"
            "scheduled:   0 (0 armed)\n",
            "unverified",
        ),
        ("supervisor:  not installed\n             (run 'lop wake install')\n", "absent"),
        ("supervisor:  systemd (unit local-operator-wakes.service)\n", "reported"),
    ],
)
def test_wake_status_names_the_state_it_actually_read(
    status_text: str, state: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=status_text, stderr="")

    monkeypatch.setattr(xplat_probe, "run", fake_run)
    result = xplat_probe.probe_wake_status({})
    assert result.extra["supervisor_state"] == state
    assert result.extra["supervisor_available"] is (state == "reported")
    assert result.status == ("PASS" if state == "reported" else "WARN")


def test_the_mobile_install_failure_detail_is_the_end_of_the_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An installer prints progress first, so the FIRST line is a success step.

    The Linux artifact's `mobile.install` FAIL detail was "generated a new portal
    password (a 0600 file ...)" -- a step that worked -- while the actual failure
    was four lines further down (reviewer B, A6).

    The transcript here deliberately does NOT contain "did not come up
    healthy": that case has its own, better evidence now (the daemon's own log)
    and is covered below, so what stays pinned here is the generic failure.
    """
    transcript = (
        "generated a new portal password (a 0600 file (/tmp/x/password))\n"
        "  built the web bundle\n"
        "  wrote the unit\n"
        "could not register the daemon: schtasks exited 1: ERROR: Access is denied.\n"
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout=transcript, stderr="")

    monkeypatch.setattr(xplat_probe, "run", fake_run)
    # `LOP_XPLAT_ALLOW_KEYCHAIN` defeats the darwin skip (which exists because the
    # REAL installer writes to the login keychain); nothing is installed here.
    result = xplat_probe.probe_mobile_install({"LOP_XPLAT_ALLOW_KEYCHAIN": "1"})
    assert result.status == "FAIL"
    # The detail ENDS at the failure rather than starting at the progress: the
    # old `_first_line` opened the reader's evidence with a step that worked.
    assert result.detail.splitlines()[-1].startswith("could not register the daemon")


def test_a_daemon_that_never_came_up_reports_the_daemons_own_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The installer answers with a POINTER, and the artifact outlives the runner.

    The Windows run of 35457478927 read exactly this: `daemon did not come up
    healthy; see C:\\Users\\RUNNER~1\\...\\logs\\mobile.log`. The log is on a
    runner that is already destroyed by the time anybody reads the artifact, so
    the one fact that would have explained the failure -- the daemon's own last
    words -- was the one fact not carried. Reported as PASS=20/FAIL=1 with no way
    to tell a crash from a slow start.

    The registry PATH dump is dropped rather than truncated: it is a single line
    of several hundred characters on Windows, and keeping it pushed the actual
    failure out of the tail and into the bit-bucket.
    """
    config = tmp_path / "config"
    (config / "logs").mkdir(parents=True)
    (config / "logs" / "mobile.log").write_text(
        "2026-09-19 12:50:04,816 - INFO - Retrieved System PATH from registry: "
        + "C:\\Program Files\\x;" * 60
        + "\n2026-09-19 12:50:05,000 - INFO - starting server at http://127.0.0.1:4098\n"
        "RuntimeError: no mobile password set. Run `lop mobile install`\n",
        encoding="utf-8",
    )
    transcript = (
        "  built the web bundle\n"
        "  registered the scheduled task (Local Operator Mobile)\n"
        "  started the task\n"
        "daemon did not come up healthy; see /tmp/x/logs/mobile.log\n"
    )

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout=transcript, stderr="")

    monkeypatch.setattr(xplat_probe, "run", fake_run)
    result = xplat_probe.probe_mobile_install(
        {"LOP_XPLAT_ALLOW_KEYCHAIN": "1", "LOCAL_OPERATOR_CONFIG_DIR": str(config)}
    )

    assert result.status == "FAIL"
    assert "no mobile password set" in result.detail, result.detail
    assert "Retrieved System PATH" not in result.detail, "the PATH dump is not evidence"
    assert "no mobile password set" in str(result.extra["daemon_log"])


def test_a_daemon_that_never_came_up_without_a_log_still_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The absent-log case is a sentence, not a crash and not an empty detail.

    A probe that dies while explaining a failure is worse than one that reports
    it plainly, and the isolated stores the battery uses legitimately have no
    daemon log to read until the installer writes one.
    """
    transcript = "  started the task\ndaemon did not come up healthy; see /tmp/x/logs/mobile.log\n"

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=1, stdout=transcript, stderr="")

    monkeypatch.setattr(xplat_probe, "run", fake_run)
    result = xplat_probe.probe_mobile_install(
        {
            "LOP_XPLAT_ALLOW_KEYCHAIN": "1",
            "LOCAL_OPERATOR_CONFIG_DIR": str(tmp_path / "absent"),
        }
    )

    assert result.status == "FAIL"
    assert "never became healthy" in result.detail
    assert "no daemon log" in result.detail


def test_import_package_separates_a_refusal_from_a_breakage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The exact two shapes off the Windows artifact of run 35405383805.

    Five modules refused with a named platform reason (the deliberate design of
    `evaluation/adapters/supervisor.py`), one broke
    (`os.register_at_fork` does not exist on Windows). Only the second is a
    defect; both have to be listed (reviewer B, A4/Q1).
    """
    failed = {
        "local_operator.evaluation.adapters.supervisor": (
            "RuntimeError: evaluation adapter supervision requires POSIX process groups"
        ),
        "local_operator.evaluation.evidence.store": (
            "AttributeError: module 'os' has no attribute 'register_at_fork'"
        ),
        "local_operator.some.unguarded_module": "ModuleNotFoundError: No module named 'fcntl'",
    }

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {"total": 502, "failed": failed}
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(xplat_probe, "run", fake_run)
    result = xplat_probe.probe_import_package({})
    assert result.status == "FAIL"
    assert list(result.extra["refused"]) == ["local_operator.evaluation.adapters.supervisor"]
    assert set(result.extra["unexpected"]) == {
        "local_operator.evaluation.evidence.store",
        "local_operator.some.unguarded_module",
    }
    # Both halves are named in the detail, so the artifact does not hide a
    # refusal behind a pass or a breakage behind a refusal count.
    assert "refuse off POSIX" in result.detail and "unexpected" in result.detail


def test_import_package_passes_when_every_failure_is_a_named_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        payload = {
            "total": 6,
            "failed": {
                "local_operator.evaluation.runner.episode": (
                    "RuntimeError: evaluation adapter supervision requires POSIX process groups"
                )
            },
        }
        return subprocess.CompletedProcess(
            args=[], returncode=0, stdout=json.dumps(payload), stderr=""
        )

    monkeypatch.setattr(xplat_probe, "run", fake_run)
    result = xplat_probe.probe_import_package({})
    assert result.status == "PASS"
    assert "5/6 modules import" in result.detail


def test_the_daemon_surface_probe_claims_only_what_it_measures() -> None:
    """It runs `install --help`, which exists everywhere (reviewer B, A4).

    The old docstring asked "does this OS have a way to keep it running?" and
    answered the smaller question; a reader of a PASS is owed the smaller claim.
    """
    doc = xplat_probe.probe_daemon_supervisors.__doc__ or ""
    assert "install --help" in doc and "renders" in doc
    assert (
        "says nothing about" in doc
    ), "the docstring no longer disclaims the supervisor claim it does not measure"


# --------------------------------------------------------------------------- #
# The comparison report: one column per FILE
# --------------------------------------------------------------------------- #


def _payload(system: str, status: str) -> dict[str, object]:
    return {
        "host": {"system": system, "release": "1", "machine": "arm64", "python": "3.13.12"},
        "counts": {status: 1},
        "results": [{"name": "probe.one", "status": status, "detail": "", "extra": {}}],
    }


def test_two_runs_of_the_same_os_are_two_columns(tmp_path: Path) -> None:
    """Reviewer B, A7: keyed by host label, the later run REPLACED the earlier.

    A before/after pair from one host is the reading this tool exists for, and a
    re-run into the same directory used to make one of them vanish silently.
    """
    (tmp_path / "before.json").write_text(json.dumps(_payload("Darwin", "FAIL")), encoding="utf-8")
    (tmp_path / "after.json").write_text(json.dumps(_payload("Darwin", "PASS")), encoding="utf-8")
    runs = xplat_report.load(tmp_path)
    assert sorted(runs) == ["after", "before"]
    assert runs["before"][1].startswith("Darwin") and runs["after"][1].startswith("Darwin")


def test_the_report_names_the_file_every_column_came_from(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "macos.json").write_text(json.dumps(_payload("Darwin", "PASS")), encoding="utf-8")
    (tmp_path / "ubuntu.json").write_text(json.dumps(_payload("Linux", "FAIL")), encoding="utf-8")
    assert xplat_report.main([str(tmp_path)]) == 1, "a FAIL anywhere must fail the report"
    printed = capsys.readouterr().out
    for name in ("macos", "ubuntu"):
        assert name in printed
    assert "columns:" in printed and "Darwin" in printed and "Linux" in printed
