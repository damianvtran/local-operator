"""Serial, pinned Linux evidence, not a fix or a broader-suite retry.

Both refs use ONE editable install at the same checkout path: their dependency
metadata must match before either payload runs. A cap here describes this new
experiment only; it cannot establish what killed the historical gw2. Two passes
leave historical shard interaction unknown. Never escalate to a suite here.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import runpy
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = "87a3741ceb062979989d134acd18d3320c372313"
HEAD = "9db425ea9b3c1f6c6cdfe236644d88c7b0e94ebd"
SELECTOR = (
    "tests/unit/test_ci_scope_scoping.py::"
    "test_a_repo_python_file_a_test_names_is_always_selected"
)


def capture(argv: list[str], cwd: Path | None = None) -> str:
    return subprocess.check_output(argv, cwd=cwd, text=True, timeout=10)


def cgroup_snapshot() -> dict[str, str]:
    """Record the actual process cgroup and ancestors, not a guessed root limit.

    Counters are shared with other processes in that cgroup. A delta is context,
    NOT attribution of an OOM to the selector. Missing v2 files stay explicit.
    """
    result = {}
    membership = Path("/proc/self/cgroup")
    if not membership.exists():
        return {"unavailable": "no /proc/self/cgroup (non-Linux control)"}
    result["membership"] = membership.read_text()
    for line in result["membership"].splitlines():
        if line.startswith("0::"):
            directory = Path("/sys/fs/cgroup") / line[3:].lstrip("/")
            while directory.is_relative_to("/sys/fs/cgroup"):
                names = ("memory.events", "memory.events.local", "memory.current", "memory.max")
                for name in names:
                    path = directory / name
                    try:
                        result[str(path)] = path.read_text()
                    except OSError as exc:
                        result[str(path)] = type(exc).__name__
                if directory == Path("/sys/fs/cgroup"):
                    break
                directory = directory.parent
    return result


def isolated_env(directory: Path) -> dict[str, str]:
    home = directory / "home"
    temporary = directory / "tmp"
    home.mkdir()
    temporary.mkdir()
    # Allowlist instead of inheriting tokens or operator configuration. No git,
    # gh, glab or Docker auth lookup may reach the synthetic home's keychain.
    env = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "SYSTEMROOT")
        if key in os.environ
    }
    env.update(
        HOME=str(home),
        USERPROFILE=str(home),
        TMPDIR=str(temporary),
        LOCAL_OPERATOR_CONFIG_DIR=str(home / ".local-operator"),
        LOCAL_OPERATOR_SHARD_STALL_SECONDS="240",
        CI="true",
        LOCAL_OPERATOR_SHARD_MEMORY_DIR=str(directory / "memory-dumps"),
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_TERMINAL_PROMPT="0",
        GIT_ASKPASS="/bin/false",
        GH_CONFIG_DIR=str(home / "gh"),
        GLAB_CONFIG_DIR=str(home / "glab"),
        GH_TOKEN="diagnostic-no-auth",
        GITLAB_TOKEN="diagnostic-no-auth",
        DOCKER_CONFIG=str(home / "docker"),
        COVERAGE_FILE=str(directory / ".coverage"),
    )
    return env


def limit_logs() -> None:
    # Full streams, never tail/truncate them silently. If a runaway writer hits
    # 32 MiB per file, SIGXFSZ makes this experiment fail with explicit evidence.
    resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024,) * 2)


def observe_bounded(receipt: Path, script: str, arguments: list[str]) -> None:
    """Run the unchanged bounded script, observing its child's raw wait status.

    Only the wrapper's Popen is instrumented. The pytest interpreter and its
    xdist worker load no shim and retain the original report-only watchdog.
    A raw wait receipt distinguishes explicit exit 137 from delivered SIGKILL.
    """
    original = subprocess.Popen

    class ObservedChild(original):
        def __init__(self, *args, **kwargs):
            # The guard also invokes ps; only the owned, new-session payload
            # is the child whose status is evidence (never the ps probe's 0).
            self.is_payload = kwargs.get("start_new_session") is True
            super().__init__(*args, **kwargs)

        def _receipt(self) -> None:
            if self.is_payload and self.returncode is not None:
                receipt.write_text(
                    json.dumps(
                        {
                            "pid": self.pid,
                            "pgid": self.pid,
                            "raw_returncode": self.returncode,
                            "signal": -self.returncode if self.returncode < 0 else None,
                            "exit_code": self.returncode if self.returncode >= 0 else None,
                        },
                        indent=2,
                    )
                    + "\n"
                )

        def poll(self):
            status = super().poll()
            self._receipt()
            return status

        def wait(self, *args, **kwargs):
            status = super().wait(*args, **kwargs)
            self._receipt()
            return status

    subprocess.Popen = ObservedChild
    sys.argv = [script, *arguments]
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        subprocess.Popen = original


def run_case(
    repo: Path,
    directory: Path,
    payload: list[str],
    *,
    timeout: float = 600,
    grace: float = 5,
    memory_mb: int = 2048,
) -> dict:
    directory.mkdir(parents=True)
    env = isolated_env(directory)
    command = [
        sys.executable,
        str(repo / "scripts/run_bounded.py"),
        "--timeout",
        str(timeout),
        "--grace",
        str(grace),
        "--memory-mb",
        str(memory_mb),
        "--",
        *payload,
    ]
    # Execute this exact bounded argv with only a parent-side raw wait observer.
    invocation = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--observe-bounded",
        str(directory / "child-status.json"),
        *command[1:],
    ]
    before = cgroup_snapshot()
    started = time.monotonic()
    observations = []
    outer_fired = False
    with (
        (directory / "stdout.log").open("w") as stdout,
        (directory / "stderr.log").open("w") as stderr,
        (directory / "processes.jsonl").open("w") as samples,
    ):
        proc = subprocess.Popen(
            invocation,
            cwd=repo,
            env=env,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
            preexec_fn=limit_logs,
        )
        try:
            while proc.poll() is None:
                # Only PID/PPID/PGID/RSS: argv and environment can contain secrets.
                raw = capture(["ps", "-axo", "pid=,ppid=,pgid=,rss="])
                rows = [tuple(map(int, line.split())) for line in raw.splitlines()]
                owned = {proc.pid}
                for _ in range(len(rows)):
                    found = {pid for pid, ppid, _, _ in rows if ppid in owned}
                    if found <= owned:
                        break
                    owned |= found
                selected = [row for row in rows if row[0] in owned]
                observations = selected or observations
                samples.write(
                    json.dumps(
                        {"wall": time.monotonic() - started, "pid_ppid_pgid_rss_kib": selected}
                    )
                    + "\n"
                )
                samples.flush()
                disk_bytes = 0
                for path in directory.rglob("*"):
                    try:
                        if path.is_file():
                            disk_bytes += path.stat().st_size
                    except FileNotFoundError:
                        # Pytest owns transient files under this TMPDIR.
                        continue
                if (
                    disk_bytes > 256 * 1024 * 1024
                    or time.monotonic() - started > timeout + grace + 10
                ):
                    outer_fired = True
                    break
                time.sleep(1)
        finally:
            # Signal ONLY the wrapper we spawned. It forwards TERM to its own
            # separately-sessioned child group and reaps on every exit path.
            if proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
                try:
                    proc.wait(timeout=grace + 3)
                except subprocess.TimeoutExpired:
                    # Last-resort owned-group teardown, only if a recorded PID
                    # still belongs to its sampled PGID. Never kill by name.
                    for pid, _, pgid, _ in observations:
                        try:
                            if os.getpgid(pid) == pgid:
                                os.killpg(pgid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                    proc.kill()
                    proc.wait(timeout=3)
    stderr_text = (directory / "stderr.log").read_text()
    result = {
        "command": command,
        "observer_invocation": invocation,
        "wrapper_pid": proc.pid,
        "child_status": (
            json.loads((directory / "child-status.json").read_text())
            if (directory / "child-status.json").exists()
            else None
        ),
        "wrapper_returncode": proc.returncode,
        "wrapper_signal": -proc.returncode if proc.returncode < 0 else None,
        # run_bounded normalizes child signals to 128+signal and prints the
        # reason. Preserve those raw diagnostic lines rather than guessing from
        # 137 (also a child's explicit exit or this experiment's memory cap).
        "bounded_diagnostics": [
            line for line in stderr_text.splitlines() if line.startswith("[run_bounded]")
        ],
        "outer_watchdog_fired": outer_fired,
        "wall_seconds": time.monotonic() - started,
        "cgroup_before": before,
        "cgroup_after": cgroup_snapshot(),
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    repo, out = args.repo.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if platform.system() != "Linux" or platform.python_version() != "3.12.15":
        raise RuntimeError("requires native Linux / Python 3.12.15; no substitute proof")
    os_release = Path("/etc/os-release").read_text()
    if 'VERSION_ID="26.04"' not in os_release:
        raise RuntimeError("requires Ubuntu 26.04")
    # Same path + byte-identical metadata makes one editable install valid for
    # both checkouts. No lockfile exists in this repo; ci.yml uses pip .[dev].
    for name in ("pyproject.toml",):
        base_metadata = capture(["git", "show", f"{BASE}:{name}"], repo)
        head_metadata = capture(["git", "show", f"{HEAD}:{name}"], repo)
        if base_metadata != head_metadata:
            raise RuntimeError(f"dependency metadata differs: {name}; stop setup")
    installed_source = capture(
        [
            sys.executable,
            "-c",
            "import local_operator; print(local_operator.__file__)",
        ],
        repo,
    ).strip()
    if Path(installed_source).resolve() != repo / "local_operator/__init__.py":
        raise RuntimeError("editable install points outside the target checkout")
    provenance = {
        "installed_source": installed_source,
        "python": sys.version,
        "linux": platform.uname()._asdict(),
        "os_release": os_release,
        "base": BASE,
        "head": HEAD,
        "pip_freeze": capture([sys.executable, "-m", "pip", "freeze"], repo),
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    def interrupted(signum, _frame):
        # Turn runner cancellation into stack unwinding through run_case's
        # owned-wrapper cleanup, rather than abandoning a new-session child.
        raise InterruptedError(f"diagnostic received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGQUIT):
        signal.signal(signum, interrupted)
    results = []
    for label, sha in (("base", BASE), ("head", HEAD)):
        subprocess.run(["git", "checkout", "--detach", sha], cwd=repo, check=True, timeout=30)
        actual = capture(["git", "rev-parse", "HEAD"], repo).strip()
        if actual != sha:
            raise RuntimeError("checkout identity mismatch")
        directory = out / label
        payload = [
            sys.executable,
            "-X",
            "faulthandler",
            "-m",
            "pytest",
            SELECTOR,
            "-n",
            "1",
            "-vv",
            "--cov",
            "--cov-fail-under=0",
            "--cov-report=",
            f"--junit-xml={directory / 'junit.xml'}",
        ]
        result = run_case(repo, directory, payload)
        result["sha"] = actual
        # Zero exit is not proof: require the one requested test to complete.
        try:
            cases = list(ET.parse(directory / "junit.xml").iter("testcase"))
            result["selector_pass"] = (
                len(cases) == 1
                and cases[0].get("name") == SELECTOR.split("::")[1]
                and not any(
                    cases[0].find(tag) is not None for tag in ("skipped", "failure", "error")
                )
            )
        except (OSError, ET.ParseError):
            result["selector_pass"] = False
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        results.append(result)
        print(
            f"{label} {sha}: rc={result['wrapper_returncode']} "
            f"wall={result['wall_seconds']:.1f}s",
            flush=True,
        )
    # Retain failures while still collecting the second, independent experiment.
    return int(
        any(
            r["wrapper_returncode"] != 0 or r["outer_watchdog_fired"] or not r["selector_pass"]
            for r in results
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--observe-bounded":
        observe_bounded(Path(sys.argv[2]), sys.argv[3], sys.argv[4:])
    else:
        sys.exit(main())
