#!/usr/bin/env python3
"""Prove a generation-pinned relay moves under real Linux user systemd.

Run ONLY on a disposable Linux runner. The fixture starts a private
``systemd --user`` manager, installs the base and PR builds into a previously
absent generation layout, and drives the real ``update --refresh-daemons``
child from each generation. It is not a unit test and never targets a paired
peer or an existing user manager.

Safety contract:

* Refuse a pre-existing stable root, relay unit, or launcher. Nothing owned by a
  person is stopped, rewritten, removed, or read for credentials.
* Keep the real HOME because ``systemd_unit_is_addressable`` correctly rejects
  a redirected home for a user-manager call. Config, identity, logs and state
  live in an isolated LOCAL_OPERATOR_CONFIG_DIR with no peer records. The relay
  listens on loopback only; only dependency installation uses external network.
* Preflight the private manager before generation installation or service
  dispatch. Cleanup addresses only the unit and exact manager PID created here.
* A PATH wrapper fails exactly the first ``ps`` probe and delegates every later
  call to real ``ps``. This makes the old process-reading gap reproducible while
  leaving the fixed post-restart success check fully capable of passing.

The JSON receipt separates installed source, live process generation and the
relay's advertised build. It contains no environment dump, control key, private
key, network secret, or credential value.
"""

from __future__ import annotations

import argparse
import json
import os
import pwd
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from typing import Any

from local_operator import procname, update
from local_operator.network import relay, store


class ProbeBlocked(RuntimeError):
    """A prerequisite is unavailable; never convert this to a skip or a pass."""


class ProbeFailure(RuntimeError):
    """The fixture or a product assertion failed."""


def _run(
    argv: list[str],
    *,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    timeout: float = 30.0,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProbeFailure(f"could not execute {argv[0]}: {exc}") from exc


def _require(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
    result = _run(argv, **kwargs)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[-1800:]
        raise ProbeFailure(f"{argv[0]} exited {result.returncode}: {detail}")
    return result


def _sha(path: Path) -> str:
    return _require(["git", "-C", str(path), "rev-parse", "HEAD"]).stdout.strip()


def _version(path: Path) -> str:
    with (path / "pyproject.toml").open("rb") as handle:
        value = tomllib.load(handle)["project"]["version"]
    if not isinstance(value, str) or not value:
        raise ProbeFailure(f"no project.version in {path / 'pyproject.toml'}")
    return value


def _source_identity(generation: Path) -> dict[str, str]:
    install_root = update._generation_install_root(generation)  # noqa: SLF001
    return {
        "generation": str(generation),
        "install_root": str(install_root),
        "source_ref": update.source_ref(install_root),
        "version": update.generation_version(generation),
    }


def _observation() -> dict[str, Any]:
    """Read loaded-process and advertised-build identities independently."""
    pid = relay._managed_service_pid()  # noqa: SLF001
    generation = update.generation_of_process(pid) if pid is not None else None
    record = store.find_own_relay()
    answering = relay.health(timeout=0.5) is not None
    # PeerRecord also carries a control key. Copy only its identity/build
    # fields; never serialize the record itself.
    return {
        "pid": pid or 0,
        "loaded_generation": str(generation or ""),
        "loaded_stale": update.generation_staleness_of_process(pid) if pid is not None else None,
        "advertised_record_pid": record.pid if record is not None else 0,
        "advertised_version": record.version if record is not None else "",
        "advertised_source_ref": record.source_ref if record is not None else "",
        "advertised_install_root": record.install_root if record is not None else "",
        "answering": answering,
    }


def _assert_identity(
    observed: dict[str, Any], generation: Path, sha: str, label: str, *, expected_stale: bool
) -> None:
    loaded = Path(str(observed["loaded_generation"]))
    if loaded.resolve() != generation.resolve():
        raise ProbeFailure(f"{label}: loaded process is {loaded}, expected {generation}")
    if observed["loaded_stale"] is not expected_stale:
        raise ProbeFailure(
            f"{label}: loaded_stale={observed['loaded_stale']!r}, expected {expected_stale}"
        )
    if observed["advertised_record_pid"] != observed["pid"]:
        raise ProbeFailure(f"{label}: systemd and advertised PIDs differ: {observed}")
    if observed["advertised_source_ref"] != sha:
        raise ProbeFailure(
            f"{label}: advertised source {observed['advertised_source_ref']!r}, expected {sha}"
        )
    install_root = str(update._generation_install_root(generation))  # noqa: SLF001
    if observed["advertised_install_root"] != install_root:
        raise ProbeFailure(f"{label}: advertised install root is not the expected generation")
    if not observed["answering"]:
        raise ProbeFailure(f"{label}: relay did not answer its real loopback health probe")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _systemd_binary() -> str | None:
    """The user-manager executable on systemd's PATH or standard library paths."""
    found = shutil.which("systemd")
    if found is not None:
        return found
    for value in ("/lib/systemd/systemd", "/usr/lib/systemd/systemd"):
        candidate = Path(value)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _log_tail(log: Any) -> str:
    """Bounded manager diagnostics; this log contains no application credentials."""
    log.flush()
    return Path(log.name).read_text(encoding="utf-8", errors="replace")[-2000:].strip()


def _start_private_manager(scratch: Path, log: Any) -> subprocess.Popen[str]:
    """Start and prove the private manager BEFORE any install/service work."""
    xdg = scratch / "xdg"
    xdg.mkdir(mode=0o700)
    os.chmod(xdg, 0o700)
    env = os.environ.copy()
    env["XDG_RUNTIME_DIR"] = str(xdg)
    env["SYSTEMD_LOG_LEVEL"] = "debug"
    env["SYSTEMD_LOG_TARGET"] = "console"
    env.pop("DBUS_SESSION_BUS_ADDRESS", None)
    # Set this in the parent BEFORE Popen too: failure cleanup must still talk
    # to the private manager, never fall back to the runner's real user bus.
    os.environ["XDG_RUNTIME_DIR"] = str(xdg)
    os.environ.pop("DBUS_SESSION_BUS_ADDRESS", None)
    executable = _systemd_binary()
    if executable is None:
        raise ProbeBlocked("systemd user-manager executable is unavailable")
    proc = subprocess.Popen(  # noqa: S603 — fixed executable, disposable manager
        [executable, "--user", "--log-level=debug"],
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            detail = _log_tail(log) or "no manager log"
            version = _run([executable, "--version"], timeout=3.0)
            status = _run(["systemctl", "--user", "is-system-running"], env=env, timeout=3.0)
            diagnostics = (
                f"uid={os.getuid()} exe={executable} xdg={xdg} "
                f"version={(version.stdout or version.stderr).strip()[:300]} "
                f"systemctl_status={(status.stdout + status.stderr).strip()[:500]} "
                f"manager_log={detail}"
            )
            raise ProbeBlocked(
                f"private systemd --user exited with {proc.returncode}: {diagnostics}"
            )
        ready = _run(["systemctl", "--user", "daemon-reload"], env=env, timeout=3.0)
        if ready.returncode == 0:
            return proc
        time.sleep(0.1)
    proc.terminate()
    proc.wait(timeout=5)
    detail = _log_tail(log) or "no manager log"
    raise ProbeBlocked(f"private systemd --user did not answer within 15s: {detail}")


def _install_source(source: Path, sha: str, version: str, ref: str) -> Path:
    generation = update.install_into_generation(
        source,
        version=version,
        commit=sha,
        ref=ref,
        origin="native-relay-update-probe",
        note="disposable Linux user-systemd base/fixed control",
    )
    identity = _source_identity(generation)
    if identity["source_ref"] != sha or identity["version"] != version:
        raise ProbeFailure(f"installed source identity mismatch for {source}: {identity}")
    return generation


def _write_pinned_unit(unit: Path, generation: Path, port: int) -> None:
    rendered = relay.render_systemd(port=port)
    current_image = procname.supervised_image()
    if current_image not in rendered:
        raise ProbeFailure("could not identify the rendered unit image for the historical fixture")
    old_image = update._generation_install_root(generation) / "bin" / "python3"  # noqa: SLF001
    if not old_image.is_file():
        raise ProbeFailure(f"generation interpreter is missing: {old_image}")
    pinned = rendered.replace(current_image, str(old_image), 1)
    unit.parent.mkdir(parents=True, exist_ok=True)
    unit.write_text(pinned, encoding="utf-8")
    _require(["systemctl", "--user", "daemon-reload"])


def _start_and_wait(generation: Path, sha: str) -> dict[str, Any]:
    _require(["systemctl", "--user", "start", relay.SYSTEMD_UNIT])
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        observed = _observation()
        pid = observed["pid"]
        if pid and observed["advertised_source_ref"] == sha and observed["answering"]:
            _assert_identity(observed, generation, sha, "service start", expected_stale=True)
            return observed
        time.sleep(0.1)
    raise ProbeFailure(f"relay did not become ready on its advertised source {sha}")


def _stop_and_wait() -> None:
    _run(["systemctl", "--user", "stop", relay.SYSTEMD_UNIT])
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        if relay._managed_service_pid() is None:  # noqa: SLF001
            return
        time.sleep(0.1)
    raise ProbeFailure("the fixture relay did not stop within 15s")


def _refresh(generation: Path, scratch: Path, label: str) -> dict[str, Any]:
    """Run the real hidden update child with a ONE-CALL process-probe fault."""
    real_ps = shutil.which("ps")
    if real_ps is None:
        raise ProbeBlocked("ps is required for both the fault and the success proof")
    wrapper_dir = scratch / "fault-bin"
    wrapper_dir.mkdir(exist_ok=True)
    real_ps_link = wrapper_dir / "real-ps"
    if not real_ps_link.exists():
        real_ps_link.symlink_to(real_ps)
    marker = scratch / f"fail-next-ps-{label}"
    marker.touch()
    wrapper = wrapper_dir / "ps"
    wrapper.write_text(
        "#!/bin/sh\n"
        "if [ -f \"$LOP_NATIVE_PS_FAIL_ONCE\" ]; then\n"
        "  rm -f \"$LOP_NATIVE_PS_FAIL_ONCE\"\n"
        "  echo 'intentional one-shot ps failure' >&2\n"
        "  exit 1\n"
        "fi\n"
        "exec \"$LOP_NATIVE_REAL_PS\" \"$@\"\n",
        encoding="utf-8",
    )
    wrapper.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{wrapper_dir}:{env.get('PATH', '')}"
    env["LOP_NATIVE_PS_FAIL_ONCE"] = str(marker)
    env["LOP_NATIVE_REAL_PS"] = str(real_ps_link)
    interpreter = update._generation_install_root(generation) / "bin" / "python3"  # noqa: SLF001
    result = _run(
        [str(interpreter), "-m", "local_operator.cli", "update", "--refresh-daemons"],
        env=env,
        cwd=scratch,
        timeout=90.0,
    )
    if marker.exists():
        raise ProbeFailure(f"{label}: the one-shot ps fault was never consumed")
    lines = [
        line
        for line in (result.stdout + "\n" + result.stderr).splitlines()
        if line.startswith("daemon_report:") or "network relay" in line
    ]
    return {
        "returncode": result.returncode,
        "receipt_lines": lines[-20:],
        "post_fault_ps": "real ps delegated; mandatory post-restart check could pass",
    }


def _cleanup(
    manager: subprocess.Popen[str] | None,
    unit: Path,
    stable: Path,
    launchers: tuple[Path, ...],
    repo: Path,
    base_source: Path | None,
    *,
    owns_artifacts: bool,
    owns_base: bool,
) -> list[str]:
    notes: list[str] = []
    if manager is not None:
        _run(["systemctl", "--user", "disable", "--now", relay.SYSTEMD_UNIT], timeout=10.0)
        _run(["systemctl", "--user", "halt"], timeout=10.0)
        try:
            manager.wait(timeout=10)
        except subprocess.TimeoutExpired:
            manager.terminate()
            try:
                manager.wait(timeout=5)
            except subprocess.TimeoutExpired:
                manager.kill()
                manager.wait(timeout=5)
        if manager.poll() is None:  # pragma: no cover — defensive exact-PID backstop
            os.kill(manager.pid, signal.SIGKILL)
            manager.wait(timeout=5)
        notes.append(f"stopped exact private manager pid {manager.pid}")
    if owns_artifacts:
        if unit.exists():
            unit.unlink()
            notes.append("removed only fixture unit")
        for path in launchers:
            if path.exists() or path.is_symlink():
                path.unlink()
        if stable.exists():
            shutil.rmtree(stable)
            notes.append("removed fixture generation layout")
    if owns_base and base_source is not None:
        _run(["git", "-C", str(repo), "worktree", "remove", "--force", str(base_source)])
        notes.append("removed owned base worktree")
    return notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base commit for the negative control")
    parser.add_argument("--json", required=True, type=Path, help="receipt destination")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    result: dict[str, Any] = {
        "schema": 1,
        "status": "FAIL",
        "base": {},
        "fixed": {},
        "cleanup": [],
    }
    manager: subprocess.Popen[str] | None = None
    base_source: Path | None = None
    owns_artifacts = False
    owns_base = False
    stable = update.stable_root()
    unit = relay.systemd_path()
    launchers = (update._local_bin_dir() / "lop", update._local_bin_dir() / "lo")  # noqa: SLF001
    scratch = Path("")
    exit_code = 1
    try:
        if not sys.platform.startswith("linux"):
            raise ProbeBlocked("real Linux user systemd is required")
        missing = [
            name
            for name in ("systemctl", "ps", "git")
            if shutil.which(name) is None
        ]
        if _systemd_binary() is None:
            missing.append("systemd")
        if missing:
            raise ProbeBlocked(f"missing required native tools: {', '.join(missing)}")
        real_home = Path(pwd.getpwuid(os.getuid()).pw_dir).resolve()
        if Path(os.environ.get("HOME", "")).resolve() != real_home:
            raise ProbeBlocked("HOME must be the runner's real home for manager addressability")
        occupied = [
            str(path)
            for path in (stable, unit, *launchers)
            if path.exists() or path.is_symlink()
        ]
        if occupied:
            raise ProbeBlocked(
                f"refusing pre-existing relay/install artifacts: {', '.join(occupied)}"
            )
        runner_temp = Path(os.environ.get("RUNNER_TEMP") or tempfile.gettempdir())
        scratch = Path(tempfile.mkdtemp(prefix="lop-native-relay-", dir=runner_temp))
        config = scratch / "config"
        config.mkdir()
        os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(config)
        os.environ["UV_CACHE_DIR"] = str(scratch / "uv-cache")
        os.environ["GIT_CONFIG_SYSTEM"] = "/dev/null"
        os.environ["GIT_TERMINAL_PROMPT"] = "0"
        port = _free_port()
        (config / "config.yml").write_text(
            f"network:\n  listen_address: 127.0.0.1\n  port: {port}\n",
            encoding="utf-8",
        )
        if store.networks_dir(config).exists() and any(store.networks_dir(config).glob("*.json")):
            raise ProbeFailure("the isolated config unexpectedly carries a peer network")

        head_source = args.repo.resolve()
        head_sha = _sha(head_source)
        base_sha = _require(["git", "-C", str(head_source), "rev-parse", args.base]).stdout.strip()
        if head_sha == base_sha:
            raise ProbeFailure("base and head must differ for the product-code control")
        base_source = scratch / "base-source"
        _require(
            [
                "git",
                "-C",
                str(head_source),
                "worktree",
                "add",
                "--detach",
                str(base_source),
                base_sha,
            ]
        )
        owns_base = True
        head_version = _version(head_source)
        if _version(base_source) != head_version:
            raise ProbeFailure(
                "control requires identical package versions; only product code may differ"
            )

        with (scratch / "private-systemd.log").open("w", encoding="utf-8") as log:
            manager = _start_private_manager(scratch, log)
        result["preflight"] = {
            "private_manager": "ready before generation install or service dispatch",
            "real_home_for_manager_only": True,
            "config": "isolated, synthetic identity, no peer records",
            "relay_listen": "127.0.0.1 only",
            "external_dialing": "none by the relay; dependency installation only",
            "port": port,
        }
        _require(
            ["systemctl", "--user", "set-environment", f"LOCAL_OPERATOR_CONFIG_DIR={config}"]
        )
        owns_artifacts = True

        base_generation = _install_source(base_source, base_sha, head_version, args.base)
        head_generation = _install_source(head_source, head_sha, head_version, "PR-head")
        if update.current_generation() != head_generation:
            raise ProbeFailure("current pointer is not on the fixed generation after installation")
        result["installed"] = {
            "base": _source_identity(base_generation),
            "fixed": _source_identity(head_generation),
        }

        # Negative control: identical fixture, old refresh code. The one failed
        # process read used to produce `current` and leave the old relay live.
        _write_pinned_unit(unit, base_generation, port)
        before = _start_and_wait(base_generation, base_sha)
        receipt = _refresh(base_generation, scratch, "base")
        after = _observation()
        _assert_identity(
            after, base_generation, base_sha, "base negative control", expected_stale=True
        )
        if after["pid"] != before["pid"] or receipt["returncode"] != 0:
            raise ProbeFailure(
                "base negative control did not retain the old success receipt: " f"{receipt}"
            )
        if not any('"already"' in line for line in receipt["receipt_lines"]):
            raise ProbeFailure(f"base did not emit the old current/already receipt: {receipt}")
        result["base"] = {
            "before": before,
            "refresh": receipt,
            "after": after,
            "verdict": (
                "reproduced old current receipt while process and advertisement stayed on base"
            ),
        }
        _stop_and_wait()

        # Fixed case: only the generation carrying the new product code changes.
        _write_pinned_unit(unit, base_generation, port)
        before = _start_and_wait(base_generation, base_sha)
        receipt = _refresh(head_generation, scratch, "fixed")
        after = _observation()
        _assert_identity(
            after, head_generation, head_sha, "fixed native relay", expected_stale=False
        )
        if after["pid"] == before["pid"] or receipt["returncode"] != 0:
            raise ProbeFailure(f"fixed relay did not restart successfully: {receipt}")
        if not any('"refreshed"' in line for line in receipt["receipt_lines"]):
            raise ProbeFailure(f"fixed relay did not emit a refreshed receipt: {receipt}")
        result["fixed"] = {
            "before": before,
            "refresh": receipt,
            "after": after,
            "verdict": (
                "unit drift drove restart; real post-fault ps proved loaded and "
                "advertised fixed build"
            ),
        }
        result["status"] = "PASS"
        exit_code = 0
    except ProbeBlocked as exc:
        result["status"] = "BLOCKED"
        result["reason"] = str(exc)
        exit_code = 2
    except Exception as exc:  # noqa: BLE001 — always emit the native receipt and cleanup
        result["status"] = "FAIL"
        result["reason"] = f"{type(exc).__name__}: {exc}"
        exit_code = 1
    finally:
        try:
            result["cleanup"] = _cleanup(
                manager,
                unit,
                stable,
                launchers,
                args.repo,
                base_source,
                owns_artifacts=owns_artifacts,
                owns_base=owns_base,
            )
        except Exception as exc:  # noqa: BLE001 — cleanup failure is itself a red result
            result["status"] = "FAIL"
            result["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            exit_code = 1
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(
            f"native relay update: {result['status']} — "
            f"{result.get('reason', 'see JSON receipt')}"
        )
        print(f"receipt: {args.json}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
