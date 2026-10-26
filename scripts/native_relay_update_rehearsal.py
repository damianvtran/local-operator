#!/usr/bin/env python3
"""Non-privileged rehearsal for the delegated native relay-update fixture.

WHY THIS EXISTS. Three privileged runs were spent finding defects in the FIXTURE
rather than measuring the product: the historical unit image, the missing
``append:`` log parent, and an ``install_root`` assertion against a field the
relay never publishes. Each was invisible until a runner was spent. This script
drives the SAME probe functions (``_observation``, ``_assert_identity``,
``_start_and_wait``, ``_write_readiness_diagnostics``) with the systemd
supervisor faked, so that class of defect fails here in seconds.

WHAT IT PROVES:

* the real relay's published record satisfies the fixture's identity contract
  (a LIVE cell, unprivileged, loopback only);
* every assertion in ``_assert_identity`` both accepts a correct observation and
  REJECTS its own mutant — a rehearsal that cannot fail is not evidence;
* readiness returns when the relay answers, and a timeout raises while leaving
  the diagnostics artifact behind;
* the base negative control (stale and unchanged) and the fixed case (moved)
  are distinguishable by the same assertion set.

WHAT IT CANNOT COVER, and must never be read as covering: real systemd cgroup
delegation, ``append:`` handling by systemd itself, the private manager's own
unit setup, ``systemd-run`` privilege boundaries, or the real ``lop update``
child process. Those remain the privileged run's business.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import native_relay_update_probe as probe  # noqa: E402


class RehearsalFailure(RuntimeError):
    """A rehearsal cell found a defect; never converted into a pass."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RehearsalFailure(message)


def _observation(
    *,
    generation: Path,
    pid: int = 4321,
    stale: bool | None = False,
    source_ref: str = "a" * 40,
    install_root: str = "",
    listen_address: str = "127.0.0.1",
    answering: bool = True,
) -> dict[str, Any]:
    """A record shaped like the one a real relay publishes."""
    return {
        "pid": pid,
        "loaded_generation": str(generation),
        "loaded_stale": stale,
        "advertised_record_pid": pid,
        "advertised_version": "0.90.2",
        "advertised_source_ref": source_ref,
        "advertised_install_root": install_root,
        "advertised_listen_address": listen_address,
        "answering": answering,
    }


def _must_fail(label: str, call: Any, match: str) -> None:
    try:
        call()
    except probe.ProbeFailure as exc:
        _check(match in str(exc), f"{label}: failed for the wrong reason: {exc}")
        return
    raise RehearsalFailure(f"{label}: was accepted; the assertion does not discriminate")


def cell_identity_contract(root: Path) -> str:
    generation = root / "generations" / "head"
    sha = "a" * 40
    good = _observation(generation=generation, source_ref=sha)
    probe._assert_identity(good, generation, sha, "rehearsal", expected_stale=False)  # noqa: SLF001

    _must_fail(
        "loaded generation",
        lambda: probe._assert_identity(  # noqa: SLF001
            _observation(generation=root / "generations" / "other", source_ref=sha),
            generation,
            sha,
            "rehearsal",
            expected_stale=False,
        ),
        "loaded process",
    )
    _must_fail(
        "non-loopback bind",
        lambda: probe._assert_identity(  # noqa: SLF001
            _observation(generation=generation, source_ref=sha, listen_address="0.0.0.0"),
            generation,
            sha,
            "rehearsal",
            expected_stale=False,
        ),
        "not loopback",
    )
    _must_fail(
        "source ref",
        lambda: probe._assert_identity(  # noqa: SLF001
            _observation(generation=generation, source_ref="b" * 40),
            generation,
            sha,
            "rehearsal",
            expected_stale=False,
        ),
        "advertised source",
    )
    _must_fail(
        "pid mismatch",
        lambda: probe._assert_identity(  # noqa: SLF001
            {**_observation(generation=generation, source_ref=sha), "advertised_record_pid": 9},
            generation,
            sha,
            "rehearsal",
            expected_stale=False,
        ),
        "PIDs differ",
    )
    _must_fail(
        "not answering",
        lambda: probe._assert_identity(  # noqa: SLF001
            _observation(generation=generation, source_ref=sha, answering=False),
            generation,
            sha,
            "rehearsal",
            expected_stale=False,
        ),
        "did not answer",
    )
    # The published-root rule: empty (what every real relay sends) is accepted,
    # a wrong non-empty root is refused, the correct one is accepted.
    wrong_root = str(root / "somewhere" / "tools" / "local-operator")
    _must_fail(
        "wrong published root",
        lambda: probe._assert_identity(  # noqa: SLF001
            _observation(generation=generation, source_ref=sha, install_root=wrong_root),
            generation,
            sha,
            "rehearsal",
            expected_stale=False,
        ),
        "advertised install root",
    )
    right_root = str(probe.update._generation_install_root(generation))  # noqa: SLF001
    probe._assert_identity(  # noqa: SLF001
        _observation(generation=generation, source_ref=sha, install_root=right_root),
        generation,
        sha,
        "rehearsal",
        expected_stale=False,
    )
    return "identity contract accepts a real-shaped record and rejects every mutant"


def cell_negative_control_and_fixed(root: Path) -> str:
    generation = root / "generations" / "head"
    sha = "c" * 40
    # Base negative control: old code leaves the process on base, not moved.
    probe._assert_identity(  # noqa: SLF001
        _observation(generation=generation, source_ref=sha, stale=True),
        generation,
        sha,
        "base negative control",
        expected_stale=True,
    )
    _must_fail(
        "base did not stay stale",
        lambda: probe._assert_identity(  # noqa: SLF001
            _observation(generation=generation, source_ref=sha, stale=False),
            generation,
            sha,
            "base negative control",
            expected_stale=True,
        ),
        "loaded_stale",
    )
    # Fixed case: the move must be observed, not assumed.
    probe._assert_identity(  # noqa: SLF001
        _observation(generation=generation, source_ref=sha, stale=False),
        generation,
        sha,
        "fixed native relay",
        expected_stale=False,
    )
    _must_fail(
        "fixed did not move",
        lambda: probe._assert_identity(  # noqa: SLF001
            _observation(generation=generation, source_ref=sha, stale=True),
            generation,
            sha,
            "fixed native relay",
            expected_stale=False,
        ),
        "loaded_stale",
    )
    return "base-negative-control and fixed cases are distinguishable"


def cell_readiness(root: Path) -> str:
    """Ready passes; a timeout raises AND leaves the diagnostics artifact."""
    generation = root / "generations" / "head"
    sha = "d" * 40
    calls = {"n": 0}

    def ready_observation() -> dict[str, Any]:
        return _observation(generation=generation, source_ref=sha, stale=True)

    def never_ready() -> dict[str, Any]:
        """A relay that is up but never advertises this source — no host reads."""
        return _observation(generation=generation, source_ref="0" * 40, stale=True, answering=False)

    original_observation = probe._observation
    original_require = probe._require
    try:
        probe._observation = ready_observation  # type: ignore[assignment]
        probe._require = lambda _argv: None  # type: ignore[assignment]
        probe._start_and_wait(generation, sha, root / "private-systemd.log")  # noqa: SLF001
    finally:
        probe._observation = original_observation  # type: ignore[assignment]
        probe._require = original_require  # type: ignore[assignment]

    # Now the failure path: never answers, so the loop must exhaust and record.
    # The environment is SAVED and restored: this cell used to leave both keys
    # pointing at a directory it then deleted, which leaked into later cells.
    saved_env = {key: os.environ.get(key) for key in ("RUNNER_TEMP", "LOCAL_OPERATOR_CONFIG_DIR")}
    os.environ["RUNNER_TEMP"] = str(root)
    os.environ["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / "config")
    (root / "config" / "logs").mkdir(parents=True, exist_ok=True)
    (root / "config" / "logs" / "network.log").write_text("network-tail\n", encoding="utf-8")
    (root / "private-systemd.log").write_text("manager-tail\n", encoding="utf-8")

    class Clock:
        def __init__(self) -> None:
            self.values = iter([0.0, 0.0, 31.0])

        def monotonic(self) -> float:
            return next(self.values)

        def sleep(self, _seconds: float) -> None:
            return None

    saved: dict[str, Any] = {}
    try:
        # The supervisor stays faked for BOTH halves: this rehearsal exists to
        # exercise the fixture, never to reach a real systemd.
        saved["require"] = probe._require
        probe._require = lambda _argv: None  # type: ignore[assignment]
        # Q7-2/R12-1: ``_observation`` is faked here too. The real one reads this
        # host's supervisor and, on macOS, the operator's LIVE relay — a read-only
        # leak the comment above claimed did not exist.
        probe._observation = never_ready  # type: ignore[assignment]
        # R13-1: ``_write_readiness_diagnostics`` would otherwise shell out to
        # ``journalctl --user -u <live unit>`` on a Linux box and copy a tail of
        # the operator's own relay journal into this artifact. Read-only, but the
        # same isolation break as R12-1 one level down.
        saved["run"] = probe._run

        def no_run(*_a: Any, **_k: Any) -> SimpleNamespace:
            """Never reaches a host binary; the diagnostics read a fake receipt."""
            return SimpleNamespace(returncode=1, stdout="", stderr="")

        probe._run = no_run  # type: ignore[assignment]
        saved["time"] = probe.time
        probe.time = Clock()  # type: ignore[assignment]
        calls["n"] = 0
        try:
            probe._start_and_wait(generation, sha, root / "private-systemd.log")  # noqa: SLF001
        except probe.ProbeFailure as exc:
            _check("did not become ready" in str(exc), f"wrong timeout error: {exc}")
        else:
            raise RehearsalFailure("readiness timeout was accepted")
    finally:
        probe.time = saved["time"]  # type: ignore[assignment]
        probe._require = saved["require"]  # type: ignore[assignment]
        probe._observation = original_observation  # type: ignore[assignment]
        probe._run = saved["run"]  # type: ignore[assignment]
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    artifact = root / "native-relay-readiness.json"
    _check(artifact.is_file(), "readiness timeout left no diagnostics artifact")
    receipt = json.loads(artifact.read_text(encoding="utf-8"))
    _check("manager-tail" in receipt.get("manager_log_tail", ""), "manager tail missing")
    _check("network-tail" in receipt.get("network_log_tail", ""), "network tail missing")
    _check("redaction" in receipt, "redaction receipt missing")
    return "readiness passes when answered and fails loudly with evidence when not"


def cell_live_relay(root: Path) -> str:
    """The contract against a REAL relay record — the assumption that cost a run."""
    python = sys.executable
    config = root / "config"
    config.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    # The probe's own writer, so the cell fails if that shape regresses: the
    # relay must bind loopback BECAUSE the config was read, not because of a flag.
    probe._write_isolated_config(config, port)  # noqa: SLF001
    env = dict(os.environ)
    env.update(
        HOME=str(root),
        LOCAL_OPERATOR_CONFIG_DIR=str(config),
        LOCAL_OPERATOR_NO_NOTIFICATIONS="1",
        PYTHONPATH=str(Path(__file__).resolve().parents[1]),
    )
    process = subprocess.Popen(
        [python, "-m", "local_operator.network.relay", "--port", str(port)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    # A typed local rather than `process.stderr` at the use site: pyright cannot
    # narrow the Optional on a second read, and a rehearsal that silently dropped
    # the relay's own dying words would hide the reason it exists.
    stderr = process.stderr
    if stderr is None:  # pragma: no cover - PIPE guarantees a stream
        process.kill()
        process.wait(timeout=10)
        raise RehearsalFailure("real relay stderr is not captured; cannot report its failure")
    try:
        record_path = None
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            matches = sorted((config / "run" / "peers").glob("*.json"))
            if matches:
                record_path = matches[0]
                break
            if process.poll() is not None:
                raise RehearsalFailure(f"real relay exited: {stderr.read()[-500:]}")
            time.sleep(0.2)
        _check(record_path is not None, "real relay published no record within 20s")
        record = json.loads(Path(str(record_path)).read_text(encoding="utf-8"))
        # The measured contract: the relay publishes no install root at all.
        _check(
            record.get("install_root") == "",
            f"relay now publishes install_root={record.get('install_root')!r}; "
            "the fixture's relaxed rule must be revisited",
        )
        _check(record.get("pid") == process.pid, "record pid is not the started process")
        bound = str((record.get("listen") or {}).get("address") or "")
        _check(
            bound in ("127.0.0.1", "::1"),
            f"relay bound {bound!r}, not loopback: the isolated config was not read",
        )
        _check("source_ref" in record, "record carries no source_ref field")
        # R12-3: run the REAL record through the fixture's own assertion rather
        # than only eyeballing three fields. Two values cannot come from the
        # record: a relay publishes no loaded generation (that is the process's),
        # and this checkout has no install stamp, so source_ref may be "". Both
        # are supplied here and named in the returned sentence, so the cell says
        # exactly how much of the contract it exercised.
        generation = root / "generations" / "head"
        # R13-3: comparing sha with itself proved nothing. Assert the measured
        # state instead, so a tree that starts publishing a wrong ref fails here.
        sha = str(record.get("source_ref") or "")
        _check(sha == "", f"an unstamped checkout advertised source_ref={sha!r}")
        observation: dict[str, Any] = {
            "pid": process.pid,
            "loaded_generation": str(generation),
            "loaded_stale": False,
            "advertised_record_pid": int(record.get("pid") or 0),
            "advertised_source_ref": sha,
            "advertised_install_root": str(record.get("install_root") or ""),
            "advertised_listen_address": bound,
            "answering": True,
        }
        probe._assert_identity(  # noqa: SLF001
            observation, generation, sha, "live relay record", expected_stale=False
        )
        return (
            "a real relay record passes _assert_identity with install_root='' "
            f"(source_ref={sha!r}; loaded generation supplied, not advertised)"
        )
    finally:
        process.kill()
        process.wait(timeout=10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=None, help="write a receipt here")
    parser.add_argument("--skip-live", action="store_true", help="skip the real-relay cell")
    args = parser.parse_args(argv)

    root = Path(tempfile.mkdtemp(prefix="lop-native-rehearsal-"))
    cells = [
        ("identity", lambda: cell_identity_contract(root)),
        ("control", lambda: cell_negative_control_and_fixed(root)),
        ("readiness", lambda: cell_readiness(root)),
    ]
    if not args.skip_live:
        cells.append(("live-relay", lambda: cell_live_relay(root)))

    results: list[dict[str, str]] = []
    failed = False
    try:
        for name, run in cells:
            try:
                detail = run()
            except (RehearsalFailure, probe.ProbeFailure, OSError) as exc:
                failed = True
                results.append({"cell": name, "status": "FAIL", "detail": str(exc)})
                print(f"FAIL {name}: {exc}", file=sys.stderr)
            else:
                results.append({"cell": name, "status": "PASS", "detail": detail})
                print(f"PASS {name}: {detail}")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    # R12-4: the receipt must carry its own limits. A receipt read on its own
    # must not be able to imply that native systemd behaviour was exercised.
    not_covered = [
        "real systemd cgroup delegation and systemd-run privilege boundaries",
        "systemd opening the unit's append: log targets",
        "the private user manager's own unit setup",
        "the real 'lop update' child process and its one-shot ps fault",
        "generation installation by uv, and any Linux-only path",
    ]
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "cells": results,
                    "status": "FAIL" if failed else "PASS",
                    "not_covered": not_covered,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    # R13-2: unconditional, so a bare run's stdout summary carries the same
    # limits as the receipt.
    print("not covered: " + "; ".join(not_covered))
    print(f"rehearsal {'FAILED' if failed else 'passed'} ({len(results)} cells)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
