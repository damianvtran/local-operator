"""Real owned-process negative controls; no harness, credentials or paid traffic."""

import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import driver


class DriverSafetyTests(unittest.TestCase):
    def test_owned_group_cleanup(self):
        # A separate fixture session is deliberately NOT registered with driver.
        # Cleanup must leave it alone while sweeping only each owned session.
        unrelated = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True
        )
        try:
            driver.stop(unrelated)
            for resistant in (False, True):
                for exited in (False, True):
                    with self.subTest(term_resistant=resistant, leader_exited=exited):
                        self.assert_owned_group_cleaned(resistant, exited)
                        self.assertIsNone(unrelated.poll())
        finally:
            try:
                os.killpg(unrelated.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            unrelated.wait(timeout=5)

    def assert_owned_group_cleaned(self, resistant, exited):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            ready = Path(tmp) / "child.pid"
            child = (
                "import os,signal,time; from pathlib import Path; "
                f"signal.signal(signal.SIGTERM, "
                f"{'signal.SIG_IGN' if resistant else 'signal.SIG_DFL'}); "
                f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)"
            )
            leader = (
                "import subprocess,sys,time; from pathlib import Path; "
                f"subprocess.Popen([sys.executable, '-c', {child!r}]); "
                f"p=Path({str(ready)!r}); "
                "exec('while not p.exists(): time.sleep(.01)'); "
                + ("" if exited else "time.sleep(60)")
            )
            proc = driver.owned_process([sys.executable, "-c", leader], start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not ready.exists() and time.monotonic() < deadline:
                    time.sleep(0.01)
                if exited:
                    proc.wait(timeout=5)
                else:
                    self.assertIsNone(proc.poll())
                pid = int(ready.read_text())
                os.kill(pid, 0)  # Negative control: cleanup starts with a live child.
                with patch.object(driver, "STOP_GRACE_SECONDS", 0.15, create=True):
                    driver.stop(proc)
                    driver.stop(proc)  # A second cleanup must not signal a recycled ID.
                deadline = time.monotonic() + 3
                while time.monotonic() < deadline:
                    result = subprocess.run(
                        ["ps", "-o", "stat=", "-p", str(pid)],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if not result.stdout.strip() or result.stdout.strip().startswith("Z"):
                        break
                    time.sleep(0.02)
                else:
                    self.fail("owned TERM-resistant child still running after stop")
            finally:
                # This fixture creates only this fresh session; never sweep other jobs.
                driver._signal_owned_group(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)

    def test_unowned_or_unsafe_groups_are_never_signalled(self):
        with patch.object(driver.os, "killpg") as kill:
            driver.stop(object())
            kill.assert_not_called()

            class Process:
                pid = os.getpgrp()

            proc = Process()
            driver._OWNED_GROUPS[proc] = (os.getpid(), proc.pid)
            try:
                with self.assertRaisesRegex(ValueError, "outside"):
                    driver.stop(proc)
                kill.assert_not_called()
            finally:
                driver._OWNED_GROUPS.pop(proc, None)

    def test_zombie_only_permission_failure_is_not_a_live_group(self):
        with (
            patch.object(driver.os, "killpg", side_effect=PermissionError),
            patch.object(driver.subprocess, "run") as ps,
        ):
            ps.return_value.stdout = "123 Z\n456 S\n"
            self.assertFalse(driver._signal_owned_group(123, 0))
            ps.return_value.stdout = "123 S\n"
            with self.assertRaises(PermissionError):
                driver._signal_owned_group(123, 0)

    # Real Darwin ``ps -o stat=`` shapes (R14 M1): the FIRST character is the
    # run state (I/R/S/T/U/Z or ?); "E" (trying to exit) is a trailing flag.
    GONE = ("Z", "Zs", "?E", "RE", "UE", "SE", "SEs")
    LIVE = ("S", "R+", "Ss", "U", "T", "I")

    def test_exiting_members_are_not_a_live_group(self):
        # S9e/R14 M1: a zombie or a member flagged "E" can no longer be
        # signalled and must not read as a live group; anything else is live.
        with (
            patch.object(driver.os, "killpg", side_effect=PermissionError),
            patch.object(driver.subprocess, "run") as ps,
        ):
            ps.return_value.stdout = "".join(f"123 {s}\n" for s in self.GONE) + "456 S\n"
            self.assertFalse(driver._signal_owned_group(123, 0))
            for live in self.LIVE:
                with self.subTest(stat=live):
                    ps.return_value.stdout = f"123 ?E\n123 {live}\n"
                    with self.assertRaises(PermissionError):
                        driver._signal_owned_group(123, 0)

    def _owned_stub(self, pid):
        class Proc:
            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        proc = Proc()
        proc.pid = pid
        driver._OWNED_GROUPS[proc] = (os.getpid(), pid)
        self.addCleanup(driver._OWNED_GROUPS.pop, proc, None)
        return proc

    def test_every_stop_signal_tolerates_an_exiting_group(self):
        # R14 M1: the first SIGTERM (stop_edge after the leader exited), the
        # grace-loop probe and the final SIGKILL sweep all meet EPERM with
        # only exiting members; none may raise and lose result.json.
        for stat in ("?E", "RE", "UE", "SE", "Z"):
            for failing in ({signal.SIGTERM, 0, signal.SIGKILL}, {0, signal.SIGKILL}):
                with self.subTest(stat=stat, failing=sorted(failing)):
                    proc = self._owned_stub(2_000_000_011)
                    sent = []

                    def killpg(group, sig, failing=failing):
                        sent.append((group, sig))
                        if sig in failing:
                            raise PermissionError

                    with (
                        patch.object(driver, "STOP_GRACE_SECONDS", 0.2),
                        patch.object(driver.os, "killpg", side_effect=killpg),
                        patch.object(driver.subprocess, "run") as ps,
                    ):
                        ps.return_value.stdout = f"{proc.pid} {stat}\n"
                        driver.stop(proc)
                    self.assertEqual({g for g, _ in sent}, {proc.pid})
                    self.assertNotIn(proc, driver._OWNED_GROUPS)

    def test_a_live_unsignallable_member_still_raises_on_the_first_signal(self):
        for stat in self.LIVE:
            with self.subTest(stat=stat):
                proc = self._owned_stub(2_000_000_013)
                with (
                    patch.object(driver.os, "killpg", side_effect=PermissionError),
                    patch.object(driver.subprocess, "run") as ps,
                ):
                    ps.return_value.stdout = f"{proc.pid} {stat}\n"
                    with self.assertRaises(PermissionError):
                        driver.stop(proc)

    def test_stop_survives_a_group_that_vanishes_mid_teardown(self):
        # S9e (d0b2b32c DEV capture, pass A): LiteLLM had already exited when
        # driver.stop signalled its group; the EPERM probe saw a member still
        # exiting, raised, and the rig wrote no result.json. A transient EPERM
        # during the grace period must keep polling, not abort teardown, and
        # only the exact owned pgid is ever signalled.
        class Proc:
            pid = 2_000_000_001

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        proc = Proc()
        driver._OWNED_GROUPS[proc] = (os.getpid(), proc.pid)
        sent = []

        def killpg(group, sig):
            sent.append((group, sig))
            if sig != signal.SIGTERM:
                raise PermissionError

        tables = iter([f"{proc.pid} R\n", f"{proc.pid} Z\n"])
        try:
            with (
                patch.object(driver.os, "killpg", side_effect=killpg),
                patch.object(driver.subprocess, "run") as ps,
            ):
                ps.side_effect = lambda *a, **k: argparse.Namespace(stdout=next(tables))
                driver.stop(proc)
        finally:
            driver._OWNED_GROUPS.pop(proc, None)
        self.assertNotIn(proc, driver._OWNED_GROUPS)
        self.assertEqual({group for group, _ in sent}, {proc.pid})
        self.assertEqual(sent[0], (proc.pid, signal.SIGTERM))
        self.assertNotIn((proc.pid, signal.SIGKILL), sent)

    def test_stop_still_raises_for_a_live_group_it_cannot_signal(self):
        # The tolerance is bounded by the grace period: a member that stays
        # live and unsignallable is a real permission failure, never swallowed.
        class Proc:
            pid = 2_000_000_003

            def poll(self):
                return 0

            def wait(self, timeout=None):
                return 0

        proc = Proc()
        driver._OWNED_GROUPS[proc] = (os.getpid(), proc.pid)

        def killpg(group, sig):
            if sig != signal.SIGTERM:
                raise PermissionError

        try:
            with (
                patch.object(driver, "STOP_GRACE_SECONDS", 0.2),
                patch.object(driver.os, "killpg", side_effect=killpg),
                patch.object(driver.subprocess, "run") as ps,
            ):
                ps.return_value.stdout = f"{proc.pid} S\n"
                with self.assertRaises(PermissionError):
                    driver.stop(proc)
        finally:
            driver._OWNED_GROUPS.pop(proc, None)

    def test_paid_preflight_refuses_before_any_evidence_or_side_effect(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            manifest = Path(tmp) / "self-asserted.json"
            manifest.write_text(
                json.dumps(
                    {
                        "frozen": True,
                        "verified": True,
                        "linux_verified": True,
                        "files": {"not-a-runtime.txt": "a" * 64},
                    }
                )
            )
            self.assertEqual(driver.frozen(manifest, driver.digest(manifest))["verified"], True)
            for harness in driver.HARNESS:
                args = argparse.Namespace(
                    harnesses=harness,
                    preregistration=manifest,
                    preregistration_sha256=driver.digest(manifest),
                )
                with self.subTest(harness=harness), ExitStack() as stack:
                    for target in ("frozen", "check_space", "owned_process"):
                        stack.enter_context(
                            patch.object(
                                driver,
                                target,
                                side_effect=AssertionError("dispatch boundary crossed"),
                            )
                        )
                    stack.enter_context(
                        patch.object(
                            driver.os.environ,
                            "get",
                            side_effect=AssertionError("credential lookup"),
                        )
                    )
                    stack.enter_context(
                        patch.object(
                            driver.subprocess,
                            "Popen",
                            side_effect=AssertionError("subprocess started"),
                        )
                    )
                    with self.assertRaisesRegex(ValueError, "HARDBLOCK PAID CODING"):
                        driver.preflight(args)
                    # Direct dispatch/service entrypoints must not bypass preflight.
                    with self.assertRaisesRegex(ValueError, "HARDBLOCK PAID CODING"):
                        driver.run_trial(args, harness, {})
                    with self.assertRaisesRegex(ValueError, "HARDBLOCK PAID CODING"):
                        with driver.lane_services(args):
                            self.fail("services started")

    def test_cli_execute_cannot_be_licensed_by_self_asserted_manifest(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get("LOCAL_OPERATOR_SCRATCHPAD")) as tmp:
            path = Path(tmp) / "self-asserted.json"
            path.write_text(
                json.dumps(
                    {
                        "frozen": True,
                        "verified": True,
                        "linux_verified": True,
                        "tools_observed": True,
                    }
                )
            )
            argv = []
            for name in (
                "toolchain",
                "task-root",
                "output",
                "request-log",
                "preregistration",
                "runtime-attestation",
                "toolwire-attestation",
                "oracle-attestation",
                "subset",
            ):
                argv.extend(["--" + name, str(path)])
            for name in (
                "preregistration-sha256",
                "runtime-sha256",
                "toolwire-sha256",
                "oracle-sha256",
            ):
                argv.extend(["--" + name, driver.digest(path)])
            for name in (
                "total-budget-usd",
                "trial-budget-usd",
                "input-usd-per-m",
                "output-usd-per-m",
            ):
                argv.extend(["--" + name, "1"])
            argv += [
                "--billing-tier",
                "explicit",
                "--model",
                "not-a-model",
                "--tasks",
                "stub",
                "--rep",
                "0",
                "--meter-url",
                "http://127.0.0.1:1",
                "--translator-url",
                "http://127.0.0.1:1",
                "--meter-current-run-file",
                str(Path(tmp) / "current_run"),
                "--execute",
            ]
            for harness in driver.HARNESS:
                with self.subTest(harness=harness):
                    result = subprocess.run(
                        [
                            sys.executable,
                            str(driver.HERE / "driver.py"),
                            *argv,
                            "--harnesses",
                            harness,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=5,
                    )
                    self.assertEqual(result.returncode, 2, result.stderr)
                    self.assertIn("HARDBLOCK PAID CODING", result.stderr)
                    self.assertEqual(result.stdout, "")
                    self.assertEqual(list(Path(tmp).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
