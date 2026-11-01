"""Tiny payload controls; deliberately never collect the repository AST test.

Run with the existing interpreter, not a new environment:
python -m unittest discover -s .github/diagnostics -p 'test_ci_selector_native.py' -v
"""

import importlib.util
import json
import os
import signal
import sys
import tempfile
import unittest
from pathlib import Path

SOURCE = Path(__file__).with_name("ci_selector_native.py")
spec = importlib.util.spec_from_file_location("native_diag", SOURCE)
diag = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diag)
REPO = SOURCE.parents[2]


class OrchestrationControls(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(
            dir=os.environ["LOCAL_OPERATOR_SCRATCHPAD"], prefix="selector-control-"
        )
        self.addCleanup(self.scratch.cleanup)
        self.out = Path(self.scratch.name) / "case"

    def run_payload(self, code, **kwargs):
        return diag.run_case(REPO, self.out, [sys.executable, "-c", code], **kwargs)

    def assert_reaped(self, result):
        status = result["child_status"]
        self.assertIsNotNone(status)
        # The tested child is reaped by Popen.wait, not just signalled.
        with self.assertRaises(ProcessLookupError):
            os.kill(status["pid"], 0)
        samples = [
            json.loads(line) for line in (self.out / "processes.jsonl").read_text().splitlines()
        ]
        self.assertTrue(samples)
        self.assertIn("cgroup_before", result)
        self.assertIn("cgroup_after", result)

    def test_success_and_full_streams(self):
        result = self.run_payload(
            "import sys; print('stdout-control'); print('stderr-control', file=sys.stderr)",
            timeout=5,
            memory_mb=256,
        )
        self.assertEqual(result["wrapper_returncode"], 0)
        self.assertEqual(result["child_status"]["raw_returncode"], 0)
        self.assertIn("stdout-control", (self.out / "stdout.log").read_text())
        self.assertIn("stderr-control", (self.out / "stderr.log").read_text())
        self.assert_reaped(result)

    def test_nonzero_is_not_a_pass(self):
        result = self.run_payload("raise SystemExit(23)", timeout=5, memory_mb=256)
        self.assertEqual(result["wrapper_returncode"], 23)
        self.assertEqual(result["child_status"]["exit_code"], 23)
        self.assertIsNone(result["child_status"]["signal"])
        self.assert_reaped(result)

    def test_signal_is_not_explicit_exit(self):
        result = self.run_payload(
            "import os,signal; os.kill(os.getpid(), signal.SIGKILL)",
            timeout=5,
            memory_mb=256,
        )
        self.assertEqual(result["wrapper_returncode"], 137)
        self.assertEqual(result["child_status"]["signal"], signal.SIGKILL)
        self.assertIsNone(result["child_status"]["exit_code"])
        self.assert_reaped(result)

    def test_timeout_escalates_and_reaps_owned_tree(self):
        result = self.run_payload(
            "import subprocess,sys,signal,time; "
            "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
            "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
            "print(p.pid,flush=True); time.sleep(30)",
            timeout=2,
            grace=0.2,
            memory_mb=256,
        )
        self.assertEqual(result["wrapper_returncode"], 124)
        self.assertLess(result["wall_seconds"], 10)
        self.assertIn("SIGKILL", (self.out / "stderr.log").read_text())
        self.assert_reaped(result)
        child = int((self.out / "stdout.log").read_text().strip())
        # A Linux container's PID 1 can retain a reparented zombie. That is dead,
        # not a live leaked sleeper; ps exposes the distinction on both hosts.
        probe = diag.subprocess.run(
            ["ps", "-o", "stat=", "-p", str(child)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
        state = probe.stdout.strip()
        self.assertTrue(not state or state.startswith("Z"), state)

    def test_memory_cap_has_distinct_evidence(self):
        result = self.run_payload(
            "import time; a=bytearray(80*1024*1024); time.sleep(15)",
            timeout=8,
            grace=0.2,
            memory_mb=32,
        )
        self.assertEqual(result["wrapper_returncode"], 137)
        self.assertIn("MEMORY LIMIT EXCEEDED", (self.out / "stderr.log").read_text())
        self.assertEqual(result["child_status"]["signal"], signal.SIGKILL)
        self.assertLess(result["wall_seconds"], 10)
        self.assert_reaped(result)
        disk_bytes = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
        self.assertLess(disk_bytes, 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
