"""Bounded stdlib checks only: do not execute any recovered task or service."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import restore_sources
import verify_static

HERE = Path(__file__).resolve().parent


class ReconstructionTests(unittest.TestCase):
    def test_hash_negative_control(self):
        with tempfile.TemporaryDirectory(dir=os.environ['LOCAL_OPERATOR_SCRATCHPAD']) as scratch:
            path = Path(scratch) / 'source.txt'
            path.write_bytes(b'public source fixture')
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertTrue(verify_static.verify_payload(path, expected))
            path.write_bytes(b'changed source fixture')
            self.assertFalse(verify_static.verify_payload(path, expected))

    def test_path_escape_rejected_without_network(self):
        with self.assertRaises(ValueError):
            restore_sources.recover(('task', '../escape.py', '0' * 64))

    def test_image_never_downloaded(self):
        row = restore_sources.recover(('task', 'environment/image.png', '0' * 64))
        self.assertEqual(row['status'], 'not_downloaded_non_source')
        self.assertIsNone(row['observed_sha256'])

    def test_privacy_scanner_negative_control(self):
        self.assertIn('github-token', verify_static.sensitive_hits('gh' + 'p_' + 'A' * 40))
        self.assertEqual(verify_static.sensitive_hits('sha256:' + 'a' * 64), [])

    def test_static_verification_real_command_and_dispatch_refusal(self):
        for args, expected in (([], 0), (['--require-complete'], 1)):
            result = subprocess.run([sys.executable, '-B', str(HERE / 'verify_static.py'), *args],
                                    capture_output=True, text=True, timeout=30, check=False)
            self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
            report = json.loads(result.stdout)
            self.assertFalse(report['runtime_qualified'])
            if expected:
                self.assertIn('campaign incomplete', report['errors'][0])

    def test_proposed_host_lock_retains_dependency_edges(self):
        lock = json.loads((HERE / 'host-services.lock.json').read_text())
        names = {package['name'] for package in lock['packages']}
        for package in lock['packages']:
            self.assertTrue(all(dep['name'] in names for dep in package.get('dependencies', [])))
            self.assertTrue(all(wheel['hash'].startswith('sha256:') for wheel in package['wheels']))
        self.assertFalse(lock['historical_recovery'])


if __name__ == '__main__':
    unittest.main()
