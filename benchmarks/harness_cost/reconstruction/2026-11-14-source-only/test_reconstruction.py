"""Bounded stdlib checks only: do not execute any recovered task or service."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

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
        self.assertIn('huggingface-token', verify_static.sensitive_hits('h' + 'f_' + 'A' * 40))
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

    def test_tampered_per_task_complete_and_gaps_are_rejected(self):
        # Aggregate counts do not change: the verifier must check each claimed
        # task eligibility row, not merely the campaign-level totals.
        matrix_path = HERE / 'task-matrix.json'
        matrix = json.loads(matrix_path.read_text())
        task = next(row for row in matrix['tasks'] if not row['complete_exact'])
        task['complete_exact'] = True
        task['gaps'] = []
        original_read_text = Path.read_text

        def read_text(path, *args, **kwargs):
            if path == matrix_path:
                return json.dumps(matrix)
            return original_read_text(path, *args, **kwargs)

        with mock.patch.object(Path, 'read_text', read_text), mock.patch.object(sys, 'argv', ['verify_static.py']):
            with contextlib.redirect_stdout(io.StringIO()) as output:
                status = verify_static.main()
        self.assertTrue(status, output.getvalue())
        self.assertIn('per-task matrix differs', output.getvalue())

    def test_public_fixture_classification_is_path_digest_and_rule_bound(self):
        path = HERE / 'payloads/sanitize-git-repo/tests/test_outputs.py'
        self.assertTrue(verify_static.classified_public_fixture(path, ['aws-access-id', 'github-token', 'huggingface-token']))
        self.assertFalse(verify_static.classified_public_fixture(path, ['private-key']))
        self.assertFalse(verify_static.classified_public_fixture(path.with_name('other.py'), ['github-token']))
        with mock.patch.object(verify_static, 'sha256', return_value='0' * 64):
            self.assertFalse(verify_static.classified_public_fixture(path, ['github-token']))

    def test_exact_local_canary_is_not_in_publication_index(self):
        inputs = json.loads((HERE / 'local-only-inputs.json').read_text())['inputs']
        for row in inputs:
            path = HERE / row['relative_path']
            self.assertEqual(verify_static.sha256(path), row['sha256'])
            result = subprocess.run(['git', 'ls-files', '--error-unmatch', str(path)],
                                    cwd=HERE, capture_output=True, text=True, timeout=10, check=False)
            self.assertEqual(result.returncode, 1, 'blocked raw fixture must remain outside the published index')

    def test_new_image_observations_match_retained_metadata_not_historical_claims(self):
        bindings = json.loads((HERE / 'image-bindings.json').read_text())
        self.assertFalse(bindings['historical_recovery'])
        for row in bindings['observations']:
            path = HERE / row['source_metadata_file']
            self.assertEqual(verify_static.sha256(path), row['source_metadata_sha256'])
            metadata = json.loads(path.read_text())
            self.assertEqual(row['image_digest'], metadata['digest'])
            self.assertEqual(row['full_size'], metadata['full_size'])
            self.assertEqual(row['origin'], 'new-public-registry-observation-not-historical-recovery')
            if row['historical_control_digest']:
                self.assertTrue(row['control_matches'])

    def test_proposed_host_lock_retains_dependency_edges(self):
        lock = json.loads((HERE / 'host-services.lock.json').read_text())
        names = {package['name'] for package in lock['packages']}
        for package in lock['packages']:
            self.assertTrue(all(dep['name'] in names for dep in package.get('dependencies', [])))
            self.assertTrue(all(wheel['hash'].startswith('sha256:') for wheel in package['wheels']))
        self.assertFalse(lock['historical_recovery'])


if __name__ == '__main__':
    unittest.main()
