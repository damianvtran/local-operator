"""Static provenance checks only; never import/download/execute upstream payloads."""
import base64
import hashlib
import json
from pathlib import Path
import unittest

import collect_metadata

HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text())


class EnvironmentSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.findings = load('source-closure-findings.json')
        cls.receipts = {row['name']: row for row in map(json.loads, (HERE / cls.findings['source_receipt']).read_text().splitlines())}

    def test_receipt_bytes_and_collection_urls_are_source_metadata_only(self):
        for row in self.receipts.values():
            if row['status'] == 'available':
                self.assertEqual(hashlib.sha256((HERE / row['file']).read_bytes()).hexdigest(), row['sha256'])
        for url in self.findings['collection_sources'].values():
            self.assertFalse(url.split('?', 1)[0].endswith(('.tgz', '.exe', '.png', '.jpg', '.mp4', '.pth')))

    def test_omp_statement_binds_source_and_distribution_without_signature_claim(self):
        row = self.receipts['omp-provenance']
        attestations = load(row['file'])['attestations']
        npm = load(next((HERE / 'metadata').glob('omp-b659*.json')).relative_to(HERE))
        expected = base64.b64decode(npm['dist']['integrity'].split('-', 1)[1]).hex()
        provenance = next(a for a in attestations if a['predicateType'] == 'https://slsa.dev/provenance/v1')
        statement = json.loads(base64.b64decode(provenance['bundle']['dsseEnvelope']['payload']))
        self.assertEqual(statement['subject'][0]['digest']['sha512'], expected)
        dependencies = statement['predicate']['buildDefinition']['resolvedDependencies']
        self.assertIn(self.findings['omp']['source_revision'], [d['digest'].get('gitCommit') for d in dependencies])
        self.assertFalse(self.findings['omp']['provenance_signature_verified'])

    def test_dsh_source_version_and_all_patch_hashes_are_lock_bound(self):
        dsh = self.findings['dsh']
        self.assertEqual(load(dsh['source_cli_manifest']['file'])['version'], dsh['version'])
        lock = (HERE / dsh['source_lock']['file']).read_text()
        self.assertEqual(len(dsh['patches']), 7)
        for row in dsh['patches']:
            self.assertIn(row['sha256'], lock)
            self.assertIn(dsh['source_revision'], row['url'])

    def test_opencode_source_files_match_listing_but_binary_is_unacquired(self):
        oc = self.findings['opencode']
        for row in oc['published_source_checks']:
            self.assertEqual(row['observed_sha256'], row['listing_sha256'])
            self.assertTrue(row['matches'])
        build = (HERE / self.receipts['oc-source-build']['file']).read_text()
        self.assertIn('Bun.build', build)
        self.assertFalse(oc['linux_binary_acquired'])
        self.assertIn('unrecoverable', oc['historical_runtime_identity'])

    def test_licensing_and_build_admission_remain_blocked(self):
        eligibility = load('binary-eligibility.json')
        inventory = load('non-source-acquisition.json')
        self.assertEqual({(r['task'], r['file']) for r in eligibility['inputs']}, {(r['task'], r['file']) for r in inventory['inputs']})
        self.assertEqual(len(eligibility['inputs']), 26)
        self.assertTrue(all(not row['eligible_for_acquisition'] and not row['acquired'] for row in eligibility['inputs']))
        self.assertFalse(self.findings['runtime_build_ready'])
        self.assertFalse(self.findings['local_operator']['root_python_lock_available'])
        self.assertFalse(self.findings['local_operator']['nix_flake_locks_python_dependencies'])

    def test_collector_replays_retained_source_urls_without_network(self):
        sources = collect_metadata.metadata_sources()
        for name, url in self.findings['collection_sources'].items():
            self.assertEqual(sources[name], url)


if __name__ == '__main__':
    unittest.main()
