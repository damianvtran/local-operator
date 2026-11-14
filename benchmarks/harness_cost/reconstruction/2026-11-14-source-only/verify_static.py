"""Verify reconstruction claims from bytes only; no task imports or execution."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

from restore_sources import REVISION, SOURCE_OVERRIDES

HERE = Path(__file__).resolve().parent
SENSITIVE = {
    'private-key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'aws-access-id': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'github-token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
    'huggingface-token': r'\bhf_[A-Za-z0-9]{30,}\b',
    'bearer-value': r'Bearer\s+[A-Za-z0-9_.-]{25,}',
    'credential-url': r'https?://[^\s/@:]+:[^\s/@]+@',
    'operator-home': r'/Users/' + 'damian/',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_payload(path, expected):
    return path.is_file() and sha256(path) == expected


def sensitive_hits(text):
    return [name for name, pattern in SENSITIVE.items() if re.search(pattern, text)]


def classified_public_fixture(path, hits):
    """Only immutable, independently sourced public canaries are non-secrets.

    A path alone is insufficient: adding even one byte invalidates this narrow
    classification. Unexpected pattern families still fail the privacy gate.
    """
    return (path == HERE / 'payloads/sanitize-git-repo/tests/test_outputs.py'
            and sha256(path) == '57e4904177acac5807b51f7c32444bde1968ba0445855fa4333d9eb73865b59e'
            and set(hits) <= {'aws-access-id', 'github-token', 'huggingface-token'})


def matrix_errors(matrix, rows, tasks):
    # A correct aggregate can hide a fabricated per-task eligibility claim.
    # Compare complete rows, including gaps and source identity, in prereg order.
    expected = []
    for task in tasks:
        files = [row for row in rows if row['task'] == task]
        expected.append({'task': task, 'source_commit': files[0]['source_commit'],
                         'source_commits': sorted({row['source_commit'] for row in files}),
                         'counts': dict(collections.Counter(row['status'] for row in files)),
                         'complete_exact': all(row['status'] == 'exact_match' for row in files),
                         'gaps': [row for row in files if row['status'] != 'exact_match']})
    return [] if matrix['tasks'] == expected else ['per-task matrix differs from latest receipt']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    errors = []
    prereg = HERE.parents[1] / 'experimental/PREREGISTRATION-HEADLINE-2026-11-09-v2.json'
    preregistration = json.loads(prereg.read_text())
    historical = json.loads((HERE / 'historical-bindings.json').read_text())
    if not verify_payload(prereg, historical['preregistration_sha256']):
        errors.append('historical preregistration bytes changed')
    for receipt in HERE.glob('source-receipt-*.jsonl'):
        for line in receipt.read_text().splitlines():
            row = json.loads(line)
            expected = preregistration['task_files'].get(row['task'], {}).get(row['file'])
            if expected != row['expected_sha256']:
                errors.append('receipt differs from historical binding: ' + row['task'] + '/' + row['file'])
            override = SOURCE_OVERRIDES.get((row['task'], row['file']))
            allowed_revisions = {REVISION} | ({override[0]} if override else set())
            if row['source_commit'] not in allowed_revisions:
                errors.append('unexpected public source revision')
            if override and row['status'] == 'exact_match' and (row['source_commit'], row['url']) != override:
                errors.append('exact recovered override lacks immutable source provenance')
            if row['status'] == 'exact_match':
                path = HERE / 'payloads' / row['task'] / row['file']
                if row['observed_sha256'] != row['expected_sha256'] or not verify_payload(path, row['expected_sha256']):
                    errors.append('invalid exact-match source claim: ' + row['task'] + '/' + row['file'])
    metadata_count = 0
    for receipt in HERE.glob('metadata-receipt-*.jsonl'):
        for line in receipt.read_text().splitlines():
            row = json.loads(line)
            if row['status'] == 'available':
                metadata_count += 1
                if not verify_payload(HERE / row['file'], row['sha256']):
                    errors.append('metadata changed: ' + row['name'])
    for row in historical['checks']:
        path = HERE.parents[1] / row['path']
        actual = sha256(path) if path.is_file() else None
        if actual != row['observed_sha256']:
            errors.append('historical source changed: ' + row['path'])
    privacy = []
    public_fixtures = []
    for path in HERE.rglob('*'):
        if path.is_file():
            hits = sensitive_hits(path.read_text(errors='replace'))
            if hits and classified_public_fixture(path, hits):
                public_fixtures.append({'path': str(path.relative_to(HERE)), 'rules': hits, 'classification': 'exact-public-FAKE-canary-source'})
                continue
            for name in hits:
                privacy.append({'path': str(path.relative_to(HERE)), 'rule': name})
    errors.extend('sensitive pattern: ' + row['path'] + ' (' + row['rule'] + ')' for row in privacy)
    matrix = json.loads((HERE / 'task-matrix.json').read_text())
    counts = collections.Counter()
    for task in matrix['tasks']:
        counts.update(task['counts'])
    latest_rows = [json.loads(line) for line in sorted(HERE.glob('source-receipt-*.jsonl'))[-1].read_text().splitlines()]
    expected_keys = {(task, name) for task, files in preregistration['task_files'].items() for name in files}
    actual_keys = [(row['task'], row['file']) for row in latest_rows]
    if len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys:
        errors.append('latest receipt has missing or duplicate bindings')
    if dict(collections.Counter(row['status'] for row in latest_rows)) != dict(counts):
        errors.append('derived matrix differs from latest receipt')
    if set(actual_keys) == expected_keys and len(actual_keys) == len(expected_keys):
        errors.extend(matrix_errors(matrix, latest_rows, preregistration['tasks']))
    if args.require_complete and any(row['status'] != 'exact_match' for row in latest_rows):
        errors.append('campaign incomplete: exact recovery gate refuses dispatch')
    print(json.dumps({'kind': 'static-source-verification', 'runtime_qualified': False,
                      'historical_checks': len(historical['checks']), 'metadata_receipt_rows_checked': metadata_count,
                      'file_statuses': dict(counts), 'privacy_pattern_hits': privacy,
                      'classified_public_fixtures': public_fixtures, 'errors': errors}, sort_keys=True))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
