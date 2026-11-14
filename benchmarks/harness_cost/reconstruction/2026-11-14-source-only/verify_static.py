"""Verify reconstruction claims from bytes only; no task imports or execution."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SENSITIVE = {
    'private-key': r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----',
    'aws-access-id': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'github-token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})',
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
            if row['source_commit'] != '7131e4375048a0e408a8fb404b5f499d726b695b':
                errors.append('unexpected public source revision')
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
    for path in HERE.rglob('*'):
        if path.is_file():
            for name in sensitive_hits(path.read_text(errors='replace')):
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
    if args.require_complete and any(row['status'] != 'exact_match' for row in latest_rows):
        errors.append('campaign incomplete: exact recovery gate refuses dispatch')
    print(json.dumps({'kind': 'static-source-verification', 'runtime_qualified': False,
                      'historical_checks': len(historical['checks']), 'metadata_receipt_rows_checked': metadata_count,
                      'file_statuses': dict(counts), 'privacy_pattern_hits': privacy, 'errors': errors}, sort_keys=True))
    return bool(errors)


if __name__ == '__main__':
    raise SystemExit(main())
