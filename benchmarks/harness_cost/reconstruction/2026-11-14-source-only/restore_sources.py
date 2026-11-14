"""Source-only, immutable public recovery; never executes downloaded task code.

Only preregistered text-source suffixes are downloaded. Media, archives, models
and encrypted payloads stay explicit gaps: a byte hash alone is not permission
to collect arbitrary binary inputs. Receipts are exclusive-created per attempt,
so repeating recovery cannot rewrite earlier observations.
"""
import concurrent.futures
import datetime
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
PREREG = HERE.parents[1] / 'experimental/PREREGISTRATION-HEADLINE-2026-11-09-v2.json'
REVISION = '7131e4375048a0e408a8fb404b5f499d726b695b'
BASE = f'https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/{REVISION}'
# TB2.1 rewrote public FAKE_* string literals without changing semantics. The
# immutable TB2 ancestor, not a locally reconstructed substitution, matches the
# historical preregistered bytes exactly for this one verifier source file.
SOURCE_OVERRIDES = {
    ('sanitize-git-repo', 'tests/test_outputs.py'): (
        '3dbbcbbedc34919bd044f5780a87b00eb99b2b35',
        'https://raw.githubusercontent.com/harbor-framework/terminal-bench-2/3dbbcbbedc34919bd044f5780a87b00eb99b2b35/sanitize-git-repo/tests/test_outputs.py',
    ),
}
TEXT_SUFFIXES = {'.md', '.toml', '.py', '.sh', '.txt', '.json', '.yaml', '.yml', '.scm', '.red', '.ttl', '.xml', '.c', '.h', '.cpp', '.js', '.ts', '.csv', '.tex', '.v', '.sv', '.s', '.asm', '.html', '.css', '.sql', '.patch', '.diff', '.cfg', '.ini', '.conf', '.log', '.fasta', '.pov', '.jsonl'}


def recover(item):
    task, name, expected = item
    revision, url = SOURCE_OVERRIDES.get((task, name), (REVISION, f'{BASE}/tasks/{task}/{name}'))
    row = {'task': task, 'file': name, 'expected_sha256': expected,
           'source_commit': revision, 'url': url}
    if (task, name) in SOURCE_OVERRIDES:
        row['publication'] = 'local-only: remote push protection; exact bytes excluded from git'
    path = Path(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('unsafe preregistration path')
    if path.suffix not in TEXT_SUFFIXES and path.name not in {'Dockerfile', 'Makefile', 'CMakeLists.txt', 'LICENSE'}:
        return dict(row, status='not_downloaded_non_source', observed_sha256=None)
    target = HERE / 'payloads' / task / name
    try:
        if target.exists():
            data = target.read_bytes()
        else:
            with urllib.request.urlopen(row['url'], timeout=30) as response:
                data = response.read(8_000_001)
            if len(data) > 8_000_000:
                return dict(row, status='source_size_limit', observed_sha256=None)
        observed = hashlib.sha256(data).hexdigest()
        status = 'exact_match' if observed == expected else 'mismatch'
        if status == 'exact_match' and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as out:
                out.write(data)
        return dict(row, status=status, observed_sha256=observed, bytes=len(data))
    except (OSError, urllib.error.URLError) as error:
        return dict(row, status='unavailable', observed_sha256=None, error=str(error))


def main():
    prereg = json.loads(PREREG.read_text())
    items = [(task, name, expected) for task, files in prereg['task_files'].items()
             for name, expected in files.items()]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(recover, items))
    receipt = HERE / ('source-receipt-' + datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ') + '.jsonl')
    with receipt.open('x') as out:
        for row in rows:
            out.write(json.dumps(row, sort_keys=True) + '\n')
    counts = {status: sum(r['status'] == status for r in rows) for status in sorted({r['status'] for r in rows})}
    print(json.dumps({'receipt': receipt.name, 'files': len(rows), 'counts': counts}, sort_keys=True))


if __name__ == '__main__':
    main()
