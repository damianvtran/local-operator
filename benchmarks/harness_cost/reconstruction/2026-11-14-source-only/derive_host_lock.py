"""Extract a proposed service dependency slice from a pinned upstream uv.lock.

This is not the deleted host environment. Keeping the original package records
preserves markers and wheel hashes without pretending to solve a new environment.
The separate Harbor/LoP dispatcher closure is intentionally not included.
"""
import hashlib
import json
from pathlib import Path
import tomllib

HERE = Path(__file__).resolve().parent
ROOTS = ('httpx', 'starlette', 'uvicorn', 'cryptography')


def main():
    source = next((HERE / 'metadata').glob('hermes-uv-lock-*.lock'))
    lock = tomllib.loads(source.read_text())
    by_name = {}
    for package in lock['package']:
        by_name.setdefault(package['name'], []).append(package)
    wanted = set()
    pending = list(ROOTS)
    packages = []
    while pending:
        name = pending.pop()
        if name in wanted:
            continue
        candidates = by_name.get(name, [])
        if len(candidates) != 1:
            raise ValueError(f'ambiguous or unavailable static package: {name}')
        package = candidates[0]
        if package.get('source', {}).get('registry') != 'https://pypi.org/simple':
            raise ValueError(f'non-registry dependency: {name}')
        if not package.get('wheels'):
            raise ValueError(f'no prebuilt wheel hashes: {name}')
        wanted.add(name)
        packages.append(package)
        pending.extend(dep['name'] for dep in package.get('dependencies', []))
    packages.sort(key=lambda package: package['name'])
    artifact = {'status': 'proposed-static-closure-not-runtime-qualified', 'roots': ROOTS,
                'source_file': str(source.relative_to(HERE)), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'historical_recovery': False, 'supported_python': 'CPython 3.14; platform-wheel availability unverified',
                'source_lock_resolution_markers': lock.get('resolution-markers'), 'packages': packages}
    (HERE / 'host-services.lock.json').write_text(json.dumps(artifact, indent=2, sort_keys=True) + '\n')
    lines = ['# PROPOSED service closure only; NOT the historical host lock.',
             '# Derived from pinned Hermes uv.lock, all base-dependency edges retained.',
             '# Future install must use --require-hashes --only-binary=:all: --no-deps.',
             '# No install/build was performed; CPython 3.14 platform qualification required.']
    for package in packages:
        hashes = sorted({wheel['hash'] for wheel in package['wheels']})
        lines.append(package['name'] + '==' + package['version'] + ' \\\n    ' + ' \\\n    '.join('--hash=' + value for value in hashes))
    (HERE / 'host-services.requirements.lock').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'service_roots': list(ROOTS), 'locked_packages': len(packages), 'closure_names': sorted(wanted)}, sort_keys=True))


if __name__ == '__main__':
    main()
