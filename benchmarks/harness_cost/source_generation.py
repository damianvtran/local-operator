"""Standalone source-only generation gate. No Git, credentials, network or builds.

The approved overlay digest is an explicit trust anchor from independent review,
not a value inferred from a receipt beside it. The overlay hashes this validator
and every current public file except itself and the separately checked derived
reconstruction manifest; this is the normal manifest/control-file cycle break.
Passing proves source bytes, NEVER resource admission or runtime qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
R = 'reconstruction/2026-11-14-source-only/'
BASE_COMMIT = '9dee883473441bc1db300abb975cd11297c6ef01'
BASELINE = R + 'generation-baseline-v1.json'
BASELINE_SHA = 'a27405d15656ba385fed1c8b3bfb4752a7658a301483a950c9cb737fe3ad412b'
HISTORICAL = R + 'historical-meter-pre-privacy.py.txt'
OLD_METER_SHA = 'e5d5d626e36e89dc7544eac3f826623fbd264e285dc35f2431fdf35540e7cc5a'
METER_SHA = '23ee793b1c8a4ed71cba1a677bced86f7017899979e787f744ba4619f7270376'
METER_TEST_SHA = '972bad24e91a73e68fc3d0656699965b6b7818dc7afa6a91c16538df2b4bebcc'
METER = 'experimental/meter_proxy.py'
METER_TEST = 'experimental/test_meter_privacy.py'
OVERLAY = R + 'generation-overlay-v4.json'
DERIVED = R + 'SHA256SUMS'
GENERATION = 'reconstruction-20261114-source-v1-live-authority-v4'
# These are the ONLY baseline edits: the reviewed meter fix and minimal gate
# entrypoint/recipe wiring. All other baseline files must remain byte-identical.
WIRING = frozenset({'experimental/coding/driver.py', 'experimental/runner_v2.py',
    R+'lifecycle_proposal.py', R+'RESOLVER-COMMAND.md', R+'BUILD-RECIPES.md',
    R+'resolver-pins.json', R+'test_lifecycle_proposal.py'})
# Existing-file wiring is exact too: resealing an overlay must not authorize
# additional lifecycle/runner behavior beyond this independently reviewed delta.
WIRING_SHA256 = {'experimental/coding/driver.py': 'c74867dbe56502592a9510178728501f06c084608273672d8e745bfb33534c8a',
 'experimental/runner_v2.py': '0c398062d4f93815fd4a95187445f729b4e07d75d5c674f2533a55df5f935549',
 'reconstruction/2026-11-14-source-only/BUILD-RECIPES.md': 'abd483bd04178204af3295d1cb8d13caaab61c350e7ca8cc0e2bb2bf035270a0',
 'reconstruction/2026-11-14-source-only/RESOLVER-COMMAND.md': 'eade643e7c995399ed852496d7a8615f999ee24909dd719b5ebe941c1a024d94',
 'reconstruction/2026-11-14-source-only/lifecycle_proposal.py': 'a17547707db838e7f3a0562c3023421da51c36def31f73ee787489be8372af39',
 'reconstruction/2026-11-14-source-only/resolver-pins.json': '184ef7ffb66c0cb5a612f1d16645df015fedf9e0d57ec9d4129506e2e610932e',
 'reconstruction/2026-11-14-source-only/test_lifecycle_proposal.py': '32038b7ae3985c2ad3aef87e136bf907f50dacfd7a331387a35c38b230078744'}
ADDITIONS = frozenset({METER_TEST, 'source_generation.py',
    'experimental/test_source_generation.py', BASELINE, HISTORICAL, OVERLAY,
    R+'METER-PROVIDER-PRIVACY-v1.md', R+'PROSPECTIVE-METER-PRIVACY-v1.json',
    R+'PROVIDER-PRIVACY-RED-v1.txt', R+'PROVIDER-PRIVACY-GREEN-v1.txt',
    R+'PROVIDER-PRIVACY-STATIC-v1.json', R+'GENERATION-VALIDATION-v2.md',
    R+'GENERATION-VALIDATION-v2-RESULTS.txt', R+'generation-overlay-v2.json',
    R+'final_attempt_admission.py', R+'test_final_attempt_admission.py',
    R+'FINAL-ATTEMPT-ADMISSION-v3.md', R+'FINAL-ATTEMPT-ADMISSION-v3-RESULTS.txt',
    R+'generation-overlay-v3.json', R+'test_resolver_authority_r20.py',
    R+'R20-LIVE-AUTHORITY-v4.md', R+'R20-LIVE-AUTHORITY-v4-RESULTS.txt'})
PRIVATE_FIXTURE = R+'payloads/sanitize-git-repo/tests/test_outputs.py'
PRIVATE_FIXTURE_SHA = '57e4904177acac5807b51f7c32444bde1968ba0445855fa4333d9eb73865b59e'

def require(value, message):
    if not value:
        raise ValueError('generation gate: '+message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def pairs(items):
    result = {}
    for key, value in items:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result

def load(data):
    return json.loads(data, object_pairs_hook=pairs)

def record(data):
    return {'sha256': digest(data), 'bytes': len(data)}

def public_files(root):
    """No Git index dependency; standalone kit and checkout use identical rules.

    Bytecode/VCS metadata is outside this SOURCE-only proof, not runtime proof.
    One immutable local-only public canary is never exported; when present its
    exact original bytes are checked, never used to exempt arbitrary content.
    """
    result = set()
    for path in root.rglob('*'):
        rel = path.relative_to(root).as_posix()
        if any(part in {'.git', '__pycache__'} for part in path.relative_to(root).parts):
            continue
        require(not path.is_symlink(), 'symlink in source kit')
        if path.is_file():
            if rel == PRIVATE_FIXTURE:
                require(digest(path.read_bytes()) == PRIVATE_FIXTURE_SHA, 'private canary changed')
                continue
            result.add(rel)
    return result

def validate(root, manifest, expected_sha256):
    root = Path(root).resolve(strict=True)
    require(type(expected_sha256) is str and re.fullmatch('[0-9a-f]{64}', expected_sha256),
            'explicit independently reviewed overlay SHA256 required')
    manifest = Path(manifest)
    require(manifest == root/OVERLAY and manifest.is_file() and not manifest.is_symlink()
            and manifest.resolve(strict=True) == manifest, 'exact overlay path required')
    raw = manifest.read_bytes()
    require(digest(raw) == expected_sha256, 'overlay SHA256 mismatch')
    overlay = load(raw)
    require(type(overlay) is dict and set(overlay) == {'schema_version','generation','base_commit',
            'baseline_sha256','historical_meter_sha256','runtime_qualified',
            'execution_authorized','changes','files'}, 'unknown overlay record/schema')
    require(type(overlay['schema_version']) is int and overlay['schema_version'] == 4,
            'unsupported generation schema')
    require(overlay['generation'] == GENERATION and overlay['base_commit'] == BASE_COMMIT,
            'unknown generation or wrong base')
    require(overlay['baseline_sha256'] == BASELINE_SHA and
            overlay['historical_meter_sha256'] == OLD_METER_SHA, 'baseline identity changed')
    require(overlay['runtime_qualified'] is False and overlay['execution_authorized'] is False,
            'source receipt cannot grant execution/runtime authority')
    baseline_raw = (root/BASELINE).read_bytes()
    require(digest(baseline_raw) == BASELINE_SHA, 'immutable baseline tampered')
    baseline = load(baseline_raw)
    require(baseline['schema_version'] == 1 and baseline['base_commit'] == BASE_COMMIT,
            'baseline version/base mismatch')
    historical = (root/HISTORICAL).read_bytes()
    old = baseline['files'][METER]
    require(digest(historical) == OLD_METER_SHA and len(historical) == old['bytes'] and
            hashlib.sha1(('blob '+str(len(historical))).encode()+b'\0'+historical).hexdigest()
            == old['git_blob_sha1'], 'exact historical Git object unavailable/tampered')
    expected_paths = set(baseline['files']) | set(ADDITIONS)
    require(public_files(root) == expected_paths, 'missing/unknown source file')
    changed = WIRING | {METER}
    require(overlay['changes'] == sorted(changed), 'undeclared or unknown source delta')
    rows = overlay['files']
    require(type(rows) is dict and set(rows) == expected_paths-{OVERLAY, DERIVED},
            'incomplete/unknown current file manifest')
    for name, row in rows.items():
        require(type(row) is dict and set(row) == {'sha256','bytes'} and
                type(row['bytes']) is int and row['bytes'] >= 0 and
                type(row['sha256']) is str and re.fullmatch('[0-9a-f]{64}',row['sha256']),
                'invalid file hash record')
        data = (root/name).read_bytes()
        require(record(data) == row, 'current source mismatch: '+name)
        if name in baseline['files'] and name not in changed:
            require(row == {k:baseline['files'][name][k] for k in ('sha256','bytes')},
                    'restricted unchanged baseline mismatch: '+name)
    require(all(rows[name]['sha256'] == sha for name,sha in WIRING_SHA256.items()),
            'reviewed wiring identity changed')
    require(rows[METER]['sha256'] == METER_SHA and rows[METER_TEST]['sha256'] == METER_TEST_SHA,
            'reviewed meter/test identity changed')
    # The normal reconstruction manifest hashes the overlay, not vice versa.
    # Verify its entire derived byte representation rather than exempting drift.
    recon = root/R
    paths = sorted(name for name in expected_paths if name.startswith(R) and name != DERIVED)
    expected = ''.join(digest((root/name).read_bytes())+'  '+name[len(R):]+'\n' for name in paths)
    require((root/DERIVED).read_bytes() == expected.encode('utf-8'), 'derived source manifest mismatch')
    return {'generation':GENERATION,'overlay_sha256':expected_sha256,'source_verified':True,
            'baseline_files':len(baseline['files']),'current_files':len(expected_paths),
            'runtime_qualified':False,'execution_authorized':False,'git_required':False}

def validate_selection(manifest, expected_sha256):
    require(manifest is not None, 'explicit NEW generation selection required; old default ineligible')
    return validate(ROOT, manifest, expected_sha256)

def require_execution_authority():
    # The accepted extension is scoped ONLY to final03's protected lifecycle.
    # It cannot authorize paid trials, accounts, models or general builds here.
    raise ValueError('paid/general execution remains HOLD; final03 is permitted only through its reviewed one-shot lifecycle admission, never a source receipt or allow flag')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-generation', type=Path)
    parser.add_argument('--source-generation-sha256')
    args = parser.parse_args()
    try:
        print(json.dumps(validate_selection(args.source_generation,args.source_generation_sha256),sort_keys=True))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2,str(exc)+'\n')

if __name__ == '__main__':
    main()
