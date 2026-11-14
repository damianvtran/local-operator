"""Derive gap inventories from retained bytes; never synthesize historical evidence."""
import collections
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENTAL = HERE.parents[1] / 'experimental'
CAMPAIGN = 'reconstruction-20261114-source-v1'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    prereg_path = EXPERIMENTAL / 'PREREGISTRATION-HEADLINE-2026-11-09-v2.json'
    prereg = json.loads(prereg_path.read_text())
    source_receipt = sorted(HERE.glob('source-receipt-*.jsonl'))[-1]
    rows = [json.loads(line) for line in source_receipt.read_text().splitlines()]
    matrix = []
    for task in prereg['tasks']:
        files = [row for row in rows if row['task'] == task]
        counts = dict(collections.Counter(row['status'] for row in files))
        matrix.append({'task': task, 'source_commit': files[0]['source_commit'],
                       'source_commits': sorted({row['source_commit'] for row in files}),
                       'counts': counts, 'complete_exact': all(row['status'] == 'exact_match' for row in files),
                       'gaps': [row for row in files if row['status'] != 'exact_match']})
    write('task-matrix.json', {'campaign_id': CAMPAIGN, 'receipt': source_receipt.name,
                             'files': len(rows), 'tasks': matrix})
    subset_path = EXPERIMENTAL / 'coding/subset.json'
    subset = json.loads(subset_path.read_text())
    known = {row['name']: row for row in subset['tasks']}
    public = json.loads((EXPERIMENTAL / 'coding/public-provenance.json').read_text())['tasks']
    names = prereg['headline_sample']['dev_tasks_excluded'] + prereg['tasks']
    reconstructed = []
    image_path = HERE / 'image-bindings.json'
    images = {row['task']: row for row in json.loads(image_path.read_text())['observations']} if image_path.exists() else {}
    for name in names:
        row = {'name': name, 'selection': 'dev' if name in prereg['headline_sample']['dev_tasks_excluded'] else 'held-out',
               'historical_metadata': known.get(name), 'image_digest': known.get(name, {}).get('image_digest') or public.get(name, {}).get('image_digest'),
               'missing_fields': [] if name in known else ['task8', 'image tag', 'hub_tag_full_size_bytes', 'task_checksum'],
               'source': 'surviving subset' if name in known else 'prereg task list; optional public-provenance image binding'}
        if name in images:
            observation = images[name]
            # Fresh tag metadata is useful prospective acquisition identity, but
            # must never masquerade as recovered original subset/runtime bytes.
            row['new_image_observation'] = observation
            if row['image_digest'] is None:
                row['image_digest'] = observation['image_digest']
                row['image_digest_origin'] = 'new-public-registry-observation'
        if row['image_digest'] is None:
            row['missing_fields'].append('image_digest')
        reconstructed.append(row)
    write('derived-subset.json', {'campaign_id': CAMPAIGN, 'historical_exact_recovery': False,
        'historical_expected_sha256': prereg['subset_sha256'], 'surviving_actual_sha256': digest(subset_path),
        'surviving_count': len(known), 'derived_count': len(reconstructed), 'tasks': reconstructed,
        'note': 'This is a new derivation, never the missing frozen subset bytes.'})
    checks = []
    for field, base in [('source_sha256', EXPERIMENTAL / 'coding'), ('container_closure_sha256', EXPERIMENTAL / 'coding')]:
        for name, expected in prereg[field].items():
            path = base / name
            observed = digest(path) if path.is_file() else None
            checks.append({'binding': field, 'path': str(path.relative_to(HERE.parents[1])), 'expected_sha256': expected,
                           'observed_sha256': observed, 'status': 'exact_match' if observed == expected else 'missing' if observed is None else 'mismatch'})
    write('historical-bindings.json', {'preregistration_sha256': digest(prereg_path), 'checks': checks,
        'external_missing_bindings': {k: prereg[k] for k in ('builder_sha256', 'runtime_sha256', 'oracle_sha256', 'toolwire_sha256', 'catalog_sha256', 'prereg_tools_sha256')},
        'note': 'External digests are historical assertions, not recovered artifacts or fresh validation.'})
    # Incremental observations do not erase earlier package pins. Keep the
    # latest observation of each named source across all append-only receipts.
    metadata_rows = {}
    for path in sorted(HERE.glob('metadata-receipt-*.jsonl')):
        for line in path.read_text().splitlines():
            receipt = json.loads(line)
            metadata_rows[receipt['name']] = receipt
    pins = []
    for receipt in metadata_rows.values():
        item = dict(receipt)
        if receipt['status'] == 'available' and receipt['file'].endswith('.json'):
            doc = json.loads((HERE / receipt['file']).read_text())
            if isinstance(doc, dict) and 'dist' in doc:
                item.update(package=doc.get('name'), version=doc.get('version'), dist=doc['dist'],
                            dependencies=doc.get('dependencies', {}), optional_dependencies=doc.get('optionalDependencies', {}),
                            dependency_closure_complete=False)
            elif isinstance(doc, dict) and 'info' in doc:
                item.update(package=doc['info']['name'], version=doc['info']['version'],
                            requires_dist=doc['info']['requires_dist'],
                            distributions=[{'filename': row['filename'], 'url': row['url'], 'sha256': row['digests']['sha256']} for row in doc['urls']],
                            dependency_closure_complete=False)
        pins.append(item)
    write('distribution-pins.partial.json', {'campaign_id': CAMPAIGN, 'runtime_installable_lock': False,
          'pins': pins, 'note': 'Static distribution hashes only; proposed latest host releases are not historical runtime pins. No resolver or install was run.'})
    lines = ['# Restored task matrix', '', '| Task | Exact files | Mismatch | Not collected | Fully exact |', '|---|---:|---:|---:|---|']
    for task in matrix:
        c = task['counts']
        lines.append(f"| {task['task']} | {c.get('exact_match', 0)} | {c.get('mismatch', 0)} | {sum(v for k,v in c.items() if k not in ('exact_match','mismatch'))} | {task['complete_exact']} |")
    (HERE / 'TASK-MATRIX.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'files': len(rows), 'file_statuses': dict(collections.Counter(row['status'] for row in rows)),
                      'fully_exact_tasks': sum(t['complete_exact'] for t in matrix), 'historical_checks': dict(collections.Counter(c['status'] for c in checks)),
                      'derived_subset_tasks': len(reconstructed), 'surviving_subset_tasks': len(known)}, sort_keys=True))


if __name__ == '__main__':
    main()
