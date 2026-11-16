"""Read-only re-evaluation of run2's exact tombstones; NOT a native rerun."""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys

repo = Path('/home/ec2-user/local-operator/.worktrees/node-browser-relay-harness')
sys.path.insert(0, str(repo))
import local_operator
assert Path(local_operator.__file__).resolve() == repo / 'local_operator/__init__.py'
from local_operator.notifications.store import NotificationStore
from scripts.e2e.node_browser_mesh import MeshRig

pad = Path('/home/ec2-user/.local-operator/sessions/4b22292b70a8/scratchpad')
run = pad / 'pr2-run2'
raw = run / 'results.json'
source_db = run / 'operator/notifications.db'
output = pad / 'pr2-recorded-state'
output.mkdir(exist_ok=True)
backup_path = output / 'notifications.db'
assert not backup_path.exists(), 'use one fresh snapshot, never overwrite evidence'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
raw_before, db_before = sha(raw), sha(source_db)
with sqlite3.connect(f'file:{source_db}?mode=ro', uri=True) as source:
    with sqlite3.connect(backup_path) as backup:
        source.backup(backup)
# sqlite backup copies committed WAL content too; no key/config/anchor files are
# exported. Only the synthetic notifications database belongs in evidence.
report = json.loads(raw.read_text())
sid = report['facts']['session_id']
node = report['facts']['mesh']['node_device']
failed = [row for row in report['rows'] if 'retires from the operator inbox' in row['row']]
assert len(failed) == 3 and all(row['verdict'] == 'FAIL' for row in failed)
with sqlite3.connect(f'file:{backup_path}?mode=ro', uri=True) as check:
    assert check.execute('SELECT COUNT(*) FROM items').fetchone()[0] == 3
    assert check.execute("SELECT COUNT(*) FROM items WHERE dismissed_at IS NOT NULL AND dismissed_via='remote-attention'").fetchone()[0] == 3
snapshot_before = sha(backup_path)
mesh = MeshRig(output / 'absent-node', output)
mesh.node_id = node
old, _ = NotificationStore(output).list_items(state='all', limit=100)
rows = []
for failed_row in failed:
    key = json.loads(failed_row['actual'])['expected_key']
    ask = key.rsplit(':', 1)[1]
    result = mesh.notice(sid, ask)
    assert result['expected_key'] == key and result['count'] == 1, result
    item = result['items'][0]
    assert item['session_id'] == sid and item['key'] == key
    assert item['dismissed_at'] and item['dismissed_via'] == 'remote-attention'
    old_count = len([item for item in old if item['session_id'] == sid and item['key'] == key])
    assert old_count == 0
    rows.append({
        'original_raw_row': failed_row['row'], 'key': key, 'session_id': sid,
        'source_device': node, 'source_ask': ask, 'old_visible_query_count': old_count,
        'corrected_helper_count': result['count'], 'dismissed_at': item['dismissed_at'],
        'dismissed_via': item['dismissed_via'], 'created_at': item['created_at'],
        'verdict': 'PASS',
    })
assert sha(raw) == raw_before and sha(source_db) == db_before
assert sha(backup_path) == snapshot_before
result = {
    'kind': 'recorded-state re-evaluation, NOT a native rerun',
    'source_head': subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip(),
    'helper_sha256': sha(repo / 'scripts/e2e/node_browser_mesh.py'),
    'raw_run2_sha256': raw_before, 'raw_run2_preserved': True,
    'source_db_sha256': db_before, 'source_db_preserved': True,
    'sqlite_backup_sha256': snapshot_before, 'backup_readback_unchanged': True,
    'native_run2_result': {'rc': 1, 'PASS': 25, 'FAIL': 3},
    'rows': rows,
}
(output / 'correction.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
print('RECORDED-STATE PASS: exact3 existing tombstones; old query0/new helper1 each; raw run2 and source DB unchanged')
