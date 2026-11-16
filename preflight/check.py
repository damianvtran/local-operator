import ast
import importlib.util
import json
import os
from pathlib import Path
import sys

repo = Path('/home/ec2-user/local-operator/.worktrees/node-browser-relay-harness')
sys.path.insert(0, str(repo))
import local_operator
assert Path(local_operator.__file__).resolve() == repo / 'local_operator/__init__.py'
print('SOURCE PASS', local_operator.__file__, flush=True)
for name in ['scripts/e2e/node_browser_harness.py', 'scripts/e2e/node_browser_mesh.py', 'tests/unit/scripts/test_node_browser_harness.py']:
    ast.parse((repo / name).read_text(), filename=name)
    print('AST PASS', name, flush=True)
for module in ['PIL', 'pytest', 'httpx', 'uvicorn', 'fastapi', 'pyright', 'flake8']:
    assert importlib.util.find_spec(module), module
    print('DEPENDENCY PASS', module, flush=True)
from scripts.e2e import node_browser_harness as harness
assert harness.CHROME.is_file() and os.access(harness.CHROME, os.X_OK)
extension, crx, version = harness._extension_source()
assert crx.is_file()
print('CRX PASS', extension, version, crx, flush=True)
story = Path('/home/ec2-user/local-operator-ui/.worktrees/sb-pr1833/storybook-static')
assert (story / 'iframe.html').is_file()
index = json.loads((story / 'index.json').read_text())
assert 'browser-approvals-dock--approved' in index['entries']
print('STORYBOOK PASS', story, flush=True)
if '--namespace' not in sys.argv:
    raise SystemExit(0)
assert harness.netns_interfaces() == ['lo']
assert not harness.REAL_POLICY.exists()
assert harness.port_refused(4099)
print('NAMESPACE PASS lo-only; real policy masked; host bridge unreachable', flush=True)
root = Path(os.environ['LOCAL_OPERATOR_CONFIG_DIR']).parent
node, operator = root / 'node', root / 'operator'
for base in [node, operator]:
    for sub in ['home', 'tmp']:
        (base / sub).mkdir(parents=True, exist_ok=True)
from local_operator.operator.trust import anchor_bytes
anchor = harness._make_operator_key(operator, 'preflight-synthetic')
for base in [node, operator]:
    (base / 'fixture-owned-anchor.json').write_bytes(anchor_bytes(anchor))
os.environ.update(LOCAL_OPERATOR_CONFIG_DIR=str(node), HOME=str(node / 'home'), TMPDIR=str(node / 'tmp'))
from tests.fixtures.anchor_fixture import install_fixture_anchor
counts = install_fixture_anchor(node)
from scripts.e2e.node_browser_mesh import MeshRig
mesh = MeshRig(node, operator)
try:
    facts = mesh.start()
    assert facts['signed_binding'] == (mesh.operator_id, mesh.network_id), facts
    assert 'approve' in facts['operator_member_capabilities'], facts
    assert mesh.observe() == []
    print('RELAYS PASS', json.dumps(facts), flush=True)
    from fastapi import FastAPI
    from local_operator.server.routes import desktop_sessions
    from local_operator.server.utils.desktop_sessions import DesktopSessions
    app = FastAPI()
    app.state.desktop_sessions = DesktopSessions(operator)
    app.include_router(desktop_sessions.router)
    # FastAPI keeps included routers lazy in this installed version. Its public
    # OpenAPI view resolves includes, unlike a flat app.routes path scan.
    paths = [path for path in app.openapi()['paths'] if path.endswith('/answers')]
    assert paths == ['/v1/desktop/sessions/{session_id}/answers'], paths
    assert 'post' in app.openapi()['paths'][paths[0]], app.openapi()['paths'][paths[0]]
    print('ROUTE PASS POST', paths[0], flush=True)
    assert counts['native_anchor_reads'] == 0, counts
finally:
    mesh.close()
print('PREFLIGHT BOOTSTRAP PASS; relays closed', flush=True)
