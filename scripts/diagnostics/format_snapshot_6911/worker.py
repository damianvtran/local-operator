"""Offhost PUBLIC source formatting only; no product/App imports."""
import ast
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from scripts.diagnostics.lfs_pair_6911.controller import finish_owned, owned, table

ROOT=Path(__file__).resolve().parents[3]
FILES=('tests/worker_code_identity.py','tests/unit/test_worker_code_identity.py',
       'tests/unit/test_tui_responsiveness.py','tests/loop_callback_spans.py')


def semantic(text):
    tree=ast.parse(text)
    # isort changes ONLY import order, including imports inside functions.
    def normalize(node):
        for key,value in ast.iter_fields(node):
            if isinstance(value,list):
                imports=[x for x in value if isinstance(x,(ast.Import,ast.ImportFrom))]
                if imports:
                    imports.sort(key=lambda x:ast.dump(x,include_attributes=False))
                    iterator=iter(imports)
                    value[:]=[next(iterator) if isinstance(x,(ast.Import,ast.ImportFrom)) else x for x in value]
                for child in value:
                    if isinstance(child,ast.AST):normalize(child)
            elif isinstance(value,ast.AST):normalize(value)
    normalize(tree)
    return ast.dump(tree,include_attributes=False)


def protected(text,name):
    node=next(x for x in ast.parse(text).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and x.name==name)
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])
    return ''.join(text.splitlines(keepends=True)[start-1:node.end_lineno])


def check_sources(root, manifest):
    expected={'scripts/diagnostics/format_snapshot_6911/worker.py','scripts/diagnostics/format_snapshot_6911/controller.py','scripts/diagnostics/lfs_pair_6911/controller.py'}
    if set(manifest['sources'])!=expected:raise ValueError('source schema STOP')
    for name,sha in manifest['sources'].items():
        path=root/name
        if path.resolve()!=path or not path.is_file() or any(x.is_symlink() for x in (path,*path.parents)) or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
            raise ValueError('source binding STOP')


def nested_tool_observer(child):
    """Nested ownership is a birth-bound observed subtree, NEVER inherited PG.

    The outer controller owns the PG. Borrowing that boundary here includes
    our own ancestors; successful formatter reaping must not kill its parent.
    Unknown/reused identities latch failure even if later observations settle.
    """
    known={}
    failed=False
    try:
        initial=table().get(child.pid)
        if initial is None or type(initial['start']) is not str or not initial['start']:
            raise ValueError('birth unknown')
        known[child.pid]=initial['start']
    except Exception:failed=True
    def observe():
        nonlocal failed
        try:
            if failed:raise ValueError('ownership unknown')
            rows=table()
            if any(pid in rows and rows[pid]['start']!=birth for pid,birth in known.items()):
                raise ValueError('identity reused')
            changed=True
            while changed:
                changed=False
                for pid,row in rows.items():
                    parent=row['ppid']
                    if pid not in known and parent in known and parent in rows and rows[parent]['start']==known[parent]:
                        known[pid]=row['start']
                        changed=True
            return {pid:rows[pid] for pid,birth in known.items() if pid in rows and rows[pid]['start']==birth}
        except Exception:
            failed=True
            raise
    def kill_owned():
        nonlocal failed
        try:
            for pid,row in observe().items():
                fresh=table().get(pid)
                if fresh is None:continue
                if fresh['start']!=row['start']:raise ValueError('kill identity reused')
                os.kill(pid,9)
        except Exception:
            failed=True
            raise
    return observe,kill_owned


def tool(command,cwd,deadline):
    status=dict(reason='LAUNCH_FAILURE',rc=None,launched=False,timeout=False,cap=False,reaped=False,settled=False,monitor_failed=False,completed=False)
    try:child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        env={'PATH':os.environ['PATH'],'HOME':str(cwd),'TMPDIR':str(cwd),'BLACK_CACHE_DIR':str(cwd/'cache'),'PYTHONDONTWRITEBYTECODE':'1'})
    except Exception:return status
    status['launched']=True
    observe,kill=nested_tool_observer(child)
    completed=False
    try:
        observe()
        size=0
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout,selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic()>=deadline:
                    status['timeout']=True
                    raise ValueError('time')
                observe()
                for key,_ in selector.select(.05):
                    data=os.read(key.fileobj.fileno(),16384)
                    if not data:selector.unregister(key.fileobj)
                    size+=len(data)
                    if size>262144:
                        status['cap']=True
                        raise ValueError('cap')
        completed=True
    except Exception:status['monitor_failed']=not(status['timeout'] or status['cap'])
    finally:
        state=finish_owned(child,observe,kill,lambda:None)
        status.update(rc=state['rc'],reaped=state['reaped'],settled=state['owned_settled'])
    status['completed']=completed
    status['reason']=('TIMEOUT' if status['timeout'] else 'OUTPUT_CAP' if status['cap'] else 'MONITOR_FAILURE' if not completed or status['monitor_failed'] else 'SETTLEMENT_FAILURE' if state['errors'] or not state['owned_settled'] else 'PASS' if state['rc']==0 else 'TOOL_FAILURE')
    return status


def run_tools(commands,work,deadline,proof,out):
    for i,cmd in enumerate(commands):
        proof['stage']='TOOLS'
        label=proof['tools'][i]['tool']
        proof['tools'][i]=dict(tool=label,**tool([sys.executable,'-B','-m']+cmd,work,deadline))
        (out/'format.json').write_text(json.dumps(proof)+'\n')
        if proof['tools'][i]['reason']!='PASS':raise ValueError('tool STOP')


def main():
    work=Path(sys.argv[1]);work.mkdir();out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'scripts/diagnostics/format_snapshot_6911/manifest.json').read_text())
    check_sources(ROOT,manifest)
    actual_tools={k:importlib.metadata.version(k) for k in ('black','isort','flake8')}
    if actual_tools != manifest['tools']:raise ValueError('tool version STOP')
    before={}
    for p,sha in manifest['snapshot'].items():
        source=ROOT/p
        if source.is_symlink() or not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest()!=sha:
            raise ValueError('frozen snapshot STOP')
        target=work/p;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        if p in FILES:before[p]=source.read_text()
    deadline=time.monotonic()+25
    commands=[['isort',*FILES],['black','--workers','1',*FILES],
              ['black','--check','--workers','1',*FILES],['isort','--check-only',*FILES],['flake8','--jobs','1',*FILES]]
    run_tools(commands,work,deadline,PROOF,out)
    PROOF['stage']='COMPATIBILITY'
    after={p:(work/p).read_text() for p in FILES}
    for p in FILES:
        if semantic(before[p])!=semantic(after[p]):raise ValueError('AST semantics STOP')
    for name in ('test_s2_send_to_agent_end_keeps_the_loop_under_the_bar','_SlowListing'):
        if protected(before[FILES[2]],name)!=protected(after[FILES[2]],name):
            raise ValueError('protected body bytes STOP')
    patch=''.join(''.join(difflib.unified_diff(before[p].splitlines(keepends=True),after[p].splitlines(keepends=True),fromfile='a/'+p,tofile='b/'+p)) for p in FILES)
    if len(patch.encode())>262144:raise ValueError('patch cap STOP')
    PROOF['stage']='PATCH_READY'
    (out/'format.patch').write_text(patch)
    PROOF.update(qualified=True,stage='COMPLETE',ast_compatible=True,protected_byte_equal=True,
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),before={p:hashlib.sha256(before[p].encode()).hexdigest() for p in FILES},after={p:hashlib.sha256(after[p].encode()).hexdigest() for p in FILES})


def entry():
    global PROOF
    out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
    PROOF=dict(qualified=False,stage='SOURCE_SETUP',tools=[dict(tool=k,reason='UNRUN',rc=None) for k in ('isort-format', 'black-format', 'black-check', 'isort-check', 'flake8-check')])
    try:main()
    except Exception:pass
    finally:
        (out/'format.json').write_text(json.dumps(PROOF)+'\n')
    if not PROOF['qualified']:raise SystemExit(1)

if __name__=='__main__':entry()
