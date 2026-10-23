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


def tool(command,cwd,deadline):
    known,pgid={},os.getpgrp()
    def observe():return owned(known,pgid)
    def kill():
        for pid,facts in observe().items():
            fresh=table().get(pid)
            if fresh and fresh['start']==facts['start']:os.kill(pid,9)
    child=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
        env={'PATH':os.environ['PATH'],'HOME':str(cwd),'TMPDIR':str(cwd),'BLACK_CACHE_DIR':str(cwd/'cache'),'PYTHONDONTWRITEBYTECODE':'1'})
    completed=False
    try:
        size=0
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout,selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic()>=deadline:raise ValueError('format time STOP')
                observe()
                for key,_ in selector.select(.05):
                    data=os.read(key.fileobj.fileno(),16384)
                    if not data:selector.unregister(key.fileobj)
                    size+=len(data)
                    if size>262144:raise ValueError('format output STOP')
        completed=True
    finally:state=finish_owned(child,observe,kill,lambda:None)
    if not completed or state['errors'] or not state['owned_settled'] or state['rc']!=0:
        raise ValueError('format tool STOP')
    return state['rc']


def main():
    work=Path(sys.argv[1]);work.mkdir();out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'scripts/diagnostics/format_snapshot_6911/manifest.json').read_text())
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
    rcs=[tool([sys.executable,'-B','-m']+cmd,work,deadline) for cmd in commands]
    after={p:(work/p).read_text() for p in FILES}
    for p in FILES:
        if semantic(before[p])!=semantic(after[p]):raise ValueError('AST semantics STOP')
    for name in ('test_s2_send_to_agent_end_keeps_the_loop_under_the_bar','_SlowListing'):
        if protected(before[FILES[2]],name)!=protected(after[FILES[2]],name):
            raise ValueError('protected body bytes STOP')
    patch=''.join(''.join(difflib.unified_diff(before[p].splitlines(keepends=True),after[p].splitlines(keepends=True),fromfile='a/'+p,tofile='b/'+p)) for p in FILES)
    if len(patch.encode())>262144:raise ValueError('patch cap STOP')
    (out/'format.patch').write_text(patch)
    (out/'format.json').write_text(json.dumps(dict(rcs=rcs,ast_compatible=True,protected_byte_equal=True,
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),before={p:hashlib.sha256(before[p].encode()).hexdigest() for p in FILES},after={p:hashlib.sha256(after[p].encode()).hexdigest() for p in FILES}))+'\n')


if __name__=='__main__':main()
