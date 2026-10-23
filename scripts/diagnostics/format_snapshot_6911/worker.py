import struct
from types import CodeType
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
    # Only contiguous import groups normalize; executable/docstring barriers
    # and every other AST node stay exact. Duplicate binding order is unsafe.
    def group(statements):
        normalized=[];bound=set()
        for statement in statements:
            for alias in statement.names:
                if alias.name=='*':raise ValueError('ambiguous import STOP')
                name=alias.asname or (alias.name.split('.')[0] if isinstance(statement,ast.Import) else alias.name)
                if name in bound:raise ValueError('duplicate import binding STOP')
                bound.add(name)
                item=ast.alias(name=alias.name,asname=alias.asname)
                normalized.append(ast.Import(names=[item]) if isinstance(statement,ast.Import) else ast.ImportFrom(module=statement.module,names=[item],level=statement.level))
        return sorted(normalized,key=lambda item:ast.dump(item,include_attributes=False))
    def normalize(node):
        for key,value in ast.iter_fields(node):
            if isinstance(value,list):
                for child in value:
                    if isinstance(child,ast.AST):normalize(child)
                output=[];pending=[]
                for child in value:
                    if isinstance(child,(ast.Import,ast.ImportFrom)):pending.append(child)
                    else:
                        if pending:output.extend(group(pending));pending=[]
                        output.append(child)
                if pending:output.extend(group(pending))
                value[:]=output
            elif isinstance(value,ast.AST):normalize(value)
    normalize(tree)
    return ast.dump(tree,include_attributes=False)


def protected(text,name):
    node=next(x for x in ast.parse(text).body if isinstance(x,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and x.name==name)
    start=min([node.lineno]+[d.lineno for d in node.decorator_list])
    return ''.join(text.splitlines(keepends=True)[start-1:node.end_lineno])


S2_NAME='test_s2_send_to_agent_end_keeps_the_loop_under_the_bar'
S2_PATH='tests/unit/test_tui_responsiveness.py'
OBSERVATION_FILE='protected-s2-observation.txt'


def exact_value(left,right):
    """Raw code structure only: line tables/positions are NOT semantics proof."""
    if type(left) is not type(right):return False
    kind=type(left)
    if kind is CodeType:
        fields=('co_code','co_consts','co_names','co_varnames','co_argcount','co_posonlyargcount','co_kwonlyargcount','co_nlocals','co_stacksize','co_flags','co_freevars','co_cellvars','co_name','co_qualname','co_exceptiontable','co_filename')
        results=[exact_value(getattr(left,k),getattr(right,k)) for k in fields]
        return False if False in results else None if None in results else True
    if kind in (str,bytes,int,bool,type(None),type(Ellipsis)):return left==right
    if kind is float:return struct.pack('>d',left)==struct.pack('>d',right)
    if kind is complex:return struct.pack('>dd',left.real,left.imag)==struct.pack('>dd',right.real,right.imag)
    if kind is tuple:
        if len(left)!=len(right):return False
        values=[exact_value(a,b) for a,b in zip(left,right)]
        return False if False in values else None if None in values else True
    if kind is frozenset:
        if len(left)!=len(right):return False
        if len(left)>256:return None
        remaining=list(right)
        for value in left:
            matches=[i for i,other in enumerate(remaining) if exact_value(value,other) is True]
            if not matches:return None if any(exact_value(value,other) is None for other in remaining) else False
            remaining.pop(matches[0])
        return True
    return None


def observation_template():
    return dict(before_ok=None,after_ok=None,compare_ok=None,before_sha256=None,after_sha256=None,byte_equal=None,raw_ast_equal=None,raw_compiled_equal=None,artifact_sha256=None,compiler=dict(python=list(sys.version_info[:3]),executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),flags=0,dont_inherit=True,optimize=0,filename=S2_PATH,scope='RAW_WHOLE_MODULE_COMPILE_NOT_PYTEST_REWRITE'))


def observe_protected(before,after,out,proof):
    observation=observation_template();proof['observation']=observation
    proof['stage']='PROTECTED_S2_EXTRACT_BEFORE'
    original=protected(before,S2_NAME);observation['before_sha256']=hashlib.sha256(original.encode('utf-8')).hexdigest();observation['before_ok']=True
    proof['stage']='PROTECTED_S2_EXTRACT_AFTER'
    candidate=protected(after,S2_NAME);observation['after_sha256']=hashlib.sha256(candidate.encode('utf-8')).hexdigest();observation['after_ok']=True
    proof['stage']='PROTECTED_S2_COMPARE'
    observation['byte_equal']=original==candidate
    def definition(text):
        nodes=[n for n in ast.parse(text).body if isinstance(n,ast.AsyncFunctionDef) and n.name==S2_NAME]
        if len(nodes)!=1:raise ValueError('definition STOP')
        return ast.dump(nodes[0],include_attributes=False)
    observation['raw_ast_equal']=definition(before)==definition(after)
    try:
        roots=[]
        for text in (before,after):
            code=compile(text,S2_PATH,'exec',flags=0,dont_inherit=True,optimize=0)
            found=[c for c in code.co_consts if type(c) is CodeType and c.co_name==S2_NAME and c.co_qualname==S2_NAME]
            if len(found)!=1:raise ValueError('compiled scope STOP')
            roots.append(found[0])
        observation['raw_compiled_equal']=exact_value(*roots)
    except (SyntaxError,ValueError,TypeError,OverflowError):observation['raw_compiled_equal']=None
    data=''.join(difflib.unified_diff(original.splitlines(keepends=True),candidate.splitlines(keepends=True),fromfile='a/'+S2_PATH,tofile='b/'+S2_PATH)).encode('utf-8')
    if len(data)>262144 or b'\0' in data:raise ValueError('observation cap STOP')
    path=out/OBSERVATION_FILE
    if path.is_symlink() or path.exists():raise ValueError('observation destination STOP')
    path.write_bytes(data)
    if path.is_symlink() or not path.is_file() or path.read_bytes()!=data:raise ValueError('observation readback STOP')
    observation['artifact_sha256']=hashlib.sha256(data).hexdigest();observation['compare_ok']=True
    return observation['byte_equal']


def read_observation(value,out):
    """Reject unknown/private keys; public source diff is observation, NOT patch."""
    if value is None:return None
    keys={'before_ok','after_ok','compare_ok','before_sha256','after_sha256','byte_equal','raw_ast_equal','raw_compiled_equal','artifact_sha256','compiler'}
    if type(value) is not dict or set(value)!=keys:raise ValueError('observation schema STOP')
    for key in ('before_ok','after_ok','compare_ok','byte_equal','raw_ast_equal','raw_compiled_equal'):
        if value[key] is not None and type(value[key]) is not bool:raise ValueError('observation bool STOP')
    for key in ('before_sha256','after_sha256','artifact_sha256'):
        v=value[key]
        if v is not None and (type(v) is not str or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)):raise ValueError('observation hash STOP')
    compiler=value['compiler']
    if type(compiler) is not dict or set(compiler)!={'python','executable_sha256','flags','dont_inherit','optimize','filename','scope'}:raise ValueError('compiler schema STOP')
    if type(compiler['python']) is not list or len(compiler['python'])!=3 or any(type(v) is not int or v<0 for v in compiler['python']):raise ValueError('compiler Python STOP')
    sha=compiler['executable_sha256']
    if type(sha) is not str or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):raise ValueError('compiler SHA STOP')
    if type(compiler['flags']) is not int or compiler['flags']!=0 or type(compiler['optimize']) is not int or compiler['optimize']!=0 or compiler['dont_inherit'] is not True or compiler['filename']!=S2_PATH or compiler['scope']!='RAW_WHOLE_MODULE_COMPILE_NOT_PYTEST_REWRITE':raise ValueError('compiler context STOP')
    if value['compare_ok'] is True:
        if not all(value[k] is True for k in ('before_ok','after_ok')) or any(value[k] is None for k in ('before_sha256','after_sha256','byte_equal','raw_ast_equal','artifact_sha256')):raise ValueError('observation complete STOP')
        path=out/OBSERVATION_FILE
        if path.is_symlink() or not path.is_file() or path.stat().st_size>262144:raise ValueError('observation file STOP')
        data=path.read_bytes();text=data.decode('utf-8')
        if hashlib.sha256(data).hexdigest()!=value['artifact_sha256'] or b'\0' in data:raise ValueError('observation data STOP')
        for line in text.splitlines():
            if line.startswith('--- ') and line!='--- a/'+S2_PATH:raise ValueError('observation path STOP')
            if line.startswith('+++ ') and line!='+++ b/'+S2_PATH:raise ValueError('observation path STOP')
            if line.startswith(('diff --git','GIT binary patch','Binary files','old mode','new mode')):raise ValueError('observation format STOP')
    elif value['artifact_sha256'] is not None:raise ValueError('premature observation STOP')
    return dict(value)


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
    PROOF['stage']='READBACK'
    after={p:(work/p).read_text() for p in FILES}
    for index,p in enumerate(FILES):
        PROOF['stage']='AST_COMPARE_'+str(index)
        if semantic(before[p])!=semantic(after[p]):raise ValueError('AST semantics STOP')
    for name in ('test_s2_send_to_agent_end_keeps_the_loop_under_the_bar','_SlowListing'):
        PROOF['stage']='PROTECTED_S2' if name.startswith('test_') else 'PROTECTED_SLOW'
        equal=observe_protected(before[FILES[2]],after[FILES[2]],out,PROOF) if name==S2_NAME else protected(before[FILES[2]],name)==protected(after[FILES[2]],name)
        if not equal:
            raise ValueError('protected body bytes STOP')
    PROOF['stage']='PATCH_BUILD'
    patch=''.join(''.join(difflib.unified_diff(before[p].splitlines(keepends=True),after[p].splitlines(keepends=True),fromfile='a/'+p,tofile='b/'+p)) for p in FILES)
    PROOF['stage']='PATCH_CAP'
    if len(patch.encode())>262144:raise ValueError('patch cap STOP')
    PROOF['stage']='PATCH_READY'
    (out/'format.patch').write_text(patch)
    PROOF.update(qualified=True,stage='COMPLETE',ast_compatible=True,protected_byte_equal=True,
        patch_sha256=hashlib.sha256(patch.encode()).hexdigest(),before={p:hashlib.sha256(before[p].encode()).hexdigest() for p in FILES},after={p:hashlib.sha256(after[p].encode()).hexdigest() for p in FILES})


def entry():
    global PROOF
    out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
    PROOF=dict(qualified=False,stage='SOURCE_SETUP',observation=None,tools=[dict(tool=k,reason='UNRUN',rc=None) for k in ('isort-format', 'black-format', 'black-check', 'isort-check', 'flake8-check')])
    try:main()
    except Exception:pass
    finally:
        (out/'format.json').write_text(json.dumps(PROOF)+'\n')
    if not PROOF['qualified']:raise SystemExit(1)

if __name__=='__main__':entry()
