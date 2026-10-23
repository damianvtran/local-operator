"""Bounded PUBLIC source formatting carrier; no runtime/App imports."""
import os
import json
import hashlib
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from scripts.diagnostics.lfs_pair_6911.controller import finish_owned, owned, table
from scripts.diagnostics.format_snapshot_6911.worker import check_sources
ROOT=Path(__file__).resolve().parents[3]


def read_worker_proof(out,expected_before):
    unknown=dict(qualified=False,stage='UNKNOWN',tools=None)
    try:
        path=out/'format.json'
        if path.is_symlink() or not path.is_file() or path.stat().st_size>65536:return unknown
        proof=json.loads(path.read_bytes())
        base={'qualified','stage','tools'}
        success={'ast_compatible','protected_byte_equal','patch_sha256','before','after'}
        if type(proof) is not dict or type(proof.get('qualified')) is not bool:return unknown
        if set(proof)!=(base|success if proof['qualified'] else base):return unknown
        if proof['stage'] not in ('SOURCE_SETUP','TOOLS','COMPATIBILITY','READBACK','AST_COMPARE_0','AST_COMPARE_1','AST_COMPARE_2','AST_COMPARE_3','PROTECTED_S2','PROTECTED_SLOW','PATCH_BUILD','PATCH_CAP','PATCH_READY','COMPLETE'):return unknown
        labels=('isort-format','black-format','black-check','isort-check','flake8-check')
        tools=proof['tools']
        if type(tools) is not list or len(tools)!=5:return unknown
        seen_unrun=False
        flags={'launched','timeout','cap','reaped','settled','monitor_failed','completed'}
        for item,label in zip(tools,labels):
            if type(item) is not dict or item.get('tool')!=label:return unknown
            if item.get('reason')=='UNRUN':
                if set(item)!={'tool','reason','rc'} or item['rc'] is not None:return unknown
                seen_unrun=True
                continue
            if seen_unrun or set(item)!=flags|{'tool','reason','rc'}:return unknown
            if any(type(item[k]) is not bool for k in flags):return unknown
            if item['rc'] is not None and type(item['rc']) is not int:return unknown
            if item['reason'] not in ('PASS','TOOL_FAILURE','LAUNCH_FAILURE','TIMEOUT','OUTPUT_CAP','MONITOR_FAILURE','SETTLEMENT_FAILURE'):return unknown
            if item['reason']=='PASS' and not (item['rc']==0 and type(item['rc']) is int and all(item[k] for k in ('launched','completed','reaped','settled')) and not any(item[k] for k in ('timeout','cap','monitor_failed'))):return unknown
        if not proof['qualified']:
            if proof['stage']=='COMPLETE':return unknown
            return dict(qualified=False,stage=proof['stage'],tools=tools)
        if proof['stage']!='COMPLETE' or any(x['reason']!='PASS' for x in tools):return unknown
        if proof['ast_compatible'] is not True or proof['protected_byte_equal'] is not True:return unknown
        def sha(v):return type(v) is str and len(v)==64 and all(c in '0123456789abcdef' for c in v)
        if proof['before']!=expected_before or type(proof['after']) is not dict or set(proof['after'])!=set(expected_before):return unknown
        if not all(sha(v) for v in proof['after'].values()) or not sha(proof['patch_sha256']):return unknown
        patch=out/'format.patch'
        if patch.is_symlink() or not patch.is_file() or patch.stat().st_size>262144:return unknown
        data=patch.read_bytes()
        if hashlib.sha256(data).hexdigest()!=proof['patch_sha256'] or b'\x00' in data:return unknown
        text=data.decode('utf-8')
        for line in text.splitlines():
            if line.startswith('--- ') and line[4:] not in {'a/'+p for p in expected_before}:return unknown
            if line.startswith('+++ ') and line[4:] not in {'b/'+p for p in expected_before}:return unknown
            if line.startswith(('diff --git','GIT binary patch','Binary files','old mode','new mode')):return unknown
        return {k:proof[k] for k in base|success}
    except Exception:return unknown


def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
    proof=dict(qualified=False,stage='UNKNOWN',tools=None)
    (out/'format.json').write_text(json.dumps(proof)+'\n')
    manifest=json.loads((ROOT/'scripts/diagnostics/format_snapshot_6911/manifest.json').read_text())
    check_sources(ROOT,manifest)
    known,pgid={},os.getpgrp()
    def observe():return owned(known,pgid)
    def kill():
        for pid,facts in observe().items():
            fresh=table().get(pid)
            if fresh and fresh['start']==facts['start']:os.kill(pid,9)
    child=subprocess.Popen([sys.executable,'-B','-m','scripts.diagnostics.format_snapshot_6911.worker',str(root),str(out)],cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    completed=False
    try:
        size=0;deadline=time.monotonic()+30
        with selectors.DefaultSelector() as selector:
            selector.register(child.stdout,selectors.EVENT_READ)
            while selector.get_map():
                if time.monotonic()>=deadline:raise ValueError('format child bound STOP')
                observe()
                for key,_ in selector.select(.05):
                    data=os.read(key.fileobj.fileno(),16384)
                    if not data:selector.unregister(key.fileobj)
                    size+=len(data)
                    if size>262144:raise ValueError('format output cap STOP')
        completed=True
    except Exception:pass
    finally:
        state=finish_owned(child,observe,kill,lambda:shutil.rmtree(root) if root.exists() else None)
        proof=read_worker_proof(out,{k:v for k,v in manifest['snapshot'].items() if k.endswith('.py')})
        proof['qualified']=proof['qualified'] is True and completed and not state['errors'] and state['synthetic_root_removed'] and state['rc']==0
        proof['lifecycle']={k:state[k] for k in ('rc','reaped','owned_settled','synthetic_root_removed')}
        (out/'format.json').write_text(json.dumps(proof)+'\n')
    if not proof['qualified'] or not completed or state['errors'] or not state['synthetic_root_removed'] or state['rc']!=0:
        raise SystemExit(1)
    for p in ('format.patch','format.json'):
        if not (out/p).is_file():raise SystemExit(1)

if __name__=='__main__':main()
