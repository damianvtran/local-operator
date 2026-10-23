"""Bounded PUBLIC source formatting carrier; no runtime/App imports."""
import os
import json
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from scripts.diagnostics.lfs_pair_6911.controller import finish_owned, owned, table
from scripts.diagnostics.format_snapshot_6911.worker import check_sources
ROOT=Path(__file__).resolve().parents[3]


def read_worker_proof(out):
    try:
        raw=(out/'format.json').read_bytes()
        if len(raw)>65536:raise ValueError('metadata cap')
        proof=json.loads(raw)
        if type(proof.get('qualified')) is not bool:raise ValueError('schema')
        return proof
    except Exception:return dict(qualified=False,stage='UNKNOWN',tools=None)


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
        proof=read_worker_proof(out)
        proof['qualified']=proof['qualified'] is True and completed and not state['errors'] and state['synthetic_root_removed'] and state['rc']==0
        proof['lifecycle']={k:state[k] for k in ('rc','reaped','owned_settled','synthetic_root_removed')}
        (out/'format.json').write_text(json.dumps(proof)+'\n')
    if not proof['qualified'] or not completed or state['errors'] or not state['synthetic_root_removed'] or state['rc']!=0:
        raise SystemExit(1)
    for p in ('format.patch','format.json'):
        if not (out/p).is_file():raise SystemExit(1)

if __name__=='__main__':main()
