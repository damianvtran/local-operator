"""Bounded PUBLIC source formatting carrier; no runtime/App imports."""
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import sys
import time
from scripts.diagnostics.lfs_pair_6911.controller import finish_owned, owned, table
ROOT=Path(__file__).resolve().parents[3]


def main():
    root=Path(sys.argv[1]);out=Path(sys.argv[2]);out.mkdir(exist_ok=True)
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
    finally:state=finish_owned(child,observe,kill,lambda:shutil.rmtree(root))
    if not completed or state['errors'] or not state['synthetic_root_removed'] or state['rc']!=0:
        raise SystemExit(1)
    for p in ('format.patch','format.json'):
        if not (out/p).is_file():raise SystemExit(1)

if __name__=='__main__':main()
