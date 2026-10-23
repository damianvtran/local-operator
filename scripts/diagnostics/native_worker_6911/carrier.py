"""Reviewed offhost-only carrier: no runtime imports before source preflight."""
import asyncio
import hashlib
import importlib.metadata
import json
import marshal
import os
from pathlib import Path
import sys
import threading
import types

ROOT = Path(__file__).resolve().parents[3]
RELATIVE = 'scripts/diagnostics/native_worker_6911/producer.py'
API = ('textual/worker.py', 'textual/app.py', 'textual/_work_decorator.py',
       'textual/dom.py', 'textual/timer.py')


def native_sources():
    dist = importlib.metadata.distribution('textual')
    if dist.version != '8.2.8':
        raise ValueError('native Textual version STOP')
    return {p: hashlib.sha256(Path(dist.locate_file(p)).read_bytes()).hexdigest() for p in API}


def compiled_hash():
    # Independent static compilation BEFORE importing the producer; same exact
    # source filename as the module loader, no arbitrary runtime frame access.
    code = compile((ROOT / RELATIVE).read_bytes(), str(ROOT / RELATIVE), 'exec')
    run = next(x for x in code.co_consts if type(x) is types.CodeType and x.co_name == 'run')
    return hashlib.sha256(marshal.dumps(run)).hexdigest()


STAGE = 'SOURCE'


def build_return(identity, native, compiled, loaded, data):
    # Called only AFTER asyncio.run returns and thread closure is checked.
    return dict(identity=identity, native_sources=native, compiled_producer=compiled,
                loaded=loaded, data=data, executor_settled=True, threads_settled=True)


def main():
    global STAGE
    expected = json.loads(os.environ['NATIVE_WORKER_EXPECTED'])
    for p, sha in expected['sources'].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha:
            raise ValueError('source STOP')
    if native_sources() != expected['native_sources'] or compiled_hash() != expected['compiled_producer']:
        raise ValueError('native/compiled source STOP')
    for key in ('HOME','LOCAL_OPERATOR_CONFIG_DIR','TMPDIR','LOCAL_OPERATOR_RUNTIME_ROOT'):
        if not Path(os.environ[key]).resolve().is_relative_to(Path(os.environ['NATIVE_WORKER_ROOT']).resolve()):
            raise ValueError('isolation STOP')
    STAGE = 'NATIVE_IMPORT'
    from scripts.diagnostics.native_worker_6911 import producer
    from tests import worker_code_identity, loop_callback_spans
    from textual import worker, app, dom, timer, _work_decorator
    modules = {'tests/worker_code_identity.py': worker_code_identity,
               'tests/loop_callback_spans.py': loop_callback_spans, RELATIVE: producer}
    loaded = {}
    for p, module in modules.items():
        path = Path(module.__file__).resolve()
        if path != ROOT / p:
            raise ValueError('actual loaded source path STOP')
        loaded[p] = hashlib.sha256(path.read_bytes()).hexdigest()
    dist = importlib.metadata.distribution('textual')
    for label, module in zip(API, (worker, app, _work_decorator, dom, timer)):
        path = Path(module.__file__).resolve()
        if path != Path(dist.locate_file(label)).resolve():
            raise ValueError('actual native path STOP')
    actual_code = hashlib.sha256(marshal.dumps(producer.run.__code__)).hexdigest()
    if actual_code != expected['compiled_producer']:
        raise ValueError('loaded compiled producer STOP')
    STAGE = 'IDENTITY'
    identity = dict(python=list(sys.version_info[:3]),
        executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
        pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(), sources=loaded)
    from scripts.diagnostics.lfs_pair_6911.controller import validate_identity
    parent = dict(expected['identity'], pid=os.getpid())
    validate_identity(identity, parent)
    # asyncio.run returns AFTER App/worker/sampler cleanup AND default executor
    # shutdown. Do not mistake a worker completion event for a physical join.
    STAGE = 'PRODUCER'
    result = asyncio.run(producer.run(ROOT, RELATIVE))
    STAGE = 'EXECUTOR'
    if threading.active_count() != 1:
        raise ValueError('owned live Python threads STOP')
    STAGE = 'COMPLETE'
    return build_return(identity, native_sources(), actual_code, loaded, result)


if __name__ == '__main__':
    try:
        proof = main()
        code = 0
    except BaseException:
        proof = {'stage': STAGE}
        code = 1
    raw = json.dumps(proof)
    if len(raw.encode()) > 262144:
        raw = '{"failure":"OUTPUT_CAP_STOP"}'
        code = 1
    Path(os.environ['NATIVE_WORKER_PROOF']).write_text(raw + '\n')
    raise SystemExit(code)
