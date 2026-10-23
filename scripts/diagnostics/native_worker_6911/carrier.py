"""Reviewed offhost-only carrier: no runtime imports before source preflight."""
import asyncio
import hashlib
import importlib.metadata
import importlib
import json
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


def compiled_equal(actual, source, filename, source_sha):
    if type(actual) is not types.FunctionType or type(actual.__code__) is not types.CodeType:
        return False
    if hashlib.sha256(source).hexdigest() != source_sha or actual.__code__.co_filename != filename:
        return False
    module = compile(source, filename, 'exec', flags=0, dont_inherit=True,
                     optimize=sys.flags.optimize)
    runs = [c for c in module.co_consts if type(c) is types.CodeType and c.co_name == 'run']
    return len(runs) == 1 and actual.__code__ == runs[0]


def observed_stage(stage, key, operation):
    global STAGE
    STAGE = stage
    value = operation()
    OBSERVED[key] = True
    return value


STAGE = 'SOURCE'
OBSERVED = {}


def build_return(identity, native, source_sha, equal, loaded, data):
    # Called only AFTER asyncio.run returns and thread closure is checked.
    return dict(identity=identity, native_sources=native, producer_source_sha256=source_sha, compiled_equal=equal,
                loaded=loaded, data=data, executor_settled=True, threads_settled=True)


def main():
    global STAGE
    expected = json.loads(os.environ['NATIVE_WORKER_EXPECTED'])
    for p, sha in expected['sources'].items():
        if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != sha:
            raise ValueError('source STOP')
    if native_sources() != expected['native_sources']:
        raise ValueError('native/compiled source STOP')
    for key in ('HOME','LOCAL_OPERATOR_CONFIG_DIR','TMPDIR','LOCAL_OPERATOR_RUNTIME_ROOT'):
        if not Path(os.environ[key]).resolve().is_relative_to(Path(os.environ['NATIVE_WORKER_ROOT']).resolve()):
            raise ValueError('isolation STOP')
    producer = observed_stage('IMPORT_PRODUCER', 'producer_imported',
        lambda: importlib.import_module('scripts.diagnostics.native_worker_6911.producer'))
    worker_code_identity, loop_callback_spans = observed_stage('IMPORT_HELPERS', 'helpers_imported',
        lambda: tuple(importlib.import_module(p) for p in ('tests.worker_code_identity','tests.loop_callback_spans')))
    worker, app, _work_decorator, dom, timer = observed_stage('IMPORT_TEXTUAL', 'textual_imported',
        lambda: tuple(importlib.import_module(p.replace('/', '.')[:-3]) for p in API))
    STAGE = 'REPO_PATHS'
    modules = {'tests/worker_code_identity.py': worker_code_identity,
               'tests/loop_callback_spans.py': loop_callback_spans, RELATIVE: producer}
    loaded = {}
    for p, module in modules.items():
        path = Path(module.__file__).resolve()
        if path != ROOT / p:
            raise ValueError('actual loaded source path STOP')
        loaded[p] = hashlib.sha256(path.read_bytes()).hexdigest()
    OBSERVED['repo_paths'] = True
    STAGE = 'NATIVE_PATHS'
    dist = importlib.metadata.distribution('textual')
    for label, module in zip(API, (worker, app, _work_decorator, dom, timer)):
        path = Path(module.__file__).resolve()
        if path != Path(dist.locate_file(label)).resolve():
            raise ValueError('actual native path STOP')
    OBSERVED['native_paths'] = True
    STAGE = 'COMPILED_COMPARE'
    source = (ROOT / RELATIVE).read_bytes()
    source_sha = hashlib.sha256(source).hexdigest()
    equal = compiled_equal(producer.run, source, str(ROOT / RELATIVE), expected['producer_source_sha256'])
    if equal is not True:
        raise ValueError('compiled structural comparison STOP')
    OBSERVED['compiled_equal'] = True
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
    return build_return(identity, native_sources(), source_sha, equal, loaded, result)


if __name__ == '__main__':
    try:
        proof = main()
        code = 0
    except BaseException:
        proof = {'stage': STAGE, 'observed': dict(OBSERVED)}
        code = 1
    raw = json.dumps(proof)
    if len(raw.encode()) > 262144:
        raw = '{"failure":"OUTPUT_CAP_STOP"}'
        code = 1
    Path(os.environ['NATIVE_WORKER_PROOF']).write_text(raw + '\n')
    raise SystemExit(code)
