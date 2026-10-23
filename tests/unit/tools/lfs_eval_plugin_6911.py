"""QA-owned SOURCE CANDIDATE: original eval heartbeat cell, two sleep variants.
Integrate only after independent review. Never import/run on the operator host.
No copied test, no threshold/block_s rewrite, no arbitrary frame locals/messages.
Controller must reuse the qualified owned-PG lifecycle and module-root entry.
"""
import hashlib
import json
import os
import inspect
import sys
import threading
import time
import types
from pathlib import Path

import pytest
from scripts.diagnostics.lfs_pair_6911.controller import validate_identity
from scripts.diagnostics.lfs_eval_6911.controller import validate_row

NODE = 'tests/unit/tools/test_eval_tool.py::test_render_does_not_block_the_event_loop'
# Integrator must populate this from the frozen Git blob, not a runtime self-hash.
EXPECTED_TEST_SHA = '2fbfcd580375c92ef69fe277900e69398b98f12d967fda9b99adb5964dcaa50c'
ROWS = []
REPORTS = []
CALLS = 0


@pytest.fixture(autouse=True, params=('baseline', 'noinject'), ids=('baseline', 'noinject'))
def _eval_sleep_variant(request):
    return request.param


def collection_modes(items):
    if len(items) != 2 or len({id(x) for x in items}) != 2:
        raise RuntimeError('EVAL distinct two items required')
    modes = [x.callspec.params['_eval_sleep_variant'] for x in items]
    if modes != ['baseline', 'noinject'] or any(x.nodeid.split('[', 1)[0] != NODE for x in items):
        raise RuntimeError('EVAL original ordered selectors required')
    return modes


def pytest_collection_modifyitems(items):
    collection_modes(items)


def spill_code(fn):
    # This trusted nested code object is the exact injected-sleep caller. No
    # function-name-only filter, arbitrary frame introspection or locals capture.
    found = [x for x in fn.__code__.co_consts if isinstance(x, types.CodeType)
             and x.co_name == '_blocking_spill_truncate']
    if len(found) != 1:
        raise RuntimeError('EVAL trusted sleep callsite missing')
    return found[0]


def matches_sleep(caller_code, trusted_code, seconds):
    return caller_code is trusted_code and type(seconds) in (float, int) and seconds == 0.3


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    global CALLS
    CALLS += 1
    mode = item.callspec.params['_eval_sleep_variant']
    if CALLS > 2 or mode != ('baseline', 'noinject')[CALLS - 1]:
        raise RuntimeError('EVAL actual call order/cardinality STOP')
    fn = inspect.unwrap(getattr(item.module, 'test_render_does_not_block_the_event_loop'))
    source = Path(fn.__code__.co_filename).resolve()
    if hashlib.sha256(source.read_bytes()).hexdigest() != EXPECTED_TEST_SHA:
        raise RuntimeError('EVAL source binding STOP')
    loaded_identity = verify_identity(item.module)
    code = spill_code(fn)
    original_sleep = time.sleep
    row = dict(mode=mode, token=CALLS, injected_calls=0, delegated=0, skipped=0,
               active=0, loop_thread=threading.get_ident(), spill_threads=[],
               restored=False, call_pass=False, assertion=False, heartbeat_assertion=False, identity=loaded_identity)

    def observed_sleep(seconds):
        caller = sys._getframe(1).f_code
        if not matches_sleep(caller, code, seconds):
            return original_sleep(seconds)
        row['injected_calls'] += 1
        row['spill_threads'].append(threading.get_ident())
        row['active'] += 1
        try:
            if mode == 'noinject':
                row['skipped'] += 1
                return None
            row['delegated'] += 1
            return original_sleep(seconds)
        finally:
            row['active'] -= 1

    try:
        time.sleep = observed_sleep
        outcome = yield
        row['call_pass'] = outcome.excinfo is None
        if outcome.excinfo is not None:
            exc = outcome.excinfo[1]
            row['assertion'] = isinstance(exc, AssertionError)
            # Numeric trusted-source location only, never exception message or
            # frame locals. Distinguish the heartbeat oracle from is_error.
            tb = exc.__traceback__
            while tb is not None:
                if tb.tb_frame.f_code is fn.__code__ and tb.tb_lineno == 870:
                    row['heartbeat_assertion'] = row['assertion']
                tb = tb.tb_next
    finally:
        time.sleep = original_sleep
        row['restored'] = time.sleep is original_sleep
        ROWS.append(row)
        if row['active'] != 0:
            raise RuntimeError('EVAL injected sleep still active at restore STOP')


def pytest_runtest_logreport(report):
    REPORTS.append(dict(phase=report.when, outcome=report.outcome))
    if report.when == 'call':
        try:
            if not ROWS or not 1 <= CALLS <= 2 or len(ROWS) != CALLS:
                raise ValueError('EVAL missing completed call STOP')
            validate_row(ROWS[-1], ('baseline', 'noinject')[CALLS - 1], CALLS, report.outcome)
        except (ValueError, KeyError, TypeError, IndexError):
            SESSION.shouldstop = 'EVAL invalid completed call STOP'
    elif report.outcome != 'passed':
        SESSION.shouldstop = 'EVAL setup/fixture STOP'




def qualify_rows(rows, reports):
    """Controller calls AFTER full module fixtures and owned-process settlement.
    Identity/isolation/source-binding validation is MANDATORY before this helper;
    this helper is not a replacement controller or a lifecycle proof.
    """
    if len(rows) != 2 or [r['mode'] for r in rows] != ['baseline', 'noinject'] or [r['token'] for r in rows] != [1, 2]:
        raise ValueError('EVAL missing/order/duplicate STOP')
    if len(reports) != 6 or [r['phase'] for r in reports] != ['setup', 'call', 'teardown'] * 2:
        raise ValueError('EVAL phases STOP')
    if any(r['outcome'] != 'passed' for r in reports if r['phase'] != 'call'):
        raise ValueError('EVAL fixture STOP')
    for i, row in enumerate(rows):
        validate_row(row, ('baseline', 'noinject')[i], i + 1, reports[3 * i + 1]['outcome'])
    if all(r['call_pass'] for r in rows):
        return 'INCONCLUSIVE_NOT_EXONERATED'
    if all(not r['call_pass'] for r in rows):
        return 'INJECTION_NOT_NECESSARY_IN_THIS_COHORT_ROOT_UNKNOWN'
    return 'N1_VARIANT_DIFFERENCE_NOT_CAUSAL_PROOF'


SESSION = None


def pytest_sessionstart(session):
    global SESSION
    SESSION = session


def verify_identity(module):
    import tests.conftest as policy
    root = Path(__file__).resolve().parents[3]
    files = {'tests/unit/tools/test_eval_tool.py': module.__file__,
             'tests/unit/tools/lfs_eval_plugin_6911.py': __file__,
             'tests/conftest.py': policy.__file__,
             'local_operator/tools/eval.py': module.eval_tool.__file__,
             'local_operator/tools/builtin.py': module.builtin.__file__}
    files['scripts/diagnostics/lfs_eval_6911/controller.py'] = sys.modules[validate_row.__module__].__file__
    hashes = {}
    for label, actual in files.items():
        path = Path(actual).resolve()
        if path != root / label:
            raise RuntimeError('EVAL loaded path STOP')
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = dict(python=list(sys.version_info[:3]),
                  executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                  pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(), sources=hashes)
    expected = json.loads(os.environ['LFS_EVAL_EXPECTED'])
    expected['pid'] = os.getpid()
    validate_identity(actual, expected)
    bound = Path(os.environ['LFS_EVAL_ROOT']).resolve()
    for key in ('HOME', 'LOCAL_OPERATOR_CONFIG_DIR', 'TMPDIR', 'LOCAL_OPERATOR_RUNTIME_ROOT'):
        if not Path(os.environ[key]).resolve().is_relative_to(bound):
            raise RuntimeError('EVAL isolation STOP')
    return actual


def pytest_sessionfinish(session, exitstatus):
    # Session finish follows all normal module fixtures, including real kernel
    # registry cleanup. Parent ALSO requires whole owned-process disappearance.
    Path(os.environ['LFS_EVAL_PROOF']).write_text(json.dumps(dict(rows=ROWS, reports=REPORTS, exitstatus=int(exitstatus))) + '\n')


@pytest.fixture(autouse=True)
def _eval_diag_environment(isolate_environment, monkeypatch):
    # Normal policy intentionally scrubs inherited config vars. Rebind ONLY our
    # synthetic root after that policy; never replace module/kernel fixtures.
    bound = Path(os.environ['LFS_EVAL_ROOT']).resolve()
    monkeypatch.setenv('LOCAL_OPERATOR_CONFIG_DIR', str(bound / 'config'))
