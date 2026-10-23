"""ONE unchanged original S2; measurement-only hooks, no new load or thresholds."""
import hashlib
import cProfile
import json
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch
import pytest
from scripts.diagnostics.lfs_pair_6911.controller import validate_identity
from scripts.diagnostics.native_worker_6911.carrier import native_sources
from scripts.diagnostics.s2_static_6911.controller import project_measurement, serialize_profile

NODE = 'tests/unit/test_tui_responsiveness.py::test_s2_send_to_agent_end_keeps_the_loop_under_the_bar'
ROWS, REPORTS = [], []
CALLS = 0
SESSION = None


def pytest_sessionstart(session):
    global SESSION
    SESSION = session


def pytest_collection_modifyitems(items):
    if len(items) != 1 or items[0].nodeid != NODE:
        raise ValueError('one original S2 selector STOP')


@pytest.fixture(autouse=True)
def _owned_config(isolate_environment, monkeypatch):
    monkeypatch.setenv('LOCAL_OPERATOR_CONFIG_DIR', str(Path(os.environ['S2_PROFILE_ROOT'])/'config'))


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    global CALLS
    CALLS += 1
    if CALLS != 1: raise ValueError('S2 call cardinality STOP')
    from local_operator.tui import app as app_module
    from tests import worker_code_identity as workers, loop_callback_spans as observer
    from tests.unit.tui import test_app_pilot as pilot_fixture
    import tests.conftest as root_policy
    root = Path(__file__).resolve().parents[2]
    expected = json.loads(os.environ['S2_PROFILE_EXPECTED'])
    files = {'tests/unit/test_tui_responsiveness.py': item.module.__file__,
        'tests/unit/lfs_s2_static_plugin_6911.py': __file__,
        'tests/conftest.py':root_policy.__file__,
        'tests/unit/tui/test_app_pilot.py':pilot_fixture.__file__,
        'tests/worker_code_identity.py':workers.__file__,
        'tests/loop_callback_spans.py':observer.__file__,
        'local_operator/tui/app.py':app_module.__file__}
    hashes = {}
    for label, actual in files.items():
        path=Path(actual).resolve()
        if path != root/label: raise ValueError('actual source path STOP')
        hashes[label]=hashlib.sha256(path.read_bytes()).hexdigest()
    identity=dict(python=list(sys.version_info[:3]),executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),pid=os.getpid(),ppid=os.getppid(),pgid=os.getpgrp(),sources=hashes)
    validate_identity(identity, dict(expected,pid=os.getpid()))
    bound=Path(os.environ['S2_PROFILE_ROOT']).resolve()
    for key in ('HOME','LOCAL_OPERATOR_CONFIG_DIR','TMPDIR','LOCAL_OPERATOR_RUNTIME_ROOT'):
        if not Path(os.environ[key]).resolve().is_relative_to(bound):raise ValueError('S2 isolation STOP')
    native = native_sources()
    if workers.ACTIVE is None: raise ValueError('metadata fixture not armed BEFORE App STOP')
    labels=set(workers.ACTIVE.allowlist.values())
    module=item.module
    row=dict(token=CALLS,identity=identity,native_sources=native,phases=[],summary=None,
             effective=None,call_pass=False,performance_oracle=False,restored=False,cpu_profile=None)
    original_stop=module.StallRecorder.stop
    original_begin=module.StallRecorder.begin_noise_window
    original_end=module.StallRecorder.end_noise_window
    original_bar=module.StallRecorder.assert_no_stall_loaded
    def mark(name):
        row['phases'].append(dict(phase=name,cpu_s=time.thread_time(),wall_s=time.perf_counter()))
        if len(row['phases'])>4: raise ValueError('phase cap STOP')
    def begin(recorder):
        mark('IDLE_BEGIN')
        return original_begin(recorder)
    def end(recorder):
        mark('TURN_BEGIN')
        return original_end(recorder)
    async def stop(recorder):
        try:return await original_stop(recorder)
        finally:
            mark('TURN_END')
            row['summary']=project_measurement(recorder.measurement_summary(), labels)
    def bar(recorder,cpu_ceiling_ms=500.0,wall_ceiling_ms=2000.0,noise_floor_ms=None):
        # Observe exact original formula, then DELEGATE original assertion.
        cpu,wall=cpu_ceiling_ms,wall_ceiling_ms
        if noise_floor_ms is not None:
            cpu=min(max(module.NOISE_K*noise_floor_ms,cpu*recorder.cpu_calibration_ratio),module.NOISE_CPU_CAP_MS)
            wall=min(max(module.NOISE_K*noise_floor_ms,wall),module.NOISE_WALL_CAP_MS)
        row['effective']=dict(cpu_ms=cpu,wall_ms=wall)
        return original_bar(recorder,cpu_ceiling_ms,wall_ceiling_ms,noise_floor_ms)
    profiler=None
    previous=sys.getprofile()
    try:
        if previous is not None:raise ValueError('preexisting profiler STOP')
        profiler=cProfile.Profile(timer=time.thread_time_ns,timeunit=1e-9)
        profiler.enable(subcalls=True,builtins=False)
        with patch.object(module.StallRecorder,'begin_noise_window',begin), patch.object(module.StallRecorder,'end_noise_window',end), patch.object(module.StallRecorder,'stop',stop), patch.object(module.StallRecorder,'assert_no_stall_loaded',bar):
            outcome=yield
            row['call_pass']=outcome.excinfo is None
            if outcome.excinfo is not None:
                exc=outcome.excinfo[1]
                tb=exc.__traceback__
                while tb is not None:
                    if isinstance(exc,AssertionError) and tb.tb_frame.f_code is module.StallRecorder.assert_no_stall.__code__ and tb.tb_lineno in (285, 291):
                        row['performance_oracle']=True
                    tb=tb.tb_next
    finally:
        try:
            if profiler is not None:
                profiler.disable()
                row['cpu_profile']=serialize_profile(profiler.getstats(),workers.ACTIVE.allowlist,sys.getprofile() is previous)
        finally:
            row['restored']=all((module.StallRecorder.stop is original_stop,module.StallRecorder.begin_noise_window is original_begin,module.StallRecorder.end_noise_window is original_end,module.StallRecorder.assert_no_stall_loaded is original_bar))
            ROWS.append(row)


def pytest_runtest_logreport(report):
    REPORTS.append(dict(phase=report.when,outcome=report.outcome))
    if report.when != 'call' and report.outcome != 'passed': SESSION.shouldstop='S2 fixture STOP'


def pytest_sessionfinish(session, exitstatus):
    Path(os.environ['S2_PROFILE_PROOF']).write_text(json.dumps(dict(rows=ROWS,reports=REPORTS,guards=[],exitstatus=int(exitstatus)))+'\n')
