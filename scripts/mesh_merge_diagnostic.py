"""Read-only predicate observation for exact failed-merge diagnosis, not a fix."""
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import sys
import threading

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ['MESH_DIAG_OUT'])
MANIFEST=json.loads('{"merge": "e134ee7a44ca0fd7528a954af5d2aa87d561281a", "parents": ["d825a26343bcac566a6ded451eb7b251bfe049b5", "0a66ccf1e5660b4ebbc724ba74fc49514e59c2b6"], "files": {"local_operator/network/enrollment.py": "c5754ee70f619a90e356f5dcb6d275016d7787831a47c9a2961cb9072115f91a", "local_operator/network/readiness.py": "8c6065ab3e1d06a78b0112571eb502aac3b2fda224ac14b5e352614c19ed4b00", "local_operator/network/onboard.py": "1525349a16fe1f988c8192a58c5771037688e1e658f597f616db1ca1fc1c63b6", "local_operator/network/credentials/offers.py": "d0ebe30a9e03c4377813198899cb703ebb195a1517e7f7a73afd36797641b1e9", "local_operator/network/credentials/owner.py": "4a8c3253278a784738018dd00039a60b6ff497afafcf74a6d2e7bb653f24e973", "local_operator/server/routes/desktop_approvals.py": "54ad5c8c6d90cace0dc33cd24cf779ef576fad66e469a352fcd84f346fbf5c67", "tests/unit/network/test_signed_desktop_onboard_e2e.py": "21d684e03a50f1b0f1b2f8a1c2217d094da0181c14d939d2f8ecd4d977f6a1f2", "tests/fixtures/mesh_process_node.py": "9a7cb44cabd23f170cc11fc2527d1738b884f923aec706f16835d7d2a6ef8dbb"}, "authority_at_core": "0a66ccf1e5660b4ebbc724ba74fc49514e59c2b6", "authority_files": {"local_operator/network/approvals.py": "1890ee284cf1d8360c6e2e4479114be1bec4286170189377d45e23d06d455f4c", "local_operator/network/enrollment.py": "c5754ee70f619a90e356f5dcb6d275016d7787831a47c9a2961cb9072115f91a", "local_operator/network/onboard_approvals.py": "558b63782c6152cc12cc12daf4e11a28d3a94338df860487f356fe2782e7cfa3", "local_operator/operator/trust.py": "bce729cd8552f9e2379a7138f978fc856016c4355a9ba44a07eb1abfd2290a39", "local_operator/server/models/desktop_approvals.py": "e55738adea83f37c413ca7ea195878406f205fb59bc26aa599ba49ac6fbbb9eb", "local_operator/server/routes/desktop_approvals.py": "54ad5c8c6d90cace0dc33cd24cf779ef576fad66e469a352fcd84f346fbf5c67", "local_operator/server/utils/desktop_onboard.py": "ebf22cbf8cebefed51a49434f0e2f4d10fe1b8a4160f315dee4cbd99bb68da5b"}}')
LOCK=threading.Lock()
COUNT=0
PHASE='preflight'

def emit(event, **data):
    global COUNT
    with LOCK:
        if COUNT>=128:return
        line=json.dumps({'phase':PHASE,'event':event,**data},sort_keys=True)
        assert len(line)<20000
        with OUT.open('a') as stream:stream.write(line+'\n')
        COUNT+=1

def code(value):
    if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.:-]{0,128}',value):return value
    return '<not-canonical>'

def row_projection(row,device_id=None,network_id=None):
    if not isinstance(row,dict):return {'type':type(row).__name__}
    observed=row.get('observed')
    return {**{key:code(row[key]) for key in ('check','class','capability','code','source') if key in row},
            'ok':row.get('ok') if type(row.get('ok')) is bool else '<not-bool>',
            'device_match':row.get('device_id')==device_id if device_id is not None else None,
            'has_network_id':'network_id' in row,'network_match':row.get('network_id')==network_id if network_id is not None else None,
            'outcome':code(observed.get('outcome')) if isinstance(observed,dict) else None}

def pytest_configure(config):
    global PHASE
    assert sys.platform=='linux', 'native Linux is required; never simulate platform identity'
    for name,expected in {**MANIFEST['files'],**MANIFEST['authority_files']}.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for name in ('local_operator.network.enrollment','local_operator.network.readiness',
                 'local_operator.network.onboard','local_operator.network.credentials.offers',
                 'local_operator.network.credentials.owner'):
        module=importlib.import_module(name)
        assert Path(module.__file__).resolve()==ROOT/Path(*name.split('.')).with_suffix('.py'),name
    from local_operator.network import enrollment as e
    from local_operator.network.types import MeshRefusal
    emit('identity',merge=MANIFEST['merge'],parents=MANIFEST['parents'],platform=sys.platform,
         python=sys.version.split()[0],installed_metadata=importlib.metadata.version('local-operator'),
         full_file_sha256=MANIFEST['files'],authority_sha256=MANIFEST['authority_files'],explicit_pythonpath=True)
    original_require=e.require
    original_ready=e.ready_evidence
    def require(condition,message='The network enrollment authority is invalid'):
        try:return original_require(condition,message)
        except MeshRefusal as error:
            frame=inspect.currentframe().f_back
            try:
                name=Path(frame.f_code.co_filename).resolve()
                relative=str(name.relative_to(ROOT)) if name.is_relative_to(ROOT) else '<outside-source>'
                context=frame.f_locals
                emit('guard_false',function=frame.f_code.co_name,file=relative,line=frame.f_lineno,
                     exception_class=type(error).__name__,exception_code=code(error.code),
                     condition_type=type(condition).__name__,
                     row=row_projection(context.get('row'),context.get('device_id'),context.get('network_id')))
            finally:del frame
            raise
    def ready(report,*,device_id,network_id):
        if isinstance(report,dict):
            rows=report.get('checks')
            emit('actual_ready_input',identity_present=report.get('identity_present'),
                 ok=report.get('ok'),has_schema='schema' in report,has_version='version' in report,
                 checks_type=type(rows).__name__,checks_count=len(rows) if isinstance(rows,list) else None,
                 rows=[row_projection(row,device_id,network_id) for row in rows[:48]] if isinstance(rows,list) else [])
        else:emit('actual_ready_input',type=type(report).__name__)
        result=original_ready(report,device_id=device_id,network_id=network_id)
        emit('actual_ready_result',warnings=[{'capability':code(r.get('capability')),'code':code(r.get('code'))} for r in result])
        return result
    e.require=require
    e.ready_evidence=ready
    PHASE='observer_negative_control'
    try:e.ready_evidence({'identity_present':False,'ok':False,'checks':[]},device_id='d_control',network_id='n_control')
    except MeshRefusal as error:assert error.code=='network_enrollment_invalid'
    else:raise AssertionError('observer negative control did not refuse')
    PHASE='two_actual_positive_scenarios'

def pytest_sessionfinish(session,exitstatus):
    emit('test_complete',exitstatus=int(exitstatus),collected=session.testscollected,failed=session.testsfailed)


def pytest_runtest_logreport(report):
    if report.when in {'setup','call','teardown'}:
        name=report.nodeid.rsplit('::',1)[-1]
        allowed={'test_signed_card_existing_runner_reaches_two_process_members',
                 'test_lost_join_response_same_card_retry_and_single_target_writer'}
        emit('case',case=name if name in allowed else '<unexpected-case>',when=report.when,outcome=report.outcome)


def main():
    import signal,subprocess,tempfile,time
    profile=Path(os.environ['RUNNER_TEMP'])/'mesh-e134-isolated'
    profile.mkdir(exist_ok=True)
    for name in ('home','config','tmp'): (profile/name).mkdir(exist_ok=True)
    env={'HOME':str(profile/'home'),'LOCAL_OPERATOR_CONFIG_DIR':str(profile/'config'),
         'TMPDIR':str(profile/'tmp'),'PATH':os.environ['PATH'],'TERM':'xterm-256color',
         'GIT_CONFIG_SYSTEM':os.devnull,'GIT_TERMINAL_PROMPT':'0',
         'GH_CONFIG_DIR':str(profile/'gh'),'GLAB_CONFIG_DIR':str(profile/'glab'),
         'PYTHONPATH':str(ROOT),'MESH_DIAG_OUT':str(OUT),
         'LOCAL_OPERATOR_SHARD_STALL_SECONDS':'240'}
    public=('CI','GITHUB_ACTIONS','RUNNER_OS','RUNNER_ARCH')
    pytest_flags=('PYTEST_ADDOPTS','PYTEST_DISABLE_PLUGIN_AUTOLOAD','PYTEST_TIMEOUT',
                  'PYTEST_CURRENT_TEST','PYTEST_XDIST_WORKER','PYTEST_XDIST_WORKER_COUNT')
    for name in (*public,*pytest_flags):
        if name in os.environ: env[name]=os.environ[name]
    emit('environment',public_flags={name:env[name] for name in public if name in env},
         preserved_pytest_flag_names=[name for name in pytest_flags if name in env],
         original_shard_stall_seconds=240,serial_two_cases=True,coverage_enabled=False,
         prior_suite_state=False,isolated_home_config_tmp=True,cmux_and_xpc_removed=True,
         limit='Isolated targeted diagnosis; does not reproduce original xdist ordering, coverage, prior test state or all ambient environment')
    args=[sys.executable,str(ROOT/'scripts/run_bounded.py'),'--timeout','180','--',
          sys.executable,'-m','pytest','-o','addopts=','-n0','-q','--tb=no',
          '-p','scripts.mesh_merge_diagnostic','--basetemp',str(profile/'tests'),
          'tests/unit/network/test_signed_desktop_onboard_e2e.py::test_signed_card_existing_runner_reaches_two_process_members',
          'tests/unit/network/test_signed_desktop_onboard_e2e.py::test_lost_join_response_same_card_retry_and_single_target_writer']
    started=time.monotonic();peak=0;owned=set();reason=''
    # Raw pytest reprs can contain bootstrap reference paths. Keep them private
    # in the disposable runner, never logs/artifacts; the plugin publishes only
    # source identity, structural guards and fixed case/outcome vocabulary.
    with tempfile.TemporaryFile(mode='w+') as private_output:
        child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=private_output,
                               stderr=subprocess.STDOUT,start_new_session=True)
        try:
            while True:
                rows=[tuple(map(int,line.split())) for line in subprocess.check_output(
                    ['/bin/ps','-axo','pid=,ppid=,rss='],text=True).splitlines() if len(line.split())==3]
                ids={child.pid};again=True
                while again:
                    extra={pid for pid,parent,rss in rows if parent in ids}
                    again=not extra.issubset(ids);ids|=extra
                owned|=ids;rss=sum(rss for pid,parent,rss in rows if pid in ids);peak=max(peak,rss)
                if rss>2*1024*1024:reason='rss_2gib';break
                if time.monotonic()-started>185:reason='outer_safety_185s';break
                if child.poll() is not None:break
                time.sleep(1)
        finally:
            if child.poll() is None:
                for pid in sorted(owned,reverse=True):
                    try:os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                try:child.wait(5)
                except subprocess.TimeoutExpired:
                    for pid in sorted(owned,reverse=True):
                        try:os.kill(pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                    child.wait(5)
    emit('lease_result',rc=child.returncode,wall_s=round(time.monotonic()-started,2),
         sampled_peak_tree_rss_mib=round(peak/1024,2),rss_limit_mib=2048,
         test_deadline_s=180,reason=reason,owned_pids=sorted(owned))
    print('Native diagnostic completed; only scrubbed structural JSONL is published. Exit status:',child.returncode)
    return child.returncode or (124 if reason else 0)

if __name__=='__main__':
    raise SystemExit(main())
