"""QA SOURCE fragment: original MOVE target baseline versus explicit fresh member dial.
No product changes; integrate/export via existing qualified controller only.
"""
from contextlib import ExitStack
import hashlib
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch
import pytest
from scripts.diagnostics.lfs_pair_6911.controller import validate_identity
from scripts.diagnostics.lfs_move_6911.controller import validate_row
from local_operator.network.relay import PeerLink

NODE = 'tests/unit/network/test_mobility_promotion_identity.py::test_a_peer_that_answers_the_old_way_cannot_make_this_device_promote'
SHA = '45e4030c7a19418600430ce7f6db3bef1520d465991a4f161eb11b661f6fdab8'
ROWS, REPORTS = [], []
CALLS = 0
GUARDS = []

def mark_guard(kind):
    if len(GUARDS) < 8:
        GUARDS.append(kind)

def stop_guard(kind):
    mark_guard(kind)
    raise RuntimeError("MOVE fixed guard STOP")

@pytest.fixture(autouse=True, params=('baseline', 'fresh'), ids=('baseline', 'fresh'))
def _move_variant(request):
    return request.param


def same_handshake(accepted, returned, network_id, a_device, b_device):
    """Private causal binding only; never serialize handshake IDs/endpoints."""
    left, right = accepted.link_id, returned.link_id
    return (type(left) is bytes and bool(left) and type(right) is bytes
            and left == right and accepted.network_id == returned.network_id == network_id
            and accepted.device_id == b_device and returned.device_id == a_device)


def pytest_collection_modifyitems(items):
    if len(items) != 2 or len({id(x) for x in items}) != 2:
        raise RuntimeError('MOVE distinct two original items required')
    if [x.callspec.params['_move_variant'] for x in items] != ['baseline', 'fresh'] or any(x.nodeid.split('[', 1)[0] != NODE for x in items):
        raise RuntimeError('MOVE ordered original selectors required')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    global CALLS
    CALLS += 1
    mode = item.callspec.params['_move_variant']
    if CALLS > 2 or mode != ('baseline', 'fresh')[CALLS - 1]:
        raise RuntimeError('MOVE actual call order STOP')
    module = item.module
    if hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest() != SHA:
        raise RuntimeError('MOVE original source STOP')
    identity = verify_identity(module)
    original_pair, original_move = module._pair_settled, module._move
    row = dict(mode=mode, token=CALLS, pair_calls=0, move_calls=0,
               dial_attempted=False, dial_ok=False, new_generation=False,
               selected_new=False, correlated=False, selected_alive=False, selected_closing=False,
               request_seen=False, reply_seen=False, first_ok=False,
               first_unreachable=False, call_pass=False, restored=False, first_assertion=False, identity=identity)
    state = dict(a=None, b=None, selected=None, admitted_selected=None, fresh=None, previous=set())
    stack = ExitStack()

    def settled(both, monkeypatch, **kwargs):
        row['pair_calls'] += 1
        answer = original_pair(both, monkeypatch, **kwargs)
        a, b, _, _ = both
        state['a'], state['b'] = a, b
        # Exact existing helper returns (_record_from_A, A_host, A_port), not B.
        record, host, port = answer
        state['previous'] = {id(x) for x in a.links.values() if x.device_id == b.identity.device_id}
        if mode == 'fresh':
            row['dial_attempted'] = True
            # B -> listening A is already proven by the transport test using
            # this exact helper result; no guessed B listener endpoint.
            link, _reason = b.dial(record.network_id, host=f'{host}:{port}',
                                   epoch=record.epoch, expected_device=a.identity.device_id)
            state['fresh'] = link
            row['dial_ok'] = link is not None
            if link is None:
                stop_guard('DIAL')
            # Actual generation on A must be NEW AND SELECTED for A -> B's first
            # move. An extra connection whose selection is old does NOT qualify.
            selected = a._link_for(b.identity.device_id)
            candidates = [x for x in a.links.values() if x.device_id == b.identity.device_id and id(x) not in state['previous']]
            row['new_generation'] = len(candidates) == 1
            row['correlated'] = row['new_generation'] and same_handshake(
                candidates[0], link, record.network_id, a.identity.device_id, b.identity.device_id
            )
            if not row['correlated']:
                stop_guard('CORRELATION')
            row['selected_new'] = row['new_generation'] and selected is candidates[0]
            if not row['selected_new']:
                stop_guard('SELECTION')
            state['admitted_selected'] = selected
        return answer

    def move(server, *args, **kwargs):
        row['move_calls'] += 1
        if row['move_calls'] != 1:
            # Preserve the later old-protocol authority/adoption check unchanged.
            return original_move(server, *args, **kwargs)
        if server is not state['a']:
            raise RuntimeError('MOVE first operation direction STOP')
        original_request = PeerLink.request
        def request(link, *request_args, **request_kwargs):
            relevant = link.server is server and link.device_id == state['b'].identity.device_id
            if relevant:
                # Observe the ACTUAL first-move transport, including baseline's
                # lazy dial; a pre-call table snapshot alone can be empty.
                state['selected'] = link
                row['request_seen'] = True
                row['selected_alive'], row['selected_closing'] = bool(link.alive), bool(link.closing)
                if mode == 'fresh' and link is not state['admitted_selected']:
                    stop_guard('REQUEST')
            reply = original_request(link, *request_args, **request_kwargs)
            if relevant and reply is not None:
                row['reply_seen'] = True
            return reply
        with patch.object(PeerLink, 'request', request):
            result = original_move(server, *args, **kwargs)
        row['first_ok'] = result.get('ok') is True
        row['first_unreachable'] = result.get('ok') is False and result.get('code') == 'unreachable'
        return result

    try:
        stack.enter_context(patch.object(module, '_pair_settled', settled))
        stack.enter_context(patch.object(module, '_move', move))
        outcome = yield
        row['call_pass'] = outcome.excinfo is None
        if outcome.excinfo is not None:
            exc = outcome.excinfo[1]
            tb = exc.__traceback__
            while tb is not None:
                if (isinstance(exc, AssertionError)
                        and tb.tb_frame.f_code is module._staged_copy_of_the_receivers_move.__code__
                        and tb.tb_lineno == 78):
                    row['first_assertion'] = True
                tb = tb.tb_next

    finally:
        # Production dial admitted/started the returned member link and the
        # fixture owns both relays. Close extra B link before fixture teardown;
        # controller MUST prove all relay threads/processes settled afterward.
        try:
            if state['fresh'] is not None:
                state['fresh'].close('qa-fresh-control')
        finally:
            stack.close()
            row['restored'] = module._pair_settled is original_pair and module._move is original_move
            ROWS.append(row)


def pytest_runtest_logreport(report):
    REPORTS.append(dict(phase=report.when, outcome=report.outcome))
    if report.when == 'call':
        try:
            if not ROWS or not 1 <= CALLS <= 2 or len(ROWS) != CALLS:
                raise ValueError('MOVE missing call STOP')
            validate_row(ROWS[-1], ('baseline','fresh')[CALLS-1], CALLS, report.outcome)
        except (ValueError, KeyError, TypeError, IndexError):
            mark_guard('ROW')
            SESSION.shouldstop = 'MOVE invalid completed call STOP'
    elif report.outcome != 'passed':
        mark_guard('FIXTURE')
        SESSION.shouldstop = 'MOVE setup/fixture STOP'



def qualify_rows(rows, reports):
    if len(rows) != 2 or [r['mode'] for r in rows] != ['baseline', 'fresh'] or [r['token'] for r in rows] != [1, 2]:
        raise ValueError('MOVE cardinality/order STOP')
    if len(reports) != 6 or [r['phase'] for r in reports] != ['setup', 'call', 'teardown'] * 2:
        raise ValueError('MOVE phases STOP')
    if any(r['outcome'] != 'passed' for r in reports if r['phase'] != 'call'):
        raise ValueError('MOVE fixtures STOP')
    for i, row in enumerate(rows):
        validate_row(row, ('baseline', 'fresh')[i], i+1, reports[3*i+1]['outcome'])
    baseline, fresh = rows
    if not all(fresh[k] for k in ('dial_attempted', 'dial_ok', 'new_generation', 'selected_new', 'correlated', 'selected_alive', 'request_seen', 'reply_seen', 'call_pass')) or fresh['selected_closing']:
        raise ValueError('MOVE fresh selected transport STOP')
    if baseline['call_pass']:
        return 'INCONCLUSIVE_NOT_EXONERATED'
    # The original staging helper immediately asserts first move ok; the
    # expected negative is THAT failure, not an unrelated later authority guard.
    if baseline['move_calls'] == 1 and baseline['first_unreachable']:
        return 'N1_TRANSPORT_CANDIDATE_NOT_ROOT_CURE'
    raise ValueError('MOVE non-transport baseline failure STOP')


SESSION = None


def pytest_sessionstart(session):
    global SESSION
    SESSION = session


def verify_identity(module):
    import tests.conftest as policy
    import tests.unit.network.conftest as network_policy
    root = Path(__file__).resolve().parents[3]
    files = {'tests/unit/network/test_mobility_promotion_identity.py': module.__file__,
             'tests/unit/network/lfs_move_plugin_6911.py': __file__,
             'tests/conftest.py': policy.__file__,
             'tests/unit/network/conftest.py': network_policy.__file__,
             'tests/unit/network/test_relay_e2e.py': sys.modules[module._pair_settled.__module__].__file__,
             'tests/unit/network/test_mobility.py': sys.modules[module._move.__module__].__file__,
             'local_operator/network/relay.py': sys.modules[PeerLink.__module__].__file__,
             'local_operator/network/mobility.py': module.mobility.__file__,
             'scripts/diagnostics/lfs_move_6911/controller.py': sys.modules[validate_row.__module__].__file__}
    hashes = {}
    for label, actual in files.items():
        path = Path(actual).resolve()
        if path != root / label:
            raise RuntimeError('MOVE actual loaded path STOP')
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = dict(python=list(sys.version_info[:3]),
                  executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                  pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(), sources=hashes)
    expected = json.loads(os.environ['LFS_MOVE_EXPECTED'])
    expected['pid'] = os.getpid()
    validate_identity(actual, expected)
    bound = Path(os.environ['LFS_MOVE_ROOT']).resolve()
    for key in ('HOME', 'LOCAL_OPERATOR_CONFIG_DIR', 'TMPDIR', 'LOCAL_OPERATOR_RUNTIME_ROOT'):
        if not Path(os.environ[key]).resolve().is_relative_to(bound):
            raise RuntimeError('MOVE isolation STOP')
    return actual


@pytest.fixture(autouse=True)
def _move_diag_environment(isolate_environment, monkeypatch):
    bound = Path(os.environ['LFS_MOVE_ROOT']).resolve()
    monkeypatch.setenv('LOCAL_OPERATOR_CONFIG_DIR', str(bound / 'config'))


def pytest_sessionfinish(session, exitstatus):
    Path(os.environ['LFS_MOVE_PROOF']).write_text(json.dumps(dict(rows=ROWS, reports=REPORTS, guards=GUARDS, exitstatus=int(exitstatus))) + '\n')
