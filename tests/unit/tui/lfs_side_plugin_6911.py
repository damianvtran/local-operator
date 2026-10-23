"""Offhost-only instrumentation of the ORIGINAL selector, twice serially.
No copied test body or fixture generator. Hookwrapper yields unconditionally.
Only the second invocation renders once after real prepare and before its spy.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import pytest
from scripts.diagnostics.lfs_pair_6911.controller import validate_identity

NODE = "tests/unit/tui/test_sidebar_repaint.py::test_identical_catalog_does_not_invalidate_rendered_frame"
ROWS, REPORTS = [], []
CALLS = 0
SESSION = None


def pytest_sessionstart(session):
    global SESSION
    SESSION = session


@pytest.fixture(autouse=True, params=("baseline", "render"), ids=("baseline", "render"))
def _lfs_side_variant(request):
    # Standard pytest parametrization creates genuinely distinct Function items;
    # no repeated-selector cache or mutable user_properties label assumption.
    return request.param


def collection_variants(items):
    if len(items) != 2 or len({id(item) for item in items}) != 2:
        raise RuntimeError("distinct parametrized items required")
    modes = [item.callspec.params["_lfs_side_variant"] for item in items]
    if modes != ["baseline", "render"] or any(item.nodeid.split("[", 1)[0] != NODE for item in items):
        raise RuntimeError("exact ordered original variants required")
    return modes


def pytest_collection_modifyitems(items):
    collection_variants(items)


def differences(sidebar):
    # Positions only, no entries, IDs, text, titles or reprs. These are current
    # complete paint-state vs last-rendered-field comparisons, not catalog-only.
    actual, painted = sidebar._paint_state(), sidebar._painted_state
    if painted is None:
        return dict(baseline_missing=True, changed_positions=list(range(len(actual))))
    return dict(baseline_missing=False,
                changed_positions=[i for i, value in enumerate(actual) if i >= len(painted) or value != painted[i]])


def identity(module):
    from local_operator.tui import app
    from local_operator.tui.widgets import session_sidebar
    import tests.conftest as policy
    root = Path(__file__).resolve().parents[3]
    files = {NODE.split("::")[0]: module.__file__,
             "tests/unit/tui/lfs_side_plugin_6911.py": __file__,
             "local_operator/tui/app.py": app.__file__,
             "local_operator/tui/widgets/session_sidebar.py": session_sidebar.__file__,
             "tests/conftest.py": policy.__file__}
    hashes = {}
    for label, actual in files.items():
        path = Path(actual).resolve()
        if path != root / label:
            raise RuntimeError("loaded source mismatch")
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
    actual = dict(python=list(sys.version_info[:3]),
                  executable_sha256=hashlib.sha256(Path(sys.executable).resolve().read_bytes()).hexdigest(),
                  pid=os.getpid(), ppid=os.getppid(), pgid=os.getpgrp(), sources=hashes)
    expected = json.loads(os.environ["LFS_SIDE_EXPECTED"])
    expected["pid"] = os.getpid()  # Parent independently compares Popen identity.
    validate_identity(actual, expected)
    bound = Path(os.environ["LFS_SIDE_ROOT"]).resolve()
    for key in ("HOME", "LOCAL_OPERATOR_CONFIG_DIR", "TMPDIR", "LOCAL_OPERATOR_RUNTIME_ROOT"):
        if not Path(os.environ[key]).resolve().is_relative_to(bound):
            raise RuntimeError("isolation STOP")
    return actual


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    global CALLS
    CALLS += 1
    variant = item.callspec.params["_lfs_side_variant"]
    if CALLS > 2 or variant != ("baseline", "render")[CALLS - 1]:
        raise RuntimeError("actual call cardinality/order STOP")
    module = item.module
    original_prepare, original_patch = module.prepare, module.patch
    row = dict(variant=variant, token=CALLS, count=None,
               prepare_calls=0, renders=0, assertion=False, call_pass=False,
               restored=False, identity=None)

    async def prepare(*args, **kwargs):
        sidebar = await original_prepare(*args, **kwargs)
        row["prepare_calls"] += 1
        row["before"] = differences(sidebar)
        if row["variant"] == "render":
            sidebar.render()
            row["renders"] += 1
        row["after"] = differences(sidebar)
        return sidebar

    class Patch:
        def __call__(self, *args, **kwargs):
            return original_patch(*args, **kwargs)

        @contextmanager
        def object(self, target, name, *args, **kwargs):
            with original_patch.object(target, name, *args, **kwargs) as spy:
                try:
                    yield spy
                finally:
                    if name == "refresh":
                        row["count"] = spy.call_count

    try:
        row["identity"] = identity(module)
        module.prepare, module.patch = prepare, Patch()
        outcome = yield  # ALL invocations, irrespective of variant.
        row["call_pass"] = outcome.excinfo is None
        if outcome.excinfo:
            kind, _exc, tb = outcome.excinfo
            # Attribute failure to the original target's assert_not_called site;
            # no exception text or raw traceback/source is recorded.
            target = inspect.unwrap(item.obj)
            lines, first = inspect.getsourcelines(target)
            site = first + next(i for i, line in enumerate(lines) if "refresh.assert_not_called()" in line)
            while tb:
                if tb.tb_frame.f_code is target.__code__ and tb.tb_lineno == site and issubclass(kind, AssertionError):
                    row["assertion"] = True
                tb = tb.tb_next
    finally:
        module.prepare, module.patch = original_prepare, original_patch
        row["restored"] = module.prepare is original_prepare and module.patch is original_patch
        ROWS.append(row)


def pytest_runtest_logreport(report):
    REPORTS.append(dict(node=report.nodeid.split("[", 1)[0] == NODE, phase=report.when, outcome=report.outcome))
    if report.failed and report.when != "call":
        # Setup/isolation/teardown failure stops rather than dispatching more.
        SESSION.shouldstop = "diagnostic setup/teardown STOP"


    if report.failed and report.when == "call":
        row = ROWS[-1] if ROWS else {}
        if not (row.get("variant") == "baseline" and row.get("assertion") and row.get("count") == 50):
            SESSION.shouldstop = "unexpected diagnostic call STOP"


def pytest_sessionfinish(session, exitstatus):
    proof = dict(rows=ROWS, reports=REPORTS, exitstatus=int(exitstatus))
    Path(os.environ["LFS_SIDE_PROOF"]).write_text(json.dumps(proof) + "\n")
