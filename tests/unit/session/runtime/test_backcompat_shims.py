"""The old ``mobile/`` import paths must keep resolving to the moved code.

The runtime package move (registrant/registry/owned/child -> session/runtime/) left
re-export shims behind, and they are not decoration: two lop binaries of
different versions coexist in running processes on one machine, and an
out-of-tree caller may hold either path. Nothing else in the suite imports the
shims any more — every real call site was repointed — so without this file
they would be dead code that rots unnoticed and breaks a mixed-version
upgrade window.

The identity assertions are the point: a shim that re-defined a class instead
of re-exporting one would still import cleanly, and would then fail an
``isinstance`` check across the two paths at runtime.
"""

from __future__ import annotations

import importlib


def test_registrant_shim_re_exports_the_same_objects() -> None:
    shim = importlib.import_module("local_operator.mobile.registrant")
    server = importlib.import_module("local_operator.session.runtime.server")

    assert shim.RuntimeServer is server.RuntimeServer
    assert shim.SessionHandle is server.SessionHandle
    assert shim.image_blocks is server.image_blocks


def test_registrant_alias_is_the_runtime_server_class() -> None:
    """``Registrant`` is the pre-move name, kept so the rename cost no caller."""
    shim = importlib.import_module("local_operator.mobile.registrant")
    server = importlib.import_module("local_operator.session.runtime.server")

    assert shim.Registrant is server.RuntimeServer
    assert server.Registrant is server.RuntimeServer


def test_registry_shim_re_exports_the_same_functions() -> None:
    shim = importlib.import_module("local_operator.mobile.registry")
    registry = importlib.import_module("local_operator.session.runtime.registry")

    for name in ("run_dir", "publish", "unpublish", "pid_alive", "scan", "RecordPublisher"):
        assert getattr(shim, name) is getattr(registry, name), name


def test_owned_shim_re_exports_the_same_objects() -> None:
    shim = importlib.import_module("local_operator.mobile.owned")
    owned = importlib.import_module("local_operator.session.runtime.owned")

    assert shim.OwnedSessionHandle is owned.OwnedSessionHandle
    assert shim.spawn_owned_session is owned.spawn_owned_session


def test_child_shim_re_exports_the_same_entrypoint() -> None:
    """``python -m local_operator.mobile.child`` is a cross-process spawn
    contract: a daemon of one version spawns a child of another. The shim
    keeps that module runnable, so it must expose the real ``main``."""
    shim = importlib.import_module("local_operator.mobile.child")
    process = importlib.import_module("local_operator.session.runtime.process")

    assert shim.main is process.main
    assert shim.amain is process.amain
    assert shim._reaper is process._reaper


def test_child_shim_is_runnable_as_a_module() -> None:
    """``-m local_operator.mobile.child`` must actually REACH the runtime's
    ``main``, not merely re-export it.

    This is the one thing the identity assertions above cannot see. The
    ``if __name__ == "__main__"`` guard is the single line that makes the
    old spawn path work, no other shim has one, and deleting it leaves every
    identity check green while silently breaking the upgrade window: a daemon
    of one version spawns ``python -m local_operator.mobile.child`` and would
    get a process that imports cleanly and exits 0 having done nothing.

    So the module is executed for real in a subprocess under ``run_name
    ="__main__"``, with the runtime's ``main`` patched to a sentinel exit
    code. A dropped guard yields 0 instead of the sentinel and fails here.
    """
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[4]
    # 87 is arbitrary but distinctive: it cannot be confused with a clean exit,
    # a Python error exit (1/2), or a signal-derived status.
    driver = (
        "import runpy, sys;"
        "import local_operator.session.runtime.process as p;"
        "p.main = lambda: 87;"
        "import local_operator.mobile.child as c;"
        "c.main = p.main;"
        "runpy.run_module('local_operator.mobile.child', run_name='__main__')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", driver], capture_output=True, text=True, cwd=str(repo)
    )

    assert proc.returncode == 87, (
        "`python -m local_operator.mobile.child` did not reach main(); "
        f"rc={proc.returncode}\n{proc.stderr[-2000:]}"
    )


def test_record_constants_are_identical_across_both_type_paths() -> None:
    """The record literals are a wire contract, not an implementation detail.

    ``mobile.types`` re-exports them rather than redefining them; a redefined
    copy that drifted by one value would split a mixed-version machine's view
    of its own live sessions.
    """
    mobile_types = importlib.import_module("local_operator.mobile.types")
    runtime_types = importlib.import_module("local_operator.session.runtime.types")

    assert mobile_types.SessionRecord is runtime_types.SessionRecord
    assert mobile_types.RUN_DIRNAME == runtime_types.RUN_DIRNAME == "run/mobile"
    assert mobile_types.HEARTBEAT_INTERVAL_S == runtime_types.HEARTBEAT_INTERVAL_S == 15.0
    assert mobile_types.HEARTBEAT_TIMEOUT_S == runtime_types.HEARTBEAT_TIMEOUT_S == 45.0
    assert mobile_types.ATTACH_MAX_CLIENTS == runtime_types.ATTACH_MAX_CLIENTS == 4
    # This move must not be read as a protocol change by anything on the wire.
    assert mobile_types.PROTOCOL_VERSION == runtime_types.PROTOCOL_VERSION == 5
