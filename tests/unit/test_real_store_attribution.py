"""The real-store tripwire excuses ONLY a provably independent retention sweep.

Regression for test run ebc3f4ee0319: a live ``lop`` (pid 34093) reaped 24
stale delegated sessions from the operator's store while pytest ran, and the
root conftest failed the run as though the suite had deleted them. The fix
reads the product's own cleanup-log rows; every test below except the first
pins one way that evidence must still FAIL, because a false excuse would hide
the very incident the tripwire exists to catch.

Every store here is a synthetic one under ``tmp_path``; nothing reads, lists or
writes the operator's real store.
"""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import time
import types
from pathlib import Path
from typing import Any

import pytest

from local_operator.session import cleanup, delegated_retention
from tests import real_store_attribution as rsa

CONFTEST = Path(__file__).resolve().parents[2] / "conftest.py"
ROOT = rsa.ProcessIdentity(pid=900_001, ppid=1, start=2_000_000_000)
SWEEPER = rsa.ProcessIdentity(pid=900_002, ppid=1, start=ROOT.start - 3600)


def _store(tmp_path: Path, names: tuple[str, ...] = ("s1", "s2")) -> Path:
    config = tmp_path / "config"
    store = config / "sessions"
    store.mkdir(parents=True)
    cleanup.mark_store(store)
    for name in names:
        (store / name).mkdir()
        (store / name / "transcript.jsonl").write_text("{}\n")
    return store.resolve()


def _remove_genuinely(
    store: Path, name: str, *, policy: str = "delegated_max_age"
) -> dict[str, Any]:
    """Remove ``name`` through the PRODUCT writer, so the row schema is the real one."""
    assert cleanup.remove_session_dir(
        store / name, config_dir=store.parent, policy=policy, reason="stale", actor="startup"
    )
    rows = (store / rsa.CLEANUP_LOG_NAME).read_text().splitlines()
    return json.loads(rows[-1])


def _rewrite_pids(store: Path, pid: int, **override: object) -> None:
    """Attribute every row to ``pid``: the writer here is this test, not a sweeper."""
    log = store / rsa.CLEANUP_LOG_NAME
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    # In-place rewrite keeps the inode, so only the content claim changes.
    with open(log, "r+b") as handle:
        handle.truncate(0)
        for row in rows:
            row.update(pid=pid, **override)
            handle.write((json.dumps(row) + "\n").encode())


def _probe(*known: rsa.ProcessIdentity):
    table = {identity.pid: identity for identity in known}
    return table.get


def _attribute(store: Path, mark: rsa.LogMark | None, started_at: float, **kw):
    kw.setdefault("root", ROOT)
    kw.setdefault("probe", _probe(ROOT, SWEEPER))
    return rsa.attribute(["s1"], store=store, mark=mark, started_at=started_at, **kw)


def _scenario(tmp_path: Path, pid: int = SWEEPER.pid, **override: object):
    """Mark the log, let a 'sweeper' remove s1 with a genuine row, attribute it."""
    store = _store(tmp_path)
    # A pre-run row: lives below the mark, so it must never count.
    _remove_genuinely(store, "s2")
    started_at = time.time() - 1
    mark = rsa.mark_log(store)
    _remove_genuinely(store, "s1")
    _rewrite_pids(store, pid, **override)
    # The rewrite changed the pre-run row too; re-mark that prefix so only the
    # rows the test means to alter are seen as altered.
    head = (store / rsa.CLEANUP_LOG_NAME).read_bytes()
    first = head.index(b"\n") + 1
    assert mark is not None
    mark = rsa.LogMark(
        identity=mark.identity, offset=first, digest=rsa.hashlib.sha256(head[:first]).hexdigest()
    )
    return store, mark, started_at


def test_independent_sweeper_with_genuine_row_is_explained(tmp_path: Path) -> None:
    store, mark, started_at = _scenario(tmp_path)
    verdict = _attribute(store, mark, started_at)
    assert verdict.unexplained == {}
    assert verdict.explained["s1"]["policy"] == "delegated_max_age"
    assert verdict.explained["s1"]["pid"] == SWEEPER.pid


@pytest.mark.parametrize(
    ("override", "expect"),
    [
        ({"policy": cleanup.EXPLICIT_DELETE_POLICY}, "not automatic retention"),
        ({"policy": cleanup.MESH_MOVE_POLICY}, "not automatic retention"),
        ({"actor": "cli"}, "not automatic retention"),
        ({"guards": ["refusal"]}, "lack"),
        ({"path": "/elsewhere/sessions/s1"}, "is not under"),
        ({"session": "s1", "path": None}, "is not this entry"),
        ({"at": "2001-01-01T00:00:00+0000"}, "outside this run"),
        ({"at": "not a time"}, "outside this run"),
    ],
)
def test_row_that_is_not_independent_retention_fails(
    tmp_path: Path, override: dict[str, Any], expect: str
) -> None:
    store, mark, started_at = _scenario(tmp_path, **override)
    verdict = _attribute(store, mark, started_at)
    assert verdict.explained == {}
    assert expect in verdict.unexplained["s1"]


def test_own_pid_fails_even_with_a_perfect_row(tmp_path: Path) -> None:
    store, mark, started_at = _scenario(tmp_path, pid=os.getpid())
    me = rsa.ProcessIdentity(pid=os.getpid(), ppid=1, start=SWEEPER.start)
    verdict = _attribute(store, mark, started_at, probe=_probe(ROOT, me))
    assert "this pytest run" in verdict.unexplained["s1"]


@pytest.mark.parametrize("start", [ROOT.start, ROOT.start + 5], ids=["same-second", "after"])
def test_actor_not_started_strictly_before_root_fails(tmp_path: Path, start: int) -> None:
    """A descendant (or an orphan re-parented to launchd) starts after the root."""
    store, mark, started_at = _scenario(tmp_path)
    child = rsa.ProcessIdentity(pid=SWEEPER.pid, ppid=1, start=start)
    verdict = _attribute(store, mark, started_at, probe=_probe(ROOT, child))
    assert "not before this run's root" in verdict.unexplained["s1"]


def test_dead_actor_fails(tmp_path: Path) -> None:
    store, mark, started_at = _scenario(tmp_path)
    verdict = _attribute(store, mark, started_at, probe=_probe(ROOT))
    assert "dead or unreadable" in verdict.unexplained["s1"]


def test_no_row_and_pre_run_row_fail(tmp_path: Path) -> None:
    store, mark, started_at = _scenario(tmp_path)
    verdict = rsa.attribute(
        ["s2", "s3"],
        store=store,
        mark=mark,
        started_at=started_at,
        root=ROOT,
        probe=_probe(ROOT, SWEEPER),
    )
    # s2's row predates the mark; s3 has none at all.
    assert set(verdict.unexplained) == {"s2", "s3"}
    assert verdict.explained == {}


def test_missing_snapshot_evidence_fails_everything(tmp_path: Path) -> None:
    store, mark, started_at = _scenario(tmp_path)
    assert "unreadable" in _attribute(store, None, started_at).unexplained["s1"]
    assert "root" in _attribute(store, mark, started_at, root=None).unexplained["s1"]


def _break(store: Path, how: str) -> None:
    log = store / rsa.CLEANUP_LOG_NAME
    data = log.read_bytes()
    if how == "truncated":
        with open(log, "r+b") as handle:
            handle.truncate(10)
    elif how == "rewritten-prefix":
        with open(log, "r+b") as handle:
            handle.write(b"X")
    elif how == "rotated":
        # A real rotation must give the log a NEW (st_dev, st_ino): identical
        # bytes mean identity is the only thing that can tell the files apart.
        # unlink-then-write let ext4/tmpfs hand the freed inode number straight
        # back, so CI run 43010912496 saw an "unchanged" log and excused s1
        # (APFS never reuses one immediately, so it passed on macOS). Writing
        # the replacement while the old inode is still held, then renaming it
        # over the log, makes the two inodes coexist and therefore differ.
        before = log.stat()
        replacement = log.with_name(log.name + ".rotated")
        replacement.write_bytes(data)
        os.replace(replacement, log)
        after = log.stat()
        assert (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
    elif how == "removed":
        log.unlink()
    elif how == "malformed":
        with open(log, "ab") as handle:
            handle.write(b"{not json\n")
    elif how == "partial":
        with open(log, "ab") as handle:
            handle.write(b'{"session": "s1"')


@pytest.mark.parametrize(
    "how", ["truncated", "rewritten-prefix", "rotated", "removed", "malformed", "partial"]
)
def test_untrustworthy_log_fails(tmp_path: Path, how: str) -> None:
    store, mark, started_at = _scenario(tmp_path)
    _break(store, how)
    if how == "rotated":
        # Precondition, not product behaviour: a "rotation" that kept the
        # snapshot's identity is undetectable by design, so it must fail here
        # loudly rather than as a misleading verdict (run 43010912496).
        assert mark is not None
        rotated = (store / rsa.CLEANUP_LOG_NAME).stat()
        assert (rotated.st_dev, rotated.st_ino) != mark.identity
    verdict = _attribute(store, mark, started_at)
    assert verdict.explained == {}
    assert "cleanup log" in verdict.unexplained["s1"]


def test_root_marker_is_verified_against_real_ancestry(monkeypatch: pytest.MonkeyPatch) -> None:
    me = rsa.read_process(os.getpid())
    assert me is not None and me.ppid == os.getppid()
    assert me.start <= time.time()
    monkeypatch.setenv(rsa.ROOT_ENV, f"{me.pid}:{me.start}")
    assert rsa.establish_root() == me
    parent = rsa.read_process(os.getppid())
    assert parent is not None
    monkeypatch.setenv(rsa.ROOT_ENV, f"{parent.pid}:{parent.start}")
    assert rsa.establish_root() == parent
    # A forged marker (wrong start, unrelated pid, garbage, absent) is no root.
    for forged in (f"{me.pid}:{me.start - 1}", f"{2**22}:{me.start}", "junk", ""):
        monkeypatch.setenv(rsa.ROOT_ENV, forged)
        assert rsa.establish_root() is None
    monkeypatch.delenv(rsa.ROOT_ENV)
    rsa.record_root()
    assert os.environ[rsa.ROOT_ENV] == f"{me.pid}:{me.start}"
    rsa.record_root(probe=_probe(ROOT))  # an existing marker is never replaced
    assert os.environ[rsa.ROOT_ENV] == f"{me.pid}:{me.start}"


def test_dead_pid_reads_as_none() -> None:
    assert rsa.read_process(2**22 + 7) is None
    assert rsa.read_process(0) is None


def test_names_are_pinned_to_the_product() -> None:
    """A renamed log, policy or actor must fail here, not silently stop excusing."""
    assert rsa.CLEANUP_LOG_NAME == cleanup.CLEANUP_LOG_NAME
    assert delegated_retention.DELEGATED_POLICY in rsa.AUTOMATIC_POLICIES
    planner = inspect.getsource(cleanup)
    for policy in rsa.AUTOMATIC_POLICIES - {delegated_retention.DELEGATED_POLICY}:
        assert f'"{policy}"' in planner, policy
    assert 'actor="startup"' in planner
    for deliberate in (
        cleanup.EXPLICIT_DELETE_POLICY,
        cleanup.MESH_MOVE_POLICY,
        cleanup.MESH_REMNANT_POLICY,
        cleanup.REMOVAL_FAILED_POLICY,
    ):
        assert deliberate not in rsa.AUTOMATIC_POLICIES


# -- the conftest hook itself, on a private copy pointed at a synthetic store --


def _conftest_copy(monkeypatch: pytest.MonkeyPatch, store: Path) -> types.ModuleType:
    """Load the root conftest privately and aim its snapshot at ``store``.

    Private for the same reason as ``test_xdist_worker_budget``: the live copy
    guards this very session. Its wrappers are already installed, so the load
    only re-takes the snapshot, here of the synthetic store.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", os.fspath(store.parent))
    # Registered with monkeypatch so the marker the load may set is undone.
    # And pre-mark ``shutil.rmtree`` as guarded when no live guard is installed
    # (a host with no real store, e.g. CI): otherwise the copy would install
    # process-wide wrappers aimed at this synthetic store for the rest of the
    # session. With the mark the copy takes its snapshot and returns.
    if not getattr(shutil.rmtree, "_local_operator_real_store_guard", False):
        monkeypatch.setattr(shutil.rmtree, "_local_operator_real_store_guard", True, raising=False)
    monkeypatch.setenv(rsa.ROOT_ENV, os.environ.get(rsa.ROOT_ENV, ""))
    spec = importlib.util.spec_from_file_location("_tripwire_under_test", CONFTEST)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module._REAL_STORE == store
    return module


def _session(basetemp: Path) -> types.SimpleNamespace:
    factory = types.SimpleNamespace(_basetemp=basetemp)
    return types.SimpleNamespace(config=types.SimpleNamespace(_tmp_path_factory=factory))


def test_hook_warns_for_an_independent_sweep_and_saves_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _store(tmp_path / "s")
    module = _conftest_copy(monkeypatch, store)
    _remove_genuinely(store, "s1")
    _rewrite_pids(store, SWEEPER.pid)
    monkeypatch.setattr(rsa, "establish_root", lambda: ROOT)
    monkeypatch.setattr(rsa, "read_process", _probe(ROOT, SWEEPER))
    now = frozenset(os.listdir(store))
    module._check_real_store_shrink(_session(tmp_path), now)  # must not exit
    err = capsys.readouterr().err
    assert "INDEPENDENT retention sweep" in err and f"pid {SWEEPER.pid}" in err
    assert "policy=delegated_max_age" in err
    saved = json.loads(next(tmp_path.glob("real-store-shrink-*.json")).read_text())
    assert saved["missing"] == ["s1"] and "s1" in saved["explained"]


def test_hook_still_fails_for_own_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path / "s")
    module = _conftest_copy(monkeypatch, store)
    _remove_genuinely(store, "s1")  # the row carries THIS process's pid
    monkeypatch.setattr(rsa, "establish_root", lambda: ROOT)
    with pytest.raises(pytest.exit.Exception, match="REAL SESSION STORE SHRANK") as raised:
        module._check_real_store_shrink(_session(tmp_path), frozenset(os.listdir(store)))
    assert "this pytest run" in str(raised.value)


def test_hook_fails_closed_without_a_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = _store(tmp_path / "s")
    module = _conftest_copy(monkeypatch, store)
    _remove_genuinely(store, "s1")
    _rewrite_pids(store, SWEEPER.pid)
    monkeypatch.delenv(rsa.ROOT_ENV, raising=False)  # unverifiable root: no excuse
    with pytest.raises(pytest.exit.Exception, match="could not be verified"):
        module._check_real_store_shrink(_session(tmp_path), frozenset(os.listdir(store)))


def test_wrapped_remover_still_refuses_the_protected_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Attribution is after-the-fact only; the in-process refusal is unchanged."""
    store = _store(tmp_path / "s")
    module = _conftest_copy(monkeypatch, store)
    original = getattr(shutil.rmtree, "__wrapped__", shutil.rmtree)
    guarded = module._guarded(original, positions=(0,))
    with pytest.raises(module.RealStoreTouched):
        guarded(store / "s1")
    assert (store / "s1").is_dir()
