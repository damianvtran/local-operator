"""The runtime's self-reaper: the three-term residency predicate (design
§6.1), the grace window, and the clean-exit ordering."""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
import time

import pytest

from local_operator.session.runtime import process as child_mod
from local_operator.session.runtime.process import (
    _clean_exit,
    _clean_ordering_already_ran,
    _reaper,
    _should_exit,
)


class FakeRegistrant:
    def __init__(self, *, supported: bool = False, watchers: int = 0, attaches: int = 0) -> None:
        self.watch_supported = supported
        self.phone_watchers = watchers
        self._attaches = attaches
        self.closed = False

    def attach_clients(self) -> int:
        return self._attaches

    async def aclose(self) -> None:
        self.closed = True


class FakeHandle:
    def __init__(self, *, busy: bool = False, next_wake_ms: int | None = None) -> None:
        self._busy = busy
        self._next_wake_ms = next_wake_ms
        self.disposed = False
        self.denied = False
        self.dispose_order: list[str] = []

    def is_busy(self) -> bool:
        return self._busy

    def next_wake_due_at(self) -> int | None:
        return self._next_wake_ms

    def _deny_pending_gates(self) -> None:
        self.denied = True

    async def dispose(self) -> None:
        self.dispose_order.append("dispose")
        self.disposed = True


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _finishes_with(value: bool) -> bool:
    """A stand-in for one of ``_reaper``'s two normal returns."""
    return value


async def _never_finishes() -> bool:
    """A stand-in for a reaper that is still running (or was cancelled)."""
    await asyncio.sleep(5)
    return True


@pytest.mark.parametrize(
    "reg",
    [
        FakeRegistrant(supported=False),
        FakeRegistrant(supported=True, watchers=1),
        FakeRegistrant(supported=True, watchers=5),
    ],
)
def test_daemon_and_phone_watchers_never_pin_idle_runtime(reg: FakeRegistrant) -> None:
    # Term 3 counts INTERACTIVE viewers only. The phone daemon's adoption
    # connection and its SSE watcher count are daemon-class signals: the
    # daemon adopts every session on the machine, so if they held runtimes
    # warm nothing would ever exit.
    assert _should_exit(FakeHandle(), reg) is True


@pytest.mark.parametrize("attaches", [1, 3])
def test_attached_viewer_holds_idle_runtime(attaches: int) -> None:
    # Term 3: a follower terminal (ClientKind "attach") is the user's
    # attention, and the next thing they do is type — hold the process warm.
    assert _should_exit(FakeHandle(), FakeRegistrant(attaches=attaches)) is False


def test_active_work_holds() -> None:
    assert _should_exit(FakeHandle(busy=True), FakeRegistrant()) is False


def test_wake_inside_warm_window_holds() -> None:
    # Term 2: a wake due in 60 s (the tightest recurrence the wake layer
    # allows) is inside WARM_WINDOW_S, so the runtime stays to fire it itself
    # rather than exiting and cold-starting a minute later.
    assert child_mod.WARM_WINDOW_S > 60.0
    handle = FakeHandle(next_wake_ms=_now_ms() + 60_000)
    assert _should_exit(handle, FakeRegistrant()) is False
    overdue = FakeHandle(next_wake_ms=_now_ms() - 1_000)
    assert _should_exit(overdue, FakeRegistrant()) is False


def test_wake_beyond_warm_window_does_not_hold() -> None:
    # A wake an hour out is cheaper to leave to a cold spawn than to hold
    # ~283 MB for.
    handle = FakeHandle(next_wake_ms=_now_ms() + 3_600_000)
    assert _should_exit(handle, FakeRegistrant()) is True


def test_warm_window_exceeds_min_wake_interval() -> None:
    # The constant pairs with MIN_WAKE_INTERVAL_MS: the window must be wider
    # than the tightest allowed recurrence or a 1-minute wake thrashes
    # exit → spawn → exit forever.
    from local_operator.harness.wake import MIN_WAKE_INTERVAL_MS

    assert child_mod.WARM_WINDOW_S * 1000 > MIN_WAKE_INTERVAL_MS


def test_predicate_tolerates_reduced_handles_and_runtimes() -> None:
    # Older hosts and reduced test doubles lack the accessors; each missing
    # or broken term reads as "does not hold" rather than crashing the reaper
    # or pinning the process.
    class Bare:
        pass

    class Broken:
        def is_busy(self) -> bool:
            return False

        def next_wake_due_at(self) -> int:
            raise RuntimeError("scheduler gone")

        def attach_clients(self) -> int:
            raise RuntimeError("registry gone")

    assert _should_exit(Bare(), Bare()) is True
    assert _should_exit(Broken(), Broken()) is True


@pytest.mark.asyncio
async def test_clean_exit_orders_gates_dispose_unpublish(monkeypatch) -> None:
    order: list[str] = []
    handle = FakeHandle()
    reg = FakeRegistrant(supported=True)

    async def dispose() -> None:
        order.append("dispose")

    async def aclose() -> None:
        order.append("unpublish")

    handle._deny_pending_gates = lambda: order.append("deny")  # type: ignore[method-assign]
    handle.dispose = dispose  # type: ignore[method-assign]
    reg.aclose = aclose  # type: ignore[method-assign]
    await _clean_exit(handle, reg)
    assert order == ["dispose", "unpublish"]


@pytest.mark.asyncio
async def test_grace_elapses_then_clean_exit(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.05)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.2")
    reg = FakeRegistrant(supported=True)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        if stop.is_set():
            break
        await asyncio.sleep(0.05)
    assert stop.is_set()
    assert handle.disposed and not handle.denied and reg.closed
    # The exit leg SAYS it ran the clean ordering: that return is what earns
    # ``amain`` the right to skip its own deny → dispose → aclose block.
    assert await task is True


@pytest.mark.asyncio
async def test_a_reaper_that_wakes_to_a_stop_reports_no_clean_exit(monkeypatch) -> None:
    """The stop-wins race (issue #1250): the reaper parks, someone else sets ``stop``.

    This is the return ``amain`` must still owe its own deny → dispose → aclose
    ordering for. It happens for real whenever a signal drain's bound expires in
    the same loop iteration as a 0.25 s reaper tick: the drain sets ``stop``, the
    reaper wakes to find it already set and leaves with nothing disposed. The
    parent commit read that return through ``reaper.exception() is None`` — which
    is also ``None`` here — so the whole exit block was skipped: no turn
    journal exit note (the row kept ``exit_cause=''``, and the SIGTERM cell
    asserting ``signal_exit_token(row.exit_cause) == "SIGTERM"`` failed with
    ``'' == 'SIGTERM'`` on ``macos-latest`` over four days (issue #1250), no
    gate deny and no dispose.

    DETERMINED HERE RATHER THAN RACED: park the reaper on its first tick, set
    ``stop`` from outside, and read what it says about itself.
    """
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.05)
    handle = FakeHandle()
    reg = FakeRegistrant(supported=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0)  # the first tick parks the reaper on its sleep
    stop.set()
    assert await task is False
    # ...and it really did leave the exit to its caller: nothing was touched.
    assert not handle.disposed and not handle.denied and not reg.closed


@pytest.mark.asyncio
async def test_the_skip_is_earned_only_by_a_clean_exit_return() -> None:
    """``_clean_ordering_already_ran``: ask the reaper, not the absence of a raise.

    The two normal returns of ``_reaper`` differ only in that value, and a
    cancelled task raises rather than answering ``None``, so the three cases
    below are the whole question ``amain`` asks after ``await stop.wait()``.
    """

    clean = asyncio.ensure_future(_finishes_with(True))
    stopped = asyncio.ensure_future(_finishes_with(False))
    await asyncio.gather(clean, stopped)
    assert _clean_ordering_already_ran(clean) is True
    assert _clean_ordering_already_ran(stopped) is False

    cancelled = asyncio.ensure_future(_never_finishes())
    cancelled.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cancelled
    assert _clean_ordering_already_ran(cancelled) is False

    running = asyncio.ensure_future(_never_finishes())
    try:
        assert _clean_ordering_already_ran(running) is False
    finally:
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running


def _own_body(node: ast.AST) -> list[ast.AST]:
    """Every node in a function's OWN body — nested scopes excluded.

    The idiom is ``test_inbox.test_the_drain_is_wired_before_the_socket_starts_
    listening``'s, and for its reason: a call inside a nested ``def`` is not part
    of the statement order an assertion about ``amain``'s own body is about, so
    it must not be able to satisfy one.
    """
    out: list[ast.AST] = []
    stack = list(ast.iter_child_nodes(node))
    while stack:
        child = stack.pop()
        out.append(child)
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(child))
    return out


def test_the_exit_block_is_gated_on_the_reapers_own_return() -> None:
    """THE WIRING, asserted against the source that provides it (#1250, R1-1).

    ``amain`` must ask whether the reaper ran the clean exit ordering by reading
    the reaper's RETURN VALUE. Its predecessor asked ``reaper.exception() is
    None``, and BOTH of ``_reaper``'s normal returns are exception-free, so that
    read credited the skip to a reaper that had disposed nothing and the whole
    exit block below it was skipped — the defect this PR fixes.

    WHY A SOURCE ASSERTION. Nothing that runs here can discriminate: the one cell
    that reaches this branch is the ``tui-e2e`` journal cell
    (``tests/e2e/test_session_survival_journal_e2e.py``), and it fails only ~6%
    of runs on the parent — so a revert to ``reaper.exception() is None`` would
    pass it ~94% of the time, and a green run proves nothing.

    AT THE ROUND-1 REVIEW THAT LEG DID NOT RUN AT ALL, and naming the real
    reason is deliberate so that nobody reads it as a rule about diffs:
    ``tui-e2e`` is ``needs: [changes, lint, type-check]`` and gates on
    ``needs.lint.result == 'success'``, so the unrelated ``lint`` red inherited
    from the tree (an ``isort`` failure in ``test_session_delete.py``, since
    fixed by #1371) skipped all six shards — while the ``changes`` classifier
    itself printed ``tui = true`` / ``tui-e2e: run``. ``tui`` is ``unit``'s
    predicate, not "this diff reaches an e2e file" (see ``FLAG_REASONS`` in
    ``scripts/ci_scope.py``), and this PR's e2e legs ran on every head whose
    ``lint`` was green.

    The other tests here pin the pieces (``_reaper``'s return value, and
    ``_clean_ordering_already_ran``'s contract against stand-in futures); this
    pins the call site that joins them, which is the line whose absence caused
    the defect.

    PARSED, NOT SUBSTRING-MATCHED, for the reason ``test_inbox`` spells out: a
    substring match can be satisfied by prose, and this module's own comments
    necessarily name ``reaper.exception()`` while explaining why it is gone.

    WHAT IT PINS, SO A CORRECT CHANGE UPDATES THIS TEST RATHER THAN BEING
    REWORKED AROUND IT. Two narrownesses are deliberate, and both are about the
    answer coming from the reaper's RETURN VALUE at this call site. The verdict
    must be the ``if``'s test DIRECTLY: hoisting it into a local
    (``verdict = _clean_ordering_already_ran(reaper)``, then ``elif verdict:``)
    is correct code this assertion rejects. And ``offenders`` forbids ANY read
    of ``<reaper>.exception()`` in ``amain``'s own body, a logging-only one
    included. A change that keeps the property but moves those shapes is a
    change to this pin, not a reason to rework the change.
    """
    source = textwrap.dedent(inspect.getsource(child_mod.amain))
    tree = ast.parse(source)
    body = _own_body(tree.body[0])

    def starts_a_reaper(value: ast.AST) -> bool:
        """Does this expression start ``_reaper``? ``ensure_future`` is one wrapper of many."""
        return any(
            isinstance(inner, ast.Call)
            and isinstance(inner.func, ast.Name)
            and inner.func.id == "_reaper"
            for inner in ast.walk(value)
        )

    def reaper_local() -> str:
        """The local ``_reaper``'s task is bound to — ``reaper`` today, whatever after.

        Resolved from the construction rather than hard-coded, so renaming the
        local cannot fail a valid classification (the convention
        ``test_inbox``'s ``runtime_name`` sets).
        """
        for node in body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and starts_a_reaper(node.value)
            ):
                return node.targets[0].id
        raise AssertionError("amain no longer starts a _reaper task")

    name = reaper_local()

    asks = [
        node
        for node in body
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_clean_ordering_already_ran"
        and any(isinstance(arg, ast.Name) and arg.id == name for arg in node.args)
    ]
    assert asks, f"amain must gate the exit block on _clean_ordering_already_ran({name})"

    def assigns_ran_clean(node: ast.AST) -> bool:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            return False
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        return any(
            isinstance(target, ast.Name) and target.id == "reaper_ran_clean_exit"
            for target in targets
        )

    assert any(
        isinstance(node, ast.If)
        and node.test is asks[0]
        and any(assigns_ran_clean(stmt) for stmt in node.body)
        for node in body
    ), "the reaper's answer must be what sets reaper_ran_clean_exit"

    offenders = [
        node
        for node in body
        if isinstance(node, ast.Attribute)
        and node.attr == "exception"
        and isinstance(node.value, ast.Name)
        and node.value.id == name
    ]
    assert not offenders, (
        f"amain must not read {name}.exception() itself: BOTH of _reaper's normal "
        "returns are exception-free, so that read credits the skip to a reaper that "
        "ran no exit ordering and drops the exit note entirely (issue #1250, R1-1)"
    )


@pytest.mark.asyncio
async def test_phone_watchers_do_not_change_idle_timing(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.01)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.08")
    elapsed: list[float] = []
    for watchers in (0, 1, 5):
        reg = FakeRegistrant(supported=bool(watchers), watchers=watchers)
        handle = FakeHandle()
        stop = asyncio.Event()
        started = asyncio.get_running_loop().time()
        await _reaper(handle, reg, stop)
        elapsed.append(asyncio.get_running_loop().time() - started)
    assert max(elapsed) - min(elapsed) < 0.04


@pytest.mark.asyncio
async def test_attached_viewer_holds_then_release_starts_drain(monkeypatch) -> None:
    """The TUI-closes-while-idle path (design §6.1): the attach count drops
    to zero, the drain starts THEN, and the runtime exits one grace later."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.15")
    reg = FakeRegistrant(attaches=1)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.4)  # well past the grace had the viewer not held it
    assert not stop.is_set() and not handle.disposed
    reg._attaches = 0  # the terminal closed
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        if stop.is_set():
            break
        await asyncio.sleep(0.02)
    assert stop.is_set() and handle.disposed and reg.closed
    await task


@pytest.mark.asyncio
async def test_viewer_attaching_mid_drain_cancels_it(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.15")
    reg = FakeRegistrant()
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.08)  # inside the drain
    reg._attaches = 1
    await asyncio.sleep(0.3)
    assert not stop.is_set() and not handle.disposed
    reg._attaches = 0
    await task
    assert stop.is_set()


@pytest.mark.asyncio
async def test_wake_in_window_holds_until_it_passes(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.1")
    reg = FakeRegistrant()
    handle = FakeHandle(next_wake_ms=_now_ms() + 30_000)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.3)
    assert not stop.is_set()
    handle._next_wake_ms = None  # the wake was cancelled (or retired)
    await task
    assert stop.is_set() and handle.disposed


@pytest.mark.asyncio
async def test_busy_session_defers_grace_start(monkeypatch) -> None:
    """A turn mid-flight when the last front end leaves must NOT start the
    clock; grace begins at turn end and outlives the turn by construction."""
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.05)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.2")
    reg = FakeRegistrant(supported=True)
    handle = FakeHandle(busy=True)
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.5)  # past the whole grace had it started at t=0
    assert not stop.is_set()
    handle._busy = False  # the turn ends NOW
    deadline = asyncio.get_running_loop().time() + 3
    while asyncio.get_running_loop().time() < deadline:
        if stop.is_set():
            break
        await asyncio.sleep(0.05)
    assert stop.is_set()
    await task


@pytest.mark.asyncio
async def test_new_activity_resets_idle_drain(monkeypatch) -> None:
    monkeypatch.setattr(child_mod, "REAP_CHECK_S", 0.02)
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "0.12")
    # Phone watchers only: an attach client would (correctly) hold forever.
    reg = FakeRegistrant(supported=True, watchers=3)
    handle = FakeHandle()
    stop = asyncio.Event()
    task = asyncio.ensure_future(_reaper(handle, reg, stop))
    await asyncio.sleep(0.08)
    handle._busy = True
    await asyncio.sleep(0.1)
    handle._busy = False
    await asyncio.sleep(0.08)
    assert not stop.is_set()
    await task
    assert stop.is_set()


def test_grace_env_override_and_defaults(monkeypatch) -> None:
    monkeypatch.delenv("LOP_SESSION_GRACE_S", raising=False)
    assert child_mod._grace_seconds() == child_mod.DEFAULT_GRACE_S == 3.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "10")
    assert child_mod._grace_seconds() == 10.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "not-a-number")
    assert child_mod._grace_seconds() == 3.0
    monkeypatch.setenv("LOP_SESSION_GRACE_S", "-5")
    assert child_mod._grace_seconds() == 3.0


class TestSpeculativeSessionCleanup:
    """The exit path no longer removes a session directory — ever.

    #622 added `_remove_unwritten_session_dir` here to tidy the lease-only
    directory a warm-but-unused runtime leaves behind. It fired on the
    operator's real store (`removed unwritten speculative session directory
    <id>` in the runtime logs) during the incident that lost 225 sessions,
    and the rule since is that no exit hook removes a session directory on
    its own judgement. The lease-only directory is what the user-enabled
    `session.cleanup.remove_empty` policy is for.
    """

    def test_the_exit_path_has_no_directory_remover(self) -> None:
        import inspect

        from local_operator.session.runtime import process

        assert not hasattr(process, "_remove_unwritten_session_dir")
        source = inspect.getsource(process)
        assert "rmdir(" not in source and "rmtree(" not in source


def test_the_runtime_child_logs_to_its_own_bounded_file_apart_from_the_daemon(
    tmp_path, monkeypatch
) -> None:
    """The runtime's log must be its OWN, bounded, and attributable.

    Three properties, each from a measured failure on the operator's machine:

    * it goes to ``logs/runtime.log``, NOT the daemon's ``logs/mobile.log``. The
      daemon's file is a launchd ``StandardOutPath`` it appends to through an fd
      it never reopens, and bounding a file means RENAMING it: measured after one
      rename of that path, nine runtime children held the renamed inode while
      only the daemon held the fresh file, so ``lop mobile logs`` (a
      ``tail mobile.log``) showed the daemon and none of its children. A test
      that let the child write the daemon's path again would reintroduce exactly
      that split.
    * it is BOUNDED — the ``logging.basicConfig(level=INFO, filename=...)`` this
      replaces wrote a 420 MB file with nothing rotating it.
    * the wire clients are pinned, because at the root's INFO they emitted one
      record per HTTP request — 557,352 of the 420 MB file's lines.

    It also has to say who it is: the file is written by every runtime child, so
    a line with no pid is a line no one can attribute.
    """
    import logging
    import logging.handlers
    import os
    from pathlib import Path

    from local_operator.logger import LOG_BACKUP_COUNT, LOG_MAX_BYTES
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.session.runtime import process

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))

    async def fake_amain(**_kwargs: object) -> int:
        # ``**kwargs`` because ``main`` passes the operator capability through
        # to the real ``amain`` (issue #1310); a double that pins the signature
        # would fail on a parameter this test is not about.
        return 0

    monkeypatch.setattr(process, "amain", fake_amain)
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    client = logging.getLogger("httpx2")
    saved_client_level = client.level
    try:
        assert process.main() == 0
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        runtime_log = tmp_path / "logs" / "runtime.log"
        assert Path(handler.baseFilename) == runtime_log
        assert not (tmp_path / "logs" / "mobile.log").exists(), (
            "the runtime wrote the daemon's launchd-owned log; a rotation there "
            "renames the file out of what `lop mobile logs` reads"
        )
        assert handler.maxBytes == LOG_MAX_BYTES
        assert handler.backupCount == LOG_BACKUP_COUNT
        assert client.level >= logging.WARNING
        assert f"pid {os.getpid()}" in runtime_log.read_text(encoding="utf-8")
    finally:
        for open_handler in list(root.handlers):
            open_handler.close()
            root.removeHandler(open_handler)
        for handler in saved_handlers:
            root.addHandler(handler)
        root.setLevel(saved_level)
        client.setLevel(saved_client_level)
