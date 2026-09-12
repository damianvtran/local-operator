"""Suite-wide isolation from the developer's machine.

This file was previously a commented-out `event_loop` fixture — zero effective
lines — so NOTHING isolated the environment. Two defects of the same shape were
found in the same week as a result:

* a test patched ``exec_mode._ensure_logs_dir`` but not the ``LOGS_DIR``
  constant resolved from ``Path.home()`` at import time, so it passed on a
  machine that happened to have that directory and failed on a clean one;
* a test asserting a "missing API key" error failed on any machine that
  exported that provider's key, because ``AuthStore`` consults the process
  environment as one tier of its resolution cascade.

Both are the same root cause: the suite read ambient state. Isolating it once
here is the fix; per-test monkeypatching is a fix that has to be remembered
every time, and it was not.

Deliberately NOT isolated: ``OPENROUTER_API_KEY`` and the other keys are
cleared, which means a test that genuinely wants a live provider must set it
explicitly. That is the right default — a unit test reaching the network
because the developer happened to have a key exported is a worse failure than
one that has to opt in.
"""

from __future__ import annotations

import logging
import os
import shutil
import signal
import sys
import time
from collections.abc import Iterator
from contextlib import suppress
from pathlib import Path

import pytest

#: Environment variables that steer credential resolution, config discovery,
#: provider selection — or NAME A REAL-MACHINE RESOURCE. Any of these leaking
#: in from the developer's shell can change what the code under test does, or
#: point it at something live. ``tests/unit/test_ambient_env_isolation.py``
#: walks the package for every variable production code reads and fails when
#: one is neither here nor explained there; add new entries HERE when the
#: variable names a session, window, socket, directory or credential.
_AMBIENT_VARS = (
    "LOCAL_OPERATOR_CONFIG_DIR",
    "LOCAL_OPERATOR_DESKTOP_TOKEN",
    "LOCAL_OPERATOR_DESKTOP_ORIGINS",
    "LOCAL_OPERATOR_HOME",
    "LOCAL_OPERATOR_DEBUG",
    # Names the session a `lop secret` retrieval is attributed to in the audit
    # trail. Inherited from the operator's own runtime it would write their
    # real session id into a sandboxed store's audit rows.
    "LOCAL_OPERATOR_SESSION_ID",
    # Tests launched from a detached operator inherit these runtime-only flags.
    # They turn strict --resume validation into adoption of a brand-new id.
    "LOP_RUNTIME_ADOPT_SESSION",
    "LOP_RUNTIME_DEFER_MATERIALISE",
    # The e2e stage's fake install prefix for the runtime self-refresh: a
    # runtime that inherited it would compare its boot stamp against a
    # directory the test owns rather than its real install, and could
    # retire (or refuse to) on a stranger's marker.
    "LOP_BUILD_PREFIX",
    # A runtime child spawned by the mobile daemon carries its session id,
    # provider, model and cwd here. A test suite run from inside such a
    # session (agents do this) inherited LOP_MOBILE_CHILD_RESUME and created
    # THAT id inside its store (QA round 1 of #645).
    "LOP_MOBILE_CHILD_RESUME",
    "LOP_MOBILE_CHILD_PROVIDER",
    "LOP_MOBILE_CHILD_MODEL",
    "LOP_MODEL_SELECTION_OVERRIDE",
    # The eval worker's scrub-channel transport (R1). The parent sets it per
    # spawn, but a worker that inherited a STALE value from the operator's own
    # runtime would publish retrieved secret values onto whatever that number
    # names in the test process — an arbitrary fd, not the pipe the parent is
    # holding. The parent's explicit env override makes that unreachable in
    # practice; scrubbing it keeps the guarantee at the fixture rather than
    # resting on one caller always remembering to set it.
    "LOCAL_OPERATOR_EVAL_SCRUB_FD",
    "LOP_MOBILE_CHILD_CWD",
    "LOP_MOBILE_PASSWORD",
    # The calling cmux workspace/surface. A headless fork e2e test inherited
    # these through an isolated HOME and renamed the operator's LIVE window
    # (#648). Nothing in a test may address a real pane.
    "CMUX_WORKSPACE_ID",
    "CMUX_SURFACE_ID",
    "CMUX_PANEL_ID",
    "CMUX_TAB_ID",
    "CMUX_SOCKET",
    # The calling Herdr pane: the identical hazard as CMUX_* — a test run from
    # inside a Herdr pane would take over that pane's Agents row and then
    # RELEASE it on exit. Nothing in a test may address a real Herdr pane.
    "HERDR_ENV",
    "HERDR_PANE_ID",
    "HERDR_BIN_PATH",
    "HERDR_SOCKET_PATH",
    "HERDR_TAB_ID",
    "HERDR_WORKSPACE_ID",
    # Redirects the imported user-scope instruction paths. HOME is already
    # scrubbed below, which covers the DEFAULT ``~/.agents/AGENTS.md``, but the
    # override names absolute paths and would survive that — a developer who
    # exports it gets a different prompt than CI from the same tree.
    "LOCAL_OPERATOR_ECOSYSTEM_INSTRUCTIONS",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    # The provider registry's ONLY callable ``env_keys`` resolver prefers this
    # over the API key, so leaving it ambient can flip both the value a test
    # resolves and the credential KIND reported for it.
    "ANTHROPIC_OAUTH_TOKEN",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "GOOGLE_API_KEY",
    "GOOGLE_AI_STUDIO_API_KEY",
    "GEMINI_API_KEY",
    "XAI_API_KEY",
    "KIMI_API_KEY",
    "MOONSHOT_API_KEY",
    "ALIBABA_CLOUD_API_KEY",
    "DASHSCOPE_API_KEY",
    "RADIENT_API_KEY",
    "SERPAPI_API_KEY",
    "TAVILY_API_KEY",
    "FAL_API_KEY",
    "ZAI_API_KEY",
    "HF_TOKEN",
)


@pytest.fixture(scope="session", autouse=True)
def warm_tiktoken_encoding() -> None:
    """Download the BPE table ONCE, before any test measures the event loop.

    ``tiktoken.get_encoding`` caches under ``tempfile.gettempdir()`` and
    DOWNLOADS the table on a miss. Measured here: 1239 ms cold, 0 ms warm. The
    compaction rulers call it inline on the event loop for small histories —
    correct, because the thread hop costs more than the encode it saves — but
    nothing in that threshold anticipated a multi-second network call hiding
    behind the first invocation.

    That made ``test_the_loop_stays_responsive_while_several_subagents_run``
    fail on CI and pass locally: a developer box has run the tokenizer before,
    a fresh runner has not. It is the same class of problem
    ``isolate_environment`` exists for — a test must not depend on ambient
    machine state, and "has this machine downloaded the BPE table?" is exactly
    that.

    Warming rather than loosening the assertion is deliberate. That bound is
    calibrated evidence (1353 ms before the compaction fix, 139 ms after), so
    widening it to swallow a cold download would blind it to the regression it
    exists to catch.

    Warmed through the project's own ``_get_encoding`` rather than by calling
    ``tiktoken.get_encoding`` directly, so it primes the module-level cache the
    rulers actually read as well as the on-disk BPE file. Under ``xdist`` each
    worker is its own process and runs this fixture itself; only the first pays
    the download, because the disk cache is shared.

    WHY THIS CHECKS THE CACHE FILE INSTEAD OF JUST CATCHING THE FAILURE. An
    earlier version wrapped the call in ``except Exception: pass``, reasoning
    that an offline box would simply fall through. It does not: with the
    network unreachable, ``_get_encoding()`` blocks for **75.7 s** (measured,
    via a dead proxy) inside urllib's retry ladder before giving up and
    returning ``None``. Swallowing the exception makes the fixture free only
    once it has already cost every offline test session more than a minute.
    tiktoken has no connect timeout to configure here, so the fix is not to
    catch the failure faster but to avoid attempting the download at all.

    So: derive the cache path exactly as tiktoken does — SHA-1 of the BPE URL,
    under ``TIKTOKEN_CACHE_DIR`` / ``DATA_GYM_CACHE_DIR`` / ``<tmp>/
    data-gym-cache`` — and warm ONLY when the file is already there. A machine
    with a cold cache and no network skips instantly and keeps the chars/4
    fallback it would have used anyway. A machine with a cold cache and a
    working network is the one case still paying the download, and it pays it
    inside whichever test touches the tokenizer first, exactly as before this
    fixture existed.

    ``LOCAL_OPERATOR_WARM_TIKTOKEN=1`` forces the download for a CI image that
    wants to populate the cache deliberately.
    """
    import hashlib
    import tempfile

    bpe_url = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
    cache_dir = (
        os.environ.get("TIKTOKEN_CACHE_DIR")
        or os.environ.get("DATA_GYM_CACHE_DIR")
        or os.path.join(tempfile.gettempdir(), "data-gym-cache")
    )
    cached = os.path.join(cache_dir, hashlib.sha1(bpe_url.encode()).hexdigest())
    if not os.path.exists(cached) and os.environ.get("LOCAL_OPERATOR_WARM_TIKTOKEN") != "1":
        return

    try:
        from local_operator.compaction.tokens import _get_encoding

        _get_encoding()
    except Exception:  # noqa: BLE001 — warming is an optimisation, never a gate
        pass


@pytest.fixture(autouse=True)
def isolate_environment(tmp_path_factory, monkeypatch):
    """Point HOME at a scratch dir and clear provider/config env vars.

    Autouse and function-scoped: every test gets a fresh HOME, so nothing can
    read or write the developer's real ``~/.local-operator`` (auth.db included)
    and no test can be made to pass by ambient credentials.

    Deliberately ONLY the environment. An earlier version also patched
    ``Path.home`` and ``os.path.expanduser``, which broke 38 tests: several set
    HOME themselves and assert on path shortening, and ``expanduser`` is called
    with ``Path`` objects as well as ``str``. ``Path.home()`` reads HOME on
    POSIX and USERPROFILE on Windows, so setting both is sufficient and leaves
    a test free to override HOME for its own purposes.
    """
    home = tmp_path_factory.mktemp("home")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))  # Windows equivalent
    for name in _AMBIENT_VARS:
        monkeypatch.delenv(name, raising=False)
    # The suite must never reach the operator's DESKTOP either. A runtime with
    # no attached client announces a parked gate through `detached_notify`,
    # which on darwin spawns a real `osascript display notification` — so six
    # tests driving real gates put 100 genuine toasts in Notification Centre,
    # titled "lop needs you" with fixture strings as bodies. Nothing in a
    # green suite reveals that: the spawn is fire-and-forget and its failure
    # is swallowed by design.
    #
    # This is the same defect class as the launchd escapes (a test reaching
    # the real machine through a side effect the assertions never look at),
    # and it gets the same answer: gate it centrally, once, for every test.
    # A test that specifically exercises the notification path unsets or
    # monkeypatches around this, which is the visible, deliberate opt-in.
    monkeypatch.setenv("LOCAL_OPERATOR_NO_NOTIFICATIONS", "1")
    yield home


@pytest.fixture(autouse=True)
def isolate_implicit_local_discovery(monkeypatch):
    """Keyless discovery must not read a developer's default-port runtimes.

    Activation now refreshes live capacity, unlike a hosted catalogue cache.
    Explicit test clients and configured random-port HTTP fixtures remain real;
    an implicit preset lookup gets the ordinary unavailable-server fallback.
    """
    from local_operator.model import discovery
    from local_operator.providers.local import LOCAL_PRESETS, resolve_base_url

    fetch = discovery.fetch_models

    def isolated(provider_id, **kwargs):
        if provider_id in LOCAL_PRESETS and kwargs.get("client") is None:
            endpoint = resolve_base_url(provider_id, override=kwargs.get("base_url"))
            if endpoint == LOCAL_PRESETS[provider_id][1]:
                return None
        return fetch(provider_id, **kwargs)

    monkeypatch.setattr(discovery, "fetch_models", isolated)


@pytest.fixture(autouse=True)
def reset_store_maintenance() -> Iterator[None]:
    """Give every test a process that has not yet swept the session store.

    ``session_factory`` runs its four whole-store maintenance passes ONCE per
    process (see ``_start_store_maintenance``): a store does not become dirty
    again because the user pressed ``/resume``, and re-sweeping cost every
    resume the full walk. That guard is module-global, so without this fixture
    the FIRST test to call ``_prepare`` in an interpreter is the only one whose
    sweeps run, and every later test silently gets a no-op — which is how
    ``test_prepare_store_scans_do_not_stall_the_loop`` fails only when it runs
    after ``test_prepare_claims_before_a_concurrent_sweep_can_reap_the_dir``
    and passes alone. Same class of leaked global state as the root logger
    below, answered the same way rather than per-test.
    """
    from local_operator.session_factory import reset_store_maintenance_for_tests

    reset_store_maintenance_for_tests()
    try:
        yield
    finally:
        reset_store_maintenance_for_tests()


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """Give every test the process-global logging state back as it found it.

    Logging is the one piece of global state a Python test suite cannot avoid
    sharing, and this suite had no isolation for it at all. The failure that
    forced this fixture: importing ``local_operator.server.app`` — which
    collection does for the whole session the moment one server test module is
    selected — left a stderr ``StreamHandler`` on the root logger, so
    ``tests/unit/mcp/test_auth.py``'s "the browser's chatter goes to the LOG,
    not the terminal" assertion saw the log ON the terminal and failed. Alone
    it passed. That is the signature of leaked global state, and answering it
    per-test is a fix that has to be remembered every time.

    Restores what the modules under test actually mutate: root handlers and
    level, plus ``lastResort``, ``raiseExceptions`` and ``Logger.addHandler``,
    which :mod:`local_operator.logger`'s silencing patches in place. This
    replaces the identical fixtures that ``tests/unit/test_logger.py`` and
    ``tests/unit/tui/test_logger_silence.py`` each kept locally: they were
    right, they were just scoped to the two files that already knew.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_last_resort = logging.lastResort
    saved_raise = logging.raiseExceptions
    saved_add_handler = logging.Logger.addHandler
    try:
        yield
    finally:
        logging.Logger.addHandler = saved_add_handler  # type: ignore[method-assign]
        logging.lastResort = saved_last_resort
        logging.raiseExceptions = saved_raise
        root.handlers[:] = saved_handlers
        root.setLevel(saved_level)


@pytest.fixture
def terminal_output(tmp_path) -> Iterator[Path]:
    """Everything written to file descriptor 2 during the test, as a file.

    For asserting that nothing reached the TERMINAL. Monkeypatching
    ``sys.stderr`` to a ``StringIO`` — the usual move, and the right one for an
    in-process ``StreamHandler`` — cannot see this: a spawned child inherits the
    DESCRIPTOR, not the Python object, so it paints the real screen while the
    buffer stays empty. A test built on the buffer passes with the defect fully
    present; that is how an MCP server's startup banner reached a user's boot
    splash with a green suite behind it.

    ``os.dup2`` and not ``contextlib.redirect_stderr`` for the same reason.
    Yields the sink path; read it back at the end of the test. Restored on the
    way out, including on failure, or pytest loses its own error stream.
    """
    path = tmp_path / "terminal-fd2.bin"
    sink = open(path, "wb")
    saved = os.dup(2)
    try:
        os.dup2(sink.fileno(), 2)
        yield path
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        sink.close()


@pytest.fixture(autouse=True)
def fresh_served_selectors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with no model selector recorded as served.

    ``failover._SERVED_SELECTORS`` is process-wide by design — a served id is
    proof the id exists for every session in the process — which makes it
    cross-test state: any test that streams a success on ``openai/gpt-4o``
    would turn a later test's flat unknown-model 400 on that id into a
    catalogue flap and re-ask it three times instead of aborting at once.
    Several suites drive ``stream_with_failover``, so the reset lives here
    rather than in one test module. Tests that need served evidence set it
    explicitly, which is also the honest way to state that precondition.
    """
    import local_operator.providers.failover as failover

    monkeypatch.setattr(failover, "_SERVED_SELECTORS", set())


#: Where `pytest_runtest_call` leaves the temp roots it saw while they existed.
#: Read back by `_secret_config_dirs` in the sweep's teardown; see that function
#: and the hook for why a path that can still be NAMED after its directory is
#: gone is the whole trick.
_SWEEP_ROOT_KEY: pytest.StashKey[tuple[Path, ...]] = pytest.StashKey()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item) -> Iterator[None]:
    """Reap this test's brokers, and record its temp roots, while both still exist.

    Two jobs, and the sweep's own teardown can do neither of them:

    * **Reap.** A config dir shallow enough that its socket lives INSIDE it —
      ``sun_path`` is 104 bytes, and Linux CI runs with ``TMPDIR=/tmp``, where a
      pytest basetemp is short enough to qualify — loses that socket file when
      pytest's ``tmp_path`` fixture removes the directory at ITS teardown. From
      then on nothing path-derived can reach the daemon, so a sweep running only
      at teardown cannot stop it: measured with ``--basetemp=/tmp/...``, one run
      of ``tests/unit/secrets/test_cli.py`` left 31 live key-holding brokers on
      exactly that layout. This is the last moment the socket exists in EVERY
      layout, so the reap for that case has to be here.

    * **Record.** `tmp_path` removes its own directory at ITS teardown whenever
      the retention policy is ``failed`` and the test passed — unconditionally,
      in pytest's own fixture generator (`_pytest/tmpdir.py`) — and it always
      tears down BEFORE an autouse fixture declared here: those are finalised
      last, because they are set up first. So by the time the sweep looked for
      its candidates, the directories a DEEP config dir's socket is derivable
      from were gone and ``rglob`` found nothing; measured on this machine, one
      run of ``tests/unit/secrets/test_cli.py`` left 28 live brokers that way.
      ``socket_path()`` is a pure function of the directory NAME, so recording
      the paths here lets the teardown sweep still reach a fallback-layout
      socket — and catch a broker a fixture finaliser restarts after this point.

    The teardown sweep stays: it is idempotent, and it is what covers that
    restart. The two together cover both layouts.
    """
    try:
        yield
    finally:
        roots = _temp_roots(item)
        item.stash[_SWEEP_ROOT_KEY] = roots
        _record_for_session_sweep(roots)
        _reap_brokers_of_this_test(item)


#: Config dirs this worker's tests could have started a broker under, kept for
#: the session-end net below. A module-level set rather than a stash entry
#: because the net runs after every item's stash is out of reach, and a plain
#: set of paths is ~100 bytes per entry for a suite that creates a few thousand.
_SESSION_SWEEP_ROOTS: set[Path] = set()

#: This session's ``basetemp``, captured by the fixture below.
#:
#: Captured through the PUBLIC ``tmp_path_factory`` fixture rather than read off
#: ``config._tmp_path_factory`` at session end: that attribute is private (and
#: pyright rejects it), and building a second ``TempPathFactory`` from the config
#: would allocate a DIFFERENT basetemp than the one the tests actually used,
#: which would make the net walk an empty directory and silently reclaim nothing.
_SESSION_BASETEMP: Path | None = None


@pytest.fixture(scope="session", autouse=True)
def _capture_session_basetemp(tmp_path_factory: pytest.TempPathFactory) -> None:
    """Record this session's basetemp for the session-end broker net.

    Session-scoped and autouse so it is resolved once per worker, at the first
    test, and costs nothing thereafter; ``getbasetemp()`` is what the factory has
    already computed for ``tmp_path``, so this creates no directory of its own.
    """
    global _SESSION_BASETEMP
    with suppress(Exception):
        _SESSION_BASETEMP = tmp_path_factory.getbasetemp().resolve()


def _record_for_session_sweep(roots: tuple[Path, ...]) -> None:
    """Remember this test's temp roots for the session-end net, scoped to basetemp.

    THE BASETEMP FILTER IS THE SAFETY PROPERTY, and it is the same one the
    per-test sweep relies on: a path is recorded only when it lies under this
    run's own ``basetemp``, so the net can never reach the operator's real
    ``~/.local-operator`` or another agent's worktree — exactly the processes
    `stop_secret_brokers_started_by_this_test` documents it must not touch. The
    isolated HOME is deliberately NOT recorded: `isolate_environment` may point
    it outside basetemp, and the per-test reap already covers it while it exists.
    """
    base = _SESSION_BASETEMP
    if base is None:
        return
    for root in roots:
        with suppress(ValueError, OSError):
            if root.resolve().is_relative_to(base):
                _SESSION_SWEEP_ROOTS.add(root)


def _session_sweep_candidates(basetemp: Path | None) -> list[Path]:
    """Config dirs under ``basetemp`` that a session-end sweep should ask about.

    Two sources, because neither alone is complete:

    * the roots recorded per test, which is the only way to name a directory
      pytest has since RECLAIMED — a fallback-layout socket outlives its config
      dir, so the name is still enough to reach the daemon (see `_temp_roots`);
    * a walk of ``basetemp`` for ``secrets/`` directories, which is the only way
      to see a config dir that never belonged to any test's ``tmp_path`` at all.
      A module- or session-scoped fixture takes its directory from
      ``tmp_path_factory.mktemp``, so it appears in NO test's ``tmp_path`` and no
      per-test record can contain it. Measured: a broker started in a
      module-scoped fixture's teardown survived a full inner run.

    ``secrets/`` is the marker because it is what the store actually creates
    (`keys.secrets_dir`), so the walk names config dirs rather than guessing at
    directory shapes — the same argument `_secret_config_dirs` records for not
    hardcoding a list of known layouts. The walk runs ONCE per session, not per
    test, which is what keeps it affordable.
    """
    candidates = set(_SESSION_SWEEP_ROOTS)
    _SESSION_SWEEP_ROOTS.clear()
    if basetemp is not None:
        with suppress(Exception):
            from local_operator.secrets.keys import SECRETS_DIRNAME

            for marker in basetemp.rglob(SECRETS_DIRNAME):
                with suppress(OSError):
                    if marker.is_dir():
                        candidates.add(marker.parent)
    return sorted(candidates)


def _sweep_session_leftovers(basetemp: Path | None) -> int:
    """Stop every leftover broker under ``basetemp``; return how many were live.

    **Belt and braces, not the primary mechanism.** The call-phase reap is what
    stops brokers in practice, and the per-test teardown sweep covers a
    function-scoped finaliser that restarts one. This exists for what neither can
    structurally see: a broker whose config dir was never a test's ``tmp_path``
    (see `_session_sweep_candidates`). Issue #958 asked for a net that also
    *reports what it reclaimed*, so a non-zero count here is a signal that some
    path leaks past the per-test reap and should be traced, rather than a number
    quietly absorbed.

    THE BASETEMP SCOPE IS THE SAFETY PROPERTY, and it is the same one the
    per-test sweep rests on: every candidate lies under this run's own basetemp,
    so this can never reach the operator's real ``~/.local-operator`` or another
    agent's worktree. Without a basetemp there is nothing safe to scope to and
    the net does nothing at all.

    Returns the number of config dirs that still had a LIVE broker, which is the
    count worth reporting: candidates that were already clean are the expected
    case and say nothing.
    """
    if not _broker_daemon_is_available():
        return 0
    candidates = _session_sweep_candidates(basetemp)
    if not candidates:
        return 0
    try:
        from local_operator.secrets import client
    except Exception:  # noqa: BLE001 — a net that cannot load is simply absent
        return 0

    # `is_running` is the cheap question and the same one `_stop_brokers_in`
    # asks; counting first is what makes the reported number "reclaimed" rather
    # than merely "considered".
    live = []
    for candidate in candidates:
        with suppress(Exception):
            if client.is_running(candidate):
                live.append(candidate)
    if live:
        _stop_brokers_in(live)
    return len(live)


def pytest_sessionfinish(session: pytest.Session) -> None:
    """Run the session-end broker net and say what it reclaimed.

    Written to ``sys.stderr`` rather than through the terminal reporter: under
    ``-n auto`` this hook runs in each xdist WORKER, which has no reporter
    plugin, and the worker's stderr is what ends up in the CI log where an
    orphaned-process report is actually read. Silent at zero — the expected
    outcome must not add a line to every run.
    """
    del session
    with suppress(Exception):
        reclaimed = _sweep_session_leftovers(_SESSION_BASETEMP)
        if reclaimed:
            print(
                f"[broker-sweep] session-end net reclaimed {reclaimed} live broker(s) "
                "the per-test reap did not stop; see tests/conftest.py",
                file=sys.stderr,
            )


def _broker_daemon_is_available() -> bool:
    """Is there a broker daemon on this platform to sweep at all?

    A named seam rather than a bare ``os.name`` read inside the sweep, because the
    one platform where this is False is the one a developer's box cannot run:
    `test_broker_sweep.py` pins the guard by patching THIS and watching a real
    broker survive, which the bare form cannot be asked without having `pathlib`
    hand out `WindowsPath` objects on POSIX.
    """
    return os.name != "nt"


def _reap_brokers_of_this_test(item: pytest.Item) -> None:
    """Stop every broker reachable from this test's own temp roots.

    The isolated HOME is read off the fixture's VALUE rather than the module
    constant, because `isolate_environment` is what put it there; the candidate
    set is built by the same `_sweep_candidates` the teardown sweep uses, so the
    two can never drift into disagreeing about what this test owns.
    """
    home = getattr(item, "funcargs", {}).get("isolate_environment")
    _stop_brokers_in(_sweep_candidates(item, home if isinstance(home, Path) else None))


def _temp_roots(item: pytest.Item) -> tuple[Path, ...]:
    """``tmp_path`` and every directory under it, as they were at call time.

    Every directory, rather than a list of the shapes the suite is known to use:
    the module docstring of `_secret_config_dirs` records where a hardcoded list
    already failed. `Exception`, not `OSError`, for the same reason the sweep
    itself suppresses broadly — a test may have replaced the path machinery this
    walk depends on, and discovering nothing is the right failure here.
    """
    tmp_path = getattr(item, "funcargs", {}).get("tmp_path")
    if not isinstance(tmp_path, Path):
        return ()
    with suppress(Exception):
        if tmp_path.is_dir():
            return (tmp_path, *(child for child in tmp_path.rglob("*") if child.is_dir()))
    return ()


@pytest.fixture(autouse=True)
def stop_secret_brokers_started_by_this_test(request, isolate_environment) -> Iterator[None]:
    """Kill any secret broker a test caused to start, and remove its runtime dir.

    **Why this is suite-wide rather than in ``tests/unit/secrets`` (QA Q7).** It
    began there, because retrieval lazily starts a daemon (design §13) and the
    CLI tests drive real ``lop secret`` subprocesses — a full run of that one
    directory left ~90 brokers alive, each holding a master key in memory and
    idling for 30 minutes. But the TUI now registers itself as a session at
    startup, so **TUI tests start brokers too**, and a fixture scoped to that
    directory could not see them: a full-tree run leaked live brokers parented
    to pytest workers plus their ``$TMPDIR/lop-secrets-<uid>-<digest>`` runtime
    directories. On the shared machine this repo is worked on, with many
    concurrent agent sessions, a leaked key-holding daemon is a resource an
    agent inflicts on the operator rather than a harmless artifact — so the
    sweep belongs where every test that CAN spawn one is covered, not where the
    tests that obviously do live.

    Scoped to this test's own config dirs, never a global sweep by process name:
    another xdist worker — or the operator's own live session — may legitimately
    be running a broker at that moment, and killing by name would take those
    down too.

    Teardown-only: each test gets fresh temporary directories, so there is
    nothing to clean up beforehand. The CANDIDATES are captured during the call
    phase (`pytest_runtest_call` below), because pytest reclaims `tmp_path`
    before this fixture is finalised — see `_temp_roots` for the measurement.
    """
    yield

    candidates = _secret_config_dirs(request, isolate_environment)
    if not candidates:
        return

    _stop_brokers_in(candidates)


def _stop_brokers_in(candidates: list[Path]) -> None:
    """SIGTERM the broker in each candidate config dir, then drop its runtime dir.

    A function rather than an in-fixture loop so the behaviour is reachable from
    a test: `test_broker_sweep.py` drives it against a REAL broker, and against a
    real broker it must NOT touch. The safety property is the caller's — the
    candidate list is derived from the test's own temp roots, never from a
    process-name sweep — and this function is what makes that property testable
    rather than merely asserted in a docstring.

    The platform guard lives HERE rather than in each caller: this runs for every
    test, from the call phase and again from teardown, and the broker is a POSIX
    daemon — `client.py` imports `fcntl` and `peer.py` authenticates over a unix
    socket — so an unguarded call would raise `ModuleNotFoundError` for every
    test in a Windows run. It did exactly that on the `filesystem-boundaries-windows`
    job when the call-phase reap was added and this guard was still only in the
    fixture, which is the argument for one guard at the bottom of the stack
    rather than one per entry point.
    """
    if not _broker_daemon_is_available():
        return

    # Imported here rather than at module scope: this conftest is loaded for
    # every test session, and the secrets client drags in fcntl/socket
    # machinery a run that touches no store never needs.
    from local_operator.secrets import client
    from local_operator.secrets.keys import secrets_dir
    from local_operator.secrets.protocol import _runtime_fallback_dir, socket_path

    for candidate in candidates:
        # Only ask candidates that actually have a socket. A broker is a
        # per-config-dir singleton, so this is the complete question, and it
        # skips the connect attempt for the many directories that never held
        # one.
        #
        # Every filesystem call here is defensive because THIS RUNS AS TEARDOWN
        # FOR EVERY TEST, including tests that deliberately break the calls it
        # makes: `test_unexpected_exception_becomes_error_result` monkeypatches
        # `Path.exists` to raise, and monkeypatch has not unwound yet when an
        # autouse fixture declared here tears down. A cleanup fixture that can
        # fail a passing test is worse than the leak it prevents.
        try:
            if not socket_path(candidate).exists():
                continue
            status = client.broker_status(candidate)
        except Exception:  # noqa: BLE001 - see above; cleanup never fails a test
            continue
        if status is None:
            continue
        pid = status.get("pid")
        # Three refusals, because the third one is how this sweep killed the run
        # that was running it. `test_broker.py`'s `broker` fixture serves the
        # broker IN-PROCESS (`threading.Thread(target=loop, daemon=True)`), so its
        # status reports THIS test process's pid, and `kill(os.getpid(),
        # SIGTERM)` takes down the xdist worker — "worker 'gw0' crashed while
        # running ...", 17 failed across 3.12 shards 2 and 3, and a `-n0` run
        # with it. The old teardown-only sweep never met it because the fixture's
        # own teardown has already closed that socket by then; the call-phase
        # reap runs while the socket is up, which is exactly why it works. An
        # in-process broker dies with the process that owns it, so skipping it
        # loses nothing.
        #
        # `pid <= 0` is not a process at all: `kill(0, ...)` signals this
        # process's whole GROUP and `kill(-1, ...)` every process this user may
        # signal, neither of which is ever a broker to reap.
        if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
            continue
        with suppress(OSError):
            os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and client.is_running(candidate):
            time.sleep(0.05)

    # The fallback runtime dir is derived from the SECRETS dir and is created
    # for any config dir whose socket path would exceed sun_path — pytest's
    # tmp_path is routinely ~145 bytes against a 103-byte limit, so here that is
    # the common case rather than the exotic one. It is created even for a
    # config dir whose store was never initialised, hence the unconditional
    # removal. AFTER the broker is stopped, since the socket lives inside it.
    for candidate in candidates:
        # `Exception`, not `OSError`, for the reason above: a test may have
        # replaced the path machinery this line depends on.
        with suppress(Exception):
            shutil.rmtree(_runtime_fallback_dir(secrets_dir(candidate)), ignore_errors=True)


def _secret_config_dirs(request: pytest.FixtureRequest, home: Path) -> list[Path]:
    """Every directory this test could have used as a config dir.

    See `_sweep_candidates` for the three sources; this wrapper exists because
    the fixture has a `request` and the call-phase hook has an `item`.
    """
    return _sweep_candidates(request.node, home)


def _sweep_candidates(node: pytest.Item, home: Path | None) -> list[Path]:
    """Every directory this test could have used as a config dir.

    Three sources, all of them the test's OWN temp roots:

    * ``home/.local-operator`` — the config dir a test that leaves HOME alone
      resolves to, since `isolate_environment` points HOME at a scratch dir.
    * the roots `pytest_runtest_call` recorded while they existed, which is the
      only way to see a config dir pytest has since reclaimed (see `_temp_roots`).
    * whatever is still under ``tmp_path`` right now — the only source that is
      non-empty for a test that failed during setup and so never reached the call
      phase, and the only one that is fresh when this runs IN the call phase.

    Derived from the test's temporary directories rather than read back from
    ``LOCAL_OPERATOR_CONFIG_DIR``: ``monkeypatch`` has already undone the test's
    ``setenv`` by the time an autouse fixture declared here tears down, so the
    environment no longer names the directory the test actually used.

    Every directory under ``tmp_path`` is a candidate, because enumerating the
    known shapes does not hold. The suite uses ``tmp_path`` itself
    (``test_logger_silence`` points the override straight at it),
    ``tmp_path/config`` (the secrets conftest and ``test_cli``), and arbitrary
    depths — ``test_a_deep_config_dir_still_gets_a_bindable_socket`` nests one
    160 bytes down deliberately, and a hardcoded list silently missed it,
    leaking one runtime dir per run.

    Cheap because it only runs for a test that HAS a ``tmp_path``, and a test's
    tmp_path holds a handful of entries; the ~16k tests that never touch a
    store pay one ``getattr``.
    """
    candidates = [home / ".local-operator"] if isinstance(home, Path) else []
    candidates.extend(node.stash.get(_SWEEP_ROOT_KEY, ()))
    return candidates + [root for root in _temp_roots(node) if root not in set(candidates)]
