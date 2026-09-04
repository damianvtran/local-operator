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
from collections.abc import Iterator
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
    # Tests launched from a detached operator inherit these runtime-only flags.
    # They turn strict --resume validation into adoption of a brand-new id.
    "LOP_RUNTIME_ADOPT_SESSION",
    "LOP_RUNTIME_DEFER_MATERIALISE",
    # A runtime child spawned by the mobile daemon carries its session id,
    # provider, model and cwd here. A test suite run from inside such a
    # session (agents do this) inherited LOP_MOBILE_CHILD_RESUME and created
    # THAT id inside its store (QA round 1 of #645).
    "LOP_MOBILE_CHILD_RESUME",
    "LOP_MOBILE_CHILD_PROVIDER",
    "LOP_MOBILE_CHILD_MODEL",
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
