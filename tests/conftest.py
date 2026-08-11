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

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

#: Environment variables that steer credential resolution, config discovery or
#: provider selection. Any of these leaking in from the developer's shell can
#: change what the code under test does.
_AMBIENT_VARS = (
    "LOCAL_OPERATOR_CONFIG_DIR",
    "LOCAL_OPERATOR_DEBUG",
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
    yield home


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
