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

import pytest

#: Environment variables that steer credential resolution, config discovery or
#: provider selection. Any of these leaking in from the developer's shell can
#: change what the code under test does.
_AMBIENT_VARS = (
    "LOCAL_OPERATOR_CONFIG_DIR",
    "LOCAL_OPERATOR_DEBUG",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "DEEPSEEK_API_KEY",
    "MISTRAL_API_KEY",
    "GOOGLE_API_KEY",
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
