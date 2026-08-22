"""Where the app keeps its own files.

One function, because the rule it encodes was previously written out three
times — in ``providers/auth_store.py``, ``providers/oauth/kimi.py`` and
``model/configure.py`` — while ``cli.py`` hardcoded ``~/.local-operator`` at a
dozen call sites and honoured no override at all. The two halves disagreed in
a way that is invisible until someone sets the variable: ``credential update``
wrote a key to the home directory while the catalogue looked for it under the
override, so the key was simply not found.

Deliberately its own module with only stdlib imports. Both the CLI's startup
path and the provider stores need it, and anything heavier here would put the
provider graph on the CLI's import path for the sake of one path join.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: Environment variable that relocates everything below. Tests set it to a
#: tmp_path so a run can never touch a developer's real credentials, which is
#: also why honouring it consistently matters more than it looks: a code path
#: that ignores it is a code path tests cannot isolate.
CONFIG_DIR_ENV = "LOCAL_OPERATOR_CONFIG_DIR"

#: Directory name under the home directory when no override is set.
DEFAULT_CONFIG_DIRNAME = ".local-operator"

#: Application name used under the platform-standard log root. Spelled with a
#: hyphen and no leading dot: ``~/Library/Logs/.local-operator`` would be a
#: hidden directory inside a folder the user is meant to browse in Console.app.
APP_DIRNAME = "local-operator"

#: Subdirectory holding log files. Only used when the logs live inside the
#: config directory (the override case) or under the XDG state root, both of
#: which hold other things too.
LOG_DIRNAME = "logs"

#: Environment variable that relocates the agent's working-directory home. Its
#: own variable rather than sharing :data:`CONFIG_DIR_ENV`: the config dir holds
#: credentials and transcripts (private, small), while the agent home is where
#: the model reads and writes files during a task (a workspace, potentially
#: large) \u2014 a user isolating one does not necessarily want the other moved.
AGENT_HOME_ENV = "LOCAL_OPERATOR_HOME"

#: Directory name under the home directory when no override is set. Spelled
#: WITHOUT a leading dot on purpose: it is a workspace the user is meant to
#: browse, not hidden state, and every existing install already has
#: ``~/local-operator-home`` \u2014 this change is about WHEN it is created and WHERE
#: the path is resolved, never the default location.
AGENT_HOME_DIRNAME = "local-operator-home"


def config_dir() -> Path:
    """The app's configuration directory: the override, else ``~/.local-operator``.

    Read from the environment on every call rather than resolved once at import.
    Tests monkeypatch the variable after the module is imported, and a module
    constant would freeze whatever the first importer saw — including, for a test
    session, the developer's real home directory.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override)
    return Path.home() / DEFAULT_CONFIG_DIRNAME


def agent_home_dir() -> Path:
    """The agent's working-directory home: the override, else ``~/local-operator-home``.

    Read from the environment on every call, for the same reason
    :func:`config_dir` is (tests monkeypatch the variable after import; a module
    constant would freeze the developer's real home into the test session).

    This does NOT honour :data:`CONFIG_DIR_ENV`. Before this existed, ``main()``
    hardcoded ``~/local-operator-home`` and created it unconditionally on every
    invocation \u2014 ``config list`` on a fresh machine created an agent workspace
    it never used \u2014 and it ignored any override entirely, so a test or isolated
    run that relocated the config dir still wrote a workspace into the real home
    directory. Callers create it lazily at the point of use (session/agent start,
    the server app) rather than at import or dispatch.
    """
    override = os.environ.get(AGENT_HOME_ENV)
    if override:
        return Path(override)
    return Path.home() / AGENT_HOME_DIRNAME


def default_agent_cwd() -> str:
    """The default working-directory string stored in an agent record.

    Returns the portable ``~/local-operator-home`` when no override is set \u2014 an
    agent record is exported and shared, so a literal home path in it would not
    replicate on another machine \u2014 and the absolute override path when one IS
    set, so the stored cwd and the directory :func:`agent_home_dir` actually
    creates cannot diverge. That divergence is precisely the class of bug
    :func:`config_dir`'s module docstring documents: one half honouring the
    override while the other hardcodes home, invisible until the variable is set.
    """
    override = os.environ.get(AGENT_HOME_ENV)
    if override:
        return str(Path(override))
    return f"~/{AGENT_HOME_DIRNAME}"


def ensure_agent_home_dir() -> Path:
    """Create and return :func:`agent_home_dir`, creating parents as needed.

    Unlike :func:`ensure_log_dir` this DOES surface a creation failure: the
    agent home is the working directory a task runs in, so a run that cannot
    create it has nowhere to operate and should fail loudly rather than silently
    fall back to the process cwd.
    """
    directory = agent_home_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def log_dir() -> Path:
    """Where the app writes its rotating log files.

    Honours :data:`CONFIG_DIR_ENV` first and unconditionally. Logs are not
    configuration, but the override exists so that a test or an isolated run
    can be certain the process touches nothing outside one directory — a log
    file escaping to the developer's real ``~/Library/Logs`` would break that
    promise and leave litter behind that nothing cleans up.

    Otherwise the platform's conventional location, because a log the user
    cannot find with the tools their OS gives them (Console.app, ``journalctl``
    habits, ``~/.local/state``) is barely better than no log at all:

    - macOS: ``~/Library/Logs/local-operator``
    - Windows: ``%LOCALAPPDATA%\\local-operator\\Logs``
    - Linux/BSD: ``$XDG_STATE_HOME/local-operator/logs``, defaulting to
      ``~/.local/state/local-operator/logs`` per the XDG base directory spec,
      which places "state that should persist but is not config or cache" —
      exactly a log — under the state root rather than under data or cache.
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override) / LOG_DIRNAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Logs" / APP_DIRNAME

    if os.name == "nt":
        # LOCALAPPDATA rather than APPDATA: logs are machine-local and must not
        # be synced to a roaming profile, where they would count against the
        # user's roaming quota and be copied between machines on every login.
        local_app_data = os.environ.get("LOCALAPPDATA")
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / APP_DIRNAME / "Logs"

    xdg_state = os.environ.get("XDG_STATE_HOME")
    base = Path(xdg_state) if xdg_state else Path.home() / ".local" / "state"
    return base / APP_DIRNAME / LOG_DIRNAME


def ensure_log_dir() -> Path | None:
    """Create and return :func:`log_dir`, or ``None`` if it cannot be created.

    Never raises. Logging is a diagnostic, not a startup requirement: a
    read-only home directory, a full disk or a file sitting where the log
    directory should be must degrade to "no log file", never to a CLI that
    refuses to start. The caller is expected to carry on with ``None``.

    Mode 0o700 because the log records prompts, model identifiers and error
    text from an interactive session — the same sensitivity class as
    ``credentials.env`` next door, and the default 0o755 would expose it to
    every other account on a shared machine.
    """
    directory = log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        return None
    return directory
