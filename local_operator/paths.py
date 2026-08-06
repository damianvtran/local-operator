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
