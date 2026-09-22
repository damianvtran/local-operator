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

# `O_BINARY` deliberately does NOT live here -- see
# `local_operator.procstate.O_BINARY`. This module is on the runner core's
# forbidden-import list, so anything the runner needs must live below it.

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

    **The same home dot-directory on every platform, Windows included, and that
    is deliberate.** ``log_dir`` below asks ``%LOCALAPPDATA%`` for its root while
    this one does not, which reads as an omission; it is not. On Windows
    ``%APPDATA%`` is the roaming profile, so relocating credentials, the secret
    store and session transcripts there would SYNC them between machines — the
    opposite of what this directory is for. A single known root also means one
    path for an operator to back up, one for every platform's docs to name, and
    no migration for existing installs (audit D16). Do not "fix" this to match
    ``log_dir``: logs are disposable and roam safely; these are not and do not.
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

    A WRITER wants this function; a READER looking for a file some OTHER process
    wrote wants :func:`log_dirs`, because the override is per process and the two
    processes need not agree (see that function for the measured case).
    """
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override) / LOG_DIRNAME
    return platform_log_dir()


def platform_log_dir() -> Path:
    """The platform's conventional log directory, IGNORING :data:`CONFIG_DIR_ENV`.

    Split out of :func:`log_dir` so :func:`log_dirs` can name BOTH places a log may
    be without a second copy of the platform table below: this is the directory a
    process WITHOUT the override writes into, and it is therefore the second place
    :func:`log_dirs` has to look.
    """
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


def log_dirs() -> tuple[Path, ...]:
    """Every directory a log file for THIS machine may be in, this process's first.

    A SET RATHER THAN ONE PATH, because the override is PER PROCESS and a reader is
    routinely not the process that wrote the file it is looking for. Measured on
    this fleet (2026-09-22): the session runtimes carry
    ``LOCAL_OPERATOR_CONFIG_DIR`` (``ps eww`` on pid 92815 shows it), so
    ``arm`` writes ``runtime-stall-<pid>.log`` under ``$CONFIG_DIR/logs``; the
    ``lop serve`` daemon that supervises and classifies them does NOT carry it
    (``ps eww`` on pid 1276), so its :func:`log_dir` is the platform default. A
    reader that resolves ONE directory therefore misses evidence that exists, and
    a dead runtime whose dump fired is narrated as if no dump had been written at
    all. A pid is all the reader has, and it cannot know which process wrote the
    file — so it looks in both.

    ORDER IS THE PREFERENCE, not a sort: this process's own :func:`log_dir` first
    is where the writer put it whenever the two agree (and the only directory an
    isolated run can see, since every candidate is HOME- or override-derived), then
    the two directories a writer that does NOT share this process's environment can
    have used — the DEFAULT config directory's logs (``$HOME`` +
    :data:`DEFAULT_CONFIG_DIRNAME`) and the platform default. Deduplicated, so a
    process without the override gets two entries rather than one path twice.

    THE DEFAULT CONFIG DIR IS NOT REDUNDANT WITH THE PLATFORM ONE, and this is the
    measured case rather than a theory: the runtimes on this host are started with
    ``LOCAL_OPERATOR_CONFIG_DIR`` set to the DEFAULT config directory, so their
    dumps land in ``~/.local-operator/logs`` -- a directory no reader resolving the
    platform default would ever open. Both are listed for that reason.

    WHAT THIS DOES NOT COVER, stated rather than implied: a writer started with a
    CUSTOM override (``LOCAL_OPERATOR_CONFIG_DIR=/somewhere/else``) is
    undiscoverable from here, because the reader holding a pid knows nothing about
    the environment of the process that died. Every candidate is a directory this
    machine's own defaults produce; a dump written anywhere else is still invisible
    to a reader, and closing that for good means a pid-to-path index at arm time
    rather than a search.
    """
    primary = log_dir()
    candidates = [
        primary,
        Path.home() / DEFAULT_CONFIG_DIRNAME / LOG_DIRNAME,
        platform_log_dir(),
    ]
    ordered: list[Path] = []
    for candidate in candidates:
        if candidate not in ordered:
            ordered.append(candidate)
    return tuple(ordered)


#: The session runtimes' own log, beside the mobile daemon's ``mobile.log``.
RUNTIME_LOG_FILENAME = "runtime.log"


def runtime_log_path() -> Path:
    """The session runtimes' shared log file, deliberately NOT the daemon's.

    ``mobile.log`` is a launchd ``StandardOutPath``: the daemon appends through an
    fd it never reopens, while a ``RotatingFileHandler`` bounds a file by
    RENAMING it. One shared path therefore means a runtime's rotation moves the
    daemon's stream — and every other runtime's — into ``mobile.log.1``, out of
    what ``lop mobile logs`` reads. Measured on the operator's machine after a
    single rename: nine runtime children held the renamed inode while only the
    daemon held the fresh ``mobile.log``, so the command showed one writer and
    the flood lived in a backup. One file per writer class, one command to read
    both (see the ``logs`` subcommand in :mod:`local_operator.cli`).
    """
    return log_dir() / RUNTIME_LOG_FILENAME


def ensure_log_dir() -> Path | None:
    """Create and return :func:`log_dir`, or ``None`` if it cannot be created.

    Never raises. Logging is a diagnostic, not a startup requirement: a
    read-only home directory, a full disk or a file sitting where the log
    directory should be must degrade to "no log file", never to a CLI that
    refuses to start. The caller is expected to carry on with ``None``.

    Mode 0o700 because the log records prompts, model identifiers and error
    text from an interactive session — the same sensitivity class as the
    encrypted credential store beside it, and the default 0o755 would expose it
    to every other account on a shared machine.
    """
    directory = log_dir()
    try:
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    except OSError:
        return None
    return directory
