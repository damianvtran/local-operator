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
from pathlib import Path

#: Environment variable that relocates everything below. Tests set it to a
#: tmp_path so a run can never touch a developer's real credentials, which is
#: also why honouring it consistently matters more than it looks: a code path
#: that ignores it is a code path tests cannot isolate.
CONFIG_DIR_ENV = "LOCAL_OPERATOR_CONFIG_DIR"

#: Directory name under the home directory when no override is set.
DEFAULT_CONFIG_DIRNAME = ".local-operator"


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
