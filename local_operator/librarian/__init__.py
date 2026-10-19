"""The Librarian — the Library's always-on organizing agent.

WHAT SHE IS. A single, long-lived conversation created on first need and
resolved through ``<config>/librarian/state.json``; she runs as an ordinary
session whose attached role is the packaged seed
``agent_seeds/librarian.md``, so every runtime feature — tools, wakes,
sessions, the ``library`` tool — is standard. She is Aida's twin, not a new
scheduler: the same lifecycle, one daily pass instead of a check-in. The
modules beside this one:

- :mod:`local_operator.librarian.state`     — her files and their cross-process lock
- :mod:`local_operator.librarian.bootstrap` — :func:`ensure_session`, the ONE creator
- :mod:`local_operator.librarian.proactive` — the cadence engine (arm / pause / resume)
- :mod:`local_operator.librarian.profile`   — the guarded writer for learned
  taxonomy conventions (the Library store, never ``system_prompt.md``)

DISABLING HER. ``librarian.enabled = false`` in config, or
``LOCAL_OPERATOR_NO_LIBRARIAN=1`` in the environment, is a supported steady
state: the gate is the first statement of every entry point and no file is
written.

PUBLIC SURFACE, kept deliberately small and import-light: the writers call
their submodule functions directly; boot paths and routes use :func:`enabled`
and :func:`ensure_session`.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.librarian import bootstrap, proactive, state
from local_operator.librarian.bootstrap import ROLE_NAME, SESSION_TITLE, ensure_session
from local_operator.librarian.state import ENV_DISABLE

__all__ = [
    "ENV_DISABLE",
    "ROLE_NAME",
    "SESSION_TITLE",
    "bootstrap",
    "enabled",
    "ensure_session",
    "proactive",
    "state",
]


def enabled(config_dir: Path | str | None = None) -> bool:
    """Whether the Librarian is enabled for this install: config key AND env switch.

    Deliberately the same predicate :func:`ensure_session` gates on (it calls
    through to this), so "may a surface mention her" and "may she be created"
    cannot disagree. Either half disables.
    """
    return bootstrap.config_enabled(config_dir)
