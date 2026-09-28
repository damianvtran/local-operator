"""Aida — the built-in chief of staff, shipped with local-operator.

WHAT SHE IS. A single, long-lived conversation (R7) created on first need and
resolved through ``<config>/aida/state.json``; she runs as an ordinary session
whose attached role is the packaged seed ``agent_seeds/aida.md``, so every
runtime feature — tools, wakes, teams, projects, subagents — is standard. The
proactive cadence, the pause switch and the disable gate live beside this
module:

- :mod:`local_operator.aida.state`      — her files and their cross-process lock
- :mod:`local_operator.aida.bootstrap`  — :func:`ensure_session`, the ONE creator
- :mod:`local_operator.aida.proactive`  — the cadence engine (arm / pause / resume)
- :mod:`local_operator.aida.onboarding` — the greeting baseline, the first-run
  predicate and the R25 nudge ledger
- :mod:`local_operator.aida.activation` — who may auto-activate her at boot
  (terminal / desktop-managed daemon, never plain cloud/automation)
- :mod:`local_operator.aida.naming`     — her display name (``aida.name``) and
  the sync between it and her conversation's title
- :mod:`local_operator.aida.profile`    — the guarded write path for the profile
  notes she records into the operator's custom instructions (``lop aida note``)

DISABLING HER. ``aida.enabled = false`` in config, or
``LOCAL_OPERATOR_NO_AIDA=1`` in the environment, is a supported steady state:
the gate is the first statement of every entry point and no file is written
(R17/R18). ``README.md`` beside this file is the operator-facing summary.

PUBLIC SURFACE, kept deliberately small and import-light: the writers call
their submodule functions directly; boot paths and routes use :func:`enabled`
and :func:`ensure_session`.
"""

from __future__ import annotations

from pathlib import Path

from local_operator.aida import naming, onboarding, proactive, state
from local_operator.aida.bootstrap import ROLE_NAME, SESSION_TITLE, ensure_session
from local_operator.aida.state import ENV_DISABLE

__all__ = [
    "ENV_DISABLE",
    "ROLE_NAME",
    "SESSION_TITLE",
    "enabled",
    "ensure_session",
    "naming",
    "onboarding",
    "proactive",
    "state",
]


def enabled(config_dir: Path | str | None = None) -> bool:
    """Whether Aida is enabled for this install: config key AND env switch.

    Deliberately the same predicate :func:`ensure_session` gates on (it calls
    through to this), so "may a surface mention her" and "may she be created"
    cannot disagree. Either half disables.
    """
    from local_operator.aida.bootstrap import config_enabled

    return config_enabled(config_dir)
