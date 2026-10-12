"""Process-local record of the agent-messages feature's last observed surface state.

WHY THIS EXISTS. RFC §5.6 makes the settings section's visibility DYNAMIC —
absent in state A (no eligible organization), absent in G (dark launch) — and
§5.4 says the static settings registry is not the gate. The two surfaces that
DRAIN the registry — the TUI's ``/settings`` page and the desktop's
``GET /v1/settings`` — therefore need an answer to "may this section render
here, right now" that is cheap, synchronous and network-free. This module is
that answer's home: :class:`~local_operator.team_bridge.client.StatusDiscipline`
(the ONE status reader, §5.1) records every answer it produces here, and the
surfaces ask :func:`section_visible`, which never fetches and never imports the
client stack.

FAIL CLOSED, BY DEFAULT. A process that has never observed the feature shows
NOTHING: the default is invisible. The asymmetry is deliberate — a wrong
absence is a missing on-ramp that the next boundary read fixes, while a wrong
presence is the up-sell §1 forbids and no read can undo on screen.

SCOPE. The record is per (process, config root). The desktop daemon records
through its ``/v1/desktop/team-bridge/status`` route and its settings payload
reads the same process's record; a TUI process shows the section only once a
boundary read has run IN THAT PROCESS (its app-start/notice wiring is a later
lane — until then the TUI is conservative, never wrong in the visible
direction).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: The one section whose visibility is dynamic (see the module docstring).
BRIDGE_SECTION = "team_bridge"

#: ``config root -> {"visible": bool, "state": str, "checked_at": float}``.
_STATE: dict[str, dict[str, Any]] = {}


def _key(config_dir: Path | str | None) -> str:
    return str(config_dir) if config_dir is not None else ""


def record(
    config_dir: Path | str | None,
    *,
    visible: bool,
    state: str,
    checked_at: float,
    read_only: bool = False,
    note: str | None = None,
) -> None:
    """Remember the last status answer this process observed for ``config_dir``.

    ``read_only``/``note`` are the §5.6 F posture: the flow cannot resume, and
    ``note`` is the contract's own sentence that the drainers render instead of
    pretending an edit resumes anything — v1 emits the neutral "Agent messages
    are turned off", with "Plan ended — reading only" reserved for a future
    positively-evidenced lapse (review round 2, R9). They travel with the SAME
    record the status boundary wrote — one answer, one posture — so a surface
    never re-derives the sentence nor fetches around this module.
    """

    _STATE[_key(config_dir)] = {
        "visible": bool(visible),
        "state": str(state),
        "checked_at": float(checked_at),
        "read_only": bool(read_only),
        "note": note if isinstance(note, str) else None,
    }


def observed(config_dir: Path | str | None = None) -> dict[str, Any] | None:
    """The last record for ``config_dir``, or ``None`` when nothing was seen."""

    return _STATE.get(_key(config_dir))


def section_visible(name: str, config_dir: Path | str | None = None) -> bool:
    """Whether the settings section ``name`` may render for this config root.

    Static sections are always visible; :data:`BRIDGE_SECTION` follows the last
    observed state — visible only when the feature was last seen READY and
    present (an eligible organization, or an existing endpoint whose plan or
    organization state keeps it readable: §5.6 states B through F). States A
    (observed ineligible; no endpoint in any organization) and G (dark launch,
    observed as ``messaging_unavailable``) and "never observed at all" all read
    as invisible, which is the direction the contract asks for.
    """

    if name != BRIDGE_SECTION:
        return True
    record_ = _STATE.get(_key(config_dir))
    return bool(record_ and record_.get("visible"))


def clear() -> None:
    """Forget every record (tests; a deleted config root)."""

    _STATE.clear()
