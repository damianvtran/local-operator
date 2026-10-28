"""Readiness check and placement for agent messages — NEVER a consent path.

WHY THIS FILE NO LONGER ENROLS (operator decision, 2026-10-28; #3116 round 4).
An earlier revision accepted consent from a terminal behind a "human gate" (a
terminal check plus the agent-shell marker). An independent review showed the
gate was forgeable — the marker is an environment variable and a TTY can be
simulated — so the command could have accepted the served consent on the user's
behalf. That is the one thing it must never do: #3022 made consent the DESKTOP
BEARER's act, and agent child processes have that bearer stripped precisely so a
prompt injection cannot promote itself into an enrolled machine. A terminal path
that can grant consent defeats that boundary no matter how the terminal is
checked, so the path is GONE rather than hardened.

What is left, and the whole list: read what this machine already has, and place
the receiver when the user's own decision (made in the app) is present. The
verbs that install, start, stop, report and tail the service live in
:mod:`local_operator.team_bridge.connector.cli`; this module is the one-shot
"is this machine ready, and if so place the receiver" answer a person (or an
agent setting the machine up) runs first.

THE LIMIT THIS CREATES, stated because a user will hit it: a headless-only or
server-only machine cannot enrol itself. Consent is granted on a machine where
the app is signed in as the owner; that commit lives in the config root, so a
server needs the app once against the same root, or a config root copied from
the machine that did it. There is no terminal substitute, by design.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

#: Returned when there is no enrolled owner: consent has not been granted in the
#: app, and this command will not grant it.
CODE_CONSENT_UI_REQUIRED = "consent_ui_required"

#: Enrolment exists but there is no chief-of-staff session committed for it.
CODE_DESIGNATION_MISSING = "designation_missing"

#: Enrolled, but the user has not enabled agent messages (the app writes the key).
CODE_ENABLE_IN_APP_REQUIRED = "enable_in_app_required"

#: The receiver was placed (or was already in place).
CODE_READY = "ready"

#: Enrolled and enabled, but the service could not be placed; see ``service``.
CODE_NOT_READY = "not_ready"


def main(args: argparse.Namespace) -> int:
    """Dispatch only this command; plain English by default, JSON on request."""

    from local_operator.paths import config_dir

    from local_operator.team_bridge.store import StoreRefused

    root = config_dir()
    try:
        result = setup(root)
    except (OSError, ValueError, sqlite3.Error, StoreRefused):
        # StoreRefused is a RuntimeError raised for an unreadable or corrupt
        # state database (review round 5, MAJOR-2). It is a refusal to answer,
        # not a crash: the command reports it like every other refusal rather
        # than dying with a traceback.
        # Exception prose can carry filesystem or transport content. The stable
        # machine code is enough.
        result = {"ok": False, "code": "setup_failed"}
    if getattr(args, "json", False):
        print(json.dumps(result, ensure_ascii=False))
    else:
        for line in render_lines(result):
            print(line)
    return 0 if result["ok"] else 1


def setup(root: Path) -> dict[str, Any]:
    """What this machine has, and — if allowed — place the receiver.

    Deliberately synchronous and offline: no credential, no HTTP, no consent, no
    designation, no settings write. The only side effect is
    :func:`sync_connector_service`, which is the same locked, guarded entry point
    every surface uses, and which does nothing at all unless the store already
    holds a committed owner and the user's enabled setting.
    """

    from local_operator.team_bridge.store import Store

    store = Store(root)
    owner = store.active_owner_readonly()
    if owner is None:
        return {"ok": False, "code": CODE_CONSENT_UI_REQUIRED}
    session = store.designated_session_for_readonly(owner)
    if not session:
        return {
            "ok": False,
            "code": CODE_DESIGNATION_MISSING,
            "account_id": owner.account_id,
            "tenant_id": owner.tenant_id,
        }
    if not _enabled(root):
        return {
            "ok": False,
            "code": CODE_ENABLE_IN_APP_REQUIRED,
            "account_id": owner.account_id,
            "tenant_id": owner.tenant_id,
            "session": session,
        }

    from local_operator.team_bridge.connector.install import sync_connector_service

    outcome = sync_connector_service(root)
    # ``not_addressable`` is NOT success (review round 5, MAJOR-1): it means the
    # unit file was written but the real service manager was never addressed — a
    # sandboxed or redirected home. Nothing is registered or running, and the
    # guide tells the user to re-run from their own shell, so claiming "the
    # receiver is in place" here would be false.
    placed = outcome.code in {"installed", "current", "registered"}
    return {
        "ok": placed,
        "code": CODE_READY if placed else CODE_NOT_READY,
        "account_id": owner.account_id,
        "tenant_id": owner.tenant_id,
        "session": session,
        "service": outcome.code,
        "detail": outcome.detail,
    }


def _enabled(root: Path) -> bool:
    """Whether the user has agent messages ON for this root, as the fences read it.

    ``team_bridge.enabled`` is written by the app when the user enables agent
    messages (the settings route). It is read, never written, here, and it is
    read through :class:`TeamBridgeSettings` so a hand-edited string value means
    what the product's own fences mean by it. Only an explicit ON counts: this
    command's answer is "the user has enabled it", and an absent key means they
    have not been asked yet — the installer treats absent as "may run" for a
    different question (does the receiver have work), which is not this one.
    """

    from local_operator.config import read_config_values
    from local_operator.team_bridge.settings import TeamBridgeSettings

    try:
        values = read_config_values(root)
    except Exception:  # noqa: BLE001 — an unreadable config is "not enabled"
        return False
    if values is None:
        return False
    return TeamBridgeSettings.from_values(values).enabled is True


def render_lines(result: dict[str, Any]) -> list[str]:
    """Plain-English lines for one result. Copy is keyed; nothing is invented here."""

    from local_operator.i18n.cli_help import help_locale as resolve_help_locale
    from local_operator.i18n.keys import cli_team_bridge as _k

    locale = resolve_help_locale()

    def text(message: Any, **kwargs: Any) -> str:
        return str(message.envelope(locale=locale, **kwargs)["text"])

    code = result.get("code")
    if code == CODE_CONSENT_UI_REQUIRED:
        return [text(_k.setup_consent_ui_required()), text(_k.setup_consent_ui_next())]
    if code == CODE_ENABLE_IN_APP_REQUIRED:
        return [
            text(
                _k.setup_enable_in_app_required(),
                account_id=result.get("account_id", ""),
            )
        ]
    if code == CODE_DESIGNATION_MISSING:
        return [text(_k.setup_designation_missing(), account_id=result.get("account_id", ""))]
    if code == CODE_READY:
        return [
            text(
                _k.setup_ready(),
                account_id=result.get("account_id", ""),
                session=result.get("session", ""),
            )
        ]
    if code == CODE_NOT_READY:
        return [
            text(
                _k.setup_not_ready(),
                service=result.get("service", ""),
                detail=result.get("detail", ""),
            )
        ]
    return [text(_k.setup_failed())]
