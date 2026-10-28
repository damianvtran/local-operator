"""Explicit backend-only consent and CoS selection, without a desktop bearer.

The first invocation only exposes accepted, freshly served consent. A write
requires the exact version plus an account, credential row, tenant and session
chosen by the operator. JSON is the machine-facing contract; argparse copy is
owned by team_bridge_arguments. Never turn a preview into default-on enrollment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

from local_operator.team_bridge.client import (
    Boundary,
    BridgeRefusal,
    BridgeSignInRequired,
    StatusDiscipline,
    TeamBridgeClient,
    consent_for_display,
    select_tenant,
)


def confirm_consent(
    version: str, text: str, *, account_id: str, tenant_id: str, session: str
) -> str | None:
    """THE human gate on accepting the served consent, or ``None`` to proceed.

    WHY A COMMAND THAT CAN CONSENT NEEDS ONE (code review MAJOR-3). The desktop
    equivalent is bearer-gated, and agent child processes have that bearer
    stripped precisely so an agent cannot self-consent. This command is the
    backend-only replacement for that consent dialog, so it must not be a way
    around the gate: otherwise a prompt injection — or an agent following a
    convenient instruction — reads the version out of the read-only preview and
    accepts on the user's behalf in a second command.

    So acceptance is a HUMAN ACT at a terminal: the full served text is printed,
    the account, organization and chief of staff are named, and an explicit ``y``
    is required. STRENGTH, stated rather than implied: this is a SPEED BUMP, not
    a security boundary — the agent-shell marker is an environment variable and a
    TTY can be simulated (measured by review round 2 through ``script``), so an
    agent that deliberately works around both can still accept. What it reliably
    stops is the ordinary path (an agent following a convenient instruction, a
    piped/headless caller) and it makes the act visible in the transcript. A
    real capability gate needs an authority this process does not hold (the
    desktop's bearer, or a code the user reads off their own screen) and is a
    separate decision. Refused, never guessed, when:

    * the caller descends from an agent's ``bash`` tool call
      (``agent_shell.in_agent_shell``) — the agent may preview, print the exact
      command for the user, and nothing more;
    * stdin or stdout is not a terminal — there is nobody to ask, or the answer
      would be read from something that already captured this text.

    Returns a stable machine code (``human_required`` /
    ``consent_not_accepted``) or ``None``. Never raises: a terminal that goes
    away is not consent.
    """

    import sys

    from local_operator.agent_shell import in_agent_shell

    if in_agent_shell():
        return "human_required"
    stdin_tty = bool(sys.stdin and sys.stdin.isatty())
    stdout_tty = bool(sys.stdout and sys.stdout.isatty())
    if not (stdin_tty and stdout_tty):
        return "human_required"
    print(text)
    print()
    print(f"Consent version: {version}")
    print(f"Account: {account_id}")
    print(f"Organization: {tenant_id}")
    print(f"Chief of staff session: {session}")
    try:
        answer = input("Accept this consent for this machine, as the account owner? [y/N] ")
    except (EOFError, KeyboardInterrupt, OSError):
        return "consent_not_accepted"
    return None if answer.strip().lower() in ("y", "yes") else "consent_not_accepted"


def main(args: argparse.Namespace) -> int:
    """Dispatch only this command; all results are JSON, including refusals."""

    from local_operator.paths import config_dir

    root = config_dir()
    try:
        result = asyncio.run(
            setup(
                root,
                credential_id=args.credential_id,
                account_id=args.account_id,
                tenant_id=args.tenant_id,
                accept_consent_version=getattr(args, "accept_consent_version", None),
                session=getattr(args, "session", None),
            )
        )
    except BridgeSignInRequired:
        result = {"ok": False, "code": "sign_in_required"}
    except BridgeRefusal as exc:
        result = {"ok": False, "code": exc.code, "refresh_status": exc.refreshes_status}
    except (OSError, ValueError, sqlite3.Error):
        # Exception prose can carry filesystem or transport content. The stable
        # machine code is enough; never print credentials or unkeyed English.
        result = {"ok": False, "code": "setup_failed"}
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result["ok"] else 1


async def setup(
    root: Path,
    *,
    credential_id: int,
    account_id: str | None = None,
    tenant_id: str | None = None,
    accept_consent_version: str | None = None,
    session: str | None = None,
) -> dict[str, Any]:
    """Run the same client authority as desktop, with an explicit OAuth pin.

    No HTTP setters, preferences or native enrollment writes are reached until
    the served text, captured account and selected tenant all match. Refused
    writes leave local preferences alone; a failed local commit is not success.
    """

    if credential_id <= 0:
        return {"ok": False, "code": "invalid_request"}
    if (accept_consent_version is None) != (session is None):
        return {"ok": False, "code": "acceptance_and_session_required"}
    client = TeamBridgeClient(root, credential_id=credential_id)
    discipline = StatusDiscipline(root, credential_id=credential_id, read_only=True)
    access = await client.current_access()
    if account_id is not None and access.account_id != account_id:
        return {"ok": False, "code": "account_mismatch"}
    account_id = access.account_id
    check = await discipline.check(Boundary.SURFACE_OPEN)
    if check.account_id != account_id:
        return {"ok": False, "code": "account_mismatch"}
    selected = select_tenant(check.payload)
    # v1 has one selected tenant, in server order. Do not silently enroll a
    # different membership from the one the production consent surface shows.
    selected_tenant = selected.get("tenant_id") if selected is not None else None
    if selected is None or not isinstance(selected_tenant, str):
        return {"ok": False, "code": "tenant_unavailable"}
    if tenant_id is not None and selected_tenant != tenant_id:
        return {"ok": False, "code": "tenant_mismatch"}
    tenant_id = selected_tenant
    consent = consent_for_display(check)
    if consent is None or consent[2] != "served":
        return {"ok": False, "code": "served_consent_required"}
    version, text, source = consent
    preview: dict[str, Any] = {
        "ok": True,
        "code": "consent_preview",
        "credential_id": credential_id,
        "account_id": account_id,
        "tenant_id": tenant_id,
        "consent": {"version": version, "text": text, "source": source},
    }
    if accept_consent_version is None:
        return preview
    if accept_consent_version != version:
        return {**preview, "ok": False, "code": "consent_version_mismatch"}
    if not account_id or not tenant_id or not session:
        return {"ok": False, "code": "invalid_request"}
    from local_operator.session.catalog import session_directory_name
    from local_operator.session.compression import transcript_source

    if not session or not session_directory_name(session):
        return {"ok": False, "code": "invalid_session"}
    if not transcript_source(root / "sessions" / session)[0].is_file():
        return {"ok": False, "code": "session_not_found"}
    from local_operator.team_bridge.store import Store

    owner = Store(root).active_owner_readonly()
    if owner is not None and (
        owner.account_id != account_id
        or owner.tenant_id != tenant_id
        or owner.credential_id != credential_id
    ):
        return {"ok": False, "code": "local_owner_conflict"}
    refusal = confirm_consent(
        version,
        text,
        account_id=account_id,
        tenant_id=tenant_id,
        session=session,
    )
    if refusal is not None:
        return {**preview, "ok": False, "code": refusal}
    answer = await client.put_settings(
        state="active",
        tenant_id=tenant_id,
        consent_version=version,
        consented_via="settings",
        expected_account_id=account_id,
        endpoint=selected.get("endpoint"),
    )
    enrollment = answer.get("local_enrollment")
    if not isinstance(enrollment, dict) or enrollment.get("committed") is not True:
        return {"ok": False, "code": "local_enrollment_failed", "local_enrollment": enrollment}

    from local_operator.config import ConfigManager
    from local_operator.settings_io import BY_KEY, write_setting
    from local_operator.team_bridge.connector.install import sync_connector_service

    manager = ConfigManager(config_dir=root)
    write_setting(manager, BY_KEY["team_bridge.session"], session)
    write_setting(manager, BY_KEY["team_bridge.enabled"], True)
    discipline.reconcile_designation(check)
    # The generation-fenced reconciliation is best-effort for normal status
    # reads. A setup command must instead verify the native result before it
    # promises a receiver destination, including on an already-owned endpoint.
    store = Store(root)
    owner = store.active_owner_readonly()
    if (
        owner is None
        or owner.account_id != account_id
        or owner.tenant_id != tenant_id
        or owner.credential_id != credential_id
        or store.designated_session_for_readonly(owner) != session
    ):
        return {"ok": False, "code": "designation_not_committed"}
    await asyncio.to_thread(sync_connector_service, root)
    return {
        "ok": True,
        "code": "setup_complete",
        "account_id": account_id,
        "tenant_id": tenant_id,
        "session": session,
        "consent_version": version,
        "local_enrollment": enrollment,
    }
