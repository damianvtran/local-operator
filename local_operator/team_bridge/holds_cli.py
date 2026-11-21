"""``lop team-bridge holds [list]``: this machine's held agent messages, READ-ONLY.

WHY THIS EXISTS (operator report 2026-11-18). Twelve held messages sat on the
operator's machine for up to ~18 days, visible only in the desktop app's Agent
messages holds list. With no app attached nobody saw them, and a message that
should have been raised was not. A terminal needs to answer "what is held here,
from whom, why, and can I act on it" without the app, the daemon or the
receiver running.

ONE SOURCE OF TRUTH. The rows are exactly
:func:`local_operator.server.utils.desktop_team_bridge_holds.list_holds`, the
reader behind ``GET /v1/desktop/team-bridge/holds``: the same conversation-hold
selection, the same releasability rule, the same ``can_deliver`` /
``can_discard``. The sender and excerpt are the SAME join the desktop renderer
makes — the owner-audit message page for that conversation and seq
(``TeamBridgeClient.owner_audit_messages``). Nothing here re-derives a policy;
if the desktop list and this command disagree, one of them is not calling that
function.

WHAT IT NEEDS. The local bridge store (opened read-only: an absent store reads
as "not enrolled" and stays absent) and, for sender/excerpt only, this machine's
signed-in bridge credential to reach the hub. No desktop bearer, no daemon, and
the receiver may be stopped. When the hub cannot be reached the local rows still
print, with the sender column saying why it is missing.

IT CHANGES NO HOLD AND WRITES NOTHING TO THE BRIDGE STORE. No deliver, discard
or admit verb exists here (those stay the desktop's bearer-gated routes,
``desktop_team_bridge_holds.decide``), and there is no body cache: the excerpt is
printed and forgotten. The one write it can cause is OUTSIDE the bridge store:
the hub call resolves the signed-in credential through the shared auth store,
which may persist a refreshed token on a 401 exactly as any signed-in command
does. A machine with no ``auth.db`` is answered ``sign_in_required`` without
building the client, so the command never creates one (#4251 QA Q2).

HUMAN-ONLY JOIN, FAIL-CLOSED. The owner audit serves bodies the agent's
screening HELD or REJECTED — the exact text screening keeps out of a model's
context (``client.py``'s owner-audit section, ``test_owner_audit_isolation.py``).
So the join needs a POSITIVE sign of a person (:func:`_withheld_reason`): the
agent-descent marker absent (:func:`local_operator.agent_shell.in_agent_shell`,
set by the ``bash`` tool and the ``eval`` worker), no app console pane
(``LOCAL_OPERATOR_CONSOLE_SURFACE`` absent — the ``console`` tool's pty), AND
stdin a terminal. Anything
else — an agent tool, a cron job, ``env -i`` with no terminal — gets the
metadata (ids, codes, ages, flags — what the hold-alert already discloses) and
``sender: withheld``, never a held body or a sender-chosen label. The first
cut gated on the marker alone, which failed OPEN on the ``eval`` worker that
never carried it (#4251 security review S-1). Stdin, not stdout, so a person's
``lop team-bridge holds --json | jq`` still works.

THE LIMIT, STATED. Not a same-uid boundary: a command that deliberately scrubs
its environment (``env -u LOCAL_OPERATOR_AGENT_SHELL -u
LOCAL_OPERATOR_CONSOLE_SURFACE``) and allocates a pty (``script``) still gets the
join, exactly as same-user code could by importing the client. That is the
accepted limit; every ordinary agent tool path — ``bash``, ``eval``, ``console``
— is withheld. A person uses their own terminal app, not the app's console pane.

Terminal copy here is developer-facing CLI output naming codes and states —
plain English like ``team_bridge/cli.py``, pending the CLI copy-extraction
tranche.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

#: Excerpt ceiling, in characters, after redaction (operator spec 2026-11-18).
EXCERPT_MAX = 160

#: Messages read when a hold has no seq to aim at (see ``_join_audit``).
_NO_SEQ_PAGE = 50

#: Plain words for the codes a person is likely to meet. A code not listed falls
#: back to its kind's sentence; the code itself is always printed beside it.
_REASON_WORDS: dict[str, str] = {
    "model_hold": "the screener was unsure and held it for your review",
    "model_refused": "the screening model refused to judge it",
    "reply_truncated": "the screener's answer was cut off",
    "bounce_without_category": "the screener bounced it without saying why",
    "work_scope_hold": "it asks your agent for new work; waiting for your decision",
    "write_once_conflict": "a stored screening record conflicted; held so nothing is overwritten",
    "screening_model_unavailable": "no screening model could be reached",
    "provider_error": "the screening provider failed; it will retry",
    "budget_exhausted": "the screening budget is spent; it will retry after the refill",
    "breaker_open": "screening is paused after repeated provider failures",
    "open_chief_of_staff": "waiting for you to open the chief-of-staff session",
    "parse_failed": "the screener's answer could not be read",
    "empty_reply": "the screener answered blank",
    "unknown_hold": "held for a reason this build does not recognise",
}
_KIND_WORDS = {
    "owner_decision": "waiting for your decision",
    "screening": "held by screening",
    "waiting": "waiting on this machine; it should clear on its own",
}


def _config_dir() -> Path:
    """The config root this command reads. A named seam: tests inject one."""

    from local_operator.paths import config_dir

    return Path(config_dir())


def _client_for(root: Path) -> Any:
    """The bridge client for the owner-audit join. A named seam: tests inject a fake."""

    from local_operator.team_bridge.client import TeamBridgeClient

    return TeamBridgeClient(root)


def _stdin_is_terminal() -> bool:
    """A named seam: tests set the answer rather than allocate a pty."""

    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError, OSError):
        return False  # a closed or exotic stdin is no evidence of a person


def _withheld_reason() -> str | None:
    """Why the human-only join is refused here, or ``None`` when a person is present.

    Fail-closed by construction: the join runs only on the positive answer.
    """

    import os

    from local_operator.agent_shell import in_agent_shell
    from local_operator.terminals import CONSOLE_SURFACE_ENV

    if in_agent_shell():
        return "agent_shell"
    # The app's console pane: a real pty, so it passes the terminal test, and
    # its environment is the app's, so it lacks the bash marker. Every pane is
    # readable by the agent (`console read`), whoever opened it, so held text
    # printed there reaches a model. PRESENCE, not value: a creator's `env`
    # override can assign the key (even "") but cannot delete it; only an
    # `env -u` in the command can, the same deliberate scrub as for the marker
    # (#4251 security review S-3).
    if CONSOLE_SURFACE_ENV in os.environ:
        return "console"
    if not _stdin_is_terminal():
        return "no_terminal"
    return None


def reason_words(code: str, kind: str) -> str:
    words = _REASON_WORDS.get(code) or _KIND_WORDS.get(kind, "held")
    from local_operator.team_bridge.hold_release import RELEASABLE_HOLD_CODES

    # A releasable CODE whose decision did not qualify (e.g. a sender over its
    # budget) is not waiting on the person: say so rather than invite an action
    # the app will not offer.
    if code in RELEASABLE_HOLD_CODES and kind != "owner_decision":
        return f"{words.split(';')[0]}; not releasable from the app"
    return words


def _printable(text: str) -> str:
    """Drop control characters from sender-chosen text before it reaches a terminal.

    A label or body is written by a REMOTE sender; an ESC sequence in it would
    otherwise be interpreted by the operator's terminal (cursor moves, title
    writes, hidden text). Whitespace runs collapse to one space.
    """

    return " ".join("".join(c for c in text if c.isprintable() or c.isspace()).split())


def excerpt_of(body: str | None) -> str | None:
    """Redact first, then cut: a secret straddling the cut is still one shape."""

    if body is None:
        return None
    from local_operator.redaction_shapes import scrub_shapes

    text = _printable(scrub_shapes(str(body)))
    if len(text) <= EXCERPT_MAX:
        return text
    return text[: EXCERPT_MAX - 1].rstrip() + "…"


def _age_seconds(held_at: str | None, now: float) -> float | None:
    if not held_at:
        return None
    try:
        return max(0.0, now - datetime.fromisoformat(held_at).timestamp())
    except ValueError:
        return None


def _age_words(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    minutes = int(seconds // 60)
    if minutes < 60:
        return f"{minutes}m"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h"
    return f"{hours // 24}d"


async def _join_audit(
    root: Path, owner: Any, items: list[dict[str, Any]], *, want_body: bool
) -> tuple[str, str | None, dict[str, dict[str, Any]]]:
    """Sender (and body) per held message id, from the owner-audit page at its seq.

    Returns ``(audit, detail, by_message_id)``. ``audit`` is closed: ``ok``,
    ``partial`` (``detail`` = how many rows the audit did not carry),
    ``withheld`` (``agent_shell`` | ``no_terminal``), ``sign_in_required``,
    ``unreachable`` or ``refused`` (``detail`` = the refusal code). Only a
    per-conversation 404 — the hub's uniform answer for a conversation outside
    the audit window — costs just that row; every other refusal (auth, account,
    a refused destination) holds for every row and stops the lookup, so it is
    never reported as "not in the audit" (#4251 QA Q1).
    """

    withheld = _withheld_reason()
    if withheld is not None:
        return "withheld", withheld, {}
    if not (root / "auth.db").exists():
        # Building the client would CREATE the auth store; with none there is
        # no credential to find anyway.
        return "sign_in_required", None, {}
    from local_operator.team_bridge.client import (
        BridgeRefusal,
        BridgeSignInRequired,
        BridgeUnreachable,
    )

    client = _client_for(root)
    found: dict[str, dict[str, Any]] = {}
    missed = 0
    for item in items:
        seq = item.get("seq")
        conversation = item.get("conversation_id")
        if not conversation:
            continue
        try:
            # The captured-owner receipt is the COMMITTED owner's account, so a
            # machine signed in as someone else is refused by the client rather
            # than shown another account's audit (the desktop route's rule).
            page = await client.owner_audit_messages(
                str(conversation),
                expected_account_id=owner.account_id,
                tenant_id=owner.tenant_id,
                # The page ending at the hold's own seq; a hold row stored under
                # the bare id has no seq, so its conversation's newest page is
                # searched instead (bounded; only the matching id is kept).
                limit=1 if seq is not None else _NO_SEQ_PAGE,
                before_seq=int(seq) + 1 if seq is not None else None,
            )
        except BridgeSignInRequired:
            return "sign_in_required", None, found
        except BridgeUnreachable:
            return "unreachable", None, found
        except BridgeRefusal as refusal:
            if refusal.http_status == 404:
                missed += 1
                continue
            return "refused", str(refusal.code), found
        for message in (page or {}).get("messages") or []:
            if str(message.get("id")) != item["message_id"]:
                continue
            sender = message.get("sender") or {}
            found[item["message_id"]] = {
                "sender_kind": str(sender.get("kind") or ""),
                "sender_label": _printable(str(sender.get("display_label") or "")),
                "agent_label": (
                    _printable(str(message["agent_label"])) if message.get("agent_label") else None
                ),
                # Held in memory for this print only; never cached or written.
                "body": message.get("body") if want_body else None,
            }
    if missed:
        return "partial", f"{missed}_not_in_audit", found
    return "ok", None, found


def collect(
    root: Path, *, excerpt: bool = False, audit: bool = True, now: float | None = None
) -> dict[str, Any]:
    """The whole answer as data: what ``--json`` prints and the table renders."""

    from local_operator.server.utils import desktop_team_bridge_holds as holds
    from local_operator.team_bridge.store import OWNER_RELEASE_LOCAL_TERMINAL, Store

    now = time.time() if now is None else float(now)
    # ``active_owner_readonly``: the same exactly-one-committed-row rule the
    # read slice resolves, over a read-only connection (no create, no migrate).
    store = Store(root)
    owner = store.active_owner_readonly()
    if owner is None:
        # The same top-level keys as an enrolled answer, so a `--json` consumer
        # never branches on which keys exist (#4251 code review m3).
        return {
            "enrolled": False,
            "release_enabled": None,
            "key_protection": None,
            "audit": "no_owner",
            "audit_detail": None,
            "count": 0,
            "items": [],
        }
    payload = holds.list_holds(root, owner, now=now)
    # #4371 review m2: a hold whose owner release is SETTLED LOCALLY
    # (``discarded_local_unconfirmed``) is answered and nothing will move it, so
    # it is not "held" here: not listed, not in ``count``, and not a reason to
    # print "Deliver or discard them". ``list_holds`` keeps the row (the
    # desktop list shows the settled outcome), so the filter lives here, keyed
    # on the same release row that makes the row non-``actionable`` there.
    items = [
        item
        for item in payload["items"]
        if (item.get("decision") or {}).get("outcome") != OWNER_RELEASE_LOCAL_TERMINAL
    ]
    audit_state, audit_detail, joined = ("skipped", None, {})
    if audit and items:
        audit_state, audit_detail, joined = asyncio.run(
            _join_audit(root, owner, items, want_body=excerpt)
        )
    rows = []
    for item in items:
        join = joined.get(item["message_id"])
        # AGE IS FROM THE FIRST HOLD (the reader's ``first_held_at``), not
        # ``held_at``, which is the decision time; see ``list_holds``.
        age = _age_seconds(item.get("first_held_at") or item.get("held_at"), now)
        row = {
            "id": item["message_id"],
            "conversation_id": item["conversation_id"],
            "seq": item["seq"],
            "reason_code": item["hold_code"],
            "reason": reason_words(item["hold_code"], item["kind"]),
            "kind": item["kind"],
            "first_held_at": item.get("first_held_at"),
            "held_at": item["held_at"],
            "age_seconds": None if age is None else int(age),
            "sender_kind": join["sender_kind"] if join else None,
            "sender_label": join["sender_label"] if join else None,
            "agent_label": join["agent_label"] if join else None,
            "can_deliver": item["can_deliver"],
            "can_discard": item["can_discard"],
            "deliver_until": item["deliver_until"],
            "decision": item["decision"],
        }
        if excerpt:
            row["excerpt"] = excerpt_of(join["body"]) if join else None
        rows.append(row)
    return {
        "enrolled": True,
        "release_enabled": payload["enabled"],
        "key_protection": payload["key_protection"],
        "audit": audit_state,
        "audit_detail": audit_detail,
        "count": len(rows),
        "items": rows,
    }


def _sender_cell(row: dict[str, Any], audit_state: str) -> str:
    if row["sender_label"] is None:
        return "withheld" if audit_state == "withheld" else "unavailable"
    label = row["sender_label"] or "?"
    kind = row["sender_kind"] or "?"
    agent = f" via {row['agent_label']}" if row.get("agent_label") else ""
    return f"{label} ({kind}){agent}"


def render(answer: dict[str, Any]) -> str:
    if not answer["enrolled"]:
        return "No agent-messages owner is enrolled on this machine; nothing is held here."
    items = answer["items"]
    if not items:
        return "Nothing is held on this machine."
    lines = [
        f"{answer['count']} held message(s). Releasing from the app: "
        f"{'on' if answer['release_enabled'] else 'off'}. Sender lookup: {answer['audit']}"
        f"{' (' + answer['audit_detail'] + ')' if answer.get('audit_detail') else ''}."
    ]
    for row in items:
        flags = (
            f"deliver={'yes' if row['can_deliver'] else 'no'} "
            f"discard={'yes' if row['can_discard'] else 'no'}"
        )
        lines.append("")
        lines.append(
            f"{row['id']}  age {_age_words(row['age_seconds'])}  {flags}"
        )  # i18n: ignore developer CLI output (see the module docstring)
        lines.append(
            f"  from:          {_sender_cell(row, answer['audit'])}"
        )  # i18n: ignore developer CLI output (see the module docstring)
        seq = f" #{row['seq']}" if row["seq"] is not None else ""
        lines.append(
            f"  conversation:  {row['conversation_id'] or '?'}{seq}"
        )  # i18n: ignore developer CLI output (see the module docstring)
        lines.append(
            f"  reason:        {row['reason']} ({row['reason_code']})"
        )  # i18n: ignore developer CLI output (see the module docstring)
        if "excerpt" in row:
            lines.append(  # i18n: ignore developer CLI output (see the module docstring)
                f"  excerpt:       {row['excerpt'] if row['excerpt'] is not None else '-'}"
            )
    if any(row["can_deliver"] or row["can_discard"] for row in items):
        lines.append("")
        lines.append(  # i18n: ignore developer CLI output (see the module docstring)
            "Deliver or discard them in the desktop app's Agent messages holds list."
        )
    return "\n".join(lines)


def main(args: argparse.Namespace) -> int:
    """Print the holds list. Exit 0 when it was read, 1 when the store could not be."""

    from local_operator.team_bridge.store import StoreRefused

    try:
        answer = collect(_config_dir(), excerpt=bool(getattr(args, "excerpt", False)))
    except StoreRefused as refusal:
        print(  # i18n: ignore developer CLI output (see the module docstring)
            f"team-bridge holds: the local bridge store could not be read ({refusal.code})"
        )
        return 1
    if getattr(args, "json", False):
        print(json.dumps(answer, ensure_ascii=False, indent=2))
    else:
        print(render(answer))
    return 0
