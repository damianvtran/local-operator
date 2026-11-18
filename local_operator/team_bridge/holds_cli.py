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

IT WRITES NOTHING. No deliver, discard or admit verb exists here (those stay the
desktop's bearer-gated routes, ``desktop_team_bridge_holds.decide``), no store
write, and no body cache: the excerpt is printed and forgotten.

HUMAN-ONLY JOIN. The owner audit serves bodies the agent's screening HELD or
REJECTED — the exact text screening keeps out of a model's context
(``client.py``'s owner-audit section, ``test_owner_audit_isolation.py``). So the
join is refused when this process descends from an agent's ``bash`` tool call
(:func:`local_operator.agent_shell.in_agent_shell`): an agent running this gets
the metadata (ids, codes, ages, flags — what the hold-alert already discloses)
and ``sender: withheld``, never a held body or a sender-chosen label. THE LIMIT,
STATED: that marker removes the ordinary path, it is not a same-uid boundary —
a process that scrubs its environment can still reach the hub with the local
credential, exactly as it could by importing the client. The excerpt is further
opt-in (``--excerpt``), capped at :data:`EXCERPT_MAX` characters, and scrubbed
by the harness's one credential-shape table BEFORE it is cut, so a token split
by the cut cannot escape the shape match.

Terminal copy here is developer-facing CLI output naming codes and states —
plain English like ``team_bridge/cli.py``, pending the CLI copy-extraction
tranche.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


def _audit_withheld() -> bool:
    from local_operator.agent_shell import in_agent_shell

    return in_agent_shell()


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
) -> tuple[str, dict[str, dict[str, Any]]]:
    """Sender (and body) per held message id, from the owner-audit page at its seq.

    Returns ``(audit_state, by_message_id)``. ``audit_state`` is ``ok`` or a
    closed reason the whole join is missing (``withheld_agent_shell``,
    ``sign_in_required``, ``unreachable``, ``refused:<code>``). A single message
    the page does not carry (outside the audit window, or no seq) is simply
    absent from the map.
    """

    if _audit_withheld():
        return "withheld_agent_shell", {}
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
            return "sign_in_required", found
        except BridgeUnreachable:
            return "unreachable", found
        except BridgeRefusal as refusal:
            # An auth/account refusal holds for every row: stop asking. Anything
            # else (a conversation outside the audit window is the hub's uniform
            # 404) costs only that row its sender.
            if refusal.http_status in (401, 403) or refusal.code.startswith("account"):
                return f"refused:{refusal.code}", found
            missed += 1
            continue
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
    return ("ok" if not missed else f"partial:{missed}_not_in_audit"), found


def collect(
    root: Path, *, excerpt: bool = False, audit: bool = True, now: float | None = None
) -> dict[str, Any]:
    """The whole answer as data: what ``--json`` prints and the table renders."""

    from local_operator.server.utils import desktop_team_bridge_holds as holds
    from local_operator.team_bridge.store import Store

    now = time.time() if now is None else float(now)
    # ``active_owner_readonly``: the same exactly-one-committed-row rule the
    # read slice resolves, over a read-only connection (no create, no migrate).
    store = Store(root)
    owner = store.active_owner_readonly()
    if owner is None:
        return {"enrolled": False, "audit": "no_owner", "items": []}
    payload = holds.list_holds(root, owner, now=now)
    # AGE IS FROM THE FIRST HOLD, not ``held_at``. The route's ``held_at`` is the
    # decision time (the Deliver window's start), and a review decision can be
    # written days after the message was first held — measured on the
    # operator's store 2026-11-18: oldest hold 17.9 d, its decision 8.1 d. The
    # question this command answers is "how long has this sat here", so it
    # reads the hold row's own time through the route module's same join.
    first_held = {
        bare: row["created_at"] for bare, row in holds._conversation_rows(store, owner).items()
    }
    items = list(payload["items"])
    audit_state, joined = ("skipped", {})
    if audit and items:
        audit_state, joined = asyncio.run(_join_audit(root, owner, items, want_body=excerpt))
    rows = []
    for item in items:
        join = joined.get(item["message_id"])
        first = first_held.get(item["message_id"])
        age = max(0.0, now - first) if first is not None else _age_seconds(item.get("held_at"), now)
        row = {
            "id": item["message_id"],
            "conversation_id": item["conversation_id"],
            "seq": item["seq"],
            "reason_code": item["hold_code"],
            "reason": reason_words(item["hold_code"], item["kind"]),
            "kind": item["kind"],
            "first_held_at": holds._iso(first),
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
        "count": len(rows),
        "items": rows,
    }


def _sender_cell(row: dict[str, Any], audit_state: str) -> str:
    if row["sender_label"] is None:
        return "withheld" if audit_state == "withheld_agent_shell" else "unavailable"
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
        f"{'on' if answer['release_enabled'] else 'off'}. Sender lookup: {answer['audit']}."
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
