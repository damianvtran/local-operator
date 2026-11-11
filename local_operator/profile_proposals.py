"""Account profile proposals: changes an agent suggested, reviewed in Agents.

WHY THIS EXISTS. An agent may want to change an account-wide agent or team
profile when the person did not explicitly ask for that edit in their own
words (a wake run tuning itself, a subagent, a Team Bridge or peer message, a
phrasing that does not name the profile). Those writes used to be filed as
SIGNED mesh approvals (``profile_approval``), which surfaced in Mesh → Approvals
— a section about devices — and could only be accepted with a presence-backed
operator key that most machines do not have, so nothing could accept them.

Now they are plain proposals in ``<config>/profile-proposals/pp_*.json``:

* the desktop Agents section lists and accepts / rejects / edit-then-accepts
  them (``server/routes/desktop_profile_proposals.py``) — the bearer-authenticated
  human surface, the same authority the profile editor already has with no
  signature (``desktop_profiles.py`` "THE USER IS THE ACTOR");
* the ``profile_proposal`` tool applies one from the conversation when the
  host-held consent witnesses say the owner agreed (``Session.
  decide_profile_proposal``), or behind the never-auto-approved
  ACCOUNT_PROFILE_GATE card otherwise.

WHAT IS KEPT from the signed path: CAS on the base commitment, the definition
lock + write-ahead journal (``definition_transaction.transact``), a backup in
the same format the agent/team tools write (so ``op=revert backup_id=…`` keeps
working from the filing conversation) and an append-only audit line.

CONSTRAINTS.
* Filing is idempotent: the id is derived from the content tuple.
* ``stale`` is computed on read (current image ≠ base), never stored, so a
  proposal never silently lands on a row that moved. There is no rebase: a
  stale proposal is refused (409 / "nothing was applied") and the originating
  conversation is asked for a fresh one.
* Only ``accept``/``reject``/``supersede`` write state; ``expired`` is folded on
  read (30 days) like ``network.approvals.presented``.
* Same-uid code can write these files: like the signed path before it, this is
  discipline (an agent cannot press Accept with the desktop bearer, which is
  scrubbed from agent shells), not a sandbox boundary.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Literal

from local_operator.definition_transaction import (
    _durable_json,
    commitment,
    definition_lock,
    prepare,
    transact,
)

logger = logging.getLogger(__name__)

PROPOSALS_DIRNAME = "profile-proposals"
AUDIT_FILENAME = "audit.jsonl"
SCHEMA = 1
#: Long on purpose: a proposal waits for a person to look at Agents, and the
#: stale-base check (not the clock) is what protects a late accept.
EXPIRY_SECONDS = 30 * 86400.0

STATE_PENDING = "pending"
STATE_ACCEPTED = "accepted"
STATE_REJECTED = "rejected"
STATE_SUPERSEDED = "superseded"
STATE_EXPIRED = "expired"
STATES = frozenset({STATE_PENDING, STATE_ACCEPTED, STATE_REJECTED, STATE_SUPERSEDED, STATE_EXPIRED})

Trust = Literal["owner", "agent", "untrusted"]
#: Fields whose value is free text: shown verbatim and editable in Agents.
_TEXT_FIELDS = frozenset({"instructions", "description", "label", "project", "manager", "model"})
_LIST_FIELDS = frozenset({"tags", "categories", "aliases"})
_FIELD_LABELS = {
    "tags": "settings & tags",
    "members": "members",
    "model": "model",
}


class ProposalError(Exception):
    """A refusal with a stable ``code`` the HTTP route and the tool both map.

    ``fields`` names the conflicting fields for ``profile_proposal_stale``.
    """

    def __init__(self, code: str, message: str, *, fields: Iterable[str] = ()) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.fields = tuple(fields)


def proposals_dir(root: Path) -> Path:
    return Path(root) / PROPOSALS_DIRNAME


def _path(root: Path, proposal_id: str) -> Path:
    if (
        not proposal_id.startswith("pp_")
        or len(proposal_id) != 19
        or any(c not in "0123456789abcdef" for c in proposal_id[3:])
    ):
        raise ProposalError("profile_proposal_not_found", f"no proposal {proposal_id!r}")
    return proposals_dir(root) / f"{proposal_id}.json"


def _image_model(kind: str) -> Any:
    from local_operator.profile_approval import AgentImage, TeamImage

    if kind == "agent":
        return AgentImage
    if kind == "team":
        return TeamImage
    raise ProposalError("profile_proposal_invalid", f"unknown profile kind {kind!r}")


def _validated(kind: str, image: Any) -> dict[str, Any] | None:
    if image is None:
        return None
    return _image_model(kind).model_validate(image).model_dump(mode="json")


def _changed_fields(before: dict[str, Any] | None, after: dict[str, Any]) -> list[str]:
    old = before or {}
    return [key for key in after if key not in {"id", "name"} and old.get(key) != after[key]]


def _snapshot(root: Path, kind: str, target_id: str) -> dict[str, Any] | None:
    from local_operator.profile_approval import snapshot

    return snapshot(root, kind, target_id)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("schema") == SCHEMA else None


def _audit(root: Path, record: dict[str, Any], action: str, actor: str) -> None:
    """Append one line per state change; the trail outlives the record files."""
    line = {
        "ts": time.time(),
        "id": record["id"],
        "kind": record["kind"],
        "name": record["name"],
        "action": action,
        "actor": actor,
        "fields": record.get("fields") or [],
        "base": record.get("base_commitment"),
        "after": record.get("after_commitment"),
    }
    directory = proposals_dir(root)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / AUDIT_FILENAME).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(line, sort_keys=True) + "\n")


def file(
    root: Path,
    *,
    profile_kind: str,
    operation: str,
    target_id: str,
    name: str,
    before: Any,
    after: Any,
    session_id: str,
    tool_call_id: str = "",
    trust: Trust = "agent",
    trigger: str = "user",
    legacy_approval_id: str | None = None,
    created_at: float | None = None,
) -> dict[str, Any]:
    """File (or re-confirm) a pending proposal; changes no profile.

    Idempotent: the id commits to kind, target, both images and the filing
    session, so a retried tool call answers the same record. A NEW pending
    proposal for the same target supersedes older pending ones — the person
    reviews the agent's latest idea, not a queue of drafts.

    An identical re-filing of a record that is already decided (rejected,
    superseded, accepted) or expired returns THAT record with its real state —
    never revived to pending — so the caller's receipt can say what actually
    happened instead of claiming a fresh proposal exists.
    """
    if operation not in {"create", "update", "revert"}:
        raise ProposalError("profile_proposal_invalid", f"cannot propose {operation!r}")
    before_image = _validated(profile_kind, before)
    after_image = _validated(profile_kind, after)
    if after_image is None:
        raise ProposalError("profile_proposal_invalid", "a proposal needs an after image")
    if (before_image is None) != (operation == "create"):
        raise ProposalError("profile_proposal_invalid", "only create has no before image")
    for image in (before_image, after_image):
        if image is not None and (image["id"] != target_id or image["name"] != name):
            raise ProposalError("profile_proposal_invalid", "image does not match its target")
    base, after_c = commitment(before_image), commitment(after_image)
    digest = commitment([profile_kind, target_id, base, after_c, session_id]).split(":")[1]
    proposal_id = "pp_" + digest[:16]
    now = time.time() if created_at is None else float(created_at)
    with definition_lock(root):
        path = _path(root, proposal_id)
        existing = _read(path)
        if existing is not None:
            shown = dict(existing)
            if shown.get("state") == STATE_PENDING and now >= float(shown.get("expires_at") or 0):
                shown["state"] = STATE_EXPIRED
            return shown
        record = {
            "schema": SCHEMA,
            "id": proposal_id,
            "kind": profile_kind,
            "operation": operation,
            "target_id": target_id,
            "name": name,
            "label": str(after_image.get("label") or ""),
            "before": before_image,
            "after": after_image,
            "base_commitment": base,
            "after_commitment": after_c,
            "fields": _changed_fields(before_image, after_image),
            "origin": {
                "session_id": session_id,
                "tool_call_id": tool_call_id,
                "trust": trust,
                "trigger": trigger,
                "legacy_approval_id": legacy_approval_id,
            },
            "state": STATE_PENDING,
            "created_at": now,
            "expires_at": now + EXPIRY_SECONDS,
            "decided_at": None,
            "decided_by": None,
            "applied": None,
            "note": None,
        }
        for other in _iter_raw(root):
            if (
                other.get("state") == STATE_PENDING
                and other.get("kind") == profile_kind
                and other.get("target_id") == target_id
            ):
                other.update(state=STATE_SUPERSEDED, decided_at=now, decided_by=proposal_id)
                _durable_json(_path(root, other["id"]), other)
                _audit(root, other, "superseded", proposal_id)
        _durable_json(path, record)
        _audit(root, record, "filed", f"session:{session_id}")
        return record


def _iter_raw(root: Path) -> Iterable[dict[str, Any]]:
    directory = proposals_dir(root)
    if not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.glob("pp_*.json")):
        data = _read(path)
        if data is not None and data.get("id") == path.stem:
            rows.append(data)
    return rows


def _fold(root: Path, record: dict[str, Any], now: float) -> dict[str, Any]:
    """Presentation fold: expiry by clock, ``stale`` by the live image."""
    folded = dict(record)
    if folded.get("state") == STATE_PENDING and now >= float(folded.get("expires_at") or 0):
        folded["state"] = STATE_EXPIRED
    folded["stale"] = False
    if folded["state"] == STATE_PENDING:
        try:
            current = _snapshot(root, folded["kind"], folded["target_id"])
        except Exception:  # noqa: BLE001 — an unreadable row is stale, never applicable
            logger.debug("profile proposal snapshot failed", exc_info=True)
            folded["stale"] = True
        else:
            folded["stale"] = commitment(current) != folded["base_commitment"]
    return folded


def list_proposals(
    root: Path,
    *,
    state: str | None = STATE_PENDING,
    name: str | None = None,
    now: float | None = None,
) -> list[dict[str, Any]]:
    """Folded proposals, newest first; imports legacy mesh records lazily."""
    import_legacy(root)
    moment = time.time() if now is None else now
    rows = [_fold(root, row, moment) for row in _iter_raw(root)]
    if state:
        rows = [row for row in rows if row["state"] == state]
    if name:
        rows = [row for row in rows if row.get("name") == name]
    rows.sort(key=lambda row: float(row.get("created_at") or 0), reverse=True)
    return rows


def get(root: Path, proposal_id: str, *, now: float | None = None) -> dict[str, Any] | None:
    import_legacy(root)
    record = _read(_path(root, proposal_id))
    if record is None:
        return None
    return _fold(root, record, time.time() if now is None else now)


# --- presentation (the §4 Summary / Detail shapes) -------------------------


def summary(record: dict[str, Any]) -> dict[str, Any]:
    origin = record.get("origin") or {}
    return {
        "id": record["id"],
        "kind": record["kind"],
        "name": record["name"],
        "label": record.get("label") or "",
        "operation": record["operation"],
        "fields": list(record.get("fields") or []),
        "state": record["state"],
        "stale": bool(record.get("stale")),
        "origin": {
            "trust": origin.get("trust") or "agent",
            "session_id": origin.get("session_id") or "",
            "trigger": origin.get("trigger") or "",
        },
        "created_at": record.get("created_at"),
        "expires_at": record.get("expires_at"),
    }


def _display(field: str, value: Any) -> tuple[str, str]:
    if value is None:
        return "", "text"
    if field in _LIST_FIELDS:
        return "\n".join(str(item) for item in value), "list"
    if isinstance(value, str):
        return value, "text"
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), "json"


def changes(
    before: dict[str, Any] | None, after: dict[str, Any], fields: Iterable[str]
) -> list[dict[str, Any]]:
    """One display row per changed field; ``after_text`` is what Accept writes."""
    from local_operator.profile_scope import profile_field_label

    rows = []
    for field in fields:
        before_text, fmt = _display(field, (before or {}).get(field))
        after_text, fmt_after = _display(field, after.get(field))
        rows.append(
            {
                "field": field,
                "label": _FIELD_LABELS.get(field, profile_field_label(field)),
                "before_text": before_text,
                "after_text": after_text,
                "format": fmt_after if after.get(field) is not None else fmt,
                "after": after.get(field),
            }
        )
    return rows


def detail(root: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Summary and display rows; a refresh never resets the proposal's base."""
    data = summary(record)
    data["changes"] = changes(record.get("before"), record["after"], record.get("fields") or [])
    data["base_commitment"] = record["base_commitment"]
    data["after_commitment"] = record["after_commitment"]
    return data


#: Hard cap on the approval card body. WHY: the card used to carry every
#: changed field's full before → after text, so a team brief of a few KB made
#: the "The agent is asking" card fill the conversation and hide its own
#: answer options. The full diff lives in Agents → <name>; the card only has to
#: say what changes and where to review it.
APPROVAL_CARD_MAX_CHARS = 600
#: Per summary line cap (contract: "truncated to ~80 chars").
_SUMMARY_LINE_MAX_CHARS = 80


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _item_counts(before: list[str], after: list[str]) -> tuple[int, int]:
    """Added/removed counts of an ordered sequence, via a difflib opcode walk."""
    import difflib

    added = removed = 0
    matcher = difflib.SequenceMatcher(a=before, b=after, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in {"replace", "delete"}:
            removed += i2 - i1
        if tag in {"replace", "insert"}:
            added += j2 - j1
    return added, removed


def _json_items(text: str) -> list[str] | None:
    """A JSON list field's items as canonical strings, or ``None`` if not a list.

    WHY: a pretty-printed JSON list re-flows its brackets and trailing commas,
    so adding one tool to ``["a"]`` diffs as ``+3 −1 lines``. Counting items
    says what actually changed.
    """
    if not text:
        return []
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, list):
        return None
    return [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in value]


def summarize_changes(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Count what changed per field from :func:`changes` rows.

    ``status`` says how to read ``added``/``removed``:

    - ``lines``: multi-line free text (and newline-joined list fields), counted
      over ``splitlines()`` — the lines the Agents diff viewer shows.
    - ``items``: a JSON list field, counted in list items (see
      :func:`_json_items`).
    - ``whitespace``/``reordered``: both counts are 0 yet the text differs;
      "+0 −0" would read as "nothing changed" on an apply card.
    - ``set``/``cleared``/``changed``: a single-line value on both sides, where
      "+1 −1 lines" says nothing useful about a model or label switch.

    Pure: no I/O, safe to call from any surface.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        before = str(row.get("before_text") or "")
        after = str(row.get("after_text") or "")
        before_lines, after_lines = before.splitlines(), after.splitlines()
        before_items = _json_items(before) if row.get("format") == "json" else None
        after_items = _json_items(after) if before_items is not None else None
        if before_items is not None and after_items is not None:
            added, removed = _item_counts(before_items, after_items)
            status = "items" if added or removed else "reordered"
        else:
            added, removed = _item_counts(before_lines, after_lines)
            scalar = len(before_lines) <= 1 and len(after_lines) <= 1
            if scalar:
                status = "set" if not before else "cleared" if not after else "changed"
            elif added or removed:
                status = "lines"
            else:
                status = "whitespace"
        out.append(
            {
                "field": row.get("field") or "",
                "label": row.get("label") or row.get("field") or "",
                "added": added,
                "removed": removed,
                "status": status,
            }
        )
    return out


_STATUS_TEXT = {"whitespace": "whitespace only", "reordered": "order changed"}


def _counts_text(item: dict[str, Any]) -> str:
    # ``status`` is always one of the str literals set above; typing it as
    # ``str`` (not ``Any``) lets ``.get(status, status)`` resolve to the
    # ``str`` overload instead of ``str | None``.
    status: str = str(item["status"])
    if status in {"lines", "items"}:
        return f"+{item['added']} −{item['removed']} {status}"
    return _STATUS_TEXT.get(status, status)


def summary_line(item: dict[str, Any]) -> str:
    """One ``<label>: +N −M lines`` (or ``<label>: changed``) receipt line."""
    return _clip(f"{item['label']}: {_counts_text(item)}", _SUMMARY_LINE_MAX_CHARS)


#: Cap on the card's ``Changes:`` line (contract line 2).
CHANGES_LINE_MAX_CHARS = 160


def changes_line(items: list[dict[str, Any]]) -> str:
    """Contract line 2: ``Changes: <label> +A −R lines; <label> changed; …``.

    One line, at most :data:`CHANGES_LINE_MAX_CHARS`; fields that do not fit
    collapse into a trailing ``… N more``. local-operator-ui parses this line
    (``Changes: `` prefix, ``; `` separator) to render its own compact
    summary, so keep the shape stable.
    """
    prefix = "Changes: "
    if not items:
        return prefix + "none listed"
    phrases = [_clip(f"{i['label']} {_counts_text(i)}", _SUMMARY_LINE_MAX_CHARS) for i in items]
    kept: list[str] = []
    for index, phrase in enumerate(phrases):
        remaining = len(phrases) - index - 1
        reserve = len(f"; … {remaining} more") if remaining else 0
        candidate = "; ".join([*kept, phrase])
        if len(prefix) + len(candidate) + reserve > CHANGES_LINE_MAX_CHARS:
            kept.append(f"… {len(phrases) - index} more")
            break
        kept.append(phrase)
    return prefix + "; ".join(kept)


def save_changes_line(labels: Iterable[str]) -> str:
    """Line 2 of a DIRECT (non-proposal) save card: ``Changes: a, b, … N more``.

    WHY names only, no ``+N −M`` counts: a direct save carries no before/after
    images — :class:`~local_operator.harness.types.DurableWriteIntent` exposes
    only the changed field/path names and ``summary``, a human-readable prose
    with several producer formats (a unified instructions diff, a replaced
    body, a whole new skill file, ``path: old → new`` edits, ``Remove:``
    lines). Every heuristic that recovered counts from that prose mis-counted
    some shape (PR #3777 R2-1, R2-3, R3-1: bullets read as removals, a ``→``
    inside a new file split into fake old/new halves). The card shows the
    clipped content below this line (:func:`clip_save_detail`), so the
    approver reads the change itself; this line only says what it touches.
    Proposal cards, which do have structured images, keep
    :func:`changes_line`'s counts.

    Same bound as :data:`CHANGES_LINE_MAX_CHARS`; names that do not fit
    collapse into ``… N more``. Duplicate names are listed once, in order.
    """
    prefix = "Changes: "
    names = list(dict.fromkeys(str(label) for label in labels if label))
    if not names:
        return prefix + "(no fields named)"
    phrases = [_clip(name, _SUMMARY_LINE_MAX_CHARS) for name in names]
    kept: list[str] = []
    for index, phrase in enumerate(phrases):
        remaining = len(phrases) - index - 1
        reserve = len(f", … {remaining} more") if remaining else 0
        candidate = ", ".join([*kept, phrase])
        if len(prefix) + len(candidate) + reserve > CHANGES_LINE_MAX_CHARS:
            kept.append(f"… {len(phrases) - index} more")
            break
        kept.append(phrase)
    return prefix + ", ".join(kept)


#: Cap on the change content a DIRECT (non-proposal) save card carries below
#: its summary. WHY it is shown at all: a direct save has no review page —
#: skills have no proposal path and no Agents diff — so the card is the only
#: place the approver can read what they approve (R2-2). WHY a cap: an
#: unbounded body was the original "card fills the conversation" report; ask
#: cards are now height-bounded and scroll on every surface, so ~4 KB reads
#: as a scrollable excerpt rather than a layout hazard.
SAVE_DETAIL_MAX_CHARS = 4000


def clip_save_detail(text: str, limit: int = SAVE_DETAIL_MAX_CHARS) -> str:
    """``text`` cut at a line boundary within ``limit`` chars, plus a tail line.

    The tail ``… N more lines not shown`` says how much was dropped, so a
    clipped excerpt is never mistaken for the whole change. A single first
    line longer than ``limit`` is clipped mid-line rather than dropped.
    """
    lines = text.splitlines()
    kept: list[str] = []
    used = 0
    for line in lines:
        cost = len(line) + (1 if kept else 0)
        if used + cost > limit:
            break
        kept.append(line)
        used += cost
    if not kept and lines:
        kept = [_clip(lines[0], limit)]
    hidden = len(lines) - len(kept)
    if hidden:
        kept.append(f"… {hidden} more line{'s' if hidden != 1 else ''} not shown")
    return "\n".join(kept)


def approval_card_text(record: dict[str, Any], rows: Iterable[dict[str, Any]]) -> str:
    """The ACCOUNT_PROFILE_GATE card for one proposal: exactly three lines.

    CONTRACT (shared with local-operator-ui, which parses it; TUI and mobile
    show it raw):

    1. ``Apply proposal <pp_id> to <kind> <name-repr>?`` — the name is
       Python-repr quoted (``'name'``, or ``"name"`` when it holds an
       apostrophe), so the UI links ``/agents?kind=…&name=…&proposal=…`` from
       the text alone and the proposal id stays visible: a spoken
       "apply pp_…" in chat binds by ``proposal_id in text``.
    2. :func:`changes_line` — per-field counts, never before/after text.
    3. ``Origin: <trust>. Review changes in Agents → <name>.``

    The full diff lives in Agents → <name>; the card stays under
    :data:`APPROVAL_CARD_MAX_CHARS` (names are capped at 128 chars).
    """
    head = f"Apply proposal {record['id']} to {record['kind']} {record['name']!r}?"
    origin = (record.get("origin") or {}).get("trust") or "agent"
    tail = _clip(f"Origin: {origin}. Review changes in Agents → {record['name']}.", 200)
    return "\n".join([head, changes_line(summarize_changes(rows)), tail])


def counts(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {"agent": {}, "team": {}}
    for row in rows:
        bucket = out.setdefault(row["kind"], {})
        bucket[row["name"]] = bucket.get(row["name"], 0) + 1
    return out


# --- decisions -------------------------------------------------------------


def _backup(
    root: Path, record: dict[str, Any], before_image: dict[str, Any], actor: str
) -> tuple[str | None, Path | None]:
    """Write the prior row in the agent/team tool backup format.

    Lands in the FILING session's ``profile-backups`` so ``agent|team
    op=revert backup_id=…`` works from that conversation; falls back to the
    proposals dir when retention already collected the session directory.
    """
    from local_operator.profile_scope import save_profile_backup, team_revision

    session_id = str((record.get("origin") or {}).get("session_id") or "")
    session_dir = Path(root) / "sessions" / session_id if session_id else None
    if session_dir is None or not session_dir.is_dir():
        session_dir = proposals_dir(root)
    kind, target_id = record["kind"], record["target_id"]
    authorized_by = f"profile-proposal:{record['id']}:{actor}"
    if kind == "agent":
        from local_operator.agent_profiles import profile_from_agent
        from local_operator.agents import AgentRegistry
        from local_operator.hub_sync.definition import definition_revision

        registry = AgentRegistry(root, migrate=False)
        row = registry.get_agent(target_id)
        fields = dict(registry.definition_revision_fields(target_id))
        fields["saver"] = profile_from_agent(registry, row).saver
        payload = {
            "kind": "agent",
            "name": str(row.name),
            "id": target_id,
            "scope": "global",
            "row": row.model_dump(mode="json"),
            "instructions": registry.get_agent_system_prompt(target_id) or "",
            "definition_fields": fields,
            "base_revision": definition_revision("agent", target_id, fields),
            "authorized_by": authorized_by,
            "proposal_id": record["id"],
        }
    else:
        from local_operator.teams import TeamRegistry

        team = TeamRegistry(root).get_team(target_id)
        payload = {
            "kind": "team",
            "name": team.name,
            "id": target_id,
            "scope": "global",
            "row": team.model_dump(mode="json"),
            "base_revision": team_revision(team),
            "authorized_by": authorized_by,
            "proposal_id": record["id"],
        }
    backup_id = save_profile_backup(str(session_dir), payload)
    return backup_id, session_dir


def _saved_revision(root: Path, kind: str, target_id: str) -> str:
    from local_operator.profile_scope import team_revision

    if kind == "agent":
        from local_operator.agent_profiles import profile_from_agent
        from local_operator.agents import AgentRegistry
        from local_operator.hub_sync.definition import definition_revision

        registry = AgentRegistry(root, migrate=False)
        fields = dict(registry.definition_revision_fields(target_id))
        fields["saver"] = profile_from_agent(registry, registry.get_agent(target_id)).saver
        return definition_revision("agent", target_id, fields)
    from local_operator.teams import TeamRegistry

    return team_revision(TeamRegistry(root).get_team(target_id))


def _pending(root: Path, proposal_id: str) -> dict[str, Any]:
    record = _read(_path(root, proposal_id))
    if record is None:
        raise ProposalError("profile_proposal_not_found", f"no proposal {proposal_id!r}")
    folded = _fold(root, record, time.time())
    if folded["state"] != STATE_PENDING:
        raise ProposalError(
            "profile_proposal_not_pending",
            f"{proposal_id} is {folded['state']}; nothing was applied.",
        )
    return record


def apply(
    root: Path,
    proposal_id: str,
    *,
    actor: str,
    edited_after: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply a pending proposal. The CALLER owns the consent decision.

    ``edited_after`` (edit-then-accept) may only replace values of fields the
    proposal already changes. A moved base always refuses: the originating
    conversation must file a fresh proposal. Returns ``{proposal, backup_id,
    backup_path}``; raises :class:`ProposalError` with ``profile_proposal_stale``
    / ``profile_proposal_not_pending`` / ``profile_proposal_invalid``.
    """
    import_legacy(root)
    with definition_lock(root):
        record = _pending(root, proposal_id)
        kind, target_id = record["kind"], record["target_id"]
        fields = list(record.get("fields") or [])
        current = _snapshot(root, kind, target_id)
        base = record.get("before")
        after = dict(record["after"])
        if commitment(current) != record["base_commitment"]:
            raise ProposalError(
                "profile_proposal_stale",
                f"{record['name']} changed since this was proposed; nothing was applied. "
                "Ask the originating conversation for a fresh proposal.",
            )
        if edited_after:
            extra = sorted(set(edited_after) - set(fields))
            if extra:
                raise ProposalError(
                    "profile_proposal_invalid",
                    f"only proposed fields can be edited (not {', '.join(extra)}).",
                )
            for field, value in edited_after.items():
                after[field] = value
        try:
            after = _validated(kind, after) or {}
        except Exception as exc:  # noqa: BLE001 — pydantic detail is the reason
            raise ProposalError("profile_proposal_invalid", f"invalid edit: {exc}") from exc
        if record["operation"] == "create":
            from local_operator.agents import AgentRegistry
            from local_operator.teams import TeamRegistry

            existing = (
                AgentRegistry(root, migrate=False).get_agent_by_name(record["name"])
                if kind == "agent"
                else TeamRegistry(root).get_team_by_name(record["name"])
            )
            if existing is not None and str(existing.id) != target_id:
                raise ProposalError(
                    "profile_proposal_stale", f"the name {record['name']!r} is now in use."
                )
        from local_operator.profile_approval import write_image

        backup: dict[str, Any] = {}

        def take_backup(image: Any) -> None:
            if image is not None:
                backup["id"], backup["dir"] = _backup(root, record, image, actor)

        # One journal per (proposal, written image): an edit-then-accept writes
        # a different image than the proposal and must not collide with it.
        transaction_id = proposal_id + "-" + commitment(after).split(":")[1][:16]
        binding = {"proposal_id": proposal_id, "actor": actor}
        # Prepared and executed under the SAME lock hold: the authority for this
        # write is the caller's consent decision (desktop bearer or the session
        # seam), not the journal; the journal only gives crash bookkeeping.
        prepare(
            root,
            transaction_id=transaction_id,
            before=base,
            after=after,
            read=lambda: _snapshot(root, kind, target_id),
            binding=binding,
        )
        transact(
            root,
            transaction_id=transaction_id,
            before=base,
            after=after,
            read=lambda: _snapshot(root, kind, target_id),
            write=lambda image: write_image(root, kind, target_id, image),
            backup=take_backup,
            binding=binding,
        )
        backup_path = None
        if backup.get("id"):
            from local_operator.profile_scope import update_profile_backup

            revision = _saved_revision(root, kind, target_id)
            update_profile_backup(str(backup["dir"]), backup["id"], saved_revision=revision)
            backup_path = str(Path(backup["dir"]) / "profile-backups" / f"{backup['id']}.json")
        revision = _saved_revision(root, kind, target_id)
        now = time.time()
        record.update(
            after=after,
            after_commitment=commitment(after),
            state=STATE_ACCEPTED,
            decided_at=now,
            decided_by=actor,
            applied={
                "backup_id": backup.get("id"),
                "backup_path": backup_path,
                "after_commitment": commitment(after),
                "after_revision": revision,
                "edited": bool(edited_after),
                "rebased": False,
            },
        )
        _durable_json(_path(root, proposal_id), record)
        _audit(root, record, "accepted", actor)
        return {
            "proposal": _fold(root, record, now),
            "backup_id": backup.get("id"),
            "backup_path": backup_path,
        }


def reject(root: Path, proposal_id: str, *, actor: str, note: str | None = None) -> dict[str, Any]:
    """Reject a pending proposal; the profile is untouched (the safe direction)."""
    import_legacy(root)
    with definition_lock(root):
        record = _pending(root, proposal_id)
        record.update(
            state=STATE_REJECTED,
            decided_at=time.time(),
            decided_by=actor,
            note=(note or None) and str(note)[:2000],
        )
        _durable_json(_path(root, proposal_id), record)
        _audit(root, record, "rejected", actor)
        return _fold(root, record, time.time())


# --- legacy mesh records ---------------------------------------------------


def import_legacy(root: Path) -> int:
    """Copy PENDING legacy mesh ``profile_save`` requests into this store.

    Lazy (runs on every list/get) and idempotent, with no one-shot marker on
    purpose: a runtime from the previous generation keeps filing into the mesh
    store until it restarts. Only records still ``requested`` AS A READER SEES
    THEM are imported: mesh expiry is folded on read (``A.presented``), so the
    stored field of a lapsed request still says ``requested`` and reading it raw
    would resurrect an expired request as pending; denied,
    expired, connected and withdrawn records are explicit outcomes and are NEVER
    revived. The mesh record itself is not touched — the mesh listing hides
    the kind (``network.approvals.NON_MESH_KINDS``) and retention prunes it.
    """
    from local_operator.network import approvals as A

    directory = A.approvals_dir(root)
    if not directory.is_dir():
        return 0
    imported = 0
    for path in sorted(directory.glob("ap_*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(raw, dict) or raw.get("kind") != A.KIND_PROFILE_SAVE:
            continue
        if A.presented(raw)["state"] != A.STATE_REQUESTED:
            continue
        try:
            what = raw["what"][A.KIND_PROFILE_SAVE]
            from local_operator.profile_approval import ProfileProposal

            # Validates images/commitments; the signature is not required here
            # because a proposal grants nothing until a person accepts it.
            legacy = ProfileProposal.model_validate(what)
        except Exception:  # noqa: BLE001 — leave an invalid record alone, hidden
            logger.warning("legacy profile approval %s is not a valid proposal", path.stem)
            continue
        if any(
            (row.get("origin") or {}).get("legacy_approval_id") == raw.get("approval_id")
            for row in _iter_raw(root)
        ):
            continue
        try:
            file(
                root,
                profile_kind=legacy.profile_kind,
                operation=legacy.operation,
                target_id=legacy.target_id,
                name=legacy.name,
                before=legacy.before,
                after=legacy.after,
                session_id=legacy.session_id,
                tool_call_id=legacy.tool_call_id,
                trust="agent",
                trigger="legacy",
                legacy_approval_id=str(raw.get("approval_id") or ""),
                created_at=legacy.created_at,
            )
        except ProposalError:
            logger.warning("legacy profile approval %s could not be imported", path.stem)
            continue
        imported += 1
    return imported
