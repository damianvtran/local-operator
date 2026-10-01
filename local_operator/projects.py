"""Projects: a tracked workstream row, one JSON file per project.

WHY THIS EXISTS
---------------
Work regularly spans several sessions — parallel features, a migration, a
release train — and nothing durable in the harness names that stream or says
where it stands. A *project* is that row. It is deliberately **not** an agent,
not a schedule and not a second team: it is the operator's own metadata about a
workstream, stored beside the teams they authored. (A team's own ``project``
brief — ``teams/<id>/project.md`` — is a different thing that happens to share
the word; see ``guide://projects``.)

STORAGE
-------
``<config_dir>/projects/<id>.json`` — one small document per project, plus a
store-wide lock file ``<config_dir>/projects/.lock``. Writes are
read-modify-write under a bounded ``flock``, published by a temp-write +
``fsync`` + ``os.replace``, so a reader never sees a torn row and a crashed
writer leaves at most a temp file. Reads are lock-free: ``os.replace`` is
atomic, so a reader sees complete bytes of one revision. This is the deliberate
simplification of :mod:`local_operator.teams`: a project has no briefs, so it
needs no per-row directory or swap machinery — and the ``schema`` int in the
row is the hook if that ever changes.

THE LINK IS STORED ONE WAY. A linked session id lives in the project row's
``sessions`` list; the reverse ("which projects is this session in") is derived
by scanning loaded rows (:meth:`ProjectRegistry.projects_for_session`). A
linked session that no longer exists is MARked, never auto-removed: silent
mutation on a read path is the class of bug this store refuses.

THE WRITE GUARD. Reads are lenient (``extra="ignore"``), so an older build can
read a newer row. A build REFUSES TO MUTATE a row whose ``schema`` exceeds
:data:`PROJECT_SCHEMA` — that is the rule which makes any future additive
format change bump safely while an older build is still in the field.

THE VIEW IS ONE COMPOSITION. :func:`build_project_view` is the single reading
of "what state is each linked session in" (runtime records, durable session
files, and an optional in-process overlay), used by the tool, the desktop
route and the mobile daemon alike — no second derivation per surface.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal, Sequence

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from local_operator.procstate import O_BINARY

logger = logging.getLogger(__name__)

#: The row format this build writes. A row whose ``schema`` is HIGHER than this
#: is readable but not mutable (see the module docstring): it was written by a
#: newer local-operator, and this build cannot promise to preserve fields it
#: does not know about. Schema 2 adds ``coordination_sessions`` (the
#: "filed by" provenance list) and the ``progress_refreshed_at/_by`` assertion
#: pair; the guard is what stops an older build's rewrite from silently
#: dropping either.
PROJECT_SCHEMA = 2

#: A project name is also a slash-command argument and an ``@project:<name>``
#: token, so it cannot contain spaces — the exact rule team names follow
#: (``teams._NAME_RE``).
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")

#: IDs are filesystem row addresses. ``uuid4().hex`` at write time; the shape
#: check mirrors ``validate_team_id`` so a transported or hand-edited row can
#: never point a path outside the store.
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

#: Session ids are the UI contract's shape (``src/shared/desktop-contract.ts``),
#: which is also the transcript directory name the harness generates.
_SESSION_ID_RE = re.compile(r"^[a-f0-9]{12}$")

#: ``q4``, ``payments-v2`` — short, lowercase, no spaces: a tag is a filter key.
_TAG_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,23}$")

#: Markdown prose describing the workstream; the surfaces RENDER it, not
#: dump it. Raised 240 -> 2000 (issue #1815): the tool schema and
#: guide://projects promised multi-paragraph markdown while this cap refused
#: the prose they invited, and the refusal was a bare pydantic sentence.
#: 2000 matches the sibling prose fields (teams, instruction-set
#: descriptions) and every render surface stays bounded without it: listing
#: rows clamp to PROJECT_ROW_CAP CELLS, the @project: snapshot trims with a
#: marker, detail pages scroll. An over-cap write from the tool is refused
#: with the submitted size and the remedy (project_tool._project_edit).
DESCRIPTION_MAX = 2000
PROGRESS_MAX = 1000
TAGS_MAX = 8
SESSIONS_MAX = 64
MILESTONES_MAX = 20
MILESTONE_NAME_MAX = 80
ESTIMATE_MAX = 1000.0

#: One cap for the ``owner``/``team`` attributions, at the milestone-name scale
#: (80): both are short single-line labels, and a value that would not fit one
#: line is refused rather than silently truncated into a name nobody wrote.
ATTRIBUTION_MAX = 80

#: The optional display ``title``: a short single-line name, the same scale as
#: the other one-line labels (80). Longer is refused rather than truncated.
TITLE_MAX = 80

#: The append-only history is BOUNDED: the newest 500 entries are kept, oldest
#: dropped first — deep enough for weeks of real reporting, bounded so one row
#: cannot grow without limit.
UPDATES_MAX = 500

#: Attachments are COPIED into the store, so both the count per update and
#: each file's size are bounded: 10 files of at most 5 MB. Beyond either, the
#: update is refused rather than trimmed — a silently dropped file is evidence
#: the writer believes was kept.
ATTACHMENTS_MAX = 10
ATTACHMENT_MAX_BYTES = 5 * 1024 * 1024

#: Extensions copied in as ``image`` attachments; everything else is ``data``
#: (the kinds renderers branch on, so the classification is stated once).
_IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"})

#: How long a progress snippet stays "fresh" when the operator has not said
#: otherwise. The live threshold is the settings key ``projects.stale_after_hours``
#: (int hours, default 4, bounds 1–168), resolved in ONE place —
#: :func:`stale_after_s` — which :func:`progress_is_stale` reads, so the badge,
#: the tool rows, the store payloads, the completion check and the wake-trigger
#: snapshot can never disagree about the window: a work turn yielding within it
#: of the last report is not asked to re-report; anything older is. This
#: constant stays as the code's DEFAULT (the settings registry's row is pinned
#: against it by ``tests/unit/test_settings_io.py``), and only the LIVE statuses
#: (``planning``/``active``/``qa``/``validation`` — :data:`PROJECT_LIVE_STATUSES`)
#: are ever stale at all (see :func:`progress_is_stale`: settled rows never read
#: stale).
PROJECT_PROGRESS_STALE_S: float = 14400.0

#: The settings key that configures the window. Spelled once; the resolver
#: below is its only reader, and the settings registry row carries the same
#: spelling.
STALE_AFTER_HOURS_KEY = "projects.stale_after_hours"

#: ``config.yml`` mtime (ns) per config dir → the resolved window. Read inside
#: per-row loops, so this must not re-parse the config per call, while an edit
#: must still land "at the next computation" (the section's LIVE promise): one
#: stat per call, a re-read only when the file actually moved. Bounded because
#: tests and multi-root callers each mint a key.
_STALE_AFTER_CACHE: dict[str, tuple[int | None, float]] = {}

#: Registry mutation lock budget: a bounded waiter, so a dead peer can never
#: park a tool call forever (the ``teams`` constants' shape).
_LOCK_TIMEOUT_S = 10.0
_LOCK_RETRY_S = 0.01

#: Cap on rows loaded from one store. Past this, the scan stops and warns: a
#: store with thousands of rows is a runaway writer, and a read path must stay
#: bounded.
_MAX_ROWS = 500

ProjectStatus = Literal["planning", "active", "qa", "validation", "paused", "done", "archived"]
EstimateUnit = Literal["points", "days"]

#: The status lifecycle, in the order the guide teaches it: planning (RFC or
#: research) → active (implementation) → qa (review/QA/design/copy cycles) →
#: validation (deployed and being validated, observation and fix-forward
#: included) → done (fully validated: requirements closed, todos addressed,
#: milestones complete), with ``paused`` a side-state and ``archived`` the end
#: of the drawer. The model literal above is the ONE source; a test pins it to
#: this tuple so a word can never exist in one surface and not another.
PROJECT_STATUSES: tuple[str, ...] = (
    "planning",
    "active",
    "qa",
    "validation",
    "paused",
    "done",
    "archived",
)

#: The statuses that are WORK IN FLIGHT: only these can read stale. ``paused``,
#: ``done`` and ``archived`` are deliberate statements that the record is
#: settled — the guard in :func:`progress_is_stale` and the filter in
#: :func:`stale_projects_for_session` both read this ONE set, so a badge and
#: the completion check cannot disagree about which rows can nag.
PROJECT_LIVE_STATUSES: frozenset[str] = frozenset({"planning", "active", "qa", "validation"})


class ProjectRegistryLockTimeout(TimeoutError):
    """The bounded wait for the cross-process projects registry lock expired."""


class ProjectNameConflictError(ValueError):
    """A create or rename landed on a name that is already taken (case-insensitive).

    A ValueError subclass so every existing ``except ValueError`` arm keeps
    catching it — the tool renders the same sentence it always did — while the
    desktop route can map THIS subclass to its 409 without matching on prose:
    a duplicate name is a state conflict the client resolves differently from a
    malformed value (422).
    """


class ProjectSchemaGuardError(RuntimeError):
    """A mutation was refused because the row was written by a newer build.

    Reads stay lenient on purpose; only writes are guarded, so a future format
    change can ship without an older build silently rewriting a row into a
    shape it does not understand (see the module docstring).
    """


def validate_project_id(project_id: str) -> str:
    """Return an ID that is exactly one safe filesystem path segment."""
    candidate = project_id or ""
    if not _ID_RE.fullmatch(candidate) or candidate in {".", ".."}:
        raise ValueError(
            "project id must be 1-128 characters of letters, digits, dot, "
            "underscore or hyphen, start with a letter or digit, and contain no "
            "path separators"
        )
    return candidate


def validate_project_name(name: str) -> str:
    """Return a stripped, legal project name or raise ``ValueError``."""
    candidate = (name or "").strip()
    if not _NAME_RE.match(candidate):
        raise ValueError(
            "project name must be 1-64 characters of letters, digits, dot, "
            "underscore or hyphen, and cannot start with a hyphen"
        )
    return candidate


def _iso_date_or_none(value: object, label: str) -> str | None:
    """Canonical ``YYYY-MM-DD``, or ``None`` for "not set".

    An empty string reads as unset: a row that echoed a cleared form can carry
    ``""`` meaning exactly what absent means, and a strict reader would drop a
    whole project row over it. Anything else must parse as an ISO date — the
    field is day-granular by design, and a value that is not a date is a bug
    worth refusing rather than storing.
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO YYYY-MM-DD date") from exc
    return parsed.isoformat()


def _edit_date(value: str | None, label: str) -> str | None:
    """A date as the EDIT vocabulary spells it.

    Three states reach an update: absent (leave as it is), cleared (``""`` or
    an explicit ``null``), and a real date. The empty string must NOT collapse
    into ``None`` at this layer — ``None`` is what "the caller said nothing"
    looks like after ``model_dump`` — so this validator keeps ``""`` intact and
    only canonicalises real dates. The apply path maps both clear spellings to
    ``None``.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO YYYY-MM-DD date, or '' to clear it")
    text = value.strip()
    if not text:
        return ""
    return _iso_date_or_none(text, label)


def _validate_estimate(value: object) -> float | None:
    """One estimate bound, shared by the row and by every input model.

    ``0 < estimate <= ESTIMATE_MAX``, fractional allowed: the two vocabularies
    Linear itself offers (points, days) are both summable numbers, and a value
    that could be zero or negative could not be rendered on any view.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("estimate must be a number")
    number = float(value)
    if not (0.0 < number <= ESTIMATE_MAX):
        raise ValueError(f"estimate must be greater than 0 and at most {ESTIMATE_MAX:g}")
    return number


def _validate_milestone_name(value: str | None) -> str:
    """One milestone-name rule, shared by the row and by every input model."""
    candidate = (value or "").strip()
    if not candidate:
        raise ValueError("milestone name must not be empty")
    if len(candidate) > MILESTONE_NAME_MAX:
        raise ValueError(f"milestone name must be at most {MILESTONE_NAME_MAX} characters")
    return candidate


def _validate_tags(tags: list[str] | None) -> list[str]:
    values = list(tags or [])
    if len(values) > TAGS_MAX:
        raise ValueError(f"at most {TAGS_MAX} tags")
    for tag in values:
        if not isinstance(tag, str) or not _TAG_RE.match(tag):
            raise ValueError(
                f"tag {tag!r} must be 1-24 characters of lowercase letters, "
                "digits, underscore or hyphen, and cannot start with a hyphen"
            )
    return values


def _validate_sessions(sessions: list[str] | None) -> list[str]:
    values = list(sessions or [])
    if len(values) > SESSIONS_MAX:
        raise ValueError(f"at most {SESSIONS_MAX} linked sessions")
    for session_id in values:
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
            raise ValueError(f"session id {session_id!r} is not a 12-character hex id")
    return values


def _attribution(value: str, label: str) -> str:
    """One attribution field: a session id, ``"operator"``, or ``""`` (unknown).

    Shared by the freshness pair's ``progress_reported_by`` and the assertion
    pair's ``progress_refreshed_by`` so the two can never accept different
    vocabularies — a checker name that the content side would refuse must be
    refused on the assertion side too.
    """
    candidate = (value or "").strip()
    if candidate in {"", "operator"} or _SESSION_ID_RE.fullmatch(candidate):
        return candidate
    raise ValueError(f"{label} must be a session id, 'operator', or ''")


def _short_text_or_none(value: object, label: str, cap: int) -> str | None:
    """One optional short-text field (``owner``/``team``/``title``), trimmed.

    Optional-not-defaulted is the contract: absent means UNKNOWN (``None``),
    never a placeholder string — a row written before these fields existed must
    read as unknown, and a cleared field returns to exactly that state rather
    than to ``""``. One rule, shared by the row and every input model, so the
    tool, the desktop route and the phone refuse the same values with the same
    sentence.
    """
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string (or '' to clear it)")
    text = value.strip()
    if not text:
        return None
    if len(text) > cap:
        raise ValueError(f"{label} must be at most {cap} characters")
    return text


def validate_milestones(milestones: list[Any] | None) -> list["ProjectMilestone"]:
    """The ONE list validator, shared by the row and by every input model.

    One validator and two entry points is the design's rule (the ``todo``
    tool's ``init``/``add`` split is the precedent): the full-replace form and
    the surgical ``milestone`` op must refuse the same lists, or a bulk edit
    could smuggle what the surgical op rejects.
    """
    values: list[ProjectMilestone] = []
    for item in milestones or []:
        if isinstance(item, ProjectMilestone):
            values.append(item)
        else:
            values.append(ProjectMilestone.model_validate(item))
    if len(values) > MILESTONES_MAX:
        # The remedy clause is what makes the two cap refusals one shape: the
        # link cap names ``unlink`` (which the guide's housekeeping section
        # promises of BOTH), so a caller told "at most 20" without "remove one
        # first" is the inconsistent one. The removal verb differs because the
        # surfaces differ: ``unlink`` is a slash verb, milestone removal is the
        # tool's ``op='milestone' remove=true``.
        raise ValueError(
            f"at most {MILESTONES_MAX} milestones; remove one with op='milestone' "
            "remove=true first"
        )
    seen: set[str] = set()
    for milestone in values:
        key = milestone.name.casefold()
        if key in seen:
            raise ValueError(f"duplicate milestone name {milestone.name!r}")
        seen.add(key)
    return values


class ProjectMilestone(BaseModel):
    """One milestone: a name, a target date, and the completion date.

    Milestone *status* (``completed`` / ``overdue`` / ``upcoming``) is
    deliberately NOT stored: it is derived at render from ``completed_at`` and
    ``target_date`` (:func:`milestone_status`), so a stored status can never
    drift from the dates that contradict it — the same "derived, not stored"
    rule the view composer applies to session liveness.
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(description="Short milestone name, unique per project.")
    target_date: str | None = Field(
        default=None, description="ISO YYYY-MM-DD the milestone is due, or null."
    )
    completed_at: str | None = Field(
        default=None, description="ISO YYYY-MM-DD the milestone was completed, or null."
    )

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _validate_milestone_name(value)

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "milestone target_date")

    @field_validator("completed_at")
    @classmethod
    def _completed_at(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "milestone completed_at")


class ProjectAttachment(BaseModel):
    """One file copied into the store beside an update entry.

    The FILE lives under ``<projects root>/attachments/<project id>/`` —
    copied in when the entry is written, never referenced from its original
    location (a scratch directory gets reaped; the evidence must survive).
    ``kind`` is classified from the extension at copy time, ``bytes`` is the
    size then, and ``path`` is where the copy lives — the resolvable handle a
    renderer opens. A reader tolerates junk here exactly as it does for the
    history itself (see :func:`_coerce_attachment`).
    """

    model_config = ConfigDict(extra="ignore")

    name: str = Field(default="", description="Original file name, for display.")
    kind: Literal["image", "data"] = Field(
        default="data", description="``image`` by extension; ``data`` otherwise."
    )
    path: str = Field(default="", description="Where the COPY lives, resolvable.")
    bytes: int = Field(default=0, description="Size at copy time.")
    added_at: str = Field(default="", description="ISO-8601 UTC copy time.")


class ProjectUpdateEntry(BaseModel):
    """One append-only progress report — the project's history entry.

    ``at`` is ISO-8601 UTC, ``text`` is the report as written (markdown,
    trimmed like every string field), ``by`` names the reporter (a session id
    or agent label; ``""`` when unknown), and ``attachments`` holds the files
    copied beside it. The log is append-only, newest last, bounded at
    :data:`UPDATES_MAX`; malformed or missing history reads as ``[]``.
    """

    model_config = ConfigDict(extra="ignore")

    at: str = ""
    text: str = ""
    by: str = ""
    attachments: list[ProjectAttachment] = Field(default_factory=list)


def _coerce_attachment(raw: dict[str, Any]) -> ProjectAttachment:
    """One attachment OBJECT from a stored row, tolerantly: junk degrades.

    Junk FIELDS degrade (an unknown ``kind`` reads ``data``, a non-numeric
    ``bytes`` reads ``0``); a non-object list item never reaches here —
    :func:`_coerce_updates` drops those, the same rule entries follow — so a
    bare string cannot become an empty attachment nobody can explain.
    """
    kind = str(raw.get("kind") or "").strip().lower()
    try:
        size = max(int(raw.get("bytes") or 0), 0)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        size = 0
    return ProjectAttachment(
        name=str(raw.get("name") or "").strip(),
        kind="image" if kind == "image" else "data",
        path=str(raw.get("path") or "").strip(),
        bytes=size,
        added_at=str(raw.get("added_at") or "").strip(),
    )


def _coerce_updates(value: object) -> list[ProjectUpdateEntry]:
    """The stored history, tolerantly: malformed or missing reads as ``[]``.

    No migration and no refusal: an entry that is not an object is dropped,
    fields that lost their shape degrade to their empty values, and the log
    stays bounded (the newest :data:`UPDATES_MAX` entries) exactly as a write
    keeps it — so a hand-edited row reads, and the next write re-saves it clean.
    """
    if not isinstance(value, list):
        return []
    entries: list[ProjectUpdateEntry] = []
    for item in value:
        if isinstance(item, ProjectUpdateEntry):
            # The in-process write path hands over already-validated entries
            # (``create_project``/``update_project`` build them); only rows
            # that came off DISK arrive as plain objects.
            entries.append(item)
            continue
        if not isinstance(item, dict):
            continue
        raw_attachments = item.get("attachments")
        attachments = [
            _coerce_attachment(attachment)
            for attachment in (raw_attachments if isinstance(raw_attachments, list) else [])
            if isinstance(attachment, dict)
        ][:ATTACHMENTS_MAX]
        entries.append(
            ProjectUpdateEntry(
                at=str(item.get("at") or "").strip(),
                text=str(item.get("text") or "").strip(),
                by=str(item.get("by") or "").strip(),
                attachments=attachments,
            )
        )
    return entries[-UPDATES_MAX:]


def _refuse_done_if_incomplete(milestones: Sequence[ProjectMilestone]) -> None:
    """The ``done`` gate's one refusal: what is incomplete, and the way through.

    Setting ``status='done'`` promises the plan is finished — requirements
    closed, milestones complete. An incomplete milestone is exactly the
    contradiction the gate exists for, so the refusal NAMES them and carries
    the deliberate escape hatch (``force_done=true``) rather than letting a
    false statement through. A plan with no milestones has nothing to prove
    and closes normally: the gate blocks an unfinished plan, not a plan-less
    row. Raises ``ValueError`` so every ``except ValueError`` arm (the tool,
    the routes) renders the same sentence.
    """
    incomplete = [milestone.name for milestone in milestones if milestone.completed_at is None]
    if not incomplete:
        return
    names = ", ".join(repr(name) for name in incomplete)
    raise ValueError(
        f"cannot set status 'done': {len(incomplete)} milestone"
        f"{'' if len(incomplete) == 1 else 's'} still incomplete ({names}) — "
        "complete them, or pass force_done=true to close with them open"
    )


def milestone_status(
    milestone: ProjectMilestone, *, today: date | None = None
) -> Literal["completed", "overdue", "upcoming"]:
    """The milestone's status, DERIVED at render, never stored.

    ``today`` exists for tests; left ``None`` the basis is :func:`_local_today`
    — the SAME date ``completed_at`` is stamped from — so "overdue" and "done
    today" cannot disagree about which day it is. The rule itself lives in
    :func:`milestone_state`, shared with the JSON-row readers.
    """
    moment = today or _local_today()
    return milestone_state(milestone.completed_at, milestone.target_date, today=moment.isoformat())


def milestone_state(
    completed_at: str | None,
    target_date: str | None,
    *,
    today: str | None = None,
) -> Literal["completed", "overdue", "upcoming"]:
    """The derived milestone status from its two raw fields — model OR JSON row.

    One rule, two shapes: :func:`milestone_status` passes a model's fields and
    the terminal renderers pass a composed row's, so the status the tool
    reports and the colour a footer or timeline paints cannot disagree about
    one milestone. Malformed dates compare as text rather than raising: the
    store tolerates a bad row, so a reader of that row must too (a receipt is
    a keystroke path and a canvas is a frame — neither may crash).
    """
    if completed_at:
        return "completed"
    target = str(target_date or "")
    if not target:
        return "upcoming"
    moment = today or _today_iso()
    try:
        overdue = date.fromisoformat(target) < date.fromisoformat(moment)
    except ValueError:
        overdue = target < moment
    return "overdue" if overdue else "upcoming"


class Project(BaseModel):
    """One project row, as stored in ``<config_dir>/projects/<id>.json``.

    The field name in the file is ``schema`` (the alias on ``schema_version``):
    a field literally named ``schema`` shadows ``BaseModel``'s own attribute
    and pydantic warns at class definition, so the attribute is named and the
    wire spelling is carried by the alias. Dump with ``by_alias=True`` for disk
    or the wire.
    """

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    schema_version: int = Field(
        default=PROJECT_SCHEMA,
        validation_alias="schema",
        serialization_alias="schema",
        description="Row format version; a higher value than this build knows refuses mutation.",
    )
    id: str = Field(description="uuid4 hex; also the row's file name.")
    name: str = Field(description="Unique (case-insensitive), 1-64 [A-Za-z0-9._-].")
    description: str = Field(default="", max_length=DESCRIPTION_MAX)
    #: Who owns the stream / which team manages it. Optional by design: absent
    #: means UNKNOWN (``None``), never a default string — a row written before
    #: these fields existed reads as unknown, and clearing returns the field to
    #: that state rather than to ``""``.
    owner: str | None = None
    team: str | None = None
    #: The DISPLAY name when it differs from the addressing key ``name``:
    #: listings show it first with the key as secondary. Optional by design —
    #: absent falls back to ``name`` on every surface (:func:`display_name`).
    title: str | None = None
    #: NOT typed as the strict ``ProjectStatus`` literal: a row written by a
    #: NEWER build must still LOAD (QA round 1, Q1) — the word is preserved
    #: verbatim and rendered everywhere (the board's leading column, a
    #: ``[? word]`` chip). The WRITE surfaces constrain the vocabulary instead
    #: (``ProjectEdit``, the tool params and the routes all type it
    #: ``ProjectStatus``), so an unknown word can never be SET — only carried
    #: through untouched.
    status: str = "active"
    progress: str = Field(default="", max_length=PROGRESS_MAX)
    progress_updated_at: float | None = None
    #: Session id, ``"operator"`` (a surface with no session), or ``""``.
    progress_reported_by: str = ""
    #: The ASSERTION pair: when a writer last checked that the recorded line
    #: still describes reality (an identical-normalized re-send, or the
    #: textless ``op='refresh'``), and who checked. Set by a refresh; CLEARED by
    #: the next append — the old assertion described superseded text. The
    #: content clock (``progress_updated_at``) never moves on a refresh: the
    #: badge keeps telling the truth about content age, while the assertion
    #: quiets the completion check for one window.
    progress_refreshed_at: float | None = None
    progress_refreshed_by: str = ""
    tags: list[str] = Field(default_factory=list)
    #: Sessions that WORK on the stream — the working set every liveness count
    #: and the completion check read.
    sessions: list[str] = Field(default_factory=list)
    #: Sessions that FILED the stream without working on it (the chief of
    #: staff's create-time auto-link lands here — see ``link_session``'s role).
    #: Provenance only: disjoint from ``sessions`` (the ``_links_disjoint``
    #: validator), never counted as a working link, never a liveness claim.
    coordination_sessions: list[str] = Field(default_factory=list)
    created_at: float = 0.0
    updated_at: float = 0.0
    # -- v2 planning fields (they shipped in schema 1; the schema-2 bump is the
    # coordination/refresh pair above) -----------------------------------------
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = Field(default=None, description="0 < estimate <= 1000.")
    estimate_unit: EstimateUnit = "points"
    milestones: list[ProjectMilestone] = Field(default_factory=list)
    # -- slice 2: the append-only history (rides any schema; the coordination
    # bump above does not touch it) --------------------------------------------
    #: Every write of a NEW progress line appends one entry here (newest last,
    #: bounded :data:`UPDATES_MAX`); the freshness pair above remains the
    #: "current" pointer at its tail. Malformed history reads as ``[]`` — a
    #: broken log never costs the row (:func:`_coerce_updates`).
    updates: list[ProjectUpdateEntry] = Field(default_factory=list)

    @field_validator("id")
    @classmethod
    def _id(cls, value: str) -> str:
        return validate_project_id(value)

    @field_validator("updates", mode="before")
    @classmethod
    def _updates(cls, value: object) -> list[ProjectUpdateEntry]:
        # BEFORE-validation: the tolerance is about SHAPE (a list of objects),
        # which must be read before the nested models ever see it.
        return _coerce_updates(value)

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return validate_project_name(value)

    @field_validator("owner")
    @classmethod
    def _owner(cls, value: object) -> str | None:
        return _short_text_or_none(value, "owner", ATTRIBUTION_MAX)

    @field_validator("team")
    @classmethod
    def _team(cls, value: object) -> str | None:
        return _short_text_or_none(value, "team", ATTRIBUTION_MAX)

    @field_validator("title")
    @classmethod
    def _title(cls, value: object) -> str | None:
        return _short_text_or_none(value, "title", TITLE_MAX)

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: list[str]) -> list[str]:
        return _validate_tags(value)

    @field_validator("sessions")
    @classmethod
    def _sessions(cls, value: list[str]) -> list[str]:
        return _validate_sessions(value)

    @field_validator("coordination_sessions")
    @classmethod
    def _coordination(cls, value: list[str]) -> list[str]:
        return _validate_sessions(value)

    @field_validator("progress_reported_by")
    @classmethod
    def _reported_by(cls, value: str) -> str:
        return _attribution(value, "progress_reported_by")

    @field_validator("progress_refreshed_by")
    @classmethod
    def _refreshed_by(cls, value: str) -> str:
        return _attribution(value, "progress_refreshed_by")

    @field_validator("start_date")
    @classmethod
    def _start_date(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "start_date")

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "target_date")

    @field_validator("completed_at")
    @classmethod
    def _completed_at(cls, value: object) -> str | None:
        return _iso_date_or_none(value, "completed_at")

    @field_validator("estimate")
    @classmethod
    def _estimate(cls, value: object) -> float | None:
        return _validate_estimate(value)

    @field_validator("milestones")
    @classmethod
    def _milestones(cls, value: list[Any]) -> list[ProjectMilestone]:
        return validate_milestones(value)

    @model_validator(mode="after")
    def _range_order(self) -> "Project":
        """An inverted range renders as nonsense, so it is refused at write.

        Clearing one side expresses "TBD", so only the both-set case is
        checked. Milestone dates are independent of this range on purpose (a
        milestone may deliberately sit outside it).
        """
        if self.start_date and self.target_date and self.target_date < self.start_date:
            raise ValueError("target_date cannot be before start_date")
        return self

    @model_validator(mode="after")
    def _links_disjoint(self) -> "Project":
        """The two link lists are one link set: an id sits in at most one of them.

        Disjointness is what makes every reader correct BY CONSTRUCTION — a
        reader of ``sessions`` counts working links and nothing else, and
        overlaps could otherwise smuggled a filing into a liveness count. The
        union is capped by the same discipline as a single list (the store's
        ``SESSIONS_MAX``), so ``link_session`` moving an id between lists can
        never grow past the cap.
        """
        overlap = sorted(set(self.sessions) & set(self.coordination_sessions))
        if overlap:
            raise ValueError(
                "a session link cannot be both working and coordination: " + ", ".join(overlap)
            )
        if len(self.sessions) + len(self.coordination_sessions) > SESSIONS_MAX:
            raise ValueError(
                f"a project holds at most {SESSIONS_MAX} session links across both "
                "working and coordination lists"
            )
        return self


class ProjectEdit(BaseModel):
    """The fields one create/update call may set.

    ABSENT means "leave it as it is"; every mutation path reads
    ``model_fields_set`` rather than treating ``None`` as a value, so a caller
    can set one field without clobbering the others. Dates accept ``""``/``null``
    as an explicit clear (see :func:`_edit_date`), as do ``owner`` and ``team``;
    ``milestones`` replaces the whole list.
    """

    model_config = ConfigDict(extra="forbid")

    name: str | None = None
    description: str | None = Field(default=None, max_length=DESCRIPTION_MAX)
    owner: str | None = None
    team: str | None = None
    title: str | None = None
    status: ProjectStatus | None = None
    progress: str | None = Field(default=None, max_length=PROGRESS_MAX)
    tags: list[str] | None = None
    start_date: str | None = None
    target_date: str | None = None
    completed_at: str | None = None
    estimate: float | None = None
    estimate_unit: EstimateUnit | None = None
    milestones: list[ProjectMilestone] | None = None

    @field_validator("name")
    @classmethod
    def _name(cls, value: str | None) -> str | None:
        return None if value is None else validate_project_name(value)

    @field_validator("owner")
    @classmethod
    def _owner(cls, value: object) -> str | None:
        return _short_text_or_none(value, "owner", ATTRIBUTION_MAX)

    @field_validator("team")
    @classmethod
    def _team(cls, value: object) -> str | None:
        return _short_text_or_none(value, "team", ATTRIBUTION_MAX)

    @field_validator("title")
    @classmethod
    def _title(cls, value: object) -> str | None:
        return _short_text_or_none(value, "title", TITLE_MAX)

    @field_validator("tags")
    @classmethod
    def _tags(cls, value: list[str] | None) -> list[str] | None:
        return None if value is None else _validate_tags(value)

    @field_validator("start_date")
    @classmethod
    def _start_date(cls, value: str | None) -> str | None:
        return _edit_date(value, "start_date")

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: str | None) -> str | None:
        return _edit_date(value, "target_date")

    @field_validator("completed_at")
    @classmethod
    def _completed_at(cls, value: str | None) -> str | None:
        return _edit_date(value, "completed_at")

    @field_validator("estimate")
    @classmethod
    def _estimate(cls, value: object) -> float | None:
        return _validate_estimate(value)

    @field_validator("milestones")
    @classmethod
    def _milestones(cls, value: list[Any] | None) -> list[ProjectMilestone] | None:
        return None if value is None else validate_milestones(value)


class MilestoneEdit(BaseModel):
    """The ``milestone`` op's arguments: add-or-update by name, or remove.

    ``target_date``'s ALIAS is the tri-state: absent (leave it), ``""``/null
    (clear it), or a date (set it) — the same vocabulary the full-replace list
    speaks, validated by the same function.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Milestone name; a name that does not exist is added.")
    target_date: str | None = None
    completed: bool | None = Field(
        default=None,
        description="True sets completed_at to today, False clears it, null leaves it.",
    )
    remove: bool = False

    @field_validator("name")
    @classmethod
    def _name(cls, value: str) -> str:
        return _validate_milestone_name(value)

    @field_validator("target_date")
    @classmethod
    def _target_date(cls, value: str | None) -> str | None:
        return _edit_date(value, "milestone target_date")


class ProjectUpdate:
    """What one :meth:`ProjectRegistry.update_project` call did.

    ``changed`` is False when every supplied field already held the value the
    caller sent (the no-op the model is told about rather than a silent write);
    ``refreshed`` is True when the call recorded an assertion that a stored
    line still describes reality (an identical-normalized re-send of stale
    text, or ``refresh_project``) — no append, content clock unmoved.
    """

    __slots__ = ("project", "changed", "refreshed")

    def __init__(self, project: Project, *, changed: bool, refreshed: bool) -> None:
        self.project = project
        self.changed = changed
        self.refreshed = refreshed


def stale_after_s(config_dir: Path | str | None = None) -> float:
    """The configured staleness window, in SECONDS.

    THE single reader of ``projects.stale_after_hours`` (int hours, default
    :data:`PROJECT_PROGRESS_STALE_S` worth of hours, bounds 1–168 enforced by
    the settings registry). Every staleness consumer — the badge, the tool
    rows, the store payloads, the completion check and the wake trigger's
    snapshot — resolves through :func:`progress_is_stale` and here, so they
    cannot disagree about the number; the wake-trigger side evaluates against
    the same key through the trigger settings snapshot.

    Never raises: an unreadable config, an unregistered key or a malformed
    value falls back to :data:`PROJECT_PROGRESS_STALE_S`. Cached per config dir
    by the config file's mtime — the read sits inside per-row loops and must
    not re-parse the config per call, while an edit must still land at the
    next computation. One stat per call is the price of that promise.
    """
    try:
        from local_operator.paths import config_dir as resolve_config_dir

        root = Path(config_dir) if config_dir is not None else resolve_config_dir()
        key = str(root)
        try:
            mtime: int | None = (root / "config.yml").stat().st_mtime_ns
        except OSError:
            mtime = None
        cached = _STALE_AFTER_CACHE.get(key)
        if cached is not None and cached[0] == mtime:
            return cached[1]
        value = PROJECT_PROGRESS_STALE_S
        if mtime is not None:
            # Only when a config file EXISTS: constructing a ConfigManager on
            # a directory whose config.yml is absent would create the
            # directory, and a read path must stay a read path. The registry
            # reads the value through the same reader the settings page uses,
            # so a hand-rolled YAML scan cannot silently disagree with it.
            from local_operator import settings_io
            from local_operator.config import ConfigManager

            setting = settings_io.resolve_key(STALE_AFTER_HOURS_KEY)
            if setting is not None:
                hours = settings_io.read_setting(ConfigManager(config_dir=root), setting)
                if isinstance(hours, (int, float)) and not isinstance(hours, bool) and hours > 0:
                    value = float(hours) * 3600.0
        if len(_STALE_AFTER_CACHE) > 16:
            _STALE_AFTER_CACHE.clear()
        _STALE_AFTER_CACHE[key] = (mtime, value)
        return value
    except Exception:  # noqa: BLE001 — a read path must never raise on its config
        logger.warning(
            "projects: could not read the staleness window; using the default", exc_info=True
        )
        return PROJECT_PROGRESS_STALE_S


def progress_is_stale(
    project: Project, *, now: float | None = None, window: float | None = None
) -> bool:
    """Whether the project's recorded progress needs refreshing.

    THE single staleness rule: no report yet is stale by construction (the
    first honest line is still owed), and a report older than the CONFIGURED
    window (:func:`stale_after_s`, default :data:`PROJECT_PROGRESS_STALE_S`)
    is stale. Computed here so the tool, the routes and the completion check
    cannot disagree about one record. ``window`` is seconds — the default
    resolves through :func:`stale_after_s`; a caller holding a config dir
    resolves it once per payload and passes it down, so every row of one
    payload is measured against the same boundary.

    ONLY ``planning``, ``active``, ``qa`` AND ``validation`` RECORDS CAN READ
    STALE — the statuses that are work in flight. Paused, done and archived are
    deliberate statements that the record is settled — the same filter
    ``stale_projects_for_session`` applies before it can name a project — so a
    settled row's badge must not nag about a snippet that is simply old. The
    set lives in :data:`PROJECT_LIVE_STATUSES`, and the guard HERE, in the one
    derivation every payload renders, rather than in each surface.
    """
    if project.status not in PROJECT_LIVE_STATUSES:
        return False
    if not project.progress:
        return True
    if project.progress_updated_at is None:
        return True
    moment = time.time() if now is None else now
    window_s = stale_after_s() if window is None else window
    return (moment - project.progress_updated_at) > window_s


def progress_asserted_within(
    project: Project, *, now: float | None = None, window: float | None = None
) -> bool:
    """Whether a refresh assertion is recent enough to quiet the completion check.

    THE nudge's assertion arm (P3): a writer that refreshed within the window
    has already acted on this record, so the reminder is retired for one
    window — while the operator-facing badge keeps reading the content clock's
    truth (two clocks, one name each: *content age* drives the badge and every
    summary, *assertion age* quiets the nudge). ``window`` defaults to the
    CONFIGURED window (:func:`stale_after_s`); the nudge passes the same
    resolved value staleness uses.
    """
    if project.progress_refreshed_at is None:
        return False
    moment = time.time() if now is None else now
    window_s = stale_after_s() if window is None else window
    return (moment - project.progress_refreshed_at) <= window_s


def _normalize_progress(text: str) -> str:
    """Normalized progress text for the refresh-vs-update classification.

    THE dedupe rule (P3): strip and collapse internal whitespace runs,
    case-SENSITIVE and punctuation-significant, so an exact-normalized re-send
    classifies as a refresh while everything else — including near-identical
    text — is an update that appends. There is deliberately NO similarity
    threshold: tiny edits flip meaning constantly (``QA 7/7`` vs ``QA 0/7``,
    ``merged as c9326d8`` vs ``merged as 0b2dc18``), and a threshold that
    swallows "(Refreshed …)" annotations swallows real outcomes too.
    """
    return " ".join((text or "").split())


def progress_date_text(updated_at: float | None) -> str | None:
    """The content clock as a LOCAL calendar day (``YYYY-MM-DD``), or ``None``.

    The day a reader sees in "no new content since <date>" / "the line still
    dates from <date>". Local because the stamp is a human-facing day (the
    same basis milestone derivation uses); ISO because that is the date
    vocabulary every other project field speaks.
    """
    if updated_at is None:
        return None
    return datetime.fromtimestamp(updated_at).date().isoformat()


def age_text(updated_at: float | None, *, now: float | None = None) -> str | None:
    """``3s``/``5m``/``2h``/``1d`` age of a raw stamp, or ``None``.

    The 90 s / 90 min / 48 h cut points live HERE — the one copy — and every
    surface imports this: :func:`reported_age` layers the ``progress`` guard on
    top for model callers, while the terminal renderers (which hold composed
    JSON rows, not ``Project`` objects) call it directly, and the row cap below
    travels with it (agent review round 1, F3/F4: the cut points and the caps
    had three copies whose comments each claimed canonicity).
    """
    if updated_at is None:
        return None
    moment = time.time() if now is None else now
    age = max(0.0, moment - updated_at)
    if age < 90:
        return f"{int(age)}s"
    if age < 5400:
        return f"{int(age // 60)}m"
    if age < 172800:
        return f"{int(age // 3600)}h"
    return f"{age / 86400:.0f}d"


#: One listing/receipt row stays scannable in a transcript. THE number, one
#: copy: the operator's listing (``slash_commands.project_listing_rows``) and
#: the model's listing (``tools/project_tool``) both import it, so the two
#: surfaces cannot truncate the same row at two different widths.
PROJECT_ROW_CAP = 160


def reported_age(project: Project, *, now: float | None = None) -> str | None:
    """``2h``-style age of the progress snippet, or ``None`` when none is stored.

    The caller composes the sentence, so one implementation of the age
    arithmetic serves the listing row, the ``show`` block, the update receipt
    and the completion-time reminder (``Session._project_reminder_text``)
    without any of them disagreeing about how old the snippet is. Lived in
    ``tools/project_tool.py`` until the completion check gained a fourth
    sentence to compose; it moved here with ``progress_is_stale`` for the same
    reason — the tool layer imports this module, never the other way round.
    The 90 s / 90 min / 48 h cut points themselves are :func:`age_text`'s,
    one copy for this and the JSON-row readers.
    """
    if not project.progress or project.progress_updated_at is None:
        return None
    return age_text(project.progress_updated_at, now=now)


def is_session_id(value: object) -> bool:
    """Is ``value`` shaped like a session id (the 12-hex UI contract)?

    Public because a RENDERER needs the answer too — the updates feed labels a
    reporter ``session <id>`` only when it is one, and an agent label otherwise
    — and reaching for the module's private ``_SESSION_ID_RE`` from another
    layer is a coupling nobody can change safely (agent review round 1, NIT-2).
    The shape itself stays defined HERE, beside the validator that enforces it.
    """
    return isinstance(value, str) and bool(_SESSION_ID_RE.fullmatch(value))


def _live_refresh_assertion(
    project: Project | Mapping[str, Any],
) -> tuple[float, float] | None:
    """``(refreshed_at, content_at)`` while the assertion is NEWER, else ``None``.

    "Live" is the one visibility rule the refresh annotation has: an assertion
    set before the line it describes was superseded (an append clears it, so
    this is a hand-edit guard as much as an invariant), and an assertion about
    no content at all paints nothing rather than a sentence without a date.
    Accepts the row model or its JSON dump, like :func:`display_name`.
    """
    if isinstance(project, Mapping):
        refreshed_at = project.get("progress_refreshed_at")
        updated_at = project.get("progress_updated_at")
    else:
        refreshed_at = project.progress_refreshed_at
        updated_at = project.progress_updated_at
    if refreshed_at is None or updated_at is None:
        return None
    if not isinstance(refreshed_at, (int, float)) or not isinstance(updated_at, (int, float)):
        return None
    if refreshed_at <= updated_at:
        return None
    return float(refreshed_at), float(updated_at)


def refreshed_age_text(
    project: Project | Mapping[str, Any], *, now: float | None = None
) -> str | None:
    """``1h``-style age of a live refresh assertion, or ``None``.

    The TOKEN every compact surface renders beside the content age ("the
    refreshed token beside the age"); the fuller :func:`refreshed_note` is the
    same fact as a sentence. Both read :func:`_live_refresh_assertion`, so a
    token and a sentence can never disagree about whether a refresh shows.
    """
    live = _live_refresh_assertion(project)
    if live is None:
        return None
    return age_text(live[0], now=now)


def refreshed_note(project: Project | Mapping[str, Any], *, now: float | None = None) -> str | None:
    """``refreshed 1h ago by session X — no new content since 2026-09-29``, or ``None``.

    THE operator-visible sentence for the assertion pair (P3 §4.3), shown only
    while the assertion is newer than the content. The by-clause follows the
    freshness pair's vocabulary: ``operator`` is named as such because a reader
    must not take it for a session id, and no checker reads as no clause at
    all.
    """
    live = _live_refresh_assertion(project)
    if live is None:
        return None
    age = age_text(live[0], now=now)
    if age is None:
        return None
    if isinstance(project, Mapping):
        checked_by = project.get("progress_refreshed_by") or ""
    else:
        checked_by = project.progress_refreshed_by or ""
    if checked_by == "operator":
        who = " by operator"
    elif checked_by:
        who = f" by session {checked_by}"
    else:
        who = ""
    return f"refreshed {age} ago{who} — no new content since {progress_date_text(live[1])}"


def display_name(project: Project | Mapping[str, Any]) -> str:
    """The human-facing name: the display ``title`` when set, else ``name``.

    THE fallback rule, here so every surface falls back the same way — the
    tool's rows and receipts, the terminal listing, the TUI canvases and
    footers. Accepts the row model or its JSON dump (the composed view's
    ``project`` mapping), because both shapes are read by those surfaces and
    neither may invent its own precedence. ``name`` stays the ADDRESSING key
    throughout (slash verbs, ``@project:<name>``, file names): this function
    decides what a reader SEES, never what a caller types.
    """
    if isinstance(project, Mapping):
        title = project.get("title")
        name = project.get("name")
    else:
        title, name = project.title, project.name
    return str(title or name or "")


def file_size_text(size: int) -> str:
    """Human size for refusals and attachment lines: ``12 B``/``4.0 KB``/``5.2 MB``.

    ONE decimal place for both units, deliberately: the natural ``:g`` format
    printed up to six significant digits (``4.00781 KB``, ``97.6562 KB``) —
    false precision for a screenshot size (design review round 1, D2) — and a
    conditional format (integer under 10, decimal above) would render the same
    magnitude in two shapes.
    """
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.1f} MB"
    if size >= 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size} B"


#: How many history entries a plain-text ``show`` reader prints by default —
#: a bounded tail, because the full log stays one explicit count away.
HISTORY_DEFAULT_TAIL = 5


def _history_field(entry: object, key: str) -> Any:
    """One history field, from the row model or the view mapping (see below)."""
    if isinstance(entry, Mapping):
        return entry.get(key)
    return getattr(entry, key, None)


def history_lines(
    project: Project | Mapping[str, Any], *, tail: int = HISTORY_DEFAULT_TAIL
) -> list[str]:
    """One ``show`` reader's history section: a bounded tail of the log.

    Accepts the row model or the composed view's ``project`` mapping — the
    tool's ``show`` renders the model, the routed receipt renders the view —
    so the two readers state the same section from one copy (the rule
    :func:`display_name` follows). Newest entries LAST (the log's own order),
    with the total in the header whenever older entries are elided; attachment
    lines carry the stored path, the resolvable handle to the copied-in file.

    Rows pass through :func:`collapse_adjacent_updates` first (ruling §4.4): a
    legacy or hand-edited run of identical-normalized entries prints as its
    newest row with a ``(re-sent N×)`` note. The header keeps counting STORED
    entries while 'latest N shown' counts printed rows, so a fully collapsed
    tail can read ``history (5, latest 1 shown)`` with the row saying why.
    """
    raw_entries = project.get("updates") if isinstance(project, Mapping) else project.updates
    entries = list(raw_entries) if isinstance(raw_entries, list) else []
    if tail <= 0:
        # `0` OMITS the section — including the "none recorded" line, so the
        # tool description's "0 omits the section" holds for an empty log too
        # (agent review round 1, F2).
        return []
    total = len(entries)
    if total == 0:
        return ["history: none recorded"]
    # Collapse BEFORE the tail slice so a run's count is the run's WHOLE
    # count, not just the entries the window happens to hold.
    shown = collapse_adjacent_updates(entries)[-tail:]
    if len(shown) == total:
        lines = [f"history ({total}):"]
    else:
        lines = [f"history ({total}, latest {len(shown)} shown):"]
    for entry, repeats in shown:
        stamp = str(_history_field(entry, "at") or "").strip() or "(no timestamp)"
        by = str(_history_field(entry, "by") or "").strip()
        # One line per entry: whitespace-collapsed so a multi-paragraph
        # markdown report cannot wrap the block; the row keeps the raw text.
        text = " ".join(str(_history_field(entry, "text") or "").split()) or "(empty)"
        by_text = f" by {by}" if by else ""
        note = f" (re-sent {repeats}×)" if repeats > 1 else ""
        lines.append(f"  - {stamp}{by_text}: {text}{note}")
        raw_attachments = _history_field(entry, "attachments")
        for attachment in raw_attachments if isinstance(raw_attachments, list) else []:
            name = str(_history_field(attachment, "name") or "").strip()
            kind = str(_history_field(attachment, "kind") or "data").strip()
            raw_size = _history_field(attachment, "bytes")
            try:
                size = int(raw_size or 0)
            except (TypeError, ValueError):
                size = 0
            path = str(_history_field(attachment, "path") or "").strip()
            lines.append(f"      attachment: {name} [{kind}, {file_size_text(size)}] {path}")
    return lines


def collapse_adjacent_updates(entries: Sequence[Any]) -> list[tuple[Any, int]]:
    """Display-only: adjacent identical-NORMALIZED history entries → newest + count.

    Ruling §4.4's dedupe, renderer-side and nothing more: the stored log is
    append-only and is never rewritten, and the write path cannot produce
    adjacent duplicates any more (an exact-normalized re-send records a refresh
    and appends nothing), so a collapse only ever reads a run an OLDER build or
    a hand-edit left behind — the corpus holds zero such pairs today. The
    NEWEST entry of a run is the one kept, so the surviving row is the one a
    reader would verify against reality; attachment sub-lines follow it.

    Normalization is the refresh classifier's (:func:`_normalize_progress`:
    strip + collapse whitespace runs, case-sensitive, punctuation
    significant), and near-identical variants are deliberately NOT collapsed —
    the same false-positive argument the refresh rule carries.
    """
    collapsed: list[tuple[Any, int, str]] = []
    for entry in entries:
        key = _normalize_progress(str(_history_field(entry, "text") or ""))
        if collapsed and collapsed[-1][2] == key:
            collapsed[-1] = (entry, collapsed[-1][1] + 1, key)
        else:
            collapsed.append((entry, 1, key))
    return [(entry, count) for entry, count, _key in collapsed]


def truncate_row(row: str, *, cap: int = PROJECT_ROW_CAP) -> str:
    """Truncate a listing row to ``cap`` CELLS (not characters) with ``…``.

    Cells, not ``len``: the cap exists to bound what the reader SEES, and a
    CJK description measured 272 cells in a 160-character row — the row wrapped
    to three visual lines on the surface whose constant was supposed to bound
    it (agent review round 1, F4).
    """
    from rich.cells import cell_len

    if cell_len(row) <= cap:
        return row
    clipped = ""
    for char in row:
        if cell_len(clipped + char) > cap - 1:
            break
        clipped += char
    return clipped.rstrip() + "…"


def store_error_text(exc: Exception) -> str:
    """A path-free one-line detail for a store failure.

    An ``OSError`` stringifies with the absolute path it touched
    (``[Errno 13] Permission denied: '/home/…/.lock'``); a user-facing receipt
    must not carry a filesystem path, so only the errno sentence survives.
    Unknown exceptions fall back to their own text (they carry no path).
    """
    if isinstance(exc, OSError):
        detail = exc.strerror or f"errno {exc.errno}"
        return detail[0].lower() + detail[1:] if detail else "i/o error"
    return str(exc)


def stale_projects_for_session(
    registry: ProjectRegistry,
    session_id: str,
    *,
    now: float | None = None,
    window: float | None = None,
) -> list[Project]:
    """The active, stale projects ``session_id`` WORKS ON, name-sorted.

    THE single reading behind the completion-time project check: the producer
    (``Session._project_continuation``) and the expiry scan
    (``Session._live_project_reminders``) compute their set through here, so
    the nudge and the check that retires it can never disagree about which
    projects are stale. Only the LIVE statuses (``planning``/``active``/
    ``qa``/``validation`` — :data:`PROJECT_LIVE_STATUSES`) can be named —
    paused, done and archived are deliberate statements that the record is
    settled, and a reminder about one would nag the session to revive it — and
    a project with no progress yet is stale by construction, because the first
    honest line is still owed.

    WORK LINKS ONLY: a coordination link ("filed by" — the chief of staff's
    create-time auto-link) is provenance, not participation, so it can never
    make an unrelated session answer for a project it does not work on; the
    filter reads ``sessions`` itself. And the assertion arm applies here, in
    the one derivation both ends share: a refresh recorded within the window
    retires the reminder for one window
    (:func:`progress_asserted_within`) even though the operator-facing badge
    keeps reading stale — a reminder that repeats after the session checked
    is the bug the exit list exists to prevent.
    """
    moment = time.time() if now is None else now
    window_s = stale_after_s(registry.config_dir) if window is None else window
    return [
        project
        for project in registry.projects_for_session(session_id)
        if session_id in project.sessions
        and project.status in PROJECT_LIVE_STATUSES
        and progress_is_stale(project, now=moment, window=window_s)
        and not progress_asserted_within(project, now=moment, window=window_s)
    ]


def stale_projects_fingerprint(
    projects: Sequence[Project],
) -> tuple[tuple[str, str, int, int], ...]:
    """The latch/expiry identity of a stale set: ``(id, status, int(stamp), int(refreshed))``.

    Sorted so two reads of the same set compare equal; the integer stamps are
    ``progress_updated_at`` and ``progress_refreshed_at`` floored (``0`` when
    unset). A report, a status change OR a refresh ALWAYS moves it — a
    replaced content stamp is at least the staleness window old, and the
    assertion stamp is newer than the content it asserts about — so the
    remaining stale projects earn another nudge in the same turn, and a
    refresh is what retires the reminder whose assertion it just answered.
    """
    return tuple(
        sorted(
            (
                project.id,
                project.status,
                int(project.progress_updated_at or 0),
                int(project.progress_refreshed_at or 0),
            )
            for project in projects
        )
    )


def _utc_now() -> float:
    return time.time()


def _utc_stamp(now: float) -> str:
    """ISO-8601 UTC, second granularity — the history log's ``at`` format.

    ``time.gmtime`` + ``strftime`` rather than ``datetime.isoformat`` so the
    suffix is ``Z`` — the same shape ``session.goal`` stamps — and two readers
    can compare the strings as text.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now))


def _local_today() -> date:
    """The ONE "today" this module reasons about: the OPERATOR'S local day.

    ``completed_at`` is stamped from this date and every date COMPARISON
    reads it, so the two cannot drift apart (agent review round 1, n1). The
    basis is LOCAL, deliberately: the stamp is a human-facing day, and a UTC
    basis stored tomorrow's date for an evening toggle west of Greenwich
    (UX round 1 follow-up; measured: a 2026-09-29 20:5x EDT toggle stored
    2026-09-30). The store is this machine's, so the machine's clock is the
    human who saw the toggle — and the one-basis rule this helper exists for
    is unchanged; only the basis moved.
    """
    return datetime.now().date()


def _today_iso() -> str:
    return _local_today().isoformat()


def _try_lock_exclusive(fd: int) -> bool:
    """Take one non-blocking exclusive lock attempt on ``fd``."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        try:
            if os.fstat(fd).st_size == 0:
                os.write(fd, b"\0")
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError as exc:
            if exc.errno in (errno.EDEADLOCK, errno.EACCES, errno.EAGAIN):  # noqa: F821
                return False
            raise

    import fcntl

    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):  # noqa: F821
            return False
        raise


def _unlock(fd: int) -> None:
    """Release a lock acquired by :func:`_try_lock_exclusive`."""
    if os.name == "nt":  # pragma: no cover - platform specific
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(fd, fcntl.LOCK_UN)


def _fsync_dir(path: Path) -> None:
    """Flush a directory entry, suppressing only unsupported implementations."""
    if os.name == "nt":  # pragma: no cover - exercised by Windows CI
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    except OSError as exc:
        unsupported = {
            errno.EINVAL,
            errno.EBADF,
            getattr(errno, "ENOTSUP", errno.EINVAL),
            getattr(errno, "EOPNOTSUPP", errno.EINVAL),
        }
        if exc.errno not in unsupported:
            raise
    finally:
        os.close(fd)


def _atomic_write_text(path: Path, text: str) -> None:
    """Publish one complete file without exposing a truncated target."""
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


class ProjectRegistry:
    """On-disk registry of projects under ``<config_dir>/projects``."""

    def __init__(self, config_dir: Path, refresh_interval: float = 5.0) -> None:
        self.config_dir = Path(config_dir)
        self.projects_dir = self.config_dir / "projects"
        self._projects: dict[str, Project] = {}
        self._last_refresh_time = 0.0
        self._refresh_interval = refresh_interval
        # The directory mtime the snapshot was taken at. A peer's publish is a
        # create/rename inside this directory, so a changed mtime is proof the
        # snapshot is behind even when the interval has not elapsed — the
        # refresh stays bounded (ONE stat per read) and a fresh row is visible
        # without waiting out the interval.
        self._dir_mtime_ns: int | None = None
        #: The ``OSError`` that made the last snapshot empty, or ``None``.
        #: Kept so a surface can tell "nothing is stored" from "nothing could
        #: be read": the store deliberately degrades to EMPTY on an unreadable
        #: directory (a reader must not lose its page to a permission blip),
        #: and answering the empty-store sentence to a broken store is exactly
        #: the misleading receipt the QA round flagged (Q4).
        self._load_error: OSError | None = None
        self._load()

    @property
    def load_error(self) -> OSError | None:
        """The error that emptied the last load, or ``None`` when it was read.

        Refreshed with every :meth:`_load`, so a store that becomes readable
        again clears itself without old state (a peer surface re-reads this on
        each verb).
        """
        return self._load_error

    def _load(self) -> None:
        """Replace the snapshot with every valid row currently on disk."""
        loaded: dict[str, Project] = {}
        try:
            children = sorted(self.projects_dir.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            # A store that has never been written is EMPTY, not unreadable:
            # `projects/` is created on first write, so its absence is the
            # normal initial state and the empty-store sentence is the right
            # answer there. Anything else — permissions, I/O — is a failure a
            # surface must not paper over as "nothing stored".
            self._load_error = None if isinstance(exc, FileNotFoundError) else exc
            self._projects = {}
            self._dir_mtime_ns = self._dir_mtime()
            self._last_refresh_time = time.time()
            return
        rows = 0
        for child in children:
            # Dot-prefixed entries are NEVER rows: the lock file and every
            # in-flight temp write live there, and a half-written temp must
            # stay invisible.
            if child.name.startswith("."):
                continue
            # A crafted symlink must never turn a read or a delete into an
            # operation outside the registry root.
            if child.is_symlink() or not child.is_file():
                continue
            if not child.name.endswith(".json"):
                continue
            rows += 1
            if rows > _MAX_ROWS:
                logger.warning(
                    "projects store at %s has more than %d rows; the rest are not loaded",
                    self.projects_dir,
                    _MAX_ROWS,
                )
                break
            try:
                with child.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if not isinstance(data, dict):
                    continue
                project = Project.model_validate(data)
                if project.id != child.name[: -len(".json")]:
                    logger.warning(
                        "ignoring project row %s: row id %s does not match",
                        child.name,
                        project.id,
                    )
                    continue
                if project.status not in PROJECT_STATUSES:
                    # QA round 1, Q1: an unknown status is a row from a NEWER
                    # build — it loads (the field is intentionally permissive)
                    # and renders in the board's leading column, but the drift
                    # is ANNOUNCED, because a silent load is how a vocabulary
                    # gap goes unnoticed until a surface misbehaves.
                    logger.warning(
                        "project row %s carries unknown status %r; loading it as-is",
                        child.name,
                        project.status,
                    )
                loaded[project.id] = project
            except FileNotFoundError:
                # The row was replaced between the scan and this open — the
                # documented atomic-publish gap, not a corrupt row.
                continue
            except Exception as exc:  # noqa: BLE001 — one bad file must not hide the rest
                logger.warning("invalid project row in %s: %s", child.name, exc)
        self._projects = loaded
        self._dir_mtime_ns = self._dir_mtime()
        self._last_refresh_time = time.time()
        # A readable store clears the flag (see ``load_error``). Set LAST so a
        # partial load never reports itself readable mid-scan.
        self._load_error = None

    def _dir_mtime(self) -> int | None:
        try:
            return self.projects_dir.stat().st_mtime_ns
        except OSError:
            return None

    def _refresh_if_needed(self) -> None:
        """Reload when the interval elapsed OR a peer changed the directory."""
        stale = (time.time() - self._last_refresh_time) > self._refresh_interval
        if not stale and self._dir_mtime() != self._dir_mtime_ns:
            stale = True
        if stale:
            self._load()

    def refresh(self) -> None:
        """Force one bounded re-read, ignoring the snapshot interval.

        The refusal surfaces read :attr:`load_error` BEFORE any read reaches
        the store, and a read is what refreshes the snapshot — so a store
        repaired since the flag was set kept refusing on the live surface
        (QA round 2, Q5: eight receipts over eleven seconds after a `chmod`
        back, every one stale). The snapshot's own staleness test cannot see
        a permission repair either: `chmod` moves the inode's ctime, not the
        directory's mtime, and the interval may not have elapsed. One stat +
        one listing per refusal is the bounded cost of telling the truth.
        """
        self._load()

    def list_projects(self) -> list[Project]:
        """Every project, metadata only, sorted by name (case-insensitive)."""
        self._refresh_if_needed()
        return sorted(self._projects.values(), key=lambda project: project.name.casefold())

    def get_project(self, project_id: str) -> Project:
        """One project by id. Raises ``KeyError`` when the id is unknown."""
        project_id = validate_project_id(project_id)
        self._refresh_if_needed()
        found = self._projects.get(project_id)
        if found is None:
            raise KeyError(f"Project with id {project_id} not found")
        return found

    def _find_cached_project_by_name(self, name: str) -> Project | None:
        wanted = (name or "").strip().casefold()
        if not wanted:
            return None
        for project in self._projects.values():
            if project.name.casefold() == wanted:
                return project
        return None

    def get_project_by_name(self, name: str) -> Project | None:
        """One project by name (case-insensitive), or ``None``."""
        self._refresh_if_needed()
        return self._find_cached_project_by_name(name)

    def projects_for_session(self, session_id: str) -> list[Project]:
        """Derived reverse lookup: the projects this session is linked to.

        MEMBERSHIP, either list: a coordination link ("filed by") is still a
        link, and the scoping sets that use this reading — nameless
        ``/project show``, the TUI's ◆ set — are sets of ROWS, not liveness
        claims. The completion check narrows to ``sessions`` itself
        (``stale_projects_for_session``), so a filing can never make an
        unrelated session answer for a project it does not work on.
        """
        self._refresh_if_needed()
        return sorted(
            (
                p
                for p in self._projects.values()
                if session_id in p.sessions or session_id in p.coordination_sessions
            ),
            key=lambda project: project.name.casefold(),
        )

    @contextmanager
    def _persistence_lock(self, *, wait: float | None = None) -> Iterator[None]:
        """Serialize one registry mutation across processes.

        The sidecar lives INSIDE ``projects/`` as ``.lock`` and is never
        mistaken for a row (dot-prefixed entries are skipped by ``_load``).
        Every kernel attempt is non-blocking and the retry loop is bounded; a
        timeout raises :class:`ProjectRegistryLockTimeout` so the tool boundary
        can present contention as a recoverable state rather than a traceback.
        Modifications keep the full budget — a write that gives up has failed
        to do the thing the user asked for.
        """
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.projects_dir / ".lock", os.O_CREAT | os.O_RDWR | O_BINARY, 0o600)
        acquired = False
        budget = _LOCK_TIMEOUT_S if wait is None else max(0.0, wait)
        deadline = time.monotonic() + budget
        try:
            while not acquired:
                acquired = _try_lock_exclusive(fd)
                if acquired:
                    break
                if time.monotonic() >= deadline:
                    raise ProjectRegistryLockTimeout(
                        "Timed out waiting for the projects registry lock; "
                        "retry after the other lop process finishes"
                    )
                time.sleep(_LOCK_RETRY_S)
            yield
        finally:
            if acquired:
                _unlock(fd)
            os.close(fd)

    def _save_project_locked(self, project: Project) -> Project:
        """Validate and publish ``project`` while ``_persistence_lock`` is held."""
        project_id = validate_project_id(project.id)
        # The write guard: a row from the future is readable but never
        # rewritten by this build (see the module docstring).
        if project.schema_version > PROJECT_SCHEMA:
            raise ProjectSchemaGuardError(
                f"project {project.name!r} was written by a newer local-operator "
                f"(schema {project.schema_version}); update this build to change it"
            )
        # The row is written in THIS build's format: stamp the current schema so
        # the guard above protects the fields this build adds — from its first
        # write on, an older build's later rewrite must refuse rather than
        # silently drop ``coordination_sessions`` / the assertion pair. A row
        # already at PROJECT_SCHEMA is untouched by this line, and an UNTOUCHED
        # old row stays at its old schema until something writes it (the
        # migration's "untouched rows stay 1" rule).
        if project.schema_version != PROJECT_SCHEMA:
            project.schema_version = PROJECT_SCHEMA
        name_key = project.name.casefold()
        occupant = next(
            (
                stored
                for stored in self._projects.values()
                if stored.id != project.id and stored.name.casefold() == name_key
            ),
            None,
        )
        if occupant is not None:
            raise ProjectNameConflictError(f"project {project.name!r} already exists")
        target = self.projects_dir / f"{project_id}.json"
        if target.is_symlink():
            raise ValueError("refusing to save through a symlinked project row")
        payload = project.model_dump(mode="json", by_alias=True)
        text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        _atomic_write_text(target, text)
        try:
            _fsync_dir(self.projects_dir)
        except BaseException:
            # A NEW row is not acknowledged unless its directory entry is
            # durable, so remove the unacknowledged row and a retry cannot
            # discover phantom success. An UPDATE keeps the just-replaced
            # bytes: ``os.replace`` has already overwritten the old revision,
            # and unlinking here would turn "possibly not durable" into
            # "definitely gone".
            if not existed:
                target.unlink(missing_ok=True)
                _fsync_dir(self.projects_dir)
            raise
        self._projects[project_id] = project
        self._dir_mtime_ns = self._dir_mtime()
        return project

    def _store_attachments(
        self, project_id: str, paths: Sequence[str | Path], *, now: float
    ) -> list[ProjectAttachment]:
        """Copy ``paths`` beside the project's row and describe the copies.

        Two-phase on purpose: every path is checked (exists, is a file, within
        the size cap) BEFORE any byte is copied, so one bad file cannot leave
        half an update's files orphaned with no metadata pointing at them. The
        copies live under ``<projects_dir>/attachments/<project id>/`` with a
        fresh unique name each (``<uuid><suffix>``), so two updates attaching
        the same source never collide and the original location is never
        referenced — a scratch directory gets reaped; the evidence must
        survive. Returns the metadata for the caller's new entry.
        """
        if not paths:
            return []
        if len(paths) > ATTACHMENTS_MAX:
            raise ValueError(f"at most {ATTACHMENTS_MAX} attachments per update (got {len(paths)})")
        sources: list[Path] = []
        for raw in paths:
            source = Path(str(raw)).expanduser()
            if not source.exists():
                raise ValueError(f"no file at {source}")
            if not source.is_file():
                raise ValueError(f"{source} is not a file")
            size = source.stat().st_size
            if size > ATTACHMENT_MAX_BYTES:
                raise ValueError(
                    f"{source.name!r} is {file_size_text(size)} — attachments "
                    f"are limited to {file_size_text(ATTACHMENT_MAX_BYTES)} each"
                )
            sources.append(source)
        target_dir = self.projects_dir / "attachments" / project_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = _utc_stamp(now)
        stored: list[ProjectAttachment] = []
        try:
            for source in sources:
                suffix = source.suffix.lower()
                destination = target_dir / f"{uuid.uuid4().hex}{suffix}"
                shutil.copyfile(source, destination)
                stored.append(
                    ProjectAttachment(
                        name=source.name,
                        kind="image" if suffix in _IMAGE_SUFFIXES else "data",
                        path=str(destination),
                        bytes=destination.stat().st_size,
                        added_at=stamp,
                    )
                )
        except BaseException:
            # A copy that dies mid-loop (disk full, revoked read) must not
            # leave earlier copies with no entry pointing at them; the
            # caller's own guard covers everything after this returns
            # (agent review round 1, F1).
            self._reclaim_attachment_files(attachment.path for attachment in stored)
            raise
        return stored

    def _reclaim_attachment_files(self, paths: Iterable[str]) -> None:
        """Best-effort unlink of stored attachment files; never raises.

        Only paths UNDER this store's ``attachments/`` root are ever removed:
        a row is a JSON file a human can edit, so a hand-written path must not
        turn a reclaim into an arbitrary-file delete. Failures are ignored —
        a leftover file is a leak, not corruption, and a reclaim error must
        never fail the operation that triggered it (agent review round 1, F1).
        """
        root = (self.projects_dir / "attachments").resolve()
        for raw in paths:
            try:
                target = Path(str(raw))
                if target.resolve().is_relative_to(root):
                    target.unlink(missing_ok=True)
            except OSError:  # pragma: no cover - best-effort reclaim
                continue

    def _row_references_paths(self, project_id: str, paths: Sequence[str]) -> bool:
        """True when the ON-DISK row references any of ``paths``.

        Used by ``update_project``'s take-back guard to tell a failure raised
        BEFORE the row replace (nothing references the new files — take them
        back) from one raised AFTER it: ``_save_project_locked`` keeps the
        just-replaced bytes when the follow-up directory fsync fails, and the
        file policy must mirror the row policy — unlinking here would leave a
        live row pointing at a deleted file (agent review round 2, F3). The
        disk artifact is the ground truth for "did the replace stand", so the
        check reads it rather than trusting a flag through the call. An
        unreadable or unparseable row reads as True: keeping a file can at
        worst leak, while unlinking one a landed row points at corrupts.
        """
        wanted = {str(path) for path in paths}
        if not wanted:
            return False
        try:
            payload = json.loads(
                (self.projects_dir / f"{validate_project_id(project_id)}.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, ValueError):
            return True
        entries = payload.get("updates") if isinstance(payload, dict) else None
        for entry in entries if isinstance(entries, list) else []:
            if not isinstance(entry, dict):
                continue
            for attachment in entry.get("attachments") or []:
                if isinstance(attachment, dict) and str(attachment.get("path") or "") in wanted:
                    return True
        return False

    def create_project(
        self,
        fields: ProjectEdit,
        *,
        sessions: Sequence[str] = (),
        coordination_sessions: Sequence[str] = (),
        progress_reported_by: str = "",
        force_done: bool = False,
    ) -> Project:
        """Create one project from ``fields``; the name must be free.

        ``sessions`` is the WORKING auto-link; ``coordination_sessions`` is the
        provenance ("filed by") link. The ``project`` tool and ``/project new``
        pass the calling session to exactly one of them — the role decision is
        the write surface's (:func:`local_operator.aida.state.is_aida_session`
        separates the chief of staff's filings from a worker's link, and a
        manager filing for a worker will read the same way). The desktop create
        route passes none, so a UI-created project starts unlinked and is
        linked from the projects surface.
        """
        name = validate_project_name(fields.name or "")
        now = _utc_now()
        with self._persistence_lock():
            # Unconditional refresh INSIDE the lock: an interval-gated snapshot
            # cannot prove uniqueness when another process wrote since our last
            # read.
            self._load()
            if self._find_cached_project_by_name(name) is not None:
                raise ProjectNameConflictError(f"project {name!r} already exists")
            supplied = fields.model_fields_set
            status = fields.status or "active"
            if status == "done" and not force_done:
                # The done gate, same rule as update: a create that declares
                # the plan finished must have finished milestones, or say so.
                _refuse_done_if_incomplete(fields.milestones or [])
            if "completed_at" in supplied:
                completed_at = None if not fields.completed_at else fields.completed_at
            elif status == "done":
                # Setting status='done' with completed_at OMITTED stamps today
                # (the one convenience of the pair; the full rule is stated in
                # the tool description).
                completed_at = _today_iso()
            else:
                completed_at = None
            project = Project(
                schema_version=PROJECT_SCHEMA,
                id=uuid.uuid4().hex,
                name=name,
                description=(fields.description or "").strip(),
                owner=fields.owner,
                team=fields.team,
                title=fields.title,
                status=status,
                progress=fields.progress or "",
                progress_updated_at=now if (fields.progress or "") else None,
                progress_reported_by=(progress_reported_by if (fields.progress or "") else ""),
                updates=(
                    [
                        ProjectUpdateEntry(
                            at=_utc_stamp(now),
                            text=fields.progress or "",
                            by=progress_reported_by,
                        )
                    ]
                    if (fields.progress or "")
                    else []
                ),
                tags=list(fields.tags or []),
                sessions=list(sessions),
                coordination_sessions=list(coordination_sessions),
                created_at=now,
                updated_at=now,
                start_date=fields.start_date or None,
                target_date=fields.target_date or None,
                completed_at=completed_at,
                estimate=fields.estimate,
                estimate_unit=fields.estimate_unit or "points",
                milestones=list(fields.milestones or []),
            )
            return self._save_project_locked(project)

    def update_project(
        self,
        project_id: str,
        fields: ProjectEdit,
        *,
        reporter: str = "",
        attachments: Sequence[str | Path] = (),
        force_done: bool = False,
    ) -> ProjectUpdate:
        """Merge ``fields`` into one project under the store lock.

        Per-field last-writer-wins: only the fields present in
        ``fields.model_fields_set`` are touched, so two sessions updating
        different fields serialize without clobbering. The progress rule is
        refresh ≠ update (P3): an identical-NORMALIZED re-send on a STALE record
        is a refresh — no append, an assertion pair written, the content clock
        UNMOVED so the stale badge keeps telling the truth; an identical
        re-send on a FRESH record is a no-op; everything else — near-identical
        text included, there is deliberately no similarity heuristic — is an
        update that writes, stamps, and APPENDS one entry to the history
        (clearing the assertion, which described superseded text).

        ``attachments`` rides that new entry: the paths are copied into the
        store and described on it. Attaching without a new line (a refresh, an
        identical re-send, a clear, or an update that omits ``progress``) is
        refused, because there would be no entry for the files to belong to.
        """
        project_id = validate_project_id(project_id)
        now = _utc_now()
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            candidate = current.model_copy(deep=True)
            supplied = fields.model_fields_set
            changed = False
            refreshed = False

            if "name" in supplied and fields.name is not None:
                candidate.name = fields.name
            if "description" in supplied and fields.description is not None:
                candidate.description = fields.description.strip()
            if "tags" in supplied and fields.tags is not None:
                candidate.tags = list(fields.tags)
            if "owner" in supplied:
                candidate.owner = fields.owner
            if "team" in supplied:
                candidate.team = fields.team
            if "title" in supplied:
                candidate.title = fields.title
            if "start_date" in supplied:
                candidate.start_date = fields.start_date or None
            if "target_date" in supplied:
                candidate.target_date = fields.target_date or None
            if "estimate" in supplied and fields.estimate is not None:
                candidate.estimate = fields.estimate
            if "estimate_unit" in supplied and fields.estimate_unit is not None:
                candidate.estimate_unit = fields.estimate_unit
            if "milestones" in supplied and fields.milestones is not None:
                candidate.milestones = list(fields.milestones)

            # completed_at and status interact, so the rule lives here once:
            #   * completed_at provided ("" clears / a date sets) wins.
            #   * else a transition INTO "done" stamps today.
            #   * moving away from done leaves the date untouched.
            if "completed_at" in supplied:
                candidate.completed_at = fields.completed_at or None
            if "status" in supplied and fields.status is not None:
                if fields.status == "done" and not force_done:
                    # The done gate: a plan closes when its milestones do, or
                    # when the caller says so explicitly. Checked against the
                    # MERGED list, so a deliberate same-call replace to a
                    # complete set can close the plan in one move; no
                    # milestones means nothing to prove (see the helper).
                    _refuse_done_if_incomplete(candidate.milestones)
                was_done = current.status == "done"
                candidate.status = fields.status
                if fields.status == "done" and not was_done and "completed_at" not in supplied:
                    candidate.completed_at = _today_iso()

            new_text = (fields.progress or "") if "progress" in supplied else ""
            # The classification, computed before any mutation: normalized
            # equality is refresh, normalized difference is update, empty is
            # clear. There is NO similarity threshold (see _normalize_progress).
            new_normalized = _normalize_progress(new_text)
            stored_normalized = _normalize_progress(current.progress)
            appends = bool(new_normalized) and new_normalized != stored_normalized
            # The attachment contract, checked BEFORE any mutation: files
            # attach to the entry a NEW line appends, so a refresh, an identical
            # re-send, a clear or an update without ``progress`` has nothing to
            # carry them.
            if attachments and not appends:
                raise ValueError(
                    "attachments ride a NEW progress line: send progress=<text> "
                    "with attach in the same update (a refresh, an identical "
                    "re-send or a clear appends no entry to carry them)"
                )
            stored_attachments: list[ProjectAttachment] = []
            evicted_entries: list[ProjectUpdateEntry] = []
            if "progress" in supplied:
                if appends:
                    candidate.progress = new_text
                    candidate.progress_updated_at = now
                    candidate.progress_reported_by = reporter
                    # The old assertion described superseded text; an append is
                    # exactly what clears it (T3's clock rule).
                    candidate.progress_refreshed_at = None
                    candidate.progress_refreshed_by = ""
                    stored_attachments = self._store_attachments(project_id, attachments, now=now)
                    appended = [
                        *candidate.updates,
                        ProjectUpdateEntry(
                            at=_utc_stamp(now),
                            text=new_text,
                            by=reporter,
                            attachments=stored_attachments,
                        ),
                    ]
                    # The cap evicts oldest-first, and the evicted entries'
                    # files are reclaimed only AFTER the save lands, so a
                    # failed save can never leave the live row pointing at
                    # files this call removed (agent review round 1, F1).
                    evicted_entries = appended[:-UPDATES_MAX]
                    candidate.updates = appended[-UPDATES_MAX:]
                    changed = True
                elif not new_normalized:
                    # An empty (or whitespace-only) snippet IS "no progress
                    # recorded": clearing the text clears BOTH pairs with it —
                    # the freshness pair and the refresh assertion, which
                    # described text that no longer exists — so a reader never
                    # sees a timestamp over nothing.
                    candidate.progress = ""
                    candidate.progress_updated_at = None
                    candidate.progress_reported_by = ""
                    candidate.progress_refreshed_at = None
                    candidate.progress_refreshed_by = ""
                    changed = True
                elif progress_is_stale(candidate, now=now, window=stale_after_s(self.config_dir)):
                    # REFRESH: identical-normalized text on a stale record. No
                    # append and the content clock is NEVER moved — the stale
                    # badge keeps reading the truth about content age — while
                    # the assertion pair quiets the completion check for one
                    # window. The checker is attributed separately from the
                    # reporter: the line's authorship did not change.
                    candidate.progress_refreshed_at = now
                    candidate.progress_refreshed_by = reporter
                    changed = True
                    refreshed = True
                # else: identical-normalized text on a fresh record — the
                # documented no-op, no write ("no reason to send it every turn").

            # Re-validate the merged candidate through the model's own rules
            # before anything touches disk (dates order, caps, grammar).
            try:
                candidate = Project.model_validate(candidate.model_dump(mode="json", by_alias=True))
                if candidate == current:
                    # Pydantic equality covers every field, so "nothing moved" is
                    # provable rather than inferred from the changed flags.
                    return ProjectUpdate(current, changed=False, refreshed=refreshed)
                if not changed:
                    # A merged value differs from the stored one only through
                    # equality above; reaching here with changed=False means the
                    # diff came from normalisation, which still deserves a write.
                    changed = True
                candidate.updated_at = now
                saved = self._save_project_locked(candidate)
            except BaseException:
                # A refusal raised BEFORE the row replace (name conflict,
                # schema guard, validation, a write that never published) must
                # take back the files this call already copied: an orphaned
                # copy has no entry to point at it (agent review round 1, F1).
                # AFTER the replace the file policy MIRRORS the row policy:
                # ``_save_project_locked`` deliberately keeps the just-replaced
                # bytes when the follow-up directory fsync fails, so the files
                # those bytes reference must be kept with them — unlinking
                # would leave a live row pointing at a deleted file (agent
                # review round 2, F3). "The replace stood" is read from disk,
                # the artifact the decision is about.
                if stored_attachments and not self._row_references_paths(
                    project_id, [attachment.path for attachment in stored_attachments]
                ):
                    self._reclaim_attachment_files(
                        attachment.path for attachment in stored_attachments
                    )
                raise
            for evicted in evicted_entries:
                self._reclaim_attachment_files(a.path for a in evicted.attachments)
            return ProjectUpdate(saved, changed=changed, refreshed=refreshed)

    def refresh_project(self, project_id: str, *, reporter: str = "") -> ProjectUpdate:
        """Record a CHECK that the stored progress line still describes reality.

        The textless ``op='refresh'``: an identical-normalized re-send reaches
        the same mechanics through :meth:`update_project`. Appends nothing and
        never moves the content clock (``progress_updated_at``) — the line is
        unchanged — but sets the assertion pair, which is what quiets the
        completion check for one window. Allowed only when the record is
        content-stale: on a fresh record a refresh is a no-op ("no reason to
        send it every turn"), and a record with no progress at all has nothing
        to assert about (the first honest line is the right act — and one with
        no text would render a sentence about content that does not exist).
        """
        project_id = validate_project_id(project_id)
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            if not _normalize_progress(current.progress):
                return ProjectUpdate(current, changed=False, refreshed=False)
            now = _utc_now()
            window = stale_after_s(self.config_dir)
            if not progress_is_stale(current, now=now, window=window):
                return ProjectUpdate(current, changed=False, refreshed=False)
            candidate = current.model_copy(deep=True)
            candidate.progress_refreshed_at = now
            candidate.progress_refreshed_by = _attribution(reporter, "progress_refreshed_by")
            candidate.updated_at = now
            saved = self._save_project_locked(candidate)
            return ProjectUpdate(saved, changed=True, refreshed=True)

    def link_session(
        self,
        project_id: str,
        session_id: str,
        *,
        role: Literal["work", "coordination"] = "work",
    ) -> tuple[Project, bool]:
        """Link one session to one project under ``role``. Returns ``(project, changed)``.

        ``role`` names WHICH list the link belongs to: ``work`` (default — the
        session drives the stream, feeds liveness counts and the completion
        check) or ``coordination`` ("filed by" — provenance only). The caller
        DECIDES the role at the write surface (the tool and ``/project new``
        ask ``is_aida_session``); the store only enforces the one representation
        rule. An id already linked under the requested role is a no-op; an id
        linked under the OTHER role MOVES between lists — that move is how a
        misfired migration demotion is restored with one ``op='link'``. The
        two lists share the one cap, and a move never grows it.
        """
        if role not in ("work", "coordination"):
            raise ValueError("role must be 'work' or 'coordination'")
        project_id = validate_project_id(project_id)
        session_id = _validate_sessions([session_id])[0]
        target = "sessions" if role == "work" else "coordination_sessions"
        other = "coordination_sessions" if role == "work" else "sessions"
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            if session_id in getattr(current, target):
                return current, False
            linked = session_id in current.sessions or session_id in current.coordination_sessions
            if (
                not linked
                and len(current.sessions) + len(current.coordination_sessions) >= SESSIONS_MAX
            ):
                raise ValueError(
                    f"project {current.name!r} already has {SESSIONS_MAX} linked "
                    "sessions; unlink one first"
                )
            candidate = current.model_copy(deep=True)
            setattr(candidate, other, [s for s in getattr(candidate, other) if s != session_id])
            setattr(candidate, target, [*getattr(candidate, target), session_id])
            candidate.updated_at = _utc_now()
            return self._save_project_locked(candidate), True

    def unlink_session(self, project_id: str, session_id: str) -> tuple[Project, bool]:
        """Unlink one session from one project, from EITHER list. Returns ``(project, removed)``.

        Target by id across both lists: the caller states "this session should
        not be linked", and which list held it is bookkeeping a reader should
        not have to look up first.
        """
        project_id = validate_project_id(project_id)
        session_id = _validate_sessions([session_id])[0]
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            if (
                session_id not in current.sessions
                and session_id not in current.coordination_sessions
            ):
                return current, False
            candidate = current.model_copy(deep=True)
            candidate.sessions = [s for s in candidate.sessions if s != session_id]
            candidate.coordination_sessions = [
                s for s in candidate.coordination_sessions if s != session_id
            ]
            candidate.updated_at = _utc_now()
            return self._save_project_locked(candidate), True

    def set_milestone(self, project_id: str, fields: MilestoneEdit) -> tuple[Project, str]:
        """Add, update or remove one milestone by name. Returns ``(project, action)``.

        ``action`` is ``"added"``, ``"updated"`` or ``"removed"`` — the receipt
        the tool and the routes report. Renaming is remove + add, documented
        and deliberate: two names in one call is a rename vocabulary the surgical
        op does not need.
        """
        project_id = validate_project_id(project_id)
        supplied = fields.model_fields_set
        with self._persistence_lock():
            self._load()
            current = self._projects.get(project_id)
            if current is None:
                raise KeyError(f"Project with id {project_id} not found")
            candidate = current.model_copy(deep=True)
            wanted = fields.name.casefold()
            index = next(
                (i for i, m in enumerate(candidate.milestones) if m.name.casefold() == wanted),
                None,
            )
            if fields.remove:
                if index is None:
                    raise KeyError(f"no milestone named {fields.name!r}")
                del candidate.milestones[index]
                action = "removed"
            elif index is None:
                candidate.milestones.append(
                    ProjectMilestone(
                        name=fields.name,
                        target_date=(
                            None if fields.target_date in (None, "") else fields.target_date
                        ),
                        completed_at=_today_iso() if fields.completed else None,
                    )
                )
                action = "added"
            else:
                milestone = candidate.milestones[index]
                before = milestone.model_dump(mode="json")
                if "target_date" in supplied:
                    milestone.target_date = fields.target_date or None
                if fields.completed is not None:
                    # NOTE: a re-completion re-stamps to today; the prior date
                    # is not kept (restoring it would need a completion
                    # history — a store-schema change, deferred in the UX
                    # round 1 remediation record).
                    milestone.completed_at = _today_iso() if fields.completed else None
                if milestone.model_dump(mode="json") == before:
                    # Nothing supplied (or nothing moved): report the truthful
                    # no-change instead of churning ``updated_at`` on the row.
                    return current, "unchanged"
                candidate.milestones[index] = ProjectMilestone.model_validate(
                    milestone.model_dump(mode="json")
                )
                action = "updated"
            # The shared validator runs on the whole list, so the surgical op
            # and the full-replace form cannot disagree about what is legal.
            candidate.milestones = validate_milestones(candidate.milestones)
            candidate.updated_at = _utc_now()
            return self._save_project_locked(candidate), action

    def delete_project(self, project_id: str) -> None:
        """Permanently remove one row. Session directories are never touched."""
        project_id = validate_project_id(project_id)
        with self._persistence_lock():
            # Delete participates in the same ordering as save, so a
            # delete-recreate-stale-save sequence has one unambiguous winner.
            self._load()
            if project_id not in self._projects:
                raise KeyError(f"Project with id {project_id} not found")
            current = self._projects.get(project_id)
            if current is not None and current.schema_version > PROJECT_SCHEMA:
                # The GUARD covers removal too, deliberately: a row written by a
                # newer build holds fields this one cannot render, so deleting it
                # would destroy data the deleter was never shown. The remedy is
                # the same sentence every mutation uses (update this build).
                raise ProjectSchemaGuardError(
                    f"project {current.name!r} was written by a newer local-operator "
                    f"(schema {current.schema_version}); update this build before deleting it"
                )
            target = self.projects_dir / f"{project_id}.json"
            if target.is_symlink():
                raise ValueError("refusing to delete a symlinked project row")
            # Remove the on-disk copy FIRST: if the unlink fails, the cache
            # still agrees with disk and the row remains visible.
            if target.exists():
                target.unlink()
                _fsync_dir(self.projects_dir)
            # The row is gone; reclaim its copied-in files. Best-effort — a
            # leftover directory is a leak, never a reason to fail a delete
            # that already happened (agent review round 1, F1).
            shutil.rmtree(self.projects_dir / "attachments" / project_id, ignore_errors=True)
            self._projects.pop(project_id, None)
            self._dir_mtime_ns = self._dir_mtime()


# ---------------------------------------------------------------------------
# The view composer
# ---------------------------------------------------------------------------


def _load_subagent_summary(session_dir: Path) -> dict[str, Any] | None:
    """``{"running", "settled", "names"}`` from the roster sidecar, or ``None``.

    A missing sidecar and an unreadable one both read ``None`` ("no roster"),
    never zeroes: a session that never launched a child has no sidecar, and a
    0/0 summary would claim an empty roster it cannot prove.
    """
    path = session_dir / "subagent-roster.v1.json"
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return None
    jobs = payload.get("jobs")
    if not isinstance(jobs, list):
        return None
    running = 0
    settled = 0
    names: list[str] = []
    for job in jobs:
        if not isinstance(job, dict):
            continue
        status = str(job.get("status", ""))
        label = str(job.get("label", "") or job.get("id", ""))
        if status == "running":
            running += 1
            if len(names) < 5:
                names.append(label)
        else:
            settled += 1
    for job in jobs:
        if len(names) >= 5:
            break
        if isinstance(job, dict) and str(job.get("status", "")) != "running":
            names.append(str(job.get("label", "") or job.get("id", "")))
    return {"running": running, "settled": settled, "names": names[:5]}


def _load_todo_summary(session_dir: Path) -> dict[str, int] | None:
    """``{"open", "total"}`` from the newest ``todo_snapshot`` row, or ``None``.

    The bounded backward scan (``read_latest_custom``) answers a one-row
    question without parsing the journal: transcripts reach tens of MB, and a
    project view must never pay a full parse for a count. A session that never
    persisted a snapshot reads ``None`` ("unknown"), never 0.
    """
    from local_operator.session.transcript import read_latest_custom

    details = read_latest_custom(session_dir, "todo_snapshot")
    if not details:
        return None
    items = details.get("items")
    if not isinstance(items, list):
        return None
    total = 0
    open_count = 0
    for phase in items:
        if not isinstance(phase, dict):
            continue
        rows = phase.get("items")
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, dict):
                continue
            total += 1
            if item.get("status") == "pending":
                open_count += 1
    return {"open": open_count, "total": total}


def scan_runtime_states(config_dir: Path) -> dict[str, dict[str, Any]]:
    """One runtime scan, keyed by session id, in the view payload's own shape.

    THE one audit of "is each session running, and where", read by
    :func:`build_project_view` and by callers that compose MANY projects (the
    desktop listing counts live sessions per row): one scan serves the whole
    call, so a listing of N projects does not walk the runtime registry N times.

    READER MODE (``reap=False``): a view must leave ``run/mobile``
    byte-identical — the record on disk is evidence a later "why did this die"
    question reads, and reaping is somebody else's job (every ``lop sessions``
    and every interactive boot sweeps). The run directory's EXISTENCE is checked
    first, so a machine that has never run a runtime does not get one created by
    a project view. Any failure degrades to an empty map: reading a project
    never fails on runtime state.
    """
    from local_operator.session.runtime import registry as runtime_registry
    from local_operator.session.runtime.types import RUN_DIRNAME

    root = Path(config_dir)
    by_session: dict[str, list[Any]] = {}
    if (root / RUN_DIRNAME).is_dir():
        try:
            for record, state in runtime_registry.scan(root, reap=False):
                by_session.setdefault(record.session_id, []).append((record, state))
        except Exception:  # noqa: BLE001 — a view never fails on runtime state
            logger.debug("could not scan runtime records", exc_info=True)

    states: dict[str, dict[str, Any]] = {}
    for session_id, records in by_session.items():
        picked = _pick_record(records)
        if picked is None:
            continue
        record, state = picked
        states[session_id] = {
            "state": state,
            "busy": bool(getattr(record, "busy", False)),
            "heartbeat_age_s": max(0.0, time.time() - record.heartbeat_at),
            "pid": record.pid,
        }
    return states


def _pick_record(records: list[Any]) -> Any | None:
    """The best record for one session id: live beats wedged beats stale, then newest."""
    rank = {"live": 2, "wedged": 1, "stale": 0}
    best: Any | None = None
    for record, state in records:
        if best is None:
            best = (record, state)
            continue
        best_record, best_state = best
        if (rank.get(state, 0), record.heartbeat_at) > (
            rank.get(best_state, 0),
            best_record.heartbeat_at,
        ):
            best = (record, state)
    return best


def build_project_view(
    project: Project,
    *,
    config_dir: Path,
    live: dict[str, dict[str, Any]] | None = None,
    records: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compose the ONE view of a project: its row plus one row per linked session.

    Three sources, in priority order, exactly as the design states:

    1. **Runtime records** — one ``registry.scan`` for the whole call, filtered
       per session id. A record classifies ``live`` / ``wedged`` / ``stale``;
       no record at all reads ``stopped``, which is the common, honest case.
    2. **Durable session directory** — exists? title, creation date, archived
       flag; the subagent roster sidecar and the newest ``todo_snapshot`` row.
    3. **In-process overlay** (``live``, TUI only) — narrow per-session overrides
       merged over the computed row for the calling process's own session, whose
       in-memory values are fresher than disk.

    Work links get rows as before; a coordination link ("filed by") ALSO gets
    one, tagged ``role="coordination"`` and stripped of every liveness fact —
    no ``runtime``, ``subagents`` or ``todos`` key at all — so no renderer can
    misread a filing as a working session and every existing reader of
    ``sessions`` stays correct without a filter of its own.

    ``None`` means unknown and is never rendered as 0: a session that never
    launched a subagent has ``subagents: null``, and a session that never
    persisted a todo snapshot has ``todos: null``.

    ``records`` is :func:`scan_runtime_states`'s answer for a caller composing
    several projects from ONE scan (the desktop listing); ``None`` scans here.
    """
    from local_operator.resume import read_title_state
    from local_operator.session.archived import archived_ids
    from local_operator.session.creation import session_created_at

    root = Path(config_dir)
    sessions_root = root / "sessions"
    states = scan_runtime_states(root) if records is None else records
    window = stale_after_s(root)

    try:
        archived = archived_ids(root)
    except Exception:  # noqa: BLE001
        archived = frozenset()

    rows: list[dict[str, Any]] = []
    for session_id in project.sessions:
        session_dir = sessions_root / session_id
        exists = session_dir.is_dir()
        title: str | None = None
        created_at: float | None = None
        if exists:
            state = read_title_state(session_dir)
            title = state.text if state is not None else None
            created_at = session_created_at(session_dir) or None

        runtime = states.get(session_id) or {
            "state": "stopped",
            "busy": None,
            "heartbeat_age_s": None,
            "pid": None,
        }

        row: dict[str, Any] = {
            "session_id": session_id,
            "role": "work",
            "exists": exists,
            "title": title,
            "created_at": created_at,
            "archived": session_id in archived,
            "runtime": runtime,
            "subagents": _load_subagent_summary(session_dir) if exists else None,
            "todos": _load_todo_summary(session_dir) if exists else None,
        }
        if live and session_id in live:
            # The overlay is shallow and per field: the caller owns fresher
            # values for the fields it passes and nothing else.
            row.update(live[session_id])
        rows.append(row)

    # Coordination links ("filed by") get a row too, tagged and stripped of
    # every liveness fact — no runtime, no subagents, no todos — so nothing
    # exists on the row for a renderer to misread as a working session. The
    # provenance fields a detail list can honestly show stay.
    for session_id in project.coordination_sessions:
        session_dir = sessions_root / session_id
        exists = session_dir.is_dir()
        title: str | None = None
        created_at: float | None = None
        if exists:
            state = read_title_state(session_dir)
            title = state.text if state is not None else None
            created_at = session_created_at(session_dir) or None
        rows.append(
            {
                "session_id": session_id,
                "role": "coordination",
                "exists": exists,
                "title": title,
                "created_at": created_at,
                "archived": session_id in archived,
            }
        )

    return {
        "project": project.model_dump(mode="json", by_alias=True),
        "progress_stale": progress_is_stale(project, window=window),
        "sessions": rows,
    }


def readable_error(exc: Exception) -> str:
    """A one-line, human-readable rendering of a validation failure.

    ``ValidationError`` stringifies to a multi-line report with the input value
    repeated; both consumers (the ``project`` tool's result and the desktop
    route's 422 body) carry prose a model or a user reads, so the first error's
    message is quoted and the rest are dropped. Shared HERE rather than
    duplicated per surface: a caller must not be able to tell which one
    answered it.
    """
    if isinstance(exc, KeyError):
        # ``str(KeyError("x"))`` quotes its message; the sentence inside is the
        # one a caller wrote, so it is quoted out here.
        return str(exc.args[0]) if exc.args else str(exc)
    if isinstance(exc, ValidationError):
        problems = exc.errors()
        if problems:
            first = problems[0]
            location = ".".join(str(part) for part in first.get("loc", ()))
            message = str(first.get("msg", "invalid value"))
            return f"{location}: {message}" if location else message
    return str(exc)


# ---------------------------------------------------------------------------
# The schema-1 -> 2 coordination migration (run from the startup seam)
# ---------------------------------------------------------------------------
#
# WHY this exists at all: the chief of staff's create-time auto-link wrote her
# session id into ``sessions`` — the WORK set — so 30 rows on the operator's
# store count her as a worker, satisfy liveness, and earn her session a
# completion-check nudge for projects she filed but does not work on. Schema 2
# splits the two meanings; this migration moves the existing links to the side
# they always meant, PRESERVATIVELY (re-kind, never delete): a row where she is
# the actual owner, the sole link, or the author of every recorded entry keeps
# her as a working link.


def _coordination_keep_rule(
    project: Project, *, cos_session_id: str, cos_display_name: str
) -> str | None:
    """The KEEP clause that protects a row from demotion, or ``None`` to demote.

    The three clauses, in the ruling's order:

    1. ``owner`` names her — casefold against her CURRENT display name. A
       rename since the row was written breaks the match; the dry run shows
       it, and a wrong demotion is one ``op='link'`` from restored.
    2. She is the SOLE linked session — the row is hers (nothing else can
       work on it, so demoting her would leave it unowned).
    3. She authored EVERY non-empty ``by`` entry on the row (at least one) —
       the record is hers even where other session files exist.
    """
    owner = (project.owner or "").strip().casefold()
    if cos_display_name and owner and owner == cos_display_name.strip().casefold():
        return "owner"
    if len(project.sessions) == 1:
        return "sole_link"
    authors = [(entry.by or "").strip() for entry in project.updates if (entry.by or "").strip()]
    if authors and all(author == cos_session_id for author in authors):
        return "authored_all"
    return None


def migrate_coordination_links(
    config_dir: Path | str,
    *,
    dry_run: bool = False,
    cos_session_id: str | None = None,
    cos_display_name: str | None = None,
) -> list[dict[str, Any]]:
    """Re-kind the chief of staff's create-time auto-links. Returns the plan.

    Selector: every SCHEMA-1 row whose WORK links include her session
    (:func:`local_operator.aida.state.session_id_of`). Decision per row:
    :func:`_coordination_keep_rule` or demote (move her id
    ``sessions`` -> ``coordination_sessions``, written ``schema=2``). Kept rows
    are NOT rewritten — "a wrong keep just leaves one row as today" — so they
    still match the selector on a later run and are re-decided (to keep) at
    the cost of one read each, writing nothing. That no-op IS the gate, exactly
    the config-migrations doctrine's shape: no stamp file, an idempotent
    predicate, backup-first, abort-if-no-backup.

    ``dry_run=True`` returns the full per-row plan (``id``, ``name``,
    ``decision`` in ``demote|keep``, the deciding ``rule``, and the planned
    ``sessions``/``coordination_sessions`` lists) and writes NOTHING — no lock,
    no backup dir, no row. Apply takes the store's own lock, re-derives the
    targets from a fresh read under it, backs every row it will rewrite into
    ``projects/.migrations-backup-<stamp>/`` BEFORE any rewrite, and aborts
    the whole run if any backup cannot be written (a later launch retries).
    ``cos_session_id``/``cos_display_name`` are injection points for tests and
    for the seam; both default to the live aida state/config.
    """
    root = Path(config_dir)
    if cos_session_id is None:
        from local_operator.aida.state import session_id_of

        cos_session_id = session_id_of(root)
    if not cos_session_id:
        # No chief of staff on this install (or no state yet): nothing can
        # match the selector, and the caller needs no error for that.
        return []
    if cos_display_name is None:
        from local_operator.aida.naming import display_name as aida_display_name

        cos_display_name = aida_display_name(root)

    registry = ProjectRegistry(root)

    def plan_row(project: Project) -> dict[str, Any] | None:
        if project.schema_version != 1 or cos_session_id not in project.sessions:
            return None
        rule = _coordination_keep_rule(
            project, cos_session_id=cos_session_id, cos_display_name=cos_display_name
        )
        if rule is not None:
            sessions_after = list(project.sessions)
            coordination_after = list(project.coordination_sessions)
            decision = "keep"
        else:
            sessions_after = [s for s in project.sessions if s != cos_session_id]
            coordination_after = [*project.coordination_sessions, cos_session_id]
            decision = "demote"
            rule = "no_keep_clause"
        return {
            "id": project.id,
            "name": project.name,
            "decision": decision,
            "rule": rule,
            "sessions": sessions_after,
            "coordination_sessions": coordination_after,
        }

    plan = [entry for project in registry.list_projects() if (entry := plan_row(project))]
    if dry_run:
        return plan
    if not any(entry["decision"] == "demote" for entry in plan):
        return plan

    with registry._persistence_lock():
        registry._load()
        # Re-derive under the lock: the pre-lock scan chose the backup-dir
        # decision only; the rows actually rewritten are TODAY's, from this
        # read, with the same clauses applied (a frame that moved in between
        # must not be re-kinded on a stale decision).
        targets = [
            project
            for project in registry._projects.values()
            if project.schema_version == 1
            and cos_session_id in project.sessions
            and _coordination_keep_rule(
                project, cos_session_id=cos_session_id, cos_display_name=cos_display_name
            )
            is None
        ]
        if not targets:
            return plan
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_dir = registry.projects_dir / f".migrations-backup-{stamp}"
        try:
            backup_dir.mkdir(parents=True, exist_ok=False)
            for project in targets:
                source = registry.projects_dir / f"{project.id}.json"
                (backup_dir / f"{project.id}.json").write_bytes(source.read_bytes())
        except OSError as exc:
            # Abort-if-no-backup: the rows are untouched and the next launch
            # retries (nothing records the attempt as done). The partial dir
            # is removed — it guards bytes that no rewrite ever replaced.
            logger.warning(
                "project migration: could not back up every row to %s (%s); "
                "no row rewritten, retrying at the next launch",
                backup_dir,
                exc,
            )
            shutil.rmtree(backup_dir, ignore_errors=True)
            return plan
        for project in targets:
            candidate = project.model_copy(deep=True)
            candidate.sessions = [s for s in candidate.sessions if s != cos_session_id]
            candidate.coordination_sessions = [*candidate.coordination_sessions, cos_session_id]
            # The re-kinded row carries the new field, so its schema must say
            # so or an older build would be free to drop it again.
            candidate.schema_version = PROJECT_SCHEMA
            # Deliberately NOT moving ``updated_at``: this is a provenance
            # correction, not operator activity, and the board sorts on it.
            registry._save_project_locked(candidate)
        logger.warning(
            "project migration: re-kinded %d chief-of-staff link%s into "
            "coordination (backup at %s)",
            len(targets),
            "" if len(targets) == 1 else "s",
            backup_dir,
        )
    return plan
