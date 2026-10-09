"""``/api/projects*`` — the phone's read-write surface over the project store.

WHAT THIS IS. The daemon half of the projects primitive: the same
:class:`~local_operator.projects.ProjectRegistry` operations the ``project``
tool and the desktop routes call, carried to the phone's own REST conventions.
It deliberately mirrors ``local_operator/server/routes/desktop_projects.py``
route for route — list, create, detail, patch, delete, links, milestones — so
the two HTTP surfaces cannot drift apart in behaviour:

* **Key resolution** is the desktop's rule: an exact project id first, then a
  case-insensitive name. Ids are ``uuid4().hex`` and names are grammar-limited,
  so a collision needs a project literally named as another's hex id — the id
  arm wins, documented rather than left to chance.
* **The store is the only writer.** No function here writes a file directly;
  every mutation goes through the registry's locked read-modify-write, so the
  tool, the desktop and the phone cannot validate differently.
* **One composition.** The composed detail view comes from
  :func:`~local_operator.projects.build_project_view` and the wire shapes come
  from :mod:`local_operator.server.models.desktop_projects` — the very models
  the desktop routes serve. There is no second derivation of "what are the
  linked sessions doing", of a milestone's derived status, or of the staleness
  verdict, so a card on the phone and a row in the desktop app cannot disagree
  about one record.
* **Refusals** map to the desktop's status codes and machine codes:
  ``404 project_not_found`` (with up to two prefix-matched names as a remedy),
  ``409 project_name_exists`` / ``409 project_schema_newer``,
  ``422 project_invalid`` / ``422 project_confirm_mismatch`` /
  ``422 project_done_incomplete`` (the ``done`` gate: open milestones; the way
  through is ``force_done: true`` on the PATCH), and
  ``503 project_store_busy`` for a lock timeout — retryable, not malformed.

TRANSPORT DIFFERENCES, and only these. The daemon's handlers speak flat JSON
bodies rather than the desktop ``CRUDResponse`` envelope, and this module
raises :class:`ProjectRouteError` instead of ``HTTPException`` so it stays
importable without FastAPI's request stack. A body that parses but is not a
JSON OBJECT gets the daemon's own body-shape refusal, ``400`` — the desktop's
model validation answers that same body ``422``, and the daemon's five
pre-existing handlers carry the identical 400/422 split. The *payloads* are
the desktop wire models, dumped to JSON — field-for-field what the desktop
serves, which is what lets the phone render the same summary/view fields
without a translation layer; the request-side vocabulary is the desktop's too,
derived from its request models (see ``_CREATE_FIELDS`` below).

THREADING. Every function here is synchronous and touches the filesystem (and,
for the live-session counts, the runtime record directory), so the daemon runs
them through ``asyncio.to_thread`` — the same off-loop shape the desktop routes
use for the identical calls. Nothing here dials a runtime or mutates a session
directory; the linked-session rows are read from files exactly as the tool's
view is.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from local_operator.projects import (
    MilestoneEdit,
    Project,
    ProjectDoneGateError,
    ProjectEdit,
    ProjectNameConflictError,
    ProjectRegistry,
    ProjectRegistryLockTimeout,
    ProjectSchemaGuardError,
    build_project_view,
    closed_with_open_milestones,
    readable_error,
    scan_runtime_states,
    stale_after_s,
)
from local_operator.server.models.desktop_projects import (
    STATUS_RANK,
    LinkMutation,
    MilestoneMutation,
    ProjectCreate,
    ProjectDelete,
    ProjectPatch,
    linked_session_view,
    project_patched,
    project_summary,
    project_view,
)

#: The keys each write body may carry, DERIVED from the desktop request models
#: (round-1 review, [m]3): a literal copy of these five tuples drifted silently
#: — the parity test compared response shapes only — so adding a field to a
#: desktop model now widens the phone's gate in the same commit. These tuples
#: answer "is this key part of the vocabulary"; values still validate in the
#: store, whose refusals are the same sentences the desktop receives.
_CREATE_FIELDS = tuple(ProjectCreate.model_fields)
_DELETE_FIELDS = tuple(ProjectDelete.model_fields)
_LINK_FIELDS = tuple(LinkMutation.model_fields)
_PATCH_FIELDS = tuple(ProjectPatch.model_fields)
_MILESTONE_FIELDS = tuple(MilestoneMutation.model_fields)


class ProjectRouteError(Exception):
    """One refusal, in the shape the daemon's JSON error body carries.

    ``status`` and ``code`` are the desktop route's own numbers and machine
    codes (see the module docstring); ``message`` is written for the reader,
    the way every daemon refusal is.

    ``extra`` is the ADDITIVE carriage for machine-readable fields beyond
    ``error``/``code`` (today only ``incomplete`` on ``project_done_incomplete``,
    the milestone names a confirm sheet lists). The daemon merges it into the
    body next to ``error`` and ``code``, so an existing client that reads only
    those two keys is unaffected; it defaults empty so every other refusal is
    byte-identical to before.
    """

    def __init__(
        self, status: int, code: str, message: str, extra: dict[str, Any] | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.extra: dict[str, Any] = dict(extra or {})


def _registry(config_dir: Path) -> ProjectRegistry:
    """A fresh adapter over ``<config_dir>/projects`` for one operation.

    Fresh metadata per call (the ``desktop_profiles`` convention): another
    process's write is visible without waiting out a polling interval. This is
    an adapter over the same directory the tool writes, not a second registry.
    """
    return ProjectRegistry(config_dir)


def _find(registry: ProjectRegistry, key: str) -> Project | None:
    """Resolve a key: exact id first, then a case-insensitive name."""
    try:
        return registry.get_project(key)
    except (KeyError, ValueError):
        # Not an existing row id — either the id is unknown or the key is not
        # id-shaped at all. Either way the name arm is the only one left.
        return registry.get_project_by_name(key)


def _not_found(registry: ProjectRegistry, key: str) -> ProjectRouteError:
    """The 404, with up to two prefix-matched names as the remedy.

    "check the projects list" tells a user nothing about a typo; naming the
    nearest names is the difference between a dead end and a fix. Offered only
    when something actually shares a leading fragment — an unrelated list of
    names would be noise.
    """
    prefix = key.strip().casefold()[:4]
    near = (
        [
            project.name
            for project in registry.list_projects()
            if prefix and project.name.casefold().startswith(prefix)
        ][:2]
        if prefix
        else []
    )
    names = f" — closest: {', '.join(near)}" if near else ""
    return ProjectRouteError(404, "project_not_found", f"no project with id or name {key!r}{names}")


def _reader_error(exc: Exception) -> str:
    """``readable_error``, minus pydantic's wrapper, for the reader's eyes.

    A store validator raising ``ValueError`` reaches :func:`readable_error` as
    ``<field>: Value error, <the store's sentence>``; the wrapper is machinery
    no reader should meet first, and every store sentence names its own subject
    ("project name must be…", "estimate must be…"), so the field prefix goes
    with it. Anything not in that exact shape passes through untouched — the
    wrapper is all this removes (round-1 design, D4).
    """
    message = readable_error(exc)
    _, marker, sentence = message.partition(": Value error, ")
    return sentence if marker else message


def _refusal(exc: Exception) -> ProjectRouteError:
    """Map one store refusal to the refusal the client can act on."""
    if isinstance(exc, ProjectNameConflictError):
        # The name is taken (case-insensitive): a STATE conflict, not a
        # malformed value — the client's move is to pick another name or open
        # the existing project, which 409 (with a machine code) says and 422
        # does not.
        return ProjectRouteError(409, "project_name_exists", str(exc))
    if isinstance(exc, ProjectSchemaGuardError):
        # The row EXISTS and is readable; this build is too old to rewrite it.
        # 409, not 404 (the operator can see it) and not 422 (nothing about the
        # request is malformed): the state of the target is what refuses.
        return ProjectRouteError(409, "project_schema_newer", str(exc))
    if isinstance(exc, ProjectRegistryLockTimeout):
        # Contention is transient; the client retries.
        return ProjectRouteError(503, "project_store_busy", str(exc))
    if isinstance(exc, ProjectDoneGateError):
        # The desktop route's mapping, field for field: 422 with its own code
        # because the remedy (complete the milestones, or resend with
        # `force_done: true`) is a choice the phone can offer only if it can
        # tell this from a malformed value. The sentence is the store's own;
        # the names ride `extra` so the body stays `{error, code, incomplete}`.
        return ProjectRouteError(
            422,
            "project_done_incomplete",
            _reader_error(exc),
            {"incomplete": list(exc.incomplete)},
        )
    return ProjectRouteError(422, "project_invalid", _reader_error(exc))


#: What one store mutation can raise and this module answers as a refusal.
#: ``ProjectSchemaGuardError`` is named last but explicitly because it
#: subclasses ``RuntimeError`` — anything catching only the value errors would
#: let the guard escape as a traceback.
_STORE_REFUSALS = (
    KeyError,
    ValueError,
    ValidationError,
    ProjectSchemaGuardError,
    ProjectRegistryLockTimeout,
)


def _summary_payload(project: Project, *, live_sessions: int, window: float) -> dict[str, Any]:
    """One wire summary, as JSON — the same model the desktop route serves.

    ``window`` is the staleness window resolved once per request
    (:func:`local_operator.projects.stale_after_s`), so every row of a
    listing is measured against the configured boundary.
    """
    return project_summary(project, live_sessions=live_sessions, window=window).model_dump(
        mode="json"
    )


def _live_counts(registry: ProjectRegistry, projects: list[Project]) -> dict[str, int]:
    """Live-session counts for a batch of projects, from ONE runtime scan."""
    if not projects:
        return {}
    states = scan_runtime_states(registry.config_dir)
    return {
        project.id: sum(1 for sid in project.sessions if states.get(sid, {}).get("state") == "live")
        for project in projects
    }


def _selected(body: dict[str, Any], allowed: tuple[str, ...]) -> dict[str, Any]:
    """The keys ``body`` actually carries from ``allowed``; anything else is refused.

    ``ProjectEdit``/``MilestoneEdit`` read ``model_fields_set`` so absent keys
    stay untouched — which only works if the dict this builds contains exactly
    the keys the caller sent. An unknown key is a 422, matching the desktop
    models' ``extra="forbid"``.
    """
    unknown = sorted(set(body) - set(allowed))
    if unknown:
        raise ProjectRouteError(
            422,
            "project_invalid",
            f"unknown field(s): {', '.join(unknown)}",
        )
    return {key: body[key] for key in allowed if key in body}


def list_payload(config_dir: Path) -> dict[str, Any]:
    """``GET /api/projects`` — the listing (summaries), board order."""
    registry = _registry(config_dir)
    listed = registry.list_projects()
    live = _live_counts(registry, listed)
    rows = [
        project_summary(
            project,
            live_sessions=live.get(project.id, 0),
            window=stale_after_s(registry.config_dir),
        )
        for project in listed
    ]
    rows.sort(key=lambda row: (STATUS_RANK.get(row.status, 99), -row.updated_at))
    return {"projects": [row.model_dump(mode="json") for row in rows]}


def detail_payload(config_dir: Path, key: str) -> dict[str, Any]:
    """``GET /api/projects/{key}`` — one project plus its linked sessions."""
    registry = _registry(config_dir)
    found = _find(registry, key)
    if found is None:
        raise _not_found(registry, key)
    view = build_project_view(found, config_dir=registry.config_dir)
    return {
        "project": project_view(found, window=stale_after_s(registry.config_dir)).model_dump(
            mode="json"
        ),
        "links": [linked_session_view(row).model_dump(mode="json") for row in view["sessions"]],
    }


def create_payload(config_dir: Path, body: dict[str, Any]) -> dict[str, Any]:
    """``POST /api/projects`` — create one row; the name must be free."""
    payload = _selected(body, _CREATE_FIELDS)
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise ProjectRouteError(422, "project_invalid", "name is required")
    registry = _registry(config_dir)
    try:
        # Construction inside the try: the edit model runs the same validators
        # as the store, and its ValidationError must reach this call's 422.
        fields = ProjectEdit(**payload)
        created = registry.create_project(fields)
    except _STORE_REFUSALS as exc:
        raise _refusal(exc) from exc
    # A created project starts unlinked (the desktop create body carries no
    # sessions either), so the live count is 0 without a scan.
    return {
        "ok": True,
        "project": _summary_payload(
            created, live_sessions=0, window=stale_after_s(registry.config_dir)
        ),
    }


def patch_payload(config_dir: Path, key: str, body: dict[str, Any]) -> dict[str, Any]:
    """``PATCH /api/projects/{key}`` — a partial edit; omitted keys are untouched."""
    payload = _selected(body, _PATCH_FIELDS)
    # `force_done` is a flag about THIS call, not a field of the row (the
    # desktop route's rule): it never reaches `ProjectEdit` and rides the
    # registry keyword instead. There is no pydantic model on this path, so the
    # desktop's strict-bool rule is restated here — only a real JSON boolean
    # may arm a flag that overrides a safety check; `"yes"`/`1` are a 422.
    force_done = payload.pop("force_done", False)
    if not isinstance(force_done, bool):
        raise ProjectRouteError(422, "project_invalid", "force_done must be true or false")
    registry = _registry(config_dir)
    found = _find(registry, key)
    if found is None:
        raise _not_found(registry, key)
    try:
        fields = ProjectEdit(**payload)
        outcome = registry.update_project(
            found.id, fields, reporter="operator", force_done=force_done
        )
    except _STORE_REFUSALS as exc:
        raise _refusal(exc) from exc
    updated = outcome.project
    live = _live_counts(registry, [updated])
    return {
        "ok": True,
        # The desktop's `ProjectPatched` shape: the summary plus `forced_done`,
        # the only place a forced close is visible (the store records none).
        "project": project_patched(
            updated,
            live_sessions=live.get(updated.id, 0),
            window=stale_after_s(registry.config_dir),
            forced_done=closed_with_open_milestones(updated, force_done=force_done),
        ).model_dump(mode="json"),
    }


def delete_payload(config_dir: Path, key: str, body: dict[str, Any]) -> dict[str, Any]:
    """``DELETE /api/projects/{key}`` — confirm-by-name delete."""
    payload = _selected(body, _DELETE_FIELDS)
    confirm = payload.get("confirm")
    if not isinstance(confirm, str) or not confirm.strip():
        raise ProjectRouteError(422, "project_invalid", "confirm must repeat the project name")
    registry = _registry(config_dir)
    found = _find(registry, key)
    if found is None:
        raise _not_found(registry, key)
    if confirm.strip().casefold() != found.name.casefold():
        # A mismatch is a 422 rather than a silent success: a client that sends
        # the wrong name is not asking for this deletion (the same ladder the
        # desktop route climbs).
        raise ProjectRouteError(
            422,
            "project_confirm_mismatch",
            f"confirm must repeat the project name {found.name!r} exactly (the name, not the id)",
        )
    try:
        registry.delete_project(found.id)
    except _STORE_REFUSALS as exc:
        raise _refusal(exc) from exc
    return {"ok": True, "deleted": True}


def milestone_payload(config_dir: Path, key: str, body: dict[str, Any]) -> dict[str, Any]:
    """``POST /api/projects/{key}/milestones`` — add-or-update one milestone.

    Add-or-update is keyed by name, case-insensitively; ``completed=True``
    stamps today and ``False`` clears it, which is the phone's toggle. The
    store's validator runs on the whole list, so this route and the tool cannot
    disagree about what is legal.
    """
    payload = _selected(body, _MILESTONE_FIELDS)
    if not isinstance(payload.get("name"), str) or not payload["name"].strip():
        raise ProjectRouteError(422, "project_invalid", "name is required")
    registry = _registry(config_dir)
    found = _find(registry, key)
    if found is None:
        raise _not_found(registry, key)
    try:
        fields = MilestoneEdit(**payload)
        project, _action = registry.set_milestone(found.id, fields)
    except _STORE_REFUSALS as exc:
        raise _refusal(exc) from exc
    return {
        "ok": True,
        "project": project_view(project, window=stale_after_s(registry.config_dir)).model_dump(
            mode="json"
        ),
    }


def milestone_remove_payload(config_dir: Path, key: str, name: str) -> dict[str, Any]:
    """``DELETE /api/projects/{key}/milestones/{name}`` — remove one by name."""
    registry = _registry(config_dir)
    found = _find(registry, key)
    if found is None:
        raise _not_found(registry, key)
    try:
        project, _action = registry.set_milestone(found.id, MilestoneEdit(name=name, remove=True))
    except _STORE_REFUSALS as exc:
        raise _refusal(exc) from exc
    return {
        "ok": True,
        "project": project_view(project, window=stale_after_s(registry.config_dir)).model_dump(
            mode="json"
        ),
    }


def link_payload(config_dir: Path, key: str, body: dict[str, Any]) -> dict[str, Any]:
    """``POST /api/projects/{key}/links`` — link one session."""
    payload = _selected(body, _LINK_FIELDS)
    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ProjectRouteError(422, "project_invalid", "session_id is required")
    registry = _registry(config_dir)
    found = _find(registry, key)
    if found is None:
        raise _not_found(registry, key)
    try:
        project, _added = registry.link_session(found.id, session_id)
    except _STORE_REFUSALS as exc:
        raise _refusal(exc) from exc
    live = _live_counts(registry, [project])
    return {
        "ok": True,
        "project": _summary_payload(
            project,
            live_sessions=live.get(project.id, 0),
            window=stale_after_s(registry.config_dir),
        ),
    }


def unlink_payload(config_dir: Path, key: str, session_id: str) -> dict[str, Any]:
    """``DELETE /api/projects/{key}/links/{session_id}`` — unlink one session."""
    registry = _registry(config_dir)
    found = _find(registry, key)
    if found is None:
        raise _not_found(registry, key)
    try:
        project, _removed = registry.unlink_session(found.id, session_id)
    except _STORE_REFUSALS as exc:
        raise _refusal(exc) from exc
    live = _live_counts(registry, [project])
    return {
        "ok": True,
        "project": _summary_payload(
            project,
            live_sessions=live.get(project.id, 0),
            window=stale_after_s(registry.config_dir),
        ),
    }
