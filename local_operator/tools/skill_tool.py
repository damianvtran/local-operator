"""The ``skill`` tool: let a CONFIGURATION RUN revise the operator's own skills.

WHO GETS IT. Only a session whose declaration names it and whose host merges it
in -- today the Agents page's configuration run (``session_factory``'s
``AGENTS_CONFIG_TOOLS`` + ``apply_config_run_shape``). It is in ``TOOL_BUILDERS``
but deliberately NOT in ``DEFAULT_TOOL_NAMES``, so no ordinary session builds it
and no other session's provider-visible tool array (the prompt-cache prefix)
moves. That is rung 3 of the AGENTS.md footprint ladder: opt-in, not host
identity. The run has no file tools by design (its declaration is its boundary),
and handing it ``read``/``write``/``edit`` to change one skill would hand it the
whole disk; this tool reaches ``<config_dir>/skills`` and nothing else.

ONE WRITE PATH. Every write goes through ``server/utils/desktop_skill_writes``,
the same module the Skills tab's routes use, so the rules are stated once:
built-in, external (ecosystem) and project skills are read-only
(``skill-not-editable``); bundle-form and unpulled hub links are read-only; a
local edit of a hub-linked skill leaves the link's baseline and policy untouched
(the link check then reports ``local-only``); paths, sizes, symlinks and binaries
are refused by the storage validators, never by a second set here. Two rules are
the run's own, because the run is a model acting on a typed request rather than
the operator clicking:

* ``update`` is a PATCH (``desktop_skill_writes.patch``): the named files change,
  ``remove_paths`` go, every other file is kept. The route's whole-set replace
  would silently delete reference files a model forgot to resend.
* ``create`` refuses a name a read-only root already supplies
  (``require_unshadowed``): a same-named local copy would replace a built-in for
  every later session. "Copy to my skills" stays a click the operator makes.

NO DELETE, PUBLISH, PULL OR SUBSCRIBE. The ops are the agent/team verbs
(list/show/create/update), so the desktop's run-activity projection already
reads list/show as reads and create/update as writes. Every committed edit
leaves the previous bytes in ``<config_dir>/hub/backups``.

``show`` returns ``SKILL.md`` WHOLE (no spill): the run has no ``read`` tool to
expand a ``spill://`` handle, and a model that saw only the head of a file and
then rewrote it would truncate it. The bound is the resolver's own 200 KiB
(``skills.protocol.MAX_READ_BYTES``), which is also the per-file storage limit.

The server import is LAZY (inside ``execute``): ``local_operator.tools`` imports
nothing from ``local_operator.server`` at module load, the same discipline
``agent_tool`` keeps for ``hub_sync.service``.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from local_operator.harness.types import (
    AbortSignal,
    AgentTool,
    AgentToolUpdate,
    ToolContext,
    ToolResult,
)
from local_operator.tools.builtin import (
    _error,
    _guard,
    _text,
    _validation_error,
    identifier_text,
)

_TOOL = "skill"

#: The manual path a read-only refusal names. Model-facing, so English by design
#: (AGENTS.md i18n: model-facing prompts are exempt); the words match the Skills
#: tab's button so the operator recognises what the run points them to.
_COPY_HINT = (
    "Ask the operator to use Copy to my skills on the Skills tab first; do not "
    "create a same-named copy yourself."
)


class SkillFileArg(BaseModel):
    """One file of a skill. Bounds mirror the route's ``SkillFileIn``.

    The REAL limits (200 KiB per file, 1 MiB per skill, in bytes; allowed paths)
    are enforced by ``SkillDocument``; these only refuse an absurd argument before
    it is decoded.
    """

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=240, description="e.g. SKILL.md, references/x.md")
    content: str = Field(max_length=1024 * 1024, description="The file's full new text.")


class SkillParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "create", "update"] = Field(
        description=(
            "list: every skill, whether you may edit it, and its fingerprint; "
            "show: a skill's SKILL.md and file list (or one file via path); "
            "create: a new skill of your own; update: change some files of one "
            "you may edit."
        )
    )
    name: str | None = Field(
        default=None, min_length=1, max_length=128, description="Skill name (all ops but list)."
    )
    path: str | None = Field(
        default=None,
        min_length=1,
        max_length=240,
        description="show only: return this one file whole instead of SKILL.md.",
    )
    files: list[SkillFileArg] | None = Field(
        default=None,
        max_length=64,
        description=(
            "create: every file, SKILL.md required. update: ONLY the files you "
            "change or add; the others are kept."
        ),
    )
    remove_paths: list[str] | None = Field(
        default=None,
        max_length=64,
        description="update only: files to delete from the skill.",
    )
    expected_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
        description="update only, required: the fingerprint your last show returned.",
    )


def _config_dir(context: ToolContext | None) -> Path | None:
    value = getattr(context, "memory_config_dir", None)
    return Path(value) if value else None


def _catalogue(config_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(rows, discovered-by-name)`` the way the Skills tab's list builds them.

    The root set is ``_discovered``'s (home-derived), so "editable" here agrees
    with what ``require_managed`` and ``require_unshadowed`` decide on write --
    plus THIS profile's managed root when the home walk does not reach it. With
    ``LOCAL_OPERATOR_CONFIG_DIR`` moved, ``<home>/.local-operator/skills`` is
    another profile's store and ``<config_dir>/skills`` is scanned by nothing, so
    the run could ``update`` a skill it could not ``show`` (and so never learn its
    fingerprint). It goes where the home native root sits: after the project
    roots, ahead of the ecosystem and packaged ones, the precedence it has in the
    default layout.
    """
    from local_operator.server.utils import desktop_skills
    from local_operator.skills import default_skill_roots, discover_skills
    from local_operator.skills.api import ecosystem_skill_roots

    roots = default_skill_roots(Path.home())
    managed = Path(config_dir) / "skills"
    if os.path.realpath(managed) not in {os.path.realpath(root) for root in roots}:
        home_store = os.path.realpath(Path.home() / ".local-operator" / "skills")
        at = next(
            (i for i, root in enumerate(roots) if os.path.realpath(root) == home_store),
            len(roots) - 1,
        )
        roots.insert(at, managed)
    discovered, _warnings = discover_skills(roots, config_dir=config_dir)
    links = desktop_skills.link_index(config_dir)
    classifier = desktop_skills.SourceClassifier(config_dir, ecosystem_skill_roots(), links)
    rows = desktop_skills.build_rows(discovered, classifier, links)
    return rows, {skill.name: skill for skill in discovered}


def _why_read_only(row: dict[str, Any]) -> str:
    source = row.get("source")
    if source == "hub" and not row.get("installed", True):
        return "a hub subscription with no local copy"
    if source == "hub":
        return "a hub skill in bundle form"
    return {
        "builtin": "built-in",
        "external": "installed by another tool",
        "project": "a project's own skill",
    }.get(str(source), str(source))


def _list(config_dir: Path, context: ToolContext | None) -> str:
    rows, _ = _catalogue(config_dir)
    if not rows:
        return "no skills found. Create one with op='create'."
    lines = []
    for row in sorted(rows, key=lambda item: item["name"]):
        mark = "editable" if row["editable"] else f"read-only ({_why_read_only(row)})"
        description = " ".join(str(row.get("description") or "").split())
        if len(description) > 120:
            description = description[:119] + "…"
        lines.append(
            f"- {identifier_text(context, row['name'])} [{row['source']}, {mark}] {description}"
        )
    return (
        "skills (show a skill before you update it):\n"
        + "\n".join(lines)
        + "\n\nread-only skills cannot be changed here. "
        + _COPY_HINT
    )


def _show(config_dir: Path, name: str, path: str | None, context: ToolContext | None) -> str:
    from local_operator.hub_sync.identity import LinkError
    from local_operator.hub_sync.skill_document import SkillDocument, validate_path
    from local_operator.hub_sync.skill_storage import managed_skill_read
    from local_operator.skills.protocol import MAX_READ_BYTES, _read_text_capped

    rows, by_name = _catalogue(config_dir)
    row = next((item for item in rows if item["name"] == name), None)
    skill = by_name.get(name)
    if row is None:
        raise LinkError("skill-not-found", "no such skill")
    if skill is None:
        raise LinkError("skill-not-editable", "this skill has no local copy to show")
    # A managed folder is read under the root lock so a concurrent swap is never
    # seen half-done; any other root is not ours to lock.
    base = Path(os.path.realpath(skill.base_dir))
    with managed_skill_read(config_dir):
        try:
            document: Any = SkillDocument.from_directory(base)
        except Exception:  # noqa: BLE001 - not a text document: show SKILL.md only
            document = None
        skill_md = (
            next(file.content for file in document.files if file.path == "SKILL.md")
            if document is not None
            else _read_text_capped(Path(skill.file_path))
        )
    header = [
        f"skill {identifier_text(context, name)}",
        f"source: {row['source']}",
        (
            "editable: yes"
            if row["editable"]
            else f"editable: no ({_why_read_only(row)}). {_COPY_HINT}"
        ),
    ]
    if row["editable"] and document is not None:
        header.append(f"fingerprint: {document.fingerprint}")
    if path is not None:
        validate_path(path)
        if document is None:
            raise LinkError("skill-not-editable", "this skill's files are not readable as text")
        found = next((file for file in document.files if file.path == path), None)
        if found is None:
            raise LinkError("skill-not-found", f"no file {path!r} in this skill")
        return "\n".join(header) + f"\n\n--- {path} ---\n{found.content}"
    others: list[str] = []
    if document is not None:
        others = [
            f"- {file.path} ({len(file.content.encode('utf-8'))} bytes)"
            for file in document.files
            if file.path != "SKILL.md"
        ]
    body = "\n".join(header) + f"\n\n--- SKILL.md ---\n{skill_md}"
    if len(skill_md.encode("utf-8")) > MAX_READ_BYTES:
        body += "\n[SKILL.md is longer than this view]"
    if others:
        body += "\n\nother files (show with path=... to read one):\n" + "\n".join(others)
    return body


def _write_receipt(result: dict[str, Any], verb: str, context: ToolContext | None) -> str:
    name = identifier_text(context, result["name"])
    if not result.get("changed", True):
        return f"skill {name} already had this content; nothing changed."
    lines = [f"{verb} skill {name}.", f"fingerprint: {result['fingerprint']}"]
    if result.get("source") == "hub":
        lines.append(
            "it is linked to the hub: this is a local change and was not published; "
            "publishing is the operator's choice."
        )
    elif verb == "updated":
        lines.append("the previous version is kept in the profile's hub/backups folder.")
    return "\n".join(lines)


_HINTS: dict[str, str] = {
    "skill-not-editable": _COPY_HINT,
    "local-concurrent-edit": (
        "The skill changed since you read it (or your fingerprint is not the one "
        "show returned). Show it again and redo the edit on the current text."
    ),
    "alias-occupied": "Choose another name, or update the existing skill if it is editable.",
    "skill-not-found": "Check the name with op='list'.",
}


def _work(params: SkillParams, config_dir: Path, context: ToolContext | None) -> str:
    """The blocking body (filesystem, fsync, root lock), run off the event loop."""
    from local_operator.server.utils import desktop_skill_writes as writes

    name = str(params.name)
    files = [item.model_dump() for item in params.files or ()]
    if params.op == "list":
        return _list(config_dir, context)
    if params.op == "show":
        return _show(config_dir, name, params.path, context)
    if params.op == "create":
        writes.require_unshadowed(config_dir, name)
        return _write_receipt(writes.create(config_dir, name, files=files), "created", context)
    result = writes.patch(
        config_dir,
        name,
        files,
        params.remove_paths or (),
        str(params.expected_fingerprint),
    )
    return _write_receipt(result, "updated", context)


def _arguments_refusal(params: SkillParams) -> str | None:
    """Cross-field rules the schema cannot state; ``None`` when the call is well-formed."""
    if params.op != "list" and not params.name:
        return f"op={params.op!r} needs 'name'."
    if params.path is not None and params.op != "show":
        return "'path' is for op='show' only."
    if params.op in ("list", "show") and (
        params.files or params.remove_paths or params.expected_fingerprint
    ):
        return f"op={params.op!r} takes no files, remove_paths or expected_fingerprint."
    if params.op == "create":
        if not params.files:
            return "op='create' needs 'files' with at least SKILL.md."
        if params.remove_paths or params.expected_fingerprint:
            return "op='create' takes no remove_paths or expected_fingerprint."
    if params.op == "update":
        if not params.expected_fingerprint:
            return (
                "op='update' needs 'expected_fingerprint': show the skill first and pass "
                "the fingerprint it returned."
            )
        if not params.files and not params.remove_paths:
            return "op='update' needs 'files' to change or 'remove_paths' to delete."
    return None


@_guard(_TOOL)
async def execute_skill(
    tool_call_id: str,
    args: dict[str, Any],
    signal: AbortSignal | None = None,
    on_update: Callable[[AgentToolUpdate], None] | None = None,
    context: ToolContext | None = None,
) -> ToolResult:
    """List, read, create and revise the operator's own skills."""
    from local_operator.hub_sync.identity import LinkError
    from local_operator.hub_sync.skill_document import SkillValidationError
    from local_operator.hub_sync.transactions import TransactionError

    try:
        params = SkillParams(**args)
    except ValidationError as exc:
        return _validation_error(tool_call_id, _TOOL, exc)
    refusal = _arguments_refusal(params)
    if refusal is not None:
        return _error(tool_call_id, _TOOL, refusal)
    config_dir = _config_dir(context)
    if config_dir is None:
        return _error(tool_call_id, _TOOL, "no profile is attached to this session.")
    try:
        text = await asyncio.to_thread(_work, params, config_dir, context)
    except (LinkError, SkillValidationError, TransactionError) as error:
        code = getattr(error, "code", "") or ""
        hint = _HINTS.get(code)
        message = f"{code}: {error}" + (f". {hint}" if hint else "")
        return _error(tool_call_id, _TOOL, message, details={"code": code})
    return _text(tool_call_id, _TOOL, text)


def build_skill_tool(context: ToolContext) -> AgentTool | None:
    """createIf: the tool needs the profile root its writes land under.

    Reaching a session at all is the declaration's job, not this factory's: the
    tool is absent from ``DEFAULT_TOOL_NAMES``, so only a host that names it (the
    configuration run) builds it.
    """
    if _config_dir(context) is None:
        return None
    return AgentTool(
        name=_TOOL,
        label="Skills",
        description=(
            "The operator's skills: list them, show one, create a new one, or "
            "update one marked editable. Show before you update and pass the "
            "fingerprint it returned; send only the files you change."
        ),
        parameters=SkillParams.model_json_schema(),
        # The same tier as ``agent``/``team`` in this run: the operator's typed
        # request on the page that watches the run is the authority, and every
        # committed edit keeps the previous bytes in hub/backups.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_skill,
    )
