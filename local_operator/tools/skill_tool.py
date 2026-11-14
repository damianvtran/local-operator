"""The ``skill`` tool: owner-requested authoring of managed account skills.

WHO GETS IT. Ordinary owner conversations and the Agents configuration run.
The factory gates on the profile root, not the host identity. Every mutation
uses the session's host-held account-write seam: an owner's explicit edit
request applies directly, remote/peer input needs the never-auto-approved card,
and hosts without a consent seam refuse. The tool reaches managed skills only;
no delete/publish/pull/subscribe operation is exposed.

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
(list/show/create/update) plus ``edit``; the desktop's run-activity projection
reads list/show as reads, create/update as writes, and an op it does not know
(``edit``) as a write once it landed. Every committed edit leaves the previous
bytes in ``<config_dir>/hub/backups``.

EVERY RESULT FITS THE REPLAY BOUND, AND NO WRITE IS BUILT FROM UNSEEN TEXT. The
loop cuts each tool result to ``DEFAULT_REPLAY_BOUND_CHARS`` (8 KiB) before the
next request (``harness/replay_bound.py``) and keeps head and tail, so a long
file returned whole reaches the model with its middle missing -- while the
fingerprint line survives, so a rewrite composed from that view would pass the
CAS and commit a truncated skill (review R1). Hence three rules, each pinned by a
test that drives the real ``AgentLoop``:

* ``show`` PAGES a file by whole lines (``offset``), every result held
  under :data:`RESULT_MAX_CHARS`, saying which lines it holds and where the next
  page starts; ``list`` pages its rows the same way.
* ``edit`` changes a file by exact ``old_text`` -> ``new_text`` replacements (the
  file ``edit`` tool's rule: ``old_text`` must occur exactly once), so a change
  needs only the text it replaces and every other byte is carried over.
* ``update`` refuses to replace or remove an EXISTING file longer than one page
  (:data:`PAGE_CHARS`): only a file the model could have read in one ``show`` may
  be replaced whole, and refusing the remove too is what stops a remove-then-re-add
  across two calls doing the same (review R8). New and short files are unaffected.

The replay bound is not the only lossy view: the session masks credential-looking
text in every tool result before the model reads it, while the fingerprint line
passes through. So ``update`` also refuses to replace a file whose masked view
differs from its bytes, and every write refuses text that ADDS the
``[redacted]`` marker (review R6). Both use the session's own mask
(``ToolContext.variables.redact``), the function that made the model's view.

No spill handle is ever returned either: the run has no ``read`` tool to open one.

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

from local_operator.harness.replay_bound import DEFAULT_REPLAY_BOUND_CHARS
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

#: The longest result this tool returns: the replay bound minus headroom for the
#: loop's own framing. Everything a page carries is measured against it.
RESULT_MAX_CHARS = DEFAULT_REPLAY_BOUND_CHARS - 1024
#: File text per ``show`` page, and so the longest existing file ``update`` may
#: replace whole. Fixed rather than "whatever the header leaves" so the answer to
#: "could the model have read this file in one show?" does not depend on which
#: page carried the file listing. Sized so the worst first page -- header
#: (~350), page, notes (~300) and the capped listing -- stays under
#: :data:`RESULT_MAX_CHARS`; ``test_every_result_fits_the_replay_bound`` pins it.
PAGE_CHARS = 4 * 1024
#: The file listing on a SKILL.md first page, capped so header + listing + page
#: stays under :data:`RESULT_MAX_CHARS`.
_LISTING_MAX_CHARS = 1536
#: Per-file bound at the schema: the storage limit (``skill_document``'s 200 KiB,
#: in bytes; a character is at least one byte, so this never refuses what storage
#: would accept). Review R5: 1 MiB x 64 entries was parsed before being refused.
_FILE_MAX_CHARS = 200 * 1024

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
    content: str = Field(max_length=_FILE_MAX_CHARS, description="The file's full new text.")


class SkillEditArg(BaseModel):
    """One exact replacement in one file (``op='edit'``)."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=240, description="The file to change.")
    old_text: str = Field(
        min_length=1,
        max_length=_FILE_MAX_CHARS,
        description="Exact text copied from show; must occur exactly once in the file.",
    )
    new_text: str = Field(max_length=_FILE_MAX_CHARS, description="Its replacement.")


class SkillParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal["list", "show", "create", "update", "edit"] = Field(
        description=(
            "list: skills and whether you may edit them; show: one page of a "
            "skill's file (SKILL.md unless path); create: a new skill; edit: "
            "exact replacements in files of an editable skill; update: replace "
            "or add whole files, or remove files."
        )
    )
    name: str | None = Field(
        default=None, min_length=1, max_length=128, description="Skill name (all ops but list)."
    )
    path: str | None = Field(
        default=None,
        min_length=1,
        max_length=240,
        description="show only: the file to page through instead of SKILL.md.",
    )
    # No ``limit``: a page is always as long as the bound allows, so a file short
    # enough to replace whole is always shown whole by a plain ``show``.
    offset: int | None = Field(
        default=None,
        ge=1,
        description="show: first line (1-based); list: first row. From the previous page.",
    )
    files: list[SkillFileArg] | None = Field(
        default=None,
        max_length=64,
        description=(
            "create: every file, SKILL.md required. update: ONLY the files you "
            "replace or add; the others are kept. A file longer than one show "
            "page cannot be replaced whole: use edit."
        ),
    )
    edits: list[SkillEditArg] | None = Field(
        default=None, max_length=32, description="edit only: applied in order, in one write."
    )
    remove_paths: list[str] | None = Field(
        default=None,
        max_length=64,
        description="update only: files to delete from the skill.",
    )
    expected_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
        description="update/edit, required: the fingerprint your last show returned.",
    )


def _config_dir(context: ToolContext | None) -> Path | None:
    value = getattr(context, "memory_config_dir", None)
    return Path(value) if value else None


def _catalogue(config_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """``(rows, discovered-by-name)`` the way the Skills tab's list builds them.

    Over ``desktop_skill_writes.run_skill_roots`` -- the SAME roots
    ``require_unshadowed`` judges a create by, so what this shows as editable and
    what a create refuses agree, with the same codes, even with the profile moved.
    """
    from local_operator.server.utils import desktop_skills
    from local_operator.server.utils.desktop_skill_writes import run_skill_roots
    from local_operator.skills import discover_skills
    from local_operator.skills.api import ecosystem_skill_roots

    discovered, _warnings = discover_skills(run_skill_roots(config_dir), config_dir=config_dir)
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


def _list(config_dir: Path, offset: int, context: ToolContext | None) -> str:
    """One page of rows, held under the replay bound, saying where the next starts.

    Review R2: one unpaged line per skill elides the middle rows on replay once a
    host carries ~50 skills, and the model would miss skills (and their editable
    flag) without being told.
    """
    rows, _ = _catalogue(config_dir)
    if not rows:
        return "no skills found. Create one with op='create'."
    ordered = sorted(rows, key=lambda item: item["name"])
    footer = "read-only skills cannot be changed here. " + _COPY_HINT
    budget = RESULT_MAX_CHARS - len(footer) - 200
    lines: list[str] = []
    used = 0
    index = offset - 1
    while index < len(ordered):
        row = ordered[index]
        mark = "editable" if row["editable"] else f"read-only ({_why_read_only(row)})"
        description = " ".join(str(row.get("description") or "").split())
        if len(description) > 120:
            description = description[:119] + "…"
        line = f"- {identifier_text(context, row['name'])} [{row['source']}, {mark}] {description}"
        if lines and used + len(line) + 1 > budget:
            break
        lines.append(line)
        used += len(line) + 1
        index += 1
    if not lines:
        return f"no skills at offset {offset}; there are {len(ordered)}."
    header = f"skills {offset}-{index} of {len(ordered)} (show a skill before you change it)"
    if index < len(ordered):
        header += f"; MORE: op='list' offset={index + 1}"
    return header + ":\n" + "\n".join(lines) + "\n\n" + footer


def _page(text: str, offset: int) -> tuple[str, int, int, int, bool]:
    """``(page, first, last, total, cut)``: whole lines from ``offset`` within :data:`PAGE_CHARS`.

    Splits on ``keepends`` lines so the page is the file's bytes exactly -- what
    the model copies into an ``edit``'s ``old_text`` must match. A single line
    longer than a page is shown up to the budget and ``cut`` says so; the rest of
    that line is never shown, which is safe because nothing can be written from it
    (``edit`` needs the exact text it replaces, and the file is too long for a
    whole-file ``update``).
    """
    lines = text.splitlines(keepends=True)
    total = len(lines)
    taken: list[str] = []
    used = 0
    index = offset - 1
    cut = False
    while index < total:
        line = lines[index]
        if used + len(line) > PAGE_CHARS:
            if not taken:
                taken.append(line[:PAGE_CHARS])
                cut = True
                index += 1
            break
        taken.append(line)
        used += len(line)
        index += 1
    return "".join(taken), offset, index, total, cut


def _show(
    config_dir: Path,
    name: str,
    path: str | None,
    offset: int,
    context: ToolContext | None,
) -> str:
    from local_operator.hub_sync.identity import LinkError
    from local_operator.hub_sync.skill_document import SkillDocument, validate_path
    from local_operator.hub_sync.skill_storage import managed_skill_read
    from local_operator.skills.protocol import _read_text_capped

    rows, by_name = _catalogue(config_dir)
    row = next((item for item in rows if item["name"] == name), None)
    skill = by_name.get(name)
    if row is None:
        raise LinkError("skill-not-found", "no such skill")
    if skill is None:
        raise LinkError("skill-not-editable", "this skill has no local copy to show")
    target = path or "SKILL.md"
    validate_path(target)
    # A managed folder is read under the root lock so a concurrent swap is never
    # seen half-done; any other root is not ours to lock.
    base = Path(os.path.realpath(skill.base_dir))
    with managed_skill_read(config_dir):
        try:
            document: Any = SkillDocument.from_directory(base)
        except Exception:  # noqa: BLE001 - not a text document: SKILL.md only
            document = None
        if document is not None:
            found = next((file for file in document.files if file.path == target), None)
            text = found.content if found is not None else None
        elif target == "SKILL.md":
            text = _read_text_capped(Path(skill.file_path))
        else:
            raise LinkError("skill-not-editable", "this skill's files are not readable as text")
    if text is None:
        raise LinkError("skill-not-found", f"no file {target!r} in this skill")
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
    page, first, last, total, cut = _page(text, offset)
    if first > max(total, 1):
        past = f"{target} has {total} lines; offset {offset} is past the end."
        return "\n".join(header) + "\n\n" + past
    where = f"--- {target}: lines {first}-{last} of {total}, {len(text)} chars ---"
    notes: list[str] = []
    if cut:
        notes.append(
            f"[line {last} is longer than one page; only its first {PAGE_CHARS} chars are "
            "shown. Change it with op='edit' on text shown here.]"
        )
    if last < total:
        notes.append(f"[MORE: op='show' path={target!r} offset={last + 1}]")
    elif first > 1 or cut:
        notes.append("[end of file]")
    if len(text) > PAGE_CHARS:
        notes.append("[longer than one page: change it with op='edit', not a whole-file update]")
    body = "\n".join(header) + f"\n\n{where}\n{page}"
    if not page.endswith("\n"):
        body += "\n"
    body += "\n".join(notes)
    if target == "SKILL.md" and offset == 1 and document is not None:
        body += _listing(document)
    return body


def _listing(document: Any) -> str:
    """The other files, capped at :data:`_LISTING_MAX_CHARS` with an explicit count."""
    others = [file for file in document.files if file.path != "SKILL.md"]
    if not others:
        return ""
    lines: list[str] = []
    used = 0
    for file in others:
        line = f"- {file.path} ({len(file.content)} chars)"
        if used + len(line) + 1 > _LISTING_MAX_CHARS:
            break
        lines.append(line)
        used += len(line) + 1
    text = "\n\nother files (show with path=... to read one):\n" + "\n".join(lines)
    if len(lines) < len(others):
        # Said, never silent: the cap keeps the page under the replay bound.
        text += f"\n- … and {len(others) - len(lines)} more, not listed to fit this page"
    return text


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
    "alias-occupied": (
        "For a same-named Hub skill, use `lop skills pull` with the same skill ID, "
        "scope and options plus `--dry-run` to review an in-place adoption merge, "
        "then confirm its receipt with the same options. Do not move or "
        "overwrite the local folder to bypass review. For a new local skill, "
        "choose another name or edit the existing skill."
    ),
    "skill-not-found": "Check the name with op='list'.",
    "skill-file-too-long-to-replace": (
        "Show the part you want to change and use op='edit' with exact old_text."
    ),
    "skill-edit-no-unique-match": "Show that part of the file again and copy old_text exactly.",
    "skill-file-masked": (
        "Change it with op='edit', quoting only text that was not masked; the masked "
        "value is kept as it is."
    ),
    "skill-marker-introduced": (
        "Write the file's real text, never the marker; a value you saw masked cannot "
        "be written from here."
    ),
}

#: Hints that depend on more than the code (QA r3 N1/N2): one code covers refusals
#: whose remedies differ, and a hint naming a parameter the call did not have, or
#: a remedy for the other case, sends the model after the wrong fix.
_REMOVAL_HINT = "A long file cannot be removed here; the operator can remove it on the Skills tab."
_EDIT_MARKER_HINT = (
    "Quote only unmasked text in old_text/new_text; the real value cannot be written " "from here."
)


def _hint(code: str, op: str, message: str) -> str | None:
    if code == "skill-file-too-long-to-replace" and "cannot be removed whole" in message:
        # The storage message carries the verb ("replaced"/"removed"); it is the
        # only place that knows which path tripped the bound.
        return _REMOVAL_HINT
    if code == "skill-marker-introduced" and op == "edit":
        return _EDIT_MARKER_HINT
    return _HINTS.get(code)


def _work(params: SkillParams, config_dir: Path, context: ToolContext | None) -> str:
    """The blocking body (filesystem, fsync, root lock), run off the event loop."""
    from local_operator.server.utils import desktop_skill_writes as writes

    name = str(params.name)
    files = [item.model_dump() for item in params.files or ()]
    if params.op == "list":
        return _list(config_dir, params.offset or 1, context)
    if params.op == "show":
        return _show(config_dir, name, params.path, params.offset or 1, context)
    if params.op == "create":
        writes.require_unshadowed(config_dir, name)
        writes.refuse_new_markers_in_files(files)
        return _write_receipt(writes.create(config_dir, name, files=files), "created", context)
    if params.op == "edit":
        result = writes.edit_files(
            config_dir,
            name,
            [item.model_dump() for item in params.edits or ()],
            str(params.expected_fingerprint),
        )
        return _write_receipt(result, "updated", context)
    result = writes.patch(
        config_dir,
        name,
        files,
        params.remove_paths or (),
        str(params.expected_fingerprint),
        whole_file_max_chars=PAGE_CHARS,
        mask=_session_mask(context),
    )
    return _write_receipt(result, "updated", context)


def _plan(params: SkillParams, config_dir: Path, context: ToolContext | None) -> dict[str, Any]:
    """The proposal plan for a create/update/edit: ``{before, after, ...}`` images.

    Built by the SAME validators the write runs (``plan_only``), so a filed
    proposal is exactly the write that would have happened -- including the
    page bound, mask and marker refusals that keep a write from being composed
    from text the model never saw. The base image and its fingerprint are read
    together and checked against ``expected_fingerprint``: the after image was
    overlaid on that revision, so a base read from a later one would let an
    accept silently revert the change made in between.
    """
    from local_operator import profile_proposals as P
    from local_operator.hub_sync.identity import LinkError
    from local_operator.server.utils import desktop_skill_writes as writes

    name = str(params.name)
    files = [item.model_dump() for item in params.files or ()]
    if params.op == "create":
        writes.require_unshadowed(config_dir, name)
        writes.refuse_new_markers_in_files(files)
        writes.document_from_files(name, files)
        if P.skill_image(config_dir, name) is not None:
            raise LinkError(
                "alias-occupied", f"a skill named {name!r} already exists on this machine"
            )
        before = None
        after_files = {item["path"]: item["content"] for item in files}
    else:
        if params.op == "edit":
            planned = writes.edit_files(
                config_dir,
                name,
                [item.model_dump() for item in params.edits or ()],
                str(params.expected_fingerprint),
                plan_only=True,
            )
        else:
            planned = writes.patch(
                config_dir,
                name,
                files,
                (),
                str(params.expected_fingerprint),
                whole_file_max_chars=PAGE_CHARS,
                mask=_session_mask(context),
                plan_only=True,
            )
        # Same fingerprint as the overlay's base => same bytes (content hash).
        before, fingerprint = P._skill_read(config_dir, name)
        if before is None or fingerprint != params.expected_fingerprint:
            raise LinkError("local-concurrent-edit", "the skill changed since it was read")
        after_files = planned["files"]
    return {
        "profile_kind": "skill",
        "operation": params.op if params.op == "create" else "update",
        "target_id": name,
        "name": name,
        "before": before,
        "after": {"id": name, "name": name, "files": after_files},
    }


def _session_mask(context: ToolContext | None) -> Callable[[str], str] | None:
    """The credential mask the loop applies to this tool's results (review R6).

    A session hands its loop ``redact_tool_result`` and its tools
    ``ToolContext.variables`` from ONE store, and both mask through
    ``VariableStore.redact``'s pass -- so judging "did the model see this file as
    it is?" with this function asks exactly the question the loop answered. No
    store, no mask: the loop then masks nothing either.
    """
    store = context.variables if context is not None else None
    return store.redact if store is not None else None


def _arguments_refusal(params: SkillParams) -> str | None:
    """Cross-field rules the schema cannot state; ``None`` when the call is well-formed."""
    if params.op != "list" and not params.name:
        return f"op={params.op!r} needs 'name'."
    if params.path is not None and params.op != "show":
        return "'path' is for op='show' only."
    if params.offset is not None and params.op not in ("list", "show"):
        return "'offset' is for op='list' and op='show' only."
    if params.edits and params.op != "edit":
        return "'edits' is for op='edit' only."
    if params.op in ("list", "show") and (
        params.files or params.remove_paths or params.expected_fingerprint
    ):
        return f"op={params.op!r} takes no files, remove_paths or expected_fingerprint."
    if params.op == "create":
        if not params.files:
            return "op='create' needs 'files' with at least SKILL.md."
        if params.remove_paths or params.expected_fingerprint:
            return "op='create' takes no remove_paths or expected_fingerprint."
    if params.op in ("update", "edit") and not params.expected_fingerprint:
        return (
            f"op={params.op!r} needs 'expected_fingerprint': show the skill first and pass "
            "the fingerprint it returned."
        )
    if params.op == "update" and not params.files and not params.remove_paths:
        return "op='update' needs 'files' to replace or 'remove_paths' to delete."
    if params.op == "edit":
        if not params.edits:
            return "op='edit' needs 'edits'."
        if params.files or params.remove_paths:
            return "op='edit' takes 'edits', not files or remove_paths."
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
    from local_operator.hub_sync.identity import LinkError, refusal_text_en
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
        if params.op not in {"list", "show"}:
            from local_operator.harness.types import DurableWriteIntent
            from local_operator.profile_scope import (
                durable_write_decision,
                durable_write_refusal,
            )

            # Create/update/edit carry a proposal plan, exactly as the agent and
            # team tools do: the host's ladder applies it directly for a clean
            # local run and FILES it for review in Agents → <name> when the run
            # read untrusted input (peer, Team Bridge, remote). ``remove_paths``
            # is a delete and carries no plan, so it keeps the account card.
            proposal = None
            if not params.remove_paths:
                proposal = await asyncio.to_thread(_plan, params, config_dir, context)
                if proposal["before"] == proposal["after"]:
                    # Nothing would change: no write, so no consent to ask for.
                    return _text(
                        tool_call_id,
                        _TOOL,
                        f"skill {identifier_text(context, str(params.name))} already had "
                        "this content; nothing changed.",
                    )
            intent = DurableWriteIntent(
                kind="skill",
                name=str(params.name),
                operation=(
                    "delete"
                    if params.remove_paths
                    else "create" if params.op == "create" else "update"
                ),
                changes=tuple(f.path for f in params.files or ())
                + tuple(e.path for e in params.edits or ()),
                summary="\n".join(
                    [f"{f.path}:\n{f.content}" for f in params.files or ()]
                    + [f"{e.path}: {e.old_text} → {e.new_text}" for e in params.edits or ()]
                    + [f"Remove: {path}" for path in params.remove_paths or ()]
                ),
                proposal=proposal,
                tool_call_id=tool_call_id,
            )
            origin, reason = await durable_write_decision(context, intent)
            if reason == "proposed":
                from local_operator.tools.agent_tool import _proposed_profile_receipt

                return _proposed_profile_receipt(tool_call_id, _TOOL, origin)
            if origin is None:
                return _error(tool_call_id, _TOOL, durable_write_refusal(intent, reason))
        text = await asyncio.to_thread(_work, params, config_dir, context)
    except (LinkError, SkillValidationError, TransactionError) as error:
        code = getattr(error, "code", "") or ""
        # Model-facing tool result: catalogued refusals are pinned to English.
        sentence = refusal_text_en(error)
        hint = _hint(code, params.op, sentence)
        message = f"{code}: {sentence}" + (f". {hint}" if hint else "")
        return _error(tool_call_id, _TOOL, message, details={"code": code})
    return _text(tool_call_id, _TOOL, text)


def build_skill_tool(context: ToolContext) -> AgentTool | None:
    """createIf: any host with the managed profile root can use the tool.

    Mutations independently require the host consent seam; availability itself
    is not authority and does not depend on which page launched the session.
    """
    if _config_dir(context) is None:
        return None
    return AgentTool(
        name=_TOOL,
        label="Skills",
        description=(
            "The operator's skills: list them, show a file page by page, create "
            "one, or change one marked editable (edit: exact replacements; update: "
            "whole short files). Pass the fingerprint show returned."
        ),
        parameters=SkillParams.model_json_schema(),
        # Per-call host consent owns mutation authority. Read-tier availability
        # must not create a second blanket approval before the exact decision.
        approval_tier="read",
        concurrency="exclusive",
        interruptible=False,
        execute=execute_skill,
    )
