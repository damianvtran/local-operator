"""Lossless edits of a role's or team's long-text fields (instructions, project).

WHY THIS EXISTS. An agent edits a brief by reading it with ``show`` and writing
it back with ``update``. Two failures turned that round trip into silent data
loss (the minervadev incident, 2026-11): ``show`` passed a 12.5k-char brief
through the generic tool-output preview, which keeps head and tail and elides
the middle, and ``update`` accepted "that preview plus one appended paragraph"
as the new whole field -- deleting ~6k chars the writer never saw. The
receipt then under-reported the loss (it counted ``-`` bullets of the new text
as removed lines), so nobody noticed.

THE CONTRACT, every rule structural rather than a size or ratio threshold (no
distribution exists to calibrate one against, and any ratio both blocks a
legitimate rewrite and admits a smaller loss):

1. ``show`` never hands the model a clipped field it could mistake for the
   whole. "Whole in the ToolResult" was not enough (QA round 1, Q2): the
   transcript kept a 13,024-char show complete, yet the NEXT model request
   carried 4,261 chars of it -- ``harness.replay_bound`` elides any replayed
   text block over its bound, after the fact, on the wire only. So a field
   longer than one page is never printed inline: default ``show`` prints it
   as a pointer (size, line count, sha256, part count), and ``show
   field=<f> part=<k>`` returns ONE page, sized (:data:`FIELD_PAGE_CHARS`)
   so header+page stays far below every replay bound. Each page states the
   row revision, the field's sha256, its exact char/line span and the next
   part; the bodies concatenate byte-exactly to the stored value
   (:func:`field_pages`). A ``base_revision`` on a page read refuses a row
   that moved between pages, so two versions can never be stitched.
2. ``edits`` -- an exact ``old`` -> ``new`` patch, or an append when ``old``
   is omitted -- is computed against the STORED text, so it cannot drop bytes
   it does not name. Each ``old`` must occur exactly once; the list is
   ordered and all-or-nothing.
3. A whole-field replacement that would remove any existing line must carry
   ``replace=true`` AND ``base_revision`` equal to the row's current
   revision: the agent's own explicit acknowledgement that the loss is
   intended, written from THIS version read in full (only a lossless
   ``show`` prints it) -- never a human approval. A ``base_revision`` on any
   update is also a CAS: a row that moved since that read refuses.
4. A value carrying the harness's own truncation/spill marker text that the
   stored field did not already contain is refused even when acknowledged:
   it is a pasted preview, never an intended brief.

``field_line_stats`` is the ONE line-count computation for a field change, so
a receipt, a proposal summary and a refusal can never disagree again.
"""

from __future__ import annotations

import difflib
import hashlib
import re
from typing import Any, Iterable, Mapping

from pydantic import BaseModel, ConfigDict, Field

#: Text the harness itself inserts when it clips or spills a tool result, or
#: elides one on replay. Exact literals from their producers (``tools.builtin``
#: spill footer, ``text_bounds.OUTPUT_TRUNCATION_MARKER``,
#: ``harness.replay_bound``) rather than a fuzzy "looks truncated" guess, so
#: ordinary prose about spills or truncation is never refused.
_PREVIEW_MARKERS = (
    re.compile(r"\[Full output \(\d+ lines\) is SAVED at spill://"),
    re.compile(re.escape("... [output truncated] ...")),
    re.compile(r"\[\.\.\. \d+ characters elided on replay"),
    # ``show``'s own page header and long-field pointer (below): a value
    # quoting them is a copied page or pointer, never an intended brief.
    re.compile(r"part \d+/\d+ · revision \S+ · sha256 [0-9a-f]{16}"),
    re.compile(r": NOT shown here \(\d+ chars, \d+ lines, sha256 [0-9a-f]{16}\)"),
)


#: Largest page body ``show field=<f> part=<k>`` returns, in characters.
#: DERIVED, not chosen: ``harness.replay_bound.DEFAULT_REPLAY_BOUND_CHARS``
#: (8,192) elides any longer replayed text block, and QA round 1 (Q2) measured
#: a real next request keeping only 4,261 chars of one result -- so a page
#: must survive a bound of HALF the default with room for its header. 3/8 of
#: the default (3,072) plus a header under ~300 chars stays under 4,096
#: (``SHOW_RESULT_BUDGET_CHARS``); the regression test replays every page
#: through ``bound_replay_payloads`` at both bounds. Not a new knob: it moves
#: only if that bound moves.
FIELD_PAGE_CHARS = 3 * 1024
#: Largest DEFAULT ``show`` result that may print fields inline (half the
#: replay bound, the same derivation); a longer one points at pages instead.
SHOW_RESULT_BUDGET_CHARS = 4 * 1024


def field_digest(value: str) -> str:
    """Short sha256 of a whole field value: what proves pages belong together."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def field_pages(value: str, page_chars: int = FIELD_PAGE_CHARS) -> list[str]:
    """``value`` split into pages that concatenate back to it byte-exactly.

    A page ends after its last newline when that keeps it at least half full
    (readable pages); otherwise -- one huge paragraph -- it is a hard cut at
    ``page_chars`` code points. Python strings index code points, so a cut
    never splits a character; a ``\r\n`` split across two pages still
    reassembles exactly because nothing is added or removed at the seam.
    """
    if not value:
        return [""]
    pages: list[str] = []
    start = 0
    while start < len(value):
        end = min(start + page_chars, len(value))
        if end < len(value):
            newline = value.rfind("\n", start, end)
            if newline >= start + page_chars // 2:
                end = newline + 1
        pages.append(value[start:end])
        start = end
    return pages


def field_pointer(field: str, value: str) -> str:
    """The default-``show`` line for a field too long to print inline."""
    parts = len(field_pages(value))
    return (
        f"{field}: NOT shown here ({len(value)} chars, {len(value.splitlines())} lines, "
        f"sha256 {field_digest(value)}). Read it whole, page by page: op='show' "
        f"field={field} part=1..{parts} (pass base_revision to pin one version)."
    )


def render_field_page(
    field: str, value: str, part: int, revision: str, *, base_revision: str | None = None
) -> str:
    """One page of ``field`` with the header that makes it verifiable.

    Raises :class:`TextEditRefusal` for a part out of range or a
    ``base_revision`` that no longer matches (the row moved between pages).
    The body follows the header's ``---`` line verbatim: concatenating every
    part's body gives the stored value, whose sha256 the header states.
    """
    if base_revision is not None and base_revision != revision:
        raise TextEditRefusal(
            "base_revision does not match the stored revision: the row changed since "
            "the earlier page, so pages of two versions would not fit together. "
            "Re-read from part=1."
        )
    pages = field_pages(value)
    if not 1 <= part <= len(pages):
        raise TextEditRefusal(f"{field} has {len(pages)} part(s); part={part} does not exist.")
    start = sum(len(page) for page in pages[: part - 1])
    body = pages[part - 1]
    first_line = value.count("\n", 0, start) + 1
    last_line = first_line + body.rstrip("\n").count("\n") if body else first_line
    nxt = f"next: part={part + 1}" if part < len(pages) else "last part"
    return (
        f"{field} part {part}/{len(pages)} · revision {revision} · sha256 "
        f"{field_digest(value)} · chars {start + 1}-{start + len(body)} of {len(value)} "
        f"· lines {first_line}-{last_line} of {len(value.splitlines())} · {nxt}. "
        "Body below the --- line, verbatim; parts concatenate exactly.\n---\n" + body
    )


class TextEditRefusal(ValueError):
    """A lossless-edit rule refused the call; nothing was written."""


# One ``edits`` entry on ``agent``/``team`` update. No docstring on purpose:
# pydantic publishes a model docstring as the schema description, and this
# schema rides every request on two tools.
class TextEdit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Wire copy kept minimal: this schema rides every request on two tools.
    field: str = Field(description="instructions|project")
    old: str | None = Field(default=None, description="exact text, once; omit=append")
    new: str


def field_line_stats(before: str, after: str) -> tuple[int, int]:
    """``(added, removed)`` lines between two WHOLE field values.

    Same matcher ``difflib.unified_diff`` uses, so the counts equal the ``+``/
    ``-`` lines of any unified diff of the two values regardless of context
    width -- and they are computed from the values, never from rendered text
    (where a markdown bullet looked like a removed line).
    """
    old_lines = (before or "").splitlines()
    new_lines = (after or "").splitlines()
    added = removed = 0
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag in ("replace", "delete"):
            removed += i2 - i1
        if tag in ("replace", "insert"):
            added += j2 - j1
    return added, removed


def stats_heading(field: str, before: str, after: str) -> str:
    """The summary heading every renderer uses: ``<field> (+A/-R lines)``."""
    added, removed = field_line_stats(before, after)
    return f"{field} (+{added}/-{removed} lines)"


_STATS_HEADING = re.compile(r"^(?P<field>[a-z_]+) \(\+(?P<added>\d+)/-(?P<removed>\d+) lines\):")


def parse_stats_heading(line: str) -> tuple[str, int, int] | None:
    """Read back a :func:`stats_heading` line (for ``profile_diff_summary``)."""
    match = _STATS_HEADING.match(line)
    if match is None:
        return None
    return match["field"], int(match["added"]), int(match["removed"])


def has_preview_marker(value: str, stored: str) -> bool:
    """Whether ``value`` carries harness clip/spill text ``stored`` lacked."""
    return any(p.search(value) and not p.search(stored) for p in _PREVIEW_MARKERS)


def apply_edits(
    stored: Mapping[str, str], edits: Iterable[Any], allowed: Iterable[str]
) -> dict[str, str]:
    """Apply ``edits`` (objects with ``field``/``old``/``new``) to ``stored``.

    Sequential, against the stored text: each ``old`` must occur exactly once
    in the field as the previous edits left it. Any failure refuses the WHOLE
    call -- a partial patch is a write nobody asked for. Returns only the
    fields an edit touched.
    """
    names = tuple(allowed)
    values = {field: str(stored.get(field) or "") for field in names}
    touched: dict[str, str] = {}
    for index, edit in enumerate(edits, start=1):
        field = str(edit.field)
        if field not in names:
            raise TextEditRefusal(
                f"edits[{index}]: field {field!r} cannot be patched here; use one of "
                f"{', '.join(names)}. Nothing was written."
            )
        current = values[field]
        old = edit.old
        new = str(edit.new)
        if not old:
            if not new:
                # An empty append changes nothing yet used to add a bare
                # newline: refuse instead of writing a no-op "edit".
                raise TextEditRefusal(
                    f"edits[{index}]: an append (old omitted) needs non-empty new text; "
                    "nothing was written."
                )
            # Append on its own line: a brief is line-structured, and gluing the
            # new text onto the last sentence is never what an append means. A
            # ``new`` that already starts with its own newline brings the
            # separator itself; adding another would insert a byte nobody wrote.
            joiner = (
                "\n" if current and not current.endswith("\n") and not new.startswith("\n") else ""
            )
            values[field] = current + joiner + new
        else:
            found = current.count(old)
            if found != 1:
                what = "not found" if found == 0 else f"found {found} times"
                raise TextEditRefusal(
                    f"edits[{index}]: old text {what} in {field}; nothing was written. "
                    "Re-read with op='show' and quote a longer, unique passage."
                )
            values[field] = current.replace(old, new, 1)
        touched[field] = values[field]
    return touched


def replacement_refusal(
    field: str,
    stored: str,
    value: str,
    *,
    base_revision: str | None,
    current_revision: str | None,
    replace: bool,
) -> str | None:
    """Refuse a whole-field replacement that would silently drop stored text.

    Returns the refusal sentence, or ``None`` when the write may proceed. A
    replacement that keeps every stored line (a pure insertion) needs nothing;
    one that removes lines needs ``replace=true`` plus the current
    ``base_revision`` (rule 3). The refusal deliberately does not print the
    current revision: echoing it would let a caller that never read the text
    copy it back and pass.
    """
    if has_preview_marker(value, stored):
        return (
            f"the new {field} contains a tool-output truncation/spill marker, so it is "
            "a clipped preview or copied page, not the whole text; nothing was written. "
            "Use edits=[...] to append or patch (they apply to the stored text), or "
            "read every part with op='show' field=<f> part=k before a rewrite."
        )
    added, removed = field_line_stats(stored, value)
    if removed == 0:
        return None
    if replace and base_revision is not None and base_revision == current_revision:
        return None
    lost = len(stored) - len(value)
    size = f"; it is {lost} chars shorter" if lost > 0 else ""
    return (
        f"replacing {field} would remove {removed} of {len(stored.splitlines())} stored "
        f"lines (+{added}/-{removed}){size}; nothing was written. To add or change part "
        "of it, use edits=[{field, old, new}] (omit old to append). To rewrite it on "
        "purpose, read it in full (op='show' field=<f> part=1..) and pass replace=true "
        "(your own acknowledgement that the removal is intended; not confirm_replace, "
        "which is the person's confirmation for other ops) with its revision as "
        "base_revision."
    )


def resolve_text_fields(
    stored: Mapping[str, str],
    *,
    whole: Mapping[str, str | None],
    edits: Iterable[Any] | None,
    base_revision: str | None,
    current_revision: str,
    replace: bool,
) -> dict[str, str] | str:
    """The resolved new values of the long-text fields, or a refusal sentence.

    ``whole`` maps each long-text field to the caller's whole-field value
    (``None`` = not given). Shared by the ``agent`` and ``team`` tools so the
    two can never apply different rules.
    """
    if base_revision is not None and base_revision != current_revision:
        return (
            "base_revision does not match the stored revision: the row changed since "
            "that read (or the value is not one show printed); nothing was written. "
            "Re-read with op='show'."
        )
    edit_list = list(edits or ())
    given = {field for field, value in whole.items() if value is not None}
    clash = given & {str(edit.field) for edit in edit_list}
    if clash:
        return (
            f"{', '.join(sorted(clash))}: pass either the whole field or edits for it, "
            "not both; nothing was written."
        )
    try:
        out = apply_edits(stored, edit_list, whole.keys()) if edit_list else {}
    except TextEditRefusal as exc:
        return str(exc)
    for field in sorted(given):
        value = str(whole[field])
        refusal = replacement_refusal(
            field,
            str(stored.get(field) or ""),
            value,
            base_revision=base_revision,
            current_revision=current_revision,
            replace=replace,
        )
        if refusal is not None:
            return refusal
        out[field] = value
    return out
