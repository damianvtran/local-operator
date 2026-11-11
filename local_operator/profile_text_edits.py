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

1. ``show`` returns the stored fields whole, with the row's revision. The
   field caps (``MAX_TEAM_INSTRUCTIONS_CHARS``, ``MAX_INSTRUCTIONS_CHARS``)
   already bound what that costs.
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
            # Append on its own line: a brief is line-structured, and gluing the
            # new text onto the last sentence is never what an append means.
            joiner = "\n" if current and not current.endswith("\n") else ""
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
            "a clipped preview, not the whole text; nothing was written. Re-read with "
            "op='show' (it returns the full text) and use edits=[...] to append or patch."
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
        "purpose, re-read with op='show' and pass replace=true with its revision as "
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
