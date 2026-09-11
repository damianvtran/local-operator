"""``@path`` references — turning tokens in a submitted message into content.

THE GOVERNING RULE
------------------
A token that does not resolve to an existing path is not a reference. It is
prose, and it is left byte-identical.

That one sentence is the whole safety argument for D7 (expansion fires on ANY
text, with no provenance flag threaded through fifteen call sites). ``glab mr
create --assignee @me`` is a boundary ``@`` and therefore a candidate token —
but ``me`` names no file under the cwd, so nothing is captured, no
``<operator-references>`` block is emitted, and the text reaches the model
exactly as typed. ``@`` inherits the discipline ``skill_token``'s docstring
already states for ``$``: *nothing is ever captured that is not an existing
path*.

WHY EXPANSION HAPPENS ONCE, AT SUBMIT
-------------------------------------
The expanded payload IS the message, and it persists that way. The rejected
alternative was to keep the bare ``@path`` in the text and re-resolve it at
replay time, the way :class:`~local_operator.session.attachments.AttachmentStore`
re-resolves a digest. That analogy breaks on a correctness property: a digest
resolves to bytes that CANNOT change (the sha256 is the identity), while a path
resolves to whatever is on disk *now* — a different file, or none. A
conversation that replays differently from how it ran is strictly worse than a
large transcript row, because the model would answer about code the user never
showed it. So expansion runs once, the result is persisted, and replay is
byte-identical to the live turn for free.

Two callers expand, and the second pass must be a no-op: the TUI expands at its
submit exits so the transcript row can stay short, and :meth:`Session.prompt`
expands unconditionally so every non-TUI surface (CLI, headless, server,
scheduler, mobile, subagent) gets the feature with zero per-surface work. That
makes IDEMPOTENCE a hard requirement rather than a nicety —
``expand_references(expand_references(t).sent).expanded is False`` — and it is
true BY CONSTRUCTION here: :func:`_reference_spans` skips any ``@`` sitting
inside an already-emitted block. Were it true only by luck, every operator
message would carry a doubled block.

NO PROVENANCE FLAG (D7)
-----------------------
There is no ``operator_typed=`` keyword and the protocol signature does not
widen. Threading one correctly would mean setting it at ten call sites and
deliberately leaving it unset at five more, where the DEFAULT is the safe
answer and every operator-facing site is an opt-in somebody must remember; a
missed site fails silently by just not working. ``serving.py:1403`` already
inspects ``signature(...).parameters`` to cope with sessions predating a
keyword, and a third such probe is a real maintenance cost. The strict resolver
above buys what the flag would have bought.

CONSTRAINT — this is a SESSION-layer leaf. It may import stdlib,
:mod:`local_operator.sigils`, :mod:`local_operator.tools.builtin` and
:mod:`local_operator.harness.approval`, and nothing from
``local_operator.tui.*`` at module scope (see :func:`scan_directory`) or from
``local_operator.session.*`` (that direction is the cycle).

The ``@`` GRAMMAR IS NOT HERE. :func:`~local_operator.sigils.at_token` and
:func:`~local_operator.sigils.split_token` live in :mod:`local_operator.sigils`
with the boundary rule they are built on, below both this module and the
composer, so neither surface owns the parse. They are re-exported from here for
their original callers; a NEW caller should import them from ``sigils``.
"""

from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from local_operator.harness.approval import ApprovalGate, ask_approval

# ``media`` is stdlib-only (struct, dataclasses) and is ALREADY imported
# unconditionally by ``tools.builtin`` below, so naming it here adds zero
# modules to ``sys.modules`` and zero edges to the graph ``test_import_graph``
# guards. Imported directly rather than through ``builtin``'s namespace because
# a re-export that happens to exist is not a contract.
from local_operator.media import sniff_image_file

# ``at_token``/``split_token`` are re-exported (see ``__all__``) because they
# were this module's public surface before the grammar moved to its correct
# home in ``sigils.py``, and a caller that imported them from here must keep
# resolving to the very same object. ``_token_end`` is imported for
# :func:`_reference_tokens`, which is resolver logic and stayed behind: the
# submit side reads where a token ends from the leaf rather than reimplementing
# it, which is precisely what ``sigils.py`` exists for.
from local_operator.sigils import _token_end, at_token, is_boundary, split_token
from local_operator.tools.builtin import (
    _GREP_PRUNE_DIRS,
    INTERNAL_READ_HEAD_CHARS,
    INTERNAL_READ_LIMIT_CHARS,
    READ_FILE_LIMIT_BYTES,
    TOOL_OUTPUT_LIMIT_CHARS,
    _collect_headings,
    _ignored,
    _list_dir_entries,
    _load_ignore_rules,
    _resolve_workspace_path,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.tui.autocomplete import ArgumentChoice


class ExpansionResult(NamedTuple):
    """What one expansion pass produced.

    ``sent`` is the text the model receives. When ``expanded`` is False it is
    the caller's own string OBJECT, not an equal copy — callers assert that
    with ``is``, because the guarantee being made is "we did not touch your
    text", and an equal-but-rebuilt string cannot prove it.
    """

    sent: str
    expanded: bool
    notices: list[str]


#: The element wrapping every reference. Also the fence :func:`_reference_spans`
#: reads to make a second expansion pass a no-op — see the module docstring.
REFERENCE_BLOCK_OPEN = "<operator-references>"
REFERENCE_BLOCK_CLOSE = "</operator-references>"

#: Kill switch, read PER CALL and never cached at import. Two precedents, and
#: both are deliberate: ``LOCAL_OPERATOR_CONTEXT_FILES``
#: (``context_files.py:344``) supplies the exact string vocabulary matched
#: below, and ``_internal_read_limit`` (``builtin.py:2598-2614``) supplies the
#: per-call discipline — its docstring says why, "so the override can be set
#: after this module is imported". Env-only by design: neither precedent is in
#: the ``/settings`` registry, and that rule governs config keys.
AT_REFERENCES_ENV = "LOCAL_OPERATOR_AT_REFERENCES"

#: Whole-block ceiling, four tool results' worth. The per-reference caps bound
#: one file; this bounds a message that names twenty. References past it are
#: listed BY PATH ONLY with a line saying so, which keeps the worst case
#: bounded regardless of who typed the text or how many tokens they typed —
#: and under D7 the typer need not be the operator.
BLOCK_LIMIT_CHARS = TOOL_OUTPUT_LIMIT_CHARS * 4

#: Ceiling on the candidate list one scan builds. The picker windows to 8 rows,
#: so the ROWS were never the cost; this stops a pathological directory from
#: making the SORT the cost on a keystroke path (design §2.5, R6).
SCAN_CANDIDATE_LIMIT = 2000

#: Names, suffixes and ancestor directories that force an approval prompt even
#: when the path is INSIDE the workspace. This is the one guard ``read`` does
#: not have, and the asymmetry is deliberate: ``read`` is a deliberate act by
#: an agent under instructions, while ``@`` expansion is AUTOMATIC AND SILENT —
#: nobody decided to read that file, a token in a sentence pulled it in.
#:
#: Matching does NOT refuse. It escalates to the same prompt an outside path
#: gets, with a description naming why, so the operator can still reference
#: their own ``.env`` deliberately; they just have to say yes.
SENSITIVE_NAMES = frozenset(
    {".env", ".netrc", ".npmrc", ".pypirc", "credentials", "id_rsa", "id_ed25519"}
)
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".keystore"})
SENSITIVE_DIR_PARTS = frozenset({".ssh", ".gnupg", ".credentials", ".aws", ".kube"})

#: Bytes sampled for the NUL probe, and the probe itself — the SAME detection
#: ``read`` uses at ``builtin.py:3441``. One answer about what "binary" means,
#: not two that can drift.
_BINARY_SNIFF_BYTES = 8000

#: Entries listed inside one ``<reference>`` for a directory before the tail is
#: named rather than shown. A directory reference is an orientation aid, not a
#: recursive dump; the agent has ``glob`` for the rest.
_DIRECTORY_ENTRY_LIMIT = 200


def at_references_enabled() -> bool:
    """Whether ``@`` expansion runs at all.

    Read per call. See :data:`AT_REFERENCES_ENV` for both precedents and why
    caching this at import would be wrong.
    """
    return os.environ.get(AT_REFERENCES_ENV, "1").strip() not in ("0", "false", "no")


def _is_sensitive(path: Path) -> bool:
    """Whether ``path`` trips the deny-list, and therefore must be asked about."""
    name = path.name
    if name in SENSITIVE_NAMES:
        return True
    # A `.env*` PREFIX counts: `.env.local` holds the same class of secret
    # `.env` does, and a name-set membership test alone would wave it through.
    if name.startswith(".env"):
        return True
    if path.suffix in SENSITIVE_SUFFIXES:
        return True
    return bool(SENSITIVE_DIR_PARTS.intersection(path.parts))


def _entry_detail(entry: os.DirEntry[str], is_dir: bool) -> str:
    """A size for a file, and deliberately NOTHING for a directory.

    A directory's entry COUNT would be the natural detail and it is what the
    obvious implementation reaches for — but producing it means opening every
    listed subdirectory, which is a second (third, eleventh) ``scandir`` on a
    path that runs per keystroke with no debounce. Measured in this worktree,
    on a directory holding 10 subdirectories one of which is wide:

        no detail             0.158 ms
        file size only        0.240 ms
        entry count per dir  23.891 ms

    A 100x regression to fill one cosmetic column, on the exact path the
    one-level scan exists to protect, and it breaks the structural guarantee
    ``test_exactly_one_directory_is_scanned_per_keystroke`` pins. The trailing
    ``/`` already tells the reader it is a directory, which is what the column
    would mostly have been conveying. Cost, not taste, decides this.

    Never raises: an unreadable entry costs a blank column rather than the
    picker, because Textual turns an escaped error into a full-screen crash.
    """
    if is_dir:
        return ""
    try:
        return f"{entry.stat(follow_symlinks=False).st_size} B"
    except OSError:
        return ""


def scan_directory(directory: str, cwd: str) -> list["ArgumentChoice"]:
    """One directory's listable entries, as picker rows. Never raises.

    SYNCHRONOUS AND ONE LEVEL, and both halves are load-bearing. This runs from
    ``_sync_picker`` on EVERY keystroke (``editor.py:3011``) and every buffer
    mutation, with no debounce, no cancellation and no generation counter. The
    measured costs decide the shape: one ``os.scandir`` of the repo root is
    0.04 ms and of ``local_operator/`` 0.07 ms, against 68 ms for an
    ignore-aware walk of this repo and 8556 ms for one of a real workspace —
    three orders of magnitude over any frame budget, per keystroke. So there is
    no worker, no debounce and no cancellation here, because at 0.04 ms none of
    them needs to exist. ``tests/unit/test_reference_scan.py`` pins that
    STRUCTURALLY (exactly one ``scandir`` per call), not as a wall-clock bound,
    which is what makes reaching for a walk fail deterministically.

    A directory resolving OUTSIDE the workspace still scans. Listing is not
    reading: the picker shows you what is there, and the approval gate sits at
    EXPANSION time in :func:`expand_references`. Splitting the two is not
    obvious, and collapsing them would mean either prompting on a keystroke or
    reading an outside file without asking.
    """
    # Lazy, and this is the §3.1 correction rather than a style choice. This
    # module is imported at module scope by ``session/session.py``, so a
    # module-level ``from local_operator.tui.autocomplete import
    # ArgumentChoice`` puts ``local_operator.tui`` on the ``session_factory``
    # import path. Measured in this worktree: module-level adds
    # ['local_operator.tui', 'local_operator.tui.autocomplete'] to
    # ``sys.modules``, lazy adds []. ``autocomplete.py`` is Textual-free TODAY,
    # so the eager form would not go red — it would quietly establish a
    # session->TUI edge that the next ``autocomplete.py`` import turns into a
    # ``test_import_graph.py:163`` failure in a module nobody connected to it.
    # Same pattern and same stated reason as ``harness/rows.py:103-108``.
    from local_operator.tui.autocomplete import ArgumentChoice

    path, _inside, resolvable = _resolve_workspace_path(directory or ".", cwd)
    if not resolvable:
        return []
    try:
        rules = [("", _load_ignore_rules(path, ""))]
        with os.scandir(path) as scan:
            entries = sorted(scan, key=lambda entry: entry.name)
    except OSError:
        # A missing, unreadable or racing directory is an empty list, never an
        # exception: the caller is a keystroke handler and Textual turns an
        # escaped error into a full-screen crash.
        return []

    choices: list[ArgumentChoice] = []
    for entry in entries:
        if len(choices) >= SCAN_CANDIDATE_LIMIT:
            break
        name = entry.name
        # The walker's own exclusion vocabulary, imported rather than retyped so
        # ``@`` agrees with ``grep`` and ``glob`` about what is worth showing. A
        # second copy of these names is the drift defect this repo names
        # repeatedly.
        if name in _GREP_PRUNE_DIRS or name.startswith("."):
            continue
        try:
            # ``DirEntry.is_dir(follow_symlinks=False)`` reads the ``d_type``
            # the kernel already returned with the listing, so classification
            # costs ZERO extra syscalls — the reasoning is written out at
            # ``builtin.py:5086-5096``, where ``iterdir`` + per-entry ``Path``
            # predicates paid three stat(2) calls per entry.
            is_dir = entry.is_dir(follow_symlinks=False)
        except OSError:
            continue
        if _ignored(name, is_dir, rules):
            continue
        choices.append(
            ArgumentChoice(
                # Trailing ``/`` on a directory, matching ``_list_dir_entries``
                # (``builtin.py:3232-3238``) so one listing convention serves
                # both the picker and the expanded payload.
                name=name + ("/" if is_dir else ""),
                detail=_entry_detail(entry, is_dir),
                alert=_is_sensitive(Path(entry.path)),
            )
        )
    return choices


#: The attribute carrying the token as typed, named once because two places
#: depend on the exact spelling: :func:`_render` writes it and
#: :func:`_already_expanded` reads it back to hold idempotence.
_TYPED_ATTRIBUTE = 'typed="'

#: A zero-width space, the character :func:`_defuse` splices into a block
#: marker found in a reference's body. Invisible to a reader and harmless to
#: the model's understanding of the text, while no longer being the literal
#: sequence :func:`_block_spans` matches.
_ZERO_WIDTH_SPACE = "\u200b"
_DEFUSED_OPEN = REFERENCE_BLOCK_OPEN.replace("<", "<" + _ZERO_WIDTH_SPACE, 1)
_DEFUSED_CLOSE = REFERENCE_BLOCK_CLOSE.replace("<", "<" + _ZERO_WIDTH_SPACE, 1)

#: One sentence of framing inside the block. The model must be able to tell
#: operator prose from file content, which is also why expansion APPENDS rather
#: than substituting in place — inline substitution would lose the typed token
#: and leave the transcript row nothing short to paint.
_BLOCK_PREAMBLE = "The operator's message references these paths. Content is included below."


class _Token(NamedTuple):
    """One candidate reference found in a submitted message."""

    typed: str
    raw: str


def _attribute(value: str) -> str:
    """Escape a value for an XML-ish attribute, reversibly.

    Character-for-character ``skills/invoke.py::_escape_attr`` (``:210-224``),
    and the completeness is load-bearing rather than tidy. Escaping only ``"``
    is not enough: a value carrying ``<`` and ``>`` can spell
    ``</operator-references>`` inside the attribute, which terminates the block
    EARLY for :func:`_block_spans`. Every ``typed=`` after the forgery then
    falls outside the recovered span, :func:`_already_expanded` returns nothing
    for it, and pass 2 re-expands — contract guarantee 3 broken, and a second
    block appended to a message that is persisted and re-sent every turn.

    Reachable two ways, and neither is theoretical. The marker contains ``/``,
    a legal POSIX separator, so ``a</operator-references>b.txt`` is a REAL
    filename (parent ``a<``, name ``operator-references>b.txt``) whose path
    attribute would carry the marker verbatim. Reproduced before this fix: one
    such reference beside an ordinary one made pass 2 return ``expanded=True``
    and emit a second block.

    ``@"my file.txt"`` is the case that motivated escaping ``"`` at all — the
    quoted form exists to carry spaces, and an unescaped delimiter truncates
    the attribute so the recovery reads back ``@``.

    A newline becomes ``&#10;`` for ``invoke.py``'s stated reason: a multi-line
    value must not break the single-line tag the scanner matches.
    """
    return (
        value.replace("&", "&amp;")
        .replace('"', "&quot;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "&#10;")
    )


def _unattribute(value: str) -> str:
    """Inverse of :func:`_attribute`; ``&amp;`` last so it cannot double-undo.

    Reverse order is the whole correctness argument, and getting it wrong fails
    silently: unescaping ``&amp;`` FIRST would turn a literal ``&amp;lt;`` back
    into ``&lt;`` and then into ``<``, inventing a character the operator never
    typed and breaking the round trip idempotence depends on.
    """
    return (
        value.replace("&#10;", "\n")
        .replace("&gt;", ">")
        .replace("&lt;", "<")
        .replace("&quot;", '"')
        .replace("&amp;", "&")
    )


def _block_spans(text: str) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans of every already-emitted block."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start = text.find(REFERENCE_BLOCK_OPEN, cursor)
        if start == -1:
            return spans
        close = text.find(REFERENCE_BLOCK_CLOSE, start)
        end = len(text) if close == -1 else close + len(REFERENCE_BLOCK_CLOSE)
        spans.append((start, end))
        cursor = end


def _already_expanded(text: str, spans: list[tuple[int, int]]) -> set[str]:
    """Tokens a previous pass over ``text`` already resolved.

    THIS is what makes idempotence structural, and skipping the spans alone
    would NOT be enough to get it. The user's typed token deliberately SURVIVES
    in the prose — R4 needs it to paint a short transcript row, and the model
    reads the sentence rather than the block — so it sits OUTSIDE every span
    and a second pass would resolve it again and append a doubled block to
    every operator message. ``Session.prompt`` expands text the TUI already
    expanded, so that second pass is the normal case, not an edge.

    Reading the ``typed=`` attributes back out is the same recovery
    ``rows.typed_line_of`` performs on a persisted ``$skill`` payload
    (``harness/rows.py:103-108``), and it is why :func:`_render` writes the
    attribute at all. A token NEWLY added to already-expanded text still
    expands, which keeps a steered or edited draft working; after that pass it
    too is named in a block, so the property holds however many passes run.
    """
    typed: set[str] = set()
    for start, end in spans:
        region = text[start:end]
        cursor = 0
        while True:
            found = region.find(_TYPED_ATTRIBUTE, cursor)
            if found == -1:
                break
            value_start = found + len(_TYPED_ATTRIBUTE)
            close = region.find('"', value_start)
            if close == -1:
                break
            typed.add(_unattribute(region[value_start:close]))
            cursor = close + 1
    return typed


def _reference_tokens(text: str) -> list[_Token]:
    """Every candidate ``@`` token in ``text``, in order, already-resolved excluded."""
    skip = _block_spans(text)
    resolved = _already_expanded(text, skip)
    tokens: list[_Token] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char != "@" or not is_boundary(text, index):
            index += 1
            continue
        if any(start <= index < end for start, end in skip):
            index += 1
            continue
        # The token never spans a newline, so the line slice the composer
        # parses and this slice are the same string — the property
        # ``sigils.py`` exists to hold.
        line_end = text.find("\n", index)
        line_end = len(text) if line_end == -1 else line_end
        end, query = _token_end(text[index:line_end], 0)
        typed = text[index : index + end]
        if query and typed not in resolved:
            tokens.append(_Token(typed=typed, raw=query))
        index += max(end, 1)
    return tokens


def _describe(path: Path, inside: bool, resolvable: bool, sensitive: bool) -> str:
    """The approval sentence for one reference.

    Built here rather than through ``builtin._approval_description`` because
    the sensitive case is a sentence that function has no marker for, and the
    operator needs to know WHICH of the two reasons they are being asked —
    "outside the workspace" on a path visibly inside it argues with itself.
    """
    if sensitive:
        return f"read (referenced, may hold secrets): {path}"
    if not resolvable:
        return f"read (referenced, unresolvable): {path}"
    if not inside:
        return f"read (referenced, outside workspace): {path}"
    return f"read (referenced): {path}"


def _directory_payload(path: Path) -> tuple[str, dict[str, str]]:
    """A FLAT one-level listing, and the attributes describing it.

    Flat because that is what ``read`` does with a directory today, and one
    consistent answer beats a second mechanism; recursive-tree semantics have
    no precedent in this codebase. ``_list_dir_entries`` is reused verbatim so
    the trailing-slash convention cannot drift from ``read``'s.
    """
    entries = _list_dir_entries(path)
    shown = entries[:_DIRECTORY_ENTRY_LIMIT]
    body = "\n".join(shown)
    if len(entries) > len(shown):
        body += f"\n[{len(entries) - len(shown)} more entries; use glob or read to list them]"
    return body, {"entries": str(len(entries))}


def _shaped_text(path: Path, text: str, shown: str) -> tuple[str, dict[str, str]]:
    """Head + heading outline for a file past the per-reference char cap.

    The shape mirrors ``_shape_internal_document``, but the POINTER it emits
    must be the real path and line range and NEVER a ``spill://`` handle. The
    reason is stated at ``session/attachments.py:36-39`` and
    ``tools/spill.py:132``: spill is LRU-evicted under a byte ceiling with a
    30-minute session grace and is allowed to forget content a transcript still
    references. This payload rides a PERSISTED USER MESSAGE, so a handle in it
    is a dead link after lunch — and the reference block is exactly the thing a
    resumed session replays. The real path costs nothing and never expires,
    which is also what ``context_files._render_index_rows`` (``:496-530``) does
    for repo guidance.
    """
    lines = text.splitlines()
    head = text[:INTERNAL_READ_HEAD_CHARS]
    head_lines = len(head.splitlines())
    rows = [
        f"  - L{max(heading.start, head_lines + 1)}-{heading.end}: {heading.text}"
        for heading in _collect_headings(lines)
        if heading.end > head_lines
    ]
    outline = "\n".join(rows)
    # ``shown``, not ``str(path)``: the footer must name the SAME path the
    # element's ``path=`` attribute carries, or the model reads one address and
    # is told to follow another. ``read`` resolves it through
    # ``_resolve_workspace_path``, which joins a relative path onto cwd exactly
    # as ``context_files``' index rows already rely on — pinned by
    # ``test_the_read_pointer_in_a_shaped_file_still_resolves``, because a
    # footer pointing at a path ``read`` cannot resolve is a dead pointer the
    # model will follow.
    footer = (
        f"\n\n[shown: the first {head_lines} of {len(lines)} lines. "
        f"Read any section with read(path={shown!r}, range='start-end').]"
    )
    if outline:
        footer = f"\n\nSections not shown:\n{outline}{footer}"
    return head + footer, {"lines": str(len(lines)), "shown": "head+outline"}


def _file_payload(path: Path, size: int, limit: int, shown: str) -> tuple[str, dict[str, str]]:
    """One file's body and attributes, bounded, never bytes for a binary.

    ``shown`` is the display path every pointer in the payload must quote; see
    :func:`_shaped_text`'s footer comment for why it is not ``str(path)``.
    """
    if size > READ_FILE_LIMIT_BYTES:
        return (
            f"[not included: {size} bytes exceeds the {READ_FILE_LIMIT_BYTES}-byte "
            f"reference cap. Read a slice with read(path={shown!r}, range='start-end').]",
            {"bytes": str(size), "shown": "metadata"},
        )
    if sniff_image_file(str(path)) is not None:
        # v1 ships the STUB, not ``ImageContent``. Carrying an image would
        # change ``ExpansionResult``'s shape, and that shape is frozen while
        # Slice A codes against it; ``read`` already returns images to an agent
        # that asks for one.
        return (
            f"[image, {size} bytes — not included. View it with read(path={shown!r}).]",
            {"bytes": str(size), "kind": "image"},
        )
    data = path.read_bytes()
    if b"\x00" in data[:_BINARY_SNIFF_BYTES]:
        # The SAME detection ``read`` uses at ``builtin.py:3441``. Metadata
        # only: bytes in a user message are tokens spent on noise, and under
        # compaction a user turn is long-lived.
        guessed = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return (
            f"[binary ({guessed}), {size} bytes — not included.]",
            {"bytes": str(size), "kind": "binary"},
        )
    text = data.decode("utf-8", errors="replace")
    if len(text) > limit:
        body, attributes = _shaped_text(path, text, shown)
        return body, {"bytes": str(size), **attributes}
    return text, {"bytes": str(size), "lines": str(len(text.splitlines()))}


def _shown(path: Path, cwd: str) -> str:
    """``path`` relative to ``cwd``, absolute only when it lies outside.

    Mirrors ``context_files.py:591-593``, the established shape for this exact
    element, and the design's §2.3 example (``path="src/app.py"``). Three
    reasons, and the first is the one that compounds: this string is PERSISTED
    and re-sent on every turn of the session, so an absolute path bills its
    length forever — measured here at 79 chars against 28 relative, per
    reference. It also writes the operator's home directory into a transcript
    that gets shared and replayed, and a relative path is what the model
    already sees everywhere else.

    ``ValueError`` (outside the base) falls back to absolute, exactly as the
    precedent does: for a path genuinely elsewhere the absolute form is the
    only honest answer, and the approval gate has already been consulted about
    it by the time this runs.
    """
    try:
        return str(path.relative_to(Path(cwd).resolve()))
    except ValueError:
        return str(path)


def _defuse(body: str) -> str:
    """Neutralize block markers appearing in a reference's BODY.

    A referenced file whose CONTENT contains ``</operator-references>`` closes
    the block early for :func:`_block_spans`, with the same consequence the
    attribute escaping above prevents: the tokens after it fall outside the
    recovered span and pass 2 re-expands them, reading files the operator never
    referenced. Reproduced before this fix — a file containing the close marker
    followed by ``@victim.txt`` made pass 2 pull in ``victim.txt``.

    ONLY the two marker sequences are touched, and nothing else about the body
    is stripped, reformatted or escaped. The model must read the file's content
    verbatim — that is the entire point of the feature, and a body that arrives
    HTML-escaped would be a worse defect than the one being fixed. A
    zero-width space inside each marker is invisible to a reader, keeps the
    text legible, and stops the literal sequence the scanner matches.
    """
    return body.replace(REFERENCE_BLOCK_CLOSE, _DEFUSED_CLOSE).replace(
        REFERENCE_BLOCK_OPEN, _DEFUSED_OPEN
    )


def _render(path: Path, typed: str, body: str, attributes: dict[str, str], cwd: str) -> str:
    """One ``<reference>`` element.

    ``typed=`` carries the token EXACTLY as written, mirroring
    ``render_invocation``'s ``invocation=`` attribute
    (``skills/invoke.py:198-201``) and for the same reason: the payload is what
    gets PERSISTED, so a resumed session replays this string and the surfaces
    recover the short row by reading the attribute back out of it.

    Attributes are escaped and the body is defused, because BOTH are attacker
    controlled — a path and a file's content are whatever is on disk. See
    :func:`_attribute` and :func:`_defuse` for the vector each one closes.
    """
    rendered = " ".join(f'{key}="{_attribute(value)}"' for key, value in attributes.items())
    head = (
        f'<reference path="{_attribute(_shown(path, cwd))}" '
        f'{_TYPED_ATTRIBUTE}{_attribute(typed)}"'
    )
    if rendered:
        head += " " + rendered
    return f"{head}>\n{_defuse(body)}\n</reference>"


async def _approved(
    path: Path,
    inside: bool,
    resolvable: bool,
    request_approval: ApprovalGate | None,
    job_id: str | None,
) -> bool:
    """Whether this reference may be read.

    ``request_approval=None`` means auto-approved. That is the documented
    contract CLI ``--yolo`` and headless tests rely on (``_check_approval``,
    ``builtin.py:1336-1351``), and routing through ``ask_approval`` is what
    keeps this ask identical in shape to a tool's — a second approval
    convention would reach a host scoping its answers with half the picture.
    """
    sensitive = _is_sensitive(path)
    if inside and not sensitive:
        return True
    if request_approval is None:
        return True
    return await ask_approval(
        request_approval, "read", _describe(path, inside, resolvable, sensitive), job_id
    )


async def expand_references(
    text: str,
    cwd: str,
    *,
    request_approval: ApprovalGate | None = None,
    job_id: str | None = None,
) -> ExpansionResult:
    """Resolve ``@path`` tokens in ``text`` and append their content.

    NEVER RAISES. Every failure degrades to ``ExpansionResult(text, False,
    [notice])`` — the same contract ``_discovered_skills`` and
    ``_expand_invocation`` hold, and the reason is that this sits on the submit
    path of every surface: a raised exception here does not lose a reference,
    it loses the user's message.

    When ``expanded`` is False the returned ``sent`` IS ``text``, by identity.
    """
    try:
        if not at_references_enabled():
            # `text` itself, not a copy: the identity guarantee is the whole
            # point of the kill switch — off means untouched.
            return ExpansionResult(text, False, [])
        return await _expand(text, cwd, request_approval, job_id)
    except Exception as exc:  # noqa: BLE001 — a submit path must never raise
        return ExpansionResult(text, False, [f"references could not be expanded: {exc}"])


async def _expand(
    text: str,
    cwd: str,
    request_approval: ApprovalGate | None,
    job_id: str | None,
) -> ExpansionResult:
    """The body :func:`expand_references` wraps. May raise; the caller degrades."""
    tokens = _reference_tokens(text)
    if not tokens:
        return ExpansionResult(text, False, [])

    notices: list[str] = []
    rendered: list[str] = []
    listed_only: list[str] = []
    seen: set[Path] = set()
    used = len(REFERENCE_BLOCK_OPEN) + len(REFERENCE_BLOCK_CLOSE) + len(_BLOCK_PREAMBLE)

    for token in tokens:
        path, inside, resolvable = _resolve_workspace_path(token.raw, cwd)
        if not resolvable or not path.exists():
            # THE GOVERNING RULE. Not a reference, so no block entry and no
            # change to the prose — this is the branch `@me` takes.
            notices.append(f"{token.typed} — no such path; sent as written")
            continue
        # Dedupe by RESOLVED path, so `@./README.md` and `@README.md` are one
        # reference. Both tokens stay in the prose: the user wrote them, and
        # the model reads the sentence, not the block.
        if path in seen:
            continue
        seen.add(path)
        if not await _approved(path, inside, resolvable, request_approval, job_id):
            notices.append(f"{token.typed} — not included; approval declined")
            continue
        # One display path per reference, resolved once and used by BOTH the
        # element's ``path=`` attribute and every ``read(path=...)`` pointer
        # inside its body, so the two can never name different addresses.
        shown = _shown(path, cwd)
        try:
            if path.is_dir():
                body, attributes = _directory_payload(path)
            else:
                body, attributes = _file_payload(
                    path, path.stat().st_size, INTERNAL_READ_LIMIT_CHARS, shown
                )
        except OSError as exc:
            notices.append(f"{token.typed} — could not be read ({exc.strerror or exc})")
            continue
        element = _render(path, token.typed, body, attributes, cwd)
        if used + len(element) > BLOCK_LIMIT_CHARS:
            # Past the whole-block cap the reference is named, not carried.
            # Naming it beats dropping it silently: the model can `read` a path
            # it has been told about.
            listed_only.append(shown)
            continue
        used += len(element) + 2
        rendered.append(element)

    if not rendered and not listed_only:
        return ExpansionResult(text, False, notices)

    block = [REFERENCE_BLOCK_OPEN, _BLOCK_PREAMBLE, *rendered]
    if listed_only:
        block.append(
            f"[the reference block reached its {BLOCK_LIMIT_CHARS}-character cap; "
            "these paths are named but not included — read them if you need them]\n"
            + "\n".join(listed_only)
        )
    block.append(REFERENCE_BLOCK_CLOSE)
    return ExpansionResult(text + "\n\n" + "\n\n".join(block), True, notices)


__all__ = [
    "AT_REFERENCES_ENV",
    "BLOCK_LIMIT_CHARS",
    "REFERENCE_BLOCK_CLOSE",
    "REFERENCE_BLOCK_OPEN",
    "SCAN_CANDIDATE_LIMIT",
    "ExpansionResult",
    "at_references_enabled",
    "at_token",
    "expand_references",
    "scan_directory",
    "split_token",
]
