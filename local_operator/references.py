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

Two callers expand: the TUI's aside worker — the one model call that never
reaches :meth:`Session.prompt` — and :meth:`Session.prompt` itself, the SINGLE
expansion site for every other surface (CLI, headless, server, scheduler,
mobile, subagent), which is what gives them the feature with zero per-surface
work. That makes IDEMPOTENCE a hard requirement rather than a nicety —
``expand_references(expand_references(t).sent).expanded is False``. Were it
true only by luck, every FORWARDED message would carry a doubled block.

It rests on TWO mechanisms, and both are load-bearing. :func:`_block_spans`
skips any ``@`` sitting inside an already-emitted block, AND every token a pass
CONSUMED is named inside that block with a ``typed=`` attribute that
:func:`_already_expanded` reads back. The second half is what the first cannot
supply: a token can be consumed without its content being carried — the block
cap was reached, or it duplicated an earlier token — and such a token used to
leave no trace in the block at all. It was therefore invisible to pass 2 and
expanded AGAIN. Measured through the real TUI-then-``Session.prompt`` sequence
(the PRE-RULING one, back when the composer still expanded at its submit exits;
post-ruling ``prompt`` is the only site a composer draft reaches, so the double
pass arises via FORWARDING — a subagent launch re-prompting text that already
carries a block): a 63-character message reached 27,584 chars after pass 1 and
55,084 chars with TWO blocks after pass 2, from three ordinary 16,383-byte
files and no attacker.
So overflowed and deduplicated tokens are named as ``<listed>`` elements, and
the one case where naming them all cannot fit inside
:data:`BLOCK_LIMIT_CHARS` expands nothing at all (:func:`_too_many`) — a
verdict that is a pure function of the text, so pass 2 reaches it too.

NO PROVENANCE FLAG (D7)
-----------------------
There is no ``operator_typed=`` keyword and the protocol signature does not
widen. Threading one correctly would mean setting it at ten call sites and
deliberately leaving it unset at five more, where the DEFAULT is the safe
answer and every operator-facing site is an opt-in somebody must remember; a
missed site fails silently by just not working. ``serving.py:1883`` already
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

import asyncio
import errno
import heapq
import mimetypes
import os
import re
import stat
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple, TypeVar, cast

from local_operator.harness.approval import (
    ApprovalGate,
    ApprovalUnavailableError,
    ask_approval,
)

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
    _BINARY_PEEK_BYTES,
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

#: The return of whatever blocking call :func:`_off_loop` is handed.
_T = TypeVar("_T")

#: The name every abandoned-read thread carries, so a leaked one is
#: identifiable in a dump and assertable in a test.
_READ_THREAD_NAME = "lop-reference-read"


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
#: (``context_files.py:404``) supplies the exact string vocabulary matched
#: below, and ``_internal_read_limit`` (``builtin.py:2638-2654``) supplies the
#: per-call discipline — its docstring says why, "so the override can be set
#: after this module is imported". Env-only by design: neither precedent is in
#: the ``/settings`` registry, and that rule governs config keys.
AT_REFERENCES_ENV = "LOCAL_OPERATOR_AT_REFERENCES"

#: Whole-block ceiling, four tool results' worth. The per-reference caps bound
#: one file; this bounds a message that names twenty. References past it are
#: listed BY PATH ONLY with a line saying so, which keeps the worst case
#: bounded regardless of who typed the text or how many tokens they typed —
#: and under D7 the typer need not be the operator, which is why this bound
#: exists at all.
#:
#: HOW IT IS ACTUALLY HELD: every append goes through :class:`_Block`, which
#: charges the markers, the preamble, each element and the overflow tail
#: against one counter, and reserves the tail's worst case BEFORE carrying an
#: element. It was previously tracked for carried elements only, with the
#: preamble, the notice and the by-path-only tail appended afterwards and
#: unaccounted — so this was not a bound: 300 tokens × 35-char names measured
#: 42,576 chars (1.30×), 1000 × 35 measured 67,776 (2.07×) and 300 × 120
#: measured 67,164 (2.05×). When even naming the overflow will not fit, the
#: pass expands NOTHING rather than overrunning; see :meth:`_Block.list_only`
#: for why those two requirements genuinely collide at that scale.
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
#:
#: FOUR RULES, ALL SET MEMBERSHIP, and the uniformity is the point rather than
#: tidiness. Three of these were frozensets and the fourth was an inline
#: ``name.startswith(".env")`` with no set behind it — which matched ``.env``
#: and ``.env.local`` but NOT a name ENDING in ``.env``. Measured before the
#: fix: ``prod.env``, ``secrets.env``, ``config.env`` and ``workspace.env`` all
#: reached the model with their contents and no prompt. Design §2.7 claims
#: ``~/.credentials/workspace.env`` is caught by the directory part AND the
#: name, "two independent gates"; only the directory gate fired, so the stated
#: property was one gate — and this repo's own ``AGENTS.md`` names that exact
#: file as the live credential file. A gap hiding in a branch is invisible; a
#: gap in a frozenset is visible at the data, which is why the fourth rule now
#: has a set of its own.
SENSITIVE_NAMES = frozenset(
    {".env", ".netrc", ".npmrc", ".pypirc", "credentials", "id_rsa", "id_ed25519"}
)
#: ``.env`` sits here as well as in :data:`SENSITIVE_NAMES` because the two
#: rules answer different questions: the name set catches the file CALLED
#: ``.env``, this suffix set catches ``prod.env`` and ``workspace.env``.
SENSITIVE_SUFFIXES = frozenset({".pem", ".key", ".p12", ".pfx", ".keystore", ".env"})
#: A ``.env*`` PREFIX counts: ``.env.local`` holds the same class of secret
#: ``.env`` does, and its suffix is ``.local``, so neither set above sees it.
SENSITIVE_NAME_PREFIXES = frozenset({".env"})
SENSITIVE_DIR_PARTS = frozenset({".ssh", ".gnupg", ".credentials", ".aws", ".kube"})

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


def _sensitive_name(name: str) -> bool:
    """Whether a BARE entry name trips one of the three name-shaped rules.

    Split out from :func:`_is_sensitive` because the fourth rule (ancestor
    directories) is a property of the PARENT and is therefore constant across
    every entry in one listing — see :func:`scan_directory`, which evaluates it
    once instead of ten thousand times.

    ``os.path.splitext`` rather than ``Path.suffix`` so this costs no ``Path``
    construction: it agrees with ``Path.suffix`` on every name in the sets
    above (``.env`` -> ``''``, ``.env.local`` -> ``'.local'``, ``prod.env`` ->
    ``'.env'``), and constructing a ``Path`` per entry measured 0.74 ms per
    2000 entries on the keystroke path.
    """
    # CASEFOLDED before every test, because the filesystem this runs on is
    # commonly case-insensitive and the deny-list was not. Verified on macOS
    # (APFS, the default case-insensitive configuration): `.ENV` and `.env`
    # report the SAME INODE, so `@.ENV` opened the real secret while matching
    # nothing in the lowercase sets — the gate was never consulted.
    # `WORKSPACE.ENV` took the same route past the suffix rule. NTFS is
    # case-insensitive by the same default, though that was not tested here.
    #
    # The candidate is folded rather than the sets: the sets are already
    # lowercase, and folding them at definition would read as if the DATA had
    # changed when what changed is the COMPARISON.
    folded = name.casefold()
    if folded in SENSITIVE_NAMES:
        return True
    if any(folded.startswith(prefix) for prefix in SENSITIVE_NAME_PREFIXES):
        return True
    return os.path.splitext(folded)[1] in SENSITIVE_SUFFIXES


def _is_sensitive(path: Path) -> bool:
    """Whether ``path`` trips the deny-list, and therefore must be asked about."""
    if _sensitive_name(path.name):
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

    The rows alone: :func:`scan_directory_report` is the same scan plus the
    number of entries :data:`SCAN_CANDIDATE_LIMIT` kept out, and this is the
    first element of that call. It is a separate name rather than a keyword
    argument because the great majority of callers — the expansion path, the
    approval descriptions, the tests — have no use for the count, and a
    required second return value would have made every one of them unpack a
    tuple to throw half of it away.
    """
    rows, _unlisted = scan_directory_report(directory, cwd)
    return rows


def scan_directory_report(directory: str, cwd: str) -> tuple[list["ArgumentChoice"], int]:
    """One directory's listable entries AND how many the cap kept out.

    ``(rows, unlisted)``. ``unlisted`` is the number of listable entries the
    directory holds beyond :data:`SCAN_CANDIDATE_LIMIT` — 0 for every directory
    under the cap, and the reason a 2500-entry directory's picker can say how
    many it is not showing instead of reporting the capped set as if it were the
    whole directory (design round 1, D6).

    It is EXACT, not an estimate, and it costs no extra syscall: the cap is
    applied to ``eligible`` after the cheap pass has already enumerated every
    listable entry, so ``len(eligible)`` is a number this function had in hand
    and dropped.

    Drop-in for :func:`scan_directory` on the three questions it answers — one
    level, synchronous, never raises.

    SYNCHRONOUS AND ONE LEVEL, and both halves are load-bearing. This runs from
    ``_sync_picker`` on EVERY keystroke (``editor.py:3101``) and every buffer
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
    # Same pattern and same stated reason as ``harness/rows.py:320-325``.
    from local_operator.tui.autocomplete import ArgumentChoice

    path, _inside, resolvable = _resolve_workspace_path(directory or ".", cwd)
    if not resolvable:
        return [], 0
    try:
        rules = [("", _load_ignore_rules(path, ""))]
        # The CHEAP pass: stream the listing and keep only what survives the
        # predicates that cost no syscall. Not sorted here — see the cap below.
        eligible: list[tuple[str, bool, os.DirEntry[str]]] = []
        with os.scandir(path) as scan:
            for entry in scan:
                name = entry.name
                # The walker's own exclusion vocabulary, imported rather than
                # retyped so ``@`` agrees with ``grep`` and ``glob`` about what
                # is worth showing. A second copy of these names is the drift
                # defect this repo names repeatedly.
                if name in _GREP_PRUNE_DIRS or name.startswith("."):
                    continue
                try:
                    # ``DirEntry.is_dir(follow_symlinks=False)`` reads the
                    # ``d_type`` the kernel already returned with the listing,
                    # so classification costs ZERO extra syscalls — the
                    # reasoning is written out at ``builtin.py:5415-5425``,
                    # where ``iterdir`` + per-entry ``Path`` predicates paid
                    # three stat(2) calls per entry.
                    is_dir = entry.is_dir(follow_symlinks=False)
                except OSError:
                    continue
                # TWO arguments, not three: ``_ignored`` does not read a directory
                # flag — the directory-only distinction lives in the COMPILED rule
                # (``_IgnoreRule.dir_only``), so this call site was computing one for
                # it and would have been the one left behind when the parameter was
                # removed (QA round 1, Q-1: the three-arg call survived that change
                # here and broke the ``@`` picker). ``is_dir`` is still read below, so
                # nothing else about this loop changes.
                if _ignored(name, rules):
                    continue
                eligible.append((name, is_dir, entry))
    except OSError:
        # A missing, unreadable or racing directory is an empty list, never an
        # exception: the caller is a keystroke handler and Textual turns an
        # escaped error into a full-screen crash.
        return [], 0

    # THE CAP BOUNDS THE EXPENSIVE WORK, not just the list length. The previous
    # shape sorted every entry and then `break`ed at the cap, so the ordering
    # cost scaled with the directory while the cap only trimmed the result.
    # `nsmallest` is O(n log k) and, more importantly, hands back exactly the
    # rows that will be built — so the per-row `stat` below runs at most
    # SCAN_CANDIDATE_LIMIT times no matter how large the directory is. That is
    # the bound `test_reference_scan.py` asserts structurally, by counting
    # syscalls rather than by timing the call.
    #
    # Alphabetical, matching the previous behaviour exactly: `nsmallest` on the
    # name key returns the same rows the old sort-then-truncate did, so the
    # picker's ordering is unchanged.
    if len(eligible) > SCAN_CANDIDATE_LIMIT:
        kept = heapq.nsmallest(SCAN_CANDIDATE_LIMIT, eligible, key=lambda item: item[0])
        # What the cap cost, counted HERE because this is the last point at
        # which the whole enumeration is still in hand. The picker's overflow
        # row is the only consumer, and without this number it reported the
        # capped set as if it were the directory: a 2500-entry directory read
        # ``… 1992 more`` and never mentioned the 500 it had not even looked at
        # (design round 1, D6).
        unlisted = len(eligible) - len(kept)
    else:
        kept = sorted(eligible, key=lambda item: item[0])
        unlisted = 0

    # Hoisted out of the loop because it is a property of the PARENT and so is
    # constant for every entry in one listing. Inside the loop it rebuilt a
    # ``Path`` per entry to ask the same question 2000 times: measured 3.94 ms
    # per 2000 entries, against 0.13 ms for the ignore check beside it, on a
    # path with a 16.7 ms frame budget.
    under_sensitive_dir = bool(SENSITIVE_DIR_PARTS.intersection(path.parts))

    rows = [
        ArgumentChoice(
            # Trailing ``/`` on a directory, matching ``_list_dir_entries``
            # (``builtin.py:3272-3278``) so one listing convention serves
            # both the picker and the expanded payload.
            name=name + ("/" if is_dir else ""),
            detail=_entry_detail(entry, is_dir),
            alert=under_sensitive_dir or _sensitive_name(name),
        )
        for name, is_dir, entry in kept
    ]
    return rows, unlisted


def reference_resolves(query: str, cwd: str) -> bool:
    """Whether ``@query`` names something that EXISTS — the composer's ink gate.

    THE RESOLVER'S OWN QUESTION, asked the same way, because the composer's ink
    is a promise about what happens at submit: :func:`expand_references` calls a
    token whose path does not resolve PROSE and sends it as written
    (``@me — no such path``). Painting that token as a reference would be a
    false positive on precisely the token the Q-2 fix exists to keep as prose,
    and a highlight that lies is worse than no highlight — it is the same
    "structured token, not text" claim ``text-area--at-reference`` makes, made
    about text the operator is not, in fact, referencing.

    THE SAME THREE PARTS as one entry of :func:`_resolve_tokens`, in the same
    order, so the ink and the expansion cannot drift:

    1. the kill switch, read per call — with ``@`` expansion off, no token is a
       reference and none of them gets reference ink;
    2. :func:`_resolve_workspace_path`, whose ``resolvable`` is the first half of
       the governing rule;
    3. :func:`_kind_of`, which is one ``stat`` answering ``is_dir``/``is_file``,
       the second half.

    One ``stat`` per call, and the caller memoizes per frame, so this is the
    same order of cost as the ``scandir`` the picker already runs per keystroke
    — measured at 0.04-0.07 ms — rather than a new budget.

    ``OSError`` is ``False``, matching the resolver rather than the operating
    system's opinion: an unstatable path takes the ``could not be read``
    branch there, which is a notice and therefore prose, so it is not a
    reference here either. The one deliberate difference is that a directory is
    not required to be LISTABLE — a listing can be empty and the token is still
    a reference, which is exactly the distinction :func:`scan_directory` draws
    for the approval gate.
    """
    if not query or not at_references_enabled():
        return False
    try:
        path, _inside, resolvable = _resolve_workspace_path(query, cwd)
        if not resolvable:
            return False
        is_dir, is_file = _kind_of(path)
    except OSError:
        return False
    return is_dir or is_file


#: The attribute carrying the token as typed, named once because two places
#: depend on the exact spelling: :func:`_render` writes it and
#: :func:`_already_expanded` reads it back to hold idempotence.
_TYPED_ATTRIBUTE = 'typed="'

#: One ELEMENT HEAD — the whole single-line ``<reference …>`` / ``<listed …>``
#: tag :func:`_render` writes, attributes and all.
#:
#: It is line-anchored and spelled from :data:`_TYPED_ATTRIBUTE` rather than
#: hard-coded, and its whole job is to stop :func:`_already_expanded` reading a
#: ``typed=`` out of a reference BODY. A body is the file's content, verbatim
#: apart from :func:`_defuse`'s two markers, so a body that merely QUOTED
#: ``typed="@x"`` — ordinary text in a config, a test fixture, or this
#: feature's own source — read back as "already expanded": a token newly added
#: beside it was dropped silently, with no notice and no read. Reproduced:
#: ``read @forge.txt`` then ``<pass 1> and also check @secret.txt`` expanded
#: ``False`` with ``notices == []``.
#:
#: ATTRIBUTE ORDER is contractual here (``path=`` then ``typed=``), which is
#: exactly what :func:`_render` emits. ``[^"]*`` per value is sound because
#: :func:`_attribute` escapes ``"`` to ``&quot;``, and ``[^>]*`` before the
#: closing ``>`` for the same reason with ``>`` and ``&gt;``.
#:
#: RESIDUAL, accepted: a body line that spells a complete element head verbatim
#: is still read as one. Closing that needs the body escaped or defused (see
#: :func:`_defuse` on why neither is acceptable — it is the file's content), and
#: its cost is one suppressed token in the same message, not a doubled block.
_ELEMENT_HEAD_RE = re.compile(
    r'^<(?:reference|listed) path="[^"]*" ' + re.escape(_TYPED_ATTRIBUTE) + r'([^"]*)"[^>]*>$'
)

#: RESIDUAL HAZARD, accepted deliberately: a marker copied out of the payload
#: does not match the file on disk. Verified — ``grep -F`` for the copied form
#: returns rc=1 against the source file, while the ZWSP-stripped form returns
#: rc=0. It fires only on a file whose body contains the literal markers (this
#: feature's own source and docs, a transcript quoting one), and every
#: alternative is worse: escaping the body changes every file's content, and
#: leaving the marker raw is the injection this defuses. Recorded because the
#: next reader will otherwise meet it as a bug.
#:
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

#: The notice introducing the by-path-only tail, as a template over the block's
#: ACTUAL size. Sized into the budget UP FRONT by :class:`_Block` rather than
#: appended afterwards — see that class for the measured overshoot the old
#: "append and hope" shape produced.
#:
#: It reports the real number instead of asserting the cap was reached, because
#: the fixed-text version was a LIE in the common case: a message naming one
#: real file and several hundred nonexistent ones emitted a 343-character block
#: — 1.0% of the cap — under a notice saying the 32,768-character cap had been
#: reached. A path can be named rather than carried because the block truly
#: filled up OR because room had to be kept for the other paths still to be
#: named, and the operator cannot act on the difference unless told which.
_OVERFLOW_NOTICE_TEMPLATE = (
    "[not every referenced path could be included; the reference block holds "
    "{used} of its {limit}-character budget. These paths are named but not "
    "included — read them if you need them]"
)

#: The widest the notice can render, which is what :class:`_Block` CHARGES. The
#: real notice is never longer, so the charge is an upper bound and the cap
#: stays a bound; charging the exact string is impossible because the number it
#: contains is not known until every element has been decided.
_OVERFLOW_NOTICE_MAX_LEN = len(
    _OVERFLOW_NOTICE_TEMPLATE.format(used=BLOCK_LIMIT_CHARS, limit=BLOCK_LIMIT_CHARS)
)

#: The separator between block elements, counted with each element because an
#: element is never appended without it.
_BLOCK_JOIN = "\n\n"

#: The body of a ``<listed>`` element. A path that was NAMED carries no content
#: by definition, but it still renders through :func:`_render` so the escaping
#: cannot be skipped for it — which is exactly the bug the raw tail had.
_LISTED_BODY = "not included — read this path if you need it"


class _Token(NamedTuple):
    """One candidate reference found in a submitted message."""

    typed: str
    raw: str


class _Resolved(NamedTuple):
    """One token after its kind is known, and what naming it would COST.

    Exists so the tail reservation can be computed from the tokens that can
    actually spend block characters. Resolution and carrying are separate
    passes because a reservation made during the first token must already know
    how many of the LAST tokens are real — a fact a single interleaved pass
    does not have, which is how a real file's body came to be dropped to pay
    for paths that did not exist.

    ``notice`` set means the token is prose: it never reaches the block, so it
    costs nothing and is excluded from the reservation. The notice is carried
    rather than emitted during resolution so the caller can append it at the
    token's own position and keep notice order identical to the typed order.

    ``listed`` is the rendered ``<listed>`` element, precomputed so the
    reservation sums each remaining token's REAL length instead of multiplying
    the current token's length by a count.
    """

    token: _Token
    path: Path
    inside: bool
    resolvable: bool
    is_dir: bool
    notice: str | None
    listed: str

    @property
    def chargeable(self) -> bool:
        """Whether this token can still spend characters inside the block.

        Prose cannot: it takes the governing rule's branch, emits a notice and
        returns. Counting it in the reservation is the phantom charge that
        dropped a real reference's body.

        A token that is later declined at the approval gate DOES count here.
        Its verdict is not known until pass 2 asks, and over-reserving by a
        decided-late token is safe in the direction that matters — the cap
        holds — whereas under-reserving would breach it.
        """
        return self.notice is None


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


def reference_block_spans(text: str) -> list[tuple[int, int]]:
    """Half-open ``[start, end)`` spans of every COMPLETE reference block.

    The block grammar as :func:`_Block.render` writes it, and all three parts
    are load-bearing for a caller deciding what to paint in a transcript row:

    - the open marker (a body cannot contain a literal one — :func:`_defuse`),
    - the block's own preamble line, which is what separates a block the
      resolver APPENDED from a marker the operator merely quoted or pasted,
    - the close marker.

    An UNCLOSED block reports nothing, deliberately: it is not provably a block
    (truncated history, a message cut mid-write), and the cost of keeping its
    text is a dangling marker on a history that was already truncated, while
    the cost of removing text on its say-so is the unanchored-``find`` bug that
    silently showed less than the operator typed (``harness/rows.py``).

    "Reports nothing" means the OPENER contributes no span — it does NOT mean
    the scan stops there. An unclosed opener used to pair with the NEXT block's
    closer, forging one span that swallowed the opener, everything between the
    two, and that whole block; the strip then deleted the operator's own prose,
    silently, which is precisely the R4 failure this function exists to prevent
    and the one branch the sentence above claimed was protected (review round 2,
    MINOR-1). The forged span needs a LATER complete block to exist at all, so
    the tell is an open marker between this opener and the closer a bare
    ``find`` would take: that closer belongs to the later block, not to this
    opener. Skip the opener and rescan from the next one.

    EVERY block, not just the last one. A message that was expanded twice —
    pass 2 adding a token to text that already carried a block, which is the
    ordinary FORWARDING shape — holds two, and a row painter that strips only
    the trailing one paints the whole first block's file content into the row
    (reproduced: a 12,965-character message painted a 6,496-character row).

    Public because the strip belongs on the harness side where both surfaces
    paint (``harness/rows.py``, design R5), while the block's grammar belongs
    here, with the code that writes it.
    """
    spans: list[tuple[int, int]] = []
    opener = REFERENCE_BLOCK_OPEN + _BLOCK_JOIN + _BLOCK_PREAMBLE
    cursor = 0
    while True:
        start = text.find(opener, cursor)
        if start == -1:
            return spans
        close = text.find(REFERENCE_BLOCK_CLOSE, start + len(opener))
        # An OPEN MARKER between this opener and that closer means this opener is
        # UNCLOSED and the closer belongs to the block that marker starts: a
        # bare `find` would forge ONE span across both, and the strip would
        # delete the operator's prose between them (review round 2, MINOR-1).
        # A body can never spell an open marker — `_defuse` neutralises it — so
        # this cannot fire on a well-formed block. Rescanning FROM the next
        # opener (not past it) is what lets that later block keep its own span,
        # and `start + len(REFERENCE_BLOCK_OPEN)` cannot re-find this one.
        nested = text.find(REFERENCE_BLOCK_OPEN, start + len(REFERENCE_BLOCK_OPEN))
        if close == -1 or (nested != -1 and nested < close):
            cursor = start + len(REFERENCE_BLOCK_OPEN)
            continue
        end = close + len(REFERENCE_BLOCK_CLOSE)
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
    (``harness/rows.py:320-325``), and it is why :func:`_render` writes the
    attribute at all. A token NEWLY added to already-expanded text still
    expands, which keeps a steered or edited draft working; after that pass it
    too is named in a block, so the property holds however many passes run.

    ELEMENT HEADS ONLY — never the bodies between them. That is the whole of
    :data:`_ELEMENT_HEAD_RE`'s job, and it is the difference between
    idempotence and a reference that silently disappears: a body quoting
    ``typed="…"`` used to answer for a token the operator had just added.
    """
    typed: set[str] = set()
    for start, end in spans:
        for line in text[start:end].splitlines():
            head = _ELEMENT_HEAD_RE.match(line)
            if head is not None:
                typed.add(_unattribute(head.group(1)))
    return typed


def _reference_tokens(text: str) -> tuple[list[_Token], int]:
    """``(tokens, suppressed)`` — candidate ``@`` tokens, already-resolved excluded.

    ``suppressed`` counts candidate tokens that fell INSIDE a block span, which
    is normally zero: a real block's tokens are recovered by their ``typed=``
    attribute and never reach the scan as candidates. It is non-zero when the
    operator's own message contains a block marker they typed or pasted —
    :func:`_block_spans` cannot tell that from a previous pass's block, and an
    UNCLOSED marker takes ``end=len(text)`` and swallows every token after it.

    That fails closed, which is why it is minor, but it used to fail SILENTLY,
    and under D7 the typer need not be the operator — so pasted text could
    disable references for the rest of a message with no trace. Every other
    non-expansion in this module emits a notice; the count is returned so this
    one can too.
    """
    skip = _block_spans(text)
    resolved = _already_expanded(text, skip)
    tokens: list[_Token] = []
    suppressed = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char != "@" or not is_boundary(text, index):
            index += 1
            continue
        if any(start <= index < end for start, end in skip):
            # Only a token with a QUERY would have been a candidate, so a bare
            # `@` inside a block is not a suppression worth reporting.
            line_stop = text.find("\n", index)
            line_stop = len(text) if line_stop == -1 else line_stop
            _end, query = _token_end(text[index:line_stop], 0)
            if query and text[index : index + _end] not in resolved:
                suppressed += 1
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
    return tokens, suppressed


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
    if not entries:
        # An empty directory still gets a BODY. A zero-length one renders as
        # ``<reference … entries="0">\n\n</reference>``, which reads as a
        # truncated payload rather than as the answer "there is nothing here" —
        # the attribute is one the model has to weigh, and every other empty
        # outcome in this module says so in prose.
        return "[this directory is empty]", {"entries": "0"}
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
    which is also what ``context_files._render_index_rows`` (``:564-598``) does
    for repo guidance.
    """
    lines = text.splitlines()
    head = text[:INTERNAL_READ_HEAD_CHARS]
    # COMPLETE lines only. ``len(head.splitlines())`` counts the line the
    # character cap cut in half as a whole one, so the footer below promised a
    # line it had shown only part of (for a head cut mid-first-line it reported
    # "the first 1 of 2 lines" on a body that had one complete line). The
    # footer's promise is what the model navigates by, so the count is of
    # terminators: a line is shown when its newline came along.
    head_lines = head.count("\n")
    # The head is a CHARACTER slice, so it normally ends MID-LINE, and counting
    # terminators alone then understates what the model actually holds: a
    # 20,000-character single-line file reported "the first 0 of 1 lines" while
    # 6,144 characters of that line were in the prompt (review round 2,
    # MINOR-3), which sends the model into a pointless re-`read`. Naming the
    # partial line is the honest form — it neither claims a complete line the
    # cap cut in half (the defect the count of terminators fixed) nor denies the
    # fragment that is really there.
    partial = bool(head) and not head.endswith("\n") and head_lines < len(lines)
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
    shown_lines = f"the first {head_lines} of {len(lines)} lines"
    if partial:
        shown_lines += f", plus part of line {head_lines + 1}"
    footer = (
        f"\n\n[shown: {shown_lines}. "
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
    if b"\x00" in data[:_BINARY_PEEK_BYTES]:
        # The SAME detection, and the same byte count, that ``read`` uses at
        # ``builtin.py:3804``: ``_BINARY_PEEK_BYTES`` is imported rather than
        # mirrored, so the one answer about what "binary" means cannot drift
        # into two. Metadata only: bytes in a user message are tokens
        # spent on noise, and under compaction a user turn is long-lived.
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


async def _off_loop(fn: Callable[..., _T], *args: object) -> _T:
    """Run a blocking filesystem call in a DAEMON thread and await its result.

    :func:`asyncio.to_thread` is the convention everywhere else in this
    codebase — but its executor threads are NON-DAEMON, and that is load
    bearing here in a way it is not at the other call sites. This module's
    whole reason for going off the loop is that a read can WEDGE forever (a
    FIFO with no writer, an NFS mount that never answers), and the only
    recovery is to abandon the thread. ``to_thread``'s thread then keeps the
    interpreter alive, and the join that hangs is ``asyncio.run``'s own:
    it calls ``loop.shutdown_default_executor()``, which is
    ``ThreadPoolExecutor.shutdown(wait=True)`` and blocks joining the worker
    still stopped in the kernel. Confirmed by a fault handler traceback —
    ``_do_shutdown`` -> ``concurrent.futures.thread.shutdown`` -> ``join``.
    (Those workers are also non-daemon, so interpreter shutdown would join
    them as well; ``asyncio.run`` simply gets there first.)

    Measured before this change, on an abandoned FIFO read: ``wait_for``
    regained control in 2.0 s and the process was STILL alive 30 s later with
    all its work done — a hang at exit, not just a leak. With a daemon thread
    the same run returned from ``asyncio.run`` at 2.0 s and the process exited.

    The abandoned thread still leaks for the life of the process; nothing can
    reclaim a thread stopped in the kernel. What changes is that the leak no
    longer outranks shutdown.

    ``call_soon_threadsafe`` rather than ``run_coroutine_threadsafe`` because
    the worker only needs to settle a future, and the ``done()`` guard is for
    the abandoned case \u2014 the awaiting task may be long gone when the read
    finally returns, and setting a cancelled future raises.
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[_T] = loop.create_future()

    def deliver(error: BaseException | None, result: object) -> None:
        # The awaiting task may have been cancelled while this thread was
        # blocked, and setting an already-settled future raises.
        if future.done():
            return
        if error is not None:
            future.set_exception(error)
        else:
            future.set_result(cast("_T", result))

    def run() -> None:
        # The outcome is passed as ARGUMENTS, never captured in a closure: an
        # `except ... as exc` name is deleted at the end of its block, so a
        # lambda closing over it raises `NameError` when the loop runs it. That
        # left the future permanently unsettled and the awaiting coroutine hung
        # forever — caught here by `test_expansion_never_raises` hanging rather
        # than failing.
        try:
            result = fn(*args)
        except BaseException as exc:  # noqa: BLE001 — re-raised on the awaiting side
            loop.call_soon_threadsafe(deliver, exc, None)
        else:
            loop.call_soon_threadsafe(deliver, None, result)

    threading.Thread(target=run, name=_READ_THREAD_NAME, daemon=True).start()
    return await future


def _file_payload_of(path: Path, limit: int, shown: str) -> tuple[str, dict[str, str]]:
    """:func:`_file_payload` with its ``stat`` done on the SAME thread.

    The ``st_size`` lookup is a blocking syscall like the read it sizes, so it
    belongs on the worker thread rather than on the event loop. One function to
    hand to :func:`asyncio.to_thread`, so the caller cannot offload the read and
    leave the ``stat`` behind.
    """
    return _file_payload(path, path.stat().st_size, limit, shown)


def _shown(path: Path, cwd: str) -> str:
    """``path`` relative to ``cwd``, absolute only when it lies outside.

    Mirrors ``context_files.py:659-661``, the established shape for this exact
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


def _render(
    path: Path,
    typed: str,
    body: str,
    attributes: dict[str, str],
    cwd: str,
    tag: str = "reference",
) -> str:
    """One ``<reference>`` element, or with ``tag="listed"`` one ``<listed>``.

    Both shapes go through this one function so the escaping cannot be skipped
    for one of them — which is precisely what happened to the overflow tail.
    See :func:`_render_listed`.

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
        f'<{tag} path="{_attribute(_shown(path, cwd))}" ' f'{_TYPED_ATTRIBUTE}{_attribute(typed)}"'
    )
    if rendered:
        head += " " + rendered
    return f"{head}>\n{_defuse(body)}\n</{tag}>"


def _render_listed(path: Path, typed: str, cwd: str) -> str:
    """One ``<listed>`` element — a path NAMED but not carried.

    An element rather than a bare line, and that is a correctness fix rather
    than a formatting preference. The overflow tail used to be
    ``"\\n".join(listed_only)`` of raw display paths, which broke two
    guarantees at once.

    IDEMPOTENCE (module docstring, guarantee 3). :func:`_already_expanded`
    recovers a consumed token from its ``typed=`` attribute, so a token named
    only as a bare line was invisible to pass 2 and expanded AGAIN. Measured
    through the real TUI-then-``Session.prompt`` sequence: a 63-character
    message became 27,584 chars after pass 1 and 55,084 chars with TWO blocks
    after pass 2. The docstring called that impossible. Carrying ``typed=``
    here is what makes it impossible in fact.

    INJECTION. The bare line was neither escaped nor defused, one line from
    :func:`_attribute`'s fix, so a filename spelling the close marker
    (``c</operator-references>d.txt``, legal on POSIX) forged a close marker
    inside the block \u2014 the exact vector ``_attribute`` exists to close, reached
    by the one path that skipped it. Going through :func:`_render`'s escaping
    closes it by construction rather than by a second remembered call.
    """
    return _render(path, typed, _LISTED_BODY, {}, cwd, tag="listed")


def _kind_of(path: Path) -> tuple[bool, bool]:
    """``(is_dir, is_file)`` for ``path`` — one ``stat``, answering both.

    ``Path.is_dir()``/``is_file()`` each stat, so asking separately paid twice
    for one answer.

    ABSENCE IS NOT AN ERROR, and the distinction is the governing rule. The
    errors ``Path.exists()`` itself swallows — ENOENT, ENOTDIR, EBADF, ELOOP —
    mean "nothing is there", so they return ``(False, False)`` and the caller
    leaves the token as prose. Everything else (``PermissionError`` above all)
    PROPAGATES, so an unstatable path reaches the caller's per-token handler
    instead of being reported as "no such path" — which would be a lie about a
    path that does exist, and would make an unreadable file indistinguishable
    from a typo.
    """
    try:
        mode = path.stat().st_mode
    except (FileNotFoundError, NotADirectoryError):
        return False, False
    except OSError as exc:
        if exc.errno in (errno.EBADF, errno.ELOOP):
            return False, False
        raise
    except ValueError:
        # An embedded NUL byte in the name is a `ValueError`, not an `OSError`
        # — the shape `media.sniff_image_file` documents catching for the same
        # reason. Nothing can be at such a path, so it is prose.
        return False, False
    return stat.S_ISDIR(mode), stat.S_ISREG(mode)


def _too_many(text: str, notices: list[str]) -> ExpansionResult:
    """Expand nothing: too many tokens to name them all inside the cap.

    The escape hatch for the collision :meth:`_Block.list_only` documents. It
    returns the caller's own string OBJECT, so guarantee 2 (``expanded is
    False`` implies ``sent is text``) holds, and the verdict is a pure function
    of the text, so pass 2 takes this same branch and guarantee 3 holds too.

    A notice rather than silence, because every other non-expansion in this
    module emits one and an operator who named 1000 paths is owed the reason.
    """
    return ExpansionResult(
        text,
        False,
        [
            *notices,
            f"too many references to include within the {BLOCK_LIMIT_CHARS}-character "
            "block cap; none were expanded — reference fewer paths",
        ],
    )


class _Block:
    """The reference block under construction, and its ONE budget.

    Every append goes through :meth:`fits`, which is the whole design. The
    previous shape tracked ``used`` for carried elements only and appended the
    preamble, the overflow notice and the by-path-only tail afterwards with no
    accounting \u2014 the budget was tracked in one place and spent in three.
    :data:`BLOCK_LIMIT_CHARS` was consequently not a bound at all. Measured:
    300 tokens \u00d7 35-char names produced 42,576 chars (1.30\u00d7 the 32,768 cap),
    1000 \u00d7 35 produced 67,776 (2.07\u00d7) and 300 \u00d7 120 produced 67,164 (2.05\u00d7).

    That cap is not cosmetic. Under D7 the typer need not be the operator (see
    :data:`AT_REFERENCES_ENV`'s neighbours and the module docstring), so an
    unbounded tail removes the containment the whole no-provenance-flag
    argument rests on.

    The tail's worst case is therefore RESERVED before any element is carried,
    not discovered afterwards: :meth:`fits` refuses a carried element that
    would leave no room for the overflow notice plus a ``<listed>`` entry for
    every token still outstanding. A reservation can only shrink as tokens are
    consumed, so a block that fits at the start still fits at the end.
    """

    def __init__(self, limit: int = BLOCK_LIMIT_CHARS) -> None:
        self._limit = limit
        self._elements: list[str] = []
        # The preamble and the two markers are spent the moment the block
        # exists, so they are charged at construction rather than at render.
        self._used = (
            len(REFERENCE_BLOCK_OPEN)
            + len(_BLOCK_JOIN)
            + len(_BLOCK_PREAMBLE)
            + len(_BLOCK_JOIN)
            + len(REFERENCE_BLOCK_CLOSE)
        )
        self._overflow: list[str] = []
        self._notice_charged = False
        # Where the notice sits in `_elements`. The slot is filled with a
        # placeholder when charged and replaced at render, because the notice
        # states the block's final size and that is not known until every
        # element has been decided.
        self._notice_at: int | None = None

    def _charge(self, element: str) -> None:
        self._elements.append(element)
        self._used += len(_BLOCK_JOIN) + len(element)

    def fits(self, element: str, reserved: int) -> bool:
        """Whether ``element`` can be CARRIED with ``reserved`` chars still due.

        ``reserved`` is the caller's worst case for the tokens not yet decided
        — one ``<listed>`` element each. The overflow notice is added here
        rather than by the caller, because whether it is still owed is this
        object's state and not the caller's: it is charged at most once.

        This reservation is what stops the last carried element from consuming
        the room the tail needs, which is how the cap was overshot before.
        """
        due = reserved
        if reserved and not self._notice_charged:
            due += len(_BLOCK_JOIN) + _OVERFLOW_NOTICE_MAX_LEN
        return self._used + len(_BLOCK_JOIN) + len(element) + due <= self._limit

    def carry(self, element: str) -> None:
        """Append a full ``<reference>``. Call only when :meth:`fits` said yes."""
        self._charge(element)

    def list_only(self, element: str) -> bool:
        """Name a path without carrying it; False when even THAT will not fit.

        Charged through the same counter as a carried element, including the
        one-off overflow notice, so the tail cannot escape the budget the way
        the appended-afterwards version did.

        THE FALSE RETURN IS NOT A DETAIL — it is where two of this module's
        guarantees genuinely collide. Guarantee 3 (idempotence) requires every
        token a pass consumed to be recoverable from the block, which costs at
        minimum a ``typed="@…"`` per token. :data:`BLOCK_LIMIT_CHARS` requires
        the block to fit in 32,768 characters. For 1000 tokens of 35 characters
        the minimum possible tail is ~45,000 characters, so at that scale the
        two requirements cannot BOTH hold: naming everything overruns the cap,
        and capping the tail leaves unnamed tokens that pass 2 re-expands.

        The caller resolves it by expanding NOTHING — see
        :func:`expand_references`. That keeps the cap absolutely, keeps
        idempotence (the decision is a pure function of the text, so pass 2
        reaches it again and also expands nothing), and fails closed with a
        notice rather than silently. Returning the verdict instead of raising
        keeps this object free of control flow that the ``never raises``
        contract would have to catch.
        """
        due = 0 if self._notice_charged else len(_BLOCK_JOIN) + _OVERFLOW_NOTICE_MAX_LEN
        if self._used + due + len(_BLOCK_JOIN) + len(element) > self._limit:
            return False
        if not self._notice_charged:
            # A placeholder of the CHARGED width: `render` replaces it with the
            # real sentence, which is never longer, so `_used` stays an upper
            # bound on what is emitted.
            self._notice_at = len(self._elements)
            self._charge(" " * _OVERFLOW_NOTICE_MAX_LEN)
            self._notice_charged = True
        self._charge(element)
        self._overflow.append(element)
        return True

    @property
    def elements(self) -> list[str]:
        """Everything the block will render, carried and listed alike.

        Empty means no token resolved, which is the caller's signal to return
        the operator's own string OBJECT untouched (guarantee 2).
        """
        return self._elements

    @property
    def overflowed(self) -> bool:
        """Whether anything was listed rather than carried."""
        return bool(self._overflow)

    def render(self) -> str:
        elements = list(self._elements)
        if self._notice_at is not None:
            elements[self._notice_at] = self._notice_for(elements)
        body = _BLOCK_JOIN.join([_BLOCK_PREAMBLE, *elements])
        return f"{REFERENCE_BLOCK_OPEN}{_BLOCK_JOIN}{body}{_BLOCK_JOIN}{REFERENCE_BLOCK_CLOSE}"

    def _notice_for(self, elements: list[str]) -> str:
        """The overflow notice, stating the block's ACTUAL size.

        The old fixed string asserted the cap had been reached, which was false
        whenever a path was listed to keep room for other paths rather than
        because the block filled: one real file among several hundred
        nonexistent tokens emitted 343 characters — 1.0% of the cap — under a
        notice claiming 32,768 had been used.

        The size is SELF-REFERENTIAL: the number is inside the string whose
        length it counts. Solved by iterating to a fixed point, reached as soon
        as the reported digit count stops changing — one round in practice,
        since only a digit-count change can move the length. The loop is
        bounded rather than `while True` because an unbounded fixed-point
        search on a render path is a hang waiting to happen; on the bound being
        exhausted it returns the last iterate, whose length is within a digit
        of correct and never exceeds the charged width, so the cap still holds.

        Whatever it returns is no longer than :data:`_OVERFLOW_NOTICE_MAX_LEN`,
        which is what `fits` and `list_only` charged: ``used`` cannot exceed
        ``limit``, so the number can never be wider than the one that sized the
        charge.
        """
        fixed = sum(
            len(_BLOCK_JOIN) + len(part)
            for index, part in enumerate(elements)
            if index != self._notice_at
        )
        fixed += (
            len(REFERENCE_BLOCK_OPEN)
            + len(_BLOCK_JOIN)
            + len(_BLOCK_PREAMBLE)
            + len(_BLOCK_JOIN)
            + len(REFERENCE_BLOCK_CLOSE)
            + len(_BLOCK_JOIN)
        )
        notice = _OVERFLOW_NOTICE_TEMPLATE.format(used=fixed, limit=self._limit)
        for _ in range(4):
            settled = _OVERFLOW_NOTICE_TEMPLATE.format(used=fixed + len(notice), limit=self._limit)
            if len(settled) == len(notice):
                return settled
            notice = settled
        return notice


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
    ``builtin.py:1352-1367``), and routing through ``ask_approval`` is what
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

    AND IT RETURNS, which is a separate claim and was the weaker one. "Never
    raises" is worthless if the coroutine can simply never finish, and a FIFO
    on the submit path did exactly that: the path exists, is not a directory,
    took the file branch, and ``read_bytes()`` blocked forever with no writer.
    The block was UNCANCELLABLE — ``asyncio.wait_for(..., timeout=4)`` never
    regained control, because the thread was stopped in the kernel rather than
    at an await — so the operator's message was lost in precisely the way this
    contract exists to prevent. Two mechanisms now stand behind the claim: only
    a regular file or a directory is read at all (:func:`_kind_of`), and the
    reads run via :func:`asyncio.to_thread`, so the residual stat-then-read
    race leaves the event loop responsive and recoverable instead of wedged.

    Non-terminating pathologies that remain are the ordinary ones every reader
    on this machine shares — an NFS mount that never answers, a disk that never
    completes a read — and they are now survivable rather than fatal to the
    loop.

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


async def _resolve_tokens(tokens: list[_Token], cwd: str) -> list[_Resolved]:
    """Decide what every token IS, before any of them is carried.

    Separated from carrying so the tail reservation can be computed from the
    tokens that can actually spend block characters — see :class:`_Resolved`
    and the reservation in :func:`_expand`.

    Reads NOTHING. Only the one ``stat`` per token that was already on this
    path, so splitting the loop adds no syscalls; the file bodies are still
    read in the second pass, and only for references that survive the cap and
    the approval gate. A token that is prose here never costs a read at all,
    exactly as before.

    Failures become a ``notice`` rather than an exception, because the caller's
    contract is per-token degradation: one unreadable path must not abandon the
    whole message.
    """
    entries: list[_Resolved] = []
    for token in tokens:
        path, inside, resolvable = _resolve_workspace_path(token.raw, cwd)
        # `Path.exists()` PROPAGATES `PermissionError` — it swallows only
        # ENOENT/ENOTDIR/EBADF/ELOOP — so an unstatable path used to escape past
        # the per-token handler here to the catch-all in `expand_references`
        # and abandon the WHOLE message: `@README.md and @noperm/s.txt`
        # expanded neither, though §3's governing rule is per-token
        # degradation. `is_file()`/`is_dir()` raise the same way, so the one
        # statement covers all three.
        #
        # `is_file()` rather than `exists()` is also the FIFO fix. A FIFO
        # exists, is not a directory, and therefore took the file branch, where
        # `read_bytes()` blocks forever with no writer — not a raise but a
        # WEDGE, on the submit path of every surface, and `asyncio.wait_for`
        # cannot cancel it because the thread is blocked in the kernel rather
        # than at an await. `expand_references` is documented never to raise;
        # it must also be able to return. A device, socket or FIFO is not a
        # file to include, so the governing rule applies: it is prose.
        notice: str | None = None
        is_dir = False
        try:
            # One `stat` for both questions, off the loop with the reads below.
            is_dir, is_file = await _off_loop(_kind_of, path)
        except OSError as exc:
            notice = f"{token.typed} — could not be read ({exc.strerror or exc})"
        else:
            if not resolvable or not (is_dir or is_file):
                # THE GOVERNING RULE. Not a reference, so no block entry and no
                # change to the prose — this is the branch `@me` takes.
                notice = f"{token.typed} — no such path; sent as written"
        entries.append(
            _Resolved(
                token=token,
                path=path,
                inside=inside,
                resolvable=resolvable,
                is_dir=is_dir,
                notice=notice,
                # Rendered even for prose (cheap, no I/O) so the reservation
                # can sum real lengths; `chargeable` is what excludes it.
                listed=_render_listed(path, token.typed, cwd),
            )
        )
    return entries


async def _expand(
    text: str,
    cwd: str,
    request_approval: ApprovalGate | None,
    job_id: str | None,
) -> ExpansionResult:
    """The body :func:`expand_references` wraps. May raise; the caller degrades."""
    tokens, suppressed = _reference_tokens(text)
    # A marker the operator typed or pasted reads as a previous pass's block, so
    # the tokens inside it are skipped. Failing closed is right; failing
    # silently is not — see `_reference_tokens`.
    suppressed_notices = (
        [
            f"{suppressed} reference"
            f"{'' if suppressed == 1 else 's'} not expanded: the message contains a "
            f"{REFERENCE_BLOCK_OPEN} marker, so that text is treated as an "
            "already-expanded block"
        ]
        if suppressed
        else []
    )
    if not tokens:
        return ExpansionResult(text, False, suppressed_notices)

    notices: list[str] = list(suppressed_notices)
    block = _Block()

    # PASS 1 — decide what each token IS, before anything is carried.
    #
    # The tail reservation has to know how many of the LATER tokens can still
    # spend block characters, and interleaving resolution with carrying does
    # not have that fact at the first token. It charged the tail for every
    # token not yet looked at, including ones that turn out to be prose and
    # cost nothing, and the phantom charge demoted a REAL reference from
    # `<reference>` to `<listed>` — its body never reached the model. Measured
    # with one real file: at 340 nonexistent tokens the body was dropped while
    # the emitted block was 343 chars, 1.0% of the 32,768 cap.
    entries = await _resolve_tokens(tokens, cwd)

    # The tail's worst case, as chars still owed by tokens not yet DECIDED.
    # Only a token that can reach `list_only` is counted: prose never enters
    # the block. Each entry contributes its own rendered length rather than a
    # count times the current token's length, so the reservation is the real
    # remainder and not an estimate of it.
    pending = sum(len(_BLOCK_JOIN) + len(entry.listed) for entry in entries if entry.chargeable)
    seen: set[Path] = set()
    # Paths the gate DECLINED, so a later spelling of the same path cannot be
    # named as if it were carried. See the dedupe branch below.
    declined: set[Path] = set()

    # PASS 2 — carry what fits, name what does not.
    for entry in entries:
        token = entry.token
        if entry.notice is not None:
            # Prose: no block entry and no change to the text. Emitted here
            # rather than in pass 1 so notices keep the operator's typed order.
            notices.append(entry.notice)
            continue
        path = entry.path
        # Decided now, so `reserved` below is what the OTHER pending tokens
        # still owe. The reservation can only shrink as tokens are consumed,
        # which is what lets `_Block.fits` treat it as a bound.
        pending -= len(_BLOCK_JOIN) + len(entry.listed)
        # Dedupe by RESOLVED path, so `@./README.md` and `@README.md` are one
        # reference. Both tokens stay in the prose: the user wrote them, and
        # the model reads the sentence, not the block.
        #
        # The duplicate is still NAMED, because a token that is silently
        # skipped is invisible to `_already_expanded` and expands again on pass
        # 2 — `@n.md and @./n.md` produced a doubled block. Naming it costs one
        # short `<listed>` element and makes guarantee 3 true for this path.
        #
        # UNLESS the gate already declined that same path. Dedupe is decided
        # before the approval verdict, so `check @.env and also @./.env` named
        # the second spelling as a `<listed>` element whose body reads "not
        # included — read this path if you need it": the block advertised the
        # very path the operator had just refused, in the model's own
        # vocabulary, and reaching the dedupe arm is not a reason to disclose
        # one. It degrades exactly as a declined token does — a notice, no
        # element — so the block is silent about the path either way.
        if path in seen:
            if path in declined:
                notices.append(f"{token.typed} — not included; approval declined")
                continue
            if not block.list_only(entry.listed):
                return _too_many(text, notices)
            continue
        seen.add(path)
        try:
            approved = await _approved(
                path, entry.inside, entry.resolvable, request_approval, job_id
            )
        except ApprovalUnavailableError as exc:
            # A gate that could not ask anyone declines THIS token and nothing
            # more — the same degradation a refusal gets, plus the reason in
            # the notice. The exception must never reach
            # ``expand_references``'s never-raise catch-all: that arm degrades
            # by ABORTING the whole expansion (``expanded=False``, the raw
            # message sent, no reference carried), so one outside or sensitive
            # ``@ref`` silently dropped every other reference in the same
            # message — the post-merge regression the typed raise introduced
            # (review of #1597).
            declined.add(path)
            notices.append(f"{token.typed} — not included; approval unavailable ({exc.reason})")
            continue
        if not approved:
            declined.add(path)
            notices.append(f"{token.typed} — not included; approval declined")
            continue
        # One display path per reference, resolved once and used by BOTH the
        # element's ``path=`` attribute and every ``read(path=...)`` pointer
        # inside its body, so the two can never name different addresses.
        shown = _shown(path, cwd)
        try:
            # OFF THE EVENT LOOP, matching `execute_read`'s precedent
            # (`builtin.py:3768`, `:3810`): every read here was inline, and
            # this sits on the submit path of every surface, so a slow disk
            # stalled the loop for the whole read. For ordinary files the
            # inline cost was minor (8 near-cap files measured 10.1 ms expand,
            # 13.2 ms max loop stall), so this is not a latency fix.
            #
            # It is a RECOVERABILITY fix for the residual TOCTOU window the
            # `is_file()` check above cannot close: the check and the read are
            # separate syscalls, so a path that is a regular file at the stat
            # and a FIFO at the read still blocks. Measured: inline, that block
            # is uncancellable — `asyncio.wait_for` never regains control
            # because the thread is stopped in the kernel, not at an await. In
            # a worker thread the same block leaves the loop responsive and
            # `wait_for` recovers in 3.0 s. The thread still leaks — nothing
            # can reclaim one stopped in the kernel — but it is a DAEMON
            # thread (`_off_loop`, not `asyncio.to_thread`), so the leak does
            # not also block interpreter shutdown the way the executor's
            # non-daemon thread did. A leaked daemon thread is recoverable; a
            # wedged event loop is not.
            if entry.is_dir:
                body, attributes = await _off_loop(_directory_payload, path)
            else:
                body, attributes = await _off_loop(
                    _file_payload_of, path, INTERNAL_READ_LIMIT_CHARS, shown
                )
        except OSError as exc:
            notices.append(f"{token.typed} — could not be read ({exc.strerror or exc})")
            continue
        element = _render(path, token.typed, body, attributes, cwd)
        # The tail still owed if every remaining token overflows. Reserving it
        # BEFORE carrying this element is what makes `BLOCK_LIMIT_CHARS` a
        # bound rather than a suggestion.
        if not block.fits(element, pending):
            # Past the whole-block cap the reference is named, not carried.
            # Naming it beats dropping it silently: the model can `read` a path
            # it has been told about, and pass 2 can see it was consumed.
            if not block.list_only(entry.listed):
                return _too_many(text, notices)
            continue
        block.carry(element)

    if not block.elements:
        return ExpansionResult(text, False, notices)

    return ExpansionResult(text + _BLOCK_JOIN + block.render(), True, notices)


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
    "reference_block_spans",
    "reference_resolves",
    "scan_directory",
    "scan_directory_report",
    "split_token",
]
