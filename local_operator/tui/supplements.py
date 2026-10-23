"""Turn supplements on the TUI — the surface's ONE mapping place (memo §2.8, §3.2).

The image-gen lane's ``tui/imagegen.py`` is the module this one sits beside, and
for the same reason: the whole vocabulary of the feature has exactly one
definition per surface, so a wire rename or a copy change is a one-line edit
here and no renderer changes. What lives here:

- :func:`preparing_line` — the loading line under the answer, whose lead is
  ``imagegen.progress_graphic`` (the shimmering canvas, settings-gated by the
  same machinery as every other animated surface). The memo's copy table shows
  ``◌ Preparing highlights… · Adjust… · Cancel``; the TUI's interactions are the
  ``/highlights`` slash commands (memo §3.2 T5 — ``ctrl+c`` is the turn
  interrupt and stays unbound), so the pointer-surface action pair is NOT
  painted here: a terminal line advertising "Cancel" that cannot be clicked is
  the D10 defect in the other direction. The copy that does apply is kept
  byte-identical to the table.
- :func:`display_text` — the ONE paint-boundary strip for every host-rendered
  string (names, titles, paths, the "N more" line). C0/C1 controls are removed
  and whitespace is collapsed, because these strings are authored by models
  and tools and are painted on every frame; the UI and native lanes reuse the
  same fixture set, so all four surfaces agree on what a display string is.
- The row FORMATTERS the block authors from: the ≤ 2 files one-liner, the
  "N more" text and the byte size. Pure functions, so they are pinned without
  a widget.

This module imports no widget code, deliberately: ``widgets/supplement_block.py``
imports this one, and the direction must stay one-way.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from rich.text import Text

from local_operator.ansi import strip_control_sequences
from local_operator.tui import imagegen as imagegen_mod

__all__ = [
    "COPY_CANCELLED",
    "COPY_FAILED",
    "COPY_PREPARING",
    "cause_phrase",
    "display_text",
    "file_row_parts",
    "file_size_text",
    "files_summary_line",
    "more_text",
    "preparing_line",
    "sound_row_text",
    "sound_rows",
    "sounds_more_text",
    "state_rank",
]

#: The fixed display copy (memo §2.8, round-1 design D7/D10; the failed line's
#: typed-cause variant is licensed there too — round-1 design D1/D3, 2026-10-17).
#: One spelling per line, byte-identical to the copy table's sentence up to the
#: pointer-surface action pair this surface does not paint (see the module
#: docstring) and up to the failed line's ` — <phrase>` suffix, whose word set
#: `_CAUSE_PHRASES` below owns for every surface.
COPY_PREPARING = "Preparing highlights…"
COPY_CANCELLED = "Highlights cancelled"
COPY_FAILED = "Couldn't prepare highlights"

#: The failed line's typed-cause phrases, one per allowlisted ``failure_kind`` (the
#: ``provider_<kind>`` vocabulary the row's ``cause`` carries). The state line says WHICH
#: wall was hit, because a Retry that will hit the same wall is not useful (operator,
#: 2026-10-17). Memo §2.8 licenses the variant, and THIS table is the one word set every
#: surface transcribes (the phone web's ``supplement.ts`` ports it verbatim). ``provider_quota``
#: says "limit reached", never "exhausted": the kind covers rate limits too, and a spent-quota
#: reading stops a user retrying a retryable wall (design round 1, D1). ``provider_unknown`` --
#: and any token this build does not know -- renders NO phrase: ``unknown`` folds ``aborted``
#: in, and inventing a failure mode for a catch-all is worse than the bare sentence (D2).
_CAUSE_PHRASES: dict[str, str] = {
    "provider_quota": "provider limit reached",
    "provider_auth": "provider sign-in needed",
    "provider_transient": "provider temporarily unavailable",
    "provider_timeout": "provider timed out",
    "provider_request": "provider refused the request",
    "provider_content": "provider blocked the content",
    "provider_context": "provider context limit",
}


def cause_phrase(cause: object) -> str:
    """The muted phrase for a failed row's typed ``cause``; ``""`` when absent/unknown."""
    return _CAUSE_PHRASES.get(str(cause), "") if cause else ""


#: How far along a state is, for "is this row newer than what we hold".
#: The journaled arms only (memo §2.4): live beats are never journaled, so a
#: read of the SAME version can still carry an EARLIER state than a beat the
#: surface already applied (``running`` held, ``decided`` read). The terminal
#: arms tie because one terminal row per version wins in the journal.
_STATE_RANK: dict[str, int] = {
    "decided": 0,
    "queued": 1,
    "running": 2,
    "cancelling": 3,
    "done": 4,
    "failed": 4,
    "cancelled": 4,
    "skipped": 4,
}


def state_rank(state: object) -> int:
    """The state's position in the lifecycle; an unknown word reads as 0.

    The order is a READER fact, not a wire one: it exists so a lazy read can be
    told apart from the live beats it races — the app applies a read only when
    ``(version, rank)`` is not older than what it holds.
    """
    return _STATE_RANK.get(str(state), 0)


def display_text(value: object) -> str:
    """One host-rendered string, safe to paint on every frame.

    Strips ANSI/C0/C1 control sequences (the shared
    :func:`~local_operator.ansi.strip_control_sequences`) and collapses every
    run of whitespace — including newlines, which survive sequence stripping by
    design — to a single space. A file name carrying a newline would otherwise
    split the row it is painted on; a name carrying an erase-display would
    repaint the terminal. Non-strings read as ``""`` so a malformed row paints
    nothing rather than raising on a render path.
    """
    if not isinstance(value, str):
        return ""
    return " ".join(strip_control_sequences(value).split())


def preparing_line(time_ms: float | None = None) -> Text:
    """The loading line: the shimmering graphic, then the fixed sentence.

    The lead is :func:`~local_operator.tui.imagegen.progress_graphic` with no
    fraction — the honest indeterminate read, and the same graphic the
    image-gen card paints while a call is running, so the app's two "working"
    surfaces share one vocabulary. It obeys ``display.shimmer`` /
    ``LOCAL_OPERATOR_NO_SHIMMER`` for free, through the machinery both
    surfaces ride.
    """
    line = Text()
    line.append_text(imagegen_mod.progress_graphic(None, time_ms))
    line.append(" ")
    line.append(COPY_PREPARING)
    return line


def files_summary_line(names: Sequence[str]) -> str:
    """The files-only one-liner (memo §2.8): the contents themselves.

    ``2 files: report.md, bench.csv`` — never the bare word "Highlights". Just
    the names: it is the one-line arm, and the paths would not fit two of them
    on a row. Names arrive already display-stripped from the caller's
    :func:`display_text` pass.
    """
    count = len(names)
    label = "1 file" if count == 1 else f"{count} files"
    return f"{label}: {', '.join(names)}"


def more_text(count: int) -> str:
    """The disclosure row's words: ``3 more files`` (memo §2.8)."""
    count = max(0, int(count))
    return f"{count} more file" if count == 1 else f"{count} more files"


def file_size_text(size_bytes: object) -> str:
    """A byte count as the short column the file rows carry.

    ``912 B`` / ``4.1 KB`` / ``1.2 MB``: one decimal on the scaled units,
    trailing ``.0`` dropped so a round number stays round. A value that is not
    a non-negative number reads as ``""`` (the column is simply absent), never
    as a wrong number.
    """
    if isinstance(size_bytes, bool) or not isinstance(size_bytes, (int, float)):
        return ""
    size = float(size_bytes)
    if size < 0:
        return ""
    if size < 1024:
        return f"{int(size)} B"
    for unit, scale in (("KB", 1024.0), ("MB", 1024.0**2), ("GB", 1024.0**3)):
        if size < scale * 1024 or unit == "GB":
            value = size / scale
            text = f"{value:.1f}".rstrip("0").rstrip(".")
            return f"{text} {unit}"
    return ""


def sound_row_text(sound: Mapping[str, object]) -> str:
    """One generated sound as the compact terminal row: ``♪ Storm swell · 2.0 s``.

    A TEXT ROW, NOT A PLAYER. A terminal cannot play audio, so the row states only
    what the artifact IS -- title and length -- and never grows a glyph or label
    implying it can be played (the block's D10 rule: no affordance a terminal cannot
    honour). It is painted for EVERY sound the turn generated, at every state that
    paints the block, because a sound the user generated exists whether or not this
    surface can render its waveform.

    ``title`` is display-stripped (it originates in a prompt or a provider), and an
    absent/zero duration is OMITTED rather than printed as ``0.0 s``: the trigger
    writes 0.0 when a result stated no usable length, so a printed zero would claim
    a fact nobody knows. One decimal keeps a row short and reads as a length.
    """
    title = display_text(sound.get("title")) or "Generated audio"
    duration = sound.get("duration_s")
    if isinstance(duration, bool) or not isinstance(duration, (int, float)):
        return f"♪ {title}"
    seconds = float(duration)
    if not seconds > 0:
        return f"♪ {title}"
    return f"♪ {title} · {seconds:.1f} s"


def sound_rows(sounds: Sequence[Mapping[str, object]]) -> list[str]:
    """Every sound's row, in generation order (the row's own order, never re-sorted)."""
    return [sound_row_text(sound) for sound in sounds]


def sounds_more_text(count: int) -> str:
    """The sound bound's disclosure row: ``12 more sounds`` (``more_text``'s twin).

    ``SOUNDS_MAX`` is 200, so this row is unreachable in practice -- and that is
    exactly why it must exist: the one case the bound bites is the case where the
    user would otherwise never learn that a sound they generated is missing.
    """
    count = max(0, int(count))
    return f"{count} more sound" if count == 1 else f"{count} more sounds"


def file_row_parts(file: Mapping[str, object]) -> tuple[str, str, str]:
    """``(name, path, size)`` for one callout row, each display-stripped.

    ``name`` is the short label and ``path`` the session-relative target the
    row fixes; a row whose name IS its path (a file at the session root)
    answers ``""`` for the path so the row does not print it twice. The caller
    decides what is clickable — this reader only says what the row says.
    """
    name = display_text(file.get("name"))
    path = display_text(file.get("path"))
    if not name:
        name = path
    if path == name:
        path = ""
    return name, path, file_size_text(file.get("size_bytes"))
