"""The human sentences for a move's working set (design §11.2, §12), rendered from keys.

WHY A MODULE OF ITS OWN: the CLI prompt, the CLI receipt and the TUI confirmation
screen show the SAME summary, and the core ``working_set`` block deliberately carries
no English beyond paths and commands (design §3.4) — every sentence is a catalogue key
in ``cli.sessions_move_workset``. One renderer keeps the three surfaces from drifting.

Rendering uses ``render_resolved`` (the language the config resolves at call time),
because the CLI is a one-shot process whose active locale is never moved, and the TUI
calls this from its worker thread with the same resolution.

``carries_files`` decides the wording, never the caller: PR A's owner answers
``False`` ("the move does not bring them yet; they stay on this device"), the transfer
half flips it to ``True`` ("sends its working files there"). A surface that hardcoded
either sentence would lie in one of the two builds.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from local_operator.i18n.keys import cli_sessions_move_workset as _ws
from local_operator.i18n.messages import Msg, render_resolved

#: Reason code → message builder. A code this build does not know renders as the code
#: itself (an older client reading a newer owner's block), never as a blank.
_REASONS = {
    "contended": _ws.reason_contended,
    "outside_home": _ws.reason_outside_home,
    "unmerged": _ws.reason_unmerged,
    "in_progress_operation": _ws.reason_in_progress_operation,
    "submodule": _ws.reason_submodule,
    "lfs": _ws.reason_lfs,
    "history_secret": _ws.reason_history_secret,
    "history_unscannable": _ws.reason_history_unscannable,
    "over_cap": _ws.reason_over_cap,
    "destination_conflict": _ws.reason_destination_conflict,
    "destination_cannot_clone": _ws.reason_destination_cannot_clone,
    "unsupported_by_peer": _ws.reason_unsupported_by_peer,
    "not_transferred": _ws.reason_not_transferred,
    "unsafe_path": _ws.reason_unsafe_path,
    "insufficient_space": _ws.reason_insufficient_space,
}

#: How many withheld / left-behind rows a summary prints before it stops listing. The
#: block itself is complete; this only bounds a terminal (a repo with hundreds of
#: secret-shaped fixtures would otherwise scroll the prompt off screen).
_LIST_LIMIT = 20


def _member(block: Mapping[str, Any], key: str) -> dict[str, Any]:
    """``block[key]`` when it is an object, else ``{}`` — a block from an older or
    newer owner may omit or reshape any member, and a summary must still render."""
    value = block.get(key)
    return value if isinstance(value, dict) else {}


def _t(message: Msg) -> str:
    return render_resolved(message)


def _size(count: int) -> str:
    from local_operator.i18n import resolve
    from local_operator.i18n.format import format_bytes

    return format_bytes(int(count), resolve.resolve_language())


def reason_text(code: str) -> str:
    builder = _REASONS.get(code)
    return _t(builder()) if builder is not None else code


def device_label(block: Mapping[str, Any], fallback: str = "") -> str:
    destination = _member(block, "destination")
    return str(destination.get("name") or destination.get("device_id") or fallback)


def source_label(block: Mapping[str, Any], fallback: str = "") -> str:
    """The device that KEEPS the files (the owner). Never "this device": a recall's
    surfaces run on the destination, where that phrase names the wrong machine."""
    source = _member(block, "source")
    return str(source.get("name") or source.get("device_id") or fallback)


def summary_lines(block: Mapping[str, Any], *, session: str, device: str = "") -> list[str]:
    """The confirmation summary: warning, counts, one line per worktree, withheld
    files, what is never copied, and the closing advice (design §11.2 text shape)."""
    state = str(block.get("state") or "")
    label = device_label(block, device)
    keeper = source_label(block, "?")
    if state == "unknown":
        failure = _member(block, "failure")
        return [
            _t(
                _ws.unknown(
                    source=keeper,
                    step=str(failure.get("step") or "?"),
                    repo=str(failure.get("repo") or "?"),
                )
            )
        ]
    if state == "unsupported":
        owner = _member(block, "owner")
        return [
            _t(_ws.unsupported(device=str(owner.get("name") or owner.get("device_id") or label)))
        ]
    worktrees = [row for row in block.get("worktrees") or [] if isinstance(row, dict)]
    if not worktrees:
        return [_t(_ws.none())]
    sends = bool(block.get("carries_files"))
    totals = _member(block, "totals")
    lines = [
        _t(
            _ws.warning_sends(session=session, device=label)
            if sends
            else _ws.warning_stays(session=session, device=label, source=keeper)
        ),
        "  "
        + _t(
            _ws.brings_summary(
                worktrees=len(worktrees),
                commits=sum(int(row.get("unpushed_commits") or 0) for row in worktrees),
                files=sum(
                    int(row.get("staged") or 0)
                    + int(row.get("unstaged") or 0)
                    + int(row.get("untracked") or 0)
                    for row in worktrees
                ),
                size=_size(int(totals.get("bytes") or 0)),
            )
        ),
    ]
    for row in worktrees:
        path = str(row.get("path") or "")
        if row.get("state") == "left_behind":
            lines.append(
                "    "
                + _t(_ws.worktree_left(path=path, reason=reason_text(str(row.get("reason") or ""))))
            )
            continue
        lines.append(
            "    "
            + _t(
                _ws.worktree_line(
                    path=path,
                    branch=str(row.get("branch") or "HEAD"),
                    unpushed=str(int(row.get("unpushed_commits") or 0)),
                    staged=str(int(row.get("staged") or 0)),
                    unstaged=str(int(row.get("unstaged") or 0)),
                    untracked=str(int(row.get("untracked") or 0)),
                )
            )
        )
    excluded = [item for item in block.get("excluded") or [] if isinstance(item, dict)]
    for item in excluded[:_LIST_LIMIT]:
        why = str(item.get("why") or "")
        path = str(item.get("path") or "")
        lines.append(
            "  "
            + _t(
                _ws.withheld_history(path=path)
                if why.startswith("history_")
                else _ws.withheld_secret(path=path)
            )
        )
    left = [item for item in block.get("left_behind") or [] if isinstance(item, dict)]
    for item in left[:_LIST_LIMIT]:
        recipe = _member(item, "recipe")
        command = str(recipe.get("run") or "")
        if command:
            lines.append(
                "  " + _t(_ws.leaves_behind(path=str(item.get("path") or ""), command=command))
            )
    lines.append(
        _t(
            _ws.advice_sends(device=label)
            if sends
            else _ws.advice_stays(source=keeper, device=label)
        )
    )
    return lines


def prompt_text(block: Mapping[str, Any], *, session: str, device: str = "") -> str:
    label = device_label(block, device)
    if block.get("carries_files"):
        return _t(_ws.prompt_sends(device=label))
    return _t(_ws.prompt_stays(session=session, device=label, source=source_label(block, "?")))


def receipt_lines(block: Mapping[str, Any], *, source: str) -> list[str]:
    """What the receipt says about the working set after the move. ``source`` is the
    device the files stayed on (the move's ``from_device`` name)."""
    if block.get("transferred") is not False:
        return []
    consent = str(block.get("consent") or "")
    if consent == "not_applicable_keep":
        return [
            _t(_ws.receipt_keep(device=device_label(block), source=source_label(block, source)))
        ]
    if consent == "without":
        return [_t(_ws.receipt_without(device=source))]
    count = sum(
        1
        for row in block.get("worktrees") or []
        if isinstance(row, dict) and row.get("state") == "left_behind"
    )
    return [_t(_ws.receipt_not_transferred(count=count, device=source))]
