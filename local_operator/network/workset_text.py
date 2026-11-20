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

import re
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


#: C0/C1 control characters, DEL and ESC sequences. Stripped from EVERY rendered
#: line (security S3, round 1): paths, remotes, device names and branch names are
#: interpolated from the owner's tree and the mesh, and a ``\x1b[2K\r\x1b[1A``
#: inside one rewrote the lines above the CLI's y/N prompt.
_CONTROL = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b.?|[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean(text: str) -> str:
    """``text`` without terminal control characters or escape sequences."""
    return _CONTROL.sub("", str(text))


def _t(message: Msg) -> str:
    return clean(render_resolved(message))


#: ``workset.GIT_FAILURE_CAUSES`` -> the sentence naming git's error class (U6).
_CAUSES = {
    "not_a_repository": _ws.cause_not_a_repository,
    "dubious_ownership": _ws.cause_dubious_ownership,
    "index_corrupt": _ws.cause_index_corrupt,
    "object_corrupt": _ws.cause_object_corrupt,
    "git_missing": _ws.cause_git_missing,
    "timeout": _ws.cause_timeout,
    "git_error": _ws.cause_git_error,
}


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


def summary_lines(
    block: Mapping[str, Any], *, session: str, device: str = "", without_button: str = ""
) -> list[str]:
    """The confirmation summary: warning, counts, one line per worktree, withheld
    files, what is never copied, and the closing advice (design §11.2 text shape).

    ``without_button`` is the label of a "Move without the working files" control the
    surface shows next to this text (the TUI card). With it, the ``unknown`` and
    ``unsupported`` summaries point at that control instead of telling the user to
    re-run with ``--without-working-set`` (review U12, round 2): a flag is the CLI's
    way forward, and the card has a button for it.
    """
    state = str(block.get("state") or "")
    label = device_label(block, device)
    keeper = source_label(block, "?")
    if state == "unknown":
        failure = _member(block, "failure")
        repo = str(failure.get("repo") or "?")
        cause = _CAUSES.get(str(failure.get("cause") or ""), _ws.cause_git_error)
        # No internal ``step`` (review U6): git's reason class and the command that
        # shows it are what a person can act on.
        lead = (
            _ws.unknown_card(source=keeper, repo=repo, button=without_button)
            if without_button
            else _ws.unknown(source=keeper, repo=repo)
        )
        return [_t(lead), "  " + _t(_ws.unknown_cause(cause=_t(cause()), repo=repo))]
    if state == "unsupported" and _member(block, "update_device").get("role") == "destination":
        # The DESTINATION dropped the consent on its way (review R3): only updating it
        # helps, so the copy names it and offers no flag.
        peer = _member(block, "update_device")
        return [_t(_ws.unsupported_destination(device=str(peer.get("name") or label)))]
    if state == "unsupported":
        owner = _member(block, "owner")
        who = str(owner.get("name") or owner.get("device_id") or label)
        if without_button:
            return [_t(_ws.unsupported_card(device=who, button=without_button))]
        return [_t(_ws.unsupported(device=who))]
    worktrees = [row for row in block.get("worktrees") or [] if isinstance(row, dict)]
    mentioned = [str(p) for p in block.get("mentioned") or [] if isinstance(p, str)]
    if not worktrees:
        lines = [_t(_ws.none())]
        if mentioned:
            lines.append("  " + _t(_ws.mentioned(paths=", ".join(mentioned[:_LIST_LIMIT]))))
        return lines
    sends = bool(block.get("carries_files"))
    totals = _member(block, "totals")
    lines = [
        _t(
            _ws.warning_sends(session=session, device=label)
            if sends
            else _ws.warning_stays(session=session, device=label, source=keeper)
        )
    ]
    # THE COUNTS ARE THE PLANNED ROWS', the same definition as the block's ``totals``
    # that the JSON and the desktop dialog read (QA round 1, Q-1): a row already
    # ``left_behind`` (a history secret, a contended tree) is listed on its own line
    # below with its reason, never folded into the numbers.
    planned = [row for row in worktrees if row.get("state") == "planned"]
    if planned:
        lines.append(
            "  "
            + _t(
                _ws.brings_summary(
                    worktrees=len(planned),
                    commits=sum(int(row.get("unpushed_commits") or 0) for row in planned),
                    files=sum(
                        int(row.get("staged") or 0)
                        + int(row.get("unstaged") or 0)
                        + int(row.get("untracked") or 0)
                        for row in planned
                    ),
                    size=_size(int(totals.get("bytes") or 0)),
                )
            )
        )
    # ONE PROBE ANSWER PER REMOTE (review U7, round 1): three worktrees of one
    # repository printed the same cannot-fetch / HTTPS / remedy triple three times,
    # which was most of what pushed the TUI card's controls off screen. The answer is
    # per remote by construction (``workset_probe.probe_request``), so it is printed
    # under the first worktree that uses that remote.
    probed: set[str] = set()
    for row in worktrees:
        path = str(row.get("path") or "")
        if row.get("state") == "left_behind":
            lines.append(
                "    "
                + _t(
                    _ws.worktree_left(
                        path=path,
                        source=keeper,
                        reason=reason_text(str(row.get("reason") or "")),
                    )
                )
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
        remote = str(row.get("remote_url") or "")
        if remote in probed:
            continue
        probed.add(remote)
        lines.extend("      " + text for text in probe_row_lines(row, device=label))
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
    lines.extend("  " + text for text in probe_block_lines(block, device=label))
    if mentioned:
        lines.append("  " + _t(_ws.mentioned(paths=", ".join(mentioned[:_LIST_LIMIT]))))
    lines.append(
        _t(
            _ws.advice_sends(device=label)
            if sends
            else _ws.advice_stays(source=keeper, device=label)
        )
    )
    return lines


#: ``clone_check`` -> the sentence for a row the destination could not use. ``ok``,
#: ``no_remote`` and the unprobed default say nothing; ``unknown_destination`` is said
#: once for the block (``probe_block_lines``), never per row.
_CLONE_CHECKS = {
    "destination_cannot_fetch": _ws.probe_cannot_fetch,
    "destination_cannot_push": _ws.probe_cannot_push,
    "destination_cannot_clone": _ws.probe_cannot_clone,
}


def probe_row_lines(row: Mapping[str, Any], *, device: str) -> list[str]:
    """The destination's answer for one worktree (``workset_probe.apply_probe``):
    what it cannot do with the remote, the offered HTTPS spelling, the exact remedy."""
    out: list[str] = []
    remote = str(row.get("remote_url") or "")
    builder = _CLONE_CHECKS.get(str(row.get("clone_check") or ""))
    if builder is not None:
        out.append(_t(builder(device=device, remote=remote)))
    rewrite = _member(row, "https_rewrite")
    if rewrite.get("url"):
        out.append(_t(_ws.probe_https_rewrite(device=device, url=str(rewrite["url"]))))
    remedy = _member(row, "remedy")
    if remedy.get("command"):
        out.append(_t(_ws.probe_remedy(device=device, command=str(remedy["command"]))))
    return out


def probe_block_lines(block: Mapping[str, Any], *, device: str) -> list[str]:
    """Block-level destination warnings, all warn-only: the probe could not be asked,
    toolchains the destination lacks, Team Bridge enrolment the session needs."""
    probe = _member(block, "destination_probe")
    state = str(probe.get("state") or "")
    if state == "unknown_destination":
        return [_t(_ws.probe_unknown(device=device))]
    out: list[str] = []
    if state == "answered":
        # WHAT THE DESTINATION WAS TOLD (security S6, round 2): the probe runs during
        # the dry run, before the user has decided to trust the destination, so the
        # summary says that it was asked and what it was sent (``probe_request``).
        out.append(_t(_ws.probe_asked(device=device)))
    tools = [str(tool) for tool in probe.get("missing_tools") or [] if isinstance(tool, str)]
    if tools:
        out.append(_t(_ws.prereq_missing_tools(device=device, tools=", ".join(tools))))
    bridge = _member(probe, "team_bridge")
    if bridge.get("needed") and not bridge.get("enrolled"):
        out.append(_t(_ws.prereq_team_bridge(device=device)))
    return out


def prompt_text(block: Mapping[str, Any], *, session: str, device: str = "") -> str:
    label = device_label(block, device)
    if block.get("carries_files"):
        return _t(_ws.prompt_sends(device=label))
    return _t(_ws.prompt_stays(session=session, device=label, source=source_label(block, "?")))


def receipt_lines(block: Mapping[str, Any], *, source: str) -> list[str]:
    """What the receipt says about the working set after the move. ``source`` is the
    device the files stayed on (the move's ``from_device`` name).

    THE LIST, NOT ONLY THE COUNT (QA round 1, Q-2): with ``--yes`` or a passed digest
    no prompt was shown, so the receipt is the only place a text-mode user learns
    WHICH worktrees, withheld files and dependency trees stayed. The headline is
    followed by the same row lines the confirmation summary prints."""
    if block.get("transferred") is not False:
        return []
    consent = str(block.get("consent") or "")
    keeper = source_label(block, source)
    if consent == "not_applicable_keep":
        head = _t(_ws.receipt_keep(device=device_label(block), source=keeper))
    elif consent == "without":
        head = _t(_ws.receipt_without(device=source))
    else:
        count = sum(
            1
            for row in block.get("worktrees") or []
            if isinstance(row, dict) and row.get("state") == "left_behind"
        )
        head = _t(_ws.receipt_not_transferred(count=count, device=source))
    return [head, *_stayed_lines(block, source=keeper)]


def _stayed_lines(block: Mapping[str, Any], *, source: str) -> list[str]:
    """One indented line per worktree, withheld file and dependency tree that stayed."""
    out: list[str] = []
    for row in block.get("worktrees") or []:
        if not isinstance(row, dict):
            continue
        out.append(
            "  "
            + _t(
                _ws.worktree_left(
                    path=str(row.get("path") or ""),
                    source=source,
                    reason=reason_text(str(row.get("reason") or "not_transferred")),
                )
            )
        )
    for item in [i for i in block.get("excluded") or [] if isinstance(i, dict)][:_LIST_LIMIT]:
        path = str(item.get("path") or "")
        why = str(item.get("why") or "")
        out.append(
            "  "
            + _t(
                _ws.withheld_history(path=path)
                if why.startswith("history_")
                else _ws.withheld_secret(path=path)
            )
        )
    for item in [i for i in block.get("left_behind") or [] if isinstance(i, dict)][:_LIST_LIMIT]:
        command = str(_member(item, "recipe").get("run") or "")
        if command:
            out.append(
                "  " + _t(_ws.leaves_behind(path=str(item.get("path") or ""), command=command))
            )
    return out
