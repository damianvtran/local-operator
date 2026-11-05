"""The ``deliver`` pin's call-time notice (bash): one line, only when a pin is refused.

WHY AT CALL TIME. A pin is the agent saying "show this under my answer". The row's
``withheld`` notice tells the OPERATOR when that failed, but only after the turn; the agent
would end its answer promising a file that never appears. So the tool result gains ONE
compact line naming the refused pins, e.g.
``deliver: 1 not shown (outside allowed locations: x.key)``, while the agent can still move
the file somewhere allowed. Accepted pins add nothing: the line costs tokens on every call
that carries it, and silence is the common case.

ONE VERDICT. The check runs the real :func:`~local_operator.supplements.candidates.prefilter`
over a synthetic one-call turn, so the call-time line and the end-of-turn row can never
disagree about which pin is refused or why. Model-facing text, so English by design (the
i18n exception for model-facing prompts); the operator-facing copy is the surfaces' keyed
withheld line.
"""

from typing import Any, Final, Mapping, Sequence

#: Reason id -> the phrase the agent reads. Same vocabulary as the row's ``withheld``.
_REASON_TEXT: Final[Mapping[str, str]] = {
    "outside": "outside allowed locations",
    "sensitive": "sensitive file",
    "missing": "file does not exist",
    "too-large": "too large to preview",
}
#: Names listed per reason before ``+N``; the line must stay one short line.
_NAMES_SHOWN: Final = 3

#: The ``deliver`` field's schema description. The schema rides EVERY request of every
#: session and the start-context budget guard has ~50 tokens to spare on the child arm,
#: so the pin is advertised on ``bash`` only, in the shortest wording that still says what
#: it does; path forms and the eval route (name the file in the answer) live in
#: guide://highlights. ``items`` stays: strict-schema providers reject a bare array.
DELIVER_DESCRIPTION: Final = "Files to show."


def _deny_prefixes() -> tuple[str, ...]:
    """The operator's ``supplements.denyPrefixes``, read the way the runner reads them.

    Without them the call-time verdict could say nothing while the turn-end row withholds
    the same pin (review R1-2): ONE verdict means the same inputs. Read per pinned call --
    pins are rare and the read is a small config load; a failure is "no prefixes", never a
    failed tool call.
    """
    try:
        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir
        from local_operator.supplements.policy import SupplementSettings

        values = getattr(ConfigManager(config_dir()).get_config(), "values", None)
        return tuple(SupplementSettings.from_values(values).deny_prefixes)
    except Exception:  # noqa: BLE001 -- see docstring
        return ()


def refusal_line(
    paths: Sequence[str],
    *,
    cwd: str,
    scratchpad: str | None,
    tool: str = "bash",
    deny_prefixes: Sequence[str] | None = None,
) -> str:
    """``""`` when every pin is accepted, else the one-line notice. Never raises."""
    pins = [p for p in paths if isinstance(p, str) and p.strip()]
    if not pins:
        return ""
    try:
        from local_operator.supplements.candidates import prefilter
        from local_operator.supplements.trigger import ToolCallItem, TurnItem

        item = TurnItem(
            id="deliver-check",
            role="assistant",
            tool_calls=(ToolCallItem(id="deliver-check", name=tool, args={}, deliver=tuple(pins)),),
        )
        result = prefilter(
            [item],
            "",
            cwd=cwd,
            want_graphics=False,
            scratchpad=scratchpad or "",
            deny_prefixes=_deny_prefixes() if deny_prefixes is None else deny_prefixes,
        )
    except Exception:  # noqa: BLE001 -- a notice must never fail the call it rides on
        return ""
    if not result.withheld:
        return ""
    by_reason: dict[str, list[str]] = {}
    for name, reason in result.withheld:
        by_reason.setdefault(reason, []).append(name)
    parts = []
    for reason, names in by_reason.items():
        shown = ", ".join(names[:_NAMES_SHOWN])
        extra = f" +{len(names) - _NAMES_SHOWN}" if len(names) > _NAMES_SHOWN else ""
        parts.append(f"{_REASON_TEXT.get(reason, reason)}: {shown}{extra}")
    return f"deliver: {len(result.withheld)} not shown ({'; '.join(parts)})"


def with_deliver_notice(execute: Any, tool: str) -> Any:
    """Wrap the bash ``execute`` so a refused ``deliver`` pin adds the one-line notice.

    Checked AFTER the call: a pin routinely names the file the same command creates. A
    background call is skipped -- its files do not exist yet when the job id returns, and a
    "file does not exist" line there would be wrong; the row's ``withheld`` still discloses a
    refusal at the end of the turn. A failed call is skipped too: its own error is the news.
    """

    async def run(
        tool_call_id: str,
        args: dict[str, Any],
        signal: Any = None,
        on_update: Any = None,
        context: Any = None,
    ) -> Any:
        result = await execute(tool_call_id, args, signal, on_update, context)
        pins = args.get("deliver") if isinstance(args, Mapping) else None
        if not isinstance(pins, list) or not pins or args.get("background") is True:
            return result
        if getattr(result, "is_error", False):
            return result
        import asyncio

        from local_operator.scratchpad import scratchpad_dir_of

        line = await asyncio.to_thread(
            refusal_line,
            pins,
            cwd=str(getattr(context, "cwd", "") or "."),
            scratchpad=scratchpad_dir_of(context),
            tool=tool,
        )
        return append_notice(result, line)

    return run


def append_notice(result: Any, line: str) -> Any:
    """Append ``line`` as a trailing text block of a ``ToolResult`` (in place), if any."""
    if not line:
        return result
    from local_operator.harness.types import TextContent

    content = getattr(result, "content", None)
    if isinstance(content, list):
        content.append(TextContent(text=line))
    return result
