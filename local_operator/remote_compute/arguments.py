"""The ``exec --remote`` flags and the classification of every exec flag.

STDLIB AND ONE GENERATED KEY MODULE ONLY. ``cli.py`` imports this at module
scope to ask "is this a remote invocation?" on the exec path, so it must not
pull the client, ``httpx`` or the exec machinery with it — the same rule that
keeps ``remote_compute/__init__`` light.

WHY THE TABLES LIVE BESIDE THE PARSER. ``--remote`` meets twenty-odd flags that
predate it, and each one needs a decision: meaningful remotely, or refused when
combined. Keeping that judgement in one table, next to the arguments it judges,
is what makes it reviewable and what lets the tests iterate the same list the
refusal does — a flag added to ``exec`` later is then either classified here
consciously or caught as an unclassified one.
"""

from __future__ import annotations

import argparse
from typing import Any

from local_operator.i18n.keys import cli_exec_remote_help as _erh

#: ``exec`` destinations that cannot mean anything for a remote run.
#:
#: ``--hosting``/``--model`` are NOT here: they are resolved through the same
#: ladder a local run uses and sent to the far end as declared. ``--name`` is
#: not here either: it labels the run, which the contract carries.
LOCAL_ONLY_FLAGS: tuple[tuple[str, str], ...] = (
    ("background", "--background"),
    ("json_mode", "--json"),
    ("tools", "--tools"),
    ("control", "--control"),
    ("supervisor_fd", "--supervisor-fd"),
    ("yolo", "--yolo"),
    ("train", "--train"),
    ("loop", "--loop"),
    ("loop_goal", "--loop-goal"),
    ("clear_goal", "--clear-goal"),
    ("goal", "--goal"),
    ("resume", "--resume"),
    ("run_in", "--run-in"),
    ("workstream", "--workstream"),
    ("status", "--status"),
    ("output_format", "--output-format"),
    ("output_schema", "--output-schema"),
    ("output_retries", "--output-retries"),
    ("team", "--team"),
    ("agent_name", "--agent"),
    ("agent_id", "--agent-id"),
    ("profile", "--profile"),
    ("effort", "--effort"),
)

#: Flags whose only meaning is remote; without ``--remote`` they are refused
#: rather than left for a local run to ignore.
REMOTE_ONLY_FLAGS: tuple[tuple[str, str], ...] = (
    ("network", "--network"),
    ("size", "--size"),
    ("max_hours", "--max-hours"),
    ("budget_usd", "--budget-usd"),
    ("repo", "--repo"),
    ("sha", "--sha"),
    ("detach", "--detach"),
    ("idempotency_key", "--idempotency-key"),
)


def is_set(value: object) -> bool:
    """Whether an argparse destination was actually given.

    ``False`` counts as unset because every ``store_true`` flag defaults to it,
    and ``0`` deliberately does NOT (``--budget-usd 0`` is a real choice) — the
    check is identity against the two sentinels, not truthiness.
    """
    return value is not None and value is not False


def first_set(args: argparse.Namespace, flags: tuple[tuple[str, str], ...]) -> str | None:
    """The spelling of the first flag in ``flags`` that was passed, if any."""
    for name, spelling in flags:
        if is_set(getattr(args, name, None)):
            return spelling
    return None


def wants_remote(args: argparse.Namespace) -> bool:
    """Whether this invocation is a remote one — including its refusals.

    True for ``--remote``, for the two remote commands, and for any flag whose
    only meaning is remote: an invocation that passes one of those without
    ``--remote`` must reach the remote command to be REFUSED, not fall through
    to a local run that silently ignores it.
    """
    if is_set(getattr(args, "remote", None)):
        return True
    if is_set(getattr(args, "remote_status", None)):
        return True
    if is_set(getattr(args, "remote_cancel", None)):
        return True
    return any(is_set(getattr(args, name, None)) for name, _ in REMOTE_ONLY_FLAGS)


def add_remote_arguments(parser: Any, *, help_locale: str = "en") -> None:
    """Register the tier-1 remote flags on the ``exec`` parser.

    ``--size`` and ``--max-hours`` parse as ``None`` rather than as their
    defaults ON PURPOSE: the default is applied where the request is built, so
    "not passed" stays distinguishable from "passed with the default" and
    ``--size`` without ``--remote`` can be refused instead of ignored.
    """

    def text(msg: Any) -> str:
        return msg.envelope(locale=help_locale)["text"]

    parser.add_argument("--remote", action="store_true", help=text(_erh.remote()))
    parser.add_argument("--network", type=str, help=text(_erh.network()))
    parser.add_argument("--size", type=str, choices=["s", "m", "l"], help=text(_erh.size()))
    parser.add_argument("--max-hours", type=float, dest="max_hours", help=text(_erh.max_hours()))
    parser.add_argument("--budget-usd", type=float, dest="budget_usd", help=text(_erh.budget_usd()))
    parser.add_argument("--repo", type=str, help=text(_erh.repo()))
    parser.add_argument("--sha", type=str, help=text(_erh.sha()))
    parser.add_argument("--detach", action="store_true", help=text(_erh.detach()))
    parser.add_argument(
        "--idempotency-key", type=str, dest="idempotency_key", help=text(_erh.idempotency_key())
    )
    parser.add_argument(
        "--remote-status", metavar="RUN_ID", dest="remote_status", help=text(_erh.status())
    )
    parser.add_argument(
        "--remote-cancel", metavar="RUN_ID", dest="remote_cancel", help=text(_erh.cancel())
    )


__all__ = [
    "LOCAL_ONLY_FLAGS",
    "REMOTE_ONLY_FLAGS",
    "add_remote_arguments",
    "first_set",
    "is_set",
    "wants_remote",
]
