"""No store walk may ever hop onto the shared default executor again.

THE DEFECT THIS PINS, so a reader does not have to reconstruct it: the
store-maintenance passes were dispatched with ``await asyncio.to_thread(work)``,
which puts the walk on the process's shared default executor. A runtime that had
decided to leave then sat in ``asyncio.Runner.close()`` joining that worker —
and because the join's own bound is CPython's 300-second constant, it outlived
the stall bound and was reported as a stalled turn: the main thread inside
``Runner.close`` was the second-largest class in the retained corpus (43 of 182
dumps, 24%), and the joined workers were this pass family (40 of 46 of them the
analytics session-name sweep).

The PATH was closed by giving the passes their own daemon thread. The SHAPE is
what this test guards, and the reason it must be a SOURCE audit rather than a
behavioural one is the same reason the boundary audits next door are static: the
returning defect is a one-line change that no test of the passes' *results* can
see, because a walk on the executor produces identical results — right up to the
moment it holds a departing process open for minutes. It has already been
diagnosed twice from a dump, and both times the fix was one call site.

WHAT THIS COVERS, exactly:

* every executor hop (``to_thread``, ``run_in_executor``) in the maintenance
  body — the pass runner and the two functions that dispatch and retry it;
* the same in the four store-walking passes themselves, where a hop would put a
  walk back on the pool from the other side;
* the one-line reason the ALLOWED site carries: the dedicated daemon thread the
  runner documents, so removing the reason means deleting a test rather than a
  comment.

WHAT IT DOES NOT COVER, stated so a green run is not read as more than it is: a
hop reached INDIRECTLY — a helper these functions call which itself submits to
the pool, or a callable handed in from outside (``AnalyticsStore`` methods,
``cleanup_from_config``) — and a name assembled at runtime. The pin is a
tripwire on the direct, literal hop, which is the shape the defect actually
took.

A note on the ALLOWED ledger below: it exists so an exception to the rule has to
be written down WITH its reason, and the assertions make each entry prove it
still describes a real hop — an allowlist that can rot into a list of names that
match nothing is how a check quietly stops checking.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import pytest

from local_operator import resume, session_factory
from local_operator.analytics import backfill as analytics_backfill

#: The two spellings an off-loop submission takes in this codebase.
_EXECUTOR_HOP_NAMES = frozenset({"to_thread", "run_in_executor"})

#: Sites the rule permits, as ``(function name, one-line reason)``. EMPTY, and
#: that is the shipped state: the dedicated daemon thread the runner documents is
#: a thread, not a hop, so there is no legitimate executor hop anywhere in this
#: body. An entry must name a function that still hops (asserted below) and carry
#: a reason — the only way to add one is to describe why a walk may block the
#: shared pool, in the same commit.
ALLOWED_EXECUTOR_HOPS: tuple[tuple[str, str], ...] = ()

#: The body: the runner, and the dispatch/retry pair around it. Named
#: POSITIVELY, so a future author who splits these has to come back here rather
#: than silently auditing a function that no longer exists.
_MAINTENANCE_BODY = (
    session_factory._run_store_maintenance,
    session_factory._store_maintenance_thread_main,
    session_factory._start_store_maintenance,
)

#: The walks themselves. A hop in any of these submits a store walk to the pool
#: from the inside, which is the same defect wearing the other hat.
_STORE_WALKING_PASSES = (
    resume.backfill_session_origins,
    resume.backfill_session_titles,
    analytics_backfill.backfill_analytics_session_names,
    analytics_backfill._name_pending_sessions,
    analytics_backfill.backfill_analytics_session_daily,
)


def _executor_hops(source: str) -> list[tuple[int, str]]:
    """Every literal executor hop in ``source``, as ``(line, text)``."""
    lines = source.splitlines()
    found: list[tuple[int, str]] = []
    for node in ast.walk(ast.parse(textwrap.dedent(source))):
        name: str | None = None
        if isinstance(node, ast.Attribute):
            name = node.attr
        elif isinstance(node, ast.Name):
            name = node.id
        if name in _EXECUTOR_HOP_NAMES:
            # ``ast.walk`` yields bare ``AST`` nodes; only the located ones carry a
            # line, and every node this loop matches is one of those.
            line = getattr(node, "lineno", 0)
            found.append((line, lines[line - 1].strip()))
    return found


def _reason_for(function: Callable[..., Any]) -> str:
    """The allowlist's reason for ``function``, or ``""`` when it is not listed."""
    for name, reason in ALLOWED_EXECUTOR_HOPS:
        if name == function.__name__:
            return reason
    return ""


@pytest.mark.parametrize("function", _MAINTENANCE_BODY, ids=lambda fn: fn.__name__)
def test_the_maintenance_body_contains_no_executor_hop(function: Callable[..., Any]) -> None:
    """The pin: the passes run on their own daemon, never on the shared pool."""
    hops = _executor_hops(inspect.getsource(function))
    if hops and _reason_for(function):
        return
    assert hops == [], (
        f"{function.__name__} submits to the shared default executor at line(s) "
        f"{[line for line, _ in hops]}: {[text for _, text in hops]}. A store walk on "
        "that pool is joined by asyncio.Runner.close() for up to 300 s, which is how a "
        "runtime that had decided to leave came to be reported as a stalled turn. Run "
        "it on the dedicated store-maintenance daemon thread instead."
    )


@pytest.mark.parametrize("function", _STORE_WALKING_PASSES, ids=lambda fn: fn.__name__)
def test_no_store_walking_pass_contains_an_executor_hop(function: Callable[..., Any]) -> None:
    """The same rule one level down: a walk must not submit itself to the pool."""
    assert _executor_hops(inspect.getsource(function)) == [], (
        f"{function.__name__} is a store walk that submits to the shared executor; it "
        "runs ON the maintenance thread already, and a hop from there re-creates the "
        "held-open teardown the daemon thread exists to prevent."
    )


def test_the_runner_still_carries_the_reason_its_allowed_site_has() -> None:
    """The daemon thread is the one allowed site, and its reason is pinned.

    Pinned rather than trusted because the reason IS the safeguard: the next
    author deciding between ``to_thread`` and a thread reads it here, and the
    sentence names both the bound it avoids and the mechanism that replaced it.
    """
    # Whitespace-normalised: the sentence wraps across two source lines, and a
    # pin on line breaks would fail on a reflow rather than on a change of rule.
    doc = " ".join((inspect.getdoc(session_factory._run_store_maintenance) or "").split())
    assert "No ``to_thread`` call is used" in doc
    assert "dedicated daemon" in doc


def test_the_allowlist_cannot_rot_into_names_that_match_nothing() -> None:
    """Every allowed entry must name a real function that really still hops."""
    for name, reason in ALLOWED_EXECUTOR_HOPS:
        candidates = {
            function.__name__: function for function in (*_MAINTENANCE_BODY, *_STORE_WALKING_PASSES)
        }
        function = candidates.get(name)
        assert function is not None, f"the allowlist names {name!r}, which is not in the body"
        assert _executor_hops(
            inspect.getsource(function)
        ), f"the allowlist exempts {name!r}, which no longer hops — drop the entry"
        assert reason.strip(), f"the allowlist entry for {name!r} carries no reason"
