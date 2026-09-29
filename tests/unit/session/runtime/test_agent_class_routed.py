"""``/agent class …`` on the ROUTED path — the seam a live terminal uses.

The TUI-local handler is pinned next door
(``tests/unit/tui/test_agent_class_slash.py``), but ``/agent`` is
AUTHORITATIVE_SESSION: a real (attached) terminal routes the command to the
runtime, whose handler is what the user's words actually reach. Round-1 UX
found the switch implemented ONLY at the local seam, so on every real session
``/agent class`` was answered as "no agent named 'class'" (U1). These cells
drive ``ServingSessionHandle.run_slash_authoritative`` — the public method the
server calls — so the routed half cannot go missing behind a local double
again.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio

from local_operator.action_class import class_from_tags
from local_operator.agent_profiles import install_seed
from local_operator.agents import AgentRegistry
from local_operator.session.runtime.serving import ServingSessionHandle
from tests.unit.session.runtime.test_serving import FakeSession


class _ClassSession(FakeSession):
    """FakeSession + the two surfaces the routed class handler reads.

    A SUBCLASS rather than attributes bolted on at runtime, so the shape stays
    part of the double's contract for the type checker every gate runs.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        super().__init__()
        self.agent_registry = registry
        self.cleanups: list[str] = []
        self.cleanup_outcome: dict[str, Any] = {
            "patience_cancelled": ["patience-1"],
            "cadence_dropped": False,
        }

    async def cleanup_after_class_switch(self, profile_name: str) -> dict[str, Any]:
        self.cleanups.append(profile_name)
        return self.cleanup_outcome


@pytest_asyncio.fixture
async def routed(tmp_path: Path) -> AsyncIterator[tuple[ServingSessionHandle, _ClassSession]]:
    """A handle over a session with a real registry and a REAL installed role."""
    registry = AgentRegistry(tmp_path)
    installed = install_seed("reviewer", registry=registry)
    assert installed is not None
    session = _ClassSession(registry)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))
    yield handle, session


def _row(session: _ClassSession) -> Any:
    row = session.agent_registry.get_agent_by_name("reviewer")
    assert row is not None, "the routed switch must not lose the row it flipped"
    return row


@pytest.mark.asyncio
async def test_the_routed_switch_flips_the_tag_and_reports_the_cleanup(routed) -> None:
    handle, session = routed
    outcome = await handle.run_slash_authoritative("agent", "class reviewer proactive", [])

    assert outcome["data"] == {"type": "agent_class", "agent": "reviewer", "class": "proactive"}
    assert "agent reviewer is now proactive" in outcome["text"]
    assert "1 pending wait(s) cancelled" in outcome["text"]
    assert class_from_tags(_row(session).tags) == "proactive"
    # The cleanup runs HERE (this process owns the session), and it sees the
    # RESOLVED name — not the raw spelling the user typed.
    assert session.cleanups == ["reviewer"]


@pytest.mark.asyncio
async def test_the_routed_report_form_writes_nothing(routed) -> None:
    handle, session = routed
    outcome = await handle.run_slash_authoritative("agent", "class reviewer", [])

    assert "reviewer: class reactive" in outcome["text"]
    assert "set with /agent class reviewer proactive|reactive" in outcome["text"]
    assert class_from_tags(_row(session).tags) == "reactive"
    assert session.cleanups == []


@pytest.mark.asyncio
async def test_the_routed_switch_refuses_a_bad_word_before_any_write(routed) -> None:
    handle, session = routed
    outcome = await handle.run_slash_authoritative("agent", "class reviewer sideways", [])

    assert "class must be one of proactive or reactive; got 'sideways'." in outcome["text"]
    assert class_from_tags(_row(session).tags) == "reactive"
    assert session.cleanups == []


@pytest.mark.asyncio
async def test_the_routed_switch_refuses_an_unknown_name_as_a_notice(routed) -> None:
    handle, session = routed
    outcome = await handle.run_slash_authoritative("agent", "class ghost proactive", [])

    assert outcome["style"] == "warning"
    assert "no agent named 'ghost'" in outcome["text"]


@pytest.mark.asyncio
async def test_the_routed_switch_accepts_a_case_variant_spelling(routed) -> None:
    """``Aida`` must reach the row ``aida`` addresses (review round 1, R2)."""
    handle, session = routed
    outcome = await handle.run_slash_authoritative("agent", "class Reviewer proactive", [])

    assert "agent reviewer is now proactive" in outcome["text"]
    assert class_from_tags(_row(session).tags) == "proactive"


@pytest.mark.asyncio
async def test_a_flip_with_no_cleanup_seam_still_lands_the_tag(routed) -> None:
    """A session double without the cleanup hook is the reduced-host case."""
    handle, session = routed
    session.cleanup_after_class_switch = None  # type: ignore[assignment]
    outcome = await handle.run_slash_authoritative("agent", "class reviewer proactive", [])

    assert "is now proactive" in outcome["text"]
    assert class_from_tags(_row(session).tags) == "proactive"
