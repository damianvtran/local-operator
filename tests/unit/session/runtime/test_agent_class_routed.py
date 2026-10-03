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
    """FakeSession + the surfaces the routed class and attach handlers read.

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
        self.attached: list[str] = []

    async def cleanup_after_class_switch(self, profile_name: str) -> dict[str, Any]:
        self.cleanups.append(profile_name)
        return self.cleanup_outcome

    def attach_agent_profile(self, name: str) -> str:
        """The attach seam, recording the name the handler resolved to."""
        self.attached.append(name)
        return name


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


def _fields(**overrides: Any):  # noqa: ANN202 — the slash suite's helper shape
    """``AgentEditFields`` with every field spelled out (strict mode)."""
    from local_operator.agents import AgentEditFields

    base: dict[str, Any] = dict(
        name=None,
        description=None,
        tags=None,
        categories=None,
        security_prompt=None,
        hosting=None,
        model=None,
        last_message=None,
        temperature=None,
        top_p=None,
        top_k=None,
        max_tokens=None,
        stop=None,
        frequency_penalty=None,
        presence_penalty=None,
        seed=None,
        current_working_directory=None,
    )
    base.update(overrides)
    return AgentEditFields(**base)


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
async def test_the_routed_attach_strips_the_escape_for_an_agent_named_class(routed) -> None:
    """U4's companion: the ``=`` escape reaches the router too.

    The reserved word made an agent literally named ``class`` unreachable on
    the attach path; the escape resolves it. The escape is positional (``=``
    is not reserved in profile names) and the strip removes exactly ONE, so
    the doubled form reaches a name that itself starts with ``=``. Without
    the strip the owner looked up the literal ``=class`` (the drift class U1
    found — the local seam had the strip, this one did not).
    """
    handle, session = routed
    session.agent_registry.create_agent(
        _fields(name="class", description="The named-class agent", tags=["role"])
    )
    outcome = await handle.run_slash_authoritative("agent", "=class", [])

    assert session.attached == ["class"], session.attached
    assert "must be one of" not in outcome.get("text", "")
    # Control: without the escape the same word IS the verb, and nothing attaches.
    control = await handle.run_slash_authoritative("agent", "class", [])
    assert session.attached == ["class"]
    assert "class" in control.get("text", "").lower()


@pytest.mark.asyncio
async def test_the_routed_class_grammar_strips_exactly_one_escape(routed) -> None:
    """U7: ``class ==odd`` reaches the profile literally named ``=odd``.

    The class resolver stripped ALL leading ``=`` (``lstrip("=")``), so no
    spelling of the class grammar could reach a name whose own spelling
    starts with the escape — while the four attach seams strip exactly one
    and resolve the doubled form. The picker offers exactly this compound;
    it must resolve the name it shows.
    """
    handle, session = routed
    session.agent_registry.create_agent(_fields(name="=odd", description="Odd name", tags=["role"]))
    outcome = await handle.run_slash_authoritative("agent", "class ==odd", [])

    assert "no agent named" not in outcome.get("text", ""), outcome
    assert "=odd" in outcome.get("text", ""), outcome
    # Control: the SINGLE escape names ``odd`` — which does not exist — so the
    # doubled spelling is the only reachable one, exactly as on attach.
    control = await handle.run_slash_authoritative("agent", "class =odd", [])
    assert "no agent named 'odd'" in control.get("text", ""), control


@pytest.mark.asyncio
async def test_a_flip_with_no_cleanup_seam_still_lands_the_tag(routed) -> None:
    """A session double without the cleanup hook is the reduced-host case."""
    handle, session = routed
    session.cleanup_after_class_switch = None  # type: ignore[assignment]
    outcome = await handle.run_slash_authoritative("agent", "class reviewer proactive", [])

    assert "is now proactive" in outcome["text"]
    assert class_from_tags(_row(session).tags) == "proactive"


@pytest.mark.asyncio
async def test_the_routed_bare_agent_lists_the_roster_instead_of_going_silent(
    routed: tuple[ServingSessionHandle, _ClassSession],
) -> None:
    """A bare ``/agent`` must DELIVER its listing to a surface with no terminal.

    ``serving.py`` used to answer this with ``noop {"type": "agent_list"}``, on the
    argument that only the terminal's own resolver draws those rows. True — and it
    made the phone's sheet, which offers ``/agent`` and whose receipt is the only
    thing the reader sees, paint a tap as silence (review round 1, R1-2/U2). The
    rows now come from the ONE enumeration ``OperatorApp._agent_profile_rows``
    also delegates to, so this asserts the routed half carries the same shape the
    app-hosted half does rather than a second assembly of it.
    """
    handle, _session = routed

    result = await handle.run_slash_authoritative("agent", "")

    assert result["kind"] == "block", result
    assert result["data"]["type"] == "agent_list"
    rows = result["data"]["items"]
    assert all(len(row) == 3 for row in rows), rows
    # The first slot is the COMPOSED display form the viewer paints verbatim
    # (the shared bounded form), so a canonical label paints alone -- the same
    # bytes the app-hosted listing block shows (design round 1, D1).
    assert "Reviewer" in [row[0] for row in rows], rows


@pytest.mark.asyncio
async def test_a_routed_bare_agent_lists_the_packaged_starters_too(tmp_path: Path) -> None:
    """A registry with no installed agents still lists the seeds.

    ``resolve_profile`` falls through to the packaged starters, so ``/agent
    reviewer`` works on a fresh machine — and the listing must therefore offer
    them, or it would deny names the attach path accepts. That is why the empty
    registry here is not an empty answer, and why the two hosts share ONE
    enumeration instead of each drawing its own boundary.
    """
    registry = AgentRegistry(tmp_path)
    session = _ClassSession(registry)
    handle = ServingSessionHandle(session, asyncio.get_running_loop(), cwd=str(tmp_path))

    result = await handle.run_slash_authoritative("agent", "")

    assert result["kind"] == "block", result
    # The seeds paint their canonical labels too, through the one shared rule.
    names = [row[0] for row in result["data"]["items"]]
    assert "Reviewer" in names, names
