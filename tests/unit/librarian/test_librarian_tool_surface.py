"""The Librarian's tool surface: a read-tier allow-list, enforced on her duty session.

The measured defect (D-2b, 2026-10-27): her seed declared an empty ``tools:``
(= every tool), and a role's list is only ENFORCED where a host declares it
(``lop exec --profile``, the SDK, a delegated child). Her duty session is
opened by the wake supervisor as a plain runtime, so an unattended daily pass
on a headless worker under ``tool_approval_mode: ask`` reached a write-tier
tool and parked on an approval nobody could answer — for a day, holding the
node ``busy`` and refusing every mesh update.

These cells pin both halves: the packaged seed's list is read-tier only and has
no ``send`` hand-off, and her REAL duty session (state.json names it, the
attachment restores her role) reaches exactly that list — while a session that
merely attached ``librarian`` with ``/agent`` is left alone.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.agent_profiles import load_seed
from local_operator.harness.types import ModelSpec, StreamEndEvent
from local_operator.librarian import state
from local_operator.resume import write_session_attachment
from local_operator.session.session import Session
from local_operator.session.transcript import Transcript
from local_operator.tools.library_tool import build_library_tool
from local_operator.tools.registry import TOOL_BUILDERS, create_tools
from tests.e2e.harness import ScriptedStream

MODEL = ModelSpec(provider="test", model_id="librarian-tools-model", context_window=100_000)
SEED_TOOLS = ("library", "read", "glob", "grep", "todo")


def test_the_seed_declares_a_read_tier_allow_list_and_no_hand_off() -> None:
    seed = load_seed("librarian")
    assert seed is not None
    assert tuple(seed.tools or ()) == SEED_TOOLS, seed.tools
    # Every tool on the list must be one an unattended pass can call without a
    # gate: read tier, checked against the real built tools, not a table.
    import tempfile

    from local_operator.harness.types import ToolContext

    with tempfile.TemporaryDirectory() as cwd:
        context = ToolContext(cwd=cwd)
        built = {}
        for name in SEED_TOOLS:
            builder = TOOL_BUILDERS.get(name)
            tool = builder(context) if builder is not None else None
            if tool is not None:
                built[name] = tool
    built.setdefault("library", build_library_tool())
    for name in SEED_TOOLS:
        assert name in built, f"{name} is not a tool this build has"
        assert built[name].approval_tier == "read", (name, built[name].approval_tier)
    # The send-to-Aida step is gone: nothing in her instructions may tell an
    # unattended pass to message another session.
    assert "`send`" not in seed.instructions, "the seed still tells her to send"


def _session(root: Path, session_id: str, *, duty: bool, attach: bool = True) -> Session:
    directory = root / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    if attach:
        write_session_attachment(directory, team="", agent="librarian", goal="")
    if duty:
        state.update_state(root, session_id=session_id)
    from local_operator.harness.types import ToolContext

    return Session(
        model=MODEL,
        model_source="config",
        stream_fn=ScriptedStream([[StreamEndEvent(stop_reason="stop")]]),
        tools=[
            *create_tools(ToolContext(cwd=str(root))),
            *([] if "library" in TOOL_BUILDERS else [build_library_tool()]),
        ],
        transcript=Transcript(directory),
        system_blocks_provider=lambda: [],
        cwd=str(root),
    )


@pytest.mark.asyncio
async def test_her_duty_session_reaches_only_her_list(isolated_root: Path) -> None:
    session = _session(isolated_root, "l1brar1an0a1", duty=True)
    try:
        assert session._librarian_duty is True
        names = {t.name for t in session._tools}
        assert names <= set(SEED_TOOLS), sorted(names - set(SEED_TOOLS))
        assert {"library", "read"} <= names, names
        for banned in ("bash", "write", "edit", "send", "wake", "task"):
            assert banned not in names, f"{banned} reached her duty session"
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_a_row_predating_the_list_falls_back_to_the_packaged_one(
    isolated_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed row from before 1.1.0 records no list; the seed's still binds."""
    original = Session.attach_agent_profile

    def _attach_without_list(self: Session, name: str) -> str | None:
        result = original(self, name)
        self._attached_profile_tools = ()  # the pre-1.1.0 row's shape
        return result

    monkeypatch.setattr(Session, "attach_agent_profile", _attach_without_list)
    session = _session(isolated_root, "l1brar1an0a2", duty=True)
    try:
        names = {t.name for t in session._tools}
        assert names <= set(SEED_TOOLS), sorted(names - set(SEED_TOOLS))
        assert "bash" not in names
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_agent_librarian_attach_elsewhere_is_not_narrowed(isolated_root: Path) -> None:
    """Only her duty session is bound; ``/agent librarian`` on another keeps its reach."""
    session = _session(isolated_root, "n0td4ty00001", duty=False)
    try:
        assert session._librarian_duty is False
        assert "bash" in {t.name for t in session._tools}
    finally:
        await session.dispose()


@pytest.mark.asyncio
async def test_an_unresolved_attachment_still_binds_her_duty_session(isolated_root: Path) -> None:
    """m1 (agent review round 1 on #3028): the duty flag is the identity, not the attachment.

    A duty session whose attachment did not resolve (``ProfileRegistryUnavailable``
    leaves the agent name empty — modelled here by no attachment at all) used to
    return early and run her unattended pass with every tool. It now falls back
    to the packaged seed's list.
    """
    session = _session(isolated_root, "l1brar1an0a3", duty=True, attach=False)
    try:
        assert session._librarian_duty is True
        names = {t.name for t in session._tools}
        assert names <= set(SEED_TOOLS), sorted(names - set(SEED_TOOLS))
        assert "bash" not in names and "send" not in names, names
    finally:
        await session.dispose()
