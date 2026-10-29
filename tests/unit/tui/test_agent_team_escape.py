"""The `=` literal-name escape at the OWNER dispatcher seams.

`/agent` and `/team` are authoritative-session commands: in a live session the
line reaches `ServingSessionHandle._agent_slash` / `_team_attach_slash`, and the
app-hosted handler never runs. The TUI-local grammar gained "a leading `=`
suppresses the clear/none detach" in the command-palette round; without the
same guard here, `/agent =clear` attached on the terminal but detached on an
attached session — one command, two meanings, depending on where it ran.

These call the real dispatcher methods (the `_owner_slash` pattern
`test_promote_verb.py` established), with the registries stubbed to the one
lookup the attach half performs.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, cast

from local_operator.session.frontend_state import SlashResult
from local_operator.session.runtime.serving import ServingSessionHandle


def _handle() -> Any:
    # The seams call `self._notify()` on a successful attach; no other handle
    # state is touched, so a stub is enough and keeps the REAL dispatcher under
    # test (see test_promote_verb.py for the same pattern).
    return cast("ServingSessionHandle", SimpleNamespace(_notify=lambda: None))


class _AgentSession:
    def __init__(self) -> None:
        self.attached: list[str] = []
        self.cleared = 0

    def attach_agent_profile(self, name: str) -> str:
        self.attached.append(name)
        return name

    def clear_agent_profile(self) -> None:
        self.cleared += 1


class _TeamSession:
    def __init__(self) -> None:
        self.attached: list[Any] = []
        self.active_team_name = ""
        self.team = SimpleNamespace(name="clear", label="", manager="manager")
        self.team_registry = SimpleNamespace(get_team_by_name=self._lookup)

    def _lookup(self, name: str) -> Any:
        return self.team if name in ("clear", "none") else None

    def attach_team(self, team: Any) -> None:
        self.attached.append(team)
        return None


def _agent_slash(session: Any, arg: str) -> Any:
    return asyncio.run(ServingSessionHandle._agent_slash(_handle(), session, arg, SlashResult))


def _team_attach(session: Any, arg: str) -> Any:
    return ServingSessionHandle._team_attach_slash(_handle(), session, arg, SlashResult)


def test_owner_agent_clear_escape_attaches_literally() -> None:
    session = _AgentSession()

    result = _agent_slash(session, "=clear")

    assert session.attached == ["clear"], result.text
    assert session.cleared == 0, "the escape must suppress the detach verb"
    assert "clear is answering" in result.text


def test_owner_agent_none_escape_attaches_literally() -> None:
    session = _AgentSession()

    result = _agent_slash(session, "=none")

    assert session.attached == ["none"], result.text
    assert session.cleared == 0


def test_owner_agent_bare_clear_still_detaches() -> None:
    session = _AgentSession()

    result = _agent_slash(session, "clear")

    assert session.cleared == 1
    assert session.attached == []
    assert "no agent active" in result.text


def test_owner_team_clear_escape_attaches_literally() -> None:
    session = _TeamSession()

    result = _team_attach(session, "=clear")

    assert [getattr(team, "name", None) for team in session.attached] == ["clear"], result.text
    assert "no team active" not in result.text


def test_owner_team_none_escape_attaches_literally() -> None:
    session = _TeamSession()

    result = _team_attach(session, "=none")

    assert [getattr(team, "name", None) for team in session.attached] == ["clear"], result.text


def test_owner_team_bare_clear_still_detaches() -> None:
    session = _TeamSession()

    result = _team_attach(session, "clear")

    assert session.attached == [None], "the bare word must remain the detach verb"
    assert "nothing was detached" in result.text
