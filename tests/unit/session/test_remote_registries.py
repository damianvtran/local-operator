"""A viewer answers `/team` and `/agent` from its OWN config dir.

Teams and agent profiles are LOCAL CONFIG — files under ``<config_dir>/teams``
and ``<config_dir>/agents`` — not runtime state. The TUI reads both registries
off the SESSION object (``_team_registry()``, ``_agent_profile_rows()``), so
when ``lop`` stopped building a ``Session`` and started handing the app a
``RemoteSession``, both surfaces silently went empty: every team the user had
vanished from `/team`, and `/agent` fell back to the packaged starters.

That is what these tests pin. The regression is invisible to any test that
drives a ``Session`` or a ``FakeSession``, because the only broken thing was
the attribute the viewer did NOT carry — which is exactly why it shipped.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.session.remote import RemoteSession
from local_operator.teams import TeamEditFields, TeamRegistry

SESSION_ID = "registryviewer1"


async def _never():
    raise AssertionError("takeover was not expected")


def _seed_team(config_dir: Path, name: str) -> None:
    """Write one real team through the registry's own writer."""
    TeamRegistry(config_dir).create_team(
        TeamEditFields(name=name, description=f"{name} description", manager="manager")
    )


@pytest.mark.asyncio
async def test_a_cold_viewer_lists_the_teams_on_this_machine(tmp_path: Path, monkeypatch) -> None:
    """The reported bug: three teams on disk, none of them in `/team`.

    A cold viewer has no runtime to ask and — until the first message — never
    will, so a registry served from the owner would leave `/team` permanently
    empty. Reading the viewer's own config dir is what makes the listing
    answerable at all in this state.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    for name in ("alpha", "beta", "gamma"):
        _seed_team(tmp_path, name)

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        registry = viewer.team_registry
        assert registry is not None, "a viewer must serve teams from its own config dir"
        assert sorted(team.name for team in registry.list_teams()) == ["alpha", "beta", "gamma"]
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_a_cold_viewer_serves_the_agent_registry_too(tmp_path: Path, monkeypatch) -> None:
    """`/agent` reads `agent_registry` off the session by the same route.

    One root cause, two user-visible surfaces: a fix that restored only teams
    would ship a half-fixed picker.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        registry = viewer.agent_registry
        assert registry is not None, "a viewer must serve agent profiles from its own config dir"
        # The registry is rooted at the viewer's config dir, not at some
        # process-global default — that is what makes it the USER's agents.
        assert Path(registry.config_dir) == tmp_path
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_the_registries_are_built_once_and_cached(tmp_path: Path, monkeypatch) -> None:
    """Constructing a registry walks the config tree; a repaint must not.

    ``TeamRegistry.__init__`` also runs crash recovery, and the picker
    re-derives its rows on EVERY keystroke, so an uncached property would put
    a directory walk plus a recovery probe on the typing path.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    _seed_team(tmp_path, "alpha")

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.team_registry is viewer.team_registry
        assert viewer.agent_registry is viewer.agent_registry
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_an_unusable_registry_degrades_one_feature_not_the_session(
    tmp_path: Path, monkeypatch
) -> None:
    """A broken teams dir costs `/team`, never the conversation.

    Same discipline ``session_factory`` applies to its own construction: the
    session must still open, and `/team` reports teams are unavailable rather
    than taking the whole session down with it.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))

    def _explode(*_args, **_kwargs):
        raise OSError("teams directory is unreadable")

    monkeypatch.setattr("local_operator.teams.TeamRegistry", _explode)

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.team_registry is None
    finally:
        await viewer.dispose()
