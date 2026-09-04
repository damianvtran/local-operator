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

    # Patching the MODULE attribute works only because the property imports
    # `TeamRegistry` inside its own body, so each call re-reads the patched
    # name (R6). Hoisting that import to module scope for performance would
    # make this test silently stop exercising the guard while still passing
    # green — if you move the import, patch `local_operator.session.remote`'s
    # binding instead.
    monkeypatch.setattr("local_operator.teams.TeamRegistry", _explode)

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.team_registry is None
    finally:
        await viewer.dispose()


@pytest.mark.asyncio
async def test_every_session_attribute_the_tui_reads_exists_on_the_viewer(
    tmp_path: Path, monkeypatch
) -> None:
    """The durable guard for the CLASS of bug this file's other tests fix (R5).

    `/team` and `/agent` broke because `app.py` reads local machine config off
    whatever object happens to be in ``self._session``, through ~50 unchecked
    ``getattr`` calls, and ``RemoteSession`` silently lacked two of the names
    ``Session`` provides. Nothing raised: every surface degraded quietly, which
    is why it reached a user rather than a test.

    So this asserts PRESENCE, not behaviour: every public attribute the TUI
    reads off a session, which ``Session`` provides, must also exist on a cold
    ``RemoteSession`` — or be named in ``KNOWN_VIEWER_GAPS`` below with the
    reason. A NEW divergence fails here, at the seam, instead of as an empty
    picker on the reporter's machine.

    The allowlist is deliberately explicit rather than a blanket skip. Each
    entry is a real user-visible or quiet degradation on the viewer path,
    catalogued during this fix and scoped to their own follow-up work; the
    point of listing them is that adding a fifteenth requires a deliberate
    edit here rather than passing unnoticed.
    """
    import inspect
    import re

    from local_operator.session.session import Session

    # Attributes the viewer genuinely cannot serve today. Shrinking this list is
    # the follow-up work; GROWING it must be a conscious decision.
    known_viewer_gaps = {
        # Attach/detach seam: needs a control op or engage-path routing. The
        # `/team` and `/agent` guards refuse loudly rather than half-execute.
        "attach_team",
        "attach_agent_profile",
        "clear_agent_profile",
        "agent_brief",
        # `/credential`'s store: the picker resolves an empty name list, so
        # stored credentials are invisible on a viewer. Its own follow-up.
        "variables",
        "journal_credential_change",
        # Guarded fallbacks in app.py that degrade to a documented default
        # (fork clones from disk, routing reads config.yml, usage/context
        # measurement is skipped, bang-mode results are not journalled).
        "request_fork",
        "has_pending_fork",
        "routing_settings",
        "record_shell",
        "measure_preloaded_context",
        "preflight_usage",
        "refresh_frontend_usage",
        "wears_inherited_title",
        # The attached team OBJECT. The viewer reports the attached roster by
        # name through `active_team_name` (which it does provide) and cannot
        # hold the Team itself without the attach seam above.
        "active_team",
        # Set by `Session._restore_attachment` to report a team/agent that
        # could not be re-resolved on resume. A viewer performs no restore, so
        # it has nothing to report; `app.py` reads it with a "" default.
        "attachment_restore_notice",
    }

    source = (Path(__file__).parents[3] / "local_operator" / "tui" / "app.py").read_text(
        encoding="utf-8"
    )
    read_by_tui = {
        match.group(1)
        for match in re.finditer(
            r'getattr\(\s*(?:self\._session|session)\s*,\s*["\']([a-z_][a-z_0-9]*)["\']',
            source,
        )
    }
    # `Session`'s surface includes the attributes it assigns in `__init__`, not
    # just class-level ones. That distinction is load-bearing: `team_registry`
    # and `agent_registry` — the two this PR restored — are INSTANCE attributes,
    # so a plain `hasattr(Session, name)` excludes exactly the names whose
    # absence caused the bug, and the guard would pass while blind to its own
    # subject. Parsed rather than instantiated because building a real Session
    # needs a model, a transcript and a config tree.
    # Scans the WHOLE class, not just `__init__` (R8). Restricting the scan to
    # the constructor is the same class of blindness as the class-level
    # `hasattr` this test was first written with: an attribute assigned in any
    # other method — a lazy cache, a late-wired handle — would be invisible to
    # the guard while being perfectly reachable from `app.py`. No public
    # attribute is assigned outside `__init__` today, so this is latent rather
    # than live, which is exactly when it is cheap to close.
    assigned_on_self = set(
        re.findall(
            r"^\s+self\.([a-z_][a-z_0-9]*)\s*(?:[:=])",
            inspect.getsource(Session),
            re.M,
        )
    )
    session_surface = assigned_on_self | {
        name for name in dir(Session) if not name.startswith("__")
    }
    assert "team_registry" in session_surface, "the self-assignment scan missed a known attribute"

    # Private names are implementation details of one side or the other, and
    # the shared contract is what this guards.
    shared = {name for name in read_by_tui if not name.startswith("_") and name in session_surface}
    assert shared, "the getattr scan found nothing — the pattern has drifted"

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        missing = {name for name in shared if not hasattr(viewer, name)}
    finally:
        await viewer.dispose()

    new_gaps = missing - known_viewer_gaps
    assert not new_gaps, (
        f"the TUI reads {sorted(new_gaps)} off the session and a viewer does not provide "
        "it, so that surface fails only on the viewer path — the exact shape of the "
        "/team and /agent regression. Implement it on RemoteSession, or add it to "
        "known_viewer_gaps with the reason."
    )
    # The registries this PR restored must never reappear as gaps.
    assert "team_registry" not in missing
    assert "agent_registry" not in missing

    stale = known_viewer_gaps - missing
    assert not stale, (
        f"{sorted(stale)} is listed as a viewer gap but the viewer now provides it — "
        "remove it from known_viewer_gaps so the list keeps meaning something."
    )


@pytest.mark.asyncio
async def test_a_failed_registry_is_constructed_once_not_once_per_keystroke(
    tmp_path: Path, monkeypatch
) -> None:
    """The FAILURE path is latched too, not just the success path (R3).

    Returning ``None`` out of the ``except`` without recording it left the
    cache empty, so the next read re-entered the constructor: 25 property
    reads measured 25 constructions. ``_team_choices`` runs on every keystroke
    while the picker is open, so an unreadable registry put a directory walk
    plus a recovery probe on the typing path — the cost ``teams.py`` is
    explicit about keeping off that loop.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    constructions = {"count": 0}

    def _explode(*_args, **_kwargs):
        constructions["count"] += 1
        raise OSError("teams directory is unreadable")

    monkeypatch.setattr("local_operator.teams.TeamRegistry", _explode)

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        for _ in range(25):
            assert viewer.team_registry is None
    finally:
        await viewer.dispose()

    assert constructions["count"] == 1, (
        f"a broken registry was rebuilt {constructions['count']} times for 25 reads; "
        "the failure must be latched so a keystroke burst costs one construction"
    )


@pytest.mark.asyncio
async def test_a_transient_registry_failure_recovers_without_a_restart(
    tmp_path: Path, monkeypatch
) -> None:
    """The latch is a COOLDOWN, not a tombstone (R7).

    A permanent latch means any transient failure — a full disk that clears, a
    directory momentarily being rewritten by a concurrent `lop team` — costs
    `/team` and `/agent` for the whole life of the session, silently, with no
    way back short of restarting. The cooldown keeps R3's guarantee (a burst of
    keystrokes pays one construction) while letting a repaired registry heal.
    """
    import time as _time

    from local_operator.teams import _READ_RECOVERY_COOLDOWN_S

    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    real = TeamRegistry
    attempts = {"count": 0}

    def _flaky(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise OSError("a transient blip")
        return real(*args, **kwargs)

    monkeypatch.setattr("local_operator.teams.TeamRegistry", _flaky)

    viewer = await RemoteSession.cold(
        SESSION_ID, config_dir=tmp_path, cwd=str(tmp_path), takeover_factory=_never
    )
    try:
        assert viewer.team_registry is None
        # Still inside the cooldown: no re-entry, so typing stays cheap.
        for _ in range(25):
            assert viewer.team_registry is None
        assert attempts["count"] == 1, "the cooldown must absorb a keystroke burst"

        _time.sleep(_READ_RECOVERY_COOLDOWN_S + 0.05)
        assert viewer.team_registry is not None, (
            "a repaired registry must recover on a later read rather than staying "
            "dead for the life of the session"
        )
    finally:
        await viewer.dispose()
