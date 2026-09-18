"""The classification layer's wiring, exercised through the knowledge-block hook.

Why this file exists beside the package's own suite (``tests/unit/classification``
is the package's): the layer is only worth anything if it is wired into the TURN
PATH, and the two properties the contract makes load-bearing there are not
properties of the package at all — they are properties of the wiring:

* **Degradation is byte-identical** (§7): with the layer off, unavailable, timed
  out or empty, the prompt must be exactly what it was before the seam existed.
  Asserted here by comparing the joined block against the block a hooks object
  without any classifier returns.
* **Selection and classification run CONCURRENTLY** (§7 step 2), so the added
  wall-clock is the difference rather than the sum.

Every test injects a seam double rather than building the real
``ClassificationService``: the wiring must work with the package ABSENT (the
layer is optional), and a test that reached for the real service would quietly
stop covering that. The two tests that DO touch the package
(``test_auto_off...``/``test_auto_on...``) exist to pin the build decision and
the ``values.classification`` reads; they are named as such.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator import session_factory
from local_operator.config import ConfigManager
from local_operator.skills.discovery import Skill


def _skill(name: str, description: str, kind: str = "skill", hide: bool = False) -> Skill:
    """A real ``Skill`` row, which is what ``render_block`` and the roster read."""
    base = Path("/tmp/skills") / name
    return Skill(
        name=name,
        description=description,
        file_path=base / "SKILL.md",
        base_dir=base,
        source="user",
        hide=hide,
        resource_type=kind,  # type: ignore[arg-type]
    )


@dataclass
class _Recommendation:
    """The contract's ``Recommendation`` (§4), field for field.

    Local rather than imported, for the reason this module's docstring gives: the
    wiring reads these by attribute, and the tests must pass with the package
    absent.
    """

    resources: tuple[Any, ...] = ()
    block: str = ""
    vendor: str | None = None
    cost_usd: float | None = None
    latency_s: float = 0.0
    skipped: str | None = None


@dataclass
class _Candidate:
    """The contract's ``Candidate`` (§4), field for field."""

    kind: str
    name: str
    description: str
    resource_url: str


class _FakeClassifier:
    """The seam double: ``recommend_resources`` + ``notice``, that is all."""

    def __init__(
        self,
        recommendation: _Recommendation | None = None,
        *,
        delay: float = 0.0,
        timeout_s: float | None = None,
        notice_line: str | None = None,
        raises: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.recommendation = recommendation or _Recommendation()
        self.delay = delay
        self.timeout_s = timeout_s
        self.notice_line = notice_line
        self.raises = raises
        self.events = events if events is not None else []
        self.requests: list[Any] = []

    async def recommend_resources(self, request: Any) -> _Recommendation:
        self.requests.append(request)
        self.events.append("classify-start")
        if self.delay:
            await asyncio.sleep(self.delay)
        self.events.append("classify-end")
        if self.raises is not None:
            raise self.raises
        return self.recommendation

    def notice(self, recommendation: _Recommendation) -> str | None:
        return self.notice_line


class _FakeIndex:
    """The router half: ``select`` plus the ``skills`` the roster walks."""

    def __init__(
        self,
        skills: list[Skill] | None = None,
        *,
        picked: list[Skill] | None = None,
        delay: float = 0.0,
        events: list[str] | None = None,
        calls: list[str] | None = None,
    ) -> None:
        self.skills = list(skills if skills is not None else (_skill("alpha", "Alpha skill."),))
        self.picked = list(picked if picked is not None else [])
        self.delay = delay
        self.events = events if events is not None else []
        self.calls = calls if calls is not None else []

    async def select(self, query: str, **kwargs: Any) -> list[Skill]:
        self.calls.append(query)
        self.events.append("select-start")
        if self.delay:
            await asyncio.sleep(self.delay)
        self.events.append("select-end")
        return list(self.picked)


def _hooks(
    index: _FakeIndex | None = None,
    *,
    classifier: Any = None,
    catalogue: str = "",
    servers: tuple[str, ...] = (),
) -> session_factory._KnowledgeHooks:
    hooks = session_factory._KnowledgeHooks(
        index=index if index is not None else _FakeIndex(),  # type: ignore[arg-type]
        classifier=classifier,
        mcp_server_names=servers,
    )
    if catalogue:
        hooks.mcp_catalogue = lambda query: catalogue
    return hooks


OFF_QUERY = "do the thing"


async def _off_block(index: _FakeIndex, catalogue: str = "") -> str:
    """Today's prompt: the same hook with no classifier at all."""
    return await session_factory._select_knowledge_block(
        _hooks(index, catalogue=catalogue), OFF_QUERY, task_id="t1"
    )


# ---------------------------------------------------------------------------
# §7 degradation: byte-identical when the layer adds nothing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_disabled_layer_renders_the_prompt_byte_for_byte() -> None:
    """``Recommendation(skipped="disabled")`` must add nothing at all."""
    index = _FakeIndex(picked=[_skill("alpha", "Alpha skill.")])
    off = await _off_block(index, catalogue="<mcps>catalogue</mcps>")

    on = _hooks(
        _FakeIndex(picked=[_skill("alpha", "Alpha skill.")]),
        classifier=_FakeClassifier(_Recommendation(skipped="disabled")),
        catalogue="<mcps>catalogue</mcps>",
    )
    assert await session_factory._select_knowledge_block(on, OFF_QUERY, task_id="t1") == off


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "skipped", ["no-vendor", "empty-roster", "timeout", "error", "circuit-open"]
)
async def test_every_empty_skip_reason_renders_the_prompt_byte_for_byte(skipped: str) -> None:
    """The skip taxonomy is the package's; the wiring's promise is that all of it degrades."""
    index = _FakeIndex(picked=[_skill("alpha", "Alpha skill.")])
    off = await _off_block(index)

    on = _hooks(
        _FakeIndex(picked=[_skill("alpha", "Alpha skill.")]),
        classifier=_FakeClassifier(_Recommendation(skipped=skipped)),
    )
    assert await session_factory._select_knowledge_block(on, OFF_QUERY, task_id="t1") == off


@pytest.mark.asyncio
async def test_a_raising_seam_renders_the_prompt_byte_for_byte() -> None:
    """A classifier fault is the layer's problem, never the turn's (§4)."""
    index = _FakeIndex(picked=[_skill("alpha", "Alpha skill.")])
    off = await _off_block(index)

    on = _hooks(
        _FakeIndex(picked=[_skill("alpha", "Alpha skill.")]),
        classifier=_FakeClassifier(raises=RuntimeError("vendor exploded")),
    )
    assert await session_factory._select_knowledge_block(on, OFF_QUERY, task_id="t1") == off


@pytest.mark.asyncio
async def test_a_recommendation_with_no_surviving_resource_renders_nothing() -> None:
    """Everything recommended was already in the prompt: the block must vanish."""
    picked = [_skill("alpha", "Alpha skill.")]
    index = _FakeIndex(picked=picked)
    off = await _off_block(index)

    on = _hooks(
        _FakeIndex(picked=picked),
        classifier=_FakeClassifier(
            _Recommendation(
                resources=(_Candidate("skill", "alpha", "Alpha skill.", "skill://alpha"),),
                vendor="typesafe",
            )
        ),
    )
    assert await session_factory._select_knowledge_block(on, OFF_QUERY, task_id="t1") == off


# ---------------------------------------------------------------------------
# §7 step 3: additive, deduped, bounded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_recommendation_block_is_additive_and_deduped() -> None:
    """Appended after the selection and the catalogue, minus what they already say."""
    picked = [_skill("alpha", "Alpha skill.")]
    # The REAL catalogue shape: at most one server, named with the SAME
    # ``mcp://`` URL the roster builds, which is what the dedupe keys on.
    catalogue = "<mcps>\n- hubspot: CRM contacts. Read `mcp://hubspot`.\n</mcps>"
    index = _FakeIndex(picked=picked)
    off = await _off_block(index, catalogue=catalogue)

    classifier = _FakeClassifier(
        _Recommendation(
            resources=(
                _Candidate("skill", "alpha", "Alpha skill.", "skill://alpha"),
                _Candidate("guide", "tunnel", "tunnel guide", "guide://tunnel"),
                _Candidate("mcp", "hubspot", "CRM", "mcp://hubspot"),
                _Candidate("mcp", "slack", "Team messages", "mcp://slack"),
            ),
            vendor="typesafe",
            cost_usd=0.000042,
            latency_s=0.31,
        ),
        notice_line="Classification: 2 via typesafe",
    )
    on = _hooks(
        _FakeIndex(picked=picked),
        classifier=classifier,
        catalogue=catalogue,
    )
    block = await session_factory._select_knowledge_block(on, OFF_QUERY, task_id="t1")

    assert block.startswith(off + "\n\n")
    tail = block[len(off) + 2 :]
    assert tail == "\n".join(
        [
            "<resource_recommendations>",
            "These may help with this request — read the ones that actually fit, "
            "ignore the rest:",
            "- guide://tunnel",
            "- mcp://slack",
            "</resource_recommendations>",
        ]
    )
    # Advisory, never imperative: the two words the phrasing rules (§7) turn on.
    assert "may help" in tail and "ignore the rest" in tail


@pytest.mark.asyncio
async def test_the_block_is_capped_by_the_configured_maximum() -> None:
    """``values.classification.maxRecommendations`` bounds what is injected."""
    classifier = _FakeClassifier(
        _Recommendation(
            resources=tuple(
                _Candidate("skill", f"s{i}", f"skill {i}", f"skill://s{i}") for i in range(5)
            ),
            vendor="typesafe",
        )
    )
    hooks = _hooks(classifier=classifier)
    hooks.classification_max_recommendations = 2

    block = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")

    assert block.count("- skill://") == 2
    # ...and the REQUEST carries the same number, because the package takes the
    # field as an upper bound over its own settings read (``min`` of the two).
    assert classifier.requests[0].max_recommendations == 2


@pytest.mark.asyncio
async def test_the_notice_rides_the_session_notice_path_once_per_message() -> None:
    """One line per admitted user message, through the bound sink, at info."""
    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "tunnel guide", "guide://tunnel"),),
            vendor="typesafe",
        ),
        notice_line="Classification: 1 resource recommendation via typesafe",
    )
    hooks = _hooks(
        _FakeIndex(picked=[_skill("alpha", "Alpha skill.")]),
        classifier=classifier,
    )
    delivered: list[tuple[str, str]] = []
    hooks.notice_sink = lambda text, kind="warning": delivered.append((text, kind))

    await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    # Same task, tool continuation: frozen, so no second call and no second line.
    await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    assert delivered == [("Classification: 1 resource recommendation via typesafe", "info")]

    # A new admitted user message is a new notice (and a new classification pass).
    await session_factory._select_knowledge_block(hooks, "a second question", task_id="t2")
    assert len(delivered) == 2
    assert len(classifier.requests) == 2


@pytest.mark.asyncio
async def test_a_silent_seam_emits_no_notice() -> None:
    """The seam owns the gate: ``notice()`` returning ``None`` means say nothing."""
    hooks = _hooks(
        classifier=_FakeClassifier(
            _Recommendation(
                resources=(_Candidate("guide", "tunnel", "g", "guide://tunnel"),),
                vendor="typesafe",
            ),
            notice_line=None,
        )
    )
    delivered: list[str] = []
    hooks.notice_sink = lambda text, kind="warning": delivered.append(text)

    await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")

    assert delivered == []


@pytest.mark.asyncio
async def test_a_provider_without_a_session_binds_no_sink() -> None:
    """The benchmark preflight renders the prompt without a facade; nothing breaks."""
    hooks = _hooks(
        classifier=_FakeClassifier(
            _Recommendation(
                resources=(_Candidate("guide", "tunnel", "g", "guide://tunnel"),),
                vendor="typesafe",
            ),
            notice_line="Classification: 1 via typesafe",
        )
    )
    session_factory.attach_classification_notices(cast(Any, object()), hooks)

    assert hooks.notice_sink is None
    block = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    assert "guide://tunnel" in block


def test_a_bound_sink_is_the_sessions_own_notice_event() -> None:
    """``attach_classification_notices`` binds ``Session._stream_notice``."""

    async def _stream_notice(text: str, kind: str = "warning") -> None:
        return None

    class _Session:
        pass

    session = _Session()
    session._stream_notice = _stream_notice  # type: ignore[attr-defined]
    hooks = session_factory._KnowledgeHooks()

    session_factory.attach_classification_notices(session, hooks)  # type: ignore[arg-type]

    assert hooks.notice_sink is _stream_notice


# ---------------------------------------------------------------------------
# §7 step 2 and the latency budget: concurrent, bounded, roster built once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_classification_runs_concurrently_with_the_selection() -> None:
    """The two legs INTERLEAVE: neither waits for the other to finish.

    Interleaving rather than a wall-clock threshold, because this host runs many
    concurrent suites and a duration bound would flake; the order of the four
    events is a property of the code, not of the machine. The wall-clock ceiling
    below is the loose second witness (sum would be 0.4 s).
    """
    events: list[str] = []
    index = _FakeIndex(picked=[], delay=0.2, events=events)
    classifier = _FakeClassifier(_Recommendation(skipped="disabled"), delay=0.2, events=events)
    hooks = _hooks(index, classifier=classifier)

    started = time.monotonic()
    await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    elapsed = time.monotonic() - started

    assert events == ["select-start", "classify-start", "select-end", "classify-end"]
    assert elapsed < 0.4, elapsed


@pytest.mark.asyncio
async def test_a_slow_classifier_cannot_hold_the_turn_past_its_deadline() -> None:
    """``values.classification.timeoutMs`` bounds the turn, whatever the seam does."""
    index = _FakeIndex(picked=[_skill("alpha", "Alpha skill.")])
    off = await _off_block(index)

    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "g", "guide://tunnel"),),
            vendor="typesafe",
        ),
        delay=5.0,
        timeout_s=0.05,
    )
    hooks = _hooks(_FakeIndex(picked=[_skill("alpha", "Alpha skill.")]), classifier=classifier)

    started = time.monotonic()
    block = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, elapsed
    assert block == off


def test_the_deadline_comes_from_the_seam_or_the_documented_default() -> None:
    """A seam that publishes ``timeout_s`` wins; a foreign one gets the §8 default."""
    assert session_factory._classification_deadline_s(_FakeClassifier(timeout_s=0.25)) == 0.25
    assert (
        session_factory._classification_deadline_s(_FakeClassifier())
        == session_factory.DEFAULT_CLASSIFICATION_TIMEOUT_MS / 1000.0
    )


@pytest.mark.asyncio
async def test_the_candidate_roster_is_built_once_per_roster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per SESSION, not per user message: rebuilding it is pure per-turn overhead."""
    builds = 0
    original = session_factory._build_classification_roster

    def counting(hooks: session_factory._KnowledgeHooks) -> tuple[Any, ...]:
        nonlocal builds
        builds += 1
        return original(hooks)

    monkeypatch.setattr(session_factory, "_build_classification_roster", counting)
    hooks = _hooks(
        classifier=_FakeClassifier(_Recommendation(skipped="disabled")), servers=("hubspot",)
    )

    await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    await session_factory._select_knowledge_block(hooks, "second", task_id="t2")
    await session_factory._select_knowledge_block(hooks, "third", task_id="t3")

    assert builds == 1


@pytest.mark.asyncio
async def test_the_roster_is_rebuilt_when_the_servers_change() -> None:
    """The cache key is the INPUTS: a new server set is a new roster."""
    hooks = _hooks(
        classifier=_FakeClassifier(_Recommendation(skipped="disabled")), servers=("hubspot",)
    )
    first = session_factory._classification_roster(hooks)

    hooks.mcp_server_names = ("hubspot", "slack")
    second = session_factory._classification_roster(hooks)

    assert first is not second
    assert [row.name for row in second] == ["alpha", "hubspot", "slack"]


def test_the_roster_is_the_routers_own_view_with_harness_owned_text() -> None:
    """Non-hidden skills and guides, plus each server's capability hint (§6, §7)."""
    index = _FakeIndex(
        [
            _skill("alpha", "Alpha skill."),
            _skill("secret", "Hidden skill.", hide=True),
            _skill("tunnel", "Tunnel guide.", kind="guide"),
            _skill("nodesc", ""),
        ]
    )
    hooks = _hooks(index, servers=("hubspot", "my-custom-server"))

    roster = session_factory._classification_roster(hooks)

    assert [(row.kind, row.name, row.resource_url) for row in roster] == [
        ("skill", "alpha", "skill://alpha"),
        ("guide", "tunnel", "guide://tunnel"),
        ("mcp", "hubspot", "mcp://hubspot"),
        ("mcp", "my-custom-server", "mcp://my-custom-server"),
    ]
    # The release-owned hint for a known server, and the harness's own fallback
    # for a custom one — never config-authored or remote-authored prose.
    assert roster[2].description == "CRM contacts, companies, deals, marketing, and sales."
    assert roster[3].description == session_factory._MCP_DEFAULT_CAPABILITY


def test_a_warm_request_only_carries_the_message_and_the_cached_roster() -> None:
    """The request the seam receives reuses the roster object, per message."""
    hooks = _hooks(classifier=None, servers=("hubspot",))

    first = session_factory._classification_request(hooks, "one")
    second = session_factory._classification_request(hooks, "two")

    assert first.candidates is second.candidates
    assert (first.user_message, second.user_message) == ("one", "two")
    assert first.max_recommendations == hooks.classification_max_recommendations
    assert first.context is None


# ---------------------------------------------------------------------------
# The build decision (§8): off imports nothing, on builds the real service
# ---------------------------------------------------------------------------


def test_auto_off_builds_no_seam_and_never_imports_the_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The default install's import graph must be what it was before the layer."""
    monkeypatch.delitem(sys.modules, "local_operator.classification", raising=False)
    monkeypatch.delitem(sys.modules, "local_operator.classification.service", raising=False)
    manager = ConfigManager(tmp_path)
    hooks = session_factory._KnowledgeHooks()
    warnings: list[str] = []

    session_factory._attach_classification(hooks, manager, cast(Any, None), warnings)

    assert hooks.classifier is None
    assert warnings == []
    assert "local_operator.classification" not in sys.modules
    assert "local_operator.classification.service" not in sys.modules


def test_auto_on_builds_the_service_from_the_snapshot(tmp_path: Path) -> None:
    """On: the real service, over a SNAPSHOT of the section (NEW_SESSIONS scope)."""
    manager = ConfigManager(tmp_path)
    manager.set_config_value("classification", {"auto": True, "maxRecommendations": 5})
    hooks = session_factory._KnowledgeHooks()
    warnings: list[str] = []

    session_factory._attach_classification(hooks, manager, cast(Any, None), warnings)

    assert warnings == []
    assert hooks.classifier is not None
    assert type(hooks.classifier).__name__ == "ClassificationService"
    assert hooks.classification_max_recommendations == 5

    # The snapshot is a copy: a later edit does not reach the built service.
    manager.set_config_value("classification", {"auto": True, "maxRecommendations": 9})
    assert hooks.classification_max_recommendations == 5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, False),
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("off", False),
        ("yes", True),
        ("maybe", False),
    ],
)
def test_the_enablement_read_matches_the_services_own(raw: Any, expected: bool) -> None:
    """``auto`` is read the way the service reads it, including YAML 1.1 spellings."""
    section: dict[str, Any] = {} if raw is None else {"auto": raw}

    assert session_factory._classification_enabled(section) is expected


def test_the_wiring_defaults_match_the_registry_rows() -> None:
    """The two constants the wiring restates are the registry's defaults too."""
    from local_operator import settings_io

    assert (
        session_factory.DEFAULT_CLASSIFICATION_MAX_RECOMMENDATIONS
        == settings_io.BY_KEY["classification.maxRecommendations"].default
    )
    assert (
        session_factory.DEFAULT_CLASSIFICATION_TIMEOUT_MS
        == settings_io.BY_KEY["classification.timeoutMs"].default
    )


# ---------------------------------------------------------------------------
# Costs are logged, not accrued (see ``_log_classification_cost``)
# ---------------------------------------------------------------------------


def test_the_cost_line_carries_the_vendors_own_figures(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """vendor/model/tokens/cost, at INFO, in the units the vendor reported."""
    with caplog.at_level("INFO", logger="local_operator.session_factory"):
        session_factory._log_classification_cost(
            _Recommendation(
                resources=(_Candidate("guide", "tunnel", "g", "guide://tunnel"),),
                vendor="typesafe",
                cost_usd=0.000042,
                latency_s=0.31,
            )
        )

    assert "vendor=typesafe" in caplog.text
    assert "$0.000042" in caplog.text
    assert "resources=1" in caplog.text


def test_a_skipped_pass_logs_nothing_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """No vendor answered, so nothing was spent: no INFO line, no noise."""
    with caplog.at_level("INFO", logger="local_operator.session_factory"):
        session_factory._log_classification_cost(_Recommendation(skipped="no-vendor"))

    assert caplog.text == ""
