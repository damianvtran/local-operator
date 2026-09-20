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
import contextlib
import logging
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
    #: The package's ``Recommendation`` carries the vendor's token counts (they are
    #: what ``_log_classification_cost`` prints), so the stand-in does too — "field
    #: for field" is the contract this double exists to keep.
    input_tokens: int | None = None
    output_tokens: int | None = None
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
    """The seam double: ``recommend_resources``, and nothing else.

    It used to carry a ``notice`` half as well, because the harness called one; the
    render path is deleted, so a double that still published it would be testing a
    contract nothing asks for.
    """

    def __init__(
        self,
        recommendation: _Recommendation | None = None,
        *,
        delay: float = 0.0,
        timeout_s: float | None = None,
        raises: BaseException | None = None,
        events: list[str] | None = None,
    ) -> None:
        self.recommendation = recommendation or _Recommendation()
        self.delay = delay
        self.timeout_s = timeout_s
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

    # The first three events ARE the property: the classify leg started before the
    # select leg ended. Its END is deliberately not in the list — the turn waits 50 ms
    # and the fake takes 200 ms, so the answer lands after the turn has moved on, which
    # is what the bounded wait means. Asserting the end here made the test a race
    # between two 200 ms sleeps (it passed alone and failed under load).
    assert events[:3] == ["select-start", "classify-start", "select-end"], events
    assert elapsed < 0.4, elapsed
    for call in [call for call in hooks.classification_outstanding if not call.task.done()]:
        with contextlib.suppress(Exception):
            await call.task
    assert "classify-end" in events, events


@pytest.mark.asyncio
async def test_a_slow_seam_gets_only_the_wait_and_no_deadline_of_its_own() -> None:
    """The turn waits ``waitMs``, never the seam's own ``timeout_s``.

    The wait is ``min(waitMs, the call's deadline)``: waiting longer than the call
    could possibly take would spend budget on nothing. Both halves are asserted
    here, because the pair is what makes the number trustworthy — the seam's 50 ms
    deadline is reached only because the configured wait is larger than it.
    """
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
    first_hooks = _hooks(
        _FakeIndex(picked=[_skill("alpha", "Alpha skill.")]), classifier=classifier
    )

    # A wait budget FAR past the seam's own deadline: the deadline wins.
    first_hooks.classification_wait_s = 5.0
    started = time.monotonic()
    block = await session_factory._select_knowledge_block(first_hooks, OFF_QUERY, task_id="t1")
    elapsed = time.monotonic() - started

    assert elapsed < 1.0, elapsed
    assert block == off
    # ...and a wait budget well under it: the BUDGET wins, not the seam's delay.
    hooks = _hooks(_FakeIndex(picked=[_skill("alpha", "Alpha skill.")]), classifier=classifier)
    hooks.classification_wait_s = 0.05
    started = time.monotonic()
    block = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    elapsed = time.monotonic() - started

    assert elapsed < 0.5, elapsed
    assert block == off
    # No task is left behind by either turn's own budget: the slow call is the
    # SAME one, still outstanding, and the harvest will look at it next message.
    assert len(hooks.classification_outstanding) == 1
    for call in (*first_hooks.classification_outstanding, *hooks.classification_outstanding):
        call.task.cancel()  # keep the loop quiet at teardown
    await asyncio.sleep(0)


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
    """An EXPLICIT ``auto: false`` must still cost nothing — not even an import.

    The default is ON since 2026-09-18, so this test names the off value rather
    than relying on the absent key; the absent key's behaviour (seam built) is
    ``test_auto_on_builds_the_service_from_the_snapshot``'s sibling below.
    """
    monkeypatch.delitem(sys.modules, "local_operator.classification", raising=False)
    monkeypatch.delitem(sys.modules, "local_operator.classification.service", raising=False)
    manager = ConfigManager(tmp_path)
    manager.set_config_value("classification", {"auto": False})
    hooks = session_factory._KnowledgeHooks()
    warnings: list[str] = []

    session_factory._attach_classification(hooks, manager, cast(Any, None), warnings)

    assert hooks.classifier is None
    assert warnings == []
    assert "local_operator.classification" not in sys.modules
    assert "local_operator.classification.service" not in sys.modules


def test_an_absent_key_builds_the_seam(tmp_path: Path) -> None:
    """The flip's whole content: no key at all means the layer is ON."""
    manager = ConfigManager(tmp_path)
    hooks = session_factory._KnowledgeHooks()
    warnings: list[str] = []

    session_factory._attach_classification(hooks, manager, cast(Any, None), warnings)

    assert warnings == []
    assert hooks.classifier is not None
    assert type(hooks.classifier).__name__ == "ClassificationService"


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
        # The absent case is the default, and the default is ON since
        # 2026-09-18 — this row is the flip (it read False before).
        (None, True),
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        ("off", False),
        ("yes", True),
        # Unreadable reads as the DEFAULT, not as off: the service's own reader
        # passes ``DEFAULT_AUTO`` as its fallback, and two readings of one toggle
        # must agree on garbage as well as on YAML 1.1 spellings.
        ("maybe", True),
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
                input_tokens=4238,
                output_tokens=380,
                latency_s=0.31,
            )
        )

    assert "vendor=typesafe" in caplog.text
    assert "$0.000042" in caplog.text
    assert "resources=1" in caplog.text
    # Input is the whole bill at this vendor's pricing, so the counts have to be
    # on the line rather than a placeholder.
    assert "tokens=4238/380" in caplog.text


def test_a_call_that_reports_no_counts_prints_none_rather_than_a_zero(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A missing or cleared count is ``-``: ``0`` would claim a call reported zero.

    Three shapes reach this: a seam that publishes no counts at all, the
    package's own cache hit (which clears them because that call spent nothing),
    and a 200 whose vendor omitted ``usage`` (``vendors._count`` turns that
    absence into ``None``). Both calls below must render ``-`` — asserted by
    COUNTING the placeholder lines, so a single accidental match cannot pass this.
    """
    with caplog.at_level("INFO", logger="local_operator.session_factory"):
        session_factory._log_classification_cost(
            _Recommendation(vendor="typesafe", cost_usd=None, input_tokens=None, output_tokens=None)
        )
        session_factory._log_classification_cost(
            _Recommendation(vendor="typesafe", cost_usd=None, latency_s=0.1)
        )
        # The other side of the distinction, pinned where it RENDERS: a leg that
        # really reported zero must still print ``0``. Without this, collapsing
        # zero into the placeholder (``if not value: return "-"``) would leave the
        # suite green while re-erasing the difference this round exists to keep.
        session_factory._log_classification_cost(
            _Recommendation(
                vendor="typesafe", cost_usd=0.000021, input_tokens=0, output_tokens=0, latency_s=0.2
            )
        )

    assert caplog.text.count("tokens=-/-") == 2
    assert "tokens=0/0" in caplog.text


def test_a_skipped_pass_logs_nothing_at_info(caplog: pytest.LogCaptureFixture) -> None:
    """No vendor answered, so nothing was spent: no INFO line, no noise."""
    with caplog.at_level("INFO", logger="local_operator.session_factory"):
        session_factory._log_classification_cost(_Recommendation(skipped="no-vendor"))

    assert caplog.text == ""


# ---------------------------------------------------------------------------
# §5a rule 6: the wait is bounded, and a late answer is carried, not lost
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hanging_seam_costs_the_turn_only_the_wait_budget() -> None:
    """The turn pays ``waitMs``, and the answer is delivered by the next message.

    The operator's budget is our OWN overhead per user message, and on a real roster
    the vendor's answer takes ~250 ms median — waiting for it would put the vendor's
    model time on the critical path. So the turn waits ``values.classification.waitMs``
    and no longer, and what it gave up on is not lost: the call keeps running, and the
    next admitted user message re-renders the knowledge block WITH the late answer.
    (That is the case this test covers — the turn had ENDED. An answer that lands while
    its own turn is still running is delivered into that turn's next model step
    instead, which is the sibling test below.)
    That re-render is the delivery mechanism (a changed knowledge section is
    journaled by the harness as a ``[session-state]`` update), which is why nothing
    here writes a bespoke host-state row.
    """
    index = _FakeIndex(picked=[_skill("alpha", "Alpha skill.")])
    off = await _off_block(index)
    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "Tunnel guide.", "guide://tunnel"),),
            vendor="typesafe",
        ),
        delay=0.35,
        timeout_s=1.5,
    )
    hooks = _hooks(_FakeIndex(picked=[_skill("alpha", "Alpha skill.")]), classifier=classifier)
    hooks.classification_wait_s = 0.05

    started = time.monotonic()
    first = await session_factory._select_knowledge_block(hooks, "first task", task_id="t1")
    waited = time.monotonic() - started

    # The BUDGET, not the vendor's 350 ms — and the prompt is byte-identical to
    # what it would have been with no layer at all.
    assert waited < 0.2, waited
    assert first == off

    await asyncio.sleep(0.4)  # the call lands after its own turn ended

    second = await session_factory._select_knowledge_block(hooks, "second task", task_id="t2")

    assert "guide://tunnel" in second
    assert second.count("guide://tunnel") == 1
    # DELIVERED ONCE, by the turn that actually carries it (the second one).


@pytest.mark.asyncio
async def test_a_late_answer_is_delivered_once_and_only_when_it_exists() -> None:
    """Consumed exactly once; a session whose answers never arrive appends nothing."""
    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "Tunnel guide.", "guide://tunnel"),),
            vendor="typesafe",
        ),
        delay=0.3,
    )
    hooks = _hooks(classifier=classifier)
    hooks.classification_wait_s = 0.05

    await session_factory._select_knowledge_block(hooks, "first", task_id="t1")
    await asyncio.sleep(0.35)  # call 1 lands, late
    # NOTHING MORE WILL EVER BE RECOMMENDED: every later call is answered empty, so
    # anything that appears in a later block can only be call 1's answer arriving
    # twice — which is exactly what must not happen.
    classifier.recommendation = _Recommendation(skipped="empty-roster")

    second = await session_factory._select_knowledge_block(hooks, "second", task_id="t2")
    assert second.count("guide://tunnel") == 1
    await asyncio.sleep(0.35)  # call 2 lands, empty

    third = await session_factory._select_knowledge_block(hooks, "third", task_id="t3")
    assert "guide://tunnel" not in third, "a delivered answer must never be delivered twice"
    assert "guide://tunnel" not in third


@pytest.mark.asyncio
async def test_the_breaker_counts_the_vendor_deadline_once_per_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hung vendor opens the breaker after three CALLS, and no turn pays one (§4).

    The wiring's deadline and the service's used to be the same number started
    microseconds apart, and the service's in-flight call is shielded — so when the
    turn's copy won, the service recorded its failure AFTER the turn had already
    been handed an empty block: four messages paid the full deadline and the breaker
    opened on the fourth (QA round 1, Q1). The turn's number is now a WAIT and the
    service's is the deadline, so the count is the vendor's own and the turn's cost
    is the budget. Both halves are asserted here, on the WIRING path with the real
    service: the vendor-leg call count after four messages, and that no message cost
    the deadline.
    """
    from local_operator.classification.service import ClassificationService
    from local_operator.classification.vendors import VENDOR_CLASSES
    from local_operator.credentials import CredentialManager
    from tests.unit.classification.support import LegBehaviour, leg_class

    behaviour = LegBehaviour(name="openrouter", delay_s=5.0)
    monkeypatch.setattr(
        "local_operator.classification.vendors.VENDOR_CLASSES",
        {**VENDOR_CLASSES, "openrouter": leg_class(behaviour)},
        raising=True,
    )
    service = ClassificationService(
        manager=CredentialManager(tmp_path),
        settings={"classification": {"auto": True, "vendor": "openrouter", "timeoutMs": 150}},
    )
    hooks = _hooks(_FakeIndex(picked=[_skill("alpha", "Alpha skill.")]), classifier=service)
    hooks.classification_wait_s = 0.02

    waits: list[float] = []
    for index in range(4):
        started = time.monotonic()
        await session_factory._select_knowledge_block(
            hooks, f"message {index}", task_id=f"t{index}"
        )
        waits.append(time.monotonic() - started)
        # Past the service's OWN deadline, so its accounting has landed before the
        # next message. The claim is that the deadline belongs to the CALL, not that
        # a message may pay it.
        await asyncio.sleep(0.2)

    # Warm messages: the BUDGET, far under the 150 ms deadline they would have paid
    # under the old wiring. The FIRST message of a session additionally pays the
    # package's local cold path (the state serialization and lazy imports behind the
    # first call), measured at ~95 ms on this host — a one-off the wait cannot bound
    # because it runs before the first await, and named here rather than hidden: it is
    # the package's cost, not the wait's.
    assert max(waits[1:]) < 0.1, waits
    assert waits[0] < 0.5, waits
    assert len(behaviour.calls) == 3, "three consecutive failures open the breaker"
    for call in hooks.classification_outstanding:
        call.task.cancel()
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# Skill-tree fixtures for the roster tests
# ---------------------------------------------------------------------------


def _write_skill(root: Path, name: str, description: str) -> Path:
    """A real skill tree entry, so the fingerprint and the scanner both see it."""
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\n\nBody for {name}.\n",
        encoding="utf-8",
    )
    return directory


def test_a_skill_installed_mid_session_becomes_a_candidate(tmp_path: Path) -> None:
    """The roster is resolved from the tree AS IT IS, not as it was at session start.

    The index is a startup snapshot (its vectors are far too expensive to rebuild
    per message), so a skill authored after boot was invisible to the roster — and
    the roster is the surface whose whole purpose is making installed resources
    reachable. Here the fingerprint-gated rescan is what puts it back.
    """
    root = tmp_path / "skills"
    _write_skill(root, "alpha", "Alpha skill.")
    index = _FakeIndex([_skill("alpha", "Alpha skill.")])
    hooks = _hooks(index, classifier=None, servers=())
    hooks.skill_roots = [root]
    hooks.skills_fingerprint = session_factory._skills_fingerprint(hooks)

    before = [row.name for row in session_factory._classification_roster(hooks)]
    assert before == ["alpha"]

    # The operator installs a skill while the session is live. The dict handed to
    # the resolver (and to every running child) must be updated IN PLACE — a
    # rebound name would leave them all on the stale mapping.
    shared = hooks.skills_by_name
    _write_skill(root, "beta", "Beta skill, authored mid-session.")

    after = [row.name for row in session_factory._classification_roster(hooks)]

    assert sorted(after) == ["alpha", "beta"]
    assert "beta" in hooks.skills_by_name
    assert hooks.skills_by_name is shared


@pytest.mark.asyncio
async def test_a_skill_tree_change_reopens_the_frozen_knowledge_block(tmp_path: Path) -> None:
    """The freeze must not outlive the tree it describes.

    Selection is frozen per admitted user message so a tool loop does not re-render
    it, and that freeze is what hid a new skill from the rest of the session — and
    from subagents, which inherit the parent's frozen block. The previous render is
    PARKED rather than dropped, because a child's block is built synchronously and
    must not come back empty in the window before the next render.
    """
    root = tmp_path / "skills"
    _write_skill(root, "alpha", "Alpha skill.")
    index = _FakeIndex([_skill("alpha", "Alpha skill.")], picked=[_skill("alpha", "Alpha skill.")])
    hooks = _hooks(index, classifier=None, servers=())
    hooks.skill_roots = [root]
    hooks.skills_fingerprint = session_factory._skills_fingerprint(hooks)

    rendered = await session_factory._select_knowledge_block(hooks, "alpha please", task_id="t1")
    assert hooks.frozen_block == rendered
    assert hooks.frozen_block

    _write_skill(root, "beta", "Beta skill, authored mid-session.")
    session_factory._refresh_knowledge_freshness(hooks)

    assert hooks.frozen_block is None
    assert hooks.superseded_block == rendered
    assert hooks.frozen_task_id is None


@pytest.mark.asyncio
async def test_an_unchanged_tree_leaves_the_frozen_block_alone(tmp_path: Path) -> None:
    """The gate is a fingerprint, not a clock: no change, no re-render, no cost."""
    root = tmp_path / "skills"
    _write_skill(root, "alpha", "Alpha skill.")
    index = _FakeIndex([_skill("alpha", "Alpha skill.")], picked=[_skill("alpha", "Alpha skill.")])
    hooks = _hooks(index, classifier=None, servers=())
    hooks.skill_roots = [root]
    hooks.skills_fingerprint = session_factory._skills_fingerprint(hooks)

    rendered = await session_factory._select_knowledge_block(hooks, "alpha please", task_id="t1")
    session_factory._refresh_knowledge_freshness(hooks)

    assert hooks.frozen_block == rendered
    assert hooks.superseded_block == ""


def test_a_large_roster_is_shortlisted_by_relevance() -> None:
    """Hundreds of skills: the dozen that travel are chosen, not just the first dozen."""
    rows = [_skill(f"fleet-{index:03d}", "Fleet automation.") for index in range(40)]
    rows.append(_skill("flavia-adverse-media", "Adverse media screening for a person."))
    hooks = _hooks(_FakeIndex(rows), classifier=None, servers=())
    hooks.classification_max_candidates = 12
    # The wiring captures the package's shortlist at attach time so the MESSAGE path
    # never imports the package; a seam without it sends the roster unchanged.
    from local_operator.classification import shortlist

    hooks.classification_shortlist = shortlist

    request = session_factory._classification_request(
        hooks, "run an adverse media screen for Flavia"
    )

    names = [candidate.name for candidate in request.candidates]
    assert "flavia-adverse-media" in names
    assert len(names) == 12


def test_a_skill_named_like_a_guide_does_not_evict_the_guide(tmp_path: Path) -> None:
    """The rescan's union is keyed on (kind, name), not on name alone.

    A user skill named after a packaged guide (`tunnel`, `browser`, `mcp`, …) is a
    real collision: keying the merge on the name let the skill REPLACE the guide row,
    so a resource the router still offers silently stopped being a candidate (agent
    review round 1).
    """
    root = tmp_path / "skills"
    _write_skill(root, "alpha", "Alpha skill.")
    index = _FakeIndex(
        [
            _skill("alpha", "Alpha skill."),
            _skill("tunnel", "Tunnel guide.", kind="guide"),
        ]
    )
    hooks = _hooks(index, classifier=None, servers=())
    hooks.skill_roots = [root]
    hooks.skills_fingerprint = session_factory._skills_fingerprint(hooks)

    # A user skill whose NAME is the guide's, installed after the index was built.
    _write_skill(root, "tunnel", "A user skill that happens to share the guide's name.")

    rows = [(row.kind, row.name) for row in session_factory._classification_roster(hooks)]

    assert ("guide", "tunnel") in rows
    assert ("skill", "tunnel") in rows


# ---------------------------------------------------------------------------
# A late answer belongs to ITS OWN message (live defect, 2026-09-18)
#
# Measured on the operator's machine: the decision vendor answers in 540-1500 ms
# against a 50 ms wait, so before this the answer for a message could never reach
# that message — it rode the NEXT user message instead. That is how a child session
# whose message asked about a Slack support thread was handed `guide://mcp` (the
# previous message's answer) while its roster plainly contained `mcp://slack`.
# ---------------------------------------------------------------------------


def _slow_classifier(recommendation: _Recommendation) -> _FakeClassifier:
    """A seam whose answer outlives the wait, so the turn has to carry it."""
    return _FakeClassifier(recommendation, delay=0.25)


def _slack_recommendation() -> _Recommendation:
    return _Recommendation(
        resources=(_Candidate("mcp", "slack", "Team messages, channels, threads.", "mcp://slack"),),
    )


@pytest.mark.asyncio
async def test_an_answer_that_misses_the_wait_lands_in_the_same_turn() -> None:
    """The model must see it on its NEXT STEP of the same turn, not next message."""
    index = _FakeIndex(picked=[])
    classifier = _slow_classifier(_slack_recommendation())
    hooks = _hooks(index, classifier=classifier)
    hooks.classification_wait_s = 0.01

    first = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    assert "mcp://slack" not in first, "the turn gave up before the vendor answered"

    await asyncio.sleep(0.3)  # the answer lands while the turn is still running

    second = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    assert "mcp://slack" in second, "the answer must reach its own message"

    # Delivered once: the following step of the SAME task re-renders the frozen block.
    third = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    assert "mcp://slack" in third

    # And a NEW message does not repeat it: the pending slot was consumed.
    fourth = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t2")
    assert "mcp://slack" not in fourth


@pytest.mark.asyncio
async def test_an_answer_for_an_older_message_still_rides_the_next_one() -> None:
    """The old behaviour, kept for the case it exists for: the turn has ENDED.

    A call that is still in flight when its message's turn finishes has no next step
    to reach, so it is carried — and labelled — as an answer to the previous message.
    """
    index = _FakeIndex(picked=[])
    classifier = _slow_classifier(_slack_recommendation())
    hooks = _hooks(index, classifier=classifier)
    hooks.classification_wait_s = 0.01

    assert "mcp://slack" not in await session_factory._select_knowledge_block(
        hooks, OFF_QUERY, task_id="t1"
    )
    await asyncio.sleep(0.3)

    later = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t2")
    assert "mcp://slack" in later


@pytest.mark.asyncio
async def test_a_service_with_no_provider_is_not_waited_on_and_logs_nothing(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No recommender provider is a configuration, not an incident: silence and no wait.

    The probe is asked BEFORE the request is built, so an install with nothing logged in
    pays neither the wait nor a line in the log — which is what the operator asked for,
    while a provider that IS configured and fails stays loud (``test_service.py``).
    """

    class _NoProvider(_FakeClassifier):
        async def provider_available(self) -> bool:
            self.events.append("probe")
            return False

    index = _FakeIndex(picked=[])
    classifier = _NoProvider(_slack_recommendation())
    hooks = _hooks(index, classifier=classifier)
    hooks.classification_wait_s = 5.0  # a wait this test would time out on

    started = time.monotonic()
    with caplog.at_level(logging.INFO, logger="local_operator.session_factory"):
        block = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    elapsed = time.monotonic() - started

    assert classifier.events == ["probe"], "no call was built, let alone waited on"
    assert classifier.requests == []
    assert elapsed < 0.5, f"the probe must not be followed by a wait (took {elapsed:.2f}s)"
    assert [record for record in caplog.records if record.levelno >= logging.INFO] == []
    assert block == await _off_block(_FakeIndex(picked=[]))


@pytest.mark.asyncio
async def test_superseding_the_freeze_keeps_what_the_frozen_block_selected() -> None:
    """The provider hands ``""`` to a render it believes is unchanged — the re-render
    that must supersede the block has to reuse the query that selected it.

    Reproduced before this existed (review round 1, R1-2) by calling the callee the way
    the provider does on a model's second step: step 1 carried the selected skill,
    step 2 (same task, ``query=""``) carried the advisory but the skill was GONE — and
    children inherit that emptied block. That is a worse outcome than the late answer it
    was trying to deliver, and it fires on essentially every tool-using turn.
    """
    index = _FakeIndex(picked=[_skill("alpha", "Alpha skill.")])
    classifier = _slow_classifier(_slack_recommendation())
    hooks = _hooks(index, classifier=classifier)
    hooks.classification_wait_s = 0.01

    first = await session_factory._select_knowledge_block(
        hooks, "a question about tunnels", task_id="t1"
    )
    assert "alpha" in first, "the selection rides the frozen block"

    await asyncio.sleep(0.3)  # the answer arrives mid-turn

    # Exactly what the provider does for an unchanged render: an EMPTY query.
    second = await session_factory._select_knowledge_block(hooks, "", task_id="t1")
    assert "mcp://slack" in second, "the in-turn answer must be delivered"
    assert "alpha" in second, "and it must not cost the block its selection"
    assert len(index.calls) == 2, "the re-render re-selects with the frozen query"
    assert len(classifier.requests) == 1, "and never asks a second time for this message"


@pytest.mark.asyncio
async def test_the_provider_probe_is_inside_the_wait_budget() -> None:
    """Resolution is not free and not always local — it must be bounded by ``waitMs``.

    ``resolve_vendor`` can refresh an expired Radient OAuth grant over the network, so a
    probe awaited BEFORE the budgeted task would put that I/O outside the one bound the
    layer promises (review round 1, R1-4). Here the probe alone takes 200x the wait.
    """

    class _SlowProbe(_FakeClassifier):
        async def provider_available(self) -> bool:
            await asyncio.sleep(0.5)
            self.events.append("probe-done")
            return True

    classifier = _SlowProbe(_slack_recommendation())
    hooks = _hooks(_FakeIndex(picked=[]), classifier=classifier)
    hooks.classification_wait_s = 0.05

    started = time.monotonic()
    block = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    elapsed = time.monotonic() - started

    assert elapsed < 0.2, f"the probe must be bounded by the wait (took {elapsed:.2f}s)"
    assert block == ""
    assert len(hooks.classification_outstanding) == 1, "and it is still running, harvestable"
    await asyncio.sleep(0.6)
    for call in hooks.classification_outstanding:
        call.task.cancel()


@pytest.mark.asyncio
async def test_an_invalidated_freeze_does_not_ask_for_the_same_message_twice(
    tmp_path: Path,
) -> None:
    """The freeze is not the only thing a later render of a message can have lost.

    A skill installed or edited mid-turn invalidates it — that is what
    ``_refresh_knowledge_freshness`` exists for — and that render still has this message's
    answer waiting. Keying the "do not ask again" guard on the freeze alone left that
    window placing a second vendor call for a message that already had an answer (review
    round 2, R2-2), which is a real call at a real price for nothing.
    """
    root = tmp_path / "skills"
    _write_skill(root, "alpha", "Alpha skill.")
    index = _FakeIndex([_skill("alpha", "Alpha skill.")], picked=[_skill("alpha", "Alpha skill.")])
    classifier = _slow_classifier(_slack_recommendation())
    hooks = _hooks(index, classifier=classifier)
    hooks.skill_roots = [root]
    # ``knowledge_fingerprint`` is what the freshness check compares against, and the real
    # provider records it at session build; without it set, the check cannot tell the tree
    # changed and no invalidation is possible to exercise.
    hooks.knowledge_fingerprint = session_factory._skills_fingerprint(hooks)
    hooks.classification_wait_s = 0.01

    first = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    assert "mcp://slack" not in first, "the turn gave up before the vendor answered"
    assert len(classifier.requests) == 1

    await asyncio.sleep(0.3)  # the answer lands mid-turn

    # A real tree change under the running turn: the freeze is invalidated, for real.
    _write_skill(root, "beta", "Beta skill, authored mid-turn.")

    # The provider re-derives the query when it sees the invalidation (its own freshness
    # check runs BEFORE it decides whether to hand one over), so this render is given the
    # real query — unlike the frozen case, where it is handed ``""``.
    second = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")

    # A second selection proves the freeze really was invalidated: a frozen render returns
    # the cached block without touching the index. (``superseded_block`` is parked and then
    # dropped BY that render, so it cannot witness this after the fact.)
    assert len(index.calls) == 2, "the tree change must have forced a real re-render"
    assert "mcp://slack" in second, "the answer still reaches its own message"
    assert "alpha" in second
    assert len(classifier.requests) == 1, "and a message that has an answer is never asked twice"
