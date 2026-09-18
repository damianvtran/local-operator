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
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
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
    #: §4's per-resource attribution. Part of "field for field": the renderer reads it,
    #: and a stand-in without it makes the shipped ``notice`` raise, which the wiring
    #: swallows — so the missing field showed up as a notice that never appeared.
    late_urls: tuple[str, ...] = ()


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
        render_with: Any = None,
    ) -> None:
        self.recommendation = recommendation or _Recommendation()
        self.delay = delay
        self.timeout_s = timeout_s
        self.notice_line = notice_line
        self.raises = raises
        self.events = events if events is not None else []
        self.requests: list[Any] = []
        #: A real service whose ``notice`` renders the line, when the COPY is what a test
        #: is about. ``notice_line`` is a canned string, so a test that asserts how the
        #: sentence reads has to render it with the shipped renderer or it asserts its
        #: own fixture.
        self.render_with = render_with

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
        if self.render_with is not None:
            return self.render_with.notice(recommendation)
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
    """One line per admitted user message, through the bound sink, at info.

    ONCE PER MESSAGE is the sentence the contract states, and it is what this now
    checks at the seam: a message that delivers a LATE answer and its own prints ONE
    line for both, because the resources of the whole prompt are announced together
    (review round 2, MINOR 1 — the previous shape printed two).

    The second half is D7: the same resource set two messages running says nothing
    the line before it did not, so the line is suppressed while the BLOCK still goes
    into the prompt. The set has to be identical AND consecutive — a different set
    speaks again.
    """
    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "tunnel guide", "guide://tunnel"),),
            vendor="typesafe",
        ),
        notice_line="Suggestion added for this message: guide://tunnel",
    )
    hooks = _hooks(
        _FakeIndex(picked=[_skill("alpha", "Alpha skill.")]),
        classifier=classifier,
    )
    delivered: list[tuple[str, str]] = []
    hooks.notice_sink = lambda text, kind="warning": delivered.append((text, kind))

    first = await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    # Same task, tool continuation: frozen, so no second call and no second line.
    await session_factory._select_knowledge_block(hooks, OFF_QUERY, task_id="t1")
    # "info", and the pinned kind is the point: the design round's D1 proposed
    # `note` for its contrast (`info`'s `dim` ink measures 3.77:1 on the light
    # theme), and `note` is NOT a legal ``NoticeEvent.kind`` — a real Session
    # rejects it and the line never reaches the user (agent review round 2,
    # blocker). This assertion is what would catch that attempt again.
    assert delivered == [("Suggestion added for this message: guide://tunnel", "info")]
    assert "guide://tunnel" in first

    # The SAME set on the next message: the prompt still gains the block, the line
    # stays quiet (D7 — four identical rows is what the design round saw).
    second = await session_factory._select_knowledge_block(hooks, "second", task_id="t2")
    assert len(delivered) == 1
    assert "guide://tunnel" in second

    # A DIFFERENT set speaks again.
    classifier.recommendation = _Recommendation(
        resources=(_Candidate("skill", "beta", "Beta skill.", "skill://beta"),),
        vendor="typesafe",
    )
    await session_factory._select_knowledge_block(hooks, "third", task_id="t3")
    assert len(delivered) == 2
    assert len(classifier.requests) == 3


@pytest.mark.asyncio
async def test_one_message_gaining_both_sets_announces_them_without_lying(tmp_path: Path) -> None:
    """A late answer AND this message's own: one line, each resource attributed.

    THE TEST THE DOCSTRING ABOVE DESCRIBED WITHOUT CHECKING (QA round 4, Q1). The line
    used to be built from the answer that arrived LAST and labelled from it, so a
    message that gained both sets announced the union as "for your previous message" —
    telling the user a resource chosen for the question they had just asked came from
    the one before it. Neither whole-line label is true of a union, so each resource
    carries its own; the two uniform cases keep the short sentence the row budget wants.

    Rendered with the SHIPPED renderer (``ClassificationService.notice``) rather than a
    canned string: the finding is about what the sentence says, so a fixture that
    returns its own text would assert nothing.
    """
    from local_operator.classification.service import ClassificationService
    from local_operator.credentials import CredentialManager

    renderer = ClassificationService(
        manager=CredentialManager(tmp_path), settings={"classification": {"auto": True}}
    )
    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "Tunnel guide.", "guide://tunnel"),),
        ),
        delay=0.3,
        render_with=renderer,
    )
    hooks = _hooks(classifier=classifier)
    hooks.classification_wait_s = 0.05
    delivered: list[str] = []
    hooks.notice_sink = lambda text, kind="warning": delivered.append(text)

    await session_factory._select_knowledge_block(hooks, "first", task_id="t1")
    await asyncio.sleep(0.35)  # call 1 lands, late — nothing announced yet

    # This message's OWN call answers inside the wait, so the prompt gains both sets.
    classifier.delay = 0.0
    classifier.recommendation = _Recommendation(
        resources=(_Candidate("skill", "beta", "Beta skill.", "skill://beta"),),
    )
    block = await session_factory._select_knowledge_block(hooks, "second", task_id="t2")

    assert "guide://tunnel" in block and "skill://beta" in block
    assert len(delivered) == 1, delivered
    line = delivered[0]
    assert "guide://tunnel (your previous message)" in line, line
    assert "skill://beta (this message)" in line, line
    # …and the whole-line label of the uniform case is what would have lied here.
    assert "for your previous message:" not in line, line

    await renderer.aclose()


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


def test_a_bound_sink_is_the_real_sessions_own_notice_event() -> None:
    """The bound sink is ``Session.queue_notice``, the REAL attribute.

    WHY this is asserted against the class and not against a stand-in: the first
    version of this test hand-built an object with the method assigned to it, which
    proves the binder's shape and NOT that the facade has that attribute — rename
    or drop it and every test in this file kept passing while notices silently never
    appeared, because ``attach_classification_notices`` fails soft on purpose
    (``getattr(..., None)``, for a benchmark preflight that renders a prompt with no
    facade at all). This is the assertion that can fail: reach for the attribute on
    the real class, then check the binder put THAT callable on the hooks, bound to
    THIS session.

    ``queue_notice`` and not ``_stream_notice`` since design round 1's D1: emitting
    during prompt build put the line in the ANSWER's slot, so the sink is the
    session's post-turn queue. The fallback is pinned too, because a facade-shaped
    double without the queue must still get its notice.

    ``Session.__new__`` rather than a constructed session: neither method needs
    construction state, and paying for a real boot here would tempt the next reader
    into driving a turn to observe the notice. The end-to-end half — a session built
    by the composition root, with the layer on — is
    ``test_the_classification_seam_is_closed_on_dispose`` in the factory suite.
    """
    from local_operator.session.session import Session

    assert hasattr(Session, "queue_notice"), (
        "the notice path the wiring binds (Session.queue_notice) must still exist; "
        "without it a recommendation is delivered silently"
    )
    session = Session.__new__(Session)
    hooks = session_factory._KnowledgeHooks()

    session_factory.attach_classification_notices(session, hooks)

    assert hooks.notice_sink is not None
    assert hooks.notice_sink.__func__ is Session.queue_notice  # type: ignore[attr-defined]
    assert hooks.notice_sink.__self__ is session  # type: ignore[attr-defined]

    # A double with only the stream method still gets the line, just unparked.
    plain = cast(Any, SimpleNamespace(_stream_notice=lambda text, kind="info": None))
    fallback = session_factory._KnowledgeHooks()
    session_factory.attach_classification_notices(plain, fallback)
    assert fallback.notice_sink is plain._stream_notice


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

    # The first three events ARE the property: the classify leg started before the
    # select leg ended. Its END is deliberately not in the list — the turn waits 50 ms
    # and the fake takes 200 ms, so the answer lands after the turn has moved on, which
    # is what the bounded wait means. Asserting the end here made the test a race
    # between two 200 ms sleeps (it passed alone and failed under load).
    assert events[:3] == ["select-start", "classify-start", "select-end"], events
    assert elapsed < 0.4, elapsed
    for task in [task for task in hooks.classification_outstanding if not task.done()]:
        with contextlib.suppress(Exception):
            await task
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
    for task in (*first_hooks.classification_outstanding, *hooks.classification_outstanding):
        task.cancel()  # keep the loop quiet at teardown
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
# §5a rule 6: the wait is bounded, and a late answer rides the next message
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_hanging_seam_costs_the_turn_only_the_wait_budget() -> None:
    """The turn pays ``waitMs``, and the answer is delivered by the next message.

    The operator's budget is our OWN overhead per user message, and on a real roster
    the vendor's answer takes ~250 ms median — waiting for it would put the vendor's
    model time on the critical path. So the turn waits ``values.classification.waitMs``
    and no longer, and what it gave up on is not lost: the call keeps running, and the
    next admitted user message re-renders the knowledge block WITH the late answer.
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
        notice_line="Classification: 1 resource recommendation via typesafe",
    )
    hooks = _hooks(_FakeIndex(picked=[_skill("alpha", "Alpha skill.")]), classifier=classifier)
    hooks.classification_wait_s = 0.05
    delivered: list[str] = []
    hooks.notice_sink = lambda text, kind="warning": delivered.append(text)

    started = time.monotonic()
    first = await session_factory._select_knowledge_block(hooks, "first task", task_id="t1")
    waited = time.monotonic() - started

    # The BUDGET, not the vendor's 350 ms — and the prompt is byte-identical to
    # what it would have been with no layer at all.
    assert waited < 0.2, waited
    assert first == off
    # ...and NO notice: nothing was delivered to that prompt.
    assert delivered == []

    await asyncio.sleep(0.4)  # the call lands after its own turn ended

    second = await session_factory._select_knowledge_block(hooks, "second task", task_id="t2")

    assert "guide://tunnel" in second
    assert second.count("guide://tunnel") == 1
    # Announced once, at delivery — by the turn that actually carries it.
    assert delivered == ["Classification: 1 resource recommendation via typesafe"]


@pytest.mark.asyncio
async def test_a_late_answer_is_delivered_once_and_only_when_it_exists() -> None:
    """Consumed exactly once; a session whose answers never arrive appends nothing."""
    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "Tunnel guide.", "guide://tunnel"),),
            vendor="typesafe",
        ),
        delay=0.3,
        notice_line="Classification: 1 via typesafe",
    )
    hooks = _hooks(classifier=classifier)
    hooks.classification_wait_s = 0.05
    delivered: list[str] = []
    hooks.notice_sink = lambda text, kind="warning": delivered.append(text)

    await session_factory._select_knowledge_block(hooks, "first", task_id="t1")
    await asyncio.sleep(0.35)  # call 1 lands, late
    # NOTHING MORE WILL EVER BE RECOMMENDED: every later call is answered empty, so
    # anything that appears in a later block can only be call 1's answer arriving
    # twice — which is exactly what must not happen.
    classifier.recommendation = _Recommendation(skipped="empty-roster")

    second = await session_factory._select_knowledge_block(hooks, "second", task_id="t2")
    assert second.count("guide://tunnel") == 1
    assert delivered == ["Classification: 1 via typesafe"]

    await asyncio.sleep(0.35)  # call 2 lands, empty

    third = await session_factory._select_knowledge_block(hooks, "third", task_id="t3")
    assert "guide://tunnel" not in third, "a delivered answer must never be delivered twice"
    assert delivered == ["Classification: 1 via typesafe"], "and never announced twice"


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
    for task in hooks.classification_outstanding:
        task.cancel()
    await asyncio.sleep(0)


# ---------------------------------------------------------------------------
# The notice's memory is what the user SAW (review round 3, NIT 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_line_its_own_gate_suppressed_is_not_remembered() -> None:
    """A notice nobody painted must not silence the next message's identical set.

    ``classification_last_announced`` means "the set the user last SAW". Recording it
    before the paint made a suppressed line — the seam's own gate, or a sink that
    failed — read as delivered, so the next message with the same resources stayed
    quiet about something the user had never been told. Two messages, same one-resource
    set: the first says nothing (its seam renders no line), the second must speak.
    """
    classifier = _FakeClassifier(
        _Recommendation(
            resources=(_Candidate("guide", "tunnel", "tunnel guide", "guide://tunnel"),),
        ),
        notice_line=None,
    )
    hooks = _hooks(_FakeIndex(picked=[]), classifier=classifier)
    delivered: list[str] = []
    hooks.notice_sink = lambda text, kind="info": delivered.append(text)

    await session_factory._select_knowledge_block(hooks, "first", task_id="t1")
    assert delivered == [], "the seam's gate said nothing to announce"
    assert (
        hooks.classification_last_announced is None
    ), "a line nobody saw must not be recorded as announced"

    classifier.notice_line = "Suggestion added for this message: guide://tunnel"
    await session_factory._select_knowledge_block(hooks, "second", task_id="t2")

    assert delivered == ["Suggestion added for this message: guide://tunnel"], delivered


# ---------------------------------------------------------------------------
# Freshness: a skill installed while the session is RUNNING (operator
# requirement, 2026-09-18). Resolved per message, so it is a candidate on the
# very next one -- mid-conversation, after a steer, and for later subagents.
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
