"""The cold resolver's two answers: the level a birth runs at, and the spec a
resumed conversation's FIRST frame is painted from.

``resolve_birth_effort`` is the single owner of "the level a conversation born on
this model will RUN at", shared by the draft pane, the draft preview and the
resumed-conversation resolver. Its third case is DEFINED on the spec's seed, so a
caller holding a seed-cleared spec gets a silently wrong answer rather than an
error — which is exactly how the preview and the cold frame disagreed (review
round 2, R6). These tests pin the guard that turns that into a loud failure
(review round 3, R12 / Q-R3-1).

The SAVED-selection half is the other answer, and this file pins it against the
spec a live runtime builds rather than against the band's text: a cold state that
answers only ``provider/model_id`` leaves the display name to the curated
registry and the context window to ``ModelSpec``'s 128k default, which is what a
first frame used to paint — the operator's report of a resumed conversation
opening on a bare model id and no denominator until the full load finished.
"""

from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.harness.types import ModelSpec
from local_operator.session.cold_model import resolve_birth_effort, resolve_saved_model
from local_operator.session.model_selection import StoredModelSelection

#: A direct-provider pair the SHIPPED registry names and carries a window for, so
#: the resolution under test is answered from shipped data and never from a
#: listing fetch (which would make this file's result depend on the network).
DIRECT_PROVIDER, DIRECT_MODEL = "deepseek", "deepseek-flash"
DIRECT_SELECTOR = f"{DIRECT_PROVIDER}/{DIRECT_MODEL}"


async def _seed_selection(
    config_dir: Path, selector: str, *, effort: str | None, fast: bool | None = None
) -> None:
    """A conversation whose journal records ``selector`` as its own selection.

    ``fast`` is omitted from the row when it is ``None``, which is the exact shape
    every row written before the dial was journalled has (``paste`` of a snapshot
    with no key, never an explicit ``null``).
    """
    from local_operator.session.transcript import Transcript

    session_id = "resumed"
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    row: dict[str, object] = {
        "version": 2,
        "selector": selector,
        "effort": effort,
        "boot": selector,
    }
    if fast is not None:
        row["fast"] = fast
    await transcript.append_custom("selected_model", row)


async def _cold_state(config_dir: Path):
    from local_operator.session.cold_model import synthesise_cold_state

    return await synthesise_cold_state(
        config_dir=config_dir, session_id="resumed", cwd=str(config_dir)
    )


def _seeded_spec() -> ModelSpec:
    """The shape ``build_model_spec`` returns: the default IS the seed it computed."""
    return ModelSpec(
        provider="deepseek",
        model_id="deepseek-flash",
        reasoning_efforts=("none", "low", "high", "max"),
        reasoning_effort="high",
        reasoning_default_effort="high",
    )


def test_a_seeded_spec_answers_its_seed_when_nothing_else_has_an_opinion(tmp_path) -> None:
    """No choice and no configured level: the spec's own rung is the answer.

    The config dir is a path that does not exist on purpose — the resolver reads a
    configured ``model_effort`` without materialising the directory it is pointed
    at, and a test must not read (or create) the developer's real config.
    """
    spec = _seeded_spec()
    assert resolve_birth_effort(spec, None, tmp_path / "absent") == "high"


def test_a_seed_cleared_spec_is_refused_instead_of_answered_no_level(tmp_path) -> None:
    """Clearing the seed is detectable, so it fails loudly rather than silently.

    ``reasoning_effort is None`` beside a non-``None`` default is the shape only a
    seed-cleared spec has, and the third case would otherwise answer ``None`` —
    "no level" — for a birth that runs the seed. This is the assertion that makes
    the guard's promise real: before the guard, this call returned ``None``.
    """
    cleared = _seeded_spec().model_copy(update={"reasoning_effort": None})
    with pytest.raises(ValueError, match="still carries its seed"):
        resolve_birth_effort(cleared, None, tmp_path / "absent")


@pytest.mark.asyncio
async def test_the_cold_state_resolves_the_saved_selection_like_the_runtime(tmp_path) -> None:
    """The state a resumed conversation's FIRST frame paints is the runtime's spec.

    WHY the assertion is on the SPEC and not on the band's text: the band's model
    segment is a pure function of the spec it is handed, so a state that answers
    only ``provider/model_id`` hands it ``display_name == ""`` (naming then falls
    back to the curated registry, and to the BARE ID for any model the shipped
    rows do not cover) and ``context_window == 128_000`` verbatim from
    ``ModelSpec`` — a placeholder the band then divides a REAL, measured token
    reading by.

    Fails on the pre-fix code on both halves. The comparison is against
    ``build_model_spec`` rather than a literal name so the two cannot drift
    apart, with the shipped registry's own non-empty answer asserted separately
    — an empty name on both sides would otherwise pass.
    """
    from local_operator.model.configure import build_model_spec

    await _seed_selection(tmp_path, DIRECT_SELECTOR, effort="high")
    runtime = build_model_spec(DIRECT_PROVIDER, DIRECT_MODEL)
    assert runtime.display_name, "the shipped registry no longer names this pair"
    assert runtime.display_name != runtime.model_id

    state = await _cold_state(tmp_path)

    assert state.selected_model is not None and state.effective_model is not None
    for spec in (state.selected_model, state.effective_model):
        assert (spec.provider, spec.model_id) == (DIRECT_PROVIDER, DIRECT_MODEL)
        assert spec.display_name == runtime.display_name
        assert spec.context_window == runtime.context_window
        assert spec.context_window != ModelSpec.model_fields["context_window"].default
        assert spec.reasoning_efforts == runtime.reasoning_efforts


@pytest.mark.asyncio
async def test_the_saved_selections_own_level_wins_over_the_configured_one(tmp_path) -> None:
    """The journal's level, not this machine's current default for the model.

    The resolution carries the level a hop may ride from the spec the session
    BOOTS on (``spec_for_target``), so the configured ``model_effort`` must not
    displace the level the conversation itself recorded: one resumed on a machine
    configured for ``max`` while the conversation chose ``low`` cold-opens on
    ``low``, or the band names a level the first turn will not send.
    """
    ConfigManager(tmp_path).update_config(
        {"hosting": "deepseek", "model_name": "deepseek-flash", "model_effort": "max"}
    )
    await _seed_selection(tmp_path, DIRECT_SELECTOR, effort="low")

    state = await _cold_state(tmp_path)

    assert state.selected_model is not None
    assert state.selected_model.reasoning_effort == "low"


def test_an_unresolvable_saved_selection_still_answers_its_pair(tmp_path) -> None:
    """Best-effort: a pair this machine cannot resolve is still a selector.

    The resolution is metadata, and metadata is never worth a failed cold open —
    a local endpoint tag no catalogue covers must return the pair the journal
    recorded (an empty spec would paint "no model" over a conversation that has
    one).
    """
    spec = resolve_saved_model(
        StoredModelSelection(provider="ollama", model_id="a-tag-no-row-covers"), tmp_path
    )

    assert (spec.provider, spec.model_id) == ("ollama", "a-tag-no-row-covers")


@pytest.mark.asyncio
async def test_the_cold_state_resolves_a_birth_selection_like_the_runtime(tmp_path) -> None:
    """Q3: a caller-named pair is a SELECTOR too, and gets the same resolution.

    The third source of a conversation's model. The CLI hands over
    ``ModelSpec(provider, model_id)`` and nothing else, so a legacy conversation
    with no journalled selection (and no checkpoint) cold-opened on the pair plus
    ``ModelSpec``'s own defaults: a 128k placeholder standing in as the band's
    denominator under a real 287,491-token reading, and no name, ladder or level
    (QA round 1, Q3 — ``224.6%/128k``, unchanged by the saved-selection fix
    because this is the branch beside it).

    Asserted against ``build_model_spec`` rather than a literal so the two cannot
    drift, exactly as the saved-selection test above does.
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.session.cold_model import synthesise_cold_state

    ConfigManager(tmp_path).update_config({"hosting": DIRECT_PROVIDER, "model_name": DIRECT_MODEL})
    runtime = build_model_spec(DIRECT_PROVIDER, DIRECT_MODEL)
    assert runtime.display_name, "the shipped registry no longer names this pair"

    state = await synthesise_cold_state(
        config_dir=tmp_path,
        # No session directory at all: this is the branch a pair the caller named
        # takes when the journal has nothing to say (a legacy conversation, or a
        # fresh one), which is why it cannot be reached through ``_seed_selection``.
        session_id="",
        cwd=str(tmp_path),
        birth_model=ModelSpec(provider=DIRECT_PROVIDER, model_id=DIRECT_MODEL),
    )

    spec = state.selected_model
    assert spec is not None and spec.provider == DIRECT_PROVIDER and spec.model_id == DIRECT_MODEL
    assert spec.display_name == runtime.display_name
    assert spec.context_window == runtime.context_window
    assert (
        spec.context_window != ModelSpec.model_fields["context_window"].default
    ), "the placeholder window is exactly what this branch used to paint"
    assert spec.reasoning_efforts == runtime.reasoning_efforts
    assert spec.reasoning_effort == runtime.reasoning_effort


@pytest.mark.asyncio
async def test_a_birth_selections_own_choices_are_never_replaced(tmp_path) -> None:
    """FILL ONLY: the desktop preview hands over a spec that already has opinions.

    ``_preview_birth_model`` builds ``build_model_spec``'s result and sets the level
    its user picked, so replacing the whole spec with this machine's configured
    answer would put the draft pane and the first turn on different rungs — the R6
    defect, one module over. A field the caller expressed must survive.
    """
    from local_operator.model.configure import build_model_spec
    from local_operator.session.cold_model import synthesise_cold_state

    ConfigManager(tmp_path).update_config(
        {
            "hosting": DIRECT_PROVIDER,
            "model_name": DIRECT_MODEL,
            # The machine configures a DIFFERENT level, so a wholesale re-resolution
            # would answer this and the caller's pick would disappear.
            "model_effort": "max",
        }
    )
    chosen = build_model_spec(DIRECT_PROVIDER, DIRECT_MODEL)
    chosen = chosen.model_copy(update={"reasoning_effort": "low"})

    state = await synthesise_cold_state(
        config_dir=tmp_path, session_id="", cwd=str(tmp_path), birth_model=chosen
    )

    spec = state.selected_model
    assert spec is not None
    assert spec.reasoning_effort == "low", "the caller's chosen level is its own opinion"


#: A shipped pair that SELLS the fast tier (``model/speed.py``'s Anthropic list),
#: resolved entirely from the registry so this file never touches the network.
FAST_PROVIDER, FAST_MODEL = "anthropic", "claude-opus-5"


@pytest.mark.asyncio
async def test_the_saved_dial_is_the_one_the_first_frame_paints(tmp_path) -> None:
    """A resumed conversation's FIRST frame shows the speed the row recorded.

    The cold state is what a desktop pane paints before a runtime exists, so it
    is the frame the reopen in ui#1266 lands on: the journal restore in
    ``Session`` cannot help here because there is no ``Session`` yet. Reading
    the row's pair and level but not its dial painted ``fast_mode: false`` over
    a conversation that recorded ``true``, and the toggle flipped OFF at exactly
    the moment the user reopened it.
    """
    await _seed_selection(tmp_path, f"{FAST_PROVIDER}/{FAST_MODEL}", effort="high", fast=True)

    state = await _cold_state(tmp_path)

    assert state.selected_model is not None
    assert state.selected_model.fast_mode is True, "the row's dial must reach the first frame"
    assert state.selected_model.supports_fast_mode is True, "and the route does sell it"


@pytest.mark.asyncio
async def test_the_saved_dial_is_clamped_to_a_route_without_the_tier(tmp_path) -> None:
    """A row can outlive the model table; the frame must not promise a 400.

    ``spec_for_target`` builds the TARGET's own spec, so the route's answer is
    authoritative here exactly as it is in ``Session._restore_selected_model``:
    a stale ``fast: true`` on a pair with no fast tier is dropped rather than
    painted on a chip whose first request would be refused.
    """
    await _seed_selection(tmp_path, DIRECT_SELECTOR, effort="high", fast=True)

    state = await _cold_state(tmp_path)

    assert state.selected_model is not None
    assert state.selected_model.supports_fast_mode is False
    assert state.selected_model.fast_mode is False


@pytest.mark.asyncio
async def test_a_row_without_a_dial_paints_a_standard_speed_frame(tmp_path) -> None:
    """Absence is not a decision, and the frame's own default is the honest answer.

    Every row written before the dial was journalled lands here; the frame must
    not invent an ON it cannot evidence, and the row's absence must not be
    written back as an OFF anyone acts on.
    """
    await _seed_selection(tmp_path, f"{FAST_PROVIDER}/{FAST_MODEL}", effort="high")

    state = await _cold_state(tmp_path)

    assert state.selected_model is not None
    assert state.selected_model.fast_mode is False


#: The two modules the cold frame used to pull, plus ``httpx`` (which
#: ``providers.failover`` drags in at module scope, and whose ``_main`` CLI module
#: ``httpx/__init__.py`` imports unconditionally). Named as strings because this
#: process must never import them to check.
_ENGAGE_STACK = (
    "local_operator.providers.failover",
    "local_operator.secrets.access",
    "httpx",
)

#: Run in a FRESH interpreter: "does a cold call import X" is a question about a
#: process that has not imported it yet, and this suite's own module imports (and
#: every test that ran before it, in file or random order) have already paid some
#: of that graph. A ``sys.modules`` delta in-process would be measuring the suite.
_COLD_IMPORT_PROBE = """
import sys
from pathlib import Path
import local_operator.session.cold_model as cold_model
from local_operator.session.model_selection import StoredModelSelection

# The module under test is imported FIRST, so the delta below is the CALL's own
# cost and not the resolver's import graph.
before = set(sys.modules)
spec = cold_model.resolve_saved_model(
    StoredModelSelection(provider={provider!r}, model_id={model!r}, effort={effort!r}),
    Path({config_dir!r}),
)
pulled = [name for name in {watched!r} if name in set(sys.modules) - before]
print("PULLED:" + ",".join(pulled))
print("WINDOW:%s" % spec.context_window)
print("NAME:%s" % spec.display_name)
"""


def test_the_cold_display_spec_pulls_nothing_from_the_engage_stack(tmp_path) -> None:
    """The first frame must not import the fallback driver or the credential reader.

    THE OPERATOR'S REPORT: clicking a conversation in the desktop app took 5-10 s
    to paint. Measured on this host (fresh process per run, isolated config),
    ``resolve_saved_model`` for a selection the catalogue ships was 1654 ms and
    pulled all three of :data:`_ENGAGE_STACK`; with the metadata answered from the
    seeded catalogue row instead it is 86 ms and pulls none of them. A
    ``sys.modules`` delta is the right instrument because the cost IS the import:
    ``providers.failover`` -> ``httpx`` (0.89 s) and
    ``configure._catalogue_credential`` -> ``registry.provider_secret_value`` ->
    ``local_operator.secrets.access`` (0.25 s).

    Run in a FRESH interpreter (see :data:`_COLD_IMPORT_PROBE`) so the assertion
    is about a cold call rather than about whatever this suite already imported.
    The pair is the shipped direct-provider one above, which is the branch the
    fast path owns; the branch that genuinely needs the driver is covered
    separately below.
    """
    import os
    import subprocess
    import sys

    probe = _COLD_IMPORT_PROBE.format(
        provider=DIRECT_PROVIDER,
        model=DIRECT_MODEL,
        effort="high",
        config_dir=str(tmp_path),
        watched=list(_ENGAGE_STACK),
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[3])},
    )
    assert completed.returncode == 0, completed.stderr
    reported = dict(
        line.split(":", 1) for line in completed.stdout.strip().splitlines() if ":" in line
    )
    assert reported["PULLED"] == "", (
        f"the cold display path imported {reported['PULLED']!r}; the first frame is "
        f"paying the engage stack again (stderr: {completed.stderr[-400:]})"
    )
    # ...and it still answered from the catalogue row, which is what makes the
    # saving worth having: a path that skipped the imports by giving up on the
    # display would be the 128k-placeholder bug this module exists to prevent.
    assert int(reported["WINDOW"]) > 0
    assert reported["NAME"] and reported["NAME"] != DIRECT_MODEL


def test_the_cold_display_spec_keeps_the_display_and_the_carry_rules(tmp_path) -> None:
    """The fast path must answer EXACTLY what ``spec_for_target`` answers.

    The saved-selection branch is a local spelling of the failover driver's
    carry rules (it cannot import that module on the click path — see
    ``_paint_spec``), so the two are compared cell by cell here and a drift fails
    the build rather than repainting a band.

    ``resolve_model_info`` is answered from the SAME metadata both sides would
    see (``resolve_model_info_paint``), which is what makes this a test of the
    CARRY logic rather than of who fetched what: with that pinned, a difference
    can only come from the local spelling. Network-independent by construction.
    """
    from local_operator.model.configure import resolve_model_info_paint
    from local_operator.providers.failover import FallbackTarget, spec_for_target
    from local_operator.session.cold_model import _display_target_spec

    info, _memo_hit = resolve_model_info_paint(DIRECT_PROVIDER, DIRECT_MODEL)
    assert info.context_window and info.context_window > 0, (
        "this pair is no longer answered from the shipped rows, so the fast path "
        "is not the branch under test any more"
    )
    import local_operator.model.configure as configure

    original = configure.resolve_model_info
    configure.resolve_model_info = lambda provider, model_id, **kwargs: info
    try:
        for effort in (None, "auto", "none", "low", "high", "max"):
            for base_effort, base_fast in (("max", True), ("min", False), (None, False)):
                base = ModelSpec(
                    provider="deepseek",
                    model_id="deepseek-flash",
                    reasoning_effort=base_effort,
                    reasoning_default_effort="high",
                    fast_mode=base_fast,
                )
                expected = spec_for_target(base, FallbackTarget(DIRECT_SELECTOR, effort))
                actual = _display_target_spec(base, DIRECT_SELECTOR, effort)
                assert actual.model_dump() == expected.model_dump(), (
                    f"effort={effort!r} base_effort={base_effort!r} "
                    f"base_fast={base_fast}: the first frame would disagree with the "
                    f"runtime's own spec"
                )
    finally:
        configure.resolve_model_info = original


def test_an_unshipped_pair_keeps_the_full_resolution(tmp_path, monkeypatch) -> None:
    """The one branch that MUST stay on the heavy path, proven to stay there.

    A model the catalogue has no window for can only be described by the
    provider's listing, and fetching it needs the fallback driver and the
    credential reader. Painting the placeholder instead is measured-wrong: the
    agent model this document's own runs use (``openai/gpt-6.1-sol``) is absent
    from the shipped rows and would paint ``128_000``/``gpt-6.1-sol`` against the
    ``872_000``/``GPT-6.1-Sol`` the full resolution reports — a 6.8x wrong
    denominator under the band's percentage.

    ``spec_for_target`` is what proves the branch is taken, so it is wrapped
    rather than mocked-away; the resolver is also stubbed so this stays
    network-independent.
    """
    from local_operator.providers import failover
    from local_operator.session.cold_model import _display_target_spec

    seen: list[tuple[str, str | None]] = []
    original = failover.spec_for_target

    def recording(base, target):
        seen.append((target.selector, target.effort))
        return original(base, target)

    monkeypatch.setattr(failover, "spec_for_target", recording)
    monkeypatch.setattr("local_operator.model.configure.resolve_model_info", lambda *a, **k: None)

    _display_target_spec(ModelSpec(provider="", model_id=""), "openai/no-such-model", "high")

    assert seen == [("openai/no-such-model", "high")], (
        "a pair the seeded catalogue cannot describe was answered from the fast "
        "path, which would paint the 128k placeholder as a real window"
    )
