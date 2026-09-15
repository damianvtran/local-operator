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


async def _seed_selection(config_dir: Path, selector: str, *, effort: str | None) -> None:
    """A conversation whose journal records ``selector`` as its own selection."""
    from local_operator.session.transcript import Transcript

    session_id = "resumed"
    directory = config_dir / "sessions" / session_id
    directory.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(directory)
    await transcript.append_custom(
        "selected_model",
        {"version": 2, "selector": selector, "effort": effort, "boot": selector},
    )


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
