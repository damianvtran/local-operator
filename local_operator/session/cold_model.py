"""Canonical state for a conversation that is not running.

Two callers need the same answer to "what should this session's strip say?" and
must not answer it twice:

* the COLD viewer, which synthesises state from config and the journal because
  there is no owner to ask (``AttachedSession._synthesise_cold_state``);
* the desktop DRAFT PREVIEW, which describes the session a new-conversation pane
  WOULD create, without creating one (``routes/desktop_sessions``).

Both must resolve the model the same way. The UI swaps the preview payload for
the first cold frame at ``finishDraft``, so a second policy would show the model
chip flicker — or worse, name one model while the session runs another. One
implementation, two callers.

Deliberately free of ``AttachedSession``: this module owns the RESOLUTION, not
the lifecycle around it. Nothing here starts a runtime, writes a directory, or
reads history beyond the model-selection rows.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from local_operator.harness.types import ModelSpec
from local_operator.session.frontend_state import (
    FrontendModelSpec,
    FrontendSessionState,
    WakeState,
)
from local_operator.session.model_selection import (
    StoredModelSelection,
    read_model_selection,
)
from local_operator.session.usage_seed import denominator_window

logger = logging.getLogger(__name__)


def resolve_birth_effort(spec: ModelSpec, chosen: str | None, config_dir: Path) -> str | None:
    """The reasoning level a conversation born on ``spec`` will RUN at.

    THREE cases, in order — every caller that reports a conversation which is not
    running yet needs all three:

    * a level was CHOSEN: clamped against this spec's ladder (``resolve_effort_in``)
      rather than refused. A stored choice is a durable record the catalogue moves
      under, so a level the route can no longer express must land on the nearest
      rung it can instead of reaching the child as one it would 400 on; the route
      refuses such a level at the moment it is CHOSEN (422).
    * no level was chosen and the machine CONFIGURES one (``model_effort``): that
      level, clamped — exactly what a launch that named no model resolves for
      itself, so picking a model alone never silently replaces the configured
      level with the model's seeded rung.
    * no level was chosen and the config expresses no opinion: the spec's own
      SEEDED rung, which is what the child would construct for itself and
      therefore what a reader has to report.

    IT HAS TO BE CARRIED, not left absent, in all three cases: the value travels
    the owner's model RPC as well as the spawn environment, and that RPC rebuilds
    the spec from the model's metadata — reseating the conversation on the seed.
    Carrying the seed is what makes it a no-op instead of a reseat.

    CALLERS MUST PASS A SPEC THAT STILL CARRIES ITS SEED. The seed-cleared spec a
    marker stores (``_draft_model_spec``'s null-level answer) makes the third case
    unreachable, which is how the preview and the plane came to disagree whenever
    no ``model_effort`` was configured (review round 2, R6) — pass
    ``build_model_spec``'s result, not the marker's value. That is now ENFORCED
    rather than merely documented: the mistake is detectable, because clearing the
    seed is the only thing that produces ``reasoning_effort is None`` beside a
    non-``None`` default, so the guard below refuses such a spec loudly instead of
    answering "no level" for a birth that will run a rung (review round 3, R12 /
    Q-R3-1).

    Every return is a rung this spec's ladder accepts, or ``None`` (nothing to
    send) — the membership contract ``resolve_effort_in`` documents.
    """
    if spec.reasoning_effort is None and spec.reasoning_default_effort is not None:
        # The shape only a seed-CLEARED spec has: ``build_model_spec`` sets the
        # default to the seed it computed, so the two are equal in every spec it
        # returns and a ``None`` effort beside a stated default cannot come from
        # it. Fail loudly rather than fall through to the third case, which would
        # report "no level" while the launch runs the seed.
        raise ValueError(
            "resolve_birth_effort needs a spec that still carries its seed, but "
            f"this one has reasoning_effort=None beside reasoning_default_effort="
            f"{spec.reasoning_default_effort!r} — the shape a seed-cleared spec "
            "has. Pass build_model_spec's result, not the marker's null-level "
            "value (see this function's docstring)."
        )
    from local_operator.model.effort import resolve_effort_in

    if chosen:
        return resolve_effort_in(spec.reasoning_efforts, spec.reasoning_default_effort, chosen)
    configured = _configured_effort_without_writing(config_dir)
    if configured:
        return resolve_effort_in(spec.reasoning_efforts, spec.reasoning_default_effort, configured)
    return spec.reasoning_effort


def _configured_effort_without_writing(config_dir: Path) -> str | None:
    """``configured_effort`` without materialising the config directory.

    WHY the guard: ``ConfigManager``'s loader mkdirs the directory it is pointed
    at, and one of this module's callers is the DRAFT PREVIEW, which is documented
    as side-effect free. Through HTTP that directory always exists (``pool.root``
    comes from a live config manager), so the write was unreachable by accident
    rather than by construction (QA round 2, Q-R2-3). A root that does not exist
    simply has no configured level to read.
    """
    if not config_dir.exists():
        return None
    from local_operator.config import ConfigManager
    from local_operator.model.effort import configured_effort

    return configured_effort(ConfigManager(config_dir=config_dir))


def resolve_configured_model(provider: str, model_id: str, config_dir: Path) -> FrontendModelSpec:
    """The configured pair as the spec a first turn will actually RUN on.

    WHY metadata rather than just the pair: the desktop strip reads the LADDER
    (``reasoning_efforts``) and the LEVEL off ``selected_model`` and gates its
    effort chip on them, so a config-only projection — an empty ladder, no level —
    hides the reading and leaves the picker unreachable on every conversation that
    has not chosen a model, which is the operator's own report of this feature. The
    ladder is a MODEL-derived field, so it can be answered without an account: this
    is the same resolution the runtime's own spec construction performs.

    WHAT IS DELIBERATELY NOT ANSWERED HERE: the EFFECTIVE WINDOW. A real cold open
    may apply account metadata (:func:`resolve_context_metadata`) and a window the
    account's plan scopes, while a draft must not read account metadata at all (a
    synthetic stickiness key would move a real account's stickiness). So the window
    here is the MODEL's own, exactly as ``build_model_spec`` reports it, and it can
    still differ from the first cold frame's on an account-scoped plan.
    """
    bare = FrontendModelSpec(provider=provider, model_id=model_id)
    if not provider or not model_id:
        return bare
    try:
        from local_operator.model.configure import build_model_spec

        spec = build_model_spec(provider, model_id)
    except Exception:  # noqa: BLE001 — metadata is best-effort; the name is still an answer
        logger.debug("configured model metadata could not be resolved", exc_info=True)
        return bare
    effort = resolve_birth_effort(spec, None, config_dir)
    if effort != spec.reasoning_effort:
        spec = spec.model_copy(update={"reasoning_effort": effort})
    return FrontendModelSpec(**spec.model_dump())


def configured_model_pair(config_dir: Path) -> tuple[str, str]:
    """The ``(provider, model_id)`` this machine is configured for, model defaulted.

    The pair alone, so the two resolvers below can share one reading of it: the
    CONFIGURED model is both an answer in its own right (a conversation with no
    journalled selection) and the BASE a saved selection is resolved against.
    Falling back to the provider's own default model when config names a provider
    but no model belongs here rather than to either caller.
    """
    from local_operator.config import ConfigManager

    config = ConfigManager(config_dir=config_dir)
    provider = str(config.get_config_value("hosting", "") or "")
    model_id = str(config.get_config_value("model_name", "") or "")
    if provider and not model_id:
        from local_operator.model.defaults import default_model_for

        model_id = default_model_for(provider) or ""
    return provider, model_id


def configured_base_model(config_dir: Path) -> FrontendModelSpec:
    """The spec a runtime constructed HERE would boot on, or an empty one.

    The carry source for :func:`resolve_saved_model`: ``spec_for_target`` takes the
    spec the session currently holds and carries only the sampling choices that may
    legitimately cross a hop, so the base has to be the CONFIGURED pair — which is
    what ``Session.__init__`` builds before ``_restore_selected_model`` maps the
    journal onto it.

    An unreadable or empty config is the empty spec, never an error: ``spec_for_target``
    carries nothing from a spec with no effort, which is the honest answer when this
    machine has no opinion to carry.
    """
    try:
        provider, model_id = configured_model_pair(config_dir)
    except Exception:  # noqa: BLE001 — an unreadable config carries nothing
        logger.debug("configured base could not be read", exc_info=True)
        return FrontendModelSpec(provider="", model_id="")
    return resolve_configured_model(provider, model_id, config_dir)


def resolve_birth_model(birth: ModelSpec, config_dir: Path) -> FrontendModelSpec:
    """The caller's own selection, resolved the way the runtime resolves it.

    The THIRD source of a conversation's model, beside config and the journal: a
    pair the CALLER named for this session (the CLI's own ``resolve_hosting_model``,
    a desktop draft's picked row). It arrives as a SELECTOR — the CLI builds
    ``ModelSpec(provider, model_id)`` and nothing else — so a cold frame painted
    from it carried the pair plus ``ModelSpec``'s own defaults: the 128k
    placeholder standing in as the band's denominator under a real restored
    reading, and no name, ladder or level (QA round 1, Q3). Resolving the pair is
    the same treatment the configured and saved branches already get, and for the
    same reason: the first frame must be the frame the runtime will paint.

    FILL ONLY, field by field. A caller may hand over a spec that is already
    resolved and already states its own opinions — the desktop preview passes
    ``build_model_spec``'s result carrying the level its user picked
    (``_preview_birth_model``) — and answering THOSE with this machine's
    ``model_effort`` would put the preview and the first turn on different rungs,
    which is the R6 defect. So a field the caller expressed is kept, and only the
    ones it left at a default are answered: the name, the effort LADDER, the
    LEVEL, and a window that is still ``ModelSpec``'s placeholder.
    """
    carried = FrontendModelSpec(**birth.model_dump())
    if not carried.provider or not carried.model_id:
        return carried
    try:
        resolved = resolve_configured_model(carried.provider, carried.model_id, config_dir)
    except Exception:  # noqa: BLE001 — metadata is best-effort; the pair is still an answer
        logger.debug("birth model metadata could not be resolved", exc_info=True)
        return carried
    update: dict[str, Any] = {}
    if not carried.display_name:
        update["display_name"] = resolved.display_name
    if not carried.reasoning_efforts:
        update["reasoning_efforts"] = resolved.reasoning_efforts
        update["reasoning_default_effort"] = resolved.reasoning_default_effort
    if carried.reasoning_effort is None:
        update["reasoning_effort"] = resolved.reasoning_effort
    # An unvouched window is the PLACEHOLDER, never a budget: the same value rule
    # the receipt seed applies (``usage_seed.denominator_window``), so a caller's
    # genuine 128k row is not overwritten and a defaulted one is still replaced.
    if denominator_window(carried) is None and denominator_window(resolved) is not None:
        update.update(
            {
                "context_window": resolved.context_window,
                "default_context_window": resolved.default_context_window,
                "max_context_window": resolved.max_context_window,
                "context_metadata_resolved": resolved.context_metadata_resolved,
            }
        )
    return carried.model_copy(update=update) if update else carried


def resolve_saved_model(saved: StoredModelSelection, config_dir: Path) -> FrontendModelSpec:
    """The conversation's OWN saved selection as the spec a resumed turn runs on.

    WHY metadata rather than just the pair, one branch over from
    :func:`resolve_configured_model`: this spec is what the BAND paints while the
    conversation is still cold, and it is what the runtime compares against the
    moment it attaches. A bare ``FrontendModelSpec(provider, model_id, effort)``
    answers the pair and nothing else, so the first frame carried ``ModelSpec``'s
    own defaults — a 128_000 window in place of the model's real one, and no
    ``display_name`` at all.

    Both halves of that were user-visible, and the window is the worse of the two.
    The band's percentage is a MEASURED reading divided by the spec's window, so a
    conversation holding 287_491 tokens against a 1_000_000 budget opened on a
    confident ``224.6%/128k`` — and, where the receipt seed filled the numerator
    while ``usage_seed.reading_window`` refused to vouch a denominator for an
    unresolved spec, on ``287.5k/—`` — then healed seconds later when the runtime's
    own spec arrived. The missing name degrades to ``naming.py``'s curated
    registry, which can name only what the shipped rows carry, so a model they do
    not cover (a release newer than the rows, any resold route) painted its bare id
    until the full load finished.

    ``spec_for_target`` rather than ``build_model_spec`` directly: it is the
    derivation the failover driver and ``Session._spec_for_route`` already build a
    resumed selection's spec with, and its own docstring is explicit that deriving
    the display spec any other way is how the band and the wire end up disagreeing
    about effort and the context window. Its base is
    :func:`configured_base_model` — the same base the runtime holds when it restores
    this selection — so the cold frame and the resumed frame agree by construction
    rather than by coincidence.

    Best-effort throughout: a resolution that raises leaves the bare pair, which is
    the selector the band has always known how to render.
    """
    bare = FrontendModelSpec(
        provider=saved.provider, model_id=saved.model_id, reasoning_effort=saved.effort
    )
    if not saved.provider or not saved.model_id:
        return bare
    try:
        from local_operator.providers.failover import FallbackTarget, spec_for_target

        # The base exists for the two things a hop may CARRY, and neither is
        # reachable when the journal named a level: ``spec_for_target`` reads the
        # base's effort only under ``target.effort is None``, and reads its
        # ``fast_mode`` only through an ``and`` — where a base built from
        # ``build_model_spec`` is always False, because that field is NOT seeded
        # ("Only the AVAILABILITY is seeded", ``configure.py``; measured: 0 of the
        # 120 shipped rows carry it). So a saved selection that names its level
        # skips a SECOND full metadata resolution on the path whose whole purpose
        # is the first frame: ``configured_base_model`` costs 7-200ms warm here and
        # far more cold, and the resolution below already resolves the target
        # (review round 1, minor 4).
        base = (
            configured_base_model(config_dir)
            if saved.effort is None
            else FrontendModelSpec(provider="", model_id="")
        )
        resolved = spec_for_target(base, FallbackTarget(saved.selector, saved.effort))
    except Exception:  # noqa: BLE001 — metadata is best-effort; the pair is still an answer
        logger.debug("saved model metadata could not be resolved", exc_info=True)
        return bare
    return FrontendModelSpec(**resolved.model_dump())


def resolve_conversation_model(
    config_dir: Path,
    session_dir: Path | None = None,
    *,
    birth_model: ModelSpec | None = None,
    model_selection_override: bool = False,
    selection_sink: Callable[[StoredModelSelection | None], None] | None = None,
) -> FrontendModelSpec:
    """The model this conversation is on, from config and its own journal.

    Precedence, and why: the conversation's SAVED selection wins over config,
    because a conversation that switched models must resume on the model it was
    using rather than on whatever the machine is configured for now. An explicit
    ``/model`` override (``model_selection_override``) is the user saying so for
    THIS session, so it outranks the journal — but only when there is a birth
    model to apply. With neither, the configured pair is the answer, falling back
    to the provider's own default model when config names a provider but no model.

    NEVER returns ``None``. ``AttachedSession.model`` raises without a spec, and a
    cold viewer is exactly the state where config may be empty (a first run,
    before ``/login``) — so the band would crash on the very screen that exists to
    help the user fix it. An empty spec renders as "no model" and is replaced by
    the runtime's own on first engage.

    ``selection_sink`` receives the selection this function read anyway. It is
    not decoration, and it is a deliberate extension of the documented signature
    (the design lists five parameters; the documented call is unchanged and still
    works): ``AttachedSession`` needs the same value to attribute a usage receipt
    that predates the serving-identity stamp (``usage_seed.reading_identity``),
    and re-reading the journal for it would scan a transcript that reaches 103 MB
    a second time on every cold open. The alternative — having the caller read the
    selection itself — either duplicates that read or duplicates this resolution,
    and a second resolution is the defect this module exists to remove. Pass
    ``None`` (the default) if the value is not wanted.
    """
    model: FrontendModelSpec | None = None
    saved: StoredModelSelection | None = None
    try:
        provider, model_id = configured_model_pair(config_dir)
        if session_dir is not None:
            saved = read_model_selection(session_dir)
        if selection_sink is not None:
            selection_sink(saved)
        if birth_model is not None and (saved is None or model_selection_override):
            model = resolve_birth_model(birth_model, config_dir)
        elif saved is not None:
            # The conversation's own pair, resolved through its OWN metadata — the
            # same treatment the configured pair below already gets, and for the
            # same reason. Left bare, this branch handed the first cold frame the
            # pair plus ``ModelSpec``'s defaults: no display name (so the band fell
            # back to the curated registry, and to the BARE ID for anything it does
            # not ship) and a 128k window standing in for the model's real one (so
            # the band divided a measured reading by it). See
            # :func:`resolve_saved_model`.
            model = resolve_saved_model(saved, config_dir)
        else:
            # The configured pair, resolved through its OWN metadata. A bare
            # ``FrontendModelSpec(provider, model_id)`` answers no ladder and no
            # level, so a reader that gates an effort reading on those fields —
            # the desktop strip does — showed nothing on every conversation that
            # had not chosen a model (the operator's report, and the preview's
            # half of it). See ``resolve_configured_model`` for what is
            # model-derived here (the ladder, the level) and what is still the
            # account's to answer (the effective window).
            model = resolve_configured_model(provider, model_id, config_dir)
    except Exception:  # noqa: BLE001 — an unreadable config is not fatal
        logger.debug("cold state could not read the configured model", exc_info=True)
        if selection_sink is not None and saved is None:
            # An unreadable config must still tell a caller that no selection was
            # read, or a sink that was never called would keep a stale value.
            selection_sink(None)
    if model is None:
        model = FrontendModelSpec(provider="", model_id="")
    return model


async def resolve_context_metadata(
    config_dir: Path, model: FrontendModelSpec, *, stickiness_key: str
) -> FrontendModelSpec:
    """Resolve the account context metadatas a live dispatch would use.

    Resolves the same account as dispatch, not a saved denominator from before
    maximum-context support. Never moves account stickiness merely because a
    viewer opened a cold session — ``read_only=True`` plus the conversation's own
    key is what keeps a look from billing a login's account choice.

    OpenAI-only by its caller's decision, and moved here verbatim so the cold
    viewer and the draft preview agree; the preview SKIPS this step entirely
    (see :func:`synthesise_cold_state`).
    """
    from local_operator.config import ConfigManager
    from local_operator.model.configure import context_spec_for_access
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.failover import (
        AuthRetryKeyState,
        _resolve_access_for_provider,
    )

    configured_model = model

    async def _resolve() -> ModelSpec:
        # AuthStore's SQLite connection is thread-affine. Creation, credential
        # resolution and close belong to this ONE worker's event loop, not
        # separately scheduled default-executor jobs.
        auth = AuthStore(config_dir / "auth.db")
        try:
            access = await _resolve_access_for_provider(
                auth,
                "openai",
                stickiness_key,
                AuthRetryKeyState(),
                None,
                read_only=True,
                model_id=configured_model.model_id,
                scoped_blocks=True,
            )
            settings = ConfigManager(config_dir=config_dir).get_config().values
            return context_spec_for_access(configured_model, access, settings)
        finally:
            auth.close()

    resolved = await asyncio.to_thread(lambda: asyncio.run(_resolve()))
    return FrontendModelSpec.model_validate(resolved.model_dump(mode="python"))


async def synthesise_cold_state(
    *,
    config_dir: Path,
    session_id: str,
    cwd: str,
    birth_model: ModelSpec | None = None,
    model_selection_override: bool = False,
    wakes: Iterable[WakeState] = (),
    selection_sink: Callable[[StoredModelSelection | None], None] | None = None,
) -> FrontendSessionState:
    """Canonical state for a session with no runtime to ask.

    Off the loop: it reads the config file. ``wakes`` is an INPUT rather than
    something read here, because the wake index is a live derived file whose
    reader belongs to the caller that owns the session
    (``AttachedSession._cold_wakes``) — and because a draft has no session to
    look up, so the preview passes nothing.

    ``session_id`` may be empty, which is the draft case: there is then no
    session directory to read a saved selection from, and the resolution falls
    through to config exactly as it does for a conversation that never switched
    models. The metadata step is deliberately NOT applied here — a caller that
    wants it calls :func:`resolve_context_metadata` — so a preview cannot move
    account stickiness on the strength of a synthetic key.
    """
    session_dir = (config_dir / "sessions" / session_id) if session_id else None

    def _build() -> FrontendSessionState:
        model = resolve_conversation_model(
            config_dir,
            session_dir,
            birth_model=birth_model,
            model_selection_override=model_selection_override,
            selection_sink=selection_sink,
        )
        return FrontendSessionState(
            session_id=session_id,
            epoch=f"cold-{session_id}",
            cwd=cwd,
            selected_model=model,
            effective_model=model,
            wakes=list(wakes),
        )

    return await asyncio.to_thread(_build)
