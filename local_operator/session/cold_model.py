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

logger = logging.getLogger(__name__)


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
        from local_operator.config import ConfigManager

        config = ConfigManager(config_dir=config_dir)
        provider = str(config.get_config_value("hosting", "") or "")
        model_id = str(config.get_config_value("model_name", "") or "")
        if session_dir is not None:
            saved = read_model_selection(session_dir)
        if selection_sink is not None:
            selection_sink(saved)
        if birth_model is not None and (saved is None or model_selection_override):
            model = FrontendModelSpec(**birth_model.model_dump())
        elif saved is not None:
            model = FrontendModelSpec(
                provider=saved.provider,
                model_id=saved.model_id,
                reasoning_effort=saved.effort,
            )
        else:
            if provider and not model_id:
                from local_operator.model.defaults import default_model_for

                model_id = default_model_for(provider) or ""
            model = FrontendModelSpec(provider=provider, model_id=model_id)
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
