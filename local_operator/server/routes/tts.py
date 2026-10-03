"""The text-to-speech availability surface.

One route, gated on the managed-mode boundary like its STT twin
(``_LEGACY_CONTROL_PATHS`` in ``server/app.py``):

* ``GET /v1/tts/paths`` — the resolver's report: every rung in cascade order,
  whether it is available, and why; plus ``servable``, the surface's own
  enable/explain bit.

It is the TTS half of ``GET /v1/stt/paths`` and exists for the same reason: a
speak control must be able to ask "can this machine speak aloud, and through
what?" without spending a call to find out. Availability answers "a persisted
credential exists" — a probe, not a promise; the per-request truth surfaces at
synthesis time and falls forward.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from local_operator.config import ConfigManager
from local_operator.env import EnvConfig
from local_operator.providers.auth_store import AuthStore
from local_operator.server.dependencies import (
    get_config_manager,
    get_env_config,
    get_provider_auth_store,
)
from local_operator.server.models.schemas import CRUDResponse
from local_operator.tts import VoicePathResolution
from local_operator.tts.cascade import resolve_voice_path

router = APIRouter()


@router.get(
    "/v1/tts/paths",
    response_model=CRUDResponse[VoicePathResolution],
    summary="Resolve the text-to-speech path",
    tags=["Tools"],
)
async def tts_paths_endpoint(
    config_manager: Annotated[ConfigManager, Depends(get_config_manager)],
    env_config: Annotated[EnvConfig, Depends(get_env_config)],
    store: Annotated[AuthStore, Depends(get_provider_auth_store)],
) -> CRUDResponse[VoicePathResolution]:
    """Report the cascade's rungs for this daemon, and the chosen path.

    Unlike the STT twin there is no model to probe: every TTS rung synthesizes
    from text alone, so the report depends only on which credentials are
    stored.
    """
    resolution = await resolve_voice_path(
        config_dir=config_manager.config_dir,
        base_url=env_config.radient_api_base_url,
        store=store,
    )
    return CRUDResponse(
        status=200,
        message="Speech paths resolved",
        result=resolution,
    )
