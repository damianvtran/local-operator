"""Pick Bedrock as the hosting when nothing else chose one.

WHY THIS EXISTS. ``docs/providers/BEDROCK.md`` ("When does Bedrock turn on by
itself?") promises that an explicit Bedrock signal -- ``AWS_BEARER_TOKEN_BEDROCK``,
``LOP_BEDROCK_AUTH=ambient`` / ``providers.bedrock.auth: ambient``, or an
AWS-compute marker -- turns Bedrock on. Before this module that was only true
for the provider census: a fresh worker (Lambda, ECS/AgentCore, a container
with only the bearer token) running ``lop exec`` or the SDK without
``--hosting`` still failed "Hosting platform is not configured", because the
hosting precedence (agent > flag > team > config) never consulted the census.
The documented ``bedrock/<id>`` model-id form did not route either.

THE RULE, applied only when that precedence produced NO hosting at all:

1. A model id written ``bedrock/<id>`` names its provider: hosting ``bedrock``,
   model ``<id>``.
2. Otherwise, Bedrock's own explicit env signal (the SAME
   :func:`~local_operator.providers.bedrock.connection.bedrock_env_key` the
   registry's ``env_keys`` tier uses, so the signal list lives in one place)
   is present AND Bedrock is the ONLY chat provider holding a usable
   credential. Then hosting ``bedrock`` with the given model (or ``None``, which
   the caller resolves to Bedrock's default model).

CONSTRAINTS.

* Never overrides a hosting someone chose (agent row, flag, team, config):
  callers invoke this only when that precedence came back empty.
* A laptop exporting only ``AWS_PROFILE`` / ``AWS_ACCESS_KEY_ID`` never picks
  Bedrock: those are not a Bedrock signal (design D6), and ``bedrock_env_key``
  is what enforces it.
* "Only provider" is deliberate: with two usable providers, picking one would
  be a guess, and the not-configured error is the honest answer.
* A census that cannot tell (unreadable store -> ``None``) picks nothing.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROVIDER_ID = "bedrock"
_MODEL_PREFIX = f"{PROVIDER_ID}/"


def implied_bedrock_hosting(
    model_name: str | None,
    *,
    config_dir: Path | str | None,
    config_values: Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, str | None] | None:
    """``("bedrock", model)`` when the run implies Bedrock, else ``None``.

    ``model`` is the id with any ``bedrock/`` prefix stripped, or ``None`` when
    no model was given (the caller applies ``default_model_for("bedrock")``).
    Call ONLY when no agent/flag/team/config hosting exists.
    """
    if model_name and model_name.startswith(_MODEL_PREFIX):
        stripped = model_name[len(_MODEL_PREFIX) :].strip()
        return PROVIDER_ID, (stripped or None)

    from local_operator.providers.bedrock.connection import bedrock_env_key

    if config_values is None and config_dir is not None:
        # Read THIS run's config dir, not ``paths.config_dir()`` (what
        # ``bedrock_env_key`` falls back to): a ConfigManager pointed elsewhere
        # (tests, the SDK's own dir) must see its own ``providers.bedrock.auth``.
        from local_operator.config import read_config_values

        try:
            config_values = read_config_values(Path(config_dir)) or {}
        except Exception:  # noqa: BLE001 - unreadable config means "unset"
            config_values = {}

    if bedrock_env_key(environ, config_values) is None:
        return None

    # The census reads the env tier through the registry's ``env_keys``
    # callable, i.e. os.environ: an ``environ`` override only narrows the
    # signal check above (tests), never what the store/env census reports.
    from local_operator.providers.model_access import credentialed_chat_providers_here

    try:
        credentialed = credentialed_chat_providers_here(
            config_dir=Path(config_dir) if config_dir is not None else None,
            config_values=config_values,
        )
    except Exception:  # noqa: BLE001 - an inference must never break boot
        logger.debug("bedrock auto-select: provider census failed", exc_info=True)
        return None
    if credentialed is None or credentialed != {PROVIDER_ID}:
        return None
    return PROVIDER_ID, (model_name or None)
