"""Sync credential probes for the audio rungs (the ``createIf`` gate's substrate).

**The frozen rule (mirrors ``imagegen/availability`` and the D9 precedent):**
every probe here is SYNC and LOCAL — sqlite rows, the encrypted store, the
process environment; NO sockets. A session-build path must not open a
connection, and a probe that cannot answer reads as "not available" rather
than raising — a broken probe must never take a session's tool surface down.

**The one deliberate divergence between rungs, reused verbatim from the image
lane's FAL/OpenAI rule:**

- **Radient** is advertised from PERSISTED LOGIN ROWS ONLY
  (``list_credentials`` non-empty; never ``RADIENT_API_KEY`` from the
  environment). A device must not offer a rung the user never signed in to.
- **ElevenLabs / OpenAI** accept a stored row **or an exported key**
  (``ELEVENLABS_API_KEY`` / ``OPENAI_API_KEY``). An exported key genuinely
  runs the call, and the strictness argument that governs a phone has no
  local equivalent here.

The call-time resolvers keep their FULL precedence — availability decides
whether a tool EXISTS; the executor decides what a request authenticates with.
What must never differ is a rung being advertised with one credential class
and spending with a REFUSED one, which is why OpenAI's probe and its call
path both read ``api_key`` rows only — never a ChatGPT OAuth grant, which is
not valid at ``/v1/audio/speech``.

**Reachability gates.** The two tools have different provider spans (the
frozen contract: sound effects run on Radient and ElevenLabs; OpenAI BYOK is
voiceover only), so each builder has its own gate —
:func:`voiceover_reachable` and :func:`sound_effect_reachable` — rather than
one shared "any audio provider" answer that would advertise a tool which can
only fail.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from pathlib import Path

from local_operator.providers.auth_store import AuthStore

logger = logging.getLogger(__name__)

#: The credential namespace the OpenAI audio rung reads: the ``openai-key``
#: login's own, holding a platform API key. NOT ``openai`` — that provider's
#: only logins are ChatGPT OAuth grants, whose tokens are not valid at the
#: audio endpoints.
OPENAI_AUDIO_NAMESPACE = "openai-key"

#: Rung accepts API-key rows only, at probe AND call time. One constant so the
#: two cannot drift: an availability answer about one credential class and a
#: request sent with another is the defect this closes.
OPENAI_AUDIO_KINDS = frozenset({"api_key"})

ELEVENLABS_ENV_KEY = "ELEVENLABS_API_KEY"
OPENAI_ENV_KEY = "OPENAI_API_KEY"


def _open_store(config_dir: Path | None) -> AuthStore:
    """A store rooted at ``config_dir`` (mirrors ``imagegen.availability``).

    The db path is spelled explicitly: ``config_dir`` alone does not set
    ``AuthStore``'s DATABASE (it feeds the env-override tier only), so a
    caller whose ``config_dir`` is not the ambient root — tests, embedded
    roots — would silently read the AMBIENT ``auth.db`` after a naive swap.
    """
    db_path = (Path(config_dir) / "auth.db") if config_dir is not None else None
    return AuthStore(db_path, config_dir=Path(config_dir) if config_dir is not None else None)


def _api_key_from_rows(rows: Sequence[object]) -> str | None:
    """The first usable ``api_key`` row's key, or ``None``.

    Only for rows the store already filtered as usable (``list_credentials``
    excludes disabled ones). Tolerant of a non-row seam: a probe must never
    raise.
    """
    for row in rows:
        credential_type = getattr(row, "credential_type", None)
        data = getattr(row, "data", None)
        if credential_type != "api_key" or not isinstance(data, dict):
            continue
        key = data.get("key")
        if isinstance(key, str) and key:
            return key
    return None


def radient_available(config_dir: Path | None = None) -> bool:
    """Whether Radient is LOGGED IN, from persisted rows alone (never env).

    ``list_credentials`` returns the stored rows and does not touch the
    network. Any failure reads as "not available".
    """
    try:
        store = _open_store(config_dir)
        try:
            return bool(store.list_credentials("radient"))
        finally:
            store.close()
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("radient availability probe failed; reporting unavailable", exc_info=True)
        return False


def elevenlabs_key(config_dir: Path | None = None) -> str | None:
    """The ElevenLabs key the BYOK rung would use: login row → store row → env.

    Store-first, matching every other provider-key reader in the tree. The
    login row wins because it is the credential the user actively signed in
    with; the store row (``lop credential update ELEVENLABS_API_KEY``) and an
    exported environment value are kept so an operator who set the rung up
    that way still runs. Never raises.
    """
    try:
        store = _open_store(config_dir)
        try:
            key = _api_key_from_rows(store.list_credentials("elevenlabs"))
        finally:
            store.close()
        if key:
            return key
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("elevenlabs login-row probe failed; falling back to store/env", exc_info=True)

    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value(ELEVENLABS_ENV_KEY, base=config_dir)
    if stored:
        return stored
    exported = os.environ.get(ELEVENLABS_ENV_KEY)
    return exported or None


def openai_audio_key(config_dir: Path | None = None) -> str | None:
    """The OpenAI audio key (sync form): ``api_key`` rows → store row → env.

    Same precedence as :func:`elevenlabs_key`; the row filter is the point (a
    ChatGPT OAuth row never answers — see the module docstring). Never raises.
    """
    try:
        store = _open_store(config_dir)
        try:
            key = _api_key_from_rows(store.list_credentials(OPENAI_AUDIO_NAMESPACE))
        finally:
            store.close()
        if key:
            return key
    except Exception:  # noqa: BLE001 - a probe must never take its caller down
        logger.debug("openai-key login-row probe failed; falling back to store/env", exc_info=True)

    from local_operator.providers.registry import provider_secret_value

    stored = provider_secret_value(OPENAI_ENV_KEY, base=config_dir)
    if stored:
        return stored
    exported = os.environ.get(OPENAI_ENV_KEY)
    return exported or None


def voiceover_reachable(config_dir: Path | None = None) -> bool:
    """Whether ANY voiceover route is reachable — the tool's createIf gate.

    Radient (persisted login), ElevenLabs (stored/exported key) or OpenAI
    (``api_key`` rows / store row / export) — every probe sync and socket-free.
    """
    return (
        radient_available(config_dir)
        or bool(elevenlabs_key(config_dir))
        or bool(openai_audio_key(config_dir))
    )


def sound_effect_reachable(config_dir: Path | None = None) -> bool:
    """Whether any SOUND-EFFECT route is reachable (frozen contract: no OpenAI).

    Deliberately narrower than :func:`voiceover_reachable`: advertising the
    sound-effect tool on an OpenAI-only machine would offer a tool whose
    every call can only end in "no provider for this request shape".
    """
    return radient_available(config_dir) or bool(elevenlabs_key(config_dir))
