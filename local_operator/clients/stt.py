"""Speech-to-text backends the mobile daemon can reach — a TABLE, not a code path.

One row per path token in the cascade resolver's vocabulary (``provider_stt_*``
plus the native ``model_audio_sidecar``), and this table is the single dispatch
surface: adding a provider is adding a row, never a new branch in the daemon
route. Nothing here invents a token spelling; the resolver owns the vocabulary
and the ordering, this module only answers "can THIS token execute in THIS tree".

Two things this module deliberately does NOT do:

* It does not rank or choose a path. Priority belongs to the cascade resolver
  (``local_operator.stt.cascade``, owned by the STT-cascade session); the daemon
  asks the resolver what is available and only asks here whether that token is
  executable.
* It does not carry HTTP clients for the bring-your-own providers. The single
  implementation of the ElevenLabs and OpenAI transcription calls lives in the
  cascade session's ``local_operator/stt/clients.py`` and is reached through the
  ONE lazy-import probe below (:func:`byo_executor`). A tree that predates that
  module cannot execute those rungs at all, and :func:`backend_ready` says so —
  rather than importing the executor at module scope or duplicating it (the
  duplicated-client failure mode this boundary exists to prevent).

The Radient rung IS implemented here, against the shipped ``RadientClient`` —
the live path, independent of the cascade module.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SttOutcome:
    """One successful transcription, in the shape every surface reads.

    ``path`` is the token that ACTUALLY ran, echoed by the daemon in the HTTP
    answer so the web stores it as ``input_path`` without re-deriving it from a
    cached availability (a TTL race would then record a path that did not run).

    Field names are deliberately the ones the cascade's ``SttOutcome`` carries
    (``text``/``provider``/``model``) so the daemon route reads either object
    the same way; see :func:`_adopt_outcome`.
    """

    text: str
    provider: str
    model: Optional[str] = None
    path: str = ""


@dataclass(frozen=True)
class SttBackend:
    """One row: a resolver token, its provider, and how this tree executes it.

    ``servable`` is a property of the MOBILE surface (not of the provider):
    ``False`` means the phone must never execute this token, whatever the
    resolver says about machine-wide availability — the daemon filters the
    resolver's answer through it and advertises the row's ``reason`` instead.

    ``executor`` names HOW a call for this row is made:
    ``"radient"`` — the adapter below (live);
    ``"cascade"`` — the cascade session's executor, present only once their
        module is in the tree (:func:`byo_executor`);
    ``"none"`` — nothing executes this row anywhere on this surface.
    """

    path: str
    provider: str
    servable: bool
    reason: str = ""
    executor: str = "none"


#: Cascade order is the operator-fixed one (Radient → ElevenLabs → SuperWhisper
#: (excluded) → native model-audio + sidecar); the dict order documents it. The
#: resolver owns the actual priority — nothing here reads this order.
#:
#: `provider_stt_openai` is the frozen resolver token; the rung exists because a
#: BYO OpenAI key is storable today, and it lights up only when the cascade
#: executor lands.
STT_BACKENDS: Mapping[str, SttBackend] = {
    "provider_stt_radient": SttBackend(
        path="provider_stt_radient",
        provider="radient",
        servable=True,
        executor="radient",
    ),
    "provider_stt_elevenlabs": SttBackend(
        path="provider_stt_elevenlabs",
        provider="elevenlabs",
        servable=True,
        executor="cascade",
    ),
    "provider_stt_openai": SttBackend(
        path="provider_stt_openai",
        provider="openai",
        servable=True,
        executor="cascade",
    ),
    "provider_stt_superwhisper": SttBackend(
        # Excluded by research (2026-09-28): SuperWhisper ships no
        # audio-in→transcript-out API at all (its public API is org stats only;
        # the CLI/MCP surface reads the local database). The row stays so the
        # token resolves to an HONEST sentence rather than "unknown path".
        path="provider_stt_superwhisper",
        provider="superwhisper",
        servable=False,
        reason="SuperWhisper has no transcription API.",
    ),
    "model_audio_sidecar": SttBackend(
        # Pending by design: whether the daemon can obtain TEXT through the
        # native model-audio + sidecar path is not settled, and a mic that
        # appears and fails is worse than one that does not. Flips in one place
        # when the cascade answers the mobile contract.
        path="model_audio_sidecar",
        provider="",
        servable=False,
        reason="Voice input here is only available in the desktop app for now.",
    ),
}

#: The probe's two constants. The freeze the cascade session was settling is
#: SETTLED here (agents review convergence, B1): the executor is the cascade's
#: token-targeted ``transcribe_backend`` — it runs the ONE rung the dispatch
#: names, over bytes, and raises the rung's own typed failures.
BYO_EXECUTOR_MODULE = "local_operator.stt.cascade"
BYO_EXECUTOR_ATTR = "transcribe_backend"


class SttBackendUnavailable(RuntimeError):
    """A token cannot execute on this surface right now.

    Carries ``path`` (when one was named) and ``reason`` so the route can answer
    the fixed 503 sentence and a log reader can see which rung was asked for.
    """

    def __init__(self, message: str, *, path: str = "", reason: str = "") -> None:
        super().__init__(message)
        self.path = path
        self.reason = reason or message


def _token_text(value: Any) -> Optional[str]:
    """A token's text, tolerating an enum/str-enum member.

    The cascade's ``AudioPath`` type is theirs to shape; a plain ``Enum`` would
    render as ``AudioPath.provider_stt_radient`` under ``str()``, and its
    ``.value``/``.name`` both carry the token for the StrEnum spelling. Reading
    through this seam is what keeps that freeze out of every call site.
    """
    if isinstance(value, str):
        return value
    for attr in ("value", "name"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str):
            return candidate
    return None


def resolve_backend_key(backend: Any) -> Optional[str]:
    """The ``STT_BACKENDS`` key a resolver token (or provider name) names.

    Tolerant on purpose: the wire may carry the token, the bare provider id
    (``elevenlabs``), or an enum member of the cascade's own type. An unknown
    value answers ``None`` — "not executable here" is a claim only a resolved
    row may make.
    """
    token = _token_text(backend)
    if not token:
        return None
    token = token.strip().lower()
    if token in STT_BACKENDS:
        return token
    prefixed = f"provider_stt_{token}"
    if prefixed in STT_BACKENDS:
        return prefixed
    for key, row in STT_BACKENDS.items():
        if row.provider and row.provider == token:
            return key
    return None


def byo_executor() -> Optional[Callable[..., Any]]:
    """The cascade session's transcription executor, or ``None`` when absent.

    THE ONE PROBE. The executor is the single implementation of the BYO HTTP
    calls (ElevenLabs/OpenAI); this repo's copy of it lands with the cascade
    session's PR, so a tree that predates that module must advertise those rungs
    as non-executable rather than failing at call time. ANY exception during the
    probe reads as absent (fail-closed): availability advertising must never
    raise, and a half-importable executor is not one this build can vouch for.
    """
    try:
        module = __import__(BYO_EXECUTOR_MODULE, fromlist=[BYO_EXECUTOR_ATTR])
        executor = getattr(module, BYO_EXECUTOR_ATTR, None)
    except Exception as exc:  # noqa: BLE001 — deliberately fail-closed
        logger.debug("BYO STT executor unavailable: %s", exc)
        return None
    return executor if callable(executor) else None


def backend_ready(path: str) -> bool:
    """Whether this build can EXECUTE ``path`` right now (executability only).

    Radient is ready wherever the daemon runs (the adapter below is shipped);
    the BYO rungs are ready only when the cascade executor is in the tree. This
    is deliberately separate from credential checks and from the resolver's
    priority: it answers "does the code to run this exist", nothing else.
    """
    row = STT_BACKENDS.get(path)
    if row is None or not row.servable:
        return False
    if row.executor == "radient":
        return True
    if row.executor == "cascade":
        return byo_executor() is not None
    return False


#: File suffix per upload mime. The Radient client posts the audio as a
#: multipart file whose name reaches the upstream decoder, so a neutral
#: ``audio.bin`` is worse than the container's own extension; the mapping is the
#: daemon's allowlist, and anything else keeps `.bin` (the upstream then decides
#: by content, which is what would happen with no extension at all).
_SUFFIX_BY_MIME = {
    "audio/mp4": ".mp4",
    "audio/webm": ".webm",
    "audio/ogg": ".ogg",
    "audio/mpeg": ".mp3",
    "audio/wav": ".wav",
    "audio/x-m4a": ".m4a",
    "audio/aac": ".aac",
}


def _suffix_for_mime(mime: str) -> str:
    base = (mime or "").split(";", 1)[0].strip().lower()
    return _SUFFIX_BY_MIME.get(base, ".bin")


def _transcribe_radient_sync(
    audio: bytes,
    mime: str,
    *,
    api_key: Any,
    base_url: str,
    language: Optional[str],
    prompt: Optional[str],
    model: Optional[str],
) -> SttOutcome:
    """Blocking Radient leg: temp file in, ``create_transcription`` out.

    Runs on a ``to_thread`` worker (see :func:`_transcribe_radient`): the client
    is sync ``requests`` and must not park the daemon's event loop.
    The credential is resolved by the caller, off this thread, so this function
    stays a pure "bytes in, result out" seam that tests can drive directly.

    The temp directory is created BEFORE the try so the ``finally: rmtree``
    owns it unconditionally — the desktop route documents (``transcription.py``
    §"Save the uploaded file temporarily") how the other ordering leaked a temp
    directory on every failed upload, and that lesson is not repeated here.
    """
    from local_operator.clients.radient import RadientClient

    temp_dir = tempfile.mkdtemp(prefix="lop-stt-")
    try:
        file_path = os.path.join(temp_dir, f"audio{_suffix_for_mime(mime)}")
        with open(file_path, "wb") as handle:
            handle.write(audio)
        client = RadientClient(api_key=api_key, base_url=base_url)
        result = client.create_transcription(
            file_path=file_path,
            model=model,
            prompt=prompt,
            language=language,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return SttOutcome(
        text=str(getattr(result, "text", "") or ""),
        provider=str(getattr(result, "provider", "") or "radient"),
        model=getattr(result, "model", None) or model,
        path="provider_stt_radient",
    )


async def _transcribe_radient(
    audio: bytes,
    mime: str,
    *,
    language: Optional[str],
    prompt: Optional[str],
    model: Optional[str],
    config_root: Any,
    store: Any,
) -> SttOutcome:
    """Resolve the Radient credential, then transcribe off the event loop.

    The credential resolution is the SAME resolver every other Radient surface
    asks (store-first central sign-in, then the per-key extension seam), so the
    phone can never run on a credential the desktop would refuse.
    """
    from local_operator.env import resolve_radient_api_base_url
    from local_operator.paths import config_dir as default_config_dir
    from local_operator.providers.radient_credentials import resolve_radient_credential

    root = config_root if config_root is not None else default_config_dir()
    base_url = resolve_radient_api_base_url()
    api_key = await resolve_radient_credential(root, base_url, store=store)
    value = api_key.get_secret_value() if hasattr(api_key, "get_secret_value") else ""
    if not value:
        # Availability normally prevents reaching here (the phone-facing filter
        # requires a persisted credential, and the persisted credential is what
        # this resolver reads); a race — the row deleted between the repaint and
        # the tap — still lands on an honest sentence instead of a 500.
        raise SttBackendUnavailable(
            "Radient is not signed in on this machine.",
            path="provider_stt_radient",
            reason="Sign in to Radient to use voice input on the phone.",
        )
    return await asyncio.to_thread(
        _transcribe_radient_sync,
        audio,
        mime,
        api_key=api_key,
        base_url=base_url,
        language=language,
        prompt=prompt,
        model=model,
    )


def _adopt_outcome(raw: Any, path: str) -> SttOutcome:
    """Normalise whatever an executor returned into :class:`SttOutcome`.

    The cascade's ``SttOutcome`` is their type; reading the fields it carries
    (plus an optional path/provider of its own) instead of importing the class
    keeps this boundary thin — if their shape ever moves, this function is the
    single edit. Their outcome has no ``provider`` (the rung IS the provider),
    so an absent one falls back to the dispatched row's provider id rather
    than answering the phone with an empty string.
    """
    if raw is None:
        raise SttBackendUnavailable("The transcription executor returned no result.", path=path)
    text = getattr(raw, "text", None)
    if not isinstance(text, str):
        raise SttBackendUnavailable(
            "The transcription executor returned an unreadable result.", path=path
        )
    provider = str(getattr(raw, "provider", "") or "")
    if not provider:
        row = STT_BACKENDS.get(path)
        provider = (row.provider if row is not None else "") or ""
    return SttOutcome(
        text=text,
        provider=provider,
        model=getattr(raw, "model", None),
        path=str(getattr(raw, "path", "") or path),
    )


async def _transcribe_via_cascade(
    token: str,
    audio: bytes,
    mime: str,
    *,
    language: Optional[str],
    prompt: Optional[str],
    config_root: Any,
    store: Any,
) -> SttOutcome:
    """Run the ONE rung ``token`` names through the cascade executor.

    THE SETTLED CALL (agents review convergence, B1): token first, then the
    audio BYTES and their mime — the executor re-checks the rung's availability
    through the same resolver the surface read and runs exactly that rung, so
    the phone cannot silently land on a path it never advertised. The phone's
    ``model`` form field is deliberately NOT forwarded: it is an STT-model
    hint the Radient adapter honors, and the BYO rungs run the fixed model ids
    their clients own (see ``stt/clients.py``).

    The executor may be a coroutine function or a blocking one. Wrapping the
    call in ``to_thread`` handles the blocking case without parking the loop,
    and a returned awaitable is awaited on the loop (it has not started yet, so
    nothing was lost by running the CALL off-thread).

    A :class:`~local_operator.stt.cascade.SttUnavailable` — the rung cannot
    serve (no stored key at dispatch time, or a token the cascade does not
    run) — is RE-RAISED as this surface's typed
    :class:`SttBackendUnavailable` (the route answers 503), never left to fall
    into the route's generic ``RuntimeError`` arm and answer 500. Upstream
    failures keep their own typing (``APIError``) so the shared classifier
    still maps 402/502.
    """
    executor = byo_executor()
    if executor is None:
        raise SttBackendUnavailable(
            "Bring-your-own voice providers are not available in this build yet.",
            path=token,
        )
    from local_operator.paths import config_dir as default_config_dir
    from local_operator.stt.cascade import SttUnavailable

    root = config_root if config_root is not None else default_config_dir()
    try:
        result = await asyncio.to_thread(
            executor,
            token,
            audio,
            mime,
            config_dir=root,
            store=store,
            language=language,
            prompt=prompt,
        )
        if inspect.isawaitable(result):
            result = await result
    except SttUnavailable as exc:
        raise SttBackendUnavailable(str(exc), path=token, reason=str(exc)) from None
    return _adopt_outcome(result, token)


async def transcribe_with_backend(
    path: str,
    audio: bytes,
    mime: str,
    *,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    model: Optional[str] = None,
    config_root: Any = None,
    store: Any = None,
) -> SttOutcome:
    """Dispatch one clip to the backend ``path`` names. The dispatch seam.

    Raises :class:`SttBackendUnavailable` for a token this surface cannot
    execute (unknown, unservable, executor absent); upstream failures from the
    clients raise their own typed errors (:class:`APIError` for Radient) and
    pass through for the route to classify.
    """
    key = resolve_backend_key(path)
    if key is None:
        raise SttBackendUnavailable(f"Unknown voice path '{path}'.", path=str(path))
    row = STT_BACKENDS[key]
    if not row.servable:
        raise SttBackendUnavailable(
            row.reason or "This voice path is not available.", path=key, reason=row.reason
        )
    if row.executor == "radient":
        return await _transcribe_radient(
            audio,
            mime,
            language=language,
            prompt=prompt,
            model=model,
            config_root=config_root,
            store=store,
        )
    if row.executor == "cascade":
        return await _transcribe_via_cascade(
            key,
            audio,
            mime,
            language=language,
            prompt=prompt,
            config_root=config_root,
            store=store,
        )
    raise SttBackendUnavailable(
        "This voice path is not available here.", path=key, reason=row.reason
    )
