"""Per-agent profile pictures, owned locally.

WHY THIS MODULE EXISTS. An agent's picture is shown wherever a session bound to
that agent is drawn (desktop chat header and sidebar, the phone relay, mesh peer
rows). The desktop proof of concept (local-operator-ui#1517, by bbqben) kept the
picture in the renderer's ``localStorage``, which no other surface can read. This
module makes the backend the owner: ONE file per agent, beside its ``agent.yml``,
plus a metadata block in ``agent.yml`` that every listing can carry without a
per-row file read.

THE FROZEN LOCAL CONTRACT (agreed with the UI and hub lanes; do not rename):

* Storage is ``<agents_dir>/<agent_id>/avatar.png`` -- a square 512x512 PNG -- or,
  only when that PNG would exceed :data:`STORED_MAX_BYTES`, ``avatar.jpg`` at the
  same size. EXACTLY ONE of the two exists; a write removes the stale sibling.
* The stored bytes are RE-ENCODED PIXELS ONLY: EXIF orientation is baked in first,
  then nothing ancillary survives (no EXIF, no GPS, no ICC, no text chunks). A
  phone photo carries the location it was taken at, and a picture that is later
  published to a hub or a peer must not carry it along.
* ``agent.yml`` gains ``avatar: {sha256, mime, bytes, updated_at, filename} | null``;
  the ``sha256`` is the digest of the STORED file's bytes, so it doubles as the HTTP
  ETag and as a cache-busting ``?v=`` query a client builds without asking.
  ``filename`` is the stored basename (``avatar.png`` / ``avatar.jpg``); a legacy
  block written before the field existed falls back by mime (see
  :func:`stored_name`). ``agent.yml`` is the authority: a stray file with no
  metadata is not a picture, and a block whose file is missing, is a symlink, or
  hashes to something other than ``sha256`` reads as NO picture.
* Input gate: PNG, JPEG, WebP, GIF (first frame) and HEIC (only when the
  ``images`` extra's pillow-heif loads). <= :data:`INPUT_MAX_BYTES` in and
  <= :data:`INPUT_MAX_PIXELS` decoded, identified by magic bytes, never by name.
  Refusals carry one of the :data:`REASONS` words so every surface (route, CLI,
  agent tool) can render its own copy from the same machine answer.
* Change signal: :func:`subscribe` registers an in-process callback fired with an
  :class:`AvatarChange` AFTER the write commits. A write from another process (the
  CLI) cannot reach that hook, so cross-process consumers reconcile on
  ``agent.yml`` (the desktop feed's authoring probe already watches it).

There is deliberately NO path by which a remote source writes a picture here: the
cloud/hub direction is publish-only.

Pillow is imported lazily through :func:`local_operator.helpers.pillow_image_module`
-- ``tests/unit/test_import_graph.py`` pins that no module on the CLI's import path
imports PIL at module level, and this module is reached from ``agents.py``.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import stat
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

#: The two stored names. PNG is the normal case; JPEG only rescues an image whose
#: lossless encoding is over the cap (photographic noise at 512^2 can do that).
PNG_NAME = "avatar.png"
JPEG_NAME = "avatar.jpg"
STORED_NAMES = (PNG_NAME, JPEG_NAME)
_MIME_BY_NAME = {PNG_NAME: "image/png", JPEG_NAME: "image/jpeg"}

#: The stored edge in pixels. 512 covers the largest display (the editor's preview
#: at 2x density) with headroom, and keeps every stored file comfortably small.
EDGE = 512
#: Over this the PNG is replaced by a JPEG.
STORED_MAX_BYTES = 1024 * 1024
#: JPEG fallback quality: visually lossless at this size, and far under the cap.
JPEG_QUALITY = 90
#: Input caps. The byte cap bounds the request body; the pixel cap bounds the
#: DECODE, which a byte cap cannot see (a decompression bomb is small on disk).
#: 40 MP rather than ``imaging.IMAGE_MAX_PIXELS`` (50 MP): the contract froze 40,
#: and every current phone sensor is under it.
INPUT_MAX_BYTES = 10 * 1024 * 1024
INPUT_MAX_PIXELS = 40_000_000

#: Machine refusal words, the whole vocabulary. Surfaces translate from these.
UNSUPPORTED_TYPE = "unsupported_type"
TOO_LARGE = "too_large"
TOO_MANY_PIXELS = "too_many_pixels"
UNDECODABLE = "undecodable"
NOT_FOUND = "not_found"
#: Pillow (the ``images`` extra) is absent or broken on this install. Not in the
#: frozen four-plus-one because it is not a fault of the INPUT; surfaces render it
#: as an install problem.
IMAGING_UNAVAILABLE = "imaging_unavailable"
REASONS = frozenset(
    {UNSUPPORTED_TYPE, TOO_LARGE, TOO_MANY_PIXELS, UNDECODABLE, NOT_FOUND, IMAGING_UNAVAILABLE}
)

#: What the gate accepts, as sniffed by ``media.sniff_image``.
_ACCEPTED_MIMES = frozenset({"image/png", "image/jpeg", "image/webp", "image/gif"})
_HEIF_MIMES = frozenset({"image/heic", "image/heif"})


class AvatarError(ValueError):
    """A refused picture write or read, carrying one :data:`REASONS` word.

    A ``ValueError`` so existing ``except ValueError`` boundaries treat it as bad
    input; ``reason`` is the stable machine answer and ``str()`` an English
    developer sentence (logs, tracebacks), never user-facing copy.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        if reason not in REASONS:
            raise ValueError(f"unknown avatar refusal reason {reason!r}")
        self.reason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}" if detail else reason)


@dataclass(frozen=True)
class AvatarChange:
    """One committed change, as delivered to :func:`subscribe` callbacks.

    ``sha256``/``mime``/``path`` are all ``None`` for a removal. Field names are
    part of the frozen contract (the SSE event and the hub lane read them).
    """

    agent_id: str
    sha256: Optional[str]
    mime: Optional[str]
    path: Optional[Path]


@dataclass(frozen=True)
class NormalisedAvatar:
    """The bytes to store, and what they are."""

    data: bytes
    mime: str
    name: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.data).hexdigest()


# -- the change hook ---------------------------------------------------------

_subscribers: list[Callable[[AvatarChange], None]] = []
_subscribers_lock = threading.Lock()


def subscribe(callback: Callable[[AvatarChange], None]) -> Callable[[], None]:
    """Register ``callback`` for committed changes; returns the unsubscribe.

    Fired synchronously on the WRITING thread, after the file and ``agent.yml``
    both landed. A callback must be cheap and must not raise -- a raise is logged
    and swallowed, because a broken consumer must not turn a committed write into
    a reported failure. Async consumers hop to their loop themselves
    (``loop.call_soon_threadsafe``).
    """
    with _subscribers_lock:
        _subscribers.append(callback)

    def _unsubscribe() -> None:
        with _subscribers_lock:
            try:
                _subscribers.remove(callback)
            except ValueError:
                pass

    return _unsubscribe


def _notify(change: AvatarChange) -> None:
    with _subscribers_lock:
        callbacks = list(_subscribers)
    for callback in callbacks:
        try:
            callback(change)
        except Exception:  # noqa: BLE001 — see subscribe(): a consumer cannot fail a commit
            logger.exception("agent avatar subscriber failed for %s", change.agent_id)


# -- normalise ---------------------------------------------------------------


def _image_module_for(mime: str) -> Any:
    from local_operator.helpers import heif_image_module, pillow_image_module

    module = heif_image_module() if mime in _HEIF_MIMES else pillow_image_module()
    if module is None:
        if mime in _HEIF_MIMES:
            # HEIC needs pillow-heif specifically; with Pillow alone it is a type
            # this install cannot read, which is the input's problem to the user.
            if pillow_image_module() is not None:
                raise AvatarError(UNSUPPORTED_TYPE, "HEIC needs the images extra (pillow-heif)")
        raise AvatarError(IMAGING_UNAVAILABLE, "Pillow is not importable on this install")
    return module


def normalise(data: bytes) -> NormalisedAvatar:
    """Validate ``data`` and re-encode it to the stored form.

    Order matters: the byte cap and the magic-byte sniff run before Pillow is
    touched; the pixel cap runs on the header size BEFORE ``load()`` decodes, so a
    bomb is refused without allocating it. Centre-crop to a square: the desktop
    editor already sends a square crop, so this only shapes the CLI/agent path,
    where "the middle of the photo" is the least-surprising framing.
    """
    from local_operator.media import sniff_image

    if len(data) > INPUT_MAX_BYTES:
        raise AvatarError(TOO_LARGE, f"{len(data)} bytes, cap {INPUT_MAX_BYTES}")
    info = sniff_image(data)
    if info is None or info.mime_type not in (_ACCEPTED_MIMES | _HEIF_MIMES):
        raise AvatarError(UNSUPPORTED_TYPE, "not a PNG, JPEG, WebP, GIF or HEIC image")
    if info.width and info.height and info.width * info.height > INPUT_MAX_PIXELS:
        raise AvatarError(TOO_MANY_PIXELS, f"{info.width}x{info.height}")

    image_module = _image_module_for(info.mime_type)
    from PIL import ImageOps  # lazy: the module above proved Pillow imports

    try:
        image = image_module.open(io.BytesIO(data))
        width, height = image.size
        if width * height > INPUT_MAX_PIXELS:
            raise AvatarError(TOO_MANY_PIXELS, f"{width}x{height}")
        # First frame only: ``open`` is positioned on frame 0 and ``load`` decodes
        # just that one, so an animated GIF/WebP becomes a still.
        image.load()
        # Orientation into the PIXELS before any geometry, or a phone portrait
        # is cropped and stored sideways once the tag is dropped below.
        image = ImageOps.exif_transpose(image)
        if image.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N"):
            # Pillow's 16-bit -> 8-bit ``convert`` CLAMPS (anything >= 255 is
            # white), so a 16-bit greyscale PNG -- what scientific and some
            # medical/scanner exports produce -- would saturate to a white disc.
            # Scale the 16-bit range down to 8 first; ``point`` on a mode "I"/
            # "I;16" image evaluates the callable per-value on the 16-bit data.
            image = image.point(lambda value: value * (1 / 256)).convert("L")
        has_alpha = image.mode in ("RGBA", "LA", "PA") or (
            image.mode == "P" and "transparency" in image.info
        )
        image = image.convert("RGBA" if has_alpha else "RGB")
        side = min(image.size)
        left = (image.width - side) // 2
        top = (image.height - side) // 2
        image = image.crop((left, top, left + side, top + side))
        if image.size != (EDGE, EDGE):
            image = image.resize((EDGE, EDGE), image_module.Resampling.LANCZOS)
    except AvatarError:
        raise
    except Exception as exc:  # noqa: BLE001 — any decoder failure is "undecodable"
        raise AvatarError(UNDECODABLE, str(exc)) from exc

    # A FRESH image with no ``info``: ``save`` copies ancillary data from
    # ``image.info`` (ICC, EXIF, text) on some codecs, and the contract is pixels
    # only. ``frombytes`` of the raw buffer is the one construction guaranteed to
    # carry nothing else.
    clean = image_module.frombytes(image.mode, image.size, image.tobytes())
    buffer = io.BytesIO()
    clean.save(buffer, format="PNG", optimize=True)
    encoded = buffer.getvalue()
    if len(encoded) <= STORED_MAX_BYTES:
        return NormalisedAvatar(encoded, "image/png", PNG_NAME)
    buffer = io.BytesIO()
    # JPEG has no alpha: flatten onto white, the usual light-surface backdrop.
    if clean.mode == "RGBA":
        backdrop = image_module.new("RGB", clean.size, (255, 255, 255))
        backdrop.paste(clean, mask=clean.getchannel("A"))
        clean = backdrop
    clean.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return NormalisedAvatar(buffer.getvalue(), "image/jpeg", JPEG_NAME)


# -- store / remove / read ---------------------------------------------------


def _write_bytes_atomically(target: Path, data: bytes) -> None:
    """Stage beside ``target`` and rename, so a reader never sees a short file."""
    temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink(missing_ok=True)


def store(agent_dir: Path, avatar: NormalisedAvatar) -> dict[str, Any]:
    """Write the normalised file into ``agent_dir`` and return its metadata block.

    ONLY the new file is written here. The stale sibling is removed by the caller
    AFTER ``agent.yml`` names the new file (:func:`remove_siblings`): deleting it
    first left a PNG->JPEG flip that died in between with metadata naming a PNG
    that no longer existed. In this order every interruption reads as either the
    old picture (died before the yml write) or the new one (died after it, with a
    stray sibling that is harmless because ``agent.yml`` is the authority).
    """
    _write_bytes_atomically(agent_dir / avatar.name, avatar.data)
    return {
        "sha256": avatar.sha256,
        "mime": avatar.mime,
        "bytes": len(avatar.data),
        "updated_at": datetime.now(timezone.utc),
        "filename": avatar.name,
    }


def remove_siblings(agent_dir: Path, keep: str) -> None:
    """Delete every stored picture file except ``keep`` (the committed one)."""
    for name in STORED_NAMES:
        if name != keep:
            (agent_dir / name).unlink(missing_ok=True)


def remove(agent_dir: Path) -> None:
    """Delete both possible stored files (absent is fine)."""
    for name in STORED_NAMES:
        (agent_dir / name).unlink(missing_ok=True)


def stored_name(meta: Any) -> Optional[str]:
    """The stored basename the metadata block names, or ``None`` when it is unusable.

    ``filename`` wins when present; a LEGACY block (written before the field
    existed) falls back by mime. The name must be exactly one of
    :data:`STORED_NAMES` and agree with ``mime`` -- ``agent.yml`` is writable by
    the agent itself, so a ``../x`` or ``/etc/passwd`` filename (or a name that
    contradicts the mime) is never joined onto a path: it reads as no picture.
    """
    mime = getattr(meta, "mime", None)
    expected = next((n for n, m in _MIME_BY_NAME.items() if m == mime), None)
    if expected is None:
        return None
    filename = getattr(meta, "filename", None)
    if filename in (None, ""):
        return expected
    if not isinstance(filename, str) or filename != expected:
        return None
    return filename


def stored_path(agent_dir: Path, meta: Any) -> Optional[Path]:
    """The stored file ``meta`` names, if it is a REGULAR file on disk.

    ``lstat``, not ``is_file``: a symlink planted as ``avatar.png`` would be
    followed by ``is_file`` and served from outside the agent directory.
    """
    name = stored_name(meta)
    if name is None:
        return None
    path = agent_dir / name
    try:
        return path if stat.S_ISREG(os.lstat(path).st_mode) else None
    except (OSError, ValueError):
        return None


def read(agent_dir: Path, meta: Any) -> tuple[bytes, str, str]:
    """``(bytes, mime, sha256)`` for the stored picture, or ``AvatarError(not_found)``.

    ``meta`` is the ``agent.yml`` block (the authority). The file is opened
    WITHOUT following a symlink, and the bytes actually read are hashed: a file
    that does not hash to ``meta.sha256`` (tampered, truncated, replaced behind
    the metadata's back) is ``not_found`` rather than served under an ETag that is
    not the digest of the body. The sha returned is therefore the body's own.
    """
    if meta is None:
        raise AvatarError(NOT_FOUND, "no picture set")
    mime = getattr(meta, "mime", None)
    sha = getattr(meta, "sha256", None)
    path = stored_path(agent_dir, meta)
    if path is None or not sha or not mime:
        raise AvatarError(NOT_FOUND, "picture file missing")
    # O_NONBLOCK: a FIFO swapped in after ``stored_path``'s lstat (the agent can
    # write its own directory) would otherwise block this open forever, never
    # reaching the ``S_ISREG`` fstat check below, which then refuses it. It has no
    # effect on a regular-file read.
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        fd = os.open(path, flags)
        with os.fdopen(fd, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                raise AvatarError(NOT_FOUND, "picture is not a regular file")
            # Bounded: a stored picture is <= STORED_MAX_BYTES by construction, so
            # a larger file is not ours and must not be read whole.
            data = handle.read(STORED_MAX_BYTES + 1)
    except OSError as exc:
        raise AvatarError(NOT_FOUND, "picture file unreadable") from exc
    if len(data) > STORED_MAX_BYTES or hashlib.sha256(data).hexdigest() != sha:
        raise AvatarError(NOT_FOUND, "picture does not match its recorded digest")
    return data, mime, sha


def emit(agent_id: str, agent_dir: Path, meta: Any) -> None:
    """Fire :func:`subscribe` callbacks for a committed set (``meta``) or removal."""
    # This process's row index must not serve the old sha for its TTL after a
    # write it made itself (another process's write waits out the TTL instead).
    with _index_lock:
        _index_cache.clear()
    if meta is None:
        _notify(AvatarChange(agent_id, None, None, None))
        return
    mime = getattr(meta, "mime", None)
    _notify(AvatarChange(agent_id, meta.sha256, mime, stored_path(agent_dir, meta)))


# -- surface helpers -----------------------------------------------------------

#: The HTTP status each refusal maps to, so the desktop route and the phone relay
#: answer one refusal the same way. 413 for the byte cap (the body is too big),
#: 415 for an unrecognised type, 422 for content that is the right kind but
#: unusable, 503 for a missing install-side capability.
HTTP_STATUS = {
    UNSUPPORTED_TYPE: 415,
    TOO_LARGE: 413,
    TOO_MANY_PIXELS: 422,
    UNDECODABLE: 422,
    NOT_FOUND: 404,
    IMAGING_UNAVAILABLE: 503,
}


def refusal_message(reason: str) -> Any:
    """The catalogued sentence for a refusal word (a ``Msg``; render it per surface).

    Lazy import: the generated key module pulls the i18n runtime, which this
    module's storage half does not need.
    """
    from local_operator.i18n.keys import wire_errors_agent_avatar as _w

    if reason == UNSUPPORTED_TYPE:
        return _w.unsupported_type()
    if reason == TOO_LARGE:
        return _w.too_large(limit_mib=str(INPUT_MAX_BYTES // (1024 * 1024)))
    if reason == TOO_MANY_PIXELS:
        return _w.too_many_pixels(limit_mp=str(INPUT_MAX_PIXELS // 1_000_000))
    if reason == UNDECODABLE:
        return _w.undecodable()
    if reason == IMAGING_UNAVAILABLE:
        return _w.imaging_unavailable()
    return _w.not_found()


def refusal_envelope(reason: str, *, locale: Optional[str] = None) -> dict[str, Any]:
    """``{code, params, text}`` with the REASON WORD as ``code``.

    The reason word rides as the wire code (``Msg.envelope(code=...)``'s
    machine-category override) so a client switches on ``unsupported_type``
    etc. by equality; the sentence is the catalogued, translated half.
    """
    return refusal_message(reason).envelope(code=reason, locale=locale)


def etag(sha256: str) -> str:
    """The strong quoted ETag for a stored picture."""
    return f'"{sha256}"'


def meta_json(meta: Any) -> Optional[dict[str, Any]]:
    """The wire form of an ``AgentAvatarMeta`` (or ``None``): ISO ``updated_at``."""
    if meta is None:
        return None
    updated = getattr(meta, "updated_at", None)
    return {
        "sha256": meta.sha256,
        "mime": meta.mime,
        "bytes": meta.bytes,
        "updated_at": updated.isoformat() if isinstance(updated, datetime) else updated,
        # The EFFECTIVE name (a legacy block's mime fallback included), so a
        # consumer never has to re-derive it; None only for an unusable block.
        "filename": stored_name(meta),
    }


# -- session row identity ------------------------------------------------------
#
# Every session listing (desktop, phone relay, mesh peer rows) adds
# ``agent_id`` + ``avatar_sha256`` to a row bound to an agent, so no surface
# fetches per row. A session's binding is an agent NAME (``attachment.json``), so
# the name has to be resolved to the registry row -- and a listing does that for
# up to hundreds of rows per paint. The index below reads ``agent.yml`` files
# DIRECTLY instead of constructing an ``AgentRegistry`` (whose constructor
# mkdirs and migrates: a listing must never create registry state), and is
# memoised per config root on the agents directory's per-file stat fingerprint,
# so a warm listing pays one readdir plus one stat per agent and no YAML parse.


@dataclass(frozen=True)
class AgentIdentity:
    """What a session row needs to draw an agent's picture."""

    agent_id: str
    avatar_sha256: Optional[str]


_INDEX_TTL_S = 1.0
_index_lock = threading.Lock()
#: root -> (computed_at, fingerprint, {name: AgentIdentity})
_index_cache: dict[str, tuple[float, tuple[Any, ...], dict[str, AgentIdentity]]] = {}


def _agents_fingerprint(agents_dir: Path) -> tuple[Any, ...]:
    parts: list[Any] = []
    try:
        with os.scandir(agents_dir) as entries:
            names = sorted(entry.name for entry in entries if not entry.name.startswith("."))
    except OSError:
        return ()
    for name in names:
        try:
            st = os.stat(agents_dir / name / "agent.yml")
            parts.append((name, st.st_ino, st.st_size, st.st_mtime_ns))
        except OSError:
            continue
    return tuple(parts)


class _BlockView:
    """A raw ``agent.yml`` ``avatar`` mapping, read like an ``AgentAvatarMeta``.

    The phone relay and the row index read ``agent.yml`` without constructing an
    ``AgentRegistry`` (which mkdirs and migrates), so they hold a dict, not the
    pydantic model; this gives :func:`stored_name` / :func:`read` the attribute
    shape they take, with one reader of the filename rule instead of two.
    """

    def __init__(self, block: dict[str, Any]) -> None:
        self.sha256 = block.get("sha256")
        self.mime = block.get("mime")
        self.filename = block.get("filename")


def is_plain_agent_id(agent_id: Any) -> bool:
    """Is ``agent_id`` safe to join under ``agents_dir`` as ONE path segment?

    The shared id guard for every surface that takes an id from a URL. A NUL byte
    is refused here because ``Path.read_text`` raises a bare ``ValueError`` on it
    (a 500 on the relay route) rather than the ``OSError`` the callers expect.
    """
    return (
        isinstance(agent_id, str)
        and bool(agent_id)
        and "\x00" not in agent_id
        and agent_id not in (".", "..")
        and Path(agent_id).name == agent_id
    )


def _build_index(agents_dir: Path) -> dict[str, AgentIdentity]:
    import yaml

    index: dict[str, AgentIdentity] = {}
    try:
        with os.scandir(agents_dir) as entries:
            dirs = sorted(entry.name for entry in entries if not entry.name.startswith("."))
    except OSError:
        return index
    for name in dirs:
        try:
            payload = yaml.safe_load(_read_manifest_text(agents_dir / name / "agent.yml"))
        except (OSError, UnicodeDecodeError, yaml.YAMLError):
            continue
        if not isinstance(payload, dict):
            continue
        agent_name = payload.get("name")
        agent_id = payload.get("id")
        if not isinstance(agent_name, str) or not isinstance(agent_id, str) or not agent_id:
            continue
        avatar = payload.get("avatar")
        sha = avatar.get("sha256") if isinstance(avatar, dict) else None
        if isinstance(avatar, dict) and stored_name(_BlockView(avatar)) is None:
            # A block naming a file we would never serve (hostile filename, or a
            # name that contradicts its mime) advertises no picture on the rows.
            sha = None
        # First row wins for a duplicated name, matching ``get_agent_by_name``'s
        # first-match rule closely enough for a picture (names are unique in
        # practice: the registry refuses a duplicate on create and rename).
        index.setdefault(
            agent_name, AgentIdentity(agent_id, sha if isinstance(sha, str) and sha else None)
        )
    return index


def agent_index(config_root: Path) -> dict[str, AgentIdentity]:
    """``{agent name: AgentIdentity}`` for every readable agent under ``config_root``."""
    import time

    agents_dir = Path(config_root) / "agents"
    key = str(agents_dir)
    now = time.monotonic()
    with _index_lock:
        cached = _index_cache.get(key)
    if cached is not None and now - cached[0] < _INDEX_TTL_S:
        return cached[2]
    fingerprint = _agents_fingerprint(agents_dir)
    if cached is not None and cached[1] == fingerprint:
        index = cached[2]
    else:
        index = _build_index(agents_dir)
    with _index_lock:
        _index_cache[key] = (now, fingerprint, index)
    return index


def _chief_of_staff_agent(config_root: Path, session_id: str) -> Optional[str]:
    """The agent name of a chief-of-staff session that carries no attachment.

    Aida's bootstrap writes ``attachment.json`` with her role name, so the
    ordinary path already resolves her; this covers a session whose sidecar is
    absent or unreadable but which ``aida.state`` still names as hers. Generic
    rather than a hard-coded ``"aida"`` string here: the role name comes from the
    module that owns it, and any OTHER proactive/chief-of-staff agent is bound
    through its own attachment like every agent session.
    """
    try:
        from local_operator.aida.bootstrap import ROLE_NAME
        from local_operator.aida.state import is_aida_session
    except Exception:  # noqa: BLE001 — an absent/broken Aida module is "not hers"
        return None
    try:
        return ROLE_NAME if is_aida_session(config_root, session_id) else None
    except Exception:  # noqa: BLE001 — a listing never fails on this decoration
        return None


def session_agent_identity(
    config_root: Path, session_id: str, agent_name: Optional[str] = None
) -> AgentIdentity | None:
    """The bound agent's id and picture sha for one session, or ``None``.

    ``agent_name`` is the binding when the caller already read the attachment
    (every listing does); otherwise it is read here. Never raises: a row that
    cannot be resolved simply carries ``null``s, which every client draws as the
    glyph fallback.
    """
    try:
        name = agent_name
        if name is None:
            from local_operator.resume import read_session_attachment

            stored = read_session_attachment(Path(config_root) / "sessions" / session_id)
            name = stored.agent if stored else ""
        if not name:
            name = _chief_of_staff_agent(Path(config_root), session_id) or ""
        if not name:
            return None
        return agent_index(Path(config_root)).get(name)
    except Exception:  # noqa: BLE001 — see docstring
        logger.debug("agent identity unresolved for session %s", session_id, exc_info=True)
        return None


def session_row_fields(
    config_root: Path, session_id: str, agent_name: Optional[str] = None
) -> dict[str, Optional[str]]:
    """``{"agent_id", "avatar_sha256"}`` for a listing row -- BOTH keys, always.

    Always both keys, ``None`` when unbound: the renderers' row merge is
    "an absent key is not a claim", so a row whose agent lost its picture must
    say ``null`` or the stale picture stays painted.
    """
    identity = session_agent_identity(config_root, session_id, agent_name)
    return {
        "agent_id": identity.agent_id if identity else None,
        "avatar_sha256": identity.avatar_sha256 if identity else None,
    }


def _read_manifest_text(path: Path) -> str:
    """Text of an ``agent.yml`` that is a regular file; ``OSError`` otherwise.

    ``agent.yml`` sits in a directory the agent itself can write, and
    ``Path.read_text`` opens a FIFO planted there with a blocking ``open(2)`` that
    never returns, wedging the relay request or the row index forever.
    ``O_NONBLOCK`` makes the open return at once and the ``fstat`` refuses
    anything that is not a regular file (FIFO, device). Symlinks are still
    followed, as ``read_text`` always did: a linked ``agent.yml`` is a layout the
    registry tolerates, and only the picture itself must never follow one.
    """
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise OSError(f"{path} is not a regular file")
        return handle.read().decode("utf-8")


def read_stored(config_root: Path, agent_id: str) -> tuple[bytes, str, str]:
    """``(bytes, mime, sha256)`` for ``agent_id`` read straight from its ``agent.yml``.

    For surfaces that serve the picture WITHOUT an ``AgentRegistry`` (the phone
    relay: a registry's constructor mkdirs and migrates, and a read route must
    not create registry state). ``agent_id`` arrives from a URL, so it must be a
    single plain path segment -- anything else is ``not_found``, never a path.
    """
    import yaml

    if not is_plain_agent_id(agent_id):
        raise AvatarError(NOT_FOUND, "not an agent id")
    agent_dir = Path(config_root) / "agents" / agent_id
    try:
        payload = yaml.safe_load(_read_manifest_text(agent_dir / "agent.yml"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ValueError) as exc:
        raise AvatarError(NOT_FOUND, "no such agent") from exc
    block = payload.get("avatar") if isinstance(payload, dict) else None
    if not isinstance(block, dict):
        raise AvatarError(NOT_FOUND, "no picture set")
    return read(agent_dir, _BlockView(block))


def stored_sha(config_root: Path, agent_id: str) -> Optional[str]:
    """The metadata sha for ``agent_id`` from the memoised index, or ``None``.

    Lets a relay answer a revalidation (304) without reading the file.
    """
    if not is_plain_agent_id(agent_id):
        return None
    for identity in agent_index(Path(config_root)).values():
        if identity.agent_id == agent_id:
            return identity.avatar_sha256
    return None
