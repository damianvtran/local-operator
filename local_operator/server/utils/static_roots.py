"""Which files ``/v1/static/*`` may serve, and the response policy those routes carry.

WHY THIS EXISTS. The static routes used to ``expanduser().resolve()`` whatever
``path`` the caller sent and serve it when its extension was on a mime
allowlist. They sit outside both the managed desktop boundary and the legacy
gate (``server/app.py``), and the CORS middleware echoes origins until a desktop
allowlist is installed, so any page the operator visits while the daemon is up
(or any local process) could ``fetch`` any readable image/audio/video/HTML file
on the machine -- absolute paths and symlinks included. The turn-supplements
security rounds recorded this as S-6 / F6 (``docs/design/turn-supplements.md``
§4.1, §8); the UI-side fix (local-operator-ui PR #931) closes the app's own door
only.

This module closes the route's side of it, in two parts that need no UI change:

1. **A served-root allowlist** (:func:`resolve_servable`). ``path`` must resolve,
   symlinks followed, to a regular file INSIDE one of the roots below.
2. **A response policy** (:func:`response_policy`): ``nosniff``, a CSP, and a
   ``frame-ancestors`` naming only the app. The CORS half lives in the app
   middleware that calls this module (``server/app.py``).

WHAT IS STILL OPEN, stated so nobody reads this as more than it is: the routes
remain unauthenticated. The UI loads them through ``<img>``/``<video>``/
``<iframe src>``, which cannot carry an ``Authorization`` header, so the bearer
cannot simply be required. The follow-up is a short-lived signed query token
minted by an authenticated endpoint and verified here, which needs the UI lane
to request it. Until then a hostile local caller can still *trigger* a read of
anything inside the roots below -- but no longer anything outside them, and a
web page can no longer *read the response* cross-origin.

THE ROOTS, and why each is there (all compared as realpaths):

* the agent home (``paths.agent_home_dir()``, ``~/local-operator-home``): the
  default workspace, where agents are told to put what they produce;
* ``<config>/sessions``: each session's scratchpad lives in its session
  directory, and tools write screenshots and renders there;
* ``<config>/uploads``: where a decoded chat attachment lands;
* the working directory of every non-stale live session (``run/mobile``
  discovery records) and of every registered agent: the canvas previews "the
  file the agent just wrote", wherever the user pointed the session. This is the
  deliberately wide arm -- a session started in ``~`` makes ``~`` a root -- and
  it is what keeps "preview a file in my workspace" working with no setting;
* explicitly configured roots: ``static.roots`` (settings page / ``config.yml``)
  and ``LOCAL_OPERATOR_STATIC_ROOTS`` (an ``os.pathsep``-separated list, for a
  daemon launched from a script).

Two refusals apply on top of the roots, to every root:

* a path component BELOW the root that starts with ``.`` is refused. A session
  rooted at ``~`` would otherwise expose ``~/.ssh`` and ``~/.aws`` to the
  extension allowlist; dot-directories are where credentials live and nothing
  the agent is asked to *preview* sits in one. A root that is itself under a
  dot-directory (``~/.local-operator/sessions``) is unaffected: only the part
  after the root is judged.
* the filesystem root is never a root, however it got configured: it would turn
  the allowlist back into "anywhere on disk".

KNOWN COST OF A ROOT ALLOWLIST: the composer's attachment thumbnails ask this
route for files the user picked or dropped from ANYWHERE (``~/Downloads``). One
outside every root above now gets a 403 and the thumbnail is broken (the
attachment itself is unaffected; it is read server-side by the chat route). The
operator's remedy today is ``static.roots``; the durable one is for the UI to
read those bytes over IPC the way the canvas image viewer already does.

No FastAPI import here: this raises :class:`StaticPathDenied` and the route turns
it into an ``HTTPException``, which keeps the module cheap to import from the
settings registry and its tests.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger("local_operator.server.utils.static_roots")

#: ``os.pathsep``-separated extra roots. Read here and nowhere else, so the
#: policy has one reader (the desktop plane's env vars follow the same rule).
ROOTS_ENV = "LOCAL_OPERATOR_STATIC_ROOTS"

#: The ``config.yml`` path of the configured roots (``static.roots`` in the
#: settings registry). Nested, so read with ``get_nested_value``.
CONFIG_PATH = ("static", "roots")

#: What ``static.roots`` falls back to when unset: nothing extra. The registry's
#: default is pinned to this by ``tests/unit/test_settings_io.py``.
DEFAULT_CONFIGURED_ROOTS: list[str] = []

#: How long the live-session roots are reused. One image grid fires a request per
#: thumbnail and each would otherwise re-read every discovery record. Short on
#: purpose: a session started a moment ago must become previewable quickly.
LIVE_ROOTS_TTL_S = 5.0

#: The CSP for an HTML preview. Deliberately NOT ``default-src 'none'`` alone: a
#: generated page is an inline-script document that pulls a library from a CDN,
#: and denying that breaks the thing the preview is for. So this mirrors the
#: policy the app adds to the same response (UI ``PREVIEW_CSP``, PR #931) -- the
#: two are enforced together, so this one is the floor for any other client.
#: ``connect-src https:`` is the load-bearing line: it keeps a previewed page
#: from reaching this daemon or any other local service over plain http.
#: ``sandbox allow-scripts`` makes a DIRECT navigation to the URL (a hostile
#: page opening it in a window) run in an opaque origin too, not just the app's
#: iframe, which already carries its own sandbox attribute.
HTML_CSP = "; ".join(
    [
        "default-src 'none'",
        "script-src 'unsafe-inline' 'unsafe-eval' https:",
        "style-src 'unsafe-inline' https:",
        "img-src data: blob: https:",
        "font-src data: https:",
        "media-src data: blob: https:",
        "connect-src https:",
        "worker-src blob:",
        "form-action 'none'",
        "base-uri 'none'",
        "sandbox allow-scripts",
    ]
)

#: The CSP for image/audio/video bytes. These are embedded by element and never
#: need to run anything; the policy matters for the one document-shaped member,
#: ``image/svg+xml``, navigated to directly. ``style-src`` lets such an SVG keep
#: its inline styling when it is only *viewed*.
MEDIA_CSP = "default-src 'none'; style-src 'unsafe-inline'; sandbox"

#: Always allowed to embed a static response: the daemon's own origin, and a
#: ``file:`` document -- the SHIPPED renderer is loaded with ``loadFile`` and so
#: runs at ``file://``, whose origin is the opaque ``"null"`` that the desktop
#: allowlist deliberately never admits, so it cannot be named by origin.
_ALWAYS_FRAMING = ("'self'", "file:")

#: The dev renderer's origin (``ELECTRON_RENDERER_URL``, a localhost dev server on
#: a port that varies) when NO origin allowlist is installed. Framing is not
#: reading: the same-origin policy still stops the framing page reading the
#: response, so this admits clickjacking-class exposure on loopback only.
_LOOPBACK_DEV_FRAMING = ("http://localhost:*", "http://127.0.0.1:*")


class StaticPathDenied(Exception):
    """A path the static routes will not serve; ``status`` is the HTTP code."""

    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


@dataclass(frozen=True)
class ServedRoots:
    """The realpath roots a request may be served from."""

    roots: tuple[Path, ...]


def _realpath(value: str | os.PathLike[str]) -> Path | None:
    """``value`` as an absolute realpath, or ``None`` when it cannot be a root.

    Refused: relative spellings (resolved against the DAEMON's cwd they would
    name a directory nobody configured), anything unresolvable, and the
    filesystem root (see the module docstring).
    """
    try:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            return None
        real = candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if real == Path(real.anchor):
        logger.warning("static root %s is the filesystem root and is ignored", value)
        return None
    return real


def configured_root_strings(config_values: Mapping[str, Any] | None) -> list[str]:
    """The operator's explicit roots: ``static.roots`` then the environment's."""
    configured: list[str] = []
    node: Any = config_values or {}
    for part in CONFIG_PATH:
        node = node.get(part) if isinstance(node, Mapping) else None
    if isinstance(node, (list, tuple)):
        configured.extend(item for item in node if isinstance(item, str) and item.strip())
    configured.extend(item for item in os.environ.get(ROOTS_ENV, "").split(os.pathsep) if item)
    return configured


_live_cache: dict[Path, tuple[float, tuple[Path, ...]]] = {}


def clear_live_cache() -> None:
    """Forget the memoised live-session roots (tests; a config-dir switch)."""
    _live_cache.clear()


def _live_session_roots(config_dir: Path) -> tuple[Path, ...]:
    """Working directories of the sessions that are running now.

    ``reap=False`` and ``check_zombie=False``: this is a READER on a request path
    and must not move another process's discovery records aside or fork a
    process-table probe per thumbnail. A stale record (dead pid) is skipped.
    """
    now = time.monotonic()
    hit = _live_cache.get(config_dir)
    if hit is not None and now - hit[0] < LIVE_ROOTS_TTL_S:
        return hit[1]
    roots: list[Path] = []
    try:
        from local_operator.session.runtime import registry

        for record, state in registry.scan(config_dir, reap=False, check_zombie=False):
            if state == "stale":
                continue
            real = _realpath(str(getattr(record, "cwd", "") or ""))
            if real is not None:
                roots.append(real)
    except Exception:  # a broken discovery dir must cost the live arm, not the route
        logger.warning("could not read live session records for static roots", exc_info=True)
    result = tuple(dict.fromkeys(roots))
    _live_cache[config_dir] = (now, result)
    return result


def build_roots(
    config_dir: Path,
    config_values: Mapping[str, Any] | None = None,
    agent_cwds: Iterable[str] = (),
) -> ServedRoots:
    """Assemble the roots for one request. See the module docstring for the why."""
    from local_operator.paths import agent_home_dir

    candidates: list[Path | None] = [
        _realpath(agent_home_dir()),
        _realpath(config_dir / "sessions"),
        _realpath(config_dir / "uploads"),
    ]
    candidates.extend(_realpath(item) for item in configured_root_strings(config_values))
    candidates.extend(_realpath(item) for item in agent_cwds)
    candidates.extend(_live_session_roots(config_dir))
    return ServedRoots(roots=tuple(dict.fromkeys(root for root in candidates if root is not None)))


def _containing_root(path: Path, roots: Sequence[Path]) -> Path | None:
    """The first root that contains ``path``, or ``None``."""
    for root in roots:
        if path.is_relative_to(root):
            return root
    return None


def resolve_servable(raw: str, roots: ServedRoots) -> Path:
    """The realpath of ``raw`` if it may be served, else :class:`StaticPathDenied`.

    Order is the security property: every refusal that depends on the filesystem
    (404 missing, 400 not a file) comes AFTER the root check, so a path outside
    the roots gets the same 403 whether or not it exists -- the route is not an
    existence oracle for the rest of the disk.
    """
    if not raw or "\x00" in raw:
        raise StaticPathDenied(400, "Invalid path.")
    # Rejected on the RAW spelling, before resolution, so a traversal attempt is
    # a clear refusal rather than something that happens to land back inside.
    if ".." in Path(raw).parts:
        raise StaticPathDenied(403, "Path may not contain '..'.")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        raise StaticPathDenied(400, "Path must be absolute.")
    try:
        real = candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        # RuntimeError is a symlink loop. Same answer as any unservable path.
        raise StaticPathDenied(400, "Path could not be resolved.") from None

    root = _containing_root(real, roots.roots)
    if root is None:
        raise StaticPathDenied(403, "Path is outside the directories this server may serve.")
    # Dot-directories below the root (``~/.ssh`` under a session rooted at ``~``).
    if any(part.startswith(".") for part in real.relative_to(root).parts):
        raise StaticPathDenied(403, "Hidden paths may not be served.")

    if not real.exists():
        raise StaticPathDenied(404, f"File not found: {raw}")
    # ``real`` is already the resolved target, so this refuses a directory and
    # any FIFO, socket or device node (which would hang a read).
    if not real.is_file():
        raise StaticPathDenied(400, f"Not a file: {raw}")
    if not os.access(real, os.R_OK):
        raise StaticPathDenied(403, f"File not accessible: {raw}")
    return real


def frame_ancestors(app_origins: Iterable[str]) -> str:
    """The ``frame-ancestors`` directive: who may embed a static response.

    The app only. ``app_origins`` is the desktop plane's admitted origin
    allowlist (``desktop_posture().origins``); with none admitted the dev
    renderer's loopback origins stand in (see :data:`_LOOPBACK_DEV_FRAMING`).
    """
    origins = sorted(app_origins)
    return "frame-ancestors " + " ".join([*_ALWAYS_FRAMING, *(origins or _LOOPBACK_DEV_FRAMING)])


def is_static_path(path: str) -> bool:
    """Whether a request path is one of the static file routes."""
    return path == "/v1/static" or path.startswith("/v1/static/")


def response_policy(path: str, app_origins: Iterable[str]) -> dict[str, str]:
    """The security headers every ``/v1/static/*`` response carries.

    Applied to ERROR responses too (a JSON 403 needs ``nosniff`` as much as a
    200), which is why the app middleware calls it by path prefix instead of each
    handler remembering to attach it.
    """
    csp = HTML_CSP if path.rstrip("/") == "/v1/static/html" else MEDIA_CSP
    return {
        "Content-Security-Policy": f"{csp}; {frame_ancestors(app_origins)}",
        "X-Content-Type-Options": "nosniff",
    }
