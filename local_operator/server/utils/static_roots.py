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

THE ROUTE'S BOUND IS THE CONNECTION, NOT A DIRECTORY LIST (provenance). Core
#2134 (v0.68.23) closed the route's side of it with a served-root allowlist.
Under the operator's direction of 2026-10-10 (quoted in
``docs/design/file-serving-and-surface-convergence.md``) the allowlist's
*rationale* was replaced by the loopback condition, because the clamp refused
exactly the paths the product exists to preview (``~/Downloads`` picks, session
scratchpads under ``~/.local-operator``, agent output anywhere). The design is
``docs/design/file-serving-and-surface-convergence.md`` §3 (D1-D3); this module
keeps the #2134 hardening, which becomes MORE load-bearing without the roots.

This module carries both halves that need no UI change:

1. **The file predicate** (:func:`resolve_servable`), in two postures chosen per
   CONNECTION, never per configuration -- **general** (unclamped; served only on
   loopback-accepted connections) and **rooted** (the v0.68.23 clamp, kept as
   the fail-closed fallback). The per-connection choice lives in
   :func:`connection_mode` / :func:`is_loopback_host`; the bind-refusal half
   lives in ``cli.py``.
2. **A response policy** (:func:`response_policy`): ``nosniff``, a CSP, and a
   ``frame-ancestors`` naming only the app. The CORS half lives in the app
   middleware that calls this module (``server/app.py``).

THE TWO POSTURES, and what enforces them (RFC §3.1-§3.2):

* **General (loopback) mode.** When the connection's accepted local address is
  loopback -- ``request.scope["server"]``, which uvicorn sets from the socket
  the kernel accepted on, so no header or path can influence it -- ``path`` may
  be any absolute path that resolves to a readable regular file; the per-route
  mime allowlists are unchanged. No root list, and no dot-component rule: the
  rule was an inner bound of the root list, and under no roots it cannot be a
  boundary while it refuses session-scratchpad paths. Measured on the pinned
  uvicorn (0.54, macOS): a loopback client reports a loopback address even to a
  wildcard bind, and a LAN client reports the LAN address -- so the condition
  holds even for a bind that leaked through a future embedding path.
* **Rooted mode (the fallback).** Every other connection -- non-loopback,
  unknown, or a missing ``scope["server"]`` (fail-closed) -- gets the v0.68.23
  behaviour: the roots below, the dot rule, the live-session arm.

The bind is refused at the source as well: ``lop serve`` refuses every
non-loopback bind (``0.0.0.0``, ``::``, LAN addresses, non-loopback-resolving
names -- exit 1, no override flag) and checks a ``--listener-fd`` adopt too;
the per-connection gate exists because a future composition path that never
sees the CLI flag (a bare ``uvicorn local_operator.server.app:app``, an
unannounced embed) must still not serve unclamped to a non-loopback connection.

An SSH port-forward or a local tunnel IS unclamped: the connection is
loopback-accepted, which is exactly the condition -- and whoever holds the
forward reads exactly what this account can read. That is the operator's
"explicit tunnels" path, stated rather than hidden (RFC §3.4).

WHAT IS STILL OPEN, stated so nobody reads this as more than it is: the routes
remain unauthenticated. The UI loads them through ``<img>``/``<video>``/
``<iframe src>``, which cannot carry an ``Authorization`` header, so the bearer
cannot simply be required; the follow-up is a short-lived signed query token
minted by an authenticated endpoint and verified here, which needs the UI lane
to request it (RFC §4, Phase 2). Until then:

* a hostile local PAGE can still *trigger* a read of anything a loopback
  connection serves -- and, since an ``<img>`` reports ``load``/``error`` and
  ``naturalWidth``/``naturalHeight`` back to the page, still *observe* which
  image paths (of allowlisted media types) exist, decode and how large they
  are -- across the whole disk now, not a root list. Bytes cannot be read
  cross-origin (the CORS grant is stripped) and the Host check is what keeps a
  rebinding page from becoming same-origin (§3.4). Accepted residual, RFC §7e;

* TOCTOU: :func:`resolve_servable` validates a realpath and the handlers then
  open BY PATH, so a symlink swapped between the check and the open is followed
  -- not a narrow window: a 30k-request swap run (security round 1, 2026-10-10)
  won ~1 request in 4 (7,343 served the outside file), while swapping an
  intermediate symlink never won (0 of 15,872, the handler opening the
  already-resolved path). In general mode the swap target is the same class of
  file the requester may ask for directly, so what remains there is robustness
  (a swap to a FIFO/device can hang a worker); in rooted mode it is the
  shared-root boundary case as before. Fix shape unchanged (``O_NOFOLLOW`` open
  + ``fstat`` + stream-from-fd), scheduled Phase 3 (RFC §7b), not blocking;
* hardlinks: a hardlink is indistinguishable from a file that lives where it is,
  and no realpath test can tell. Moot for confidentiality under the no-roots
  rationale (any same-user process reads the target directly; a page cannot
  create links) -- noted, no work (RFC §7c);
* Host validation covers DNS NAMES only (:func:`host_is_acceptable`): an IP
  literal Host is admitted, since a rebinding page cannot present one. It is also
  scoped to this module's routes -- :func:`is_static_path` is what the middleware
  gates on, so the rest of the legacy surface (``/v1/agents``, ``/health``) still
  answers under a rebinding Host (security round 1 measured it; out of scope).

THE ROOTS (ROOTED MODE ONLY), and why each is there (all compared as realpaths):

* the agent home (``paths.agent_home_dir()``, ``~/local-operator-home``): the
  default workspace, where agents are told to put what they produce;
* ``<config>/sessions``: each session's scratchpad lives in its session
  directory, and tools write screenshots and renders there;
* ``<config>/uploads``: where a decoded chat attachment lands;
* the working directory of every non-stale live session (``run/mobile``
  discovery records): the canvas previews "the file the agent just wrote",
  wherever the user pointed the session. This is the IMPLICIT arm, and it is
  bounded: a cwd that equals or CONTAINS ``$HOME`` yields no root (review R1 --
  34 live sessions on the reference host were started in ``~``, which would have
  made ``~/Pictures``, ``~/Downloads`` and ``~/Desktop`` servable and the list
  no list at all). Why a session's cwd is a safe input and an agent's is not:
  a record is written by the session process into a ``0700`` directory, and the
  only HTTP route that names a cwd (``POST /v1/desktop/sessions``) is behind the
  desktop bearer, so no ungated caller can move it. A registered agent's
  ``current_working_directory`` IS writable by the ungated ``PATCH /v1/agents/<id>``
  (review R2: one cross-origin PATCH widened the list and it persisted to
  ``agent.yml``), so that arm was removed rather than guarded;
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
* a root that is an ANCESTOR of ``$HOME`` (``/``, ``/Users``, ``~/..``, and on
  macOS the ``/System/Volumes/Data`` firmlink of ``/``) is never a root, however
  it got configured -- ``static.roots`` and the environment variable go through
  the same :func:`root_refusal` -- because it would turn the allowlist back into
  "anywhere on disk". ``$HOME`` itself is an operator choice and is accepted
  ONLY from the explicit arms: widening past the built-ins is the operator's
  opt-in through ``static.roots`` / ``LOCAL_OPERATOR_STATIC_ROOTS``, never
  something a session's cwd can do for them.

WHAT THE LOOPBACK CONDITION CHANGED FOR PREVIEWS. The clamp's user-facing cost
(formerly: every preview of a file outside every root -- a ``~/Downloads``
pick, agent output in ``/tmp``, a file on another volume -- was a 403) is
retired where it bit: a loopback-accepted connection, which is every
first-party surface today, serves those files. The cost text survives only as
the ROOTED posture's story, where the 403 body still names ``static.roots`` as
the remedy.

macOS TCC (documented, not a mystery): reads of ``~/Downloads``,
``~/Documents``, ``~/Pictures`` and the like can still be refused by OS privacy
controls depending on the daemon's launch context; the errno classification in
:func:`resolve_servable` maps that to the uniform 403 rather than a crash.

No FastAPI import here: this raises :class:`StaticPathDenied` and the route turns
it into an ``HTTPException``, which keeps the module cheap to import from the
settings registry and its tests.
"""

from __future__ import annotations

import errno
import ipaddress
import logging
import os
import stat
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


#: The ONE body for "not inside a served root" in ROOTED (fallback) mode,
#: shared by every route and every reason there (see :func:`resolve_servable`).
#: It names the remedy -- the rooted posture's 403 is what a developer debugging
#: an embedded/off-loopback server sees, and the answer they need is the roots
#: list, not a grep.
OUTSIDE_ROOTS_DETAIL = (
    "Path is outside the directories this server may serve. To allow it, add the "
    "directory to static.roots (Settings > File previews, or config.yml) or to "
    f"{ROOTS_ENV}."
)

#: The ONE body for the uniform 403s in GENERAL (loopback) mode: a resolve
#: failure, a stat failure that is not a clean miss. It names no reason and no
#: remedy on purpose -- there is no root list to add to, and the uniform answer
#: is the property (no refusal reason becomes a signal of its own; security S-4).
UNSERVABLE_DETAIL = "Path could not be served."

#: The stat errnos that mean "this path is not there", answered 404 inside a root:
#: missing, a component that is not a directory, and a symlink loop. Everything
#: else an ``os.stat`` can raise (``ENAMETOOLONG`` on an over-long component,
#: ``EACCES`` on a directory the daemon may not traverse) is the uniform 403: the
#: caller learns nothing about the disk, and no such path can be a clean answer.
_MISSING_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.ELOOP})


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


def _home() -> Path | None:
    """``$HOME`` as a realpath, or ``None`` when it cannot be determined."""
    try:
        return Path.home().resolve()
    except (OSError, RuntimeError):
        return None


def _contains(root: Path, home: Path) -> bool:
    """Whether ``home`` is inside ``root``, by name OR by identity.

    The name test alone misses a firmlink: macOS exposes the whole disk a second
    time at ``/System/Volumes/Data``, whose realpath is itself, so it does not
    lexically contain ``/Users/me`` although ``/System/Volumes/Data/Users/me`` IS
    that directory (review R6). So the tail of ``home`` is also looked up under
    ``root`` and compared with ``samefile``.
    """
    if home.is_relative_to(root):
        return True
    for start in range(1, len(home.parts)):
        probe = root.joinpath(*home.parts[start:])
        try:
            if probe.samefile(home):
                return True
        except OSError:
            continue
    return False


def _same_directory(left: Path, right: Path) -> bool:
    """Whether two paths are the same directory: by identity, falling back to the name.

    ``samefile`` is the identity test the rest of this module uses
    (:func:`_contains`) -- a string comparison is not the same question: macOS
    ``resolve()`` canonicalises neither case nor the ``/System/Volumes/Data``
    firmlink, so ``/Users/me``, ``/users/me`` and
    ``/System/Volumes/Data/Users/me`` are one directory spelled three ways (review
    R2-1, security S-1). The name fallback keeps a path that does not exist
    answerable -- a missing entry is nobody's home directory.
    """
    try:
        return left.samefile(right)
    except OSError:
        return left == right


def root_refusal(real: Path, *, implicit: bool = False) -> str | None:
    """Why ``real`` (an absolute realpath) may not be a served root, else ``None``.

    THE ONE PREDICATE behind every root: the settings validator, the environment
    variable, the built-ins and the live-session arm all call it, so there is no
    second spelling to drift (review R1/R6).

    * A root that CONTAINS ``$HOME`` is refused outright -- ``/``, ``/Users``,
      ``~/..``, and ``/System/Volumes/Data`` on macOS, which is ``/`` by another
      name. Matching on the filesystem root's NAME alone missed all but one.
    * ``$HOME`` ITSELF is decided by identity, not by the spelling (review R2-1,
      security S-1): ``/users/me`` and ``/System/Volumes/Data/Users/me`` are the
      same directory as ``$HOME`` and compare unequal to it, so a live-session
      record whose cwd was written that way used to pass this predicate -- as
      neither equal to nor an ancestor of ``home`` -- and serve ``~/Pictures``.
    * ``implicit=True`` is the stricter rule for a root nobody typed (a live
      session's cwd): it must not BE ``$HOME`` either. Sessions started in
      ``~`` are the normal case, and honouring them would serve ``~/Pictures``,
      ``~/Downloads`` and ``~/Desktop``. Serving ``~`` is the operator's explicit
      opt-in through ``static.roots``, whatever spelling they typed it in.
    """
    if real == Path(real.anchor):
        return "the filesystem root cannot be a served root"
    home = _home()
    if home is None:
        return None
    if _same_directory(real, home):
        if implicit:
            return "your home directory is only served when configured explicitly"
        return None
    if not _contains(real, home):
        return None
    return f"{real} contains your home directory, so it would serve nearly everything"


def _realpath(value: str | os.PathLike[str], *, implicit: bool = False) -> Path | None:
    """``value`` as an absolute realpath, or ``None`` when it cannot be a root.

    Refused: relative spellings (resolved against the DAEMON's cwd they would
    name a directory nobody configured), anything unresolvable, and whatever
    :func:`root_refusal` rejects.
    """
    try:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            return None
        real = candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    reason = root_refusal(real, implicit=implicit)
    if reason is not None:
        # An implicit refusal is routine (most sessions run in ~) and says so quietly.
        log = logger.debug if implicit else logger.warning
        log("static root %s is ignored: %s", value, reason)
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
            real = _realpath(str(getattr(record, "cwd", "") or ""), implicit=True)
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
) -> ServedRoots:
    """Assemble the roots for one request. See the module docstring for the why.

    There is deliberately no registered-agent arm: ``current_working_directory``
    is writable through the ungated ``PATCH /v1/agents/<id>`` (review R2).
    """
    from local_operator.paths import agent_home_dir

    candidates: list[Path | None] = [
        _realpath(agent_home_dir()),
        _realpath(config_dir / "sessions"),
        _realpath(config_dir / "uploads"),
    ]
    candidates.extend(_realpath(item) for item in configured_root_strings(config_values))
    candidates.extend(_live_session_roots(config_dir))
    return ServedRoots(roots=tuple(dict.fromkeys(root for root in candidates if root is not None)))


def _containing_root(path: Path, roots: Sequence[Path]) -> Path | None:
    """The prefix of ``path`` that is one of ``roots``, or ``None``.

    Lexical first (the cheap, common answer). On a miss, fall back to comparing
    each existing ancestor of ``path`` with each root by identity (``samefile``):
    on a case-insensitive filesystem (default APFS) ``/Users/x/Workspace`` and a
    root spelled ``/Users/x/workspace`` are one directory, and the byte-wise test
    would give a false 403 for a file that is plainly in the workspace (review
    R10). Identity is the right test and cannot admit anything outside a root:
    ``path`` is already symlink-resolved, so an ancestor that IS a root puts the
    file inside it.
    """
    for root in roots:
        if path.is_relative_to(root):
            return root
    root_ids = set()
    for root in roots:
        try:
            info = root.stat()
        except OSError:
            continue
        root_ids.add((info.st_dev, info.st_ino))
    for ancestor in path.parents:
        try:
            info = ancestor.stat()
        except OSError:
            continue  # a missing parent (a 404 later, but 403 first when outside)
        if (info.st_dev, info.st_ino) in root_ids:
            return ancestor
    return None


def resolve_servable(raw: str, roots: ServedRoots, *, general: bool) -> Path:
    """The realpath of ``raw`` if it may be served, else :class:`StaticPathDenied`.

    ``general=True`` is the loopback posture (RFC §3.2): any absolute path that
    resolves to a readable regular file -- no root check, no dot rule; the
    per-route mime allowlists are unchanged. ``general=False`` is the rooted
    fallback: the roots, the dot rule and the live-session arm apply.

    Order is the security property, per posture. Rooted: every refusal that
    depends on the filesystem (404 missing, 400 not a file) comes AFTER the root
    check, so a path outside the roots gets the same 403 whether or not it
    exists -- the route is not an existence oracle for the rest of the disk.
    General: there is no root check, and the uniform 403
    (:data:`UNSERVABLE_DETAIL`) still speaks for every refusal reason; the raw
    ``..`` scan comes first in both postures and rejects before any resolution.
    """
    if not raw or "\x00" in raw:
        raise StaticPathDenied(400, "Invalid path.")
    # Rejected on the RAW spelling, before resolution, so a traversal attempt is
    # a clear refusal rather than something that happens to land back inside.
    if ".." in Path(raw).parts:
        raise StaticPathDenied(403, "Path may not contain '..'.")
    # ONE refusal for everything that cannot be placed, whatever the reason: an
    # unknown ``~user`` (``expanduser`` raises RuntimeError and would otherwise
    # be a 500 that also tells a caller which local users exist), a symlink loop
    # (which would otherwise be a distinct 400, a loop-existence oracle for the
    # rest of the disk), any other resolve failure (e.g. a component over
    # NAME_MAX, reproduced as a 500 in security round 1). Rooted mode answers
    # with the remedy-shaped body; general mode with the neutral one.
    unplaced = UNSERVABLE_DETAIL if general else OUTSIDE_ROOTS_DETAIL
    try:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            raise StaticPathDenied(400, "Path must be absolute.")
        real = candidate.resolve()
    except (OSError, RuntimeError, ValueError):
        raise StaticPathDenied(403, unplaced) from None

    if not general:
        root = _containing_root(real, roots.roots)
        if root is None:
            raise StaticPathDenied(403, OUTSIDE_ROOTS_DETAIL)
        # Dot-directories below the root (``~/.ssh`` under a session rooted at
        # ``~``). Rooted mode only: general mode dropped the rule -- without
        # roots it cannot be a boundary, and it refuses the session-scratchpad
        # previews the product exists for (RFC §3.3).
        if any(part.startswith(".") for part in real.relative_to(root).parts):
            raise StaticPathDenied(403, "Hidden paths may not be served.")

    # ONE ``os.stat``, not ``exists()``/``is_file()``: those SWALLOW some OS
    # errors and re-raise others, and which ones changed between Python 3.12 and
    # 3.14 (``ENAMETOOLONG`` is ignored from 3.13 on only). Uncaught, a refusal
    # here escaped as the handler's generic 500 -- the one answer that was not a
    # clean 4xx and so a signal in its own right (security S-4). Classifying the
    # errno answers the same way on every interpreter.
    try:
        info = os.stat(real)
    except OSError as exc:
        if exc.errno in _MISSING_ERRNOS:
            raise StaticPathDenied(404, f"File not found: {raw}") from None
        raise StaticPathDenied(403, unplaced) from None
    # ``real`` is already the resolved target, so this refuses a directory and
    # any FIFO, socket or device node (which would hang a read).
    if not stat.S_ISREG(info.st_mode):
        raise StaticPathDenied(400, f"Not a file: {raw}")
    if not os.access(real, os.R_OK):
        raise StaticPathDenied(403, f"File not accessible: {raw}")
    return real


def is_loopback_host(host: str | None) -> bool:
    """Whether ``host`` is a loopback address: ``127.0.0.0/8``, ``::1``, ``localhost``.

    The RFC §3.2 condition, one spelling for both the per-connection gate and
    the CLI's bind policy (``cli.py`` imports this). ``::ffff:127.0.0.1`` -- an
    IPv4-mapped address a v6 listener reports for a v4 loopback client -- is
    unwrapped to its v4 form first, because it IS loopback by identity; anything
    that does not parse as an address and is not ``localhost`` is not loopback.
    """
    if host is None:
        return False
    name = host.strip().lower()
    if name == "localhost":
        return True
    try:
        address = ipaddress.ip_address(name)
    except ValueError:
        return False
    if isinstance(address, ipaddress.IPv6Address):
        mapped = address.ipv4_mapped
        if mapped is not None:
            address = mapped
    return address.is_loopback


def connection_mode(host: str | None) -> str:
    """``"general"`` when the accepted connection's local address is loopback, else ``"rooted"``.

    The caller passes the host half of ``request.scope["server"]`` (or ``None``
    when it is missing/foreign). Everything that is not provably loopback -- a
    LAN address, a wildcard, an unknown or absent value -- is rooted: the
    fail-closed half of §3.2, so the unclamped predicate is only ever reached
    on a connection the kernel accepted on a loopback address. Measured on the
    pinned uvicorn (0.54, macOS): a wildcard bind still reports the
    PER-CONNECTION local address, loopback for a loopback client and the LAN
    address for a LAN client.
    """
    return "general" if is_loopback_host(host) else "rooted"


def frame_ancestors(app_origins: Iterable[str]) -> str:
    """The ``frame-ancestors`` directive: who may embed a static response.

    The app only. ``app_origins`` is the desktop plane's admitted origin
    allowlist (``desktop_posture().origins``); with none admitted the dev
    renderer's loopback origins stand in (see :data:`_LOOPBACK_DEV_FRAMING`).
    """
    origins = sorted(app_origins)
    return "frame-ancestors " + " ".join([*_ALWAYS_FRAMING, *(origins or _LOOPBACK_DEV_FRAMING)])


#: Bind addresses that mean "every interface": the daemon is then reachable under
#: names this process cannot enumerate, so a Host check has nothing to compare to
#: and a DNS name fails closed there (see :func:`host_is_acceptable`).
_WILDCARD_BINDS = frozenset({"0.0.0.0", "::", ""})


def _host_name(host_header: str) -> str:
    """The bare, lower-cased host of a ``Host`` header (port and IPv6 brackets stripped)."""
    value = host_header.strip().lower()
    if value.startswith("["):  # [::1]:8080
        return value[1:].split("]", 1)[0]
    return value.rsplit(":", 1)[0] if value.count(":") == 1 else value


def host_is_acceptable(host_header: str | None, bound_host: str | None) -> bool:
    """Whether a request's ``Host`` may reach a static route (DNS-rebinding guard).

    WHY. Dropping the CORS grant stops a foreign page READING a response, but a
    DNS-rebinding page is not foreign: it re-points ``rebind.attacker.test`` at
    ``127.0.0.1`` and is then same-origin with this daemon, so it needs no grant
    at all (review R8). The one thing it cannot change is the ``Host`` it sends,
    which is the attacker's name. So the route accepts only:

    * an IP literal (a rebinding page cannot present one: no DNS is involved);
    * ``localhost`` -- what the dev renderer and a hand-typed URL use;
    * the host the daemon was told to bind, when that is a specific name (an
      operator who bound ``--host mac.lan`` and reaches it as such).

    Refused before any of that is even parsed: a ``Host`` carrying userinfo
    (``localhost:80@attacker.test``, ``[::1]@attacker.test``) or present but empty
    (security S-3). Neither is an authority a browser can send, and the split on
    ``:`` would otherwise read the attacker's name out of such a value.

    No header at all is accepted: rebinding always carries a name. A wildcard or
    empty announced host (``--host 0.0.0.0`` -- no longer producible through the
    CLI, which refuses every non-loopback bind; kept fail-closed for an
    embedding path) has nothing to compare a DNS name against, so a name fails
    closed: only an IP literal or ``localhost`` passes (the wildcard-bind bypass
    that admitted every name here is deleted; RFC §3.4). The app's own renderer
    dials ``http://127.0.0.1:<port>`` (backend-service.ts), which passes.
    """
    if host_header is None:
        return True
    authority = host_header.strip()
    if not authority or "@" in authority:
        return False
    name = _host_name(authority)
    try:
        ipaddress.ip_address(name)
        return True
    except ValueError:
        pass
    if name == "localhost":
        return True
    if not bound_host or bound_host.strip().lower() in _WILDCARD_BINDS:
        # Nothing to compare a DNS name against: fail closed. (This branch is
        # the fail-closed replacement for the wildcard-bind bypass that used to
        # return True here -- RFC §3.4.)
        return False
    return name == _host_name(bound_host)


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
