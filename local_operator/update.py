"""PyPI-facing updater for the installed ``local-operator`` distribution.

WHY THIS EXISTS
---------------
End users install from PyPI (``uv tool``, pipx, or pip). They need one
command that upgrades whatever they actually have, without being pointed at
``lop-update`` — that script archives local git ``main`` into the uv-tool
env, which is the opposite audience.

"Latest" is always ``https://pypi.org/pypi/local-operator/json`` →
``info.version``, compared to ``importlib.metadata.version("local-operator")``.
That is the same source the splash version row and ``lop --version`` already
use. A second version channel (git tags, ``lop-update``, a pin file) would
diverge from what the running process reports.

WHY THE CACHE
-------------
The splash paints immediately and a background probe fills an optional ``!``
row. Hitting PyPI on every launch would stall a flaky network under the first
frame and turn a quiet check into a toast. Modelled on
``model/catalogue.py``: fresh cache skips the network, a stale copy is kept
when the fetch fails, and a total miss is ``None`` — never an error the user
has to dismiss. The probe is news, not a prerequisite, so it must not compete
with the credentials-fallback line.

``/update`` and ``lop update`` bypass the TTL (they need a live answer) but
still rewrite the cache so the next splash is free.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import (
    PackageNotFoundError,
    distribution,
    distributions,
    version,
)
from pathlib import Path
from typing import Any, Callable, Literal, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

from local_operator.interpreter import SAFE_PATH_FLAG

logger = logging.getLogger(__name__)

#: Same cache root the model catalogue uses, so there is one place to clear.
_CACHE_DIR = Path("~/.local-operator/cache")
_CACHE_NAME = "pypi-local-operator.json"

#: Six hours. Shorter re-fetches on every other launch for a number that
#: moves on the order of days; a day would hide a release the user just
#: saw announced. The splash worker runs once per process, so the TTL is
#: for the *next* launch, not for keystrokes.
TTL_S = 6 * 60 * 60

PYPI_JSON_URL = "https://pypi.org/pypi/local-operator/json"

#: Short enough that a hung PyPI cannot stall a splash worker across the
#: TUI suite; long enough for a slow but living mirror.
_FETCH_TIMEOUT_S = 5.0


class InstallKind(str, Enum):
    UV_TOOL = "uv-tool"
    PIPX = "pipx"
    PIP = "pip"
    EDITABLE = "editable"
    UNKNOWN = "unknown"


class UpdateError(Exception):
    """Refused or failed upgrade; the message is what the CLI/TUI print."""


#: Bound the child so a hung ``launchctl kickstart`` cannot stall the
#: successful-upgrade path. The daemon itself is not waited on.
_MOBILE_RESTART_TIMEOUT_S = 30.0

#: Bound on the child that repairs the OTHER supervised daemons. Larger than the
#: mobile bounds because it is one child doing four plists, each with a
#: bootout/bootstrap pair; still bounded so a hung ``launchctl`` cannot stall a
#: successful upgrade.
_DAEMON_REFRESH_TIMEOUT_S = 60.0

#: The supervised daemons a combined release must leave branded, as the plist
#: filenames that prove each one is installed. Labels are repeated here rather
#: than imported for the same reason ``_mobile_plist_path`` is: importing
#: ``mobile.install`` pulls Starlette into the updater, and this probe runs in
#: the CLI and in the TUI's update worker.
_DAEMON_PLIST_LABELS = (
    "com.local-operator.mobile",
    "com.local-operator.browser",
    "com.local-operator.tunnel",
    "com.local-operator.wakes",
)

#: Default loopback probe used only for the unsupervised warning. Must
#: match ``mobile.daemon.DEFAULT_PORT``; do not import that module here.
_MOBILE_HEALTHZ = "http://127.0.0.1:4098/healthz"

MobileRefreshKind = Literal["skipped", "restarted", "failed", "unsupervised"]


@dataclass(frozen=True)
class MobileRefresh:
    """Outcome of the post-upgrade LaunchAgent bounce. Never an exception."""

    kind: MobileRefreshKind
    error: str = ""


@dataclass(frozen=True)
class DaemonRefresh:
    """One daemon group's outcome, already rendered for the upgrade summary.

    Both callers (the CLI's ``lop update`` and the TUI's ``/update``) print the
    same sentences from this, so one outcome cannot be described two ways. The
    lines are rendered where the outcome is KNOWN — the mobile half from
    :class:`MobileRefresh`, the rest from the new wheel's own report — because
    only that side can tell a repaired plist from an untouched one.

    ``name`` is for the reader of a failure, not for printing: every sentence
    already names its daemon.
    """

    name: str
    lines: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class VersionCheck:
    installed: str
    latest: str | None
    behind: bool


@dataclass(frozen=True)
class BuildStamp:
    """One comparable build token: distribution version plus the git ref of
    the install, when ``lop-update`` recorded one.

    Two long-lived populations coexist on a developer host — viewer TUIs and
    the runtime processes they spawn — while ``lop-update`` replaces the
    on-disk install under them, often several times a day. Comparing what a
    process LOADED with what is on disk now is how that skew becomes visible
    instead of silent.

    The ref is the primary key and the version the fallback, because this
    host's common drift is a same-version rebuild: ``lop-update`` builds from
    ``main`` while ``pyproject.toml`` still names the last released version,
    so two genuinely different builds share one version string and only the
    recorded commit tells them apart. Installs with no ``.lop-source`` (PyPI
    wheels, pipx, editable checkouts) carry ``""`` and compare on version
    alone — dev-tree skew is out of scope by design.

    Frozen so a snapshot taken at process start cannot be mutated by the code
    that later compares against it.
    """

    version: str
    source_ref: str = ""

    def label(self) -> str:
        """How this build is named in a user-facing notice.

        ``0.49.0`` when there is no ref to disambiguate, ``0.49.0@4d3ce1d``
        when there is — short-form because a notice line is read, not copied,
        and seven characters is the length git itself abbreviates to.
        """
        base = self.version or "unknown"
        if not self.source_ref:
            return base
        return f"{base}@{self.source_ref[:7]}"


def installed_version() -> str:
    """Distribution version, or ``""`` when this interpreter has no install.

    Empty is the source-checkout case: there is nothing to compare to PyPI
    and :func:`install_kind` will refuse rather than guess.

    Install metadata is written ONCE and never refreshed when the version in
    ``pyproject.toml`` moves, so a checkout at 0.49.0 kept reporting the
    0.46.23 it had been installed at -- and the app showed that stale number to
    users in Settings > Updates (QA Q3 / UX U13).

    PRECEDENCE: the checkout's ``pyproject.toml`` when that checkout IS this
    install (an editable install, verified through ``direct_url.json`` by
    :func:`_editable_source_version`), otherwise the install metadata.

    Not a maximum. This used to take the HIGHER of the two, on the argument that
    both are stale in opposite directions -- metadata never moves with the tree,
    while a stray ``pyproject.toml`` declaring an older version produced a
    spurious "update available" (review round 2, MINOR-2) -- and that a maximum
    therefore failed in the SAFE direction.

    That argument only covers a stray NEWER file. A stray OLDER one drags the
    number DOWN, and there is no direction of ``max`` that protects against
    both, because the real defect was never the comparison: it was trusting a
    file whose relationship to the running code had not been established. On a
    real install a spawned child read a 0.51.0 checkout while running 0.51.5,
    and the maximum could not help -- the wrong input simply won or lost on
    magnitude.

    Once the source side is required to prove identity, the two are no longer
    rival guesses about one unknown: a matched checkout is the live version of
    the code that is executing and its metadata is a snapshot of an earlier
    state, so the checkout is authoritative outright and the ordering is
    irrelevant. When nothing proves identity there is only one input left.

    A packaged (non-editable) install has no editable ``direct_url.json``, so it
    reports its metadata exactly as before -- including when a checkout of this
    project is sitting in the working directory, which is the case that broke.
    """
    source = _editable_source_version()
    try:
        metadata = version("local-operator")
    except PackageNotFoundError:
        metadata = ""
    # ``source`` is non-empty only when the imported tree was proven to be this
    # install's editable source, so it describes the running code and wins
    # outright. No comparison: an older matched checkout is a real downgrade of
    # the code in memory, not a stale reading to be corrected upward.
    return source or metadata


def _editable_install_root() -> Path | None:
    """The source tree an EDITABLE install of this project points at, if any.

    PEP 610 records the origin of an editable install in ``direct_url.json`` as
    a ``file://`` URL with ``dir_info.editable`` true. That URL is the one piece
    of evidence that says which checkout the installed distribution actually
    resolves to, as opposed to which checkout merely happens to be lying around.

    ``None`` whenever this is not an editable install, the marker is missing, or
    the URL is not a readable local path — every one of which means "no checkout
    is authoritative here", which is the safe answer for the only caller.
    """
    data = _direct_url_payload()
    if data is None:
        return None
    dir_info = data.get("dir_info")
    editable = data.get("editable") is True or (
        isinstance(dir_info, dict) and dir_info.get("editable") is True
    )
    if not editable:
        return None
    url = data.get("url")
    if not isinstance(url, str) or not url.startswith("file:"):
        return None
    try:
        return Path(url2pathname(urlparse(url).path)).resolve()
    except (OSError, ValueError):
        return None


def _editable_source_version() -> str:
    """The checkout's ``pyproject.toml`` version — only when it IS the install.

    An editable install's ``dist-info`` is written once and never moves with the
    tree, so a checkout at 0.49.0 reported the 0.46.23 it was installed at and
    the app showed that stale number in Settings > Updates (QA Q3 / UX U13).
    Reading the adjacent ``pyproject.toml`` fixes that — but only if the file
    describes the code that is actually running.

    IDENTITY, NOT ADJACENCY. This previously trusted any ``pyproject.toml``
    sitting beside the imported package, on the reasoning that a released build
    has no adjacent project file and so could never read a stray one. That
    reasoning was false, and the failure was measured on a real install: a
    spawned child whose working directory was a checkout of this project
    imported THAT checkout (``-m`` puts the cwd on ``sys.path`` ahead of
    site-packages — the defect :mod:`local_operator.interpreter` now prevents),
    so ``Path(__file__)`` pointed into a tree the install had nothing to do
    with. A 0.51.5 install reported 0.51.0, and its stale ``*.egg-info``
    shadowed the real dist-info down to 0.49.3 as well.

    So the checkout wins only when ``direct_url.json`` names it as the editable
    source of the installed distribution. A stray tree — a scratch clone, a
    worktree that merely happens to be the cwd, a fixture — is now ignored, and
    the caller falls back to metadata that at least describes a real install.

    Still cheap and total: any doubt (no editable install, a mismatched root, an
    unreadable or unexpected file) returns ``""``, and nothing here may raise.
    """
    try:
        import tomllib

        # local_operator/update.py -> local_operator/ -> the checkout root.
        root = Path(__file__).resolve().parent.parent
        # The imported tree must BE the tree this install was made editable
        # from; adjacency alone proves nothing about which code is running.
        if root != _editable_install_root():
            return ""
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():
            return ""
        with pyproject.open("rb") as handle:
            project = tomllib.load(handle)["project"]
        if project.get("name") != "local-operator":
            return ""
        return str(project["version"])
    except Exception:  # noqa: BLE001 — a version readout must never raise
        return ""


def parse_version(value: str) -> tuple[int, int, int] | None:
    """``X.Y.Z`` or ``None``. No ``packaging`` — this project ships that shape.

    An unparseable side (a local ``0.28.0rc1``, a yanked extra) is treated as
    "not behind" by the caller: a banner we cannot defend is worse than none.
    """
    parts = value.strip().split(".")
    if len(parts) != 3:
        return None
    try:
        return int(parts[0]), int(parts[1]), int(parts[2])
    except ValueError:
        return None


def is_behind(installed: str, latest: str | None) -> bool:
    """True only when both sides parse and installed is strictly older."""
    if not installed or not latest:
        return False
    left = parse_version(installed)
    right = parse_version(latest)
    if left is None or right is None:
        return False
    return left < right


def default_cache_dir() -> Path:
    return _CACHE_DIR.expanduser()


def _cache_path(cache_dir: Path | None) -> Path:
    return (cache_dir or default_cache_dir()) / _CACHE_NAME


def _read_cache(path: Path) -> tuple[dict[str, Any] | None, float]:
    """Return ``(payload, age_seconds)``; ``(None, inf)`` when unusable.

    Corrupt and future-dated documents are missing, not raised: this is an
    optimisation store. A future timestamp treated as age-zero would pin the
    document forever after a clock skew (same rule as ``model/catalogue.py``).
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        payload = raw["payload"]
        fetched_at = float(raw["fetched_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None, float("inf")
    if not isinstance(payload, dict):
        return None, float("inf")
    age = time.time() - fetched_at
    if not (age >= 0):
        return payload, float("inf")
    return payload, age


def _umask() -> int:
    current = os.umask(0o022)
    os.umask(current)
    return current


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    """Atomic temp+rename, best-effort. A failed write must not fail the check."""
    fd: int | None = None
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
        fd, tmp = handle, Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None
            json.dump({"fetched_at": time.time(), "payload": payload}, stream)
        os.chmod(tmp, 0o644 & ~_umask())
        tmp.replace(path)
        tmp = None
    except OSError:
        pass
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def _fetch_pypi_version(*, client: Any | None = None) -> str | None:
    """Live ``info.version``, or ``None`` on any transport/HTTP/JSON error.

    httpx is imported here so ``import local_operator.cli`` (and the splash
    module) never pay for it. The splash worker already swallows ``None``.
    """
    import httpx

    try:
        if client is None:
            response = httpx.get(PYPI_JSON_URL, timeout=_FETCH_TIMEOUT_S)
        else:
            response = client.get(PYPI_JSON_URL, timeout=_FETCH_TIMEOUT_S)
        response.raise_for_status()
        version_s = response.json()["info"]["version"]
    except Exception:
        return None
    if not isinstance(version_s, str) or not version_s.strip():
        return None
    return version_s.strip()


def check_latest(
    *,
    force: bool = False,
    cache_dir: Path | None = None,
    client: Any | None = None,
) -> VersionCheck:
    """Installed vs PyPI. ``force`` bypasses the TTL but still rewrites the cache.

    Failure mode is silent: ``latest is None`` and ``behind is False``. The
    splash must not grow a toast or steal ``info.notice`` for a probe.
    """
    installed = installed_version()
    path = _cache_path(cache_dir)
    payload, age = _read_cache(path)
    cached: str | None = None
    if payload is not None:
        raw = payload.get("version")
        if isinstance(raw, str) and raw.strip():
            cached = raw.strip()

    latest: str | None
    if cached is not None and age < TTL_S and not force:
        latest = cached
    else:
        fetched = _fetch_pypi_version(client=client)
        if fetched is not None:
            latest = fetched
            _write_cache(path, {"version": fetched})
        else:
            latest = cached

    return VersionCheck(installed=installed, latest=latest, behind=is_behind(installed, latest))


def cached_latest(cache_dir: Path | None = None) -> tuple[str | None, float | None]:
    """The last PyPI answer we already have, and how old it is. NEVER fetches.

    :func:`check_latest` is the *upgrade* path and is not a cached read despite
    looking like one: only its ``cached is not None and age < TTL_S and not
    force`` branch is served from disk, and every other path — cold cache,
    corrupt document, an age past :data:`TTL_S` — falls through to
    :func:`_fetch_pypi_version`, a live HTTP call bounded at
    :data:`_FETCH_TIMEOUT_S`, and then *writes* the cache.

    ``/info`` is the *diagnostic* path. It is opened precisely when something is
    already broken, and frequently because the network is the thing that is
    broken, so a 5 s stall on a captive portal or a DNS blackhole is the worst
    available behaviour for the one screen that exists to explain a failure.
    Measured on this host with an injected client that raises immediately:
    ``check_latest()`` on a cold cache cost **156.86 ms** (all of it DNS/connect
    setup before the raise) against **0.0002 ms** for the read below. A
    diagnostic must also not MUTATE state, and ``check_latest`` rewrites the
    cache on success.

    Returns ``(None, None)`` when there is nothing usable cached. The caller
    must render that as "unknown (never checked)" and never as "up to date":
    those are different facts, and collapsing them tells a user on a broken
    network that they are on the newest release.
    """
    payload, age = _read_cache(_cache_path(cache_dir))
    if payload is None:
        return None, None
    raw = payload.get("version")
    if not isinstance(raw, str) or not raw.strip():
        return None, None
    # ``_read_cache`` returns ``inf`` for a future-dated document (a clock
    # skew), which is not an age any caller can render. The version is still
    # good, so the value survives and only its staleness is unknown.
    return raw.strip(), (age if age != float("inf") else None)


def _direct_url_payload() -> dict[str, Any] | None:
    """PEP 610 ``direct_url.json`` from the distribution that actually has one.

    NOT ``distribution("local-operator")``. That returns the FIRST name match in
    ``sys.path`` order, and a leftover ``local_operator.egg-info/`` in a checkout
    -- a gitignored build artifact any ``pip install -e``/``setup.py`` run leaves
    behind, present in real checkouts -- sits earlier than site-packages whenever
    the cwd is on the path. Egg-info metadata predates PEP 610 and carries no
    ``direct_url.json``, so the shadow made a genuine editable install look like
    no install at all: ``_editable_install_root()`` returned ``None``, the
    identity check in :func:`_editable_source_version` failed against its own
    tree, and ``installed_version()`` fell through to the stale ``PKG-INFO``
    number the egg-info advertised. Measured on a real ``uv pip install -e`` with
    ``pyproject.toml`` at 0.51.7 and a leftover ``PKG-INFO`` at 0.46.23, that
    reported 0.46.23 -- the very Settings > Updates staleness (QA Q3 / UX U13)
    this module exists to prevent, reintroduced by the shadow rather than by any
    version comparison.

    So scan every installed distribution of this name and take the first that
    publishes the marker. The marker is the evidence; a distribution without one
    cannot answer the question and must not be allowed to answer it negatively.

    Deliberately NOT "accept an egg-info found inside the candidate root": that
    reasoning is adjacency again -- a STRAY checkout's egg-info also lives inside
    that stray checkout, so it would vouch for exactly the unrelated tree this
    function's caller was rewritten to reject.
    """
    for dist in distributions(name="local-operator"):
        text = dist.read_text("direct_url.json")
        if not text:
            continue
        try:
            data = json.loads(text)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _is_editable_direct_url() -> bool:
    """PEP 610 / 660: ``dir_info.editable`` is what pip and uv write for ``-e``."""
    data = _direct_url_payload()
    if data is None:
        return False
    if data.get("editable") is True:
        return True
    dir_info = data.get("dir_info")
    return isinstance(dir_info, dict) and dir_info.get("editable") is True


def _installer_metadata() -> str:
    """Lower-cased dist-info ``INSTALLER``, or ``""`` when there is none.

    pip, uv and pipx each write the file, so it states outright which tool
    owns the install rather than inferring it from a path. Absent is no
    evidence at all (a vendored tree, a distro package), never a negative.
    """
    try:
        dist = distribution("local-operator")
    except PackageNotFoundError:
        return ""
    text = dist.read_text("INSTALLER")
    if not text:
        return ""
    return text.strip().lower()


def _is_uv_tool(prefix: Path) -> bool:
    """uv tool is the documented end-user path (README + the ``lop`` launcher).

    Two probes because the layout has drifted across uv versions: older
    installs put ``uv-receipt.toml`` next to ``sys.prefix``; every current
    one still nests the env under ``…/uv/tools/local-operator``. Either
    signal is enough — requiring both would miss a valid install.
    """
    if (prefix / "uv-receipt.toml").is_file():
        return True
    if (prefix.parent / "uv-receipt.toml").is_file():
        return True
    parts = prefix.parts
    try:
        uv_at = parts.index("uv")
    except ValueError:
        return False
    rest = parts[uv_at + 1 :]
    return len(rest) >= 2 and rest[0] == "tools" and "local-operator" in rest


def _is_pipx(prefix: Path) -> bool:
    """pipx is the other installer the README already tells people to use.

    ``PIPX_HOME`` first so a relocated pipx (the documented escape hatch on
    Linux PEP-668 hosts) still matches; the default ``~/.local/pipx`` is
    what an unconfigured install actually writes.
    """
    pipx_home = Path(os.environ.get("PIPX_HOME", Path.home() / ".local" / "pipx"))
    expected = (pipx_home / "venvs" / "local-operator").resolve()
    try:
        resolved = prefix.resolve()
    except OSError:
        resolved = prefix
    if resolved == expected or expected in resolved.parents:
        return True
    parts = prefix.parts
    return "pipx" in parts and "venvs" in parts and "local-operator" in parts


def _is_ordinary_pip(prefix: Path) -> bool:
    """A venv prefix, or a dist-info that names pip: the README ``pip install`` path.

    The two layout probes both miss a *base* interpreter (#396). A ``mise``,
    ``pyenv`` or ``asdf`` toolchain reports ``sys.prefix == sys.base_prefix``
    and writes no ``pyvenv.cfg``, so an ordinary ``pip install
    local-operator`` there fell through to ``UNKNOWN`` and ``/update``
    refused an upgrade that ``pip install -U`` performs fine — a reporter on
    0.42.13 could not reach 0.42.19 at all.

    ``INSTALLER`` is consulted last and only for the exact value ``pip``.
    ``uv`` and ``pipx`` reach this line only once their own probes above have
    declined, and answering "pip" for them would be the guess this module
    exists to refuse: ``uv`` is also what ``uv pip install --system`` writes
    into a base prefix, where ``uv tool upgrade`` is the wrong command.
    """
    if (prefix / "pyvenv.cfg").is_file():
        return True
    if sys.prefix != getattr(sys, "base_prefix", sys.prefix):
        return True
    return _installer_metadata() == "pip"


def install_kind(
    *,
    prefix: str | Path | None = None,
    executable: str | Path | None = None,
) -> InstallKind:
    """How this interpreter was installed. Refuse-don't-guess for the rest.

    ``prefix`` / ``executable`` are test seams so a tmp tree can stand in
    for ``sys.prefix`` without mutating the running process.
    """
    del executable  # reserved: unknown-layout messages print the real one
    root = Path(prefix) if prefix is not None else Path(sys.prefix)

    # No distribution, or an editable checkout: this is the repo ``.venv``.
    # ``pip install -U`` into it would either no-op or smash the editable
    # link. Developers update the *global* runtime with ``lop-update``.
    try:
        distribution("local-operator")
        has_dist = True
    except PackageNotFoundError:
        has_dist = False
    if not has_dist or _is_editable_direct_url():
        return InstallKind.EDITABLE

    if _is_uv_tool(root):
        return InstallKind.UV_TOOL
    if _is_pipx(root):
        return InstallKind.PIPX
    if _is_ordinary_pip(root):
        return InstallKind.PIP
    return InstallKind.UNKNOWN


#: First token written into ``.lop-source`` for an install that came from a
#: PyPI wheel rather than from a git snapshot. A sentinel rather than a fake
#: commit: a PyPI upgrade genuinely HAS no git ref, and copying the previous
#: install's sha forward is exactly the lie this module exists to stop.
#:
#: Expect one cosmetic artefact in the upgrade window that introduces this
#: token: a runtime still on PRE-SENTINEL code reads the whole first token as
#: a ref and renders ``/info`` as ``source git snapshot @ pypi``. It is
#: self-clearing rather than sticky — the marker's mtime still CHANGES, so
#: those runtimes see a new build and retire on their own (measured at ~1.5 s,
#: QA round 1, Q2) — and it is unavoidable for a format change that must keep
#: the marker a single file shared with ``lop-update``.
PYPI_SOURCE_TOKEN = "pypi"


def _looks_like_git_sha(token: str) -> bool:
    """Is this ``.lop-source`` token a commit, as opposed to a sentinel?

    The marker's first token is either an abbreviated-or-full git sha (what
    ``lop-update`` writes) or :data:`PYPI_SOURCE_TOKEN`. Discriminating on
    SHAPE rather than on an allow-list keeps the two writers — this module and
    the out-of-tree ``lop-update`` shell script — from having to agree on
    anything but the format, and means an unrecognised future sentinel degrades
    to "no ref" instead of being rendered as a bogus commit.

    A PyPI version can never collide: it carries dots, which are not hex.

    The shape test is deliberately loose: it admits any 7-40 hex token, so a
    hex-looking BRANCH name (``deadbeef``) in the second writer's ref position
    would read as a commit. It cannot fire today — ``lop-update`` only ever
    writes ``git rev-parse --verify`` output into the first token — but it is
    the constraint on anyone adding a sentinel later: A FUTURE SENTINEL MUST
    NOT BE HEX, or it will render as a bogus commit instead of degrading to
    "no ref" (review round 1, R1-3).
    """
    if not (7 <= len(token) <= 40):
        return False
    return all(char in "0123456789abcdefABCDEF" for char in token)


def is_git_snapshot(prefix: str | Path | None = None) -> bool:
    """Was this install built from a git ref, as opposed to a PyPI wheel?

    Presence of ``.lop-source`` used to be the whole test, which was true only
    while ``lop-update`` was the sole writer. It is not any more: a PyPI
    ``/update`` now records ``pypi <version>`` at the same path (see
    :func:`write_source_marker`), so the marker survives the transition from a
    git snapshot to a wheel and the FIRST TOKEN — not the file's existence —
    is what says which one is installed.

    Reading existence alone here is what made ``lop update`` keep printing
    "this runtime was built from git" on a host whose git snapshot had already
    been replaced by a wheel, and made ``/info`` label that wheel a snapshot.

    Default is still PyPI: the caller prints one line and upgrades. We do not
    invoke ``lop-update`` — developers who want git ``main`` keep using that
    script.
    """
    return bool(source_ref(prefix))


def source_ref(prefix: str | Path | None = None) -> str:
    """The git commit this install was built from, or ``""``.

    ``.lop-source`` holds two whitespace-separated tokens at the root
    :func:`is_git_snapshot` probes, in one of two shapes:

    * ``<git-sha> <ref>`` — a ``lop-update`` snapshot. The sha is the commit;
      the ref half is a label that repeats across rebuilds of one release and
      therefore cannot distinguish two builds, which is the whole job here.
    * ``pypi <version>`` — a PyPI wheel installed by :func:`perform_upgrade`.
      There is no commit, so this returns ``""`` and the caller falls back to
      the distribution version alone.

    Absent (never upgraded through either writer, an editable checkout) is
    ``""`` too, for the same reason.
    """
    root = Path(prefix) if prefix is not None else Path(sys.prefix)
    try:
        raw = (root / ".lop-source").read_text(encoding="utf-8")
    except (OSError, ValueError):
        # Missing, unreadable, a directory (OSError) — or not valid UTF-8,
        # which raises UnicodeDecodeError, a ValueError rather than an
        # OSError. Both are caught because this must never raise into an
        # adopt or a bind: ``RuntimeServer.__init__`` stamps the record from
        # here, so an unhandled decode error on a corrupt marker would stop
        # every runtime on the host from being constructed at all — total
        # blast radius for a token that is only ever decoration on a
        # diagnostic path (review round 1, R1-2).
        return ""
    parts = raw.split()
    if not parts:
        return ""
    return parts[0] if _looks_like_git_sha(parts[0]) else ""


def write_source_marker(
    root: str | Path,
    *,
    version: str,
    commit: str = "",
    ref: str = "",
) -> bool:
    """Record what is installed at ``root`` in ``.lop-source``. Never raises.

    WHY THIS EXISTS
    ---------------
    ``perform_upgrade`` replaced the payload under ``root`` and nothing wrote
    the marker back, so the file kept describing the build it had DISPLACED.
    On the reporting host that left ``.lop-source`` naming the 0.51.7 bump
    commit while site-packages carried 0.51.9 — every ``version@ref`` label,
    every :class:`BuildStamp` comparison and the settle clock below all read
    from a file about a build that was no longer there.

    THE FORMAT IS A CONTRACT WITH A SECOND WRITER
    ---------------------------------------------
    ``~/.local/bin/lop-update`` (a shell script, out of this tree) writes
    ``printf '%s %s\\n' "$COMMIT" "$REF"``. That shape is preserved exactly, so
    the two writers stay interchangeable and neither has to know about the
    other; :func:`source_ref` discriminates on token shape, not on which writer
    produced the line. A PyPI upgrade has no commit, so it writes
    ``pypi <version>`` — honest about having no ref rather than carrying the
    previous install's sha forward.

    ORDERING IS LOAD-BEARING
    ------------------------
    Callers must write this only AFTER the installer has exited successfully.
    :func:`build_marker_age_s` uses this file's mtime as the moment the install
    became whole, and a runtime that acted on a marker written mid-install
    could spawn a successor that imports a torn tree.

    ATOMIC, AND BEST-EFFORT
    -----------------------
    Temp-and-rename within the destination directory, mirroring
    :func:`_write_cache`: a torn or interrupted write cannot leave a partial
    marker behind, which matters because ``RuntimeServer.__init__`` reads this
    file and a corrupt one would otherwise reach every runtime on the host
    (review round 1, R1-2). A failure to write returns ``False`` rather than
    raising — a missing marker degrades to "compare on version alone", while a
    failed upgrade report would be a worse outcome than an unrecorded one.
    """
    first = commit if commit else PYPI_SOURCE_TOKEN
    second = ref if commit else version
    line = f"{first} {second}\n"

    path = Path(root) / ".lop-source"
    fd: int | None = None
    tmp: Path | None = None
    try:
        handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
        fd, tmp = handle, Path(name)
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            fd = None
            stream.write(line)
            # fsync before the rename: the marker's whole value is that it is
            # true about the tree beside it, and an unflushed write that
            # survives as an empty file after a crash reads as "no ref".
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(tmp, 0o644 & ~_umask())
        tmp.replace(path)
        tmp = None
        return True
    except OSError:
        return False
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
        if tmp is not None:
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass


def installed_build(prefix: str | Path | None = None) -> BuildStamp:
    """This interpreter's comparable build token, read fresh from disk.

    Called at the seams that spawn or bind a runtime (adopt, engage, bind) —
    a handful of times per process, never on a hot path — because that is
    exactly where a stale in-memory build meets a newer on-disk one. Both
    reads are deliberately live: ``importlib.metadata.version`` re-reads the
    dist-info directory (verified on this project's 3.12 and on 3.14.3: the
    directory NAME carries the version, so even a path-keyed cache misses),
    and ``.lop-source`` is one small file read.
    """
    return BuildStamp(version=installed_version(), source_ref=source_ref(prefix))


def classify_import_failure(
    exc: BaseException, module: str, *, boot: BuildStamp | None
) -> str | None:
    """Name a mid-install import as ``install-mid-update``, or ``None``.

    ``lop-update`` and :func:`perform_upgrade` replace the installed tree IN
    PLACE, so a process that loaded the old build can hit a lazy
    ``from local_operator… import x`` the new tree no longer satisfies — the
    observed shape is precise: the module still resolves, the NAME does not
    (``ImportError: cannot import name '_journal_injection_ids' from
    'local_operator.session.transcript'``, 605 times on one machine's log).

    What separates that from a genuine packaging bug is the STAMP MOVING UNDER
    THE PROCESS. If the install on disk still matches the build this process
    booted from, the miss is ours and must stay an ordinary traceback — so
    ``None``. A ``boot`` we could not read (``None``) also answers ``None``:
    without a baseline there is nothing to compare, and guessing here would
    relabel a real packaging error as an install race.

    ``module`` is what the CALLER was importing, used when the exception itself
    names nothing (some wrappers drop ``name``).
    """
    if not isinstance(exc, ImportError):
        return None
    named = str(getattr(exc, "name", "") or "")
    text = str(exc) or ""
    if not (
        named.startswith("local_operator")
        or module.startswith("local_operator")
        or "local_operator" in text
    ):
        return None
    if boot is None:
        return None
    try:
        current = installed_build()
    except Exception:  # noqa: BLE001 — an unreadable stamp is not evidence
        return None
    if current == boot:
        return None
    from local_operator.incidents import render_cut_off_reason

    detail = "" if current.label() == boot.label() else f" ({boot.label()} → {current.label()})"
    return render_cut_off_reason("install-mid-update", detail=detail)


def build_marker_age_s(prefix: str | Path | None = None) -> float | None:
    """Seconds since the install on disk was last written, or ``None``.

    The settle input for a runtime's self-refresh (``process._build_changed``):
    an installer rewrites site-packages over several seconds, and a runtime
    that acted inside that window could spawn a successor importing a torn
    tree. Two files record when the install was last touched: ``.lop-source``,
    written last by both writers (``lop-update`` and
    :func:`write_source_marker`) precisely so it marks the moment the install
    became whole, and the ``dist-info`` directory, written last by the
    installer itself.

    WHY THE NEWEST OF THE TWO, NOT THE MARKER FIRST
    -----------------------------------------------
    Reading the marker first and the dist-info only as a fallback assumed the
    marker is rewritten whenever the payload is — which was not true before
    :func:`write_source_marker` existed, and still is not for an install
    upgraded by some path that writes neither (a hand-run ``uv tool install
    --force``). A marker older than the payload then reported a minutes-old
    install as hours old: on the reporting host, ~19000 s for a tree written
    minutes earlier, which is the settle guard reading the wrong clock and
    disarming itself exactly when it was needed.

    Taking the MOST RECENT of the two mtimes cannot have that failure: any
    write to either file only makes the reported age smaller, and a smaller
    age means the guard waits longer. Erring toward "not settled yet" is the
    safe direction — the cost is one more refresh check, against the cost of
    spawning from a half-written tree.

    ``None`` when neither can be read — an editable checkout has no dist-info
    of its own and no marker, and the caller treats "unknown" as "not
    settled", which is the same safe side.

    THE TWO TERMS ARE NOT SCOPED THE SAME WAY, ON PURPOSE
    -----------------------------------------------------
    The marker is read from ``prefix``; the dist-info is always the RUNNING
    INTERPRETER's, because :func:`distribution` resolves through ``sys.path``
    and takes no prefix. In production the two are the same tree — a
    runtime's ``prefix`` IS its own install — so the distinction is invisible.
    It shows only through the ``LOP_BUILD_PREFIX`` test seam, where a caller
    passing a foreign prefix gets an age mixing that prefix's marker with this
    interpreter's dist-info (review round 1, R1-2).

    That is deliberate rather than merely tolerated. Scoping the dist-info to
    ``prefix`` means globbing ``<prefix>/lib/*/site-packages/*.dist-info``,
    and a layout that glob does not match degrades this back to reading the
    MARKER ALONE — which is precisely the stale-marker failure the max-of-two
    was introduced to fix, reintroduced on the production path to sharpen a
    seam only the e2e stage uses. The mixed answer is also safe in the one
    direction that matters: an extra mtime can only make the age SMALLER, so
    the settle guard waits longer, never less.
    """
    root = Path(prefix) if prefix is not None else Path(sys.prefix)
    mtimes: list[float] = []

    try:
        mtimes.append((root / ".lop-source").stat().st_mtime)
    except OSError:
        pass

    try:
        dist = distribution("local-operator")
        located = getattr(dist, "_path", None)
        if located is not None:
            mtimes.append(Path(located).stat().st_mtime)
    except (PackageNotFoundError, OSError):
        pass

    if not mtimes:
        return None
    return max(0.0, time.time() - max(mtimes))


def installer_invocation(
    kind: InstallKind,
    *,
    executable: str | None = None,
) -> tuple[list[str], str | None]:
    """``(argv, executable)`` for the installer of ``kind``.

    Returned as a PAIR because the two are not independent: ``executable`` is
    the image to run the argv with, or ``None`` when argv[0] already is the
    image. A caller that took the argv alone and passed no ``executable=`` would
    have POSIX ``execve`` the argv[0] string, and for the pip path that string is
    a label with spaces in it.

    UV, PIPX (and any future git/pipx-shaped installer) keep their own argv[0]
    and get ``None``: they are third-party binaries, they are named already, and
    replacing their argv[0] would both mislabel the row and lose the binary the
    user's PATH resolves.

    The PIP path is OURS, and it is the EDR profile the operator has already
    been bitten by: an interpreter named ``python3.x`` performing a network
    install from a process the user did not start. Its argv[0] is therefore the
    role label and its image is the interpreter — the same pairing
    ``secrets/client.py`` uses for the broker. ``executable=`` is honoured for
    the argv-only case the pip kind had before, so a caller pinning an
    interpreter still pins it.
    """
    if kind is InstallKind.UV_TOOL:
        # Re-install with --force rather than `uv tool upgrade`:
        # `uv tool upgrade` fails when installed from a temporary git snapshot
        # (the build directory no longer exists) or when installed with an exact
        # version pin (`specifier = "==..."` in uv-receipt.toml causes "Nothing to upgrade").
        # `uv tool install --force local-operator` always fetches and replaces with
        # the latest PyPI distribution regardless of previous installation receipt.
        return ["uv", "tool", "install", "--force", "local-operator"], None
    if kind is InstallKind.PIPX:
        return ["pipx", "upgrade", "local-operator"], None
    if kind is InstallKind.PIP:
        from local_operator import procname

        argv0, image = procname.spawn_identity(procname.LABEL_INSTALL)
        return [argv0, "-m", "pip", "install", "-U", "local-operator"], executable or image
    raise UpdateError(f"no installer for {kind.value}")


def installer_argv(
    kind: InstallKind,
    *,
    executable: str | None = None,
) -> list[str]:
    """The installer's argv alone, for callers that only print or compare it.

    Spawning callers use :func:`installer_invocation`: printing an argv is a
    legitimate use of the list on its own, running one is not, because the pip
    path's argv[0] is a label and needs the interpreter beside it.
    """
    return installer_invocation(kind, executable=executable)[0]


def installer_label(kind: InstallKind) -> str:
    if kind is InstallKind.UV_TOOL:
        return "uv tool"
    if kind is InstallKind.PIPX:
        return "pipx"
    if kind is InstallKind.PIP:
        return "pip"
    return kind.value


def editable_refusal() -> str:
    return (
        "this interpreter is the repo .venv, not an installed distribution. "
        "update the global runtime with lop-update after the change is merged."
    )


def tui_editable_refusal() -> str:
    """Same refusal as :func:`editable_refusal`, worded for the person in the TUI.

    The CLI line names ``.venv`` and ``lop-update`` because that is the
    contributor path. ``/update`` is typed by someone sitting in the app;
    they need to know this is the checkout, not the installed ``lop``.
    """
    return "this is the repo checkout, not the installed lop — run lop-update after merge"


def tui_installer_failure(kind: InstallKind) -> str:
    """User-facing next step after a non-zero installer, keyed by install kind."""
    if kind is InstallKind.PIPX:
        hint = "pipx upgrade local-operator"
    elif kind is InstallKind.PIP:
        hint = "python -m pip install -U local-operator"
    else:
        hint = "uv tool install --force local-operator"
    return f"upgrade failed; try `{hint}` in a shell"


def unknown_refusal(
    *,
    prefix: str | None = None,
    executable: str | None = None,
) -> str:
    return (
        "cannot tell how this install was launched\n"
        f"  sys.prefix: {prefix or sys.prefix}\n"
        f"  sys.executable: {executable or sys.executable}\n"
        "supported upgrades:\n"
        "  uv tool upgrade local-operator\n"
        "  pipx upgrade local-operator\n"
        "  python -m pip install -U local-operator"
    )


def git_snapshot_notice() -> str:
    return "this runtime was built from git; " "lop update will replace it with the PyPI wheel"


def _run_installer(argv: list[str], *, executable: str | None = None) -> int:
    import subprocess

    # stderr/stdout pass through: the installer is what the user is watching.
    # ``executable`` is what makes a labelled argv[0] runnable at all — without
    # it POSIX would try to exec the label itself (see ``installer_invocation``).
    completed = subprocess.run(argv, check=False, executable=executable)
    return int(completed.returncode)


def perform_upgrade(
    *,
    target: str,
    kind: InstallKind | None = None,
    run: Callable[[list[str]], int] | None = None,
    prefix: str | Path | None = None,
    executable: str | None = None,
) -> str:
    """Run the detected installer. Returns ``target`` (this process cannot re-read it).

    The new wheel is not imported into this interpreter; callers print
    ``target`` rather than asking :func:`installed_version` again.

    On success the ``.lop-source`` marker is rewritten to describe what was
    just installed — see :func:`write_source_marker`. Without that step the
    marker kept naming the DISPLACED build: it is written only by the
    ``lop-update`` shell script, so an upgrade driven from here (``lop
    update`` and the TUI's ``/update``, which share this function) moved
    site-packages and left the marker behind, and every ``version@ref`` label
    and settle-clock reading on the host went on describing a build that was
    no longer installed.

    Ordering is deliberate and load-bearing: the marker is written only after
    the installer has exited 0, because its mtime is the signal a runtime uses
    to decide the install has settled.
    """
    detected = kind if kind is not None else install_kind(prefix=prefix, executable=executable)
    if detected is InstallKind.EDITABLE:
        raise UpdateError(editable_refusal())
    if detected is InstallKind.UNKNOWN:
        raise UpdateError(
            unknown_refusal(prefix=str(prefix) if prefix else None, executable=executable)
        )
    argv, image = installer_invocation(detected, executable=executable)
    if run is not None:
        # The injected runner sees the argv alone: it is a seam for tests and for
        # callers that observe the installer, not a way to spawn anything.
        code = run(argv)
    else:
        code = _run_installer(argv, executable=image)
    if code != 0:
        raise UpdateError(f"installer exited {code}")

    # Only the uv-tool layout has a ``.lop-source`` root to record into, and
    # it is the layout ``lop-update`` shares. pipx and pip installs never had
    # a marker and gain nothing from one: they compare on version alone.
    if detected is InstallKind.UV_TOOL:
        root = Path(prefix) if prefix is not None else Path(sys.prefix)
        # ``target`` is the PyPI version just installed, and this path is
        # always a PyPI wheel: ``installer_argv`` runs `uv tool install
        # --force local-operator` with no --from, so no git ref exists to
        # record. Passing no commit is what makes the marker say so.
        if not write_source_marker(root, version=target):
            # Deliberately not fatal: the upgrade itself SUCCEEDED, and a
            # failed marker only costs accuracy in the labels. But without a
            # line here the host silently reverts to the pre-fix behaviour —
            # a marker naming the displaced build — with nothing to find
            # afterwards, so log it rather than discarding the result
            # (review round 1, R1-4).
            logger.warning(
                "Upgraded to %s but could not record it in %s/.lop-source; "
                "version labels will keep naming the previous build.",
                target,
                root,
            )
    return target


def _mobile_plist_path() -> Path:
    """Well-known LaunchAgent path. Isolated so tests can patch it.

    Same one-liner as ``install.plist_path`` / ``install.LABEL``
    (``com.local-operator.mobile.plist``). Duplicated on purpose:
    ``local_operator.mobile.install`` imports ``daemon`` (Starlette),
    and folding that into the updater would pull the web stack into
    every ``lop update`` and every TUI ``/update`` worker.
    """
    return Path.home() / "Library" / "LaunchAgents" / "com.local-operator.mobile.plist"


def _mobile_healthz_answers() -> bool:
    """True only when something already answers on the default port.

    Used solely to warn about an unsupervised ``lop mobile serve``. Do
    not SIGTERM that process: it is not ours to bounce.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(_MOBILE_HEALTHZ, timeout=1.0) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _mobile_restart_invocation() -> tuple[list[str], str] | None:
    """``(argv, executable)`` for the *new* distribution's ``mobile restart``.

    ``sys.executable -m local_operator.cli`` is the post-upgrade
    interpreter — the same interpreter the LaunchAgent's ProgramArguments
    already name, so its site-packages are the wheel the installer just
    wrote. There is deliberately NO PATH ``lop`` fallback: a PATH hit can
    be a *different* installation (another tool env, a brew shim) than
    the one just upgraded, and restarting that would serve the wrong
    build while reporting success. If this interpreter is gone after the
    upgrade, the refresh fails honestly and the copy names the recovery.

    ``SAFE_PATH_FLAG`` (via ``python_argv``), because the sentence above is
    only true with it. :func:`refresh_mobile_after_upgrade` runs this argv with
    no ``cwd=``, so the child inherits the directory the update was started
    from — ``update.py``'s own ``lop update`` and the in-TUI ``/update`` worker
    both run with a user or session cwd. When that directory is a checkout of
    this project, ``-m`` puts it on ``sys.path`` ahead of site-packages and the
    bounce restarts the daemon through the CHECKOUT: pre-upgrade code, running
    under the post-upgrade interpreter, reporting success. See
    :mod:`local_operator.interpreter`.

    The argv[0] is the role label and the interpreter travels BESIDE it, as in
    :func:`installer_invocation`: this is a process the product spawns, and
    naming every such process is the point of the change this belongs to. A
    daemon bounce is also the kind of activity an EDR watches.
    """
    if sys.executable and Path(sys.executable).exists():
        from local_operator import procname

        argv0, image = procname.spawn_identity(procname.LABEL_MOBILE_RESTART)
        return [argv0, SAFE_PATH_FLAG, "-m", "local_operator.cli", "mobile", "restart"], image
    return None


def refresh_mobile_after_upgrade() -> MobileRefresh:
    """Bounce the supervised mobile daemon after a successful wheel install.

    Kept out of :func:`perform_upgrade` so existing installer tests cannot
    kickstart a real LaunchAgent. Never raises: the package upgrade already
    succeeded, and a failed bounce must not roll it back.

    ``restart``, not ``install``: the wheel already ships ``mobile/web/dist``,
    cookies live in the Keychain, and ``install`` would regenerate a
    password. In-process ``service_action`` would run *this* (old) code
    and import Starlette into the TUI worker.
    """
    import subprocess

    try:
        if not _mobile_plist_path().exists():
            if _mobile_healthz_answers():
                return MobileRefresh(kind="unsupervised")
            return MobileRefresh(kind="skipped")
        invocation = _mobile_restart_invocation()
        if invocation is None:
            return MobileRefresh(
                kind="failed",
                error="this interpreter vanished after the upgrade",
            )
        argv, executable = invocation
        completed = subprocess.run(
            argv,
            executable=executable,
            check=False,
            capture_output=True,
            text=True,
            timeout=_MOBILE_RESTART_TIMEOUT_S,
        )
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip()
            detail = tail.splitlines()[-1][:200] if tail else f"exit {completed.returncode}"
            return MobileRefresh(kind="failed", error=detail)
        return MobileRefresh(kind="restarted")
    except subprocess.TimeoutExpired:
        return MobileRefresh(kind="failed", error="timed out")
    except FileNotFoundError as exc:
        return MobileRefresh(kind="failed", error=str(exc))
    except Exception as exc:  # noqa: BLE001 — bounce must never fail the update
        return MobileRefresh(kind="failed", error=str(exc))


def _daemon_refresh_invocation() -> tuple[list[str], str] | None:
    """``(argv, executable)`` for the *new* distribution's daemon repair.

    Same argument as :func:`_mobile_restart_invocation`, and it matters more
    here: the repair RENDERS a plist, so running it in-process would render it
    with THIS process's already-imported (pre-upgrade) modules and write the
    previous build's plist shape — a no-op wearing the costume of a fix. The
    child is started from the wheel the installer just wrote, which is also why
    the installers are imported inside :func:`daemons_refresh_command` rather
    than here.

    No PATH ``lop`` fallback, for the reason recorded on the mobile argv: a PATH
    hit can be a different installation entirely.
    """
    if sys.executable and Path(sys.executable).exists():
        from local_operator import procname

        argv0, image = procname.spawn_identity(procname.LABEL_DAEMONS_REFRESH)
        return (
            [argv0, SAFE_PATH_FLAG, "-m", "local_operator.cli", "update", "--refresh-daemons"],
            image,
        )
    return None


def _installed_daemon_plists() -> list[Path]:
    """The supervised daemons that are installed, as plist paths.

    A pure filesystem probe that decides whether the repair is worth a child
    process at all, and it is deliberately built from ``Path.home()``: with
    ``HOME`` redirected — a test, a sandbox — it finds nothing and the whole
    repair is inert before a single ``launchctl`` is reached. The installers
    have their own, stronger identity guard; this one only has to be cheap and
    safe, because it runs on every upgrade.
    """
    directory = Path.home() / "Library" / "LaunchAgents"
    found: list[Path] = []
    for label in _DAEMON_PLIST_LABELS:
        try:
            path = directory / f"{label}.plist"
            if path.exists():
                found.append(path)
        except OSError:  # noqa: PERF203 — a probe must not fail an upgrade
            continue
    return found


def refresh_service_daemons_after_upgrade() -> DaemonRefresh:
    """Repair the supervised daemons ``lop-update`` used to leave behind.

    THE GAP THIS CLOSES: ``lop-update`` bounced mobile and touched nothing else,
    so the browser bridge, the tunnel and (until a session wrote a wake) the
    wakes supervisor kept running a plist written by whatever build installed
    them. On a machine installed before branding, that is a permanent
    ``python3.14`` row in Activity Monitor — the colleague's symptom.

    One child for all of them, because the child is the only place the NEW
    wheel's renderers exist, and because four children would pay four
    interpreter startups on every upgrade. Never raises: the upgrade already
    succeeded, so the worst outcome here is a warning.
    """
    name = "service daemons"
    if not _installed_daemon_plists():
        return DaemonRefresh(name)
    import subprocess

    invocation = _daemon_refresh_invocation()
    if invocation is None:
        return DaemonRefresh(
            name,
            warnings=(
                "warning: this interpreter vanished after the upgrade, so the "
                "installed daemons were not refreshed; run lop browser restart, "
                "lop mobile restart and lop tunnel restart to pick it up",
            ),
        )
    argv, executable = invocation
    try:
        completed = subprocess.run(
            argv,
            executable=executable,
            check=False,
            capture_output=True,
            text=True,
            timeout=_DAEMON_REFRESH_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return DaemonRefresh(name, warnings=("warning: daemon refresh timed out",))
    except Exception as exc:  # noqa: BLE001 — a failed repair must not fail the update
        warning = f"warning: could not refresh installed daemons: {exc}"
        return DaemonRefresh(name, warnings=(warning,))
    lines = tuple(line for line in (completed.stdout or "").splitlines() if line.strip())
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout or "").strip()
        detail = tail.splitlines()[-1][:200] if tail else f"exit {completed.returncode}"
        warning = f"warning: could not refresh installed daemons: {detail}"
        return DaemonRefresh(name, warnings=(warning,))
    warnings = tuple(line for line in (completed.stderr or "").splitlines() if line.strip())
    return DaemonRefresh(name, lines=lines, warnings=warnings)


def _mobile_daemon_refresh(result: MobileRefresh) -> DaemonRefresh:
    """The mobile bounce as a summary entry, in the sentences it always used.

    Moved out of the CLI's own printer unchanged, so the U1/U2 copy below lives
    in one place whatever prints it.
    """
    if result.kind == "restarted":
        return DaemonRefresh("mobile", lines=("mobile daemon restarted — refresh the phone UI",))
    if result.kind == "failed":
        # U1: name the recovery, not just the failure — the update itself
        # succeeded, so the only action left is the bounce the update could
        # not perform.
        return DaemonRefresh(
            "mobile",
            warnings=(
                f"warning: mobile daemon did not restart: {result.error}; "
                "run lop mobile restart",
            ),
        )
    if result.kind == "unsupervised":
        # U2: no LaunchAgent owns this daemon, so `lop mobile restart` is not
        # the fix — that path is launchd-only. The operator of a foreground
        # serve must stop and relaunch the process they started.
        return DaemonRefresh(
            "mobile",
            warnings=(
                "warning: a mobile daemon is running unsupervised; stop and "
                "relaunch the foreground lop mobile serve process to pick up the new UI",
            ),
        )
    return DaemonRefresh("mobile")


def refresh_daemons_after_upgrade() -> list[DaemonRefresh]:
    """Every supervised daemon this build knows, refreshed with the NEW wheel.

    Order is load-bearing. The service child runs FIRST because it REWRITES
    plists; the mobile bounce that follows is a ``mobile restart``, so running it
    first would restart mobile from the previous plist and then restart it again
    — the second start being the only one on the new definition. "First" here
    means "before", not "printed first": the service lines are printed ahead of
    the mobile line for the same reason.

    Never raises. A daemon that did not restart is a warning on a successful
    upgrade, which is the same disposition mobile has always had.

    This is the entry point ``lop update`` prints from. The TUI composes the two
    halves itself (:func:`refresh_service_daemons_after_upgrade` and
    :func:`refresh_mobile_after_upgrade`) because it renders the mobile outcome
    as its own notice with a token, before relaunching.
    """
    services = refresh_service_daemons_after_upgrade()
    mobile = _mobile_daemon_refresh(refresh_mobile_after_upgrade())
    return [services, mobile]


def _print_daemon_refreshes(refreshes: Sequence[DaemonRefresh]) -> None:
    """Report each daemon's outcome in the upgrade summary."""
    for refresh in refreshes:
        for line in refresh.lines:
            print(line)
        for warning in refresh.warnings:
            print(warning, file=sys.stderr)


def daemons_refresh_command() -> int:
    """``lop update --refresh-daemons``: the repair, run under the NEW wheel.

    Internal, and spawned by :func:`refresh_service_daemons_after_upgrade`
    rather than invoked by hand (it is reachable by hand for a machine whose
    upgrade predates this fix). Prints one line per daemon that CHANGED and one
    warning per daemon that could not be repaired; silent when everything is
    already current, because that is the normal state of a machine and this runs
    on every upgrade.

    The installer imports are function-local: ``mobile.install`` imports the
    Starlette daemon, and this module is imported by the TUI, so a module-level
    import would put the web stack in every session. In THIS process that cost
    is correct — it is a short-lived child whose entire job is the repair.

    RUNS ONLY FROM AN INSTALLED DISTRIBUTION, which is the same refusal
    ``perform_upgrade`` makes for the upgrade itself, applied at the point that
    WRITES. A source checkout must never rewrite an installed daemon: the
    ``Program`` these plists record is an INTERPRETER, so a repair from a
    worktree points the operator's daemons at that worktree's own venv —
    reproduced exactly that way during this change's development, from a
    worktree, against the live LaunchAgents (all four were restored afterwards).
    The visible entry points cannot reach this state (an editable install is
    refused before the refresh), but the hidden flag can, and the guard belongs
    where the writing happens rather than in the caller.
    """
    if install_kind() in (InstallKind.EDITABLE, InstallKind.UNKNOWN):
        print(
            "warning: installed daemons are only refreshed by an installed "
            "distribution; this is a source checkout, so nothing was touched",
            file=sys.stderr,
        )
        return 0
    from local_operator.browser_bridge import install as browser_install
    from local_operator.mobile import install as mobile_install
    from local_operator.tunnels import install as tunnel_install
    from local_operator.wakes import install as wakes_install

    refreshers = (
        mobile_install.refresh_plist_if_stale,
        browser_install.refresh_plist_if_stale,
        tunnel_install.refresh_plist_if_stale,
        wakes_install.refresh_plist_if_stale,
    )
    for refresh in refreshers:
        # Every one of these is no-raise by contract, so no guard is needed here
        # and a failure in one daemon cannot stop the next.
        outcome = refresh()
        line = outcome.summary()
        if line:
            print(line)
        warning = outcome.warning()
        if warning:
            print(warning, file=sys.stderr)
    return 0


def update_command(*, check: bool = False, refresh_daemons: bool = False) -> int:
    """``lop update``, ``lop update --check`` and the internal daemon repair.

    ``--refresh-daemons`` is not an upgrade: it is the repair step that the
    upgrade path runs in a CHILD process from the newly installed wheel, so that
    the plists it renders are this build's and not the previous one's. See
    :func:`daemons_refresh_command`. It is checked before the PyPI call because
    it must work on any machine, including one whose network is down, and it
    never reports a version. See the architect table for the other codes.
    """
    if refresh_daemons:
        return daemons_refresh_command()

    result = check_latest(force=True)
    if result.latest is None:
        print("could not reach PyPI to learn the latest version", file=sys.stderr)
        return 1

    if check:
        if result.behind:
            print(f"local-operator {result.installed}")
            print(f"latest on PyPI: {result.latest}")
            print("run `lop update` to install")
            return 2
        print(f"local-operator {result.installed} is the latest")
        return 0

    if not result.behind:
        print(f"local-operator {result.installed} is the latest")
        return 0

    kind = install_kind()
    if kind is InstallKind.EDITABLE:
        print(editable_refusal(), file=sys.stderr)
        return 1
    if kind is InstallKind.UNKNOWN:
        print(unknown_refusal(), file=sys.stderr)
        return 1

    if is_git_snapshot():
        print(git_snapshot_notice())

    print(f"local-operator {result.installed} (latest is {result.latest})")
    print(f"upgrading via {installer_label(kind)}…")
    try:
        installed = perform_upgrade(target=result.latest, kind=kind)
    except UpdateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"installed {installed}")
    _print_daemon_refreshes(refresh_daemons_after_upgrade())
    return 0
