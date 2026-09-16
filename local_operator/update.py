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
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from importlib.metadata import (
    Distribution,
    PackageNotFoundError,
    distribution,
    distributions,
    version,
)
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence
from urllib.parse import urlparse
from urllib.request import url2pathname

from local_operator.interpreter import SAFE_PATH_FLAG

logger = logging.getLogger(__name__)

#: Same cache root the model catalogue uses, so there is one place to clear.
_CACHE_DIR = Path("~/.local-operator/cache")
_CACHE_NAME = "pypi-local-operator.json"

#: The distribution name this module installs and inspects. Named once because
#: the generation reader resolves it by NAME out of a foreign tree's
#: ``site-packages`` rather than by importing it.
DISTRIBUTION_NAME = "local-operator"

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

#: The first token for an install built from a local DIRECTORY whose commit
#: could not be read (``lop update --from-snapshot <dir>``). Not hex, per
#: :func:`_looks_like_git_sha`'s constraint on new sentinels, and not
#: :data:`PYPI_SOURCE_TOKEN` either: a local build is not a wheel from the
#: index, and the marker is the one place that says where a build came from.
SNAPSHOT_SOURCE_TOKEN = "snapshot"


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
    origin: str = PYPI_SOURCE_TOKEN,
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

    ``origin`` is the first token for an install that is NOT a PyPI wheel and
    has no commit to name (a locally built directory passed to ``lop update
    --from-snapshot``). It must not be hex or it would read as a commit — see
    :func:`_looks_like_git_sha`, which anticipates exactly this — and any
    reader that does not recognise it degrades to "no ref", which for such an
    install is the truth.

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
    first = commit if commit else origin
    second = ref if commit else version
    # A bare sentinel when there is nothing to say about the second token: the
    # shape stays ``<token> [<label>]``, and no reader has to cope with a
    # trailing space that means "the label was empty".
    line = f"{first} {second}\n" if second else f"{first}\n"

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
    exc: BaseException,
    module: str,
    *,
    boot: BuildStamp | None,
    recorded_boot: BuildStamp | None = None,
) -> str | None:
    """Name a mid-install import as ``install-mid-update``, or ``None``.

    THE LEGACY SHAPE, KEPT AS A SAFETY NET. It was written for a layout that
    replaced the installed tree IN PLACE — ``lop-update`` and
    :func:`perform_upgrade` writing 929 files over the tree a long-lived process
    was importing from — so a process that loaded the old build could hit a lazy
    ``from local_operator… import x`` the new tree no longer satisfied. The
    observed shape was precise: the module still resolved, the NAME did not
    (``ImportError: cannot import name '_journal_injection_ids' from
    'local_operator.session.runtime.transcript'``, 605 times on one machine's log).

    In-place replacement is no longer how the uv-tool install works: each build
    lands in its OWN generation and a running process's tree is never rewritten
    (see ``local_operator.update``'s layout section). So for such a process this
    classifier now answers ``None`` by construction, and correctly: its own tree
    did not move, so an ImportError IS ours and must stay an ordinary
    traceback. It is kept because three populations still have the old shape and
    a half-replaced tree is exactly what they would see — a pip/pipx install
    (neither has a layout this product can make atomic), a process launched
    before this machine migrated, and a ``lop`` started out of the old fixed
    uv-tool tree.

    What separates that from a genuine packaging bug is the STAMP MOVING UNDER
    THE PROCESS. If the install on disk still matches the build this process
    booted from, the miss is ours and must stay an ordinary traceback — so
    ``None``. ``installed_build`` is the right question here rather than
    ``disk_build``: it asks whether THIS process's tree moved, and a pointer
    that moved onto another generation is not that (the files this process
    imports were never touched). A ``boot`` we could not read (``None``) also
    answers ``None``: without a baseline there is nothing to compare, and
    guessing here would relabel a real packaging error as an install race.

    ``module`` is what the CALLER was importing, used when the exception itself
    names nothing (some wrappers drop ``name``).

    ``recorded_boot`` IS THE DURABLE SECOND OPINION, consulted ONLY when the
    live stamp is unavailable. ``session/runtime/journal.py`` publishes a boot
    record per runtime (``run/host/<pid>.json``) holding the build that process
    was born on, and which survives precisely the case this function exists to
    name: an install torn badly enough that ``installed_build()`` — which reads
    the dist-info directory and ``.lop-source`` off the very tree being
    replaced — cannot answer. Before this, that case fell to ``boot is None``
    and the tear was reported as a genuine packaging bug.

    The live stamp still WINS whenever it is readable, and not as a matter of
    taste: it is the same process's own reading at the same instant, while the
    recorded one is a snapshot from boot time. Consulting the record first would
    make a process that has legitimately been re-pointed at a new install
    compare against a stale baseline. When neither is readable the answer stays
    ``None``, which is the legacy path unchanged.
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
        boot = recorded_boot
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
    and takes no prefix. In the PRE-generation layout the two were the same
    tree — a runtime's ``prefix`` WAS its own install — so the distinction was
    invisible. It is visible in exactly two shapes now, and both are safe:
    the ``LOP_BUILD_PREFIX`` test seam, where a caller passing a foreign prefix
    gets an age mixing that prefix's marker with this interpreter's dist-info
    (review round 1, R1-2); and the generation layout's disk read, where the
    marker comes from the generation the POINTER names and the dist-info from
    this process's own — an OLDER mtime, so the max is still the fresh marker
    and the settle asks exactly the question it is meant to ("has the install
    the pointer just moved to stopped being written?").

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


# ---------------------------------------------------------------------------
# The generation layout
# ---------------------------------------------------------------------------
#
# WHY (2026-09-15, measured on the reporting host)
# -----------------------------------------------
# ``uv tool install --force`` RECREATES ``~/.local/share/uv/tools/local-operator``
# in place. Every process importing from that tree was reading files that were
# being deleted and rewritten underneath it: 36 sessions died with no exit
# record ("the runtime disappeared without exiting cleanly while this turn was
# running"), and 113 crash reports in ~/Library/Logs/DiagnosticReports named the
# planted libpython dylib, clustered inside the install window — a process
# LAUNCHED during the rewrite dies at load. The runtime's own self-refresh is
# idle-gated BY DESIGN (a busy one never checks), so a runtime with work in
# flight had no defence at all.
#
# There is no fix available inside that shape: the installer owns the tree, and
# a tree that is rewritten in place cannot be handed over from. So each build
# gets its OWN generation root and the stable path becomes a POINTER resolved
# once per process, at exec:
#
#     ~/.local/bin/lop -\
#     ~/.local/bin/local-operator --+--> ~/.local/share/lop/current
#                                             |
#                                             v
#                       ~/.local/share/lop/generations/<id>/
#                           bin/lop          -> tools/local-operator/bin/lop
#                           tools/local-operator/         (the venv, sys.prefix)
#                           tools/local-operator/.lop-source
#
# Nothing a running process holds is ever rewritten: its ``sys.path`` names the
# generation it was launched from, and superseding it is a temp symlink plus
# ``os.rename`` — atomic, with no instant at which ``current`` is absent or
# unresolved. A ``lop`` exec that races the flip therefore resolves either the
# old generation or the new one, never nothing.
#
# RESOLUTION IS DONE BY THE KERNEL, NOT BY ``pwd``. ``~/.local/bin/lop`` is a
# symlink chain and a uv console script's shebang names its OWN generation's
# interpreter ABSOLUTELY, so the child's ``sys.prefix`` comes out concrete
# (verified: launching ``<gen>/tools/local-operator/bin/python3`` through
# ``current`` reports the pointer path, because CPython detects a venv from the
# invoked path's parent directory — which is why the spawn sites resolve the
# pointer themselves rather than handing ``current`` to a child; see
# :func:`current_interpreter`).
#
# WHAT THIS MAKES OF THE OLD DEFENCES. The build watch, the settle window and
# the files-gone probe all still exist and still work; after this they are a
# CONVERGENCE path rather than a safety path. A mixed-generation fleet is an
# accepted steady state: a runtime older than ``current`` keeps serving until it
# goes idle, and the next engage constructs on the current build.

#: The stable root. Deliberately NOT under ``~/.local/share/uv/tools``: uv owns
#: that tree and rewrites it, which is the whole incident. Read through
#: ``Path(_STABLE_ROOT).expanduser()`` at every use so a redirected ``HOME``
#: (an isolated test, a sandbox) moves it — an absolute path captured at import
#: time would write into the operator's real home from inside a sandbox.
_STABLE_ROOT = "~/.local/share/lop"

#: Where the console scripts uv would normally write land, so the stable
#: launchers can be recognised and refreshed without a second constant.
_LOCAL_BIN = "~/.local/bin"

#: How many UNREFERENCED generations survive a prune. "Unreferenced" means no
#: live or persisted session record names it and it is not the pointer's
#: target, so this is the whole margin for a session that has a generation in
#: flight but no record yet (an engage's first ~1.2 s) and for a terminal whose
#: record has aged out. Two rather than one because the previous generation is
#: exactly the one a just-flipped fleet is still reading from.
DEFAULT_KEEP_GENERATIONS = 2

#: Age at which a generation with no ``.lop-source`` marker is crash debris
#: rather than an install in flight. Nothing else can leave one: every failure
#: path removes its own tree, and a finished generation always carries a marker
#: (written before its first flip) — so "no marker yet" means "uv is still
#: working in there", and only a ``kill -9`` stretches that past an hour.
_PARTIAL_TTL_S = 3600.0


def stable_root() -> Path:
    """The one directory whose path never changes for the life of the machine."""
    return Path(_STABLE_ROOT).expanduser()


def generations_dir() -> Path:
    """Where the generation roots live, one per build."""
    return stable_root() / "generations"


def pointer_path() -> Path:
    """The ``current`` symlink: the single mutable artefact of the layout.

    Everything else in a generation is written once and never touched, so this
    is the only file a flip has to be atomic about.
    """
    return stable_root() / "current"


def daemon_image_path() -> Path:
    """The stable interpreter path a supervised unit (launchd, systemd) names.

    A SHIM rather than a symlink to the current generation's interpreter, and
    the difference is measured rather than stylistic: CPython decides "am I in a
    venv" from the parent directory of the path it was EXECUTED through, so a
    symlink at this path loses the venv entirely (verified: ``sys.prefix`` came
    out as the uv-managed base interpreter, with no ``site-packages``), while a
    script that resolves the pointer itself and execs the concrete path keeps it.
    It is also the only shape that survives a prune: the unit names THIS path,
    so a restart after a flip or a prune cannot hit the deleted libpython
    dylib pin that killed 113 processes on 2026-09-15.
    """
    return stable_root() / "bin" / "python3"


#: The shim above, written verbatim by :func:`ensure_daemon_image`. Kept as one
#: constant so the file on disk is comparable byte-for-byte — a rewrite only
#: happens when it genuinely changed.
_DAEMON_SHIM = """#!/bin/sh
# lop's supervised daemons (launchd LaunchAgents, systemd user units) name THIS
# path. It resolves the install pointer ONCE, here, and execs that generation's
# interpreter by an absolute path, so the child's sys.path names its own
# generation rather than the mutable `current` symlink -- and a restart after a
# flip or a prune can never hit the libpython dylib pin of a tree that is gone.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd -P)
gen=$(CDPATH= cd -- "$here/../current" 2>/dev/null && pwd -P) || gen=""
if [ -z "$gen" ]; then
    echo "lop: no current install generation ($here/../current is unreadable)" >&2
    exit 78
fi
# Prefer the branded image when it is planted, because that is what Activity
# Monitor reads (p_comm); the interpreter is the always-present fallback.
if [ -x "$gen/tools/local-operator/bin/Local Operator" ]; then
    exec "$gen/tools/local-operator/bin/Local Operator" "$@"
fi
exec "$gen/tools/local-operator/bin/python3" "$@"
"""


def _venv_interpreter(root: Path) -> Path:
    """The interpreter inside the venv ``root``, spelled as its platform does."""
    if os.name == "nt":  # pragma: no cover — the layout below is POSIX-shaped
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python3"


def _generation_install_root(generation: Path) -> Path:
    """The venv (``sys.prefix``) of one generation root.

    Spelled once because three readers depend on agreeing: the installer aims uv
    at it, ``.lop-source`` is written into it, and pruning matches a record's
    ``install_root`` against it.
    """
    return generation / "tools" / DISTRIBUTION_NAME


def current_generation() -> Path | None:
    """The generation root ``current`` resolves to, or ``None``.

    WHAT THIS GUARANTEES, stated narrowly because the narrow version is the one
    the callers can rely on: a DANGLE is impossible — the pointer is replaced by
    ``os.rename``, so it always names a generation that exists, and a pointer to
    a tree that is gone (or to nothing at all) answers ``None``. A READ CAN
    STILL FAIL, transiently, on the platform this ships on: macOS raises
    ``OSError: [Errno 22] Invalid argument`` from both ``os.readlink`` and
    ``Path.resolve`` while a symlink is being renamed underneath the reader
    (reproduced by a tight rename loop; the numbers are in
    ``docs/design-install-generations.md`` §3.2). Every caller has a documented
    fallback for that answer — a spawn falls back to this process's interpreter,
    the build watch sees no move, and :func:`prune_generations` REFUSES TO
    DELETE ANYTHING, because "I could not read which tree is live" must never
    be answered by deleting trees.

    ``os.readlink`` rather than ``Path.resolve`` on top of that, because it also
    answers the right QUESTION: readlink returns what the link NAMES (one check,
    no chasing), which is what the caller wants to pass on; ``resolve`` walks the
    chain and can be caught between steps, returning a path the link had already
    stopped naming.
    """
    try:
        target = os.readlink(pointer_path())
    except OSError:
        return None
    generation = Path(target)
    return generation if generation.is_dir() else None


def current_install_root() -> Path | None:
    """The install root the POINTER resolves to, or ``None``.

    This is "what a fresh ``lop`` would load", which is NOT necessarily what
    this process loaded — the two differ for the whole mixed-generation window
    and comparing them is the point (see :func:`disk_build` and
    ``buildwatch.build_changed``).

    ``LOP_INSTALL_ROOT`` overrides it, test-only, exactly as
    ``LOP_BUILD_PREFIX`` does for the boot sample: the e2e stage has to be able
    to point a real process at a generation without owning the host's pointer.
    """
    override = os.environ.get("LOP_INSTALL_ROOT", "")
    if override:
        return Path(override)
    generation = current_generation()
    if generation is None:
        return None
    return _generation_install_root(generation)


def current_interpreter() -> Path | None:
    """The interpreter of the CURRENT generation, or ``None`` if unreadable.

    CONCRETE, never through ``current``: the pointer is resolved here, once, so
    the child's ``sys.prefix`` and ``sys.path`` name a generation that no later
    flip can redirect. Handing ``<pointer>/bin/python3`` to a child instead
    would leave it importing through the mutable symlink — the failure this
    whole layout exists to remove.
    """
    root = current_install_root()
    if root is None:
        return None
    candidate = _venv_interpreter(root)
    if not os.access(candidate, os.X_OK):
        return None
    return candidate


def process_install_root() -> str:
    """The install root THIS process imports from, as a path string.

    Its own generation, not the pointer's: that is what a session record has to
    name so pruning can never delete a tree a live session is still reading
    from. Resolved, because a process launched through the pointer would
    otherwise name the mutable path and the record would follow a later flip.
    """
    try:
        return str(Path(sys.prefix).resolve())
    except OSError:  # pragma: no cover — an unresolvable prefix is not fatal
        return str(sys.prefix)


def _site_packages(root: Path) -> Path | None:
    """The ``site-packages`` directory inside the venv ``root``, or ``None``."""
    candidates = [root / "Lib" / "site-packages"]
    candidates.extend(sorted((root / "lib").glob("python*/site-packages")))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return None


def _distribution_at(root: Path) -> Distribution | None:
    """The ``local-operator`` distribution installed under ``root``, or ``None``.

    SCOPED ON PURPOSE. ``importlib.metadata`` resolves through ``sys.path``
    unless it is handed a path, and a generation is by construction NOT on this
    process's path — that is the whole layout — so a bare ``distribution()``
    would answer about the running build no matter which root was asked about.
    ``distributions(path=...)`` is the documented scoped read, and the name
    comparison is what keeps it to our distribution inside a tree that also
    carries every dependency.
    """
    site = _site_packages(root)
    if site is None:
        return None
    try:
        for found in distributions(path=[str(site)]):
            name = str(found.metadata["Name"] or "").lower().replace("_", "-")
            if name == DISTRIBUTION_NAME:
                return found
    except Exception:  # noqa: BLE001 — an unreadable tree is "no answer here"
        logger.debug("distribution lookup failed under %s", root, exc_info=True)
    return None


def _stamp_at(root: Path) -> BuildStamp | None:
    """The stamp of the install sitting at ``root``, or ``None``.

    Both halves come from that tree: the version out of its own ``dist-info``
    (via :func:`_distribution_at`, which scopes the lookup to that tree's
    ``site-packages``) and the ref out of its own ``.lop-source``.
    """
    found = _distribution_at(root)
    if found is None:
        return None
    return BuildStamp(version=found.version, source_ref=source_ref(root))


def disk_build(root: str | Path | None = None) -> BuildStamp | None:
    """The build a FRESH ``lop`` would load, or ``None`` when there is none.

    The counterpart of :func:`installed_build`, which keeps "THIS process"
    semantics: ``lop --version`` and a runtime's boot sample must describe the
    code in memory, while the build watch, the TUI's skew notice and ``lop
    refresh`` all have to describe the POINTER. Comparing the two is the only
    way "the install moved under me" survives a layout where the running tree is
    never rewritten.

    ``None`` covers three shapes, all of them "no install to fall behind":

    * **this process is not an install at all** — an editable checkout's install
      on disk is its own working tree. Left unguarded, a developer's runtime
      would read the global pointer, retire on a flip and be respawned onto the
      installed ``lop``: a worktree session silently converted into a global
      one, which is the failure ``design-build-skew`` §6.5 rules out;
    * **the pointer is unreadable** — no migration yet, a pruned target, an
      interrupted flip. Every caller treats it as "no evidence of a move", which
      is the safe direction;
    * **the generation carries no distribution** — then the version half cannot
      be answered honestly, and a version-only stamp would be read as a move by
      whoever compares labels (see ``buildwatch.proves_a_move``).

    ``root`` overrides where the stamp is read from, and it is the e2e seam
    (``LOP_BUILD_PREFIX``): a temp directory carrying a fake ``.lop-source`` and
    no distribution of its own, so — exactly as before this layout — the VERSION
    half comes from this interpreter and only the ref is read there.
    """
    if root is not None:
        target = Path(root)
        found = _distribution_at(target)
        if found is not None:
            # A tree that carries its own metadata answers wholesale, which is
            # what makes this usable for any root and not just the seam.
            return BuildStamp(version=found.version, source_ref=source_ref(target))
        return BuildStamp(version=installed_version(), source_ref=source_ref(target))
    try:
        kind = install_kind()
    except Exception:  # noqa: BLE001 — an unreadable kind is "no install"
        return None
    if kind in (InstallKind.EDITABLE, InstallKind.UNKNOWN):
        return None
    target = current_install_root()
    if target is None:
        return None
    return _stamp_at(target)


def _new_generation_id(token: str, attempt: int = 1) -> str:
    """A sortable, collision-free directory name for one generation.

    Timestamp first so ``ls`` reads chronologically and a prune can use name
    order and mtime interchangeably; the build token (short commit, or the
    version, or ``pypi``) rides along so a human can tell two generations apart
    without opening them. ``attempt`` is what makes two installs starting in the
    same second land in two directories instead of one.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    safe = re.sub(r"[^0-9A-Za-z.+-]", "-", token).strip("-") or "build"
    base = f"{stamp}-{safe[:24]}"
    return base if attempt <= 1 else f"{base}-{attempt}"


def _reserve_generation(token: str) -> Path:
    """Create this generation's directory exclusively, and return it.

    ``os.mkdir`` WITHOUT ``exist_ok`` is the whole point: two installs starting
    in the same second must not both aim uv at one path, because uv would then
    upgrade the tree the other one is still writing instead of building beside
    it. The loser takes the next name.
    """
    try:
        generations_dir().mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise UpdateError(f"could not create {generations_dir()}: {exc}") from exc
    for attempt in range(1, 100):
        candidate = generations_dir() / _new_generation_id(token, attempt)
        try:
            os.mkdir(candidate)
        except FileExistsError:
            continue
        except OSError as exc:
            raise UpdateError(f"could not create {candidate}: {exc}") from exc
        return candidate
    raise UpdateError("too many generations share one timestamp")


def _generation_env(generation: Path) -> dict[str, str]:
    """The environment that aims uv at ONE generation root.

    ``UV_TOOL_DIR``/``UV_TOOL_BIN_DIR`` are the whole mechanism — uv honours both
    (verified on uv 0.9.x: ``<gen>/tools/local-operator`` becomes the venv and the
    console scripts land in ``<gen>/bin``). The bin directory exists only to keep
    uv away from the REAL ``~/.local/bin``, whose entries are this layout's
    stable launchers and must not be rewritten by an installer.
    """
    return {
        **os.environ,
        "UV_TOOL_DIR": str(generation / "tools"),
        "UV_TOOL_BIN_DIR": str(generation / "bin"),
    }


def _remove_tree(path: Path) -> bool:
    """Best-effort removal of a tree this module owns; ``True`` when it is gone.

    Never raises: it runs on the failure paths of an install, where the error the
    caller is about to report is the one worth keeping.

    IT RESTORES WRITE BITS AND RETRIES, because the trees this is called on can be
    read-only. The shape is not hypothetical: a migration from a ``bin/`` without
    write permission fails the rebinding step (that is the shape the refusal is
    for), and ``shutil.rmtree`` needs write permission on every directory it
    empties — so the refusal promising "the copy is removed" left the ~136 MB copy
    behind, and ``lop install prune``, which removes through this same helper,
    reported such a generation as removed while it stayed on disk (review round 3,
    R3-2). The retry only ever ADDS write permission, and only to paths that are
    really INSIDE the tree it was asked to delete — see ``_inside``, because the
    obvious version of that scoping was wrong (review round 4, R4-1). That
    combination is what makes it safe in a function that must not raise.

    The return value exists for the same reason: a caller that PRINTS a removal
    has to know whether one happened.
    """
    try:
        present = path.exists() or path.is_symlink()
    except (OSError, RuntimeError):
        # ``Path.exists`` reaches ELOOP through ``stat()``, and pathlib raises
        # that as a RuntimeError rather than answering False (measured on 3.12:
        # a self-referential link). Something is there under a name this process
        # cannot resolve, and "there" is the answer this guard needs.
        present = True
    if not present:
        return True

    try:
        root = path.resolve()
    except (OSError, RuntimeError):
        # A dangling path, or a symlink LOOP — which is the input rmtree is
        # about to refuse, and the reason ``resolve()`` cannot be left unguarded
        # in a function documented never to raise (review round 5, R5-2: the
        # loop escaped as a RuntimeError where the previous head returned
        # False). Falling back to the spelling we were given keeps the
        # containment test below meaningful.
        root = path

    def _inside(candidate: Path) -> bool:
        """Is ``candidate`` a REAL path inside the tree being deleted?

        BOTH HALVES ARE LOAD-BEARING. ``shutil.rmtree`` reports a top-level
        SYMLINK by handing the callback ``os.path.islink`` and the link itself,
        and the unlink of a symlink entry does the same with the entry — while
        ``stat`` and ``chmod`` FOLLOW links. Restoring owner bits through one
        would reach the link's target, a directory this call was never asked to
        touch (review round 4, R4-1, measured: the retry chmod'ed the target of a
        link handed to ``rmtree``). ``resolve()`` alone is not enough either: it
        would resolve the link INTO ``root`` and answer "inside" for a path that
        is not the tree.
        """
        try:
            if candidate.is_symlink():
                return False
            return candidate.resolve().is_relative_to(root)
        except (OSError, RuntimeError):  # pragma: no cover — gone, or a loop
            return False

    def _with_write_bits(
        function: Callable[[str], object], target: str, error: BaseException
    ) -> None:
        # ``rmtree`` hands us the call that failed. For an unlink inside a
        # directory that is not writable, the missing bit is the PARENT's, so both
        # it and the target get their owner bits back before the one retry.
        for candidate in {Path(target).parent, Path(target)}:
            if not _inside(candidate):
                continue
            try:
                os.chmod(candidate, os.stat(candidate).st_mode | 0o700)
            except OSError:  # pragma: no cover — gone, or not ours to chmod
                # ``exc_info=True``, not the exception that triggered the handler:
                # that traceback is about a different call (review round 4, R4-2).
                logger.debug("could not make %s writable", candidate, exc_info=True)
        if function not in (os.unlink, os.rmdir):
            # ONLY THE TWO ONE-PATH REMOVALS ARE RETRIED, which is a different
            # test from the one this had. ``rmtree`` reports two other shapes:
            #
            # * ``os.path.islink`` — the TOP-LEVEL notification, where the path it
            #   was handed is a symlink and it refused to touch it;
            # * ``os.open`` — its own ``ELOOP`` from the fd walk, which re-issued
            #   with a single path is a ``TypeError``, not a retry (measured:
            #   ``open() missing required argument 'flags'``; QA round 3, Q3 — the
            #   function's "never raises" contract was still broken on a loop).
            #
            # The INNER form — unlinking a symlink ENTRY inside the tree — arrives
            # as ``os.unlink`` with a target that is also a symlink, and that retry
            # is both needed and safe: ``unlink`` removes the link itself, never its
            # target. Gating on ``Path(target).is_symlink()`` instead made a
            # read-only ``bin/`` holding a venv's symlinks unremovable — the R3-2
            # shape, reintroduced by its own fix (review round 5, R5-1).
            return
        try:
            function(target)
        except (OSError, TypeError):
            # TypeError for the same reason: a callable this code cannot re-issue
            # with a path must not take the caller down with it.
            logger.debug("could not remove %s", target, exc_info=True)

    try:
        shutil.rmtree(path, onexc=_with_write_bits)
    except OSError:
        logger.debug("could not remove %s", path, exc_info=True)
    if path.exists() or path.is_symlink():
        # Announced rather than swallowed: every caller's message about this tree
        # (a refusal, a ``removed:`` line) claims it is gone.
        logger.warning("could not remove %s; it is still on disk", path)
        return False
    return True


def flip_pointer(generation: Path) -> None:
    """Point ``current`` at ``generation``, atomically.

    A staged symlink plus ``os.rename``, and the staging name is a SIBLING of the
    pointer so the rename cannot cross a filesystem. ``os.rename`` over an
    existing symlink is the atomic step: there is no instant at which
    ``current`` is missing or dangling, so a process that execs ``lop`` while a
    flip is in flight resolves one generation or the other. The generation is
    checked to exist first — a pointer to a directory that is not there is the
    one state every reader of this layout gets to avoid by construction.
    """
    if not generation.is_dir():
        raise UpdateError(f"refusing to point current at a missing generation: {generation}")
    pointer = pointer_path()
    pointer.parent.mkdir(parents=True, exist_ok=True)
    staged = pointer.with_name(f"{pointer.name}.tmp-{os.getpid()}")
    try:
        staged.unlink(missing_ok=True)
        os.symlink(generation, staged)
        os.rename(staged, pointer)
    finally:
        try:
            staged.unlink(missing_ok=True)
        except OSError:  # pragma: no cover — already renamed, or unreadable
            pass


def _write_executable(path: Path, text: str) -> bool:
    """Write ``text`` at ``path`` mode 0755, atomically, if it differs.

    The comparison is what keeps the common path free of writes: a daemon
    installer runs on every ``lop update``, and rewriting a shim that is already
    byte-identical would churn the file's mtime for nothing. The write itself is
    a temp file plus ``os.rename`` in the same directory, so no reader can
    observe a half-written shim (a launchd restart landing mid-write would
    otherwise execute whatever prefix had reached the disk).
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and path.read_text(encoding="utf-8") == text:
            return True
        handle, name = tempfile.mkstemp(dir=str(path.parent), prefix=f"{path.name}.", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(name, 0o755)
            os.rename(name, path)
        except BaseException:
            Path(name).unlink(missing_ok=True)
            raise
        return True
    except OSError:
        logger.debug("could not write %s", path, exc_info=True)
        return False


def _may_name_the_shim() -> bool:
    """May THIS process render a plist / systemd unit against the stable shim?

    TWO SHAPES QUALIFY, and the second is the one review round 1 (R-2) found
    missing: a process that already runs out of a generation (the ordinary case),
    and an INSTALLED distribution that is not one yet — ``uv tool``, pipx or pip
    (every kind except ``editable``/``unknown``) on a machine whose pointer has
    already moved to a generation, which is exactly the post-migration state. In
    that state the shim exists and names the current build, so a unit rendered
    against it runs the machine's install; keeping the
    legacy venv in the plist instead would name the tree that is no longer
    current.

    A SOURCE CHECKOUT never qualifies, and that is the guard this predicate
    inherits from ``_repair_refusal``: a dev tree must not repoint the operator's
    daemons at itself, and rendering a unit that runs the machine's install from
    a worktree would be the same surprise in the other direction.
    """
    if _is_generation_install():
        return True
    return install_kind() in (InstallKind.UV_TOOL, InstallKind.PIPX, InstallKind.PIP)


def daemon_image() -> Path | None:
    """The stable interpreter path a supervised unit should name, or ``None``.

    THE READ HALF of :func:`ensure_daemon_image`, and separate from it because
    rendering a plist is not allowed to write: ``lop mobile status`` and every
    test of ``render_plist`` go through here, and a status command that plants a
    file is a surprise with no upside.

    ``None`` unless this process may name a machine-level artefact
    (:func:`_may_name_the_shim`) AND the machine has a pointer AND the shim is
    already there. All three matter: a source checkout has no business naming the
    operator's install, a machine with no layout has nothing to point at, and a
    shim that does not exist is a unit that cannot start. With ``None`` every
    caller keeps its pre-generation shape, which still works.
    """
    if not _may_name_the_shim():
        return None
    if not pointer_path().is_symlink():
        return None
    path = daemon_image_path()
    return path if path.is_file() else None


def ensure_daemon_image(generation: Path | None = None) -> Path | None:
    """Write the stable interpreter shim supervised units name, or ``None``.

    Called from the install paths (a new generation, and the migration), so the
    file a plist names exists before any plist is rendered against it.

    ``generation`` is how a caller that is NOT itself a generation install asks
    for the shim anyway, and the migration is exactly that caller: it runs from
    the legacy tree while creating a generation, so a gate on "is THIS process a
    generation install" answered ``None`` there and left the shim unwritten.
    Measured after a real ``lop install migrate`` (QA round 1, Q3: ``<stable>/bin/
    python3`` did not exist) and named by review round 1 (R-2): the four
    installers then kept rendering plists that name a path inside the legacy
    venv — the shape this shim exists to remove. The shim is a MACHINE-level
    artefact (it resolves ``current`` for whatever generation is current), so a
    caller that has just created a generation may offer it.

    An explicitly passed path must still BE one of our generations. This is the
    one writer of a file the operator's daemons will execute, and "a caller said
    so" is not a licence to plant it anywhere.
    """
    if generation is not None:
        if not _is_generation_install(generation):
            return None
    elif not _may_name_the_shim():
        return None
    if not pointer_path().is_symlink():
        # The shim execs ``current``. With no pointer there is nothing for it to
        # name, and a shim that can only exit 78 is worse than no shim: the
        # installers would render a plist naming a program that is guaranteed to
        # fail. Asked of the LINK rather than of ``current_generation()`` so a
        # transient read failure cannot silently skip the write.
        return None
    path = daemon_image_path()
    return path if _write_executable(path, _DAEMON_SHIM) else None


def _is_generation_install(root: Path | None = None) -> bool:
    """Is ``root`` (this process's own install by default) one of OUR generations?

    Asked of the path rather than of a marker file because the path is the
    claim: a generation lives under ``generations_dir()`` and nothing else does.
    """
    candidate = Path(process_install_root()) if root is None else root
    try:
        resolved = candidate.resolve()
        generations = generations_dir().resolve()
    except OSError:  # pragma: no cover — an unresolvable path is not a generation
        return False
    return generations in resolved.parents


def _local_bin_dir() -> Path:
    """``~/.local/bin``: where the stable launchers live."""
    return Path(_LOCAL_BIN).expanduser()


def _atomic_symlink(link: Path, target: Path) -> bool:
    """Point ``link`` at ``target`` with a rename, so no reader sees a gap."""
    try:
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink() and os.readlink(link) == str(target):
            return True
        staged = link.with_name(f"{link.name}.tmp-{os.getpid()}")
        staged.unlink(missing_ok=True)
        os.symlink(target, staged)
        os.rename(staged, link)
        return True
    except OSError:
        logger.debug("could not point %s at %s", link, target, exc_info=True)
        return False


#: An absolute POSIX path inside a text file: a shebang target, or a
#: ``VIRTUAL_ENV=`` value. Deliberately not exhaustive — it exists to find the
#: paths uv writes into an installed tree, and anything it does not match is left
#: byte-identical.
_ABSOLUTE_PATH = re.compile(rb"(?<![A-Za-z0-9_.\-/])(/[^\s'\"\n]*)")


def _repoint_paths(data: bytes, source_root: Path, install_root: Path) -> bytes:
    """Rewrite every absolute path in ``data`` whose PREFIX is ``source_root``.

    THE HALF OF THE REBINDING THAT MAKES IT WORK ON A REAL MACHINE.
    ``str(source_root)`` is the spelling the caller has, and the caller derives it
    from ``sys.prefix`` — which CPython has already RESOLVED. The spelling in the
    file is the one ``uv`` was given when the tree was installed, and on any host
    where the install path has a symlinked component the two never match:

        file:   #!/tmp/…/legacy/tools/local-operator/bin/python     (uv's spelling)
        caller: /private/tmp/…/legacy/tools/local-operator          (sys.prefix)

    A string search for the caller's spelling therefore found nothing, nothing
    failed, and the migration reported success with a copy that still executed the
    legacy venv (design review round 1, D2 — reproduced with the design's own §8
    walkthrough shape under an isolated ``HOME``, which on macOS is usually under
    ``/tmp``). This compares the venv's IDENTITY instead, so every spelling of it
    is found.

    THE MATCH IS ON A PREFIX, RESOLVED ONE COMPONENT AT A TIME, and that detail is
    load-bearing rather than fussy: resolving the WHOLE token does not work for
    the shebang, because a venv's ``bin/python3`` is itself a symlink to the base
    interpreter, so the full path resolves straight OUT of the venv (measured: a
    shebang naming the legacy tree resolved to ``/usr/bin/python3`` and was left
    untouched by the first version of this). The longest prefix that resolves to
    the source root is what gets rewritten; the remainder of the path — ``/bin/
    python3`` — is preserved verbatim.
    """
    real_source = str(_real(source_root))
    out = bytearray()
    cursor = 0
    for match in _ABSOLUTE_PATH.finditer(data):
        token = match.group(1)
        try:
            spelled = token.decode("utf-8")
        except UnicodeDecodeError:  # pragma: no cover — not text we wrote
            continue
        parts = spelled.split("/")
        matched = 0
        for cut in range(2, len(parts) + 1):
            try:
                if os.path.realpath("/".join(parts[:cut])) == real_source:
                    matched = cut
            except (OSError, ValueError):  # pragma: no cover — a path the OS rejects
                break
        if not matched:
            continue
        prefix = "/".join(parts[:matched])
        out += data[cursor : match.start(1)]
        out += str(install_root).encode("utf-8") + spelled[len(prefix) :].encode("utf-8")
        cursor = match.end(1)
    if not cursor:
        return data
    out += data[cursor:]
    return bytes(out)


def _rebind_scripts(install_root: Path, source_root: Path) -> tuple[list[Path], list[Path]]:
    """Point a COPIED tree's own scripts at itself instead of at the original.

    THE MIGRATION'S ONE REWRITE, and without it the migration does not work at
    all. ``uv tool install`` writes every console script with an ABSOLUTE
    shebang naming the venv it was installed into, so a verbatim copy executes
    the tree it was copied from:

        #!/Users/…/.local/share/uv/tools/local-operator/bin/python3

    Measured on the reporting host (review round 1, R-1): the chain
    ``~/.local/bin/lop → current/bin/lop → <gen>/…/bin/lop`` ended at the LEGACY
    interpreter, so every process started from the stable launcher after ``lop
    install migrate`` still had ``sys.prefix`` = ``~/.local/share/uv/tools/
    local-operator`` — the tree the host rewrites in place, which is exactly the
    hazard this layout exists to remove.

    Rewrites every TEXT file under ``<install_root>/bin`` that names the source
    root: the shebang of each console script, and the ``VIRTUAL_ENV`` line in
    ``activate``/``activate.csh``/``activate.fish``, which is the same claim in
    another form. The search is on the VENV'S IDENTITY rather than on a
    spelling: every absolute path in the file is resolved and compared against
    the resolved source root, so a file written with the unresolved spelling is
    re-pointed too (see :func:`_repoint_paths` — a plain string search missed
    exactly that case, design review round 1, D2). Symlinks are skipped:
    ``bin/python3`` points at a base interpreter OUTSIDE the tree,
    which must not be touched. Anything with a NUL byte in its first block is
    skipped as binary.

    Written atomically with the mode preserved, because a half-rewritten script
    is a script ``lop`` cannot execute and the pointer is about to name this
    tree. Bytes, not text: nothing here decodes or re-encodes a file it does not
    change.

    RETURNS WHAT IT COULD NOT REWRITE rather than only logging it, because the
    caller's promise IS this rewrite: a migration that flipped the pointer with a
    console script still naming the legacy venv would print "copied … into …" and
    hand the operator the R-1 symptom (review round 2, R2-4). The one caller
    fails the migration on a non-empty second element.
    """
    bin_dir = install_root / ("Scripts" if os.name == "nt" else "bin")
    if not bin_dir.is_dir():
        return [], []
    replacements = [(str(source_root).encode(), str(install_root).encode())]
    rewritten: list[Path] = []
    failed: list[Path] = []
    for entry in sorted(bin_dir.iterdir()):
        if entry.is_symlink() or not entry.is_file():
            continue
        try:
            data = entry.read_bytes()
        except OSError:
            # A file this process cannot read is very likely one it cannot
            # rewrite either, so it is reported rather than skipped: the caller
            # decides what a partial migration means, and "silently unchanged" is
            # the one answer it must not get.
            failed.append(entry)
            continue
        if b"\x00" in data[:4096]:
            continue
        patched = data
        for old, new in replacements:
            patched = patched.replace(old, new)
        patched = _repoint_paths(patched, source_root, install_root)
        if patched == data:
            continue
        staged = entry.with_name(f"{entry.name}.rebind-{os.getpid()}")
        try:
            mode = entry.stat().st_mode
            staged.write_bytes(patched)
            os.chmod(staged, mode & 0o7777)
            os.rename(staged, entry)
        except OSError:
            # The temp goes with the failure: ``bin/`` is meant to be a closed
            # set, and a ``.rebind-<pid>`` left inside a generation is litter in
            # the one directory a future reader globs (review round 2, R2-5).
            try:
                staged.unlink(missing_ok=True)
            except OSError:  # pragma: no cover — nothing further to do about it
                pass
            logger.warning("could not rebind %s to %s", entry, install_root, exc_info=True)
            failed.append(entry)
            continue
        rewritten.append(entry)
    return rewritten, failed


def _link_generation_bin(generation: Path) -> None:
    """Give a generation its own ``bin``, for a generation uv did not build.

    ``install_into_generation`` gets this for free: uv is aimed at
    ``<generation>/bin`` with ``UV_TOOL_BIN_DIR`` and writes the console-script
    shims there itself. :func:`clone_into_generation` runs no installer, so it
    has to lay the same shape down by hand — and it MUST, because that directory
    is what the stable launchers point through (``~/.local/bin/lop ->
    <stable>/current/bin/lop``): a generation without it leaves every launcher on
    the machine DANGLING, which is a worse state than not having migrated.

    Found by the live walkthrough, not by the unit tests: the migration's
    "copied" and "pointer" lines printed cleanly while ``~/.local/bin/lop``
    pointed at nothing.
    """
    install_root = _generation_install_root(generation)
    interpreter_dir = "Scripts" if os.name == "nt" else "bin"
    for name in _console_script_names(install_root) or (DISTRIBUTION_NAME, "lop"):
        script = install_root / interpreter_dir / name
        if script.exists():
            _atomic_symlink(generation / interpreter_dir / name, script)


def write_stable_launchers(generation: Path) -> tuple[list[Path], list[Path]]:
    """Make ``~/.local/bin`` name the pointer for every console script.

    ``~/.local/bin/lop -> <stable>/current/bin/lop``: one file per entry point,
    written once per install and never otherwise touched, resolving through
    ``current`` at exec so it needs no maintenance on a flip. The entry points
    are read out of the generation's own metadata rather than hardcoded here, so
    a new ``[project.scripts]`` line gets a stable launcher without a second
    list to keep in step.

    The launchers are written AFTER the flip, and that order is deliberate: the
    legacy ``~/.local/bin/lop`` (uv's own symlink into the old fixed tree) keeps
    working until the pointer is in place, so an interrupted install leaves the
    machine on the build it had rather than with a launcher pointing nowhere.

    A launcher is only written when the path it will name EXISTS. Pointing
    ``~/.local/bin/lop`` at a target that is not there would replace a working
    command with a broken one, and that is strictly worse than leaving the
    previous install's launcher in place — so the missing case is a warning and
    a skip, never a link.

    RETURNS ``(written, failed)``, and the failures are a RETURN VALUE rather than
    a debug note because the caller's message claims the machine adopted the
    layout: with ``~/.local/bin`` unwritable, the migration printed the whole
    success block and exited 0 while ``lop`` on PATH still resolved to the LEGACY
    tree and ``<stable>/bin/python3`` (planted a moment earlier) had the
    supervised units on the new one — a machine split between two layouts,
    reported as a clean adoption (design review round 1, D1).
    """
    install_root = _generation_install_root(generation)
    names = _console_script_names(install_root) or (DISTRIBUTION_NAME, "lop")
    interpreter_dir = "Scripts" if os.name == "nt" else "bin"
    written: list[Path] = []
    failed: list[Path] = []
    for name in names:
        if not (install_root / interpreter_dir / name).exists():
            continue
        link = _local_bin_dir() / name
        if not (generation / interpreter_dir / name).exists():
            logger.warning(
                "generation %s has no %s/%s for %s; leaving that launcher alone",
                generation.name,
                interpreter_dir,
                name,
                link,
            )
            failed.append(link)
            continue
        if _atomic_symlink(link, pointer_path() / interpreter_dir / name):
            written.append(link)
        else:
            failed.append(link)
    return written, failed


def _console_script_names(install_root: Path) -> tuple[str, ...]:
    """Every console script this distribution declares, from its own metadata."""
    found = _distribution_at(install_root)
    if found is None:
        return ()
    try:
        return tuple(
            entry.name
            for entry in found.entry_points
            if entry.group == "console_scripts" and entry.name
        )
    except Exception:  # noqa: BLE001 — unreadable entry points fall back to the names
        logger.debug("entry points unreadable at %s", install_root, exc_info=True)
        return ()


def _run_installer_env(argv: list[str], env: dict[str, str]) -> int:
    """Run one installer argv with ``env``, returning its exit status."""
    import subprocess

    return int(subprocess.run(argv, check=False, env=env).returncode)


def install_into_generation(
    source: str | Path | None = None,
    *,
    runner: Callable[[list[str], dict[str, str]], int] | None = None,
    version: str = "",
    commit: str = "",
    ref: str = "",
    origin: str = PYPI_SOURCE_TOKEN,
) -> Path:
    """Install ONE build into its own generation and point ``current`` at it.

    ``source`` is what uv installs: ``None`` for ``local-operator`` from PyPI
    (the ``lop update`` path), or a directory holding this project's source
    (``lop update --from-snapshot``; the host script that pre-builds the mobile
    web bundle passes its prepared directory here).

    ``runner`` is the test seam — it receives ``(argv, env)`` because the
    environment IS the mechanism under test: ``UV_TOOL_DIR``/``UV_TOOL_BIN_DIR``
    are what keep the installer away from every other tree, so a seam that could
    only see the argv would not be watching the thing that matters.

    ORDER, and every step is load-bearing:

    1. reserve ``generations/<id>`` exclusively; uv installs into it. Nothing
       references a generation until the flip, so a tree being built is
       invisible to every reader — and reserving it with ``os.mkdir`` is what
       keeps a second install from aimng uv at the same path;
    2. write ``.lop-source`` into the new tree, so the marker is in place BEFORE
       the generation becomes visible to any reader (and so "no marker" means
       "still installing" for :func:`prune_generations`);
    3. flip ``current``;
    4. write the stable launchers and the daemon shim.

    A failure at step 1 or 2 removes the tree and raises: nothing observable has
    changed — the pointer never moved — and the caller reports the installer's
    own error. Step 4 is best-effort by design: the build is already current at
    that point, and a machine whose sandbox denies a write must not be told the
    upgrade failed. PRUNING IS THE CALLER'S, not this function's: it is a
    deletion, and every caller that wants it says so where it can report what
    went (see ``perform_upgrade``).

    NO STAGING RENAME, deliberately. Building in ``<id>.partial`` and renaming
    it into place looks tidier and is WRONG here: uv bakes the installation path
    into the console-script shims it writes under ``UV_TOOL_BIN_DIR``, so every
    one of them would name the renamed-away directory and dangle. Reserved-and-
    built-in-place keeps uv's own artefacts pointing at paths that survive.
    """
    token = commit[:12] or version or "pypi"
    generation = _reserve_generation(token)
    argv = installer_argv(InstallKind.UV_TOOL)
    if source is not None:
        # ``--from <dir> local-operator`` is the invocation that gets uv to
        # resolve the project under ``dir`` and read its name from the tree.
        argv = [*argv[:-1], "--from", str(source), argv[-1]]
    try:
        code = (runner or _run_installer_env)(argv, _generation_env(generation))
    except OSError as exc:
        # A missing ``uv``, an unwritable generations dir: the caller gets one
        # refusal sentence, not a traceback out of a spawn it did not make.
        _remove_tree(generation)
        raise UpdateError(f"could not run uv: {exc}") from exc
    try:
        if code != 0:
            raise UpdateError(f"installer exited {code}")
        write_source_marker(
            _generation_install_root(generation),
            version=version,
            commit=commit,
            ref=ref,
            origin=origin,
        )
        # The flip is INSIDE this handler so a refusal from the stable root
        # (``EACCES``, ``ENOSPC``) arrives as the one sentence the callers print
        # instead of a bare ``OSError``, and so a built tree nobody can reach is
        # not left behind (review round 1, R-9). Safe to remove on failure:
        # ``flip_pointer`` either renamed the pointer or raised before it did,
        # never both.
        try:
            flip_pointer(generation)
        except OSError as exc:
            raise UpdateError(f"could not point current at {generation.name}: {exc}") from exc
    except BaseException:
        # Nothing has been flipped, so this tree is nobody's but ours. A
        # ``kill -9`` cannot reach here — which is exactly what
        # ``prune_generations``' marker-age rule is for.
        _remove_tree(generation)
        raise
    _written, not_written = write_stable_launchers(generation)
    if not_written:
        # A WARNING here, not a refusal: this machine's ``~/.local/bin/lop`` already
        # points THROUGH ``current`` (an earlier install wrote it), so it feeds off
        # the new generation the moment the flip lands and the install is complete
        # either way. The MIGRATION is the path where a missing launcher leaves the
        # machine split, and that one refuses and rolls back.
        logger.warning(
            "could not write %s; that launcher still names the previous layout",
            ", ".join(str(path) for path in not_written),
        )
    # ``generation`` explicitly, for the same reason the migration passes it: the
    # process running this may be a LEGACY uv-tool install (``lop update`` from a
    # tree that predates the layout), so the this-process gate is False on the
    # very run that creates the layout (review round 1, R-2).
    ensure_daemon_image(generation)
    return generation


def clone_into_generation(
    source: str | Path | None = None,
) -> Path:
    """Clone an INSTALLED tree into a generation and point ``current`` at it.

    THE MIGRATION, and it is non-destructive on purpose: the legacy fixed tree is
    copied, never moved or deleted, so a machine that has just adopted the
    layout still has the install it was running on and can fall back to it by
    hand. ``source`` defaults to ``sys.prefix`` — the tree the running ``lop``
    imports from, which for a pre-layout machine IS the fixed tree.

    A real copy rather than hardlinks (the tempting cheap shape): a hardlinked
    generation shares inodes with a tree that ``uv tool install --force`` is
    about to rewrite, and this layout's entire promise is that a generation's
    bytes are written once. 136 MB and ~4.8k files is the honest price of that
    promise, paid once per machine.

    ``.lop-source`` rides along inside the copied venv, so the generation is
    stamped with the build it really is — including the ``commit ref`` form that
    makes two same-version builds distinguishable. A source tree that carries no
    marker gets one here, because every other reader treats an unmarked
    generation as an install still in flight (``prune_generations``) and a
    generation we just adopted is finished by definition.
    """
    origin = Path(source) if source is not None else Path(sys.prefix)
    if not origin.is_dir():
        raise UpdateError(f"nothing to clone: {origin} is not a directory")
    token = f"migrate-{source_ref(origin)[:12] or 'legacy'}"
    generation = _reserve_generation(token)
    try:
        shutil.copytree(origin, _generation_install_root(generation), symlinks=True)
    except (OSError, shutil.Error) as exc:
        _remove_tree(generation)
        raise UpdateError(f"could not copy {origin} into {generation.name}: {exc}") from exc
    install_root = _generation_install_root(generation)
    if not (install_root / ".lop-source").is_file():
        found = _distribution_at(install_root)
        write_source_marker(
            install_root,
            version=found.version if found is not None else "",
            origin=SNAPSHOT_SOURCE_TOKEN,
        )
    # The copied scripts still name the tree this was copied FROM (see
    # ``_rebind_scripts``), so they are re-pointed at the copy before anything
    # executes them — and before the pointer flip below, which is what makes
    # ``~/.local/bin/lop`` mean the generation afterwards.
    #
    # A PARTIAL REWRITE FAILS THE MIGRATION. This is a one-time step whose entire
    # promise is that the copy runs from the copy, so flipping the pointer with a
    # console script still naming the legacy venv would print success and hand
    # the operator exactly the symptom R-1 exists to remove (review round 2,
    # R2-4). Nothing has been flipped or linked yet, so the copy is removed and
    # the machine is left as it was.
    _rewritten, unrepointed = _rebind_scripts(install_root, origin)
    if unrepointed:
        _remove_tree(generation)
        raise UpdateError(
            "could not re-point "
            + ", ".join(sorted(path.name for path in unrepointed))
            + f" in {install_root}; the migration was abandoned before the pointer moved, "
            "so this machine is unchanged"
        )
    # No installer ran, so this tree has no ``bin`` of its own: lay one down
    # before anything points through it (see ``_link_generation_bin``).
    _link_generation_bin(generation)
    shim_was_absent = not daemon_image_path().exists()
    # Captured BEFORE the flip, because after it the pointer names our generation
    # whether or not this run created it (review round 6, R6-1).
    previous = current_generation()
    flip_pointer(generation)
    _written, not_written = write_stable_launchers(generation)
    if not_written:
        # THE MIGRATION REFUSES AND ROLLS BACK, because a migration that cannot
        # write ``~/.local/bin/lop`` has adopted nothing: `lop` on PATH still runs
        # the legacy tree while the shim planted a moment later would have the
        # supervised units on the new layout. Half a layout is harder to reason
        # about than none (design review round 1, D1).
        _undo_migration(generation, remove_shim=shim_was_absent, previous=previous)
        raise UpdateError(
            "could not write "
            + ", ".join(str(path) for path in not_written)
            + "\n`lop` on your PATH would still run the old install, so the migration was "
            "rolled back: the pointer, the daemon shim and the copied tree are gone and "
            "this machine is as it was"
        )
    # ``generation`` explicitly: this process is the LEGACY tree, so the
    # this-process gate in :func:`ensure_daemon_image` cannot see the layout
    # that now exists.
    ensure_daemon_image(generation)
    return generation


def _sweep_staging_links(moment: float) -> list[Path]:
    """Remove ``current.tmp-*`` leaves an interrupted flip left in the stable root.

    ``flip_pointer`` unlinks its own staging name in a ``finally``, which covers
    every failure INSIDE that call but not a ``kill -9`` between ``os.symlink``
    and ``os.rename``. Nothing else reclaims one: ``prune_generations`` walks
    ``generations/``, and the stable root is the one directory whose contents are
    meant to be a closed set (``current``, ``bin/``, ``generations/``), so a
    stray link there is permanent litter that also happens to look like a
    pointer (review round 1, R-6).

    Aged by :data:`_PARTIAL_TTL_S`, the same "young means in flight" rule the
    generation sweep uses: a concurrent ``flip_pointer`` that is between its
    symlink and its rename must not have its staging name deleted underneath it.
    """
    root = stable_root()
    if not root.is_dir():
        return []
    swept: list[Path] = []
    for entry in sorted(root.glob(f"{pointer_path().name}.tmp-*")):
        try:
            # ``lstat``: the entry is a SYMLINK, and ``stat`` would follow it to
            # the generation and report that tree's (fresh) mtime — so every
            # staging link would look in-flight and none would ever be swept.
            if moment - entry.lstat().st_mtime < _PARTIAL_TTL_S:
                continue
            entry.unlink()
        except OSError:  # pragma: no cover — vanished or unreadable: nothing to do
            continue
        logger.info("removed an interrupted flip's staging link: %s", entry)
        swept.append(entry)
    return swept


def referenced_install_roots() -> tuple[Path, ...]:
    """The install roots named by live and persisted records.

    TWO NAMESPACES, because two kinds of long-lived process read a generation:
    a session runtime (``run/mobile``) and the ``lop serve`` daemon
    (``run/serve``, whose record has carried a ``prefix`` field all along for
    the update path). Both are read through their own registry so pruning and
    the processes that publish records cannot drift apart, and both imports are
    FUNCTION-LOCAL because they reach the session and server layers: this module
    is on ``lop --version``'s path and must not drag either in.

    AND TWO CONFIG ROOTS. ``registry.scan()`` reads ``config_dir()``, which is the
    AMBIENT one — an isolated run, a second profile, a QA pass each have their
    own — so a prune invoked under a different root could not see sessions
    published under the default one, and their generations fell back to the
    ``keep`` margin alone (review round 1, R-7). Both roots are read, deduped;
    the default (``~/.local-operator``) is the one every ordinary session on the
    machine publishes under, and reading it is what makes rule 2 mean what its
    docstring says. The failure direction is unchanged: MORE trees kept.

    A failure in either namespace, under either root, reads as "no records from
    there", which only ever keeps MORE trees — the direction that costs disk
    rather than a running session.
    """
    roots: list[Path] = []
    seen: set[str] = set()
    for config_root in _config_roots():
        for value in _session_roots_under(config_root) + _serve_roots_under(config_root):
            key = str(value)
            if key and key not in seen:
                seen.add(key)
                roots.append(Path(key))
    return tuple(roots)


def _config_roots() -> tuple[Path, ...]:
    """The ambient config root and the home default, deduped, when both exist."""
    from local_operator.paths import DEFAULT_CONFIG_DIRNAME, config_dir

    ambient = config_dir()
    default = Path.home() / DEFAULT_CONFIG_DIRNAME
    if _real(ambient) == _real(default):
        return (ambient,)
    return (ambient, default)


def _session_roots_under(config_root: Path) -> list[Path]:
    """``install_root`` from every session record under one config root."""
    found: list[Path] = []
    try:
        from local_operator.session.runtime import registry
        from local_operator.session.runtime.types import SessionRecord

        for record, _state in registry.scan(config_root, parse=SessionRecord.from_json):
            value = str(getattr(record, "install_root", "") or "")
            if value:
                found.append(Path(value))
    except Exception:  # noqa: BLE001 — no readable records means no objection to keep
        logger.debug("session records unreadable under %s", config_root, exc_info=True)
    return found


def _serve_roots_under(config_root: Path) -> list[Path]:
    """``prefix`` from every ``lop serve`` record under one config root."""
    found: list[Path] = []
    try:
        from local_operator.server import registry as serve_registry

        for serve_record, _state in serve_registry.scan(config_root):
            value = str(getattr(serve_record, "prefix", "") or "")
            if value:
                found.append(Path(value))
    except Exception:  # noqa: BLE001 — same direction as above
        logger.debug("serve records unreadable under %s", config_root, exc_info=True)
    return found


def _real(path: Path) -> Path:
    """``path`` with links resolved, and never raising.

    Every comparison in :func:`prune_generations` is between a directory the
    pointer NAMED and a directory this function LISTED, and the two are two
    spellings of one path (``/tmp`` is a symlink to ``/private/tmp`` on this
    platform, and the pointer is written from ``Path.home()``). They must
    compare equal, so both sides go through here. ``Path.resolve`` is the wrong
    tool for it: it raises when a component is replaced underneath it
    (measured — see :func:`current_generation`), and a prune that raised
    halfway through would be the worst of both outcomes.
    """
    return Path(os.path.realpath(path))


def _undo_migration(generation: Path, *, remove_shim: bool, previous: Path | None = None) -> None:
    """Put the machine back when a migration fails after the flip (D1, R6-1).

    Best-effort and never raising, for the same reason ``_remove_tree`` is: the
    caller is about to report the real error, and cleanup that raises would
    replace it. Each step is narrow on purpose:

    * the pointer is put back to whatever it named BEFORE this run (``previous``),
      and unlinked only when this run created it. "It names our generation" cannot
      tell those two apart — after ``flip_pointer`` it is true either way — and
      unlinking an already-adopted machine's pointer leaves ``lop`` on PATH
      DANGLING while the refusal claims the machine is as it was (review round 6,
      R6-1: measured, `lop` on PATH stopped resolving and the generation that had
      been current became unreferenced);
    * the shim is removed only when this run planted it (``remove_shim``);
    * the copy goes through ``_remove_tree``, which reports what it could not
      remove instead of claiming it.
    """
    target = current_generation()
    if target is not None and _real(target) == _real(generation):
        if previous is not None:
            try:
                flip_pointer(previous)
            except OSError:  # pragma: no cover — nothing further to do about it
                logger.warning("could not put %s back", pointer_path(), exc_info=True)
        else:
            try:
                pointer_path().unlink(missing_ok=True)
            except OSError:  # pragma: no cover — same
                logger.warning("could not undo %s", pointer_path(), exc_info=True)
    if remove_shim:
        try:
            daemon_image_path().unlink(missing_ok=True)
        except OSError:  # pragma: no cover — same
            logger.warning("could not undo %s", daemon_image_path(), exc_info=True)
    if not _remove_tree(generation):
        logger.warning("the refused migration's copy is still at %s", generation)


@dataclass(frozen=True)
class PruneDecision:
    """One generation's fate, and the reason for it, in the CLI's own words."""

    path: Path
    removed: bool
    reason: str


@dataclass(frozen=True)
class PrunePlan:
    """What a prune decided: the removals, and the keeps WITH their reasons.

    The keeps are part of the answer because the command that runs this exists to
    explain a retention decision. Reporting only the removals left ``nothing to
    remove`` with three different meanings, and made the one tree a live session
    was still running from — the single most important thing that output could
    say — invisible (design review round 1, D3).
    """

    removed: tuple[Path, ...]
    decisions: tuple[PruneDecision, ...]

    @property
    def kept(self) -> tuple[PruneDecision, ...]:
        """The generations that survived, in the order they were considered."""
        return tuple(decision for decision in self.decisions if not decision.removed)


def prune_generations(
    *,
    keep: int = DEFAULT_KEEP_GENERATIONS,
    referenced: Iterable[str | Path] = (),
    now: float | None = None,
) -> PrunePlan:
    """Delete the generations nothing can still be importing from.

    RETENTION IS STRUCTURAL, NOT A COUNT. A generation is kept when it is the
    pointer's target, when a live or persisted record names its install root, or
    when it is one of the last ``keep`` generations nothing refers to. The first
    two are the reason this is safe to run while the machine is busy: a runtime's
    own tree is never a candidate, and the count is only the margin for a session
    that has no record yet.

    Deletion is what makes the layout affordable rather than a leak — each
    generation is a whole venv — so it runs after every successful install as
    well as on demand from ``lop install prune``. Removals are returned rather
    than only logged, because both callers report them: a silent 136 MB delete is
    not something this tool gets to do. The KEEPS are returned too, with their
    reasons, for the same argument in the other direction (design review D3).

    ``.lop-source``-less generations are skipped unless they are older than
    :data:`_PARTIAL_TTL_S`, which is the only shape a ``kill -9`` mid-install
    leaves behind (every ordinary failure removes its own tree, and a finished
    generation always carries a marker). That rule is also what makes a prune
    safe to run while an install is in flight: the tree being built has no
    marker yet, so it is skipped even though nothing references it.

    THE MARGIN COUNTS MARKER-CARRYING GENERATIONS ONLY. An in-flight tree used to
    hold one of the ``keep`` places while itself being skipped by the age rule,
    so the margin protected one fewer finished generation than it promises,
    exactly while an install was running (QA round 2, Q3).
    """
    generations = generations_dir()
    if not generations.is_dir():
        return PrunePlan(removed=(), decisions=())
    moment = time.time() if now is None else now
    wanted: set[Path] = set()
    current = current_generation()
    if current is None and pointer_path().is_symlink():
        # A pointer that EXISTS and does not resolve: a flip being renamed, or a
        # link left dangling by a deleted generation. Either way this function
        # cannot tell which tree is live, and the answer to "I cannot tell" is
        # to delete nothing. The marker-age rule below is written the same way,
        # for the same reason: doubt keeps trees.
        logger.warning(
            "install pointer %s does not resolve; keeping every generation", pointer_path()
        )
        everything = sorted(
            (path for path in generations.iterdir() if path.is_dir() and not path.is_symlink()),
            key=lambda path: (path.stat().st_mtime, path.name),
        )
        return PrunePlan(
            removed=(),
            decisions=tuple(
                PruneDecision(path, False, "the pointer does not resolve, so nothing is removed")
                for path in everything
            ),
        )
    if current is not None:
        wanted.add(_real(current))
    _sweep_staging_links(moment)
    named_by_a_record: set[Path] = set()
    for root in referenced:
        try:
            # A record names the venv (``<gen>/tools/local-operator``), and the
            # generation it belongs to is two levels up.
            named_by_a_record.add(_real(Path(root).parent.parent))
        except OSError:  # pragma: no cover — an unresolvable reference keeps nothing extra
            continue
    wanted |= named_by_a_record
    entries = sorted(
        # ``is_dir()`` follows symlinks, so it admits a link to a directory; the
        # layout never writes one there (``_reserve_generation`` uses ``mkdir``),
        # and ``_remove_tree`` refuses a symlink by design — but a link is not one
        # of our generations, so it is not a pruning candidate at all (review
        # round 4, R4-1: this is the shape that handed one to the remover).
        (path for path in generations.iterdir() if path.is_dir() and not path.is_symlink()),
        key=lambda path: (path.stat().st_mtime, path.name),
    )

    def _marker(path: Path) -> bool:
        return (path / "tools" / DISTRIBUTION_NAME / ".lop-source").is_file()

    survivors = [path for path in entries if _real(path) not in wanted]
    # The margin is over SURVIVORS THAT CARRY A MARKER: an in-flight tree is
    # skipped by the age rule below, so letting it hold a place would shrink the
    # margin for finished builds whenever an install was running (QA Q3).
    carrying = [path for path in survivors if _marker(path)]
    margin = max(0, len(carrying) - max(0, keep))
    # Resolved for the same reason ``wanted`` is: a set of unresolved paths
    # would never match, and pruning would silently keep every generation
    # forever (found by its own test, not by inspection).
    removable = {_real(path) for path in carrying[:margin]}

    removed: list[Path] = []
    decisions: list[PruneDecision] = []
    for path in entries:
        resolved = _real(path)
        if current is not None and resolved == _real(current):
            decisions.append(PruneDecision(path, False, "the pointer's target"))
            continue
        if resolved in named_by_a_record:
            decisions.append(PruneDecision(path, False, "a live or saved session record names it"))
            continue
        removal_reason = "superseded, unreferenced"
        if not _marker(path):
            removal_reason = f"no .lop-source: crash debris, older than {_ttl_label()}"
            try:
                in_flight = moment - path.stat().st_mtime < _PARTIAL_TTL_S
            except OSError:  # pragma: no cover — vanished under us, nothing to do
                continue
            if in_flight:
                decisions.append(PruneDecision(path, False, "no .lop-source: an install in flight"))
                continue
        elif resolved not in removable:
            decisions.append(PruneDecision(path, False, f"unreferenced, but within --keep {keep}"))
            continue
        if _remove_tree(path):
            removed.append(path)
            decisions.append(PruneDecision(path, True, removal_reason))
        else:
            # A removal candidate that survived the attempt is NOT a removal: it
            # is kept, and said so, rather than dropped from the answer.
            decisions.append(PruneDecision(path, False, "could not be removed; it is still there"))
    return PrunePlan(removed=tuple(removed), decisions=tuple(decisions))


def _ttl_label() -> str:
    """``_PARTIAL_TTL_S`` in the words a person reads: ``1h``, ``30m``."""
    hours = _PARTIAL_TTL_S / 3600
    if hours >= 1:
        return f"{hours:g}h"
    return f"{_PARTIAL_TTL_S / 60:g}m"


def install_prune_command(*, keep: int = DEFAULT_KEEP_GENERATIONS) -> int:
    """``lop install prune``: apply the retention policy, and say what it decided."""
    if not generations_dir().is_dir():
        print("no install generations on this machine — nothing to prune")
        return 0
    plan = prune_generations(keep=keep, referenced=referenced_install_roots())
    for line in prune_lines(plan):
        print(line)
    return 0


#: One label width for the whole ``lop install`` group (the CLI's house column
#: for blocks like this is 22, so labels are padded to 21 and the value starts at
#: 22; ``status`` was ragged within itself and ``prune`` matched nothing else —
#: design review D6).
_PRUNE_LABEL_WIDTH = 21


def _field(label: str, value: str) -> str:
    """``label`` in the group's column, ``value`` after it.

    A label as long as the column itself keeps its single separating space rather
    than running into its value, which is what the first cut of this did; nothing
    in the ``install`` group is that long now (design review round 2, D16 shortened
    ``a new lop would load:`` to ``next lop would load:`` so it fits).
    """
    if len(label) < _PRUNE_LABEL_WIDTH:
        return f"{label:<{_PRUNE_LABEL_WIDTH}}{value}"
    return f"{label} {value}"


def prune_lines(plan: PrunePlan) -> list[str]:
    """The lines a prune prints — one shape for every caller that renders them.

    SHARED RATHER THAN WRITTEN TWICE: the two front ends used to print the same
    event as two different sentences with two different identifiers — a bare
    ``pruned superseded generation <name>`` from the snapshot path and a
    timestamped ``INFO`` record naming an absolute path from the upgrade path,
    where it also landed ABOVE the lines that explained what had happened
    (design review D4).

    THE KEEPS ARE PRINTED TOO, with their reasons (D3). This is the one command
    whose entire job is a retention decision: naming only the removals left
    ``nothing to remove`` with three meanings, and made the one tree a live
    session was still running from invisible.
    """
    lines: list[str] = []
    if not plan.removed:
        # NO PARENTHETICAL. Round 1's text for this case was a short single line
        # that REPLACED the block; the implementation kept the block and added a
        # summary, which for four reasons became a 146-character run-on restating
        # the rows beneath it in a grammar they do not use (design review round 2,
        # D13). The rows answer "why" already.
        lines.append(_field("generations:", f"{len(plan.decisions)}, nothing to remove"))
    else:
        lines.append(_field("generations:", str(len(plan.decisions))))
    for decision in plan.decisions:
        if decision.reason == "the pointer's target":
            label = "current:"
        else:
            label = "removed:" if decision.removed else "kept:"
        lines.append(_field(label, f"{decision.path.name}  ({decision.reason})"))
    return lines


def prune_notice_lines(plan: PrunePlan) -> list[str]:
    """What an UPGRADE prints about a prune: one line per removal, and no more.

    Separate from :func:`prune_lines` because the two audiences are different: an
    upgrade is reporting a side effect it had to perform, while ``lop install
    prune`` is answering a question about the retention decision, which needs the
    keeps and their reasons (design review D3/D4).
    """
    lines: list[str] = []
    for decision in plan.decisions:
        if not decision.removed:
            continue
        if decision.reason == "superseded, unreferenced":
            lines.append(f"pruned superseded generation {decision.path.name}")
        else:
            # NOT "superseded": a marker-less tree never completed an install, and
            # calling crash debris a superseded build is the one claim in this
            # sentence an operator could act on wrongly (design review round 2,
            # D14). The wording is the prune command's own reason string.
            lines.append(f"pruned unfinished generation {decision.path.name} ({decision.reason})")
    return lines


def install_migrate_command() -> int:
    """``lop install migrate``: adopt the generation layout for this machine.

    Idempotent by outcome rather than by check: a second run clones the tree the
    running ``lop`` imported from, which is now the first generation's own venv —
    a faithful copy of a copy, which is wasteful but harmless. The caller-facing
    guard is :func:`_is_generation_install`: once this process IS a generation,
    there is nothing to migrate, and saying so is better than growing a
    generation per invocation.

    REFUSES A SOURCE CHECKOUT, and that guard is load-bearing rather than
    tidy: the migration COPIES ``sys.prefix`` into the layout and flips the
    machine's pointer at the copy, so a developer running it from ``repo/.venv``
    would point the whole machine's ``lop`` at a copy of a worktree venv — the
    same accident :func:`_repair_refusal` exists to prevent for the supervised
    daemons, where a worktree venv repointed the operator's four live plists at
    itself. Only a durable installed tree has something worth adopting.
    """
    if _is_generation_install():
        print(f"already using the generation layout: {process_install_root()}")
        # Printed WITH its target, like every other pointer line in the product:
        # the one question this block invites is which generation the machine is
        # on, and the answer was a second command away (design review D9).
        print(f"pointer: {pointer_path()} -> {current_generation() or '(unresolved)'}")
        return 0
    kind = install_kind()
    # EVERY REFUSAL ON THIS SURFACE GOES TO STDERR, which is what its siblings on
    # ``lop update`` already do and what round 1's remediation wrongly claimed of
    # this command: on stdout, `lop install migrate | tee log` files a refusal as
    # output and anything grepping stdout for success reads a failure as one
    # (design review round 2, D12). The success block below stays on stdout.
    if kind == InstallKind.EDITABLE:
        print(
            "refusing to migrate: this interpreter is a source checkout's venv, not an "
            "installed distribution, so the tree it imports from is one it is still "
            "being edited in. run `lop install migrate` from an installed `lop`.",
            file=sys.stderr,
        )
        return 1
    if kind == InstallKind.UNKNOWN:
        print(
            "refusing to migrate: this interpreter has no install this command can "
            "identify (no dist-info, no venv of its own), so there is no tree to copy",
            file=sys.stderr,
        )
        return 1
    try:
        generation = clone_into_generation()
    except UpdateError as exc:
        print(f"could not migrate: {exc}", file=sys.stderr)
        return 1
    print(f"copied {Path(sys.prefix)} into {generation}")
    print(f"pointer {pointer_path()} -> {generation}")
    print(f"the previous install is untouched at {Path(sys.prefix)}")
    return 0


@dataclass(frozen=True)
class SnapshotSource:
    """Where a ``--from-snapshot`` build came from, and how to name it.

    ``path`` is a directory uv can install from. ``commit``/``ref``/``version``
    are what the ``.lop-source`` marker records; all three may be empty (a
    hand-passed directory that is not a git repository and has no readable
    ``pyproject.toml``), which the marker answers with the bare
    :data:`SNAPSHOT_SOURCE_TOKEN`. ``temporary`` marks a directory THIS module
    extracted and must therefore remove.
    """

    path: Path
    commit: str = ""
    ref: str = ""
    version: str = ""
    temporary: bool = False

    @property
    def label(self) -> str:
        """How this build is named to a person: the ref, else the commit, else the path."""
        return self.ref or self.commit or str(self.path)

    @property
    def install_label(self) -> str:
        """What ``--from-snapshot`` says it is INSTALLING (design review D8).

        ``label`` alone is the branch name for the directory shape, and a branch
        name is a claim about the source rather than about the bytes: a directory
        is installed AS IT STANDS — uncommitted work included — while a ref is
        archived from ``HEAD``, so the two are different builds of one version.
        Naming the branch in both cases told the operator nothing about which of
        the two they were about to get.
        """
        if not self.temporary:
            return str(self.path)
        return f"{self.label} @ {self.commit[:7]}" if self.commit else self.label

    @property
    def install_shape(self) -> str:
        """The other half of that sentence: which SHAPE of source this is."""
        if not self.temporary:
            return "working tree as it stands"
        return "archived from the repository"


def _git(repo: Path, *args: str) -> str:
    """One ``git`` query against ``repo``; ``""`` on any failure.

    Total by design: every caller is recording provenance for a label, and a
    missing git binary or a directory that is not a repository must cost a
    blank field rather than the install.
    """
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _project_version(root: Path) -> str:
    """The ``version`` in ``root``'s ``pyproject.toml``, or ``""``.

    Only ever decoration on the marker: the generation's REAL version is read
    from the installed distribution metadata (``disk_build``), which uv writes
    from the same tree a moment later.
    """
    try:
        import tomllib

        with (root / "pyproject.toml").open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except Exception:  # noqa: BLE001 — provenance for a label may never raise
        return ""


def resolve_snapshot(value: str) -> SnapshotSource:
    """Resolve ``--from-snapshot``'s argument to a directory uv can install.

    Two accepted shapes, and the distinction is what the caller has to hand:

    * **a directory** — installed as it stands. This is the shape a caller that
      prepares its own tree (a bundle build, a patch set, a CI artifact) needs,
      and the reason this command does not insist on a git ref;
    * **a git ref** — archived out of the repository the command runs in. The
      caller keeps its working tree (including uncommitted work) out of the
      build by construction, and the commit is recorded, so two builds of one
      unchanged version stay distinguishable — which is the whole job of the
      ref half of the marker.

    Raises :class:`UpdateError` with something a person can act on: a ref that
    does not resolve, a repository that is not there, or an archive with no
    ``pyproject.toml`` in it are all "this is not a tree I can install" rather
    than a traceback.
    """
    candidate = Path(value).expanduser()
    if candidate.is_dir():
        resolved = candidate.resolve()
        commit = _git(resolved, "rev-parse", "HEAD")
        return SnapshotSource(
            path=resolved,
            commit=commit,
            ref=_git(resolved, "rev-parse", "--abbrev-ref", "HEAD") if commit else "",
            version=_project_version(resolved),
        )
    repo = Path.cwd()
    commit = _git(repo, "rev-parse", "--verify", f"{value}^{{commit}}")
    if not commit:
        raise UpdateError(
            f"{value!r} is neither a directory nor a git ref in {repo} "
            "— pass a path to a source tree, or a ref of the repository you are in"
        )
    import io
    import tarfile

    archive = _git_bytes(repo, "archive", "--format=tar", commit)
    if archive is None:
        raise UpdateError(f"could not archive {value!r} from {repo}")
    target = Path(tempfile.mkdtemp(prefix="lop-snapshot-"))
    try:
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            # ``filter="data"`` refuses absolute paths and links on extraction.
            # The bytes come from our own ``git archive``, so this is about the
            # documented default changing under us in 3.14 rather than about a
            # hostile tarball — a future interpreter rejects them by default and
            # the explicit filter keeps the behaviour identical either way.
            tar.extractall(target, filter="data")
    except (OSError, tarfile.TarError) as exc:
        _remove_tree(target)
        raise UpdateError(f"could not unpack {value!r}: {exc}") from exc
    if not (target / "pyproject.toml").is_file():
        _remove_tree(target)
        raise UpdateError(f"the archive of {value!r} has no pyproject.toml")
    return SnapshotSource(
        path=target,
        commit=commit,
        ref=value,
        version=_project_version(target),
        temporary=True,
    )


def _git_bytes(repo: Path, *args: str) -> bytes | None:
    """``git``'s binary stdout, or ``None`` when it could not be read."""
    import subprocess

    try:
        completed = subprocess.run(
            ["git", "-C", str(repo), *args], check=False, capture_output=True
        )
    except OSError:
        return None
    return completed.stdout if completed.returncode == 0 else None


def _human_bytes(total: int) -> str:
    """``136 MB``, for a number a person reads in a status block."""
    for unit, step in (("GB", 1 << 30), ("MB", 1 << 20), ("KB", 1 << 10)):
        if total >= step:
            return f"{total / step:.0f} {unit}"
    return f"{total} B"


def _tree_size(root: Path) -> int:
    """Bytes under ``root``, best-effort.

    A ``stat`` walk rather than shelling out to ``du``: this runs inside
    ``lop install status``, and the number is the one that drives the decision the
    command is asked about — each generation is a whole venv that ``prune`` exists
    to reclaim (design review D10). Unreadable entries are skipped rather than
    raised: a status command must not fail on a tree it cannot fully read.
    """
    total = 0
    for directory, _dirs, files in os.walk(root, onerror=lambda _error: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(directory, name)).st_size
            except OSError:  # pragma: no cover — vanished mid-walk
                continue
    return total


def install_status_command() -> int:
    """``lop install status``: the layout, and what a new ``lop`` would load.

    The one surface that separates "the build this process loaded" from "the
    build the pointer names". Every label in the product describes the former;
    on a mixed-generation machine they disagree, and this is where that is
    legible without reading symlinks by hand.

    THE EMPTY STATE EXPLAINS ITSELF (design review D7): ``(unresolved)``,
    ``(unknown)`` and a header with nothing under it read as three separate
    failures rather than as the one fact they are — this machine has not adopted
    the layout — and the command that fixes it was not named.
    """
    print(_field("stable root:", str(stable_root())))
    pointer = pointer_path()
    generation = current_generation()
    if generation is None:
        # TWO STATES, not one (review round 6, R6-2): "no layout at all" and "a
        # pointer that resolves to nothing" printed identically — including while
        # the very next line listed the generations that DO exist. Both are
        # reachable (a hand-deleted pointer; the migration's undo used to produce
        # the dangling one), and the surface whose whole job is to make the pointer
        # legible is the one place that must not blur them. The layout question is
        # answered by the LAYOUT, not by the pointer: the D7 sentence is for a
        # machine that has none.
        if pointer.is_symlink():
            print(_field("pointer:", f"{pointer} -> (unresolved)"))
        elif generations_dir().is_dir() and any(generations_dir().iterdir()):
            print(_field("pointer:", f"{pointer} -> (absent)"))
        else:
            print(_field("pointer:", "(no generation layout on this machine)"))
    else:
        print(_field("pointer:", f"{pointer} -> {generation}"))
    print(_field("this process:", process_install_root()))
    root = current_install_root()
    fresh = _stamp_at(root) if root is not None else None
    if fresh is None:
        print(_field("next lop would load:", "(unknown — nothing resolves behind the pointer)"))
    else:
        print(_field("next lop would load:", fresh.label()))
    generations = (
        sorted(path for path in generations_dir().iterdir() if path.is_dir())
        if generations_dir().is_dir()
        else []
    )
    if not generations:
        print(_field("generations:", "none"))
        print("run `lop install migrate` from an installed `lop` to adopt the layout")
        return 0
    total = sum(_tree_size(path) for path in generations)
    print(f"generations ({len(generations)}, {_human_bytes(total)}):")
    for path in generations:
        marker = "  <- current" if generation is not None and path.name == generation.name else ""
        print(f"  {path.name}{marker}")
    for held in referenced_install_roots():
        # Indented under the list deliberately: the label column above is for
        # this command's own fields, and a record-named tree is a property of one
        # of the generations rather than a fifth field (design review D6).
        #
        # Named by GENERATION ID, the vocabulary of the lines above it (design
        # review round 2, D17). A record names the venv (``<gen>/tools/
        # local-operator``), so the id is two levels up; the absolute path stays the
        # fallback for a record whose root is not under this layout.
        root = Path(held)
        nested = root.parent.parent
        named = nested.name if nested.parent == generations_dir() else str(held)
        print(f"  held by a live session: {named}")
    return 0


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
    source: str | Path | None = None,
    commit: str = "",
    ref: str = "",
    on_prune: Callable[["PrunePlan"], None] | None = None,
) -> str:
    """Run the detected installer. Returns ``target`` (this process cannot re-read it).

    The new wheel is not imported into this interpreter; callers print
    ``target`` rather than asking :func:`installed_version` again.

    THE uv-tool LAYOUT INSTALLS INTO A NEW GENERATION. That branch routes through
    :func:`install_into_generation`, which is what makes an upgrade safe to run
    while ~24 runtimes are importing from the install: the tree they hold is not
    the tree that changes, and the handover is a pointer flip
    (:func:`flip_pointer`). Both front ends share this function — ``lop update``
    and the TUI's ``/update`` — so both get the generation path from one place.

    pip and pipx KEEP TODAY'S BEHAVIOUR, deliberately and with a documented
    consequence: neither has a directory layout this module can make atomic, so
    they still rewrite site-packages in place under the running fleet. There is
    no generation story for them to route through, and inventing one here would
    be a second installer's worth of work in a change that exists to stop a
    known, measured failure. ``lop-update`` and the wheel path are the ones the
    host actually uses.

    ``run`` IS THE OBSERVER SEAM AND IT DOUBLES AS A SAFETY FENCE. A caller that
    substituted the installer has not produced a tree to point at, and flipping
    the host's ``current`` onto a directory an injected runner never filled would
    break every session on the machine — so the injected-runner shape keeps the
    old behaviour exactly (argv, exit status, marker at ``prefix``) and never
    touches the pointer.

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
    if detected is InstallKind.UV_TOOL and run is None:
        # ``target`` is the PyPI version just installed, and this path is always
        # a PyPI wheel unless the caller passed ``source`` (a git snapshot):
        # ``commit``/``ref`` describe that case, and leaving them empty is what
        # makes the marker say ``pypi <version>``.
        install_into_generation(source, version=target, commit=commit, ref=ref)
        # Retention, not tidiness: a generation is a whole venv, so a machine
        # that never pruned would grow by one per release. The policy is
        # structural (see ``prune_generations``) and best-effort — the upgrade
        # has already succeeded, and a caller that cannot read the record
        # directory must not be told otherwise.
        try:
            plan = prune_generations(referenced=referenced_install_roots())
        except Exception:  # noqa: BLE001 — pruning never fails an upgrade
            logger.debug("generation prune failed", exc_info=True)
        else:
            # THE CALLER SAYS IT, not this function: the CLI has already printed
            # ``installed``/``current install:`` by the time it can, so the
            # removal lands where it belongs — after the lines that explain it —
            # instead of as a bare timestamped record above them. ``on_prune`` is
            # how the two front ends share the sentence (design review D4); with
            # no caller listening, the log file is still the record.
            if on_prune is not None:
                on_prune(plan)
            else:
                for line in prune_notice_lines(plan):
                    logger.info("%s", line)
        return target
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
    #
    # Reached only through the ``run`` seam now: the real uv-tool upgrade was
    # handled above, and there the marker is written into the new generation
    # before it becomes visible (``install_into_generation`` step 2) rather than
    # into a tree that is already in use.
    if detected is InstallKind.UV_TOOL:
        root = Path(prefix) if prefix is not None else Path(sys.prefix)
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

    THE PROBE IS MACHINE-WIDE, NOT HOME-SCOPED, and that is deliberate rather
    than an oversight: the process it warns about is one a person started by hand
    in a terminal (``lop mobile serve``), which has no plist and no relationship
    to ``Path.home()`` — so narrowing it to "this HOME has a plist" would silence
    exactly the case the warning exists for. The consequence is recorded here
    because a QA run with an isolated ``HOME`` still sees it: this function can
    answer 200 for the operator's REAL daemon (QA round 2, Q4).
    ``_mobile_refresh``'s verdict for that answer is ``unsupervised``, which takes
    no action by design. A future change that ACTED on this answer would reach a
    daemon outside the caller's HOME, and would have to solve that first.
    """
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(_MOBILE_HEALTHZ, timeout=1.0) as response:
            return 200 <= int(response.status) < 300
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _mobile_restart_invocation() -> tuple[list[str], str | None] | None:
    """``(argv, executable)`` for the *new* distribution's ``mobile restart``.

    ``sys.executable -m local_operator.cli`` is the post-upgrade
    interpreter — the same interpreter the LaunchAgent's ProgramArguments
    already name, so its site-packages are the wheel the installer just
    wrote. There is deliberately NO PATH ``lop`` fallback: a PATH hit can
    be a *different* installation (another tool env, a brew shim) than
    the one just upgraded, and restarting that would serve the wrong
    build while reporting success. If this interpreter is gone after the
    upgrade, the refresh fails honestly and the copy names the recovery.

    ``SAFE_PATH_FLAG``, written literally rather than through
    ``python_argv`` (this argv's ``argv[0]`` is a label, and ``python_argv``
    builds an interpreter-first argv), because the sentence above is
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

    WHICH interpreter travels is :func:`_post_upgrade_invocation`'s argument,
    and since the generation layout it is NOT necessarily ``sys.executable``.
    """
    from local_operator import procname

    return _post_upgrade_invocation(procname.LABEL_MOBILE_RESTART, ["mobile", "restart"])


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


def _daemon_refresh_invocation() -> tuple[list[str], str | None] | None:
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
    from local_operator import procname

    return _post_upgrade_invocation(procname.LABEL_DAEMONS_REFRESH, ["update", "--refresh-daemons"])


def _post_upgrade_invocation(label: str, tail: list[str]) -> tuple[list[str], str | None] | None:
    """``(argv, executable)`` for a child that must run the build just installed.

    THE INTERPRETER MOVED WITH THE GENERATION LAYOUT, and this is where that has
    teeth. Both helpers above used to name ``sys.executable`` and were right to:
    the installer rewrote the tree this process runs from, so the running
    interpreter WAS the new wheel. The generation installer never touches this
    process's tree — it builds a generation and flips the pointer — which makes
    ``sys.executable`` precisely the SUPERSEDED build. A repair run from it would
    render the previous build's LaunchAgent shape and report success, which is
    the failure those docstrings already call "a no-op wearing the costume of a
    fix".

    So the child runs the interpreter the pointer resolves to, CONCRETELY
    (:func:`current_interpreter` — never the mutable ``current`` path, or the
    child would import through a symlink that a second upgrade can redirect
    mid-run). ``sys.executable`` stays the answer for a pip/pipx upgrade and for
    a machine that has not migrated: those installers still rewrite in place, so
    there the running interpreter genuinely is the new wheel.

    ``None`` — the caller reports "no interpreter to run it with" — only when
    neither a pointer nor a usable ``sys.executable`` exists.
    """
    from local_operator import procname

    interpreter = current_interpreter()
    if interpreter is None or str(interpreter) == sys.executable:
        if not (sys.executable and Path(sys.executable).exists()):
            return None
        argv0, image = procname.spawn_identity(label)
        return [argv0, SAFE_PATH_FLAG, "-m", "local_operator.cli", *tail], image
    # The branded link is planted per venv on first use, so the new tree may not
    # have one yet; the label rides on the argv either way, which is the axis a
    # ``ps`` reader sees.
    return [
        procname.branded_argv0(label),
        SAFE_PATH_FLAG,
        "-m",
        "local_operator.cli",
        *tail,
    ], str(interpreter)


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


def _repair_refusal() -> str | None:
    """Why this process must not rewrite the installed daemons, or ``None``.

    TWO QUESTIONS, and the second is the invariant this guard exists for: a
    repair may change how a daemon is NAMED, never WHICH INSTALL it runs.

    1. **Is this an installation at all?** An editable or unknown install is
       refused outright — that is the incident this guard came from, where a
       worktree venv rewrote the operator's four live plists to point at
       itself.
    2. **Is it the SAME installation the plists already run?** A durable
       install — a uv tool, pipx — IS the interpreter ``lop`` runs from, so it
       may repair what it owns. Anything else (a hand-made venv with a PyPI
       install, a second tool env) is refused unless its prefix is the prefix
       the installed plists already record, so that such a venv cannot repoint
       the operator's daemons at itself and then be deleted.

    Prefix equality, not path equality, is the test: a stale plist recording
    ``<prefix>/bin/python3`` and the branded shape recording
    ``<prefix>/bin/Local Operator`` are the SAME install.

    The generation layout adds a THIRD recorded shape — the stable shim
    (``<stable>/bin/python3``, whose ``parent.parent`` is the stable root rather
    than a venv) — and it never reaches this comparison: only a generation
    install renders it, a generation install is a uv tool, and a uv tool answers
    ``None`` above for the reason question 2 is about (it IS the installation
    ``lop`` runs from).
    """
    kind = install_kind()
    if kind in (InstallKind.EDITABLE, InstallKind.UNKNOWN):
        return (
            "installed daemons are only refreshed by an installed "
            "distribution; this is a source checkout, so nothing was touched"
        )
    if kind in (InstallKind.UV_TOOL, InstallKind.PIPX):
        return None
    from local_operator import launchd

    mine = Path(sys.prefix).resolve()
    others: list[str] = []
    for path in _installed_daemon_plists():
        recorded = launchd.recorded_install_prefix(launchd.load(path))
        if recorded is not None and recorded != mine:
            others.append(f"{path.name} runs {recorded}")
    if others:
        return (
            f"the installed daemons belong to another installation "
            f"({'; '.join(others)}), so this one ({mine}) left them alone; "
            "upgrade from that installation to repair them"
        )
    return None


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
    refusal = _repair_refusal()
    if refusal is not None:
        # A refusal is printed rather than silent: it is the difference between
        # "nothing needed repairing" and "this process is not allowed to".
        print(f"warning: {refusal}", file=sys.stderr)
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


def _print_current_generation() -> None:
    """Name the generation ``current`` now points at, or say nothing.

    Silent when there is no generation layout on this machine (a pip/pipx
    install, a machine that has not migrated), because there is nothing to say
    and a line reading "(none)" after a successful upgrade would look like a
    failure.
    """
    generation = current_generation()
    if generation is not None:
        print(f"current install: {generation}")


def _generation_upgrade(total: int) -> int:
    """The tail every successful install shares: report, prune, refresh, succeed.

    ``lop update --from-snapshot`` uses this directly; the PyPI path prints its own
    ``installed`` line and then runs the same tail. The prune notice comes from
    :func:`prune_notice_lines`, the one renderer both front ends use, and it prints
    AFTER ``current install:`` — the removal reported where it belongs rather than
    as a bare record above the lines that explain it (design review D4).
    """
    _print_current_generation()
    for line in prune_notice_lines(prune_generations(referenced=referenced_install_roots())):
        print(line)
    _print_daemon_refreshes(refresh_daemons_after_upgrade())
    return total


def _snapshot_command(value: str) -> int:
    """``lop update --from-snapshot <dir-or-ref>``: install a local build.

    The in-repo half of what the out-of-tree ``lop-update`` script does today,
    and the reason that script can be reduced to a delegator: the archive, the
    install and the pointer flip all happen here, under this repo's tests.

    Deliberately NOT gated on a PyPI version check. A snapshot's version comes
    from the tree being installed (its ``pyproject.toml`` still names the last
    release), so "am I behind PyPI" is not the question being answered —
    installing the tree is. A git snapshot also still upgrades from PyPI on a
    plain ``lop update``; nothing here changes that.
    """
    kind = install_kind()
    if kind is InstallKind.EDITABLE:
        print(editable_refusal(), file=sys.stderr)
        return 1
    if kind is not InstallKind.UV_TOOL:
        # The generation layout is uv-tool only, and so is this command: a pip
        # or pipx install has no per-generation root for a snapshot to land in,
        # and pretending otherwise would rewrite site-packages under the
        # running fleet — the failure this change exists to remove.
        print(
            "lop update --from-snapshot installs into a uv-tool generation, "
            f"and this install is {kind.value} — install it with `uv tool "
            "install --force --from <dir> local-operator` instead.",
            file=sys.stderr,
        )
        return 1
    try:
        snapshot = resolve_snapshot(value)
    except UpdateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    shape = (
        f"{snapshot.version}, {snapshot.install_shape}"
        if snapshot.version
        else snapshot.install_shape
    )
    print(f"installing {snapshot.install_label} ({shape})")
    try:
        install_into_generation(
            snapshot.path,
            version=snapshot.version,
            commit=snapshot.commit,
            ref=snapshot.ref,
            origin=SNAPSHOT_SOURCE_TOKEN,
        )
    except UpdateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if snapshot.temporary:
            # ``uv`` has copied what it needs; the extract is ours to reclaim,
            # and leaving 136 MB of tree per install in TMPDIR is how a machine
            # with a small /tmp dies on a day nobody is looking.
            _remove_tree(snapshot.path)
    return _generation_upgrade(0)


def update_command(
    *, check: bool = False, refresh_daemons: bool = False, from_snapshot: str | None = None
) -> int:
    """``lop update``, ``lop update --check``, ``--from-snapshot`` and the repair.

    ``--refresh-daemons`` is not an upgrade: it is the repair step that the
    upgrade path runs in a CHILD process from the newly installed wheel, so that
    the plists it renders are this build's and not the previous one's. See
    :func:`daemons_refresh_command`. It is checked before the PyPI call because
    it must work on any machine, including one whose network is down, and it
    never reports a version. See the architect table for the other codes.

    ``--from-snapshot`` is checked before the PyPI call for the same reason: it
    installs a build that is already on this machine, so a host with no route to
    the index (or no wish to use one) must be able to run it. Combining it with
    ``--check`` is a refusal rather than a precedence rule — the two answer
    different questions and a caller that asked for both has asked for neither.
    """
    if refresh_daemons:
        return daemons_refresh_command()

    if from_snapshot is not None:
        if check:
            print("--check compares against PyPI; --from-snapshot installs a tree", file=sys.stderr)
            return 1
        return _snapshot_command(from_snapshot)

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
    pruned: list[str] = []
    try:
        installed = perform_upgrade(
            target=result.latest,
            kind=kind,
            on_prune=lambda plan: pruned.extend(prune_notice_lines(plan)),
        )
    except UpdateError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"installed {installed}")
    # Names the layout, not the build: on a machine with generations the build
    # now lives in its own tree and `current` names it, which is the one fact a
    # person watching an upgrade wants to see and cannot otherwise know. Pruning
    # happens inside ``perform_upgrade`` (one place, both front ends); it hands
    # the decision back through ``on_prune`` so the removal is printed HERE,
    # after the lines that explain it, in the same words the snapshot path uses
    # (design review D4).
    _print_current_generation()
    for line in pruned:
        print(line)
    _print_daemon_refreshes(refresh_daemons_after_upgrade())
    return 0
