"""The macOS notification sender that carries Local Operator's own identity.

macOS attributes a notification to the PROCESS that posts it. `osascript -e
'display notification'` therefore arrives as **Script Editor** — its name, its
icon, its notification settings — no matter what title text it is handed. The
operator saw exactly that: a banner reading "Local Operator" wearing Script
Editor's generic icon.

This package builds a tiny app bundle we own (``notifier.m``, compiled with
the Command Line Tools) and posts through it, so the banner carries our name
and the project's icon and can carry a click action.

**Everything here is best-effort and cached.** The build happens once, into
the user's config dir, and every later notification reuses it; a machine with
no compiler, no ``iconutil``, or an unwritable config dir simply reports
failure and the caller falls back to the bare ``osascript`` route it used
before. Nothing in this module may raise into a runtime's gate path, and
nothing may block it: the build is guarded by a lock so two runtimes racing on
first use cannot corrupt a half-written bundle, and the caller decides whether
it can afford to wait for it (``ensure_bundle(block=False)`` never does).
"""

from __future__ import annotations

import logging
import os
import plistlib
import shutil
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

#: Our own bundle id. Stable across versions: the notification database keys
#: a sender's settings (and the user's "allow notifications" answer) on this,
#: so changing it would silently reset the user's preferences.
BUNDLE_ID = "me.damiantran.localoperator"

#: The name the banner shows.
BUNDLE_NAME = "Local Operator"

#: Bumped when `notifier.m`, the Info.plist shape, or the icon changes, so an
#: upgraded install rebuilds instead of reusing a stale binary. Compared
#: against the marker written beside the built bundle.
#:
#: BUMPING THIS IS ALSO HOW A STALE ICON IS FIXED. Notification Centre caches
#: a sender's icon against its bundle id, and it does NOT re-read the bundle
#: because the file on disk changed: during development a bundle first
#: registered without an icon kept showing the generic placeholder even after
#: `icon.icns` was added, until `lsregister -f` ran again. A rebuild re-runs
#: registration (see `build_bundle`), so raising this stamp is what pushes the
#: new icon out to users who already have the old bundle.
BUILD_STAMP = "2"

_MARKER = ".build-stamp"

#: How long a `.building` marker is believed before it is reclaimed as stale.
#: The build is one `clang` invocation (sub-second here, seconds on a cold or
#: loaded machine); this is generous enough that a genuinely running build is
#: never stolen, and short enough that a crashed one costs at most this long
#: rather than forever.
_BUILD_LOCK_STALE_S = 300.0


def bundle_root(config_dir: Path) -> Path:
    """Where the built bundle lives: inside the user's config dir.

    NOT in the wheel's site-packages: that tree may be read-only, is replaced
    wholesale on upgrade, and is shared between environments. The config dir
    is per-user, writable, and already where this app keeps its state.
    """
    return config_dir / "notifier" / "LocalOperator.app"


def _source_dir() -> Path:
    return Path(__file__).resolve().parent


def is_supported() -> bool:
    """Whether this platform can host the bundle at all."""
    return sys.platform == "darwin"


def _stamp_path(app: Path) -> Path:
    return app.parent / _MARKER


def is_built(app: Path) -> bool:
    """Whether a CURRENT bundle is already present.

    Checked by content stamp rather than mere existence: a bundle built by an
    older release names an older binary, and treating "a directory is there"
    as "built" would pin the user to it forever.
    """
    binary = app / "Contents" / "MacOS" / "notifier"
    if not binary.exists():
        return False
    try:
        return _stamp_path(app).read_text(encoding="utf-8").strip() == BUILD_STAMP
    except OSError:
        return False


def _info_plist() -> dict[str, object]:
    return {
        "CFBundleIdentifier": BUNDLE_ID,
        "CFBundleName": BUNDLE_NAME,
        "CFBundleDisplayName": BUNDLE_NAME,
        "CFBundleExecutable": "notifier",
        "CFBundleIconFile": "icon",
        "CFBundlePackageType": "APPL",
        "CFBundleInfoDictionaryVersion": "6.0",
        "CFBundleShortVersionString": "1.0",
        # Background helper: no Dock tile, no menu bar, no window. Without
        # this the bundle flashes into the Dock every time it posts.
        "LSUIElement": True,
    }


def build_bundle(config_dir: Path) -> Path | None:
    """Compile and register the bundle. Returns its path, or None.

    None is an ORDINARY answer — no compiler, no SDK, an unwritable config
    dir — and the caller falls back to the plain notification route.
    """
    if not is_supported():
        return None
    compiler = shutil.which("clang")
    if compiler is None:
        logger.debug("no clang; the identity notifier cannot be built")
        return None

    app = bundle_root(config_dir)
    macos = app / "Contents" / "MacOS"
    resources = app / "Contents" / "Resources"
    try:
        macos.mkdir(parents=True, exist_ok=True)
        resources.mkdir(parents=True, exist_ok=True)

        (app / "Contents" / "Info.plist").write_bytes(plistlib.dumps(_info_plist()))

        icon = _source_dir() / "icon.icns"
        if icon.exists():
            shutil.copyfile(icon, resources / "icon.icns")

        result = subprocess.run(  # noqa: S603 — fixed argv, no shell
            [
                compiler,
                "-framework",
                "Foundation",
                # See notifier.m: NSUserNotificationCenter is deprecated but is
                # the only API that delivers from an unsigned helper without
                # aborting. The suppression is the deliberate trade.
                "-Wno-deprecated-declarations",
                "-o",
                str(macos / "notifier"),
                str(_source_dir() / "notifier.m"),
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if result.returncode != 0:
            logger.debug("identity notifier build failed: %s", result.stderr.strip()[:400])
            return None

        # Ad-hoc signature: LaunchServices declines to register an unsigned
        # bundle on Apple silicon, and this costs nothing (no identity, no
        # certificate). Failure is not fatal — an unregistered bundle still
        # posts, it is only slower to be recognised.
        codesign = shutil.which("codesign")
        if codesign is not None:
            subprocess.run(  # noqa: S603 — fixed argv, no shell
                [codesign, "--force", "--sign", "-", str(app)],
                capture_output=True,
                timeout=60,
                check=False,
            )
        _register(app)
        _stamp_path(app).write_text(BUILD_STAMP, encoding="utf-8")
        return app
    except Exception:  # noqa: BLE001 — a toast is never worth an exception
        logger.debug("identity notifier could not be prepared", exc_info=True)
        return None


def _register(app: Path) -> None:
    """Tell LaunchServices the bundle exists, so its identity resolves."""
    lsregister = (
        "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
        "LaunchServices.framework/Support/lsregister"
    )
    if not os.path.exists(lsregister):
        return
    subprocess.run(  # noqa: S603 — fixed argv, no shell
        [lsregister, "-f", str(app)],
        capture_output=True,
        timeout=30,
        check=False,
    )


def ensure_bundle(config_dir: Path, *, block: bool = False) -> Path | None:
    """The bundle if it is ready, building it only when ``block`` allows.

    A gate announcement must not wait on a compiler, so the default answer on
    a cold machine is "not yet" — the caller sends this notification the old
    way and the NEXT one carries the identity. ``block=True`` is for the
    explicit warm-up path (`lop notify --prepare`) where waiting is the point.
    """
    if not is_supported():
        return None
    app = bundle_root(config_dir)
    if is_built(app):
        return app
    if not block:
        _build_in_background(config_dir)
        return None
    return build_bundle(config_dir)


def _build_in_background(config_dir: Path) -> None:
    """Kick off a one-shot build without waiting for it.

    Guarded by an atomic marker so a burst of notifications (or several
    runtimes on one machine) starts exactly one compiler.
    """
    lock = config_dir / "notifier" / ".building"
    try:
        lock.parent.mkdir(parents=True, exist_ok=True)
        # O_EXCL is the whole guard: the loser of the race does nothing.
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        # A MARKER IS NOT A TOMBSTONE. The builder is a daemon thread in a
        # runtime that exits as soon as its work is done — plus SIGKILL, OOM
        # and power loss — so the process can die between creating this and
        # removing it. Treating that as "a build is in progress" disabled the
        # identity bundle permanently and invisibly: every later
        # `ensure_bundle` returned None forever while notifications kept
        # working via the fallback, so nothing ever surfaced the failure
        # (round 3, B4).
        #
        # Age is the liveness test rather than a pid: the marker's writer may
        # be long gone, and a pid on a machine that has since rebooted can
        # name an unrelated live process. A build is a single clang
        # invocation; anything older than the stale window had no survivor.
        try:
            if time.time() - lock.stat().st_mtime < _BUILD_LOCK_STALE_S:
                return
            lock.unlink()
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except OSError:
            # Lost the race to another reclaimer, or the marker vanished
            # under us. Either way somebody else is building; do nothing.
            return
    except OSError:
        return
    os.close(fd)

    import threading

    def _run() -> None:
        try:
            build_bundle(config_dir)
        finally:
            try:
                lock.unlink()
            except OSError:
                pass

    thread = threading.Thread(target=_run, name="notifier-build", daemon=True)
    thread.start()


def notify_command(
    app: Path,
    title: str,
    body: str,
    click_command: str = "",
    subtitle: str = "",
) -> list[str]:
    """argv posting one notification through the bundle.

    ``subtitle`` carries the state category ("Input required") in a field of
    its own, where a long session name cannot clip it — see `notifier.m`.
    Positional after ``click_command`` because the run loop keys off the
    click argument's presence; an empty string in slot 3 keeps the subtitle
    reachable for a toast with no click action.

    Pure, so the wire shape is testable without building or posting anything —
    the same discipline `osascript_command` follows.
    """
    argv = [str(app / "Contents" / "MacOS" / "notifier"), title or BUNDLE_NAME, body]
    if click_command or subtitle:
        argv.append(click_command)
    if subtitle:
        argv.append(subtitle)
    return argv
