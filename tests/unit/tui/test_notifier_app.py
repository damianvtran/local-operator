"""The macOS notification sender that carries our own identity.

macOS attributes a notification to the process that POSTS it, so the
`osascript` route arrives as Script Editor — its name and its icon — whatever
title text it is handed. The operator caught exactly that by looking at a
banner. These tests pin the parts of the fix that can be asserted without a
Mac; the identity and the icon themselves are proven by a captured screenshot
in the PR, because that is the only surface that can show them.
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

from local_operator.tui import notifier_app


def test_the_shipped_assets_exist() -> None:
    """The wheel must carry both, or a user's first notification has nothing
    to build and nothing to show. `pyproject.toml` lists them explicitly;
    this fails if either is dropped from the tree."""
    source = Path(notifier_app.__file__).parent
    assert (source / "notifier.m").is_file()
    icon = source / "icon.icns"
    assert icon.is_file()
    # An .icns begins with 'icns'; a PNG renamed to .icns renders as nothing.
    assert icon.read_bytes()[:4] == b"icns"


def test_the_bundle_declares_our_identity_and_icon() -> None:
    """The Info.plist is what macOS reads to name and illustrate the sender."""
    info = notifier_app._info_plist()

    assert info["CFBundleIdentifier"] == "me.damiantran.localoperator"
    assert info["CFBundleName"] == "Local Operator"
    # Without an icon FILE KEY the bundle renders as the generic placeholder
    # even with icon.icns present in Resources — the exact defect reported.
    assert info["CFBundleIconFile"] == "icon"
    # A background helper must not bounce into the Dock to post a toast.
    assert info["LSUIElement"] is True


def test_an_unbuilt_bundle_is_not_reported_as_ready(tmp_path: Path) -> None:
    """`is_built` gates the fast path; a false positive would post through a
    binary that does not exist and silently deliver nothing."""
    assert notifier_app.is_built(notifier_app.bundle_root(tmp_path)) is False


def test_a_bundle_from_an_older_release_is_rebuilt(tmp_path: Path) -> None:
    """Existence is not readiness. A bundle built before the icon (or before a
    `notifier.m` fix) must be rebuilt rather than reused forever — which is
    also how a stale cached icon reaches users, since a rebuild re-runs
    LaunchServices registration."""
    app = notifier_app.bundle_root(tmp_path)
    (app / "Contents" / "MacOS").mkdir(parents=True)
    (app / "Contents" / "MacOS" / "notifier").write_text("stale")
    (app.parent / ".build-stamp").write_text("0")

    assert notifier_app.is_built(app) is False


def test_the_notify_argv_carries_the_click_command_only_when_given(
    tmp_path: Path,
) -> None:
    """A toast with no session has nothing to reopen, and the helper must not
    wait on a run loop for an activation that can never be actioned."""
    app = notifier_app.bundle_root(tmp_path)

    plain = notifier_app.notify_command(app, "title", "body")
    clickable = notifier_app.notify_command(app, "title", "body", "echo hi")

    assert plain == [str(app / "Contents" / "MacOS" / "notifier"), "title", "body"]
    assert clickable[-1] == "echo hi"


def test_a_cold_machine_does_not_block_the_gate_path(tmp_path: Path) -> None:
    """`ensure_bundle` must answer immediately when nothing is built yet.

    The caller is a runtime announcing a parked gate; waiting on a compiler
    there would put a build on the path of a notification. The first
    notification goes out the old way and the build happens behind it.
    """
    assert notifier_app.ensure_bundle(tmp_path, block=False) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="the bundle is macOS-only")
def test_a_real_build_produces_a_registered_bundle(tmp_path: Path) -> None:
    """The whole build, for real: compile, sign, register, stamp.

    Skipped off darwin and tolerant of a machine with no compiler — both are
    ordinary answers this path already degrades to.
    """
    import shutil

    if shutil.which("clang") is None:
        pytest.skip("no compiler on this machine")

    app = notifier_app.build_bundle(tmp_path)
    assert app is not None

    assert (app / "Contents" / "MacOS" / "notifier").is_file()
    assert (app / "Contents" / "Resources" / "icon.icns").is_file()
    info = plistlib.loads((app / "Contents" / "Info.plist").read_bytes())
    assert info["CFBundleIdentifier"] == "me.damiantran.localoperator"
    assert notifier_app.is_built(app) is True


def test_a_crashed_build_does_not_disable_the_bundle_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A `.building` marker is a lock, not a tombstone.

    The builder is a daemon thread in a runtime that exits as soon as its
    work is done — plus SIGKILL, OOM and power loss — so the process can die
    between creating the marker and removing it. Treating that as "a build is
    in progress" disabled the identity bundle PERMANENTLY and invisibly:
    every later `ensure_bundle` returned None while notifications kept
    working through the fallback, so nothing ever surfaced the failure
    (round 3, B4).

    Asserted on the decision rather than by building: an old marker must be
    reclaimed, a fresh one must be respected (or two runtimes racing on first
    use would both start a compiler).

    Hold a real builder thread at its entry event so its finally block cannot
    unlink the marker between exists() and stat(), the old test's CI race.
    Thread-start publication proves the decision synchronously; the timeout
    only guards a wedged thread, never determines whether reclamation happened.
    No compiler or platform-dependent build speed participates in this test.
    """
    import os
    import threading
    from types import SimpleNamespace

    entered = threading.Event()
    release = threading.Event()
    builders: list[threading.Thread] = []
    errors: list[str] = []
    test_thread_id = threading.get_ident()

    class ObservedThread(threading.Thread):
        def start(self) -> None:
            # The module imports threading inside the launch function. This
            # shared spy must ignore work surviving from earlier fixtures.
            if self.name == "notifier-build" and threading.get_ident() == test_thread_id:
                builders.append(self)
            super().start()

    def held_build(_config_dir: Path) -> None:
        entered.set()
        if not release.wait(timeout=10):
            errors.append("the test never released the builder")

    monkeypatch.setattr(threading, "Thread", ObservedThread)
    monkeypatch.setattr(notifier_app, "build_bundle", held_build)
    fresh = 1000.0
    monkeypatch.setattr(notifier_app, "time", SimpleNamespace(time=lambda: fresh))

    lock = tmp_path / "notifier" / ".building"
    lock.parent.mkdir(parents=True, exist_ok=True)
    lock.write_text("original marker")
    try:
        # A current marker must not even launch a second builder thread.
        assert notifier_app._BUILD_LOCK_STALE_S > 0
        os.utime(lock, (fresh, fresh))
        notifier_app._build_in_background(tmp_path)
        assert builders == [], "a fresh marker must be respected"
        assert lock.read_text() == "original marker"

        old = fresh - notifier_app._BUILD_LOCK_STALE_S - 60
        os.utime(lock, (old, old))
        notifier_app._build_in_background(tmp_path)
        assert len(builders) == 1, "a stale marker must start a replacement builder"
        assert entered.wait(timeout=10), "the replacement builder never entered"
        # The original marker is replaced before the thread starts; the held
        # builder owns this new marker until the test explicitly releases it.
        assert lock.read_text() == ""
    finally:
        release.set()
        for builder in builders:
            builder.join(timeout=10)
            assert not builder.is_alive(), "the builder did not finish after release"

    assert errors == []
    assert not lock.exists(), "the completed builder must remove its marker"
