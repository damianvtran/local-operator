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
