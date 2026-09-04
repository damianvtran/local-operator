"""Install-on-demand for the wake supervisor.

**These tests must never install a real LaunchAgent.** ``launchctl`` has no
sandbox: it addresses the calling user's live session whatever ``Path.home()``
says, so the obvious installer test — patch home, call the installer — wrote a
plist under a pytest tmpdir and then bootstrapped a REAL supervised unit into
the developer's session pointed at that tmpdir. It was observed here:
``launchctl print gui/501/com.local-operator.wakes`` reported a live unit whose
plist lived under ``/private/var/folders/…/pytest-of-damian/``.

``_launchd_is_addressable`` is the guard that makes this file safe: the plist
is written wherever ``plist_path()`` points, but launchd is only addressed when
that path is inside the real passwd home. Every test below runs under a
redirected home, so the file half is exercised in full and the process half
refuses. A test that needs to assert on ``launchctl`` behaviour must fake the
subprocess, never call it.
"""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path
from typing import cast

import pytest

from local_operator.wakes.install import (
    LABEL,
    UNSUPPORTED_REASON,
    ensure_supervisor_installed,
    is_supported,
    plist_path,
    render_plist,
)


@pytest.fixture
def redirected_home(tmp_path: Path, monkeypatch) -> Path:  # noqa: ANN001
    """Point ``Path.home()`` at a tmpdir. The launchd guard does the rest."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    return home


def test_the_installer_never_touches_the_real_launch_agents_directory(
    redirected_home: Path,
) -> None:
    """The guard itself, asserted rather than assumed.

    This is the test that would have caught the stray unit: under a redirected
    home the installer must decline to reach launchd, whatever else it does.
    """
    from local_operator.wakes.install import _launchd_is_addressable

    assert _launchd_is_addressable() is False


def test_install_writes_a_plist_but_declines_to_load_it(
    redirected_home: Path, tmp_path: Path
) -> None:
    outcome = ensure_supervisor_installed(tmp_path / "config")

    if not is_supported():
        assert outcome.installed is False
        assert outcome.reason == UNSUPPORTED_REASON
        return

    # The file half runs in full: this is what launchd WOULD be handed.
    assert plist_path().exists()
    written = plistlib.loads(plist_path().read_bytes())
    assert written == render_plist(tmp_path / "config")
    # The process half refuses under a redirected home.
    assert outcome.installed is False
    assert "not addressable" in outcome.reason


@pytest.mark.skipif(sys.platform != "darwin", reason="LaunchAgent plist is macOS-only")
def test_the_plist_lets_a_finished_supervisor_stay_down(tmp_path: Path) -> None:
    """``KeepAlive: {SuccessfulExit: False}`` is load-bearing, not decoration.

    The supervisor exits 0 when the wake index empties. This key is what makes
    that exit stick — a plain ``KeepAlive: true`` would restart it forever
    against an empty index, which is exactly the always-on cost the design set
    out to avoid.
    """
    plan = render_plist(tmp_path)
    assert plan["KeepAlive"] == {"SuccessfulExit": False}
    assert plan["Label"] == LABEL
    argv = cast(list[str], plan["ProgramArguments"])
    assert argv[1:] == ["-m", "local_operator.wakes.supervisor"]
    # The config dir travels in the environment: a second profile (or a test)
    # must be able to supervise its own store rather than the default one.
    env = cast(dict[str, str], plan["EnvironmentVariables"])
    assert env["LOCAL_OPERATOR_CONFIG_DIR"] == str(tmp_path)


@pytest.mark.skipif(sys.platform != "darwin", reason="LaunchAgent plist is macOS-only")
def test_a_stale_plist_is_rewritten_rather_than_trusted(
    redirected_home: Path, tmp_path: Path
) -> None:
    """Idempotent by CONTENT, not by existence.

    A plist from an older release names a different interpreter or a different
    config dir. Treating "a file is there" as "installed" would leave that
    stale unit supervising the wrong store forever.
    """
    plist_path().parent.mkdir(parents=True, exist_ok=True)
    plist_path().write_bytes(plistlib.dumps({"Label": LABEL, "ProgramArguments": ["/bin/false"]}))

    ensure_supervisor_installed(tmp_path / "config")

    rewritten = plistlib.loads(plist_path().read_bytes())
    assert rewritten == render_plist(tmp_path / "config")


def test_install_never_raises_even_on_an_unwritable_target(
    redirected_home: Path, tmp_path: Path, monkeypatch
) -> None:
    """The caller is the wake persist path, which has already succeeded.

    An installer failure must be reported through the outcome and never
    propagate: the schedule is durable, and a wake the user asked for must not
    turn into an exception because a plist could not be written.
    """

    def _boom(*_args, **_kwargs):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(Path, "write_bytes", _boom)

    outcome = ensure_supervisor_installed(tmp_path / "config")
    assert outcome.installed is False
    assert outcome.reason


def test_the_guard_holds_when_the_redirected_home_is_inside_the_real_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure mode the other guard test structurally cannot reach.

    Round 1 (R4): the guard asked whether the plist path was *under* the real
    passwd home, which is a LOCATION test. Set `TMPDIR` inside `$HOME` — not
    exotic; it is how you avoid `/var/folders` cleanup races — and pytest's
    `tmp_path`, and so a patched `Path.home()`, lands inside the real home and
    satisfies containment. The guard would then wave a real launchd bootstrap
    through against a directory about to be deleted, which is the exact
    incident it exists to prevent.

    Every existing test uses `tmp_path`, whose location follows `TMPDIR`, so
    none of them can express this. This one builds the pathological home
    explicitly under the REAL passwd home and asserts the guard still refuses.
    """
    import pwd

    from local_operator.wakes.install import _launchd_is_addressable

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    # A path shaped like the dangerous case. Never created: the guard is a
    # pure path comparison, and this test must not write into the real home.
    pathological = real_home / "tmp-pytest-sandbox" / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: pathological))

    assert _launchd_is_addressable() is False, (
        "a redirected home INSIDE the real home was accepted as addressable; "
        "the guard has regressed from an identity test to a location test"
    )


def test_the_guard_accepts_only_the_genuine_home(monkeypatch: pytest.MonkeyPatch) -> None:
    """The positive half: the guard is not vacuously False.

    Without this, replacing the body with `return False` would pass every
    other test in this file while silently disabling install-on-demand for
    every real user.
    """
    import pwd

    from local_operator.wakes.install import _launchd_is_addressable, is_supported

    if not is_supported():
        pytest.skip("launchd guard is only meaningful on darwin with launchctl")

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: real_home))

    assert _launchd_is_addressable() is True


def test_the_write_guard_refuses_config_dirs_outside_the_real_home(
    tmp_path: Path,
) -> None:
    """The round-2 escape: real HOME, redirected config dir.

    The round-1 guard covered the `launchctl` call but not the WRITE, and the
    write is the half that escapes that combination: `plist_path()` is the
    real `~/Library/LaunchAgents` whenever HOME is real, so a sandbox run
    planted a supervised unit in the operator's live launchd domain, pointed
    at a store that dies with the sandbox.

    Asserted on the decision predicate rather than by running the installer
    against the real home: a test that exercises the escape end to end would
    REPRODUCE the incident on the very machine it guards if the guard ever
    regressed. The predicate is the whole decision — the branch that consumes
    it is two lines.
    """
    import pwd

    from local_operator.wakes.install import _config_lives_in_real_home

    real_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
    assert _config_lives_in_real_home(tmp_path / "sandbox-cfg") is False
    assert _config_lives_in_real_home(real_home / ".local-operator") is True
