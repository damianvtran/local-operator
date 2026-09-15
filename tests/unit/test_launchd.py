"""Guards for the shared LaunchAgent helpers — addressability and repair.

WHAT IS AT RISK HERE, and therefore what these tests pin:

1. **The addressability test is IDENTITY, not containment.** ``launchctl``
   addresses the calling user's live session whatever ``Path.home()`` says, so a
   test or sandbox that redirects ``HOME`` into a tmpdir must not reach the real
   LaunchAgents. The containment shortcut fails on the shape that is NOT exotic:
   a redirected home that lands *inside* the real one. One such run evicted
   ``com.local-operator.browser`` from the operator's live session (recorded in
   ``browser_bridge/install.py``).
2. **A repair must not change anything it was not asked about.** Byte content is
   compared, not existence or mtime, and a rewrite preserves the on-disk mode —
   the tunnel plist is written 0600 and must stay there.
3. **Every function is no-raise by contract.** A stale plist is decoration on a
   process listing; an upgrade that has already succeeded must never fail on it.

The end-to-end behaviour (a stale plist really being rewritten and the daemon
restarted on the operator's machine) is on the PR as live ``ps``/``plutil``
captures: it cannot be asserted here without touching a real launchd session.
"""

from __future__ import annotations

import os
import plistlib
from pathlib import Path

import pytest

from local_operator import launchd
from local_operator.paths import CONFIG_DIR_ENV

PLIST = "com.local-operator.mobile"


def _plist_path(home: Path) -> Path:
    return home / "Library" / "LaunchAgents" / f"{PLIST}.plist"


class TestAddressability:
    """``is_own_plist`` — the guard that keeps a sandbox off the real session."""

    def test_the_real_home_path_is_addressable(self) -> None:
        home = launchd.real_home()
        assert home is not None
        assert launchd.is_own_plist(_plist_path(home), PLIST) is True

    def test_a_redirected_home_is_refused(self, tmp_path: Path) -> None:
        assert launchd.is_own_plist(_plist_path(tmp_path), PLIST) is False

    def test_containment_would_have_passed_and_identity_does_not(self, monkeypatch) -> None:
        """A home nested inside the real one is the failure this pins.

        ``TMPDIR`` under ``$HOME`` is ordinary on macOS, so "is the plist inside
        the real home?" answers *yes* for a sandbox — and then the guard lets a
        sandbox rewrite the operator's daemon.
        """
        home = launchd.real_home()
        assert home is not None
        nested = home / "sandbox-home"
        monkeypatch.setattr(launchd, "real_home", lambda: nested)
        # The path the REAL home produces (what launchctl would actually act on)
        # is refused once the identity is taken from the redirected home.
        assert launchd.is_own_plist(_plist_path(home), PLIST) is False
        # ... and the nested home's own path is addressable, which is only ever
        # reachable when it is genuinely the passwd home.
        assert launchd.is_own_plist(_plist_path(nested), PLIST) is True

    def test_unreadable_passwd_refuses(self, monkeypatch) -> None:
        monkeypatch.setattr(launchd, "real_home", lambda: None)
        assert launchd.is_own_plist(Path("/anywhere") / PLIST, PLIST) is False


class TestConfigDirContainment:
    """``config_lives_in_real_home`` — the store-outlives-us guard."""

    def test_a_sandbox_store_is_refused(self, tmp_path: Path) -> None:
        assert launchd.config_lives_in_real_home(tmp_path / "sandbox-cfg") is False

    def test_a_store_under_the_real_home_is_accepted(self) -> None:
        home = launchd.real_home()
        assert home is not None
        assert launchd.config_lives_in_real_home(home / ".local-operator") is True


class TestArgValue:
    """Reading a plist's argv for its ARGUMENTS, never for an interpreter."""

    def test_separate_and_equals_forms(self) -> None:
        assert launchd.arg_value({"ProgramArguments": ["x", "--port", "4098"]}, "--port") == "4098"
        assert launchd.arg_value({"ProgramArguments": ["x", "--port=5000"]}, "--port") == "5000"

    @pytest.mark.parametrize(
        "plist",
        [
            None,
            {},
            {"ProgramArguments": "not-a-list"},
            {"ProgramArguments": ["x", "--other", "1"]},
            {"ProgramArguments": ["x", "--port"]},
            {"ProgramArguments": ["x", "--port", 4098]},
        ],
    )
    def test_absent_or_unusable_is_none(self, plist) -> None:
        assert launchd.arg_value(plist, "--port") is None

    def test_int_arg_keeps_the_configured_port(self) -> None:
        """A repair must not move a daemon off a non-default port."""
        assert launchd.int_arg({"ProgramArguments": ["x", "--port=5000"]}, "--port", 4098) == 5000

    def test_int_arg_falls_back_rather_than_guessing(self) -> None:
        assert launchd.int_arg({"ProgramArguments": ["x", "--port", "abc"]}, "--port", 4098) == 4098
        assert launchd.int_arg(None, "--port", 4098) == 4098


class TestConfigDirFromPlist:
    def test_reads_the_recorded_store(self, tmp_path: Path) -> None:
        plist: dict[str, object] = {"EnvironmentVariables": {CONFIG_DIR_ENV: str(tmp_path)}}
        assert launchd.config_dir_from_plist(plist) == tmp_path

    @pytest.mark.parametrize(
        "plist", [None, {}, {"EnvironmentVariables": "nope"}, {"EnvironmentVariables": {}}]
    )
    def test_absent_is_none(self, plist) -> None:
        assert launchd.config_dir_from_plist(plist) is None


class TestRewriteIfStale:
    """Content, not existence — and nothing else touched."""

    def test_absent_is_not_installed(self, tmp_path: Path) -> None:
        outcome = launchd.rewrite_if_stale(
            name="mobile", path=tmp_path / "missing.plist", rendered={"Label": "x"}
        )
        assert outcome == launchd.PlistRefresh(name="mobile", kind="not-installed")
        assert outcome.summary() == ""
        assert outcome.warning() == ""

    def test_current_is_left_untouched(self, tmp_path: Path) -> None:
        rendered = {"Label": "com.local-operator.mobile", "ProgramArguments": ["a", "-m", "b"]}
        path = tmp_path / PLIST
        path.write_bytes(plistlib.dumps(rendered))
        before = path.stat().st_mtime_ns
        outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered=dict(rendered))
        assert outcome.kind == "current"
        assert path.stat().st_mtime_ns == before, "an up-to-date plist was rewritten anyway"

    def test_stale_is_rewritten_and_says_so(self, tmp_path: Path) -> None:
        path = tmp_path / PLIST
        path.write_bytes(plistlib.dumps({"Label": "com.local-operator.mobile", "Legacy": True}))
        rendered = {
            "Label": "com.local-operator.mobile",
            "Program": "/prefix/bin/Local Operator",
            "ProgramArguments": ["Local Operator [mobile daemon] port=4098", "-m", "svc"],
        }
        outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered=rendered)
        assert outcome.kind == "repaired"
        assert plistlib.loads(path.read_bytes()) == rendered
        assert "mobile" in outcome.summary()

    def test_a_rewrite_preserves_the_mode(self, tmp_path: Path) -> None:
        """The tunnel plist is 0600; a repair must not widen it."""
        path = tmp_path / PLIST
        path.write_bytes(plistlib.dumps({"Label": "old"}))
        path.chmod(0o600)
        launchd.rewrite_if_stale(name="tunnel", path=path, rendered={"Label": "new"})
        assert path.stat().st_mode & 0o777 == 0o600

    def test_a_corrupt_plist_is_repaired_rather_than_crashing(self, tmp_path: Path) -> None:
        path = tmp_path / PLIST
        path.write_bytes(b"not a plist at all")
        outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered={"Label": "new"})
        assert outcome.kind == "repaired"

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores the file mode")
    def test_an_unwritable_plist_fails_without_raising(self, tmp_path: Path) -> None:
        path = tmp_path / f"{PLIST}.plist"
        path.write_bytes(plistlib.dumps({"Label": "old"}))
        path.chmod(0o400)
        try:
            outcome = launchd.rewrite_if_stale(name="mobile", path=path, rendered={"Label": "new"})
        finally:
            path.chmod(0o600)
        assert outcome.kind == "failed"
        assert outcome.warning().startswith("warning: ")

    def test_the_recorded_install_prefix_reads_both_plist_shapes(self) -> None:
        """The test the repair's identity guard runs on.

        Prefix equality rather than path equality is the point: a stale plist
        recording ``<prefix>/bin/python3`` and the branded shape recording
        ``<prefix>/bin/Local Operator`` are the SAME install, so a repair that
        upgrades one into the other must not read as "another installation".
        """
        branded = {"Program": "/opt/tool/bin/Local Operator"}
        legacy = {
            "ProgramArguments": ["/opt/tool/bin/python3", "-m", "local_operator.wakes.supervisor"]
        }
        assert launchd.recorded_install_prefix(branded) == Path("/opt/tool")
        assert launchd.recorded_install_prefix(legacy) == Path("/opt/tool")

    def test_the_recorded_install_prefix_gives_up_rather_than_guessing(self) -> None:
        """``None`` is "cannot tell", which callers must treat as no objection.

        The branded shape carries the image in ``Program``, so a LABEL in
        ``ProgramArguments[0]`` is not a path and must not be read as one — a
        guess here would refuse a legitimate repair on a machine whose plist
        this code simply does not understand.
        """
        assert launchd.recorded_install_prefix({}) is None
        assert launchd.recorded_install_prefix(None) is None
        assert launchd.recorded_install_prefix({"ProgramArguments": []}) is None
        assert (
            launchd.recorded_install_prefix({"ProgramArguments": ["Local Operator [wakes]"]})
            is None
        )
