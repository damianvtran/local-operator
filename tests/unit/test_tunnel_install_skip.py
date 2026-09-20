"""The tunnel installer must not write or churn launchd when nothing changed.

WHY THIS IS A FILE OF ITS OWN, next to ``test_tunnel_install.py`` (which covers
the platform arms and the refusal messages): the behaviour added here is the
opposite of "do more". With the generation shim the plist path is stable, so a
re-install usually renders the SAME bytes, and rewriting them plus a
bootout/bootstrap is exactly what an EDR reads as "Persistence: launchd job /
plist file modification" (MITRE T1543.001) with no functional change behind it.
``wakes.install`` has compared content before acting since it shipped; these
cells pin the same shape here, in both directions, with the launchctl call log
asserted directly.

Darwin-only: off macOS this daemon is a systemd unit or a scheduled task, and
neither is re-created the same way.
"""

from __future__ import annotations

import plistlib
import sys
from pathlib import Path

import pytest

from local_operator import launchd, supervisors
from local_operator.tunnels import install

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the plist lifecycle is macOS-only; Linux/Windows have their own arms",
)


class FakeProc:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Rig:
    """``install()`` with the file system pointed at ``tmp_path`` and calls recorded."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.path = tmp_path / "com.local-operator.tunnel.plist"
        self.calls: list[list[str]] = []
        self.loaded = True
        self.reloads: list[Path] = []
        # Stated rather than inherited from the runner's OS, like the mobile
        # installer's own fixture does.
        monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
        monkeypatch.setattr(install, "service_path", lambda: self.path)
        monkeypatch.setattr(install.config, "directory", lambda base=None: tmp_path)
        monkeypatch.setattr(install, "_launchctl", self._launchctl)
        monkeypatch.setattr(install.launchd, "reload_job", self._reload_job)

    def _launchctl(self, *cmd: str) -> FakeProc:
        self.calls.append(list(cmd))
        if cmd[:1] == ("print",):
            # `pid = …` is what launchd prints only while a process is alive:
            # this is the reading `launchd.job_running` parses.
            if self.loaded:
                return FakeProc(returncode=0, stdout="\tpid = 4242\n")
            return FakeProc(returncode=113, stdout="Bad request.", stderr="Could not find service")
        if cmd[:2] == ("kickstart", "-k"):
            self.loaded = True
        return FakeProc(returncode=0)

    def _reload_job(self, **kwargs: object) -> launchd.JobReload:
        self.reloads.append(kwargs["path"])  # type: ignore[arg-type]
        return launchd.JobReload(label=str(kwargs["label"]), outcome="reloaded")

    def write_current_plist(self) -> None:
        payload = plistlib.dumps(install.render_plist())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(payload)
        self.path.chmod(0o600)


def test_a_reinstall_that_changes_nothing_does_not_write_or_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Equal bytes and a live job: no write, no bootout/bootstrap, no kickstart.

    The probe is the ONLY call allowed, and it is the call that makes the skip
    safe — launchd is the authority on whether this job is running.
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()
    before = rig.path.stat().st_mtime_ns

    install.install()

    assert rig.path.stat().st_mtime_ns == before, "the plist was rewritten"
    assert rig.reloads == [], "an unchanged plist must not be reloaded"
    assert rig.calls == [["print", f"{launchd.job_domain()}/{install.LABEL}"]], rig.calls


def test_a_reinstall_still_writes_and_reloads_a_changed_plist(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A plist from an older build is still replaced and loaded."""
    rig = _Rig(monkeypatch, tmp_path)
    rig.path.parent.mkdir(parents=True, exist_ok=True)
    rig.path.write_bytes(plistlib.dumps({"Label": install.LABEL, "stale": True}))
    rig.path.chmod(0o600)

    install.install()

    assert plistlib.loads(rig.path.read_bytes()) == install.render_plist()
    assert rig.reloads == [rig.path], "a changed plist must still be reloaded"


def test_a_wrong_mode_is_still_repaired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Content AND mode make "current": this installer is the one that sets 0600.

    A file whose bytes match but whose mode drifted is a real state (an archive
    restore, a hand edit) and the old unconditional write repaired it. Treating
    "bytes match" as enough would quietly stop doing that.
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()
    rig.path.chmod(0o644)

    install.install()

    assert (rig.path.stat().st_mode & 0o777) == 0o600
    assert rig.reloads == [rig.path]


def test_a_loaded_but_dead_job_is_restarted_without_rewriting_the_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The state the old unconditional reload repaired, still repaired.

    ``kickstart -k`` rather than bootout+bootstrap: the file is already correct,
    so the narrower operation is right here (the bootout pair is for a REWRITE,
    where launchd's in-memory definition is what has to be replaced).
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()
    rig.loaded = False
    before = rig.path.stat().st_mtime_ns

    install.install()

    assert rig.path.stat().st_mtime_ns == before, "the file was already correct"
    assert ["kickstart", "-k", f"{launchd.job_domain()}/{install.LABEL}"] in rig.calls
    assert rig.reloads == [], "a stopped job needs a restart, not a rewrite"


def test_a_label_launchd_forgot_falls_through_to_the_full_reload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Nothing registered ⇒ the reload IS the registration; skipping it would leave it dead."""
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()

    def unregistered(*cmd: str) -> FakeProc:
        rig.calls.append(list(cmd))
        # `print` fails for an unknown label AND the kickstart that follows it,
        # which is the real sequence: the only way back is bootout+bootstrap.
        return FakeProc(returncode=113, stdout="Bad request.", stderr='Could not find service "x"')

    monkeypatch.setattr(install, "_launchctl", unregistered)

    install.install()

    assert rig.reloads == [rig.path]


def test_the_log_file_is_still_prepared(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The skip is about the PLIST and launchd; the log path is still made ready.

    Pinned because "install became a no-op" is the natural over-reading of this
    change: the daemon writes its output there, and an install on a fresh store
    must still leave it writable.
    """
    rig = _Rig(monkeypatch, tmp_path)
    rig.write_current_plist()

    install.install()

    assert (tmp_path / "service.log").exists()
