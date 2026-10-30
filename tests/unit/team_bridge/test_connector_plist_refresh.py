"""The upgrade's repair for the bridge connector's LaunchAgent.

WHY THIS FILE EXISTS. ``connector/install.py`` wrote its plist at install time and
nothing ever read an old one back: the upgrade's refresh child walked four
daemons and the connector was not among them. Under the generation layout that
matters twice over, because the rendered plist is byte-identical across builds
(``Program`` is the stable shim) — so "the plist is current" says nothing about
the BUILD the running connector is serving, and an always-on connector left on an
older generation keeps HOLDING the account's machine-wide ``connector.lock``.
Measured 2026-10-30: pid 48311, generation 20261023T142851Z-0.86.1, seven days,
2,826 supervised starts refused exit 75.

WHAT IS PINNED, per cell:

* a stale plist is rewritten and the job reloaded through ``bootout`` +
  ``bootstrap`` — never ``kickstart -k``, which restarts launchd's in-memory
  definition and keeps the old argv;
* a CURRENT plist on a MOVED build is kickstarted exactly once, and a refused
  kickstart prints the one warning shape the shared vocabulary has
  (``warning: <name> daemon was not refreshed: …``);
* a current plist on a current build is read, not bounced;
* a redirected ``HOME`` is not addressable and ``launchctl`` is never reached —
  the guard that has already cost the operator one live daemon;
* a plist pointing at a store that is not the default one is not repointed at
  this process's own.

The store comes off the plist: the connector's plist carries no
``LOCAL_OPERATOR_CONFIG_DIR`` (only the ``LOCAL_OPERATOR_CONNECTOR`` marker), so
``--config-dir`` in ``ProgramArguments`` is the only honest source.
"""

from __future__ import annotations

import os
import plistlib
import sys
from pathlib import Path

import pytest

from local_operator import launchd, supervisors
from local_operator import update as update_mod
from local_operator.team_bridge.connector import install as connector_install

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="launchd's plists are macOS's; the other two platforms are out of scope here",
)

#: Two real generation shapes from the incident, one day apart.
OLD = "20261023T142851Z-0.86.1"
NEW = "20261030T145827Z-0.92.2"

#: Saved before any fixture patches them, so the ownership cells can put the REAL
#: guards back rather than testing a stub.
_REAL_SHARED_LABEL_REFUSAL = launchd.shared_label_refusal
_REAL_IS_OWN_PLIST = launchd.is_own_plist


class _Completed:
    """The shape ``launchd._call`` reads off a subprocess result."""

    def __init__(self, args: tuple[str, ...], code: int, out: str = "") -> None:
        self.args = args
        self.returncode = code
        self.stdout = out
        self.stderr = "" if code == 0 else "Bootstrap failed: 5: Input/output error"


@pytest.fixture
def calls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, ...]]:
    """A launchd that answers like launchd, and the call log it wrote.

    ``print`` reports a RUNNING job (``job_pid``'s only handle on a process),
    ``bootout`` unloads it, ``bootstrap`` loads it back, and ``kickstart`` brings
    up a NEW pid — so the reload and restart verification steps read a real
    sequence instead of a stub where everything trivially succeeds. The bounded
    waits are advanced, not spent.
    """

    log: list[tuple[str, ...]] = []
    clock = [0.0]
    monkeypatch.setattr(launchd, "_monotonic", lambda: clock[0])
    monkeypatch.setattr(launchd, "_sleep", lambda seconds: clock.__setitem__(0, clock[0] + seconds))
    monkeypatch.setattr(supervisors, "supervisor", lambda: "launchctl")
    monkeypatch.setattr(connector_install, "plist_path", lambda: tmp_path / "connector.plist")
    # The ownership guards have their own cells at the bottom of this file; the
    # happy paths open them explicitly so the repair under test is the repair.
    monkeypatch.setattr(launchd, "shared_label_refusal", lambda config: None)
    monkeypatch.setattr(launchd, "is_own_plist", lambda path, label: True)

    loaded = [True]
    live: list[int | None] = [4242]

    def runner(*args: str):
        log.append(args)
        verb = args[0]
        if verb == "print":
            if not loaded[0]:
                return _Completed(args, 1)
            return _Completed(args, 0, f"\tpid = {live[0]}\n" if live[0] is not None else "")
        if verb == "bootout":
            loaded[0], live[0] = False, None
            return _Completed(args, 0)
        if verb == "bootstrap":
            loaded[0], live[0] = True, live[0] or 4242
            return _Completed(args, 0)
        if verb == "kickstart":
            live[0] = (live[0] or 4242) + 1
            return _Completed(args, 0)
        return _Completed(args, 0)

    monkeypatch.setattr(connector_install, "_launchctl", runner)
    return log


def _layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A real generation layout: two trees and ``current`` pointing at the new one."""

    root = tmp_path / "lop"
    for name in (OLD, NEW):
        (root / "generations" / name).mkdir(parents=True, exist_ok=True)
    (root / "current").symlink_to(root / "generations" / NEW)
    monkeypatch.setattr(update_mod, "_STABLE_ROOT", str(root))


def _write(path: Path, store: Path, *, current: bool) -> None:
    """The plist this build would write, or the pre-branding shape."""

    if current:
        path.write_bytes(plistlib.dumps(connector_install.render_plist(store)))
        return
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": connector_install.LABEL,
                "ProgramArguments": [
                    sys.executable,
                    "-m",
                    connector_install.MODULE,
                    "--config-dir",
                    str(store),
                ],
                "RunAtLoad": True,
            }
        )
    )


def test_a_stale_plist_is_rewritten_and_the_connector_reloaded(
    calls: list[tuple[str, ...]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """bootout, wait for the label to go, bootstrap, verify — the shared sequence.

    A ``kickstart -k`` here would restart the job from launchd's IN-MEMORY
    definition and keep running the old argv, which is the defect the whole
    rewrite exists to avoid (measured in ``launchd``'s own docstring).
    """

    store = tmp_path / "config"
    path = tmp_path / "connector.plist"
    _write(path, store, current=False)

    outcome = connector_install.refresh_plist_if_stale()

    assert outcome.kind == "repaired", outcome
    written = plistlib.loads(path.read_bytes())
    assert written == connector_install.render_plist(store)
    assert [call[0] for call in calls] == ["bootout", "print", "bootstrap", "print"], calls
    assert calls[0] == ("bootout", f"gui/{os.getuid()}/{connector_install.LABEL}"), calls
    assert calls[2] == ("bootstrap", f"gui/{os.getuid()}", str(path)), calls


def test_a_current_plist_on_a_moved_build_is_restarted_exactly_once(
    calls: list[tuple[str, ...]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE INCIDENT'S SHAPE, and the cell that fails if the second question is
    dropped: the plist already says what this build would write, the running
    connector is serving an older generation, and a restart is the only repair
    that can move it — exactly one."""

    store = tmp_path / "config"
    path = tmp_path / "connector.plist"
    _write(path, store, current=True)
    before = path.read_bytes()
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        update_mod,
        "stale_generation_of_process",
        lambda pid: tmp_path / "lop" / "generations" / OLD,
    )

    outcome = connector_install.refresh_plist_if_stale()

    assert outcome.kind == "restarted", outcome
    assert OLD in outcome.summary()
    assert [call[0] for call in calls].count("kickstart") == 1, calls
    assert path.read_bytes() == before


def test_a_refused_kickstart_prints_the_shared_warning_shape(
    calls: list[tuple[str, ...]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failure line is the existing ``PlistRefresh.warning()``, unchanged.

    No new sentence: the upgrade summary prints ``warning: <name> daemon was not
    refreshed: <detail>`` for every daemon, and a daemon that invented its own
    format would be a second vocabulary the summary cannot read.
    """

    store = tmp_path / "config"
    path = tmp_path / "connector.plist"
    _write(path, store, current=True)
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(
        update_mod,
        "stale_generation_of_process",
        lambda pid: tmp_path / "lop" / "generations" / OLD,
    )
    # launchctl REFUSES the restart (non-zero, nothing asked to happen): the
    # job is left as it was, which is the ``failed`` shape rather than the
    # accepted-but-dead one.
    monkeypatch.setattr(
        connector_install,
        "_launchctl",
        lambda *args: _Completed(args, 1 if args[0] == "kickstart" else 0, "\tpid = 4242\n"),
    )

    outcome = connector_install.refresh_plist_if_stale()

    assert outcome.kind == "failed", outcome
    assert outcome.warning().startswith("warning: bridge connector daemon was not refreshed: ")
    assert "lop team-bridge receiver install" in outcome.warning(), outcome.warning()


def test_a_current_plist_on_a_current_build_is_read_not_bounced(
    calls: list[tuple[str, ...]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One read, no write, no reload: a healthy connector is not bounced."""

    store = tmp_path / "config"
    path = tmp_path / "connector.plist"
    _write(path, store, current=True)
    _layout(tmp_path, monkeypatch)
    monkeypatch.setattr(update_mod, "stale_generation_of_process", lambda pid: None)

    outcome = connector_install.refresh_plist_if_stale()

    assert outcome.kind == "current", outcome
    assert outcome.summary() == ""
    assert [call[0] for call in calls] == ["print"], calls


def test_a_redirected_home_is_not_addressable(
    calls: list[tuple[str, ...]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``launchctl`` addresses the REAL user's session whatever ``HOME`` says.

    The REAL ``launchd.is_own_plist`` runs here (patching it would test nothing):
    a fabricated passwd home stands in for the operator's, and the test's own
    ``HOME`` — where this file is written — is not it. Nothing outside the temp
    HOME is touched.
    """

    real_home = tmp_path / "real-home"
    (real_home / "Library" / "LaunchAgents").mkdir(parents=True)
    monkeypatch.setattr(launchd, "is_own_plist", _REAL_IS_OWN_PLIST)
    monkeypatch.setattr(launchd, "real_home", lambda: real_home)
    sandbox_plist = Path.home() / "Library" / "LaunchAgents" / f"{connector_install.LABEL}.plist"
    sandbox_plist.parent.mkdir(parents=True, exist_ok=True)
    _write(sandbox_plist, tmp_path / "config", current=True)
    monkeypatch.setattr(connector_install, "plist_path", lambda: sandbox_plist)

    assert launchd.is_own_plist(sandbox_plist, connector_install.LABEL) is False
    outcome = connector_install.refresh_plist_if_stale()

    assert outcome.kind == "not-addressable", outcome
    assert calls == []


def test_a_store_that_is_not_the_default_one_is_not_repaired(
    calls: list[tuple[str, ...]], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One shared label per user means one store may own it.

    ``launchd.shared_label_refusal`` is put back to its REAL implementation: a
    plist naming any root but the passwd home's default one is refused rather than
    repointed, which is the 2026-10-08 incident's rule (a sandbox store hijacked
    the operator's live supervisor).
    """

    real_home = tmp_path / "real-home"
    agents = real_home / "Library" / "LaunchAgents"
    agents.mkdir(parents=True)
    monkeypatch.setattr(launchd, "real_home", lambda: real_home)
    monkeypatch.setattr(launchd, "shared_label_refusal", _REAL_SHARED_LABEL_REFUSAL)
    # The plist IS the one launchd would read for this user (so `is_own_plist`
    # passes and the refusal under test is the store's), naming a scratch store.
    path = agents / f"{connector_install.LABEL}.plist"
    monkeypatch.setattr(connector_install, "plist_path", lambda: path)
    _write(path, tmp_path / "scratch-store", current=False)

    outcome = connector_install.refresh_plist_if_stale()

    assert outcome.kind == "not-addressable", outcome
    assert calls == []
    assert plistlib.loads(path.read_bytes())["Label"] == connector_install.LABEL
