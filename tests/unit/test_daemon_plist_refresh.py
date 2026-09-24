"""The stale-LaunchAgent repair: the four daemons ``lop-update`` never touched.

WHY THIS EXISTS. Three of the four installers wrote their plist at install time
and never read it again, and ``lop-update`` bounced only the mobile daemon — so a
daemon installed by a build that predates branding kept running ``python3.x``
forever. The measured instance is the reported symptom: a
``com.local-operator.mobile.plist`` (mtime Sep 5) running
``python3.14 -m local_operator.mobile.service --port 4098``, on a machine whose
*current* installer renders a branded link.

WHAT IS PINNED, per daemon:

- **stale -> rewritten AND restarted.** Restarted by ``bootout`` + ``bootstrap``,
  never ``kickstart -k``: measured on macOS, a kickstart after a rewrite restarts
  the job from launchd's in-memory definition and keeps running the OLD argv, so
  a kickstart-only repair silently repairs nothing.
- **current -> untouched**, with no ``launchctl`` call at all. A repair that
  bounces a healthy daemon on every upgrade is a different (and unpriced) change.
- **a current plist is not a current BUILD.** Under the generation layout the
  rendered plist is byte-identical across builds (``Program`` is the stable shim),
  so a daemon that is still serving a generation the pointer has left is only
  visible in its OWN argv — and the repair for it is a ``kickstart``, not a
  rewrite. The last section of this file pins that, including the unreadable-probe
  answers that must mean "no reload".
- **not installed -> untouched.** Nothing to repair is not a failure.
- **not addressable -> untouched.** ``launchctl`` always addresses the REAL
  user's session, whatever ``HOME`` says, so a sandboxed run must decline before
  it reaches launchd.
- **bootstrap failure -> reported, never raised.** The upgrade already succeeded.

The real end-to-end behaviour is on the PR as live ``ps``/``plutil`` captures.
"""

from __future__ import annotations

import os
import plistlib
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest

from local_operator import launchd, procname, supervisors
from local_operator import update as update_mod
from local_operator.browser_bridge import install as browser_install
from local_operator.mobile import install as mobile_install
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tunnels import config as tunnel_config
from local_operator.tunnels import install as tunnel_install
from local_operator.update import InstallKind
from local_operator.wakes import install as wakes_install

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="LaunchAgents are macOS-only; Linux units are re-read on every start",
)

#: ``(module, plist path, renderer, launchd label)`` for one daemon. Named
#: because the tuple is heterogeneous and would otherwise be spelled out in
#: every test signature.
Target = tuple[ModuleType, Path, Callable[[], dict[str, object]], str]

#: The pre-branding plist shape, byte-for-byte what an older build wrote: a bare
#: interpreter as element 0 and no ``Program`` key at all.
LEGACY_ARGV = ["-m"]


def _freeze_clock(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Replace ``launchd``'s clock, so a bounded wait becomes an assertion.

    ``reload_job`` waits for launchd to release a label and retries the
    bootstrap until a real deadline. Against a stubbed launchctl that never
    releases the label, the honest implementation spends that deadline in wall
    clock — seconds per test — so the tests advance it instead. Returned so a
    test can assert the retries were spaced rather than spun.
    """
    sleeps: list[float] = []
    now = [0.0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(launchd, "_monotonic", lambda: now[0])
    monkeypatch.setattr(launchd, "_sleep", sleep)
    return sleeps


def _patch_launcher(
    monkeypatch: pytest.MonkeyPatch,
    module,
    calls: list[tuple[str, ...]],
    *,
    fail: bool = False,
    pid: int | None = None,
    kickstart_fails: bool = False,
) -> None:
    """Install a launchd-shaped stand-in for one module's ``launchctl`` call.

    Three modules have a ``_launchctl(*args)`` helper and the tunnel has its
    own copy of one; the stand-in covers all four the same way.

    It answers like launchd rather than like a stub where every call succeeds,
    because that is the whole difference this change is about: ``print``
    resolves only while the label is loaded, ``bootout`` unloads it, and
    ``bootstrap`` loads it back unless the test is asking for the failure. The
    old "everything returns 0" fake is what let an un-sequenced pair look
    correct — and it would make the release wait look like a permanent stall.

    ``pid`` adds the other half of what ``print`` really answers: a RUNNING job
    prints a ``pid = <n>`` line (:func:`local_operator.launchd.job_pid`), which is
    the only handle the build probe has on the process. Without it the stand-in
    answers the stopped-job shape. ``kickstart_fails`` is a refusal of the
    restart itself, which is a different failure from ``fail``'s bootstrap one.
    """
    _freeze_clock(monkeypatch)

    class _Completed:
        def __init__(self, args: list[str], returncode: int) -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = ""
            self.stderr = "" if returncode == 0 else "Bootstrap failed: 5: Input/output error"

    loaded = [pid is not None]

    def fake(*args: str):
        calls.append(args)
        verb = args[0]
        if verb == "print":
            if not loaded[0]:
                return _Completed(list(args), 1)
            completed = _Completed(list(args), 0)
            if pid is not None:
                completed.stdout = f"\tpid = {pid}\n"
            return completed
        if verb == "bootout":
            loaded[0] = False
            return _Completed(list(args), 0)
        if verb == "kickstart" and kickstart_fails:
            return _Completed(list(args), 1)
        if verb == "bootstrap" and fail:
            return _Completed(list(args), 1)
        if verb == "bootstrap":
            loaded[0] = True
        return _Completed(list(args), 0)

    monkeypatch.setattr(module, "_launchctl", fake)


def _modules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Target]:
    """``name -> (module, plist path, render, label)`` for the four daemons.

    ``plist_path`` is patched to ``tmp_path`` on every one of them: the repair's
    job is to touch the file launchd resolves, and these tests care about which
    file that is, not about ``~/Library/LaunchAgents``.
    """
    sentinel = tmp_path / "bin" / procname.BRAND
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.touch()
    monkeypatch.setattr(procname, "ensure_branded_interpreter", lambda: sentinel)
    monkeypatch.setattr(launchd, "is_own_plist", lambda path, label: True)
    # The isolated test HOME is not the passwd home, so a real containment check
    # would (correctly) refuse the two daemons that record a config dir. Their
    # own guard is pinned in `test_launchd.py`; here it is the repair under test.
    monkeypatch.setattr(launchd, "config_lives_in_real_home", lambda config: True)
    monkeypatch.setattr(wakes_install, "_config_lives_in_real_home", lambda config: True)
    # The supervisor IDENTITY is what the four repairs branch on now (a repair is
    # launchd's, and the other two platforms' units are handled elsewhere), so
    # the seam is `supervisors.supervisor` — one patch covering mobile, wakes and
    # the browser bridge, whose own `_supervisor()` delegates to it.
    monkeypatch.setattr(supervisors, "supervisor", lambda: "launchctl")
    monkeypatch.setattr(browser_install, "_supervisor", lambda: "launchctl")

    store = tmp_path / "config-root"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(store))

    targets = {}
    for name, module, path_attr, render, label in (
        (
            "mobile",
            mobile_install,
            "plist_path",
            lambda: mobile_install.render_plist(4098),
            mobile_install.LABEL,
        ),
        (
            "browser bridge",
            browser_install,
            "plist_path",
            lambda: browser_install.render_plist(4099),
            # `label()`, not the `LABEL` constant: this root's registration is
            # addressed under its own suffixed label, which is the label the
            # reload must name. Using the bare constant here would assert the
            # wrong name and, worse, hand the fixture a plist whose Label
            # disagrees with what the installer renders.
            browser_install.label(),
        ),
        (
            "tunnel",
            tunnel_install,
            "service_path",
            lambda: tunnel_install.render_plist(),
            tunnel_install.LABEL,
        ),
        (
            "wakes supervisor",
            wakes_install,
            "plist_path",
            lambda: wakes_install.render_plist(store),
            wakes_install.LABEL,
        ),
    ):
        path = tmp_path / f"{label}.plist"
        monkeypatch.setattr(module, path_attr, lambda path=path: path)
        targets[name] = (module, path, render, label)
    return targets


@pytest.fixture
def targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Target]:
    return _modules(tmp_path, monkeypatch)


def _write_legacy(path: Path, label: str, module: str) -> None:
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": label,
                "ProgramArguments": [sys.executable, "-m", module],
                "RunAtLoad": True,
            }
        )
    )


def _module_for(name: str) -> str:
    return {
        "mobile": "local_operator.mobile.service",
        "browser bridge": "local_operator.browser_bridge.daemon",
        "tunnel": "local_operator.tunnels.service",
        "wakes supervisor": "local_operator.wakes.supervisor",
    }[name]


#: The recovery command each daemon's repair must name when it leaves the job
#: stopped. Spelled out here rather than imported so a change to one of the
#: installer's strings has to be a decision in this file too. A SECOND surface now
#: names the same command — the upgrade summary's killed-refresh report, which
#: carries it in the refresh child's own announcement — and
#: `test_the_killed_refresh_reports_the_vocabulary_this_file_pins` holds the two
#: together, because a summary that sends the operator to a command that no longer
#: exists is the same defect this file was written about.
_RECOVERY = {
    "mobile": "lop mobile install",
    "browser bridge": "lop browser install",
    "tunnel": "lop tunnel install",
    "wakes supervisor": "lop wake install",
}


def test_the_killed_refresh_reports_the_vocabulary_this_file_pins() -> None:
    """One recovery command per daemon, whichever surface is doing the naming.

    ``update._refresh_steps`` is the refresh CHILD's table: the child announces the
    daemon it is about to repair, and the parent reads that announcement out of a
    child its bound has just killed, so the daemon's name and its installer are
    printed by code that a second spelling could drift away from.
    """
    from local_operator import update as update_mod

    pinned = {name: recovery for name, recovery, _repair in update_mod._refresh_steps()}
    assert pinned == _RECOVERY


@pytest.mark.parametrize("name", ["mobile", "browser bridge", "tunnel", "wakes supervisor"])
def test_a_stale_plist_is_rewritten_and_the_daemon_restarted(
    name: str, targets: dict[str, Target], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, path, render, label = targets[name]
    _write_legacy(path, label, _module_for(name))
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls)

    outcome = module.refresh_plist_if_stale()

    assert outcome.kind == "repaired", (name, outcome)
    assert name in outcome.summary()
    written = plistlib.loads(path.read_bytes())
    assert written == render(), name
    assert written["ProgramArguments"][0].startswith(procname.BRAND), name
    # Bootout, then wait for the label to be released, then bootstrap, then
    # verify: a kickstart would restart the OLD in-memory definition and the
    # rewrite would have been pointless, and a bare bootout+bootstrap pair races
    # launchd's teardown (the defect this sequence exists to fix).
    assert [call[0] for call in calls] == ["bootout", "print", "bootstrap", "print"], calls
    # Addressed by DOMAIN and LABEL, never by plist path: `launchctl print` on a
    # path is not a thing, and the bootout must name the job launchd knows.
    assert calls[0] == ("bootout", f"gui/{os.getuid()}/{label}"), calls
    # The bootstrap hands launchd the FILE it re-reads.
    assert calls[2] == ("bootstrap", f"gui/{os.getuid()}", str(path)), calls
    assert calls[1] == ("print", f"gui/{os.getuid()}/{label}"), calls
    assert calls[3] == ("print", f"gui/{os.getuid()}/{label}"), calls


@pytest.mark.parametrize("name", ["mobile", "browser bridge", "tunnel", "wakes supervisor"])
def test_a_current_plist_is_not_touched_at_all(
    name: str, targets: dict[str, Target], monkeypatch: pytest.MonkeyPatch
) -> None:
    """No rewrite, no restart: a healthy daemon is not bounced on every upgrade.

    The zero-call property holds here for a reason worth stating, because the
    repair now asks a SECOND question after this one: a machine with no generation
    layout has no build question to ask, so the reading is not made and
    ``launchctl`` is never reached — see
    ``test_a_machine_without_the_generation_layout_is_not_probed_at_all``, which
    injects a probe that says "moved" and shows nothing happens. On a machine WITH
    the layout the same repair reads (one ``print``, no write, no reload): see
    ``test_a_current_plist_on_a_current_build_is_only_read_not_bounced``.
    """
    module, path, render, _label = targets[name]
    path.write_bytes(plistlib.dumps(render()))
    before = path.read_bytes()
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls)

    outcome = module.refresh_plist_if_stale()

    assert outcome.kind == "current", (name, outcome)
    assert outcome.summary() == ""
    assert path.read_bytes() == before
    assert calls == []


@pytest.mark.parametrize("name", ["mobile", "browser bridge", "tunnel", "wakes supervisor"])
def test_an_absent_plist_is_left_alone(
    name: str, targets: dict[str, Target], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, path, _render, _label = targets[name]
    assert not path.exists()
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls)

    outcome = module.refresh_plist_if_stale()

    assert outcome.kind == "not-installed", (name, outcome)
    assert calls == []


@pytest.mark.parametrize("name", ["mobile", "browser bridge", "tunnel", "wakes supervisor"])
def test_a_sandbox_never_reaches_launchd(
    name: str, targets: dict[str, Target], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The guard that has already cost the operator one live daemon once."""
    module, path, render, label = targets[name]
    _write_legacy(path, label, _module_for(name))
    before = path.read_bytes()
    monkeypatch.setattr(launchd, "is_own_plist", lambda path, label: False)
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls)

    outcome = module.refresh_plist_if_stale()

    assert outcome.kind == "not-addressable", (name, outcome)
    assert path.read_bytes() == before
    assert calls == []
    assert render() is not None  # the renderer is never even needed


@pytest.mark.parametrize("name", ["mobile", "browser bridge", "tunnel", "wakes supervisor"])
def test_a_failed_bootstrap_is_reported_not_raised(
    name: str, targets: dict[str, Target], monkeypatch: pytest.MonkeyPatch
) -> None:
    module, path, _render, label = targets[name]
    _write_legacy(path, label, _module_for(name))
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls, fail=True)

    outcome = module.refresh_plist_if_stale()

    assert outcome.kind == "failed", (name, outcome)
    assert "launchctl could not load it" in outcome.detail, outcome
    # The job is DOWN at this point — bootout already succeeded — so the detail
    # must name the command that brings it back, and the warning the upgrade
    # prints must carry the same sentence rather than swallowing it.
    recovery = _RECOVERY[name]
    assert f"run `{recovery}` to reinstall it" in outcome.detail, outcome
    assert "STOPPED" in outcome.detail, outcome
    assert recovery in outcome.warning(), outcome
    assert outcome.warning().startswith("warning: "), outcome


@pytest.mark.parametrize("name", ["mobile", "browser bridge", "tunnel", "wakes supervisor"])
def test_a_raising_repair_never_escapes(name: str, targets: dict[str, Target], monkeypatch) -> None:
    """``lop-update`` has already succeeded; nothing here may roll that back."""
    module, path, _render, label = targets[name]

    def explode(*args: object, **kwargs: object):
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "_launchctl", explode, raising=False)
    monkeypatch.setattr(module.subprocess, "run", explode, raising=False)
    monkeypatch.setattr(launchd, "load", explode)

    outcome = module.refresh_plist_if_stale()

    assert outcome.kind == "failed", (name, outcome)
    assert "boom" in outcome.detail


def test_the_tunnel_repairs_the_store_its_own_plist_names(
    targets: dict[str, Target], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A repair brings a unit up to date IN PLACE; it does not migrate it.

    The config dir comes off the plist being replaced, so a daemon supervising a
    non-default store is not silently pointed at this process's own.
    """
    module, path, _render, label = targets["tunnel"]
    recorded = tmp_path / "some-other-store"
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": label,
                "ProgramArguments": [sys.executable, "-m", "local_operator.tunnels.service"],
                "EnvironmentVariables": {CONFIG_DIR_ENV: str(recorded)},
            }
        )
    )
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls)

    assert module.refresh_plist_if_stale().kind == "repaired"
    written = plistlib.loads(path.read_bytes())
    assert written["EnvironmentVariables"][CONFIG_DIR_ENV] == str(recorded)
    assert str(recorded) in written["StandardOutPath"]
    assert tunnel_config.directory(recorded) == recorded / "tunnel"


def test_the_port_is_taken_from_the_plist_being_replaced(
    targets: dict[str, Target], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon installed on a non-default port is repaired on THAT port."""
    module, path, _render, label = targets["browser bridge"]
    path.write_bytes(
        plistlib.dumps(
            {
                "Label": label,
                "ProgramArguments": [
                    sys.executable,
                    "-m",
                    "local_operator.browser_bridge.daemon",
                    "--port",
                    "4199",
                ],
            }
        )
    )
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls)

    assert module.refresh_plist_if_stale().kind == "repaired"
    written = plistlib.loads(path.read_bytes())
    assert written["ProgramArguments"][-1] == "4199"


# ---------------------------------------------------------------------------
# THE SECOND STALENESS QUESTION: a current plist can still be a stale BUILD.
#
# WHY, MEASURED. Every supervised unit names the stable shim
# (`~/.local/share/lop/bin/python3`), and the shim resolves `current` ONCE at exec,
# so the rendered plist is byte-identical across builds and `rewrite_if_stale`
# answers "current" for all of them. Live on the operator's machine 2026-09-24:
# four byte-identical plists, the tunnel and mobile daemons on the current
# generation, and the wakes supervisor and browser bridge two and three generations
# behind (`ps -o pid=,ppid=,args=` against their own launchd pids). The consequence
# is why this exists at all: a RELEASED fix (PR #1509, v0.62.22) could not reach the
# tunnel connector until an explicit `lop tunnel restart` moved pid 1206 to pid
# 59435 — and the generation pid 1206 had been serving was by then pruned, so it
# had been importing from a deleted tree for three days.
#
# WHAT IS PINNED. The repair asks launchd for the running pid and the PROCESS for
# the generation it was started from (`update.stale_generation_of_process`, whose own
# answers are pinned in tests/unit/test_daemon_build_probe.py), then kicks at most
# ONCE — and only when the build provably moved. The child under test is the shipped
# `daemons_refresh_command()`, so the announce/report plumbing the upgrade summary
# reads is exercised too, not just the one function.
# ---------------------------------------------------------------------------

#: One pid per daemon, as `launchctl print` would report one. Distinct on purpose: a
#: repair that probed the wrong daemon's pid would still "work" with one shared value.
_PIDS = {"mobile": 111, "browser bridge": 222, "tunnel": 333, "wakes supervisor": 444}


def _layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    running: str,
    current: str,
    pointer: bool = True,
) -> Path:
    """A real generation layout under this test's stable root, and its running tree.

    ``_STABLE_ROOT`` is patched rather than ``current_generation``, so the question
    the repair asks FIRST is the real one — a ``readlink`` and a ``stat`` — and only
    the process probe is injected. ``running`` and ``current`` are separate
    generations, which is the situation being repaired (pass the same name for both
    to model a daemon that is already where it should be). ``pointer=False`` models
    a machine with no layout at all, which is a different answer from a daemon that
    is current.
    """
    root = tmp_path / "lop"
    for name in {running, current}:
        (root / "generations" / name).mkdir(parents=True, exist_ok=True)
    if pointer:
        (root / "current").symlink_to(root / "generations" / current)
    monkeypatch.setattr(update_mod, "_STABLE_ROOT", str(root))
    return root / "generations" / running


def _rig(
    targets: dict[str, Target],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    running: str,
    current: str,
    moved_pid: int | None = None,
    pointer: bool = True,
    probe_none: bool = False,
    probe_raises: bool = False,
    kickstart_fails: bool = False,
    daemon_pids: bool = True,
) -> dict[str, list[tuple[str, ...]]]:
    """The shipped refresh step with the runner and the build probe injected.

    Every plist is written CURRENT (``render()`` is what this build would write), so
    any reload seen here is the second question's answer rather than a rewrite's.
    ``moved_pid`` is the one daemon whose probe answers "a generation other than
    ``current``"; every other pid answers ``None``, which is what the shipped probe
    answers for a daemon that is current AND for one it could not read.
    """
    _layout(tmp_path, monkeypatch, running=running, current=current, pointer=pointer)
    for _name, (_module, path, render, _label) in targets.items():
        path.write_bytes(plistlib.dumps(render()))

    calls: dict[str, list[tuple[str, ...]]] = {name: [] for name in targets}
    for name, (module, _path, _render, _label) in targets.items():
        _patch_launcher(
            monkeypatch,
            module,
            calls[name],
            pid=_PIDS[name] if daemon_pids else None,
            kickstart_fails=kickstart_fails,
        )

    def probe(pid: int):
        if probe_raises:
            raise RuntimeError("ps exploded")
        if probe_none or pid != moved_pid:
            return None
        return tmp_path / "lop" / "generations" / running

    monkeypatch.setattr(update_mod, "stale_generation_of_process", probe)
    # The child's own refusal guard: only an installed distribution may repair a
    # daemon, and a worktree venv must be refused here (that incident is pinned
    # in tests/unit/test_update.py).
    monkeypatch.setattr(update_mod, "install_kind", lambda: InstallKind.UV_TOOL)
    return calls


def test_a_current_plist_on_a_moved_build_is_restarted_exactly_once(
    targets: dict[str, Target],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """THE DEFECT, AND THE MUTATION THIS TEST EXISTS TO CATCH.

    Delete the ``if outcome.kind == "current"`` branch from any of the four
    installers and this fails: the plist is already current, the build is one
    generation behind, and the daemon is never restarted — which is exactly the
    state the operator's tunnel connector was found in, serving a pruned tree three
    days after a released fix for it.

    It also pins what must NOT happen: ONE `kickstart`, for the daemon that moved
    only; a read-only `print` for the three that did not; and not a byte written.
    """
    running = "20260921T125352Z-0.61.12"
    calls = _rig(
        targets,
        tmp_path,
        monkeypatch,
        running=running,
        current="20260924T103058Z-509c7450dbf6",
        moved_pid=_PIDS["tunnel"],
    )

    assert update_mod.daemons_refresh_command() == 0

    domain = launchd.job_domain()
    label = targets["tunnel"][3]
    assert calls["tunnel"] == [
        ("print", f"{domain}/{label}"),
        ("kickstart", "-k", f"{domain}/{label}"),
    ], calls["tunnel"]
    for name in ("mobile", "browser bridge", "wakes supervisor"):
        assert calls[name] == [("print", f"{domain}/{targets[name][3]}")], (name, calls[name])
    for name, (_module, path, render, _label) in targets.items():
        assert plistlib.loads(path.read_bytes()) == render(), name
    captured = capsys.readouterr()
    assert (
        "tunnel daemon: restarted onto the install `current` points at "
        f"(it was still running {running})"
    ) in captured.out
    assert captured.err == ""


def test_a_current_plist_on_a_current_build_is_only_read_not_bounced(
    targets: dict[str, Target],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """The control case, and the one this repair could most easily break.

    The reading IS made — one ``launchctl print`` per daemon, which is the only cost
    this adds to an upgrade with nothing to do — and nothing else: no write, no
    ``kickstart``, no summary line. The probe answering ``None`` here is what the
    shipped probe answers for a daemon that IS current (pinned in
    tests/unit/test_daemon_build_probe.py).
    """
    calls = _rig(
        targets,
        tmp_path,
        monkeypatch,
        running="20260924T103058Z-509c7450dbf6",
        current="20260924T103058Z-509c7450dbf6",
        probe_none=True,
    )

    assert update_mod.daemons_refresh_command() == 0

    domain = launchd.job_domain()
    for name, (_module, _path, _render, label) in targets.items():
        assert calls[name] == [("print", f"{domain}/{label}")], (name, calls[name])
    captured = capsys.readouterr()
    assert "restarted onto the install" not in captured.out
    assert captured.err == ""


def test_a_machine_without_the_generation_layout_is_not_probed_at_all(
    targets: dict[str, Target],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No layout, no question — and launchd is not consulted for one.

    A pip/pipx install or a host before its first migration cannot have this
    failure: its unit names an interpreter path, and a rewrite is what repairs it.
    So the layout gate is asked FIRST, before the pid. The probe here insists the
    build moved, which proves the GATE is what stops the reload and not the probe's
    own caution.
    """
    calls = _rig(
        targets,
        tmp_path,
        monkeypatch,
        running="20260921T125352Z-0.61.12",
        current="20260924T103058Z-509c7450dbf6",
        moved_pid=_PIDS["tunnel"],
        pointer=False,
    )

    assert update_mod.daemons_refresh_command() == 0

    for name in targets:
        assert calls[name] == [], (name, calls[name])


def test_a_stale_plist_is_repaired_the_same_way_when_the_build_also_moved(
    targets: dict[str, Target], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Today's repair, unchanged, and never doubled up with the new one.

    A stale plist is rewritten AND reloaded through ``bootout`` + ``bootstrap``
    (never ``kickstart``, which would restart launchd's in-memory definition and keep
    the old argv). The second question is not asked on top of that: the rewrite's
    reload already brings the new build, so asking again could interrupt the daemon
    twice for one repair.
    """
    module, path, _render, label = targets["tunnel"]
    _layout(
        tmp_path,
        monkeypatch,
        running="20260921T125352Z-0.61.12",
        current="20260924T103058Z-509c7450dbf6",
    )
    _write_legacy(path, label, _module_for("tunnel"))
    calls: list[tuple[str, ...]] = []
    _patch_launcher(monkeypatch, module, calls, pid=_PIDS["tunnel"])
    monkeypatch.setattr(
        update_mod,
        "stale_generation_of_process",
        lambda pid: tmp_path / "lop" / "generations" / "20260921T125352Z-0.61.12",
    )

    outcome = module.refresh_plist_if_stale()

    assert outcome.kind == "repaired", outcome
    assert [call[0] for call in calls] == ["bootout", "print", "bootstrap", "print"], calls


def test_a_daemon_that_is_not_running_is_never_kicked(
    targets: dict[str, Target],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stopped daemon has no build to compare, so the repair declines.

    This is the deliberate non-goal stated in `launchd.restart_if_build_moved`: the
    repair for a stopped-but-loaded job is the installers' own `kickstart` path,
    which resumes a job whose PLIST is correct, and this question must not start a
    daemon nobody asked it to start.
    """
    calls = _rig(
        targets,
        tmp_path,
        monkeypatch,
        running="20260921T125352Z-0.61.12",
        current="20260924T103058Z-509c7450dbf6",
        moved_pid=_PIDS["tunnel"],
        daemon_pids=False,
    )

    assert update_mod.daemons_refresh_command() == 0

    for name in targets:
        assert calls[name] == [("print", f"{launchd.job_domain()}/{targets[name][3]}")], (
            name,
            calls[name],
        )


def test_a_refused_restart_is_reported_with_the_daemons_own_recovery_command(
    targets: dict[str, Target],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """A restart launchd refuses is reported, once, in the two words that fix it.

    Nothing was rewritten here, so the sentence must NOT be
    ``launchd.reload_failure``'s (which says the daemon is STOPPED because its
    ``bootout`` already landed). What is true either way is that the job was running
    an install that is not ``current`` — and the command that restores it, which is
    the same string this file pins for the other failure path
    (``_RECOVERY``), so a moved verb has to be a decision in both.
    """
    moved = "20260921T125352Z-0.61.12"
    calls = _rig(
        targets,
        tmp_path,
        monkeypatch,
        running=moved,
        current="20260924T103058Z-509c7450dbf6",
        moved_pid=_PIDS["tunnel"],
        kickstart_fails=True,
    )

    assert update_mod.daemons_refresh_command() == 0

    domain = launchd.job_domain()
    label = targets["tunnel"][3]
    # ONE attempt: a repair that retried here would be the reload loop the
    # constraint forbids, and the daemon is still serving traffic meanwhile.
    assert calls["tunnel"] == [
        ("print", f"{domain}/{label}"),
        ("kickstart", "-k", f"{domain}/{label}"),
    ], calls["tunnel"]
    captured = capsys.readouterr()
    assert "warning: tunnel daemon was not refreshed:" in captured.err
    assert moved in captured.err
    assert _RECOVERY["tunnel"] in captured.err, captured.err
    assert "STOPPED" not in captured.err, captured.err


def test_a_probe_that_raises_is_reported_and_never_escapes(
    targets: dict[str, Target],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys,
) -> None:
    """The installers' own contract, held for the NEW question: a repair never
    fails an upgrade that has already succeeded.

    The shipped probe catches its own documented failures and answers ``None`` (see
    tests/unit/test_daemon_build_probe.py); this drives the case it does not
    document — an unexpected raise — through the repair, which reports it as a failed
    refresh and carries on to the next daemon.
    """
    calls = _rig(
        targets,
        tmp_path,
        monkeypatch,
        running="20260921T125352Z-0.61.12",
        current="20260924T103058Z-509c7450dbf6",
        probe_raises=True,
    )

    assert update_mod.daemons_refresh_command() == 0

    for name in targets:
        assert [call[0] for call in calls[name]] == ["print"], (name, calls[name])
    captured = capsys.readouterr()
    assert "warning: tunnel daemon was not refreshed: ps exploded" in captured.err
