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
from local_operator.browser_bridge import install as browser_install
from local_operator.mobile import install as mobile_install
from local_operator.paths import CONFIG_DIR_ENV
from local_operator.tunnels import config as tunnel_config
from local_operator.tunnels import install as tunnel_install
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
    monkeypatch: pytest.MonkeyPatch, module, calls: list[tuple[str, ...]], *, fail: bool = False
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
    """
    _freeze_clock(monkeypatch)

    class _Completed:
        def __init__(self, args: list[str], returncode: int) -> None:
            self.args = args
            self.returncode = returncode
            self.stdout = ""
            self.stderr = "" if returncode == 0 else "Bootstrap failed: 5: Input/output error"

    loaded = [False]

    def fake(*args: str):
        calls.append(args)
        verb = args[0]
        if verb == "print":
            # `launchctl print <domain>/<label>`: non-zero exactly when launchd
            # does not know the label.
            return _Completed(list(args), 0 if loaded[0] else 1)
        if verb == "bootout":
            loaded[0] = False
            return _Completed(list(args), 0)
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
#: installer's strings has to be a decision in this file too.
_RECOVERY = {
    "mobile": "lop mobile install",
    "browser bridge": "lop browser install",
    "tunnel": "lop tunnel install",
    "wakes supervisor": "lop wake install",
}


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
    """No rewrite, no restart: a healthy daemon is not bounced on every upgrade."""
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
