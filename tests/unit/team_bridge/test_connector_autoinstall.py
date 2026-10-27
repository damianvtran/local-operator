"""D2a: a committed enrolment has a real connector service, not just a plist render.

The launchctl fake is stateful: bootout unregisters the label, bootstrap
registers it, and print answers about the label rather than returning a blanket
success. HOME is redirected and the shared-label/source guards are opened only
for the cells about the real lifecycle; the isolation cells leave them intact.
No test starts a connector or addresses the operator's real launchd domain.
"""

from __future__ import annotations

import asyncio
import plistlib
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from local_operator import launchd, procname, supervisors
from local_operator.team_bridge import client as tb_client
from local_operator.team_bridge.client import Boundary, TeamBridgeClient
from local_operator.team_bridge.connector import install
from local_operator.team_bridge.store import OwnerReceipt, OwnerSnapshot, Store
from tests.unit.team_bridge.test_client import FakeRow, FakeStore, _ok
from tests.unit.team_bridge.test_default_on import (
    Hub,
    _default_on_endpoint,
    _discipline,
)
from tests.unit.team_bridge.test_discipline import _set_local

pytestmark = pytest.mark.asyncio

OWNER = OwnerSnapshot(
    credential_id=1,
    account_id="acc_1",
    tenant_id="tnt_1",
    endpoint_id="bep_1",
    canonical_base_url="https://hub.example/v1",
    enrollment_generation=1,
)


class FakeLaunchctl:
    """One in-memory launchd label, with the verbs ``launchd.reload_job`` uses."""

    def __init__(self) -> None:
        self.loaded = False
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, *args: str) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args[0] == "print":
            return subprocess.CompletedProcess(
                list(args), 0 if self.loaded else 113, "state = running\n\tpid = 4242\n", ""
            )
        if args[0] == "bootstrap":
            self.loaded = True
        elif args[0] == "bootout":
            self.loaded = False
        return subprocess.CompletedProcess(list(args), 0, "", "")


@pytest.fixture(autouse=True)
def _destination_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)


@pytest.fixture
def service(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeLaunchctl, Path, Path]:
    """A real user-shaped home/store with all machine-side effects contained."""

    home = tmp_path / "home"
    root = home / ".local-operator"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
    # The real guards are exercised in the isolation tests below. These cells
    # test the writer, so open the decisions at the same module attributes the
    # production code reads; the launchctl fake is still mandatory.
    monkeypatch.setattr(launchd, "shared_label_refusal", lambda config_dir: None)
    monkeypatch.setattr(launchd, "is_own_plist", lambda path, label: True)
    monkeypatch.setattr(procname, "supervised_image", lambda: Path("/stable/lop/bin/python3"))
    fake = FakeLaunchctl()
    # ``raising=False``: on base source the module has no seam yet, and the
    # behavioural cells below must fail on their ASSERTION (no service) rather
    # than at fixture setup.
    monkeypatch.setattr(install, "_launchctl", fake, raising=False)
    return fake, home, root


def _commit(root: Path) -> None:
    Store(root).commit_enrollment(OWNER, OwnerReceipt("bridge_consent_v1", "bep_1"))


def _client(handler: Any, root: Path) -> TeamBridgeClient:
    return TeamBridgeClient(
        root,
        base_url="https://hub.example/v1",
        store=FakeStore(rows=[FakeRow(1, data={"access": "tok", "account_id": "acc_1"})]),
        transport=httpx.MockTransport(handler),
    )


def _active(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/messaging/events"):
        return httpx.Response(403, json={"error": "no", "code": "bridge_not_available"})
    return _ok({"endpoint": {"id": "bep_1", "state": "active"}, "settings": {"tenant_id": "tnt_1"}})


async def test_enrolment_commit_installs_and_starts_the_connector(service) -> None:
    fake, home, root = service
    await _client(_active, root).put_settings(
        state="active",
        tenant_id="tnt_1",
        consent_version="bridge_consent_v1",
        consented_via="settings",
        endpoint=None,
        expected_account_id="acc_1",
    )
    assert Store(root).committed_owner_count() == 1
    path = home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist"
    assert path.exists(), "a committed owner got no connector service"
    assert fake.loaded
    plan = plistlib.loads(path.read_bytes())
    assert plan["Label"] == install.LABEL
    assert plan["Program"] == "/stable/lop/bin/python3"
    assert plan["ProgramArguments"] == [
        procname.branded_argv0(procname.LABEL_BRIDGE_CONNECTOR),
        "-m",
        "local_operator.team_bridge.connector",
        "--config-dir",
        str(root),
    ]
    assert "generations" not in plan["Program"]


async def test_app_start_catches_up_an_already_enrolled_machine(service) -> None:
    fake, home, root = service
    _commit(root)
    await _discipline(Hub(endpoint=_default_on_endpoint("active", "settings")), root).check(
        Boundary.APP_START
    )
    assert (home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist").exists()
    assert fake.loaded


async def test_a_second_reconciliation_is_a_noop(service) -> None:
    fake, home, root = service
    _commit(root)
    assert install.sync_connector_service(root).code == install.CODE_INSTALLED
    path = install.plist_path()
    before = path.read_bytes()
    fake.calls.clear()
    assert install.sync_connector_service(root).code == install.CODE_CURRENT
    assert path.read_bytes() == before
    assert [call[0] for call in fake.calls] == ["print"], fake.calls


async def test_turn_off_boots_out_and_removes_the_agent(service) -> None:
    fake, home, root = service
    _commit(root)
    install.sync_connector_service(root)
    path = install.plist_path()
    assert path.exists() and fake.loaded
    await _client(_active, root).put_settings(state="disabled", expected_account_id="acc_1")
    assert not path.exists()
    assert not fake.loaded
    assert any(call[0] == "bootout" for call in fake.calls)


async def test_explicit_local_off_also_removes_the_agent(service) -> None:
    fake, home, root = service
    _commit(root)
    install.sync_connector_service(root)
    _set_local(root, enabled=False)
    assert install.sync_connector_service(root).code == install.CODE_REMOVED
    assert not install.plist_path().exists()
    assert not fake.loaded


async def test_isolated_config_dir_never_installs_or_addresses_launchd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety arm: no opened guards, no plist, no launchctl, no install."""

    home = tmp_path / "home"
    root = tmp_path / "isolated-config"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
    _commit(root)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(install, "_launchctl", lambda *args: calls.append(args))
    assert install.sync_connector_service(root).code == install.CODE_ISOLATED
    assert not (home / "Library" / "LaunchAgents" / f"{install.LABEL}.plist").exists()
    assert calls == []


async def test_a_source_checkout_cannot_repoint_the_users_agent(service, monkeypatch) -> None:
    fake, home, root = service
    _commit(root)
    monkeypatch.setattr(procname, "supervised_image", lambda: None)
    from local_operator import update

    monkeypatch.setattr(update, "install_kind", lambda: update.InstallKind.EDITABLE)
    assert install.sync_connector_service(root).code == install.CODE_SOURCE_CHECKOUT
    assert not install.plist_path().exists()
    assert fake.calls == []


async def test_linux_is_a_noop_with_a_status_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.SYSTEMCTL)
    _commit(tmp_path)
    assert install.sync_connector_service(tmp_path).code == install.CODE_UNSUPPORTED
    assert install.connector_state(tmp_path) == install.CODE_UNSUPPORTED
    assert not install.plist_path().exists()


async def test_status_distinguishes_missing_installed_and_loaded(service) -> None:
    fake, home, root = service
    assert install.connector_state(root) == install.STATE_MISSING
    install.plist_path().parent.mkdir(parents=True)
    install.plist_path().write_bytes(plistlib.dumps(install.render_plist(root)))
    assert install.connector_state(root) == install.STATE_INSTALLED
    fake.loaded = True
    assert install.connector_state(root) == install.STATE_LOADED


async def test_the_consent_surface_commit_also_installs(service) -> None:
    from local_operator.team_bridge.connector.enrollment import (
        commit_consent_and_designation,
    )

    fake, home, root = service
    commit_consent_and_designation(
        OWNER,
        OwnerReceipt("bridge_consent_v1", "bep_1"),
        cos_session_id=None,
        config_dir=root,
    )
    assert fake.loaded
    assert install.plist_path().exists()


async def test_a_slow_launchctl_never_blocks_the_event_loop(service, monkeypatch) -> None:
    """R1-1: the OFF write must not freeze the server while launchctl works.

    The fake sleeps for real, so the proof is a timer on the SAME loop: if the
    lifecycle ran inline no tick would land while the request was in flight.
    Inline was the defect — QA measured 0.97 s of a blocked loop on one disable,
    against every route and the SSE fan-out waiting behind it.
    """

    fake, home, root = service
    _commit(root)
    install.sync_connector_service(root)
    immediate = install._launchctl

    def sleepy(*args: str):
        time.sleep(0.8)
        return immediate(*args)

    monkeypatch.setattr(install, "_launchctl", sleepy)
    ticks: list[float] = []

    async def ticker() -> None:
        for _ in range(12):
            await asyncio.sleep(0.05)
            ticks.append(time.monotonic())

    write = asyncio.create_task(
        _client(_active, root).put_settings(state="disabled", expected_account_id="acc_1")
    )
    ticking = asyncio.create_task(ticker())
    await asyncio.sleep(0.3)
    in_flight = not write.done()
    landed = len(ticks)
    await asyncio.gather(write, ticking)

    # 0.3 s at a 0.05 s interval is ~6 ticks; a blocked loop yields none — and
    # on the inline code the whole write finished first, so ``in_flight`` was
    # already false by the time this line ran. Ordered so the message names the
    # defect rather than the setup.
    assert landed >= 3, (
        f"the event loop was blocked by launchctl: only {landed} ticks landed in "
        "0.3 s while the write was in flight"
    )
    assert in_flight, "the write finished inside the window; the fake was not slow enough"
    assert not install.plist_path().exists()


async def test_concurrent_syncs_install_once(service) -> None:
    """Q-3: two triggers at one app start must not each reload the label.

    Four threads through the shipped entry point, as the serve lifespan, a
    status boundary and a settings edit can do at the same moment. Before the
    lock this produced four reload cycles, and a bootout landing after another
    thread's bootstrap could leave the label UNLOADED while both reported
    success — an outage, not a wasted call.
    """

    import concurrent.futures

    fake, home, root = service
    _commit(root)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        codes = list(pool.map(lambda _: install.sync_connector_service(root).code, range(4)))

    assert codes.count(install.CODE_INSTALLED) == 1, codes
    assert set(codes) <= {install.CODE_INSTALLED, install.CODE_CURRENT}, codes
    assert fake.loaded, "a concurrent sync left the label unloaded"
    bootstraps = [c for c in fake.calls if c[0] == "bootstrap"]
    assert len(bootstraps) == 1, fake.calls


async def test_a_forgotten_label_is_reported_as_registered_not_installed(service) -> None:
    """Q-6: nothing is written when the plist is already correct.

    The plist is current but launchd has forgotten the job (a reboot before the
    agent loaded, a hand bootout), so the same bytes are bootstrapped again. The
    log line must say ``registered``; ``installed`` claimed a write that never
    happened.
    """

    fake, home, root = service
    _commit(root)
    install.sync_connector_service(root)
    fake.loaded = False  # launchd forgot the job; the file is untouched
    before = install.plist_path().stat().st_mtime_ns
    assert install.sync_connector_service(root).code == install.CODE_REGISTERED
    assert install.plist_path().stat().st_mtime_ns == before, "the plist was rewritten"
    assert fake.loaded
