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


async def test_a_host_with_no_supervisor_is_a_noop_with_a_status_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No launchctl, no systemctl, no schtasks: nothing is touched, honestly.

    This used to be the LINUX cell ("unsupported"); Linux now installs
    (``test_connector_platforms.py``). What stays a no-op is a host with no user
    service supervisor at all, and its answer names the foreground command
    rather than the old catch-all ``unsupported``.
    """

    monkeypatch.setattr(supervisors, "supervisor", lambda: None)
    _commit(tmp_path)
    outcome = install.sync_connector_service(tmp_path)
    assert outcome.code == install.CODE_NO_SUPERVISOR
    assert "python -m local_operator.team_bridge.connector --config-dir" in outcome.detail
    assert install.connector_state(tmp_path) == install.CODE_NO_SUPERVISOR
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
    # 0.4 s of an operation that sleeps at least 0.8 s per launchctl call, so
    # the window sits well inside it. The threshold is deliberately LOOSE (2 of
    # ~8 possible ticks): the defect is a COMPLETELY blocked loop, i.e. zero
    # ticks, and a tight bound only adds flakiness on a loaded machine.
    await asyncio.sleep(0.4)
    in_flight = not write.done()
    landed = len(ticks)
    await asyncio.gather(write, ticking)

    # 0.3 s at a 0.05 s interval is ~6 ticks; a blocked loop yields none — and
    # on the inline code the whole write finished first, so ``in_flight`` was
    # already false by the time this line ran. Ordered so the message names the
    # defect rather than the setup.
    assert landed >= 2, (
        f"the event loop was blocked by launchctl: only {landed} ticks landed in "
        "0.4 s while the write was in flight"
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


async def test_a_turn_off_that_races_the_catch_up_wins(service, monkeypatch) -> None:
    """R2-1: the wanted state must be decided INSIDE the lock.

    The reviewer's interleaving: the catch-up reads "wanted", the user turns the
    feature off, the off event takes the lock and removes, then the catch-up
    takes the lock holding a STALE decision and installs anyway — leaving an
    opted-out machine with a loaded, KeepAlive connector until the next serve
    start. The hook below reproduces the ordering deterministically instead of
    hoping for it: ``_wanted`` answers True once (the stale read), then False
    (the state the lock-holder must see).
    """

    fake, home, root = service
    _commit(root)
    answers = iter([True, False])

    def stale_then_fresh(_config_dir):
        try:
            return next(answers)
        except StopIteration:
            return False

    monkeypatch.setattr(install, "_wanted", stale_then_fresh)
    outcome = install.sync_connector_service(root)

    assert outcome.code == install.CODE_ABSENT, outcome
    assert not install.plist_path().exists(), "an opted-out machine was given a connector"
    assert not fake.loaded
    assert fake.calls == [] or all(call[0] == "print" for call in fake.calls), fake.calls


def test_an_unenrolled_machine_gets_no_lock_artifact(tmp_path, monkeypatch) -> None:
    """R2-5: the cheap ``absent`` answer must not create ``<config>/team_bridge``.

    A never-enrolled machine reaches this twice per session today (app start and
    a status boundary), and the lock's ``mkdir`` + file left a directory behind
    on a store that has none — contradicting the docstring's "no store created
    at every app start".
    """

    home = tmp_path / "home"
    root = home / ".local-operator"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setattr(supervisors, "supervisor", lambda: supervisors.LAUNCHCTL)
    monkeypatch.setattr(launchd, "shared_label_refusal", lambda config_dir: None)
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(install, "_launchctl", lambda *args: calls.append(args))

    assert install.sync_connector_service(root).code == install.CODE_ABSENT
    assert not (root / "team_bridge").exists(), "a never-enrolled store gained a lock file"
    assert calls == []


async def test_a_string_app_start_boundary_reaches_the_catch_up(tmp_path, monkeypatch) -> None:
    """The desktop route sends ``boundary=app_start`` as a STRING.

    ``StatusDiscipline.check`` used to compare with ``is`` against the enum, so
    the route's app-start read never reached the catch-up — only the lifespan
    task did, and it hid the dead trigger (QA round 2, out-of-scope note). The
    parse is lenient and preserves the boundary, so both spellings work.
    """

    seen: list[str] = []
    monkeypatch.setattr(install, "sync_connector_service", lambda root: seen.append(str(root)))

    discipline = _discipline(Hub(endpoint=_default_on_endpoint("active", "settings")), tmp_path)
    await discipline.check("app_start")
    assert len(seen) == 1, "a string app-start boundary did not reach the catch-up"

    discipline.invalidate()
    await discipline.check("surface_open")
    assert len(seen) == 1, "a surface-open read must not install anything"


async def test_a_turn_off_during_the_catch_ups_install_ends_removed(service, monkeypatch) -> None:
    """R3-1: the "absent" short cut must not race an install that holds the lock.

    The reviewer's ordering, made deterministic: the catch-up is inside
    ``_install`` (holding the lock, past its decision) when the user turns the
    feature off, so the off event's sync runs while the plist is still being
    written. Answering ``absent`` from a pre-lock check — which the R2-5 short
    cut did for any not-wanted machine — lets the catch-up finish and leaves an
    opted-out machine running a loaded KeepAlive connector with nothing to heal
    it. The short cut is now limited to machines with NO store at all, so this
    off event waits for the lock and then removes what the catch-up installed.
    """

    import threading

    fake, home, root = service
    _commit(root)
    inside_install = threading.Event()
    release = threading.Event()
    real_load = launchd.load

    def hold_the_install(path):
        inside_install.set()
        # The off event runs while we are here, i.e. mid-install under the lock.
        release.wait(10)
        return real_load(path)

    monkeypatch.setattr(launchd, "load", hold_the_install)
    outcomes: dict[str, str] = {}

    def catch_up() -> None:
        outcomes["catch_up"] = install.sync_connector_service(root).code

    def turn_off() -> None:
        _set_local(root, enabled=False)
        outcomes["turn_off"] = install.sync_connector_service(root).code

    first = threading.Thread(target=catch_up)
    first.start()
    assert inside_install.wait(10), "the catch-up never reached the install"
    second = threading.Thread(target=turn_off)
    second.start()
    # Give the off event every chance to take the short cut it must no longer
    # take; then let the install finish.
    second.join(0.4)
    release.set()
    first.join(15)
    second.join(15)

    assert not install.plist_path().exists(), outcomes
    assert not fake.loaded, outcomes
    assert outcomes["turn_off"] in (install.CODE_REMOVED, install.CODE_ABSENT), outcomes


def test_a_same_endpoint_double_enable_reports_already_owned(tmp_path, monkeypatch) -> None:
    """R3-2: the convergence put an await between the read-only check and the commit.

    #3027's ``_local_commit_outcome`` asks ``active_owner_readonly()`` whether
    this machine already holds THIS endpoint before committing. That check and
    the commit were atomic only because the whole function was synchronous; it
    is now ``async`` (the commit runs in a worker thread, R1-1), so a double
    submit can pass the check twice. One commits; the loser hits the store's
    in-transaction refusal. Reporting that as ``other_owner`` would be wrong —
    this machine DOES hold this endpoint — and it costs the caller the
    ``team_bridge.enabled`` write (and arms a pointless default-on retry), so
    the outcome must be ``already_owned`` with ``committed: true``.
    """

    import asyncio
    import concurrent.futures

    monkeypatch.setattr(tb_client, "org_oauth_destination_allowed", lambda base: True)
    # The store exists before either enable in the real sequence — the status
    # read and the pre-enable head read both open it — and two connections
    # bootstrapping a brand-new database race on ``schema_meta``, which is a
    # separate, pre-existing concern. What this cell pins is the owner outcome.
    Store(tmp_path).committed_owner_count()

    def enable() -> dict[str, Any]:
        return asyncio.run(
            _client(_active, tmp_path).put_settings(
                state="active",
                tenant_id="tnt_1",
                consent_version="bridge_consent_v1",
                consented_via="settings",
                endpoint=None,
                expected_account_id="acc_1",
            )
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: enable(), (0, 1)))

    outcomes = [result["local_enrollment"] for result in results]
    assert Store(tmp_path).committed_owner_count() == 1
    assert all(outcome["committed"] is True for outcome in outcomes), outcomes
    codes = sorted(outcome["code"] for outcome in outcomes)
    assert codes == ["already_owned", "committed"], outcomes
