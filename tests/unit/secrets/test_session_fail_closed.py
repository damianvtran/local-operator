"""A session's tool child never gets the unnotified local decrypt.

``access.retrieve_secret`` degrades to reading the key file when no broker can
serve it (§13), which is right for ``lop secret get`` in an operator's terminal
and wrong for a process whose stdout IS a session's transcript: there the
degrade skips the §6 redaction notice and paints the value raw. The routine
trigger is a broker crash mid-session — a respawned daemon has an empty session
table, denies the descendant, and the old ``BrokerDenied`` clause then decrypted
locally. These cells pin the fail-closed rule and the paths it must not touch.

The crash cells use REAL processes, as ``test_broker.py`` does and for the same
reason: the descendant relationship the broker authorizes on is what the kernel
reports, and a faked pid proves nothing about it. The policy cells around them
are in-process, with the client seam stubbed, because what they pin is the
decision ``access`` makes on each broker outcome rather than the broker itself.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.secrets import access, client
from local_operator.secrets.broker import SecretBroker
from local_operator.secrets.errors import (
    BrokerUnavailable,
    SecretNotFound,
    SessionBrokerLost,
)
from local_operator.secrets.keys import key_path, write_private_file
from local_operator.secrets.store import SecretStore

_SUPPORTED = sys.platform == "darwin" or sys.platform.startswith("linux")
_REPO = str(Path(__file__).resolve().parents[3])

SECRET_VALUE = b"s3cr3t-token-value"
SESSION_ID = "test-session-fail-closed"


@pytest.fixture
def keyfile_store(config_root: Path, master_key: bytes) -> SecretStore:
    """A keyfile-tier store with its key on disk, as the tier means (see test_broker.py)."""
    store = SecretStore(master_key, base=config_root)
    store.initialize()
    store.set("DEMO_TOKEN", SECRET_VALUE, description="fail-closed test")
    write_private_file(key_path(config_root), master_key)
    return store


@pytest.fixture
def open_store_spy(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Record every local decrypt ``retrieve_secret`` attempts, then let it run."""
    calls: list[object] = []
    real = access.open_store

    def spy(base=None, **kwargs):
        calls.append(base)
        return real(base, **kwargs)

    monkeypatch.setattr(access, "open_store", spy)
    return calls


# --- real broker: a crash mid-session ---------------------------------------


def _descendant_script(config_root: Path, wait_s: float, *, trace_denials: bool = False) -> str:
    """A session tool child: retrieves DEMO_TOKEN and reports any local decrypt.

    It runs as a real child of the test process (the registered session), with
    ``LOCAL_OPERATOR_SESSION_ID`` set the way ``session_id_env_injection`` sets
    it for ``bash``. The spy prints a marker rather than raising, so a
    regression shows up as the value being SERVED locally — the leak itself —
    not as an error the assertion could confuse with the refusal.
    """
    return textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_REPO!r})
        from pathlib import Path
        from local_operator.secrets import access
        access.SESSION_BROKER_WAIT_S = {wait_s!r}
        real = access.open_store
        def spy(base=None, **kwargs):
            print("LOCAL_DECRYPT", flush=True)
            return real(base, **kwargs)
        access.open_store = spy
        if {trace_denials!r}:
            # Announce each refusal so the test can re-register only AFTER
            # the child has been denied once; otherwise a fast re-registration
            # wins the race and the cell passes without exercising the wait.
            from local_operator.secrets import client
            real_retrieve = client.retrieve
            def traced(*args, **kwargs):
                try:
                    return real_retrieve(*args, **kwargs)
                except client.BrokerDenied:
                    print("DENIED", flush=True)
                    raise
            client.retrieve = traced
        try:
            value = access.retrieve_secret("DEMO_TOKEN", Path({str(config_root)!r}))
            print(f"OK {{value.decode()}}", flush=True)
        except Exception as exc:
            print(f"REFUSED {{type(exc).__name__}}: {{exc}}", flush=True)
        from local_operator.secrets import client
        for entry in client.spawned_brokers().values():
            for pid in entry.pids:
                print(f"SPAWNED {{pid}}", flush=True)
        """)


def _descendant_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if not key.startswith("CMUX_")}
    env["PYTHONPATH"] = _REPO
    env[access.SESSION_ID_ENV] = SESSION_ID
    return env


def _run_descendant(config_root: Path, wait_s: float) -> str:
    result = subprocess.run(
        [sys.executable, "-c", _descendant_script(config_root, wait_s)],
        capture_output=True,
        text=True,
        env=_descendant_env(),
        timeout=60,
    )
    return result.stdout + result.stderr


def _reap_respawned_brokers(output: str) -> None:
    """SIGTERM, by exact pid, every daemon the descendant's ``ensure_broker`` spawned.

    The pids come from the descendant's own spawn record (``SPAWNED`` lines):
    that names exactly the daemons it started. A first draft reaped through
    ``broker_status`` after teardown and left the respawned daemon running
    (caught by run_bounded's escaped-descendant sweep), so the record it is.
    """
    for line in output.splitlines():
        if not line.startswith("SPAWNED "):
            continue
        pid = int(line.split()[1])
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)


class _Session:
    """A session registered through the SHIPPING path, answering §6 notices.

    ``secrets.session.register_session`` rather than the bare client call, on
    purpose: it is the call that publishes the lineage marker the fail-closed
    decision reads, so a rig that registered any other way would test a
    session the descendant cannot see.
    """

    def __init__(self, config_root: Path) -> None:
        from local_operator.secrets.session import register_session

        self.received: list[tuple[str, bytes]] = []
        registration = register_session(
            lambda name, value: self.received.append((name, value)),
            session_id=SESSION_ID,
            base=config_root,
        )
        assert registration is not None, "the session could not register with the broker"
        self.registration = registration

    def close(self) -> None:
        self.registration.close()


@pytest.mark.allow_secret_broker
@pytest.mark.skipif(not _SUPPORTED, reason="the broker is implemented for macOS and Linux")
def test_a_broker_crash_mid_session_fails_closed_and_never_decrypts_locally(
    config_root: Path,
    master_key: bytes,
    keyfile_store: SecretStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The negative the security owner named: kill the broker, then retrieve.

    Before the kill the descendant is served WITH a notice, which proves the
    rig is a real session lineage. After it, the descendant's ``ensure_broker``
    respawns a daemon that has never seen the session, the daemon denies, and
    the pre-fix code took that denial to the local decrypt — printing the value
    with no notice. Now it must refuse, name the session, and never open the
    store locally.
    """
    # With #4389 in, a session re-registers with the descendant's respawned
    # daemon on its own and the read is then SERVED with a notice. That
    # automatic recovery is proven by the REAL-roll tests in
    # test_broker_roll.py (`test_roll_swaps_the_broker_and_the_session_is_
    # notified_again`, `test_a_descendant_during_reregistration_is_held_then_
    # notified`, `test_a_real_roll_keeps_the_lineage_marker_through_reregistration`),
    # not by `test_a_broker_that_comes_back_inside_the_window_…`, which turns
    # automatic re-registration off too and drives it by hand. It is switched
    # off here so this cell keeps pinning the refusal for a session that does
    # not come back.
    from local_operator.secrets import session as session_module

    monkeypatch.setattr(session_module, "_REREGISTER_DELAYS_S", ())
    broker = SecretBroker(config_root, key_provider=lambda: master_key, idle_shutdown_s=0)
    broker.start()
    session = _Session(config_root)
    after = ""
    try:
        before = _run_descendant(config_root, wait_s=1.5)
        assert f"OK {SECRET_VALUE.decode()}" in before, before
        assert session.received == [("DEMO_TOKEN", SECRET_VALUE)], "the live broker skipped §6"

        broker.stop(timeout=5)  # the crash: the registration dies with the daemon
        started = time.monotonic()
        after = _run_descendant(config_root, wait_s=1.5)
        elapsed = time.monotonic() - started

        assert "LOCAL_DECRYPT" not in after, f"fell back to the unnotified decrypt:\n{after}"
        assert SECRET_VALUE.decode() not in after, f"the value escaped:\n{after}"
        assert "REFUSED SessionBrokerLost" in after, after
        assert SESSION_ID in after, "the refusal must name the session it protected"
        assert "Restart the session" in after, "the refusal must name the remedy"
        # Bounded: the shrunken window plus interpreter start and one daemon spawn.
        assert elapsed < 30, f"the refusal took {elapsed:.1f}s"
    finally:
        session.close()
        broker.stop(timeout=5)
        _reap_respawned_brokers(after)


@pytest.mark.allow_secret_broker
@pytest.mark.skipif(not _SUPPORTED, reason="the broker is implemented for macOS and Linux")
def test_a_broker_that_comes_back_inside_the_window_serves_with_a_notice(
    config_root: Path,
    master_key: bytes,
    keyfile_store: SecretStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bounded wait's purpose: a respawn plus a re-registration is not an outage.

    The session registered, then its broker crashed. The descendant respawns a
    daemon and is denied; while it waits, the session re-registers with that
    daemon (the step the broker-roll work automates), and the next
    attempt is served THROUGH the broker — so the session hears about the value
    before the child can print it.
    """
    # The re-registration is driven by hand below, AFTER the first denial, so
    # the crashed registration's own automatic re-registration (#4389) is off:
    # left on, it races the child's first attempt and, being the same pid,
    # takes the notice meant for `session`.
    from local_operator.secrets import session as session_module

    monkeypatch.setattr(session_module, "_REREGISTER_DELAYS_S", ())
    broker = SecretBroker(config_root, key_provider=lambda: master_key, idle_shutdown_s=0)
    broker.start()
    crashed = _Session(config_root)
    broker.stop(timeout=5)  # crashed before this retrieval began; the session lives on
    session: _Session | None = None
    out = ""
    process = subprocess.Popen(
        [sys.executable, "-c", _descendant_script(config_root, wait_s=6.0, trace_denials=True)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=_descendant_env(),
    )
    try:
        # Wait until the child's respawned daemon has DENIED it once, then
        # re-register against that daemon — the step a session takes when its
        # broker comes back. Pre-fix, the denial went straight to the local
        # decrypt, so the next line the child prints is LOCAL_DECRYPT.
        assert process.stdout is not None
        first = process.stdout.readline()
        assert first.strip() == "DENIED", f"expected a denial first, got {first!r}"
        session = _Session(config_root)
        rest, _ = process.communicate(timeout=30)
        out = first + rest
        assert "LOCAL_DECRYPT" not in out, out
        assert f"OK {SECRET_VALUE.decode()}" in out, out
        assert session.received == [("DEMO_TOKEN", SECRET_VALUE)], "served without the §6 notice"
    finally:
        if process.poll() is None:
            process.kill()
            out, _ = process.communicate(timeout=5)
        if session is not None:
            session.close()
        crashed.close()
        _reap_respawned_brokers(out)


# --- the decision, per broker outcome ---------------------------------------


@pytest.fixture
def in_session(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stand in for a registered ancestor; the marker walk itself is pinned below."""
    from local_operator.secrets import session as session_module

    monkeypatch.setattr(
        session_module,
        "registered_session_ancestor",
        lambda base=None: (4242, SESSION_ID),
        raising=False,  # so a tree without the seam fails on behaviour, not setup
    )
    # `raising=False` so the cells below fail on BEHAVIOUR (a local decrypt)
    # against a tree without the window, not on a missing attribute.
    monkeypatch.setattr(access, "SESSION_BROKER_WAIT_S", 0.6, raising=False)


@pytest.mark.parametrize("outcome", ["denied", "unavailable", "no-broker"])
def test_inside_a_session_no_broker_outcome_reaches_the_local_decrypt(
    outcome: str,
    in_session: None,
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``BrokerDenied`` is policy and the other two are outages; none falls back.

    Each is retried inside the (shrunken) window and then refused, and the
    elapsed time stays inside that window plus one retry pause — the bound the
    security owner set is on the total stall, not per attempt.
    """
    attempts: list[str] = []
    timeouts: list[object] = []

    def fake_ensure(base=None, *, timeout=None, deadline=None):
        attempts.append("ensure")
        timeouts.append(timeout)
        return outcome != "no-broker"

    def fake_retrieve(name, base=None, *, role="agent", deadline=None):
        attempts.append("retrieve")
        if outcome == "denied":
            raise client.BrokerDenied("no lop session is registered with the broker")
        raise BrokerUnavailable("could not reach the secret broker")

    monkeypatch.setattr(client, "ensure_broker", fake_ensure)
    monkeypatch.setattr(client, "retrieve", fake_retrieve)

    started = time.monotonic()
    with pytest.raises(SessionBrokerLost) as caught:
        access.retrieve_secret("DEMO_TOKEN", config_root)
    elapsed = time.monotonic() - started

    assert open_store_spy == [], "a session descendant reached the local decrypt"
    assert SECRET_VALUE.decode() not in str(caught.value)
    assert attempts.count("ensure") >= 2, "the bounded wait never retried"
    # Each spawn poll is clipped to the window, so one slow start cannot carry
    # the total past the bound.
    assert all(isinstance(t, float) and t <= access.SESSION_BROKER_WAIT_S + 0.01 for t in timeouts)
    assert elapsed < access.SESSION_BROKER_WAIT_S + 0.5, f"stalled {elapsed:.2f}s"


def test_inside_a_session_a_miss_is_still_a_miss(
    in_session: None,
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """R11/Q5: the broker's ``SecretNotFound`` propagates at once, unretried.

    ``secrets.get(name, default)`` turns exactly that class into its default;
    swallowing it into the wait (or the refusal) would break that in every tool
    child of every session.
    """
    calls: list[str] = []

    def fake_retrieve(name, base=None, *, role="agent", deadline=None):
        calls.append(name)
        raise SecretNotFound(f"no secret named {name!r}")

    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)
    monkeypatch.setattr(client, "retrieve", fake_retrieve)

    with pytest.raises(SecretNotFound):
        access.retrieve_secret("MISSING", config_root)
    assert calls == ["MISSING"]
    assert open_store_spy == []


@pytest.mark.parametrize("outcome", ["denied", "unavailable", "no-broker"])
def test_outside_a_session_the_keyfile_fallback_still_serves(
    outcome: str,
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """§13 control: an operator's terminal has no session, so it keeps the fallback."""
    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)

    def fake_retrieve(name, base=None, *, role="agent", deadline=None):
        if outcome == "denied":
            raise client.BrokerDenied("no lop session is registered with the broker")
        raise BrokerUnavailable("could not reach the secret broker")

    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: outcome != "no-broker")
    monkeypatch.setattr(client, "retrieve", fake_retrieve)

    assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
    assert open_store_spy == [config_root]


def test_a_session_id_without_a_registration_keeps_the_fallback(
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``LOCAL_OPERATOR_SESSION_ID`` alone is attribution, not a redactor.

    Every session's tool child carries it, including children of a session that
    booted before the store existed or whose registration was refused — there is
    nothing to notify, so refusing would only break ``lop secret get`` there
    (``test_cli.py`` runs every verb this way).
    """
    monkeypatch.setenv(access.SESSION_ID_ENV, SESSION_ID)
    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: False)

    assert access.in_live_session_lineage(config_root) is None
    assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
    assert open_store_spy == [config_root]


def test_lineage_markers_are_found_by_ancestry_and_ignore_the_dead(
    config_root: Path, keyfile_store: SecretStore
) -> None:
    """The marker walk: a live ancestor matches; a recycled-pid marker never does.

    The test process plays the session (marker under its own identity), and a
    real child asks. A marker for a pid with the wrong start time — a session
    that crashed without closing, its pid since reused — matches nothing and is
    pruned on the next publish.
    """
    from local_operator.secrets.peer import process_info
    from local_operator.secrets.session import (
        _publish_lineage_marker,
        _withdraw_lineage_marker,
        lineage_dir,
    )

    probe = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {_REPO!r})
        from pathlib import Path
        from local_operator.secrets import access
        print(access.in_live_session_lineage(Path({str(config_root)!r})))
        """)

    def ask() -> str:
        return subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            env=_descendant_env(),
            timeout=60,
        ).stdout.strip()

    assert ask() == "None", "no registration yet"
    me = process_info(os.getpid())
    assert me is not None
    stale = lineage_dir(config_root)
    stale.mkdir(mode=0o700, parents=True, exist_ok=True)
    stale_marker = stale / f"{me.pid}-{me.start_time + 1}-deadbeef"
    stale_marker.write_text("ghost")
    assert ask() == "None", "a marker with the wrong start time matched"

    marker = _publish_lineage_marker(config_root, SESSION_ID)
    try:
        assert marker is not None
        assert not stale_marker.exists(), "the stale marker survived a publish"
        assert ask() == repr((os.getpid(), SESSION_ID))
        # The host itself is not its own descendant: its in-process reads keep
        # the fallback (see `registered_session_ancestor`).
        assert access.in_live_session_lineage(config_root) is None
    finally:
        _withdraw_lineage_marker(marker)
    assert ask() == "None", "a closed registration still counted"


def test_the_hardened_tier_is_unchanged_inside_a_session(
    in_session: None,
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passphrase mode already raised the broker's own error at once; it still does.

    No wait (the broker is the boundary there and has no disk key to race) and
    no re-labelling: callers that report ``BrokerUnavailable`` keep seeing it.
    """
    monkeypatch.setattr(access, "key_mode", lambda base=None: "passphrase")
    attempts: list[str] = []

    def fake_retrieve(name, base=None, *, role="agent", deadline=None):
        attempts.append(name)
        raise BrokerUnavailable("could not reach the secret broker")

    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)
    monkeypatch.setattr(client, "retrieve", fake_retrieve)

    with pytest.raises(BrokerUnavailable) as caught:
        access.retrieve_secret("DEMO_TOKEN", config_root)
    assert not isinstance(caught.value, SessionBrokerLost)
    assert attempts == ["DEMO_TOKEN"]
    assert open_store_spy == []


# --- #4389 convergence: one budget, a roll inside the same refusal ------------


def test_session_wait_matches_the_roll_budget() -> None:
    """#4388 R1 / #4389 S5: ONE client-side wait, and it agrees with the roll hold.

    The in-session window is the whole stall a `$(lop secret get …)` may cost
    (≤ 8 s, the security owner's ceiling), and it must still cover a held read
    during an update roll (successor hold + redaction notice), or an ordinary
    update would refuse reads the successor was about to serve.
    """
    from local_operator.secrets import broker as broker_module

    assert access.SESSION_BROKER_WAIT_S == client.ROLL_WAIT_S <= 8.0
    # #4389 R3-2: strictly inside, by the explicit margin — a held read must be
    # answered BEFORE the waiting client gives up, not at the same instant.
    assert broker_module.ROLL_HOLD_MARGIN_S >= 1.0
    assert (
        broker_module.ROLL_GRACE_S
        + broker_module.NOTIFY_ACK_TIMEOUT_S
        + broker_module.ROLL_HOLD_MARGIN_S
        <= access.SESSION_BROKER_WAIT_S
    )


def _silent_listener(config_root: Path):
    """A broker that accepts and never answers — the wedged/stopped successor shape."""
    import socket as socket_module

    from local_operator.secrets.keys import ensure_secrets_dir
    from local_operator.secrets.protocol import socket_path

    ensure_secrets_dir(config_root)
    path = socket_path(config_root)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    listener = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
    listener.bind(str(path))
    listener.listen(64)
    return listener


@pytest.mark.allow_secret_broker
@pytest.mark.skipif(not _SUPPORTED, reason="the broker is implemented for macOS and Linux")
def test_inside_a_session_one_deadline_bounds_a_silent_broker(
    in_session: None,
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#4388 R1 + QA Q2: a broker that never answers costs the window, not a request timeout.

    Before, `ensure_broker`'s liveness ping ran with a fresh REQUEST_TIMEOUT_S
    outside any budget (QA measured 14 s); now every ping, spawn poll and
    request inside the session path is clipped to one deadline. The spawn is
    stubbed so no daemon is started against the silent socket.
    """
    monkeypatch.setattr(client, "REQUEST_TIMEOUT_S", 3.0)
    monkeypatch.setattr(client, "_spawn_broker", lambda base=None: -1)
    monkeypatch.setattr(client, "_record_spawn", lambda *args, **kwargs: None)
    listener = _silent_listener(config_root)
    try:
        started = time.monotonic()
        with pytest.raises(SessionBrokerLost):
            access.retrieve_secret("DEMO_TOKEN", config_root)
        elapsed = time.monotonic() - started
    finally:
        listener.close()
    assert open_store_spy == []
    assert elapsed < access.SESSION_BROKER_WAIT_S + 0.8, f"stalled {elapsed:.2f}s"


def test_a_roll_budget_running_out_inside_a_session_is_the_same_refusal(
    in_session: None,
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S6: a roll that outlasts the budget inside a session ends in THE refusal.

    ``BrokerRolling`` used to escape as its own class from the session path;
    it is folded into ``SessionBrokerLost`` so a crash, a roll and a late
    re-registration all end the same, named, never-decrypted way.
    """
    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)

    def rolling(name, base=None, *, role="agent", deadline=None):
        raise client.BrokerRolling("the secret broker is being replaced by an update")

    monkeypatch.setattr(client, "retrieve", rolling)
    with pytest.raises(SessionBrokerLost) as caught:
        access.retrieve_secret("DEMO_TOKEN", config_root)
    assert "being replaced by an update" in str(caught.value)
    assert open_store_spy == []


@pytest.mark.parametrize("comes_back", [True, False])
def test_a_denial_during_a_live_roll_never_falls_back_inside_a_session(
    comes_back: bool,
    in_session: None,
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#4388 S2: with re-registration in, retrying a denial is how a roll is survived.

    The successor denies until the session re-registers; inside a session that
    denial is retried within the one window and then either served through the
    broker (it came back) or refused — never the keyfile fallback.
    """
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    _write_sentinel(rolling_path(config_root), 1)
    denials: list[float | None] = []
    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)

    def successor(name, base=None, *, role="agent", deadline=None):
        denials.append(deadline)
        if comes_back and len(denials) > 2:
            return SECRET_VALUE
        raise client.BrokerDenied("no lop session is registered with the broker")

    monkeypatch.setattr(client, "retrieve", successor)
    if comes_back:
        assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
    else:
        with pytest.raises(SessionBrokerLost):
            access.retrieve_secret("DEMO_TOKEN", config_root)
    assert open_store_spy == [], "a denial during a roll reached the local decrypt"
    # Every attempt carried the SAME session-wide deadline (#4388 R1).
    assert len(set(denials)) == 1 and isinstance(denials[0], float)


def test_outside_a_session_a_roll_bounds_the_liveness_ping(
    keyfile_store: SecretStore,
    config_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """QA Q2: during a roll, the out-of-session ping shares the client's roll budget."""
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    _write_sentinel(rolling_path(config_root), 1)
    seen: dict[str, object] = {}

    def fake_ensure(base=None, **kwargs):
        seen["ensure"] = kwargs.get("deadline")
        return True

    def fake_retrieve(name, base=None, *, role="agent", **kwargs):
        seen["retrieve"] = kwargs.get("deadline")
        return SECRET_VALUE

    monkeypatch.setattr(client, "ensure_broker", fake_ensure)
    monkeypatch.setattr(client, "retrieve", fake_retrieve)
    before = time.monotonic()
    assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
    deadline = seen.get("ensure")
    assert isinstance(deadline, float), "the liveness ping ran outside the roll budget"
    assert deadline <= before + client.ROLL_WAIT_S + 0.5
    assert seen.get("retrieve") == deadline


@pytest.mark.allow_secret_broker
@pytest.mark.skipif(not _SUPPORTED, reason="the broker is implemented for macOS and Linux")
def test_a_sigkilled_broker_daemon_fails_closed(
    config_root: Path,
    master_key: bytes,
    keyfile_store: SecretStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#4388 R3: the named negative on a REAL daemon killed with SIGKILL.

    Unlike a graceful ``stop()``, SIGKILL leaves the socket file behind, so the
    descendant goes through the stale-socket respawn path. Re-registration is
    disabled here so the cell pins the refusal itself; the re-registering
    outcome (served WITH a notice) is the broker-roll tests' subject.
    """
    from local_operator.secrets import session as session_module
    from local_operator.secrets.protocol import socket_path

    monkeypatch.setattr(session_module, "_REREGISTER_DELAYS_S", ())
    daemon = client._spawn_broker(config_root)
    after = ""
    session: _Session | None = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            status = client.broker_status(config_root)
            if status and status.get("pid") == daemon:
                break
            time.sleep(0.05)
        else:
            raise AssertionError("the real broker daemon never came up")
        session = _Session(config_root)
        before = _run_descendant(config_root, wait_s=1.5)
        assert f"OK {SECRET_VALUE.decode()}" in before, before

        os.kill(daemon, signal.SIGKILL)
        _await_gone(daemon)  # detached: not our child, so poll rather than wait
        assert socket_path(config_root).exists(), "SIGKILL should leave the socket behind"
        after = _run_descendant(config_root, wait_s=1.5)

        assert "LOCAL_DECRYPT" not in after, f"fell back to the unnotified decrypt:\n{after}"
        assert SECRET_VALUE.decode() not in after, f"the value escaped:\n{after}"
        assert "REFUSED SessionBrokerLost" in after, after
    finally:
        if session is not None:
            session.close()
        try:
            os.kill(daemon, signal.SIGKILL)
        except ProcessLookupError:
            pass
        _reap_respawned_brokers(after)


def _await_gone(pid: int, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)


def test_a_marker_publish_failure_is_a_warning(
    config_root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """#4388 R4/S3: losing the fail-closed property is visible, and boot goes on."""
    import logging

    from local_operator.secrets import session as session_module

    def broken(base=None):
        raise PermissionError("read-only secrets dir")

    monkeypatch.setattr(session_module, "lineage_dir", broken)
    with caplog.at_level(logging.WARNING, logger=session_module.__name__):
        assert session_module._publish_lineage_marker(config_root, SESSION_ID) is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings and "lineage marker" in warnings[0].getMessage()


def test_an_unidentifiable_live_session_keeps_its_marker(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#4388 S3: ``process_info`` → ``None`` is "unknown", not "gone".

    A transient lookup failure while ANOTHER session prunes used to delete this
    live session's marker, silently reopening its children's fallback. Only a
    pid the kernel says does not exist is pruned.
    """
    from local_operator.secrets import peer
    from local_operator.secrets.session import _prune_stale_markers, lineage_dir

    directory = lineage_dir(config_root)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    alive = directory / f"{os.getpid()}-12345-cafef00d"
    alive.write_text("live")
    gone_process = subprocess.Popen([sys.executable, "-c", "pass"])
    gone_process.wait()
    gone = directory / f"{gone_process.pid}-12345-deadbeef"
    gone.write_text("dead")
    monkeypatch.setattr(peer, "process_info", lambda pid: None)
    _prune_stale_markers(directory)
    assert alive.exists(), "a live session's marker was pruned on an unknown lookup"
    assert not gone.exists(), "a definitively dead session's marker survived"


# --- #4389 round 3 ------------------------------------------------------------


def test_marker_prune_asks_the_windows_safe_liveness_probe(
    config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R3-1: pruning decides "gone" through ``procstate.pid_liveness``, never ``os.kill``.

    ``os.kill(pid, 0)`` TERMINATES the process on Windows (the xplat battery's
    ``static.posix_attributes`` rule). The probe's three answers are pinned:
    False prunes, None (could not prove) keeps, True keeps — and ``os.kill``
    is never reached. The pid used is this test's own, which ``os.kill`` would
    call alive, so a regression to signal 0 keeps the "gone" marker and fails.
    """
    from local_operator import procstate
    from local_operator.secrets import peer
    from local_operator.secrets.session import _prune_stale_markers, lineage_dir

    directory = lineage_dir(config_root)
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    marker = directory / f"{os.getpid()}-12345-feedface"

    def no_kill(*_args: object) -> None:
        raise AssertionError("marker pruning reached os.kill")

    monkeypatch.setattr(peer, "process_info", lambda pid: None)
    monkeypatch.setattr(os, "kill", no_kill)
    for answer, survives in ((None, True), (True, True), (False, False)):
        marker.write_text("m")
        monkeypatch.setattr(procstate, "pid_liveness", lambda pid, a=answer: a)
        _prune_stale_markers(directory)
        assert marker.exists() is survives, f"pid_liveness={answer!r}"


def _tripwire(*_args: object, **_kwargs: object) -> None:
    raise AssertionError("the unnotified local decrypt was reached after a roll was observed")


def test_an_observed_roll_survives_a_liveness_ping_that_exhausts_the_budget(
    keyfile_store: SecretStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8 (a): the sentinel is seen, then the liveness ping spends the budget and
    the sentinel lapses meanwhile. "No broker" is the roll not finishing: refuse."""
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    _write_sentinel(rolling_path(config_root), 1)

    def exhausted(base=None, **_kwargs):
        rolling_path(config_root).unlink()  # the sentinel lapses during the ping
        return False

    monkeypatch.setattr(client, "ensure_broker", exhausted)
    monkeypatch.setattr(access, "open_store", _tripwire)
    with pytest.raises(client.BrokerRolling):
        access.retrieve_secret("DEMO_TOKEN", config_root)


def test_an_observed_roll_survives_a_sentinel_that_expires_before_the_retrieve(
    keyfile_store: SecretStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8 (b): liveness succeeds only AFTER the sentinel expired; the real client
    retrieval then meets silence. It must stay a roll refusal, never the fallback."""
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    monkeypatch.setattr(client, "ROLL_WAIT_S", 0.4)
    _write_sentinel(rolling_path(config_root), 1)

    def late_liveness(base=None, **_kwargs):
        rolling_path(config_root).unlink()
        return True

    def silent(*_args, **_kwargs):
        raise BrokerUnavailable("the secret broker did not answer: timed out")

    monkeypatch.setattr(client, "ensure_broker", late_liveness)
    monkeypatch.setattr(client, "_bounded_request", silent)
    monkeypatch.setattr(access, "open_store", _tripwire)
    with pytest.raises(client.BrokerRolling):
        access.retrieve_secret("DEMO_TOKEN", config_root)


def test_a_no_roll_outage_outside_a_session_still_falls_back(
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S8 control: with NO roll observed, the same silence keeps the §13 fallback."""
    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)

    def silent(*_args, **_kwargs):
        raise BrokerUnavailable("the secret broker did not answer: timed out")

    monkeypatch.setattr(client, "_bounded_request", silent)
    assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
    assert open_store_spy == [config_root]


# --- #4389 S8 (security r4): a DENIAL after an observed roll is refused too --------
#
# The security owner's invariant: once a read has observed a roll — at entry or
# anywhere during liveness/retrieval — both silence and denial end in a served
# value or `BrokerRolling`, never `open_store(...).get(...)`. The successor's hold
# only protects lineages it expects, so a late, unmarked lineage is denied after
# the hold and a denial proves nothing about the redaction notice. Reads that saw
# no roll keep §8 (denial → keyfile) and §13 (outage → local), pinned by the
# controls below. Every case goes through the PUBLIC entry with a store tripwire.


def _deny(*_args: object, **_kwargs: object) -> dict[str, Any]:
    raise client.BrokerDenied("no lop session is registered with the broker")


def test_a_denial_after_the_observed_roll_expires_never_decrypts_locally(
    keyfile_store: SecretStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8 (a): seen at entry, lapsed before the retrieve, then DENIED: refuse."""
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    monkeypatch.setattr(client, "ROLL_WAIT_S", 0.4)
    _write_sentinel(rolling_path(config_root), 1)

    def late_liveness(base=None, **_kwargs):
        rolling_path(config_root).unlink()
        return True

    monkeypatch.setattr(client, "ensure_broker", late_liveness)
    monkeypatch.setattr(client, "_bounded_request", _deny)
    monkeypatch.setattr(access, "open_store", _tripwire)
    with pytest.raises(client.BrokerRolling):
        access.retrieve_secret("DEMO_TOKEN", config_root)


def test_access_refuses_a_denial_from_retrieve_when_it_saw_the_roll_at_entry(
    keyfile_store: SecretStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8 (a, access clause): whatever `retrieve` does, a denial reaching access
    after it observed the roll is refused, not handed to the keyfile fallback."""
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    _write_sentinel(rolling_path(config_root), 1)

    def deny_after_expiry(name, base=None, **_kwargs):
        rolling_path(config_root).unlink()
        _deny()

    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)
    monkeypatch.setattr(client, "retrieve", deny_after_expiry)
    monkeypatch.setattr(access, "open_store", _tripwire)
    with pytest.raises(client.BrokerRolling):
        access.retrieve_secret("DEMO_TOKEN", config_root)


def test_a_roll_first_seen_inside_retrieval_then_a_denial_never_decrypts_locally(
    keyfile_store: SecretStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8 (b): no sentinel at entry; the first attempt meets the roll, every
    later one a denial after it lapsed. The observation inside retrieval holds."""
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    monkeypatch.setattr(client, "ROLL_WAIT_S", 0.4)
    attempts: list[int] = []

    def roll_then_deny(*_args: object, **_kwargs: object) -> dict[str, Any]:
        attempts.append(1)
        if len(attempts) == 1:
            _write_sentinel(rolling_path(config_root), 1)
            raise BrokerUnavailable("the secret broker closed the connection")
        rolling_path(config_root).unlink(missing_ok=True)
        _deny()
        raise AssertionError("unreachable")

    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)
    monkeypatch.setattr(client, "_bounded_request", roll_then_deny)
    monkeypatch.setattr(access, "open_store", _tripwire)
    with pytest.raises(client.BrokerRolling):
        access.retrieve_secret("DEMO_TOKEN", config_root)
    assert len(attempts) >= 2, "the denial after the roll must have been met"


def test_a_roll_that_starts_after_entry_refuses_when_no_broker_starts(
    keyfile_store: SecretStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M4-1 (c): the sentinel appears only during the liveness ping, which then
    reports no broker. Re-sampled, that is the roll not finishing: refuse."""
    from local_operator.secrets.protocol import rolling_path
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)

    def roll_begins(base=None, **_kwargs):
        _write_sentinel(rolling_path(config_root), 1)
        return False

    monkeypatch.setattr(client, "ensure_broker", roll_begins)
    monkeypatch.setattr(access, "open_store", _tripwire)
    with pytest.raises(client.BrokerRolling):
        access.retrieve_secret("DEMO_TOKEN", config_root)


def test_a_no_roll_denial_outside_a_session_keeps_the_keyfile_fallback(
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S8 control (d): the operator's terminal is denied by construction; with
    NO roll observed the real client re-raises and §8's keyfile fallback serves."""
    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    monkeypatch.setattr(client, "ensure_broker", lambda base=None, **_: True)
    monkeypatch.setattr(client, "_bounded_request", _deny)
    assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
    assert open_store_spy == [config_root]


class _FakeClock:
    """A `time` stand-in for client/access: `sleep` advances `monotonic`.

    Wall-clock `time()` stays real: the roll sentinel's TTL is wall-clock."""

    def __init__(self, start: float) -> None:
        self.now = start
        self.time = time.time

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def test_an_observed_roll_keeps_its_remaining_wait_and_then_serves(
    keyfile_store: SecretStore, config_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S8 coverage (security r4 MINOR): the observation is FORWARDED to
    `client.retrieve`, which keeps waiting on the ONE entry deadline — so a denial
    after the sentinel lapsed is retried and a broker that then answers serves.
    Without the forwarding the client would re-raise on the first attempt; with
    a re-armed budget the recorded deadlines would differ."""
    from local_operator.secrets.protocol import encode_bytes, rolling_path
    from local_operator.secrets.roll import _write_sentinel

    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    clock = _FakeClock(100.0)
    monkeypatch.setattr(client, "time", clock)
    monkeypatch.setattr(access, "time", clock)
    _write_sentinel(rolling_path(config_root), 1)
    entry_deadline = 100.0 + client.ROLL_WAIT_S

    def slow_liveness(base=None, *, deadline=None, **_kwargs):
        assert deadline == entry_deadline
        clock.now += 5.0  # the ping spends most of the budget...
        rolling_path(config_root).unlink()  # ...and the sentinel lapses meanwhile
        return True

    deadlines: list[float | None] = []

    def deny_then_serve(op: str, base=None, deadline=None, **_kwargs) -> dict[str, Any]:
        deadlines.append(deadline)
        if len(deadlines) == 1:
            _deny()
        return {"value": encode_bytes(SECRET_VALUE)}

    monkeypatch.setattr(client, "ensure_broker", slow_liveness)
    monkeypatch.setattr(client, "_bounded_request", deny_then_serve)
    monkeypatch.setattr(access, "open_store", _tripwire)
    assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
    assert deadlines == [entry_deadline, entry_deadline]
    assert clock.now < entry_deadline


@pytest.mark.allow_secret_broker
@pytest.mark.skipif(not _SUPPORTED, reason="the broker is implemented for macOS and Linux")
def test_outside_a_session_a_stalled_broker_costs_one_budget_then_falls_back(
    keyfile_store: SecretStore,
    config_root: Path,
    open_store_spy: list[object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Q4: a stopped broker (accepts, never answers) with no roll in sight.

    The ordinary outage rule still decrypts locally, but the WHOLE call — the
    liveness ping, and the fallback's own broker leg in ``master_key_for`` —
    shares one deadline. QA measured ~29 s before: two fresh 10 s pings, then
    two more in the fallback. Spawning is stubbed so nothing real starts.
    """
    monkeypatch.delenv(access.SESSION_ID_ENV, raising=False)
    monkeypatch.setattr(client, "ROLL_WAIT_S", 1.0)
    monkeypatch.setattr(client, "REQUEST_TIMEOUT_S", 3.0)
    monkeypatch.setattr(client, "_spawn_broker", lambda base=None: -1)
    monkeypatch.setattr(client, "_record_spawn", lambda *args, **kwargs: None)
    listener = _silent_listener(config_root)
    try:
        started = time.monotonic()
        assert access.retrieve_secret("DEMO_TOKEN", config_root) == SECRET_VALUE
        elapsed = time.monotonic() - started
    finally:
        listener.close()
    assert open_store_spy == [config_root]
    assert elapsed < client.ROLL_WAIT_S + 0.8, f"stalled {elapsed:.2f}s"
