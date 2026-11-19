"""Per-source pre-auth limits and the first-byte deadline on the mesh relay listener.

Roaming stage 4. Before this, the ONLY pre-auth bound was global
(``DEFAULT_MAX_HANDSHAKES``): one address could open eight connections, send
nothing, and hold every slot for the whole 10 s handshake budget, so a real
member's inbound dial from anywhere else was dropped at the cap. These cells
drive a REAL ``RelayServer`` over real loopback sockets:

* a flood from 127.0.0.1 holds at most the per-source cap, and a member dial from
  a SECOND source address (127.0.0.2, which Linux routes on ``lo``) still gets a
  link while the flood is held open;
* garbage is refused by the handshake as before and gives its slots back;
* a silent connection is closed at the first-byte deadline, while a peer whose
  first byte arrives late-but-inside it still completes;
* connect/close churn from one source is rate-limited without touching another;
* an established link is unaffected by a later flood from its own address.
"""

from __future__ import annotations

import selectors
import socket
import time
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import audit as audit_mod
from local_operator.network import identity, relay
from tests.unit.network import conftest as net_fixtures

# Fixtures reach this module by import. ``devices`` is pulled through
# ``request.getfixturevalue("devices")`` instead of as a parameter: a parameter would
# shadow the imported binding (pyflakes F811), and renaming the import would rename the
# fixture itself, because pytest names a fixture after the attribute it is found under.
from tests.unit.network.test_relay_e2e import (  # noqa: F401 — fixtures by import
    _pair,
    devices,
    serve_shaped_relay,
)

SECOND_SOURCE = "127.0.0.2"


def _second_source_available() -> bool:
    """Can this host bind a client socket to 127.0.0.2? Linux routes 127/8 on ``lo``;
    macOS only has 127.0.0.1 unless an alias is configured."""
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((SECOND_SOURCE, 0))
    except OSError:
        return False
    finally:
        probe.close()
    return True


needs_second_source = pytest.mark.skipif(
    not _second_source_available(),
    reason="127.0.0.2 is not bindable on this host (no 127/8 route on lo)",
)


def _connect(host: str, port: int, *, source: str = "127.0.0.1") -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((source, 0))
    sock.settimeout(5.0)
    sock.connect((host, port))
    return sock


def _closed_by_server(socks: list[socket.socket], *, within_s: float) -> set[int]:
    """Indices of ``socks`` the server closed (EOF/reset) within ``within_s``.

    A refused connection gets NO frame, so EOF is the only observable; a held one
    stays silent and never becomes readable in the window.
    """
    closed: set[int] = set()
    deadline = time.monotonic() + within_s
    with selectors.DefaultSelector() as watcher:
        for index, sock in enumerate(socks):
            watcher.register(sock, selectors.EVENT_READ, index)
        while time.monotonic() < deadline and len(closed) < len(socks):
            for key, _events in watcher.select(0.05):
                index = key.data
                if index in closed:
                    continue
                try:
                    data = socks[index].recv(1)
                except OSError:
                    data = b""
                if data == b"":
                    closed.add(index)
                    watcher.unregister(socks[index])
    return closed


def _refusals(server: relay.RelayServer, cause: str) -> list[dict[str, Any]]:
    server.audit.flush()
    return [
        row
        for row in server.audit.tail(limit=500)
        if row.get("event") == "handshake_refused" and row.get("cause") == cause
    ]


def _slots_idle(server: relay.RelayServer) -> bool:
    return (
        server._handshake_slots._value == server.settings.max_handshakes  # noqa: SLF001
        and server._preauth.inflight("127.0.0.1") == 0  # noqa: SLF001
    )


@pytest.fixture()
def lone_relay(root: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Any:
    """One serve-shaped relay on loopback with the settings a cell asks for."""
    overrides: dict[str, Any] = dict(getattr(request, "param", {}) or {})
    root_a = root / "solo"
    server = serve_shaped_relay(
        root_a,
        monkeypatch,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1", **overrides),
        identity=identity.mint(root_a, name="solo-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    host, port = server.bind()
    server.bind_control()
    server.start()
    try:
        yield server, host, port
    finally:
        server.stop()


# ---------------------------------------------------------------------------
# Source normalisation and the limiter's own bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("addr", "key"),
    [
        (("127.0.0.1", 5555), "127.0.0.1"),
        (("::ffff:203.0.113.9", 5555, 0, 0), "203.0.113.9"),
        (("2001:db8:1:2:aaaa::1", 5555, 0, 0), "2001:db8:1:2::/64"),
        (("2001:db8:1:2:bbbb::7", 6666, 0, 0), "2001:db8:1:2::/64"),
        (("fe80::1%lo0", 1, 0, 0), "fe80::/64"),
        ("not-an-address", "not-an-address"),
    ],
)
def test_a_source_is_its_ip_with_mapped_v4_folded_and_v6_at_its_slash_64(
    addr: Any, key: str
) -> None:
    assert relay.preauth_source_key(addr) == key


def test_the_per_source_cap_counts_in_flight_and_frees_on_release() -> None:
    limiter = relay.PreauthLimiter(per_source=2)
    assert limiter.admit("a") == ""
    assert limiter.admit("a") == ""
    assert limiter.admit("a") == "handshake_source_cap"
    assert limiter.admit("b") == "", "another source is charged to its own budget"
    limiter.release("a")
    assert limiter.admit("a") == ""


def test_the_rate_bucket_throttles_churn_and_refills_at_its_rate() -> None:
    now = [100.0]
    limiter = relay.PreauthLimiter(per_source=99, burst=5, rate_per_s=2.0, clock=lambda: now[0])
    admitted = 0
    for _ in range(50):
        if limiter.admit("churn") == "":
            admitted += 1
            limiter.release("churn")
    assert admitted == 5, "a burst of churn admits only the bucket's burst"
    assert limiter.admit("churn") == "handshake_source_rate"
    assert limiter.admit("other") == "", "another source has its own bucket"
    now[0] += 1.0  # 2 tokens back
    assert limiter.admit("churn") == ""
    limiter.release("churn")
    assert limiter.admit("churn") == ""
    limiter.release("churn")
    assert limiter.admit("churn") == "handshake_source_rate", "refused attempts cost no debt"


def test_the_source_map_is_bounded_under_a_spray_of_sources() -> None:
    limiter = relay.PreauthLimiter(per_source=3, map_max=64)
    for n in range(5000):
        source = f"198.18.{n // 256}.{n % 256}"
        assert limiter.admit(source) == ""
        limiter.release(source)
    assert limiter.tracked_sources() <= 64


# ---------------------------------------------------------------------------
# Real sockets against a real relay
# ---------------------------------------------------------------------------


@needs_second_source
def test_an_idle_flood_from_one_source_cannot_starve_a_member_dial_from_another(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE ACCEPTANCE CELL. Default settings (8 global, 3 per source): twenty
    silent connections from 127.0.0.1, then a REAL member handshake from
    127.0.0.2 while the flood is still open. Before the fix the flood held all
    eight global slots and this dial was dropped at the cap."""
    two_relays = request.getfixturevalue("devices")
    server_a, server_b, host, port = two_relays
    record, _host, _port = _pair(two_relays, monkeypatch)
    cap = server_a.settings.max_handshakes_per_source
    assert cap < server_a.settings.max_handshakes
    flood = [_connect(host, port) for _ in range(20)]
    try:
        closed = _closed_by_server(flood, within_s=1.0)
        held = len(flood) - len(closed)
        assert held <= cap, f"{held} silent connections from one source held slots (cap {cap})"
        assert len(closed) >= len(flood) - cap

        member = _connect(host, port, source=SECOND_SOURCE)
        link, reason = server_b.dial(
            record.network_id, host=f"{host}:{port}", epoch=record.epoch, connected=member
        )
        assert link is not None, f"a member dial from a second source was starved: {reason}"
        reply = link.request({"op": "net_catalog", "req": 7, "locality": "remote"})
        assert reply is not None and reply["op"] == "ack", reply
        rows = _refusals(server_a, "handshake_source_cap")
        assert rows, "the per-source refusal left no local audit row"
        assert len(rows) < 5, "the per-source refusal is throttled, not one row per drop"
        detail = rows[0]["detail"]
        assert detail["cause"] == "handshake_source_cap"
        assert detail["their_device"] == "" and detail["source"] == "127.0.0.1"
        assert rows[0]["detail"]["their_addr"].startswith("127.0.0.1:")
    finally:
        for sock in flood:
            sock.close()


def test_a_garbage_flood_is_refused_by_the_handshake_and_gives_its_slots_back(
    lone_relay: Any,
) -> None:
    server, host, port = lone_relay
    flood = [_connect(host, port) for _ in range(12)]
    try:
        for sock in flood:
            try:
                sock.sendall(b"\x16\x03\x01 not a hello at all\n")
            except OSError:
                pass  # already dropped at the cap
        closed = _closed_by_server(flood, within_s=5.0)
        # Every one is closed: the admitted ones by the handshake refusing garbage
        # (NOT by the first-byte timer — they sent bytes), the rest at the cap.
        assert len(closed) == len(flood)
        assert net_fixtures.wait_for(lambda: _slots_idle(server), timeout_s=5.0)
        assert not _refusals(server, "first_byte_timeout"), "garbage tripped the first-byte timer"
    finally:
        for sock in flood:
            sock.close()


@pytest.mark.parametrize("lone_relay", [{"first_byte_timeout_s": 1.0}], indirect=True)
def test_a_silent_connection_is_closed_at_the_first_byte_deadline(lone_relay: Any) -> None:
    server, host, port = lone_relay
    started = time.monotonic()
    sock = _connect(host, port)
    try:
        assert _closed_by_server([sock], within_s=4.0) == {0}
        elapsed = time.monotonic() - started
        # 1 s deadline; well under the 10 s handshake budget it used to hold.
        assert 0.8 <= elapsed < 2.5, f"closed after {elapsed:.2f}s"
        assert net_fixtures.wait_for(lambda: _slots_idle(server), timeout_s=2.0)
        rows = _refusals(server, "first_byte_timeout")
        assert rows and rows[0]["detail"]["stage"] == "first_byte", rows
    finally:
        sock.close()


@needs_second_source
def test_a_member_whose_first_byte_is_late_but_inside_the_deadline_still_links(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timer bounds silence, not a handshake: a first byte at ~1 s (default
    deadline 2 s) is followed by the ordinary handshake budget, untouched."""
    two_relays = request.getfixturevalue("devices")
    server_a, server_b, host, port = two_relays
    record, _host, _port = _pair(two_relays, monkeypatch)
    member = _connect(host, port, source=SECOND_SOURCE)
    time.sleep(1.0)
    link, reason = server_b.dial(
        record.network_id, host=f"{host}:{port}", epoch=record.epoch, connected=member
    )
    assert link is not None, f"a late-first-byte member was cut: {reason}"
    assert not _refusals(server_a, "first_byte_timeout")


@needs_second_source
def test_churn_from_one_source_is_rate_limited_and_another_source_is_not(
    lone_relay: Any,
) -> None:
    server, host, port = lone_relay
    burst = int(relay.PREAUTH_RATE_BURST)
    for _ in range(burst * 3):
        _connect(host, port).close()
    assert net_fixtures.wait_for(
        lambda: bool(_refusals(server, "handshake_source_rate")), timeout_s=5.0
    ), "a connect/close churn past the burst was never throttled"
    # The churned source is still throttled (an idle probe is dropped at accept,
    # not held for the first-byte window) while a fresh source is held.
    churned = _connect(host, port)
    other = _connect(host, port, source=SECOND_SOURCE)
    try:
        closed = _closed_by_server([churned, other], within_s=0.5)
        assert 0 in closed, "the churning source was not throttled"
        assert 1 not in closed, "another source was throttled by someone else's churn"
    finally:
        churned.close()
        other.close()


def test_an_established_link_is_unaffected_by_a_later_flood_from_its_own_address(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    two_relays = request.getfixturevalue("devices")
    server_a, server_b, host, port = two_relays
    record, _host, _port = _pair(two_relays, monkeypatch)
    link, reason = server_b.dial(record.network_id, host=f"{host}:{port}", epoch=record.epoch)
    assert link is not None, reason
    flood = [_connect(host, port) for _ in range(20)]
    try:
        _closed_by_server(flood, within_s=0.5)
        for req in range(3):
            reply = link.request({"op": "net_catalog", "req": 100 + req, "locality": "remote"})
            assert reply is not None and reply["op"] == "ack", reply
        assert link.alive
    finally:
        for sock in flood:
            sock.close()


# ---------------------------------------------------------------------------
# Round-1 remediation (PR #4290): ordering, validation, the shared budget, refunds
# ---------------------------------------------------------------------------


class _RecordingSlots:
    """A ``BoundedSemaphore`` that records each release into a shared order log."""

    def __init__(self, inner: Any, order: list[str]) -> None:
        self._inner = inner
        self._order = order

    def acquire(self, blocking: bool = True) -> bool:
        return self._inner.acquire(blocking=blocking)

    def release(self) -> None:
        self._order.append("global")
        self._inner.release()

    @property
    def _value(self) -> int:
        return self._inner._value  # noqa: SLF001


class _ScriptedListener:
    """Hands the accept loop one connection, then stops it — so the REAL loop runs
    on the test thread and its exceptions surface here, not in a daemon thread."""

    def __init__(self, server: relay.RelayServer, conn: Any, addr: Any) -> None:
        self._server = server
        self._pending = [(conn, addr)]

    def accept(self) -> tuple[Any, Any]:
        if self._pending:
            return self._pending.pop()
        self._server._stop.set()  # noqa: SLF001
        raise OSError("scripted listener exhausted")

    def close(self) -> None:
        pass


class _InlineThread:
    """Runs a handshake thread's target synchronously inside ``start``."""

    def __init__(self, *, target: Any, args: tuple[Any, ...], name: str, daemon: bool) -> None:
        self._target, self._args = target, args

    def start(self) -> None:
        self._target(*self._args)


class _UnstartableThread(_InlineThread):
    def start(self) -> None:
        raise RuntimeError("can't start new thread")


@pytest.fixture()
def unstarted_relay(root: Path, monkeypatch: pytest.MonkeyPatch) -> Any:
    root_a = root / "unstarted"
    server = serve_shaped_relay(
        root_a,
        monkeypatch,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
        identity=identity.mint(root_a, name="unstarted-device"),
        audit=audit_mod.AuditLog(root_a),
    )
    order: list[str] = []
    real_release = server._preauth.release  # noqa: SLF001

    def recording_release(source: str, **kwargs: Any) -> None:
        # ``**kwargs`` so the ORDER assertion, not a signature, is what fails
        # against a build that releases the other way round.
        order.append("source")
        real_release(source, **kwargs)

    monkeypatch.setattr(server._preauth, "release", recording_release)  # noqa: SLF001
    recording_slots = _RecordingSlots(server._handshake_slots, order)
    server._handshake_slots = recording_slots  # type: ignore[assignment]  # noqa: SLF001
    return server, order


def _run_accept_once(server: relay.RelayServer, addr: Any) -> tuple[socket.socket, socket.socket]:
    ours, theirs = socket.socketpair()
    server._listener = _ScriptedListener(  # type: ignore[assignment]  # noqa: SLF001
        server, ours, addr
    )
    server._accept_loop()  # noqa: SLF001
    return ours, theirs


def test_the_source_reservation_is_released_before_the_global_permit_on_a_normal_exit(
    unstarted_relay: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, order = unstarted_relay
    monkeypatch.setattr(relay.threading, "Thread", _InlineThread)
    monkeypatch.setattr(server, "_run_inbound_handshake", lambda sock, addr: sock.close())
    ours, theirs = _run_accept_once(server, ("203.0.113.5", 4000))
    theirs.close()
    assert order == ["source", "global"], order
    assert server._preauth.inflight("203.0.113.5") == 0  # noqa: SLF001


def test_the_source_reservation_is_released_before_the_global_permit_when_the_thread_cannot_start(
    unstarted_relay: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, order = unstarted_relay
    monkeypatch.setattr(relay.threading, "Thread", _UnstartableThread)
    with pytest.raises(RuntimeError, match="can't start"):
        _run_accept_once(server, ("203.0.113.5", 4000))
    assert order == ["source", "global"], order
    assert server._preauth.inflight("203.0.113.5") == 0  # noqa: SLF001
    assert server._handshake_slots._value == server.settings.max_handshakes  # noqa: SLF001


def test_a_global_cap_refusal_releases_and_refunds_the_source_without_touching_the_pool(
    unstarted_relay: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """R1-M1: global exhaustion caused by OTHER sources must not spend this
    source's rate tokens, and the refused attempt never held a global permit."""
    server, order = unstarted_relay
    for _ in range(server.settings.max_handshakes):
        assert server._handshake_slots.acquire(blocking=False)  # noqa: SLF001
    limiter = server._preauth  # noqa: SLF001
    ours, theirs = _run_accept_once(server, ("203.0.113.5", 4000))
    theirs.close()
    assert order == ["source"], "no global release for a permit that was never taken"
    assert limiter.inflight("203.0.113.5") == 0
    tokens, _last = limiter._buckets["203.0.113.5"]  # noqa: SLF001
    assert tokens == pytest.approx(limiter.burst), "the refused attempt's token came back"


def test_a_refund_returns_one_token_capped_at_the_burst() -> None:
    now = [10.0]
    limiter = relay.PreauthLimiter(per_source=9, burst=2, rate_per_s=0.0, clock=lambda: now[0])
    assert limiter.admit("a") == ""
    limiter.release("a", refund=True)
    assert limiter.admit("a") == "" and limiter.admit("a") == ""
    assert limiter.admit("a") == "handshake_source_rate", "two tokens, not three"
    limiter.release("a", refund=True)
    limiter.release("a", refund=True)
    limiter.release("never-seen", refund=True)  # no bucket: a no-op, not a crash
    assert limiter._buckets["a"][0] == pytest.approx(2.0)  # noqa: SLF001


@pytest.mark.parametrize(
    ("overrides", "per_source", "first_byte", "problems"),
    [
        ({}, 3, 2.0, 0),
        ({"first_byte_timeout_s": 0}, 3, 0.0, 0),  # explicit disable preserved
        ({"first_byte_timeout_s": 0.5}, 3, 0.5, 0),
        ({"first_byte_timeout_s": -1.0}, 3, 2.0, 1),
        ({"first_byte_timeout_s": float("nan")}, 3, 2.0, 1),
        ({"first_byte_timeout_s": float("inf")}, 3, 2.0, 1),
        ({"first_byte_timeout_s": 61.0}, 3, 2.0, 1),
        ({"first_byte_timeout_s": "soon"}, 3, 2.0, 1),
        ({"max_handshakes_per_source": 0}, 3, 2.0, 1),
        ({"max_handshakes_per_source": "lots"}, 3, 2.0, 1),
        ({"max_handshakes_per_source": True}, 3, 2.0, 1),
        # S4 / R1-M2: a legal value at or above the global cap is clamped below it.
        ({"max_handshakes_per_source": 8}, 7, 2.0, 0),
        ({"max_handshakes": 4, "max_handshakes_per_source": 10}, 3, 2.0, 0),
        ({"max_handshakes": 1, "max_handshakes_per_source": 3}, 1, 2.0, 0),
    ],
)
def test_the_preauth_settings_are_validated_and_clamped(
    overrides: dict[str, Any], per_source: int, first_byte: float, problems: int
) -> None:
    settings = relay.NetworkSettings(**overrides)
    got_cap, got_wait, got_problems = relay.resolve_preauth_limits(settings)
    assert (got_cap, got_wait, len(got_problems)) == (per_source, first_byte, problems)


def test_an_invalid_value_falls_back_with_one_warning_before_the_relay_listens(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    root_a = root / "invalid"
    with caplog.at_level("WARNING", logger=relay.logger.name):
        server = serve_shaped_relay(
            root_a,
            monkeypatch,
            settings=relay.NetworkSettings(
                port=0, listen_address="127.0.0.1", first_byte_timeout_s=float("nan")
            ),
            identity=identity.mint(root_a, name="invalid-device"),
            audit=audit_mod.AuditLog(root_a),
        )
    assert server._listener is None, "validated at construction, before bind/listen"  # noqa: SLF001
    assert server._first_byte_timeout_s == relay.DEFAULT_FIRST_BYTE_TIMEOUT_S  # noqa: SLF001
    lines = [r.getMessage() for r in caplog.records if "first_byte_timeout_s" in r.getMessage()]
    assert len(lines) == 1, lines


def test_a_hand_edited_string_reaches_validation_instead_of_crashing_the_reader(
    root: Path,
) -> None:
    """The real reader (``store.read_config`` over a hand-edited ``config.yml``):
    a value that will not coerce is carried to validation, which names it and
    falls back, instead of ``int("soon")`` raising inside ``from_config``."""
    cfg = root / "hand-edited"
    cfg.mkdir()
    (cfg / "config.yml").write_text(
        "values:\n"
        "  network:\n"
        "    first_byte_timeout_s: soon\n"
        "    max_handshakes_per_source: 2.5\n",
        encoding="utf-8",
    )
    settings = relay.NetworkSettings.from_config(cfg)
    assert settings.first_byte_timeout_s == "soon"
    assert settings.max_handshakes_per_source == 2.5, "never silently truncated to 2"
    cap, wait, problems = relay.resolve_preauth_limits(settings)
    assert (cap, wait) == (
        relay.DEFAULT_MAX_HANDSHAKES_PER_SOURCE,
        relay.DEFAULT_FIRST_BYTE_TIMEOUT_S,
    )
    assert len(problems) == 2, problems
    assert relay._coerce_or_raw(int, 4.0) == 4  # noqa: SLF001
    assert relay._coerce_or_raw(int, "3") == 3  # noqa: SLF001


class _BrokenSocket:
    def __init__(self) -> None:
        self.closed = 0

    def settimeout(self, _value: float) -> None:
        raise OSError(9, "Bad file descriptor")

    def recv(self, *_a: Any) -> bytes:  # pragma: no cover — settimeout raises first
        return b""

    def close(self) -> None:
        self.closed += 1


def test_a_socket_setup_failure_closes_the_socket_with_a_throttled_diagnostic(
    unstarted_relay: Any, caplog: pytest.LogCaptureFixture
) -> None:
    server, _order = unstarted_relay
    socks = [_BrokenSocket() for _ in range(5)]
    with caplog.at_level("WARNING", logger=relay.logger.name):
        for sock in socks:
            deadline = time.monotonic() + 5.0
            got = server._await_first_byte(sock, ("203.0.113.5", 1), deadline)  # noqa: SLF001
            assert got is False
    assert [s.closed for s in socks] == [1] * 5, "every socket is closed"
    lines = [r for r in caplog.records if "pre-auth socket setup failed" in r.getMessage()]
    assert len(lines) == 1, "one line per window, not one per connection"


@pytest.mark.parametrize(
    "lone_relay", [{"first_byte_timeout_s": 1.0, "handshake_timeout_s": 1.5}], indirect=True
)
def test_a_late_first_byte_spends_the_handshake_budget_rather_than_starting_a_new_one(
    lone_relay: Any,
) -> None:
    """S2: one byte at ~0.9 s then silence is closed at the 1.5 s handshake deadline
    taken at accept — not at 0.9 + 1.5 = 2.4 s as when the deadline started late."""
    server, host, port = lone_relay
    started = time.monotonic()
    sock = _connect(host, port)
    try:
        time.sleep(0.9)
        sock.sendall(b"\x00")
        assert _closed_by_server([sock], within_s=4.0) == {0}
        elapsed = time.monotonic() - started
        assert 1.3 <= elapsed < 2.05, f"held for {elapsed:.2f}s"
    finally:
        sock.close()


@pytest.mark.parametrize("lone_relay", [{"first_byte_timeout_s": 0.3}], indirect=True)
def test_the_first_byte_close_does_not_wait_on_the_audit_write(
    lone_relay: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, host, port = lone_relay
    real_note = server._note_preauth_refusal  # noqa: SLF001

    def slow_note(*args: Any, **kwargs: Any) -> None:
        time.sleep(2.0)  # a stalled audit disk
        real_note(*args, **kwargs)

    monkeypatch.setattr(server, "_note_preauth_refusal", slow_note)
    started = time.monotonic()
    sock = _connect(host, port)
    try:
        assert _closed_by_server([sock], within_s=4.0) == {0}
        elapsed = time.monotonic() - started
        assert elapsed < 1.2, f"the close waited on the audit write ({elapsed:.2f}s)"
        # The (slow) row is still written, and the handshake thread then ends.
        assert net_fixtures.wait_for(lambda: _slots_idle(server), timeout_s=4.0)
        assert _refusals(server, "first_byte_timeout")
    finally:
        sock.close()


def test_both_preauth_keys_are_registered_with_the_relays_bounds() -> None:
    """Q1: registered, so ``lop config edit``/``config list``/``/settings`` see them,
    with bounds that agree with the relay's own validation."""
    from local_operator import settings_io

    by_key = {s.key: s for s in settings_io.SETTINGS}
    cap = by_key["network.max_handshakes_per_source"]
    wait = by_key["network.first_byte_timeout_s"]
    assert cap.path == ("network", "max_handshakes_per_source")
    assert wait.path == ("network", "first_byte_timeout_s")
    assert (cap.minimum, cap.maximum) == (1, relay.MAX_HANDSHAKES_PER_SOURCE_LIMIT)
    assert (wait.minimum, wait.maximum) == (0.0, relay.FIRST_BYTE_TIMEOUT_MAX_S)
    assert settings_io.validate(wait, 0.0) is None, "0 disables and is accepted"
    assert settings_io.validate(wait, -1.0) is not None
    assert settings_io.validate(cap, 0) is not None
