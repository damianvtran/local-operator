"""Same-OS-user check on loopback connections (``server/peer_check.py``).

What these pin, and why each matters:

* the per-connection decision (same user admitted, foreign refused, root only
  where the kernel can attribute it, error -> one retry -> refuse) with injected
  lookups, so every branch runs on every OS;
* the ``/proc/net/tcp`` parser on fixture text (v4, v6, ``::ffff:``, uid 0,
  overflow 65534, TIME_WAIT, full 4-tuple strictness);
* the skip modes and their ONE notice -- the operator's "zero friction"
  constraint lives there;
* the uvicorn coupling: real servers booted through the gated protocol, so an
  uvicorn upgrade that changes ``connection_made`` fails here, including the
  proof that a refused connection never reaches the app;
* the real native mechanism (macOS libproc / Linux procfs) on a real child
  connection, and a desktop-style ``/health`` plus a ``/v1/static`` preview
  through the real app and the real gate.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http.client
import ipaddress
import logging
import os
import socket
import struct
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest
import uvicorn

from local_operator.server import peer_check as pc

NATIVE = sys.platform == "darwin" or sys.platform.startswith("linux")
MY_UID = os.getuid() if hasattr(os, "getuid") else -1
LOGGER = "local_operator.server.peer_check"


def _ep(host: str, port: int) -> pc.Endpoint:
    return pc.normalize_endpoint(host, port)


class FakeLookup:
    """A ``PeerLookup`` whose answers are scripted, recording every call."""

    mechanism = "fake"

    def __init__(self, *answers: Any, admits_root: bool = False) -> None:
        self.answers = list(answers)
        self.admits_root = admits_root
        self.calls: list[tuple[pc.Endpoint, pc.Endpoint]] = []

    def owner_uid(self, client: pc.Endpoint, server: pc.Endpoint) -> int | None:
        self.calls.append((client, server))
        answer = self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]
        if isinstance(answer, BaseException):
            raise answer
        if callable(answer):
            answer = answer()
        assert answer is None or isinstance(answer, int)
        return answer


def _enforcing(lookup: FakeLookup) -> pc.PeerGate:
    return pc.PeerGate(pc.GateStatus("enforcing", mechanism="fake"), lookup)


def _check(gate: pc.PeerGate) -> pc.Decision:
    return asyncio.run(gate.check(_ep("127.0.0.1", 50000), _ep("127.0.0.1", 1111)))


# --------------------------------------------------------------------------
# Decision policy
# --------------------------------------------------------------------------


def test_same_user_is_admitted_on_one_lookup() -> None:
    lookup = FakeLookup(MY_UID)
    assert _check(_enforcing(lookup)).admit
    assert len(lookup.calls) == 1


def test_foreign_user_is_refused_with_403_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    decision = _check(_enforcing(FakeLookup(MY_UID + 1000)))
    assert not decision.admit
    assert decision.response.startswith(b"HTTP/1.1 403 ")
    assert b"connection: close" in decision.response
    assert f"owned by uid {MY_UID + 1000}" in caplog.text


def test_refusal_log_is_rate_limited_per_uid(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    gate = _enforcing(FakeLookup(MY_UID + 1000))
    for _ in range(5):
        assert not _check(gate).admit
    assert sum("refused a loopback connection" in r.message for r in caplog.records) == 1


def test_root_is_admitted_only_where_the_mechanism_can_see_root() -> None:
    # Linux: uid 0 is positively attributed and admitted (manager decision 1).
    assert _check(_enforcing(FakeLookup(0, admits_root=True))).admit
    # macOS: root is never visible, so a uid-0 answer must not be admitted.
    assert not _check(_enforcing(FakeLookup(0, admits_root=False))).admit


@pytest.mark.skipif(MY_UID == 0, reason="the daemon user is root here")
def test_overflow_uid_is_refused() -> None:
    assert not _check(_enforcing(FakeLookup(65534, admits_root=True))).admit


def test_not_found_is_rescanned_once_then_refused() -> None:
    lookup = FakeLookup(None)
    decision = _check(_enforcing(lookup))
    assert decision.response.startswith(b"HTTP/1.1 403 ")
    assert len(lookup.calls) == 2


def test_not_found_then_found_on_rescan_is_admitted() -> None:
    lookup = FakeLookup(None, MY_UID)
    assert _check(_enforcing(lookup)).admit


def test_error_is_retried_once_then_refused_with_503(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    lookup = FakeLookup(OSError("boom"))
    decision = _check(_enforcing(lookup))
    assert decision.response.startswith(b"HTTP/1.1 503 ")
    assert len(lookup.calls) == 2, "exactly ONE immediate retry"
    assert "lookup error after one retry: OSError: boom" in caplog.text


def test_error_then_success_on_retry_is_admitted_without_a_refusal_log(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.WARNING, logger=LOGGER)
    assert _check(_enforcing(FakeLookup(OSError("transient"), MY_UID))).admit
    assert "refused" not in caplog.text


def test_timeout_counts_as_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "LOOKUP_TIMEOUT_S", 0.05)
    lookup = FakeLookup(lambda: time.sleep(0.2) or MY_UID)
    decision = _check(_enforcing(lookup))
    assert decision.response.startswith(b"HTTP/1.1 503 ")


def test_flood_beyond_pending_cap_is_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pc, "MAX_PENDING_LOOKUPS", 0)
    assert _check(_enforcing(FakeLookup(MY_UID))).response.startswith(b"HTTP/1.1 503 ")


# --------------------------------------------------------------------------
# Address normalisation and the /proc/net/tcp parser
# --------------------------------------------------------------------------


def test_ipv4_mapped_ipv6_is_normalised_to_ipv4() -> None:
    assert _ep("::ffff:127.0.0.1", 5) == (ipaddress.IPv4Address("127.0.0.1"), 5)
    assert _ep("::1", 5) == (ipaddress.IPv6Address("::1"), 5)
    assert _ep("127.0.0.1", 5) == (ipaddress.IPv4Address("127.0.0.1"), 5)


def _proc_hex(host: str, port: int) -> str:
    """Encode like the kernel does: 32-bit words in host byte order."""
    packed = ipaddress.ip_address(host).packed
    words = struct.unpack(f"={len(packed) // 4}I", packed)
    return "".join(f"{w:08X}" for w in words) + f":{port:04X}"


def _proc_row(local: tuple[str, int], remote: tuple[str, int], uid: int, state: str = "01") -> str:
    return (
        f"   0: {_proc_hex(*local)} {_proc_hex(*remote)} {state} 00000000:00000000 "
        f"00:00000000 00000000 {uid:>5}        0 12345 1 0000000000000000 20 4 30 10 -1"
    )


_HEADER = (
    "  sl  local_address rem_address   st tx_queue rx_queue tr tm->when retrnsmt"
    "   uid  timeout inode"
)


def test_proc_parser_decodes_host_order_v4_and_v6() -> None:
    assert pc.parse_proc_net_address(_proc_hex("127.0.0.1", 1111)) == _ep("127.0.0.1", 1111)
    assert pc.parse_proc_net_address(_proc_hex("::1", 1111)) == _ep("::1", 1111)
    assert pc.parse_proc_net_address(_proc_hex("::ffff:127.0.0.1", 80)) == _ep("127.0.0.1", 80)
    # The literal the kernel prints for 127.0.0.1:1111 on little-endian hosts.
    if sys.byteorder == "little":
        assert pc.parse_proc_net_address("0100007F:0457") == _ep("127.0.0.1", 1111)


def test_proc_finder_returns_the_client_rows_uid() -> None:
    text = "\n".join(
        [
            _HEADER,
            # Our own accepted socket (local = us): must never be the match.
            _proc_row(("127.0.0.1", 1111), ("127.0.0.1", 50000), 4242),
            _proc_row(("127.0.0.1", 50000), ("127.0.0.1", 1111), 1001),
        ]
    )
    assert pc.find_uid_in_proc_net(text, _ep("127.0.0.1", 50000), _ep("127.0.0.1", 1111)) == 1001


def test_proc_finder_reports_root_and_overflow_uids() -> None:
    text = "\n".join(
        [
            _HEADER,
            _proc_row(("127.0.0.1", 50000), ("127.0.0.1", 1111), 0),
            _proc_row(("::1", 50001), ("::1", 1111), 65534),
        ]
    )
    assert pc.find_uid_in_proc_net(text, _ep("127.0.0.1", 50000), _ep("127.0.0.1", 1111)) == 0
    assert pc.find_uid_in_proc_net(text, _ep("::1", 50001), _ep("::1", 1111)) == 65534


def test_proc_finder_matches_a_v4_mapped_row_in_tcp6() -> None:
    text = "\n".join(
        [_HEADER, _proc_row(("::ffff:127.0.0.1", 50000), ("::ffff:127.0.0.1", 1111), 1001)]
    )
    assert (
        pc.find_uid_in_proc_net(text, _ep("127.0.0.1", 50000), _ep("::ffff:127.0.0.1", 1111))
        == 1001
    )


def test_proc_finder_requires_the_full_four_tuple() -> None:
    """Same ports on a different loopback address are NOT the client."""
    text = "\n".join([_HEADER, _proc_row(("127.0.0.2", 50000), ("127.0.0.1", 1111), 1001)])
    assert pc.find_uid_in_proc_net(text, _ep("127.0.0.1", 50000), _ep("127.0.0.1", 1111)) is None
    text = "\n".join([_HEADER, _proc_row(("127.0.0.1", 50000), ("127.0.0.3", 1111), 1001)])
    assert pc.find_uid_in_proc_net(text, _ep("127.0.0.1", 50000), _ep("127.0.0.1", 1111)) is None


def test_proc_finder_ignores_time_wait_rows() -> None:
    """TIME_WAIT rows report uid 0; the root rule must never admit them."""
    text = "\n".join([_HEADER, _proc_row(("127.0.0.1", 50000), ("127.0.0.1", 1111), 0, state="06")])
    assert pc.find_uid_in_proc_net(text, _ep("127.0.0.1", 50000), _ep("127.0.0.1", 1111)) is None


def test_linux_lookup_rereads_its_preopened_tables(tmp_path: Path) -> None:
    tcp, tcp6 = tmp_path / "tcp", tmp_path / "tcp6"
    tcp.write_text(_HEADER + "\n")
    tcp6.write_text("\n".join([_HEADER, _proc_row(("::1", 50000), ("::1", 1111), 1001)]) + "\n")
    lookup = pc.LinuxProcNetLookup((str(tcp), str(tcp6)))
    assert lookup.owner_uid(_ep("::1", 50000), _ep("::1", 1111)) == 1001
    # A row added after open is seen: the fd is re-read from offset 0.
    tcp.write_text(
        "\n".join([_HEADER, _proc_row(("127.0.0.1", 50002), ("127.0.0.1", 1111), 7)]) + "\n"
    )
    assert lookup.owner_uid(_ep("127.0.0.1", 50002), _ep("127.0.0.1", 1111)) == 7
    assert lookup.owner_uid(_ep("127.0.0.1", 9), _ep("127.0.0.1", 1111)) is None


# --------------------------------------------------------------------------
# Skip modes and the ONE startup notice
# --------------------------------------------------------------------------


def _no_fs(_path: str) -> bool:
    return False


def _no_text(_path: str) -> str | None:
    return None


@pytest.mark.parametrize(
    ("env", "exists", "texts", "expected"),
    [
        ({}, {"/.dockerenv"}, {}, "container: /.dockerenv"),
        ({}, {"/run/.containerenv"}, {}, "container: /run/.containerenv"),
        ({"container": "podman"}, set(), {}, "container: container=podman"),
        (
            {},
            set(),
            {"/proc/1/cgroup": "0::/kubepods/besteffort/pod1"},
            "container: /proc/1/cgroup names kubepods",
        ),
        ({"WSL_DISTRO_NAME": "Ubuntu"}, set(), {}, "wsl: WSL_DISTRO_NAME is set"),
        (
            {},
            set(),
            {"/proc/sys/kernel/osrelease": "5.15.0-microsoft-standard-WSL2"},
            "wsl: kernel osrelease names microsoft",
        ),
        ({}, set(), {"/proc/1/cgroup": "0::/init.scope"}, None),
    ],
)
def test_environment_skip_detection(
    env: dict[str, str], exists: set[str], texts: dict[str, str], expected: str | None
) -> None:
    assert (
        pc.detect_environment_skip(env, exists=exists.__contains__, read_text=texts.get) == expected
    )


def _notices(caplog: pytest.LogCaptureFixture) -> list[logging.LogRecord]:
    return [
        r for r in caplog.records if r.name == LOGGER and r.message.startswith("peer-uid check:")
    ]


def _resolve(env: dict[str, str], **kw: Any) -> pc.PeerGate:
    kw.setdefault("lookup_factory", lambda: FakeLookup(MY_UID))
    kw.setdefault("detect_skip", lambda _env: None)
    kw.setdefault("probe", lambda _lookup: None)
    return pc.resolve_gate(env, **kw)


def test_auto_enforces_natively_with_one_info_notice(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    gate = _resolve({})
    assert gate.enforcing and gate.status.as_dict() == {"mode": "enforcing", "mechanism": "fake"}
    (notice,) = _notices(caplog)
    assert notice.levelno == logging.INFO and "enforcing (fake)" in notice.message


def test_auto_skips_in_a_container_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    gate = _resolve({}, detect_skip=lambda _env: "container: /.dockerenv")
    assert not gate.enforcing
    assert gate.status.as_dict() == {"mode": "skipped", "reason": "container: /.dockerenv"}
    (notice,) = _notices(caplog)
    assert notice.levelno == logging.WARNING
    assert "skipped (container: /.dockerenv)" in notice.message
    assert "LOCAL_OPERATOR_PEER_UID_CHECK=on" in notice.message


def test_on_overrides_the_container_heuristic(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    gate = _resolve({pc.ENV_VAR: "on"}, detect_skip=lambda _env: "container: /.dockerenv")
    assert gate.enforcing
    assert len(_notices(caplog)) == 1


def test_off_disables_with_one_info_notice_and_never_probes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)

    def _boom(_lookup: object) -> str | None:
        raise AssertionError("off must not probe")

    gate = _resolve({pc.ENV_VAR: "OFF"}, probe=_boom)
    assert gate.status.mode == "disabled" and not gate.enforcing
    (notice,) = _notices(caplog)
    assert (
        notice.levelno == logging.INFO
        and "disabled by LOCAL_OPERATOR_PEER_UID_CHECK=off" in notice.message
    )


def test_unsupported_platform_skips_with_one_warning(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)

    def _factory() -> pc.PeerLookup:
        return pc.default_lookup_factory("win32")

    gate = _resolve({}, lookup_factory=_factory)
    assert gate.status.as_dict() == {"mode": "skipped", "reason": "unsupported: windows"}
    assert len(_notices(caplog)) == 1


@pytest.mark.parametrize("mode", ["auto", "on"])
def test_failed_self_probe_skips_rather_than_locking_out(
    mode: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    gate = _resolve({pc.ENV_VAR: mode}, probe=lambda _lookup: "self-probe: offsets drifted")
    assert gate.status.as_dict() == {"mode": "skipped", "reason": "self-probe: offsets drifted"}
    assert asyncio.run(gate.check(_ep("127.0.0.1", 1), _ep("127.0.0.1", 2))).admit
    assert len(_notices(caplog)) == 1


def test_unrecognised_value_falls_back_to_auto_inside_the_one_notice(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=LOGGER)
    gate = _resolve({pc.ENV_VAR: "yes"})
    assert gate.enforcing
    (notice,) = _notices(caplog)
    assert "ignored unrecognised" in notice.message


def test_self_probe_rejects_a_mechanism_that_admits_everything() -> None:
    """The negative control: a lookup answering our uid for ANY tuple fails."""
    assert "unowned tuple" in (pc.self_probe(FakeLookup(MY_UID)) or "")


def test_self_probe_rejects_a_mechanism_that_cannot_see_other_processes() -> None:
    assert "resolved to uid None" in (pc.self_probe(FakeLookup(None)) or "")


# --------------------------------------------------------------------------
# The uvicorn coupling: real servers through the gated protocol
# --------------------------------------------------------------------------


class SpyApp:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            return
        self.calls += 1
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"text/plain")],
            }
        )
        await send({"type": "http.response.body", "body": b"ok"})


def _gate_source(gate: pc.PeerGate) -> Callable[[], concurrent.futures.Future[pc.PeerGate]]:
    fut: concurrent.futures.Future[pc.PeerGate] = concurrent.futures.Future()
    fut.set_result(gate)
    return lambda: fut


@pytest.fixture
def serve() -> Iterator[Callable[..., int]]:
    """Boot uvicorn on 127.0.0.1:0 in a thread; returns the bound port."""
    servers: list[tuple[uvicorn.Server, threading.Thread]] = []

    def _start(app: Any, protocol: type[asyncio.Protocol], host: str = "127.0.0.1") -> int:
        family = socket.AF_INET6 if ":" in host else socket.AF_INET
        sock = socket.socket(family, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((host, 0))
        config = uvicorn.Config(
            app, http=protocol, lifespan="off", log_level="warning", log_config=None
        )
        server = uvicorn.Server(config)
        thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
        thread.start()
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise TimeoutError("uvicorn did not start")
            time.sleep(0.01)
        servers.append((server, thread))
        return sock.getsockname()[1]

    yield _start
    for server, thread in servers:
        server.should_exit = True
        thread.join(timeout=10)


def _get(port: int, path: str = "/", host: str = "127.0.0.1", **headers: str) -> tuple[int, bytes]:
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("GET", path, headers=headers)
        resp = conn.getresponse()
        return resp.status, resp.read()
    finally:
        conn.close()


def _base_protocol() -> type[asyncio.Protocol]:
    from uvicorn.protocols.http.auto import AutoHTTPProtocol

    return AutoHTTPProtocol


def test_admitted_connection_reaches_the_app(serve: Callable[..., int]) -> None:
    app, lookup = SpyApp(), FakeLookup(MY_UID)
    port = serve(app, pc.make_gated_protocol(_base_protocol(), _gate_source(_enforcing(lookup))))
    assert _get(port) == (200, b"ok")
    assert app.calls == 1
    client, server = lookup.calls[0]
    assert server == _ep("127.0.0.1", port), "the lookup is given OUR sockname"
    assert client[0] == ipaddress.IPv4Address("127.0.0.1") and client[1] != port


def test_refused_connection_never_reaches_the_app(serve: Callable[..., int]) -> None:
    app = SpyApp()
    port = serve(
        app,
        pc.make_gated_protocol(
            _base_protocol(), _gate_source(_enforcing(FakeLookup(MY_UID + 1000)))
        ),
    )
    status, body = _get(port)
    assert status == 403 and b"not this daemon's OS user" in body
    assert app.calls == 0, "reading stayed paused: no request was parsed or dispatched"


def test_lookup_error_answers_503_and_never_reaches_the_app(serve: Callable[..., int]) -> None:
    app = SpyApp()
    port = serve(
        app,
        pc.make_gated_protocol(
            _base_protocol(), _gate_source(_enforcing(FakeLookup(OSError("x"))))
        ),
    )
    assert _get(port)[0] == 503
    assert app.calls == 0


def test_keep_alive_requests_share_one_lookup(serve: Callable[..., int]) -> None:
    app, lookup = SpyApp(), FakeLookup(MY_UID)
    port = serve(app, pc.make_gated_protocol(_base_protocol(), _gate_source(_enforcing(lookup))))
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        for _ in range(3):
            conn.request("GET", "/")
            resp = conn.getresponse()
            assert (resp.status, resp.read()) == (200, b"ok")
    finally:
        conn.close()
    assert app.calls == 3 and len(lookup.calls) == 1


def test_skipped_gate_costs_no_lookup(serve: Callable[..., int]) -> None:
    app, lookup = SpyApp(), FakeLookup(MY_UID + 1000)
    gate = pc.PeerGate(pc.GateStatus("skipped", reason="container: test"), lookup)
    port = serve(app, pc.make_gated_protocol(_base_protocol(), _gate_source(gate)))
    assert _get(port) == (200, b"ok")
    assert lookup.calls == []


def test_connection_waits_for_an_unresolved_policy(serve: Callable[..., int]) -> None:
    """A connection arriving during the self-probe is held, then decided."""
    app, lookup = SpyApp(), FakeLookup(MY_UID + 1000)
    fut: concurrent.futures.Future[pc.PeerGate] = concurrent.futures.Future()
    port = serve(app, pc.make_gated_protocol(_base_protocol(), lambda: fut))
    threading.Timer(0.2, lambda: fut.set_result(_enforcing(lookup))).start()
    assert _get(port)[0] == 403
    assert app.calls == 0


def test_ipv6_loopback_connection_is_checked(serve: Callable[..., int]) -> None:
    if not socket.has_ipv6:
        pytest.skip("no IPv6 on this host")
    app, lookup = SpyApp(), FakeLookup(MY_UID)
    try:
        port = serve(
            app,
            pc.make_gated_protocol(_base_protocol(), _gate_source(_enforcing(lookup))),
            host="::1",
        )
    except OSError:
        pytest.skip("::1 is not bindable here")
    assert _get(port, host="::1") == (200, b"ok")
    assert lookup.calls[0][1] == _ep("::1", port)


def test_default_class_is_the_auto_protocol_gated() -> None:
    cls = pc.gated_protocol_class()
    assert issubclass(cls, _base_protocol())
    assert cls.__name__.startswith("PeerChecked")
    # The --reload import string resolves to the very same class.
    from uvicorn.importer import import_from_string

    assert import_from_string(pc.PROTOCOL_IMPORT_STRING) is cls


# --------------------------------------------------------------------------
# The real native mechanism, no injection
# --------------------------------------------------------------------------


@pytest.fixture
def child_connection() -> Iterator[tuple[pc.Endpoint, pc.Endpoint]]:
    """A loopback connection whose client end lives in a CHILD process."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(10)
    port = srv.getsockname()[1]
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            f"import socket,sys;s=socket.create_connection(('127.0.0.1',{port}));sys.stdin.read()",
        ],
        stdin=subprocess.PIPE,
    )
    conn, peer = srv.accept()
    try:
        yield _ep(*peer), _ep(*conn.getsockname()[:2])
    finally:
        conn.close()
        srv.close()
        assert child.stdin is not None
        child.stdin.close()
        child.wait(timeout=10)


@pytest.mark.skipif(not NATIVE, reason="native mechanism is macOS/Linux only (Windows is slice 2)")
def test_native_lookup_resolves_a_child_connection_to_our_uid(
    child_connection: tuple[pc.Endpoint, pc.Endpoint],
) -> None:
    lookup = pc.default_lookup_factory()
    client, server = child_connection
    assert lookup.owner_uid(client, server) == MY_UID
    # An unowned tuple (port 0 is never a client port) resolves to nobody,
    # which on macOS is exactly what a foreign user's socket looks like.
    assert lookup.owner_uid((client[0], 0), server) is None
    # Full 4-tuple: the right ports on a different loopback address must not
    # match (a foreign process on 127.0.0.2:X while ours uses 127.0.0.1:X).
    other = ipaddress.ip_address("127.0.0.2")
    assert lookup.owner_uid((other, client[1]), server) is None


@pytest.mark.skipif(not NATIVE, reason="native mechanism is macOS/Linux only")
def test_native_self_probe_passes() -> None:
    assert pc.self_probe(pc.default_lookup_factory()) is None


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS libproc property")
def test_macos_cannot_inspect_another_users_process() -> None:
    """Why macOS not-found must refuse: root's launchd (pid 1) is opaque to us."""
    if MY_UID == 0:
        pytest.skip("running as root")
    lookup = pc.MacLibprocLookup()
    assert lookup._pid_owns(1, _ep("127.0.0.1", 1), _ep("127.0.0.1", 2)) is False


@pytest.mark.skipif(not NATIVE, reason="native mechanism is macOS/Linux only")
def test_real_gate_admits_a_desktop_style_request_and_a_static_preview(
    serve: Callable[..., int], tmp_path: Path, test_app_client: object
) -> None:
    """Regression: the real app behind the REAL gate still serves its clients.

    ``/health`` with the headers the desktop renderer sends, and a
    ``/v1/static/images`` preview of a file outside every served root (the
    loopback "general" posture) -- the two paths most exposed to a false
    refusal. ``test_app_client`` is requested only for the app state it
    installs (config manager on a temp dir); the requests go over real sockets.
    """
    from local_operator.server.app import app as asgi_app

    gate = pc.resolve_gate({})
    assert gate.enforcing, gate.status
    image = tmp_path / "preview.png"
    image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    port = serve(asgi_app, pc.make_gated_protocol(_base_protocol(), _gate_source(gate)))
    status, body = _get(port, "/health", Accept="application/json", Origin="null")
    assert status == 200, body
    status, body = _get(port, f"/v1/static/images?path={image}")
    assert status == 200, body
    assert body.startswith(b"\x89PNG")
