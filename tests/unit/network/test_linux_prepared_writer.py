"""Real Linux TCP mechanics only: no PeerLink/authority integration is implied.

Tiny socket buffers, payload sizes and test deadlines below are deterministic rig
choices, not product limits. The process boundary is disposable shared memory,
NOT a proposed durable fence. Keep this module dormant outside this test file.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import multiprocessing
import os
import select
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, cast

import pytest

from local_operator.network import linux_prepared_writer as writer_module
from local_operator.network.linux_prepared_writer import (
    AttemptState,
    LinuxPreparedRecordWriter,
)
from local_operator.network.wire import FrameReader, LinkCrypto, LinkKeys

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux mechanics only")


@contextlib.contextmanager
def tcp_pair():
    listener = socket.socket()
    original = socket.socket()
    peer = None
    try:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        listener.settimeout(5)
        # Set receive window before connect so the actual kernel, not a mock,
        # forces backpressure on records substantially larger than this window.
        original.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
        original.settimeout(0.15)
        original.connect(listener.getsockname())
        peer, _ = listener.accept()
        original.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        peer.settimeout(5)
        yield original, peer
    finally:
        if peer is not None:
            peer.close()
        original.close()
        listener.close()


def codecs():
    keys = LinkKeys(b"l" * 16, b"d" * 32, b"r" * 32, b"i" * 4, b"j" * 4)
    return LinkCrypto(keys, role="dialer"), LinkCrypto(keys, role="listener")


class Boundary:
    def __init__(self, state=AttemptState(0)):
        self.state = state
        self.entries = 0
        self.inside = False
        self.lock = threading.Lock()

    def __enter__(self):
        self.lock.acquire()
        self.entries += 1
        self.inside = True
        return self.state

    def __exit__(self, *args):
        self.inside = False
        self.lock.release()


class ProcessBoundary:
    def __init__(self, lock, revision):
        self.lock = lock
        self.revision = revision
        self.entries = 0
        self.inside = False

    def __enter__(self):
        self.lock.acquire()
        self.entries += 1
        self.inside = True
        return AttemptState(self.revision.value)

    def __exit__(self, *args):
        self.inside = False
        self.lock.release()


def mutate_revision(lock, revision, waiting, changed, committed_at):
    # spawn passes an independently opened OS semaphore handle to this process.
    if not waiting.wait(5):
        raise RuntimeError("writer did not reach outside-boundary backpressure")
    # Rig scheduling allowance only: let the writer enter the real select
    # syscall after announcing a non-writable socket. Timestamp containment
    # below (and the syscall evidence run) verifies the actual overlap.
    time.sleep(0.02)
    with lock:
        revision.value = 1
        committed_at.value = time.monotonic()
    changed.set()


def join(thread):
    thread.join(8)
    assert not thread.is_alive(), "owned thread was not reaped"


def receive_all(peer):
    received = bytearray()
    while chunk := peer.recv(65536):
        received.extend(chunk)
    return bytes(received)


def test_enabled_linux_writer_completes_record_without_changing_reader_timeout(
    monkeypatch,
):
    import fcntl

    with tcp_pair() as (original, peer):
        sender, receiver = codecs()
        peer_reader = FrameReader(peer)
        before = {"op": "ordinary-before"}
        protected = {"op": "protected", "body": "x" * 262144}
        after = {"op": "ordinary-after"}
        original.sendall(sender.seal(before))
        assert receiver.open(peer_reader.read_record_payload(time.monotonic() + 5)) == before
        original.settimeout(0.15)
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        duplicate = writer._duplicate
        assert original.gettimeout() == 0.15
        assert duplicate.gettimeout() == 0
        sealed = sender.seal(protected)
        boundary = Boundary()
        read_results, read_errors = [], []
        timed_out = threading.Event()
        record_done = threading.Event()
        reverse = sender.__class__(sender._keys, role="listener")
        original_decoder = sender.__class__(sender._keys, role="dialer")
        reverse_frame = {"op": "reverse"}
        reader_cycles = []

        def original_reader():
            try:
                reader = FrameReader(original)
                for _ in range(3):
                    with pytest.raises(TimeoutError):
                        reader.read_record_payload(time.monotonic() + 0.01)
                    assert duplicate.gettimeout() == 0
                    assert fcntl.fcntl(original, fcntl.F_GETFL) & os.O_NONBLOCK
                    reader_cycles.append(original.gettimeout())
                timed_out.set()
                assert record_done.wait(5)
                read_results.append(
                    original_decoder.open(reader.read_record_payload(time.monotonic() + 5))
                )
            except BaseException as exc:
                read_errors.append(exc)
                timed_out.set()

        reader_thread = threading.Thread(target=original_reader)
        reader_thread.start()
        real_select = select.select
        waits = []
        release_receiver = threading.Event()

        def observed_select(readers, writers, errors, timeout):
            assert not boundary.inside
            if writers and writers[0] is duplicate:
                ready = real_select([], writers, [], 0)[1]
                waits.append(bool(ready))
                if not ready:
                    release_receiver.set()
            return real_select(readers, writers, errors, timeout)

        monkeypatch.setattr(writer_module.select, "select", observed_select)
        receiver_results, receiver_errors = [], []

        def receive_protected():
            try:
                assert release_receiver.wait(5), "no real TCP backpressure"
                assert timed_out.wait(5)
                receiver_results.append(
                    receiver.open(peer_reader.read_record_payload(time.monotonic() + 5))
                )
            except BaseException as exc:
                receiver_errors.append(exc)

        receiver_thread = threading.Thread(target=receive_protected)
        receiver_thread.start()
        try:
            # Run the actual FrameReader timeout changes CONCURRENTLY with the
            # writer; its duplicate's timeout must remain independent throughout.
            result = writer.write_record(
                sealed, expected_revision=0, boundary=boundary, deadline=time.monotonic() + 5
            )
            join(receiver_thread)
            assert not receiver_errors
            assert receiver_results == [protected]
            assert result.complete and result.accepted_bytes == len(sealed)
            assert boundary.entries > 1 and False in waits
            assert original.gettimeout() == reader_cycles[-1]
            assert duplicate.gettimeout() == 0
            assert len(reader_cycles) == 3
            with pytest.raises(dataclasses.FrozenInstanceError):
                setattr(result, "complete", False)
            writer.close()
            writer.close()
            assert duplicate.fileno() == -1
            assert writer._wake_read.fileno() == writer._wake_write.fileno() == -1
            assert original.fileno() >= 0
            original.sendall(sender.seal(after))
            assert receiver.open(peer_reader.read_record_payload(time.monotonic() + 5)) == after
            assert sender.sent == receiver.received == 3
            peer.sendall(reverse.seal(reverse_frame))
            record_done.set()
            join(reader_thread)
            assert not read_errors
            assert read_results == [reverse_frame]
            print(
                "POSITIVE",
                {
                    "sealed": len(sealed),
                    "accepted": result.accepted_bytes,
                    "boundary_entries": boundary.entries,
                    "nonwritable_waits": waits.count(False),
                    "reader_timeout_cycles": len(reader_cycles),
                    "codec_sequence": receiver.received,
                    "original_fd_preserved": original.fileno() >= 0,
                },
            )
        finally:
            writer.cancel()
            record_done.set()
            release_receiver.set()
            peer.shutdown(socket.SHUT_RDWR)
            original.shutdown(socket.SHUT_RDWR)
            join(receiver_thread)
            join(reader_thread)


def test_linux_writer_boundary_change_abandons_partial_record_and_shuts_down(
    monkeypatch,
):
    ctx = multiprocessing.get_context("spawn")
    lock, revision = ctx.Lock(), ctx.Value("q", 0, lock=False)
    waiting, changed = ctx.Event(), ctx.Event()
    committed_at = ctx.Value("d", 0.0, lock=False)
    process = ctx.Process(
        target=mutate_revision, args=(lock, revision, waiting, changed, committed_at)
    )
    process.start()
    with tcp_pair() as (original, peer):
        sender, _ = codecs()
        sealed = sender.seal({"op": "partial", "body": "z" * 262144})
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        duplicate_fd = writer._duplicate.fileno()
        boundary = ProcessBoundary(lock, revision)
        real_select = select.select
        writer_results, errors, raw = [], [], []
        wait_spans = []

        def observed_select(readers, writers, exceptional, timeout):
            assert not boundary.inside
            if writers and writers[0] is writer._duplicate:
                if not real_select([], writers, [], 0)[1]:
                    started = time.monotonic()
                    waiting.set()
                    ready = real_select(readers, writers, exceptional, timeout)
                    wait_spans.append((started, time.monotonic()))
                    return ready
            return real_select(readers, writers, exceptional, timeout)

        monkeypatch.setattr(writer_module.select, "select", observed_select)

        def write():
            try:
                writer_results.append(
                    writer.write_record(
                        sealed,
                        expected_revision=0,
                        boundary=boundary,
                        deadline=time.monotonic() + 8,
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=write)
        thread.start()
        try:
            assert waiting.wait(5)
            assert changed.wait(5)
            assert revision.value == 1

            # Drain only after the separate process won. The next actual send
            # must be refused, even though TCP becomes writable again.
            def drain():
                try:
                    raw.append(receive_all(peer))
                except BaseException as exc:
                    errors.append(exc)

            drain_thread = threading.Thread(target=drain)
            drain_thread.start()
            try:
                join(thread)
                assert not errors
                result = writer_results[0]
                assert not result.complete and result.reason == "revision"
            finally:
                writer.cancel()
                join(drain_thread)
            assert 0 < result.accepted_bytes < len(sealed)
            assert raw[0] == sealed[: result.accepted_bytes]
            assert boundary.entries > 1
            assert any(start < committed_at.value < end for start, end in wait_spans)
            with pytest.raises(ConnectionError):
                FrameReader(peer, buffered=raw[0]).read_record_payload(time.monotonic() + 1)
            with pytest.raises(RuntimeError, match="closed"):
                writer.write_record(
                    b"later", expected_revision=1, boundary=boundary, deadline=time.monotonic() + 1
                )
            assert original.fileno() >= 0
            assert original.recv(1) == b""
            with pytest.raises(OSError):
                os.fstat(duplicate_fd)
            writer.cancel()
            writer.close()
            process.join(5)
            assert not process.is_alive() and process.exitcode == 0
            print(
                "REVOKE",
                {
                    "sealed": len(sealed),
                    "accepted_prefix": result.accepted_bytes,
                    "received": len(raw[0]),
                    "post_win_continuations": 0,
                    "mutator_committed_inside_socket_wait": any(
                        start < committed_at.value < end for start, end in wait_spans
                    ),
                    "mutator_commit_monotonic": committed_at.value,
                    "outside_wait_spans": wait_spans,
                    "boundary_entries": boundary.entries,
                    "process_exit": process.exitcode,
                    "original_fd_preserved": original.fileno() >= 0,
                },
            )
        finally:
            writer.cancel()
            waiting.set()
            changed.set()
            peer.shutdown(socket.SHUT_RDWR)
            join(thread)
            process.join(5)
            if process.is_alive():
                process.terminate()
                process.join(5)
            assert not process.is_alive()
            process.close()


@pytest.mark.parametrize("flag", [None, 0, 1, "true", False])
def test_flag_refuses_before_socket_effects(flag):
    class NoEffects:
        def gettimeout(self):
            pytest.fail("disabled/non-bool flag touched original socket")

    error = RuntimeError if flag is False else TypeError
    with pytest.raises(error):
        LinuxPreparedRecordWriter(
            cast(socket.socket, NoEffects()), linux_prepared_writer_enabled=flag
        )
    with pytest.raises(RuntimeError, match="disabled"):
        LinuxPreparedRecordWriter(cast(socket.socket, NoEffects()))


def test_nonlinux_and_original_timeout_refuse_before_dup(monkeypatch):
    class NoDup:
        def __init__(self, timeout):
            self.timeout = timeout

        def gettimeout(self):
            return self.timeout

        def dup(self):
            pytest.fail("refusal duplicated original")

    with monkeypatch.context() as patch:
        patch.setattr(writer_module.sys, "platform", "darwin")
        with pytest.raises(RuntimeError, match="requires Linux"):
            LinuxPreparedRecordWriter(
                cast(socket.socket, object()), linux_prepared_writer_enabled=True
            )
    for timeout in (None, 0):
        with pytest.raises(ValueError, match="positive"):
            LinuxPreparedRecordWriter(
                cast(socket.socket, NoDup(timeout)), linux_prepared_writer_enabled=True
            )


@pytest.mark.parametrize(
    "state,reason",
    [
        (AttemptState(1), "revision"),
        (AttemptState(0, pending=True), "pending"),
        (AttemptState(0, cancelled=True), "cancelled"),
    ],
)
def test_zero_first_send_abandons(state, reason):
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        boundary = Boundary(state)
        result = writer.write_record(
            b"sealed", expected_revision=0, boundary=boundary, deadline=time.monotonic() + 1
        )
        assert result.accepted_bytes == 0 and result.reason == reason
        assert peer.recv(1) == b""
        writer.close()
        writer.cancel()
        assert original.fileno() >= 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"sealed_record": bytearray(b"x")},
        {"sealed_record": memoryview(b"x")},
        {"sealed_record": b""},
        {"expected_revision": True},
        {"deadline": float("inf")},
        {"deadline": True},
    ],
)
def test_invalid_input_refuses_before_boundary_or_send(kwargs):
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        boundary = Boundary()
        # Deliberately ill-typed runtime inputs exercise the refusal contract.
        values: dict[str, Any] = dict(
            sealed_record=b"x",
            expected_revision=0,
            boundary=boundary,
            deadline=time.monotonic() + 1,
        )
        values.update(kwargs)
        try:
            with pytest.raises(TypeError):
                writer.write_record(**values)
            assert boundary.entries == 0
            assert not select.select([peer], [], [], 0)[0]
        finally:
            writer.close()


@pytest.mark.parametrize("state", [True, {"revision": 0}, object()])
def test_boundary_must_yield_closed_snapshot(state):
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        with pytest.raises(TypeError, match="exact AttemptState"):
            writer.write_record(
                b"x", expected_revision=0, boundary=Boundary(state), deadline=time.monotonic() + 1
            )
        assert peer.recv(1) == b""
        assert writer._duplicate.fileno() == -1


@pytest.mark.parametrize("args", [(True,), (-1,), (0, 1), (0, False, 1)])
def test_snapshot_exact_types(args):
    with pytest.raises(TypeError):
        AttemptState(*args)


def test_accepted_record_setup_exception_shuts_down(monkeypatch):
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        boundary = Boundary()

        def allocation_failure(record):
            raise MemoryError("synthetic memoryview allocation failure")

        monkeypatch.setattr(writer_module, "memoryview", allocation_failure, raising=False)
        with pytest.raises(MemoryError):
            writer.write_record(
                b"sealed",
                expected_revision=0,
                boundary=boundary,
                deadline=time.monotonic() + 1,
            )
        assert boundary.entries == 0
        assert writer._duplicate.fileno() == -1
        assert writer._wake_read.fileno() == writer._wake_write.fileno() == -1
        assert peer.recv(1) == b""
        assert original.fileno() >= 0
        writer.close()
        writer.cancel()


@pytest.mark.parametrize("stop", ["cancel", "close"])
def test_cancel_and_close_do_not_release_active_attempt_handles(monkeypatch, stop):
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        boundary = Boundary()
        waiting, release = threading.Event(), threading.Event()
        real_select = select.select
        real_release = writer._release_locked
        releases = []

        def release_once():
            releases.append(writer._duplicate.fileno())
            real_release()

        monkeypatch.setattr(writer, "_release_locked", release_once)
        results, errors = [], []

        def paused_wait(readers, writers, exceptional, timeout):
            assert not boundary.inside
            if writers and not real_select([], writers, [], 0)[1]:
                waiting.set()
                assert release.wait(5)
            return real_select(readers, writers, exceptional, timeout)

        monkeypatch.setattr(writer_module.select, "select", paused_wait)

        def write():
            try:
                results.append(
                    writer.write_record(
                        b"s" * 262144,
                        expected_revision=0,
                        boundary=boundary,
                        deadline=time.monotonic() + 8,
                    )
                )
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=write)
        thread.start()
        try:
            assert waiting.wait(5)
            with pytest.raises(RuntimeError, match="another record"):
                writer.write_record(
                    b"second", expected_revision=0, boundary=boundary, deadline=time.monotonic() + 1
                )
            fd = writer._duplicate.fileno()
            getattr(writer, stop)()
            getattr(writer, stop)()
            assert writer._duplicate.fileno() == fd
            assert releases == []
            os.fstat(fd)  # held by the active owner even after cancellation
            release.set()
            join(thread)
            assert not errors
            assert not results[0].complete
            assert writer._duplicate.fileno() == -1
            assert writer._wake_read.fileno() == writer._wake_write.fileno() == -1
            assert releases == [fd]
            writer.cancel()
            writer.close()
            assert releases == [fd]
            assert original.fileno() >= 0
            assert len(receive_all(peer)) == results[0].accepted_bytes
        finally:
            release.set()
            writer.cancel()
            join(thread)


def test_deadline_and_socket_error_abandon():
    for failure in ("deadline", "socket-error"):
        with tcp_pair() as (original, peer):
            writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
            if failure == "socket-error":
                original.shutdown(socket.SHUT_WR)
            result = writer.write_record(
                b"sealed",
                expected_revision=0,
                boundary=Boundary(),
                deadline=time.monotonic() + (1 if failure == "socket-error" else -1),
            )
            assert not result.complete and result.reason == failure
            assert result.accepted_bytes == 0
            assert writer._duplicate.fileno() == -1
            assert peer.recv(1) == b""


def test_positive_timeout_control_and_kernel_mode_race():
    import fcntl

    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        duplicate = writer._duplicate
        accepted = 0
        try:
            # Fill the *actual* buffer until writable readiness is false. A
            # positive-timeout wrapper is deliberately NOT the writer solution.
            while True:
                try:
                    accepted += duplicate.send(b"x" * 65536, socket.MSG_DONTWAIT)
                except BlockingIOError:
                    if not select.select([], [duplicate], [], 0.02)[1]:
                        break
            started = time.monotonic()
            with pytest.raises(TimeoutError):
                original.send(b"control", socket.MSG_DONTWAIT)
            elapsed = time.monotonic() - started
            original_timeout = original.gettimeout()
            assert original_timeout is not None
            assert elapsed >= original_timeout * 0.8
            # Another wrapper can clear shared O_NONBLOCK. MSG_DONTWAIT still
            # must make the helper's one actual send return EAGAIN immediately.
            original.setblocking(True)
            assert not (fcntl.fcntl(duplicate, fcntl.F_GETFL) & os.O_NONBLOCK)
            assert duplicate.gettimeout() == 0
            # ACKs can free space during the control's 150ms timeout even
            # without writable readiness. Re-establish real saturation, counting
            # every accepted byte, before judging the blocking-mode race.
            while True:
                try:
                    accepted += duplicate.send(b"x" * 65536, socket.MSG_DONTWAIT)
                except BlockingIOError:
                    try:
                        accepted += duplicate.send(b"x", socket.MSG_DONTWAIT)
                    except BlockingIOError:
                        break
            with pytest.raises(BlockingIOError):
                duplicate.send(b"x", socket.MSG_DONTWAIT)
            sealed = b"s" * 65536
            result = writer.write_record(
                sealed,
                expected_revision=0,
                boundary=Boundary(),
                deadline=time.monotonic() + 0.02,
            )
            assert not result.complete and result.reason == "deadline"
            received = receive_all(peer)
            assert received == b"x" * accepted + sealed[: result.accepted_bytes]
            assert original.gettimeout() is None
            print(
                "CONTROL",
                {
                    "saturated_bytes": accepted,
                    "positive_timeout_s": elapsed,
                    "duplicate_eagain": True,
                    "shared_blocking_mode": "exercised",
                },
            )
        finally:
            original.settimeout(0.15)
            writer.cancel()


def test_no_production_consumer_or_flag_override():
    root = Path(writer_module.__file__).resolve().parents[1]
    violations = []
    for path in root.rglob("*.py"):
        if path.resolve() == Path(writer_module.__file__).resolve():
            continue
        source = path.read_text(encoding="utf-8")
        # Every syntax form examined below contains one of these exact strings.
        # Filter first rather than parsing the unrelated production tree: the
        # dormancy check stays exhaustive without spending seconds on other ASTs.
        if "linux_prepared_writer" not in source and "LinuxPreparedRecordWriter" not in source:
            continue
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [name.name for name in node.names]
            elif isinstance(node, ast.Import):
                names = [name.name for name in node.names]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                names = [node.value]
            elif isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            else:
                continue
            if any(
                "linux_prepared_writer" in name or "LinuxPreparedRecordWriter" in name
                for name in names
            ):
                violations.append(f"{path.relative_to(root)}:{node.lineno}")
    assert violations == []
    source = Path(writer_module.__file__).read_text(encoding="utf-8")
    assert "getenv" not in source and "environ" not in source
