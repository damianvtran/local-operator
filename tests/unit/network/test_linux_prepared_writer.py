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

#: select(2)'s FD_SETSIZE ceiling: the descriptor height these tests exist to
#: clear when they lease a high slot. CPython does not export the C macro
#: (``select.FD_SETSIZE`` does not exist), so it is spelled once, here, every use
#: derives from it, and the assert below makes that derivation enforceable rather
#: than aspirational: a floor quietly lowered would otherwise stop exercising the
#: high-descriptor path while still passing.
FDSET_CEILING = 1024
# Deliberately a literal, not FDSET_CEILING -- a guard phrased in terms of the value
# it guards could never fire. Lowering the constant now fails at import instead of
# silently turning these tests back into ordinary low-descriptor runs.
assert FDSET_CEILING >= 1024, "never below select(2)'s FD_SETSIZE"


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
    # Rig scheduling allowance only: let the writer enter the real selector
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


@contextlib.contextmanager
def high_descriptor(writer, attribute):
    """Lease one slot at or above FD_SETSIZE here, not in the service.

    "At or above", not "above": the shrunken-limit branch can land on exactly 1024.
    F_DUPFD_CLOEXEC avoids opening thousands of descriptors merely to reproduce
    fd_set's ceiling. Restore the soft limit even when the old source fails.
    """
    import fcntl
    import resource

    limits = resource.getrlimit(resource.RLIMIT_NOFILE)
    try:
        if limits[0] <= FDSET_CEILING:
            # Width stays +2, not +1: it must still admit the lease when 1024 is already
            # occupied -- the dup then lands on 1025, the last slot this window allows.
            assert limits[1] >= FDSET_CEILING + 2, "high-descriptor test lease unavailable"
            resource.setrlimit(resource.RLIMIT_NOFILE, (FDSET_CEILING + 2, limits[1]))
        low = getattr(writer, attribute)
        # The floor argument IS the ceiling guarantee: F_DUPFD_CLOEXEC returns the
        # LOWEST free slot at or above the floor, so it cannot hand back a slot below
        # FD_SETSIZE. Re-asserting that here would be a tautology, not evidence.
        fd = fcntl.fcntl(low.fileno(), fcntl.F_DUPFD_CLOEXEC, FDSET_CEILING)
        high = socket.socket(fileno=fd)
        try:
            high.settimeout(low.gettimeout())
            setattr(writer, attribute, high)
            # NEVER assert the slot NUMBER. Which free slot the floor lands on is
            # incidental to clearing the ceiling: 1024/1025 only happens in a
            # pristine process, and nothing keeps a worker near FD_SETSIZE -- an
            # xdist worker that already held 1024..1112 was handed 1113 (CI run
            # 45172624920, job `test (3.13, 2)`: "assert 1113 in (1024, 1025)").
            # What the test does need is that the swap took, and that can still
            # fail, so the swapped descriptor is what is asserted. The observed
            # value is in the message: a bare range told the next reader nothing
            # about the process it ran in.
            leased = getattr(writer, attribute)
            assert (
                leased.fileno() == fd
            ), f"writer {attribute} still holds fd {leased.fileno()}, not the leased {fd}"
            low.close()
            yield fd
        finally:
            writer.close()
            high.close()
            with pytest.raises(OSError):
                os.fstat(fd)
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, limits)
        assert resource.getrlimit(resource.RLIMIT_NOFILE) == limits
        print("NOFILE_RESTORED", limits)


@pytest.mark.parametrize("attribute", [None, "_duplicate", "_wake_read"])
def test_high_descriptor_completes_authenticated_record(attribute):
    with tcp_pair() as (original, peer):
        sender, receiver = codecs()
        reader = FrameReader(peer)
        before = {"op": "ordinary-before"}
        original.sendall(sender.seal(before))
        assert receiver.open(reader.read_record_payload(time.monotonic() + 5)) == before
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        boundary = Boundary()
        frame = {"op": "high-descriptor", "body": "h" * 262144}
        sealed = sender.seal(frame)
        received, errors = [], []

        def receive():
            try:
                # Rig delay establishes actual TCP backpressure; never product policy.
                time.sleep(0.05)
                received.append(receiver.open(reader.read_record_payload(time.monotonic() + 5)))
            except BaseException as exc:
                errors.append(exc)

        lease = high_descriptor(writer, attribute) if attribute else contextlib.nullcontext(None)
        with lease as fd:
            thread = threading.Thread(target=receive)
            thread.start()
            try:
                result = writer.write_record(
                    sealed, expected_revision=0, boundary=boundary, deadline=time.monotonic() + 5
                )
                join(thread)
                assert not errors and received == [frame]
                assert result.complete and result.accepted_bytes == len(sealed)
                assert boundary.entries > 1
                writer.close()
                after = {"op": "ordinary-after"}
                original.sendall(sender.seal(after))
                assert receiver.open(reader.read_record_payload(time.monotonic() + 5)) == after
                peer.sendall(receiver.seal({"op": "reverse"}))
                reverse = sender.open(
                    FrameReader(original).read_record_payload(time.monotonic() + 5)
                )
                assert reverse == {"op": "reverse"}
                assert sender.sent == receiver.received == 3
                print(
                    "HIGHFD_COMPLETE", attribute, fd, result, "ordinary-before/after reverse PASS"
                )
            finally:
                writer.cancel()
                join(thread)


@pytest.mark.parametrize("attribute", [None, "_duplicate", "_wake_read"])
@pytest.mark.parametrize("stop", ["cancel", "close", "deadline"])
def test_backpressured_descriptor_abandons_exact_prefix(attribute, stop, monkeypatch):
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        lease = high_descriptor(writer, attribute) if attribute else contextlib.nullcontext(None)
        with lease as fd:
            boundary = Boundary()
            sender, _ = codecs()
            sealed = sender.seal({"op": "abandoned", "body": "p" * 262144})
            real_duplicate = writer._duplicate
            partial = threading.Event()
            accepted, results, errors = [], [], []

            class ObservedSend:
                def __getattr__(self, name):
                    return getattr(real_duplicate, name)

                def send(self, data, flags):
                    assert boundary.inside
                    sent = real_duplicate.send(data, flags)
                    accepted.append(sent)
                    if 0 < sent < len(data):
                        partial.set()
                    return sent

            monkeypatch.setattr(writer, "_duplicate", ObservedSend())

            def write():
                try:
                    results.append(
                        writer.write_record(
                            sealed,
                            expected_revision=0,
                            boundary=boundary,
                            deadline=time.monotonic() + (0.15 if stop == "deadline" else 5),
                        )
                    )
                except BaseException as exc:
                    errors.append(exc)

            thread = threading.Thread(target=write)
            thread.start()
            try:
                assert partial.wait(5), "no actual partial TCP send"
                if stop != "deadline":
                    # Let the actual continuation enter readiness. On old source
                    # a high fd instead escapes with ValueError, which is RED.
                    time.sleep(0.02)
                    assert thread.is_alive(), errors
                    getattr(writer, stop)()
                join(thread)
                assert not errors, errors
                result = results[0]
                assert not result.complete
                assert result.reason == ("deadline" if stop == "deadline" else "cancelled")
                assert 0 < result.accepted_bytes == sum(accepted) < len(sealed)
                raw = receive_all(peer)
                assert raw == sealed[: result.accepted_bytes]
                with pytest.raises(ConnectionError):
                    FrameReader(peer, buffered=raw).read_record_payload(time.monotonic() + 1)
                assert original.fileno() >= 0 and original.recv(1) == b""
                with pytest.raises(RuntimeError, match="closed"):
                    writer.write_record(
                        sealed,
                        expected_revision=0,
                        boundary=boundary,
                        deadline=time.monotonic() + 1,
                    )
                writer.cancel()
                writer.close()
                assert real_duplicate.fileno() == -1
                assert writer._wake_read.fileno() == writer._wake_write.fileno() == -1
                print("PREFIX_EOF", attribute, fd, stop, result, "next-refusal owned-cleanup PASS")
            finally:
                writer.cancel()
                join(thread)
                print(
                    "DESCRIPTOR_RUN",
                    attribute,
                    fd,
                    stop,
                    "accepted_bytes",
                    sum(accepted),
                    "results",
                    results,
                    "errors",
                    [repr(exc) for exc in errors],
                )


def observe_readiness(monkeypatch, observe):
    """Instrument the actual selector syscall, not a replacement readiness rig."""
    real_selector = writer_module.selectors.DefaultSelector

    class ObservedSelector(real_selector):
        def select(self, timeout=None):
            return observe(super().select, timeout)

    monkeypatch.setattr(writer_module.selectors, "DefaultSelector", ObservedSelector)


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
        waits = []
        release_receiver = threading.Event()

        def observed_select(real_select, timeout):
            assert not boundary.inside
            ready = real_select(0)
            waits.append(bool(ready))
            if not ready:
                release_receiver.set()
            return real_select(timeout)

        observe_readiness(monkeypatch, observed_select)
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
        writer_results, errors, raw = [], [], []
        wait_spans = []

        def observed_select(real_select, timeout):
            assert not boundary.inside
            if not real_select(0):
                started = time.monotonic()
                waiting.set()
                ready = real_select(timeout)
                wait_spans.append((started, time.monotonic()))
                return ready
            return real_select(timeout)

        observe_readiness(monkeypatch, observed_select)

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
        real_release = writer._release_locked
        releases = []

        def release_once():
            releases.append(writer._duplicate.fileno())
            real_release()

        monkeypatch.setattr(writer, "_release_locked", release_once)
        results, errors = [], []

        def paused_wait(real_select, timeout):
            assert not boundary.inside
            if not real_select(0):
                waiting.set()
                assert release.wait(5)
            return real_select(timeout)

        observe_readiness(monkeypatch, paused_wait)

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


def test_empty_readiness_retries_deadline_without_stale_send(monkeypatch):
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        boundary = Boundary()
        snapshots = []

        def empty_once(real_select, timeout):
            assert not boundary.inside
            snapshots.append((boundary.entries, timeout))
            if len(snapshots) == 1:
                return []  # DefaultSelector's interrupted-wait result.
            assert snapshots[1][0] == snapshots[0][0]
            assert 0 < snapshots[1][1] <= snapshots[0][1]
            writer.cancel()
            # Both handles are ready on this initially writable TCP endpoint;
            # cancellation must win over the write event, without another send.
            ready = real_select(timeout)
            assert {key.fileobj for key, _ in ready} == {writer._duplicate, writer._wake_read}
            return ready

        observe_readiness(monkeypatch, empty_once)
        result = writer.write_record(
            b"s" * 262144, expected_revision=0, boundary=boundary, deadline=time.monotonic() + 1
        )
        assert not result.complete and result.reason == "cancelled"
        assert boundary.entries == 1
        assert receive_all(peer) == b"s" * result.accepted_bytes
        assert writer._duplicate.fileno() == -1
        assert writer._wake_read.fileno() == writer._wake_write.fileno() == -1


def test_signal_interrupted_backpressure_keeps_callers_deadline():
    import signal

    # A real signal interrupts the actual epoll wait; Python/DefaultSelector
    # must retain the finite absolute deadline, not translate EINTR to success.
    with tcp_pair() as (original, peer):
        writer = LinuxPreparedRecordWriter(original, linux_prepared_writer_enabled=True)
        signals = []
        previous_handler = signal.getsignal(signal.SIGALRM)
        previous_timer = signal.getitimer(signal.ITIMER_REAL)
        assert previous_timer == (0.0, 0.0), "test process already owns an alarm"
        signal.signal(signal.SIGALRM, lambda *_: signals.append(time.monotonic()))
        started = time.monotonic()
        deadline = started + 0.15
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.04)
            result = writer.write_record(
                b"s" * 262144, expected_revision=0, boundary=Boundary(), deadline=deadline
            )
            assert signals and started < signals[0] < deadline
            assert not result.complete and result.reason == "deadline"
            assert time.monotonic() >= deadline
            assert 0 < result.accepted_bytes < 262144
            assert receive_all(peer) == b"s" * result.accepted_bytes
            assert writer._duplicate.fileno() == -1
            assert writer._wake_read.fileno() == writer._wake_write.fileno() == -1
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)
            writer.close()


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
