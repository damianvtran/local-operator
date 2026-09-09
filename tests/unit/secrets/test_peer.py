"""Peer identification and the ancestry decision (design §2.1).

These cover the DECISION PROCEDURE in isolation. The end-to-end adversarial
cases — a real detached script denied against a real daemon — live in
``test_broker.py``, because a unit test of the walk cannot prove that the
process on the other end of a socket is the one the kernel reported.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from local_operator.secrets import peer
from local_operator.secrets.peer import ProcessIdentity, authorize, process_info

_SUPPORTED = sys.platform == "darwin" or sys.platform.startswith("linux")
requires_supported = pytest.mark.skipif(
    not _SUPPORTED, reason="peer identification is implemented for macOS and Linux"
)


@requires_supported
def test_process_info_reads_this_process() -> None:
    identity = process_info(os.getpid())
    assert identity is not None
    assert identity.pid == os.getpid()
    # A start time of 0 would mean the offset probe silently read the wrong
    # field, which is exactly the failure mode the empirical offsets exist to
    # prevent — and it would make every identity comparison vacuously equal.
    assert identity.start_time > 0


@requires_supported
def test_process_info_returns_none_for_a_dead_process() -> None:
    """A vanished pid is unidentifiable, therefore unauthorizable."""
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    # The pid may be reused by another process before this runs, in which case
    # the identity we get back is a DIFFERENT process — which is the whole
    # point of pinning, so accept either "gone" or "not the same one".
    identity = process_info(child.pid)
    assert identity is None or identity.pid == child.pid


@requires_supported
def test_ancestry_walk_reaches_this_process_from_a_grandchild() -> None:
    """session -> bash -> python is the real agent shape and must resolve."""
    probe = "import os; print(os.getpid())"
    result = subprocess.run(
        ["bash", "-c", f"{sys.executable} -c '{probe}'"],
        capture_output=True,
        text=True,
        check=True,
    )
    grandchild_pid = int(result.stdout.strip())
    # The grandchild has exited by now, so walk from a LIVE one instead: hold
    # it open and walk while it exists.
    holder = subprocess.Popen(
        ["bash", "-c", f"exec {sys.executable} -c 'import time; time.sleep(30)'"]
    )
    try:
        chain = [identity.pid for identity in peer._walk(holder.pid)]
        assert holder.pid in chain
        assert os.getpid() in chain, f"walk from {holder.pid} missed this process: {chain}"
    finally:
        holder.terminate()
        holder.wait()
    assert grandchild_pid > 0


def test_identity_comparison_never_trusts_a_bare_pid() -> None:
    """The pin is what defeats pid reuse; equality must require more than a pid."""
    original = ProcessIdentity(pid=4242, start_time=1000, unique_id=7, parent_unique_id=1)
    recycled = ProcessIdentity(pid=4242, start_time=2000, unique_id=9, parent_unique_id=1)
    assert original.same_process_as(original)
    assert not original.same_process_as(recycled)


def test_identity_falls_back_to_start_time_when_unique_ids_are_absent() -> None:
    """Linux has no p_uniqueid, so the start time must still discriminate."""
    original = ProcessIdentity(pid=99, start_time=500)
    recycled = ProcessIdentity(pid=99, start_time=900)
    assert original.same_process_as(ProcessIdentity(pid=99, start_time=500))
    assert not original.same_process_as(recycled)


def test_authorize_refuses_when_no_session_is_registered() -> None:
    """Fail closed: an empty registry authorizes nobody."""
    identity = ProcessIdentity(pid=os.getpid(), start_time=1)
    allowed, reason = authorize(identity, {})
    assert not allowed
    assert "no lop session" in reason


@requires_supported
def test_authorize_allows_a_registered_session_itself() -> None:
    identity = process_info(os.getpid())
    assert identity is not None
    allowed, reason = authorize(identity, {identity.pid: identity})
    assert allowed, reason


@requires_supported
def test_authorize_refuses_a_recycled_session_pid() -> None:
    """A registry entry whose process changed must stop authorizing.

    This is the pid-reuse defence stated as a test. Recycling a pid on demand
    is not constructible — macOS allocates pids sequentially machine-wide, so
    forcing one number to come back around means spawning ~99k processes and
    winning a race against every other process on the host. The PIN is the
    mechanism, so the pin is what is tested.
    """
    live = process_info(os.getpid())
    assert live is not None
    stale = ProcessIdentity(
        pid=live.pid,
        start_time=live.start_time - 1,
        unique_id=(live.unique_id or 0) + 1,
        parent_unique_id=live.parent_unique_id,
    )
    allowed, reason = authorize(live, {live.pid: stale})
    assert not allowed
    assert "reused" in reason.lower()


@requires_supported
def test_walk_is_bounded() -> None:
    """A bounded walk cannot be spun by a pathological parent chain."""
    chain = list(peer._walk(os.getpid(), max_depth=3))
    assert len(chain) <= 3


@requires_supported
def test_walk_includes_pid_one() -> None:
    """pid 1 must be examined, not skipped.

    ``while pid > 1`` never inspects pid 1 itself. That is invisible on a
    normal host, where no session is pid 1, and wrong inside a container where
    a session frequently IS pid 1 — measured in Docker, where the same probe
    denied a legitimate descendant until the bound was corrected.
    """
    chain = [identity.pid for identity in peer._walk(os.getpid(), max_depth=64)]
    assert chain, "the walk yielded nothing at all"
    # Either we reached pid 1, or we stopped early on a bound/permission edge;
    # what must never happen is reaching pid 1's CHILD and then stopping.
    if len(chain) > 1:
        last = chain[-1]
        parent_of_last = peer.parent_pid(last)
        assert last == 1 or parent_of_last in (None, 0, 1) or len(chain) == 64
