"""Does a platform's GROUP census still list an unreaped zombie leader?

This is the measurement the pin's scope rests on, so it is written to ANSWER the
question rather than to pass: it prints what the platform said (the CI log then
carries a measurement, not a claim) and fails with the reason named when a
precondition cannot be established — never a silent skip.

Runs on macOS and Linux (POSIX process groups); skipped where there is no fork.
Signals nothing: it waits, samples, censuses, and reaps exactly its own child.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

import pytest

from local_operator import owned_process_group as owned

pytestmark = pytest.mark.skipif(
    not hasattr(os, "fork") or os.name != "posix",
    reason="POSIX process groups are required to ask this question at all",
)


def test_a_zombie_leader_is_visible_to_the_group_census_on_this_platform():
    deadline = time.monotonic() + 10.0
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - the child never returns
        os.close(write_fd)
        os.setsid()  # its own session, and therefore its own process group
        os.read(read_fd, 1)  # wait for the parent's go-ahead
        os._exit(0)
    os.close(read_fd)
    try:
        # Capture while the leader is still ALIVE: that is the wrapper's own
        # order, and the corpse arrives afterwards, exactly as in a real run.
        cap = owned.OwnedGroup.capture(
            pid,
            parent_pid=os.getpid(),
            scheme=owned.procstate.birth_scheme,
            clock=time.monotonic,
            signal_group=lambda *args: None,
            deadline=time.monotonic() + 5.0,
        )
        assert cap.last.status == "VERIFIED", ("capture of a live leader", cap.last.reason)

        os.write(write_fd, b"x")
        os.close(write_fd)
        write_fd = -1

        sample = None
        while time.monotonic() < deadline:
            sample = owned.bounded_samples([pid], time.monotonic() + 1.0).get(pid)
            if sample is not None and sample.zombie:
                break
            time.sleep(0.01)
        assert sample is not None, "the leader never became a zombie inside the bound"
        assert sample.zombie is True, "the sampler does not report the corpse as terminal"

        census = owned.strict_census(pid, time.monotonic() + 5.0)
        visible = pid in census
        verdict = cap.refresh(deadline=time.monotonic() + 5.0)
        # With a corpse and nothing else in it, the group has no live member and
        # refresh answers OBSERVED_EMPTY — that is not a lost reservation. What
        # would be a lost reservation is `pin_lost`, which is what this asserts
        # against; the live-member half is exercised by the incident-shape test.
        claimable = verdict.refusal_class != "pin_lost"
        answer = {
            "platform": sys.platform,
            "zombie_sample": sample.zombie,
            "zombie_in_group_census": visible,
            "census": census,
            "reservation_claimable": claimable,
            "anchor": verdict.anchor,
            "reason": verdict.reason,
        }
        print("DARWIN_PROBE " + json.dumps(answer), flush=True)

        # (a) the platform's answer to the unknown, asserted rather than assumed
        assert visible, (
            "the group census does NOT list the unreaped zombie leader on this "
            "platform; the pin must stay live-leader-only here and the docs must "
            "say so: " + json.dumps(answer)
        )
        # (c) therefore the reservation is claimable from the corpse: the pin did
        # not report itself lost on a group whose only member is the leader.
        assert claimable, (answer, verdict.reason)
    finally:
        if write_fd != -1:
            os.close(write_fd)
        try:
            os.waitpid(pid, 0)  # reap EXACTLY our own child
        except ChildProcessError:  # already reaped by the failed-precondition path
            pass


@pytest.mark.skipif(not sys.platform.startswith("darwin"), reason="Darwin census shape")
def test_a_new_session_child_is_visible_to_the_census():
    """F1: the runner spawns with `start_new_session=True`, so this is production.

    `proc_listpids(PROC_PGRP_ONLY, ...)` cannot list a group belonging to another
    session and answers an EMPTY list rather than an error — which is how a live
    leader's capture came back OBSERVED_EMPTY on macOS. An earlier measurement
    missed it because the probe child used `setpgid(0, 0)` and stayed in the
    caller's session; the convenient case was the wrong case. This child calls
    `setsid()` and must still be found. Fails at 53ac9c08.
    """
    import os

    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # pragma: no cover - child
        try:
            os.close(read_fd)
            os.setsid()
            os.write(write_fd, str(os.getpid()).encode())
            time.sleep(30)
        finally:
            os._exit(0)
    os.close(write_fd)
    with os.fdopen(read_fd, "rb") as handle:
        child = int(handle.read())
    try:
        assert os.getsid(child) != os.getsid(0), "the child must be in a NEW session"
        members = owned.strict_census(child, time.monotonic() + 5.0)
        assert child in members, (child, members)
    finally:
        try:
            os.kill(child, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            os.waitpid(child, 0)
        except ChildProcessError:
            pass
