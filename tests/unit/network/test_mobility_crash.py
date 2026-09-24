"""The crash windows: ``kill -9`` at each point of a move, then recover.

THE RISKIEST PIECE OF THIS SLICE, so it is proven by killing a real process rather
than by simulating the loss. Both relays (owner and destination) live in a FORKED
CHILD; the child drives a move up to a designated point and parks; the parent
kills it with SIGKILL — no cleanup, no flush, nothing gets a chance to tidy up.
What survives is exactly what the design claims is sufficient: the files on disk.

THE THREE POINTS are the ones the design's recovery table turns on (§6.5):

* ``before_ready`` — the destination finished copying into staging and died before
  it told the owner the copy verified. Nothing was committed; the owner still holds
  the whole conversation and its directory.
* ``after_handing_off`` — the owner advanced the journal and died before its
  tombstone and its delete. THE PHASE WHERE A ROLLBACK IS IMPOSSIBLE BY RULE, so
  the recovery has to COMPLETE the handoff, not undo it.
* ``after_committed`` — the owner committed (tombstone written, directory gone) and
  the destination died before it promoted. The destination's only proof it may
  adopt the id is the owner's tombstone, and it has to be able to get it.

EACH WINDOW ASSERTS THE SAME INVARIANT, which is INV-1 at rest: after recovery
EXACTLY ONE device holds a directory for the id, and it holds the WHOLE transcript
(byte-identical to what the source had before the move). ``test_..._exactly_one``
counts holders by walking both roots, so a second copy anywhere fails the test.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest

SESSION = "9f3ac1e0b7d2"
WINDOWS = ("before_ready", "after_handing_off", "after_committed")

#: How long the parent waits for the child to reach its crash point. Generous: the
#: child builds two relays, pairs them over real TCP and copies a transcript.
REACH_TIMEOUT_S = 120.0


def _child_body(setup: Path, window: str, ready_marker: Path, crash_marker: Path) -> int:
    """Everything the crashing device does, in its own process.

    Runs AFTER the fork, so the relays, their threads and their sockets belong to a
    process that can be killed outright. ``monkeypatch`` here is a real
    ``MonkeyPatch`` instance rather than a fixture, because a forked child has no
    pytest fixtures — the same patching the shared pairing helper needs.
    """
    from _pytest.monkeypatch import MonkeyPatch

    from local_operator.network import identity, mobility, relay
    from local_operator.session.cleanup import mark_store
    from local_operator.session.placement import (
        MeshStamp,
        SessionPlacement,
        write_stamp,
    )
    from tests.unit.network.test_relay_e2e import _pair

    root_a = setup / "a"
    root_b = setup / "b"
    identity.mint(root_a, name="device-a")
    identity.mint(root_b, name="device-b")
    server_a = relay.RelayServer(
        root=root_a,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
    )
    server_b = relay.RelayServer(
        root=root_b,
        settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1"),
    )
    host, port = server_a.bind()
    server_a.bind_control()
    server_b.bind_control()
    server_a.start()
    server_b.start()
    mp = MonkeyPatch()
    mp.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root_b))
    record, _host, _port = _pair((server_a, server_b, host, port), mp, role="admin")

    # The conversation, on A, owned by A — the ordinary shape of a session a move
    # is asked about.
    directory = root_a / "sessions" / SESSION
    directory.mkdir(parents=True, exist_ok=True)
    transcript = "".join(
        json.dumps({"id": f"e{index}", "type": "user", "content": f"line {index}"}) + "\n"
        for index in range(40)
    )
    (directory / "transcript.jsonl").write_text(transcript, encoding="utf-8")
    (directory / "title.json").write_text(json.dumps({"title": "mesh design"}), encoding="utf-8")
    mark_store(root_a / "sessions")
    write_stamp(
        root_a,
        MeshStamp(
            session_id=SESSION,
            network_id=record.network_id,
            home_device=server_a.identity.device_id,
            placement=SessionPlacement(
                mode="local", home_device=server_a.identity.device_id, stamp_revision=1
            ),
        ),
    )
    (setup / "expect.json").write_text(
        json.dumps(
            {
                "network_id": record.network_id,
                "epoch": record.epoch,
                "session_id": SESSION,
                "digest": "sha256:" + hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
                "owner_device": server_a.identity.device_id,
                "taker_device": server_b.identity.device_id,
                "root_a": str(root_a),
                "root_b": str(root_b),
            }
        ),
        encoding="utf-8",
    )
    ready_marker.write_text("ready", encoding="utf-8")

    def _crash_here() -> None:
        """Announce the point to the parent, then hang until it is killed."""
        crash_marker.write_text(window, encoding="utf-8")
        while True:
            time.sleep(0.2)

    # --- the crash point, one patch per window ---------------------------
    from local_operator.network import projection, sync

    if window == "before_ready":
        real_sync = sync.sync_from

        def sync_then_die(*args: Any, **kwargs: Any) -> Any:
            result = real_sync(*args, **kwargs)
            _crash_here()  # the copy is complete; `ready` was never sent
            return result

        sync.sync_from = sync_then_die  # type: ignore[assignment]
    elif window == "after_handing_off":
        real_tombstone = projection.write_tombstone

        def tombstone_then_die(*args: Any, **kwargs: Any) -> Any:
            # The journal already says `handing-off`; the tombstone and the delete
            # never happen. This is the window where a rollback is forbidden.
            _crash_here()
            return real_tombstone(*args, **kwargs)

        projection.write_tombstone = tombstone_then_die  # type: ignore[assignment]
    else:
        real_promote = mobility._promote

        def promote_then_die(*args: Any, **kwargs: Any) -> Any:
            _crash_here()  # the owner has committed; this device never adopts
            return real_promote(*args, **kwargs)

        mobility._promote = promote_then_die  # type: ignore[assignment]

    result = mobility.request_move(SESSION, to="local", root=root_b)
    # Reached only if the crash point was never hit, which is a bug in this test:
    # report it through the exit status instead of hanging.
    (setup / "no-crash.json").write_text(json.dumps(result, default=str), encoding="utf-8")
    return 3


def _kill_at(window: str, setup: Path) -> None:
    """Run the crashing device as a SUBPROCESS, then SIGKILL it at ``window``.

    A SUBPROCESS AND NOT A FORK. ``os.fork`` in a pytest process that already
    holds threads (xdist, the relays of a previous test) is a documented hazard —
    Python 3.12 warns about it at every call, and the failure mode is a child that
    deadlocks on a lock another thread held at fork time, which would present here
    as "the child never reached its crash point". ``subprocess`` gives a process
    whose only threads are the ones it starts itself, and the kill is exactly as
    real: SIGKILL to the pid.
    """
    repo = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(repo), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    child = subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "--child", window, str(setup)],
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    crash_marker = setup / "child-crashed"
    try:
        deadline = time.monotonic() + REACH_TIMEOUT_S
        while time.monotonic() < deadline:
            if crash_marker.is_file() and crash_marker.read_text(encoding="utf-8") == window:
                break
            if (setup / "no-crash.json").is_file():
                raise AssertionError(
                    f"the move finished without reaching the {window} crash point: "
                    + (setup / "no-crash.json").read_text(encoding="utf-8")
                )
            if child.poll() is not None:
                output = child.stdout.read() if child.stdout is not None else ""
                raise AssertionError(
                    f"the child exited (rc {child.returncode}) before the {window} crash "
                    f"point:\n{output[-4000:]}"
                )
            time.sleep(0.05)
        else:
            raise AssertionError(f"the child never reached the {window} crash point")
        # THE KILL. SIGKILL, so nothing the child holds is flushed or tidied: what
        # the recovery below reads is only what was already durable on disk.
        child.kill()
        child.wait(timeout=30)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=30)


def _holders(root_a: Path, root_b: Path, session_id: str) -> list[Path]:
    """Every device that holds ``session_id`` WITH a transcript, after recovery."""
    found = []
    for root in (root_a, root_b):
        directory = root / "sessions" / session_id
        if (directory / "transcript.jsonl").is_file():
            found.append(directory)
    return found


def _digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _restart(setup: Path, expect: dict[str, Any]) -> tuple[Any, Any, str, int]:
    """Fresh relays over the SAME roots — the reboot after the kill.

    The member row still advertises the dead child's port, so the link is
    re-established by dialling the new address directly, exactly as a device whose
    peer moved address does. A relay object holds no state a recovery needs: that
    is the property these tests exist to check.
    """
    from local_operator.network import relay

    root_a = Path(expect["root_a"])
    root_b = Path(expect["root_b"])
    server_a = relay.RelayServer(
        root=root_a, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    server_b = relay.RelayServer(
        root=root_b, settings=relay.NetworkSettings(port=0, listen_address="127.0.0.1")
    )
    host, port = server_a.bind()
    server_a.bind_control()
    server_b.bind_control()
    server_a.start()
    server_b.start()
    link, reason = server_b.dial(
        str(expect["network_id"]), host=f"{host}:{port}", epoch=int(expect["epoch"])
    )
    assert link is not None, f"the restarted devices could not reach each other: {reason}"
    del setup
    return server_a, server_b, host, port


@pytest.mark.parametrize("window", WINDOWS)
def test_a_kill_at_each_point_leaves_exactly_one_holder(window: str, tmp_path: Path) -> None:
    from local_operator.network import mobility
    from local_operator.network.projection import read_tombstones
    from local_operator.session.placement import read_handoff_journal

    setup = tmp_path / "device"
    setup.mkdir()
    _kill_at(window, setup)
    expect = json.loads((setup / "expect.json").read_text(encoding="utf-8"))
    root_a = Path(expect["root_a"])
    root_b = Path(expect["root_b"])
    session_id = str(expect["session_id"])
    original = str(expect["digest"])

    # --- what the kill left behind, before any recovery -------------------
    staged = root_b / "network" / "staging" / session_id
    journal_a = read_handoff_journal(root_a)
    tombstone_a = read_tombstones(root_a)
    if window == "before_ready":
        assert journal_a[session_id]["phase"] == "prepared"
        assert not tombstone_a, "nothing was committed, so there must be no tombstone"
        assert (root_a / "sessions" / session_id).exists()
        assert staged.is_dir() and not (staged / "ready.json").exists()
    elif window == "after_handing_off":
        assert journal_a[session_id]["phase"] == "handing-off"
        assert not tombstone_a, "the crash landed between the journal and the tombstone"
        assert (root_a / "sessions" / session_id).exists()
        assert (staged / "ready.json").is_file()
    else:
        assert session_id in tombstone_a
        assert not (root_a / "sessions" / session_id).exists()
        assert (staged / "ready.json").is_file()
    # NO WINDOW EVER LEAVES THE DESTINATION HOLDING A SESSION: the promote is the
    # last thing a move does, and every one of these kills is before it.
    assert not (root_b / "sessions" / session_id).exists()

    # --- recovery: restart both devices and reconcile ---------------------
    server_a, server_b, _host, _port = _restart(setup, expect)
    try:
        # The OWNER first: its half needs no peer (the tombstone and the delete are
        # local), and in the `handing-off` window it is what completes the handoff.
        mobility.reconcile(root_a, server=server_a, own_instance="recovery-instance-a")
        if window == "before_ready":
            assert read_handoff_journal(root_a) == {}, "a `prepared` handoff must roll back"
            assert not read_tombstones(root_a)
        else:
            assert str(read_tombstones(root_a)[session_id]["device_id"]) == str(
                expect["taker_device"]
            )
            assert not (root_a / "sessions" / session_id).exists()
            assert read_handoff_journal(root_a) == {}

        # The DESTINATION then reconciles against the owner's own record.
        mobility.reconcile(root_b, server=server_b, own_instance="recovery-instance-b")
    finally:
        server_a.stop()
        server_b.stop()

    # --- THE INVARIANT: exactly one holder, holding the whole conversation --
    holders = _holders(root_a, root_b, session_id)
    assert len(holders) == 1, [str(path) for path in holders]
    assert _digest(holders[0] / "transcript.jsonl") == original
    if window == "before_ready":
        # THE OWNER KEEPS IT: the copy never proved itself, so the move rolled back
        # on both sides and the conversation is where it started.
        assert holders[0] == root_a / "sessions" / session_id
    else:
        # THE DESTINATION ADOPTED IT, and the owner has nothing but the tombstone —
        # which is the design's whole claim about why the tombstone exists.
        assert holders[0] == root_b / "sessions" / session_id
        assert not (root_a / "sessions" / session_id).exists()
    # And the move's own inert artefacts are outside the session store.
    assert not (root_b / "sessions" / session_id / "ready.json").exists()


if __name__ == "__main__":  # pragma: no cover — the crashing child, run by _kill_at
    # ``python test_mobility_crash.py --child <window> <setup>``: the entry point the
    # parent's ``_kill_at`` uses. It reaches a crash point and then SLEEPS until it
    # is killed — it never returns, and if it does something went wrong.
    if "--child" not in sys.argv:
        raise SystemExit("this file is a test module; run it under pytest")
    _window = sys.argv[sys.argv.index("--child") + 1]
    _setup = Path(sys.argv[sys.argv.index("--child") + 2])
    _ready = _setup / "child-ready"
    _crashed = _setup / "child-crashed"
    while True:
        try:
            raise SystemExit(_child_body(_setup, _window, _ready, _crashed))
        except SystemExit:
            raise
        except BaseException:  # noqa: BLE001 — reported to the parent as an exit code
            import traceback

            traceback.print_exc()
            raise SystemExit(2) from None


def test_the_staging_gc_exempts_a_copy_that_is_the_only_one_left(tmp_path: Path) -> None:
    """§7's GC rule: a verified copy is not swept just because it is old.

    Those bytes are the last copy of a conversation whose owner may already have
    deleted its directory, so the age cap yields to the marker — reported to the
    user rather than decided by a sweep (design §13 Q6).
    """
    from local_operator.network import mobility
    from local_operator.network.sync import staging_dir

    root = tmp_path / "device"
    (root / "sessions").mkdir(parents=True)
    kept = staging_dir(root, "kept-session")
    kept.mkdir(parents=True)
    (kept / "transcript.jsonl").write_text("", encoding="utf-8")
    (kept / "ready.json").write_text(json.dumps({"promoted": False}), encoding="utf-8")
    abandoned = staging_dir(root, "abandoned-session")
    abandoned.mkdir(parents=True)
    (abandoned / "transcript.jsonl").write_text("{}", encoding="utf-8")
    old = time.time() - mobility.STAGING_MAX_AGE_S - 60
    os.utime(kept, (old, old))
    os.utime(abandoned, (old, old))

    swept = mobility.sweep_staging(root)

    assert swept == ["abandoned-session"]
    assert kept.is_dir()
