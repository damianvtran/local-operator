"""Real six-relay loopback measurement; counts each sealed record once.
Requests and replies are attributed by (cipher, inbound req) in both directions;
an outbound request with the same numeric req cannot overwrite an answer's label.

Run from the worktree with its dev venv, isolated HOME/config and PYTHONPATH=.
Default: 65-second settlement, then 3 x 30-second byte windows. --legacy removes
only the push capability (same-code control); --convergence measures five renames
per topology, including a real four-second membership timeout and one old peer
transcript. --repair-window also observes the full five-minute repair period.
This empty-session mesh intentionally does not synthesize placement/session work.
"""

import argparse
import collections
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import sys
import threading
import time

import pytest

from local_operator.network import relay, store, wire
from tests.unit.network.test_membership_convergence import (
    HUB,
    Device,
    _init_network,
    _join,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--convergence", action="store_true")
parser.add_argument("--repair-window", action="store_true")
parser.add_argument(
    "--legacy", action="store_true", help="Disable the push capability on established links"
)
args = parser.parse_args()
ROOT = pathlib.Path(os.environ["HOME"]) / "rig"
if not os.environ.get("LOCAL_OPERATOR_CONFIG_DIR"):
    raise SystemExit("Use isolated HOME and LOCAL_OPERATOR_CONFIG_DIR for this benchmark")
ROOT.mkdir(parents=True, exist_ok=True)
lock = threading.Lock()
requests = {}
rows = []
seal = wire.LinkCrypto.seal
open_record = wire.LinkCrypto.open


def remember(cipher, frame):
    op = frame.get("op", "")
    req = frame.get("req")
    if op not in ("ack", "refuse", "error"):
        requests[(id(cipher), req)] = op


def opened(self, payload):
    frame = open_record(self, payload)
    with lock:
        remember(self, frame)
    return frame


def sealed(self, frame):
    payload = seal(self, frame)
    with lock:
        op = frame.get("op", "")
        answer = op in ("ack", "refuse", "error")
        attributed = requests.get((id(self), frame.get("req")), op) if answer else op
        rows.append((time.monotonic(), attributed, "answer" if answer else "ask", len(payload)))
    return payload


wire.LinkCrypto.seal = sealed
wire.LinkCrypto.open = opened
built = []
mp = pytest.MonkeyPatch()
try:
    for i in range(6):
        built.append(Device(ROOT, f"rig-{i}", mode=HUB))
    record = _init_network(built[0].server)
    for device in built[1:]:
        result = _join(device, inviter=built[0], monkeypatch=mp)
        assert result.get("device_id") == device.device_id, result
    for i, a in enumerate(built):
        for b in built[i + 1 :]:
            link, reason = a.server.dial(record.network_id, host=f"{b.host}:{b.port}", epoch=1)
            assert link is not None, reason
    print(
        "PREFLIGHT",
        json.dumps(
            {
                "source": relay.__file__,
                "source_sha256": hashlib.sha256(
                    pathlib.Path(relay.__file__).read_bytes()
                ).hexdigest(),
                "members": len(built),
                "links": [sum(link.alive for link in d.server.links.values()) for d in built],
            }
        ),
        flush=True,
    )

    def rename_trials(label, count=5, wedge=False):
        samples = []
        wedge_seen = threading.Event()
        wedged = built[1]
        dispatch = wedged.server.dispatch
        request = relay.PeerLink.request
        timeout_samples = []
        if wedge:

            def timed_request(self, frame, *, timeout=None):
                started = time.monotonic()
                reply = request(self, frame, timeout=timeout)
                if (
                    self.device_id == wedged.device_id
                    and frame.get("op") == "net_member_list"
                    and reply is None
                ):
                    timeout_samples.append(round(time.monotonic() - started, 3))
                return reply

            relay.PeerLink.request = timed_request

            def stalled(link, frame):
                if frame.get("op") == "net_member_list" and "changed_members" not in frame:
                    wedge_seen.set()
                    # Intercept at dispatch, NOT the handler: a handler returning
                    # None is normalized into an ack and is not a wedged peer.
                    return None
                return dispatch(link, frame)

            wedged.server.dispatch = stalled
        for trial in range(count):
            for device in built:
                for link in device.server.links.values():
                    link.member_pulled_at = time.time()
                    link.member_push_unverified = False
                    link.member_pull_interval_s = (
                        relay.MEMBERSHIP_PUSH_PULL_INTERVAL_S
                        if device.server._member_push_enabled(link)
                        else relay.MEMBERSHIP_PULL_MAX_INTERVAL_S
                    )
                    if wedge and link.device_id == wedged.device_id:
                        link.member_pulled_at -= relay.MEMBERSHIP_PUSH_PULL_INTERVAL_S + 1
                if wedge:
                    device.server._membership_wake.set()
            if wedge:
                assert wedge_seen.wait(10), "no request entered wedged peer"
                wedge_seen.clear()
            name = f"{label}-{trial}"
            started = time.monotonic()
            with store.mutate(record.network_id, built[0].root) as copy:
                own = copy.member(built[0].device_id)
                assert own is not None
                own.name = name
                store.save(copy, built[0].root)
            deadline = started + 40
            while time.monotonic() < deadline:
                if all(
                    (member := store.load(record.network_id, d.root).member(built[0].device_id))
                    is not None
                    and member.name == name
                    for d in built[1:]
                ):
                    break
                time.sleep(0.02)
            else:
                raise AssertionError(f"{label} did not converge: {name}")
            samples.append(round(time.monotonic() - started, 3))
            print("TRIAL", label, samples[-1], flush=True)
        print(
            "CONVERGENCE",
            label,
            json.dumps(
                {
                    "n": len(samples),
                    "samples_s": samples,
                    "median_s": statistics.median(samples),
                    "max_s": max(samples),
                }
            ),
            flush=True,
        )
        if wedge:
            # Give the final in-flight request its real four-second timeout so
            # the report proves this was a blocked pull, not just a silent mock.
            time.sleep(relay.MEMBERSHIP_PULL_TIMEOUT_S + 0.25)
            wedged.server.dispatch = dispatch
            relay.PeerLink.request = request
            assert len(timeout_samples) >= 5, "fewer than five real pulls timed out"
            print("WEDGE_TIMEOUTS", json.dumps(timeout_samples), flush=True)

    if args.legacy:
        for device in built:
            for link in device.server.links.values():
                link.capabilities = frozenset(link.capabilities - {wire.MEMBER_LIST_PUSH_V1})
                link.local_capabilities = frozenset(
                    link.local_capabilities - {wire.MEMBER_LIST_PUSH_V1}
                )
    if args.convergence:
        # Negative control: this very build, with the new transcript capability
        # absent, must retain exactly the legacy 30-second convergence clock.
        for device in built:
            for link in device.server.links.values():
                link.capabilities = frozenset(link.capabilities - {wire.MEMBER_LIST_PUSH_V1})
                link.local_capabilities = frozenset(
                    link.local_capabilities - {wire.MEMBER_LIST_PUSH_V1}
                )
        rename_trials("legacy-control")
        for device in built:
            for link in device.server.links.values():
                link.capabilities = frozenset(link.capabilities | {wire.MEMBER_LIST_PUSH_V1})
                link.local_capabilities = frozenset(
                    link.local_capabilities | {wire.MEMBER_LIST_PUSH_V1}
                )
        rename_trials("push-healthy")
        # Exercise the actual shipped CLI/control socket, not a hand-shaped status
        # payload. A fresh synthetic config keeps this away from the live service.
        cli_env = dict(os.environ, LOCAL_OPERATOR_CONFIG_DIR=str(built[2].root))
        for verb in ("status", "peers"):
            command = [
                str(pathlib.Path(sys.executable).with_name("lop")),
                "network",
                verb,
                "--json",
            ]
            result = subprocess.run(
                command, env=cli_env, capture_output=True, text=True, timeout=45, check=True
            )
            print("CLI", verb, result.stdout.strip(), flush=True)
        rename_trials("push-wedged", wedge=True)
        # Model one old transcript in an otherwise new mesh, in BOTH directions.
        old = built[1]
        for device in built:
            for link in device.server.links.values():
                if device is old or link.device_id == old.device_id:
                    link.capabilities = frozenset(link.capabilities - {wire.MEMBER_LIST_PUSH_V1})
                    link.local_capabilities = frozenset(
                        link.local_capabilities - {wire.MEMBER_LIST_PUSH_V1}
                    )
        rename_trials("mixed-one-legacy")
        started = time.monotonic()
        removed = built[-1].device_id
        receipt = built[0].server._ctl_member_rm(
            {"network": record.network_id, "device_id": removed}
        )
        deadline = started + 30
        while time.monotonic() < deadline:
            if all(
                (member := store.load(record.network_id, device.root).member(removed)) is not None
                and not member.active
                for device in built[:-1]
            ):
                break
            time.sleep(0.02)
        else:
            raise AssertionError("epoch removal did not reach all five survivors")
        print(
            "REMOVAL",
            json.dumps(
                {
                    "receipt": receipt,
                    "all_survivors_removed": True,
                    "elapsed_s": round(time.monotonic() - started, 3),
                }
            ),
            flush=True,
        )
    else:
        time.sleep(65)
        lengths = [30, 30, 30] + ([300] if args.repair_window else [])
        for window, seconds in enumerate(lengths):
            start = time.monotonic()
            time.sleep(seconds)
            end = time.monotonic()
            table = collections.defaultdict(
                lambda: {"ask_bytes": 0, "answer_bytes": 0, "asks": 0, "answers": 0}
            )
            with lock:
                for stamp, op, side, size in rows:
                    if start <= stamp < end:
                        table[op][f"{side}_bytes"] += size
                        table[op]["asks" if side == "ask" else "answers"] += 1
            print(
                "WINDOW",
                window,
                json.dumps(dict(sorted(table.items())), sort_keys=True),
                "duration_s=",
                seconds,
                flush=True,
            )
finally:
    mp.undo()
    for device in built:
        device.stop()
