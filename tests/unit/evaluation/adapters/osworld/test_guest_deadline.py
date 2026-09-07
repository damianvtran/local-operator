"""The guest's command deadline and our socket deadline are one relationship.

Three deadlines govern one guest action: our HTTP socket read timeout
(``GUEST_COMMAND_TIMEOUT_S``), the guest's own ``subprocess.run`` deadline
inside its ``/execute`` route, and the parent step timeout. Nothing related the
first two. The adapter posted ``{"command", "shell"}`` and never sent
``timeout``, so the guest fell back to its route default of 120 s while our
socket waited 90 s.

That ordering resolves the WRONG way. With the guest's deadline outside ours, a
command slower than our socket expires on OUR side while it is still running in
the guest: ``requests`` raises, and a transport error cannot tell "never
started" from "half-applied". ``execute`` can then only report an unknown
outcome, and the no-retry policy — correct to refuse replaying a
possibly-committed batch — ends the episode.

The fix makes the guest's deadline a DERIVED function of ours, strictly inside
it, so the guest reaches its deadline first and ANSWERS with a real response
instead of leaving us to time out a still-running command. The tests below pin
that ordering, pin the constants so neither can drift alone (the anti-drift
property ``test_type_deadline.py`` establishes for the type bound), and prove
the wire body actually carries the derived value — an invariant that holds in
the module but never reaches the guest would fix nothing.

The diagnostic tests cover a separate, deliberately NARROW improvement: a
signal-killed guest command reports as a negative returncode, so ``exit -15``
is really "SIGTERM killed it" in a misleading costume. The message now names
the signal. It deliberately does NOT assert a CAUSE — many things kill a
process, this layer cannot distinguish them, and guessing would encode a
theory into every future incident report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from threading import Thread
from typing import Any

import pytest
from lop_osworld_v2_adapter.providers import aws as aws_mod
from lop_osworld_v2_adapter.providers.aws import (
    AwsProvider,
    GuestExecutionError,
    _exit_description,
)
from lop_osworld_v2_adapter.providers.base import (
    GUEST_COMMAND_TIMEOUT_S,
    GUEST_EXECUTE_TIMEOUT_S,
    GUEST_TRANSPORT_MARGIN_S,
    guest_deadline_for,
)

from tests.unit.evaluation.adapters.osworld.test_aws_provider import (
    CREDS,
    REGION,
    _FakeEnv,
    _Stubs,
)

# An autouse fixture is not inherited across modules, so the AWS scrub is
# imported by name and re-registered here. Reused rather than re-written: a
# second way of making these tests hermetic is how one of them later drifts.
from tests.unit.evaluation.adapters.osworld.test_aws_provider import (  # noqa: F401  isort:skip
    _hermetic_aws,
)

# Socket deadlines a caller can realistically pass. The default is the common
# case; ``guest_disk`` shrinks its own toward zero as its budget drains, which
# is why the squeezed tail is covered rather than assumed away.
SOCKET_DEADLINES = (90.0, 60.0, 30.0, 12.0, 10.0, 9.0, 5.0, 1.0, 0.5, 0.01)


# ----------------------------------------------------------------------------
# The arithmetic: the two deadlines cannot drift apart
# ----------------------------------------------------------------------------


def test_the_guest_deadline_is_strictly_inside_the_socket_deadline() -> None:
    """The ordering invariant the whole fix rests on.

    Not "close to" and not "at most": STRICTLY less. A guest deadline equal to
    ours is still a race we can lose, and losing it returns the ambiguous
    unknown-outcome state that kills episodes.
    """

    assert GUEST_EXECUTE_TIMEOUT_S < GUEST_COMMAND_TIMEOUT_S


def test_the_guest_deadline_is_derived_from_the_socket_deadline_and_the_margin() -> None:
    """Recompute the derivation independently of the module that publishes it.

    This is the test that prevents recurrence: it checks the two numbers are
    still expressions of one another, which is the property whose absence let a
    120 s guest default sit outside our 90 s socket.
    """

    assert GUEST_EXECUTE_TIMEOUT_S == GUEST_COMMAND_TIMEOUT_S - GUEST_TRANSPORT_MARGIN_S
    assert GUEST_EXECUTE_TIMEOUT_S == guest_deadline_for(GUEST_COMMAND_TIMEOUT_S)


@pytest.mark.parametrize("socket_timeout", SOCKET_DEADLINES)
def test_every_caller_socket_deadline_yields_a_positive_inner_deadline(
    socket_timeout: float,
) -> None:
    """The invariant must hold for EVERY caller, not just the default.

    ``guest_disk`` passes ``min(COMMAND_TIMEOUT_S, remaining)``, so the socket
    deadline shrinks toward zero over a preparation pass. Two failure modes are
    excluded here: a non-positive deadline (which the guest reads as "kill this
    immediately") and a deadline at or beyond the socket deadline (which
    restores the ambiguous ordering).
    """

    derived = guest_deadline_for(socket_timeout)
    assert derived > 0.0
    assert derived < socket_timeout


def test_the_deadline_constants_are_pinned_so_either_moving_is_deliberate() -> None:
    """Pin both values, so changing EITHER lands here first.

    The derivation above is invariant under a coordinated change, so it alone
    would let someone halve the socket timeout and never notice the guest
    deadline halving with it. Changing a value here is legitimate — update it,
    and say in the commit which deadline moved and why the ordering still holds.
    """

    assert GUEST_COMMAND_TIMEOUT_S == 90.0
    assert GUEST_TRANSPORT_MARGIN_S == 10.0
    assert GUEST_EXECUTE_TIMEOUT_S == 80.0


def test_the_margin_leaves_room_for_the_measured_per_command_overhead() -> None:
    """The margin pays for transport, so it must exceed transport's real cost.

    3.6 s of fixed per-command overhead was measured over 21 real batches (see
    ``GUEST_TYPE_DEADLINE_FRACTION``). A margin below that would let the guest's
    error response miss our socket, losing the determinism the ordering buys.
    """

    measured_overhead_s = 3.6
    assert GUEST_TRANSPORT_MARGIN_S > measured_overhead_s


# ----------------------------------------------------------------------------
# The wire: the derived deadline actually reaches the guest
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_execute_body_carries_the_derived_timeout() -> None:
    """An invariant that never reaches the guest fixes nothing.

    The defect was not a wrong number, it was an ABSENT field: with no
    ``timeout`` in the body the guest applies its own 120 s default no matter
    what our constants say. This asserts the field is present and equal to the
    derivation, so deleting it fails here rather than in a live run.
    """

    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        env = _FakeEnv()
        env.pkgs_prefix = "{command}"
        provider._env = env
        provider._public_ip = "127.0.0.1"
        await provider.execute(["pyautogui.click(1, 2)"])

    assert len(stubs.guest_posts) == 1
    posted = stubs.guest_posts[0]
    assert posted["timeout"] == GUEST_EXECUTE_TIMEOUT_S
    assert posted["timeout"] < GUEST_COMMAND_TIMEOUT_S
    # The rest of the body shape is unchanged: ``shell`` false keeps the guest
    # exec'ing argv directly rather than through a shell.
    assert posted["shell"] is False


def test_guest_disk_commands_also_carry_a_coherent_deadline() -> None:
    """The disk-preparation path shares the transport and must share the rule.

    It passes its own shrinking socket deadline, so this is where a fix applied
    only to the canonical constant would silently not apply.
    """

    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        provider._public_ip = "127.0.0.1"
        provider._run_guest_command(["bash", "-c", "df -B1 /"], 45.0)

    posted = stubs.guest_posts[-1]
    assert posted["timeout"] == guest_deadline_for(45.0)
    assert posted["timeout"] < 45.0


# ----------------------------------------------------------------------------
# The diagnostic: a signal death is legible, and leaks nothing
# ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("returncode", "expected"),
    [
        (1, "exit 1"),
        (7, "exit 7"),
        (127, "exit 127"),
        (-15, "terminated by signal SIGTERM"),
        (-9, "terminated by signal SIGKILL"),
        (-2, "terminated by signal SIGINT"),
    ],
)
def test_a_signal_death_is_named_and_an_ordinary_exit_is_unchanged(
    returncode: int, expected: str
) -> None:
    """``exit -15`` is a signal wearing a misleading costume.

    A reader who does not know ``subprocess``'s negative-returncode convention
    reads -15 as an exit code and hunts for a program that returned one. Naming
    the signal removes that research step. Ordinary non-zero exits keep their
    existing wording so nothing that reads them has to change.
    """

    assert _exit_description(returncode) == expected


def test_an_unrecognised_signal_number_still_reads_as_a_signal() -> None:
    """Signal numbers are platform-specific; an unknown one is still not an exit code.

    Falling back to ``exit -64`` here would reintroduce exactly the confusion
    this function exists to remove.
    """

    assert _exit_description(-64) == "terminated by signal 64"


def test_the_diagnostic_does_not_assert_why_the_process_died() -> None:
    """State WHAT happened, never WHY — this layer cannot know.

    A SIGTERM can come from the OS, a supervisor, another process, or the
    command's own actions. An earlier investigation was sent down the wrong path
    by a confidently misattributed cause, so the message must not name one.
    ``timeout`` in particular is a cause this layer cannot substantiate: the
    guest's own timeout does not even surface here (its route returns HTTP 500,
    which fails the transport branch instead).
    """

    described = " ".join(_exit_description(rc) for rc in (-15, -9, -2, -64, 1, 7))
    for causal_claim in ("timeout", "timed out", "deadline", "killed by", "because", "guest-side"):
        assert causal_claim not in described.lower()


@pytest.mark.asyncio
async def test_a_signal_killed_action_reports_the_signal_without_leaking_the_command() -> None:
    """The end-to-end path: signal name in, command text out.

    The existing code is deliberately careful that neither transport exceptions
    nor guest stderr reach the message, because both can echo the command,
    credentials or on-screen text. Improving the diagnostic must not widen that
    surface, so this asserts the new wording AND the old silence together.
    """

    secret = "s3cret-passphrase-do-not-leak"
    with _Stubs() as stubs:
        stubs.guest_default = {
            "returncode": -15,
            "output": f"stdout containing {secret}",
            "error": f"stderr containing {secret}",
        }
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        env = _FakeEnv()
        env.pkgs_prefix = "{command}"
        provider._env = env
        provider._public_ip = "127.0.0.1"
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute([f"pyautogui.typewrite({secret!r})"])

    message = str(raised.value)
    assert "terminated by signal SIGTERM" in message
    # The unrecoverable-state wording is load-bearing and must survive.
    assert "input may be partial" in message
    assert "batch not retried" in message
    # Nothing from the command, its stdout or its stderr appears.
    assert secret not in message
    assert "typewrite" not in message
    assert "stdout containing" not in message
    assert "stderr containing" not in message


@pytest.mark.asyncio
async def test_no_error_path_leaks_command_text() -> None:
    """Both failure branches stay silent about the payload.

    The transport branch (unknown outcome) and the non-zero-exit branch (partial
    input) are the only two ways ``execute`` raises, and neither may echo what it
    was asked to run.
    """

    secret = "another-s3cret-value"
    statement = f"pyautogui.typewrite({secret!r})"

    for guest_reply in (
        RuntimeError(f"connection refused while posting {secret}"),
        {"returncode": 3, "output": secret, "error": secret},
        {"returncode": -9, "output": secret, "error": secret},
    ):
        with _Stubs() as stubs:
            stubs.guest_default = guest_reply
            provider = AwsProvider(
                CREDS,
                region=REGION,
                lease_ref="lop-ttl-x",
                clients=stubs.clients,
                sleep=stubs.sleep,
            )
            env = _FakeEnv()
            env.pkgs_prefix = "{command}"
            provider._env = env
            provider._public_ip = "127.0.0.1"
            with pytest.raises(GuestExecutionError) as raised:
                await provider.execute([statement])

        message = str(raised.value)
        assert secret not in message
        assert "typewrite" not in message


# ----------------------------------------------------------------------------
# The real control boundary: a guest that honours the deadline we send
# ----------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_real_guest_honours_the_posted_deadline_and_answers_in_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Exercise the property end-to-end against upstream's own route logic.

    This is the point of the fix, so it is proven over real HTTP rather than
    asserted from constants. The handler mirrors ``execute_command``
    (osworld-server @ a3cc3f0): it reads ``timeout`` from the BODY, passes it to
    ``subprocess.run``, and on ``TimeoutExpired`` returns HTTP 500 exactly as
    upstream does.

    A command that outruns the guest's deadline must therefore come back as a
    definite answer WITHIN our socket deadline — the ambiguity the fix removes.
    The numbers are scaled down so the test costs a second, but the ordering
    (guest strictly inside socket) is the deployed one.
    """

    import subprocess
    import sys
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    bodies_seen: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies_seen.append(payload)
            # Upstream reads the deadline from the body, defaulting to 120.
            # Sending it is the entire fix; the default is the defect.
            timeout = payload.get("timeout", 120)
            try:
                completed = subprocess.run(
                    [sys.executable, *payload["command"][1:]],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                body = json.dumps(
                    {
                        "returncode": completed.returncode,
                        "output": completed.stdout,
                        "error": completed.stderr,
                    }
                ).encode()
                status = 200
            except subprocess.TimeoutExpired:
                # Upstream's own shape: HTTP 500 and no returncode.
                body = json.dumps({"status": "error", "message": "timed out"}).encode()
                status = 500
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(aws_mod, "GUEST_PORT", server.server_port)
    # A socket deadline whose derived inner deadline is short enough to fire
    # quickly, while keeping the deployed strict ordering.
    socket_timeout = GUEST_TRANSPORT_MARGIN_S + 1.0
    inner = guest_deadline_for(socket_timeout)
    assert 0.0 < inner < socket_timeout
    try:
        with _Stubs() as stubs:
            stubs.clients.http_post_json = aws_mod.build_clients(CREDS, REGION).http_post_json
            provider = AwsProvider(
                CREDS,
                region=REGION,
                lease_ref="lop-ttl-x",
                clients=stubs.clients,
                sleep=stubs.sleep,
            )
            provider._public_ip = "127.0.0.1"

            # A command that finishes inside the guest deadline returns normally.
            quick = provider._run_guest_command(["python", "-c", "print('done')"], socket_timeout)
            assert quick.returncode == 0

            # A command that outruns the guest deadline is ANSWERED by the guest
            # (HTTP 500) rather than expiring our socket. ``raise_for_status``
            # turns that into the transport branch, which is the honest
            # unknown-outcome report — and crucially it arrives promptly.
            sleeper = ["python", "-c", f"import time; time.sleep({socket_timeout * 4:.1f})"]
            with pytest.raises(Exception) as raised:
                provider._run_guest_command(sleeper, socket_timeout)
            # It was the guest's answer, not our socket giving up.
            assert "500" in str(raised.value) or "Server Error" in str(raised.value)

        assert bodies_seen[-1]["timeout"] == inner
    finally:
        server.shutdown()
        thread.join(timeout=5)
        server.server_close()


def test_the_adapter_never_relies_on_the_guest_route_default() -> None:
    """Guard the specific regression: a body without ``timeout``.

    The guest's default is 120 s, outside our 90 s socket. If a refactor ever
    drops the field, the module constants would still agree with each other and
    every arithmetic test above would still pass — only this one fails.
    """

    source = Path(aws_mod.__file__).read_text()
    posted_body = re.search(r"\{\s*\"command\": list\(command\).*?\}", source, re.DOTALL)
    assert posted_body is not None, "the /execute body literal moved; update this guard"
    assert "timeout" in posted_body.group(0)
    assert "guest_deadline_for" in posted_body.group(0)
