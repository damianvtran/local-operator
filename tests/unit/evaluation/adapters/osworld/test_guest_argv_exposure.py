"""Our guest-side process must not carry the agent's typed text in its argv.

Episode ep-0ce67ac2d3a1 died with ``GuestExecutionError: guest action 1 failed
(exit -15)``. An earlier step had legitimately started ``nohup ffmpeg -y -f
x11grab ...`` to record the screen, and the model was correctly cleaning up
after itself:

    pkill -f "ffmpeg -y -f x11grab"; sleep 0.5; xdotool getactivewindow ...

Nothing about that is wrong. The defect was ours: the statement was handed to
the guest as a literal ``python -c <source>`` argv, and argv is a PUBLIC
channel -- ``/proc/<pid>/cmdline`` is precisely what ``pkill -f``, ``pgrep -f``
and ``ps | grep`` match against. The pattern was a substring of the command
line of the process typing it, so ``pkill -f`` matched that process and
SIGTERMed it. Exit -15 is SIGTERM; the step ran ~5.9 s against a 90 s deadline,
which rules out any timeout explanation.

These tests therefore assert the property, not a spelling: the agent's text is
absent from our argv, and a real pattern matcher run against a real live
process no longer selects it. They deliberately do NOT assert that we detect,
sanitise or block ``pkill``/``killall``/``pgrep`` -- the agent's usage is
legitimate and routine, and screening its text would be both incomplete and
not ours to impose.
"""

from __future__ import annotations

import base64
import shutil
import subprocess
import sys
import time
from typing import Any

import pytest
from lop_osworld_v2_adapter import actions
from lop_osworld_v2_adapter.providers.aws import AwsProvider, GuestExecutionError

from local_operator import computer_input
from local_operator.computer_input import python_source_argv
from local_operator.evaluation.protocol import ClickAction, KeyAction, TypeAction
from tests.unit.evaluation.adapters.osworld.test_actions import _geo
from tests.unit.evaluation.adapters.osworld.test_aws_provider import (  # noqa: F401
    CREDS,
    REGION,
    _FakeEnv,
    _hermetic_aws,
    _Stubs,
)

# The exact text from the failing episode's last model_response, and the
# pattern inside it that ``pkill -f`` was given. Real evidence, not an invented
# payload: a fix that does not clear THIS string has not fixed the incident.
FATAL_TYPED_TEXT = (
    'pkill -f "ffmpeg -y -f x11grab"; sleep 0.5; xdotool getactivewindow windowclose; '
    "sleep 1; xdotool search --onlyvisible --class chrome windowactivate; sleep 0.5; "
    "xdotool key F11\n"
)
FATAL_PATTERN = "ffmpeg -y -f x11grab"

# Upstream's own controller prefix (python.py:37-45, condensed to the part that
# matters here): whatever we wrap the statement in must not reintroduce the text.
PKGS_PREFIX = "import pyautogui; import time; import platform; {command}"


def _guest_argv_for(typed_text: str) -> list[str]:
    """The argv our provider would hand the guest for one ``type`` action."""
    statement = actions.compile_action(
        TypeAction(observation_id="o", text=typed_text), _geo()
    )
    assert statement is not None
    return python_source_argv(PKGS_PREFIX.format(command=statement))


# ----------------------------------------------------------------------------
# The exposure itself


def test_the_typed_text_is_absent_from_the_guest_processes_argv() -> None:
    argv = _guest_argv_for(FATAL_TYPED_TEXT)
    joined = " ".join(argv)
    assert FATAL_PATTERN not in joined
    assert "pkill" not in joined
    # Not merely the pattern: no fragment of what the agent typed survives.
    for fragment in ("ffmpeg", "x11grab", "xdotool", "windowclose", "chrome"):
        assert fragment not in joined


def test_only_a_fixed_agent_independent_program_reaches_argv() -> None:
    """Two unrelated payloads produce byte-identical argv up to the encoding.

    This is the invariant that makes the property hold for text nobody has
    thought of yet, rather than for the strings these tests happen to name.
    """
    first = _guest_argv_for(FATAL_TYPED_TEXT)
    second = _guest_argv_for("echo something entirely different\n")
    assert first[:3] == second[:3] == ["python", "-c", computer_input._SOURCE_BOOTSTRAP]
    # The constant program mentions no action and no agent-supplied text.
    assert "pyautogui" not in computer_input._SOURCE_BOOTSTRAP


@pytest.mark.skipif(shutil.which("pgrep") is None, reason="needs pgrep")
def test_a_pkill_shaped_payload_no_longer_matches_our_own_live_process() -> None:
    """The real mechanism, against a real matcher and a real running process.

    ``pgrep -f`` rather than ``pkill -f``: it is the same matcher over the same
    ``/proc/<pid>/cmdline``, but it reports instead of signalling, so the test
    observes the selection without depending on having killed something. The
    guest runs ``pyautogui.typewrite``; the host has no GUI stack, so the
    statement's runtime is stood in for by a sleep while the argv -- the thing
    under test -- is built exactly as production builds it.
    """
    source = PKGS_PREFIX.replace("import pyautogui; ", "").format(
        command="time.sleep(30); typed = %r" % FATAL_TYPED_TEXT
    )
    argv = python_source_argv(source)
    process = subprocess.Popen(
        [sys.executable, *argv[1:]],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Wait for the kernel to publish the command line, so a negative result
        # cannot simply mean "the process had not started yet".
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if subprocess.run(["pgrep", "-f", argv[3][:40]], capture_output=True).returncode == 0:
                break
            time.sleep(0.05)
        else:
            pytest.fail("the guest-exec stand-in never became visible to pgrep")

        matched = subprocess.run(
            ["pgrep", "-f", FATAL_PATTERN], capture_output=True, text=True
        )
        pids = {int(pid) for pid in matched.stdout.split() if pid.isdigit()}
        assert process.pid not in pids
    finally:
        process.kill()
        process.wait(timeout=10)


# ----------------------------------------------------------------------------
# The statement still arrives intact


@pytest.mark.parametrize(
    "typed",
    [
        FATAL_TYPED_TEXT,
        "ordinary text",
        "quotes ' \" and a backslash \\ and a #hash",
        "a\tb\nnext\r\n",
        "$(command_substitution) `backticks` ; rm -rf /",
    ],
    ids=["fatal-payload", "ordinary", "quoting", "whitespace", "shell-metacharacters"],
)
def test_the_guest_still_receives_the_exact_statement_it_used_to(typed: str) -> None:
    """Encoding changes the transport, never the program the guest runs."""
    argv = _guest_argv_for(typed)
    decoded = base64.b64decode("".join(argv[3:])).decode("utf-8")
    assert decoded == PKGS_PREFIX.format(command=f"pyautogui.typewrite({typed!r})")


def test_the_decoded_source_executes_and_reproduces_the_text_verbatim(tmp_path: Any) -> None:
    """End to end through a real interpreter: encode, split, exec, compare.

    The guest's ``python -c`` is the only consumer of this argv, so the proof
    that matters is that a real interpreter reconstructs the text byte for byte
    -- including the newline and the embedded double quotes that a naive
    quoting scheme would drop.
    """
    marker = tmp_path / "typed.txt"
    source = (
        "from pathlib import Path; "
        f"Path({str(marker)!r}).write_text({FATAL_TYPED_TEXT!r}, encoding='utf-8')"
    )
    argv = python_source_argv(source)
    completed = subprocess.run(
        [sys.executable, *argv[1:]], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == FATAL_TYPED_TEXT


def test_every_argument_stays_within_the_single_argument_byte_limit() -> None:
    """Base64 is ASCII, so the character split is also the byte split.

    Linux caps one argv entry at MAX_ARG_STRLEN; a 100k-character paste encodes
    to ~533KB and must therefore still arrive in chunks the guest rejoins.
    """
    argv = python_source_argv("x = %r" % ("\N{SLIGHTLY SMILING FACE}" * 100_000))
    assert len(argv) > 4, "a payload this large must still be split"
    assert all(len(argument.encode("utf-8")) <= 64_000 for argument in argv[1:])


# ----------------------------------------------------------------------------
# Ordinary actions are unaffected


@pytest.mark.parametrize(
    "action, expected",
    [
        (
            ClickAction(observation_id="o", frame_id="screen", x=100, y=200, button="left"),
            "pyautogui.click(x=100, y=200, button='left')",
        ),
        (
            KeyAction(observation_id="o", keys=("ctrl", "s")),
            "pyautogui.hotkey('ctrl', 's')",
        ),
    ],
    ids=["click", "key-chord"],
)
def test_ordinary_actions_compile_and_survive_the_encoding_unchanged(
    action: Any, expected: str
) -> None:
    statement = actions.compile_action(action, _geo())
    assert statement == expected
    argv = python_source_argv(PKGS_PREFIX.format(command=statement))
    assert base64.b64decode("".join(argv[3:])).decode("utf-8").endswith(expected)


# ----------------------------------------------------------------------------
# The properties the fix had to preserve


@pytest.mark.asyncio
async def test_the_transport_contract_and_single_exec_boundary_are_unchanged() -> None:
    """One post, one exec, ``shell: false`` -- no new partial-commit boundary.

    A fix that uploaded the source to the guest and then executed it would have
    introduced a second round trip, and with it a state where the file landed
    but the exec never ran. The batch must still commit in exactly one step.
    """
    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        provider._env = _FakeEnv()
        provider._public_ip = "127.0.0.1"
        await provider.execute([f"pyautogui.typewrite({FATAL_TYPED_TEXT!r})"])

    assert len(stubs.guest_posts) == 1, "the batch must still commit in one post"
    post = stubs.guest_posts[0]
    assert post["shell"] is False
    assert isinstance(post["command"], list)
    assert FATAL_PATTERN not in " ".join(post["command"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"returncode": 7, "error": "PRIVATE_PAYLOAD"},
        RuntimeError("PRIVATE_PAYLOAD"),
    ],
    ids=["guest-stderr", "transport-exception"],
)
async def test_failures_still_redact_and_still_refuse_to_retry(
    response: dict[str, Any] | Exception,
) -> None:
    """Neither guest stderr nor a transport exception may reach the caller.

    Both can echo the command, credentials or UI text, and the outcome of a
    failed action is unknown -- so the batch stops where it failed and is never
    replayed. Encoding the source changed neither policy.
    """
    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        provider._env = _FakeEnv()
        provider._public_ip = "127.0.0.1"
        stubs.guest_default = response
        with pytest.raises(GuestExecutionError) as raised:
            await provider.execute(["first()", "must_not_run()"])

    assert "PRIVATE_PAYLOAD" not in str(raised.value)
    assert "batch not retried" in str(raised.value)
    assert len(stubs.guest_posts) == 1, "the second action must not have run"
    assert stubs.slept == [], "a failed batch does not settle"


@pytest.mark.asyncio
async def test_the_controllers_input_patch_still_wraps_every_statement() -> None:
    """``pkgs_prefix`` is upstream's isShiftCharacter fix; it must still apply.

    The encoding sits strictly below that wrapping -- we encode the already
    prefixed script -- so the guest still runs the patched program.
    """
    with _Stubs() as stubs:
        provider = AwsProvider(
            CREDS, region=REGION, lease_ref="lop-ttl-x", clients=stubs.clients, sleep=stubs.sleep
        )
        env = _FakeEnv()
        env.pkgs_prefix = PKGS_PREFIX
        provider._env = env
        provider._public_ip = "127.0.0.1"
        await provider.execute(["pyautogui.typewrite('abc')"])

    command = stubs.guest_posts[0]["command"]
    decoded = base64.b64decode("".join(command[3:])).decode("utf-8")
    assert decoded == PKGS_PREFIX.format(command="pyautogui.typewrite('abc')")
