"""No argv this adapter sends to the guest may approach Linux's argv limit.

THE REGRESSION THESE TESTS EXIST FOR. ``/execute`` takes an argv list and no
stdin, so every script the supervisor installs rides in a single argv string
under ``sudo -S bash -c``. Linux caps ONE argv string at ``MAX_ARG_STRLEN``
(131072 bytes on x86-64) and ``execve`` fails with ``E2BIG`` past it. When the
guest watchdog landed (#2739, 5b8ff262d9) the install's largest element grew
from ~118,600 to 135,579 bytes with a non-empty client password, so the real
guest rejected the exec. The provider only ever persisted the exception TYPE
(``providers/aws.py`` records ``type(error).__name__``, never the message), so a
size limit read as a transport mystery: ``{"cause": "HTTPError", "installed":
false, "status": "supervision unavailable"}`` on a real episode, while the arm
before the watchdog installed supervision in 20/20 runs.

WHY A BOUND RATHER THAN THE LIMIT. The payload is source that grows, and the
escalation ladder repeats the fragment once per sudo tier, so the shipped
number is a few times the source size. 100,000 bytes leaves real margin below
131,072 for a tier the ladder may add later, without failing the current head.

These tests assert the PROPERTY over every builder in the module, not the
spelling of one command, and they print the measured sizes so the next
regression names itself.
"""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any, Callable, Sequence

import pytest
from lop_osworld_v2_adapter import guest_disk, guest_supervisor

from tests.unit.evaluation.adapters.osworld.test_aws_provider import (  # noqa: F401
    _hermetic_aws,
    _plan,
    _Stubs,
)
from tests.unit.evaluation.adapters.osworld.test_guest_loss import _provider_over

#: Linux's per-string ceiling (``MAX_ARG_STRLEN``), in BYTES -- the unit the
#: kernel counts. Our payloads are ASCII, but measuring bytes is the property
#: that has to hold, not the character count of a particular encoding.
LINUX_MAX_ARG_STRLEN = 131_072

#: The bound the adapter ships against: real margin below the kernel's ceiling.
MAX_SAFE_ARGV_BYTES = 100_000

#: A password is what multiplies the escalation ladder (agentless, supplied, and
#: -- when the supplied value is one of upstream's own defaults -- the
#: ``upstream-default`` rung), and the ladder repeats the whole fragment once per
#: rung. Every builder is therefore measured on the runbook's documented value as
#: well: that is the WORST configuration the real guest sees, and measuring only
#: the two-tier shape left a blind window between the two-tier and three-tier
#: sizes in which this guard stayed green while the command had already crossed
#: the kernel limit. The value is read from ``guest_disk`` rather than repeated
#: here, so the ladder cannot gain a rung without this file measuring it.
PASSWORD_CASES = [
    ("no-password", ""),
    ("with-password", "two-tier-value"),
    ("documented-default-three-rungs", guest_disk._UPSTREAM_DEFAULT_PASSWORDS[0]),
]

PASSWORDS = [pytest.param(value, id=name) for name, value in PASSWORD_CASES]

UNIT = "action-server.service"

#: Every argv this module hands to a guest exec, by builder. ``restart_service_
#: argv`` is the argv ``systemd-run`` execs INSIDE the guest (the watchdog
#: dispatches the same one), so it is bounded by the same kernel rule.
BUILDERS: list[tuple[str, Callable[[str], Sequence[str]]]] = [
    ("install_command", lambda password: guest_supervisor.install_command(password)),
    (
        "restart_command",
        lambda password: guest_supervisor.restart_command(UNIT, password, 777),
    ),
    (
        "restart_result_command",
        lambda password: guest_supervisor.restart_result_command(password),
    ),
    ("layout_command", lambda password: guest_supervisor.layout_command(password)),
    (
        "report_command",
        lambda password: guest_supervisor.report_command(UNIT, password),
    ),
    (
        "watchdog restart_service_argv",
        lambda password: guest_supervisor.restart_service_argv(UNIT, 5000, 777),
    ),
]


def _largest_element(command: Sequence[str]) -> int:
    return max(len(part.encode()) for part in command)


def _measure(password: str) -> list[tuple[str, int]]:
    return [(name, _largest_element(build(password))) for name, build in BUILDERS]


@pytest.mark.parametrize("password", PASSWORDS)
def test_no_builder_puts_one_argv_element_near_the_kernel_limit(password: str) -> None:
    measured = _measure(password)
    over = [(name, size) for name, size in measured if size >= MAX_SAFE_ARGV_BYTES]
    report = ", ".join(f"{name}={size}" for name, size in measured)
    rungs = guest_supervisor.install_command(password)[-1].count("| sudo -S") + 1
    assert not over, (
        f"largest argv element over {MAX_SAFE_ARGV_BYTES} bytes "
        f"(Linux MAX_ARG_STRLEN={LINUX_MAX_ARG_STRLEN}) for password "
        f"{'set' if password else 'empty'}, {rungs} escalation rung(s): {over}. "
        f"Measured: {report}. "
        "The guest's /execute takes argv only, so an oversized element fails the "
        "exec with E2BIG and the provider can only record the exception type -- "
        "compress the payload or transport it outside argv (see TRANSPORT_SOURCE)."
    )
    # Print the inventory on success too: the failure message is only useful if
    # the sizes are cheap to read back when something grows unexpectedly.
    print(f"argv sizes (password {'set' if password else 'empty'}, " f"{rungs} rung(s)): {report}")


def test_the_documented_default_password_really_takes_the_third_rung() -> None:
    """The worst measured case must be the three-tier one, not a mislabelled copy.

    If the ladder stopped opening its ``upstream-default`` rung, the guard would
    silently fall back to measuring the two-tier shape under a three-tier name --
    the exact blind window it exists to close.
    """

    script = guest_supervisor.install_command(guest_disk._UPSTREAM_DEFAULT_PASSWORDS[0])[-1]
    assert "upstream-default" in script
    assert guest_supervisor.install_command("two-tier-value")[-1].count("| sudo -S") == 1


def test_the_install_transport_is_what_actually_rides_in_argv() -> None:
    """The compressed element is the shipped one, not a parallel copy."""

    for _, password in PASSWORD_CASES:
        command = guest_supervisor.install_command(password)
        assert any(
            guest_supervisor._install_payload() in part for part in command
        ), "the install command does not carry the compressed payload"


def test_the_install_payload_decodes_to_the_shipped_sources_verbatim() -> None:
    """Compression is transport only: the guest runs byte-identical source."""

    packed = guest_supervisor._install_payload()
    decoded = json.loads(zlib.decompress(base64.b64decode(packed)).decode())
    assert decoded == [
        guest_supervisor.INSTALL_SOURCE,
        guest_supervisor.CAPTURE_SOURCE,
        guest_supervisor.WATCHDOG_SOURCE,
        guest_supervisor.RESTART_SOURCE,
    ]


def _run_driver(payload: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", guest_supervisor.TRANSPORT_SOURCE, payload, "5000"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_the_driver_rebuilds_the_positional_contract_the_installer_reads() -> None:
    """``INSTALL_SOURCE`` still sees argv[1..4]: capture, port, watchdog, restart."""

    probe = "import json, sys\nprint(json.dumps(sys.argv))\n"
    batch = [probe, "<capture>", "<watchdog>", "<restart>"]
    payload = base64.b64encode(zlib.compress(json.dumps(batch).encode(), 9)).decode("ascii")
    result = _run_driver(payload)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [
        "-c",
        "<capture>",
        "5000",
        "<watchdog>",
        "<restart>",
    ]


def test_the_driver_keeps_the_install_refusal_vocabulary() -> None:
    """A refused install is still one JSON document on stdout, never a traceback."""

    probe = "raise RuntimeError('existing capture helper is not journalled')\n"
    batch = [probe, "", "", ""]
    payload = base64.b64encode(zlib.compress(json.dumps(batch).encode(), 9)).decode("ascii")
    result = _run_driver(payload)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "installed": False,
        "reason": "existing capture helper is not journalled",
    }


def test_the_install_the_fake_guest_receives_fits_and_arrives_once(tmp_path: Path) -> None:
    """The shipped builder, through the adapter's real guest-command path.

    ``_install_guest_supervisor`` is what a real episode runs; ``_Stubs``
    records the exact ``/execute`` body, so this measures the argv the guest
    would have received -- one command, one payload, under the bound.
    """

    with _Stubs() as stubs:
        provider = _provider_over(stubs, tmp_path)
        assert _plan().client_password
        provider._install_guest_supervisor(_plan())
        assert len(stubs.guest_posts) == 1, "the install must stay one logical operation"
        argv: Any = stubs.guest_posts[0]["command"]
        assert isinstance(argv, list)
        largest = _largest_element(argv)
        assert largest < MAX_SAFE_ARGV_BYTES, (
            f"the fake guest received a {largest}-byte argv element "
            f"(bound {MAX_SAFE_ARGV_BYTES}, Linux limit {LINUX_MAX_ARG_STRLEN})"
        )
        assert stubs.guest_posts[0]["shell"] is False


#: Every way the transport stage can fail, as the element the guest would receive.
#: These are the shapes that used to escape as an rc=1 traceback with EMPTY
#: stdout -- which the host cannot tell from a sudo rejection.
CORRUPT_PAYLOADS = [
    pytest.param(lambda good: good[: len(good) // 2], id="truncated-base64"),
    pytest.param(
        lambda good: good[:10] + ("A" if good[10] != "A" else "B") + good[11:],
        id="flipped-character",
    ),
    pytest.param(
        lambda good: base64.b64encode(b"not a zlib stream").decode("ascii"),
        id="not-zlib",
    ),
    pytest.param(
        lambda good: base64.b64encode(zlib.compress(b"not json")).decode("ascii"),
        id="not-json",
    ),
    pytest.param(
        lambda good: base64.b64encode(zlib.compress(json.dumps(["print(1)"]).encode())).decode(
            "ascii"
        ),
        id="wrong-arity",
    ),
    pytest.param(
        lambda good: base64.b64encode(
            zlib.compress(json.dumps({"a": "1", "b": "2", "c": "3", "d": "4"}).encode())
        ).decode("ascii"),
        id="object-not-list",
    ),
    pytest.param(
        lambda good: base64.b64encode(zlib.compress(json.dumps("scalar").encode())).decode("ascii"),
        id="scalar-not-list",
    ),
    pytest.param(
        lambda good: base64.b64encode(
            zlib.compress(json.dumps(["print(1)", 1, "2", "3"]).encode())
        ).decode("ascii"),
        id="non-string-member",
    ),
]


def _good_payload() -> str:
    return base64.b64encode(
        zlib.compress(json.dumps(["print(1)", "c", "w", "r"]).encode(), 9)
    ).decode("ascii")


@pytest.mark.parametrize("corrupt", CORRUPT_PAYLOADS)
def test_every_transport_corruption_is_one_fixed_vocabulary_refusal(
    corrupt: Callable[[str], str],
) -> None:
    """A broken payload must never read as a sudo failure.

    ``aws.py`` records only the cause word, so a transport crash is persisted as
    "privilege escalation or installation command failed" -- a false diagnosis
    this suite has already paid for. One literal keeps the stages distinct.
    """

    result = _run_driver(corrupt(_good_payload()))
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout) == {
        "installed": False,
        "reason": "install payload invalid",
    }


def test_the_driver_bounds_the_inflate_before_parsing() -> None:
    """The element is kernel-capped; the expansion it can produce must be too.

    Deflate reaches ~1030:1, so a guest-accepted 131,072-byte element could
    otherwise inflate to ~135 MB before anything looks at it. ``LIMIT`` in the
    driver makes that ceiling a decision -- and a payload over it is refused like
    any other transport corruption, not parsed.
    """

    oversized = ["print(1)", "A" * 1_500_000, "w", "r"]
    payload = base64.b64encode(zlib.compress(json.dumps(oversized).encode(), 9)).decode("ascii")
    assert len(payload) < MAX_SAFE_ARGV_BYTES
    result = _run_driver(payload)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "installed": False,
        "reason": "install payload invalid",
    }
