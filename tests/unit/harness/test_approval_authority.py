"""The approval gate's authority model (issue #1310).

Two units are pinned here, and they are the whole of the rule:

* :func:`local_operator.harness.approval.transition_authority` — which
  control-plane requests are AUTHORITY-INCREASING, i.e. which ones remove or
  answer the gate that constrains the caller. Getting this set too NARROW is a
  security hole (a loosening word that reads as ordinary); getting it too WIDE
  is a usability regression (a report or a tightening refused). Both directions
  are asserted below.
* :func:`local_operator.harness.approval.operator_cap_ok` — the constant-time
  comparison, whose interesting cases are all the degenerate ones.

The FD HANDOFF is pinned with a real child process, because the properties that
matter about it (the descriptor survives the exec, its NUMBER is what rides in
argv, the descriptor is gone afterwards) are properties of ``exec`` and cannot
be observed in process.
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.harness import approval as approval_module
from local_operator.harness.approval import (
    APPROVALS_LOOSENING_WORDS,
    OPERATOR_CAP_BYTES,
    OPERATOR_FD_FLAG,
    frame_authority,
    mint_operator_cap,
    open_operator_cap_handoff,
    operator_cap_for,
    operator_cap_guarantee,
    operator_cap_ok,
    read_operator_cap_from_argv,
    remember_operator_cap,
    reset_operator_caps_for_tests,
    transition_authority,
)

_TESTS_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.parametrize(
    ("command", "args", "expected"),
    [
        # The transition the whole change is about, in all three spellings the
        # handlers accept.
        ("approvals", "auto", "authority-increasing"),
        ("approvals", "off", "authority-increasing"),
        ("approvals", "yolo", "authority-increasing"),
        # Whitespace and case are normalised exactly as the sinks normalise
        # them, so a frame the sink WOULD honour cannot read as ordinary here.
        ("approvals", "  AUTO  ", "authority-increasing"),
        ("Approvals", "auto", "authority-increasing"),
        # Tightening and reporting are ordinary: they must keep working from
        # every surface, which is the asymmetry the issue asks for.
        ("approvals", "ask", "ordinary"),
        ("approvals", "on", "ordinary"),
        ("approvals", "prompt", "ordinary"),
        ("approvals", "", "ordinary"),
        ("approvals", "default auto", "ordinary"),
        ("approvals", "auto now", "ordinary"),
        ("approvals", "banana", "ordinary"),
        # Every other command, including the neighbours that look similar.
        ("model", "auto", "ordinary"),
        ("stop", "", "ordinary"),
        ("prompt", "auto", "ordinary"),
        ("gate", "auto", "ordinary"),
        ("", "", "ordinary"),
    ],
)
def test_transition_authority_classifies_the_slash_class(
    command: str, args: str, expected: str
) -> None:
    assert transition_authority(command, args) == expected


@pytest.mark.parametrize(
    ("approved", "expected"),
    [
        (True, "authority-increasing"),
        (False, "ordinary"),
        (None, "ordinary"),
    ],
)
def test_transition_authority_classifies_an_approval_answer(
    approved: bool | None, expected: str
) -> None:
    """A deny is ordinary ON PURPOSE: yes may be withheld everywhere, no may not."""
    assert transition_authority("", "", approved=approved) == expected


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        ({"op": "slash", "command": "approvals", "args": "auto"}, "authority-increasing"),
        ({"op": "slash_result", "command": "approvals", "args": "auto"}, "authority-increasing"),
        ({"op": "slash_result", "command": "approvals", "args": "ask"}, "ordinary"),
        ({"op": "approval_answer", "approved": True}, "authority-increasing"),
        ({"op": "approval_answer", "approved": False}, "ordinary"),
        # Truthiness, not identity: the dispatch reads the field in a boolean
        # position, so anything truthy IS an approval and is judged as one.
        ({"op": "approval_answer", "approved": 1}, "authority-increasing"),
        ({"op": "prompt", "text": "hello"}, None),
        ({"op": "ping"}, None),
        ({"op": "ask_answer", "value": "yes"}, None),
    ],
)
def test_frame_authority_reads_the_right_field_per_op(
    frame: dict[str, Any], expected: str | None
) -> None:
    assert frame_authority(frame) == expected


def test_the_capability_is_a_full_entropy_hex_string() -> None:
    first, second = mint_operator_cap(), mint_operator_cap()
    assert len(first) == OPERATOR_CAP_BYTES
    assert first != second
    assert len(first.hex()) == OPERATOR_CAP_BYTES * 2


@pytest.mark.parametrize(
    ("supplied", "held", "expected"),
    [
        (None, None, False),
        (None, b"x" * OPERATOR_CAP_BYTES, False),
        ("", b"x" * OPERATOR_CAP_BYTES, False),
        ("ab" * OPERATOR_CAP_BYTES, None, False),
        ("ab" * OPERATOR_CAP_BYTES, b"x" * OPERATOR_CAP_BYTES, False),
        # A short/long held value is a bug on this side, and refusing is the only
        # safe reading of a credential that is not the one we minted.
        (b"x".hex(), b"x", False),
        (12345, b"x" * OPERATOR_CAP_BYTES, False),
        # Non-ASCII must REFUSE rather than raise: `hmac.compare_digest` on the
        # str form raises TypeError, which would turn a forged frame into a 500.
        ("é" * OPERATOR_CAP_BYTES, b"x" * OPERATOR_CAP_BYTES, False),
    ],
)
def test_operator_cap_ok_refuses_every_degenerate_case(
    supplied: object, held: bytes | None, expected: bool
) -> None:
    assert operator_cap_ok(supplied=supplied, held=held) is expected


def test_operator_cap_ok_accepts_exactly_the_hex_of_what_it_holds() -> None:
    cap = mint_operator_cap()
    assert operator_cap_ok(supplied=cap.hex(), held=cap) is True
    # Uppercase hex is a DIFFERENT string, and the mint produces lowercase: the
    # comparison is over the exact bytes, not over a numeric value.
    assert operator_cap_ok(supplied=cap.hex().upper(), held=cap) is False


def test_the_registry_answers_only_for_pids_this_process_remembers() -> None:
    reset_operator_caps_for_tests()
    cap = mint_operator_cap()
    remember_operator_cap(4242, cap)
    assert operator_cap_for(4242) == cap
    # Another process's runtime is the case the whole change turns on.
    assert operator_cap_for(4243) is None
    reset_operator_caps_for_tests()
    assert operator_cap_for(4242) is None


@pytest.mark.parametrize(
    ("platform", "os_name", "scope", "expected"),
    [
        ("darwin", "posix", None, "strong"),
        ("win32", "nt", None, "weak"),
        ("linux", "posix", "1", "strong"),
        ("linux", "posix", "2", "strong"),
        ("linux", "posix", "0", "not-a-boundary"),
        ("linux", "posix", None, "unreported"),
        ("freebsd12", "posix", None, "unreported"),
    ],
)
def test_the_reported_boundary_matches_the_host(
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
    os_name: str,
    scope: str | None,
    expected: str,
) -> None:
    """The report must never claim a boundary the host does not have.

    ``ptrace_scope=0`` is the case that matters: a same-uid process can read this
    one's memory there, so the capability raises the cost of an attack rather
    than closing it, and the honest label is what the design doc's residual
    section is built on.

    ``os.name`` is patched as well as ``sys.platform`` because the Windows
    branch keys on the former — the two are separate reads, and a test that
    patched only one would report the platform it is running on rather than the
    one it names (which is how the first version of this test passed on a Linux
    row it never reached).
    """
    monkeypatch.setattr(approval_module.sys, "platform", platform)
    monkeypatch.setattr(approval_module.os, "name", os_name)
    if scope is None:
        monkeypatch.setattr(approval_module.Path, "read_text", _raise_oserror)
    else:
        monkeypatch.setattr(approval_module.Path, "read_text", lambda self, **_: scope)
    assert operator_cap_guarantee() == expected


def _raise_oserror(self: Path, **_: object) -> str:
    raise FileNotFoundError(self)


# ---------------------------------------------------------------------------
# The handoff, on a real child
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_the_capability_reaches_a_real_child_and_the_parent_keeps_none() -> None:
    """The handoff's real properties, observed through a real ``exec``.

    The child is a separate interpreter, so this exercises everything that only
    exists across a fork: the descriptor surviving the exec, its NUMBER being
    what argv carries, the exact 32 bytes arriving, and BOTH ends being closed
    on this side afterwards (which is what keeps a later ``close_fds=True`` tool
    spawn from being the only thing protecting it).
    """
    cap = mint_operator_cap()
    handoff = open_operator_cap_handoff()
    descriptor = int(handoff.argv[1])
    probe = (
        "import hashlib, sys\n"
        "from local_operator.harness.approval import read_operator_cap_from_argv\n"
        "value = read_operator_cap_from_argv(sys.argv[1:])\n"
        "print('NONE' if value is None else hashlib.sha256(value).hexdigest())\n"
        "print(sys.argv[2])\n"
    )
    process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", probe, *handoff.argv],
        pass_fds=handoff.pass_fds,
        close_fds=handoff.close_fds,
        stdout=subprocess.PIPE,
    )
    try:
        handoff.deliver(cap)
        out, _ = process.communicate(timeout=30)
    finally:
        handoff.close()
    digest, _, reported = out.decode().strip().partition("\n")
    expected = hashlib.sha256(cap).hexdigest()
    assert digest == expected, digest
    # THE NUMBER, NOT THE VALUE, is what argv carried — and the digest above is
    # the only thing the child printed, so nothing here can leak the capability
    # into a log by accident.
    assert reported == str(descriptor), f"argv carried {reported}, expected {descriptor}"
    assert cap.hex() not in out.decode()
    assert not _fd_is_open(descriptor), "the spawner kept the child's end of the handoff"


def test_a_child_with_no_flag_gets_no_capability() -> None:
    """Fail-closed for the spawners and callers that predate the field."""
    assert read_operator_cap_from_argv([]) == None  # noqa: E711 — explicitness
    assert read_operator_cap_from_argv(["--other", "1"]) is None
    assert read_operator_cap_from_argv([OPERATOR_FD_FLAG]) is None
    assert read_operator_cap_from_argv([f"{OPERATOR_FD_FLAG}=nonsense"]) is None
    # An unopened descriptor number is not an error, it is an absence.
    assert read_operator_cap_from_argv([OPERATOR_FD_FLAG, "999999"]) is None


def test_both_argv_spellings_of_the_flag_are_read() -> None:
    cap = mint_operator_cap()
    read_fd, write_fd = os.pipe()
    os.write(write_fd, cap)
    os.close(write_fd)
    try:
        assert read_operator_cap_from_argv([OPERATOR_FD_FLAG, str(read_fd)]) == cap
    finally:
        with pytest.raises(OSError):
            os.close(read_fd)


def test_a_short_handoff_is_refused_rather_than_padded() -> None:
    """A truncated write must not become a capability of the wrong length."""
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b"short")
    os.close(write_fd)
    try:
        assert read_operator_cap_from_argv([OPERATOR_FD_FLAG, str(read_fd)]) is None
    finally:
        pytest.raises(OSError, os.close, read_fd)


def test_a_closed_handoff_closes_twice_without_complaint() -> None:
    handoff = open_operator_cap_handoff()
    handoff.close()
    handoff.close()
    assert handoff.closed is True


#: The tool-shaped probe: a grandchild spawned the way ``tools/builtin.py``
#: spawns every model-run command (``close_fds=True``, ``start_new_session``),
#: asked whether a given descriptor is in its table.
_TOOL_SHAPED_PROBE = (
    "import os, sys\n"
    "try:\n"
    "    os.fstat(int(sys.argv[1]))\n"
    "    print('OPEN')\n"
    "except OSError:\n"
    "    print('CLOSED')\n"
)

#: The child: reports whether IT holds the handoff descriptor, spawns the
#: tool-shaped grandchild, reads the capability, and reports again.
_HANDOFF_OF_HANDOFF_PROBE = (
    "import subprocess, sys, os\n"
    "from local_operator.harness.approval import read_operator_cap_from_argv\n"
    "descriptor = int(sys.argv[2])\n"
    "def probe():\n"
    "    done = subprocess.run(\n"
    "        [sys.executable, '-c', PROBE, str(descriptor)],\n"
    "        capture_output=True, close_fds=True, start_new_session=True,\n"
    "    )\n"
    "    return done.stdout.decode().strip()\n"
    "try:\n"
    "    os.fstat(descriptor)\n"
    "    print('child-before', 'OPEN')\n"
    "except OSError:\n"
    "    print('child-before', 'CLOSED')\n"
    "print('tool-shaped', probe())\n"
    "read_operator_cap_from_argv(sys.argv[1:])\n"
    "try:\n"
    "    os.fstat(descriptor)\n"
    "    print('child-after', 'OPEN')\n"
    "except OSError:\n"
    "    print('child-after', 'CLOSED')\n"
)


@pytest.mark.slow
def test_a_tool_shaped_subprocess_cannot_see_the_handoff_descriptor() -> None:
    """The descriptor is unreadable to the shape of process that runs a model's tools.

    ``tools/builtin.py`` spawns every model-run command with ``close_fds=True``
    and ``start_new_session=True``, and the runtime closes the handoff descriptor
    before it serves anything — so a tool subprocess can neither inherit it nor
    find it by number. Both halves are measured here, through two real
    generations of process, because either one alone leaves the claim untested:

    * the child DOES hold the descriptor before it reads (so "the tool cannot
      see it" is a property of the tool spawn and of the close, not of a
      handoff that delivered nothing);
    * the tool-shaped grandchild does not have it, before OR after the read.
    """
    cap = mint_operator_cap()
    handoff = open_operator_cap_handoff()
    program = _HANDOFF_OF_HANDOFF_PROBE.replace("PROBE", repr(_TOOL_SHAPED_PROBE))
    process = subprocess.Popen(  # noqa: S603 — fixed argv, no shell
        [sys.executable, "-c", program, *handoff.argv],
        pass_fds=handoff.pass_fds,
        close_fds=handoff.close_fds,
        stdout=subprocess.PIPE,
    )
    try:
        handoff.deliver(cap)
        out, _ = process.communicate(timeout=60)
    finally:
        handoff.close()
    text = out.decode()
    assert "child-before OPEN" in text, text
    assert "tool-shaped CLOSED" in text, text
    assert "child-after CLOSED" in text, text
    assert cap.hex() not in text


def _fd_is_open(descriptor: int) -> bool:
    """Whether ``descriptor`` is still open in THIS process.

    ``os.fstat`` rather than ``/dev/fd``: it is the same question, it needs no
    procfs, and it cannot be confused by a descriptor that closed and was reused
    (the reuse case is a real one — the test process opens sockets while this
    runs, so a path-based probe could observe a *different* fd at that number).
    """
    try:
        os.fstat(descriptor)
    except OSError:
        return False
    return True


# ---------------------------------------------------------------------------
# The pin between this module's word set and the two sinks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "function"),
    [
        ("local_operator/session/runtime/serving.py", "_approvals_slash"),
        ("local_operator/tui/app.py", "_approvals_slash_result"),
    ],
)
def test_the_loosening_word_set_matches_both_approvals_handlers(module: str, function: str) -> None:
    """A widening of the sinks alone must fail HERE, not in production.

    ``transition_authority`` classifies frames; the two ``/approvals`` handlers
    are what ACT on them. The rule is one rule, so the strings that mean "loosen"
    must be one set — and a coder who adds a synonym to a handler without adding
    it here would otherwise leave a loosening word that reads as ordinary at the
    seam, i.e. a hole with no failing test. Read from the source rather than
    asserted against a copy of itself, so this cannot pass vacuously.
    """
    tree = ast.parse((_TESTS_ROOT / module).read_text(encoding="utf-8"))
    handler = next(
        (
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function
        ),
        None,
    )
    assert handler is not None, f"{module}::{function} moved or was renamed"

    found: set[frozenset[str]] = set()
    for node in ast.walk(handler):
        # ``argument in ("auto", "off", "yolo")`` — the sink's own spelling.
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.ops[0], (ast.In, ast.NotIn))
            and node.comparators
            and isinstance(node.comparators[0], ast.Tuple)
        ):
            literals = {
                element.value
                for element in node.comparators[0].elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            }
            if literals & APPROVALS_LOOSENING_WORDS:
                found.add(frozenset(literals))
    assert found == {frozenset(APPROVALS_LOOSENING_WORDS)}, (
        f"{module}::{function} compares against {found} while the seam classifies "
        f"{set(APPROVALS_LOOSENING_WORDS)} — the two must be the same set, or a "
        "loosening word reaches the sink that the seam read as ordinary."
    )
