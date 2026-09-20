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
import re
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
    handshake_proof,
    handshake_proof_ok,
    is_wire_hex,
    mint_operator_cap,
    open_operator_cap_handoff,
    operator_cap_for,
    operator_cap_guarantee,
    operator_nonce,
    read_operator_cap_from_argv,
    remember_operator_cap,
    request_proof,
    request_proof_ok,
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
    ("supplied", "held", "client_nonce", "server_salt", "expected"),
    [
        # The honest case, in both directions the two ends compute.
        ("__proof__", b"c" * OPERATOR_CAP_BYTES, "n" * 64, "s" * 64, True),
        # Nothing held on this side: a runtime nobody handed a capability to
        # refuses even a well-formed proof.
        ("__proof__", None, "n" * 64, "s" * 64, False),
        # A credential that is not the one we minted is a programming error on
        # this side, and refusing is the only safe reading of it.
        ("__proof__", b"short", "n" * 64, "s" * 64, False),
        # No nonce, or no salt: a client that never asked for a handshake has no
        # connection-bound value it could legitimately hold.
        ("__proof__", b"c" * OPERATOR_CAP_BYTES, "", "s" * 64, False),
        ("__proof__", b"c" * OPERATOR_CAP_BYTES, "n" * 64, "", False),
        # Forged or absent candidates, including the types a JSON frame can
        # carry.
        (None, b"c" * OPERATOR_CAP_BYTES, "n" * 64, "s" * 64, False),
        ("", b"c" * OPERATOR_CAP_BYTES, "n" * 64, "s" * 64, False),
        (12345, b"c" * OPERATOR_CAP_BYTES, "n" * 64, "s" * 64, False),
        (b"x" * 32, b"c" * OPERATOR_CAP_BYTES, "n" * 64, "s" * 64, False),
        # Non-ASCII must REFUSE rather than raise: `hmac.compare_digest` on the
        # str form raises TypeError, which would turn a forged frame into a 500.
        ("é" * 64, b"c" * OPERATOR_CAP_BYTES, "n" * 64, "s" * 64, False),
    ],
)
def test_request_proof_ok_refuses_every_degenerate_case(
    supplied: object,
    held: bytes | None,
    client_nonce: str,
    server_salt: str,
    expected: bool,
) -> None:
    if supplied == "__proof__" and isinstance(held, bytes) and len(held) == OPERATOR_CAP_BYTES:
        ready: object = request_proof(held, client_nonce=client_nonce, server_salt=server_salt)
    else:
        ready = supplied
    assert (
        request_proof_ok(
            supplied=ready, held=held, client_nonce=client_nonce, server_salt=server_salt
        )
        is expected
    )


def test_a_proof_is_not_the_capability_and_is_bound_to_one_connection() -> None:
    """The properties the record-rewriting attack (agent review round 1, R1-1) needs.

    Read as one claim: what crosses the wire is a value that (a) is not the
    capability, (b) does not repeat across connections, and (c) cannot be moved
    between the two directions. Without (a) a same-uid impostor that rewrites
    ``control_port`` in the record simply reads the credential out of the console
    and replays it at the real runtime, which is what the reviewer reproduced
    end to end with production clients.
    """
    cap = mint_operator_cap()
    first = operator_nonce()
    second = operator_nonce()
    salt = operator_nonce()

    handshake = handshake_proof(cap, client_nonce=first, server_salt=salt)
    request = request_proof(cap, client_nonce=first, server_salt=salt)

    # (a) Not the secret, and not a fixed value derived from it alone.
    assert handshake != cap.hex()
    assert request != cap.hex()
    assert cap.hex() not in handshake
    # (b) Bound to the connection's nonces: the same capability produces a
    # different proof elsewhere, so a harvested value is worthless there.
    assert request_proof(cap, client_nonce=second, server_salt=salt) != request
    assert request_proof(cap, client_nonce=first, server_salt=second) != request
    # (c) Domain separated: a transcript's worth of one direction is not a
    # credential for the other.
    assert handshake != request
    assert (
        request_proof_ok(supplied=handshake, held=cap, client_nonce=first, server_salt=salt)
        is False
    )
    assert (
        handshake_proof_ok(supplied=request, held=cap, client_nonce=first, server_salt=salt)
        is False
    )
    # ...and each direction verifies against its own construction.
    assert (
        handshake_proof_ok(supplied=handshake, held=cap, client_nonce=first, server_salt=salt)
        is True
    )
    assert (
        request_proof_ok(supplied=request, held=cap, client_nonce=first, server_salt=salt) is True
    )
    # A DIFFERENT capability cannot satisfy either, which is what makes the
    # proof a proof rather than a token.
    other = mint_operator_cap()
    assert (
        request_proof_ok(supplied=request, held=other, client_nonce=first, server_salt=salt)
        is False
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        # A LITERAL, never a computed value: a random one would make this row's
        # NODE ID differ per worker, and pytest-xdist aborts the whole module
        # ("Different tests were collected between gw3 and gwX") — which took
        # every control in this file out of CI (agent review round 2, R2-1).
        # The mint's own shape is covered by ``operator_nonce()`` inside a test
        # body, and by ``test_the_capability_is_a_full_entropy_hex_string``.
        ("a" * 64, True),
        ("A" * 64, True),
        (None, False),
        ("", False),
        ("a" * 63, False),
        ("a" * 65, False),
        ("z" * 64, False),
        (12345, False),
        (b"a" * 32, False),
    ],
)
def test_is_wire_hex_accepts_exactly_a_thirty_two_byte_hex(value: object, expected: bool) -> None:
    """One shape check, shared by the runtime's auth reader and the client's."""
    assert is_wire_hex(value) is expected


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
        # The TUI's OWN handler, and the reason it belongs in this pin: a
        # TUI-hosted registrant routes ``op: slash`` to ``TuiSessionHandle`` and
        # from there into this handler's flag, so a word it accepts but
        # ``transition_authority`` does not know would be a loosening that reads
        # as ordinary at the seam (agent review round 1, MINOR).
        ("local_operator/tui/app.py", "_cmd_approvals"),
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


# ---------------------------------------------------------------------------
# Revision 2: the sources are COMBINED here, and the combination is the policy
# ---------------------------------------------------------------------------


def test_admit_increasing_combines_the_sources_and_never_invents_one() -> None:
    """The truth table of the seam's whole policy, in one place.

    Four cells, and the third is the one that matters most: a signature that was
    OFFERED and did not hold must not be a way in. The fourth is its twin and
    matters for a different reason — a capability holder whose client also
    attached a stale signature must not be locked out, because the frame could
    not be replayed into a naked yes anyway (the runtime single-uses the
    challenge before it asks).

    ``None`` is "this source said nothing", which is why it is not simply False:
    conflating the two would make a client that predates the field look like one
    whose signature failed, and the two want different log lines and different
    follow-ups.
    """
    from local_operator.harness.approval import admit_increasing

    # (capability, signature) -> admitted
    assert admit_increasing(capability=False, signature=None) is False
    assert admit_increasing(capability=True, signature=None) is True
    assert admit_increasing(capability=False, signature=True) is True
    assert admit_increasing(capability=True, signature=True) is True
    assert admit_increasing(capability=False, signature=False) is False
    assert admit_increasing(capability=True, signature=False) is True


def test_signature_target_is_derived_from_the_frame_not_from_a_field() -> None:
    """The action and the request id come from the FRAME, so a client cannot choose them.

    A ``request_id`` the caller invents for a loosening is harmless (the command
    names no card, and it is bound into the message either way), but the ACTION
    must not be: a signature minted to answer a card would otherwise be
    presentable as a loosening. It is derived from the same classification the
    seam already uses, which is why there is exactly one reading of a frame.
    """
    from local_operator.harness.approval import signature_target

    assert signature_target({"op": "slash_result", "command": "approvals", "args": "auto"}) == (
        "loosen",
        "",
    )
    assert signature_target(
        {"op": "slash", "command": "approvals", "args": "auto", "request_id": "r1"}
    ) == ("loosen", "r1")
    assert signature_target({"op": "approval_answer", "approved": True, "request_id": "ab"}) == (
        "approve",
        "ab",
    )
    # Ordinary frames have no target at all: a signature on one would be a
    # signature the runtime never checks, which is a place for state to rot.
    assert signature_target({"op": "approval_answer", "approved": False}) is None
    assert signature_target({"op": "ping"}) is None
    assert signature_target({"op": "slash_result", "command": "approvals", "args": "ask"}) is None


# ---------------------------------------------------------------------------
# Q7-1's class, closed rather than its sentence
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[3]

#: Every phrasing that offers a reader a remedy they cannot take under the
#: revision-2 authority model. The first two name the SPAWNER's window (which a
#: phone, an attached pane and a desktop backend are not, and which a
#: background-started runtime does not have); the last three are the deleted
#: retire-and-reopen family verbatim.
_WINDOW_REMEDIES = (
    "the window that started this session",
    "app window that started this session",
    "the terminal or app window that started",
    "the window that opens a runtime owns its gate",
    "retire and reopen",
    "reopen the session here",
)


def _shipped_notices() -> dict[str, str]:
    """Every notice CONSTANT the product ships, by name.

    Collected rather than listed: a hand-maintained list is the thing a fourth
    sentence slips past, which is exactly how this finding arrived three times
    (design round 1, agent round 6, QA round 7) — each fix swept the sentences
    someone had noticed.
    """
    return {
        name: value
        for name, value in vars(approval_module).items()
        if name.endswith("_NOTICE") and isinstance(value, str)
    }


def test_no_shipped_notice_names_a_window_remedy() -> None:
    """Q7-1 as a CLASS: no shipped copy sends a reader to the spawner's window.

    Three rounds produced three sentences carrying this clause — the refusal, the
    card refusal, and finally ``LOOSENING_REFUSED_NOTICE``, which QA round 7
    measured reaching a reader on BOTH hosts (``serving.py`` and ``tui/app.py``
    render the same constant) while the design doc asserted "no remedy depends on
    owning a window any more". Each fix had swept the sentences someone had
    looked at, so this cell inventories the PRODUCT instead: every notice
    constant in ``harness/approval.py``, and then every string literal in
    ``local_operator/**.py``, against the phrasings that promise a window.

    Docstrings are excluded and that is deliberate rather than convenient: the
    code that explains WHY a remedy was deleted has to be able to quote it, and
    two of those explanations sit in this diff. Comments are not in the AST at
    all. What is asserted is the copy a reader can be shown.

    THE SUBJECT GREW TWICE, each time because a pass found the clause somewhere
    this cell could not read (agent review round 8, R8-1): the shipped
    DOCUMENTATION (`docs/**/*.md`, whose own arm is below, after QA round 8 found
    the sentence asserted as current in `docs/DESKTOP_CONTROLS.md`) and the
    shipped PORTAL SOURCES (`local_operator/mobile/web/src`, where the phone's own
    copy lives — a reader is shown those strings exactly as they are shown the
    Python ones). The portal half strips JavaScript comments first, for the same
    reason the Python half skips docstrings: prose that records the deletion must
    be able to name it.
    """
    notices = _shipped_notices()
    # The inventory has to be big enough to be the product: a refactor that
    # renamed or inlined these constants would make the sweep below vacuous.
    assert len(notices) >= 6, sorted(notices)
    for name, copy in sorted(notices.items()):
        for phrase in _WINDOW_REMEDIES:
            assert phrase not in copy, f"{name} offers a window remedy: {copy!r}"

    offenders: dict[str, list[str]] = {}
    for source in sorted((_REPO_ROOT / "local_operator" / "mobile" / "web" / "src").rglob("*.ts*")):
        relative = source.relative_to(_REPO_ROOT).as_posix()
        normalised = _js_copy(source)
        for phrase in _WINDOW_REMEDIES:
            if phrase in normalised:
                offenders.setdefault(phrase, []).append(relative)
    for module in sorted((_REPO_ROOT / "local_operator").rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            for phrase in _WINDOW_REMEDIES:
                if phrase in node.value:
                    offenders.setdefault(phrase, []).append(
                        f"{module.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}"
                    )
    assert not offenders, f"shipped copy still promises a window remedy: {offenders}"


def _js_copy(path: Path) -> str:
    """A TypeScript source with its comments removed — the copy a renderer can show.

    The analogue of skipping docstrings on the Python side, and it exists for the
    same reason: a comment that records WHY a remedy was deleted must be able to
    name it (``pair.tsx`` has one, next to the revoked-device copy).
    """
    text = re.sub(r"/\*.*?\*/", " ", path.read_text(encoding="utf-8"), flags=re.DOTALL)
    return re.sub(r"^\s*//.*$", " ", text, flags=re.MULTILINE)


#: The subjects the class sweep covers, declared here so the list is a LIST rather
#: than prose in three docstrings (agent review round 9, R9-3). Each round of this
#: class' life found a subject the sweep did not have: the notice constants, then
#: the Python literals, then the portal sources, then the documentation. Naming them
#: together is what makes "the sweep" answerable, and
#: :func:`test_the_class_sweep_covers_every_subject_it_declares` fails if one of
#: them stops being read.
_CLASS_SUBJECTS = {
    "notice constants": "the copy the runtime raises, read through _shipped_notices()",
    "python literals": "every non-docstring string literal under local_operator/",
    "portal sources": "local_operator/mobile/web/src/*.ts*, JS comments stripped",
    "documentation": "docs/**/*.md, whitespace-normalised and read sentence by sentence",
}

#: The ONE exemption, and why it is one: the design record has to be able to quote
#: revision 1's text in order to say what was deleted — the class' whole point. The
#: exemption is CHECKED rather than trusted: a mention inside an exempt file must sit
#: beside a word that makes it history (see ``test_no_shipped_document_promises_a_
#: window_remedy``), so the exemption cannot become a second home for the claim.
_CLASS_EXEMPTIONS = {"docs/design/approval-authority.md": "the design record quotes revision 1"}

#: A window-ish noun and a gate-loosening verb in one SENTENCE is not enough on its
#: own to convict documentation: "`/approvals auto` typed in the console that owns
#: the gate" is the repaired, TRUE sentence, and it matches the shape. What made the
#: deleted rule false was the SPAWNER relation — the window was named as the process
#: that started this session's runtime — so the docs arm requires that relation too.
#: "owns the gate" is deliberately not a relation; "started/launched/spawned/opened
#: it" is (agent review round 9, R9-3, which warned this widening is what flags the
#: branch's own repaired sentence if it is done naively). Measured against the
#: tree: the verb alone still missed, because "`/approvals default ...` opens the
#: default editor" sat in the same long sentence as the (true) "console that owns
#: the gate" clause — so the relation takes an OBJECT (a session, a runtime, "it")
#: as well, which is the difference between opening a session and opening an editor.
_SPAWNER_RELATION = re.compile(
    r"\b(?:start(?:ed|s|ing)?|launch(?:ed|es|ing)?|spawn(?:ed|s|ing)?|open(?:ed|s|ing)?|ran)\b"
    r"[^.\n]{0,30}?\b(?:session|runtime|it|this one|that one)\b",
    re.IGNORECASE,
)


#: THE RULE ITSELF, worded with no window at all (agent review round 9, R9-1's
#: second exhibit): "Loosening a RUNNING gate is refused unless this backend started
#: that runtime" asserts the deleted model and names no window, so neither the
#: phrase list nor the window shape can see it. A sentence that BOTH names the gate
#: action AND says who started the runtime is the claim, whatever nouns it uses —
#: and the repaired sentences survive it because they either deny the relation
#: ("Who started the runtime has nothing to do with it.") or never mention it.
_SPAWNER_CLAIM = re.compile(
    r"\b(?:loosen\w*|approvals|retire and reopen)\b[^.\n]{0,90}?"
    r"\b(?:start(?:ed|s)|launch(?:ed|es)|spawn(?:ed|s)|open(?:ed|s))\b[^.\n]{0,40}?"
    r"\b(?:session|runtime|it)\b"
    r"|\b(?:start(?:ed|s)|launch(?:ed|es)|spawn(?:ed|s)|open(?:ed|s))\b[^.\n]{0,40}?"
    r"\b(?:session|runtime|it)\b[^.\n]{0,90}?\b(?:loosen\w*|approvals|retire and reopen)\b",
    re.IGNORECASE,
)

#: The one ALLOWANCE, and it is keyed to a file AND the exact string, so it cannot
#: excuse a copy of the sentence somewhere else. Why it is allowed at all: with NO
#: operator anchor on the host, the spawn capability IS the only source a running
#: runtime accepts, so `lop operator status` stating that is the level being
#: reported rather than the rule being asserted — and the line goes on to name the
#: way out. :func:`test_the_class_sweep_covers_every_subject_it_declares` asserts the
#: string is still in that file, so an allowance cannot outlive the sentence it
#: excuses.
_CLASS_ALLOWANCES = {
    (
        "local_operator/operator/handlers.py",
        "only the process that",
    ): "the spawn-only level's own report: with no anchor installed this is the state",
}


def _allowed(relative: str, copy: str) -> bool:
    """Whether ``copy`` sits in the allowance list, for THIS file.

    Keyed on the file as well as the text so an allowance cannot be inherited by a
    copy of the sentence elsewhere, which is the failure mode an exemption list has.
    """
    return any(file == relative and text in copy for (file, text) in _CLASS_ALLOWANCES)


def _sentences(text: str) -> list[str]:
    """A document split at sentence ends — the unit the docs shape arm judges."""
    return [part for part in re.split(r"(?<=[.;])\s+", text) if part.strip()]


#: A WINDOW NAMED AS THE PLACE TO LOOSEN FROM, in any wording (agent review round
#: 8, R8-2). The phrase list above is precise and hand-maintained, so a fourth
#: sentence saying the same thing in different words would pass it; this is the
#: invariant that cannot miss a paraphrase which keeps the REMEDY (``/approvals``,
#: ``adopt``) beside a window-ish noun. Its stated bound: a paraphrase with no
#: remedy word at all in the same clause — "do it in the console that opened the
#: session" — is still the phrase list's job.
_WINDOW_REMEDY_SHAPE = (
    re.compile(
        r"\b(window|console|terminal)\b[^.\n]{0,80}?\b(approvals|retire and reopen)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(approvals|retire and reopen)\b[^.\n]{0,80}?\b(window|console|terminal)\b",
        re.IGNORECASE,
    ),
)


def test_no_shipped_copy_offers_a_window_as_the_place_to_loosen() -> None:
    """The invariant the phrase list cannot be, checked over every shipped copy.

    Every one of the four sentences this class has produced paired a window-ish noun
    (window, console, terminal) with the remedy (``/approvals auto``, adopting the
    file, retiring and reopening). That conjunction is what makes the sentence a
    remedy rather than a description, and it survives rewording in a way a fixed
    phrase list does not.

    The tokens are chosen so the invariant is TRUE of this tree rather than merely
    loud: `approvals`/`adopt`/`retire and reopen` are the gate-loosening verbs, so
    unrelated copy that mentions a terminal and a reopen — the notifications help
    line, the phone's "the terminal session went away — reopen it to answer" —
    does not match. Both were measured against this cell before it was written, and
    the narrower token set is why it has no exemptions.
    """
    offenders: dict[str, list[str]] = {}
    for module in sorted((_REPO_ROOT / "local_operator").rglob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings:
                continue
            relative = module.relative_to(_REPO_ROOT).as_posix()
            if _allowed(relative, node.value):
                continue
            if any(shape.search(node.value) for shape in _WINDOW_REMEDY_SHAPE) or (
                _SPAWNER_CLAIM.search(node.value)
            ):
                offenders.setdefault(node.value[:60], []).append(f"{relative}:{node.lineno}")
    for source in sorted((_REPO_ROOT / "local_operator" / "mobile" / "web" / "src").rglob("*.ts*")):
        copy = _js_copy(source)
        if any(shape.search(copy) for shape in _WINDOW_REMEDY_SHAPE) or _SPAWNER_CLAIM.search(copy):
            offenders.setdefault(source.name, []).append(source.relative_to(_REPO_ROOT).as_posix())
    assert not offenders, f"shipped copy offers a window as the place to loosen: {offenders}"


#: The runtime slices a raised exception's text to this many characters before it
#: answers (``server.py``'s ``str(exc)[:400]``). One number, in one place, because
#: an earlier version of the cell below carried a per-notice mapping in which every
#: value equalled the default — a budget that could never change a verdict (agent
#: review round 8, R8-3).
_ERROR_FRAME_CHARS = 400


def test_the_class_sweep_covers_every_subject_it_declares() -> None:
    """The subject list is a claim, so it is measured (agent review round 9, R9-3).

    Every round of this class' life found a subject the sweep had grown around:
    the portal sources in round 8, the documentation in the same round, this cell's
    own list in round 9. A declared subject that stops being read is exactly how the
    next one survives, so each count is asserted non-zero and the exemption set is
    asserted to be the only thing skipped — a sweep that quietly stopped visiting
    `docs/` would fail here rather than pass silently.
    """
    counts = {
        "notice constants": len(_shipped_notices()),
        "python literals": sum(
            1
            for module in (_REPO_ROOT / "local_operator").rglob("*.py")
            for node in ast.walk(ast.parse(module.read_text(encoding="utf-8")))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        ),
        "portal sources": len(
            list((_REPO_ROOT / "local_operator" / "mobile" / "web" / "src").rglob("*.ts*"))
        ),
        "documentation": len(list((_REPO_ROOT / "docs").rglob("*.md"))),
    }
    assert set(counts) == set(_CLASS_SUBJECTS), (sorted(counts), sorted(_CLASS_SUBJECTS))
    empty = {name: count for name, count in counts.items() if count == 0}
    assert not empty, f"a declared subject of the class sweep is empty: {empty}"
    assert all((_REPO_ROOT / path).exists() for path in _CLASS_EXEMPTIONS)
    # An allowance that has outlived the sentence it excuses is a hole: each one is
    # asserted to still be in ITS file.
    stale = [
        (file, text)
        for (file, text) in _CLASS_ALLOWANCES
        if text not in (_REPO_ROOT / file).read_text(encoding="utf-8")
    ]
    assert not stale, f"an allowance outlived the copy it excuses: {stale}"


def test_every_notice_fits_the_error_frame_slice() -> None:
    """M-3: the two newest copies sat outside the cap that the two older ones pin.

    ``server.py`` slices a raised exception's text to 400 characters before it
    answers, so a copy longer than that reaches a raw client as a sentence that
    stops mid-remedy (QA round 2, Q3). The cells that pin this pinned
    ``OPERATOR_AUTHORITY_REQUIRED_NOTICE`` and ``CARD_APPROVAL_REFUSED_NOTICE``
    by name — and the unconfigured pair added in round 6, now the longest copies
    in the product, were pinned by nothing. A future edit naming
    ``lop operator install`` twice more would truncate with nothing going red.

    Asserted as a property of the INVENTORY rather than of four names, for the
    same reason the cell above collects its subjects.
    """
    for name, copy in sorted(_shipped_notices().items()):
        assert (
            len(copy) <= _ERROR_FRAME_CHARS
        ), f"{name} is {len(copy)} characters (cap {_ERROR_FRAME_CHARS}): {copy!r}"


def test_no_shipped_document_promises_a_window_remedy() -> None:
    """The class cell's DOCUMENT arm (QA round 8, Q8-1).

    The Python sweep above closes the class for copy the product can emit. It
    structurally cannot see a `.md`, and the very next pass found the same clause
    asserted as current in ``docs/DESKTOP_CONTROLS.md`` — this PR's own text, in a
    document that tells the desktop team what the gate rule IS. Two sentences that
    the rest of this round's evidence (a phone loosening a session the relay never
    spawned, an attached pane with no spawn capability, the shipped copy naming the
    operator's levers) falsifies directly.

    The sweep normalises whitespace before matching, because both hits were wrapped
    across lines and a line-oriented grep is what let them survive a review round.
    ``docs/design/approval-authority.md`` is EXEMPT and that exemption is itself
    checked: its mentions are the record of the deletion, so each one must sit
    within 600 characters of a word that makes it history rather than a promise. An
    exempt file whose mentions stop being explanatory fails here, which is what
    keeps the exemption from becoming a second home for the claim.
    """
    phrases = _WINDOW_REMEDIES + (
        "the console that started it",
        "the window that started it",
        "that started this session",
        "that started the session",
    )
    offenders: dict[str, list[str]] = {}
    for doc in sorted((_REPO_ROOT / "docs").rglob("*.md")):
        relative = doc.relative_to(_REPO_ROOT).as_posix()
        normalised = " ".join(doc.read_text(encoding="utf-8").split())
        for sentence in _sentences(normalised):
            phrase_hit = next((phrase for phrase in phrases if phrase in sentence), "")
            # THE SHAPE, scoped by the spawner relation: see _SPAWNER_RELATION for
            # why a bare shape arm would convict this PR's own repaired sentence.
            shape_hit = bool(
                any(shape.search(sentence) for shape in _WINDOW_REMEDY_SHAPE)
                and _SPAWNER_RELATION.search(sentence)
            ) or bool(_SPAWNER_CLAIM.search(sentence))
            if not (phrase_hit or shape_hit):
                continue
            if relative in _CLASS_EXEMPTIONS:
                continue
            offenders.setdefault(relative, []).append(phrase_hit or sentence[:80])
    assert not offenders, f"shipped documentation still promises a window remedy: {offenders}"

    history = (
        "deleted",
        "DELETED",
        "removed",
        "gone",
        "no longer",
        "used to",
        "retired",
        "revision 1",
        "Revision 1",
        "stopped naming",
        "deletes",
    )
    unguarded: list[str] = []
    for relative in _CLASS_EXEMPTIONS:
        record = " ".join((_REPO_ROOT / relative).read_text(encoding="utf-8").split())
        sentences = _sentences(record)
        for index, sentence in enumerate(sentences):
            phrase_hit = any(phrase in sentence for phrase in phrases)
            shape_hit = bool(
                any(shape.search(sentence) for shape in _WINDOW_REMEDY_SHAPE)
                and _SPAWNER_RELATION.search(sentence)
            ) or bool(_SPAWNER_CLAIM.search(sentence))
            if not (phrase_hit or shape_hit):
                continue
            # The word can sit in the sentence beside it: this document's prose runs
            # long, and "the model that was wrong" is often introduced one clause
            # earlier than the quotation (round 9's own citation is written that way).
            around = " ".join(sentences[max(0, index - 1) : index + 2])
            if not any(word in around for word in history):
                unguarded.append(sentence[:120])
    assert not unguarded, (
        f"an exempt document mentions a window remedy as something other than history: "
        f"{unguarded}"
    )


def _occurrences(text: str, phrase: str) -> list[int]:
    """Every index ``phrase`` appears at — a scan, because ``str.find`` cannot say."""
    found: list[int] = []
    start = 0
    while (at := text.find(phrase, start)) != -1:
        found.append(at)
        start = at + 1
    return found
