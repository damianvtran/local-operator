"""Interactive credential paste over a REAL pseudo-terminal (QA Q4).

Why a pty and not a ``StringIO`` stand-in: the defect lived in the kernel's
line discipline. In canonical mode one line is capped at ``MAX_CANON`` (1024
bytes on macOS); a real GCP service-account key has a ~1.7 KiB
``private_key`` line, so the excess and the Enter were dropped and
``lop login google-vertex-sa`` hung until Ctrl-C. Only a real terminal device
reproduces that, so every test here drives ``pty.openpty()``.

Each read runs on a helper thread joined with a timeout: the failure mode is a
hang, and a test that hangs reports nothing.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from typing import Any

import pytest

pty = pytest.importorskip("pty")
termios = pytest.importorskip("termios")

from local_operator.providers import auth_cli  # noqa: E402

#: Generous for a read that completes in milliseconds; only a hang reaches it.
READ_TIMEOUT_S = 10.0


def _attrs(fd: int) -> list[Any]:
    """Terminal attributes minus PENDIN, which the kernel itself sets.

    macOS flags PENDIN on EVERY switch back to canonical mode, even with an
    empty input queue (measured: a bare ICANON off/on round trip on a fresh
    pty changes lflag by exactly PENDIN). It is the kernel's "reprint pending
    input" note, cleared on the next read, not a mode the reader left behind.
    """
    attrs = termios.tcgetattr(fd)
    attrs[3] &= ~getattr(termios, "PENDIN", 0)
    return attrs


def _service_account_json() -> str:
    """A pretty-printed key shaped like a real one, with a >1.7 KB key line."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    pem = (
        rsa.generate_private_key(public_exponent=65537, key_size=2048)
        .private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode("ascii")
    )
    payload = {
        "type": "service_account",
        "project_id": "minerva-vertex",
        "private_key_id": "0123456789abcdef0123456789abcdef01234567",
        "private_key": pem,
        "client_email": "sa@minerva-vertex.iam.gserviceaccount.com",
        "client_id": "123456789012345678901",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": (
            "https://www.googleapis.com/robot/v1/metadata/x509/"
            "sa%40minerva-vertex.iam.gserviceaccount.com"
        ),
        "universe_domain": "googleapis.com",
    }
    text = json.dumps(payload, indent=2)
    longest = max(len(line.encode()) for line in text.splitlines())
    # The precondition that makes this a regression test: one line longer than
    # macOS's MAX_CANON (1024), at real-key size.
    assert longest > 1700, longest
    return text


@pytest.fixture
def terminal() -> Iterator[tuple[int, int]]:
    master, slave = pty.openpty()
    try:
        yield master, slave
    finally:
        # Slave first: a writer still blocked on the master (after a failed
        # read) is released by the hang-up instead of wedging the close.
        for fd in (slave, master):
            try:
                os.close(fd)
            except OSError:
                pass


def _read(slave: int, master: int, typed: bytes, *, chunk: int = 0) -> Any:
    """Run ``_read_tty_credential`` on ``slave`` while ``typed`` goes into ``master``.

    Returns the value or the raised exception. ``chunk`` > 0 writes in pieces,
    the way a keyboard (or a slow paste) delivers it.
    """
    outcome: dict[str, Any] = {}

    def reader() -> None:
        try:
            outcome["value"] = auth_cli._read_tty_credential(slave)
        except BaseException as raised:  # noqa: BLE001 — the result under test
            outcome["value"] = raised

    def writer() -> None:
        pieces = (
            [typed] if chunk <= 0 else [typed[i : i + chunk] for i in range(0, len(typed), chunk)]
        )
        for piece in pieces:
            os.write(master, piece)
            if chunk > 0:
                time.sleep(0.002)

    before = _attrs(slave)
    read_thread = threading.Thread(target=reader, daemon=True)
    read_thread.start()
    # The reader must switch the mode before the bytes land, as it does for a
    # user who pastes after seeing the prompt.
    time.sleep(0.05)
    write_thread = threading.Thread(target=writer, daemon=True)
    write_thread.start()
    read_thread.join(READ_TIMEOUT_S)
    assert not read_thread.is_alive(), "credential read hung (MAX_CANON regression)"
    assert _attrs(slave) == before, "terminal attributes were not restored"
    return outcome["value"]


@pytest.mark.parametrize("newline", [b"\r", b"\n", b"\r\n"])
def test_pasted_service_account_json_longer_than_max_canon_completes(
    terminal: tuple[int, int], newline: bytes
) -> None:
    master, slave = terminal
    text = _service_account_json()
    typed = text.encode().replace(b"\n", newline) + newline
    value = _read(slave, master, typed)
    assert json.loads(value) == json.loads(text)


def test_bracketed_paste_markers_are_stripped(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    text = _service_account_json()
    typed = b"\x1b[200~" + text.encode().replace(b"\n", b"\r") + b"\x1b[201~"
    assert json.loads(_read(slave, master, typed)) == json.loads(text)


def test_slowly_typed_json_completes_on_its_closing_brace(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    typed = b'{\r  "a": "x}{\\"",\r  "b": 1\r}'  # braces and quotes inside a string
    value = _read(slave, master, typed, chunk=1)
    assert json.loads(value) == {"a": 'x}{"', "b": 1}


#: Above FD_SETSIZE (1024), where ``select.select`` raises ValueError.
HIGH_FD = 1100


@pytest.fixture
def high_fd_terminal(terminal: tuple[int, int]) -> Iterator[tuple[int, int]]:
    """The pty's slave end moved to fd ``HIGH_FD``.

    Why: the suite leaks fds, so in some shard placements ``pty.openpty()``
    returns fds above 1023 (CI job 136728521591: ``terminal = (1171, 1172)``)
    and the reader's ``select.select`` raised. ``dup2`` reproduces that
    placement deterministically instead of waiting for a leak to line up.
    ``F_DUPFD`` (lowest FREE fd >= ``HIGH_FD``), not ``dup2``: ``dup2`` would
    silently close whatever already holds that number -- under a leaky run
    that can be this very pty's master.
    """
    resource = pytest.importorskip("resource")
    fcntl = pytest.importorskip("fcntl")
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = HIGH_FD + 256  # headroom in case fds just above HIGH_FD are taken
    if hard != resource.RLIM_INFINITY and hard <= HIGH_FD:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} cannot hold fd {HIGH_FD}")
    raised = soft != resource.RLIM_INFINITY and soft < wanted
    if raised:
        new_soft = wanted if hard == resource.RLIM_INFINITY else min(wanted, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    master, slave = terminal
    high = fcntl.fcntl(slave, fcntl.F_DUPFD, HIGH_FD)
    try:
        yield master, high
    finally:
        os.close(high)
        if raised:
            resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))


def test_paste_on_a_terminal_above_fd_setsize_completes(
    high_fd_terminal: tuple[int, int],
) -> None:
    master, slave = high_fd_terminal
    assert slave >= 1024
    text = _service_account_json()
    # No Enter in the same read: the closing brace triggers the trailing-newline
    # wait, which is the select() call that raised (CI job 136728521591).
    typed = b"\x1b[200~" + text.encode().replace(b"\n", b"\r") + b"\x1b[201~"
    value = _read(slave, master, typed)
    assert not isinstance(value, BaseException), value
    assert json.loads(value) == json.loads(text)


def test_trailing_newline_is_consumed_above_fd_setsize(
    high_fd_terminal: tuple[int, int],
) -> None:
    master, slave = high_fd_terminal
    import tty

    before = termios.tcgetattr(slave)
    tty.setraw(slave)  # bytes reach the reader without waiting for a line
    try:
        os.write(master, b"\x1b[201~\rnext")
        time.sleep(0.05)
        auth_cli._consume_trailing_newline(slave)  # must not raise ValueError
        # Everything up to and including the Enter was framing; the rest stays.
        assert os.read(slave, 16) == b"next"
    finally:
        termios.tcsetattr(slave, termios.TCSANOW, before)


def test_single_line_path_finishes_on_enter(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    assert _read(slave, master, b"~/keys/sa.json\r") == "~/keys/sa.json"


def test_backspace_edits_the_value(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    assert _read(slave, master, b"~/keys/sx\x7f\x08sa.json\r", chunk=1) == "~/keys/sa.json"


def test_ctrl_d_on_nothing_typed_is_eof(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    assert isinstance(_read(slave, master, b"\x04"), EOFError)


def test_ctrl_d_after_partial_json_returns_what_was_typed(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    assert _read(slave, master, b'{\r  "a": 1\r\x04') == '{\n  "a": 1\n'


def test_blank_line_ends_an_unbalanced_paste(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    assert _read(slave, master, b'{\r  "a": 1\r\r') == '{\n  "a": 1\n'


def test_enter_on_nothing_typed_cancels(terminal: tuple[int, int]) -> None:
    master, slave = terminal
    assert _read(slave, master, b"\r") == ""


def test_ctrl_c_byte_raises_keyboard_interrupt_when_isig_is_off(terminal: tuple[int, int]) -> None:
    # With ISIG on (as the reader sets it) the kernel turns Ctrl-C into SIGINT
    # and the byte never arrives (covered by the controlling-tty test below);
    # the byte branch is the fallback for a terminal that delivers it anyway,
    # reproduced here by disabling the INTR character.
    master, slave = terminal
    attrs = termios.tcgetattr(slave)
    control = list(attrs[6])
    control[termios.VINTR] = b"\x00"  # 0 disables the INTR character
    attrs[6] = control
    termios.tcsetattr(slave, termios.TCSANOW, attrs)
    assert isinstance(_read(slave, master, b"abc\x03"), KeyboardInterrupt)


_CHILD = textwrap.dedent("""
    import asyncio, fcntl, json, os, sys, termios

    # Make the pty this process's controlling terminal, so Ctrl-C typed into
    # it is a real SIGINT to us, as it is for `lop login` at a shell.
    fcntl.ioctl(0, termios.TIOCSCTTY, 0)
    PENDIN = getattr(termios, "PENDIN", 0)  # set by the kernel; see _attrs

    def attrs():
        current = termios.tcgetattr(0)
        current[3] &= ~PENDIN
        return current

    before = attrs()

    from local_operator.providers import auth_cli
    from local_operator.providers.registry import get_provider_definition

    callbacks = auth_cli._callbacks_interactive(get_provider_definition("google-vertex-sa"))
    result = {}
    try:
        with auth_cli._terminal_state_restored():
            result["value"] = asyncio.run(callbacks.on_manual_code_input())
    except KeyboardInterrupt:
        result["interrupted"] = True
    result["restored"] = attrs() == before
    sys.stderr.write("RESULT " + json.dumps(result) + "\\n")
    sys.stderr.flush()
    os._exit(0)
    """)


def _run_child(typed: bytes) -> dict[str, Any]:
    """Drive the real CLI paste callback in a child whose controlling tty is a pty."""
    master, slave = pty.openpty()
    env = {k: v for k, v in os.environ.items() if not k.startswith("CMUX_")}
    child = subprocess.Popen(
        [sys.executable, "-c", _CHILD],
        stdin=slave,
        stdout=slave,
        stderr=subprocess.PIPE,
        start_new_session=True,
        env=env,
    )
    os.close(slave)
    try:
        # Wait for the prompt: the reader is in non-canonical mode only after it.
        seen = b""
        deadline = time.monotonic() + READ_TIMEOUT_S * 3
        while b"input hidden" not in seen and time.monotonic() < deadline:
            seen += os.read(master, 4096)
        assert b"input hidden" in seen, seen
        time.sleep(0.1)

        def drain() -> None:
            # Keep reading the child's terminal output: the last close of a
            # tty waits for unread output to drain (measured: ~30 s on macOS
            # with one undrained "\r\n"), which would read as a hang.
            while True:
                try:
                    if not os.read(master, 4096):
                        return
                except OSError:
                    return

        threading.Thread(target=drain, daemon=True).start()
        os.write(master, typed)
        _, err = child.communicate(timeout=READ_TIMEOUT_S * 3)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
        os.close(master)
    line = next(row for row in err.decode().splitlines() if row.startswith("RESULT "))
    return json.loads(line[len("RESULT ") :])


def test_cli_paste_callback_reads_a_real_size_key_over_a_controlling_tty() -> None:
    text = _service_account_json()
    result = _run_child(text.encode().replace(b"\n", b"\r") + b"\r")
    assert json.loads(result["value"]) == json.loads(text)
    assert result["restored"] is True


def test_cli_paste_callback_ctrl_c_cancels_and_restores_the_terminal() -> None:
    result = _run_child(b'{\r  "partial": \x03')
    assert result == {"interrupted": True, "restored": True}
