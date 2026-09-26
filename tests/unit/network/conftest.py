"""Fixtures and helpers for the mesh core's unit tests.

Every test builds its OWN config root: the modules take a ``root`` argument
precisely so a test never has to monkeypatch ``HOME``, and the one place that
matters is identity — a test that minted a key into the operator's real store
would be a defect wearing a green tick.
"""

from __future__ import annotations

import argparse
import socket
import time
from pathlib import Path
from typing import Any, Callable

import pytest


def wait_for(predicate: Callable[[], bool], timeout_s: float = 15.0) -> bool:
    """Poll ``predicate`` until it holds, or until the deadline passes; report which.

    THE DEADLINE IS A BACKSTOP, NOT THE ASSERTION. The wait lasts exactly as long as
    the work does, and the bound exists so that a genuine hang fails the run instead of
    blocking it (see AGENTS.md, "Wait on the event, never on the clock"). What a caller
    has to get right is the PREDICATE: it must test the condition the assertion is
    about, not a proxy that merely happens to become true first.

    That distinction is what two cells in this package got wrong, and a loaded shard
    runner is what found it. The relay RECORDS on the thread that ACTS, and it acts
    first: ``_close_stream`` pops the stream table and closes the dial before it calls
    ``_report_stream_closed``, and the pairing listener sends its abort frame before it
    records ``pairing_refused``. A test that waits for the effect and then reads the
    trail once is therefore racing the row's own write with a margin of ZERO rather
    than a margin of its bound — measured at 1 ms of injected scheduling delay on one
    cell and 100 ms on the other, against starvation gaps of 525-668 ms recorded for
    this fleet.

    ONE SPELLING, ONE DEFAULT. This replaces the two module-local copies it was moved
    out of (``test_session_plane._wait_for``, 10 s, and
    ``test_credentials_real_link._wait_for``, 15 s): the shared default is the larger of
    the two, because a backstop that fires early under load is a flake of its own.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


def subcommands_of(parser: argparse.ArgumentParser) -> dict[str, Any]:
    """``{verb: subparser}`` for ``parser``, proven to be the dict argparse built.

    argparse types an action's ``choices`` as ``Iterable[Any] | None``, so a reader
    that KNOWS this mapping is the one ``add_subparsers`` created has to prove it:
    a re-derived ``getattr`` per use is not something a checker can narrow, and
    subscripting the union blind is the error this replaces. Shared because three
    of this package's test modules ask the same question of the real parser.
    """
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict):
            return choices
    raise AssertionError(f"{parser.prog}: no subcommands registered")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A private config root for one test."""
    return tmp_path


@pytest.fixture()
def socketpair() -> Any:
    """A connected pair of TCP sockets on loopback — real sockets, not a stub.

    The handshake's failure modes are about what goes on the wire (a silent close,
    a pipelined frame, a truncated read), so a fake stream would test the fake
    rather than the protocol.
    """
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    client = socket.create_connection(server.getsockname()[:2], timeout=5)
    accepted, _addr = server.accept()
    try:
        yield client, accepted
    finally:
        for sock in (client, accepted, server):
            try:
                sock.close()
            except OSError:
                pass
