"""`lop browser pair` from the COLD state: an install connected, unpaired, no live code.

The operator report this file exists for (2026-11-02): `lop browser pair` printed
"no extension is waiting to pair" while `lop browser status` said the extension
was connected. The daemon minted a code only at the extension's handshake, with a
two-minute lifetime, so an install that had connected more than two minutes
earlier and never paired had no live code — and nothing the CLI could do would
produce one. The only way out was typing a WRONG code in the popup to make the
daemon rotate a fresh one, which nobody can be expected to guess.

These rows run the REAL daemon (``uvicorn`` in a child process, on a port the OS
picks, under a throwaway config root) and the REAL CLI entry point, with a
scripted WebSocket standing in for the extension's worker. The code's expiry is
produced by rewriting the pending file's ``expires_at`` — the daemon re-reads
that file on every decision, so this is the state a real clock would reach,
without a 15-minute sleep.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from contextlib import ExitStack, suppress
from pathlib import Path
from typing import Any, Iterator

import pytest
from websockets.sync.client import ClientConnection
from websockets.sync.client import connect as sync_connect

from local_operator.browser_bridge.protocol import PROTO_VERSION

STORE_ID = "omibaecbjdhgbbcedbnnnmjpmopfheof"
UNPACKED_ID = "jbadjeaodkoboanppmpjiifpconegdcj"

_CHILD = """
import sys
import uvicorn
from local_operator.browser_bridge.daemon import create_app
port = int(sys.argv[1])
uvicorn.run(create_app(port), host="127.0.0.1", port=port, log_level="warning")
"""


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def _health(port: int) -> dict[str, Any]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2.0) as response:
        return json.loads(response.read().decode())


def _env(root: Path) -> dict[str, str]:
    # The child daemon and the CLI must share ONE config root, and neither may
    # see the operator's: HOME and the config dir both point into tmp_path.
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("CMUX_", "LOP_", "LOCAL_OPERATOR_"))
    }
    env["HOME"] = str(root)
    env["LOCAL_OPERATOR_CONFIG_DIR"] = str(root / ".local-operator")
    env["NO_COLOR"] = "1"
    env["LOCAL_OPERATOR_LANGUAGE"] = "en"
    return env


class _Daemon:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = root / ".local-operator"
        self.port = _free_port()
        self.env = _env(root)
        self.child = subprocess.Popen(
            [sys.executable, "-c", _CHILD, str(self.port)],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            with suppress(Exception):
                if _health(self.port)["status"] == "ok":
                    return
            time.sleep(0.1)
        self.stop()
        raise AssertionError("the child daemon never answered /health")

    def stop(self) -> None:
        if self.child.poll() is None:
            self.child.terminate()
            try:
                self.child.wait(timeout=10)
            except subprocess.TimeoutExpired:  # pragma: no cover - reaped below
                self.child.kill()
                self.child.wait(timeout=5)

    def extension(self, extension_id: str, *, version: str = "0.1.26") -> ClientConnection:
        """A scripted worker: dial, say hello with no token, read the ack."""
        wire = sync_connect(
            f"ws://127.0.0.1:{self.port}/extension",
            additional_headers={"Origin": f"chrome-extension://{extension_id}"},
            open_timeout=5.0,
        )
        wire.send(
            json.dumps(
                {
                    "event": "hello",
                    "proto": PROTO_VERSION,
                    "token": "",
                    "extension_version": version,
                    "browser": "Mozilla/5.0 Chrome/153.0.0.0",
                }
            )
        )
        ack = json.loads(wire.recv(timeout=5.0))
        assert ack["event"] == "hello_ack" and ack["paired"] is False, ack
        return wire

    def pending_file(self) -> Path:
        return self.config / "run" / "browser" / "pairing-pending.json"

    def expire_every_code(self) -> None:
        """Move every live code's expiry into the past, as the clock would."""
        path = self.pending_file()
        record = json.loads(path.read_text())
        for entry in record["pending"].values():
            entry["expires_at"] = time.time() - 1.0
        path.write_text(json.dumps(record))

    def codes(self) -> dict[str, str]:
        record = json.loads(self.pending_file().read_text())
        return {
            key: str(entry["code"])
            for key, entry in record["pending"].items()
            if float(entry["expires_at"]) > time.time()
        }

    def pair(self, *extra: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "local_operator.cli", "browser", "pair", *extra],
            env=self.env,
            cwd=self.root,
            capture_output=True,
            text=True,
            timeout=60,
        )


@pytest.fixture
def daemon(tmp_path: Path) -> Iterator[_Daemon]:
    # `lop browser pair` reads the daemon's discovery file (port + session key)
    # from the config root, which the child publishes on startup.
    running = _Daemon(tmp_path)
    try:
        yield running
    finally:
        running.stop()


def _next_event(wire: ClientConnection, name: str) -> dict[str, Any]:
    """Read frames until one named ``name`` arrives; earlier frames are returned
    nowhere (the `show_pairing` nudge `pair` sends lands first)."""
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        frame = json.loads(wire.recv(timeout=max(0.1, deadline - time.monotonic())))
        if frame.get("event") == name:
            return frame
    raise AssertionError(f"no {name} frame within 5s")


def _six_digit_codes(text: str) -> list[str]:
    import re

    return re.findall(r"(?<!\d)\d{6}(?!\d)", text)


def test_cold_state_pair_prints_a_fresh_code_without_any_popup(daemon: _Daemon) -> None:
    """THE REPORTED STATE: connected, unpaired, code expired, no popup anywhere.

    Before the fix this printed "no extension is waiting to pair" and exited 1.
    """
    with daemon.extension(STORE_ID):
        daemon.expire_every_code()
        assert daemon.codes() == {}, "precondition: no live code exists"

        result = daemon.pair()

        assert result.returncode == 0, result.stdout + result.stderr
        live = daemon.codes()
        assert list(live) == [STORE_ID], "the CLI must mint a code for the connected install"
        assert live[STORE_ID] in _six_digit_codes(result.stdout), result.stdout
        assert "no extension is waiting" not in result.stdout
        # Says where to type it and how long it lasts, in plain words.
        assert "Local Operator icon" in result.stdout, result.stdout
        assert "15 minutes" in result.stdout, result.stdout


def test_running_pair_twice_shows_the_same_live_code(daemon: _Daemon) -> None:
    """A second run must not rotate a code the user may already be typing."""
    with daemon.extension(STORE_ID):
        daemon.expire_every_code()
        first = daemon.pair()
        second = daemon.pair()
        assert first.returncode == second.returncode == 0
        assert _six_digit_codes(first.stdout) == _six_digit_codes(second.stdout)
        assert daemon.codes()[STORE_ID] in _six_digit_codes(second.stdout)


def test_two_unpaired_installs_each_get_a_labelled_code(daemon: _Daemon) -> None:
    with ExitStack() as stack:
        stack.enter_context(daemon.extension(STORE_ID, version="0.1.26"))
        stack.enter_context(daemon.extension(UNPACKED_ID, version="0.1.20"))
        daemon.expire_every_code()

        result = daemon.pair()

        assert result.returncode == 0, result.stdout
        live = daemon.codes()
        assert set(live) == {STORE_ID, UNPACKED_ID}
        for extension_id, code in live.items():
            line = next(row for row in result.stdout.splitlines() if code in row)
            # Each code names its install: browser + extension version + short id.
            assert extension_id[:8] in line, line
        assert "0.1.20" in result.stdout and "0.1.26" in result.stdout
        # The OLDER install gets the update instruction; it still pairs.
        assert "chrome://extensions" in result.stdout, result.stdout


def test_no_extension_connected_says_how_to_fix_it(daemon: _Daemon) -> None:
    result = daemon.pair()
    assert result.returncode == 1
    assert _six_digit_codes(result.stdout) == []
    assert "open the extension popup" not in result.stdout.lower()
    assert "Chrome" in result.stdout and "chromewebstore.google.com" in result.stdout, (
        result.stdout
    )


def test_a_code_minted_by_the_cli_pairs_the_install(daemon: _Daemon) -> None:
    """End to end over the wire: the printed code is what authorises the install."""
    with daemon.extension(STORE_ID) as wire:
        daemon.expire_every_code()
        printed = _six_digit_codes(daemon.pair().stdout)
        assert len(printed) == 1
        wire.send(json.dumps({"event": "pair", "code": printed[0]}))
        verdict = _next_event(wire, "pair_result")
        assert verdict["event"] == "pair_result" and verdict["ok"] is True, verdict
    # Already paired now, and the CLI says so instead of minting anything.
    again = daemon.pair()
    assert again.returncode == 0
    assert _six_digit_codes(again.stdout) == []
    assert "already paired" in again.stdout.lower(), again.stdout


def test_pair_asks_the_install_to_show_its_page_and_never_sends_it_the_code(
    daemon: _Daemon,
) -> None:
    """The nudge that opens the pairing tab carries NO code.

    The pairing property is that the code travels terminal → person → browser.
    If the daemon pushed the code to the extension, a rogue local process that
    dialled with a forged extension Origin could read it off the wire and pair
    itself; the nudge must be a bare event.
    """
    with daemon.extension(STORE_ID) as wire:
        result = daemon.pair()
        code = _six_digit_codes(result.stdout)[0]
        nudge = _next_event(wire, "show_pairing")
        assert nudge == {"event": "show_pairing"}
        assert code not in json.dumps(nudge)


def test_the_codes_endpoint_refuses_a_caller_without_the_session_key(daemon: _Daemon) -> None:
    with daemon.extension(STORE_ID):
        daemon.expire_every_code()
        for headers in ({}, {"X-Bridge-Key": "not-the-key"}):
            request = urllib.request.Request(
                f"http://127.0.0.1:{daemon.port}/pairing/codes",
                data=b"{}",
                method="POST",
                headers={"Content-Type": "application/json", **headers},
            )
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(request, timeout=5.0)
            assert refused.value.code == 401
            assert code_free(refused.value.read().decode())
        # And the refusal minted nothing: a stranger cannot even make a code exist.
        assert daemon.codes() == {}


def code_free(text: str) -> bool:
    return _six_digit_codes(text) == []


def test_the_copy_lifetime_matches_the_daemon_ttl() -> None:
    from local_operator.browser_bridge.daemon import PAIR_TTL_S, PENDING_RETENTION_S
    from local_operator.browser_bridge.protocol import PAIR_TTL_MINUTES

    assert PAIR_TTL_S == 15 * 60
    assert PAIR_TTL_MINUTES * 60 == PAIR_TTL_S
    # A record outlives its code by one TTL so "that code expired" stays honest.
    assert PENDING_RETENTION_S == 2 * PAIR_TTL_S


def test_an_expired_code_says_it_expired_and_is_replaced(daemon: _Daemon) -> None:
    from local_operator.browser_bridge.protocol import PAIR_EXPIRED_MESSAGE

    with daemon.extension(STORE_ID) as wire:
        code = _six_digit_codes(daemon.pair().stdout)[0]
        daemon.expire_every_code()
        wire.send(json.dumps({"event": "pair", "code": code}))
        verdict = _next_event(wire, "pair_result")
        assert verdict["ok"] is False
        assert verdict["message"] == PAIR_EXPIRED_MESSAGE, verdict
        fresh = _six_digit_codes(daemon.pair().stdout)[0]
        assert daemon.codes()[STORE_ID] == fresh


def test_an_expired_code_typed_over_a_fresh_socket_still_says_expired(daemon: _Daemon) -> None:
    """The extension's real submit path: a NEW socket (hello, then pair).

    The handshake used to rotate the expired code first, so the guess was
    compared against a brand-new code and the person read "didn't match".
    """
    from local_operator.browser_bridge.protocol import PAIR_EXPIRED_MESSAGE

    with daemon.extension(STORE_ID):
        code = _six_digit_codes(daemon.pair().stdout)[0]
        daemon.expire_every_code()
        with daemon.extension(STORE_ID) as submit:
            submit.send(json.dumps({"event": "pair", "code": code}))
            verdict = _next_event(submit, "pair_result")
        assert verdict["ok"] is False
        assert verdict["message"] == PAIR_EXPIRED_MESSAGE, verdict
    # ...and it was replaced, so a re-run prints a code that works. The submit
    # socket evicted the worker's link (later-connection-wins within one
    # identity); the real worker re-dials within a second, modelled here.
    with daemon.extension(STORE_ID):
        again = daemon.pair()
        printed = _six_digit_codes(again.stdout)
        assert printed and printed[0] != code, again.stdout
