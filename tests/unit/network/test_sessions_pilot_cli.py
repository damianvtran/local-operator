"""``lop network sessions --send/--steer/--slash`` over a real pair of relays.

WHY A SUBPROCESS CELL, next to the faked-viewer tests in
``test_sessions_pilot.py``: those pin this verb's OWN decisions before a frame is
written, and a fake viewer cannot answer the two questions that matter most about
the verb — does a turn actually reach a conversation on the other device, and is
a device that may not do it refused with the peer's own sentence? Both are
properties of the wire and of the peer's authoriser, so this file drives the real
entry point against a real relay pair.

WHAT IT DOES NOT DUPLICATE. The pilot matrix over two real relays —
prompt/steer/rename/goal/model/stop landing on the peer's runtime — is
``test_remote_viewer.py``'s subject, and the frame-level capability gate is
``test_stream_op_gate.py``'s. These cells are the CLI's half of the same claims:
the flags reach that seam, and the refusals arrive as something a person (and a
script) can act on.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.network import store as network_store
from tests.unit.network.test_relay_e2e import devices  # noqa: F401 — fixtures
from tests.unit.network.test_session_plane import (
    Devices,
    _create_named_session_on_a_real_peer,
)

#: How long one CLI run gets: the child imports the whole application and, for a
#: prompt, waits for a turn on the peer.
RUN_TIMEOUT_S = 90.0


@pytest.fixture()
def peer_pair(request: pytest.FixtureRequest) -> Devices:
    pair: Devices = request.getfixturevalue("devices")
    return pair


def _run_cli(root: Path, home: Path, *args: str) -> tuple[int | None, str, str]:
    """Run ``lop network …`` as a user would, and reap it.

    The environment is built from scratch (``test_remote_resume_cli``'s rule):
    ``LOP_*`` decides what a child runtime IS and ``CMUX_*`` names the operator's
    real workspaces, so an inherited value would make this cell measure the
    session that started it.
    """
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(root),
        "TERM": "xterm-256color",
        "PYTHONPATH": str(Path(__file__).resolve().parents[3]),
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "local_operator.cli", "network", *args],
        env=env,
        cwd=str(home),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        out, err = process.communicate(timeout=RUN_TIMEOUT_S)
        return process.returncode, out, err
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        out, err = process.communicate()
        return None, out or "", err or ""


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "shell-home"
    home.mkdir(exist_ok=True)
    return home


def _user_texts(created: Any) -> list[str]:
    """Every user message the OWNER's transcript journalled, in order.

    The content is a list of blocks (``[{"text": …}]``), so it is dumped rather
    than stringified: ``str()`` renders the Python repr and a substring search
    against it then depends on the quoting, which is how a cell comes to pass for
    the wrong reason.
    """
    texts = []
    for entry in created.owner.transcript_entries():
        payload = entry.get("payload") or {}
        if payload.get("role") == "user":
            texts.append(json.dumps(payload.get("content")))
    return texts


def _said(created: Any, needle: str) -> bool:
    """Whether the OWNER's transcript holds ``needle`` in any user turn."""
    return any(needle in row for row in _user_texts(created))


@pytest.mark.asyncio
async def test_a_follow_up_turn_reaches_the_peer_from_a_shell(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The act the whole flag exists for: a turn into a conversation that is
    ALREADY there, driven from a shell rather than from the TUI that could
    already do it."""
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-cli",
        prompt="first turn",
    )
    try:
        root_a = created.server_a.root
        # A REAL DEVICE HAS A CONFIGURED PROVIDER and the CLI refuses to boot
        # without one, before any factory runs — without this the cell would
        # measure the "not configured" banner instead of the verb.
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        assert _said(created, "first turn"), _user_texts(created)

        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--peer",
            "device-b",
            "--send",
            created.session_id,
            "second turn from the shell",
            "--json",
        )
        assert code == 0, (code, out, err)
        payload = json.loads(out)
        assert payload["ok"] is True, payload
        assert payload["outcome"] == "finished", payload
        assert payload["session_id"] == created.session_id
        # THE OWNER RAN IT: the peer's own journal holds the turn the shell sent,
        # which is the claim a local-shadow implementation would fail.
        assert await asyncio.to_thread(_said, created, "second turn from the shell"), _user_texts(
            created
        )
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_device_without_the_prompt_capability_is_refused_in_words(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The adversarial half: a remote prompt is a remote execution path, and a peer
    that may not take one must say WHICH device and WHICH capability."""
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-caps",
        prompt="",
    )
    try:
        root_a = created.server_a.root
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        # The revoke is written straight into the PEER's own record, on A's row
        # there, because THAT is the authoriser which decides whether A's frames
        # may reach B's session. The product's own route to it is
        # `lop network member revoke`, which needs the device to be admin — B joins
        # this fixture's network as ``drive``, and pairing it as admin to reach the
        # granting verb would make this cell about the inviter's roles rather than
        # about the refusal under test. The granting verb's own round trip and its
        # audit rows are `test_member_caps.py`'s subject; what is pinned here is
        # what a device that lost the capability is TOLD.
        server_b = created.server_b
        record = network_store.list_networks(server_b.root)[0]
        a_device_id = created.server_a.identity.device_id
        with network_store.mutate(record.network_id, server_b.root) as fresh:
            member = fresh.member(a_device_id)
            assert member is not None, "the peer holds no row for this device"
            assert "prompt" in member.capabilities, member.capabilities
            member.capabilities = [cap for cap in member.capabilities if cap != "prompt"]
            network_store.save(fresh, server_b.root)
        reloaded = network_store.load(record.network_id, server_b.root).member(a_device_id)
        assert reloaded is not None
        assert "prompt" not in reloaded.capabilities

        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--peer",
            "device-b",
            "--send",
            created.session_id,
            "you may not take this",
            "--json",
        )
        assert code == 1, (code, out, err)
        # THE PEER'S OWN SENTENCE, and it names the device and the capability —
        # 'it does not hold the prompt capability' is what tells a person whether
        # to grant something or to go elsewhere.
        combined = out + err
        assert "capability" in combined, combined
        assert "device-b" in combined or "device-a" in combined, combined
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_an_unreachable_peer_fails_fast_and_truthfully(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A peer that is down is a `doctor` question, not a hang and not a success."""
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-gone",
        prompt="",
    )
    try:
        root_a = created.server_a.root
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        # The peer stops answering; A's own relay is still up, so the failure has
        # to come from the DIAL rather than from anything local.
        created.server_b.stop()
        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--peer",
            "device-b",
            "--send",
            created.session_id,
            "anyone there?",
            "--json",
        )
        assert code == 1, (code, out, err)
        # NOT a hang, NOT an empty answer, and NOT the transport's own words: the
        # DEVICE is named as the component that failed, with the command that
        # diagnoses it. The transport reports a stopped peer and a refused session
        # with one identical sentence ("the remote owner did not send its state"),
        # so this code is read from the peer's own catalogue rather than assumed —
        # which is exactly what makes it deterministic between a cached row and a
        # fresh one (CI caught the first version answering two ways for one state).
        assert '"code": "peer_unreachable"' in out, (out, err)
        combined = out + err
        assert "unreachable" in combined.lower(), combined
        assert "doctor" in combined, combined
    finally:
        await asyncio.to_thread(created.stop)
