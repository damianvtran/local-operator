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


async def _journals(created: Any, needle: str, timeout_s: float = 30.0) -> bool:
    """Whether the OWNER's journal holds ``needle``, waiting out the flush.

    ADMISSION IS NOT COMPLETION: a receipt that carries a ``request`` gets a turn
    started on the peer, and the journal line for that turn lands a moment AFTER
    the receipt comes back — while a read that lands inside the write raises rather
    than answering (CI saw ``JSONDecodeError`` at char 202 on exactly this cell).
    Both are timing, so the question is polled, and a malformed read is retried
    instead of reported as an answer.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while True:
        try:
            if await asyncio.to_thread(_said, created, needle):
                return True
        except ValueError:  # a partially flushed journal line: not an answer yet
            pass
        if loop.time() >= deadline:
            return False
        await asyncio.sleep(0.05)


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
            "--json",
            "--peer",
            "device-b",
            "--send",
            created.session_id,
            # FLAGS COME BEFORE THE PAYLOAD: the text is an argparse REMAINDER, so
            # a flag written after it is delivered as part of the prompt.
            "second turn from the shell",
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
            "--json",
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
            "--json",
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
        # diagnoses it. WHICH RUNG NOTICES IS A RACE — the reachability read can
        # refuse before any viewer is opened when its cached row already knows, and
        # the dial's own bind rung answers when the row was still fresh — so what
        # this cell pins is the CODE both rungs agree on, which is what a script
        # branches on (round 1, MINOR-4: a dial that produced no stream is the
        # device, and the guide's `session_unreachable` row moved to say so).
        assert '"code": "peer_unreachable"' in out, (out, err)
        combined = out + err
        assert "unreachable" in combined.lower(), combined
        assert "doctor" in combined, combined
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_flag_shaped_prompt_reaches_the_peers_journal_whole(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-1 MAJOR-1, through the REAL entry point over a real relay pair.

    ``--name`` is an option THIS subcommand declares, so the old parse consumed it
    as one: the owner ran a turn on ``check the`` and the command exited 0. The
    peer's own journal is what settles it — the words are in the record of the
    device that ran the turn, not in anything this side printed.
    """
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-argv",
        prompt="",
    )
    try:
        root_a = created.server_a.root
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--json",
            "--peer",
            "device-b",
            "--send",
            created.session_id,
            "check",
            "the",
            "--name",
            "field",
        )
        assert code == 0, (code, out, err)
        payload = json.loads(out)
        assert payload["outcome"] == "finished", payload
        assert await asyncio.to_thread(_said, created, "check the --name field"), _user_texts(
            created
        )
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_success_receipt_names_the_device_the_turn_ran_on(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """QA round 1, Q1, over a real relay pair: a wrong ``--peer`` still runs the turn
    (the session id is what ROUTES it), and the receipt says which device that was
    instead of echoing the caller's string back as though it were the answer.

    Both halves matter: the device named is the one the turn LANDED on — its own
    journal has the text below — and the caller's word is kept beside it so the
    mistake is visible rather than silently corrected.
    """
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-named",
        prompt="",
    )
    try:
        root_a = created.server_a.root
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--json",
            "--peer",
            "no-such-device",
            "--send",
            created.session_id,
            "name check",
        )
        assert code == 0, (code, out, err)
        payload = json.loads(out)
        assert payload["peer"] == "device-b", payload
        assert payload["peer_named"] == "no-such-device", payload
        assert await _journals(created, "name check"), _user_texts(created)
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_goal_slash_from_the_shell_runs_its_request_on_the_peer(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Round-1 MAJOR-2, end to end: an ACTION-carrying receipt runs on the device
    that owns the conversation.

    ``/goal``, ``/agent`` and ``/team`` answer with a receipt whose ``request`` the
    owner submits ITSELF — unless the client declared that it renders that
    receipt, which is what the attached vocabulary means and what this verb used to
    declare by default. A one-shot viewer that declares nothing gets the turn run
    over there, and the peer's journal is where that shows.
    """
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-goal",
        prompt="",
    )
    try:
        root_a = created.server_a.root
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--json",
            "--peer",
            "device-b",
            "--slash",
            created.session_id,
            "/goal wire the mesh end to end",
        )
        assert code == 0, (code, out, err)
        payload = json.loads(out)
        assert payload["ok"] is True, payload
        # THE RECEIPT IS THE OWNER'S, and so is the turn it asked for: the request
        # is in the peer's journal, which a declaring viewer suppresses.
        assert await _journals(created, "wire the mesh end to end"), _user_texts(created)
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_conversation_name_reaches_the_peer_over_a_real_pair(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§3B, over the wire: `--send <name>` resolves against the peer's OWN rows.

    The id-or-name read is the part no fake can prove end to end — the rows come
    from a real fan-out, and the name the selector matches is the one the OWNER
    published for its session. What the receipt must carry is the resolved id,
    and what the peer's journal must hold is the turn.
    """
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-by-name",
        prompt="first turn",
    )
    try:
        root_a = created.server_a.root
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--json",
            "--peer",
            "device-b",
            "--send",
            "pilot-by-name",
            "second turn by name",
        )
        assert code == 0, (code, out, err)
        payload = json.loads(out)
        assert payload["ok"] is True, payload
        assert payload["outcome"] == "finished", payload
        assert (
            payload["session_id"] == created.session_id
        ), "the receipt names the session the NAME resolved to"
        assert await asyncio.to_thread(_said, created, "second turn by name"), _user_texts(created)
    finally:
        await asyncio.to_thread(created.stop)


@pytest.mark.asyncio
async def test_a_peek_reads_a_live_tail_over_a_real_pair(
    peer_pair: Devices, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """§3C, over the wire: the tail window, resolved by name, served by the owner.

    A read with no fake in the path: the viewer is the real one, the rows come
    from the owner's synced display window over the real protocol, and the
    conversation's own words are what prove the window is the CONVERSATION's
    rather than an empty frame this side guessed at.
    """
    created = await asyncio.to_thread(
        _create_named_session_on_a_real_peer,
        peer_pair,
        monkeypatch,
        name="pilot-peek",
        prompt="first turn",
    )
    try:
        root_a = created.server_a.root
        (root_a / "config.yml").write_text(
            "version: 0.0.0\nvalues:\n  hosting: test\n  model_name: mock\n", encoding="utf-8"
        )
        code, out, err = await asyncio.to_thread(
            _run_cli,
            root_a,
            _home(tmp_path),
            "sessions",
            "--json",
            "--peer",
            "device-b",
            "--peek",
            "pilot-peek",
            "--steps",
            "5",
        )
        assert code == 0, (code, out, err)
        payload = json.loads(out)
        assert payload["ok"] is True and payload["verb"] == "peek", payload
        assert payload["session_id"] == created.session_id, payload
        assert payload["rows"], "the owner served no rows for a conversation with a turn"
        said = "\n".join(row["text"] for row in payload["rows"])
        assert "first turn" in said, payload
    finally:
        await asyncio.to_thread(created.stop)
