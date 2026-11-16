"""A bounded local GitHub borrow timeout is visible, and lends nothing new.

WHY THIS FILE EXISTS. On 2026-11-16 a node's bash child ran without
``GH_TOKEN`` and ``gh`` said only "set the GH_TOKEN environment variable":
the local credential round trip had hit its existing 8 s deadline, and that
timeout was indistinguishable from every other empty-env answer. The fix adds
ONE stderr line for that typed timeout only. These cells drive the REAL bash
tool, the REAL ``MeshCredentialClient`` and a real loopback control socket
whose synthetic relay is silent, closes early, or refuses -- so the
discrimination is proven on the actual path, not on a fake outcome:

* timeout       -> env still empty, exactly one note;
* early close   -> plain no-answer, no note (not every ``owner_offline`` is a timeout);
* denial        -> the owner's refusal code, no note;
* policy exclude-> no note (the exclude wins over any borrowed login anyway).

Synthetic ids and a disposable loopback socket only; no live credentials.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.network import store
from local_operator.network.credentials import github, placement
from local_operator.network.credentials.client import MeshCredentialClient
from local_operator.network.credentials.types import BorrowTimeout, BrokerError
from local_operator.tools import builtin
from local_operator.variables import VariableStore
from tests.unit.network.test_credentials_github import (
    BORROWER_DEVICE,
    OWNER_DEVICE,
    _seed_borrower_network,
)

NOTE = "GitHub credential from synthetic-owner did not arrive within 8 s; retry the command"
CHILD = (
    'import os,sys; print("TOKEN_PRESENT", bool(os.environ.get("GH_TOKEN"))); '
    'print("ordinary stderr", file=sys.stderr)'
)


def _client(root: Path) -> MeshCredentialClient:
    _seed_borrower_network(root, "n_gh")
    document = placement.PlacementDocument("n_gh", root=root, written_by=OWNER_DEVICE)
    document.declare(
        "github",
        owner_device=OWNER_DEVICE,
        owner_device_name="synthetic-owner",
        provider="github",
        identity_label="",
        by=OWNER_DEVICE,
    )
    document.grant("github", BORROWER_DEVICE, scope="device", by=OWNER_DEVICE)
    document.save()
    client = MeshCredentialClient(
        root=root, self_device=BORROWER_DEVICE, network_id="n_gh", placement=document
    )
    # The production bound, unchanged: this is the deadline the node hit.
    assert client.deadline_s == 8.0
    return client


class _Relay:
    """One disposable loopback control socket answering ONE ask by ``behaviour``."""

    def __init__(self, behaviour: str) -> None:
        self.behaviour = behaviour
        self.listener = socket.socket()
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen()
        self.listener.settimeout(15)
        self.release = threading.Event()
        self.entered = threading.Event()
        self.errors: list[str] = []
        self.record = SimpleNamespace(
            control_port=self.listener.getsockname()[1], control_key="synthetic"
        )
        self.thread = threading.Thread(target=self._serve, name="synthetic-credential-control")

    def _serve(self) -> None:
        try:
            conn, _ = self.listener.accept()
            with conn:
                stream = conn.makefile("rb")
                assert json.loads(stream.readline())["key"] == "synthetic"
                assert json.loads(stream.readline())["op"] == "credential_grant"
                stream.close()
                self.entered.set()
                if self.behaviour == "deny":
                    reply = {"code": "not_a_holder", "message": "synthetic refusal"}
                    conn.sendall((json.dumps(reply) + "\n").encode())
                elif self.behaviour == "silent":
                    assert self.release.wait(15)
                # "close": fall through and close at once -- an early no-answer.
        except BaseException as exc:  # noqa: BLE001 — reported by the cell
            self.errors.append(type(exc).__name__)
        finally:
            self.listener.close()

    def __enter__(self) -> "_Relay":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release.set()
        self.thread.join(3)
        with contextlib.suppress(OSError):
            self.listener.close()
        assert not self.thread.is_alive() and not self.errors


def _run_bash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, behaviour: str
) -> tuple[str, Any, float]:
    client = _client(tmp_path)
    outcomes: list[Any] = []
    real = client.request_grant_sync

    def recording(*args: Any, **kwargs: Any) -> Any:
        outcome = real(*args, **kwargs)
        outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(client, "request_grant_sync", recording)
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(github, "_client_for", lambda *a: client)
    context = ToolContext(
        cwd=str(tmp_path), variables=VariableStore(cwd=str(tmp_path)), session_id="synthetic"
    )
    command = shlex.join([sys.executable, "-c", CHILD])

    async def run() -> Any:
        # A coroutine wrapper: the guarded tool returns an awaitable.
        return await builtin.execute_bash("timeout-note", {"command": command}, None, None, context)

    with _Relay(behaviour) as relay:
        monkeypatch.setattr(store, "find_own_relay", lambda *a: relay.record)
        started = time.monotonic()
        result = asyncio.run(run())
        elapsed = time.monotonic() - started
        assert relay.entered.is_set(), "the real client never asked the control socket"
    assert len(outcomes) == 1
    return result.text, outcomes[0], elapsed


def test_a_real_borrow_timeout_adds_one_note_and_the_env_stays_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text, outcome, elapsed = _run_bash(tmp_path, monkeypatch, "silent")
    assert isinstance(outcome, BorrowTimeout) and outcome.code == "owner_offline"
    assert not getattr(outcome, "access_token", "")
    assert elapsed >= 8.0, "the existing deadline must not change"
    assert "TOKEN_PRESENT False" in text
    assert "ordinary stderr" in text
    assert text.count(NOTE) == 1, text


def test_an_early_close_is_not_a_timeout_and_shows_no_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text, outcome, elapsed = _run_bash(tmp_path, monkeypatch, "close")
    assert type(outcome) is BrokerError and outcome.code == "owner_offline"
    assert elapsed < 8.0
    assert "TOKEN_PRESENT False" in text
    assert "did not arrive" not in text, text


def test_a_denial_keeps_its_own_refusal_and_shows_no_timeout_note(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text, outcome, _ = _run_bash(tmp_path, monkeypatch, "deny")
    assert type(outcome) is BrokerError and outcome.code == "not_a_holder"
    assert "TOKEN_PRESENT False" in text
    assert "did not arrive" not in text, text


def test_a_policy_exclude_shows_no_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from local_operator.tools import shell_env

    (tmp_path / "config.yml").write_text("values:\n  shell_environment:\n    exclude: [GH_TOKEN]\n")
    # The policy is frozen per config dir; forget it before and after this cell
    # (shell_env's documented test seam) so no other cell inherits the exclude.
    shell_env.reset_policy_cache()
    try:
        text, outcome, _ = _run_bash(tmp_path, monkeypatch, "silent")
        # Rig check: the exclude really was the policy in force, so "no note"
        # is the suppression and not an unread config.
        assert "GH_TOKEN" in shell_env.load_policy().exclude
    finally:
        shell_env.reset_policy_cache()
    assert isinstance(outcome, BorrowTimeout)
    assert "TOKEN_PRESENT False" in text
    assert "did not arrive" not in text, text


def test_the_door_reports_only_a_typed_timeout_and_never_without_a_sink() -> None:
    """``lent_github_env``'s fences are unchanged: no sink means no borrow at all."""
    timeout = BorrowTimeout(code="owner_offline", key="github", timeout_s=8.0)
    plain = BrokerError(code="owner_offline", key="github")
    seen: list[Any] = []

    class _Client:
        def __init__(self, outcome: Any) -> None:
            self.outcome = outcome

        def should_borrow(self, key: str) -> bool:
            return True

        def request_grant_sync(self, *args: Any, **kwargs: Any) -> Any:
            return self.outcome

    assert github.lent_github_env(
        session_id="s", register=None, client=_Client(timeout), on_timeout=seen.append
    ) == ({}, "")
    assert seen == [], "a caller that cannot scrub must not even trigger a borrow"
    assert github.lent_github_env(
        session_id="s", register=lambda t: None, client=_Client(plain), on_timeout=seen.append
    ) == ({}, "")
    assert seen == [], "a plain no-answer is not a timeout"
    assert github.lent_github_env(
        session_id="s", register=lambda t: None, client=_Client(timeout), on_timeout=seen.append
    ) == ({}, "")
    assert seen == [timeout]


def test_control_request_keeps_none_for_callers_that_do_not_opt_in() -> None:
    """Every other caller of ``control_request`` still reads ``None`` on a timeout."""
    from local_operator.network.relay import control_request
    from local_operator.network.types import ControlRequestTimeout

    for opt_in in (False, True):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        held: list[socket.socket] = []
        accept = threading.Thread(target=lambda: held.append(listener.accept()[0]))
        accept.start()
        record: Any = SimpleNamespace(control_port=listener.getsockname()[1], control_key="k")
        try:
            if opt_in:
                with pytest.raises(ControlRequestTimeout) as caught:
                    control_request(record, "credential_grant", timeout=0.3, report_timeout=True)
                assert caught.value.timeout_s == 0.3
            else:
                assert control_request(record, "credential_grant", timeout=0.3) is None
        finally:
            accept.join(3)
            for conn in held:
                conn.close()
            listener.close()
