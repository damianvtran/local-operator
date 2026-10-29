"""A sync re-copy of the SAME account is never an account change.

WHY (measured on the live mesh, 2026-10-21): a member session on cloud-node-1
carried dozens of ``session_credential_binding_notice.v1`` rows reading "This
session is now using the deepseek login on <host>", one per ~15 s. Each sync
re-copy replaced the copy's store row (insert new, sweep old), the recorder
keyed "effective identity" on that row id, and every new id took the
"same device, different row" arm of the notice renderer.

These cells pin the fix's three halves: the recorder compares a STABLE account
identity (the copy's origin marker) instead of the row id; a same-account change
journals no notice even across the upgrade edge; and the notice row, whatever
its history, never reaches the provider payload or the context estimate.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from local_operator.harness.message_types import SESSION_BINDING_NOTICE_MESSAGE_TYPE
from local_operator.harness.types import ChatRequest, CustomMessage, Message, ModelSpec
from local_operator.session.credential_binding import (
    SESSION_BINDING_CUSTOM_TYPE,
    CredentialBinding,
    CredentialBindingRecorder,
    is_same_account,
    mesh_copy_account,
)
from local_operator.session.transcript import Transcript

SELF = "d_" + "1" * 32
OWNER = "d_" + "2" * 32
OTHER_OWNER = "d_" + "3" * 32
PROVIDER = "deepseek"
SESSION = "sess-same-account"
NOTICE_TEXT = "[session credential] This session is now using the deepseek login on node."


def _copy_row(row_id: int, owner: str = OWNER) -> Any:
    """A store row shaped like an applied mesh copy (the origin marker it carries)."""
    return SimpleNamespace(
        id=row_id,
        data={"key": "sk-synthetic", "mesh_origin": {"owner_device": owner, "key": PROVIDER}},
    )


def _rows(transcript: Transcript) -> list[dict[str, Any]]:
    out = []
    for line in transcript.path.read_text().splitlines():
        payload = json.loads(line).get("payload") or {}
        if payload.get("custom_type") == SESSION_BINDING_CUSTOM_TYPE:
            out.append(payload["details"])
    return out


def _recorder(
    transcript: Transcript, rows: dict[int, Any], changes: list[Any]
) -> CredentialBindingRecorder:
    recorder = CredentialBindingRecorder(
        transcript,
        session_id=SESSION,
        device_id=SELF,
        device_name="node",
        on_change=lambda new, previous: changes.append((new, previous)),
    )
    # The production resolver (``attach_credential_binding``), over a fake store.
    recorder.set_account_resolver(lambda credential_id: mesh_copy_account(rows.get(credential_id)))
    return recorder


@pytest.mark.asyncio
async def test_a_re_copied_row_of_the_same_account_appends_nothing(tmp_path: Path) -> None:
    """Row ids 10, 11, 12 ... for the same copy: ONE binding row, zero notices.

    Pre-fix this wrote one row and fired one change per id (the per-tick flood).
    """
    transcript = Transcript(tmp_path / "sess")
    rows = {row_id: _copy_row(row_id) for row_id in range(10, 20)}
    changes: list[Any] = []
    recorder = _recorder(transcript, rows, changes)
    for row_id in rows:
        recorder.observe_local(provider=PROVIDER, credential_id=row_id)
        await recorder.drain()
    written = _rows(transcript)
    assert len(written) == 1, written
    assert written[0]["account"] == f"mesh:{OWNER}:{PROVIDER}"
    assert [previous for _new, previous in changes] == [None], "only the first serve signals"


@pytest.mark.asyncio
async def test_a_real_switch_still_records_and_signals_once(tmp_path: Path) -> None:
    """A genuinely different account (another owner's copy, or a local login) is news."""
    transcript = Transcript(tmp_path / "sess")
    rows = {10: _copy_row(10), 11: _copy_row(11, owner=OTHER_OWNER), 12: SimpleNamespace(data={})}
    changes: list[Any] = []
    recorder = _recorder(transcript, rows, changes)
    for row_id in (10, 11, 12):
        recorder.observe_local(provider=PROVIDER, credential_id=row_id)
        await recorder.drain()
    assert [row["credential_id"] for row in _rows(transcript)] == [10, 11, 12]
    signalled = [(new.credential_id, previous.credential_id) for new, previous in changes[1:]]
    assert signalled == [(11, 10), (12, 11)]


@pytest.mark.asyncio
async def test_the_upgrade_edge_records_identity_without_a_notice(tmp_path: Path) -> None:
    """A row an older build wrote (no ``account``) then a re-copy: recorded, NOT announced."""
    transcript = Transcript(tmp_path / "sess")
    legacy = CredentialBinding(
        provider=PROVIDER, owner_device=SELF, owner_device_name="node", credential_id=9
    )
    await transcript.append_custom(SESSION_BINDING_CUSTOM_TYPE, legacy.to_details())
    transcript = Transcript(transcript.directory)
    changes: list[Any] = []
    recorder = _recorder(transcript, {9: _copy_row(9), 10: _copy_row(10)}, changes)
    recorder.observe_local(provider=PROVIDER, credential_id=10)
    await recorder.drain()
    assert len(_rows(transcript)) == 1, "proven same-account replacement appends nothing"
    assert changes == [], "the first identity-bearing row is not an account switch"


def test_is_same_account_is_the_notice_gate() -> None:
    account = f"mesh:{OWNER}:{PROVIDER}"

    def _b(*, credential_id: int = 1, owner: str = SELF, acct: str = "") -> CredentialBinding:
        return CredentialBinding(
            provider=PROVIDER, owner_device=owner, credential_id=credential_id, account=acct
        )

    same = _b(acct=account)
    assert is_same_account(same, _b(credential_id=2, acct=account))
    assert not is_same_account(_b(), same)
    assert not is_same_account(same, _b(acct="mesh:x:y"))
    assert not is_same_account(_b(), _b())
    assert not is_same_account(same, _b(owner=OWNER, acct=account))


@pytest.mark.asyncio
async def test_notice_and_binding_rows_never_reach_the_provider_payload(tmp_path: Path) -> None:
    """The exclusion, proved on the WIRE BODY a real provider client builds.

    ``render.py`` documents that unlisted custom types are dropped; this cell
    replays a transcript carrying a flood of notices plus binding rows through
    ``build_llm_history`` and the production converter, then builds the actual
    OpenAI-compatible request body, and asserts neither the notice text nor the
    row type is anywhere in it — nor in the context estimate's input.
    """
    from local_operator.compaction.tokens import estimate_messages_tokens
    from local_operator.harness.render import _default_convert_to_llm
    from local_operator.providers.clients import OpenAICompatClient

    transcript = Transcript(tmp_path / "sess")
    await transcript.append_message(Message.user("Synthetic question"))
    for row_id in range(3):
        binding = CredentialBinding(provider=PROVIDER, owner_device=SELF, credential_id=row_id + 1)
        await transcript.append_custom(SESSION_BINDING_CUSTOM_TYPE, binding.to_details())
        await transcript.append_message(
            CustomMessage(
                custom_type=SESSION_BINDING_NOTICE_MESSAGE_TYPE,
                attribution="system",
                details={"text": NOTICE_TEXT},
            )
        )
    history = Transcript(transcript.directory).build_llm_history()
    rendered = _default_convert_to_llm(history)
    clean = _default_convert_to_llm([Message.user("Synthetic question")])
    assert estimate_messages_tokens(rendered) == estimate_messages_tokens(clean)

    request = ChatRequest(
        model=ModelSpec(provider="openai", model_id="gpt-synthetic"),
        system_blocks=["Synthetic prefix"],
        messages=rendered,
    )
    body = OpenAICompatClient(base_url="https://provider.example.invalid/v1")._build_body(request)
    wire = json.dumps(body)
    assert "Synthetic question" in wire, "the cell must witness a real body"
    assert "session credential" not in wire
    assert SESSION_BINDING_NOTICE_MESSAGE_TYPE not in wire
    assert SESSION_BINDING_CUSTOM_TYPE not in wire


@pytest.mark.asyncio
@pytest.mark.parametrize("prior_readable", [True, False])
async def test_local_to_mesh_or_unknown_upgrade_is_a_real_signal(
    tmp_path: Path, prior_readable: bool
) -> None:
    transcript = Transcript(tmp_path / "sess")
    changes: list[Any] = []
    rows = {10: SimpleNamespace(data={"type": "api_key", "key": "local-key"}), 11: _copy_row(11)}
    recorder = _recorder(transcript, rows, changes)
    recorder.observe_local(provider=PROVIDER, credential_id=10)
    await recorder.drain()
    if not prior_readable:
        rows.pop(10)  # A swept/missing old row is not positive provenance.
    recorder.observe_local(provider=PROVIDER, credential_id=11)
    await recorder.drain()
    recorder.observe_local(provider=PROVIDER, credential_id=11)
    await recorder.drain()
    assert [row["credential_id"] for row in _rows(transcript)] == [10, 11]
    assert len(changes) == 2 and changes[1][1].credential_id == 10
