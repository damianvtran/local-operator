"""Tiny plain-text affordance, exact existing storage/approval/privacy contracts."""

import json

import pytest

from local_operator.harness.types import ToolContext
from local_operator.memory import MemoryIdentity, MemoryStore
from local_operator.tools.memory_tools import (
    build_memory_tool,
    execute_fact_memory,
    execute_memory,
)
from local_operator.tools.registry import create_tools


@pytest.mark.asyncio
async def test_store_recall_and_existing_url_correction_share_one_engine(tmp_path):
    root = tmp_path / "cfg"
    context = ToolContext(
        session_id="tiny", memory_config_dir=str(root), session_dir=str(root / "sessions" / "tiny")
    )
    created = await execute_fact_memory(
        "a",
        {
            "op": "store",
            "text": "Elara Venn lives in Dunedin; contact elara@example.invalid",
        },
        context=context,
    )
    assert not created.is_error
    item = json.loads(str(getattr(created.content[0], "text", "")))
    bank = MemoryStore(root, MemoryIdentity("tiny"))
    assert bank.show("global", item["id"])["fact"]["source"]["kind"] == "reported"
    assert bank.show("global", item["id"])["fact"]["verified_at"] is None
    recall = await execute_fact_memory("r", {"op": "recall", "text": "Elara Venn"}, context=context)
    assert not recall.is_error and item["id"] in str(recall.content)
    from local_operator.tools.memory_tools import execute_memory

    changed = await execute_memory(
        "e",
        "edit",
        {
            "path": "memory://global/" + item["id"],
            "old_text": item["version"],
            "new_text": json.dumps(
                {"title": "Elara corrected", "body": "Elara lives in Wellington"}
            ),
        },
        context,
    )
    assert not changed.is_error
    assert bank.show("global", item["id"])["fact"]["body"] == "Elara lives in Wellington"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["disabled", "legacy-scope", "quota", "oversize"])
async def test_refusals_preserve_existing_policy_and_never_truncate(tmp_path, monkeypatch, case):
    import local_operator.memory as memory

    root = tmp_path / "cfg"
    root.mkdir()
    if case == "disabled":
        (root / "config.yml").write_text('{"values":{"memory":{"enabled":false}}}')
    if case == "quota":
        monkeypatch.setattr(memory, "MAX_OWNER_OPS", 0)
    args = {
        "op": "store",
        "text": "word " * 400 if case == "oversize" else "Kestrel release goal",
    }
    if case == "legacy-scope":
        args["scope"] = "global"
    response = await execute_fact_memory(
        "r", args, context=ToolContext(session_id="s", agent_id="main", memory_config_dir=str(root))
    )
    assert response.is_error
    assert MemoryStore(root, MemoryIdentity()).inventory()["items"] == []


def test_default_advertised_but_explicit_allowlist_not_widened_and_approval_tiers():
    tool = build_memory_tool()
    assert tool.approval_tier == "write"
    assert tool.call_approval_tier is not None
    assert tool.call_approval_tier({"op": "recall"}) == "read"
    assert tool.call_approval_tier({"op": "store"}) == "write"
    assert "memory" in [t.name for t in create_tools(ToolContext())]
    assert "memory" not in [t.name for t in create_tools(ToolContext(), enabled=["read"])]


def test_schema_published_with_intent_stays_under_250_tokens():
    from local_operator.compaction.tokens import count_text_tokens

    tool = next(t for t in create_tools(ToolContext()) if t.name == "memory")
    envelope = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }
    text = json.dumps(envelope)
    assert "i" in tool.parameters["properties"]
    assert count_text_tokens(text) <= 250
    assert len(text) / 2.78 <= 250  # billed-density control, not only a generic tokenizer


def test_malformed_dedicated_args_are_recognised_and_never_assume_a_path():
    from local_operator.memory_privacy import is_memory, retained_arguments

    for args in (
        None,
        ["SSN 123-45-6789"],
        {"text": "SSN 123-45-6789"},
        {"op": "invalid", "scope": "invalid", "text": "a3bf" * 10},
    ):
        assert is_memory("memory", args)
        safe, raw, masked = retained_arguments(
            "memory", args, json.dumps({"payload": args, "alternate": "SSN 123-45-6789"})
        )
        assert masked and safe == {"memory_arguments_redacted": True}
        assert raw is not None
        assert "6789" not in raw and "a3bf" not in raw


def test_url_scheme_is_not_a_relative_windows_drive_in_protected_cwd(tmp_path, monkeypatch):
    from local_operator.memory import MemoryError, guard
    from local_operator.memory_privacy import retained_arguments

    protected = tmp_path / "config"
    protected.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(protected))
    monkeypatch.chdir(protected)
    args = {
        "op": "store",
        "text": "Elara Venn: https://example.invalid/threads/kestrel",
    }
    assert not retained_arguments("memory", args, json.dumps(args))[2]
    guard("memory://team?q=Kestrel")
    with pytest.raises(MemoryError, match="Sensitive path"):
        guard("Read ~/.ssh/id_rsa")


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd_kind", ["repo", "scratch", "neutral"])
async def test_personal_contract_arguments_replay_in_each_cwd(tmp_path, monkeypatch, cwd_kind):
    from pathlib import Path

    from local_operator.memory_privacy import retained_arguments

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = (
        root
        if cwd_kind == "scratch"
        else Path("/") if cwd_kind == "neutral" else Path(__file__).resolve().parents[2]
    )
    monkeypatch.chdir(cwd)
    args = {
        "i": "Storing Elara Venn facts",
        "op": "store",
        "text": (
            "Elara Venn lives in Dunedin; colleague Mira Vale is reachable at "
            "mira@example.invalid and https://example.invalid/people/mira"
        ),
    }
    assert not retained_arguments("memory", args, json.dumps(args))[2]
    context = ToolContext(session_id="replay", memory_config_dir=str(root))
    result = await execute_fact_memory("captured-call", args, context=context)
    assert not result.is_error, result
    bank = MemoryStore(root, MemoryIdentity("replay"))
    rows = bank.search("Elara", "global")
    assert rows[0]["body"] == args["text"]
    assert rows[0]["source"]["kind"] == "reported"


@pytest.mark.asyncio
async def test_host_denial_prevents_the_named_tool_from_writing(tmp_path):
    from local_operator.harness.loop import AgentLoop, LoopContext
    from local_operator.harness.types import Message, StreamEndEvent, StreamTextDelta
    from tests.unit.harness.test_loop import (
        ScriptedStream,
        make_config,
        tool_call_delta,
    )

    denied = []

    async def deny(name, summary):
        denied.append(name)
        return False

    stream = ScriptedStream(
        [
            [
                tool_call_delta(
                    0,
                    id="store-denied",
                    name="memory",
                    args=json.dumps({"op": "store", "text": "Kestrel goal ships Friday"}),
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ],
            [StreamTextDelta(delta="Not stored"), StreamEndEvent(stop_reason="stop")],
        ]
    )
    context = LoopContext(
        tools=[build_memory_tool()],
        tool_context=ToolContext(request_approval=deny, memory_config_dir=str(tmp_path)),
    )
    result = await AgentLoop().run_to_end(
        [Message.user("Remember the rollout goal")], context, make_config(stream), None
    )
    assert denied == ["memory"]
    assert any(m.is_error for m in result if isinstance(m, Message) and m.role == "tool")
    assert not (tmp_path / "memory").exists()


def test_native_schema_is_explicit_durable_personal_domain_not_a_scope_choice():
    tool = build_memory_tool()
    assert set(tool.parameters["properties"]) == {"op", "text"}
    assert tool.parameters["required"] == ["op", "text"]
    assert "durable personal facts across sessions" in tool.description
    assert "user/colleague names" in tool.description
    assert tool.describe_approval is not None
    assert "global memory (across sessions)" in tool.describe_approval({"op": "store"}, "")


@pytest.mark.asyncio
async def test_explicit_global_store_is_recalled_by_a_fresh_identity_without_scope_override(
    tmp_path,
):
    first = ToolContext(session_id="first", memory_config_dir=str(tmp_path))
    second = ToolContext(session_id="fresh", memory_config_dir=str(tmp_path))
    stored = await execute_fact_memory(
        "s",
        {"op": "store", "text": "Elara Venn lives in Dunedin"},
        context=first,
    )
    assert not stored.is_error
    recall = await execute_fact_memory("r", {"op": "recall", "text": "Elara Venn"}, context=second)
    assert not recall.is_error and "Dunedin" in str(recall.content)
    # Work remains in the explicit current-session URL bank, never promoted by
    # native recall. The old owner's row stays byte-for-byte in its own scope.
    from local_operator.tools.memory_tools import execute_memory

    local = await execute_memory(
        "local",
        "write",
        {
            "path": "memory://session",
            "content": json.dumps({"title": "Private work", "body": "Private session note"}),
        },
        first,
    )
    assert not local.is_error
    hidden = await execute_fact_memory(
        "h", {"op": "recall", "text": "Private session note"}, context=second
    )
    assert not hidden.is_error and "Private session note" not in str(hidden.content)
    assert MemoryStore(tmp_path, MemoryIdentity("first")).search("Private session note", "session")


@pytest.mark.asyncio
@pytest.mark.parametrize("cwd_kind", ["repo", "scratch", "neutral"])
@pytest.mark.parametrize(
    "locator", ["docs/.env", "deploy/kubeconfig-prod", "config/token.txt", "C:/.ssh/id_rsa"]
)
async def test_relative_sensitive_paths_refused_by_actual_store_and_capture_in_each_cwd(
    tmp_path, monkeypatch, cwd_kind, locator
):
    from pathlib import Path

    from local_operator.memory_capture import extract

    root = tmp_path / "config"
    root.mkdir()
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(root))
    cwd = (
        root
        if cwd_kind == "scratch"
        else Path("/") if cwd_kind == "neutral" else Path(__file__).resolve().parents[2]
    )
    monkeypatch.chdir(cwd)
    context = ToolContext(session_id="paths", memory_config_dir=str(root))
    result = await execute_fact_memory(
        "unsafe", {"op": "store", "text": "Read " + locator}, context=context
    )
    assert result.is_error
    assert not (root / "memory").exists(), "refusal precedes every DB/index/oplog write"
    assert extract("Read " + locator) == [], "automatic capture uses the same path guard"
    good = await execute_fact_memory(
        "url",
        {"op": "store", "text": "Runbook https://example.invalid/runbook"},
        context=context,
    )
    assert not good.is_error
    assert any(
        row["body"] == "https://example.invalid/runbook"
        for row in extract("https://example.invalid/runbook")
    )
    # A URL carrying credential material still uses the credential-shape guard.
    bad_url = await execute_fact_memory(
        "credential-url",
        {
            "op": "store",
            "text": "https://operator:synthetic-pass123@example.invalid/runbook",
        },
        context=context,
    )
    assert bad_url.is_error


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy_scope", ["global", "session", "team", "agent"])
@pytest.mark.parametrize("op", ["store", "recall"])
async def test_legacy_native_scope_is_rejected_not_stripped_or_promoted(tmp_path, legacy_scope, op):
    from local_operator.harness.types import Message, ToolCall
    from local_operator.memory_privacy import retained_arguments, retained_message

    args = {"op": op, "scope": legacy_scope, "text": "Elara Venn lives in Dunedin"}
    raw = json.dumps(args)
    safe, kept_raw, masked = retained_arguments("memory", args, raw)
    assert masked and safe == {"op": op, "memory_arguments_redacted": True}
    assert kept_raw is not None and "Dunedin" not in kept_raw and '"scope"' not in kept_raw
    original = Message(
        role="assistant",
        content=[],
        tool_calls=[ToolCall(name="memory", arguments=args, raw_arguments=raw)],
        provider_payload={"alternate": raw},
    )
    retained = retained_message(original)
    assert "Dunedin" not in retained.model_dump_json()
    assert original.tool_calls[0].arguments == args and original.tool_calls[0].raw_arguments == raw
    result = await execute_fact_memory(
        "legacy",
        args,
        context=ToolContext(
            session_id="old", team_id="team", agent_id="agent", memory_config_dir=str(tmp_path)
        ),
    )
    assert result.is_error and "Dunedin" not in str(result.content)
    assert not (
        tmp_path / "memory"
    ).exists(), "even legacy global is invalid before any DB write/read"


@pytest.mark.asyncio
async def test_existing_session_rows_are_never_promoted_or_rewritten_by_personal_helper(tmp_path):
    owner = MemoryStore(tmp_path, MemoryIdentity("old-session"), origin="fixture")
    old = owner.put("session", {"title": "Temporary work", "body": "Temporary parcel state only"})
    with owner.connection() as db:
        before = [
            tuple(row) for row in db.execute("SELECT op_id,data FROM ops WHERE scope='session'")
        ]
    context = ToolContext(session_id="fresh-session", memory_config_dir=str(tmp_path))
    stored = await execute_fact_memory(
        "personal", {"op": "store", "text": "Elara Venn lives in Dunedin"}, context=context
    )
    assert not stored.is_error
    receipt = json.loads(str(getattr(stored.content[0], "text", "")))
    assert receipt["scope"] == "global" and receipt["uri"].startswith("memory://global/")
    assert receipt["lifetime"] == "global/across sessions"
    recall = await execute_fact_memory(
        "lookup", {"op": "recall", "text": "Temporary parcel state"}, context=context
    )
    assert not recall.is_error
    assert "Temporary parcel state" not in str(recall.content)
    with owner.connection() as db:
        after = [
            tuple(row) for row in db.execute("SELECT op_id,data FROM ops WHERE scope='session'")
        ]
    assert before == after and owner.show("session", old["id"])["status"] == "active"


#: A device id (network.identity.device_id_for: ``d_`` + sha256 hex[:32]) whose hex
#: tail satisfies the IBAN shape + mod-97 check by chance. Held verbatim from the
#: refusal that the read-path exemption fixes.
DEVICE_ID = "d_ef73a4312e5c4686196b30c035c8a05a"


@pytest.fixture
def recall_context(tmp_path):
    return ToolContext(session_id="s", memory_config_dir=str(tmp_path / "cfg"))


async def _seeded(context: ToolContext) -> None:
    written = await execute_memory(
        "w",
        "write",
        {
            "path": "memory://global",
            "content": json.dumps({"title": "Host", "body": "staging host is stage-2"}),
        },
        context,
    )
    assert not written.is_error, written.content


@pytest.mark.asyncio
async def test_a_structural_id_recall_query_is_not_refused_by_the_read_path(recall_context):
    """The agent twin of the store refusal, on the same value.

    ``_run`` guards the WHOLE argument dict before the store sees it, so a benign
    identifier lookup failed here even with the store path fixed.
    """
    await _seeded(recall_context)
    recalled = await execute_memory(
        "r", "read", {"path": f"memory://global?q={DEVICE_ID}"}, recall_context
    )
    assert not recalled.is_error, str(recalled.content)
    assert json.loads(str(getattr(recalled.content[0], "text", ""))) == []
    # The same call with prose still answers: the exemption changed no other query.
    prose = await execute_memory(
        "p", "read", {"path": "memory://global?q=staging host"}, recall_context
    )
    assert not prose.is_error
    assert json.loads(str(getattr(prose.content[0], "text", "")))[0]["id"]


@pytest.mark.asyncio
async def test_a_structural_reference_recall_value_is_not_refused_by_the_fact_tool(recall_context):
    """The THIRD site, on the same value: the plain-text tool guards its own args first.

    ``execute_fact_memory`` runs its own ``guard_encoded(args)`` over the whole dict
    before it delegates to the read path, so ``recall(<device id>)`` still failed with
    the store-oriented message after the two layers below it were exempted.

    The recall answers ``[]`` anyway: no stored fact can carry the token, because the
    write side refuses ``d_<hex32>`` anywhere in fact text. Removing a WRONG refusal is
    the whole goal — this does not make a device id a lookup key.
    """
    await _seeded(recall_context)
    recalled = await execute_fact_memory(
        "r", {"op": "recall", "text": DEVICE_ID}, context=recall_context
    )
    assert not recalled.is_error, str(recalled.content)
    assert json.loads(str(getattr(recalled.content[0], "text", ""))) == {
        "bank": "global/across sessions",
        "items": [],
    }
    # The same call with prose still answers, so the exemption moved no other query.
    prose = await execute_fact_memory(
        "p", {"op": "recall", "text": "staging host"}, context=recall_context
    )
    assert not prose.is_error
    assert json.loads(str(getattr(prose.content[0], "text", "")))["items"][0]["id"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        # A real card/IBAN/SSN spelling stays refused on the recall branch too.
        {"op": "recall", "text": "card 4111 1111 1111 1111"},
        {"op": "recall", "text": "IBAN GB82WEST12345698765432"},
        {"op": "recall", "text": "SSN 123-45-6789"},
        # A reference inside prose is prose: the exemption is a FULL match.
        {"op": "recall", "text": DEVICE_ID + " extra"},
        # A WRITE carrying the same id-shaped value: storing a value is not the same
        # act as looking one up, so the store branch keeps the guard it has today.
        {"op": "store", "text": DEVICE_ID},
        # ...and an op this tool does not have is not a recall either, so it reaches
        # the guard (and then the contract) exactly as it did before.
        {"op": "write", "text": DEVICE_ID},
        # The exemption is that one VALUE of that one op: a secret-shaped SIBLING
        # argument of the SAME recall call is still inspected and refused.
        {"op": "recall", "text": DEVICE_ID, "i": "card 4111 1111 1111 1111"},
    ],
)
async def test_the_fact_tool_exemption_covers_only_a_recalls_text_value(recall_context, args):
    result = await execute_fact_memory("c", dict(args), context=recall_context)
    assert result.is_error and "Sensitive value refused" in str(result.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "query",
    [
        "4111111111111111",
        "GB82WEST12345698765432",
        "123-45-6789",
        "the SSN is 123-45-6789",
        f"{DEVICE_ID} extra",
    ],
)
async def test_a_secret_shaped_recall_query_is_still_refused(recall_context, query):
    refused = await execute_memory(
        "r", "read", {"path": f"memory://global?q={query}"}, recall_context
    )
    assert refused.is_error
    assert "Sensitive value refused" in str(refused.content)


@pytest.mark.asyncio
async def test_the_recall_exemption_covers_the_read_query_value_alone(recall_context):
    """A sibling argument, and the same value on a write, keep today's refusal."""
    sibling = await execute_memory(
        "r",
        "read",
        {"path": f"memory://global?q={DEVICE_ID}", "raw": "GB82WEST12345698765432"},
        recall_context,
    )
    assert sibling.is_error and "Sensitive value refused" in str(sibling.content)
    written = await execute_memory(
        "w", "write", {"path": f"memory://global?q={DEVICE_ID}", "content": "{}"}, recall_context
    )
    assert written.is_error and "Sensitive value refused" in str(written.content)
    payload = await execute_memory(
        "s",
        "write",
        {
            "path": "memory://global",
            "content": json.dumps({"title": "Payout", "body": "IBAN GB82WEST12345698765432"}),
        },
        recall_context,
    )
    assert payload.is_error and "Sensitive value refused" in str(payload.content)


@pytest.mark.asyncio
async def test_a_read_path_whose_query_does_not_parse_keeps_the_letter_of_its_refusal(
    recall_context,
):
    """Classifying the q value must not change WHICH refusal a read reports.

    ``parse_url`` parses its query with ``strict_parsing=True``, which raises
    ``ValueError`` for a field with no "=" (a trailing ``&``). Treating that as "not a
    reference" leaves the caller's own guard — which refuses an id-shaped path — to
    speak first, so the message and its order are the ones this input produced before
    the recall exemption existed. ``execute_memory`` maps a bare ``ValueError`` to a
    generic request error instead, which would have moved the guard behind it.
    """
    id_shaped = await execute_memory(
        "r", "read", {"path": f"memory://global?q={DEVICE_ID}&"}, recall_context
    )
    assert id_shaped.is_error and "Sensitive value refused" in str(id_shaped.content)
    # Not id-shaped: the guard admits it and parse_url still owns the refusal.
    prose = await execute_memory(
        "p", "read", {"path": "memory://global?q=staging host&"}, recall_context
    )
    assert prose.is_error
    assert "Invalid memory request or unavailable store" in str(prose.content)


@pytest.mark.asyncio
async def test_a_tokens_parameter_is_refused_by_a_rule_that_predates_this_change(
    recall_context,
):
    """Documents PRE-EXISTING behaviour; this is not the recall fix's [redacted].

    A whitespace-free path carrying ``tokens=`` is refused by the path-like-token arm
    (``memory.py`` extracts path-like tokens and asks ``is_sensitive``; the name match
    answers ``token-or-secret-name``), and it refuses a non-id query identically — on
    the base too, before any exemption. The exemption removes one value and stops at
    ``&``, so it never reaches this parameter; ``limit=`` is unaffected. Pinned so the
    residual cannot drift silently.
    """
    limited = await execute_memory(
        "l", "read", {"path": f"memory://global?q={DEVICE_ID}&limit=5"}, recall_context
    )
    assert not limited.is_error, str(limited.content)
    for query in (DEVICE_ID, "stage2"):
        refused = await execute_memory(
            "t", "read", {"path": f"memory://global?q={query}&tokens=50"}, recall_context
        )
        assert refused.is_error and "Sensitive path refused" in str(refused.content)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "args",
    [
        [],
        "recall",
        {"op": "recall"},
        {"op": "recall", "text": 12345},
    ],
)
async def test_a_malformed_call_still_fails_the_contract_not_the_exemption(recall_context, args):
    """The exemption reads ``args`` BEFORE validation, so its defensive branch is pinned.

    None of these is a recall whose text IS a reference, so the guard and the contract
    answer in the same order and with the same message as they did before it existed:
    a non-dict must not reach the exemption's ``args.get`` at all.
    """
    result = await execute_fact_memory("c", args, context=recall_context)
    assert result.is_error and "Invalid memory contract" in str(result.content)
