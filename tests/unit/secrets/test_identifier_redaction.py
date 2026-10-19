"""LOH-28 remediation: a structured identifier SPAN, never a value exemption.

The first cut registered identifier VALUES in the store, so a legal project or
team name such as ``token`` exempted that word for the rest of the session — and
an unrelated ``token: <32 hex>`` line came back unmasked. The redesign marks the
exact bytes a render site prints (private-use open/close markers), extracts the
spans to per-call random placeholders before either redaction pass, and restores
only the placeholders that survived. Nothing is registered: an identifier lives
for one occurrence in one result.

Both directions are the gate: the incident's long slug must survive at its
``key:`` line, and a real credential later in the same output must still mask.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.harness.loop import AgentLoop, LoopContext
from local_operator.harness.types import (
    ChatRequest,
    LoopConfig,
    Message,
    ModelSpec,
    StreamEndEvent,
    StreamTextDelta,
    StreamToolCallDelta,
    ToolContext,
    ToolExecutionEndEvent,
)
from local_operator.projects import ProjectEdit, ProjectRegistry
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tools.agent_tool import execute_agent
from local_operator.tools.builtin import (
    _redact_tool_text,
    _session_row_brief,
    _sessions_info_body,
    execute_read,
    identifier_text,
)
from local_operator.tools.project_tool import execute_project
from local_operator.tools.registry import create_tools
from local_operator.tools.spill import get_store
from local_operator.tools.team_tool import execute_team, execute_team_delete
from local_operator.variables import (
    IDENTIFIER_SPAN_CLOSE,
    IDENTIFIER_SPAN_OPEN,
    VariableStore,
)

#: The 34-char slug from the incident, assembled from words so no literal in
#: this source is a maskable spelling in the reader's own transcript.
SLUG = "-".join(["project", "keys", "export", "status", "reports"])
#: A 34-char slug that masks under base rules: three digits exceed the weak-name
#: digit cap, and it does not repeat the credential name ``key``.
SLUG_DIGITS = "-".join(["project", "alpha", "v2", "export", "status", "v77"])
#: A 60-char lowercase hyphen phrase: past the base 32-char release bound.
LONG_PHRASE = "-".join(
    ["platform", "observability", "baseline", "rollout", "status", "reports", "daily"]
)
#: A real credential shape: 32 hex characters.
HEX_32 = "abcdef0123456789" * 2
#: A real mixed-case token, assembled so no literal token appears in this file.
MIXED_TOKEN = "".join(["aB3k", "L9mN", "2pQ7", "Zr4"])
#: A credential phrase the strong-name rule must keep masking.
PASSWORD_PHRASE = "-".join(["correct", "horse", "battery", "staple"])
#: An issuer-prefixed value, assembled so it is not a literal in this source.
SK_VALUE = "sk-" + "".join(["abcdefgh", "ijklmnop"])
#: A private-use character inside the span placeholder range.
PLACEHOLDER_CHAR = chr(0xF123)


@pytest.fixture()
def store() -> VariableStore:
    return VariableStore(cwd=".")


def _plain(store: VariableStore, text: str) -> str:
    """The rendered text with span markers removed — what restoration must yield.

    Deferred to the store's own stripper so the expectation and the spill
    path cannot drift (the token is store-scoped, not a fixed shape).
    """
    return store.strip_identifier_spans(text)


def _has_private_use(text: str) -> bool:
    return any(ord(char) in (0xE000, 0xE001) or 0xF000 <= ord(char) <= 0xF8FF for char in text)


def _assert_no_marker_text(text: str) -> None:
    """No live PUA marker and no six-character ``\\ue00x`` escape either.

    A ``!r`` over a marked value escapes the markers into ASCII text, which
    looks harmless but is not a span; both spellings are failures here.
    """
    assert IDENTIFIER_SPAN_OPEN not in text and IDENTIFIER_SPAN_CLOSE not in text
    assert "\\ue000" not in text and "\\ue001" not in text
    assert _has_private_use(text) is False


# --- the protocol itself -----------------------------------------------------


def test_a_marked_identifier_survives_a_weak_name(store: VariableStore) -> None:
    marked = "key: " + store.identifier_span(SLUG)
    scrubbed, report = store.redact_with_report(marked)
    assert scrubbed == "key: " + SLUG
    assert report.labels == ()
    assert report.reached_model is False


def test_an_unmarked_long_slug_still_masks_like_base(store: VariableStore) -> None:
    """The base behaviour the fix must not widen: no span, no exemption."""
    for value in (SLUG_DIGITS, LONG_PHRASE):
        scrubbed, report = store.redact_with_report("key: " + value)
        assert value not in scrubbed
        assert "credential-assignment" in report.labels


def test_a_marked_identifier_survives_a_strong_name(store: VariableStore) -> None:
    text = "token: " + store.identifier_span(SLUG) + "\napi_key: " + HEX_32
    scrubbed, report = store.redact_with_report(text)
    assert "token: " + SLUG in scrubbed
    assert HEX_32 not in scrubbed
    assert "credential-assignment" in report.labels
    assert report.reached_model is False


# --- the blocker: an identifier VALUE must not become a store-wide exemption ---

BLOCKER_NAMES = (
    "token",
    "password",
    "secret",
    "api_key",
    "pass",
    "Bearer",
    "xoxb",
    "credentials",
)


@pytest.mark.parametrize("name", BLOCKER_NAMES)
def test_a_credential_named_identifier_does_not_release_a_later_credential(
    store: VariableStore, name: str
) -> None:
    """The exact regression both review rounds found.

    The name is legal as a project/team key; marking it at the ``key:`` line
    must not exempt the independent ``token:`` credential below it, in the same
    store and the same pass.
    """
    text = "key: " + store.identifier_span(name) + "\ntoken: " + HEX_32
    scrubbed, report = store.redact_with_report(text)
    assert "key: " + name in scrubbed
    assert HEX_32 not in scrubbed
    assert report.labels == ("credential-assignment",)
    assert report.reached_model is False


ADVERSARIAL_VALUES = (
    SK_VALUE,
    'quoted "name"',
    "back`ticked",
    "line\nbreak",
    "marker" + IDENTIFIER_SPAN_OPEN + "inside",
    "placeholder" + PLACEHOLDER_CHAR + "inside",
)


@pytest.mark.parametrize("value", ADVERSARIAL_VALUES)
def test_adversarial_identifiers_stay_readable_and_never_exempt_credentials(
    store: VariableStore, value: str
) -> None:
    text = "key: " + store.identifier_span(value) + "\npassword: " + PASSWORD_PHRASE
    scrubbed, report = store.redact_with_report(text)
    expected = value.replace(IDENTIFIER_SPAN_OPEN, "\ufffd").replace(
        IDENTIFIER_SPAN_CLOSE, "\ufffd"
    )
    expected = "".join("\ufffd" if 0xF000 <= ord(char) <= 0xF8FF else char for char in expected)
    assert "key: " + expected in scrubbed
    assert "correct-horse-battery-staple" not in scrubbed
    assert report.labels == ("credential-assignment",)
    assert report.reached_model is False
    assert _has_private_use(scrubbed) is False


def test_a_span_inside_a_longer_strong_value_masks_whole(store: VariableStore) -> None:
    text = "api_key: prefix" + store.identifier_span("value") + "suffix"
    scrubbed, report = store.redact_with_report(text)
    assert "value" not in scrubbed
    assert "prefix" not in scrubbed and "suffix" not in scrubbed
    assert _has_private_use(scrubbed) is False
    assert report.reached_model is False


def test_a_span_overlapping_a_known_secret_is_not_extracted(store: VariableStore) -> None:
    """Review R1-3 / QA Q1-3: extracting it would split the secret's bytes."""
    secret = "prefix-" + SLUG
    store.register_redaction(secret)
    text = "key: " + store.identifier_span(SLUG) + "\nvalue: " + secret
    out, _ = store.redact_with_report(text)
    assert secret not in out
    assert "value: [redacted]" in out
    assert IDENTIFIER_SPAN_OPEN not in out and IDENTIFIER_SPAN_CLOSE not in out


def test_a_span_containing_a_known_secret_is_not_extracted(store: VariableStore) -> None:
    """The other direction: the secret masks in place, never the whole span."""
    secret = "-".join(SLUG.split("-")[:2])
    store.register_redaction(secret)
    out, _ = store.redact_with_report("key: " + store.identifier_span(SLUG))
    assert secret not in out
    assert out != "key: " + SLUG
    assert IDENTIFIER_SPAN_OPEN not in out and IDENTIFIER_SPAN_CLOSE not in out


def test_malformed_markers_never_reach_the_output(store: VariableStore) -> None:
    unclosed, _ = store.redact_with_report("key: " + IDENTIFIER_SPAN_OPEN + "token")
    assert IDENTIFIER_SPAN_OPEN not in unclosed
    lone_close, _ = store.redact_with_report("x" + IDENTIFIER_SPAN_CLOSE + "y")
    assert IDENTIFIER_SPAN_CLOSE not in lone_close
    stale, _ = store.redact_with_report("value " + PLACEHOLDER_CHAR)
    assert PLACEHOLDER_CHAR not in stale


def test_identifier_span_strips_marker_and_placeholder_chars(store: VariableStore) -> None:
    assert store.identifier_span("") == ""
    wrapped = store.identifier_span(
        "a" + IDENTIFIER_SPAN_OPEN + "b" + IDENTIFIER_SPAN_CLOSE + PLACEHOLDER_CHAR + "c"
    )
    assert wrapped == (
        IDENTIFIER_SPAN_OPEN + store._span_token + ":a\ufffdb\ufffd\ufffdc" + IDENTIFIER_SPAN_CLOSE
    )
    out, _ = store.redact_with_report("key: " + wrapped)
    assert out == "key: a\ufffdb\ufffd\ufffdc"


# --- base keep-masking controls ---------------------------------------------


@pytest.mark.parametrize("name", ("password", "token", "secret", "api_key"))
def test_strong_names_still_mask_long_word_values(store: VariableStore, name: str) -> None:
    text = name + ": " + LONG_PHRASE
    scrubbed, report = store.redact_with_report(text)
    assert LONG_PHRASE not in scrubbed
    assert report.labels == ("credential-assignment",)
    assert report.reached_model is False


def test_the_base_rule_masks_a_long_word_value_under_key(store: VariableStore) -> None:
    scrubbed, report = store.redact_with_report("key: " + LONG_PHRASE)
    assert LONG_PHRASE not in scrubbed
    assert report.labels == ("credential-assignment",)


# --- the real render paths ---------------------------------------------------


@pytest.mark.asyncio
async def test_project_show_and_list_mark_the_key_and_title(tmp_path) -> None:
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path)
    registry.create_project(
        ProjectEdit(
            name=SLUG,
            title="Export status reports for the platform rollout",
            description="Span wiring fixture.",
        ),
        sessions=[],
    )
    context = ToolContext(
        cwd=".", session_id="4e92693767fa", project_registry=registry, variables=store
    )
    shown = await execute_project("tc", {"op": "show", "name": SLUG}, None, None, context)
    listed = await execute_project("tc", {"op": "list"}, None, None, context)
    assert IDENTIFIER_SPAN_OPEN in shown.text and IDENTIFIER_SPAN_CLOSE in shown.text
    assert IDENTIFIER_SPAN_OPEN in listed.text
    assert "key: " + store.identifier_span(SLUG) in shown.text
    for result in (shown, listed):
        scrubbed, report = store.redact_with_report(result.text)
        assert scrubbed == _plain(store, result.text)
        assert SLUG in scrubbed
        assert report.labels == ()
        assert report.reached_model is False


def test_session_rows_mark_the_title_and_id(store: VariableStore) -> None:
    context = ToolContext(cwd=".", session_id="4e92693767fa", variables=store)
    row = {"state": "live", "conversation_name": SLUG, "session_id": "4e92693767fa"}
    brief = _session_row_brief(row, context)
    info = _sessions_info_body(
        row, {"session_origin": "user", "sidebar_visibility": "listed"}, context
    )
    assert (
        IDENTIFIER_SPAN_OPEN in brief
        and "session " + store.identifier_span(row["session_id"]) in brief
    )
    assert IDENTIFIER_SPAN_OPEN in info
    for text in (brief, info):
        scrubbed, report = store.redact_with_report(text)
        assert scrubbed == _plain(store, text)
        assert SLUG in scrubbed and row["session_id"] in scrubbed
        assert report.labels == ()


@pytest.mark.asyncio
async def test_team_show_and_list_mark_names_and_roster(tmp_path) -> None:
    store = VariableStore(cwd=".")
    registry = TeamRegistry(tmp_path)
    registry.create_team(
        TeamEditFields(
            name=SLUG,
            label="Project Keys Export",
            manager="token",
            members=[TeamMember(role="password")],
        )
    )
    context = ToolContext(cwd=".", team_registry=registry, variables=store)
    shown = await execute_team("tc", {"op": "show", "name": SLUG}, None, None, context)
    listed = await execute_team("tc", {"op": "list"}, None, None, context)
    assert IDENTIFIER_SPAN_OPEN in shown.text and "manager: token" in _plain(store, shown.text)
    assert "- password" in _plain(store, shown.text)
    assert IDENTIFIER_SPAN_OPEN in listed.text
    for result in (shown, listed):
        scrubbed, report = store.redact_with_report(result.text)
        assert scrubbed == _plain(store, result.text)
        assert SLUG in scrubbed and "token" in scrubbed
        assert report.labels == ()
        assert report.reached_model is False


@pytest.mark.asyncio
async def test_agent_show_and_list_mark_the_role_name(tmp_path) -> None:
    store = VariableStore(cwd=".")
    registry = AgentRegistry(tmp_path)
    fields = {
        "name": SLUG,
        "label": "Project Keys Export",
        "description": "Export status reports for the platform rollout.",
        "tags": ["role", "tools:bash"],
    }
    registry.create_agent(AgentEditFields(**fields))
    context = ToolContext(cwd=".", agent_registry=registry, variables=store)
    for args in ({"op": "show", "name": SLUG}, {"op": "list"}):
        result = await execute_agent("tc", args, None, None, context)
        assert IDENTIFIER_SPAN_OPEN in result.text
        scrubbed, report = store.redact_with_report(result.text)
        assert scrubbed == _plain(store, result.text)
        assert SLUG in scrubbed
        assert report.labels == ()
        assert report.reached_model is False


# --- helper ------------------------------------------------------------------


def test_identifier_text_is_a_no_op_without_a_capable_store() -> None:
    assert identifier_text(None, SLUG) == SLUG
    assert identifier_text(ToolContext(cwd="."), SLUG) == SLUG
    store = VariableStore(cwd=".")
    assert identifier_text(ToolContext(cwd=".", variables=store), SLUG).startswith(
        IDENTIFIER_SPAN_OPEN
    )


# --- rendered reprs and receipts (round-2 self-check) ------------------------


def test_the_adversarial_issuer_value_is_the_assembled_one() -> None:
    """The redacted-on-write failure must not recur: the source carries SK_VALUE.

    A literal ``sk-`` value written here was masked by the transcript pass, so
    the case silently stopped testing an issuer value. The tuple member is the
    assembled constant; the AST-level fact pinned here is that the SOURCE names
    it rather than a literal.
    """
    assert ADVERSARIAL_VALUES[0] == SK_VALUE
    assert ADVERSARIAL_VALUES[0].startswith("sk-")
    source = Path(__file__).read_text(encoding="utf-8")
    assert "    SK_VALUE," in source


@pytest.mark.asyncio
async def test_agent_and_team_reprs_never_leak_escaped_markers(tmp_path) -> None:
    """``!r`` over a marked value escapes the PUA chars, so the repr is marked."""
    store = VariableStore(cwd=".")
    agent_registry = AgentRegistry(tmp_path / "agents")
    fields = {
        "name": SLUG,
        "label": "Project Keys Export",
        "description": "Export status reports for the platform rollout.",
        "tags": ["role", "tools:bash"],
    }
    agent_registry.create_agent(AgentEditFields(**fields))
    agent_ctx = ToolContext(cwd=".", agent_registry=agent_registry, variables=store)
    shown = await execute_agent("tc", {"op": "show", "name": SLUG}, None, None, agent_ctx)
    assert IDENTIFIER_SPAN_OPEN in shown.text
    assert repr(SLUG) in shown.text.replace(IDENTIFIER_SPAN_OPEN, "").replace(
        IDENTIFIER_SPAN_CLOSE, ""
    )

    team_registry = TeamRegistry(tmp_path / "teams")
    team_registry.create_team(
        TeamEditFields(
            name=SLUG,
            label="Project Keys Export",
            manager="manager",
            members=[TeamMember(role="coder")],
        )
    )
    team_ctx = ToolContext(cwd=".", team_registry=team_registry, variables=store)
    updated = await execute_team(
        "tc", {"op": "update", "name": SLUG, "description": "updated"}, None, None, team_ctx
    )
    deleted = await execute_team_delete("tc", {"name": SLUG}, None, None, team_ctx)
    for result in (shown, updated, deleted):
        scrubbed, report = store.redact_with_report(result.text)
        _assert_no_marker_text(scrubbed)
        assert SLUG in scrubbed
        assert report.labels == ()
        assert report.reached_model is False


@pytest.mark.asyncio
async def test_project_receipts_and_fields_mark_token_and_keep_credentials(tmp_path) -> None:
    """The class's receipts and non-key fields: marked, and never a global pass."""
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path / "config")
    context = ToolContext(
        cwd=".", session_id="4e92693767fa", project_registry=registry, variables=store
    )
    created = await execute_project(
        "tc",
        {
            "op": "create",
            "name": "token",
            "title": "Blocker regression",
            "team": "password",
            "description": "receipt and field marking",
        },
        None,
        None,
        context,
    )
    updated = await execute_project(
        "tc",
        {"op": "update", "name": "token", "progress": "still marked"},
        None,
        None,
        context,
    )
    shown = await execute_project("tc", {"op": "show", "name": "token"}, None, None, context)
    for index, result in enumerate((created, updated, shown)):
        assert IDENTIFIER_SPAN_OPEN in result.text
        # The RENDERING store must redact: a span is scoped to its minting
        # store, which is the unforgeability property. A distinct credential
        # per receipt keeps the shape pass (not exact-value recall) the thing
        # under test.
        probe = HEX_32[:-1] + format(index, "x")
        scrubbed, report = store.redact_with_report(result.text + "\ntoken: " + probe)
        assert "token" in scrubbed
        assert probe not in scrubbed
        assert report.labels == ("credential-assignment",)
        assert report.reached_model is False
        _assert_no_marker_text(scrubbed)


# --- forge resistance and boundary overlap (round 2) -------------------------


def test_a_forged_static_pair_does_not_suppress_a_credential(store: VariableStore) -> None:
    """Bare markers are not an authority: only this store's token makes a span."""
    for value in (HEX_32, "abcd1234"):
        text = "token: " + IDENTIFIER_SPAN_OPEN + value + IDENTIFIER_SPAN_CLOSE
        scrubbed, report = store.redact_with_report(text)
        assert value not in scrubbed
        assert report.labels == ("credential-assignment",)
        assert report.reached_model is False
        _assert_no_marker_text(scrubbed)


def test_a_forged_wrong_token_pair_does_not_suppress_a_credential(
    store: VariableStore,
) -> None:
    wrong = "f" * 32
    text = "token: " + IDENTIFIER_SPAN_OPEN + wrong + ":" + HEX_32 + IDENTIFIER_SPAN_CLOSE
    scrubbed, report = store.redact_with_report(text)
    assert HEX_32 not in scrubbed
    assert report.labels == ("credential-assignment",)
    assert report.reached_model is False
    _assert_no_marker_text(scrubbed)


def test_a_forged_pair_masks_through_the_live_tool_sink(store: VariableStore) -> None:
    context = ToolContext(cwd=".", variables=store)
    forged = "token: " + IDENTIFIER_SPAN_OPEN + HEX_32 + IDENTIFIER_SPAN_CLOSE
    scrubbed = _redact_tool_text(forged, context)
    assert HEX_32 not in scrubbed
    _assert_no_marker_text(scrubbed)


@pytest.mark.asyncio
async def test_a_forged_pair_in_a_project_description_still_masks(tmp_path) -> None:
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path / "config")
    context = ToolContext(cwd=".", project_registry=registry, variables=store)
    forged = IDENTIFIER_SPAN_OPEN + HEX_32 + IDENTIFIER_SPAN_CLOSE
    await execute_project(
        "tc",
        {
            "op": "create",
            "name": SLUG,
            "title": "Forgery probe",
            "description": "leaked token: " + forged,
        },
        None,
        None,
        context,
    )
    shown = await execute_project("tc", {"op": "show", "name": SLUG}, None, None, context)
    scrubbed, report = store.redact_with_report(shown.text)
    assert HEX_32 not in scrubbed
    assert report.labels == ("credential-assignment",)
    assert report.reached_model is False
    _assert_no_marker_text(scrubbed)


def test_a_legit_span_survives_beside_a_forged_pair(store: VariableStore) -> None:
    forged = IDENTIFIER_SPAN_OPEN + HEX_32 + IDENTIFIER_SPAN_CLOSE
    text = "key: " + store.identifier_span(SLUG) + "\ntoken: " + forged
    scrubbed, report = store.redact_with_report(text)
    assert SLUG in scrubbed
    assert HEX_32 not in scrubbed
    assert report.labels == ("credential-assignment",)
    assert report.reached_model is False
    _assert_no_marker_text(scrubbed)


def test_nested_markers_inside_an_identifier_are_neutralized(
    store: VariableStore,
) -> None:
    value = "a" + IDENTIFIER_SPAN_OPEN + "b" + IDENTIFIER_SPAN_CLOSE + PLACEHOLDER_CHAR + "c"
    scrubbed, report = store.redact_with_report("key: " + store.identifier_span(value))
    assert scrubbed == "key: a\ufffdb\ufffd\ufffdc"
    assert report.labels == ()
    _assert_no_marker_text(scrubbed)


def test_unrelated_marker_bytes_are_neutralized(store: VariableStore) -> None:
    text = "plain " + IDENTIFIER_SPAN_OPEN + " text " + PLACEHOLDER_CHAR + " end"
    scrubbed, _ = store.redact_with_report(text)
    assert scrubbed == "plain \ufffd text \ufffd end"
    _assert_no_marker_text(scrubbed)


def test_a_known_secret_overlapping_a_span_boundary_masks_whole(
    store: VariableStore,
) -> None:
    """QA Q2-2: the secret starts before the OPEN and ends inside the span."""
    store.register_redaction("vorn-kell-tarn")
    # The secret begins BEFORE the open and ends INSIDE the span.
    scrubbed, _ = store.redact_with_report("key: vorn-" + store.identifier_span("kell-tarn-misp"))
    assert "vorn" not in scrubbed and "kell" not in scrubbed
    assert "[redacted]" in scrubbed
    _assert_no_marker_text(scrubbed)


def test_an_identifier_containing_a_known_secret_is_not_extracted(
    store: VariableStore,
) -> None:
    store.register_redaction("kell-tarn")
    scrubbed, _ = store.redact_with_report("key: " + store.identifier_span("vorn-kell-tarn"))
    assert "kell-tarn" not in scrubbed
    assert "[redacted]" in scrubbed
    _assert_no_marker_text(scrubbed)


def test_a_known_secret_wholly_inside_a_span_masks_whole(store: VariableStore) -> None:
    store.register_redaction("kell-tarn")
    scrubbed, _ = store.redact_with_report("key: " + store.identifier_span("vorn-kell-tarn-misp"))
    assert "kell-tarn" not in scrubbed and "[redacted]" in scrubbed
    _assert_no_marker_text(scrubbed)


def test_strip_identifier_spans_removes_the_token_and_stray_markers(
    store: VariableStore,
) -> None:
    marked = (
        "key: "
        + store.identifier_span(SLUG)
        + "\n"
        + IDENTIFIER_SPAN_OPEN
        + "x"
        + IDENTIFIER_SPAN_CLOSE
    )
    stripped = store.strip_identifier_spans(marked)
    assert stripped == "key: " + SLUG + "\n\ufffdx\ufffd"
    assert store._span_token not in stripped
    _assert_no_marker_text(stripped)


def test_a_spill_file_carries_no_span_token_or_markers(
    store: VariableStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, str] = {}

    class _FakeStore:
        def write(self, text: str, **kwargs: object) -> object:
            captured["text"] = text
            return object()

    import local_operator.tools.builtin as builtin_module

    monkeypatch.setattr(builtin_module, "get_store", lambda: _FakeStore())
    context = ToolContext(cwd=".", variables=store)
    large = ("padding " * 200) + "key: " + store.identifier_span(SLUG) + "\n"
    builtin_module._spill(large, "project", context)
    stored = captured["text"]
    assert SLUG in stored
    assert store._span_token not in stored
    _assert_no_marker_text(stored)


@pytest.mark.asyncio
async def test_team_override_alias_and_launch_hints_survive_redaction(tmp_path) -> None:
    store = VariableStore(cwd=".")
    registry = TeamRegistry(tmp_path / "teams")
    registry.create_team(
        TeamEditFields(
            name=SLUG,
            label="Project Keys Export",
            manager="manager",
            aliases=["token", "password"],
            members=[TeamMember(role="token", tier="hi")],
        )
    )
    context = ToolContext(cwd=".", team_registry=registry, variables=store)
    shown = await execute_team("tc", {"op": "show", "name": SLUG}, None, None, context)
    assert "aliases: " + IDENTIFIER_SPAN_OPEN in shown.text
    assert IDENTIFIER_SPAN_OPEN + store._span_token + ":token" in shown.text
    scrubbed, report = store.redact_with_report(shown.text)
    assert "token" in scrubbed and "password" in scrubbed
    assert SLUG in scrubbed
    assert report.labels == ()
    assert report.reached_model is False
    _assert_no_marker_text(scrubbed)


@pytest.mark.asyncio
async def test_project_update_and_refresh_receipts_mark_the_name(tmp_path) -> None:
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path / "config")
    context = ToolContext(cwd=".", project_registry=registry, variables=store)
    await execute_project(
        "tc",
        {"op": "create", "name": SLUG, "title": "Receipt probe"},
        None,
        None,
        context,
    )
    updated = await execute_project(
        "tc",
        {"op": "update", "name": SLUG, "title": "Receipt probe"},
        None,
        None,
        context,
    )
    refreshed = await execute_project("tc", {"op": "refresh", "name": SLUG}, None, None, context)
    for result in (updated, refreshed):
        assert IDENTIFIER_SPAN_OPEN in result.text
        scrubbed, report = store.redact_with_report(result.text)
        assert SLUG in scrubbed
        assert report.labels == ()
        assert report.reached_model is False
        _assert_no_marker_text(scrubbed)


# --- rounds 3: live events, spill ranges, truncation, performance ----------


MODEL = ModelSpec(provider="test", model_id="m")


class _ProjectShowStream:
    """One ``project show`` call, then a plain stop."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.requests: list[ChatRequest] = []

    def __call__(self, request: ChatRequest, signal: object):
        self.requests.append(request)
        if len(self.requests) == 1:
            turn = [
                StreamToolCallDelta(
                    index=0,
                    id="c1",
                    name="project",
                    argument_delta='{"op": "show", "name": "' + self.name + '"}',
                ),
                StreamEndEvent(stop_reason="toolUse"),
            ]
        else:
            turn = [StreamTextDelta(delta="done"), StreamEndEvent(stop_reason="stop")]

        async def gen():
            for event in turn:
                yield event

        return gen()


@pytest.mark.asyncio
async def test_a_live_tool_event_never_carries_the_span_token(tmp_path) -> None:
    """R3-1: the event is read by the UI/API before the model copy is redacted."""
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path / "config")
    tool_context = ToolContext(cwd=".", session_id="s1", project_registry=registry, variables=store)
    await execute_project(
        "tc",
        {"op": "create", "name": SLUG, "title": "Live event probe"},
        None,
        None,
        tool_context,
    )
    config = LoopConfig(
        model=MODEL,
        convert_to_llm=lambda messages: [m for m in messages if isinstance(m, Message)],
        stream_fn=_ProjectShowStream(SLUG),
        redact_tool_result=store.redact,
    )
    context = LoopContext(
        system_blocks=["sys"],
        tools=create_tools(tool_context),
        tool_context=tool_context,
    )
    ends: list[ToolExecutionEndEvent] = []
    async for event in AgentLoop().run([Message.user("go")], context, config, None):
        if isinstance(event, ToolExecutionEndEvent):
            ends.append(event)
    assert ends
    pieces: list[str] = []
    for item in ends[0].result.content:
        value = getattr(item, "text", None)
        if isinstance(value, str):
            pieces.append(value)
    text = "\n".join(pieces)
    assert store._span_token not in text
    _assert_no_marker_text(text)
    assert SLUG in text


@pytest.mark.asyncio
async def test_a_spill_read_back_re_marks_identifiers(tmp_path) -> None:
    """R3-2: the sidecar keeps ranges, never markers, and read-back restores the key."""
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path / "config")
    context = ToolContext(cwd=".", project_registry=registry, variables=store)
    credential = "token: " + HEX_32
    await execute_project(
        "tc",
        {
            "op": "create",
            "name": SLUG,
            "title": "Spill probe",
            "description": ("d" * 1880) + "\n" + credential,
        },
        None,
        None,
        context,
    )
    for index in range(20):
        await execute_project(
            "tc",
            {"op": "update", "name": SLUG, "progress": f"step {index} " + "p" * 940},
            None,
            None,
            context,
        )
    shown = await execute_project("tc", {"op": "show", "name": SLUG}, None, None, context)
    assert shown.details and "spill" in shown.details
    handle = shown.details["spill"]["handle"]
    digest = handle.split("spill://", 1)[1]

    import json

    spill_store = get_store()
    on_disk = spill_store._content_path(digest).read_text(encoding="utf-8")
    sidecar = json.loads(spill_store._meta_path(digest).read_text(encoding="utf-8"))
    assert SLUG in on_disk
    assert IDENTIFIER_SPAN_OPEN not in on_disk and IDENTIFIER_SPAN_CLOSE not in on_disk
    assert store._span_token not in on_disk and store._span_token not in str(sidecar)
    assert _has_private_use(on_disk) is False
    assert sidecar["identifier_spans"]

    read = await execute_read("tc", {"path": handle}, None, None, context)
    assert IDENTIFIER_SPAN_OPEN in read.text
    scrubbed, report = store.redact_with_report(read.text)
    assert SLUG in scrubbed
    assert HEX_32 not in scrubbed
    assert report.labels == ("credential-assignment",)
    assert report.reached_model is False
    _assert_no_marker_text(scrubbed)
    assert "\ufffd" not in scrubbed


@pytest.mark.asyncio
async def test_project_list_truncation_keeps_the_key_whole(tmp_path) -> None:
    """Q3-1: the key span is never cut by the row cap, even after a long title."""
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path / "config")
    context = ToolContext(cwd=".", project_registry=registry, variables=store)
    await execute_project(
        "tc",
        {
            "op": "create",
            "name": SLUG,
            # Wide cells: 80 CJK characters are 160 visual cells, so the key
            # span starts beyond the row cap and must still be kept.
            "title": "键" * 80,
        },
        None,
        None,
        context,
    )
    listed = await execute_project("tc", {"op": "list"}, None, None, context)
    assert IDENTIFIER_SPAN_OPEN in listed.text
    scrubbed, report = store.redact_with_report(listed.text)
    assert SLUG in scrubbed
    assert "\ufffd" not in scrubbed
    assert report.labels == ()
    assert report.reached_model is False
    _assert_no_marker_text(scrubbed)


def test_span_aware_truncation_never_cuts_a_span(store: VariableStore) -> None:
    from local_operator.projects import truncate_row

    title = store.identifier_span("x" * 40)
    key = store.identifier_span(SLUG)
    clipped = truncate_row("- " + title + " (" + key + ") then trailing prose", cap=60)
    assert title in clipped and key in clipped
    assert clipped.count(IDENTIFIER_SPAN_OPEN) == clipped.count(IDENTIFIER_SPAN_CLOSE)


def test_extractor_scales_to_megabyte_text(store: VariableStore) -> None:
    """R3-3: one piece per segment/span, not one 3-tuple per character."""
    for megabytes, ceiling in ((1, 4.0), (4, 8.0)):
        unit = ("x" * 1024 + "\n") * (megabytes * 1024 + 1)
        text = unit[: megabytes * 1024 * 1024]
        chunks: list[str] = []
        step = 64 * 1024
        spans = 0
        for offset in range(0, len(text), step):
            chunks.append(text[offset : offset + step])
            if offset % (256 * 1024) == 0:
                chunks.append(store.identifier_span(SLUG))
                spans += 1
        marked = "".join(chunks)
        started = time.process_time()
        _protected, extracted = store._extract_identifier_spans(marked)
        elapsed = time.process_time() - started
        assert len(extracted) == spans
        assert elapsed < ceiling, f"{megabytes}MB took {elapsed:.1f}s"
