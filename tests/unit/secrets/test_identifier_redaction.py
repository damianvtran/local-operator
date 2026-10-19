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

import pytest

from local_operator.agents import AgentEditFields, AgentRegistry
from local_operator.harness.types import ToolContext
from local_operator.projects import ProjectEdit, ProjectRegistry
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tools.agent_tool import execute_agent
from local_operator.tools.builtin import (
    _session_row_brief,
    _sessions_info_body,
    identifier_text,
)
from local_operator.tools.project_tool import execute_project
from local_operator.tools.team_tool import execute_team
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


def _plain(text: str) -> str:
    """The rendered text with span markers removed — what restoration must yield."""
    return text.replace(IDENTIFIER_SPAN_OPEN, "").replace(IDENTIFIER_SPAN_CLOSE, "")


def _has_private_use(text: str) -> bool:
    return any(ord(char) in (0xE000, 0xE001) or 0xF000 <= ord(char) <= 0xF8FF for char in text)


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
    "sk-" + "abcdefghijklmnop",
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
    assert wrapped == IDENTIFIER_SPAN_OPEN + "a\ufffdb\ufffd\ufffdc" + IDENTIFIER_SPAN_CLOSE
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
        assert scrubbed == _plain(result.text)
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
        assert scrubbed == _plain(text)
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
    assert IDENTIFIER_SPAN_OPEN in shown.text and "manager: token" in _plain(shown.text)
    assert "- password" in _plain(shown.text)
    assert IDENTIFIER_SPAN_OPEN in listed.text
    for result in (shown, listed):
        scrubbed, report = store.redact_with_report(result.text)
        assert scrubbed == _plain(result.text)
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
        assert scrubbed == _plain(result.text)
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
