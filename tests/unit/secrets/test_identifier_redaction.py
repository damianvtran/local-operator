"""LOH-28: explicitly declared identifiers survive redaction; secrets do not.

The defect these pin: a project key is rendered as ``key: <name>``, ``key`` is a
weak credential name, and the readable-lowercase-hyphen release refused values
longer than 32 characters. A 34-character slug therefore masked — and a mask is
REGISTERED session-wide, so a title sharing any six-character window with the
slug graded the hit exposed and filed a credential incident. Stored data was
never wrong; only the transcript was.

Two directions are the gate here, and neither alone is evidence:

* a declared identifier is released BYTE-IDENTICAL under weak and strong names;
* a real credential, a known-secret overlap, and an identifier embedded in a
  longer masking value all still mask whole.

The identifier values are assembled from words rather than written as one
literal, following the corpus convention: this file is read through the pass it
describes, and a literal here would be a maskable spelling in the reader's own
transcript.
"""

from __future__ import annotations

from typing import Any

import pytest

from local_operator.harness.types import ToolContext
from local_operator.projects import ProjectEdit, ProjectRegistry
from local_operator.teams import TeamEditFields, TeamMember, TeamRegistry
from local_operator.tools.agent_tool import execute_agent
from local_operator.tools.project_tool import execute_project
from local_operator.tools.team_tool import execute_team
from local_operator.variables import VariableStore

#: The 34-char slug from the incident: just past the retired 32-char bound.
SLUG = "-".join(["project", "keys", "export", "status", "reports"])
#: The same class far past the bound, so a re-added length test reds anywhere.
SLUG_60 = "-".join(
    ["platform", "observability", "baseline", "rollout", "status", "reports", "daily"]
)
#: A real credential shape: 32 hex characters under a strong name.
HEX_32 = "abcdef0123456789" * 2
#: A real mixed-case token, the other shape the store must keep masking.
MIXED_TOKEN = "".join(["aB3k", "L9mN", "2pQ7", "Zr4"])
#: The long member role a team roster carries, in the same released class.
MEMBER_ROLE = "-".join(["release", "verification", "owner", "secondary"])
#: A 34-char slug that still MASKS without a declaration: three digits put it
#: outside the released digit cap, and it does not repeat the credential name
#: (a ``key`` value with a ``keys`` segment is released by that clause
#: instead). This makes the explicit exemption's INDEPENDENT effect visible —
#: the incident's digit-free slug is released by the rule change alone (pinned
#: below and in the corpus), so a control using it could not show what
#: ``register_identifier`` adds.
SLUG_DIGITS = "-".join(["project", "alpha", "v2", "export", "status", "v77"])
SESSION = "4e92693767fa"


@pytest.fixture()
def store() -> VariableStore:
    return VariableStore(cwd=".")


def test_an_unregistered_three_digit_long_key_masks_and_registers(
    store: VariableStore,
) -> None:
    """The control: without a declaration, a non-released 34-char key masks.

    The digit cap is the property that keeps this value a shape, so the mask is
    the store's ordinary behaviour and not a leftover. Registration is what made
    the incident session-wide, so it is asserted here rather than assumed: the
    value must be gone both in place and from a later, shape-free occurrence.
    """
    text = "key: " + SLUG_DIGITS
    scrubbed, report = store.redact_with_report(text)
    assert SLUG_DIGITS not in scrubbed
    assert report.reached_model is False  # the mask itself was whole
    assert store.redact(SLUG_DIGITS) == "[redacted]"


def test_the_incident_slug_is_released_by_the_rule_alone(store: VariableStore) -> None:
    """Change 1 already fixes the reported line; the exemption is defence.

    A digit-free lowercase hyphen phrase is the weak-name release's class at any
    length, so ``key:`` no longer masks it even with no declaration. Pinned so a
    re-added length bound reds here and in the corpus.
    """
    text = "key: " + SLUG
    scrubbed, report = store.redact_with_report(text)
    assert scrubbed == text
    assert report.reached_model is False


def test_a_registered_identifier_is_byte_identical_under_a_weak_name() -> None:
    store = VariableStore(cwd=".")
    assert store.register_identifier(SLUG) is True
    text = "key: " + SLUG
    scrubbed, report = store.redact_with_report(text)
    assert scrubbed == text
    assert report.labels == ()
    assert report.reached_model is False
    # Declared, not credential: the two namespaces stay separate.
    assert store.identifier_values() == [SLUG]
    assert SLUG not in store.redaction_values()


def test_a_registered_identifier_is_released_under_strong_names() -> None:
    """An explicit exemption outranks a strong label.

    ``token``/``secret``/``password`` are exactly the names whose masks the weak
    release does not consult; a render site that declares an identifier has
    decided the value is a name, and the strong name cannot override that.
    """
    store = VariableStore(cwd=".")
    assert store.register_identifier(SLUG_60) is True
    text = "\n".join(
        [
            "token: " + SLUG_60,
            "secret: " + SLUG_60,
            "password: " + SLUG_60,
            "api_key: " + SLUG_60,
        ]
    )
    scrubbed, report = store.redact_with_report(text)
    assert scrubbed == text
    assert report.reached_model is False


def test_the_same_store_still_masks_real_credentials() -> None:
    """Registration must not become a blanket release for the session."""
    store = VariableStore(cwd=".")
    assert store.register_identifier(SLUG) is True
    text = f"api_key: {HEX_32}\ntoken: {MIXED_TOKEN}"
    scrubbed, report = store.redact_with_report(text)
    assert HEX_32 not in scrubbed
    assert MIXED_TOKEN not in scrubbed
    assert "[redacted]" in scrubbed
    assert report.reached_model is False


def test_an_identifier_inside_a_known_secret_is_not_protected() -> None:
    """All-or-none: protecting it would split the secret's bytes.

    The identifier is a substring of a value the session already knows is a
    secret, so the guard skips the declaration and the exact pass masks the
    secret WHOLE — no readable slug fragment survives.
    """
    store = VariableStore(cwd=".")
    secret = "prefix-" + SLUG
    assert store.register_redaction(secret) is True
    assert store.register_identifier(SLUG) is True
    scrubbed, _ = store.redact_with_report("value: " + secret)
    assert secret not in scrubbed
    assert SLUG not in scrubbed
    assert scrubbed.rstrip().endswith("[redacted]")


def test_an_identifier_containing_a_known_secret_is_not_protected() -> None:
    """The other direction of the same guard."""
    store = VariableStore(cwd=".")
    secret = "-".join(SLUG.split("-")[:2])
    assert store.register_redaction(secret) is True
    assert store.register_identifier(SLUG) is True
    scrubbed, _ = store.redact_with_report("value: " + SLUG)
    assert secret not in scrubbed
    assert SLUG not in scrubbed


def test_an_identifier_inside_a_longer_unknown_value_masks_whole() -> None:
    """A mask that swallows the placeholder is never partially restored."""
    store = VariableStore(cwd=".")
    assert store.register_identifier(SLUG) is True
    value = "-".join(["assigned", SLUG, "value"])
    scrubbed, report = store.redact_with_report("api_key: " + value)
    assert SLUG not in scrubbed
    assert "assigned" not in scrubbed
    assert not any(0xE000 <= ord(char) <= 0xF8FF for char in scrubbed)
    assert report.reached_model is False


@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "line\nbreak",
        "carriage\rreturn",
        "nul\x00byte",
        "private\ue000use",
        "x" * 513,
    ],
)
def test_registration_refuses_framing_and_oversize(store: VariableStore, value: str) -> None:
    assert store.register_identifier(value) is False
    assert store.identifier_values() == []


def test_a_512_character_identifier_is_accepted(store: VariableStore) -> None:
    assert store.register_identifier("x" * 512) is True


@pytest.mark.asyncio
async def test_project_show_and_list_keep_the_key_readable(tmp_path) -> None:
    """The real render path: a title sharing words with the slug must not fire.

    This is the incident's exact shape — the slug and a title in the same
    record, both registered before either line is returned — and the assertions
    include the store's own verdict, not just the substring.
    """
    store = VariableStore(cwd=".")
    registry = ProjectRegistry(tmp_path)
    registry.create_project(
        ProjectEdit(
            name=SLUG,
            title="Export status reports for the platform rollout",
            description="Key redaction regression fixture.",
        ),
        sessions=[],
    )
    context = ToolContext(cwd=".", session_id=SESSION, project_registry=registry, variables=store)
    shown = await execute_project("tc", {"op": "show", "name": SLUG}, None, None, context)
    assert "key: " + SLUG in shown.text, shown.text
    listed = await execute_project("tc", {"op": "list"}, None, None, context)
    assert SLUG in listed.text, listed.text
    for result in (shown, listed):
        assert "[redacted]" not in result.text
        scrubbed, report = store.redact_with_report(result.text)
        assert scrubbed == result.text
        assert report.labels == ()
        assert report.reached_model is False


@pytest.mark.asyncio
async def test_team_show_and_list_keep_the_name_readable(tmp_path) -> None:
    store = VariableStore(cwd=".")
    registry = TeamRegistry(tmp_path)
    registry.create_team(
        TeamEditFields(
            name=SLUG,
            label="Project Keys Export",
            manager="manager",
            members=[TeamMember(role=MEMBER_ROLE)],
        )
    )
    context = ToolContext(cwd=".", team_registry=registry, variables=store)
    for args in ({"op": "show", "name": SLUG}, {"op": "list"}):
        result = await execute_team("tc", args, None, None, context)
        assert SLUG in result.text, result.text
        assert "[redacted]" not in result.text
        scrubbed, report = store.redact_with_report(result.text)
        assert scrubbed == result.text
        assert report.reached_model is False


@pytest.mark.asyncio
async def test_agent_show_keeps_the_role_name_readable(tmp_path) -> None:
    """A registered role with the released class as its name, rendered for real.

    The agent registry is built directly (the tool's create path is a separate
    surface); the ``role`` tag is what makes it a ROLE rather than a specialist,
    so ``show`` resolves it through the same reader the tool uses.
    """
    from local_operator.agents import AgentEditFields, AgentRegistry

    store = VariableStore(cwd=".")
    registry = AgentRegistry(tmp_path)
    # ``**fields`` like the other tests, because pyright reads the pydantic
    # Field(...) declarations without plain defaults as required arguments.
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
        assert SLUG in result.text, result.text
        assert "[redacted]" not in result.text
        scrubbed, report = store.redact_with_report(result.text)
        assert scrubbed == result.text
        assert report.reached_model is False


def _note(context: ToolContext | None, *values: Any) -> None:
    """The helper under test, imported lazily to keep this file's imports flat."""
    from local_operator.tools.builtin import note_identifiers

    note_identifiers(context, *values)


def test_note_identifiers_is_a_no_op_without_a_capable_store() -> None:
    _note(None, SLUG)
    _note(ToolContext(cwd="."), SLUG)
    assert VariableStore(cwd=".").identifier_values() == []


def test_note_identifiers_skips_falsy_values(store: VariableStore) -> None:
    _note(ToolContext(cwd=".", variables=store), None, "", SLUG)
    assert store.identifier_values() == [SLUG]
