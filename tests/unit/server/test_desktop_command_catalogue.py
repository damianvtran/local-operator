"""The command catalogue and the `commands` capability, on the wire.

Why this lives apart from the registry's own tests: these two assertions are
about what a RENDERER is told, not about what the registry holds. The catalogue
row is the one place a host-specific fact crosses to the desktop, and the
capability is the only way a renderer can tell whether a leading-slash refusal
still describes a reachable state.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from local_operator.server.app import app
from local_operator.slash_commands import SLASH_COMMANDS


def test_the_catalogue_carries_prefixes_text_on_every_row(monkeypatch):
    """The new field crosses additively, and it is the REGISTRY's value.

    Additive means an older renderer ignores the key rather than failing to
    parse the row, and a renderer facing an older backend falls back to its own
    `promptCommands ∪ inlineArgument` derivation. Comparing against
    ``SLASH_COMMANDS`` rather than a literal keeps the wire from becoming a
    second registry that can drift.
    """
    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "catalogue-token")
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)

    with TestClient(app) as client:
        response = client.get(
            "/v1/desktop/commands", headers={"Authorization": "Bearer catalogue-token"}
        )
        assert response.status_code == 200
        rows = response.json()["result"]["commands"]

    by_name = {spec.name: spec for spec in SLASH_COMMANDS if spec.desktop_destination}
    assert {row["name"] for row in rows} == set(by_name)
    for row in rows:
        assert "prefixes_text" in row, row["name"]
        assert row["prefixes_text"] == by_name[row["name"]].prefixes_text, row["name"]

    # The vocabulary itself, stated on the wire rather than only in the registry,
    # because this is the set the messages endpoint's rule reads.
    assert {row["name"] for row in rows if row["prefixes_text"]} == {
        "goal",
        "loop",
        "btw",
        "fork",
        "team",
        "agent",
        "model",
        "effort",
        "approvals",
        "theme",
    }


def test_the_catalogue_carries_the_argument_shape_and_its_vocabulary(monkeypatch):
    """The THIRD source crosses too, with the vocabulary a renderer needs.

    `prefixes_text` says "the composer can complete text here"; it does not say
    that `/mcp logout` and `/login openai` are commands the desktop RUNS. A
    renderer reading only the booleans plans those as prose — messages the
    composer would have run — so the shape and its vocabulary are the fields that
    let the two hosts answer `/mcp logout` (command) and `/mcp logout seems to
    cause a crash` (message) identically.

    `argument_words` must be the SAME vocabulary the validator reads (the provider
    registry, the MCP subcommand tuple), which is why it is compared against those
    sources rather than a literal: a hand-copied list is the drift this field
    exists to prevent.
    """
    from local_operator.mcp.config import SERVER_NAME_RE
    from local_operator.providers.registry import known_provider_ids
    from local_operator.session.frontend_state import MCP_SUBCOMMANDS
    from local_operator.slash_commands import _is_mcp_invocation, _is_provider

    monkeypatch.setenv("LOCAL_OPERATOR_DESKTOP_TOKEN", "catalogue-token")
    monkeypatch.delenv("LOCAL_OPERATOR_DESKTOP_ORIGINS", raising=False)

    with TestClient(app) as client:
        response = client.get(
            "/v1/desktop/commands", headers={"Authorization": "Bearer catalogue-token"}
        )
        assert response.status_code == 200
        rows = response.json()["result"]["commands"]

    by_name = {spec.name: spec for spec in SLASH_COMMANDS if spec.desktop_destination}
    for row in rows:
        assert row["argument_shape"] == by_name[row["name"]].argument_shape.value, row["name"]

    shapes = {row["name"]: row["argument_shape"] for row in rows}
    assert shapes["mcp"] == "subcommand"
    assert shapes["login"] == "provider"
    assert shapes["rename"] == "any"
    assert shapes["usage"] == "word"
    assert shapes["compact"] == "none"

    words = {row["name"]: row["argument_words"] for row in rows}
    assert words["mcp"] == sorted(MCP_SUBCOMMANDS)
    assert words["login"] == list(known_provider_ids())
    # Every other shape takes any word, so the list is empty rather than absent:
    # an older renderer ignores the key, and a newer one must not read "empty" as
    # "no argument may be typed".
    assert words["usage"] == [] and words["compact"] == [] and words["rename"] == []

    # And the wire vocabulary really is what the validators answer with, so a
    # renderer applying it reaches the endpoint's own decision.
    for provider in known_provider_ids():
        assert _is_provider(provider) is True
    assert _is_provider("zzz") is False
    assert _is_mcp_invocation(words["mcp"][0]) is True
    assert _is_mcp_invocation("logout seems to cause a crash") is False
    assert SERVER_NAME_RE.fullmatch("my-server") is not None


def test_the_commands_capability_reports_the_narrowed_policy():
    """`commands: 2` is the renderer's only signal for which refusal it may see.

    No surface is gated on this number (nothing is withheld by it), so it is a
    BUMP rather than a new key: the single consumer is the alert's remedy. A
    renderer on < 2 tells the user to update the backend, because on that pairing
    a legitimate message really can be refused; on >= 2 the refusal means a client
    bug and the sentence stands alone. Left unpinned, a later edit dropping the
    bump would silently put the UI's remedy back on the wrong branch.
    """
    with TestClient(app) as client:
        result = client.get("/v1/capabilities").json()["result"]

    assert result["features"]["commands"] == 2
