"""Wire-only refusal localization; launch diagnostics remain English."""

from types import SimpleNamespace

import pytest

from local_operator.harness import subagent
from local_operator.server.utils import desktop_profiles as projection
from local_operator.teams import TeamMember


@pytest.mark.parametrize(
    "raw,kind",
    [
        (None, "tier_unconfigured"),
        (42, "tier_invalid_value"),
        ({"model": "bad", "effort": "invalid"}, "tier_invalid_mapping"),
        ("bad", "tier_invalid_selector"),
    ],
)
def test_resolver_classifies_at_the_original_producer(monkeypatch, raw, kind):
    monkeypatch.setattr(subagent, "_row_effort_tier", lambda *args: None)
    result = subagent.resolve_member_model(
        "synthetic", launch_tier="hi", tier_targets={"hi": (raw, None)}
    )
    assert result.unavailable_kind == kind
    assert result.unavailable == (
        "no model configured at subagents.models.hi"
        if raw is None
        else subagent.malformed_tier_reason("hi", raw)
    )
    assert result.unavailable_quiet is (raw is None)
    assert result.selector is None
    entry = projection._effective_member(
        TeamMember(role="synthetic", tier="hi"), result, None, locale="en"
    )
    assert entry["reason_message"]["code"] == f"wire.teams.effective.{kind}"
    assert entry["reason_message"]["params"] == {"tier": "hi"}
    assert entry["reason"] == entry["reason_message"]["text"]
    assert entry["available"] is False


def test_config_read_error_classification_preserves_diagnostic(monkeypatch):
    monkeypatch.setattr(subagent, "_row_effort_tier", lambda *args: None)

    def broken():
        raise ValueError("RAW_ENGLISH_EXCEPTION_MARKER")

    monkeypatch.setattr(subagent, "read_effort_tier_targets", broken)
    result = subagent.resolve_member_model("synthetic", launch_tier="hi")
    assert result.unavailable_kind == "config_unreadable"
    assert result.unavailable == "config could not be read (RAW_ENGLISH_EXCEPTION_MARKER)"
    entry = projection._effective_member(TeamMember(role="synthetic"), result, None, locale="en")
    assert entry["reason_message"]["code"] == "wire.teams.effective.config_unreadable"
    assert entry["reason_message"]["params"] == {}
    assert "RAW_ENGLISH_EXCEPTION_MARKER" not in str(entry)


@pytest.mark.parametrize("kind", [None, "future_unknown_kind"])
def test_unknown_classification_uses_safe_whole_message(kind):
    result = SimpleNamespace(
        tier="hi",
        tier_source="team",
        effort=None,
        effort_source="model_default",
        selector=None,
        unavailable="RAW_ENGLISH_EXCEPTION_MARKER",
        unavailable_kind=kind,
    )
    entry = projection._effective_member(TeamMember(role="synthetic"), result, None, locale="en")
    assert entry["reason_message"]["code"] == "wire.teams.effective.unavailable"
    assert "RAW_ENGLISH_EXCEPTION_MARKER" not in str(entry)


def test_missing_nested_team_and_unreadable_registry(monkeypatch):
    member = TeamMember(role="synthetic-nested", kind="team")
    team = SimpleNamespace(name="synthetic", manager="manager", members=[member])
    teams = SimpleNamespace(get_team_by_name=lambda name: team if name == "synthetic" else None)
    row = projection.team_effective(None, teams, "synthetic", locale="en")["members"][0]
    assert row["reason_message"]["code"] == "wire.teams.effective.team_missing"
    assert row["reason_message"]["params"] == {"role": "synthetic-nested"}
    team.members = [TeamMember(role="synthetic-member")]

    def broken(*args, **kwargs):
        raise ValueError("RAW_ENGLISH_EXCEPTION_MARKER")

    monkeypatch.setattr(subagent, "resolve_member_model", broken)
    row = projection.team_effective(None, teams, "synthetic", locale="en")["members"][0]
    assert row["reason_message"]["code"] == "wire.teams.effective.registry_unreadable"
    assert "RAW_ENGLISH_EXCEPTION_MARKER" not in str(row)


def test_model_build_failure_does_not_leak_exception(monkeypatch):
    from local_operator.model import configure

    def broken(*args):
        raise ValueError("RAW_ENGLISH_EXCEPTION_MARKER")

    monkeypatch.setattr(configure, "build_model_spec", broken)
    result = subagent.MemberModelResolution("hi", "team", "provider/model", "high", "team")
    row = projection._effective_member(TeamMember(role="synthetic"), result, None, locale="en")
    assert row["reason_message"]["code"] == "wire.teams.effective.model_unresolvable"
    assert row["reason_message"]["params"] == {"selector": "provider/model"}
    assert row["reason"] == row["reason_message"]["text"]
    assert "RAW_ENGLISH_EXCEPTION_MARKER" not in str(row)
    assert row["effort"]["value"] == "high"


def test_success_and_sentinel_do_not_gain_a_reason(monkeypatch):
    from local_operator.model import configure

    monkeypatch.setattr(
        configure,
        "build_model_spec",
        lambda *args: SimpleNamespace(
            reasoning_effort="high",
            reasoning_efforts=None,
        ),
    )
    for selector in (None, "default", "provider/model"):
        result = subagent.MemberModelResolution("hi", "team", selector, "high", "team")
        row = projection._effective_member(TeamMember(role="synthetic"), result, None, locale="en")
        assert row["available"] is True
        assert row["reason"] == ""
        assert "reason_message" not in row
        assert row["effort"]["value"] == "high"
