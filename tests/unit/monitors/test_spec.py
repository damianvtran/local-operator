"""Spec construction and validation (contract §4.4, §10.1, §11.4)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest
from pydantic import ValidationError

from local_operator.monitors import spec as monitor_spec
from local_operator.monitors.settings import MonitorSettings


def _spec_of(outcome: Any) -> Any:
    assert "spec" in outcome
    return cast("monitor_spec.MonitorBuilt", outcome)["spec"]


def _error_of(outcome: Any) -> Any:
    assert "error" in outcome
    return cast("monitor_spec.MonitorBuildFailed", outcome)


def settings(**overrides: Any) -> MonitorSettings:
    import dataclasses

    base = MonitorSettings()
    if not overrides:
        return base
    return dataclasses.replace(base, **overrides)


def build(
    request: Mapping[str, Any], *, monitor_id: str = "m1", now_ms: int = 1_756_000_000_000
) -> Any:
    return monitor_spec.build_monitor_spec(
        request,
        monitor_id=monitor_id,
        now_ms=now_ms,
        settings=settings(),
        cwd="/work",
        validate=lambda tool, args: None,
    )


def test_a_minimal_request_builds_a_durable_spec() -> None:
    outcome = build({"tool": "bash", "arguments": {"command": "date -u"}})
    spec = _spec_of(outcome)
    assert spec.id == "m1"
    assert spec.name == "bash"  # derived from the tool when omitted
    assert spec.every_ms == 60_000
    assert spec.until_at is None
    assert spec.cwd == "/work"
    assert spec.notify is False and spec.sort_lines is False and spec.ignore == []


def test_the_interval_floor_refuses_without_calling_it_malformed() -> None:
    failed = _error_of(build({"tool": "bash", "arguments": {"command": "date"}, "every": "5s"}))
    assert failed["malformed"] is False
    assert "30s" in failed["error"]


def test_a_bad_duration_is_malformed() -> None:
    soon = _error_of(build({"tool": "bash", "arguments": {"command": "date"}, "every": "soon"}))
    assert soon["malformed"] is True


def test_until_parses_and_a_past_until_is_refused() -> None:
    outcome = build(
        {"tool": "bash", "arguments": {"command": "date"}, "until": "2030-01-01T00:00:00"}
    )
    assert _spec_of(outcome).until_at is not None
    failed = _error_of(
        build({"tool": "bash", "arguments": {"command": "date"}, "until": "2001-01-01"})
    )
    assert failed["malformed"] is False
    assert "past" in failed["error"]


def test_ignore_bounds_and_regex_validation() -> None:
    outcome = build(
        {
            "tool": "bash",
            "arguments": {"command": "date"},
            "ignore": ["^a", "^b", "^c", "^d", "^e", "^f", "^g", "^h", "^i"],
        }
    )
    assert _error_of(outcome)["malformed"] is True
    failed = _error_of(
        build({"tool": "bash", "arguments": {"command": "date"}, "ignore": ["([unclosed"]})
    )
    assert failed["malformed"] is True
    assert "invalid ignore regex" in failed["error"]


def test_name_bound_is_enforced() -> None:
    long_name = _error_of(
        build({"tool": "bash", "arguments": {"command": "date"}, "name": "x" * 200})
    )
    assert long_name["malformed"] is True


def test_shape_validation_precedes_the_read_only_gate() -> None:
    # The gate is the expensive check; a request that fails shape validation
    # must not pay for it — and must fail for the SHAPE reason.
    calls: list[tuple[str, Mapping[str, Any]]] = []

    def validate(tool: str, args: Mapping[str, Any]) -> str | None:
        calls.append((tool, args))
        return "refused"

    outcome = monitor_spec.build_monitor_spec(
        {"arguments": {"x": 1}},
        monitor_id="m1",
        now_ms=1,
        settings=settings(),
        cwd="",
        validate=validate,
    )
    assert _error_of(outcome)["malformed"] is True
    assert calls == []


def test_the_read_only_gate_refuses_the_build() -> None:
    outcome = monitor_spec.build_monitor_spec(
        {"tool": "eval", "arguments": {"code": "1+1"}},
        monitor_id="m1",
        now_ms=1,
        settings=settings(),
        cwd="",
        validate=lambda tool, args: f"monitor can't watch {tool!r}",
    )
    failed = _error_of(outcome)
    assert failed["malformed"] is False
    assert "can't watch" in failed["error"]


def test_spec_model_refuses_extra_fields_and_the_field_floor() -> None:
    with pytest.raises(ValidationError):
        monitor_spec.MonitorSpec(
            id="m1",
            name="x",
            tool="bash",
            arguments={},
            every_ms=60_000,
            surprise=1,  # type: ignore[call-arg]
        )
    with pytest.raises(ValidationError):
        monitor_spec.MonitorSpec(id="m1", name="x", tool="bash", arguments={}, every_ms=10)


def test_spec_identity_is_a_canonical_hash_not_a_name() -> None:
    first = monitor_spec.spec_identity("bash", {"command": "date", "x": 1})
    second = monitor_spec.spec_identity("bash", {"x": 1, "command": "date"})
    third = monitor_spec.spec_identity("bash", {"command": "date"})
    assert first == second
    assert first != third
    assert monitor_spec.spec_identity("web_fetch", {"url": "u"}) != first


def test_ids_are_never_reused_after_a_cancel() -> None:
    # The high-water rule: `next_seq` is persisted beside the rows, and the
    # allocation is above BOTH the stored mark and any live id, so a cancelled
    # m1 cannot come back as m1.
    assert monitor_spec.next_monitor_seq([]) == 1
    assert monitor_spec.next_monitor_seq([], 5) == 5
    assert monitor_spec.next_monitor_seq(["m1", "m3"], 2) == 4
    assert monitor_spec.next_monitor_seq(["m1"], 99) == 99
    assert monitor_spec.allocate_monitor_id(["m1", "m2"], high_water=2) == ("m3", 4)
    assert monitor_spec.allocate_monitor_id([], high_water=7) == ("m7", 8)
    # A hand-edited transcript cannot lower the mark or break the parse.
    assert monitor_spec.next_monitor_seq(["not-an-id", "m2"], "junk") == 3


def test_a_shape_refusal_from_the_real_validator_is_not_malformed() -> None:
    """The arm path's classification for the §D2 refusal: a call that is
    well-formed but NOT runnable is a conflict ("this cannot be watched"), not a
    malformed request — the same split the interval floor and the read-only gate
    take, and what makes the desktop route answer 409 rather than 422.
    """
    from local_operator.monitors.readonly import external_monitor_verdict

    outcome = monitor_spec.build_monitor_spec(
        {"tool": "glob", "arguments": {"pattern": "*.py", "path": "/tmp"}},
        monitor_id="m1",
        now_ms=1,
        settings=settings(),
        cwd="",
        validate=lambda tool, args: external_monitor_verdict(tool, args),
    )
    failed = _error_of(outcome)
    assert failed["malformed"] is False
    assert 'unknown argument(s) "path"' in failed["error"]
