"""The timeout parameter's bounds, the derivation of urgency, and the flag.

The parse is where a 1000x bug would ship (``parse_wake_duration`` returns
MILLISECONDS), so ``"2h"`` is asserted to be 7200 seconds rather than asserted to
be "not None" — design ``docs/design/ask-nonblocking.md`` §2.1/§7.
"""

from __future__ import annotations

import pytest

from local_operator.asks import policy


def test_the_default_is_one_hour():
    assert policy.DEFAULT_TIMEOUT_S == 3600
    assert policy.parse_timeout_param(None) == (3600, None)


def test_an_integer_is_seconds():
    assert policy.parse_timeout_param(600) == (600, None)
    assert policy.parse_timeout_param(86400) == (86400, None)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("2h", 7200), ("30m", 1800), ("1h30m", 5400), ("10m", 600), ("120s", 120)],
)
def test_a_duration_string_is_converted_from_milliseconds(text: str, seconds: int):
    """``"2h"`` ⇒ 7200 s. Without the ``// 1000`` this reads 7,200,000."""
    assert policy.parse_timeout_param(text) == (seconds, None)


def test_the_bounds_are_inclusive():
    assert policy.parse_timeout_param(120)[1] is None
    assert policy.parse_timeout_param(86400)[1] is None


@pytest.mark.parametrize("bad", [119, 0, -5, 86401, "90s", "25h", "junk", "", "two hours"])
def test_out_of_range_is_a_validation_error_naming_the_bounds(bad):
    seconds, error = policy.parse_timeout_param(bad)
    assert seconds == 0
    assert error is not None
    assert "120" in error and "86400" in error
    assert "seconds" in error


def test_a_bool_is_refused_rather_than_read_as_one_second():
    """``True`` is an ``int`` in Python; a model writing ``timeout: true`` means
    nothing this tool can honour, so it is rejected instead of becoming 1 s."""
    assert policy.parse_timeout_param(True)[1] is not None
    assert policy.parse_timeout_param(False)[1] is not None


def test_a_numeric_string_is_seconds():
    assert policy.parse_timeout_param("600") == (600, None)


def test_urgency_is_derived_at_fifteen_minutes():
    assert policy.is_urgent(900) is True
    assert policy.is_urgent(901) is False
    assert policy.is_urgent(policy.MIN_TIMEOUT_S) is True
    assert policy.is_urgent(policy.MAX_TIMEOUT_S) is False


def test_the_floor_is_at_least_two_wake_ticks():
    """A sub-tick deadline would be decided by timer granularity, not by the
    model's choice — the design's stated reason for the 2-minute floor."""
    from local_operator.harness.wake import MAX_ARM_MS

    assert policy.MIN_TIMEOUT_S * 1000 >= 2 * MAX_ARM_MS


def test_the_flag_defaults_on_and_the_env_var_is_a_kill_switch(monkeypatch):
    """The queue is the DEFAULT; ``LOP_ASK_NONBLOCKING=0`` is the escape hatch.

    The DIRECTION of the membership test matters as much as the set: a variable
    that is ABSENT — or present but empty — must leave the shipped default, so
    the kill set is named explicitly and everything outside it (including a
    typo) leaves the queue on. A kill switch a typo could arm would fail in the
    one direction that hurts the operator it exists for.
    """
    import importlib

    for value in ("0", "false", "no", "off", "OFF", " false "):
        monkeypatch.setenv("LOP_ASK_NONBLOCKING", value)
        module = importlib.reload(policy)
        assert module.NONBLOCKING_ASK is False, value
        assert module.enabled() is False
    for value in ("", "1", "true", "YES", "on", "maybe"):
        monkeypatch.setenv("LOP_ASK_NONBLOCKING", value)
        module = importlib.reload(policy)
        assert module.NONBLOCKING_ASK is True, value
    monkeypatch.delenv("LOP_ASK_NONBLOCKING", raising=False)
    module = importlib.reload(policy)
    assert module.NONBLOCKING_ASK is True
    # Leave the module on the shipped default: the reload above is the last
    # word on the attribute, and monkeypatch only restores the ENV.
    assert module.enabled() is True


def test_enabled_follows_the_module_attribute(monkeypatch):
    """``enabled()`` exists so ONE monkeypatch governs every path; the attribute
    stays the setting and nothing caches it."""
    assert policy.enabled() is policy.NONBLOCKING_ASK
    monkeypatch.setattr(policy, "NONBLOCKING_ASK", True)
    assert policy.enabled() is True


def test_the_late_window_is_seven_days():
    assert policy.LATE_WINDOW_S == 7 * 24 * 3600


def test_the_open_cap_is_eight():
    assert policy.OPEN_ASK_CAP == 8
