"""The ``team_bridge.*`` readers: defaults, fail-closed normalisation, nested paths."""

from __future__ import annotations

from local_operator.team_bridge import settings as tb


def test_defaults_when_the_section_is_absent():
    values = tb.TeamBridgeSettings.from_values({})
    # The consent hold: an absent key means OFF until a consent version with
    # opt-out copy is served and accepted. Everything that actually gates a run
    # (eligibility, consent, launch) is server-side too.
    assert values.enabled is False
    assert values.onboarding == ""
    assert values.session == ""
    assert values.default_enrollment == {}


def test_reads_from_a_nested_config_document():
    values = tb.TeamBridgeSettings.from_values(
        {
            "team_bridge": {
                "enabled": True,
                "onboarding": "seen",
                "session": "sess-123",
                "default_enrollment": {"acc_1": "done"},
            }
        }
    )
    assert values.enabled is True
    assert values.onboarding == "seen"
    assert values.session == "sess-123"
    assert values.default_enrollment == {"acc_1": "done"}


def test_untrusted_values_normalise_fail_closed():
    """A hand-edited value must read as the shipped default, never crash.

    ``enabled``: a non-bool in a PRESENT value reads as OFF; absence is the
    shipped default (also OFF during the consent hold), and only a real ``true``
    opts this machine in. ``onboarding``: anything but ``seen`` reads as "not
    dismissed", which allows the one-time notice to show again rather than
    suppressing it. ``session``: a non-string (including a list from the key's
    earlier draft shape) reads as "none designated" — wrongly naming a wake
    destination is worse than naming none. ``default_enrollment``: only
    ``done`` counts, so a typo cannot turn a stored marker into authority.
    """

    values = tb.TeamBridgeSettings.from_values(
        {
            "team_bridge": {
                "enabled": "true",
                "onboarding": "yes",
                "session": ["sess-1", "sess-2"],
                "default_enrollment": "yes",
            }
        }
    )
    assert values.enabled is False
    assert values.onboarding == ""
    assert values.session == ""
    assert values.default_enrollment == {}

    # A present-but-unreadable value never reads as a coerced ON; only an
    # explicit real bool does. Absence is the shipped default (OFF).
    assert tb.TeamBridgeSettings.from_values({"team_bridge": {"enabled": "false"}}).enabled is False
    assert tb.TeamBridgeSettings.from_values({"team_bridge": {"enabled": None}}).enabled is False
    assert tb.TeamBridgeSettings.from_values({}).enabled is False


def test_the_default_enrollment_marker_is_per_account(tmp_path):
    """One member enrolling must not suppress another on the same machine."""

    from local_operator.config import ConfigManager

    tb.record_default_enrollment(tmp_path, "acc_1")
    tb.record_default_enrollment(tmp_path, "acc_2")
    values = tb.TeamBridgeSettings.from_config(ConfigManager(tmp_path))

    assert values.default_enrollment == {"acc_1": "done", "acc_2": "done"}
    # A repeat is a no-op, not an error.
    tb.record_default_enrollment(tmp_path, "acc_1")
    assert tb.TeamBridgeSettings.from_config(ConfigManager(tmp_path)).default_enrollment == {
        "acc_1": "done",
        "acc_2": "done",
    }


def test_no_setting_can_disable_the_admission_gate():
    """Screening is an invariant, not a switch: nothing registered may turn it off.

    RFC §13.9/item 1 and §8: screening is the admission gate, so an editable
    "screening" switch would license exactly what the never-list forbids.

    WHY THIS IS A CAPABILITY CHECK RATHER THAN A NAME MATCH. It used to refuse
    any ``team_bridge`` row whose key or help mentioned "screen", which was
    right while screening had no knobs at all. It now has four SPEND CEILINGS
    (``team_bridge.screening.*`` — the 10x raise, on measured demand), and those
    are not the gate: they say how much screening may COST, never whether it
    runs. A name match would fail on exactly the rows the feature needs, and the
    tempting repair would be to weaken the guard. So the intent is asserted
    directly: every ``team_bridge`` row that mentions screening must be an INT
    with ``minimum >= 1``, i.e. there is no zero, negative, boolean or
    enumerated value any writer could store that stops the gate.

    The KEY SET is pinned as well. A new screening-related row must be looked at
    by a person and added here, rather than sliding in under a passing test —
    the same reason the old guard named offenders instead of counting them.
    """

    from local_operator import settings_io

    screening_rows = [
        setting
        for setting in settings_io.SETTINGS
        if setting.section == "team_bridge"
        and ("screen" in setting.key or "screen" in setting.help.lower())
    ]
    assert [setting.key for setting in screening_rows] == [
        "team_bridge.screening.budget_per_hour",
        "team_bridge.screening.ambient_budget_per_hour",
        "team_bridge.screening.unaddressed_budget_per_hour",
        "team_bridge.screening.sender_budget_per_hour",
    ]
    for setting in screening_rows:
        # A ceiling, never a switch: an INT whose floor is one. OFF would need a
        # boolean or a zero, and neither is storable here.
        assert setting.kind is settings_io.Kind.INT, setting.key
        assert setting.minimum == 1, setting.key


def test_session_ids_are_stripped():
    values = tb.TeamBridgeSettings.from_values({"team_bridge": {"session": " sess-1 "}})
    assert values.session == "sess-1"
