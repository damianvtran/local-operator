"""The ``team_bridge.*`` readers: defaults, fail-closed normalisation, nested paths."""

from __future__ import annotations

from local_operator.team_bridge import settings as tb


def test_defaults_when_the_section_is_absent():
    values = tb.TeamBridgeSettings.from_values({})
    # The opt-out default: an absent key means ON. Everything that actually
    # gates a run (eligibility, consent, launch) is server-side.
    assert values.enabled is True
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

    ``enabled``: a non-bool in a PRESENT value reads as OFF — the ON default
    writes consent upstream, so an unreadable value must not consent. Absence
    is the shipped default (ON); the key that turns it off is a real ``false``.
    ``onboarding``: anything but ``seen`` reads as "not dismissed", which
    allows the one-time notice to show again rather than suppressing it.
    ``session``: a non-string (including a list from the key's earlier draft
    shape) reads as "none designated" — wrongly naming a wake destination is
    worse than naming none. ``default_enrollment``: only ``done`` counts, so a
    typo cannot suppress the one default consent write.
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

    # A present-but-unreadable value FAILS CLOSED: the ON default performs a
    # consent write, so a value the reader cannot parse must not be the thing
    # that consents. Absence is the shipped default (ON).
    assert tb.TeamBridgeSettings.from_values({"team_bridge": {"enabled": "false"}}).enabled is False
    assert tb.TeamBridgeSettings.from_values({"team_bridge": {"enabled": None}}).enabled is False
    assert tb.TeamBridgeSettings.from_values({}).enabled is True


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
    """Screening is an invariant, not a key: nothing registered may turn it off.

    RFC §13.9/item 1 and §8: screening is the admission gate, so an editable
    "screening" switch would license exactly what the never-list forbids. The
    guard is over the LIVE registry — a future row that reintroduces one fails
    here by name, rather than relying on review attention.
    """

    from local_operator import settings_io

    offenders = [
        setting.key
        for setting in settings_io.SETTINGS
        if setting.section == "team_bridge"
        and ("screen" in setting.key or "screen" in setting.help.lower())
    ]
    assert offenders == []


def test_session_ids_are_stripped():
    values = tb.TeamBridgeSettings.from_values({"team_bridge": {"session": " sess-1 "}})
    assert values.session == "sess-1"
