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
    assert values.default_enrollment == ""


def test_reads_from_a_nested_config_document():
    values = tb.TeamBridgeSettings.from_values(
        {
            "team_bridge": {
                "enabled": True,
                "onboarding": "seen",
                "session": "sess-123",
                "default_enrollment": "done",
            }
        }
    )
    assert values.enabled is True
    assert values.onboarding == "seen"
    assert values.session == "sess-123"
    assert values.default_enrollment == "done"


def test_untrusted_values_normalise_fail_closed():
    """A hand-edited value must read as the shipped default, never crash.

    ``enabled``: a non-bool reads as the default, which is ON under the
    opt-out policy — the key that actually turns it off is a real ``false``.
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
    assert values.enabled is True
    assert values.onboarding == ""
    assert values.session == ""
    assert values.default_enrollment == ""

    # A hand-edited STRING "false" is not an opt-out: only a real bool is.
    assert tb.TeamBridgeSettings.from_values({"team_bridge": {"enabled": "false"}}).enabled is True


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
