"""Console admission settings are policy, not process termination permissions."""

import math

import pytest

from local_operator.ui_console.policy import effective_policy


def test_defaults_are_bounded_and_match_desktop_contract() -> None:
    assert effective_policy({}) == {"max_live": 128, "max_owner": 8, "user_reserved": 16}


@pytest.mark.parametrize("raw", [None, True, "unlimited", [], float("nan"), math.inf])
def test_invalid_values_never_mean_unlimited(raw: object) -> None:
    assert effective_policy({"console": {"max_live": raw}})["max_live"] == 128


def test_hand_edits_clamp_and_user_reservation_tracks_ceiling() -> None:
    assert effective_policy(
        {"console": {"max_live": 1, "max_owner": 999, "user_reserved": 999}}
    ) == {
        "max_live": 32, "max_owner": 32, "user_reserved": 8,
    }
    assert effective_policy({"console": {"max_live": 999, "max_owner": 0, "user_reserved": 0}}) == {
        "max_live": 256, "max_owner": 1, "user_reserved": 4,
    }
