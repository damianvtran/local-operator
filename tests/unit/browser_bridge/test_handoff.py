"""Handle matching and record notes for child → parent browser handoff (PR #3821)."""

from __future__ import annotations

import pytest

from local_operator.browser_bridge.handoff import surface_matches
from local_operator.browser_bridge.resources import cleanup_disposition

SURFACE = "bridge:100:6b055c9f0a1d2e3f"


@pytest.mark.parametrize("handle", [SURFACE, "bridge:100", "bridge:100:6b055c…", " bridge:100:6…"])
def test_surface_matches_accepts_listing_forms(handle: str) -> None:
    assert surface_matches(SURFACE, handle)


@pytest.mark.parametrize(
    "handle",
    ["…", "b…", "bridge…", "bridge:…", "bridge:100:…", "bridge:10", "bridge:101:6b055c…", ""],
)
def test_surface_matches_rejects_bare_or_short_prefixes(handle: str) -> None:
    """R3: any trailing '…' used to be a prefix, so '…' matched every tab."""
    assert not surface_matches(SURFACE, handle)


def test_cleanup_disposition_names_the_parent_a_tab_was_handed_to() -> None:
    """R4: the `handed_to` note is what `lop browser tabs`/`cleanup` now say."""
    eligible, reason = cleanup_disposition(
        {"retention": "pending human sign-in", "handed_to": "abc123def456"}
    )
    assert eligible is False
    assert reason.startswith("retained: pending human sign-in")
    assert "handed to parent session abc123def456" in reason
    plain = cleanup_disposition({"retention": "pending human sign-in"})[1]
    assert plain.endswith("the owning session must release it")
