"""The seeding rules, tested where they are decided.

These are unit tests of ``usage_seed`` rather than cold-viewer tests on purpose:
the window decision is a PURE function of the reading and the spec, and the
defect this file was added for (B1, agent review round 1) could not be caught by
the viewer-level guard test — that one configures anthropic, where the resolved
flag is False, so it fails on the numerator assertion long before it reaches a
window claim. A test that cannot fail on the assertion it is supposed to protect
is not evidence, so the evidence lives here, with the viewer-level cases in
``test_remote_cold.py`` covering the wiring.
"""

from __future__ import annotations

import pytest

from local_operator.harness.types import Usage
from local_operator.model.configure import UNKNOWN_CONTEXT_WINDOW
from local_operator.session.frontend_state import FrontendModelSpec
from local_operator.session.model_selection import StoredModelSelection
from local_operator.session.usage_seed import reading_identity, reading_window


def _receipt(**overrides) -> Usage:
    values = {
        "context_tokens": 322_546,
        "provider": "openai",
        "model_id": "gpt-5.6-sol",
    }
    values.update(overrides)
    return Usage(**values)


def test_a_placeholder_window_is_not_a_denominator() -> None:
    """THE B1 CASE: `resolved=True` does not vouch for the window VALUE.

    ``UNKNOWN_CONTEXT_WINDOW`` is a placeholder and ``context_spec_for_access``
    writes it together with ``context_metadata_resolved: True`` whenever the
    selected account resolved to nothing or the catalogue row carried no positive
    window. Trusting the flag alone divided a real 322_546-token receipt by the
    placeholder and printed a measured ``252.0%/128k`` — a wrong reading where the
    cold path previously showed none.

    Anchors: ``configure.py``'s ``UNKNOWN_CONTEXT_WINDOW = 128_000`` and the two
    placeholder returns in ``context_spec_for_access``.

    Without the placeholder guard in ``reading_window`` this test returns 128_000
    and fails.
    """
    spec = FrontendModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        context_window=UNKNOWN_CONTEXT_WINDOW,
        default_context_window=None,
        max_context_window=None,
        context_metadata_resolved=True,
    )
    assert reading_window(_receipt(), fallback=None, spec=spec) is None


def test_a_model_whose_own_default_is_the_placeholder_may_still_use_it() -> None:
    """The other side of B1: a DOCUMENTED 128k default is a real window.

    `context_spec_for_access` prefers `default_context_window` when the session
    budgets against the model's default rather than its maximum, and that value
    can legitimately be 128_000 on a model that serves more. The placeholder case
    always leaves that field `None`, so the two are distinguishable.
    """
    spec = FrontendModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        context_window=UNKNOWN_CONTEXT_WINDOW,
        default_context_window=UNKNOWN_CONTEXT_WINDOW,
        max_context_window=1_000_000,
        context_metadata_resolved=True,
    )
    assert reading_window(_receipt(), fallback=None, spec=spec) == UNKNOWN_CONTEXT_WINDOW


def test_a_resolved_window_is_the_denominator() -> None:
    spec = FrontendModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        context_window=400_000,
        context_metadata_resolved=True,
    )
    assert reading_window(_receipt(), fallback=None, spec=spec) == 400_000


def test_an_unresolved_spec_has_no_window() -> None:
    """A config-derived spec's window is a default, so it is not a denominator."""
    spec = FrontendModelSpec(provider="openai", model_id="gpt-5.6-sol", context_window=400_000)
    assert reading_window(_receipt(), fallback=None, spec=spec) is None


def test_a_reading_that_names_another_model_has_no_window() -> None:
    """A count measured on another model is not convertible to this one's window."""
    spec = FrontendModelSpec(
        provider="anthropic",
        model_id="claude-opus-5",
        context_window=1_000_000,
        context_metadata_resolved=True,
    )
    assert reading_window(_receipt(), fallback=None, spec=spec) is None
    assert reading_identity(_receipt(), fallback=None) == ("openai", "gpt-5.6-sol")


def test_an_unstamped_reading_borrows_the_conversations_own_selection() -> None:
    """Rows written before the serving-identity stamp are attributable.

    The saved selection is durable evidence of the model this conversation ran on,
    which is the only thing that can name an unstamped receipt — and the reason
    ``reading_identity`` takes a fallback at all.
    """
    unstamped = _receipt(provider=None, model_id=None)
    saved = StoredModelSelection(provider="openai", model_id="gpt-5.6-sol")
    spec = FrontendModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        context_window=400_000,
        context_metadata_resolved=True,
    )
    assert reading_identity(unstamped, fallback=saved) == ("openai", "gpt-5.6-sol")
    assert reading_window(unstamped, fallback=saved, spec=spec) == 400_000


def test_an_unattributable_reading_has_neither_identity_nor_window() -> None:
    """No stamp and no saved selection means "cannot be attributed", not "mine"."""
    unstamped = _receipt(provider=None, model_id=None)
    spec = FrontendModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        context_window=400_000,
        context_metadata_resolved=True,
    )
    assert reading_identity(unstamped, fallback=None) is None
    assert reading_window(unstamped, fallback=None, spec=spec) is None


@pytest.mark.parametrize("window", [0, -1])
def test_a_nonpositive_window_is_not_a_denominator(window: int) -> None:
    spec = FrontendModelSpec(
        provider="openai",
        model_id="gpt-5.6-sol",
        context_window=window,
        context_metadata_resolved=True,
    )
    assert reading_window(_receipt(), fallback=None, spec=spec) is None
