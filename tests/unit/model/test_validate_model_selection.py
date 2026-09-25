"""``validate_model_selection``: one validator for the desktop draft and the peer switch.

Two things are pinned here. The validator refuses each way a pair can fail to be
servable, before anything is built or mutated (design D3). And the desktop draft
route, which now delegates to it, answers with 422 bodies BYTE-IDENTICAL to the
ones it answered before the extraction: those bodies are a wire contract the
desktop app reads, so the expected dicts below are the pre-extraction literals,
not values derived from the new code.
"""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from local_operator.model.configure import (
    ModelSelectionRefused,
    validate_model_selection,
)
from local_operator.server.routes.desktop_sessions import DraftModel, _draft_model_spec

#: The 422 bodies ``_draft_model_spec`` produced at origin/main 213e6c1b5, copied
#: verbatim from the code that composed them. A reworded refusal fails here.
PRE_EXTRACTION_BODIES = [
    (
        ("nope-provider", "whatever"),
        {"code": "provider_unknown", "message": "'nope-provider' is not a known provider."},
    ),
    (
        ("typesafe", "jev-1.13"),
        {
            "code": "provider_decision_only",
            "message": (
                "'typesafe' serves decision-model calls, not chat completions, "
                "so no session can run on it."
            ),
        },
    ),
    (
        ("anthropic", "claude-opus-9"),
        {"code": "model_unknown", "message": "'claude-opus-9' is not a model anthropic serves."},
    ),
]


@pytest.mark.parametrize("pair,body", PRE_EXTRACTION_BODIES)
def test_the_desktop_draft_422_bodies_are_unchanged(pair, body) -> None:
    provider, model_id = pair
    with pytest.raises(HTTPException) as caught:
        _draft_model_spec(DraftModel(provider=provider, model_id=model_id))
    assert caught.value.status_code == 422
    assert caught.value.detail == body


def test_the_desktop_draft_unbuildable_body_is_unchanged(monkeypatch) -> None:
    """``model_unavailable`` is only reachable when the spec build itself raises."""
    import local_operator.model.configure as configure

    def boom(*_args, **_kwargs):
        raise RuntimeError("metadata resolver down")

    monkeypatch.setattr(configure, "build_model_spec", boom)
    with pytest.raises(HTTPException) as caught:
        _draft_model_spec(DraftModel(provider="deepseek", model_id="deepseek-flash"))
    assert caught.value.detail == {
        "code": "model_unavailable",
        "message": "'deepseek-flash' could not be resolved.",
    }


def test_the_desktop_draft_never_consults_credentials(monkeypatch) -> None:
    """The draft route never refused on credentials; the shared validator must
    not start doing so for it (its 422 set is a contract)."""
    spec = _draft_model_spec(DraftModel(provider="deepseek", model_id="deepseek-flash"))
    assert (spec.provider, spec.model_id) == ("deepseek", "deepseek-flash")


@pytest.mark.parametrize(
    "provider,model_id,code",
    [
        ("nosuchprov", "x", "provider_unknown"),
        ("typesafe", "jev-1.13", "provider_decision_only"),
        # Measured by the architect: ``build_model_spec`` BUILDS this pair, so
        # only the catalogue check stands between a typo and a dead next turn.
        ("deepseek", "not-a-model", "model_unknown"),
    ],
)
def test_each_unservable_pair_is_refused_with_its_code(provider, model_id, code) -> None:
    with pytest.raises(ModelSelectionRefused) as caught:
        validate_model_selection(provider, model_id, usable=lambda _p: True)
    assert caught.value.code == code
    assert str(caught.value) == caught.value.message


def test_a_provider_with_no_usable_credential_is_refused() -> None:
    asked: list[str] = []

    def usable(provider: str) -> bool:
        asked.append(provider)
        return False

    with pytest.raises(ModelSelectionRefused) as caught:
        validate_model_selection("deepseek", "deepseek-flash", usable=usable)
    assert caught.value.code == "provider_unusable"
    assert "deepseek" in caught.value.message
    assert asked == ["deepseek"]


def test_the_credential_probe_runs_only_after_the_pair_checks() -> None:
    """An unknown pair is refused for what it IS, never as a credential miss."""

    def usable(_provider: str) -> bool:
        raise AssertionError("the credential store must not be read for an unknown pair")

    with pytest.raises(ModelSelectionRefused) as caught:
        validate_model_selection("deepseek", "not-a-model", usable=usable)
    assert caught.value.code == "model_unknown"


def test_an_unenumerable_catalogue_accepts_the_pair() -> None:
    """``offered_model_ids`` answers ``None`` for the mock provider: "not looked"
    is not "does not exist", so the pair builds."""
    spec = validate_model_selection("test", "anything-goes", usable=lambda _p: True)
    assert (spec.provider, spec.model_id) == ("test", "anything-goes")


def test_a_servable_pair_returns_the_built_spec() -> None:
    spec = validate_model_selection("deepseek", "deepseek-flash", usable=lambda _p: True)
    assert (spec.provider, spec.model_id) == ("deepseek", "deepseek-flash")
