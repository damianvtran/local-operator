"""The stored "Screening model" choice, and the chains that honour it.

WHY THIS FILE EXISTS. ``resolve_screening_chain(..., preferred=...)`` and the
chain-level consequences of a chosen pair are pinned in
``test_screening_model_allowlist.py`` — but nothing in production passed
``preferred``, so the choice the operator was promised was unreachable. These
tests pin the SEAM instead: what the config document means (including every
value that must read as "no choice"), and that the readers which build a chain
for real use hand it the stored pair.

It is deliberately NOT a second copy of the chain-order matrix: those tests own
the order, this file owns the wiring. The one end-to-end path (a stored choice
with no credential still screens on the receiver's own default) lives in the
allowlist file beside the rig that can drive the shipped cycle.
"""

from __future__ import annotations

from typing import Any, NoReturn

import pytest

from local_operator.team_bridge import settings as tb
from local_operator.team_bridge.model_boundary import (
    resolve_screening_chain,
    resolve_screening_selection,
)

OPUS = ("anthropic", "claude-opus-5-5")
OPUS_R = ("radient", "anthropic/claude-opus-5.5")


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yml").write_text("values:\n" + body)


def _with_choice(tmp_path, hosting: str, model: str, choice: object) -> None:
    """A config document whose default is ``hosting/model`` plus a stored choice."""

    _write_config(
        tmp_path,
        f"  hosting: {hosting}\n"
        f"  model_name: {model}\n"
        "  team_bridge:\n"
        "    screening:\n"
        f"      model: {choice if not isinstance(choice, str) else repr(choice)}\n",
    )


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> set[str]:
    """The providers that hold a credential, at the seam production reads."""

    held: set[str] = set()
    monkeypatch.setattr(
        "local_operator.providers.model_access.credentialed_chat_providers_here",
        lambda **_kwargs: set(held),
    )
    return held


@pytest.fixture(autouse=True)
def _no_provider_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Nothing here may reach a provider: this file is a READ-meaning file.

    ``resolve_screening_chain`` builds a SELECTION, never a client, so the
    provider entry points are refused by name: a test that started calling one
    would be proving something else.
    """

    def refuse(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise AssertionError("a reading test must never build a provider client")

    for name in ("create_stream_fn",):
        monkeypatch.setattr(f"local_operator.model.configure.{name}", refuse, raising=False)


def test_an_absent_key_is_no_choice(tmp_path) -> None:
    _write_config(tmp_path, "  hosting: openai\n  model_name: gpt-6-astra\n")
    assert tb.read_screening_model(tmp_path) is None
    assert tb.TeamBridgeSettings.from_values({}).screening_model is None
    # ``""`` is the registry's "no preference", not a parse failure.
    assert (
        tb.TeamBridgeSettings.from_values(
            {"team_bridge": {"screening": {"model": ""}}}
        ).screening_model
        is None
    )


def test_a_listed_pair_splits_on_the_first_separator(tmp_path) -> None:
    """A Radient id carries its OWN slash, so the split cannot come from the right."""

    _with_choice(tmp_path, "openai", "gpt-6-astra", "radient/anthropic/claude-opus-5.5")
    assert tb.read_screening_model(tmp_path) == OPUS_R
    _with_choice(tmp_path, "openai", "gpt-6-astra", "anthropic/claude-opus-5-5")
    assert tb.read_screening_model(tmp_path) == OPUS


@pytest.mark.parametrize(
    "value",
    [
        # No separator: a bare model id is not a pair, and guessing the provider
        # would let a typo change which model reads someone's mail.
        "claude-opus-5-5",
        "/claude-opus-5-5",
        "anthropic/",
        "/",
        # Not a string at all: a hand-edited list, a number, an explicit null.
    ],
)
def test_a_value_that_is_not_a_pair_reads_as_no_choice(tmp_path, value) -> None:
    _with_choice(tmp_path, "openai", "gpt-6-astra", value)
    assert tb.read_screening_model(tmp_path) is None


@pytest.mark.parametrize("value", [None, 42, ["anthropic", "claude-opus-5-5"], {"a": 1}])
def test_a_non_string_reads_as_no_choice(tmp_path, value) -> None:
    _with_choice(tmp_path, "openai", "gpt-6-astra", value)
    assert tb.read_screening_model(tmp_path) is None


def test_the_settings_object_carries_the_choice(tmp_path) -> None:
    _with_choice(tmp_path, "openai", "gpt-6-astra", "anthropic/claude-opus-5-5")
    assert (
        tb.TeamBridgeSettings.from_values(
            {"team_bridge": {"screening": {"model": "anthropic/claude-opus-5-5"}}}
        ).screening_model
        == OPUS
    )


def test_an_off_table_choice_leaves_todays_chain_identical(tmp_path, credentials) -> None:
    """The table is the only authority: an unlisted pair re-orders nothing.

    ``openai`` is credentialed on purpose, so this cannot pass by the provider
    simply being unavailable — the chain must ignore the PAIR, not the provider.
    """

    credentials.update({"openai", "anthropic", "radient"})
    _with_choice(tmp_path, "openai", "gpt-6-astra", "openai/gpt-6-astra")
    chosen = [
        (hop.provider, hop.model_id)
        for hop in resolve_screening_chain(tmp_path, preferred=tb.read_screening_model(tmp_path))
    ]
    untouched = [
        (hop.provider, hop.model_id) for hop in resolve_screening_chain(tmp_path, preferred=None)
    ]
    assert chosen == untouched
    # And the reader really did hand the chain a pair, so this is not vacuous.
    assert tb.read_screening_model(tmp_path) == ("openai", "gpt-6-astra")


def test_a_listed_choice_leads_and_no_hop_is_dropped(tmp_path, credentials) -> None:
    credentials.update({"openai", "anthropic", "radient"})
    _with_choice(tmp_path, "openai", "gpt-6-astra", "anthropic/claude-opus-5-5")
    chosen = [
        (hop.provider, hop.model_id)
        for hop in resolve_screening_chain(tmp_path, preferred=tb.read_screening_model(tmp_path))
    ]
    untouched = [
        (hop.provider, hop.model_id) for hop in resolve_screening_chain(tmp_path, preferred=None)
    ]
    assert chosen[0] == OPUS
    # The cascade is RE-ORDERED, never narrowed: the same hops, one moved.
    assert sorted(chosen) == sorted(untouched)
    assert ("openai", "gpt-6-astra") in chosen


def test_a_choice_with_no_credential_falls_through_to_the_default(tmp_path, credentials) -> None:
    """The operator directive: a preference must never block a receiver.

    Nothing Anthropic is credentialed here, so the chosen pair cannot be tried;
    the receiver's own default still screens. An empty chain would be a HOLD,
    and a hold is exactly what the directive forbids in this situation.
    """

    credentials.update({"openai"})
    _with_choice(tmp_path, "openai", "gpt-6-astra", "anthropic/claude-opus-5-5")
    chain = resolve_screening_chain(tmp_path, preferred=tb.read_screening_model(tmp_path))
    assert [(hop.provider, hop.model_id) for hop in chain] == [("openai", "gpt-6-astra")]


def test_the_image_gate_selection_read_honours_the_stored_choice(tmp_path, credentials) -> None:
    """``resolve_screening_selection`` is a chain built for real use, so it obeys.

    It feeds the image gate and the canonical-provider read; a choice honoured
    only in the cycle would mean the same machine screening different models
    depending on which surface asked.
    """

    credentials.update({"openai", "anthropic", "radient"})
    _with_choice(tmp_path, "openai", "gpt-6-astra", "anthropic/claude-opus-5-5")
    selection = resolve_screening_selection(tmp_path)
    assert selection is not None
    assert (selection.provider, selection.model_id) == OPUS
    assert selection.preferred is True


def test_writing_a_choice_round_trips_and_empty_clears_it(tmp_path) -> None:
    _write_config(tmp_path, "  hosting: openai\n  model_name: gpt-6-astra\n")
    tb.write_screening_model_choice(tmp_path, "anthropic/claude-opus-5-5")
    assert tb.read_screening_model(tmp_path) == OPUS
    tb.write_screening_model_choice(tmp_path, "")
    assert tb.read_screening_model(tmp_path) is None
    # The write must not lose the document around it.
    assert "gpt-6-astra" in (tmp_path / "config.yml").read_text()
