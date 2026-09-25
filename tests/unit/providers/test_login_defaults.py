"""Login-sets-default-hosting behaviour and default-model resolution.

Covers item 2 (login adopts hosting/model when config is empty) and item 3
(per-provider default model fallback).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.config import ConfigManager
from local_operator.model.defaults import DEFAULT_MODEL_NAMES, default_model_for


def test_default_model_for_known_provider() -> None:
    assert default_model_for("deepseek") == "deepseek-flash"
    assert default_model_for("zai") == "glm-5.3"
    # noop aliases to test, which has no default.
    assert default_model_for("noop") is None


def test_default_model_for_unknown_provider() -> None:
    assert default_model_for("some-custom-host") is None


def test_default_model_names_map_reexported_from_configure() -> None:
    # configure.DEFAULT_MODEL_NAMES must be the SAME object as defaults' — one
    # map, imported cheaply on the startup path (item 3).
    from local_operator.model import configure

    assert configure.DEFAULT_MODEL_NAMES is DEFAULT_MODEL_NAMES


def test_apply_login_defaults_sets_hosting_and_model_when_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    auth_cli._apply_login_defaults("deepseek")

    manager = ConfigManager(tmp_path)
    assert manager.get_config_value("hosting") == "deepseek"
    assert manager.get_config_value("model_name") == "deepseek-flash"


def test_apply_login_defaults_leaves_existing_hosting_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "openai")
    manager.set_config_value("model_name", "gpt-4o")

    # Logging into a second provider must not repoint an existing default.
    auth_cli._apply_login_defaults("deepseek")

    reloaded = ConfigManager(tmp_path)
    assert reloaded.get_config_value("hosting") == "openai"
    assert reloaded.get_config_value("model_name") == "gpt-4o"


@pytest.mark.parametrize(
    ("provider", "expected_hosting", "expected_model"),
    [
        # Login FLAVOURS: authentication routes, not hosting ids. Each has no
        # default model of its own, which is what made the first version of the
        # repair leave the dead model in place beside the new hosting.
        ("xai-oauth", "xai", "grok-4.7"),
        ("openai-device", "openai", "gpt-6-astra"),
        ("zai-oauth", "zai", "glm-5.3"),
        # No default model even after alias resolution: the stale model must be
        # CLEARED, not kept. An empty model_name is a recoverable setup state
        # (ModelNotConfiguredError), not a fully-configured boot; a model from
        # a provider that never existed is worse (boots, then fails at stream).
        ("ollama", "ollama", ""),
        # Ordinary provider with a default: the baseline case.
        ("deepseek", "deepseek", "deepseek-flash"),
    ],
)
def test_repair_never_leaves_a_model_from_the_replaced_provider(
    provider: str, expected_hosting: str, expected_model: str
) -> None:
    """A repair must not produce a config that boots and then fails at stream time.

    The regression this pins: writing the raw login-flavour id as hosting left
    `model_name` untouched when that flavour had no default model, yielding e.g.
    `hosting='xai-oauth' model_name='claude-sonnet-4-5'`. `configure_model`
    ACCEPTS that pair, so boot succeeded and the failure moved to stream time as
    a provider-side unknown-model error -- trading a boot failure the app
    explains for a runtime failure it cannot.
    """
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults(provider, "anthropicxyq", "claude-sonnet-4-5")

    assert plan.repairing is True
    assert plan.hosting == expected_hosting
    # Never None while repairing: the stale model is always overwritten, and ""
    # (clear it) is a deliberate value rather than "leave it alone".
    assert plan.model_name == expected_model
    assert plan.model_name != "claude-sonnet-4-5"


def test_plan_is_the_single_source_of_truth_for_both_login_front_ends() -> None:
    """Both `/login` and `local-operator login` must plan identically.

    The two copies of this rule had drifted on every axis that mattered (which
    hosting id, which brokenness test, which model), which is the class of bug
    the shared planner exists to make impossible. Asserting the front ends call
    the planner rather than re-deriving the policy is what keeps them together.
    """
    import inspect

    from local_operator.providers import auth_cli
    from local_operator.tui import app as tui_app

    for source in (
        inspect.getsource(auth_cli._apply_login_defaults),
        inspect.getsource(tui_app.OperatorApp._apply_login_defaults),
    ):
        assert "plan_login_defaults" in source
        # The policy must not be re-implemented beside the call.
        assert "default_model_for" not in source
        assert "credential_provider_id" not in source


def test_plan_refuses_to_write_an_unknown_provider_id() -> None:
    """A bogus provider_id must not become a hosting the registry does not own.

    `credential_provider_id` passes unknown ids through, so without a registry
    check this planner -- whose purpose is "never write a hosting the engine
    cannot boot" -- would write exactly that. Both current callers reach here
    only after a successful login, so this is not reachable today; the no-op
    is the honest answer when there is nothing legitimate to write (M1).
    """
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults("not-a-provider", "anthropicxyq", "stale")
    assert plan.hosting is None
    assert plan.model_name is None
    assert plan.receipt is None


def test_plan_leaves_a_usable_hosting_alone() -> None:
    """Logging into a second provider must not repoint an existing default."""
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults("deepseek", "openai", "gpt-4o")
    assert plan.hosting is None
    assert plan.model_name is None
    assert plan.receipt is None


def test_plan_treats_a_legacy_alias_as_usable() -> None:
    """`noop` is not a registry id -- it maps to `test`, and must not be
    mistaken for a corrupted value and silently replaced."""
    from local_operator.providers.login_defaults import (
        is_unusable_hosting,
        plan_login_defaults,
    )

    assert is_unusable_hosting("noop") is False
    assert is_unusable_hosting("anthropicxyq") is True
    # Empty hosting is the separate nothing-configured case, not "unusable".
    assert is_unusable_hosting("") is False
    assert plan_login_defaults("deepseek", "noop", "m").hosting is None


def test_apply_login_defaults_repairs_an_unknown_hosting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`login` REPLACES a hosting the registry does not own.

    The unknown-hosting error recommends this exact command, so the "already
    set, leave it alone" rule had to gain an exception: without it the login
    stored a credential, wrote nothing, and the next run failed to boot on the
    same corrupted value — the remedy looping back to the problem it fixes.
    The stale model goes with it: it belonged to the provider being replaced.
    """
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "anthropicxyq")
    manager.set_config_value("model_name", "claude-sonnet-4-5")

    auth_cli._apply_login_defaults("deepseek")

    reloaded = ConfigManager(tmp_path)
    assert reloaded.get_config_value("hosting") == "deepseek"
    assert reloaded.get_config_value("model_name") == "deepseek-flash"


def test_apply_login_defaults_leaves_a_legacy_alias_hosting_untouched(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A legacy ALIAS is a valid hosting and must not be treated as corrupt.

    `noop` is not a registry id — it maps to `test`. The repair check asks the
    registry (which resolves aliases) rather than testing membership against
    ids, so a working alias config is not silently repointed by a later login.
    """
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "noop")

    auth_cli._apply_login_defaults("deepseek")

    assert ConfigManager(tmp_path).get_config_value("hosting") == "noop"


def test_resolve_hosting_model_falls_back_to_default_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Hosting set, model empty: resolves the provider default rather than
    raising 'Model name is not configured' (item 3)."""
    import argparse

    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.session_factory import resolve_hosting_model

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "deepseek")
    manager.set_config_value("model_name", "")

    args = argparse.Namespace(hosting=None, model=None)
    hosting, model = resolve_hosting_model(None, args, manager)
    assert hosting == "deepseek"
    assert model == "deepseek-flash"


def test_resolve_hosting_model_no_hosting_raises_hosting_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No hosting at all raises the dedicated HostingNotConfiguredError so the
    setup-state gate can classify it (item 1)."""
    import argparse

    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.session_factory import (
        HostingNotConfiguredError,
        resolve_hosting_model,
    )

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    args = argparse.Namespace(hosting=None, model=None)
    with pytest.raises(HostingNotConfiguredError):
        resolve_hosting_model(None, args, manager)


# ---------------------------------------------------------------------------
# A decision-only provider is never a chat hosting (the fifth door)
# ---------------------------------------------------------------------------


#: The one-row budget for a RECEIPT SENTENCE, and it is a measured number with a
#: provenance rather than a round figure: at 100 columns the designer's rendered frame
#: (`save_capture`, production stylesheet) gave the notice block 75 cells and the
#: sentence inside it 71 — the glyph, the gutter and the indent take the rest. QA round 4
#: (Q4) is why this replaced `<= 90`: 90 was never measured, and the sentences these
#: tests assert are 54-63 cells, so the old bound could not fail for anything a user
#: would see wrap.
#:
#: WHAT IT DOES AND DOES NOT COVER. The branches whose text is fully determined here —
#: no-hosting (54), first-run write (60), a decision-only login against a typical
#: configured pair (63) — are asserted against it. The branches whose length is the IDS'
#: are not, because no copy change can bound them: measured, `openrouter/anthropic/
#: claude-opus-5-20260101` puts the sentence at 84 and the repairing sentence
#: (`Replaced unusable hosting 'anthropicxyq' with 'deepseek', model to 'deepseek-flash'`)
#: at 83, both of which wrap. That is a real, known wrap in a branch no alternative copy
#: would fix, and it is recorded here rather than hidden by an assertion that would pass
#: for the wrong reason.
RECEIPT_ROW_CELLS = 71


def test_a_decision_only_provider_is_never_adopted_as_hosting() -> None:
    """``login typesafe`` stores a credential; it must not move the routing.

    TypeSafe's Jev rejects ``chat/completions`` on every host we reach it through
    (``registry.is_decision_only``), so a plan that adopted it as hosting — as
    cases 2 and 3 below would — writes a config whose very next session cannot
    answer a turn. That is the trap the catalogue, the ``/model`` ranking, the
    session-model resolver and the failover chain each refuse; login is the fifth
    door to it, and this is the only one where the credential IS the point: the
    layer needs the key, and nothing about that needs a chat hosting.
    """
    from local_operator.providers.login_defaults import plan_login_defaults

    # The COMMON state — a working hosting already configured — must still say
    # something: this is the state the note used to be unreachable in, and the login
    # then read as having silently done nothing (review round 1, Q4).
    configured = plan_login_defaults("typesafe", "deepseek", "deepseek-chat")
    assert configured.hosting is None
    assert configured.model_name is None
    assert configured.receipt is not None
    # The copy answers the only question the user has ("did this change my model?")
    # and names what is actually serving — the fact they can act on. Design round 1's
    # D5 took the harness's vocabulary off it: no "Jev", no "decision-model calls",
    # no "resource recommendations", and it ends with its own full stop (D6).
    assert configured.receipt == "Nothing changed — chats keep running on deepseek/deepseek-chat."
    # THIS BRANCH, named rather than assumed: the configured-hosting sentence is the
    # only one that can name what serves, and the length below is a claim about IT —
    # a bare ``<= RECEIPT_ROW_CELLS`` would pass on any branch's string, including one
    # this test never meant to describe. The other branch's tell (it points at
    # ``/model`` because there is nothing serving to name) must not appear here.
    assert "deepseek/deepseek-chat" in configured.receipt
    assert "/model" not in configured.receipt
    assert len(configured.receipt) <= RECEIPT_ROW_CELLS, len(configured.receipt)
    assert len(configured.receipt) == 63, "the measured length this test's budget is about"

    # Case 2: hosting empty (a fresh config, or `hosting: ""`).
    for empty in ("", None):
        plan = plan_login_defaults("typesafe", empty, None)
        assert plan.hosting is None, empty
        assert plan.model_name is None
        # Not a repair: nothing was written, and `repairing` is what the callers
        # use for wording.
        assert plan.repairing is False
        # The receipt is the ONE caller-facing channel, and it has to name both
        # the reason and the provider a user recognises.
        assert plan.receipt is not None
        # No routing to name, so the sentence names the way out of that state.
        assert plan.receipt == "Nothing changed — pick a chat model with /model first."
        assert len(plan.receipt) == 54, len(plan.receipt)
        assert len(plan.receipt) <= RECEIPT_ROW_CELLS
        # The wire vocabulary the design round took off the default copy.
        assert "Jev" not in plan.receipt
        assert "decision-model" not in plan.receipt

    # Case 3: hosting set but unusable. A repair must not replace a broken
    # hosting with one that cannot serve a session at all, and it must not write
    # the dead model either.
    repaired = plan_login_defaults("typesafe", "anthropicxyq", "claude-sonnet-4-5")
    assert repaired.hosting is None
    assert repaired.model_name is None
    assert repaired.receipt is not None


def test_a_normal_provider_still_adopts_on_an_empty_hosting() -> None:
    """The regression guard: the exemption is about ONE provider, not the rule.

    Identical call, same empty hosting, an ordinary provider — the adoption that
    makes a fresh ``login`` usable has to survive.
    """
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults("xai", None, None)

    assert plan.hosting == "xai"
    assert plan.model_name == "grok-4.7"
    assert plan.receipt is not None


def test_apply_login_defaults_writes_nothing_and_says_so_for_typesafe(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI half: the no-write path must still print the note.

    ``_apply_login_defaults`` returned before printing whenever the plan wrote
    nothing, which would have made this receipt unreachable — the user would see
    ``Stored API key for 'typesafe'.`` and then nothing about their routing.
    """
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    # The failure the note exists for happened in the COMMON state: a hosting already
    # configured and perfectly usable, where the planner returned only "nothing to
    # write" and the user saw `Stored API key for 'typesafe'.` and silence.
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "deepseek")
    manager.set_config_value("model_name", "deepseek-chat")

    auth_cli._apply_login_defaults("typesafe")
    printed = capsys.readouterr().out

    # THE LITERAL sentence, capital and full stop included — deliberately not the
    # planner's own output run a second time, which would move with the copy and so
    # could never fail (review round 4, NIT). The two front ends agreeing is a
    # consequence of the print sites, not of this assertion: ``auth_cli`` prints
    # ``plan.receipt`` and the TUI's ``_apply_login_defaults`` returns ``plan.receipt``
    # for ``notice(..., "note")``, so pinning the bytes HERE pins what both render.
    assert "Nothing changed — chats keep running on deepseek/deepseek-chat." in printed
    # ...and the routing is untouched: the configured pair, unchanged.
    assert manager.get_config_value("hosting") == "deepseek"
    assert manager.get_config_value("model_name") == "deepseek-chat"

    # The same on a config with no hosting at all, and still nothing written.
    fresh = tmp_path / "fresh"
    monkeypatch.setenv(CONFIG_DIR_ENV, str(fresh))
    auth_cli._apply_login_defaults("typesafe")
    fresh_out = capsys.readouterr().out
    assert "Nothing changed — pick a chat model with /model first." in fresh_out
    # One full stop, from the receipt itself — not the CLI's own punctuation (D6).
    assert fresh_out.count(".") == 1, fresh_out
    # The lowercase-start case, which is where the two front ends used to diverge: the
    # CLI upper-cased it and the TUI did not (D9). Both now print what the planner
    # wrote, so a receipt that starts lowercase stays lowercase everywhere.
    from local_operator.providers.login_defaults import plan_login_defaults

    deepseek_receipt = plan_login_defaults("deepseek", "", None).receipt
    assert deepseek_receipt is not None and deepseek_receipt.startswith("Set default")
    reloaded = ConfigManager(fresh)
    assert not reloaded.get_config_value("hosting")
    assert not reloaded.get_config_value("model_name")

    # A normal provider through the SAME command still adopts, and prints its
    # write receipt — the two paths share one print site.
    auth_cli._apply_login_defaults("deepseek")
    write_receipt = capsys.readouterr().out.strip()
    # The WRITE branch too: literal, and inside the same row budget. This is the
    # receipt that would otherwise be the silent one — the round-4 NIT's point is that
    # every branch of this copy is asserted against the bytes a user sees.
    assert write_receipt == "Set default hosting to 'deepseek', model to 'deepseek-flash'."
    assert len(write_receipt) == 61 and len(write_receipt) <= RECEIPT_ROW_CELLS, len(write_receipt)


# ---------------------------------------------------------------------------
# The suggested-model table (`model.defaults.SUGGESTED_MODELS`)
# ---------------------------------------------------------------------------


#: The operator's stated frontier picks (2026-09-24), each verified against the
#: provider's own docs or live listing -- see the table's per-row comments.
_OPERATOR_TARGETS = {
    "anthropic": "claude-opus-5-5",
    "openai": "gpt-6-astra",
    # DeepSeek's direct wire id for V4.1 Flash (its /models lists exactly this).
    "deepseek": "deepseek-flash",
    "zai": "glm-5.3",
    "alibaba": "qwen3.8-max",
    "xai": "grok-4.7",
    "google": "gemini-3.8-flash",
}


@pytest.mark.parametrize(("hosting", "model"), sorted(_OPERATOR_TARGETS.items()))
def test_the_suggestions_are_the_frontier_picks(hosting: str, model: str) -> None:
    from local_operator.model.defaults import suggested_model_for

    suggestion = suggested_model_for(hosting)
    assert suggestion is not None
    assert suggestion.id == model
    # The old table's ids must not creep back: each was two generations stale.
    assert suggestion.id not in {
        "gpt-4o",
        "claude-3-5-sonnet-latest",
        "grok-3",
        "gemini-2.0-flash-001",
        "deepseek-chat",
    }


def test_every_suggestion_names_a_model_the_registry_describes() -> None:
    """A suggestion the registry answers "unknown" for would boot with a -1 window.

    Aggregators are exempt by construction: their ids are the router's own and
    carry no registry row (the listing is authoritative there).
    """
    from local_operator.model.defaults import OAUTH_SUGGESTED_MODELS, SUGGESTED_MODELS
    from local_operator.model.registry import get_model_info

    rows = list(SUGGESTED_MODELS.items()) + list(OAUTH_SUGGESTED_MODELS.items())
    for hosting, suggestion in rows:
        if hosting in ("openrouter",):
            continue
        info = get_model_info(hosting, suggestion.id)
        assert info.id == suggestion.id, (hosting, suggestion.id, info.id)
        assert (info.context_window or 0) > 0, (hosting, suggestion.id)


def test_every_suggestion_is_named_as_the_model_picker_names_it() -> None:
    """The desktop says "Default model set to <name>" and the model picker then
    shows the registry row's name for the same selection. Two names for one
    model read as two models (review round 1, #7), so wherever a shipped row
    exists the suggestion carries that row's name verbatim. Rows without one
    (OpenRouter's namespaced id, Radient's router) name themselves.
    """
    from local_operator.model.defaults import OAUTH_SUGGESTED_MODELS, SUGGESTED_MODELS
    from local_operator.model.registry import static_models

    rows = list(SUGGESTED_MODELS.items()) + list(OAUTH_SUGGESTED_MODELS.items())
    mismatched = [
        (hosting, suggestion.id, suggestion.name, row.name)
        for hosting, suggestion in rows
        if (row := static_models(hosting).get(suggestion.id)) is not None
        and row.name != suggestion.name
    ]
    assert mismatched == []


def test_every_hosted_cloud_provider_has_a_suggestion() -> None:
    """The desktop shows "Suggested: ..." on every cloud provider card; a gap is a
    blank card and a first sign-in that leaves the model empty."""
    from local_operator.model.defaults import suggested_model_for
    from local_operator.providers.registry import (
        PROVIDER_REGISTRY,
        credential_provider_id,
    )

    for provider in PROVIDER_REGISTRY:
        storage = credential_provider_id(provider.id)
        if provider.allows_missing_api_key or provider.decision_only or provider.wire == "mock":
            continue
        assert suggested_model_for(storage) is not None, storage


def test_default_model_names_is_derived_from_the_suggestions() -> None:
    from local_operator.model.defaults import SUGGESTED_MODELS

    assert DEFAULT_MODEL_NAMES == {h: s.id for h, s in SUGGESTED_MODELS.items()}


def test_the_route_decides_the_kimi_spelling() -> None:
    """Kimi's OAuth grant reaches the coding-plan host (``k3``); a key reaches
    api.moonshot.cn (``kimi-k3``). Each must get the id its host serves."""
    from local_operator.providers.login_defaults import plan_login_defaults

    assert plan_login_defaults("kimi", "", None).model_name == "k3"
    assert plan_login_defaults("kimi", "", None, oauth=False).model_name == "kimi-k3"
    # Every other OAuth flavour serves the API-key id.
    assert plan_login_defaults("xai-oauth", "", None).model_name == "grok-4.7"
    assert plan_login_defaults("zai-oauth", "", None).model_name == "glm-5.3"
    assert plan_login_defaults("anthropic", "", None).model_name == "claude-opus-5-5"


def test_the_cli_plans_with_the_flavour_that_ran_not_the_storage_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``radient-key`` stores under ``radient``, whose own login is OAuth, so a
    storage-id derivation planned a pasted key as an OAuth grant (review round
    1, #5). Kimi is where the flavour changes the model id, so it is the probe:
    the CLI must hand the planner what the login actually returned.
    """
    from local_operator.paths import CONFIG_DIR_ENV
    from local_operator.providers import auth_cli

    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    auth_cli._apply_login_defaults("kimi", oauth=False)
    assert ConfigManager(tmp_path).get_config_value("model_name") == "kimi-k3"

    other = tmp_path / "oauth"
    other.mkdir()
    monkeypatch.setenv(CONFIG_DIR_ENV, str(other))
    auth_cli._apply_login_defaults("kimi", oauth=True)
    assert ConfigManager(other).get_config_value("model_name") == "k3"


@pytest.mark.parametrize(
    ("result", "oauth"),
    [("sk-pasted-key", False), ({"type": "oauth", "access": "a", "refresh": "r"}, True)],
)
def test_run_login_passes_the_result_shape_as_the_flavour(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, result: object, oauth: bool
) -> None:
    """``run_login`` decides the flavour from what the login RETURNED (a key
    string or an OAuth dict), not from the storage id's own definition."""
    import dataclasses

    from local_operator.providers import auth_cli, registry
    from local_operator.providers.auth_store import AuthStore

    async def login(_callbacks, **_kwargs):
        return result

    definition = registry.get_provider_definition("radient")
    assert definition is not None
    monkeypatch.setitem(registry._BY_ID, "radient", dataclasses.replace(definition, login=login))
    seen: list[tuple[str, bool | None]] = []
    monkeypatch.setattr(
        auth_cli, "_apply_login_defaults", lambda pid, *, oauth=None: seen.append((pid, oauth))
    )
    store = AuthStore(tmp_path / "auth.db", config_dir=tmp_path)
    try:
        assert auth_cli.run_login("radient", tmp_path, store) == 0
    finally:
        store.close()
    assert seen == [("radient", oauth)]


def test_an_empty_model_beside_the_same_provider_is_filled() -> None:
    """Hosting already this provider, model empty: fill the suggestion, keep hosting."""
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults("anthropic", "anthropic", "")
    assert plan.hosting is None
    assert plan.model_name == "claude-opus-5-5"
    assert plan.model_label == "Claude Opus 5.5"
    assert plan.receipt == "Set default model to 'claude-opus-5-5'."
    # A flavour counts as the same provider.
    assert plan_login_defaults("xai-oauth", "xai", None).model_name == "grok-4.7"


def test_an_empty_model_beside_another_provider_is_left_alone() -> None:
    """Writing Y's model beside X's hosting would boot and fail at stream time."""
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults("xai", "anthropic", "")
    assert (plan.hosting, plan.model_name, plan.receipt) == (None, None, None)


def test_a_chosen_model_is_never_overridden() -> None:
    from local_operator.providers.login_defaults import plan_login_defaults

    plan = plan_login_defaults("anthropic", "anthropic", "claude-sonnet-5")
    assert (plan.hosting, plan.model_name, plan.receipt) == (None, None, None)


def test_apply_writes_a_model_only_plan(tmp_path: Path) -> None:
    """The shared writer: a plan that sets only the model must still be written
    (each front end used to gate the model write on a hosting write)."""
    from local_operator.providers.login_defaults import (
        apply_login_defaults,
        plan_login_defaults,
    )

    manager = ConfigManager(tmp_path)
    manager.set_config_value("hosting", "anthropic")
    plan = plan_login_defaults("anthropic", "anthropic", "")
    assert apply_login_defaults(manager, plan) is True
    reloaded = ConfigManager(tmp_path)
    assert reloaded.get_config_value("hosting") == "anthropic"
    assert reloaded.get_config_value("model_name") == "claude-opus-5-5"
