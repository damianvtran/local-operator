"""The remappable-hotkey registry: identity, validation, resolution.

Every property here is STRUCTURAL — what is registered, what is rejected, what
resolves to what. Nothing in this feature is about how long anything takes, so
there is no timing bound anywhere in this file or its pilot counterparts (see
AGENTS.md, "Timing, flakes, and how to assert that something is fast").

The anti-drift test is the one that earns its keep. The binding id is
simultaneously the config key, the ``Binding`` id and the tip lookup, and it is
PERSISTED USER DATA — a rename does not raise, it silently orphans every user's
override and restores the default under them.
"""

from __future__ import annotations

import pytest

from local_operator import keymap, settings_io


def test_binding_ids_config_keys_and_app_bindings_are_one_vocabulary() -> None:
    """THE anti-drift test: the same id in all three places, or fail loudly.

    The three declarations are deliberate duplication (the app must not import
    the settings registry to bind a key, and the CLI must not import the TUI to
    validate one), so nothing but this test stops them diverging. It imports
    the app, which is why it lives beside the other registry tests rather than
    in the TUI directory: the assertion is about the REGISTRY agreeing with the
    app, not about anything rendering.
    """
    from local_operator.tui.app import OperatorApp

    registry_ids = {action.id for action in keymap.KEY_ACTIONS}
    settings_keys = {setting.key for setting in settings_io.SETTINGS if setting.section == "keymap"}
    # `BINDINGS` is typed as accepting tuples as well as `Binding`s, so the id
    # is read through `getattr` rather than by attribute access.
    binding_ids = {
        str(getattr(binding, "id", "") or "")
        for binding in OperatorApp.BINDINGS
        if str(getattr(binding, "id", "") or "").startswith(keymap.KEYMAP_PREFIX)
    }

    assert registry_ids == settings_keys, "settings registry drifted from KEY_ACTIONS"
    assert registry_ids == binding_ids, "OperatorApp.BINDINGS drifted from KEY_ACTIONS"


def test_every_binding_action_exists_on_the_app() -> None:
    """A binding naming a missing action fails only WHEN THE KEY IS PRESSED.

    Textual resolves the action lazily, so a typo here ships as a hotkey that
    does nothing on the one machine that presses it. Checked structurally
    instead.
    """
    from local_operator.tui.app import OperatorApp

    for action in keymap.KEY_ACTIONS:
        assert hasattr(OperatorApp, f"action_{action.action}"), action.action


def test_keymap_settings_are_flat_dotted() -> None:
    """The ``display.*`` shape, not the ``tui.*`` one.

    Flat is what makes "registered" and "propagated" the same set:
    ``config_watch._changed_registry_keys`` only diffs keys the registry knows,
    so a nested block would admit hand-written sub-keys that propagation could
    never see.
    """
    flat = set(settings_io.flat_dotted_keys())
    for action in keymap.KEY_ACTIONS:
        assert action.id in flat


def test_shipped_defaults_are_bindable_and_unreserved() -> None:
    """A default that its own validator rejects would be unfixable from the page."""
    for action in keymap.KEY_ACTIONS:
        assert keymap.validate_key(action.default) is None, action.id
        assert action.default not in keymap.RESERVED_KEYS


def test_shipped_defaults_do_not_collide_with_each_other() -> None:
    defaults = [action.default for action in keymap.KEY_ACTIONS]
    assert len(defaults) == len(set(defaults))


def test_shipped_defaults_are_not_composer_keys() -> None:
    """A non-priority binding on a composer key never fires while typing.

    Measured: a remap onto ``ctrl+u`` did not fire AND the TextArea deleted the
    line, with no clash reported. Shipping a default in that set would make the
    feature look broken out of the box.
    """
    for action in keymap.KEY_ACTIONS:
        assert action.default not in keymap.COMPOSER_KEYS, action.id


@pytest.mark.parametrize(
    "bad",
    [
        "banana",  # not a key at all
        "ctrl+",  # a dangling modifier
        "ctrl-n",  # the wrong separator
        "kontrol+n",  # a misspelled modifier
        "",  # empty
    ],
)
def test_validate_rejects_malformed_keys(bad: str) -> None:
    """Textual accepts every one of these VERBATIM and silently makes the
    action unreachable — the Claude Code pre-2.1.246 silent-disable bug, live
    in Textual 8.2.8. This validator is the only thing standing in front of it,
    which is why it is at the write boundary rather than in the capture UI."""
    assert keymap.validate_key(bad) is not None


@pytest.mark.parametrize("key", ["escape", "ctrl+c", "ctrl+d", "ctrl+q", "ctrl+m", "enter", "tab"])
def test_validate_refuses_reserved_keys_with_a_reason(key: str) -> None:
    """A refusal names the key's job. `ctrl+m` is refused because it is the
    SAME BYTE as enter — binding it would silently unbind the composer's
    submit, with nothing on screen relating the two."""
    reason = keymap.validate_key(key)
    assert reason is not None and reason.strip()


def test_validate_refuses_a_bare_printable_character() -> None:
    reason = keymap.validate_key("n")
    assert reason is not None and "composer" in reason


def test_validate_accepts_ordinary_chords_and_alternates() -> None:
    """A comma means ALTERNATES in Textual (both keys fire), so the schema
    tolerates a hand-written list even though capture stores exactly one."""
    assert keymap.validate_key("ctrl+g") is None
    assert keymap.validate_key("f5") is None
    assert keymap.validate_key("ctrl+n,f5") is None


def test_normalize_lowercases_so_the_page_cannot_display_an_unpressable_key() -> None:
    """``ctrl+N`` is accepted by Textual verbatim and binds a key nobody can
    press, while the page goes on displaying it."""
    assert keymap.normalize_key("ctrl+N") == "ctrl+n"
    assert keymap.normalize_key("  CTRL+G  ") == "ctrl+g"


def test_resolved_keymap_omits_defaults_and_absent_keys() -> None:
    """Omission is the CORRECT encoding of "unset", not a shortcut:
    ``set_keymap`` applies against the pristine class BINDINGS rather than
    cumulatively, so an empty mapping restores every shipped key."""
    action = keymap.KEY_ACTIONS[0]
    assert keymap.resolved_keymap({}) == ({}, [])
    assert keymap.resolved_keymap({action.id: action.default}) == ({}, [])


def test_resolved_keymap_drops_invalid_values_and_names_them() -> None:
    """Garbage on disk must not leave the action unreachable — it keeps the
    shipped default and the caller prints one notice."""
    action = keymap.KEY_ACTIONS[0]
    resolved, rejected = keymap.resolved_keymap({action.id: "banana"})
    assert resolved == {}
    assert rejected == [action.id]


def test_resolved_keymap_carries_a_valid_override() -> None:
    action = keymap.KEY_ACTIONS[0]
    resolved, rejected = keymap.resolved_keymap({action.id: "ctrl+g"})
    assert resolved == {action.id: "ctrl+g"}
    assert rejected == []


def test_effective_key_reads_the_persisted_value_not_the_binding() -> None:
    """``Binding.key`` still reports the OLD key after a remap (measured), so a
    tip built on it would advertise a chord that no longer works."""
    action = keymap.KEY_ACTIONS[0]
    assert keymap.effective_key(action, {}) == action.default
    assert keymap.effective_key(action, {action.id: "ctrl+g"}) == "ctrl+g"


def test_conflict_note_warns_about_composer_keys_only() -> None:
    """The class Textual's own clash detection CANNOT see: it only covers
    bindings in the same BindingsMap, so a remap onto a TextArea key reports
    no clash and then silently loses to the composer."""
    assert "composer" in keymap.conflict_note("keymap.new_session", "ctrl+u")
    assert keymap.conflict_note("keymap.new_session", "f5") == ""


def test_settings_registry_validates_and_normalizes_through_the_facade() -> None:
    """The page is not the only writer. ``lop config edit`` and a hand edit
    reach the same value, so the guard has to be in ``settings_io``."""
    setting = settings_io.BY_KEY["keymap.new_session"]
    assert settings_io.validate(setting, "banana") is not None
    assert settings_io.validate(setting, "ctrl+g") is None
    assert settings_io.coerce(setting, "CTRL+G") == "ctrl+g"


def test_write_setting_normalizes_even_when_coerce_is_bypassed(tmp_path) -> None:
    """The server's ``PATCH /v1/settings/{key}`` hands a raw JSON value to
    ``write_setting`` without passing through ``coerce``. Without the
    normalization there, ``ctrl+N`` would validate and be stored verbatim."""
    from local_operator.config import ConfigManager

    manager = ConfigManager(tmp_path)
    setting = settings_io.BY_KEY["keymap.new_session"]
    settings_io.write_setting(manager, setting, "ctrl+G")
    assert settings_io.read_setting(manager, setting) == "ctrl+g"


def test_reset_deletes_the_override_rather_than_writing_the_default(tmp_path) -> None:
    """ "Store only overrides" falls out of the existing facade for free — `r`
    on the page already routes here."""
    from local_operator.config import ConfigManager

    manager = ConfigManager(tmp_path)
    setting = settings_io.BY_KEY["keymap.new_session"]
    settings_io.write_setting(manager, setting, "ctrl+g")
    settings_io.reset_setting(manager, setting)
    assert setting.key not in manager.get_config().values
    assert settings_io.read_setting(manager, setting) == setting.default
