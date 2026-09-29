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

    Scope-aware since desktop-scoped actions exist: the settings registry and
    the registry must agree over ALL actions, while the app drives only the
    app-scope half — and a desktop id appearing in ``BINDINGS`` is its own
    failure (a binding no terminal can press, whose empty action Textual would
    happily resolve only when the impossible key is hit).
    """
    from local_operator.tui.app import OperatorApp

    registry_ids = {action.id for action in keymap.KEY_ACTIONS}
    settings_keys = {setting.key for setting in settings_io.SETTINGS if setting.section == "keymap"}
    app_ids = {action.id for action in keymap.KEY_ACTIONS if action.scope == "app"}
    desktop_ids = {action.id for action in keymap.KEY_ACTIONS if action.scope == "desktop"}
    # `BINDINGS` is typed as accepting tuples as well as `Binding`s, so the id
    # is read through `getattr` rather than by attribute access.
    binding_ids = {
        str(getattr(binding, "id", "") or "")
        for binding in OperatorApp.BINDINGS
        if str(getattr(binding, "id", "") or "").startswith(keymap.KEYMAP_PREFIX)
    }

    assert registry_ids == settings_keys, "settings registry drifted from KEY_ACTIONS"
    assert app_ids == binding_ids, "OperatorApp.BINDINGS drifted from KEY_ACTIONS' app scope"
    assert desktop_ids & binding_ids == set(), "a desktop-scope action grew a Textual binding"


def test_every_binding_action_exists_on_the_app() -> None:
    """A binding naming a missing action fails only WHEN THE KEY IS PRESSED.

    Textual resolves the action lazily, so a typo here ships as a hotkey that
    does nothing on the one machine that presses it. Checked structurally
    instead — and INVERTED for desktop scope: a desktop action must have NO
    ``action_*`` method, because its key is not a Textual binding and an
    app-level method would be the first half of a dead binding.
    """
    from local_operator.tui.app import OperatorApp

    for action in keymap.KEY_ACTIONS:
        if action.scope == "desktop":
            assert not hasattr(OperatorApp, f"action_{action.action}"), action.id
            continue
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
    """A default that its own validator rejects would be unfixable from the page.

    Per scope: each default runs through the validator for its OWN scope, and
    against the reserved set that scope has — the app's `RESERVED_KEYS` are
    terminal keys, while a desktop value answers to `DESKTOP_RESERVED_COMBOS`
    (a desktop default that was an OS-reserved chord could never register).
    """
    for action in keymap.KEY_ACTIONS:
        assert (
            keymap.validate_key(action.default, scope=action.scope, action_id=action.id) is None
        ), action.id
        if action.scope == "desktop":
            assert action.default not in keymap.DESKTOP_RESERVED_COMBOS, action.id
        else:
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


@pytest.mark.parametrize(
    "key", ["escape", "ctrl+c", "ctrl+d", "super+d", "ctrl+q", "ctrl+m", "enter", "tab"]
)
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


def test_action_holding_reads_the_persisted_key_not_the_shipped_one() -> None:
    """The stale-map hazard, as a unit fact.

    ``set_keymap`` never rewrites ``OperatorApp.BINDINGS``, so anything that
    answers "who holds this key?" from the declared map keeps naming a
    remapped action at its SHIPPED key forever. That produced both halves of
    review round 1's M1 on the real page: a false positive warning that a
    freed key was taken, and a false negative missing the action that actually
    held the captured key.
    """
    values = {"keymap.new_session": "ctrl+g"}

    # The shipped key is now FREE and must not be attributed to anyone.
    assert keymap.action_holding("ctrl+n", values) is None
    # The key it actually moved to names the action that holds it now.
    holder = keymap.action_holding("ctrl+g", values)
    assert holder is not None and holder.id == "keymap.new_session"
    # An action sitting at its DEFAULT is still found: `resolved_keymap` omits
    # defaults (omission encodes "unset"), so an inverted resolver would miss
    # exactly the actions a user is most likely to collide with.
    at_default = keymap.action_holding("ctrl+s", values)
    assert at_default is not None and at_default.id == "keymap.resume"
    # A row is never its own victim.
    assert keymap.action_holding("ctrl+g", values, excluding="keymap.new_session") is None


def test_two_actions_may_not_share_one_key() -> None:
    """Design §C.4. Both victims are inside the feature being configured.

    Without this, the survivor is decided by position in ``KEY_ACTIONS`` and
    the loser is reachable only by slash command with nothing saying why —
    the silent-disable failure this module exists to prevent, reached from the
    other direction (review round 1, M2).
    """
    values = {"keymap.new_session": "ctrl+g"}

    problem = keymap.group_conflict("keymap.resume", "ctrl+g", values)
    # The message names the OVERLAPPING key, not "that key": with alternates
    # a submitted value can have one bad member and one good one.
    assert problem is not None and "already uses ctrl+g" in problem
    # Normalization is applied before comparing, so an equivalent spelling of
    # the same chord cannot slip past the check.
    assert keymap.group_conflict("keymap.resume", "CTRL+G", values) is not None
    # A free key, and the row keeping its own key, are both allowed.
    assert keymap.group_conflict("keymap.resume", "f5", values) is None
    assert keymap.group_conflict("keymap.new_session", "ctrl+g", values) is None


def test_a_shared_alternate_is_a_collision_even_when_the_strings_differ() -> None:
    """Key strings are comma-separated SETS, so they compare as sets.

    ``"f5,ctrl+g"`` and ``"ctrl+g"`` are different strings and the same
    binding as far as ``ctrl+g`` is concerned. The first implementation of the
    group check compared whole strings and let this through, which reopened
    M2 through the very path §H.1 recommends for a second key — measured on
    the real app: three presses of ``ctrl+g`` fired ``new_session`` every time
    and ``resume`` answered to no key at all (review round 2, M4).
    """
    # alternate on the LEFT, whole key on the right
    values = {"keymap.new_session": "f5,ctrl+g"}
    assert keymap.group_conflict("keymap.resume", "ctrl+g", values) is not None
    # whole key on the left, alternate on the right
    values = {"keymap.new_session": "ctrl+g"}
    assert keymap.group_conflict("keymap.resume", "f5,ctrl+g", values) is not None
    # alternate on BOTH sides, overlapping in one member only
    values = {"keymap.new_session": "f5,ctrl+g"}
    assert keymap.group_conflict("keymap.resume", "ctrl+t,ctrl+g", values) is not None
    # an alternate that collides with the OTHER row's shipped default, which is
    # absent from `values` entirely — the case an inverted resolver would miss.
    assert keymap.group_conflict("keymap.new_session", "f5,ctrl+s", {}) is not None

    # Disjoint alternates are legitimate and must still be allowed.
    values = {"keymap.new_session": "f5,ctrl+g"}
    assert keymap.group_conflict("keymap.resume", "f2,ctrl+t", values) is None
    # A row is never its own victim, including when it keeps its alternates.
    assert keymap.group_conflict("keymap.new_session", "f5,ctrl+g", values) is None


def test_the_refusal_names_the_overlapping_key_not_the_whole_value() -> None:
    """With alternates in play, "that key" is ambiguous.

    A user who submitted ``"f5,ctrl+g"`` has one bad member and one good one;
    a refusal that does not say which cannot be acted on.
    """
    problem = keymap.group_conflict(
        "keymap.resume", "ctrl+t,ctrl+g", {"keymap.new_session": "f5,ctrl+g"}
    )
    assert problem is not None
    assert "ctrl+g" in problem
    assert "ctrl+t" not in problem, f"named a key that does not collide: {problem}"


def test_alternates_splits_and_normalizes_every_member() -> None:
    """The set contract the group check rests on."""
    assert keymap.alternates("f5,ctrl+g") == frozenset({"f5", "ctrl+g"})
    assert keymap.alternates("CTRL+G") == frozenset({"ctrl+g"})
    # Spelling variants cannot appear as distinct members.
    assert keymap.alternates("ctrl+g,CTRL+G") == frozenset({"ctrl+g"})


# ---------------------------------------------------------------------------
# Desktop scope: the accelerator grammar and its dispatch
# ---------------------------------------------------------------------------


def test_desktop_values_canonicalize_into_one_stored_spelling() -> None:
    """Aliases, order and duplicates collapse — one physical chord, one string.

    `cmd`/`command`/`super` are the literal meta key, `option` is alt and
    `control` is ctrl; the fixed order makes `ctrl+meta+space` and
    `meta+ctrl+space` the same stored value, which is what the reserved set
    below matches against.
    """
    assert keymap.normalize_desktop_key("CMD+Option+Space") == "meta+alt+space"
    assert keymap.normalize_desktop_key("  Super + ALT + f8 ") == "meta+alt+f8"
    assert keymap.normalize_desktop_key("control+shift+N") == "ctrl+shift+n"
    assert keymap.normalize_desktop_key("ctrl+meta+space") == "meta+ctrl+space"
    assert keymap.normalize_desktop_key("alt+alt+ctrl+n") == "ctrl+alt+n"


def test_desktop_validation_refuses_os_reserved_chords_with_their_jobs() -> None:
    """A refusal names the job the chord already does, and alias/case spelling
    cannot smuggle the same chord past it."""
    for combo, reason in keymap.DESKTOP_RESERVED_COMBOS.items():
        if combo in ("escape", "tab", "enter"):
            # Key TOKENS: refused in any chord, with the same reason.
            assert keymap.validate_desktop_key(f"ctrl+{combo}") == reason
        else:
            assert keymap.validate_desktop_key(combo) == reason
    assert keymap.validate_desktop_key("CMD+SPACE") is not None
    assert (
        keymap.validate_desktop_key("ctrl+meta+space")
        == keymap.DESKTOP_RESERVED_COMBOS["meta+ctrl+space"]
    )


def test_desktop_validation_refuses_comma_alternates() -> None:
    """Electron has no alternates concept and the registrar expresses exactly
    one chord, so a value Textual would read as "two keys" must not be storable
    here. The TUI scope keeps alternates, unchanged."""
    reason = keymap.validate_desktop_key("ctrl+n,f5")
    assert reason is not None and "one chord" in reason
    assert keymap.validate_key("ctrl+n,f5") is None


def test_desktop_validation_needs_a_modifier_for_typable_keys() -> None:
    """A bare letter/digit/space would fire while the user types; a bare
    function key is not typable and is allowed (it is the classic media-key
    chord)."""
    reason = keymap.validate_desktop_key("n")
    assert reason is not None and "modifier" in reason
    assert keymap.validate_desktop_key("space") is not None
    assert keymap.validate_desktop_key("f8") is None


def test_desktop_validation_refuses_shape_errors() -> None:
    """Dangling modifiers, extra keys, unknown modifiers and unknown keys all
    come back as a reason, never as a silent store."""
    assert "add a key" in (keymap.validate_desktop_key("ctrl+alt") or "")
    assert "one key" in (keymap.validate_desktop_key("n+f5") or "")
    assert "unknown modifier" in (keymap.validate_desktop_key("banana+n") or "")
    assert "not a key" in (keymap.validate_desktop_key("banana") or "")
    assert keymap.validate_desktop_key("") is not None
    assert keymap.validate_desktop_key(7) is not None


def test_validate_key_dispatches_by_scope() -> None:
    """The SAME string is acceptable or refused depending on the row's scope.

    `primary+alt+space` is the shipped desktop default and is not a key any
    terminal can send; it must be refused by the app rules and accepted by the
    desktop ones. Scope dispatch is what makes the write boundary, the capture
    widget and `lop config edit` agree.
    """
    assert keymap.validate_key("primary+alt+space") is not None
    assert (
        keymap.validate_key("primary+alt+space", scope="desktop", action_id="keymap.quick_send")
        is None
    )
    assert keymap.validate_key("n", scope="desktop") is not None
    # An id the registry does not know falls back to the app scope rather than
    # silently becoming a desktop shortcut.
    assert keymap.scope_of("keymap.not_a_row") == "app"
    assert keymap.scope_of("keymap.quick_send") == "desktop"


def test_the_desktop_grammar_mapping_is_what_dispatches(monkeypatch: pytest.MonkeyPatch) -> None:
    """The per-ACTION grammar seam, pinned.

    `keymap.push_to_talk`'s planned bare-modifier hold is not expressible as
    an Electron accelerator, so validation and normalization dispatch through
    `_DESKTOP_GRAMMARS` keyed by action id rather than on scope alone; a fake
    grammar registered in-test is what proves the mapping is consulted instead
    of the accelerator rules being a scope-wide hardcode.
    """
    fake = keymap._DesktopGrammar(
        normalize=lambda text: f"fake:{text}",
        validate=lambda value: "the fake grammar refuses everything",
    )
    monkeypatch.setitem(keymap._DESKTOP_GRAMMARS, "keymap.push_to_talk", fake)
    assert (
        keymap.validate_desktop_key("alt-right-hold", action_id="keymap.push_to_talk")
        == "the fake grammar refuses everything"
    )
    assert keymap.normalize_desktop_key("alt-right-hold", action_id="keymap.push_to_talk") == (
        "fake:alt-right-hold"
    )
    assert (
        keymap.validate_key("alt-right-hold", scope="desktop", action_id="keymap.push_to_talk")
        == "the fake grammar refuses everything"
    )
    # No registered grammar still means the accelerator rules for a desktop row.
    assert keymap.validate_desktop_key("primary+alt+space", action_id="keymap.quick_send") is None


def test_resolved_keymap_applies_the_desktop_grammar() -> None:
    """A desktop override is canonicalized on the way in, and an unstorable
    value is dropped and named rather than leaving a dead shortcut."""
    action = keymap.BY_ID["keymap.quick_send"]
    assert keymap.resolved_keymap({action.id: "CMD+SHIFT+N"}) == (
        {action.id: "meta+shift+n"},
        [],
    )
    resolved, rejected = keymap.resolved_keymap({action.id: "meta+space"})
    assert resolved == {} and rejected == [action.id]


def test_group_conflict_compares_desktop_values_canonically() -> None:
    """Cross-scope collisions are found on the canonical string: a global
    chord that the app already binds to a terminal key is held, and the
    desktop row's own default does not collide with anything."""
    problem = keymap.group_conflict("keymap.quick_send", "ctrl+n", {})
    assert problem is not None and "ctrl+n" in problem
    assert keymap.group_conflict("keymap.quick_send", "primary+alt+space", {}) is None


def test_display_key_maps_platform_tokens_only_for_desktop_scope() -> None:
    """App scope is Textual's own vocabulary; desktop scope is the one place
    `primary`/`meta` become readable, per platform."""
    assert keymap.display_key("escape") == "esc"
    assert (
        keymap.display_key("primary+alt+space", scope="desktop", platform="darwin")
        == "cmd+alt+space"
    )
    assert (
        keymap.display_key("primary+alt+space", scope="desktop", platform="win32")
        == "ctrl+alt+space"
    )
    assert keymap.display_key("meta+shift+n", scope="desktop", platform="linux") == (
        "super+shift+n"
    )
    assert keymap.alternates("") == frozenset()
    assert keymap.alternates(" f5 , ctrl+g ") == frozenset({"f5", "ctrl+g"})
