"""The remappable action registry: binding ids, defaults, and key validation.

WHY THIS MODULE EXISTS
======================

Three surfaces have to agree about one vocabulary, and they live in three
different modules:

* ``OperatorApp.BINDINGS`` declares ``Binding(..., id="keymap.new_session")``;
* ``settings_io.SETTINGS`` declares a ``Setting(key="keymap.new_session")``
  the ``/settings`` page and ``lop config edit`` both drive;
* ``tui/widgets/welcome.py`` renders a tip naming the CURRENT key for that
  action.

Those three strings are one string. Declaring it three times is drift waiting
to happen, so it is declared HERE once and the others derive from
:data:`KEY_ACTIONS`. ``tests/unit/test_keymap.py`` asserts the three-way
identity, because the failure is silent: a renamed id does not raise, it
orphans every user's stored override and quietly restores the default.

**THE BINDING ID IS PERSISTED USER DATA.** It is the literal key in the user's
``config.yml``. Renaming ``keymap.new_session`` is a config migration, not a
refactor.

NO TEXTUAL IMPORT AT MODULE LEVEL, for the reason ``settings_io`` records: the
CLI's ``config edit``/``config list`` consult this registry through
``settings_io`` and must not pay for the TUI. :func:`normalize_key` and
:func:`validate_key` import ``textual.keys`` lazily and degrade to a
conservative parse when Textual is absent, so a TUI-less install can still
read and write the keys.

WHY VALIDATION LIVES HERE AND NOT IN THE CAPTURE UI
===================================================

Textual validates NOTHING. Measured against textual 8.2.8 in this worktree::

    'banana'    -> active={'banana':   'keymap.new_session'}
    'ctrl+'     -> active={'ctrl+':    'keymap.new_session'}
    'ctrl-n'    -> active={'ctrl-n':   'keymap.new_session'}
    original ctrl+n after the bogus remap -> []   # action UNREACHABLE

A garbage key string does not raise; it silently moves the binding to a key no
terminal can emit AND takes the default away with it. That is the Claude Code
pre-v2.1.246 silent-disable bug, live in Textual today.

The capture UI cannot be the only guard, because ``lop config edit`` and a
hand-edited ``config.yml`` reach the same value and bypass the page entirely.
So the predicate lives here, ``settings_io.validate`` calls it at the write
boundary, and the capture widget calls the SAME predicate — the two can never
disagree about what is bindable.

EXTENSION POINT, DELIBERATELY NOT BUILT
=======================================

A leader key (``ctrl+x ctrl+n``) is the structural answer to this app having
spent 9 of ~12 comfortable ``ctrl+<letter>`` slots, and Textual has no chord
support: a comma in a key string means ALTERNATES, not a sequence (measured).
Building one means hand-rolled modal state in ``OperatorApp.on_key`` with its
own timeout and its own interaction with the composer's escape-coalescing
machinery, which has produced defects in three consecutive rounds. Out of
scope. The schema does not preclude it — a future leader would store
``"ctrl+x ctrl+n"`` (space-separated, a spelling Textual will never claim) and
intercept before dispatch.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable, Mapping
from typing import Any

#: Prefix every remappable action's config key and binding id carries. The
#: settings page, ``_on_config_change`` and the resolver all key on it, so
#: "is this a hotkey?" is one string comparison rather than a membership test
#: that a newly-registered action could be missing from.
KEYMAP_PREFIX = "keymap."


@dataclasses.dataclass(frozen=True)
class KeyAction:
    """One remappable action.

    :attr:`id` is simultaneously the ``config.yml`` key, the
    ``textual.binding.Binding`` id, the entry ``config_watch`` reports in
    ``changed_keys``, and the tip lookup — see the module docstring. It is
    persisted user data; renaming it orphans overrides.
    """

    id: str
    #: The ``OperatorApp`` action this binding fires, without the ``action_``
    #: prefix. Spelled out rather than derived from :attr:`id`, so that
    #: grepping for the action name finds this row and so a rename of either
    #: cannot silently produce a binding pointing at a method that no longer
    #: exists (Textual reports an unknown action only when the key is pressed).
    action: str
    #: The row label on the ``/settings`` page.
    label: str
    #: The shipped key. Restored by deleting the override, never by writing
    #: this value back — ``settings_io.reset_setting`` deletes, and an absent
    #: key is how :func:`resolved_keymap` encodes "use the default".
    default: str
    #: One-line help for the settings row.
    help: str
    #: Splash-tip template with a single ``{key}`` field. Resolved at render
    #: from the PERSISTED value rather than from ``Binding.key``, which still
    #: reads the old key after a remap (measured). Keep the sentence at or
    #: under 32 cells excluding ``{key}`` — ``welcome.TIP_MIN_WIDTH`` budgets
    #: a worst-case key against it and an over-long template would push the
    #: threshold up and drop the tip row on narrow terminals. "" means "no
    #: tip" and ``welcome.KEYED_TIPS`` skips the row.
    tip: str
    #: Which surface OWNS the key: ``"app"`` for a Textual binding declared in
    #: ``OperatorApp.BINDINGS``, ``"desktop"`` for a global shortcut owned by
    #: the desktop app. A desktop action declares no ``action_*`` method, no
    #: ``Binding`` and no splash tip — the key's MEANING lives entirely in the
    #: consumer (toggle for ``quick_send``, press-and-hold for
    #: ``push_to_talk``), and this registry is semantics-agnostic by design;
    #: it owns only the stored value's grammar (see the desktop-grammar block
    #: below), because ``lop config edit`` and the desktop's settings page must
    #: refuse the same values the capture UI does.
    scope: str = "app"


#: Every remappable action, in the order the settings page lists them.
#:
#: ``ctrl+n`` and ``ctrl+s`` are both NON-PRIORITY bindings in ``app.py``, and
#: that is a hard constraint rather than an oversight — see the comment on the
#: bindings themselves. ``ctrl+n`` coexists with the pickers' own ``ctrl+n`` =
#: "move down" because Textual dispatches focused-widget-first for
#: non-priority bindings (measured: picker focused -> picker moves and the app
#: binding does not fire; composer focused -> the app binding fires and the
#: draft is untouched).
#:
#: ``ctrl+s`` is safe from XON/XOFF flow control because Textual's own driver
#: clears it (``textual/drivers/linux_driver.py``: "Don't capture Ctrl-S and
#: Ctrl-Q", i.e. ``stty -ixon``). It is also the ONLY genuinely free
#: ``ctrl+<letter>`` left in this application — ``h``/``i``/``j``/``m`` are
#: terminal aliases for backspace/tab/LF/CR — so it is spent knowingly. An
#: outer tmux or ssh may still swallow it before the app sees it; that is what
#: the remapping this module exists for is for.
KEY_ACTIONS: tuple[KeyAction, ...] = (
    KeyAction(
        id="keymap.new_session",
        action="keymap_new_session",
        label="New session",
        default="ctrl+n",
        help="Starts a fresh conversation, the same as /new.",
        tip="{key} starts a new conversation",
    ),
    KeyAction(
        id="keymap.resume",
        action="keymap_resume",
        label="Resume a session",
        default="ctrl+s",
        help="Opens the picker of recent conversations, the same as /resume.",
        tip="{key} reopens a recent conversation",
    ),
    # The first DESKTOP-scoped action. Not a Textual binding: no `action_*`
    # method, no `Binding`, and `tip=""` so the splash cannot advertise a key
    # no terminal can press. The stored default is `primary+alt+shift+space` —
    # ONE value that registers as ⌘⌥⇧Space on macOS and Ctrl+Alt+Shift+Space
    # on Windows/Linux, which is why the token is `primary` and not a per-OS
    # default; the registry must declare exactly one value because
    # `is_default` compares against it on whichever machine reads the config.
    # It replaced `primary+alt+space` (⌘⌥Space) in September 2026, when that
    # chord was found opening macOS's Finder Spotlight window (Apple KB 102650)
    # beside this app's mini composer — the reserved set below now refuses it,
    # so a revert cannot be stored.
    KeyAction(
        id="keymap.quick_send",
        action="",
        label="Quick send",
        default="primary+alt+shift+space",
        help="Opens a small composer over other apps; messages the chief of staff.",
        tip="",
        scope="desktop",
    ),
    # The second DESKTOP-scoped action and the first with its OWN value
    # grammar: a bare-modifier HOLD is not expressible as an Electron
    # accelerator, so `alt-right-hold` is validated by the hold grammar
    # registered in `_DESKTOP_GRAMMARS` below. Semantics stay consumer-side
    # (the STT stream's press-and-hold dictation); the registry stores the
    # combination and the scope only. `tip=""` like every desktop row — no
    # terminal can press it, and the splash must not advertise it. Help is
    # verbatim from the STT freeze; renaming it is a config migration because
    # the id is persisted user data.
    KeyAction(
        id="keymap.push_to_talk",
        action="",
        label="Push to talk",
        default="alt-right-hold",
        help="Hold to dictate (desktop); release to stop.",
        tip="",
        scope="desktop",
    ),
)

#: ``id -> KeyAction`` for the lookups the page, the app and the tips all do.
BY_ID: dict[str, KeyAction] = {action.id: action for action in KEY_ACTIONS}

#: ``id -> scope``, the one-question lookup every writer and display path
#: needs: "is this value a Textual key or a desktop shortcut?". Derived so a
#: new action cannot leave it behind.
SCOPE_BY_ID: dict[str, str] = {action.id: action.scope for action in KEY_ACTIONS}


def scope_of(action_id: str | None) -> str:
    """``action_id``'s scope; ``"app"`` for ids the registry does not know.

    The fallback is deliberate: an unknown id is not a desktop shortcut just
    because someone asked, and the app rules are the pre-existing behaviour
    every caller had before this axis existed.
    """
    return SCOPE_BY_ID.get(action_id or "", "app")


# ---------------------------------------------------------------------------
# The reserved set
# ---------------------------------------------------------------------------

#: Keys that may never be bound to a remappable action, and why.
#:
#: Refused at the WRITE boundary and echoed by the capture widget, so the page
#: and ``lop config edit`` agree. A refusal always states its reason: a key
#: that appears to do nothing in capture mode is indistinguishable from a
#: broken terminal.
#:
#: The ``ctrl+m``/``ctrl+i``/``ctrl+h``/``ctrl+j``/``ctrl+@`` entries are the
#: subtle ones. They are the SAME BYTES as enter/tab/backspace/LF/NUL, so
#: binding one silently binds the other — a user who bound ``ctrl+m`` would
#: find Enter no longer submits the composer, with nothing on screen relating
#: the two.
RESERVED_KEYS: dict[str, str] = {
    "escape": "esc cancels — it cannot be bound",
    "ctrl+c": "ctrl+c interrupts the agent — it cannot be bound",
    "ctrl+d": "ctrl+d quits — it cannot be bound",
    # `super+d` is Textual's kitty-protocol spelling of macOS Cmd+D. Reserve it
    # beside Ctrl+D so remapping another action cannot steal the graceful exit.
    "super+d": "cmd+d quits — it cannot be bound",
    "ctrl+q": "ctrl+q is Textual's own quit — it cannot be bound",
    "ctrl+m": "ctrl+m is the same byte as enter — it cannot be bound",
    "ctrl+i": "ctrl+i is the same byte as tab — it cannot be bound",
    "ctrl+h": "ctrl+h is the same byte as backspace — it cannot be bound",
    "ctrl+j": "ctrl+j is the same byte as a newline — it cannot be bound",
    "ctrl+@": "ctrl+@ is the same byte as NUL — it cannot be bound",
    "enter": "enter submits the composer — it cannot be bound",
    "tab": "tab moves focus — it cannot be bound",
    "space": "space types a space — it cannot be bound",
}

#: Keys the COMPOSER owns, which a non-priority app binding silently loses to
#: while the composer has focus — which is almost always.
#:
#: Measured (textual 8.2.8): a non-priority app binding remapped onto
#: ``ctrl+u`` did not fire AND the ``TextArea`` deleted the line, while
#: ``clashed_bindings`` reported NOTHING — Textual's clash detection only
#: covers bindings in the same ``BindingsMap``, i.e. app-level ones. So this
#: static table is the only warning available for this class, and it has to be
#: in the first cut rather than deferred: without it a user binds ``ctrl+u``,
#: sees nothing happen, and reasonably concludes the feature is broken.
#:
#: Read from ``Editor.BINDINGS`` plus the ``TextArea`` keys it inherits. Not
#: derived at import: that would mean importing the TUI here, which the module
#: docstring forbids, and the set is stable enough that the anti-drift risk is
#: smaller than the import cost.
COMPOSER_KEYS: frozenset[str] = frozenset(
    {
        # Editor.BINDINGS
        "alt+left",
        "alt+right",
        "alt+shift+left",
        "alt+shift+right",
        "alt+b",
        "alt+f",
        "ctrl+v",
        "super+v",
        "ctrl+o",
        # TextArea's own editing keys
        "ctrl+a",
        "ctrl+e",
        "ctrl+w",
        "ctrl+x",
        "ctrl+k",
        "ctrl+u",
        "ctrl+y",
        "ctrl+z",
        "ctrl+f",
        "ctrl+left",
        "ctrl+right",
        "ctrl+shift+left",
        "ctrl+shift+right",
        "ctrl+shift+k",
        "f6",
        "f7",
        "super+c",
        "super+x",
        "super+y",
        "super+z",
        "super+backspace",
        "alt+backspace",
        "alt+delete",
        "ctrl+backspace",
        "backspace",
        "delete",
        "home",
        "end",
        "pageup",
        "pagedown",
        "up",
        "down",
        "left",
        "right",
    }
)

#: Values a DESKTOP-scoped action may never be bound to, and why.
#:
#: The same philosophy as :data:`RESERVED_KEYS`: a refusal always names the
#: job the combination already does, because "that value appears to do
#: nothing" is indistinguishable from a broken install. Two halves:
#:
#: * **full chords**, keyed by the CANONICAL string (the same normalization
#:   the write boundary applies, so a hand-written alias or a different
#:   modifier order hits the same entry — e.g. ``cmd+space`` and
#:   ``meta+space`` are one key);
#: * **key TOKENS** every app depends on (``escape``/``tab``/``enter``),
#:   refused in ANY chord: a global binding on one steals it system-wide, so
#:   v1 does not permit them as the key half at all.
DESKTOP_RESERVED_COMBOS: dict[str, str] = {
    "meta+space": "Spotlight uses ⌘Space — pick a chord with another modifier",
    "primary+space": "Spotlight uses ⌘Space — pick a chord with another modifier",
    # The Finder/Spotlight search window (Option-Command-Space, Apple KB
    # 102650) — the chord quick-send shipped on until September 2026, found by
    # pressing it: the Finder search window opened beside the mini composer.
    # Both spellings, because `meta` IS ⌘ on macOS; the original collision
    # survey missed this one, so it is refused rather than left to be
    # rediscovered.
    "meta+alt+space": "Spotlight uses ⌘⌥Space in Finder — pick a chord with another modifier",
    "primary+alt+space": "Spotlight uses ⌘⌥Space in Finder — pick a chord with another modifier",
    "ctrl+space": "input-source switching uses Ctrl+Space",
    "meta+ctrl+space": "the emoji picker uses ⌃⌘Space",
    "alt+space": (
        "on Windows this opens the window menu; on macOS Option+Space " "types a non-breaking space"
    ),
    "meta+tab": "⌘Tab switches apps",
    "alt+f4": "closes the window on Windows",
    "ctrl+alt+delete": "the OS owns it",
    "meta+q": "quits apps on macOS",
    "escape": "esc cancels in every app — a global binding would steal it system-wide",
    "tab": "tab moves focus in every app — a global binding would steal it system-wide",
    "enter": "enter submits in every app — a global binding would steal it system-wide",
}


# ---------------------------------------------------------------------------
# Key strings: normalize, validate
# ---------------------------------------------------------------------------


def _valid_key_names() -> frozenset[str]:
    """Textual's canonical key vocabulary, or ``frozenset()`` without Textual.

    An empty set means :func:`validate_key` falls back to a structural parse
    rather than rejecting everything: a TUI-less install must still be able to
    run ``lop config edit keymap.new_session ctrl+g``.
    """
    try:
        from textual.keys import Keys
    except Exception:  # noqa: BLE001 — a TUI-less install is a supported one
        return frozenset()
    return frozenset(member.value for member in Keys)


def normalize_key(text: str) -> str:
    """Textual's own spelling of ``text``.

    Applied at the WRITE boundary and not only at apply, because two of the
    three writers never touch the capture UI. The capture widget receives an
    already-normalized ``event.key`` from Textual, so it is a no-op there;
    ``lop config edit`` and a hand edit are why it exists. Without it the file
    holds ``ctrl+N``, the page displays ``ctrl+N``, and the runtime binds a key
    nobody can press (measured: ``'ctrl+N'`` is accepted verbatim and becomes
    an unreachable binding).

    Comma-separated ALTERNATES are preserved: Textual treats ``'ctrl+n,f5'`` as
    two keys that both fire the action (measured), and refusing a spelling the
    runtime honours would make the file and the page disagree.
    """
    parts = [part.strip().lower() for part in text.split(",")]
    parts = [part for part in parts if part]
    try:
        from textual.keys import _normalize_key_list

        return _normalize_key_list(",".join(parts))
    except Exception:  # noqa: BLE001 — no Textual, or a private API that moved
        return ",".join(parts)


def alternates(text: str) -> frozenset[str]:
    """The set of individual keys ``text`` binds, normalized.

    A Textual key string is a COMMA-SEPARATED SET, not a scalar: ``"f5,ctrl+g"``
    binds both, and any one of them firing is enough to make another action
    holding it unreachable. Every comparison between two configured keys must
    therefore be a set operation — a string comparison reports ``"f5,ctrl+g"``
    and ``"ctrl+g"`` as different values while the app has them colliding on
    ``ctrl+g``, which is how the group guard was bypassed (review round 2, M4).

    Normalized per element through the same path the write boundary uses, so
    ``"CTRL+G"`` and ``"ctrl+g"`` cannot both be present as distinct members.
    """
    return frozenset(part for part in normalize_key(text).split(",") if part)


def _is_valid_single_key(key: str, vocabulary: frozenset[str]) -> bool:
    """Whether ``key`` names something a terminal can actually send."""
    if not key:
        return False
    if vocabulary and key in vocabulary:
        return True
    # A single printable character is a legal key and is NOT a `Keys` member,
    # so the vocabulary check alone would reject `a` and `?`. Textual maps
    # these to key names during normalization, so anything still one character
    # here is a character key.
    if len(key) == 1 and key.isprintable():
        return True
    if not vocabulary:
        # No Textual: accept a conservative structural spelling
        # (`mod+mod+name`, non-empty parts) rather than rejecting everything.
        return all(part for part in key.split("+"))
    return False


def validate_key(value: Any, *, scope: str = "app", action_id: str | None = None) -> str | None:
    """``None`` when ``value`` is bindable, else the user-facing reason.

    Written FOR the user: ``settings_io``'s page prints it inline and keeps the
    editor open, so it has to say what to type rather than name a rule.

    Order matters. The RESERVED check runs before the vocabulary check so a
    user who presses ``ctrl+c`` is told what ctrl+c is for, rather than being
    told it is not a key — which would be false and would read as a bug.

    ``scope`` selects the VALUE GRAMMAR. An app-scope value is a Textual key
    (the rules below); a desktop-scope value is a global shortcut, validated by
    the grammar registered for ``action_id`` (default: the accelerator one —
    see ``_DESKTOP_GRAMMARS``). An unknown scope falls back to the app rules,
    which is the pre-existing behaviour every caller had before this axis
    existed.
    """
    if scope == "desktop":
        return validate_desktop_key(value, action_id=action_id)
    if not isinstance(value, str):
        return "expected a key, like ctrl+n or f5"
    normalized = normalize_key(value)
    if not normalized:
        return "expected a key, like ctrl+n or f5"
    vocabulary = _valid_key_names()
    for part in normalized.split(","):
        reason = RESERVED_KEYS.get(part)
        if reason is not None:
            return reason
        if len(part) == 1 and part.isprintable():
            # A bare printable character IS text. Binding `n` makes the
            # composer unusable in a way that is hard to relate back to a
            # settings row, so it is refused with the reason rather than
            # accepted as a technically-valid key.
            return "a plain character types into the composer — add ctrl, alt or super"
        if not _is_valid_single_key(part, vocabulary):
            return "not a key this terminal can send — try ctrl+n, f5, or press a key to capture"
    return None


# ---------------------------------------------------------------------------
# Desktop-scoped values: the accelerator grammar
# ---------------------------------------------------------------------------
#
# A desktop value is NOT a Textual key: it is an Electron accelerator written
# in this module's own stored vocabulary, and the grammar lives here so every
# writer — the capture widget, ``lop config edit`` and the desktop's settings
# page — refuses the same values. `normalize_key` is the WRONG tool for it:
# Textual's only transformation is single-character expansion, and every
# desktop value is multi-token, so it would pass through verbatim except for
# case (design §A.4).

#: Canonical modifier order. Load-bearing, not cosmetic: it makes
#: ``ctrl+meta+space`` and ``meta+ctrl+space`` ONE stored string, so the
#: reserved table below and every comparison see a single spelling. Matches the
#: design's own examples (``primary+alt+shift+space``, ``meta+shift+n``,
#: ``meta+ctrl+space``).
_DESKTOP_MODIFIER_ORDER = ("primary", "meta", "ctrl", "alt", "shift")

#: Aliases canonicalized on read, so one physical modifier has one spelling in
#: the file: ``cmd``/``command``/``super`` are the literal meta key, ``option``
#: is alt, ``control`` is ctrl.
_DESKTOP_MODIFIER_ALIASES = {
    "cmd": "meta",
    "command": "meta",
    "super": "meta",
    "option": "alt",
    "control": "ctrl",
}

#: Named non-typable keys a desktop value may end in. ``space``, single
#: letters/digits and ``f1``..``f24`` are handled by the token predicate;
#: ``escape``/``tab``/``enter`` are deliberately ABSENT — they are refused
#: with their reasons (the key-token half of :data:`DESKTOP_RESERVED_COMBOS`).
_DESKTOP_NAMED_KEY_TOKENS = frozenset(
    {
        "backspace",
        "delete",
        "insert",
        "home",
        "end",
        "pageup",
        "pagedown",
        "up",
        "down",
        "left",
        "right",
    }
)


def _is_desktop_modifier(token: str) -> bool:
    return token in _DESKTOP_MODIFIER_ORDER


def _is_desktop_function_key(token: str) -> bool:
    """The one key class that may stand BARE in a desktop value: F1..F24
    (design §A.5: "Bare function keys are allowed (they are not typable)")."""
    if token.startswith("f") and token[1:].isdigit():
        return 1 <= int(token[1:]) <= 24
    return False


def _is_valid_desktop_key_token(token: str) -> bool:
    """Whether ``token`` can be the ONE key half of a desktop value."""
    if token in _DESKTOP_NAMED_KEY_TOKENS or token == "space":
        return True
    if len(token) == 1 and token.isascii() and (token.isalpha() or token.isdigit()):
        return True
    return _is_desktop_function_key(token)


def _canonical_desktop_tokens(text: str) -> tuple[list[str], list[str]]:
    """``(modifiers, keys)`` for ``text``, canonicalized and de-duplicated.

    Total by design: garbage produces garbage in canonical shape, and the
    REFUSALS are :func:`_validate_accelerator`'s job. Modifiers take the fixed
    order; keys keep their relative order (there should be exactly one).
    """
    tokens = [
        _DESKTOP_MODIFIER_ALIASES.get(part.strip().lower(), part.strip().lower())
        for part in text.split("+")
    ]
    tokens = [token for token in tokens if token]
    modifiers = [m for m in _DESKTOP_MODIFIER_ORDER if m in tokens]
    keys = list(dict.fromkeys(t for t in tokens if t not in _DESKTOP_MODIFIER_ORDER))
    return modifiers, keys


def _normalize_accelerator(text: str) -> str:
    """Canonical stored form: lowercase, aliases mapped, fixed modifier order,
    duplicates collapsed, exactly one key last."""
    modifiers, keys = _canonical_desktop_tokens(text)
    return "+".join(modifiers + keys)


def _validate_accelerator(value: Any) -> str | None:
    """The accelerator rules (design §A.5), in rule order.

    Structural only: the OS is the authority on what actually registers, so a
    value this accepts can still fail registration in the desktop app — that
    is a registration failure to SURFACE (§G.5), never a silent dead key.
    """
    if not isinstance(value, str):
        return "expected a shortcut, like primary+alt+shift+space"
    if "," in value:
        # Electron has no alternates concept and the registrar can express
        # exactly one chord per action, so a value Textual would read as "two
        # keys" must not be storable here (design §A.3).
        return "a global shortcut sets exactly one chord"
    tokens = [
        _DESKTOP_MODIFIER_ALIASES.get(part.strip().lower(), part.strip().lower())
        for part in value.split("+")
    ]
    tokens = [token for token in tokens if token]
    if not tokens:
        return "expected a shortcut, like primary+alt+shift+space"
    key_token = tokens[-1]
    for token in tokens[:-1]:
        if not _is_desktop_modifier(token):
            # A reserved token in modifier position gets its JOB named, not a
            # generic "unknown modifier" — the same reason the key half does.
            reserved_reason = DESKTOP_RESERVED_COMBOS.get(token)
            if reserved_reason is not None:
                return reserved_reason
            if _is_valid_desktop_key_token(token):
                return "a shortcut has one key and it goes last — like ctrl+f5"
            return f"unknown modifier `{token}`"
    if _is_desktop_modifier(key_token):
        return "add a key to the shortcut, like primary+alt+shift+space"
    # The FULL chord first, so a chord the table names specifically (`meta+tab`
    # → "⌘Tab switches apps") states its own job rather than the generic
    # reason of its key half; then the key TOKEN, which is refused in any chord.
    combo_reason = DESKTOP_RESERVED_COMBOS.get(_normalize_accelerator(value))
    if combo_reason is not None:
        return combo_reason
    token_reason = DESKTOP_RESERVED_COMBOS.get(key_token)
    if token_reason is not None:
        return token_reason
    if not _is_valid_desktop_key_token(key_token):
        return (
            f"not a key: `{key_token}` — use space, a letter, a digit, "
            "F1–F24, or a named key like pageup"
        )
    if len(tokens) == 1 and not _is_desktop_function_key(key_token):
        # THE bare carve-out is F1..F24 ONLY (design §A.5, "Bare function
        # keys are allowed (they are not typable)"). Every other bare key is
        # refused: a printable character and `space` fire while the user
        # types, and a NAMED key (`delete`, `backspace`, an arrow, `pageup`, …)
        # is consumed by every editing app all the same — a global binding on
        # one steals it system-wide. Review round 1, F1: the first cut
        # accepted the whole non-typable class, and QA reproduced a bare
        # `delete` storing on all three write paths.
        return "a global shortcut needs a modifier — it would otherwise fire while you type"
    return None


#: The accepted tokens of the bare-modifier HOLD grammar (v1, frozen with the
#: STT stream): one bare modifier, a side, and the ``-hold`` suffix — no key
#: token, no space or letter chords, and no modifiers stacked together. Hold
#: semantics live entirely in the consumer (press-and-hold to dictate); the
#: registry stores the combination and the scope only, which is why this is a
#: second value GRAMMAR rather than a second meaning for the accelerator
#: tokens. The two grammars refuse each other's values by construction.
_HOLD_TOKENS: tuple[str, ...] = (
    "alt-right-hold",
    "alt-left-hold",
    "meta-right-hold",
    "meta-left-hold",
    "ctrl-right-hold",
    "ctrl-left-hold",
)

#: Display labels for the hold family, per platform class. macOS gets the mac
#: key names; on the other platforms ``alt`` reads "Command" because the STT
#: freeze pins the default's non-mac string as "Right-Command (hold)" —
#: display-only, one line to amend if that freeze was a placeholder — and
#: ``meta`` follows :func:`display_key`'s own meta convention (``win`` /
#: ``super``).
_HOLD_DISPLAY_LABELS: dict[str, dict[str, str]] = {
    "darwin": {"alt": "Option", "meta": "Command", "ctrl": "Control"},
    "win32": {"alt": "Command", "meta": "Win", "ctrl": "Control"},
    "other": {"alt": "Command", "meta": "Super", "ctrl": "Control"},
}


def _normalize_hold(text: str) -> str:
    """Canonical stored form of a hold value: trimmed and lowercased.

    Nothing else to canonicalize — a hold is ONE token with no separators, so
    case is the only spelling a hand-editor can vary."""
    return text.strip().lower()


def _validate_hold(value: Any) -> str | None:
    """The hold-family rules. No structural parse: membership IS the rule, and
    the refusal names the whole family so a missed spelling is one glance away.
    """
    if not isinstance(value, str):
        return "expected a hold key, like alt-right-hold"
    if _normalize_hold(value) in _HOLD_TOKENS:
        return None
    return (
        "not a hold key — use a bare modifier hold: alt-right-hold, "
        "alt-left-hold, meta-right-hold, meta-left-hold, ctrl-right-hold, "
        "or ctrl-left-hold"
    )


def _display_hold(value: str, plat: str) -> str:
    """``alt-right-hold`` -> ``Right-Option (hold)`` on macOS.

    Presentation only: the stored token is the platform-independent one and
    the consumer maps it to a physical key."""
    modifier, side = value.removesuffix("-hold").split("-", 1)
    klass = "darwin" if plat == "darwin" else ("win32" if plat == "win32" else "other")
    return f"{side.title()}-{_HOLD_DISPLAY_LABELS[klass][modifier]} (hold)"


@dataclasses.dataclass(frozen=True)
class _DesktopGrammar:
    """One value grammar for desktop-scoped rows: canonicalize + refuse.

    The registry is GENERIC over desktop-scoped actions, but "desktop scope"
    alone does not define a value grammar: dispatching on scope alone would
    hardcode the accelerator one for every desktop action and block the next
    one to arrive with a different value space.
    """

    normalize: Callable[[str], str]
    validate: Callable[[Any], str | None]
    #: Whether the /settings capture gesture can produce a value in this
    #: grammar at all. App rows always can; the hold grammar cannot — terminal
    #: input reports a KEY, never "the bare modifier is down", so its row is
    #: app-configured only and must refuse to ARM rather than paint a
    #: listening frame whose every completion the validator refuses.
    capturable_in_terminal: bool = True


_ACCELERATOR_GRAMMAR = _DesktopGrammar(
    normalize=_normalize_accelerator, validate=_validate_accelerator
)

_HOLD_GRAMMAR = _DesktopGrammar(
    normalize=_normalize_hold, validate=_validate_hold, capturable_in_terminal=False
)

#: action id -> the grammar its desktop value is written in. ``quick_send``
#: uses the accelerator default (``primary+alt+shift+space``); ``push_to_talk``
#: registered the bare-modifier HOLD grammar when the STT stream froze its
#: value space — a hold (``alt-right-hold``) is not expressible as an Electron
#: accelerator, which is exactly why this seam exists. Keep this a plain dict;
#: a plugin system is out of scope.
_DESKTOP_GRAMMARS: dict[str, _DesktopGrammar] = {
    "keymap.push_to_talk": _HOLD_GRAMMAR,
}


def _grammar_for(action_id: str | None) -> _DesktopGrammar:
    return _DESKTOP_GRAMMARS.get(action_id or "", _ACCELERATOR_GRAMMAR)


def capturable_in_terminal(action_id: str) -> bool:
    """Whether the /settings capture gesture can produce a value for this row.

    App rows always can. For a desktop row the answer comes from its grammar:
    the accelerator family is captured as a chord, while a bare modifier HOLD
    has no terminal representation at all — so the hold row is app-configured
    only and its capture affordance must not arm (design: the refusal is the
    row's desktop-scope detail, not a listening frame the validator would
    reject on every completion).
    """
    if scope_of(action_id) != "desktop":
        return True
    return _grammar_for(action_id).capturable_in_terminal


def normalize_desktop_key(text: str, *, action_id: str | None = None) -> str:
    """Canonical stored form of a desktop value, in ``action_id``'s grammar.

    One stored value maps correctly on every platform by construction:
    ``primary`` is the platform's command modifier (⌘ / Ctrl) and ``meta`` the
    literal Super/Command key, so the string in the file is
    platform-independent.
    """
    return _grammar_for(action_id).normalize(text)


def validate_desktop_key(value: Any, *, action_id: str | None = None) -> str | None:
    """``None`` when ``value`` is storable for ``action_id``, else the reason.

    Called from :func:`validate_key` for desktop-scope rows; ``action_id``
    selects the grammar (default: the accelerator one) so a future desktop
    action with a different value space cannot be validated by the wrong
    rules.
    """
    return _grammar_for(action_id).validate(value)


def _canonical_variants(action_id: str | None, text: str) -> frozenset[str]:
    """The set of strings ``text`` collides by, canonicalized in its own scope.

    App values are Textual key SETS (comma alternates); desktop values are a
    single canonical chord. Cross-scope platform equivalence (``primary`` vs
    ``meta`` on macOS) is deliberately NOT modelled in v1: nothing here can
    compare two operating systems at once, so the hard case — a TUI key and a
    global key that coincide on ONE platform — warns at most, never refuses
    (the warn-and-allow philosophy :func:`conflict_note` states for the soft
    class).
    """
    if scope_of(action_id) == "desktop":
        normalized = normalize_desktop_key(text, action_id=action_id)
        return frozenset({normalized}) if normalized else frozenset()
    return alternates(text)


def conflict_note(action_id: str, key: str) -> str:
    """A warning naming what ``key`` would cost, or ``""`` when it costs nothing.

    This covers the class Textual CANNOT see. Its own clash detection
    (``handle_bindings_clash``) only reports bindings in the same
    ``BindingsMap``, i.e. app-level ones, so a remap onto a ``TextArea``
    editing key reports nothing at all (measured) — and that is the class that
    matters most under a non-priority binding, because it silently does
    nothing while the composer has focus, which is almost always. App-level
    clashes need no table: the settings page reads the app's own declared
    binding map and names the victim by its description.

    Warn-and-allow rather than refuse. The reserved set above is already
    enormous and mostly context-scoped, so refusing every soft conflict makes
    the feature feel arbitrary; stealing silently is worse, because the victim
    is invisible until the user notices weeks later that something stopped
    working. Naming the cost at the moment of the decision is the middle path.
    """
    del action_id  # symmetric with the app-level path, which does key on it
    for part in normalize_key(key).split(","):
        if part in COMPOSER_KEYS:
            return f"the composer uses {part} — the hotkey will not fire while you are typing"
    return ""


def action_holding(
    key: str, values: Mapping[str, Any], *, excluding: str | None = None
) -> KeyAction | None:
    """The remappable action that answers to ``key`` RIGHT NOW, or ``None``.

    Resolved from the persisted values, never from ``OperatorApp.BINDINGS``.
    That is the whole point of this function and it is not a style preference:
    ``set_keymap`` does not rewrite the class-declared map, so the declared map
    keeps reporting a remapped action at its SHIPPED key forever (measured —
    declared is byte-identical before and after a remap). A caller that asks
    the declared map "who holds ctrl+n?" after ``new_session`` moved to
    ``ctrl+g`` is told "New session" about a key that is now free, and is told
    nothing about ``ctrl+g``, which is the collision that actually breaks
    something. Both were reproduced on the real app (review round 1, M1).

    ``excluding`` is the id being edited, so a row never reports itself as its
    own victim.

    Compares against :func:`effective_key` per action rather than inverting
    :func:`resolved_keymap`, because the resolver deliberately OMITS an action
    sitting at its default — omission is how "unset" is encoded — so an
    inverted map would be blind to exactly the default-keyed actions a user is
    most likely to collide with.

    Compared as ALTERNATE SETS and not as strings. A Textual key string may
    carry comma-separated alternates (``"f5,ctrl+g"`` binds both), a form
    ``normalize_key`` preserves and which §H.1 of the design deliberately
    routes users to via ``lop config edit`` since capture takes one key. A
    whole-string comparison sees ``"f5,ctrl+g" != "ctrl+g"`` and reports no
    holder while the app has both actions live on ``ctrl+g`` — measured: three
    presses fired ``new_session`` every time and ``resume`` was reachable by no
    key at all (review round 2, M4). Any overlap is a collision, because any
    single shared alternate is enough to make one action unreachable. Desktop
    values are one canonical chord instead — see :func:`_canonical_variants`.
    """
    wanted = _canonical_variants(excluding, key)
    if not wanted:
        return None
    for action in KEY_ACTIONS:
        if action.id == excluding:
            continue
        if wanted & _canonical_variants(action.id, effective_key(action, values)):
            return action
    return None


def group_conflict(action_id: str, key: str, values: Mapping[str, Any]) -> str | None:
    """Why ``key`` may not be given to ``action_id``, or ``None``.

    TWO REMAPPABLE ACTIONS MAY NOT SHARE A KEY, and this is a refusal rather
    than the warn-and-allow the soft conflicts get. The asymmetry is
    deliberate and design §C.4 argued it: a soft conflict trades one
    context-scoped binding the user may never use for a hotkey they asked for,
    which is a trade they can reasonably want, whereas two ids on one key has
    no reading under which it is intended — BOTH victims are inside the
    feature being configured, and the survivor is decided by nothing better
    than position in :data:`KEY_ACTIONS`. Reproduced before the fix: both ids
    on ``ctrl+g`` made three presses fire ``['new','new','new']`` with
    ``/resume`` reachable only by slash command and nothing anywhere saying
    why (review round 1, M2) — the silent-disable failure this module exists
    to prevent, arrived at from the other direction.

    Lives here rather than in the page because ``lop config edit`` and a hand
    edit reach the same values with no UI to warn at all, and it is called
    from ``settings_io.validate`` so every writer shares one answer.

    Takes ``values`` rather than reading config itself, so it stays pure and
    the caller decides which snapshot is authoritative.

    Compares ALTERNATE SETS via :func:`action_holding`, so ``"f5,ctrl+g"``
    against ``"ctrl+g"`` is caught. The message names the OVERLAPPING keys
    rather than the whole value, because with alternates in play "that key" is
    ambiguous — the user typed a two-key string and only one of them is the
    problem, and a refusal that does not say which is a refusal they cannot
    act on.
    """
    other = action_holding(key, values, excluding=action_id)
    if other is None:
        return None
    shared = sorted(
        _canonical_variants(action_id, key)
        & _canonical_variants(other.id, effective_key(other, values))
    )
    return (
        f"{other.label.lower()} already uses {', '.join(shared)}"
        " — pick another, or change that row first"
    )


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def resolved_keymap(values: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """``({binding id: key}, [rejected ids])`` for the overrides in ``values``.

    Pure and app-free, so the whole resolution rule is testable without a
    terminal.

    An ABSENT or default-valued key is simply omitted, and that is the correct
    encoding rather than a shortcut: ``App.set_keymap`` applies against the
    pristine class ``BINDINGS`` rather than cumulatively, so an empty mapping
    restores every default (measured: ``set_keymap({})`` -> the class default
    fires again). The resolver therefore never has to compute a diff or an
    "unset" instruction.

    An UNPARSEABLE value on disk is DROPPED and named, so the caller can print
    one notice, and the action keeps its shipped default. Never leaving the
    action unreachable is ``tui/settings.py``'s own rule — "a missing or
    unreadable config never breaks the TUI" — applied to the one setting whose
    breakage would remove the user's route to fixing it.
    """
    resolved: dict[str, str] = {}
    rejected: list[str] = []
    for action in KEY_ACTIONS:
        raw = values.get(action.id)
        if raw is None:
            continue
        # Scope-aware: a desktop value is neither a Textual key nor a comma
        # set, so the app rules would reject every one of them, and the app
        # normalization would not touch its canonical form.
        if validate_key(raw, scope=action.scope, action_id=action.id) is not None:
            rejected.append(action.id)
            continue
        key = (
            normalize_desktop_key(str(raw), action_id=action.id)
            if action.scope == "desktop"
            else normalize_key(str(raw))
        )
        if key and key != action.default:
            resolved[action.id] = key
    return resolved, rejected


def effective_key(action: KeyAction, values: Mapping[str, Any]) -> str:
    """The key ``action`` actually answers to, given ``values``.

    Read from the PERSISTED value, never from ``Binding.key``: measured, the
    class binding map still reports the OLD key after a remap, so a tip built
    on it would confidently name a key that no longer works.
    """
    resolved, _ = resolved_keymap(values)
    return resolved.get(action.id, action.default)


def format_key_display(key: str) -> str:
    """``key`` in Textual's own footer vocabulary (``escape`` -> ``esc``).

    Routed through ``textual.keys.format_key`` so the settings page, the tips
    and Textual's own surfaces cannot spell the same key two ways.
    """
    try:
        from textual.keys import format_key
    except Exception:  # noqa: BLE001 — no Textual: the raw spelling is honest
        return key
    return ",".join(format_key(part) for part in key.split(",") if part)


def display_key(value: str, *, scope: str = "app", platform: str | None = None) -> str:
    """``value`` the way a user should READ it, for its own scope.

    App scope is Textual's footer vocabulary (:func:`format_key_display`).
    Desktop scope maps the platform tokens for DISPLAY — ``primary`` reads
    ``cmd`` on macOS and ``ctrl`` on Windows/Linux, ``meta`` reads
    ``cmd``/``win``/``super`` — because neither token is a key any terminal can
    press or show. A hold token reads ``<Side>-<ModifierLabel> (hold)``
    (``alt-right-hold`` -> ``Right-Option (hold)`` on macOS). Presentation
    only: the stored value is untouched.
    """
    if scope != "desktop":
        return format_key_display(value)
    plat = platform or sys.platform
    hold = value.strip().lower()
    if hold in _HOLD_TOKENS:
        return _display_hold(hold, plat)
    shown: list[str] = []
    for part in value.split("+"):
        token = _DESKTOP_MODIFIER_ALIASES.get(part.strip().lower(), part.strip().lower())
        if not token:
            continue
        if token == "primary":
            shown.append("cmd" if plat == "darwin" else "ctrl")
        elif token == "meta":
            shown.append({"darwin": "cmd", "win32": "win"}.get(plat, "super"))
        else:
            shown.append(token)
    return "+".join(shown)


__all__ = [
    "BY_ID",
    "COMPOSER_KEYS",
    "DESKTOP_RESERVED_COMBOS",
    "KEYMAP_PREFIX",
    "KEY_ACTIONS",
    "SCOPE_BY_ID",
    "KeyAction",
    "RESERVED_KEYS",
    "capturable_in_terminal",
    "conflict_note",
    "display_key",
    "effective_key",
    "format_key_display",
    "normalize_desktop_key",
    "normalize_key",
    "resolved_keymap",
    "scope_of",
    "validate_desktop_key",
    "validate_key",
]
