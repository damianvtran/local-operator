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
from collections.abc import Mapping
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
    #: threshold up and drop the tip row on narrow terminals.
    tip: str


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
)

#: ``id -> KeyAction`` for the lookups the page, the app and the tips all do.
BY_ID: dict[str, KeyAction] = {action.id: action for action in KEY_ACTIONS}


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


def validate_key(value: Any) -> str | None:
    """``None`` when ``value`` is bindable, else the user-facing reason.

    Written FOR the user: ``settings_io``'s page prints it inline and keeps the
    editor open, so it has to say what to type rather than name a rule.

    Order matters. The RESERVED check runs before the vocabulary check so a
    user who presses ``ctrl+c`` is told what ctrl+c is for, rather than being
    told it is not a key — which would be false and would read as a bug.
    """
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
        if validate_key(raw) is not None:
            rejected.append(action.id)
            continue
        key = normalize_key(str(raw))
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


__all__ = [
    "BY_ID",
    "COMPOSER_KEYS",
    "KEYMAP_PREFIX",
    "KEY_ACTIONS",
    "KeyAction",
    "RESERVED_KEYS",
    "conflict_note",
    "effective_key",
    "format_key_display",
    "normalize_key",
    "resolved_keymap",
    "validate_key",
]
