"""Declarative registry of every user-settable ``config.yml`` value, plus the
read/validate/write facade the ``/settings`` page and the CLI both drive.

WHY THIS MODULE EXISTS
======================

:class:`~local_operator.config.ConfigManager` has **no nested-key writer**.
``set_config_value`` is a plain ``dict.__setitem__`` on ``Config.values``
followed by a whole-file ``yaml.dump``; there is no ``set("retry.maxRetries",
10)``. Every nested write in the codebase today is therefore a hand-rolled
read-modify-write — ``OperatorApp._persist_theme``,
``web_search.service.save_search_settings``,
``web_fetch.service.save_fetch_settings`` — each one re-deriving "read the
sub-mapping, copy it, poke one key, put it back". That is fine for three call
sites and untenable for a page that offers ~50 of them, so the merge rule lives
here once.

The merge is not cosmetic. ``ConfigManager._load_config`` back-fills **missing
top-level keys only**: a config carrying a partial ``retry:`` block never gets
its missing siblings back. A writer that REPLACED ``retry`` with
``{"maxRetries": 4}`` would silently destroy ``fallbackChains``,
``usageAwareFallback`` and the rest, and nothing would report it until a
failover did not happen. :func:`write_setting` merges into the existing
sub-mapping and never replaces it.

THE ``display.*`` FLAT-KEY TRAP
===============================

``display.shimmer`` is a **literal dotted key at the TOP LEVEL** of ``values``
— ``tui/settings.py`` reads ``values.get("display.shimmer")``, not
``values["display"]["shimmer"]`` — whereas ``retry.maxRetries`` is genuinely
nested. A facade that split every key on ``.`` would write a ``display:``
mapping that **nothing reads**: the toggle would report success, the config
file would gain a plausible-looking block, and the flag would never change.
That is a silent failure that looks like it worked, which is why the path is
DECLARED per setting (:attr:`Setting.path`) instead of derived from the key,
and why :func:`flat_dotted_keys` exists for the round-trip test to assert
against.

NO TEXTUAL IMPORT. The CLI's ``config edit``/``config list`` consult this
registry (a dotted key used to be rejected outright by the validator even
though the app itself instructs users to type one), and the unit tests import
it without a terminal. Keep it dependency-light: importing this module must
never drag in the TUI.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import json
import logging
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

from local_operator import keymap as _keymap
from local_operator.i18n import catalogues as _catalogues
from local_operator.i18n import runtime as _i18n_runtime
from local_operator.i18n.keys import wire_settings as _ws
from local_operator.i18n.messages import Msg as _Msg
from local_operator.model.effort import EFFORT_ORDER, SUPPORTED_EFFORTS
from local_operator.providers.local import (
    DEFAULT_MODEL_OVERRIDES,
    LOCAL_PRESETS,
    model_overrides,
    validate_endpoint_setting,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from local_operator.config import ConfigManager


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry copy: rendered through the i18n runtime
# ---------------------------------------------------------------------------
#
# The registry's user-facing strings live in the `wire.settings` catalogue
# (RFC §2.2); the module renders them at import through the local runtime.
# Resolution is PINNED to `en`, and that is not a shortcut: the shipped locale
# set is en-only while the program ships dark (RFC §9 P1), so a full resolution
# and this pin agree byte-for-byte today. When the translation waves land, the
# re-resolution point is this helper (and the clients, which render the served
# catalogue by key per §2.6).
#
# ONE catalogue read at import, not one per message: the registry builds ~740
# strings and a read-per-string would put ~740 file parses on the CLI's
# import path. A broken wheel (catalogue missing/invalid) degrades the way
# `messages.render` does — each string resolves to its own message code —
# rather than making `lop --help` impossible.
try:
    _WIRE_SETTINGS_SOURCES: dict[str, str] = _catalogues.load_catalogue("en", "wire.settings")
except Exception:  # pragma: no cover - a wheel without the catalogue is broken
    _WIRE_SETTINGS_SOURCES = {}


def _t(message: _Msg) -> str:
    """The resolved string for a generated `wire.settings` message."""
    source = _WIRE_SETTINGS_SOURCES.get(message.code)
    if source is None:
        return message.code
    return _i18n_runtime.render_message(source, dict(message.params), "en")



class ConfigUnreadableError(Exception):
    """config.yml cannot be parsed, so no write may be based on it.

    Its own type rather than ``ValueError`` because the two mean opposite
    things to a caller: a ``ValueError`` from this module is the SCHEMA
    rejecting a value the user typed, which they fix by typing something else,
    whereas this says the file underneath is broken and nothing the user types
    into the page can be safely stored until it is repaired. Callers that print
    a validation message next to an open editor must not print this one there.
    """


class Kind(enum.Enum):
    """How a setting is EDITED, which is what the page renders a row from.

    Deliberately about the interaction and not about the Python type: ``INT``
    and ``FLOAT`` are both "type a number" to a user but validate differently,
    and ``ENUM`` is "expand a list and pick", which is a different widget from
    ``TEXT`` even when the stored value is also a string.
    """

    BOOL = "bool"
    ENUM = "enum"
    INT = "int"
    FLOAT = "float"
    TEXT = "text"
    #: A comma-separated ordered list of enum members (``web_search.providers``).
    #: Edited as text because ORDER is load-bearing there — the ``ordered``
    #: strategy runs the list top to bottom — and a set of checkboxes cannot
    #: express order without inventing a second reorder affordance.
    LIST = "list"
    #: The failover cascade. Not editable as a scalar at all; the page routes
    #: this row to the two-level chain editor.
    CASCADE = "cascade"
    #: A remappable hotkey (``keymap.*``). Its own kind because the
    #: interaction is the OPPOSITE of ``TEXT``: a text editor buffers every
    #: printable key, and ``ctrl+n`` is not printable, so typing a chord into
    #: one is impossible — it falls through to the page's own bindings and
    #: moves the cursor. A hotkey row has to LISTEN instead, so the page runs
    #: a capture mode on it. Stored as a Textual key string; validated by
    #: :mod:`local_operator.keymap`, never by Textual, which accepts anything
    #: and silently makes the action unreachable.
    HOTKEY = "hotkey"
    #: Shown, never written. Retired keys stay visible so a user who set one
    #: years ago can see that it is inert rather than wondering why it does
    #: nothing.
    READONLY = "readonly"


class Scope(enum.Enum):
    """WHEN a change takes effect — the question immediate-write raises.

    A page that writes on Enter owes the user this answer, because the write
    landing and the behaviour changing are not the same moment for most of
    these keys. Rendered as a dim tag on the SECTION header rather than per
    row: ~50 per-row tags is noise, and scope is uniform within a section by
    construction (a section whose members disagree is a section that should be
    split).
    """

    #: Takes effect immediately in every running session on this machine —
    #: on the same call stack in the process that wrote it, and within
    #: ``ConfigWatcher.POLL_INTERVAL_S`` for sessions in other processes (see
    #: :mod:`local_operator.config_watch`) — SUBJECT to the per-key caveats on
    #: the SECTION description, which is where "applied" is spelled out for a
    #: key whose live half is conditional: ``model`` (a session that chose with
    #: ``/model`` keeps its choice), ``web_tools`` (the inventory catches up at
    #: the next turn while execution refuses at once), and ``approvals`` (a
    #: LOOSENING reaches only the session whose own process wrote it, while a
    #: tightening is unconditional — see the section description and
    #: :func:`local_operator.harness.approval.loosening_is_authorised`). The
    #: scope says WHEN, the description says what "applied" means for that key.
    LIVE = "live"
    #: Read when a session is built — a ``/new`` or ``/reload`` picks it up.
    #: Only ``local_providers`` carries it now: ``approvals``, ``model`` and
    #: ``web_tools`` went LIVE, and ``session`` (autosave + cleanup) turned out
    #: to be launch-time all along.
    NEW_SESSIONS = "new sessions"
    #: Read once at process start; needs a relaunch.
    NEW_LAUNCH = "new launch"


@dataclasses.dataclass(frozen=True)
class Choice:
    """One member of an :attr:`Kind.ENUM` setting's value space."""

    value: Any
    label: str
    description: str = ""


@dataclasses.dataclass(frozen=True)
class Setting:
    """One editable configuration value.

    ``path`` is the authority, not ``key``. ``key`` is the dotted name a user
    types (``lop config edit display.terminal_title false``) and the page
    displays; ``path`` is where the value actually lives inside ``values``. For
    ``display.*`` those differ in the way that matters: the key is dotted and
    the path is a ONE-element tuple holding that same dotted string, because
    the dot is part of the literal top-level key rather than a level of
    nesting. See the module docstring.
    """

    key: str
    path: tuple[str, ...]
    section: str
    label: str
    kind: Kind
    default: Any
    help: str
    choices: tuple[Choice, ...] = ()
    #: Resolves the choices at CALL time instead of declaring them here, for a
    #: value space that is a registry rather than a fixed list (``tui.theme``:
    #: the themes are ~30 and a palette module adds more). Declaring them
    #: statically would mean this file re-listing another module's registry,
    #: which is the drift the anti-drift test exists to stop. The indirection
    #: is a callable rather than an import so the import stays LAZY — the
    #: registry lives under ``local_operator.tui`` and this module is imported
    #: by the CLI, which must not pay for the TUI (see the module docstring).
    choices_source: Callable[[], tuple[Choice, ...]] | None = None
    #: Inclusive bounds for INT/FLOAT. ``None`` on either side means unbounded.
    minimum: float | None = None
    maximum: float | None = None
    #: What an INT/FLOAT value COUNTS, as a plain lowercase noun the UI can put
    #: after the number ("hours", "days", "bytes"); empty for a bare count or a
    #: non-numeric row. Together with ``minimum``/``maximum`` it is the whole
    #: contract a bounded-duration control needs, and it is authored here
    #: because a renderer cannot infer "this INT is hours" from its key. The
    #: stored value is ALWAYS in this unit — a UI that shows "2 days" must still
    #: write 48 — so the unit never changes what a writer sends.
    unit: str = ""
    #: Members a LIST setting may contain, in the order they are offered.
    #: A CLOSED allow-list: :func:`validate`/:func:`coerce` reject anything
    #: else, which is right only where this repo owns the vocabulary
    #: (``web_search.providers``). For a LIST over an OPEN, upstream-owned
    #: namespace (the OpenRouter host slugs), leave ``members`` empty — any
    #: non-empty token validates — and seed the editor with :attr:`placeholder`
    #: instead, so common values are visible without becoming a reject list.
    members: tuple[str, ...] = ()
    #: Ghost text painted in the EMPTY inline editor (faint, never part of the
    #: buffer, never committed). Two jobs, both about the empty state being
    #: the default the user must not disturb: show the vocabulary an open LIST
    #: accepts (``deepseek, groq, …``) and show the SHAPE a structured TEXT
    #: field expects (the ``max_price`` JSON example). Plain prose help lives
    #: in :attr:`help`; this is the one-line answer to "what would I type?".
    placeholder: str = ""
    #: A HARD consequence clause, painted in the page's danger ink AHEAD of
    #: :attr:`help` and never shed from the detail line — including when the
    #: row is off-default and the state clause (``default: …``) competes for
    #: the row. For the one setting whose stored value is itself the hazard
    #: (``providers.openrouter.order`` disables sticky routing): the warning
    #: must be on screen exactly when the dangerous value is stored, which is
    #: the state the ordinary shed ladder sacrifices help for. SOFT notes stay
    #: in ``help`` with the faint ink, so the two ranks cannot be confused.
    warning: str = ""
    #: An empty text field CLEARS the key rather than storing "". Off for
    #: settings whose empty string is a real value (``searxng_endpoint``
    #: unset IS ""), on where empty means "no opinion" (``hosting``).
    empty_unsets: bool = False
    #: The browsing cursor APPLIES this value to the running app without
    #: storing it, so a user can see a choice before accepting it (#440 §3).
    #:
    #: Opt-IN rather than inferred from the kind, because previewing is not a
    #: property of having choices — it is a claim that the value has a live
    #: apply path AND a reliable revert, and every target is another revert
    #: route that has to be correct on every exit from the expansion. Only
    #: ``tui.theme`` sets it: it already repaints live through
    #: ``OperatorApp._apply_theme``, and the page captures the applied value on
    #: open and restores it on every cancel route. ``display.*`` is the other
    #: candidate and is deliberately left off until that revert is earned.
    #:
    #: Consumed by the settings page only. Nothing in this module previews
    #: anything — a preview must never reach a writer, which is exactly why the
    #: flag lives beside the write facade rather than inside it.
    preview: bool = False
    #: Structured text settings validate before any persistence; the UI retains
    #: the rejected input so a typo cannot silently disable a working endpoint.
    validate_value: Callable[[Any], object] | None = None
    #: The key of a BOOL master switch this setting is INERT without. The
    #: settings page paints the value dim while the master is off (the same
    #: ink as a READONLY row) and the row's detail says so, so a leftover
    #: ``max_sessions: 200`` under a switched-off ``session.cleanup`` cannot
    #: read as a cap in force (design round 1, D1). Presentational only: the
    #: consumer of the gated key is responsible for honouring the master.
    gated_by: str | None = None
    #: The SCOPE of a HOTKEY row's value, DERIVED from ``keymap.KEY_ACTIONS``
    #: (``"app"`` / ``"desktop"``): it selects the value grammar every write
    #: path applies and is projected on the wire so the desktop app can render
    #: a desktop row with the rules that own it. Never authored twice — an
    #: empty string means "not a hotkey row".
    hotkey_scope: str = ""

    @property
    def resolved_choices(self) -> tuple[Choice, ...]:
        """The choice list, with :attr:`choices_source` resolved if it is set.

        Every consumer of an ENUM's value space must read THIS rather than
        :attr:`choices`, or a dynamically-sourced setting reads as having no
        choices at all: :func:`validate` would reject every value and the page
        would expand an empty list.
        """
        if self.choices_source is not None:
            return self.choices_source()
        return self.choices

    @property
    def is_flat_dotted(self) -> bool:
        """True when the dot in :attr:`key` is literal, not a nesting level.

        The one-line statement of the trap this module exists to avoid. A
        setting whose key contains a dot but whose path is a single element is
        stored under that dotted string verbatim.
        """
        return len(self.path) == 1 and "." in self.path[0]


@dataclasses.dataclass(frozen=True)
class Section:
    """A group of settings shown under one header."""

    name: str
    title: str
    scope: Scope
    description: str = ""


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------
#
# Ordering is the page's reading order, chosen so the settings a user is most
# likely to have come for (which model, does it fail over) are first and the
# retired keys are last.

SECTIONS: tuple[Section, ...] = (
    # Defaults are sampled at conversation birth. The watcher still announces
    # edits, but only an explicit local /model action may select for an owner.
    Section(
        "model",
        _t(_ws.model_title()),
        Scope.NEW_SESSIONS,
        _t(_ws.model_description()),
    ),
    # Split out of ``model`` (review round 1, M3). The design left this key in
    # ``model`` and proposed documenting the discrepancy, which was defensible
    # while nothing read the scope aloud — but the config-change notice now
    # says "takes effect on /new" for every non-LIVE key, and for this one that
    # is FALSE: ``Session._apply_config_change`` rebinds the stream fn on it and
    # ``configure._openai_api_mode`` reads the rebound mapping when it builds
    # the next client. Scope is uniform within a section by construction, so
    # saying something true here means a section of its own, exactly as ``fork``
    # and ``web_tools`` are. Transport policy stays live while the model
    # identity belongs to its conversation, so these must remain separate.
    Section(
        "providers",
        # Titled for the WIRE FORMAT, not the word "provider" (design review
        # round 1, D4): the pane one column to the right is headed `providers`
        # and lists the user's credentials, so two adjacent things called
        # "provider" meant two entirely different concepts. This also makes the
        # header agree with its rows instead of colliding with the pane.
        #
        # NOT the designer's other suggestion, "OpenAI API surface": that was
        # right when this section held one row, but M6 moved the Anthropic
        # cache-TTL key in beside it, so an OpenAI-specific title would now
        # mislabel half the section.
        _t(_ws.providers_title()),
        Scope.LIVE,
        _t(_ws.providers_description()),
    ),
    # The OpenRouter chat-completions ``provider`` routing object has its own
    # section rather than living under "Wire protocol" (design round 1, D1):
    # thirteen rows grafted onto the three protocol knobs made two different
    # jobs share one header, and every label carrying an "OpenRouter " prefix
    # to disambiguate itself from the OpenAI/Anthropic neighbours hid the
    # distinguishing words past the 29-cell label budget. A section of its
    # own names the job once in the header, so the rows can drop the prefix.
    # LIVE for the same reason as ``providers``: ``SessionStreamFn`` resolves
    # the object from the REBOUND settings mapping every time it builds a
    # client (``model.configure._openrouter_provider_preferences``), so an
    # edit lands on the next call — no ``/new``, no relaunch.
    Section(
        "openrouter",
        _t(_ws.openrouter_title()),
        Scope.LIVE,
        _t(_ws.openrouter_description()),
    ),
    # LIVE: every ``retry.*`` key routes through ``RetrySettings.from_settings``
    # PER CALL on the mapping ``SessionStreamFn`` holds, and the config watcher
    # rebinds that mapping on every change (``SessionStreamFn.apply_settings``).
    Section(
        "failover",
        _t(_ws.failover_title()),
        Scope.LIVE,
        _t(_ws.failover_description()),
    ),
    Section(
        "appearance",
        _t(_ws.appearance_title()),
        Scope.LIVE,
        _t(_ws.appearance_description()),
    ),
    # Its own section rather than a pair of rows under ``appearance``. Scope
    # does not force the split — ``appearance`` is LIVE too — but the section
    # DESCRIPTION is where this page teaches the capture gesture, and
    # ``appearance``'s is about theme and terminal features. "How do I even
    # use this row?" is worth a header, the same reason ``runtime`` and
    # ``fork`` split out rather than accept a description that would be a
    # distraction on half their rows.
    Section(
        "keymap",
        _t(_ws.keymap_title()),
        Scope.LIVE,
        _t(_ws.keymap_description()),
    ),
    # LIVE — the SCOPE is right (a config write reaches every running session
    # within a poll) and the description below carries the key's one caveat, so
    # it is not a lie on a section header. Two rules keep the live half safe:
    # the new mode applies at the next approval DECISION (``ServingSessionHandle``
    # reads its flag per gate call), so a prompt already parked on screen is left
    # for the human — never auto-answered, never auto-denied; and a LOOSENING is
    # only honoured when the process holding the gate made the write itself
    # through this facade (issue #1282, ``harness.approval
    # .loosening_is_authorised``) — a model-run shell command rewriting
    # ``config.yml``, an editor, another pane, the settings API elsewhere, and
    # ``lop config edit`` may all TIGHTEN a running session and none may loosen
    # one. ``--yolo`` is an explicit pin that outranks the key. Its own section
    # because ``session`` (autosave + cleanup) is launch-time and scope is
    # uniform per section.
    #
    # THE DESCRIPTION BELOW IS THE ONLY APPROVALS CAVEAT THIS REGISTRY CARRIES,
    # and it has exactly ONE consumer: the desktop settings header
    # (``server/routes/settings.py`` -> the section header component). The TUI
    # ``/settings`` page paints the section TITLE and its scope tag, then the
    # ROW and the row HELP (``Tool approval mode`` / ``tool_approval_mode``'s
    # own ``help``) — never this sentence — so the row help has to stand on its
    # own and say the same thing in its own cell budget (58 cells of the 74-cell
    # detail line, which must still fit its ``· default: ask`` clause; the
    # measurement is at that row. Design round 1, D1; agent review round 1,
    # m1). Keep the two in step: both must be TRUE for the embedded
    # pane (whose own page write IS the gate-holding process's write, and so
    # does loosen) and for the attached one (where only ``/approvals auto`` in
    # the session loosens).
    Section(
        "approvals",
        _t(_ws.approvals_title()),
        Scope.LIVE,
        _t(_ws.approvals_description()),
    ),
    # NEW_LAUNCH, honestly: ``auto_save_conversation`` is read ONCE by the CLI
    # at process start (``cli.py`` sets ``args.train``) to pick the transcript
    # DIRECTORY, and a transcript cannot move mid-session; the TUI's runtime
    # child never reads it at all. ``session.cleanup.*`` is consumed by the
    # once-per-process store-maintenance pass (``session_factory
    # ._STORE_MAINTENANCE_TASK``), so a ``/new`` does not re-run it either.
    # The former "new sessions" label promised a `/new` would adopt these,
    # which nothing did.
    Section(
        "session",
        _t(_ws.session_title()),
        Scope.NEW_LAUNCH,
        # The scope tag one column away already says "takes effect: new
        # launch", so restating "read once at launch" here said "launch" twice
        # within one row (design round 1, D7).
        _t(_ws.session_description()),
    ),
    # The two cleanup CLASSES, each its own section so each gets its own header
    # on every surface (the TUI paints the title; the desktop paints the title
    # and the description) and its own plain-language copy. They are split out
    # of ``session`` rather than left under "Session storage" because they are
    # different promises: the first is the user's own conversations and is OFF
    # until they ask; the second is machine-made bookkeeping and is ON with a
    # long, bounded window. Scope is NEW_LAUNCH for both, honestly: the policy
    # is read by the store-maintenance pass at launch and by its hourly sweep
    # in a long-lived runtime (``session_factory._store_maintenance_thread_main``),
    # and never by a ``/new``.
    Section(
        "session_cleanup",
        _t(_ws.session_cleanup_title()),
        Scope.NEW_LAUNCH,
        _t(_ws.session_cleanup_description()),
    ),
    Section(
        "session_delegated",
        # NOT "Delegated work: subagents and background sessions": at 49 cells
        # the title overran the page's 34-cell value column and the header
        # silently lost its "takes effect: new launch" tag (found in the
        # rendered frame). The longer phrase lives in the description (desktop)
        # and in the rows' help (TUI paints no section description).
        _t(_ws.session_delegated_title()),
        Scope.NEW_LAUNCH,
        _t(_ws.session_delegated_description()),
    ),
    # LIVE: ``max_running`` is pushed into the running ``AsyncJobManager`` by
    # ``Session._apply_config_change`` (raising it lets the next launch through;
    # lowering it lets running jobs finish — nothing is evicted), the
    # ``models.*`` tiers are read at every spawn, and ``model_choice`` is read
    # both by the rebuild that re-renders the two tool schemas and by the
    # tool-argument refusal itself.
    Section(
        "subagents",
        _t(_ws.subagents_title()),
        Scope.LIVE,
        _t(_ws.subagents_description()),
    ),
    # NEW_SESSIONS, and the scope is a statement about the CONSUMER: the
    # classification service is built per session (``ClassificationService`` is
    # constructed by ``session_factory._attach_classification`` and caches the
    # vendor it resolved, its circuit breaker and its LRU for that session's
    # life), and it is handed a SNAPSHOT of this section rather than the live
    # mapping the stream fn holds. An edit therefore lands on the next session,
    # which is what this tag tells the user — claiming LIVE here would be a
    # painted lie of the kind the ``effort.*`` note in
    # ``Session._apply_config_change`` records.
    Section(
        "classification",
        # THE NAME IS THE ONE THE TIPS USE, and it is the title rather than the
        # description because the TUI paints the title on every settings frame
        # while the description below reaches only the settings API. Round 1
        # added the word to a row's help line and round 2 measured what that
        # bought: the help paints on the SELECTED row's detail line, clipped
        # from 78 columns down, so at rest the page said "Resource
        # recommendations" twice and never the word a tip sent the user to look
        # for (design round 2, D11). The runtime used to have its own message as
        # well ("Suggestion added…"); that line is deleted — the layer is silent
        # in the chat — so the title is now the only name on screen.
        _t(_ws.classification_title()),
        Scope.NEW_SESSIONS,
        _t(_ws.classification_description()),
    ),
    Section(
        "supplements",
        # The user-facing name is "Highlights" (memo §0 "The names").
        _t(_ws.supplements_title()),
        # NEW_SESSIONS: the runner is built once per runtime handle and reads a SNAPSHOT
        # of this section AT BUILD (docs/design/turn-supplements.md §2.12; the read is in
        # ``SupplementRunner.__init__``), so an edit lands on the next session. Claiming
        # LIVE would be a painted lie.
        Scope.NEW_SESSIONS,
        _t(_ws.supplements_description()),
    ),
    Section(
        "monitor",
        _t(_ws.monitor_title()),
        # NEW_SESSIONS for the classification section's reason: the monitor
        # scheduler is built per session and handed a SNAPSHOT of this section
        # (docs/design/monitor-tool.md §16), so an edit lands on the next
        # session start. Claiming LIVE would be a painted lie.
        Scope.NEW_SESSIONS,
        _t(_ws.monitor_description()),
    ),
    # Its own section rather than a row under "Session", and the reason is the
    # SCOPE: scope is uniform within a section by construction, "Session" is
    # launch-time (autosave and the cleanup policy), and these keys take
    # effect on the very next /resume in this same terminal. Filing a live key
    # under a section labelled for launch is exactly the painted lie AGENTS.md
    # warns about — split the section.
    Section(
        "runtime",
        _t(_ws.runtime_title()),
        Scope.LIVE,
        # WHERE THE CAVEAT ON THE TWO WARM-RUNTIME KEYS BELONGS (review round 1,
        # F6): the Scope enum is a statement about the KEYS, and a per-key
        # exception stated only in a code comment is a claim the user cannot
        # read. Both are read when a drain window is drawn, so an edit is
        # picked up by the next window — a runtime already inside one keeps
        # the window it drew.
        _t(_ws.runtime_description()),
    ),
    # LIVE: the session re-coerces its ``CompactionSettings`` on every change,
    # and all three trigger checks read that attribute at check time.
    Section(
        "compaction",
        _t(_ws.compaction_title()),
        Scope.LIVE,
        _t(_ws.compaction_description()),
    ),
    # LIVE: ``/fork`` reads these through the config manager at the moment it
    # runs, so an edit takes effect on the very next fork.
    Section(
        "fork",
        _t(_ws.fork_title()),
        Scope.LIVE,
        _t(_ws.fork_description()),
    ),
    # The GATE comes first, then the knobs it gates (design review round 1,
    # D3): reading order is the hierarchy the user sees, and putting the
    # master switches after four tuning knobs left someone scanning for "is
    # web search on?" finding the answer next to the retired-keys graveyard.
    # LIVE by two mechanisms that together make "applied" true: execution
    # re-checks ``enabled`` on EVERY call (``execute_web_search`` /
    # ``run_fetch`` — which also covers the ``read <url>`` sugar), so a
    # disable refuses at once even mid-turn; and a top-level session
    # reconciles its advertised inventory at the next TURN boundary (never
    # mid-turn — a call the model already emitted against a just-removed tool
    # would come back "Tool not found"). Subagents keep their spawn inventory
    # and rely on the per-call gate alone.
    Section(
        "web_tools",
        _t(_ws.web_tools_title()),
        Scope.LIVE,
        _t(_ws.web_tools_description()),
    ),
    # LIVE: both tools build their settings from config on EVERY call
    # (``web_search/tool.py``, ``web_fetch/tool.py``).
    Section(
        "web_search",
        _t(_ws.web_search_title()),
        Scope.LIVE,
        _t(_ws.web_search_description()),
    ),
    Section(
        "web_fetch",
        _t(_ws.web_fetch_title()),
        Scope.LIVE,
        _t(_ws.web_fetch_description()),
    ),
    # LIVE for the same reason as the two web sections: ``execute_bash`` reads
    # ``bash.shell`` through a fresh ``ConfigManager(config_dir())`` on EVERY
    # call (``tools/builtin.py::_configured_bash_shell``), so an edit lands on
    # the very next command. Its own section rather than a row under "Session"
    # or "Runtime": scope is uniform within a section by construction, and
    # this is about how a tool executes, not how sessions behave.
    Section(
        "tools",
        _t(_ws.tools_title()),
        Scope.LIVE,
        _t(_ws.tools_description()),
    ),
    # LIVE: ``hook_forwarding.native_hooks_enabled`` / ``forwarding_enabled``
    # read these through a fresh ``ConfigManager(config_dir())`` on every
    # finished tool call.
    Section(
        "hooks",
        _t(_ws.hooks_title()),
        Scope.LIVE,
        _t(_ws.hooks_description()),
    ),
    # LIVE, and its own section, for the reason ``tools`` is LIVE: ``execute_bash``
    # reads these through a fresh ``ConfigManager(config_dir())`` per call, so an
    # edit lands on the next command. Deliberately NOT in ``shell_environment``,
    # whose members are NEW_LAUNCH because the agent's own shell can lower them:
    # the memory ceiling is a resource policy the OPERATOR owns, and a loosening
    # by the agent is no privilege escalation — the kill only ever stops the
    # agent's own command, never the runtime.
    Section(
        "memory_guard",
        _t(_ws.memory_guard_title()),
        Scope.LIVE,
        _t(_ws.memory_guard_description()),
    ),
    # Its own section for the reason ``memory_guard`` has one, and LIVE for the
    # same reason: the reader is a fresh ``ConfigManager`` per ``bash`` call. It is
    # NOT in ``memory_guard`` because scope is uniform within a section and these
    # are a different policy — a memory ceiling protects the DEVICE, a query
    # budget protects the TURN, and the two have different failure stories (the
    # one kills a command that was too big; the other kills one that was merely
    # too wide).
    Section(
        "query_budget",
        _t(_ws.query_budget_title()),
        Scope.LIVE,
        _t(_ws.query_budget_description()),
    ),
    # Split out of ``tools`` (review round 1, M2), for the reason the module's
    # own history gives for ``providers`` and ``web_tools``: scope is uniform
    # within a section by construction, and these three keys are the one part of
    # how a tool executes that must NOT re-read per call — a live policy is a
    # policy the agent's own shell can weaken between two commands.
    Section(
        "shell_environment",
        _t(_ws.shell_environment_title()),
        Scope.NEW_LAUNCH,
        _t(_ws.shell_environment_description()),
    ),
    Section(
        "local_providers",
        _t(_ws.local_providers_title()),
        Scope.NEW_SESSIONS,
        _t(_ws.local_providers_description()),
    ),
    # LIVE: the click handler is a FRESH PROCESS every time it runs
    # (``lop resume-click`` is spawned by the notification), so it reads config
    # at click time and an edit lands on the very next click. Nothing is
    # threaded through a running session, which is why this cannot be
    # NEW_SESSIONS like the knobs that gate a session's construction.
    Section(
        "desktop",
        _t(_ws.desktop_title()),
        Scope.LIVE,
        _t(_ws.desktop_description()),
    ),
    # The static file routes' served roots (``server/utils/static_roots.py``).
    # LIVE: the roots are rebuilt from the config on every request, so an edit
    # lands on the next thumbnail or preview with no restart.
    Section(
        "static",
        _t(_ws.static_title()),
        Scope.LIVE,
        _t(_ws.static_description()),
    ),
    # NEW_LAUNCH, honestly: the audit keys are read when the audit WRITER is built,
    # and the writer is built once per relay process (``AuditLog.from_config``);
    # ``max_handshakes`` is read when ``NetworkSettings.from_config`` builds the
    # relay's settings. A LIVE label would promise an effect the code cannot deliver
    # — the open file handle already exists, and the accept loop's cap is a
    # constructor argument. The repo's rule is that a scope label says when a change
    # lands, not when the user would like it to.
    Section(
        "network",
        _t(_ws.network_title()),
        Scope.NEW_LAUNCH,
        _t(_ws.network_description()),
    ),
    # The Aida keys. Scope LIVE, with the caveats stated in the description
    # rather than hidden behind a dishonest label: the pause flag is delivered
    # to every live process by the config watcher on its 2 s tick (that IS the
    # mechanism that reaches a session running in another terminal), and the
    # others are read at her next action — a boot, `/aida`, or the next fire —
    # which for an always-on assistant is the same thing in practice. The one
    # sentence a user needs before pressing Enter is written there.
    Section(
        "aida",
        _t(_ws.aida_title()),
        Scope.LIVE,
        _t(_ws.aida_description()),
    ),
    # The hub keys. LIVE is honest here: the update runner re-reads the mapping
    # on every tick (and every interval sleep), so an edit lands within one tick.
    # Both auto-update switches default ON; turning one OFF keeps the runner
    # CHECKING (the sidebar still shows "update available") and only stops it
    # from merging, so "manual" never means "blind".
    Section(
        "hub",
        _t(_ws.hub_title()),
        Scope.LIVE,
        _t(_ws.hub_description()),
    ),
    # The agents key. NEW_LAUNCH, honestly: its one consumer is the startup
    # seed-update pass, which reads it under the config-migration seam at each
    # ``lop`` process start — an edit lands on the next launch and nothing
    # re-reads it mid-session. Turning it OFF keeps REPORTING (drift is still
    # classified and noticed; only the unattended write stops) — the same
    # split the hub's auto-update switches use.
    Section(
        "agents",
        _t(_ws.agents_title()),
        Scope.NEW_LAUNCH,
        _t(_ws.agents_description()),
    ),
    # The projects store's own knobs. LIVE because the staleness window is
    # resolved at each staleness computation (the badge, the tool rows, the
    # completion check and the wake-trigger snapshot all read through
    # ``projects.stale_after_s``), so an edit lands on the next computation —
    # for a running session that is its next listing or turn-end.
    Section(
        "projects",
        _t(_ws.projects_title()),
        Scope.LIVE,
        _t(_ws.projects_description()),
    ),
    # The generic wake-trigger mechanism (``local_operator/wakes/triggers/``).
    # LIVE: the evaluation pass re-reads the published snapshot every ~5
    # minutes while the supervisor is up, and the budget/gap are read at the
    # next evaluation; nothing here needs a relaunch or a /new.
    Section(
        "wakes",
        _t(_ws.wakes_title()),
        Scope.LIVE,
        _t(_ws.wakes_description()),
    ),
    # The proactive CLASS's own bounds (R29–R38). Scope LIVE is the honest
    # label here for once: every key is read at the moment a patience wait
    # fires or re-arms — a delivery-time read by construction — so an edit
    # lands on the next wait even in a running session.
    Section(
        "proactive",
        _t(_ws.proactive_title()),
        Scope.LIVE,
        _t(_ws.proactive_description()),
    ),
    # The speak-aloud voicing dials (design note §1). One section, LIVE: the
    # descriptor is built per request, so an edit changes the next spoken
    # message and nothing needs a relaunch or a /new. It is ONE section rather
    # than keys scattered into ``model``/``providers`` because they are one
    # object — the descriptor — and a user tuning how the assistant sounds is
    # doing one thing.
    Section(
        "speech",
        _t(_ws.speech_title()),
        Scope.LIVE,
        _t(_ws.speech_description()),
    ),
    Section(
        "retired",
        _t(_ws.retired_title()),
        Scope.NEW_LAUNCH,
        _t(_ws.retired_description()),
    ),
)


def _validate_openrouter_max_price(value: Any) -> None:
    """Accept a JSON object string or mapping of optional price caps.

    Runs before persistence (see :attr:`Setting.validate_value`): the TUI
    editor types a string, the desktop ``PATCH /v1/settings/{key}`` route and a
    hand-written config.yml hand a parsed mapping, and both must be checked or
    a typo would sail into the request body as-is. Stored verbatim either way;
    :func:`local_operator.model.configure._openrouter_provider_preferences`
    parses the string form at client build time.
    """
    if isinstance(value, str):
        if not value.strip():
            return  # "" is "no cap"; `empty_unsets` clears the key on write.
        try:
            value = json.loads(value)
        except ValueError:
            raise ValueError('expected a JSON object like {"prompt": 1, "completion": 2}') from None
    if not isinstance(value, Mapping):
        raise ValueError('expected a JSON object like {"prompt": 1, "completion": 2}')
    for name, cap in value.items():
        if name not in ("prompt", "completion", "request", "image"):
            raise ValueError(f"unknown price field: {name}")
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap < 0:
            raise ValueError(f"{name} must be a non-negative number")


#: What separates a rejection's VALUE from its ADVICE.
#:
#: THE PAGE CAN ONLY SPEND ONE LINE ON A REJECTION and this widget's contract is
#: that the line CLIPS WITHOUT A MARK, so a message whose length is the USER'S
#: OWN INPUT has to be splittable "value first, advice after" — the value can be
#: shed (it is what the user typed, and it is still in the row's value column)
#: while the advice is pinned, then cut with a visible ellipsis. A message that
#: uses this separator OPTS IN to that ladder, and the opt-in is checked by
#: ASKING THE PRODUCER (:func:`split_value_rejection`) rather than by reading the
#: head's token count — so prose notices ("config.yml is unreadable, nothing was
#: written — …") are never shed while a QUOTED launcher path containing a space
#: is (QA round 1, Q1; design round 2, D16).
REJECTION_VALUE_SEP = " — "


#: The ADVICE half of an interpolated-value rejection — the copy
#: :func:`_validate_desktop_launch_command` pins after the value.
#:
#: NAMED BECAUSE THE RENDERER HAS TO RECOGNISE THE SHAPE INSTEAD OF GUESSING IT.
#: The page sheds the value half and keeps this one, and it used to decide which
#: half was which from the head's TOKEN COUNT — the wrong question, because a
#: QUOTED launcher path containing a space is ONE token to the shell grammar the
#: validator uses and SEVERAL to ``str.split``. For that shape the ladder
#: degraded to a plain clip and cut the fault, the consequence and the remedy
#: together at 80x24 (QA round 1, Q1). Writing the copy once, here, is what lets
#: the producer and :func:`split_value_rejection` agree by construction.
ADVICE_NOT_FOUND = "does not exist; clicks open a terminal. Clear this to discover the app."
ADVICE_NOT_EXECUTABLE = "not executable; clicks open a terminal. Clear this to discover the app."
ADVICE_NOT_ON_PATH = "not on PATH; clicks open a terminal. Clear this to discover the app."

#: Whether this process is on Windows, read ONCE as a module constant.
#:
#: Both places below ask a Windows question — which shell the bash row promises
#: and what "executable" means for a click launcher — and neither can be asked
#: with an inline ``os.name`` read a test can flip: patching ``os.name``
#: process-wide makes ``pathlib`` build a ``WindowsPath`` and refuse on the host
#: running the test (see :mod:`local_operator.procstate`, which holds its
#: platform the same way). One name, read once, patched in tests.
_IS_WINDOWS = os.name == "nt"
_VALUE_REJECTION_ADVICE: tuple[str, ...] = (
    ADVICE_NOT_FOUND,
    ADVICE_NOT_EXECUTABLE,
    ADVICE_NOT_ON_PATH,
)


def split_value_rejection(message: str) -> tuple[str, str] | None:
    """``(value, advice)`` when ``message`` is one THIS MODULE shaped that way.

    The renderer's question is always "which half of this message is the user's
    own input?" — the answer decides what may be shed and what has to be pinned
    — and no amount of reading the rendered string answers it reliably: the
    value is a shlex-parsed token, so it can contain spaces, and the page's
    prose notices (``config.yml is unreadable, nothing was written — …``,
    ``could not save — <path> has an unexpected structure…``) carry the same
    separator in the same place while their HEAD is exactly the part that must
    not be shed.

    So the producer answers instead. Every value-shaped rejection ENDS in one of
    :data:`_VALUE_REJECTION_ADVICE`, which nothing else in the tree writes, and
    matching the TAIL (rather than the first separator) also keeps a value that
    itself contains the separator split at the right place: the value is
    whatever precedes the advice, wherever that lands.

    ``None`` means the message is not one of ours — prose, a nested exception's
    text, a validator that pinned no advice — and the ladder leaves it alone.
    """
    for advice in _VALUE_REJECTION_ADVICE:
        suffix = REJECTION_VALUE_SEP + advice
        if message.endswith(suffix):
            return message[: -len(suffix)], advice
    return None


@functools.cache
def _user_shell_path() -> str | None:
    """The PATH a USER'S OWN SHELL would have, or ``None`` if unreadable.

    WHY THE WRITER'S PATH IS NOT THE QUESTION (agent review round 1, M3). Write
    time validation used to ask ``shutil.which`` alone, i.e. "can THIS process
    resolve the token?" — and that is not what the reader of a click needs. The
    window the validation exists for is a writer whose PATH is not the user's:
    a GUI-launched settings page, ``PATCH /v1/settings`` from a service, or
    ``lop config edit`` under a login-less PATH. Driven, on this machine:

        PATH=/opt/homebrew/bin:/usr/bin:/bin   accepted  'local-operator-ui …'
        PATH=/usr/bin:/bin:/usr/sbin:/sbin     REFUSED  'local-operator-ui …'

    — the second is the literal example in this setting's own help text, and a
    value the CLICK can run. So a bare name that the writer cannot resolve is
    re-checked against the login-shell PATH, which is the PATH the rest of the
    CLI already adopts for subprocess work (``helpers.setup_cross_platform_
    environment``).

    RESIDUAL LIMITATION, recorded rather than papered over: the click runs in
    the NOTIFIER's environment, which write-time code cannot observe, so a bare
    name can still fail at click time. This makes the check agree with the value
    the user could reach by hand instead of with the incidental PATH of whoever
    wrote it, and the refusal tells the user the route that does not depend on a
    PATH at all — clearing the field hands discovery back.

    CACHED, and consulted ONLY on the branch that is about to refuse. The read
    is a login-shell round trip (bounded inside ``helpers``); caching it is what
    keeps a second refusal free, and asking it last is what keeps it off the
    accept path, where a user is waiting on a keystroke.

    THE CACHE IS PER-PROCESS, and that is safe because it is consulted ONLY on
    the refusing side (agent review round 2, M2). A stale PATH can therefore
    produce a FALSE REFUSAL — a directory the login shell gained after this
    process started is not seen — and never a wrong ACCEPT, because an accepted
    name still has to satisfy ``shutil.which``, which re-stats. Nothing that
    refuses is silently allowed through, and the refusal names the way out.
    """
    try:
        import platform

        from local_operator import helpers

        system = platform.system()
        if system == "Windows":
            return helpers.get_windows_registry_path()
        if system in ("Darwin", "Linux"):
            return helpers.get_posix_shell_path()
    except Exception:  # noqa: BLE001 — an unreadable PATH is an ordinary answer
        logger.debug("could not read the user's shell PATH", exc_info=True)
    return None


def _resolvable_for_the_user(executable: str) -> bool:
    """Is ``executable`` on the user's own PATH as well as this writer's?

    A PATH-PREFIXED token (``./launcher``, ``/opt/app/bin/ui``) is not asked —
    ``any PATH`` cannot change where that points, which is why the separator
    branch above checks existence instead. See :func:`_user_shell_path`.
    """
    import shutil

    path = _user_shell_path()
    if not path:
        return False
    return shutil.which(executable, path=path) is not None


def _validate_desktop_launch_command(value: Any) -> None:
    """Reject a click launcher that cannot be run at all, at the moment it is written.

    WRITE-TIME BECAUSE THE CLICK HAS NOWHERE ELSE TO COMPLAIN (UX round 1, U4).
    ``desktop.launch_command`` REPLACES app discovery rather than leading it —
    a configured command is the user's own answer and is not second-guessed by
    a fallback chain — so a typo in it diverts every notification click to a
    terminal instead, indefinitely, and the handler that would have noticed is
    a detached process whose only trace was a ``logger.debug``. The value is
    checked here, where the user is still looking at the field and can be told
    what is wrong. ``tui.resume_click`` warns at click time as well, which is
    what covers a value that reached ``config.yml`` by hand.

    THE CHECK IS "CAN THIS BE RUN", NOT "IS THIS SHAPED NICELY". The first word
    has to resolve to an executable — an existing executable file for a path,
    a name on the writer's PATH or on the user's login PATH otherwise — because
    that is precisely the question ``Popen`` answers at click time with an
    ``OSError``. A launcher that is not installed yet, or whose path has moved,
    is a click that lands in a terminal, so it is refused while it is still in
    front of the user.

    TWO PORTS, NOT ONE (agent review round 1, M3). Asking only the WRITER's PATH
    refused the value in this setting's own help text whenever the writer was
    not a login shell — see :func:`_user_shell_path`, which is consulted only
    when the writer's PATH has already failed and is cached when it is.

    EVERY MESSAGE NAMES THE REMEDY, not just the fault (UX round 2, U12 / design
    round 1, D13). This rejection REPLACES the row's own help while it is on
    screen, so the sentence that answers "what do I type instead" would
    otherwise be exactly the one the error displaces. Each message is therefore
    shaped ``<token> — <fault>; <consequence>. <remedy>``, and the ADVICE half
    is capped at 74 cells — the row's budget at 80x24, the narrowest width the
    page measures — so it is never cut at all (design round 1, D11).

    THAT CAP IS WHY THE REMEDY IS ONE CLAUSE. Fault, consequence and a two-way
    remedy are ~90 cells together: one of the three has to give, and the fix
    direction for D13 itself drops the consequence to afford the remedy. The
    remedy kept is the one a user cannot derive — clearing the field is what
    hands discovery back — and the fault already names the path as the problem,
    which is the other half of what "fix the path" would say.

    EMPTY IS NOT CHECKED: ``""`` is the default and it MEANS "discover the app
    for me", and ``empty_unsets`` clears the key rather than storing it.
    """
    if not isinstance(value, str):
        raise ValueError(
            "expected a command line, e.g. 'local-operator-ui --open-session {session}'"
        )
    if not value.strip():
        return
    import os
    import shlex
    import shutil
    import sys

    try:
        # The same grammar the handler reads it with (`resume_click`), so a
        # value that validates here is one that will parse there: on Windows
        # the command line is not POSIX-quoted and `shlex` would strip the
        # backslashes out of every path.
        parts = shlex.split(value, posix=sys.platform != "win32")
    except ValueError as error:
        raise ValueError(f"not a valid command line ({error})") from None
    if not parts:
        return
    executable = parts[0]
    if os.sep in executable or (os.altsep and os.altsep in executable):
        if not os.path.exists(executable):
            raise ValueError(f"{executable}{REJECTION_VALUE_SEP}{ADVICE_NOT_FOUND}")
        # TWO QUESTIONS, ONE PER PLATFORM, and the POSIX one is unchanged: on
        # Windows "is it executable" has no mode-bit answer at all
        # (:func:`_windows_command_is_runnable`), and the vacuous `X_OK` that
        # used to stand in for it let any existing file through.
        runnable = (
            _windows_command_is_runnable(executable)
            if _IS_WINDOWS
            else os.access(executable, os.X_OK)
        )
        if not runnable:
            raise ValueError(f"{executable}{REJECTION_VALUE_SEP}{ADVICE_NOT_EXECUTABLE}")
        return
    if shutil.which(executable) is None and not _resolvable_for_the_user(executable):
        raise ValueError(f"{executable}{REJECTION_VALUE_SEP}{ADVICE_NOT_ON_PATH}")


def _windows_command_is_runnable(executable: str) -> bool:
    """Whether WINDOWS would actually run ``executable``.

    ``os.access(path, os.X_OK)`` cannot answer this (audit D23). It is a POSIX
    question — "does this file carry the execute bit?" — and on Windows there is
    no such bit: the call is satisfied for any existing file, so the check this
    replaces accepted a ``.txt`` or a ``.ps1`` as a click launcher and the
    failure surfaced later as an unexplained refusal to launch.

    The platform's own answer is the one ``cmd`` uses: a file is runnable when
    its extension is in ``PATHEXT`` (``.COM;.EXE;.BAT;.CMD;…``). A path with NO
    extension is also runnable, because ``CreateProcess`` appends ``.exe`` to a
    name that carries none — so ``C:\\tools\\notepad`` runs ``notepad.exe``.
    PATHEXT is read from the environment rather than hardcoded: it is the user's
    own list, and a host that has added an extension to it means it.
    """
    suffix = os.path.splitext(executable)[1]
    if not suffix:
        return os.path.isfile(executable + ".exe")
    # Split on ";" and not on ``os.pathsep``: the list's separator is a WINDOWS
    # fact (there it happens to equal ``os.pathsep``, which is why the two look
    # interchangeable) and reading it as the host's would make this function
    # answer differently on a machine that is not Windows — including in a test,
    # which is where the difference would otherwise hide.
    extensions = (os.environ.get("PATHEXT") or ".COM;.EXE;.BAT;.CMD").split(";")
    return suffix.upper() in {ext.strip().upper() for ext in extensions if ext.strip()}


def _bool_choices(on: str, off: str) -> tuple[Choice, ...]:
    return (Choice(True, _t(_ws.bool_on()), on), Choice(False, _t(_ws.bool_off()), off))


@functools.cache
def _theme_choices() -> tuple[Choice, ...]:
    """Every registered theme, as ENUM choices (review round 1, m1).

    Read from ``tui.theme``'s registry rather than listed here, because the
    registry is the only place that knows what is installed: the two brand
    ramps plus every curated palette, ~30 today and open to more. A hardcoded
    pair would be wrong the moment a palette is added, which is exactly the
    kind of drift this file's anti-drift test exists to catch.

    Imported function-locally, and an import failure yields an empty tuple
    rather than raising: this module is imported by the CLI, which has no TUI,
    and ``config list`` must still describe the key on a machine where the TUI
    extra is not installed. An empty tuple makes ``validate`` refuse every
    value, which fails CLOSED — refusing to write a theme is recoverable,
    writing one nothing can render is the config-and-behaviour disagreement
    this change is closing.

    CACHED because ``_build_rows`` calls it on every repaint when the theme row
    is expanded, and AGENTS.md forbids unbounded work on a paint path. The
    registry is fixed for the life of the process with ONE exception: the
    ``terminal`` theme is derived from the host terminal at TUI boot
    (``tui.host_theme.install``), which clears this cache after registering it. Without this the
    first call pays a ~95ms import of ``local_operator.tui.theme`` — never on a
    real paint, since the TUI has already imported it, but the bound relied on
    an invariant nothing stated (review round 2, m5).
    """
    try:
        from local_operator.tui.theme import available_themes, theme_spec
    except Exception:  # pragma: no cover - TUI-less install; see the docstring
        return ()
    choices: list[Choice] = []
    for name in available_themes():
        spec = theme_spec(name)
        # `label` is the NAME, not `spec.label`: in this registry the label is
        # the token a user types (`display.nerd_icons` labels its None/True/
        # False choices "auto"/"on"/"off"), and it is what `validate` lists
        # back on a rejection. Offering "Operator Dark" as the answer to
        # "expected one of" would name something `lop config edit` refuses.
        #
        # The description is `spec.description` ALONE. Prefixing it with
        # `spec.label` cost the width that the description needs — the row is
        # one line and the pane truncates — to restate what the name beside it
        # already says ("monokai" -> "Monokai").
        choices.append(Choice(name, name, spec.description))
    return tuple(choices)


def _language_choices() -> tuple[Choice, ...]:
    """The ``language`` row's value space: ``auto`` plus the SHIPPED locales.

    Deliberately NOT a static list: shipment is a property of the translation
    ledger (§6), and the resolver's ``shipped_locales()`` derives it from the
    generated set — so a wave that passes audit widens this picker by
    regenerating data, with no code change here, while an unaudited locale can
    never be selected (RFC §3.1: "unaudited locales are not listed").

    Imported function-locally like ``_theme_choices``'s registry, and read at
    CALL time through ``choices_source``: the value space must reflect the
    shipped set of the process answering, not of import.
    """
    from local_operator.i18n.resolve import language_options

    return tuple(Choice(value, label) for value, label in language_options())


#: One-line meaning for each rung of :data:`EFFORT_ORDER`, for the
#: ``model_effort`` row's expanded list. Every member is a 3-argument
#: :class:`Choice` here (value, label, description), so the descriptions live
#: beside the ladder's own names rather than in the row. A rung with no entry
#: falls back to an EMPTY description rather than raising: this table is built
#: at import, and a KeyError there would take the whole CLI down for a missing
#: sentence. ``tests/unit/test_settings_io.py`` pins that no rung is missing one,
#: which is the loud failure this avoids at runtime.
_EFFORT_LEVEL_HELP: dict[str, str] = {
    # `reasoning off`, not `no reasoning — fastest, cheapest` (review round 1,
    # m2): every other rung's description names a DEPTH, and the two benefits
    # named the one row that costs the least, in a picker where the row directly
    # above it is `auto`. A description has one job here — say what the member
    # means — and the ladder's cheapest end is not the place to sell.
    "none": _t(_ws.model_effort_choice_none_description()),
    "minimal": _t(_ws.model_effort_choice_minimal_description()),
    "low": _t(_ws.model_effort_choice_low_description()),
    "medium": _t(_ws.model_effort_choice_medium_description()),
    "high": _t(_ws.model_effort_choice_high_description()),
    "xhigh": _t(_ws.model_effort_choice_xhigh_description()),
    "max": _t(_ws.model_effort_choice_max_description()),
}


def _effort_choices() -> tuple[Choice, ...]:
    """``model_effort``'s value space: the shared ladder plus the ``auto`` rung.

    ``""`` leads as its own member rather than as an ``empty_unsets`` empty —
    the same shape ``providers.openrouter.sort`` uses — so the page shows
    ``auto`` BESIDE the real rungs as a peer to pick between (a bare empty
    field would not), and the schema knows a stored empty string means "no
    opinion" rather than a missing key.

    The ladder comes from :data:`EFFORT_ORDER`, not a re-listing: the vocabulary
    is the one place a rung is defined, and a second copy here would drift the
    moment a level is added or removed. A rung the chosen model lacks is NOT
    hidden — it is CLAMPED at use (``configure_model``), which is precisely what
    makes offering the full ladder safe: the row needs no model to render, which
    is required on the CLI path (a paint must not resolve a model) and in the
    no-model setup state.
    """
    return (
        Choice("", _t(_ws.model_effort_choice_unset_label()), _t(_ws.model_effort_choice_unset_description())),
        *(Choice(level, level, _EFFORT_LEVEL_HELP.get(level, "")) for level in EFFORT_ORDER),
    )


def _bash_shell_help(windows: bool) -> str:
    """The ``bash.shell`` row's help for a platform, and why it is platform-shaped.

    The sentence NAMES A PATH — "empty uses bash on PATH, else /bin/sh" — so it
    has to name the one this OS falls back to (audit B26). The resolver is
    ``tools.builtin.resolve_bash_shell``: the configured value, else ``bash`` on
    PATH, else that module's last resort. On Windows the last resort is NOT
    ``/bin/sh`` — there is no such file there — it is the ``bash.exe`` of a Git
    for Windows install, and when even that is absent the tool REFUSES the call
    with an install hint (``WINDOWS_NO_BASH_MESSAGE``) rather than executing
    every command in a dialect it does not advertise. Promising ``/bin/sh`` on
    that row would describe an interpreter that cannot exist on the machine the
    operator is reading it on.

    A FUNCTION RATHER THAN ONE INLINE CONDITIONAL, so both spellings are
    reachable by a test on either host: the row itself is built at import, and
    the only other way to exercise the Windows string from macOS would be to
    patch ``os.name`` process-wide. The text is spelled here rather than imported
    from ``tools.builtin`` for the reason the row's ``path`` is pinned by test
    instead of imported: this module must stay cheap for the CLI, and
    ``tools.builtin`` is not (see the module docstring on Textual).

    THE WINDOWS SPELLING IS 72 CELLS, and that is a constraint rather than a
    coincidence (design round 1, D1). This field does not wrap, its shed ladder
    has no rung below "help alone", and the floor cuts the sentence mid-clause
    with a visible ``…`` — so the 163-cell spelling this replaces was clipped at
    `else the bash.exe` at 80 columns AND at `with neit` at 120, i.e. unreadable
    on the platform it was written for at every width the page supports. The
    clause it drops is not information lost: "with neither, the tool refuses and
    says how to install one" is the first line of that module's own refusal
    (``tools.builtin.WINDOWS_NO_BASH_MESSAGE``), which the tool card renders with
    room around it. `Git for Windows` also had to go — the cells are the budget
    here, and the refusal's install hint is where a user reads the product name.
    """
    if windows:
        return _t(_ws.bash_shell_help_windows())
    return _t(_ws.bash_shell_help())


#: This host's spelling, baked into the row below at import.
_BASH_SHELL_HELP = _bash_shell_help(_IS_WINDOWS)


def _validate_github_repositories(value: object) -> object:
    """Every entry is ``owner/repo`` (``.git`` tolerated) — the shape the helper compares.

    The mint splits the owner off and sends the NAME to GitHub; the git helper
    compares the full ``owner/repo`` against this same list. A bare repo name would
    be ambiguous the moment the list spans owners, so the one shape is enforced
    where a person types it — before a half-valid allow-list can reach either side.
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError("GitHub repositories must be a list of owner/repo entries.")
    for entry in value:
        text = str(entry).strip()
        owner, sep, repo = text.partition("/")
        repo = repo.removesuffix(".git")
        if (
            not owner
            or sep != "/"
            or not repo
            or "/" in repo
            or owner in (".", "..")
            or repo in (".", "..")
            or any(ch.isspace() for ch in text)
        ):
            raise ValueError(
                f"not an owner/repo pair: {text!r} — write each designated repository as "
                "owner/repo, for example damianvtran/scratch"
            )
    return value


def _validate_advertise_hosts(value: object) -> object:
    """Every entry is ``host:port`` — the shape this row's help already promises.

    Enforced at the WRITE facade rather than at the relay because this is the only place
    a person types an endpoint, and every writer funnels through :func:`validate` (the
    page, `lop config edit`, the server's `PATCH /v1/settings`). An entry with no port
    (``203.0.113.7``) or an impossible one (``tunnel.example.com:70000``) was accepted,
    persisted into ``listen.advertised``, and then carried into every invite — where it
    reads as an address a joiner can dial and is not, which is the class of lie the rest
    of this change is about (review round 1, M1).
    """
    if not isinstance(value, (list, tuple)):
        raise ValueError("Advertised endpoints must be a list of host:port entries.")
    for entry in value:
        text = str(entry).strip()
        host, _, port = text.rpartition(":")
        if not host or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise ValueError(
                f"not a dialable host:port: {text} — every entry needs a port, "
                "for example tunnel.example.com:4100"
            )
    return value


#: The ``aida.name`` cap — a LITERAL here because this module deliberately
#: stays off the aida package's import path (see the aida block below); the
#: real constant is ``aida.naming.MAX_NAME_CHARS`` and the anti-drift test in
#: ``tests/unit/aida/test_aida_naming.py`` pins the two together the way
#: ``_consumer_defaults()`` pins the defaults. 80 matches the session title's
#: own cap (``session.naming.MAX_TITLE_CHARS``) on purpose: a session rename
#: syncs its title into this key, so a title the session layer accepted must
#: never be refused by the config layer mid-sync.
_AIDA_NAME_MAX_CHARS = 80


def _validate_delegated_max_age_hours(value: Any) -> None:
    """The range sentence for ``session.cleanup.delegated.max_age_hours``.

    The generic INT bounds would say "must be at most 720", which tells a user
    typing ``1`` nothing about WHY 2 is the floor. This names the range and what
    it means in days, once, for every writer (the page, ``lop config edit`` and
    the HTTP route share :func:`validate`). Only a whole number reaches the range
    test: a bool, a float or text keeps the generic type sentence from the INT
    arm, which is the more useful thing to say about them.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return
    if not 2 <= value <= 720:
        raise ValueError(f"max_age_hours must be between 2 and 720 (30 days); got {value}")


def _validate_static_roots(value: Any) -> None:
    """``static.roots`` entries are absolute directories that do not contain ``$HOME``.

    Enforced at the write facade so a typo cannot silently widen the file-serving
    boundary: a relative entry would be resolved against the DAEMON's cwd (the
    reader drops it, so it would be a root that quietly does nothing), and ``/``,
    ``/Users`` or ``~/..`` would turn the allowlist back into "anywhere on disk".
    The ancestor test is the reader's own predicate (``root_refusal``), imported
    here rather than restated so the two cannot disagree -- the reader applies it
    to the environment variable too, which never passes through this facade.
    """
    if not isinstance(value, list):
        return
    # Imported here: the settings registry is imported by every entry point and the
    # server package is not needed until a value is actually being written.
    from local_operator.server.utils.static_roots import root_refusal

    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{item!r} is not a path; each entry must be a directory string")
        expanded = os.path.expanduser(item.strip())
        if not os.path.isabs(expanded):
            raise ValueError(f"{item!r} is not an absolute path (use /abs/path or ~/path)")
        try:
            real = Path(os.path.realpath(expanded))
        except (OSError, RuntimeError, ValueError):
            raise ValueError(f"{item!r} cannot be resolved to a directory") from None
        reason = root_refusal(real)
        if reason is not None:
            raise ValueError(f"{item!r}: {reason}")


def _validate_aida_name(value: Any) -> None:
    """``aida.name`` is a display name: trimmed, non-empty, bounded, no controls.

    Enforced at the write facade for the same reason ``_validate_advertise_hosts``
    is — every writer (/settings, ``lop config edit``, ``PATCH /v1/settings``)
    funnels through :func:`validate` — and the rename paths reuse THIS refusal
    via ``aida.naming.validate_name``, so the settings page and a ``/aida
    rename`` receipt cannot disagree about what a name is (the drift rule:
    one rule set, one funnel). Whitespace is collapsed before the checks, so
    ``"  Maya  "`` is the valid name ``Maya`` — readers normalize identically —
    while control characters (a pasted escape sequence) are refused outright,
    because they would corrupt every surface that renders the name.
    """
    if not isinstance(value, str):
        raise ValueError("expected a name, e.g. Aida")
    name = " ".join(value.split())
    if not name:
        raise ValueError("a name is required")
    if len(name) > _AIDA_NAME_MAX_CHARS:
        raise ValueError(f"at most {_AIDA_NAME_MAX_CHARS} characters — this is {len(name)}")
    if any(ord(char) < 32 or ord(char) == 127 for char in name):
        raise ValueError("control characters are not allowed in a name")


SETTINGS: tuple[Setting, ...] = (
    # -- model --------------------------------------------------------------
    Setting(
        key="hosting",
        path=("hosting",),
        section="model",
        label=_t(_ws.hosting_label()),
        kind=Kind.TEXT,
        default="",
        # 72 cells. Every help string on this page has to clear the ~76-cell
        # footer budget at 80 columns: past it the shed ladder drops the YAML
        # key path (the thing a user maps a row to the file by) and then the
        # sentence itself is clipped mid-clause with no ellipsis (design round
        # 1, D2). Measure any edit to these four before landing it.
        help=_t(_ws.hosting_help()),
        empty_unsets=True,
    ),
    Setting(
        key="model_name",
        path=("model_name",),
        section="model",
        label=_t(_ws.model_name_label()),
        kind=Kind.TEXT,
        default="",
        # 72 cells — see the note on `hosting` above.
        help=_t(_ws.model_name_help()),
        empty_unsets=True,
    ),
    Setting(
        key="model_effort",
        path=("model_effort",),
        section="model",
        label=_t(_ws.model_effort_label()),
        kind=Kind.ENUM,
        # "" is the unset member: the stored empty string means "no opinion"
        # (the model's own default), exactly like `providers.openrouter.sort`.
        # An ENUM member rather than `empty_unsets` so the page shows `auto`
        # beside the real rungs as a peer to pick between.
        default="",
        # 57 cells, and that is the point (design round 1, D4+D5; round 2, D10):
        # it has to hold the row's own meaning AND the resting state inside the
        # 74-cell detail budget at 80 columns, which the off-default line spends
        # as `<help> · default: —`. The first cut sat EXACTLY on 74 with zero
        # headroom, so one more word anywhere would shed the whole sentence in the
        # state a user reads it in. `the model's default` rather than `the
        # model's own default` recovers 4 cells; the clamp sentence the original
        # cut carried is documented in the README and named where it happens, on
        # the `/model default` receipt.
        help=_t(_ws.model_effort_help()),
        choices=_effort_choices(),
    ),
    # -- providers ----------------------------------------------------------
    Setting(
        key="providers.openai.use_max_context_window",
        path=("providers", "openai", "use_max_context_window"),
        section="providers",
        label=_t(_ws.providers_openai_use_max_context_window_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.providers_openai_use_max_context_window_help())
        ),
    ),
    Setting(
        key="providers.openai.api",
        path=("providers", "openai", "api"),
        section="providers",
        label=_t(_ws.providers_openai_api_label()),
        kind=Kind.ENUM,
        default="responses",
        help=_t(_ws.providers_openai_api_help()),
        choices=(
            Choice("responses", _t(_ws.providers_openai_api_choice_responses_label()), _t(_ws.providers_openai_api_choice_responses_description())),
            Choice("chat_completions", _t(_ws.providers_openai_api_choice_chat_completions_label()), _t(_ws.providers_openai_api_choice_chat_completions_description())),
        ),
    ),
    Setting(
        key="providers.anthropic.cache_ttl_1h_min_context_tokens",
        path=("providers", "anthropic", "cache_ttl_1h_min_context_tokens"),
        # LIVE, not NEW_LAUNCH (review round 2, M6). `_client_for` reads this
        # off the same mapping `apply_settings` rebinds, and the session rebinds
        # on ANY `retry.*` change — so under NEW_LAUNCH the notice told the user
        # a key needed a `/new` while a neighbouring edit had already moved it.
        # Applying it live is harmless (it only affects the next client build),
        # so the honest label is the cheaper of the two fixes.
        section="providers",
        label=_t(_ws.providers_anthropic_cache_ttl_1h_min_context_tokens_label()),
        kind=Kind.INT,
        default=150_000,
        help=(
            _t(_ws.providers_anthropic_cache_ttl_1h_min_context_tokens_help())
        ),
        minimum=0,
        maximum=10_000_000,
    ),
    # -- providers.openrouter: chat-completions `provider` routing object ----
    #
    # These rows build the OpenRouter chat-completions request body's
    # ``provider`` object (openrouter.ai/docs/guides/routing/provider-selection).
    # The over-arching constraint is DeepSeek prompt-cache affinity: OpenRouter
    # normally routes follow-up calls back to the host that answered the first
    # one ("sticky routing"), which keeps the host-side KV cache warm on long
    # conversations. ANY explicit preference risks routing away from that warm
    # host, and ``order`` disables sticky routing outright — so every default
    # below is "no opinion", and the resolver emits NO ``provider`` object at
    # all until the user sets at least one of them. Do not give ``sort`` an
    # explicit default: an always-on sort is an always-on cache miss.
    #
    # Row order is the reading order the object's own docs use (design round 1,
    # D1): the primary control (sort), then the three host lists, then privacy,
    # then price/perf. Labels carry no "OpenRouter " prefix — the section
    # header already says it, and the prefix pushed the distinguishing words
    # past the 29-cell label budget (design round 1, D5).
    # Leads the section despite the reading order below, because it is the one
    # row that is ON by default and every row after it OVERRIDES it: a user who
    # sets `sort`, `order`, `only` or `ignore` has expressed a host preference,
    # and the harness then stops pinning entirely (`SessionStreamFn._affinity_
    # enabled`). Reading it first is what makes that relationship visible.
    #
    # Also the one row here that is NOT a wire key. The four `provider` object
    # keys below are resolved by `_openrouter_provider_preferences` and sent to
    # OpenRouter; this one is a HARNESS switch, deliberately not read by that
    # resolver — see the note on the parity test in tests/unit/test_settings_io.
    Setting(
        key="providers.openrouter.provider_affinity",
        path=("providers", "openrouter", "provider_affinity"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_provider_affinity_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.providers_openrouter_provider_affinity_help())
        ),
    ),
    Setting(
        key="providers.openrouter.sort",
        path=("providers", "openrouter", "sort"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_sort_label()),
        kind=Kind.ENUM,
        # "" is the unset member: the schema must know a stored empty string
        # means "no opinion", and the resolver treats it exactly like a missing
        # key. An ENUM member rather than `empty_unsets` so the page can show
        # the three real policies beside "default" as peers to pick between.
        default="",
        help=(
            _t(_ws.providers_openrouter_sort_help())
        ),
        choices=(
            Choice("", _t(_ws.providers_openrouter_sort_choice_unset_label()), _t(_ws.providers_openrouter_sort_choice_unset_description())),
            Choice("price", _t(_ws.providers_openrouter_sort_choice_price_label()), _t(_ws.providers_openrouter_sort_choice_price_description())),
            Choice("throughput", _t(_ws.providers_openrouter_sort_choice_throughput_label()), _t(_ws.providers_openrouter_sort_choice_throughput_description())),
            Choice("latency", _t(_ws.providers_openrouter_sort_choice_latency_label()), _t(_ws.providers_openrouter_sort_choice_latency_description())),
        ),
    ),
    Setting(
        key="providers.openrouter.order",
        path=("providers", "openrouter", "order"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_order_label()),
        kind=Kind.LIST,
        default=[],
        # The consequence LEADS the detail line in the page's danger ink and is
        # never shed from it — including on the off-default row, where the
        # `default:` reset clause used to crowd it out exactly when the
        # dangerous value was stored (QA round 1 Q1 / design round 1 D2). The
        # help below stays SOFT faint ink so the two ranks read apart.
        warning=_t(_ws.providers_openrouter_order_warning()),
        # Empty-first (design round 1, D3): empty is the default that must not
        # be disturbed, so the detail names what empty MEANS before the how-to.
        help=(
            _t(_ws.providers_openrouter_order_help())
        ),
        # OPEN namespace — deliberately no `members`. OpenRouter owns the slug
        # vocabulary and grows it without notice (deepinfra, novita, regional
        # variants like google-vertex/us-east5), so a closed list would reject
        # hosts the upstream docs themselves use. Any non-empty slug token
        # validates; the placeholder seeds the common hosts without gating
        # the write.
        placeholder=_t(_ws.providers_openrouter_order_placeholder()),
        # An empty routing order is "no opinion", not a validation error —
        # unlike web_search.providers, nothing breaks with zero entries.
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.only",
        path=("providers", "openrouter", "only"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_only_label()),
        kind=Kind.LIST,
        default=[],
        help=(
            _t(_ws.providers_openrouter_only_help())
        ),
        # Open namespace, same reason as `order` above.
        placeholder=_t(_ws.providers_openrouter_only_placeholder()),
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.ignore",
        path=("providers", "openrouter", "ignore"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_ignore_label()),
        kind=Kind.LIST,
        default=[],
        help=(
            _t(_ws.providers_openrouter_ignore_help())
        ),
        # Open namespace, same reason as `order` above.
        placeholder=_t(_ws.providers_openrouter_ignore_placeholder()),
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.allow_fallbacks",
        path=("providers", "openrouter", "allow_fallbacks"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_allow_fallbacks_label()),
        # ENUM, not BOOL (design round 1, D5/N3): at the wire this is tri-state
        # — OpenRouter's own default is "fall through", the only meaningful
        # override is "fail instead", and the resolver omits the key entirely
        # while at default. A BOOL defaulted to True painted the resting row
        # as a configured `on` when nothing is sent, the one row in the
        # section that looked set on a fresh install. `default`/`false` keeps
        # the no-opinion vocabulary the other rows use: `—` at rest. The
        # choice VALUE is the literal sent ("false"), so what the expansion
        # shows is what `lop config edit` accepts.
        kind=Kind.ENUM,
        default="",
        help=(
            _t(_ws.providers_openrouter_allow_fallbacks_help())
        ),
        choices=(
            Choice("", _t(_ws.providers_openrouter_allow_fallbacks_choice_unset_label()), _t(_ws.providers_openrouter_allow_fallbacks_choice_unset_description())),
            Choice("false", _t(_ws.providers_openrouter_allow_fallbacks_choice_false_label()), _t(_ws.providers_openrouter_allow_fallbacks_choice_false_description())),
        ),
    ),
    Setting(
        key="providers.openrouter.require_parameters",
        path=("providers", "openrouter", "require_parameters"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_require_parameters_label()),
        # ENUM for the same tri-state reason as `allow_fallbacks`: False in a
        # BOOL read as "parameter dropping is off" while the wire truth is
        # "no preference sent"; "" restores the shared no-opinion `—`.
        kind=Kind.ENUM,
        default="",
        help=(
            _t(_ws.providers_openrouter_require_parameters_help())
        ),
        choices=(
            Choice("", _t(_ws.providers_openrouter_require_parameters_choice_unset_label()), _t(_ws.providers_openrouter_require_parameters_choice_unset_description())),
            Choice("true", _t(_ws.providers_openrouter_require_parameters_choice_true_label()), _t(_ws.providers_openrouter_require_parameters_choice_true_description())),
        ),
    ),
    Setting(
        key="providers.openrouter.data_collection",
        path=("providers", "openrouter", "data_collection"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_data_collection_label()),
        kind=Kind.ENUM,
        default="",
        help=(
            _t(_ws.providers_openrouter_data_collection_help())
        ),
        choices=(
            Choice("", _t(_ws.providers_openrouter_data_collection_choice_unset_label()), _t(_ws.providers_openrouter_data_collection_choice_unset_description())),
            Choice("allow", _t(_ws.providers_openrouter_data_collection_choice_allow_label()), _t(_ws.providers_openrouter_data_collection_choice_allow_description())),
            Choice("deny", _t(_ws.providers_openrouter_data_collection_choice_deny_label()), _t(_ws.providers_openrouter_data_collection_choice_deny_description())),
        ),
    ),
    Setting(
        key="providers.openrouter.zdr",
        path=("providers", "openrouter", "zdr"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_zdr_label()),
        kind=Kind.ENUM,
        # ENUM, not BOOL (design round 1, D5): the wire key is tri-state —
        # absent means "no preference" (OpenRouter's default stands), `true`
        # means ZDR hosts only, and there is no meaningful `false` to send.
        # A BOOL had to fake the unset state as `off`, which read as "ZDR
        # disabled" beside ENUM rows showing `—` for the same state.
        default="",
        help=_t(_ws.providers_openrouter_zdr_help()),
        choices=(
            Choice("", _t(_ws.providers_openrouter_zdr_choice_unset_label()), _t(_ws.providers_openrouter_zdr_choice_unset_description())),
            Choice("true", _t(_ws.providers_openrouter_zdr_choice_true_label()), _t(_ws.providers_openrouter_zdr_choice_true_description())),
        ),
    ),
    Setting(
        key="providers.openrouter.enforce_distillable_text",
        path=("providers", "openrouter", "enforce_distillable_text"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_enforce_distillable_text_label()),
        # ENUM for the same tri-state reason as `zdr` directly above.
        kind=Kind.ENUM,
        default="",
        help=(
            _t(_ws.providers_openrouter_enforce_distillable_text_help())
        ),
        choices=(
            Choice("", _t(_ws.providers_openrouter_enforce_distillable_text_choice_unset_label()), _t(_ws.providers_openrouter_enforce_distillable_text_choice_unset_description())),
            Choice("true", _t(_ws.providers_openrouter_enforce_distillable_text_choice_true_label()), _t(_ws.providers_openrouter_enforce_distillable_text_choice_true_description())),
        ),
    ),
    Setting(
        key="providers.openrouter.quantizations",
        path=("providers", "openrouter", "quantizations"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_quantizations_label()),
        kind=Kind.LIST,
        default=[],
        help=(
            _t(_ws.providers_openrouter_quantizations_help())
        ),
        # CLOSED here (unlike the host lists): this vocabulary is a documented
        # finite set, not a growing upstream namespace. mxfp4/nvfp4/mxfp8 and
        # `unknown` are in the OpenRouter docs beside the classic levels
        # (review round 1, m2).
        members=(
            "int4",
            "int8",
            "fp4",
            "fp6",
            "fp8",
            "fp16",
            "bf16",
            "fp32",
            "mxfp4",
            "nvfp4",
            "mxfp8",
            "unknown",
        ),
        placeholder=_t(_ws.providers_openrouter_quantizations_placeholder()),
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.max_price",
        path=("providers", "openrouter", "max_price"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_max_price_label()),
        kind=Kind.TEXT,
        default="",
        # Example-first and short (design round 1, D4): the old prose pushed
        # the units past the ellipsis at 120 columns, and for a JSON-in-TEXT
        # row the example IS the documentation. The empty editor ghosts the
        # same example; the four accepted field names are enumerated by the
        # validator's own rejection message.
        help=_t(_ws.providers_openrouter_max_price_help()),
        placeholder=_t(_ws.providers_openrouter_max_price_placeholder()),
        empty_unsets=True,
        validate_value=_validate_openrouter_max_price,
    ),
    Setting(
        key="providers.openrouter.preferred_min_throughput",
        path=("providers", "openrouter", "preferred_min_throughput"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_preferred_min_throughput_label()),
        kind=Kind.FLOAT,
        default=0.0,
        help=(
            _t(_ws.providers_openrouter_preferred_min_throughput_help())
        ),
        minimum=0.0,
        maximum=100_000.0,
    ),
    Setting(
        key="providers.openrouter.preferred_max_latency",
        path=("providers", "openrouter", "preferred_max_latency"),
        section="openrouter",
        label=_t(_ws.providers_openrouter_preferred_max_latency_label()),
        kind=Kind.FLOAT,
        default=0.0,
        help=(
            _t(_ws.providers_openrouter_preferred_max_latency_help())
        ),
        minimum=0.0,
        maximum=600.0,
    ),
    # -- failover -----------------------------------------------------------
    Setting(
        key="retry.enabled",
        path=("retry", "enabled"),
        section="failover",
        label=_t(_ws.retry_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.retry_enabled_help()),
        choices=_bool_choices(_t(_ws.retry_enabled_choice_true_description()), _t(_ws.retry_enabled_choice_false_description())),
    ),
    Setting(
        key="retry.maxRetries",
        path=("retry", "maxRetries"),
        section="failover",
        label=_t(_ws.retry_maxretries_label()),
        kind=Kind.INT,
        default=10,
        help=_t(_ws.retry_maxretries_help()),
        minimum=0,
        maximum=100,
    ),
    Setting(
        key="retry.baseDelayMs",
        path=("retry", "baseDelayMs"),
        section="failover",
        label=_t(_ws.retry_basedelayms_label()),
        kind=Kind.INT,
        default=500,
        help=_t(_ws.retry_basedelayms_help()),
        minimum=0,
        maximum=60_000,
    ),
    Setting(
        key="retry.connectivityMaxRetries",
        path=("retry", "connectivityMaxRetries"),
        section="failover",
        label=_t(_ws.retry_connectivitymaxretries_label()),
        kind=Kind.INT,
        default=15,
        help=_t(_ws.retry_connectivitymaxretries_help()),
        minimum=0,
        maximum=200,
    ),
    Setting(
        key="retry.connectivityBackoffCapMs",
        path=("retry", "connectivityBackoffCapMs"),
        section="failover",
        label=_t(_ws.retry_connectivitybackoffcapms_label()),
        kind=Kind.INT,
        default=60_000,
        help=_t(_ws.retry_connectivitybackoffcapms_help()),
        minimum=1_000,
        maximum=600_000,
    ),
    Setting(
        key="retry.modelFallback",
        path=("retry", "modelFallback"),
        section="failover",
        label=_t(_ws.retry_modelfallback_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.retry_modelfallback_help()),
        choices=_bool_choices(_t(_ws.retry_modelfallback_choice_true_description()), _t(_ws.retry_modelfallback_choice_false_description())),
    ),
    Setting(
        key="retry.usageAwareFallback",
        path=("retry", "usageAwareFallback"),
        section="failover",
        label=_t(_ws.retry_usageawarefallback_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.retry_usageawarefallback_help()),
        choices=_bool_choices(_t(_ws.retry_usageawarefallback_choice_true_description()), _t(_ws.retry_usageawarefallback_choice_false_description())),
    ),
    Setting(
        key="retry.usageAwareAccountPick",
        path=("retry", "usageAwareAccountPick"),
        section="failover",
        label=_t(_ws.retry_usageawareaccountpick_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.retry_usageawareaccountpick_help())
        ),
        choices=_bool_choices(_t(_ws.retry_usageawareaccountpick_choice_true_description()), _t(_ws.retry_usageawareaccountpick_choice_false_description())),
    ),
    Setting(
        key="retry.usageReservePercent",
        path=("retry", "usageReservePercent"),
        section="failover",
        label=_t(_ws.retry_usagereservepercent_label()),
        kind=Kind.FLOAT,
        default=10.0,
        help=_t(_ws.retry_usagereservepercent_help()),
        minimum=0.0,
        maximum=100.0,
    ),
    Setting(
        key="retry.fallbackChains",
        path=("retry", "fallbackChains"),
        section="failover",
        label=_t(_ws.retry_fallbackchains_label()),
        kind=Kind.CASCADE,
        default={},
        help=_t(_ws.retry_fallbackchains_help()),
    ),
    Setting(
        key="retry.pinnedFallback",
        path=("retry", "pinnedFallback"),
        section="failover",
        label=_t(_ws.retry_pinnedfallback_label()),
        # ENUM: the value space is two words this module owns, and the runtime
        # degrades any other shape to the default (see `RetrySettings.
        # from_settings`), so the page should offer exactly the two.
        kind=Kind.ENUM,
        # The literal is deliberate, like every default in this file; the code
        # constant is `DEFAULT_PINNED_FALLBACK` in providers/failover.py, and
        # `_consumer_defaults()` in tests/unit/test_settings_io.py pins the
        # pair to each other. "cross-family" has been the shipped default
        # since the 2026-09-30 incident: a pinned child whose credential is
        # unusable must still reach a configured hop, cross-vendor included
        # as the announced last resort, rather than fail on a chain it was
        # forbidden to walk. "same-family" remains the strict opt-in.
        default="cross-family",
        # NOTE ON THE COPY (review round 1, D1/D3; hint shortened and bounds
        # re-derived in the design review round 1 remediation, D1): the row's
        # two choice descriptions render ONLY on the expanded rows, where the
        # painted field — truncation ellipsis included — is 17/26 cells at
        # 60x20 and 23/32 at 100x30 (captured frames), and each description
        # IS a micro-hint (the discriminating words lead). The `(default)`
        # marker now sits on `cross-family` and costs its row ~9 of those
        # cells — the frame showed the old 22-cell `any vendor (announced)`
        # clipped to `(announce…` — so that hint is the 10-cell `any vendor`;
        # `(announced)` lives in the help below, which paints in full on the
        # detail line's 93 cells at 100x30, and is the clause the overlong
        # predecessor clipped at every size.
        help=(
            _t(_ws.retry_pinnedfallback_help())
        ),
        choices=(
            Choice(
                "same-family",
                _t(_ws.retry_pinnedfallback_choice_same_family_label()),
                _t(_ws.retry_pinnedfallback_choice_same_family_description()),
            ),
            Choice(
                "cross-family",
                _t(_ws.retry_pinnedfallback_choice_cross_family_label()),
                _t(_ws.retry_pinnedfallback_choice_cross_family_description()),
            ),
        ),
    ),
    # -- appearance ---------------------------------------------------------
    Setting(
        key="tui.theme",
        path=("tui", "theme"),
        section="appearance",
        label=_t(_ws.tui_theme_label()),
        # ENUM, not TEXT: the value space is closed (the theme registry), so a
        # free-text field let the page accept a theme that does not exist and
        # then display a value the app was not using — `app.py` catches the
        # KeyError and falls back to the default, silently. Enum also gives the
        # row the same expand-and-pick affordance every other closed value
        # space here has (review round 1, m1).
        kind=Kind.ENUM,
        # "dark" restated rather than imported from `tui.theme.DEFAULT_THEME`,
        # for the reason every default in this file is a literal: importing a
        # consumer here would put it on the CLI's import path (`tui.theme`
        # pulls in `rich.style`, ~90ms), and the module docstring's "keep it
        # dependency-light" rule is what makes this registry shareable. The
        # drift that costs is bought back by the anti-drift test, which
        # compares this against DEFAULT_THEME itself and now fails loudly
        # rather than skipping (review round 1, M1).
        default="dark",
        choices_source=_theme_choices,
        help=_t(_ws.tui_theme_help()),
        # The one previewing setting. `/theme` already browses with a live
        # arrow-key preview and restores on cancel, so this makes the settings
        # page offer the same affordance through the same live-apply path
        # rather than being the one place a theme can only be tried by keeping
        # it (#440 §3).
        preview=True,
    ),
    # The five display flags below are the FLAT-DOTTED case: each `path` is a
    # single element containing a dot, because `tui/settings.py` reads
    # `values["display.shimmer"]` — a top-level key that happens to have a dot
    # in its name. Splitting these on `.` writes a `display:` mapping nothing
    # reads. See the module docstring.
    Setting(
        key="display.shimmer",
        path=("display.shimmer",),
        section="appearance",
        label=_t(_ws.display_shimmer_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_shimmer_help()),
        choices=_bool_choices(_t(_ws.display_shimmer_choice_true_description()), _t(_ws.display_shimmer_choice_false_description())),
    ),
    Setting(
        key="display.narration",
        path=("display.narration",),
        section="appearance",
        label=_t(_ws.display_narration_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_narration_help()),
        choices=_bool_choices(_t(_ws.display_narration_choice_true_description()), _t(_ws.display_narration_choice_false_description())),
    ),
    Setting(
        key="display.reasoning",
        path=("display.reasoning",),
        section="appearance",
        label=_t(_ws.display_reasoning_label()),
        kind=Kind.BOOL,
        default=False,
        # ONE sentence, sized to the field rather than to the argument: the help
        # column paints 93 characters at 100 columns (133 at 140, 152 at 160) and
        # elides the rest, so a longer string loses its tail silently — the
        # previous revision's 178-character sentence had its whole `so nothing is
        # left in the transcript` clause cut off screen (design review round 1,
        # D1; `after-d1-reasoning-help-100x30.svg` paints this one whole). The
        # `/resume` argument that was cut is recorded in `tui/settings.py`'s
        # `_DEFAULT_NOTES` and in `tui/widgets/reasoning.py`'s module docstring,
        # where it has room. The trigger says the thinking ENDS, not "the answer
        # starts", because `reasoning_end` also drops the block on a turn that
        # goes on to a tool call (round 1, D4).
        help=(
            _t(_ws.display_reasoning_help())
        ),
        choices=_bool_choices(_t(_ws.display_reasoning_choice_true_description()), _t(_ws.display_reasoning_choice_false_description())),
    ),
    Setting(
        key="display.rail",
        path=("display.rail",),
        section="appearance",
        label=_t(_ws.display_rail_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.display_rail_help())
        ),
        choices=_bool_choices(_t(_ws.display_rail_choice_true_description()), _t(_ws.display_rail_choice_false_description())),
    ),
    Setting(
        # Default changed to False by maintainer
        key="display.comfortable_rows",
        path=("display.comfortable_rows",),
        section="appearance",
        label=_t(_ws.display_comfortable_rows_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.display_comfortable_rows_help()),
        choices=_bool_choices(_t(_ws.display_comfortable_rows_choice_true_description()), _t(_ws.display_comfortable_rows_choice_false_description())),
    ),
    Setting(
        key="display.nerd_icons",
        path=("display.nerd_icons",),
        section="appearance",
        label=_t(_ws.display_nerd_icons_label()),
        kind=Kind.ENUM,
        default=None,
        help=_t(_ws.display_nerd_icons_help()),
        # The tri-state IS the None-vs-bool distinction: `settings_get` returns
        # None only when the key is ABSENT, which is what "auto" reads. So the
        # auto choice must write nothing rather than write a value — handled by
        # `write_setting`, which deletes on a None for a key with no shipped
        # default.
        choices=(
            Choice(None, _t(_ws.display_nerd_icons_choice_none_label()), _t(_ws.display_nerd_icons_choice_none_description())),
            Choice(True, _t(_ws.display_nerd_icons_choice_true_label()), _t(_ws.display_nerd_icons_choice_true_description())),
            Choice(False, _t(_ws.display_nerd_icons_choice_false_label()), _t(_ws.display_nerd_icons_choice_false_description())),
        ),
    ),
    Setting(
        key="display.heading_markers",
        path=("display.heading_markers",),
        section="appearance",
        label=_t(_ws.display_heading_markers_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.display_heading_markers_help()),
        choices=_bool_choices(_t(_ws.display_heading_markers_choice_true_description()), _t(_ws.display_heading_markers_choice_false_description())),
    ),
    Setting(
        key="display.terminal_title",
        path=("display.terminal_title",),
        section="appearance",
        label=_t(_ws.display_terminal_title_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_terminal_title_help()),
        choices=_bool_choices(_t(_ws.display_terminal_title_choice_true_description()), _t(_ws.display_terminal_title_choice_false_description())),
    ),
    Setting(
        key="display.images",
        path=("display.images",),
        section="appearance",
        label=_t(_ws.display_images_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_images_help()),
        choices=_bool_choices(_t(_ws.display_images_choice_true_description()), _t(_ws.display_images_choice_false_description())),
    ),
    Setting(
        key="display.notifications",
        path=("display.notifications",),
        section="appearance",
        label=_t(_ws.display_notifications_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_notifications_help()),
        choices=_bool_choices(_t(_ws.display_notifications_choice_true_description()), _t(_ws.display_notifications_choice_false_description())),
    ),
    Setting(
        # The observer path (a background session finishing while you are
        # looking at a DIFFERENT one) widens where a session name appears: the
        # toast is about a conversation the terminal is not showing, so its
        # name is the only thing that identifies it, and macOS repeats banner
        # titles on the lock screen. Session names are model-written and can
        # quote the work, so this is the opt-out for anyone who does not want
        # that on a screen other people can see. Default TRUE because it
        # matches what every existing notification already does; off falls back
        # to the brand name, which still says a session finished.
        key="display.notification_session_name",
        path=("display.notification_session_name",),
        section="appearance",
        # Fits the row's label column at 100 columns. "Name sessions in
        # notifications" elided to "…notificatio…", losing the one word that
        # says what the row governs.
        label=_t(_ws.display_notification_session_name_label()),
        kind=Kind.BOOL,
        default=True,
        # Leads with the CONSEQUENCE, not the mechanism: the decision being
        # asked for is whether model-written content (session name, last
        # line, error cause) appears on a lock screen, which is the one fact
        # that changes someone's answer. The neighbours ("Fires only while
        # the terminal is unfocused.") are worded the same way (design
        # round 1, D6). All three legs are named because this flag gates
        # banner BODIES too, not just names (design round 2, D1): OFF drops
        # the error cause that makes an error banner actionable, and a row
        # that hides that trade surprises someone who turns it off.
        # Implicit concatenation keeps the rendered string one sentence while
        # holding the line under the 100-column flake8/black budget; a
        # `noqa: E501` on the joined form is reserved in this repo for
        # unsplittable content (URLs, embedded code), not prose.
        help=(
            _t(_ws.display_notification_session_name_help())
        ),
        # `off` no longer means "app name only" on every route — a background
        # session with no stored title is titled "A session finished" — so the
        # label names what the user gets rather than a fallback that is now one
        # of two (design round 1, D7, folded into D2).
        choices=_bool_choices(_t(_ws.display_notification_session_name_choice_true_description()), _t(_ws.display_notification_session_name_choice_false_description())),
        # Without this the row renders `on` while `display.notifications` is
        # off, describing a banner that cannot fire — the "page states
        # something untrue about its own effect" class (#431). The detail line
        # now reads `inert: Desktop notifications is off` through the mechanism
        # four `session.cleanup.*` rows already use (design round 1, D4).
        gated_by="display.notifications",
        # Without this the row renders `on` while `display.notifications` is
        # off, describing a banner that cannot fire — the "page states
        # something untrue about its own effect" class (#431). The detail line
        # now reads `inert: Desktop notifications is off` through the mechanism
        # four `session.cleanup.*` rows already use (design round 1, D4).
    ),
    Setting(
        key="display.time_format",
        path=("display.time_format",),
        section="appearance",
        label=_t(_ws.display_time_format_label()),
        kind=Kind.ENUM,
        default="12h",
        help=_t(_ws.display_time_format_help()),
        choices=(
            Choice("12h", _t(_ws.display_time_format_choice_12h_label()), _t(_ws.display_time_format_choice_12h_description())),
            Choice("24h", _t(_ws.display_time_format_choice_24h_label()), _t(_ws.display_time_format_choice_24h_description())),
        ),
    ),
    Setting(
        key="language",
        path=("language",),
        section="appearance",
        # i18n: ignore wire.settings slice owns this row's copy (label + help).
        label=_t(_ws.language_label()),
        kind=Kind.ENUM,
        # Literal, not imported: the registry stays off the resolver's import
        # path (the same arrangement as `tui.theme` above), and
        # `test_every_default_matches_its_consumer` compares this against
        # `i18n.resolve.DEFAULT_LANGUAGE`, which is what stops the two
        # drifting into a page that lies.
        default="auto",
        help=_t(_ws.language_help()),
        # Dynamic: auto plus the locales the translation ledger has passed
        # (RFC §3.1 — unaudited locales are not offered). Widens by
        # regeneration, never by a code change here.
        choices_source=_language_choices,
    ),
    Setting(
        # The session's INITIAL dock density, not a hard override: `ctrl+g`
        # cycles freely from it and never writes it back (the same split as
        # `tool_approval_mode` vs `/approvals`). A hard override would make
        # the key a no-op again, which is the defect #525 fixed. LIVE by
        # section: a write applies to the mounted panel unless the user has
        # already cycled it this session. Named `display.dock` rather than
        # `display.subagent_dock` so a later todo-panel extension can share it.
        key="display.dock",
        path=("display.dock",),
        section="appearance",
        label=_t(_ws.display_dock_label()),
        kind=Kind.ENUM,
        default="full",
        help=_t(_ws.display_dock_help()),
        choices=(
            # Parallel descriptions: each names WHAT IS SHOWN and nothing
            # else. `hidden` used to append "; ctrl+g brings it back", which
            # made it the only choice explaining its own exit and left the
            # list reading unevenly — and the help line above already names
            # `ctrl+g` for all three (round 1, D5).
            Choice("full", _t(_ws.display_dock_choice_full_label()), _t(_ws.display_dock_choice_full_description())),
            Choice("summary", _t(_ws.display_dock_choice_summary_label()), _t(_ws.display_dock_choice_summary_description())),
            Choice("hidden", _t(_ws.display_dock_choice_hidden_label()), _t(_ws.display_dock_choice_hidden_description())),
        ),
    ),
    # Cross-session traffic — the `send` tool's own traces and the inbound
    # peer receipts. Named for what it HIDES, unlike every show-positive
    # `display.*` sibling above: a config file reads
    # `display.hide_cross_session: true` as exactly what it does, while a
    # `cross_session: true` could not be told from "show" without the help
    # text. OFF is today's rendering. Read by the TUI gates and the phone fold
    # through `local_operator.cross_session.cross_session_hidden()`; the
    # forward-only mid-session semantics and the restore path are documented
    # in `tui/settings.py`'s `_DEFAULT_NOTES`.
    #
    # The label carries the VERB ("Hide …") the way the key does: at rest
    # the row reads `Hide cross-session traffic  off`, because a skimmer
    # comparing it with `Status band on` / `Prompt chevron on` read a bare
    # `Cross-session traffic  on` as "the traffic is on" (design review
    # round 1, D4). The choices carry the FORWARD-ONLY scope, not the help:
    # they render fully at every tested width down to 60x20 (a 54-cell list;
    # the off member's 11-cell `(default)` tag is what bounds the clause
    # length), and "reopen to re-read" is the half of the sentence a user
    # flipping the switch actually needs (UX round 1, U1). Flat-dotted like
    # every `display.*` key (the dot is part of the literal top-level key —
    # see the block comment at `display.shimmer`).
    Setting(
        key="display.hide_cross_session",
        path=("display.hide_cross_session",),
        section="appearance",
        label=_t(_ws.display_hide_cross_session_label()),
        kind=Kind.BOOL,
        default=False,  # OFF = today's rendering; see tui/settings.py _DEFAULT_NOTES
        help=_t(_ws.display_hide_cross_session_help()),
        choices=_bool_choices(_t(_ws.display_hide_cross_session_choice_true_description()), _t(_ws.display_hide_cross_session_choice_false_description())),
    ),
    # The desktop transcript's answer mark: a thin rule to the left of the row
    # that CLOSES a turn (the `display.rail` idea, for the desktop app). Only
    # `local-operator-ui` draws it; the TUI has no such element, so the TUI
    # reader is settings-only (this key is in the test allow-list of keys with
    # no single-value consumer, like every registry-derived `display.*` flag).
    #
    # Default OFF, deliberately: the operator reported the rail looked heavy
    # next to the answer text, so it ships opt-in rather than as the new look.
    # "Unset" must therefore mean the transcript as it renders today. Flat-dotted
    # like every `display.*` key (see the block comment at `display.shimmer`).
    Setting(
        key="display.turn_answer_rail",
        path=("display.turn_answer_rail",),
        section="appearance",
        label=_t(_ws.display_turn_answer_rail_label()),
        kind=Kind.BOOL,
        default=False,  # opt-in: the rail looked heavy; see tui/settings.py _DEFAULT_NOTES
        help=_t(_ws.display_turn_answer_rail_help()),
        choices=_bool_choices(_t(_ws.display_turn_answer_rail_choice_true_description()), _t(_ws.display_turn_answer_rail_choice_false_description())),
    ),
    # -- the composer widget-visibility family (operator request, 2026-09-27) --
    #
    # One BOOL per composable piece of the composer: the status band and the
    # prompt chevron as whole widgets, plus one key per band segment. All
    # default True, which IS today's shape — the family exists to REMOVE
    # pieces, so "unset" must mean "the band as shipped". LIVE like the rest of
    # their section: the segment keys are read on the paint path
    # (`tui/settings.py` -> `status_line._composer_hidden_segments`) and the
    # two widget keys are applied by `OperatorApp._apply_composer_settings` off
    # the config watcher, so an edit lands on a running TUI without a relaunch.
    #
    # Flat-dotted like every `display.*` key above (see the block comment at
    # `display.shimmer`): the dot is part of a literal top-level key, so each
    # `path` is the ONE-element tuple holding the whole key.
    Setting(
        key="display.composer.band",
        path=("display.composer.band",),
        section="appearance",
        label=_t(_ws.display_composer_band_label()),
        kind=Kind.BOOL,
        default=True,
        # The disclosure leads, because the tail is the point (design review
        # round 1, D1): hiding the band also hides its STANDING alerts — the
        # disarmed-gate `!`, a parked connector, the MCP-failure lamp — and a
        # help that lists the segments first clips that clause off first.
        # Measured 67 cells, which paints whole at both 100 and 80 columns.
        help=_t(_ws.display_composer_band_help()),
        choices=_bool_choices(_t(_ws.display_composer_band_choice_true_description()), _t(_ws.display_composer_band_choice_false_description())),
    ),
    Setting(
        key="display.composer.chevron",
        path=("display.composer.chevron",),
        section="appearance",
        label=_t(_ws.display_composer_chevron_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_composer_chevron_help()),
        choices=_bool_choices(_t(_ws.display_composer_chevron_choice_true_description()), _t(_ws.display_composer_chevron_choice_false_description())),
    ),
    Setting(
        key="display.composer.model",
        path=("display.composer.model",),
        section="appearance",
        label=_t(_ws.display_composer_model_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_composer_model_help()),
        choices=_bool_choices(_t(_ws.display_composer_model_choice_true_description()), _t(_ws.display_composer_model_choice_false_description())),
    ),
    Setting(
        key="display.composer.cwd",
        path=("display.composer.cwd",),
        section="appearance",
        label=_t(_ws.display_composer_cwd_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_composer_cwd_help()),
        choices=_bool_choices(_t(_ws.display_composer_cwd_choice_true_description()), _t(_ws.display_composer_cwd_choice_false_description())),
    ),
    Setting(
        key="display.composer.context",
        path=("display.composer.context",),
        section="appearance",
        label=_t(_ws.display_composer_context_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_composer_context_help()),
        choices=_bool_choices(_t(_ws.display_composer_context_choice_true_description()), _t(_ws.display_composer_context_choice_false_description())),
    ),
    Setting(
        key="display.composer.rate",
        path=("display.composer.rate",),
        section="appearance",
        label=_t(_ws.display_composer_rate_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_composer_rate_help()),
        choices=_bool_choices(_t(_ws.display_composer_rate_choice_true_description()), _t(_ws.display_composer_rate_choice_false_description())),
    ),
    Setting(
        key="display.composer.cost",
        path=("display.composer.cost",),
        section="appearance",
        label=_t(_ws.display_composer_cost_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_composer_cost_help()),
        choices=_bool_choices(_t(_ws.display_composer_cost_choice_true_description()), _t(_ws.display_composer_cost_choice_false_description())),
    ),
    Setting(
        key="display.composer.duration",
        path=("display.composer.duration",),
        section="appearance",
        label=_t(_ws.display_composer_duration_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.display_composer_duration_help()),
        choices=_bool_choices(_t(_ws.display_composer_duration_choice_true_description()), _t(_ws.display_composer_duration_choice_false_description())),
    ),
    Setting(
        key="tui.sidebar_visible",
        path=("tui", "sidebar_visible"),
        section="appearance",
        label=_t(_ws.tui_sidebar_visible_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.tui_sidebar_visible_help()),
        choices=_bool_choices(_t(_ws.tui_sidebar_visible_choice_true_description()), _t(_ws.tui_sidebar_visible_choice_false_description())),
    ),
    Setting(
        key="tui.sidebar_position",
        path=("tui", "sidebar_position"),
        section="appearance",
        label=_t(_ws.tui_sidebar_position_label()),
        kind=Kind.ENUM,
        default="left",
        help=_t(_ws.tui_sidebar_position_help()),
        choices=(
            Choice("left", _t(_ws.tui_sidebar_position_choice_left_label()), _t(_ws.tui_sidebar_position_choice_left_description())),
            Choice("right", _t(_ws.tui_sidebar_position_choice_right_label()), _t(_ws.tui_sidebar_position_choice_right_description())),
        ),
    ),
    Setting(
        key="tui.sidebar_show_subagents",
        path=("tui", "sidebar_show_subagents"),
        section="appearance",
        label=_t(_ws.tui_sidebar_show_subagents_label()),
        kind=Kind.BOOL,
        default=False,
        help=(
            _t(_ws.tui_sidebar_show_subagents_help())
        ),
        choices=_bool_choices(_t(_ws.tui_sidebar_show_subagents_choice_true_description()), _t(_ws.tui_sidebar_show_subagents_choice_false_description())),
    ),
    # -- hotkeys ------------------------------------------------------------
    # DERIVED from `keymap.KEY_ACTIONS` rather than spelled out, because the
    # id is simultaneously this key, the `Binding` id in `OperatorApp.BINDINGS`
    # and the tip lookup in `welcome.py`. Writing it here a second time is the
    # drift `test_keymap.py`'s three-way anti-drift test exists to catch, and
    # the id is PERSISTED USER DATA (it is the literal key in the user's
    # config.yml), so drift orphans overrides silently rather than failing.
    #
    # FLAT-DOTTED, like `display.*` and unlike `tui.*`: `_changed_registry_keys`
    # only diffs keys the registry knows, so a nested `keymap:` block would
    # invite hand-written sub-keys that propagation could never see. A flat key
    # per registered action makes "registered" and "propagated" the same set by
    # construction.
    *(
        Setting(
            key=action.id,
            path=(action.id,),
            section="keymap",
            label=action.label,
            kind=Kind.HOTKEY,
            default=action.default,
            help=action.help,
            hotkey_scope=action.scope,
        )
        for action in _keymap.KEY_ACTIONS
    ),
    # -- approvals ----------------------------------------------------------
    Setting(
        key="tool_approval_mode",
        path=("tool_approval_mode",),
        section="approvals",
        label=_t(_ws.tool_approval_mode_label()),
        kind=Kind.ENUM,
        default="ask",
        # 58 cells, and the number is the LADDER's, not a taste call. The detail
        # line composes ``<help> · default: <default>`` when the row is off its
        # default and sheds whole rungs to fit — so the budget that decides
        # whether this sentence is on the frame at all is 74 cells MINUS the
        # 15-cell ``· default: ask`` suffix, i.e. 59. Measured on the head with
        # the real page: the 72-cell sentence I first wrote (design round 1,
        # D1's recommendation, which was measured against the width with no
        # clause) painted NOTHING off-default at 80x24 — the ladder fell through
        # to ``default: ask   tool_approval_mode``, which is exactly the state the
        # operator is in after the write this change is about. Verified again
        # after this edit: 80x24 off-default shows the help and the clause, and
        # only 60 cols ellipsizes (today's string already did).
        #
        # The sentence is the SECTION description's rule in the TUI's own words —
        # one clause, no head — because that description is painted by the desktop
        # header alone and this is the only approvals copy the TUI page shows
        # (design round 1, D1; agent review round 1, m1). Both must stay true for
        # the embedded pane (whose own page write IS the gate-holding process's
        # write) and for the attached one.
        help=_t(_ws.tool_approval_mode_help()),
        choices=(
            Choice("ask", _t(_ws.tool_approval_mode_choice_ask_label()), _t(_ws.tool_approval_mode_choice_ask_description())),
            Choice("auto", _t(_ws.tool_approval_mode_choice_auto_label()), _t(_ws.tool_approval_mode_choice_auto_description())),
        ),
    ),
    # -- session storage ----------------------------------------------------
    Setting(
        key="auto_save_conversation",
        path=("auto_save_conversation",),
        section="session",
        label=_t(_ws.auto_save_conversation_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.auto_save_conversation_help()),
        choices=_bool_choices(_t(_ws.auto_save_conversation_choice_true_description()), _t(_ws.auto_save_conversation_choice_false_description())),
    ),
    # -- runtime ------------------------------------------------------------
    Setting(
        # `runtime.*`, matching the section it appears in and the other key in
        # it. Round 1 (R5): filing it under `session.*` while showing it in the
        # Runtime section made the file teach two rules — `session.cleanup.*`
        # stays in the Session section, so a user reading the Runtime page
        # could not predict which YAML key they were editing. The scope
        # argument for keeping it out of the Session SECTION (that section is
        # NEW_LAUNCH, this key is LIVE) is sound and unaffected: the section
        # is the scope boundary, the namespace is the section's name. New in
        # this release, so there is no migration cost to settling it now.
        key="runtime.background_on_resume",
        path=("runtime", "background_on_resume"),
        section="runtime",
        label=_t(_ws.runtime_background_on_resume_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.runtime_background_on_resume_help()),
        choices=_bool_choices(
            _t(_ws.runtime_background_on_resume_choice_true_description()),
            _t(_ws.runtime_background_on_resume_choice_false_description()),
        ),
    ),
    Setting(
        # The two residency knobs of the keep-alive (design-runtime-prewarm §5).
        # LIVE by construction and by the section's own scope: the reaper reads
        # both at the moment it draws a drain window
        # (``session/runtime/process.py:_drain_window_s``), so an edit applies to
        # the NEXT window — a runtime already inside one keeps the window it
        # drew, which is the qualification the section description carries.
        # Neither key is a "new sessions" setting, which is why they are here.
        #
        # The consumer reads them through ``get_nested_value`` on the tuples
        # below, which is the accessor the registry's own path pairs with; the
        # round-trip test is what keeps the two spellings from drifting.
        key="runtime.keep_alive_seconds",
        path=("runtime", "keep_alive_seconds"),
        section="runtime",
        label=_t(_ws.runtime_keep_alive_seconds_label()),
        kind=Kind.INT,
        default=300,
        help=_t(_ws.runtime_keep_alive_seconds_help()),
        minimum=0,
        maximum=3600,
    ),
    Setting(
        key="runtime.keep_alive_max",
        path=("runtime", "keep_alive_max"),
        section="runtime",
        label=_t(_ws.runtime_keep_alive_max_label()),
        kind=Kind.INT,
        default=4,
        # "THIS INSTALL", not "this machine" (QA round 1, Q-5): the cap is
        # enforced over the registry of the runtime's OWN config root, so two
        # installs on one host hold up to two caps between them. See
        # ``process._keep_alive_candidates``.
        help=_t(_ws.runtime_keep_alive_max_help()),
        minimum=1,
        maximum=64,
    ),
    # -- session cleanup policy ---------------------------------------------
    # The ONE way a session directory can be removed automatically, and it is
    # OFF by default. Ordered master-switch first so the page reads as "a
    # switch, then what it controls"; the sub-rows carry a `↳` prefix and say
    # in their help that they are inert while the switch is off. Every row is
    # a genuinely NESTED path under `session.cleanup`, and the consumer
    # (`session/cleanup.py`) reads it back through
    # `ConfigManager.get_nested_value` on the SAME tuple — the previous
    # `session.reap_unused` toggle was written nested here and read flat
    # there, so the opt-out never worked and the reaper it gated deleted 225
    # named sessions. `test_settings_io.py` round-trips every nested setting
    # through that reader to keep this from recurring.
    # Help strings are budgeted: the detail row sheds the help before the
    # key path, so each sub-row LEADS with the gate ("Needs cleanup on.") and
    # stays under ~63 cells (design round 1, D3). The master's help fits the
    # 94-cell budget at 100 cols and names what ON does and what is spared.
    Setting(
        key="session.cleanup.enabled",
        path=("session", "cleanup", "enabled"),
        section="session_cleanup",
        label=_t(_ws.session_cleanup_enabled_label()),
        kind=Kind.BOOL,
        default=False,
        help=(
            _t(_ws.session_cleanup_enabled_help())
        ),
        choices=_bool_choices(
            _t(_ws.session_cleanup_enabled_choice_true_description()),
            _t(_ws.session_cleanup_enabled_choice_false_description()),
        ),
    ),
    Setting(
        key="session.cleanup.max_sessions",
        path=("session", "cleanup", "max_sessions"),
        section="session_cleanup",
        label=_t(_ws.session_cleanup_max_sessions_label()),
        kind=Kind.INT,
        default=0,
        help=_t(_ws.session_cleanup_max_sessions_help()),
        minimum=0,
        gated_by="session.cleanup.enabled",
    ),
    Setting(
        key="session.cleanup.max_inactive_days",
        path=("session", "cleanup", "max_inactive_days"),
        section="session_cleanup",
        label=_t(_ws.session_cleanup_max_inactive_days_label()),
        kind=Kind.INT,
        default=0,
        help=_t(_ws.session_cleanup_max_inactive_days_help()),
        minimum=0,
        gated_by="session.cleanup.enabled",
    ),
    Setting(
        key="session.cleanup.max_total_bytes",
        path=("session", "cleanup", "max_total_bytes"),
        section="session_cleanup",
        label=_t(_ws.session_cleanup_max_total_bytes_label()),
        kind=Kind.INT,
        default=0,
        help=_t(_ws.session_cleanup_max_total_bytes_help()),
        minimum=0,
        gated_by="session.cleanup.enabled",
    ),
    Setting(
        key="session.cleanup.remove_empty",
        path=("session", "cleanup", "remove_empty"),
        section="session_cleanup",
        label=_t(_ws.session_cleanup_remove_empty_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.session_cleanup_remove_empty_help()),
        choices=_bool_choices(_t(_ws.session_cleanup_remove_empty_choice_true_description()), _t(_ws.session_cleanup_remove_empty_choice_false_description())),
        gated_by="session.cleanup.enabled",
    ),
    # -- delegated-work retention --------------------------------------------
    # Nested one level deeper (``session.cleanup.delegated.*``) so the parent
    # keys above can never be mistaken for these and so the two classes read as
    # two blocks in config.yml. The consumer is the same module, through the
    # same ``DELEGATED_PATH`` tuple. ``enabled`` is the one cleanup switch that
    # defaults to TRUE; see ``cleanup.DEFAULT_DELEGATED_ENABLED`` for why.
    Setting(
        key="session.cleanup.delegated.enabled",
        path=("session", "cleanup", "delegated", "enabled"),
        section="session_delegated",
        label=_t(_ws.session_cleanup_delegated_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.session_cleanup_delegated_enabled_help())
        ),
        choices=_bool_choices(
            _t(_ws.session_cleanup_delegated_enabled_choice_true_description()),
            _t(_ws.session_cleanup_delegated_enabled_choice_false_description()),
        ),
    ),
    Setting(
        key="session.cleanup.delegated.max_age_hours",
        path=("session", "cleanup", "delegated", "max_age_hours"),
        section="session_delegated",
        label=_t(_ws.session_cleanup_delegated_max_age_hours_label()),
        kind=Kind.INT,
        default=48,
        # The help carries both limits because the page paints help but not
        # min/max, and "2 to 720" is what a user needs to type a valid value.
        help=_t(_ws.session_cleanup_delegated_max_age_hours_help()),
        minimum=2,
        maximum=720,
        unit="hours",
        validate_value=_validate_delegated_max_age_hours,
        gated_by="session.cleanup.delegated.enabled",
    ),
    Setting(
        key="runtime.unattended_gate_timeout",
        path=("runtime", "unattended_gate_timeout"),
        section="runtime",
        label=_t(_ws.runtime_unattended_gate_timeout_label()),
        kind=Kind.INT,
        default=24,
        help=_t(_ws.runtime_unattended_gate_timeout_help()),
        minimum=0,
        maximum=720,
    ),
    Setting(
        key="subagents.max_running",
        path=("subagents", "max_running"),
        section="subagents",
        label=_t(_ws.subagents_max_running_label()),
        kind=Kind.INT,
        default=15,
        help=_t(_ws.subagents_max_running_help()),
        minimum=1,
        maximum=64,
    ),
    Setting(
        key="subagents.slim_child_knowledge",
        path=("subagents", "slim_child_knowledge"),
        section="subagents",
        label=_t(_ws.subagents_slim_child_knowledge_label()),
        kind=Kind.BOOL,
        # The consumer constant is ``DEFAULT_SLIM_CHILD_KNOWLEDGE`` in
        # ``harness/subagent.py``; ``test_settings_io``'s ``_consumer_defaults``
        # guards the pair, which is what stops this literal and the reader's
        # fallback drifting apart.
        default=True,
        # 60 cells — with the 3-space gutter and the 30-cell key path that is
        # 93 of the 94-cell slot at 100 columns, so the key keeps its place
        # beside the help (the shedding discipline this section sizes to).
        # The sentence carries both halves the row must state: what is
        # trimmed, and the invariant — every guide/skill NAME survives, so
        # nothing a child could have read becomes unfindable, only terser.
        help=_t(_ws.subagents_slim_child_knowledge_help()),
        choices=_bool_choices(_t(_ws.subagents_slim_child_knowledge_choice_true_description()), _t(_ws.subagents_slim_child_knowledge_choice_false_description())),
    ),
    Setting(
        key="subagents.max_team_depth",
        path=("subagents", "max_team_depth"),
        section="subagents",
        label=_t(_ws.subagents_max_team_depth_label()),
        kind=Kind.INT,
        # Literal, like max_running's: the consumer default is
        # ``harness.subagent.DEFAULT_MAX_TEAM_DEPTH``, and the consumer test
        # keeps the two equal. The maximum is ``teams.MAX_ORG_DEPTH``; the
        # reader clamps to it even when the file says more (BEN-7-D3), and
        # ``test_settings_io.test_the_team_depth_maximum_matches_max_org_depth``
        # keeps this literal equal to the constant.
        default=3,
        help=_t(_ws.subagents_max_team_depth_help()),
        minimum=1,
        maximum=8,
    ),
    Setting(
        key="subagents.model_choice",
        path=("subagents", "model_choice"),
        section="subagents",
        label=_t(_ws.subagents_model_choice_label()),
        kind=Kind.ENUM,
        # The literal, not an import: this module reports defaults for the page
        # and deliberately keeps `local_operator.tools.*` off its own import
        # path (`BASH_SHELL_DEFAULT` is restated for the same reason). The
        # consumer constant the value must equal is `DEFAULT_MODEL_CHOICE` in
        # `harness/subagent.py`, and `test_settings_io`'s `_consumer_defaults`
        # guards that pair — which is what stops this literal and the reader's
        # fallback drifting apart into a page that lies about the default.
        default="operator",
        # Says what the ROW decides, never what a capability would do: the help
        # line is one string for both values, and the first version ("Lets a
        # delegating model swap a child onto a configured model tier") described
        # the capability, so at the default it read as the opposite of the
        # stored value — the resting state of the row claimed the picker was
        # open. Length matters twice: the detail line sheds the WHOLE help once
        # the key path stops fitting beside it (settings_view._detail_clause),
        # so this is 66 cells against a 69-cell budget at 100 columns, and the
        # key path keeps its place beside it (66 + 3 + 22 = 91 of the row's 94).
        # The rewrite is length-NEUTRAL: the sentence it replaced also measured
        # 66, so the 80-column rung behaves identically on both sides.
        #
        # It also carries the operator's own route to a deliberate pin, which is
        # the half this row was missing: the model-side pin is refused on
        # purpose (a pin is how the incident stayed invisible), so an operator
        # who wants one strong reviewer must be told there is a place to put it
        # — the ROLE's own profile — rather than handed the picker back.
        help=_t(_ws.subagents_model_choice_help()),
        # Both descriptions are sized to the EXPANDED choice row, which is the
        # only place they render, and they are measured on a RENDERED FRAME at
        # 100 columns — the frame, not the row-text painter, because the
        # expansion's own cursor marker and indent cost cells the row-text
        # arithmetic does not charge: the operator row renders 26 cells of
        # description there (its `(default)` marker takes 11 of the column) and
        # the model row 40. The previous pair measured 93 and 120 painted cells,
        # so "role pins still apply" and "different, costlier model" — the two
        # consequences this row exists to state — were clipped at every width the
        # page is measured at. The pins half now lives in the help above; the
        # money half is here.
        #
        # "inherits the session model" is 26 exactly, which is why it is not
        # "inherits this session's model" (29, and clipped to "…session's mo…").
        choices=(
            Choice(
                "operator",
                _t(_ws.subagents_model_choice_choice_operator_label()),
                _t(_ws.subagents_model_choice_choice_operator_description()),
            ),
            Choice(
                "model",
                _t(_ws.subagents_model_choice_choice_model_label()),
                _t(_ws.subagents_model_choice_choice_model_description()),
            ),
        ),
    ),
    Setting(
        key="subagents.models.lo",
        path=("subagents", "models", "lo"),
        section="subagents",
        label=_t(_ws.subagents_models_lo_label()),
        kind=Kind.TEXT,
        default="",
        # Billing, the sentinel, and what empty does are the three facts this
        # row has to carry. The FIRST and THIRD are the incident's: a
        # deliberate tier pin read as harmless because nothing said a child on
        # it RUNS, and is billed, at that model's rates — and the sentence that
        # used to stand here, "empty inherits", said the opposite of what the
        # code does. Empty (and absent) REMOVE the tier: the schema stops
        # advertising it and the strict launch path REFUSES a role pinned to it
        # rather than quietly inheriting (#635). The SECOND is the explicit
        # opt-in `default`, resolved at launch to the session's current model;
        # without it named here the only spelling an operator had for "use the
        # session model" was the one that deletes the tier.
        #
        # `'default':` rather than `'default'=` is the page's own notation for
        # "this literal means X" (see the retry row's "'default': nothing
        # sent"), and it names the resolved thing the way the neighbouring
        # model_choice row does ("inherits the session model"). The `=` would
        # also read as "the default VALUE is the session model", which is not
        # what the shipped default (`""`) is.
        #
        # On the sentinel row the word `default` carries three referents on the
        # bottom two lines: the VALUE the operator stored, the clause
        # `default: —` (what `r` restores), and the footer's `r default` hint.
        # The help quotes the literal and the inks differ (value `fg`, clause
        # `dim`), which is what keeps them apart; recorded because the collision
        # exists only in this new state.
        #
        # Length is budgeted, not styled, and the budget is TIGHT: the detail
        # line sheds the WHOLE help once the key path no longer fits beside it
        # (settings_view._detail_clause), and at 100 columns that row is 94 cells
        # with `subagents.models.hi` (19) plus its separator taking 22 — 72 cells
        # of help, and this string measures 71. An earlier version measured 73
        # and shed the key path, which the comment beside it wrongly claimed it
        # did not.
        #
        # Measured on rendered frames rather than derived, and on BOTH sides of
        # this change in the `subagents` state: at 100 columns the help and
        # `subagents.models.hi` both paint — but only on a row AT ITS DEFAULT.
        # An OFF-DEFAULT row also carries the `· default: —` clause (about 12
        # more cells), which pushes the key path off the line; that is base
        # behaviour (the sentence this replaced did the same) and not a
        # regression, and the HELP still paints whole at every width. At 80
        # columns the help paints whole and the key path is shed. The frame is
        # the only thing that settles it, which is why the bounds below are
        # measured rather than derived.
        #
        # This line no longer carries the `See subagents.model_choice` pointer
        # the previous wording did: three facts do not fit in 72 cells beside a
        # 26-cell cross-reference, and this row's own contract is what each
        # VALUE does. The row it pointed at is two above, and is the only one
        # labelled "Who picks a subagent's model". The design round reviewed and
        # signed off this trade.
        help=_t(_ws.subagents_models_lo_help()),
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.med",
        path=("subagents", "models", "med"),
        section="subagents",
        label=_t(_ws.subagents_models_med_label()),
        kind=Kind.TEXT,
        default="",
        # Billing, the sentinel, and what empty does are the three facts this
        # row has to carry. The FIRST and THIRD are the incident's: a
        # deliberate tier pin read as harmless because nothing said a child on
        # it RUNS, and is billed, at that model's rates — and the sentence that
        # used to stand here, "empty inherits", said the opposite of what the
        # code does. Empty (and absent) REMOVE the tier: the schema stops
        # advertising it and the strict launch path REFUSES a role pinned to it
        # rather than quietly inheriting (#635). The SECOND is the explicit
        # opt-in `default`, resolved at launch to the session's current model;
        # without it named here the only spelling an operator had for "use the
        # session model" was the one that deletes the tier.
        #
        # `'default':` rather than `'default'=` is the page's own notation for
        # "this literal means X" (see the retry row's "'default': nothing
        # sent"), and it names the resolved thing the way the neighbouring
        # model_choice row does ("inherits the session model"). The `=` would
        # also read as "the default VALUE is the session model", which is not
        # what the shipped default (`""`) is.
        #
        # On the sentinel row the word `default` carries three referents on the
        # bottom two lines: the VALUE the operator stored, the clause
        # `default: —` (what `r` restores), and the footer's `r default` hint.
        # The help quotes the literal and the inks differ (value `fg`, clause
        # `dim`), which is what keeps them apart; recorded because the collision
        # exists only in this new state.
        #
        # Length is budgeted, not styled, and the budget is TIGHT: the detail
        # line sheds the WHOLE help once the key path no longer fits beside it
        # (settings_view._detail_clause), and at 100 columns that row is 94 cells
        # with `subagents.models.hi` (19) plus its separator taking 22 — 72 cells
        # of help, and this string measures 71. An earlier version measured 73
        # and shed the key path, which the comment beside it wrongly claimed it
        # did not.
        #
        # Measured on rendered frames rather than derived, and on BOTH sides of
        # this change in the `subagents` state: at 100 columns the help and
        # `subagents.models.hi` both paint — but only on a row AT ITS DEFAULT.
        # An OFF-DEFAULT row also carries the `· default: —` clause (about 12
        # more cells), which pushes the key path off the line; that is base
        # behaviour (the sentence this replaced did the same) and not a
        # regression, and the HELP still paints whole at every width. At 80
        # columns the help paints whole and the key path is shed. The frame is
        # the only thing that settles it, which is why the bounds below are
        # measured rather than derived.
        #
        # This line no longer carries the `See subagents.model_choice` pointer
        # the previous wording did: three facts do not fit in 72 cells beside a
        # 26-cell cross-reference, and this row's own contract is what each
        # VALUE does. The row it pointed at is two above, and is the only one
        # labelled "Who picks a subagent's model". The design round reviewed and
        # signed off this trade.
        help=_t(_ws.subagents_models_med_help()),
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.hi",
        path=("subagents", "models", "hi"),
        section="subagents",
        label=_t(_ws.subagents_models_hi_label()),
        kind=Kind.TEXT,
        default="",
        # Billing, the sentinel, and what empty does are the three facts this
        # row has to carry. The FIRST and THIRD are the incident's: a
        # deliberate tier pin read as harmless because nothing said a child on
        # it RUNS, and is billed, at that model's rates — and the sentence that
        # used to stand here, "empty inherits", said the opposite of what the
        # code does. Empty (and absent) REMOVE the tier: the schema stops
        # advertising it and the strict launch path REFUSES a role pinned to it
        # rather than quietly inheriting (#635). The SECOND is the explicit
        # opt-in `default`, resolved at launch to the session's current model;
        # without it named here the only spelling an operator had for "use the
        # session model" was the one that deletes the tier.
        #
        # `'default':` rather than `'default'=` is the page's own notation for
        # "this literal means X" (see the retry row's "'default': nothing
        # sent"), and it names the resolved thing the way the neighbouring
        # model_choice row does ("inherits the session model"). The `=` would
        # also read as "the default VALUE is the session model", which is not
        # what the shipped default (`""`) is.
        #
        # On the sentinel row the word `default` carries three referents on the
        # bottom two lines: the VALUE the operator stored, the clause
        # `default: —` (what `r` restores), and the footer's `r default` hint.
        # The help quotes the literal and the inks differ (value `fg`, clause
        # `dim`), which is what keeps them apart; recorded because the collision
        # exists only in this new state.
        #
        # Length is budgeted, not styled, and the budget is TIGHT: the detail
        # line sheds the WHOLE help once the key path no longer fits beside it
        # (settings_view._detail_clause), and at 100 columns that row is 94 cells
        # with `subagents.models.hi` (19) plus its separator taking 22 — 72 cells
        # of help, and this string measures 71. An earlier version measured 73
        # and shed the key path, which the comment beside it wrongly claimed it
        # did not.
        #
        # Measured on rendered frames rather than derived, and on BOTH sides of
        # this change in the `subagents` state: at 100 columns the help and
        # `subagents.models.hi` both paint — but only on a row AT ITS DEFAULT.
        # An OFF-DEFAULT row also carries the `· default: —` clause (about 12
        # more cells), which pushes the key path off the line; that is base
        # behaviour (the sentence this replaced did the same) and not a
        # regression, and the HELP still paints whole at every width. At 80
        # columns the help paints whole and the key path is shed. The frame is
        # the only thing that settles it, which is why the bounds below are
        # measured rather than derived.
        #
        # This line no longer carries the `See subagents.model_choice` pointer
        # the previous wording did: three facts do not fit in 72 cells beside a
        # 26-cell cross-reference, and this row's own contract is what each
        # VALUE does. The row it pointed at is two above, and is the only one
        # labelled "Who picks a subagent's model". The design round reviewed and
        # signed off this trade.
        help=_t(_ws.subagents_models_hi_help()),
        empty_unsets=True,
    ),
    # -- resource classification --------------------------------------------
    #
    # docs/design/classification-layer.md §8. Ordered master-switch first, then
    # what it controls, matching the `session.cleanup.*` block above: the page
    # reads as "a switch, then its knobs", every sub-row leads with the gate
    # (`↳` label, "Needs recommendations on." help), and the numbers here are the
    # §8 defaults the classification package's own readers fall back to —
    # `_consumer_defaults()` in tests/unit/test_settings_io.py binds the two so
    # neither can drift.
    #
    # Scope is NEW_SESSIONS on the SECTION above; the reason is recorded there
    # rather than repeated per row.
    Setting(
        key="classification.auto",
        path=("classification", "auto"),
        section="classification",
        label=_t(_ws.classification_auto_label()),
        kind=Kind.BOOL,
        # Default ON since 2026-09-18, matching the package's own `DEFAULT_AUTO`
        # (`classification/service.py`, which records why the flip is worth its
        # spend). The OFF case stays a real path and stays free: with
        # `auto: false` the wiring returns before it imports the package, so the
        # prompt is byte-identical to a harness without the layer.
        #
        # The flip was a deliberate change of behaviour, so it is stated once
        # here in the comment that owns the default, and nowhere else in this
        # file. The help says what ON does rather than what the feature is: the
        # label already names the feature, and it now names it in the one term
        # the tips and `guide://classification` use (design round 2, D11).
        default=True,
        help=_t(_ws.classification_auto_help()),
        choices=_bool_choices(
            _t(_ws.classification_auto_choice_true_description()),
            _t(_ws.classification_auto_choice_false_description()),
        ),
    ),
    Setting(
        key="classification.vendor",
        path=("classification", "vendor"),
        section="classification",
        label=_t(_ws.classification_vendor_label()),
        kind=Kind.ENUM,
        default="auto",
        # `auto` is a real member here rather than the unset spelling
        # `model_effort` uses: the cascade's own default IS "first leg with a
        # usable credential", and storing that as an empty string would leave
        # the page unable to show which of the two the operator chose.
        help=_t(_ws.classification_vendor_help()),
        choices=(
            Choice("auto", _t(_ws.classification_vendor_choice_auto_label()), _t(_ws.classification_vendor_choice_auto_description())),
            Choice("radient", _t(_ws.classification_vendor_choice_radient_label()), _t(_ws.classification_vendor_choice_radient_description())),
            Choice("typesafe", _t(_ws.classification_vendor_choice_typesafe_label()), _t(_ws.classification_vendor_choice_typesafe_description())),
            Choice("openrouter", _t(_ws.classification_vendor_choice_openrouter_label()), _t(_ws.classification_vendor_choice_openrouter_description())),
        ),
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.model",
        path=("classification", "model"),
        section="classification",
        label=_t(_ws.classification_model_label()),
        kind=Kind.TEXT,
        default="",
        help=_t(_ws.classification_model_help()),
        empty_unsets=True,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.timeoutMs",
        path=("classification", "timeoutMs"),
        section="classification",
        label=_t(_ws.classification_timeoutms_label()),
        kind=Kind.INT,
        default=1500,
        # 0 is "use the default", not "no deadline": the reader refuses a
        # non-positive value rather than treating it as unlimited, so a hand-edit
        # cannot leave a turn parked on a vendor. The help says so.
        help=_t(_ws.classification_timeoutms_help()),
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.waitMs",
        path=("classification", "waitMs"),
        section="classification",
        label=_t(_ws.classification_waitms_label()),
        kind=Kind.INT,
        default=50,
        # The operator's latency budget, in the one number that enforces it: how
        # long a turn will WAIT for an answer. Deliberately NOT the call's
        # deadline — a call that misses this window is left running and its
        # answer is delivered by a later message (contract §5a, and
        # ``session_factory._harvest_classification``), so raising this buys a
        # fresher prompt at the price of turn latency and lowering it pushes more
        # answers onto the following turn. The vendor's own model time is excluded
        # from the budget by its terms, which is exactly what makes 50 ms
        # achievable while ``timeoutMs`` stays at 1500. 0 is "use the default",
        # like every other number in this section, never "wait forever".
        help=_t(_ws.classification_waitms_help()),
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.maxStateChars",
        path=("classification", "maxStateChars"),
        section="classification",
        label=_t(_ws.classification_maxstatechars_label()),
        kind=Kind.INT,
        default=6000,
        help=_t(_ws.classification_maxstatechars_help()),
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.maxCandidates",
        path=("classification", "maxCandidates"),
        section="classification",
        label=_t(_ws.classification_maxcandidates_label()),
        kind=Kind.INT,
        default=12,
        help=_t(_ws.classification_maxcandidates_help()),
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.maxRecommendations",
        path=("classification", "maxRecommendations"),
        section="classification",
        label=_t(_ws.classification_maxrecommendations_label()),
        kind=Kind.INT,
        default=3,
        help=_t(_ws.classification_maxrecommendations_help()),
        minimum=0,
        gated_by="classification.auto",
    ),
    # ``classification.notice`` used to sit here. The layer no longer renders a line
    # into the transcript at all (the operator asked for it gone: it is an internal
    # resource-selection step, and it was landing under the reply), so the switch has
    # nothing left to gate — see the retired row below.
    # -- monitors ------------------------------------------------------------
    #
    # docs/design/monitor-tool.md §16. Scope is NEW_SESSIONS on the SECTION
    # above (the scheduler is built per session and handed a snapshot), so an
    # edit lands on the next session start — the classification block's scope
    # and reason, verbatim. The numbers are the §16 defaults the monitor
    # package's own readers fall back to; `_monitor_consumer_defaults()` in
    # tests/unit/test_settings_io.py binds the two so neither can drift.
    #
    # 0 means "use the default" for every number here, exactly like the
    # classification rows: the readers refuse a non-positive value rather than
    # honouring it, so a hand-edit cannot leave a 0-second poll.
    Setting(
        key="monitor.defaultIntervalS",
        path=("monitor", "defaultIntervalS"),
        section="monitor",
        label=_t(_ws.monitor_defaultintervals_label()),
        kind=Kind.INT,
        default=60,
        help=_t(_ws.monitor_defaultintervals_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.maxMonitors",
        path=("monitor", "maxMonitors"),
        section="monitor",
        label=_t(_ws.monitor_maxmonitors_label()),
        kind=Kind.INT,
        default=8,
        help=_t(_ws.monitor_maxmonitors_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.runTimeoutMs",
        path=("monitor", "runTimeoutMs"),
        section="monitor",
        label=_t(_ws.monitor_runtimeoutms_label()),
        kind=Kind.INT,
        default=120000,
        help=_t(_ws.monitor_runtimeoutms_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.snapshotMaxChars",
        path=("monitor", "snapshotMaxChars"),
        section="monitor",
        label=_t(_ws.monitor_snapshotmaxchars_label()),
        kind=Kind.INT,
        default=32768,
        help=_t(_ws.monitor_snapshotmaxchars_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.maxDeltaLines",
        path=("monitor", "maxDeltaLines"),
        section="monitor",
        label=_t(_ws.monitor_maxdeltalines_label()),
        kind=Kind.INT,
        default=12,
        help=_t(_ws.monitor_maxdeltalines_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.deltaMaxChars",
        path=("monitor", "deltaMaxChars"),
        section="monitor",
        label=_t(_ws.monitor_deltamaxchars_label()),
        kind=Kind.INT,
        default=1200,
        help=_t(_ws.monitor_deltamaxchars_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.classifyMaxChars",
        path=("monitor", "classifyMaxChars"),
        section="monitor",
        label=_t(_ws.monitor_classifymaxchars_label()),
        kind=Kind.INT,
        default=1200,
        help=(
            _t(_ws.monitor_classifymaxchars_help())
        ),
        minimum=0,
    ),
    Setting(
        key="monitor.maxConsecutiveFailures",
        path=("monitor", "maxConsecutiveFailures"),
        section="monitor",
        label=_t(_ws.monitor_maxconsecutivefailures_label()),
        kind=Kind.INT,
        default=5,
        help=_t(_ws.monitor_maxconsecutivefailures_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.maxDeliveriesPerHour",
        path=("monitor", "maxDeliveriesPerHour"),
        section="monitor",
        label=_t(_ws.monitor_maxdeliveriesperhour_label()),
        kind=Kind.INT,
        default=12,
        help=_t(_ws.monitor_maxdeliveriesperhour_help()),
        minimum=0,
    ),
    Setting(
        key="monitor.normalizeTimestamps",
        path=("monitor", "normalizeTimestamps"),
        section="monitor",
        label=_t(_ws.monitor_normalizetimestamps_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.monitor_normalizetimestamps_help()),
        choices=_bool_choices(
            _t(_ws.monitor_normalizetimestamps_choice_true_description()),
            _t(_ws.monitor_normalizetimestamps_choice_false_description()),
        ),
    ),
    # -- turn supplements ("Highlights") --------------------------------------
    #
    # docs/design/turn-supplements.md §2.12. Defaults live beside their consumer in
    # ``supplements/policy.py``; ``_consumer_defaults()`` in tests/unit/test_settings_io.py
    # binds the two. ``graphics`` is OFF by default until a surface can render it (§6).
    Setting(
        key="supplements.enabled",
        path=("supplements", "enabled"),
        section="supplements",
        label=_t(_ws.supplements_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.supplements_enabled_help()),
        choices=_bool_choices(
            _t(_ws.supplements_enabled_choice_true_description()),
            _t(_ws.supplements_enabled_choice_false_description()),
        ),
    ),
    Setting(
        key="supplements.files",
        path=("supplements", "files"),
        section="supplements",
        label=_t(_ws.supplements_files_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.supplements_files_help()),
        choices=_bool_choices(_t(_ws.supplements_files_choice_true_description()), _t(_ws.supplements_files_choice_false_description())),
        gated_by="supplements.enabled",
    ),
    Setting(
        key="supplements.graphics",
        path=("supplements", "graphics"),
        section="supplements",
        label=_t(_ws.supplements_graphics_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.supplements_graphics_help()),
        choices=_bool_choices(
            _t(_ws.supplements_graphics_choice_true_description()), _t(_ws.supplements_graphics_choice_false_description())
        ),
        gated_by="supplements.enabled",
    ),
    Setting(
        key="supplements.model",
        path=("supplements", "model"),
        section="supplements",
        label=_t(_ws.supplements_model_label()),
        kind=Kind.TEXT,
        default="auto",
        help=_t(_ws.supplements_model_help()),
        gated_by="supplements.graphics",
    ),
    Setting(
        key="supplements.maxTurns",
        path=("supplements", "maxTurns"),
        section="supplements",
        label=_t(_ws.supplements_maxturns_label()),
        kind=Kind.INT,
        default=2,
        help=_t(_ws.supplements_maxturns_help()),
        minimum=1,
        maximum=4,
        gated_by="supplements.graphics",
    ),
    Setting(
        key="supplements.maxOutputTokens",
        path=("supplements", "maxOutputTokens"),
        section="supplements",
        label=_t(_ws.supplements_maxoutputtokens_label()),
        kind=Kind.INT,
        default=6000,
        help=_t(_ws.supplements_maxoutputtokens_help()),
        # Mirrors ``policy.MAX_OUTPUT_TOKENS_BOUNDS``, the reader's accept range: with no
        # page maximum the registry accepted values ``from_values`` substitutes with the
        # default (round-1 R4: 2,000,000 came back as 6000). The pair is pinned by
        # ``test_supplements_registry_bounds_are_honoured_by_the_reader``.
        minimum=1,
        maximum=1_000_000,
        gated_by="supplements.graphics",
    ),
    Setting(
        key="supplements.timeoutS",
        path=("supplements", "timeoutS"),
        section="supplements",
        label=_t(_ws.supplements_timeouts_label()),
        kind=Kind.INT,
        default=90,
        help=_t(_ws.supplements_timeouts_help()),
        minimum=1,
        maximum=3600,
        gated_by="supplements.enabled",
    ),
    Setting(
        key="supplements.maxCostUsd",
        path=("supplements", "maxCostUsd"),
        section="supplements",
        label=_t(_ws.supplements_maxcostusd_label()),
        kind=Kind.FLOAT,
        default=1.00,
        help=_t(_ws.supplements_maxcostusd_help()),
        # ``0`` is a VALID value -- "never spend" -- and the reader honours it as stored
        # (round-1 R4); the range is only the page's control window, not the reader's
        # limit (a hand-edited higher cap stays the user's own stated guard). The literal
        # must equal ``policy.DEFAULT_MAX_COST_USD`` (the defaults-drift test compares
        # them); $1.00 since the operator directive of 2026-10-10 (was $0.20).
        minimum=0.0,
        maximum=100.0,
        gated_by="supplements.graphics",
    ),
    Setting(
        key="supplements.maxFeatured",
        path=("supplements", "maxFeatured"),
        section="supplements",
        label=_t(_ws.supplements_maxfeatured_label()),
        kind=Kind.INT,
        default=4,
        help=_t(_ws.supplements_maxfeatured_help()),
        minimum=1,
        maximum=12,
        gated_by="supplements.enabled",
    ),
    Setting(
        key="supplements.denyPrefixes",
        path=("supplements", "denyPrefixes"),
        section="supplements",
        label=_t(_ws.supplements_denyprefixes_label()),
        kind=Kind.LIST,
        default=[],
        help=(
            _t(_ws.supplements_denyprefixes_help())
        ),
        placeholder=_t(_ws.supplements_denyprefixes_placeholder()),
        empty_unsets=True,
        gated_by="supplements.enabled",
    ),
    # -- fork ---------------------------------------------------------------
    #
    # Both paths are genuinely NESTED two-element tuples, not flat dotted keys.
    # The ``display.*`` trap above applies only to keys ``tui/settings.py`` reads
    # as literal dotted top-level keys; nothing reads ``values["fork.mode"]``,
    # and declaring it that way would write a key nothing reads while looking
    # like success from every angle.
    Setting(
        key="fork.mode",
        path=("fork", "mode"),
        section="fork",
        label=_t(_ws.fork_mode_label()),
        kind=Kind.ENUM,
        default="switch",
        help=_t(_ws.fork_mode_help()),
        choices=(
            Choice(
                "window",
                _t(_ws.fork_mode_choice_window_label()),
                _t(_ws.fork_mode_choice_window_description()),
            ),
            Choice("switch", _t(_ws.fork_mode_choice_switch_label()), _t(_ws.fork_mode_choice_switch_description())),
        ),
    ),
    Setting(
        key="fork.cmux_placement",
        path=("fork", "cmux_placement"),
        section="fork",
        # Leads with the CONDITION, not the tool name: this row is a sub-clause
        # of "Where a fork opens" above it, and "placement" is a word that
        # appears nowhere else a user has seen. The help text already phrased it
        # this way; the label was lagging behind it.
        label=_t(_ws.fork_cmux_placement_label()),
        kind=Kind.ENUM,
        default="workspace",
        help=_t(_ws.fork_cmux_placement_help()),
        choices=(
            Choice("workspace", _t(_ws.fork_cmux_placement_choice_workspace_label()), _t(_ws.fork_cmux_placement_choice_workspace_description())),
            Choice("surface", _t(_ws.fork_cmux_placement_choice_surface_label()), _t(_ws.fork_cmux_placement_choice_surface_description())),
        ),
    ),
    # -- compaction ---------------------------------------------------------
    Setting(
        key="compaction.enabled",
        path=("compaction", "enabled"),
        section="compaction",
        label=_t(_ws.compaction_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.compaction_enabled_help()),
        choices=_bool_choices(_t(_ws.compaction_enabled_choice_true_description()), _t(_ws.compaction_enabled_choice_false_description())),
    ),
    Setting(
        key="compaction.strategy",
        path=("compaction", "strategy"),
        section="compaction",
        label=_t(_ws.compaction_strategy_label()),
        kind=Kind.ENUM,
        default="auto",
        help=_t(_ws.compaction_strategy_help()),
        choices=(
            Choice("auto", _t(_ws.compaction_strategy_choice_auto_label()), _t(_ws.compaction_strategy_choice_auto_description())),
            Choice("context-full", _t(_ws.compaction_strategy_choice_context_full_label()), _t(_ws.compaction_strategy_choice_context_full_description())),
            Choice("snapcompact", _t(_ws.compaction_strategy_choice_snapcompact_label()), _t(_ws.compaction_strategy_choice_snapcompact_description())),
            Choice("off", _t(_ws.compaction_strategy_choice_off_label()), _t(_ws.compaction_strategy_choice_off_description())),
        ),
    ),
    Setting(
        key="compaction.threshold_percent",
        path=("compaction", "threshold_percent"),
        section="compaction",
        label=_t(_ws.compaction_threshold_percent_label()),
        kind=Kind.FLOAT,
        default=0.80,
        help=_t(_ws.compaction_threshold_percent_help()),
        minimum=0.0,
        maximum=100.0,
    ),
    Setting(
        key="compaction.threshold_tokens",
        path=("compaction", "threshold_tokens"),
        section="compaction",
        label=_t(_ws.compaction_threshold_tokens_label()),
        kind=Kind.INT,
        # Literal, not DEFAULT_THRESHOLD_TOKENS: importing local_operator.compaction
        # here drags the whole pass engine into every settings read. A unit test
        # pins this to the constant so the two cannot drift.
        default=400_000,
        help=_t(_ws.compaction_threshold_tokens_help()),
        minimum=1,
    ),
    Setting(
        key="compaction.keep_recent_tokens",
        path=("compaction", "keep_recent_tokens"),
        section="compaction",
        label=_t(_ws.compaction_keep_recent_tokens_label()),
        kind=Kind.INT,
        default=20_000,
        help=_t(_ws.compaction_keep_recent_tokens_help()),
        minimum=0,
    ),
    Setting(
        key="compaction.auto_continue",
        path=("compaction", "auto_continue"),
        section="compaction",
        label=_t(_ws.compaction_auto_continue_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.compaction_auto_continue_help()),
        choices=_bool_choices(_t(_ws.compaction_auto_continue_choice_true_description()), _t(_ws.compaction_auto_continue_choice_false_description())),
    ),
    Setting(
        key="compaction.mid_turn_enabled",
        path=("compaction", "mid_turn_enabled"),
        section="compaction",
        label=_t(_ws.compaction_mid_turn_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.compaction_mid_turn_enabled_help()),
        choices=_bool_choices(_t(_ws.compaction_mid_turn_enabled_choice_true_description()), _t(_ws.compaction_mid_turn_enabled_choice_false_description())),
    ),
    # The two BYTE knobs. They live in this section because they are compaction
    # triggers, but they measure a different thing from every other key here:
    # request SIZE, not context occupancy. A screenshot-heavy conversation can
    # sit at 15% of a 1M-token window and still exceed a provider's request
    # cap, because images are billed by pixel area and so carry a flat token
    # charge regardless of their byte length. The labels say "MB" for that
    # reason — a bare number here would read as tokens like its neighbours.
    Setting(
        key="compaction.wire_bytes_budget",
        path=("compaction", "wire_bytes_budget"),
        section="compaction",
        label=_t(_ws.compaction_wire_bytes_budget_label()),
        kind=Kind.INT,
        default=24_000_000,
        help=(
            _t(_ws.compaction_wire_bytes_budget_help())
        ),
        minimum=0,
    ),
    Setting(
        key="compaction.wire_bytes_trigger",
        path=("compaction", "wire_bytes_trigger"),
        section="compaction",
        label=_t(_ws.compaction_wire_bytes_trigger_label()),
        kind=Kind.INT,
        default=16_000_000,
        help=(
            _t(_ws.compaction_wire_bytes_trigger_help())
        ),
        minimum=0,
    ),
    # -- web search ---------------------------------------------------------
    # The two ``enabled`` flags sit in ``web_tools``, apart from the knobs that
    # share their YAML block, for reading order (the gate before what it
    # gates). The startup snapshot decides the BOOT inventory; execution
    # re-checks ``enabled`` per call; a flip is reconciled into a top-level
    # session's inventory at its next turn.
    Setting(
        key="web_search.enabled",
        path=("web_search", "enabled"),
        section="web_tools",
        label=_t(_ws.web_search_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.web_search_enabled_help()),
        choices=_bool_choices(_t(_ws.web_search_enabled_choice_true_description()), _t(_ws.web_search_enabled_choice_false_description())),
    ),
    Setting(
        key="web_search.strategy",
        path=("web_search", "strategy"),
        section="web_search",
        label=_t(_ws.web_search_strategy_label()),
        kind=Kind.ENUM,
        default="round_robin",
        help=_t(_ws.web_search_strategy_help()),
        choices=(
            Choice("round_robin", _t(_ws.web_search_strategy_choice_round_robin_label()), _t(_ws.web_search_strategy_choice_round_robin_description())),
            Choice("ordered", _t(_ws.web_search_strategy_choice_ordered_label()), _t(_ws.web_search_strategy_choice_ordered_description())),
        ),
    ),
    Setting(
        key="web_search.providers",
        path=("web_search", "providers"),
        section="web_search",
        label=_t(_ws.web_search_providers_label()),
        kind=Kind.LIST,
        default=["duckduckgo", "tavily"],
        # A PRIORITY PREFIX, not an allowlist: naming a provider here means "try
        # this first", and every other usable provider still follows in its
        # automatic band. The help says so, because a user who reads the list as
        # "exactly these" would be surprised by the chain -- and that surprise is
        # the documented cost of not writing a freezing migration.
        help=_t(_ws.web_search_providers_help()),
        members=(
            "duckduckgo",
            "tavily",
            "deepseek",
            "perplexity",
            "brave",
            "exa",
            "parallel",
            "serpapi",
            "searxng",
        ),
    ),
    Setting(
        key="web_search.excluded_providers",
        path=("web_search", "excluded_providers"),
        section="web_search",
        label=_t(_ws.web_search_excluded_providers_label()),
        kind=Kind.LIST,
        default=[],
        # `empty_unsets` is required HERE and is exactly what is wrong for
        # web_search.providers: [] is this key's DEFAULT (nothing excluded), so an
        # empty field clears the key rather than failing validation the way an
        # empty priority list does.
        empty_unsets=True,
        help=_t(_ws.web_search_excluded_providers_help()),
        members=(
            "duckduckgo",
            "tavily",
            "deepseek",
            "perplexity",
            "brave",
            "exa",
            "parallel",
            "serpapi",
            "searxng",
        ),
    ),
    Setting(
        key="web_search.timeout_seconds",
        path=("web_search", "timeout_seconds"),
        section="web_search",
        label=_t(_ws.web_search_timeout_seconds_label()),
        kind=Kind.FLOAT,
        default=20.0,
        help=_t(_ws.web_search_timeout_seconds_help()),
        minimum=1.0,
        maximum=120.0,
    ),
    Setting(
        key="web_search.searxng_endpoint",
        path=("web_search", "searxng_endpoint"),
        section="web_search",
        label=_t(_ws.web_search_searxng_endpoint_label()),
        kind=Kind.TEXT,
        default="",
        help=_t(_ws.web_search_searxng_endpoint_help()),
    ),
    Setting(
        key="web_search.deepseek_evidence",
        path=("web_search", "deepseek_evidence"),
        section="web_search",
        label=_t(_ws.web_search_deepseek_evidence_label()),
        kind=Kind.BOOL,
        default=False,
        # Off by default: it is a SECOND model turn (measured 4-11s on top of the
        # search) that buys a verbatim quote and a relevance score per source,
        # for the "which page do I fetch next" decision. Only the deepseek
        # provider consumes it; every other provider already returns snippets.
        help=_t(_ws.web_search_deepseek_evidence_help()),
    ),
    Setting(
        key="web_search.read_enabled",
        path=("web_search", "read_enabled"),
        section="web_search",
        label=_t(_ws.web_search_read_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        # On by default because it is inert until used: the tool refuses (telling
        # the model to fetch instead) whenever no readable page context exists,
        # so a session that never uses it pays nothing but a tool schema.
        help=_t(_ws.web_search_read_enabled_help()),
    ),
    # -- web fetch ----------------------------------------------------------
    Setting(
        key="web_fetch.enabled",
        path=("web_fetch", "enabled"),
        section="web_tools",
        label=_t(_ws.web_fetch_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        # 60 cells — see the note on `hosting`. No BACKTICKS: the footer is a
        # plain `Text`, so they render as literal characters, and this was the
        # only one of 57 help strings carrying any (design round 1, D5).
        help=_t(_ws.web_fetch_enabled_help()),
        choices=_bool_choices(_t(_ws.web_fetch_enabled_choice_true_description()), _t(_ws.web_fetch_enabled_choice_false_description())),
    ),
    Setting(
        key="web_fetch.timeout_seconds",
        path=("web_fetch", "timeout_seconds"),
        section="web_fetch",
        label=_t(_ws.web_fetch_timeout_seconds_label()),
        kind=Kind.FLOAT,
        default=20.0,
        help=_t(_ws.web_fetch_timeout_seconds_help()),
        minimum=1.0,
        maximum=300.0,
    ),
    Setting(
        key="web_fetch.max_bytes",
        path=("web_fetch", "max_bytes"),
        section="web_fetch",
        label=_t(_ws.web_fetch_max_bytes_label()),
        kind=Kind.INT,
        default=5 * 1024 * 1024,
        help=_t(_ws.web_fetch_max_bytes_help()),
        minimum=1024,
    ),
    Setting(
        key="web_fetch.max_redirects",
        path=("web_fetch", "max_redirects"),
        section="web_fetch",
        label=_t(_ws.web_fetch_max_redirects_label()),
        kind=Kind.INT,
        default=5,
        help=_t(_ws.web_fetch_max_redirects_help()),
        minimum=0,
        maximum=50,
    ),
    Setting(
        key="web_fetch.cache_ttl_seconds",
        path=("web_fetch", "cache_ttl_seconds"),
        section="web_fetch",
        label=_t(_ws.web_fetch_cache_ttl_seconds_label()),
        kind=Kind.INT,
        default=900,
        help=_t(_ws.web_fetch_cache_ttl_seconds_help()),
        minimum=0,
    ),
    Setting(
        key="web_fetch.allow_private",
        path=("web_fetch", "allow_private"),
        section="web_fetch",
        label=_t(_ws.web_fetch_allow_private_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.web_fetch_allow_private_help()),
        choices=_bool_choices(_t(_ws.web_fetch_allow_private_choice_true_description()), _t(_ws.web_fetch_allow_private_choice_false_description())),
    ),
    Setting(
        key="web_fetch.render_backend",
        path=("web_fetch", "render_backend"),
        section="web_fetch",
        label=_t(_ws.web_fetch_render_backend_label()),
        kind=Kind.ENUM,
        default="auto",
        help=_t(_ws.web_fetch_render_backend_help()),
        choices=(
            Choice("auto", _t(_ws.web_fetch_render_backend_choice_auto_label()), _t(_ws.web_fetch_render_backend_choice_auto_description())),
            Choice("stdlib", _t(_ws.web_fetch_render_backend_choice_stdlib_label()), _t(_ws.web_fetch_render_backend_choice_stdlib_description())),
        ),
    ),
    Setting(
        key="web_fetch.enrich",
        path=("web_fetch", "enrich"),
        section="web_fetch",
        label=_t(_ws.web_fetch_enrich_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.web_fetch_enrich_help()),
        choices=_bool_choices(_t(_ws.web_fetch_enrich_choice_true_description()), _t(_ws.web_fetch_enrich_choice_false_description())),
    ),
    Setting(
        key="web_fetch.max_attempts",
        path=("web_fetch", "max_attempts"),
        section="web_fetch",
        label=_t(_ws.web_fetch_max_attempts_label()),
        kind=Kind.INT,
        default=3,
        help=_t(_ws.web_fetch_max_attempts_help()),
        minimum=1,
        maximum=5,
    ),
    Setting(
        key="web_fetch.blocked_retry",
        path=("web_fetch", "blocked_retry"),
        section="web_fetch",
        label=_t(_ws.web_fetch_blocked_retry_label()),
        kind=Kind.BOOL,
        default=True,
        help=_t(_ws.web_fetch_blocked_retry_help()),
        choices=_bool_choices(_t(_ws.web_fetch_blocked_retry_choice_true_description()), _t(_ws.web_fetch_blocked_retry_choice_false_description())),
    ),
    # -- tools --------------------------------------------------------------
    # ``path`` mirrors ``tools.builtin.BASH_SHELL_PATH``; the two are pinned
    # together by ``test_bash_shell_row_shares_the_consumer_path`` rather than
    # imported, because this module must stay cheap for the CLI and
    # ``tools.builtin`` is not (see the module docstring on Textual).
    # -- hooks ------------------------------------------------------------
    # ``path`` mirrors ``hook_forwarding.NATIVE_PATH`` / ``FORWARD_*_PATH``.
    Setting(
        key="hooks.native",
        path=("hooks", "native"),
        section="hooks",
        label=_t(_ws.hooks_native_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.hooks_native_help()),
    ),
    Setting(
        key="hooks.forward_claude",
        path=("hooks", "forward_claude"),
        section="hooks",
        label=_t(_ws.hooks_forward_claude_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.hooks_forward_claude_help()),
    ),
    Setting(
        key="hooks.forward_codex",
        path=("hooks", "forward_codex"),
        section="hooks",
        label=_t(_ws.hooks_forward_codex_label()),
        kind=Kind.BOOL,
        default=False,
        help=_t(_ws.hooks_forward_codex_help()),
    ),
    Setting(
        key="bash.shell",
        path=("bash", "shell"),
        section="tools",
        label=_t(_ws.bash_shell_label()),
        kind=Kind.TEXT,
        default="",
        help=_BASH_SHELL_HELP,
        empty_unsets=True,
    ),
    # Cross-session SENDS. A `send` row rather than a `tools.send.*` one because
    # that is the key the design named and what an operator will look for in
    # ``config.yml``; it lives in this SECTION because "how a tool behaves" is
    # what the section already means and its Scope is uniform (LIVE) — the gate
    # is read per send, so an edit lands on the next one.
    #
    # OFF is the shipped behaviour and the designed default: the tool result
    # already carries the cause, the message id and the retry advice, so this is
    # the operator's opt-in SECOND copy of a fact the model has already been
    # told (design note B). The default constant the reader uses is
    # ``peer_send.JOURNAL_UNCONFIRMED_DEFAULT``, pinned to this row by
    # ``tests/unit/test_settings_io.py``.
    Setting(
        key="send.journal_unconfirmed",
        path=("send", "journal_unconfirmed"),
        section="tools",
        # Named for what it DOES, not for one of its two states (UX round 1,
        # U7): the hook fires for BOTH amber states -- ``mailbox`` and
        # ``unconfirmed`` -- and mailbox is the common one (the incident shape at
        # ~5 s), so a label that said "unconfirmed" sent an operator who wanted
        # the wake-failed notice looking in the wrong place.
        label=_t(_ws.send_journal_unconfirmed_label()),
        kind=Kind.BOOL,
        default=False,
        help=(
            _t(_ws.send_journal_unconfirmed_help())
        ),
    ),
    # -- search_interception ------------------------------------------------
    # ``path`` mirrors ``tools.builtin.SEARCH_INTERCEPTION_*_PATH`` (pinned
    # together by ``test_search_interception_rows_share_the_consumer_paths``,
    # the same split the bash.shell row uses).
    #
    # These live in ``tools`` rather than a section of their own because that
    # section is already "how a tool executes" and its ``Scope`` is uniform
    # (LIVE) — ``_search_interception_config`` reads a fresh ``ConfigManager``
    # on every ``bash`` call, so an edit lands on the next command.
    #
    # Three rows rather than one: a guard that REFUSES a command the model wrote
    # deserves a master switch, a warn-only arm so an operator can watch before
    # enforcing, and a separate lever for the ripgrep default-prune.
    Setting(
        key="tools.search_interception.enabled",
        path=("tools", "search_interception", "enabled"),
        section="tools",
        label=_t(_ws.tools_search_interception_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.tools_search_interception_enabled_help())
        ),
    ),
    Setting(
        key="tools.search_interception.block",
        path=("tools", "search_interception", "block"),
        label=_t(_ws.tools_search_interception_block_label()),
        section="tools",
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.tools_search_interception_block_help())
        ),
    ),
    Setting(
        key="tools.search_interception.rg_excludes",
        path=("tools", "search_interception", "rg_excludes"),
        label=_t(_ws.tools_search_interception_rg_excludes_label()),
        section="tools",
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.tools_search_interception_rg_excludes_help())
        ),
    ),
    # LIVE: ``Session._apply_config_change`` re-reads it into the publish
    # filter, so the next turn's tools array follows the edit. Path mirrors
    # ``tools.deferral.TOOL_DEFERRAL_PATH`` (not imported: this module must
    # stay cheap for the CLI).
    Setting(
        key="tools.defer",
        path=("tools", "defer"),
        section="tools",
        label=_t(_ws.tools_defer_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.tools_defer_help())
        ),
    ),
    # -- memory_guard -------------------------------------------------------
    # ``path`` mirrors ``memory_guard.BASH_MEMORY_*_PATH``; the four are pinned
    # together by ``test_memory_guard_rows_share_the_consumer_paths`` rather than
    # imported, for the same reason the row above is not (this module must stay
    # cheap for the CLI, ``tools.builtin``/``memory_guard`` must not be pulled in).
    # LIVE: the reader is a fresh ConfigManager per command (see the section).
    Setting(
        key="bash.memory.enabled",
        path=("bash", "memory", "enabled"),
        section="memory_guard",
        label=_t(_ws.bash_memory_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.bash_memory_enabled_help())
        ),
        choices=_bool_choices(_t(_ws.bash_memory_enabled_choice_true_description()), _t(_ws.bash_memory_enabled_choice_false_description())),
    ),
    Setting(
        key="bash.memory.mode",
        path=("bash", "memory", "mode"),
        section="memory_guard",
        label=_t(_ws.bash_memory_mode_label()),
        kind=Kind.ENUM,
        default="auto",
        help=(
            _t(_ws.bash_memory_mode_help())
        ),
        choices=(
            Choice("auto", _t(_ws.bash_memory_mode_choice_auto_label()), _t(_ws.bash_memory_mode_choice_auto_description())),
            Choice("manual", _t(_ws.bash_memory_mode_choice_manual_label()), _t(_ws.bash_memory_mode_choice_manual_description())),
        ),
    ),
    Setting(
        key="bash.memory.limit_mb",
        path=("bash", "memory", "limit_mb"),
        section="memory_guard",
        label=_t(_ws.bash_memory_limit_mb_label()),
        kind=Kind.INT,
        default=0,
        help=(
            _t(_ws.bash_memory_limit_mb_help())
        ),
    ),
    Setting(
        key="bash.memory.soft_fraction",
        path=("bash", "memory", "soft_fraction"),
        section="memory_guard",
        label=_t(_ws.bash_memory_soft_fraction_label()),
        kind=Kind.FLOAT,
        default=0.8,
        help=(
            _t(_ws.bash_memory_soft_fraction_help())
        ),
    ),
    # -- query_budget -------------------------------------------------------
    # ``path`` mirrors ``query_budget.QUERY_BUDGET_*_PATH`` in
    # ``tools/query_budget.py``; the three are pinned together by
    # ``test_query_budget_rows_share_the_consumer_paths`` rather than imported,
    # for the same reason the rows above are not (this module must stay cheap for
    # the CLI, ``tools.builtin``/``tools.query_budget`` must not be pulled in).
    #
    # Three rows rather than one, mirroring the search-interception trio: a guard
    # that KILLS a command the model wrote deserves a master switch, a warn-only
    # arm so an operator can watch before enforcing, and the number itself.
    Setting(
        key="bash.query_budget.enabled",
        path=("bash", "query_budget", "enabled"),
        section="query_budget",
        label=_t(_ws.bash_query_budget_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.bash_query_budget_enabled_help())
        ),
        choices=_bool_choices(_t(_ws.bash_query_budget_enabled_choice_true_description()), _t(_ws.bash_query_budget_enabled_choice_false_description())),
    ),
    Setting(
        key="bash.query_budget.stop",
        path=("bash", "query_budget", "stop"),
        section="query_budget",
        label=_t(_ws.bash_query_budget_stop_label()),
        kind=Kind.BOOL,
        default=True,
        help=(
            _t(_ws.bash_query_budget_stop_help())
        ),
        choices=_bool_choices(_t(_ws.bash_query_budget_stop_choice_true_description()), _t(_ws.bash_query_budget_stop_choice_false_description())),
    ),
    Setting(
        key="bash.query_budget.seconds",
        path=("bash", "query_budget", "seconds"),
        section="query_budget",
        label=_t(_ws.bash_query_budget_seconds_label()),
        kind=Kind.INT,
        default=60,
        help=(
            _t(_ws.bash_query_budget_seconds_help())
        ),
    ),
    # -- shell_environment ----------------------------------------------
    # ``path`` mirrors ``tools.shell_env.MODE_PATH`` and friends, pinned the
    # same way as the rows above. The three rows are one policy and are read
    # by one reader, but they are three settings rather than a JSON blob because
    # each answers a different question and each has a shape the editor already
    # knows: a mode to pick, and two name lists to type.
    #
    # IN THEIR OWN SECTION, and specifically NOT in ``tools`` (review round 1,
    # M2): scope is uniform within a section by construction, ``tools`` is LIVE,
    # and these keys must NOT be live. ``tools.shell_env.load_policy`` resolves
    # the policy once per process and memoises it, because ``config.yml`` is
    # writable by the agent's own shell as the same uid — a policy re-read per
    # command is a policy the constrained party can LOWER mid-run, which is how
    # a strict run gets its provider key back in the next command. NEW_LAUNCH is
    # that decision said in the vocabulary this module already has, and it is
    # also what the config-change notice tells the user.
    Setting(
        key="shell_environment.mode",
        path=("shell_environment", "mode"),
        section="shell_environment",
        label=_t(_ws.shell_environment_mode_label()),
        kind=Kind.ENUM,
        # Pinned to ``shell_env.MODE_DEFAULT`` by
        # ``test_shell_environment_rows_share_the_reader_paths`` rather than
        # imported: the default must stay the permissive one, and the pin is
        # what makes flipping it a decision rather than an edit.
        default="inherit",
        help=(
            _t(_ws.shell_environment_mode_help())
        ),
        choices=(
            Choice("inherit", _t(_ws.shell_environment_mode_choice_inherit_label()), _t(_ws.shell_environment_mode_choice_inherit_description())),
            Choice("allowlist", _t(_ws.shell_environment_mode_choice_allowlist_label()), _t(_ws.shell_environment_mode_choice_allowlist_description())),
        ),
    ),
    Setting(
        key="shell_environment.inherit",
        path=("shell_environment", "inherit"),
        section="shell_environment",
        label=_t(_ws.shell_environment_inherit_label()),
        kind=Kind.LIST,
        default=[],
        help=(
            _t(_ws.shell_environment_inherit_help())
        ),
        # OPEN namespace: these are the operator's own variable names, so there
        # is no vocabulary for this repo to bound.
        placeholder=_t(_ws.shell_environment_inherit_placeholder()),
        empty_unsets=True,
    ),
    Setting(
        key="shell_environment.exclude",
        path=("shell_environment", "exclude"),
        section="shell_environment",
        label=_t(_ws.shell_environment_exclude_label()),
        kind=Kind.LIST,
        default=[],
        help=(
            _t(_ws.shell_environment_exclude_help())
        ),
        placeholder=_t(_ws.shell_environment_exclude_placeholder()),
        empty_unsets=True,
    ),
    *[
        setting
        for provider, (name, endpoint, _url) in LOCAL_PRESETS.items()
        for setting in (
            Setting(
                f"providers.{provider}.base_url",
                ("providers", provider, "base_url"),
                "local_providers",
                _t(_ws.local_providers_base_url_label(name=name)),
                Kind.TEXT,
                endpoint,
                _t(_ws.local_providers_base_url_help()),
                validate_value=validate_endpoint_setting,
            ),
            Setting(
                f"providers.{provider}.models",
                ("providers", provider, "models"),
                "local_providers",
                _t(_ws.local_providers_models_label(name=name)),
                Kind.TEXT,
                DEFAULT_MODEL_OVERRIDES,
                _t(_ws.local_providers_models_help()),
                validate_value=model_overrides,
            ),
        )
    ],
    # -- retired ------------------------------------------------------------
    # Kept VISIBLE and read-only rather than hidden. A user who set one of
    # these years ago needs to see that it is inert; removing the row would
    # leave them believing a ceiling is still in force.
    Setting(
        key="conversation_length",
        path=("conversation_length",),
        section="retired",
        label=_t(_ws.conversation_length_label()),
        kind=Kind.READONLY,
        default=100,
        help=_t(_ws.conversation_length_help()),
    ),
    Setting(
        key="detail_length",
        path=("detail_length",),
        section="retired",
        label=_t(_ws.detail_length_label()),
        kind=Kind.READONLY,
        default=15,
        help=_t(_ws.detail_length_help()),
    ),
    Setting(
        key="max_learnings_history",
        path=("max_learnings_history",),
        section="retired",
        label=_t(_ws.max_learnings_history_label()),
        kind=Kind.READONLY,
        default=50,
        help=_t(_ws.max_learnings_history_help()),
    ),
    Setting(
        key="classification.notice",
        path=("classification", "notice"),
        section="retired",
        label=_t(_ws.classification_notice_label()),
        kind=Kind.READONLY,
        default=True,
        # RETIRED RATHER THAN DELETED, by this section's own rule above: a user who
        # set it deserves to see that it is inert rather than to find the row gone
        # and guess whether the layer stopped using it or stopped existing. The
        # surface it gated was removed outright instead of being left behind a false
        # switch — the per-call cost line the guide points at is at INFO in the
        # session log, so the diagnostic is not lost, and nothing draws into the
        # transcript any more.
        #
        # The copy names the feature the way EVERY other name on that screen does
        # ("Smart hints" — the section title, the live row's label, the tips, and the
        # guide, which says in as many words that all three use that name). "the layer"
        # is this codebase's word, not the user's; design round 1 (D1) measured that a
        # user who once saw a suggestion line has no anchor for it.
        help=_t(_ws.classification_notice_help()),
    ),
    Setting(
        key="desktop.launch_command",
        path=("desktop", "launch_command"),
        section="desktop",
        label=_t(_ws.desktop_launch_command_label()),
        # TEXT, NOT LIST, and the reason is the separator rather than the type.
        # ``Kind.LIST`` is a COMMA-SEPARATED token list (``web_search.providers``
        # and the OpenRouter host slugs), and a comma is a legal character in an
        # argv word — a path, a title, an argument. Storing argv in a
        # comma-delimited field would therefore corrupt a perfectly valid
        # command line silently, at click time, on the one path a user has no
        # other way to reach. The string is split with ``shlex`` by the reader,
        # which is the same grammar the user would type into a shell.
        #
        # ``{session}`` is substituted rather than appended, so an install whose
        # launcher wants the id somewhere other than last can say so without a
        # second setting.
        kind=Kind.TEXT,
        # RESTATED, not imported from the reader, matching the `tui.theme`
        # precedent above: this module is loaded by every CLI invocation and by
        # the TUI, and reaching into `tui.resume_click` from here would be an
        # import edge in the wrong direction for one empty string. The
        # divergence that restating invites is guarded by name:
        # `tests/unit/test_settings_io.py::_consumer_defaults` maps this key to
        # `resume_click.DESKTOP_LAUNCH_COMMAND_DEFAULT`, and
        # `test_every_default_matches_its_consumer` fails if the two part ways.
        default="",
        # The order in this copy IS the code's order and is asserted against it:
        # `_launch_desktop` appends `shutil.which(DESKTOP_BIN_NAME)` (the npm
        # bin) first and the `open -b` bundle second, and `docs/DESKTOP_API.md`
        # states the same order. It said the reverse here, on the one surface a
        # user browses to learn it (UX round 1, U3).
        #
        # BOTH CANDIDATES ARE NAMED, AND THE BUDGET IS THE REASON THE SECOND
        # SENTENCE IS THIS SHORT (design round 1, D12/D14). The first clause used
        # to say "the npm bin" and "the packaged bundle" — names that identify
        # neither artifact for a reader who has to find one — while the code
        # appends the bundle candidate ONLY on darwin, so Linux and Windows were
        # told about a discovery step that cannot happen there. Naming the bin
        # costs ~19 cells, and the old second sentence was 128 of a 194-cell
        # string against a `width − 6` slot that is at most 94 cells (100x30),
        # so its tail was dead copy at every width the page measures: the
        # `{session}` semantics are now the SHORT half and the placeholder
        # carries the command's shape.
        help=(
            _t(_ws.desktop_launch_command_help())
        ),
        placeholder=_t(_ws.desktop_launch_command_placeholder()),
        # Empty is the DEFAULT that must not be disturbed, and it has a real
        # meaning here ("discover it for me"), so it clears the key rather than
        # storing "".
        empty_unsets=True,
        # THE ONLY PLACE THIS CAN BE REPORTED (UX round 1, U4): the handler runs
        # detached from the notification, so a launcher that cannot be run turns
        # every click into a terminal with nothing on screen or in the log
        # saying why. Rejecting it here keeps the user in front of the field.
        validate_value=_validate_desktop_launch_command,
    ),
    Setting(
        key="static.roots",
        path=("static", "roots"),
        section="static",
        label=_t(_ws.static_roots_label()),
        kind=Kind.LIST,
        default=[],
        empty_unsets=True,
        validate_value=_validate_static_roots,
        # The consequence is the point of the row: every entry WIDENS what an
        # unauthenticated local caller can trigger a read of.
        warning=_t(_ws.static_roots_warning()),
        help=(
            _t(_ws.static_roots_help())
        ),
        placeholder=_t(_ws.static_roots_placeholder()),
    ),
    # -- network: where peers reach this device -------------------------------
    # THE TRIO THE DESIGN'S OWN TABLE NAMES. mesh-transport-identity.md §10.4 lists
    # `network.listen_address`, `network.port` and `network.advertise_hosts` among
    # the keys every one of which "needs its Setting, its module-level default
    # beside the reader, and its _consumer_defaults() entry". They were the three
    # that never got one, and the cost was not cosmetic: with no row, no surface in
    # the product could show or write them, so the only documented way to declare
    # an address was to hand-edit `config.yml` — and that edit went nowhere. A
    # top-level `network:` block (beside `values:`, which is how a dotted key reads)
    # is invisible to `get_nested_value`, which walks `values`, AND is deleted by
    # the next launch's cleanup migration, whose rewrite serialises only
    # metadata/values/version. So the operator's spelling was silently ignored and
    # then silently destroyed, while `init --advertise-host` looked like an
    # alternative that only worked once, at creation. Registering them is what
    # makes the canonical location `values.network.*` the one /settings shows and
    # writes, i.e. the one `get_nested_value` reads back.
    Setting(
        key="network.listen_address",
        path=("network", "listen_address"),
        section="network",
        label=_t(_ws.network_listen_address_label()),
        kind=Kind.TEXT,
        default="0.0.0.0",
        help=(
            # 53 CELLS, and the number is a constraint rather than a style: the detail
            # line is ONE row that sheds (help · clause · key → help · clause → …), and
            # the rung that matters is `help · clause` at 80 columns — 74 cells of
            # budget, less this row's 16-cell `default: 0.0.0.0` and the 3-cell joiner,
            # so the room is 55 and this string leaves 2 cells of it. The old 132-cell
            # sentence was dropped WHOLE off-default, so the reader who had set a mesh
            # address got the key path and no explanation of the field (design review
            # round 1, D1).
            _t(_ws.network_listen_address_help())
        ),
    ),
    # INT with the port range enforced, because the value is a socket bind and a
    # typo stored here would surface later as a relay that will not start — the
    # page and `lop config edit` both reach this, so the bound belongs on the row.
    Setting(
        key="network.port",
        path=("network", "port"),
        section="network",
        label=_t(_ws.network_port_label()),
        kind=Kind.INT,
        default=4097,
        minimum=1,
        maximum=65535,
        help=(
            # 50 cells: same rung and the same reason as `listen_address` above — this
            # row's clause is 13 cells (`default: 4097`) against the 74-cell budget and
            # the 3-cell joiner, so the room is 58 and this string leaves 8 of it.
            # Anything past the room sheds the help whole at 80 columns.
            _t(_ws.network_port_help())
        ),
    ),
    # LIST over an OPEN namespace — deliberately no `members`. The vocabulary is
    # hostnames, addresses and ports, which are the operator's and not this repo's;
    # a closed list would reject every address that matters. Empty means "no
    # opinion" (publish whatever the interface table shows), so the row clears
    # rather than failing, exactly as `NetworkSettings.from_config` reads an absent
    # key.
    Setting(
        key="network.advertise_hosts",
        path=("network", "advertise_hosts"),
        section="network",
        label=_t(_ws.network_advertise_hosts_label()),
        kind=Kind.LIST,
        default=[],
        help=(
            # 58 cells, the tightest of the three rows: its clause is `default: —` (10)
            # and the joiner is 3, against 74 at 80 columns — 3 cells of slack, and no
            # more, because the rung below (`clause · key`) drops the help ENTIRELY.
            # THE PORT RULE IS THE PART THAT MUST SURVIVE: `config edit` refuses a bare
            # address with exactly that rule, so a reader who never sees the help learns
            # it only by failing. The 204-cell sentence it replaces never rendered its
            # last sentence at ANY usable width, not even 200 columns (design review
            # round 1, D1).
            _t(_ws.network_advertise_hosts_help())
        ),
        # THE BARE ADDRESS LEADS. The ghost clips rather than wraps, and the example a
        # reader most needs is the one that shows an address WITH its port; the hostname
        # example it used to lead with ate the row and hid this one (design round 1, D4).
        placeholder=_t(_ws.network_advertise_hosts_placeholder()),
        empty_unsets=True,
        validate_value=_validate_advertise_hosts,
    ),
    # -- network (the mesh audit log) ---------------------------------------
    # Defaults mirror ``local_operator/network/audit.py``'s module constants, which
    # is what ``_consumer_defaults()`` in tests/unit/test_settings_io.py pins: a
    # registry default that disagrees with the reader's is a painted lie nothing
    # else reports. The numbers and their arithmetic (why 8 MiB compressed, why 5
    # generations, why 90 days) are justified in mesh-incident-response.md §4.5.
    Setting(
        key="network.audit.max_bytes",
        path=("network", "audit", "max_bytes"),
        section="network",
        label=_t(_ws.network_audit_max_bytes_label()),
        kind=Kind.INT,
        default=8_388_608,
        minimum=65_536,
        maximum=1_073_741_824,
        help=(
            _t(_ws.network_audit_max_bytes_help())
        ),
    ),
    Setting(
        key="network.audit.generations",
        path=("network", "audit", "generations"),
        section="network",
        label=_t(_ws.network_audit_generations_label()),
        kind=Kind.INT,
        default=5,
        minimum=1,
        maximum=50,
        help=_t(_ws.network_audit_generations_help()),
    ),
    Setting(
        key="network.audit.max_age_days",
        path=("network", "audit", "max_age_days"),
        section="network",
        label=_t(_ws.network_audit_max_age_days_label()),
        kind=Kind.FLOAT,
        default=90.0,
        minimum=1.0,
        maximum=3650.0,
        help=(
            _t(_ws.network_audit_max_age_days_help())
        ),
    ),
    # THE RELAY'S OWN LIMIT, and the one an operator needs exactly when a peer
    # cannot connect: §2.1's cap on UNAUTHENTICATED connections in flight. The slot
    # is held only until a handshake resolves, so this bounds neither `max_links`
    # nor an established link; past the cap a connection is dropped at accept with
    # no reply, and each drop is one audit record per window. Both ends read their
    # OWN value, so raising it on one device does not raise it on the other.
    #
    # The default mirrors ``local_operator/network/relay.py``'s
    # ``DEFAULT_MAX_HANDSHAKES``, which is what ``_consumer_defaults()`` in
    # tests/unit/test_settings_io.py pins: a registry default that disagrees with
    # the reader's is a painted lie nothing else reports.
    Setting(
        key="network.max_handshakes",
        path=("network", "max_handshakes"),
        section="network",
        label=_t(_ws.network_max_handshakes_label()),
        kind=Kind.INT,
        default=8,
        minimum=1,
        maximum=1024,
        help=(
            _t(_ws.network_max_handshakes_help())
        ),
    ),
    # -- network.sync / network.credentials (mesh build plan P0) ---------------
    # Declared by P0 so the sync and credentials slices never edit this file (the
    # plan's conflict rule). Defaults mirror the READERS' module constants —
    # ``network/sync.py`` SYNC_DEBOUNCE_S / SYNC_TICK_S and
    # ``network/credentials/__init__.py`` GRANT_TTL_S — which ``_consumer_defaults()``
    # in tests/unit/test_settings_io.py pins. Relay-restart scope like the rest of
    # the section: the watcher and the broker read them when the relay starts.
    Setting(
        key="network.sync.debounce_s",
        path=("network", "sync", "debounce_s"),
        section="network",
        label=_t(_ws.network_sync_debounce_s_label()),
        kind=Kind.FLOAT,
        default=30.0,
        minimum=1.0,
        maximum=3600.0,
        help=(
            _t(_ws.network_sync_debounce_s_help())
        ),
    ),
    Setting(
        key="network.sync.tick_s",
        path=("network", "sync", "tick_s"),
        section="network",
        label=_t(_ws.network_sync_tick_s_label()),
        kind=Kind.FLOAT,
        default=15.0,
        minimum=1.0,
        maximum=3600.0,
        help=_t(_ws.network_sync_tick_s_help()),
    ),
    Setting(
        key="network.credentials.grant_ttl_s",
        path=("network", "credentials", "grant_ttl_s"),
        section="network",
        label=_t(_ws.network_credentials_grant_ttl_s_label()),
        kind=Kind.FLOAT,
        default=900.0,
        minimum=60.0,
        maximum=3600.0,
        help=(
            _t(_ws.network_credentials_grant_ttl_s_help())
        ),
    ),
    # The github adapter's allow-list (github.py's ``REPOSITORIES_PATH``; the same
    # key is read by the owner — the mint narrows to it and REFUSES when empty —
    # and by the borrower's git helper as its use-time backstop). Empty is a
    # refusal at mint time, never "no narrowing": a mint without ``repositories``
    # would cover everything the App installation was granted.
    Setting(
        key="network.credentials.github.repositories",
        path=("network", "credentials", "github", "repositories"),
        section="network",
        label=_t(_ws.network_credentials_github_repositories_label()),
        kind=Kind.LIST,
        default=[],
        help=_t(_ws.network_credentials_github_repositories_help()),
        placeholder=_t(_ws.network_credentials_github_repositories_placeholder()),
        empty_unsets=True,
        validate_value=_validate_github_repositories,
    ),
    # -- hub ---------------------------------------------------------------
    # Defaults are LITERALS here for the same reason the aida ones are: this module
    # stays off the hub_sync import path (it loads on every CLI start), and
    # `_consumer_defaults()` in tests/unit/test_settings_io.py imports the real
    # constants from ``local_operator/hub_sync/settings.py`` to pin they cannot
    # drift. Every path is a genuinely NESTED tuple, read back through
    # ``ConfigManager.get_nested_value`` — NOT ``get_config_value`` on the dotted
    # string, which looks up a literal top-level key and would read nothing.
    # Help strings are budgeted to <=72 cells (the detail line at 100 columns).
    Setting(
        key="hub.auto_update.agents",
        path=("hub", "auto_update", "agents"),
        section="hub",
        label=_t(_ws.hub_auto_update_agents_label()),
        kind=Kind.BOOL,
        default=True,
        choices=_bool_choices(_t(_ws.hub_auto_update_agents_choice_true_description()), _t(_ws.hub_auto_update_agents_choice_false_description())),
        help=_t(_ws.hub_auto_update_agents_help()),
    ),
    Setting(
        key="hub.auto_update.teams",
        path=("hub", "auto_update", "teams"),
        section="hub",
        label=_t(_ws.hub_auto_update_teams_label()),
        kind=Kind.BOOL,
        default=True,
        choices=_bool_choices(_t(_ws.hub_auto_update_teams_choice_true_description()), _t(_ws.hub_auto_update_teams_choice_false_description())),
        help=_t(_ws.hub_auto_update_teams_help()),
    ),
    Setting(
        key="hub.check_interval_min",
        path=("hub", "check_interval_min"),
        section="hub",
        label=_t(_ws.hub_check_interval_min_label()),
        kind=Kind.INT,
        default=60,
        minimum=5,
        maximum=1440,
        help=_t(_ws.hub_check_interval_min_help()),
    ),
    Setting(
        key="hub.merge_model",
        path=("hub", "merge_model"),
        section="hub",
        label=_t(_ws.hub_merge_model_label()),
        kind=Kind.TEXT,
        default="",
        placeholder=_t(_ws.hub_merge_model_placeholder()),
        help=_t(_ws.hub_merge_model_help()),
        empty_unsets=True,
    ),
    # -- agents ------------------------------------------------------------
    # The starter-update switch. A LITERAL default for the same reason the hub
    # keys are literals: this module stays off the import path of the package
    # that reads the key (``local_operator.agent_profiles`` pulls the registry
    # machinery), and ``_consumer_defaults()`` in tests/unit/test_settings_io.py
    # imports ``AUTO_UPDATE_SEEDS_DEFAULT`` from there to pin the two cannot
    # drift. Path is a genuinely NESTED tuple, read back through
    # ``ConfigManager.get_nested_value`` by ``_auto_update_seeds_enabled``.
    Setting(
        key="agents.auto_update.seeds",
        path=("agents", "auto_update", "seeds"),
        section="agents",
        label=_t(_ws.agents_auto_update_seeds_label()),
        kind=Kind.BOOL,
        default=True,
        choices=_bool_choices(_t(_ws.agents_auto_update_seeds_choice_true_description()), _t(_ws.agents_auto_update_seeds_choice_false_description())),
        # <= 72 cells (the picker's budget). The round-1 copy measured 77 and
        # blew that budget; trimmed here to 61 on the ``cell_len`` measure
        # (design round 2, D2-2 / agent review R2-3). Names the held
        # exception too: "tool changes still ask" is what the launch pass
        # does, and leaving it out let the "on" choice over-promise
        # (design round 1, D7).
        help=_t(_ws.agents_auto_update_seeds_help()),
    ),
    # -- aida --------------------------------------------------------------
    # Defaults are LITERALS here, not imports: this module deliberately keeps the
    # aida package off its import path (it is loaded on every CLI start), and
    # `_consumer_defaults()` in tests/unit/test_settings_io.py imports the real
    # constants from ``local_operator/aida/`` to pin that these cannot drift.
    Setting(
        key="aida.enabled",
        path=("aida", "enabled"),
        section="aida",
        # ROLE-NAMED, not persona-named (design round 1, D4): "Aida enabled"
        # read as the old name beside a configured `Display name: Nova`, while
        # every sibling row in the section is a role term. The section title
        # stays "Aida" — that one is the config namespace (`aida.*`).
        label=_t(_ws.aida_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        choices=_bool_choices(_t(_ws.aida_enabled_choice_true_description()), _t(_ws.aida_enabled_choice_false_description())),
        help=_t(_ws.aida_enabled_help()),
    ),
    Setting(
        key="aida.name",
        path=("aida", "name"),
        section="aida",
        label=_t(_ws.aida_name_label()),
        kind=Kind.TEXT,
        default="Aida",
        placeholder=_t(_ws.aida_name_placeholder()),
        validate_value=_validate_aida_name,
        help=(
            # ≤94 cells, the detail line's budget at 100x30 (design round 1,
            # D2): the previous wording truncated at "upd…", eating exactly
            # the clause that says the rename APPLIES.
            _t(_ws.aida_name_help())
        ),
    ),
    Setting(
        key="aida.cadence.at",
        path=("aida", "cadence", "at"),
        section="aida",
        label=_t(_ws.aida_cadence_at_label()),
        kind=Kind.TEXT,
        default="08:30",
        placeholder=_t(_ws.aida_cadence_at_placeholder()),
        help=_t(_ws.aida_cadence_at_help()),
    ),
    Setting(
        key="aida.cadence.paused",
        path=("aida", "cadence", "paused"),
        section="aida",
        label=_t(_ws.aida_cadence_paused_label()),
        kind=Kind.BOOL,
        default=False,
        choices=_bool_choices(_t(_ws.aida_cadence_paused_choice_true_description()), _t(_ws.aida_cadence_paused_choice_false_description())),
        help=(
            _t(_ws.aida_cadence_paused_help())
        ),
    ),
    Setting(
        key="aida.cadence.max_extra_per_day",
        path=("aida", "cadence", "max_extra_per_day"),
        section="aida",
        label=_t(_ws.aida_cadence_max_extra_per_day_label()),
        kind=Kind.INT,
        default=2,
        minimum=0,
        maximum=12,
        help=_t(_ws.aida_cadence_max_extra_per_day_help()),
    ),
    Setting(
        key="aida.cadence.min_gap_minutes",
        path=("aida", "cadence", "min_gap_minutes"),
        section="aida",
        label=_t(_ws.aida_cadence_min_gap_minutes_label()),
        kind=Kind.INT,
        default=90,
        minimum=0,
        maximum=1440,
        help=_t(_ws.aida_cadence_min_gap_minutes_help()),
    ),
    Setting(
        key="aida.onboarding.nudge_days",
        path=("aida", "onboarding", "nudge_days"),
        section="aida",
        label=_t(_ws.aida_onboarding_nudge_days_label()),
        kind=Kind.INT,
        default=14,
        minimum=1,
        maximum=365,
        help=(
            _t(_ws.aida_onboarding_nudge_days_help())
        ),
    ),
    # -- projects -------------------------------------------------------------
    # The staleness window. Default is a LITERAL like the aida block's (this
    # module is loaded on every CLI start); `_consumer_defaults()` in
    # tests/unit/test_settings_io.py pins it to the constant in
    # ``local_operator/projects.py`` that the resolver falls back to, so a
    # drift between this table and the computation is a red test.
    Setting(
        key="projects.stale_after_hours",
        path=("projects", "stale_after_hours"),
        section="projects",
        label=_t(_ws.projects_stale_after_hours_label()),
        kind=Kind.INT,
        default=4,
        minimum=1,
        maximum=168,
        help=(
            _t(_ws.projects_stale_after_hours_help())
        ),
    ),
    # -- wake triggers --------------------------------------------------------
    # Defaults are LITERALS here too; `_consumer_defaults()` imports the
    # constants the trigger layer itself falls back to (``wakes/triggers/``)
    # and pins these rows against them.
    Setting(
        key="wakes.triggers.enabled",
        path=("wakes", "triggers", "enabled"),
        section="wakes",
        label=_t(_ws.wakes_triggers_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        choices=_bool_choices(_t(_ws.wakes_triggers_enabled_choice_true_description()), _t(_ws.wakes_triggers_enabled_choice_false_description())),
        help=_t(_ws.wakes_triggers_enabled_help()),
    ),
    Setting(
        key="wakes.triggers.max_per_day",
        path=("wakes", "triggers", "max_per_day"),
        section="wakes",
        label=_t(_ws.wakes_triggers_max_per_day_label()),
        kind=Kind.INT,
        default=6,
        minimum=0,
        maximum=48,
        help=(
            _t(_ws.wakes_triggers_max_per_day_help())
        ),
    ),
    Setting(
        key="wakes.triggers.min_gap_minutes",
        path=("wakes", "triggers", "min_gap_minutes"),
        section="wakes",
        label=_t(_ws.wakes_triggers_min_gap_minutes_label()),
        kind=Kind.INT,
        default=60,
        minimum=10,
        maximum=1440,
        help=_t(_ws.wakes_triggers_min_gap_minutes_help()),
    ),
    Setting(
        key="wakes.triggers.project_staleness.enabled",
        path=("wakes", "triggers", "project_staleness", "enabled"),
        section="wakes",
        label=_t(_ws.wakes_triggers_project_staleness_enabled_label()),
        kind=Kind.BOOL,
        default=True,
        choices=_bool_choices(_t(_ws.wakes_triggers_project_staleness_enabled_choice_true_description()), _t(_ws.wakes_triggers_project_staleness_enabled_choice_false_description())),
        help=(
            _t(_ws.wakes_triggers_project_staleness_enabled_help())
        ),
    ),
    # -- proactive class ------------------------------------------------------
    # Defaults are LITERALS here, not imports, for the same reason the aida
    # block's are: this module is loaded on every CLI start. The consumer
    # defaults the settings test pins live in ``local_operator/wakes/patience``
    # (DEFAULT_WAIT_MS, DEFAULT_BACKOFF, DEFAULT_MAX_ATTEMPTS, DEFAULT_TTL_MS,
    # DEFAULT_MAX_PENDING), so a drift between this table and the engine is a
    # red test rather than a page that lies about what a fire will honour.
    Setting(
        key="proactive.patience.default_ms",
        path=("proactive", "patience", "default_ms"),
        section="proactive",
        label=_t(_ws.proactive_patience_default_ms_label()),
        kind=Kind.INT,
        default=300000,
        minimum=60000,
        maximum=86400000,
        help=(
            _t(_ws.proactive_patience_default_ms_help())
        ),
    ),
    Setting(
        key="proactive.patience.backoff",
        path=("proactive", "patience", "backoff"),
        section="proactive",
        label=_t(_ws.proactive_patience_backoff_label()),
        kind=Kind.INT,
        default=3,
        minimum=1,
        maximum=10,
        help=(
            _t(_ws.proactive_patience_backoff_help())
        ),
    ),
    Setting(
        key="proactive.patience.max_attempts",
        path=("proactive", "patience", "max_attempts"),
        section="proactive",
        label=_t(_ws.proactive_patience_max_attempts_label()),
        kind=Kind.INT,
        default=3,
        minimum=1,
        maximum=10,
        help=(_t(_ws.proactive_patience_max_attempts_help())),
    ),
    Setting(
        key="proactive.patience.episode_ttl_ms",
        path=("proactive", "patience", "episode_ttl_ms"),
        section="proactive",
        label=_t(_ws.proactive_patience_episode_ttl_ms_label()),
        kind=Kind.INT,
        default=7200000,
        minimum=60000,
        maximum=86400000,
        help=(
            _t(_ws.proactive_patience_episode_ttl_ms_help())
        ),
    ),
    Setting(
        key="proactive.patience.max_pending",
        path=("proactive", "patience", "max_pending"),
        section="proactive",
        label=_t(_ws.proactive_patience_max_pending_label()),
        kind=Kind.INT,
        default=4,
        minimum=0,
        maximum=16,
        help=_t(_ws.proactive_patience_max_pending_help()),
    ),
    # --- speech voicing -----------------------------------------------------
    # Every default below is a LITERAL on purpose (``settings_io`` must stay off
    # the tts package's import path), and ``test_settings_io``'s
    # ``_consumer_defaults`` imports the real constants and asserts they match —
    # the pair is what stops a page from advertising a default the synthesizer
    # does not use.
    Setting(
        key="speech.voice.gender",
        path=("speech", "voice", "gender"),
        section="speech",
        label=_t(_ws.speech_voice_gender_label()),
        kind=Kind.ENUM,
        # ``auto`` is today's behaviour: the daemon classifies each agent and
        # resolves the result before sending, because the hub refuses ``auto``
        # (it has no agent context to classify with).
        default="auto",
        help=_t(_ws.speech_voice_gender_help()),
        choices=(
            Choice(
                "auto",
                _t(_ws.speech_voice_gender_choice_auto_label()),
                _t(_ws.speech_voice_gender_choice_auto_description()),
            ),
            Choice("female", _t(_ws.speech_voice_gender_choice_female_label()), _t(_ws.speech_voice_gender_choice_female_description())),
            Choice("male", _t(_ws.speech_voice_gender_choice_male_label()), _t(_ws.speech_voice_gender_choice_male_description())),
        ),
    ),
    Setting(
        key="speech.voice.tone",
        path=("speech", "voice", "tone"),
        section="speech",
        label=_t(_ws.speech_voice_tone_label()),
        kind=Kind.ENUM,
        default="warm",
        help=_t(_ws.speech_voice_tone_help()),
        choices=(
            Choice("warm", _t(_ws.speech_voice_tone_choice_warm_label()), _t(_ws.speech_voice_tone_choice_warm_description())),
            Choice("neutral", _t(_ws.speech_voice_tone_choice_neutral_label()), _t(_ws.speech_voice_tone_choice_neutral_description())),
            Choice("bright", _t(_ws.speech_voice_tone_choice_bright_label()), _t(_ws.speech_voice_tone_choice_bright_description())),
            Choice("calm", _t(_ws.speech_voice_tone_choice_calm_label()), _t(_ws.speech_voice_tone_choice_calm_description())),
            Choice("authoritative", _t(_ws.speech_voice_tone_choice_authoritative_label()), _t(_ws.speech_voice_tone_choice_authoritative_description())),
        ),
    ),
    Setting(
        key="speech.voice.expressiveness",
        path=("speech", "voice", "expressiveness"),
        section="speech",
        label=_t(_ws.speech_voice_expressiveness_label()),
        kind=Kind.ENUM,
        default="medium",
        help=_t(_ws.speech_voice_expressiveness_help()),
        choices=(
            Choice("low", _t(_ws.speech_voice_expressiveness_choice_low_label()), _t(_ws.speech_voice_expressiveness_choice_low_description())),
            Choice("medium", _t(_ws.speech_voice_expressiveness_choice_medium_label()), _t(_ws.speech_voice_expressiveness_choice_medium_description())),
            Choice("high", _t(_ws.speech_voice_expressiveness_choice_high_label()), _t(_ws.speech_voice_expressiveness_choice_high_description())),
        ),
    ),
    Setting(
        key="speech.voice.pace",
        path=("speech", "voice", "pace"),
        section="speech",
        label=_t(_ws.speech_voice_pace_label()),
        kind=Kind.FLOAT,
        default=1.0,
        minimum=0.5,
        maximum=2.0,
        help=_t(_ws.speech_voice_pace_help()),
    ),
    Setting(
        key="speech.voice.language",
        path=("speech", "voice", "language"),
        section="speech",
        label=_t(_ws.speech_voice_language_label()),
        kind=Kind.TEXT,
        default="auto",
        # Cleared, the key is removed and the consumer's own ``auto`` applies.
        empty_unsets=True,
        help=_t(_ws.speech_voice_language_help()),
    ),
    Setting(
        key="speech.voice.accent",
        path=("speech", "voice", "accent"),
        section="speech",
        label=_t(_ws.speech_voice_accent_label()),
        kind=Kind.TEXT,
        default="",
        empty_unsets=True,
        help=_t(_ws.speech_voice_accent_help()),
    ),
    Setting(
        key="speech.voice.instructions",
        path=("speech", "voice", "instructions"),
        section="speech",
        label=_t(_ws.speech_voice_instructions_label()),
        kind=Kind.TEXT,
        # The pre-#1835 native-dialect guidance. NOT ``empty_unsets``: an empty
        # value is meaningful (send no instructions of our own) and is a
        # DIFFERENT state from the key being absent (use this default).
        default=(
            "Speak aloud and pay attention to potentially multilingual inputs and make sure to "
            "use native accents for all different parts of the text, especially those that are "
            "not english. Strive for a casual and native-sounding conversational tone. Don't "
            "over-enunciate, consider word combinations that should have silent and natural "
            'transitions, like "raha hoon" -> "rahoon" or "je m\'appelle" -> "jm\'appelle".'
        ),
        help=_t(_ws.speech_voice_instructions_help()),
    ),
)

#: ``key -> Setting`` for the lookups the CLI and the page both do.
BY_KEY: dict[str, Setting] = {setting.key: setting for setting in SETTINGS}


def settings_for(section: str) -> tuple[Setting, ...]:
    """Every setting in ``section``, in registry order."""
    return tuple(setting for setting in SETTINGS if setting.section == section)


def flat_dotted_keys() -> tuple[str, ...]:
    """Keys whose dot is literal rather than a nesting level.

    Exported so the round-trip test can assert against the registry instead of
    hard-coding a list that would drift the moment a sixth ``display.*`` flag
    is added.
    """
    return tuple(setting.key for setting in SETTINGS if setting.is_flat_dotted)


def display_defaults() -> dict[str, Any]:
    """``{"display.shimmer": True, ...}`` — the TUI display-flag defaults.

    ``tui/settings.py`` derives its flag defaults from this so the page and the
    fast-path reader cannot disagree about what "unset" means. Returned as a
    fresh dict because the caller caches and mutates its copy.
    """
    return {
        setting.key: setting.default
        for setting in SETTINGS
        if setting.is_flat_dotted and setting.key.startswith("display.")
    }


# ---------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------


_MISSING = object()


def _walk(values: Mapping[str, Any], path: Sequence[str]) -> Any:
    """Follow ``path`` through nested mappings; ``_MISSING`` if it breaks.

    A non-mapping partway down is treated as absent rather than raising: a
    hand-edited ``retry: "yes"`` must render as "unset, showing the default"
    on the page, not crash the surface that would let the user fix it.
    """
    current: Any = values
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def strict_bool(value: Any, default: bool) -> bool:
    """A REAL boolean or ``default`` — never ``bool(value)``.

    ``enabled: "false"`` in a hand-edited YAML is a non-empty string, and
    ``bool("false")`` is True. The cleanup policy parses its master switch
    this way (review round 1, R1-6); the settings PAGE must read it the same
    way, or the page paints ``on`` for a switch the policy treats as off
    (review round 2, R2-4) — the incident's accessor-mismatch class again,
    one level up. So this lives here, beside the reader every page read goes
    through, and the consumer imports it rather than spelling its own.
    Anything that is not literally true/false, 0/1, or the YAML 1.1 spellings
    a human types (yes/no/on/off) reads as the default.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "yes", "on", "1"):
            return True
        if lowered in ("false", "no", "off", "0"):
            return False
    return default


def read_setting(manager: "ConfigManager", setting: Setting) -> Any:
    """The stored value for ``setting``, or its default when unset.

    A BOOL is normalised through :func:`strict_bool` so a hand-edited
    ``"false"`` reads as the consumer reads it (off), and ``is_default`` /
    the value ink / the toggle all agree with the policy about the state.
    """
    raw = _walk(manager.get_config().values, setting.path)
    if raw is _MISSING:
        return setting.default
    if setting.validate_value is model_overrides and isinstance(raw, Mapping):
        # Hand-written YAML maps and the text editor share the same contract.
        # Python's dict repr uses single quotes and cannot round-trip as JSON.
        return json.dumps(raw, default=str)
    if setting.validate_value is _validate_openrouter_max_price and isinstance(raw, Mapping):
        # Same contract as model_overrides above: the TUI editor is text, so a
        # stored mapping (PATCH route, hand-written YAML) is shown as JSON the
        # editor can round-trip.
        return json.dumps(raw, default=str)
    if setting.kind is Kind.BOOL and isinstance(setting.default, bool):
        return strict_bool(raw, setting.default)
    return raw


def is_default(manager: "ConfigManager", setting: Setting) -> bool:
    """Whether the stored value equals the shipped default.

    Immediate-write's one real cost is undo, so the page marks changed rows and
    offers a reset on them. Compared by VALUE and not by presence: a user who
    explicitly typed the default has not changed anything, and highlighting the
    row would be a lie about the state of their config.
    """
    return read_setting(manager, setting) == setting.default


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------


def coerce(setting: Setting, text: str) -> Any:
    """Parse a user's typed string into the stored type.

    Raises ``ValueError`` with a message written FOR the user — the page prints
    it inline under the editor and keeps the editor open, so it has to say what
    to type rather than name a Python exception.
    """
    text = text.strip()
    if setting.kind is Kind.INT:
        try:
            return int(text)
        except ValueError:
            raise ValueError("expected a whole number") from None
    if setting.kind is Kind.FLOAT:
        try:
            return float(text)
        except ValueError:
            raise ValueError("expected a number") from None
    if setting.kind is Kind.LIST:
        items = [part.strip() for part in text.split(",") if part.strip()]
        # `members` is the CLOSED allow-list and only gates where this repo
        # owns the vocabulary (web_search.providers). An empty tuple is an
        # OPEN list (the OpenRouter host slugs — an upstream-owned namespace
        # that grows without notice): every non-empty token is accepted, so
        # `deepinfra`, `novita` or `google-vertex/us-east5` cannot be rejected
        # by a list that was merely out of date (review round 1, M1).
        if setting.members:
            unknown = [item for item in items if item not in setting.members]
            if unknown:
                offered = ", ".join(setting.members)
                raise ValueError(f"unknown: {', '.join(unknown)} — pick from {offered}")
        # Stable de-duplication, matching `coerce_search_settings`: a repeated
        # provider is a typo, not a request to weight it twice.
        return list(dict.fromkeys(items))
    if setting.kind is Kind.BOOL:
        lowered = text.lower()
        if lowered in ("true", "on", "yes", "1"):
            return True
        if lowered in ("false", "off", "no", "0"):
            return False
        raise ValueError("expected on or off")
    if setting.kind is Kind.HOTKEY:
        # Normalized at the WRITE boundary, not only at apply. The capture UI
        # already receives a normalized `event.key`, so this is for the other
        # two writers — `lop config edit` and a hand edit — which otherwise
        # store `ctrl+N`: the page then DISPLAYS `ctrl+N` while the runtime
        # binds a key nobody can press (measured — Textual accepts it
        # verbatim). Rejection happens in `validate`, so the message is the
        # same whichever writer arrives. The grammar follows the ROW's scope:
        # a desktop value is an accelerator, not a Textual key string.
        if setting.hotkey_scope == "desktop":
            return _keymap.normalize_desktop_key(text, action_id=setting.key)
        return _keymap.normalize_key(text)
    if setting.kind is Kind.CASCADE:
        # JSON, because the value is a two-level structure and the page's own
        # chain editor is the only other way to build one. Without this arm
        # the kind fell through to ``return text``, so the two writers that do
        # not go through that editor — ``lop config edit`` and a hand-edited
        # ``config.yml`` — put a STRING where a mapping belongs. ``validate``
        # had no ``CASCADE`` arm either, so the write was accepted and
        # reported as a success, while ``resolve_chain`` and ``read_chains``
        # both require a ``Mapping`` and silently read a string as "no cascade
        # configured". The failover the user had just asked for never ran, and
        # nothing on any surface said so.
        try:
            return json.loads(text)
        except ValueError:
            raise ValueError(
                'expected JSON, e.g. {"default": ["anthropic/claude-opus-5"]}'
            ) from None
    return text


def validate(setting: Setting, value: Any, values: Mapping[str, Any] | None = None) -> str | None:
    """``None`` when ``value`` may be stored, else the reason it may not.

    Bounds are enforced HERE rather than left to the consumer's own clamping,
    because the consumers clamp SILENTLY (``coerce_search_settings`` pins the
    timeout to 1-120 on read). A page that accepted 500 and stored it would
    show 500 forever while the tool used 120 — the config and the behaviour
    disagreeing, with nothing on screen admitting it.

    ``values`` is the current config snapshot, needed only by the checks that
    are about a value's relationship to its SIBLINGS rather than to its own
    schema — today just the hotkey group, where two actions sharing one key
    leaves one silently unreachable. Optional because most settings are
    self-contained and the unit hosts validate without a config; when it is
    omitted the group check is skipped, and :func:`write_setting` always
    supplies it, so no write can reach disk unchecked.
    """
    if setting.validate_value is not None:
        try:
            setting.validate_value(value)
        except (ValueError, TypeError) as error:
            return str(error)
    if setting.kind is Kind.READONLY:
        return "this setting is retired and cannot be changed"
    if setting.kind is Kind.ENUM:
        # `resolved_choices`, never `choices`: a registry-sourced value space
        # (tui.theme) declares none statically, so reading the raw field would
        # reject every value including the default.
        choices = setting.resolved_choices
        if not choices:
            # An empty value space is a BROKEN HOST, not a bad value: the
            # message has to say so, because "expected one of: " lists nothing
            # and tells the user their input is wrong while offering no way to
            # be right (review round 2, m4). Only reachable when a
            # `choices_source` cannot resolve — the TUI-less install the source
            # fails closed for.
            return "this setting's choices could not be read on this install"
        # bool and int compare equal in Python, but are distinct JSON choices.
        if not any(
            type(value) is type(choice.value) and value == choice.value for choice in choices
        ):
            return f"expected one of: {', '.join(str(c.label) for c in choices)}"
        return None
    if setting.kind is Kind.LIST:
        if not isinstance(value, list):
            return "expected a comma-separated list"
        if any(not isinstance(item, str) for item in value):
            return "expected a list of names"
        # `members` gates only the CLOSED lists (see `coerce`); an OPEN list
        # still bounds each token itself — an empty slug is a trailing or
        # doubled comma, not a host any router knows.
        if setting.members:
            unknown = [item for item in value if item not in setting.members]
            if unknown:
                return f"unknown: {', '.join(str(item) for item in unknown)}"
        elif any(not item.strip() for item in value):
            return "expected non-empty names, separated by commas"
        if not value and not setting.empty_unsets:
            # An empty list is an error only where the consumer NEEDS at least
            # one entry (web_search.providers). Where empty means "no opinion"
            # and the feature simply stays off (the OpenRouter routing lists),
            # `empty_unsets` marks it and the row clears instead of failing.
            return "at least one provider is required"
        return None
    if setting.kind is Kind.BOOL:
        return None if isinstance(value, bool) else "expected on or off"
    if setting.kind in (Kind.INT, Kind.FLOAT):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "expected a number"
        # Typed HTTP edits bypass coerce(), unlike the terminal text editor.
        # Enforce the stored type here so every writer shares the same bounds.
        if setting.kind is Kind.INT and not isinstance(value, int):
            return "expected a whole number"
        if isinstance(value, float) and not math.isfinite(value):
            return "expected a finite number"
        if setting.minimum is not None and value < setting.minimum:
            return f"must be at least {_number(setting.minimum)}"
        if setting.maximum is not None and value > setting.maximum:
            return f"must be at most {_number(setting.maximum)}"
        return None
    if setting.kind is Kind.TEXT:
        # A TEXT row with its own `validate_value` (JSON-object fields like
        # `providers.openrouter.max_price`, endpoint URLs) owns the value's
        # whole contract — including accepting non-string shapes the PATCH
        # route hands in — so the generic string check only applies to
        # unvalidated text.
        if setting.validate_value is not None:
            return None
        return None if isinstance(value, str) else "expected text"
    if setting.kind is Kind.HOTKEY:
        # THE guard, and it has to be here rather than in the capture widget:
        # `lop config edit` and a hand-edited config.yml reach the same value
        # and bypass the page entirely, and Textual validates NOTHING — a
        # garbage key string silently moves the binding somewhere unreachable
        # AND takes the shipped default with it (measured; the Claude Code
        # pre-2.1.246 silent-disable bug, live in Textual today). The capture
        # widget calls the same predicate, so the two cannot disagree. The
        # scope (and, for desktop rows, the action id) selects the grammar:
        # `keymap.quick_send` must be judged by the rules of a global
        # shortcut, not the rules of a terminal key.
        problem = _keymap.validate_key(
            value,
            scope=setting.hotkey_scope or "app",
            action_id=setting.key,
        )
        if problem is not None:
            return problem
        # THE SIBLING CHECK, after the grammar rather than before it: a value
        # the row's own grammar refuses must hear THAT sentence. Adding the
        # hold family made the old order actively misleading — an accelerator
        # pasted into the hold row collided with quick_send's default string
        # and was told "pick another", as if any accelerator could be accepted
        # by a row that holds none. A value that survives its own grammar is
        # the only kind a collision can be about.
        if values is not None and isinstance(value, str) and value.strip():
            return _keymap.group_conflict(setting.key, value, values)
        return None
    if setting.kind is Kind.CASCADE:
        # Same rule the HOTKEY arm above states, for the same reason: `lop
        # config edit` and a hand-edited config.yml both reach this value
        # without passing the page's chain editor. A shape the failover layer
        # cannot read is not a cascade that merely looks odd —
        # `resolve_chain` returns ``None`` for it and routing continues as
        # though nothing were configured, which is the one outcome the user
        # cannot see. Refusing names the problem while the user is still
        # looking at the command that caused it.
        if not isinstance(value, Mapping):
            return 'expected a JSON object of chains, e.g. {"default": ["anthropic/claude-opus-5"]}'
        for key, hops in value.items():
            if not isinstance(key, str) or not key.strip():
                return 'every chain needs a name, e.g. "default"'
            # A bare string is the plausible near-miss ({"default": "a/b"}):
            # it is a Sequence, so an isinstance check alone would admit it
            # and then iterate it one character at a time.
            if isinstance(hops, str) or not isinstance(hops, Sequence):
                return f"chain {key!r} must be a list of provider/model hops"
            for hop in hops:
                # `_hop_label` is the DISPLAY formatter and answers a weaker
                # question — it returns any non-empty string unchanged, so it
                # accepts `gpt-4o`, which `expand_fallback_targets` then drops
                # for having no provider. That is this bug wearing a different
                # hat: stored, confirmed, and routing nothing. It still gates
                # the mapping form's provider/model presence, whose label
                # carries the effort a string hop may not; `validate_hop` —
                # the predicate the page's hop editor and the server's
                # `_write_cascade` already share — is what a string hop is
                # held to. The mapping arm's `effort` value below is the rest
                # of that shape's contract.
                if not _hop_label(hop):
                    return f"chain {key!r} has a hop that is not provider/model: {hop!r}"
                if isinstance(hop, str):
                    problem = validate_hop(hop)
                    if problem is not None:
                        return f"chain {key!r}: {problem}"
                elif isinstance(hop, Mapping):
                    # `_hop_label` above only checks provider/model are
                    # non-empty; it does not read `effort` at all, so a
                    # mapping hop with an unsupported effort passed it,
                    # was stored, and reported as success — the same
                    # silent-drop this whole setting exists to refuse, one
                    # shape over. `_normalize_chain_entry` (the runtime's
                    # own reader) already rejects the same value with a
                    # warning at materialization; holding it here instead
                    # of there is what makes the refusal visible to the
                    # user who typed it, at the command that caused it.
                    raw_effort = hop.get("effort")
                    if raw_effort is not None and str(raw_effort).strip().lower() not in (
                        SUPPORTED_EFFORTS
                    ):
                        return (
                            f"chain {key!r}: effort {raw_effort!r} is not one of "
                            f"{', '.join(sorted(SUPPORTED_EFFORTS))}"
                        )
        return None
    return None


def _number(value: float) -> str:
    """Render a bound without a pointless ``.0`` on an integral float."""
    return str(int(value)) if float(value).is_integer() else str(value)


# ---------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------


def write_setting(manager: "ConfigManager", setting: Setting, value: Any) -> None:
    """Store ``value``, merging into any existing sub-mapping.

    THE merge rule (see the module docstring): the sub-mapping is copied,
    the one leaf is replaced, and the copy is written back through
    ``set_config_value`` — the only writer ``ConfigManager`` has. Replacing the
    sub-mapping wholesale would destroy siblings that ``_load_config`` never
    back-fills, and a flat-dotted key has no sub-mapping at all, which is
    exactly why ``path`` is declared rather than split from ``key``.

    ``None`` on a setting whose default is ``None`` DELETES the key: that is
    the tri-state's "auto", and writing an explicit ``null`` would make
    ``settings_get`` report an explicit choice where the user asked for the
    automatic one.

    Raises ``ValueError`` when :func:`validate` rejects the value, so no caller
    can write past the schema.
    """
    if setting.kind is Kind.HOTKEY and isinstance(value, str):
        # Normalized HERE as well as in `coerce`, because the writers do not
        # all pass through `coerce`: the server's `PATCH /v1/settings/{key}`
        # hands a raw JSON value straight to this function. `validate_key`
        # normalizes internally before checking, so an un-normalized `ctrl+N`
        # would PASS validation and then be stored verbatim — the page would
        # display `ctrl+N` while the runtime bound a key nobody can press.
        # One normalization at the single point every write funnels through,
        # in the GRAMMAR of the row's own scope (a desktop value is not a
        # Textual key string).
        if setting.hotkey_scope == "desktop":
            value = _keymap.normalize_desktop_key(value, action_id=setting.key)
        else:
            value = _keymap.normalize_key(value)
    # The group check needs the CURRENT config, so it is supplied here rather
    # than left to each caller: this is the one function every writer funnels
    # through, including `lop config edit` and `PATCH /v1/settings`, neither of
    # which has a UI that could warn.
    problem = validate(setting, value, manager.get_config().values)
    if problem is not None:
        raise ValueError(problem)
    if value is None and setting.default is None:
        reset_setting(manager, setting)
        return
    _store(manager, setting.path, value)
    _invalidate_caches()
    _notify_watcher(manager)
    _publish_trigger_settings(manager, setting)
    _sync_aida_hold_marker(manager, setting)


def reset_setting(manager: "ConfigManager", setting: Setting) -> None:
    """Delete the stored value so ``setting`` reads as its default again.

    Deletion rather than "write the default", because for the flat-dotted
    tri-state (``display.nerd_icons``) absence and presence mean different
    things, and because a config that carries only what the user actually chose
    stays readable by hand. Top-level keys that ship in ``DEFAULT_CONFIG`` are
    back-filled on the next load, which lands on the same value from the other
    direction.
    """
    if setting.kind is Kind.READONLY:
        raise ValueError("this setting is retired and cannot be changed")
    _delete(manager, setting.path)
    _invalidate_caches()
    _notify_watcher(manager)
    _publish_trigger_settings(manager, setting)
    _sync_aida_hold_marker(manager, setting)


def _reload_before_write(manager: "ConfigManager") -> None:
    """Re-read config.yml so the write merges into what is on disk NOW.

    THE reason this exists (review round 1, B1): ``set_config_value`` does not
    write one key, it dumps the manager's WHOLE in-memory snapshot
    (``vars(self.config)``). A manager that was constructed a while ago is
    therefore a stale copy of the entire file, and writing one setting through
    it silently reverts every key anything else changed in the meantime.

    That is not a theoretical multi-session race. It fires inside ONE session,
    because the writers here are deliberately short-lived while the readers are
    not: ``OperatorApp._persist_theme`` builds a fresh ``ConfigManager`` per
    call, and ``SettingsView`` holds one captured when the page opened. Open
    ``/settings``, run ``/theme``, toggle any row, and the theme write is gone.

    It is done HERE, at the two primitives every write funnels through, rather
    than in ``write_setting``/``reset_setting``/``write_chains``: a facade-level
    reload has to be repeated in each new entry point and is silently missing
    from the next one added, whereas a primitive-level reload cannot be
    forgotten because there is no way to write without passing through it.

    A config that cannot be read ABORTS the write, and that is checked BEFORE
    the reload rather than caught around it. The reason is the whole of review
    round 2's B3: ``ConfigManager._load_config`` does not raise on a malformed
    config.yml. It prints, moves the file aside to ``config.yml.bad.<stamp>``,
    and returns ``_fresh_default_config()``. So ``reload()`` SUCCEEDS on a
    hand-edit with a tab in it, the manager silently becomes defaults, and the
    write that follows dumps those defaults over the user's file. The `.bad`
    backup holds only the broken two-line edit, so the last good config is then
    recoverable from nowhere — the write is the step that destroys it.

    Degrading to defaults is defensible at STARTUP, where the alternative is a
    lockout, and indefensible as the BASE OF A WRITE. Refusing is recoverable
    in a way that overwriting is not: the user fixes their YAML and the write
    works. Raising rather than swallowing is safe because every caller already
    reports a failed write instead of crashing on it — ``SettingsView._write``
    and the cascade commits hold the reason on the page, ``cli.config edit``
    exits 1 with it.
    """
    _require_readable_config(manager)
    manager.reload()


def _require_readable_config(manager: "ConfigManager") -> None:
    """Raise :class:`ConfigUnreadableError` unless config.yml parses right now.

    Parsed HERE rather than inferred from what ``reload()`` did, because by the
    time ``_load_config`` has degraded to defaults it has already renamed the
    file: there is no undo, and inspecting the manager afterwards cannot
    distinguish "the file was broken" from "the user's config genuinely holds
    default values". The check has to happen while the bytes are still there.

    The shapes that count as unreadable are exactly the ones
    ``_load_config`` degrades on, kept deliberately in step with it: a YAML
    syntax error, a top level that is not a mapping, an empty file, and
    bytes that are not UTF-8 text. The empty case matters because YAML
    parses "" to ``None`` and ``_load_config`` back-fills defaults WITHOUT
    renaming anything, so it degrades just as silently while leaving no
    `.bad` backup at all.

    A file whose top level IS a mapping but whose ``values`` is not passes
    here on purpose: widening the check would put it out of step with
    ``_load_config``, which accepts exactly a mapping. It fails later in the
    write with its bytes intact, which is the property that matters.

    A MISSING file is not unreadable — that is a first run, the reload
    correctly yields defaults, and there is no prior config to destroy.
    """
    config_file = getattr(manager, "config_file", None)
    if config_file is None:
        return
    try:
        raw = config_file.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        # A non-UTF-8 file is unreadable in exactly the sense this function
        # means, and it must say so through this type. `UnicodeDecodeError` is a
        # `ValueError` subclass, so uncaught it was caught one branch EARLIER by
        # the page's `except ValueError` — the schema's slot — and the user saw
        # "'utf-8' codec can't decode byte 0xff in position 0" sitting where
        # "the value you typed is wrong" goes, on a row whose value was fine
        # (review round 3, n2). Reachable from a Windows editor or a PowerShell
        # redirect writing UTF-16.
        raise ConfigUnreadableError(
            f"{config_file} is not valid UTF-8 text ({error.reason} at byte {error.start})"
        ) from error
    except FileNotFoundError:
        return
    except OSError:
        # A permissions or I/O problem is not a corrupt config. Let the write
        # proceed and fail on its own terms, so the caller reports the real
        # errno rather than a misleading "unreadable config".
        return
    if not raw.strip():
        raise ConfigUnreadableError(f"{config_file} is empty")
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as error:
        raise ConfigUnreadableError(f"{config_file} could not be parsed: {error}") from error
    if loaded is not None and not isinstance(loaded, Mapping):
        raise ConfigUnreadableError(
            f"{config_file} is not a configuration mapping "
            f"(top level is {type(loaded).__name__})"
        )


def _store(manager: "ConfigManager", path: Sequence[str], value: Any) -> None:
    _reload_before_write(manager)
    top = path[0]
    if len(path) == 1:
        manager.set_config_value(top, value)
        return
    existing = manager.get_config_value(top, None)
    # A shallow copy per level, so the write never mutates the manager's live
    # mapping before `set_config_value` commits it. A partially-mutated
    # in-memory config that then failed to write would leave the process
    # believing a value that is not on disk.
    root: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    cursor = root
    for part in path[1:-1]:
        child = cursor.get(part)
        cursor[part] = dict(child) if isinstance(child, Mapping) else {}
        cursor = cursor[part]
    cursor[path[-1]] = value
    manager.set_config_value(top, root)


def _delete(manager: "ConfigManager", path: Sequence[str]) -> None:
    # Same staleness trap as `_store` — a delete also dumps the whole snapshot.
    _reload_before_write(manager)
    top = path[0]
    values = manager.get_config().values
    if len(path) == 1:
        if top in values:
            del values[top]
            manager.update_config({}, write=True)
        return
    existing = manager.get_config_value(top, None)
    if not isinstance(existing, Mapping):
        return
    root: dict[str, Any] = dict(existing)
    cursor = root
    for part in path[1:-1]:
        child = cursor.get(part)
        if not isinstance(child, Mapping):
            return
        cursor[part] = dict(child)
        cursor = cursor[part]
    if path[-1] not in cursor:
        return
    del cursor[path[-1]]
    manager.set_config_value(top, root)


def _invalidate_caches() -> None:
    """Drop the process caches a write just invalidated.

    ``tui.settings`` caches the display flags for the life of the process and
    ``settings_reload`` is its ONLY invalidator, so a page that wrote
    ``display.shimmer`` without calling it would leave the running TUI reading
    the old value — the change would appear to have been lost until relaunch.

    Imported function-locally and guarded: this module is imported by the CLI,
    which has no TUI and must not pay for one.
    """
    try:
        from local_operator.tui.settings import settings_reload

        settings_reload()
    except Exception:  # pragma: no cover - a cache drop must never fail a write
        pass


def _notify_watcher(manager: "ConfigManager") -> None:
    """Hand the write to this process's config watcher, if one exists.

    The in-process FAST PATH of :mod:`local_operator.config_watch`: the
    watcher's poll would deliver this change within its interval anyway, but
    a user who toggles ``compaction.enabled`` on the page expects their OWN
    session to honour it on the same keystroke, not two seconds later. The
    watcher re-reads the file and fans out with ``source="local"`` so the TUI
    knows not to announce a change the page already showed.

    Sits beside :func:`_invalidate_caches` at the facade level rather than in
    ``_store``/``_delete`` because a write is one facade call but may be
    several primitive calls; notifying once per facade call is what keeps a
    single edit from being announced twice.

    ``existing_watcher`` rather than ``process_watcher``: the CLI's ``config
    edit`` runs in a process that never started one, and building a watcher
    there would be work with no subscriber. Keyed on the MANAGER's directory,
    not ``paths.config_dir()``, so a write through a manager pointed at some
    other directory (tests, ``--config-dir``) cannot notify the wrong watcher.

    Imported function-locally and guarded for the same reason as
    ``_invalidate_caches``: a notification must never fail a write that has
    already landed on disk.
    """
    try:
        from local_operator.config_watch import existing_watcher

        watcher = existing_watcher(getattr(manager, "config_dir", None))
        if watcher is not None:
            watcher.notify_local()
    except Exception:  # pragma: no cover - a notification must never fail a write
        pass


def _publish_trigger_settings(manager: "ConfigManager", setting: Setting) -> None:
    """Re-publish the wake-trigger settings snapshot after a write or reset.

    The wake supervisor cannot read ``config.yml`` (it is stdlib-only, no
    YAML), so the trigger evaluation pass reads a published snapshot
    (``<config>/wakes/triggers/settings.json``) instead — and THIS hook, on
    the one facade pair every settings edit funnels through, is what makes a
    TUI/UI/CLI edit land at the next evaluation pass (the Wake triggers
    section's LIVE promise). Aida's boot/ensure/reconcile publish too, so a
    hand-edit of ``config.yml`` converges without a settings write.

    GATED ON THE SNAPSHOT'S OWN KEYS: an edit of an unrelated key cannot move
    any snapshot value, so it must not create the ``wakes/triggers/``
    directory either (the publisher's value-equality skip would only make it a
    no-op write once the file exists). Best-effort — a failure must never fail
    the settings write that already landed.
    """
    try:
        from local_operator.wakes import triggers

        if setting.key not in triggers.SNAPSHOT_KEYS:
            return
        triggers.publish_settings(getattr(manager, "config_dir", None), manager=manager)
    except Exception:  # noqa: BLE001 — a cache publish never fails a settings write
        logger.warning("could not publish the trigger settings snapshot", exc_info=True)


# ---------------------------------------------------------------------------
# The failover cascade
# ---------------------------------------------------------------------------
#
# `retry.fallbackChains` is `{chain key: [hop, ...]}` where a hop is either a
# "provider/model" string or a `{provider, model, effort}` mapping. The page
# edits it as two levels (chains, then hops within one chain), so the helpers
# below are the only place that shape is known outside `providers/failover.py`.


def _sync_aida_hold_marker(manager: "ConfigManager", setting: Setting) -> None:
    """Mirror an ``aida.cadence.paused`` write onto her wake-index entry.

    ``/aida pause`` stamps the derived ``held_at`` marker itself, but the SAME
    key written through a settings surface — the /settings row, ``lop config``,
    ``PATCH /v1/settings`` — never touched the entry, so the supervisor's INDEX
    paths kept treating her armed rows as fireable while every settings
    surface said she was paused (review round 1, R1). Routing the write
    through the pause writer's own marker helpers makes both pause surfaces
    produce the same on-disk state. Best-effort by contract: a failure must
    never fail the settings write that already landed.

    A ROWLESS entry is a deliberate no-op: ``store.write_entry`` treats an
    empty schedule list as "remove the entry", so there is nothing there to
    hold — the trigger layer's publish-gated ``triggers.declines`` covers that
    shape instead, and the snapshot re-publish above carries the paused bit to
    the supervisor regardless.
    """
    if setting.key != "aida.cadence.paused":
        return
    try:
        from local_operator.aida import proactive
        from local_operator.wakes import triggers

        root = getattr(manager, "config_dir", None)
        if root is None:
            return
        target = triggers.target_session_id(root)
        if not target:
            return
        paused = strict_bool(read_setting(manager, setting), False)
        if paused:
            proactive.mark_held(root, target)
        else:
            proactive.clear_held(root, target)
    except Exception:  # noqa: BLE001 — a derived marker never fails a settings write
        logger.warning("could not sync aida's hold marker", exc_info=True)


def read_chains(manager: "ConfigManager") -> dict[str, list[str]]:
    """The cascade as ``{key: ["provider/model (effort)", ...]}``.

    Structured hops are flattened to a display LABEL that carries the effort,
    so the page shows the routing decision rather than hiding it. The label is
    also the identity :func:`write_chains` matches on to put the original
    mapping back untouched — see there. Malformed entries are dropped rather
    than rendered, mirroring ``_normalize_chains``: a chain the failover layer
    will ignore must not be shown as if it were live.
    """
    raw = _walk(manager.get_config().values, ("retry", "fallbackChains"))
    if raw is _MISSING or not isinstance(raw, Mapping):
        return {}
    chains: dict[str, list[str]] = {}
    for key, entries in raw.items():
        if not isinstance(key, str) or isinstance(entries, str):
            continue
        if not isinstance(entries, Sequence):
            continue
        hops: list[str] = []
        for entry in entries:
            hop = _hop_label(entry)
            if hop:
                hops.append(hop)
        chains[key] = hops
    return chains


def _hop_label(entry: Any) -> str:
    if isinstance(entry, str):
        return entry.strip()
    if isinstance(entry, Mapping):
        provider = str(entry.get("provider", "") or "").strip()
        model = str(entry.get("model", entry.get("model_id", "")) or "").strip()
        if provider and model:
            effort = str(entry.get("effort", "") or "").strip()
            return f"{provider}/{model}" + (f" ({effort})" if effort else "")
    return ""


def _originals_by_label(manager: "ConfigManager") -> dict[str, dict[str, Any]]:
    """``{chain key: {hop label: the entry exactly as stored}}``.

    The lookup :func:`write_chains` needs to write a hop back in the shape it
    was read in. Keyed by LABEL rather than by index because the page reorders,
    inserts and deletes hops, so an index does not survive an edit while the
    label travels with the hop it names. Two hops in one chain with the same
    label are the same hop as far as every layer here is concerned — the label
    carries provider, model and effort, which is the whole of what the failover
    layer honours — so collapsing them loses nothing THE FAILOVER LAYER
    HONOURS. It is not quite true of the FILE: two same-labelled hops carrying
    different extra keys both come back as the first one's entry.
    ``providers/failover.py`` accepts exactly ``provider``, ``model``,
    ``model_id`` and ``effort`` and already logs anything else as ignored,
    which is why that is a cosmetic loss rather than a routing change (review
    round 2, m6).
    """
    raw = _walk(manager.get_config().values, ("retry", "fallbackChains"))
    if raw is _MISSING or not isinstance(raw, Mapping):
        return {}
    originals: dict[str, dict[str, Any]] = {}
    for key, entries in raw.items():
        if not isinstance(key, str) or isinstance(entries, str):
            continue
        if not isinstance(entries, Sequence):
            continue
        by_label: dict[str, Any] = {}
        for entry in entries:
            label = _hop_label(entry)
            # First occurrence wins: a later duplicate is the same hop.
            if label and label not in by_label:
                by_label[label] = entry
        originals[key] = by_label
    return originals


def write_chains(
    manager: "ConfigManager",
    chains: Mapping[str, Sequence[str]],
    *,
    base: Mapping[str, Sequence[str]] | None = None,
) -> None:
    """Apply ``chains`` to the cascade, dropping empty ones.

    ``base`` is the snapshot the caller READ before it edited, and passing it
    turns a wholesale replace into a MERGE of just the caller's own change.
    Without it this function replaces ``retry.fallbackChains`` outright, which
    is correct for a caller that means "the cascade is exactly this" (the CLI,
    a test) and wrong for the page.

    Why the page must pass it (review round 2, M2): the page builds ``chains``
    from an earlier ``read_chains`` and edits one hop in it. Reloading the
    manager before a wholesale replace reads the fresh on-disk state and then
    discards it, so a chain another session added in the meantime is deleted,
    and a hop another session re-effortted is written back as a BARE SELECTOR
    — the page's stale label no longer matches the stored entry, the originals
    lookup misses, and ``effort`` is dropped from a hop nobody touched. The
    reload alone cannot fix that; only knowing which chains the caller actually
    changed can.

    With ``base``, a chain the caller did not touch is taken from DISK rather
    than from the caller's stale copy, so a concurrent add survives and a
    concurrent effort edit is left alone. A chain the caller did change is
    written as the caller has it, because that is the edit they just made.
    Concurrent edits to THE SAME chain still resolve last-writer-wins, which is
    the one window that cannot be closed without a lock the config format has
    no room for.

    ``chains`` holds DISPLAY LABELS (what :func:`read_chains` returned, with
    the user's edit applied to at most one of them). A hop whose label still
    matches the entry it was read from is written back as THAT ENTRY, byte for
    byte, rather than reconstructed from the label.

    That is the whole point (review round 1, B2). The page edits one hop but
    rewrites every chain, and un-labelling with ``hop.split(" (")[0]`` turned
    every structured ``{provider, model, effort}`` entry in every OTHER chain
    into a bare selector — so adding a hop to one chain silently stripped
    ``effort`` from all the others. ``effort`` is a routing decision, not
    decoration: ``providers/failover.py`` documents it as the "retry cheaper on
    failure" form and warns that flattening it "would silently discard the one
    key that makes the entry mean something different".

    A hop whose label does NOT match anything stored is genuinely new text the
    user typed, so it is stored as the bare selector it reads as. Retyping a
    structured hop's model therefore does drop its effort — correctly: they
    replaced the hop, and the page had no field in which to keep it.

    An empty chain is dropped rather than stored because ``_normalize_chains``
    already drops it on read: keeping it would put a row in the file that the
    page shows and the failover layer does not have, which is the config and
    the behaviour disagreeing again.
    """
    # Before the originals are read, not after: `_store` reloads, and a lookup
    # built from a stale snapshot would restore entries the file no longer has.
    # This also raises if config.yml has become unreadable, so a cascade write
    # aborts rather than dumping defaults over it (round 2, B3).
    _reload_before_write(manager)
    originals = _originals_by_label(manager)
    on_disk = read_chains(manager)

    if base is None:
        # No snapshot: the caller means "the cascade is exactly this".
        merged: dict[str, list[str]] = {key: list(hops) for key, hops in chains.items()}
    else:
        # Only the caller's OWN edits are applied over what is on disk now.
        # Compared by value rather than tracked by a "which chain changed"
        # flag, so the merge stays correct for a caller that edited several
        # chains, and so no caller can forget to declare its edit.
        touched = {
            key
            for key in set(base) | set(chains)
            for before, after in [(list(base.get(key, [])), list(chains.get(key, [])))]
            if before != after
        }
        merged = {key: list(hops) for key, hops in on_disk.items()}
        for key in touched:
            edited = list(chains.get(key, []))
            if edited:
                merged[key] = edited
            else:
                # The caller deleted it. Honour that even though it is still on
                # disk, or a delete would be silently undone by the merge.
                merged.pop(key, None)

    stored = {
        key: [originals.get(key, {}).get(hop, hop.split(" (")[0]) for hop in hops]
        for key, hops in merged.items()
        if key.strip() and hops
    }
    _store(manager, ("retry", "fallbackChains"), stored)
    _invalidate_caches()
    _notify_watcher(manager)


def validate_hop(text: str) -> str | None:
    """``None`` when ``text`` is a usable ``provider/model`` selector.

    A trailing ``(effort)`` is REJECTED rather than quietly dropped. The page
    displays hops as ``openai/gpt-5 (high)``, so a user copying the format it
    had just shown them typed something that was accepted, stored WITHOUT the
    effort, and re-read without the ``(high)`` they typed — the parenthetical
    vanished with nothing on screen saying so (review round 2, n1). Naming the
    boundary is better than silently narrowing the value: the page has no field
    for effort, so a hop carrying one is a hop this editor cannot express.
    """
    candidate = text.strip()
    if not candidate:
        return "expected provider/model"
    if candidate.endswith(")") and " (" in candidate:
        return "effort is not editable here — type provider/model on its own"
    provider, sep, model = candidate.partition("/")
    if not sep or not provider.strip() or not model.strip():
        return "expected provider/model (e.g. openrouter/deepseek/deepseek-chat)"
    return None


#: Description lookup for ``lop config list``, so the CLI's table and the page
#: describe a key with one sentence rather than two that drift. Callers merge
#: their own extras over this.
def descriptions() -> dict[str, str]:
    """``{key: help}`` for every registered setting."""
    return {setting.key: setting.help for setting in SETTINGS}


def resolve_key(key: str) -> Setting | None:
    """The setting named ``key``, or ``None``.

    Exact match only. A near-miss is the CLI's business to suggest — it already
    runs difflib over the key set — and guessing here would let a typo write a
    neighbouring setting.
    """
    return BY_KEY.get(key)


def valid_keys() -> tuple[str, ...]:
    """Every key the CLI's ``config edit`` accepts, sorted for difflib."""
    return tuple(sorted(BY_KEY))


__all__ = [
    "BY_KEY",
    "Choice",
    "Kind",
    "SECTIONS",
    "SETTINGS",
    "Scope",
    "Section",
    "Setting",
    "coerce",
    "descriptions",
    "display_defaults",
    "flat_dotted_keys",
    "is_default",
    "read_chains",
    "read_setting",
    "reset_setting",
    "resolve_key",
    "settings_for",
    "valid_keys",
    "validate",
    "validate_hop",
    "write_chains",
    "write_setting",
]


#: Type of the notice callback the page hands to helpers that can fail
#: partially (a write that lands but whose cache drop did not).
NoticeFn = Callable[[str], None]
