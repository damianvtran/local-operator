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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import yaml

from local_operator import keymap as _keymap
from local_operator.model.effort import EFFORT_ORDER
from local_operator.providers.local import (
    DEFAULT_MODEL_OVERRIDES,
    LOCAL_PRESETS,
    model_overrides,
    validate_endpoint_setting,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from local_operator.config import ConfigManager


logger = logging.getLogger(__name__)


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
    #: :mod:`local_operator.config_watch`). Per-key caveats live on the
    #: SECTION description (``model``: a session that chose with ``/model``
    #: keeps its choice; ``web_tools``: the inventory catches up at the next
    #: turn while execution refuses at once) — the scope says WHEN, the
    #: description says what "applied" means for that key.
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
        "Model",
        Scope.NEW_SESSIONS,
        "Provider, model and reasoning effort for new conversations. "
        "Existing sessions keep their model; /model saved adopts the default here.",
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
        "Wire protocol",
        Scope.LIVE,
        "How direct provider connections are made: API surface and cache TTL.",
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
        "OpenRouter routing",
        Scope.LIVE,
        "How OpenRouter picks a host; unset = sticky routing.",
    ),
    # LIVE: every ``retry.*`` key routes through ``RetrySettings.from_settings``
    # PER CALL on the mapping ``SessionStreamFn`` holds, and the config watcher
    # rebinds that mapping on every change (``SessionStreamFn.apply_settings``).
    Section(
        "failover",
        "Failover and retry",
        Scope.LIVE,
        "What happens when a provider call fails or a quota runs out.",
    ),
    Section(
        "appearance",
        "Appearance",
        Scope.LIVE,
        "Theme and the terminal features the TUI is allowed to use.",
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
        "Hotkeys",
        Scope.LIVE,
        "Keys for starting and resuming conversations. Press enter on a row, "
        "then press the key you want.",
    ),
    # LIVE — a reversal of the original design, which kept the approval mode
    # build-time on the theory that a gate flipping under a running turn is a
    # security-relevant surprise. The operator's actual request ("if I change
    # a setting I want it to go into effect for all my agents") is the
    # opposite: a disk write IS the machine-wide intent, and it overrides any
    # per-session ``/approvals`` toggle. Two rules keep it safe: the new mode
    # applies at the next approval DECISION (``ServingSessionHandle`` reads its
    # flag per gate call), so a prompt already parked on screen is left for
    # the human — never auto-answered, never auto-denied; and every viewer
    # prints the amber "tool approvals now auto" notice. ``--yolo`` is an
    # explicit pin that outranks the key. Its own section because ``session``
    # (autosave + cleanup) is launch-time and scope is uniform per section.
    Section(
        "approvals",
        "Approvals",
        Scope.LIVE,
        "Whether write and command tools prompt, in every running session.",
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
        "Session storage",
        Scope.NEW_LAUNCH,
        # The scope tag one column away already says "takes effect: new
        # launch", so restating "read once at launch" here said "launch" twice
        # within one row (design round 1, D7).
        "Autosave and the cleanup policy.",
    ),
    # LIVE: ``max_running`` is pushed into the running ``AsyncJobManager`` by
    # ``Session._apply_config_change`` (raising it lets the next launch through;
    # lowering it lets running jobs finish — nothing is evicted), the
    # ``models.*`` tiers are read at every spawn, and ``model_choice`` is read
    # both by the rebuild that re-renders the two tool schemas and by the
    # tool-argument refusal itself.
    Section(
        "subagents",
        "Subagents",
        Scope.LIVE,
        "Concurrency cap, who picks a child's model, and the model each tier runs on.",
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
        # for (design round 2, D11). The runtime's own message is separate and
        # unchanged: it says "Suggestion added…".
        "Smart hints",
        Scope.NEW_SESSIONS,
        "Advisory skills, guides and MCP servers a decision model may suggest "
        "per message. Off keeps the prompt unchanged.",
    ),
    # Its own section rather than a row under "Session", and the reason is the
    # SCOPE: scope is uniform within a section by construction, "Session" is
    # launch-time (autosave and the cleanup policy), and these keys take
    # effect on the very next /resume in this same terminal. Filing a live key
    # under a section labelled for launch is exactly the painted lie AGENTS.md
    # warns about — split the section.
    Section(
        "runtime",
        "Runtime",
        Scope.LIVE,
        "How sessions behave when you leave them.",
    ),
    # LIVE: the session re-coerces its ``CompactionSettings`` on every change,
    # and all three trigger checks read that attribute at check time.
    Section(
        "compaction",
        "Compaction",
        Scope.LIVE,
        "When the conversation is summarised to reclaim context.",
    ),
    # LIVE: ``/fork`` reads these through the config manager at the moment it
    # runs, so an edit takes effect on the very next fork.
    Section(
        "fork",
        "Fork",
        Scope.LIVE,
        "Where /fork opens the branched conversation.",
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
        "Web tools",
        Scope.LIVE,
        "Whether the search and fetch tools are offered and allowed to run.",
    ),
    # LIVE: both tools build their settings from config on EVERY call
    # (``web_search/tool.py``, ``web_fetch/tool.py``).
    Section(
        "web_search",
        "Web search",
        Scope.LIVE,
        "Providers and load balancing for the search tool.",
    ),
    Section(
        "web_fetch",
        "Web fetch",
        Scope.LIVE,
        "Limits and rendering for the fetch tool.",
    ),
    # LIVE for the same reason as the two web sections: ``execute_bash`` reads
    # ``bash.shell`` through a fresh ``ConfigManager(config_dir())`` on EVERY
    # call (``tools/builtin.py::_configured_bash_shell``), so an edit lands on
    # the very next command. Its own section rather than a row under "Session"
    # or "Runtime": scope is uniform within a section by construction, and
    # this is about how a tool executes, not how sessions behave.
    Section(
        "tools",
        "Tools",
        Scope.LIVE,
        "How the built-in tools execute.",
    ),
    # Split out of ``tools`` (review round 1, M2), for the reason the module's
    # own history gives for ``providers`` and ``web_tools``: scope is uniform
    # within a section by construction, and these three keys are the one part of
    # how a tool executes that must NOT re-read per call — a live policy is a
    # policy the agent's own shell can weaken between two commands.
    Section(
        "shell_environment",
        "Agent shell",
        Scope.NEW_LAUNCH,
        "What a command the agent runs can see. Resolved once per session, so an "
        "edit here applies at the next launch and cannot weaken a session that is "
        "already running.",
    ),
    Section(
        "local_providers",
        "Local servers",
        Scope.NEW_SESSIONS,
        "Server endpoints and exact-model metadata. Use /login to connect; "
        "reselect with /model saved to apply model changes.",
    ),
    # LIVE: the click handler is a FRESH PROCESS every time it runs
    # (``lop resume-click`` is spawned by the notification), so it reads config
    # at click time and an edit lands on the very next click. Nothing is
    # threaded through a running session, which is why this cannot be
    # NEW_SESSIONS like the knobs that gate a session's construction.
    Section(
        "desktop",
        "Desktop app",
        Scope.LIVE,
        "Where a notification click sends you when the desktop app is not running.",
    ),
    Section(
        "retired",
        "Retired",
        Scope.NEW_LAUNCH,
        "Keys that are read but no longer do anything.",
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
        if not os.access(executable, os.X_OK):
            raise ValueError(f"{executable}{REJECTION_VALUE_SEP}{ADVICE_NOT_EXECUTABLE}")
        return
    if shutil.which(executable) is None and not _resolvable_for_the_user(executable):
        raise ValueError(f"{executable}{REJECTION_VALUE_SEP}{ADVICE_NOT_ON_PATH}")


def _bool_choices(on: str, off: str) -> tuple[Choice, ...]:
    return (Choice(True, "on", on), Choice(False, "off", off))


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
    registry is fixed for the life of the process (themes are declared, not
    installed at runtime), so a stale cache is not reachable. Without this the
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
    "none": "reasoning off",
    "minimal": "the least reasoning the model offers",
    "low": "light reasoning",
    "medium": "moderate reasoning",
    "high": "deep reasoning",
    "xhigh": "very deep reasoning",
    "max": "the model's deepest reasoning",
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
        Choice("", "auto", "the model's own default"),
        *(Choice(level, level, _EFFORT_LEVEL_HELP.get(level, "")) for level in EFFORT_ORDER),
    )


SETTINGS: tuple[Setting, ...] = (
    # -- model --------------------------------------------------------------
    Setting(
        key="hosting",
        path=("hosting",),
        section="model",
        label="Default provider",
        kind=Kind.TEXT,
        default="",
        # 72 cells. Every help string on this page has to clear the ~76-cell
        # footer budget at 80 columns: past it the shed ladder drops the YAML
        # key path (the thing a user maps a row to the file by) and then the
        # sentence itself is clipped mid-clause with no ellipsis (design round
        # 1, D2). Measure any edit to these four before landing it.
        help="Provider for new conversations. /model saved adopts it here.",
        empty_unsets=True,
    ),
    Setting(
        key="model_name",
        path=("model_name",),
        section="model",
        label="Default model",
        kind=Kind.TEXT,
        default="",
        # 72 cells — see the note on `hosting` above.
        help="Model for new conversations. /model saved adopts it here.",
        empty_unsets=True,
    ),
    Setting(
        key="model_effort",
        path=("model_effort",),
        section="model",
        label="Default reasoning effort",
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
        help="Effort for new conversations. Unset: the model's default.",
        choices=_effort_choices(),
    ),
    # -- providers ----------------------------------------------------------
    Setting(
        key="providers.openai.use_max_context_window",
        path=("providers", "openai", "use_max_context_window"),
        section="providers",
        label="Use maximum OpenAI context",
        kind=Kind.BOOL,
        default=True,
        help=(
            "Off: provider default. On: supported max. "
            "Applies next request; compaction unchanged."
        ),
    ),
    Setting(
        key="providers.openai.api",
        path=("providers", "openai", "api"),
        section="providers",
        label="OpenAI API surface",
        kind=Kind.ENUM,
        default="responses",
        help="Direct OpenAI GPT-5 calls use the Responses API unless opted out.",
        choices=(
            Choice("responses", "responses", "the public Responses API (default)"),
            Choice("chat_completions", "chat_completions", "explicit compatibility opt-out"),
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
        label="Anthropic 1h cache above (tokens)",
        kind=Kind.INT,
        default=150_000,
        help=(
            "Context size from which Anthropic requests use the 1-hour prompt-cache "
            "TTL (2x write cost, survives idle gaps over 5 minutes). 0 disables."
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
        label="cache affinity",
        kind=Kind.BOOL,
        default=True,
        help=(
            "Reuses the host that served the previous turn so the prompt cache "
            "stays warm. Off = OpenRouter's price-weighted load balancing."
        ),
    ),
    Setting(
        key="providers.openrouter.sort",
        path=("providers", "openrouter", "sort"),
        section="openrouter",
        label="routing policy",
        kind=Kind.ENUM,
        # "" is the unset member: the schema must know a stored empty string
        # means "no opinion", and the resolver treats it exactly like a missing
        # key. An ENUM member rather than `empty_unsets` so the page can show
        # the three real policies beside "default" as peers to pick between.
        default="",
        help=(
            "How OpenRouter ranks hosts for this call. May route away from the "
            "host holding your warm prompt cache (a cold start on long "
            "conversations); 'default' sends no preference at all."
        ),
        choices=(
            Choice("", "default", "no preference — sticky routing stays on (warmest cache)"),
            Choice("price", "price", "cheapest host first"),
            Choice("throughput", "throughput", "highest tokens/sec first"),
            Choice("latency", "latency", "lowest response latency first"),
        ),
    ),
    Setting(
        key="providers.openrouter.order",
        path=("providers", "openrouter", "order"),
        section="openrouter",
        label="host order",
        kind=Kind.LIST,
        default=[],
        # The consequence LEADS the detail line in the page's danger ink and is
        # never shed from it — including on the off-default row, where the
        # `default:` reset clause used to crowd it out exactly when the
        # dangerous value was stored (QA round 1 Q1 / design round 1 D2). The
        # help below stays SOFT faint ink so the two ranks read apart.
        warning="disables sticky routing — prompt cache goes cold",
        # Empty-first (design round 1, D3): empty is the default that must not
        # be disturbed, so the detail names what empty MEANS before the how-to.
        help=(
            "Empty = no opinion (sticky routing stays). Comma-separated host "
            "slugs tried in this exact order."
        ),
        # OPEN namespace — deliberately no `members`. OpenRouter owns the slug
        # vocabulary and grows it without notice (deepinfra, novita, regional
        # variants like google-vertex/us-east5), so a closed list would reject
        # hosts the upstream docs themselves use. Any non-empty slug token
        # validates; the placeholder seeds the common hosts without gating
        # the write.
        placeholder="deepseek, groq, mistral, …",
        # An empty routing order is "no opinion", not a validation error —
        # unlike web_search.providers, nothing breaks with zero entries.
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.only",
        path=("providers", "openrouter", "only"),
        section="openrouter",
        label="allowed hosts",
        kind=Kind.LIST,
        default=[],
        help=(
            "Empty = no opinion (sticky routing stays). Comma-separated "
            "allow-list of host slugs; every other host is excluded — a "
            "forced cold start if the warm host is not on it."
        ),
        # Open namespace, same reason as `order` above.
        placeholder="deepseek, groq, mistral, …",
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.ignore",
        path=("providers", "openrouter", "ignore"),
        section="openrouter",
        label="ignored hosts",
        kind=Kind.LIST,
        default=[],
        help=(
            "Empty = no opinion (sticky routing stays). Comma-separated "
            "block-list of host slugs to omit (e.g. a host that is failing "
            "right now) — may cost you the warm prompt cache."
        ),
        # Open namespace, same reason as `order` above.
        placeholder="deepseek, groq, mistral, …",
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.allow_fallbacks",
        path=("providers", "openrouter", "allow_fallbacks"),
        section="openrouter",
        label="fallbacks",
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
            "'default' sends nothing (OpenRouter falls through to another "
            "host when the preferred one fails); 'false' fails rather than "
            "fall through to a host outside your preferences."
        ),
        choices=(
            Choice("", "default", "no preference — OpenRouter's default (fall through)"),
            Choice("false", "false", "fail rather than fall through"),
        ),
    ),
    Setting(
        key="providers.openrouter.require_parameters",
        path=("providers", "openrouter", "require_parameters"),
        section="openrouter",
        label="require parameters",
        # ENUM for the same tri-state reason as `allow_fallbacks`: False in a
        # BOOL read as "parameter dropping is off" while the wire truth is
        # "no preference sent"; "" restores the shared no-opinion `—`.
        kind=Kind.ENUM,
        default="",
        help=(
            "'true' restricts to hosts that support every parameter you send "
            "(tools, structured output); 'default' lets OpenRouter silently "
            "drop unsupported ones."
        ),
        choices=(
            Choice("", "default", "no preference — unsupported parameters may be dropped"),
            Choice("true", "true", "only hosts that support every parameter"),
        ),
    ),
    Setting(
        key="providers.openrouter.data_collection",
        path=("providers", "openrouter", "data_collection"),
        section="openrouter",
        label="data collection",
        kind=Kind.ENUM,
        default="",
        help=(
            "'deny' excludes hosts that may train on or retain your prompts. "
            "'default' sends no preference (OpenRouter's default: allow)."
        ),
        choices=(
            Choice("", "default", "no preference sent"),
            Choice("allow", "allow", "hosts may collect data"),
            Choice("deny", "deny", "exclude hosts that collect data"),
        ),
    ),
    Setting(
        key="providers.openrouter.zdr",
        path=("providers", "openrouter", "zdr"),
        section="openrouter",
        label="zero data retention",
        kind=Kind.ENUM,
        # ENUM, not BOOL (design round 1, D5): the wire key is tri-state —
        # absent means "no preference" (OpenRouter's default stands), `true`
        # means ZDR hosts only, and there is no meaningful `false` to send.
        # A BOOL had to fake the unset state as `off`, which read as "ZDR
        # disabled" beside ENUM rows showing `—` for the same state.
        default="",
        help="'true': restrict to zero-data-retention endpoints. 'default': nothing sent.",
        choices=(
            Choice("", "default", "no preference sent"),
            Choice("true", "true", "zero-data-retention endpoints only"),
        ),
    ),
    Setting(
        key="providers.openrouter.enforce_distillable_text",
        path=("providers", "openrouter", "enforce_distillable_text"),
        section="openrouter",
        label="distillable text",
        # ENUM for the same tri-state reason as `zdr` directly above.
        kind=Kind.ENUM,
        default="",
        help=(
            "'true': restrict to endpoints whose text output may be "
            "distilled. 'default': nothing sent."
        ),
        choices=(
            Choice("", "default", "no preference sent"),
            Choice("true", "true", "distillable-text endpoints only"),
        ),
    ),
    Setting(
        key="providers.openrouter.quantizations",
        path=("providers", "openrouter", "quantizations"),
        section="openrouter",
        label="quantizations",
        kind=Kind.LIST,
        default=[],
        help=(
            "Empty = no opinion. Comma-separated quantization levels the "
            "served model may use (e.g. int8, fp8, mxfp4)."
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
        placeholder="int8, fp8, mxfp4, …",
        empty_unsets=True,
    ),
    Setting(
        key="providers.openrouter.max_price",
        path=("providers", "openrouter", "max_price"),
        section="openrouter",
        label="max price",
        kind=Kind.TEXT,
        default="",
        # Example-first and short (design round 1, D4): the old prose pushed
        # the units past the ellipsis at 120 columns, and for a JSON-in-TEXT
        # row the example IS the documentation. The empty editor ghosts the
        # same example; the four accepted field names are enumerated by the
        # validator's own rejection message.
        help='{"prompt": 1, "completion": 2} — USD / million tokens',
        placeholder='{"prompt": 1, "completion": 2}',
        empty_unsets=True,
        validate_value=_validate_openrouter_max_price,
    ),
    Setting(
        key="providers.openrouter.preferred_min_throughput",
        path=("providers", "openrouter", "preferred_min_throughput"),
        section="openrouter",
        label="min throughput (tok/s)",
        kind=Kind.FLOAT,
        default=0.0,
        help=(
            "Prefer hosts serving at least this many output tokens/sec "
            "(p50). 0 sends no preference. A host below the bar is deprioritised, "
            "which may cost you a warm prompt cache."
        ),
        minimum=0.0,
        maximum=100_000.0,
    ),
    Setting(
        key="providers.openrouter.preferred_max_latency",
        path=("providers", "openrouter", "preferred_max_latency"),
        section="openrouter",
        label="max latency (s)",
        kind=Kind.FLOAT,
        default=0.0,
        help=(
            "Prefer hosts whose time-to-first-token stays under this many "
            "seconds (p50). 0 sends no preference. A host above the bar is "
            "deprioritised, which may cost you a warm prompt cache."
        ),
        minimum=0.0,
        maximum=600.0,
    ),
    # -- failover -----------------------------------------------------------
    Setting(
        key="retry.enabled",
        path=("retry", "enabled"),
        section="failover",
        label="Retry failed calls",
        kind=Kind.BOOL,
        default=True,
        help="Retry a failed provider call before surfacing the error.",
        choices=_bool_choices("retry with backoff", "fail on the first error"),
    ),
    Setting(
        key="retry.maxRetries",
        path=("retry", "maxRetries"),
        section="failover",
        label="Max retries",
        kind=Kind.INT,
        default=10,
        help="Fast budget against a reachable provider (5xx, timeout).",
        minimum=0,
        maximum=100,
    ),
    Setting(
        key="retry.baseDelayMs",
        path=("retry", "baseDelayMs"),
        section="failover",
        label="Base delay (ms)",
        kind=Kind.INT,
        default=500,
        help="First backoff step; later attempts grow from it.",
        minimum=0,
        maximum=60_000,
    ),
    Setting(
        key="retry.connectivityMaxRetries",
        path=("retry", "connectivityMaxRetries"),
        section="failover",
        label="Connectivity retries",
        kind=Kind.INT,
        default=15,
        help="Patient budget for a machine that went offline; distinct from max retries.",
        minimum=0,
        maximum=200,
    ),
    Setting(
        key="retry.connectivityBackoffCapMs",
        path=("retry", "connectivityBackoffCapMs"),
        section="failover",
        label="Connectivity backoff cap (ms)",
        kind=Kind.INT,
        default=60_000,
        help="Longest wait between connectivity retries.",
        minimum=1_000,
        maximum=600_000,
    ),
    Setting(
        key="retry.modelFallback",
        path=("retry", "modelFallback"),
        section="failover",
        label="Model fallback",
        kind=Kind.BOOL,
        default=True,
        help="Move to the next hop in the cascade when a model keeps failing.",
        choices=_bool_choices("fall back to the next hop", "stay on the chosen model"),
    ),
    Setting(
        key="retry.usageAwareFallback",
        path=("retry", "usageAwareFallback"),
        section="failover",
        label="Usage-aware fallback",
        kind=Kind.BOOL,
        default=False,
        help="Switch before a quota runs out. Costs one quota request per user message.",
        choices=_bool_choices("check quota at message boundaries", "only react to failures"),
    ),
    Setting(
        key="retry.usageAwareAccountPick",
        path=("retry", "usageAwareAccountPick"),
        section="failover",
        label="Usage-aware account pick",
        kind=Kind.BOOL,
        default=True,
        help=(
            "Start new sessions on the same-provider account with the most quota left, "
            "read from the cached /usage report. Applies to sessions only."
        ),
        choices=_bool_choices("prefer the least-loaded account", "spread by session hash only"),
    ),
    Setting(
        key="retry.usageReservePercent",
        path=("retry", "usageReservePercent"),
        section="failover",
        label="Usage reserve (%)",
        kind=Kind.FLOAT,
        default=10.0,
        help="Headroom below which an account counts as low; a running session stays on it.",
        minimum=0.0,
        maximum=100.0,
    ),
    Setting(
        key="retry.fallbackChains",
        path=("retry", "fallbackChains"),
        section="failover",
        label="Failover cascade",
        kind=Kind.CASCADE,
        default={},
        help="Ordered provider/model hops tried when a call keeps failing.",
    ),
    # -- appearance ---------------------------------------------------------
    Setting(
        key="tui.theme",
        path=("tui", "theme"),
        section="appearance",
        label="Theme",
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
        help="Colour ramp. /theme switches it live with an arrow-key preview.",
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
        label="Shimmer animation",
        kind=Kind.BOOL,
        default=True,
        help="The animated sheen on the working line.",
        choices=_bool_choices("animate the working line", "static working line"),
    ),
    Setting(
        key="display.narration",
        path=("display.narration",),
        section="appearance",
        label="Mid-turn narration",
        kind=Kind.BOOL,
        default=True,
        help="Keep the agent's mid-turn narration in the transcript after its tool calls run.",
        choices=_bool_choices("keep narration", "hide narration once tools run"),
    ),
    Setting(
        key="display.rail",
        path=("display.rail",),
        section="appearance",
        label="Assistant gutter rail",
        kind=Kind.BOOL,
        default=True,
        help=(
            "A rule marks the ANSWER: the message that ends the turn. "
            "Mid-turn narration takes no rail."
        ),
        choices=_bool_choices("rail the answer only", "no rail"),
    ),
    Setting(
        # Default changed to False by maintainer
        key="display.comfortable_rows",
        path=("display.comfortable_rows",),
        section="appearance",
        label="Comfortable action rows",
        kind=Kind.BOOL,
        default=False,
        help="Pad tool and prompt rows so they are easier to click.",
        choices=_bool_choices("padded, easier to click", "compact, more history"),
    ),
    Setting(
        key="display.nerd_icons",
        path=("display.nerd_icons",),
        section="appearance",
        label="Nerd Font glyphs",
        kind=Kind.ENUM,
        default=None,
        help="Expanded tool-row icons. Auto reads the terminal emulator's markers.",
        # The tri-state IS the None-vs-bool distinction: `settings_get` returns
        # None only when the key is ABSENT, which is what "auto" reads. So the
        # auto choice must write nothing rather than write a value — handled by
        # `write_setting`, which deletes on a None for a key with no shipped
        # default.
        choices=(
            Choice(None, "auto", "decide from the terminal emulator"),
            Choice(True, "on", "force glyphs on"),
            Choice(False, "off", "force plain icons"),
        ),
    ),
    Setting(
        key="display.heading_markers",
        path=("display.heading_markers",),
        section="appearance",
        label="Heading markers",
        kind=Kind.BOOL,
        default=False,
        help="Show the literal ### before a heading, as the markdown source writes it.",
        choices=_bool_choices("show ### markers", "colour and weight only"),
    ),
    Setting(
        key="display.terminal_title",
        path=("display.terminal_title",),
        section="appearance",
        label="Terminal title",
        kind=Kind.BOOL,
        default=True,
        help="OSC 0 window title carrying the session name and run state.",
        choices=_bool_choices("set the window title", "leave the title alone"),
    ),
    Setting(
        key="display.images",
        path=("display.images",),
        section="appearance",
        label="Inline images",
        kind=Kind.BOOL,
        default=True,
        help="Screenshots and attachments drawn in the transcript.",
        choices=_bool_choices("draw images", "text receipts only"),
    ),
    Setting(
        key="display.notifications",
        path=("display.notifications",),
        section="appearance",
        label="Desktop notifications",
        kind=Kind.BOOL,
        default=True,
        help="Fires only while the terminal is unfocused.",
        choices=_bool_choices("notify when unfocused", "never notify"),
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
        label="Notification session names",
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
            "A session's name, last line and error causes appear on banners, "
            "including the lock screen."
        ),
        # `off` no longer means "app name only" on every route — a background
        # session with no stored title is titled "A session finished" — so the
        # label names what the user gets rather than a fallback that is now one
        # of two (design round 1, D7, folded into D2).
        choices=_bool_choices("name the session", "keep names off banners"),
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
        label="Wake time format",
        kind=Kind.ENUM,
        default="12h",
        help="Scheduled wake times use your local timezone. Choose a 12- or 24-hour clock.",
        choices=(
            Choice("12h", "12-hour", "7:52 PM PDT"),
            Choice("24h", "24-hour", "19:52 PDT"),
        ),
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
        label="Subagent dock",
        kind=Kind.ENUM,
        default="full",
        help="How much room the subagent panel takes when a session starts; ctrl+g cycles it.",
        choices=(
            # Parallel descriptions: each names WHAT IS SHOWN and nothing
            # else. `hidden` used to append "; ctrl+g brings it back", which
            # made it the only choice explaining its own exit and left the
            # list reading unevenly — and the help line above already names
            # `ctrl+g` for all three (round 1, D5).
            Choice("full", "full", "one row per child, newest first"),
            Choice("summary", "summary", "a one-line count"),
            Choice("hidden", "hidden", "not shown"),
        ),
    ),
    Setting(
        key="tui.sidebar_visible",
        path=("tui", "sidebar_visible"),
        section="appearance",
        label="Session sidebar",
        kind=Kind.BOOL,
        default=False,
        help="Active and recent conversations. Ctrl+B toggles the sidebar.",
        choices=_bool_choices("show the sidebar", "keep the full conversation width"),
    ),
    Setting(
        key="tui.sidebar_position",
        path=("tui", "sidebar_position"),
        section="appearance",
        label="Sidebar position",
        kind=Kind.ENUM,
        default="left",
        help="Which side of the conversation holds the session sidebar.",
        choices=(
            Choice("left", "left", "sessions to the left of the conversation"),
            Choice("right", "right", "sessions to the right of the conversation"),
        ),
    ),
    Setting(
        key="tui.sidebar_show_subagents",
        path=("tui", "sidebar_show_subagents"),
        section="appearance",
        label="Sidebar subagent layer",
        kind=Kind.BOOL,
        default=False,
        help=(
            "List recent subagent runs below your own sessions. "
            "Ctrl+A toggles the layer while the sidebar has focus."
        ),
        choices=_bool_choices("list recent subagent runs", "show only your own sessions"),
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
        )
        for action in _keymap.KEY_ACTIONS
    ),
    # -- approvals ----------------------------------------------------------
    Setting(
        key="tool_approval_mode",
        path=("tool_approval_mode",),
        section="approvals",
        label="Tool approval mode",
        kind=Kind.ENUM,
        default="ask",
        # 74 cells — see the note on `hosting`.
        help="How every running session treats write and exec tools, from its next call.",
        choices=(
            Choice("ask", "ask", "prompt before write/exec tools"),
            Choice("auto", "auto", "run them without asking"),
        ),
    ),
    # -- session storage ----------------------------------------------------
    Setting(
        key="auto_save_conversation",
        path=("auto_save_conversation",),
        section="session",
        label="Auto-save conversation",
        kind=Kind.BOOL,
        default=False,
        help="Headless REPL launches only; the TUI's runtime does not read it.",
        choices=_bool_choices("save automatically", "save on request"),
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
        label="Keep working after /resume",
        kind=Kind.BOOL,
        default=True,
        help="Leave a running turn working when you switch away from its session.",
        choices=_bool_choices(
            "keep the turn running in the background",
            "stop the turn when you leave the session",
        ),
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
        section="session",
        label="Session cleanup",
        kind=Kind.BOOL,
        default=False,
        help=(
            "Off: nothing is ever removed. "
            "On: limits below run at launch, sparing the newest 10 + live."
        ),
        choices=_bool_choices(
            "limits run at launch; newest 10 + live kept",
            "nothing is ever removed",
        ),
    ),
    Setting(
        key="session.cleanup.max_sessions",
        path=("session", "cleanup", "max_sessions"),
        section="session",
        label="↳ max sessions",
        kind=Kind.INT,
        default=0,
        help="Needs cleanup on. Keep N newest /resume sessions; 0 = no cap.",
        minimum=0,
        gated_by="session.cleanup.enabled",
    ),
    Setting(
        key="session.cleanup.max_inactive_days",
        path=("session", "cleanup", "max_inactive_days"),
        section="session",
        label="↳ max inactive days",
        kind=Kind.INT,
        default=0,
        help="Needs cleanup on. Remove sessions idle this many days; 0 = never.",
        minimum=0,
        gated_by="session.cleanup.enabled",
    ),
    Setting(
        key="session.cleanup.max_total_bytes",
        path=("session", "cleanup", "max_total_bytes"),
        section="session",
        label="↳ max total bytes",
        kind=Kind.INT,
        default=0,
        help="Needs cleanup on. Trim oldest past this many bytes; 0 = no cap.",
        minimum=0,
        gated_by="session.cleanup.enabled",
    ),
    Setting(
        key="session.cleanup.remove_empty",
        path=("session", "cleanup", "remove_empty"),
        section="session",
        label="↳ remove empty",
        kind=Kind.BOOL,
        default=False,
        help="Needs cleanup on. Remove dirs that never got a transcript.",
        choices=_bool_choices("remove transcript-less directories", "keep them"),
        gated_by="session.cleanup.enabled",
    ),
    Setting(
        key="runtime.unattended_gate_timeout",
        path=("runtime", "unattended_gate_timeout"),
        section="runtime",
        label="Unattended question timeout (h)",
        kind=Kind.INT,
        default=24,
        help="How long a question waits when you are away. 0 never times out.",
        minimum=0,
        maximum=720,
    ),
    Setting(
        key="subagents.max_running",
        path=("subagents", "max_running"),
        section="subagents",
        label="Max background jobs",
        kind=Kind.INT,
        default=15,
        help="Ceiling on concurrent subagents and backgrounded bash, which share one pool.",
        minimum=1,
        maximum=64,
    ),
    Setting(
        key="subagents.model_choice",
        path=("subagents", "model_choice"),
        section="subagents",
        label="Who picks a subagent's model",
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
        help="Who picks a subagent's tier; the operator pins one in its profile.",
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
                "the operator",
                "inherits the session model",
            ),
            Choice(
                "model",
                "the model",
                "may run the subagent on a costlier model",
            ),
        ),
    ),
    Setting(
        key="subagents.models.lo",
        path=("subagents", "models", "lo"),
        section="subagents",
        label="Subagent model: lo",
        kind=Kind.TEXT,
        default="",
        # The billing fact and the picker pointer are the two things this row
        # was missing, and the incident is why: a deliberate tier pin read as
        # harmless because nothing said a child on it RUNS, and is billed, at
        # that model's rates, or that "Who picks a subagent's model" is what
        # decides who may choose it.
        #
        # Length is budgeted, not styled, and the budget is TIGHT: the detail
        # line sheds the WHOLE help once the key path no longer fits beside it
        # (settings_view._detail_clause), and at 100 columns that row is 94 cells
        # with `subagents.models.hi` (19) plus its separator taking 22 — 72 cells
        # of help. An earlier version measured 73 and shed the key path, which
        # the comment beside it wrongly claimed it did not; this is 71 and was
        # re-measured on a rendered frame at 80/100/140 rather than estimated.
        # At 80 the key path RENDERS (the row is 74 cells there and the rung
        # paints `help · clause · key`), so the earlier note that it was "still
        # shed" at that width was wrong in the safe direction — the frame is the
        # only thing that settles it, which is why the bounds below are measured
        # rather than derived.
        #
        # The pointer names the row (by the key the page greppable from, which is
        # also the spelling `lop config edit` takes) instead of saying "row
        # above": registry order is max_running, model_choice, lo, med, hi, so
        # "above" would point med at lo and hi at med — and `hi` is the row this
        # incident ran through.
        help="Bills at that model's rates; empty inherits. See subagents.model_choice",
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.med",
        path=("subagents", "models", "med"),
        section="subagents",
        label="Subagent model: med",
        kind=Kind.TEXT,
        default="",
        # The billing fact and the picker pointer are the two things this row
        # was missing, and the incident is why: a deliberate tier pin read as
        # harmless because nothing said a child on it RUNS, and is billed, at
        # that model's rates, or that "Who picks a subagent's model" is what
        # decides who may choose it.
        #
        # Length is budgeted, not styled, and the budget is TIGHT: the detail
        # line sheds the WHOLE help once the key path no longer fits beside it
        # (settings_view._detail_clause), and at 100 columns that row is 94 cells
        # with `subagents.models.hi` (19) plus its separator taking 22 — 72 cells
        # of help. An earlier version measured 73 and shed the key path, which
        # the comment beside it wrongly claimed it did not; this is 71 and was
        # re-measured on a rendered frame at 80/100/140 rather than estimated.
        # At 80 the key path RENDERS (the row is 74 cells there and the rung
        # paints `help · clause · key`), so the earlier note that it was "still
        # shed" at that width was wrong in the safe direction — the frame is the
        # only thing that settles it, which is why the bounds below are measured
        # rather than derived.
        #
        # The pointer names the row (by the key the page greppable from, which is
        # also the spelling `lop config edit` takes) instead of saying "row
        # above": registry order is max_running, model_choice, lo, med, hi, so
        # "above" would point med at lo and hi at med — and `hi` is the row this
        # incident ran through.
        help="Bills at that model's rates; empty inherits. See subagents.model_choice",
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.hi",
        path=("subagents", "models", "hi"),
        section="subagents",
        label="Subagent model: hi",
        kind=Kind.TEXT,
        default="",
        # The billing fact and the picker pointer are the two things this row
        # was missing, and the incident is why: a deliberate tier pin read as
        # harmless because nothing said a child on it RUNS, and is billed, at
        # that model's rates, or that "Who picks a subagent's model" is what
        # decides who may choose it.
        #
        # Length is budgeted, not styled, and the budget is TIGHT: the detail
        # line sheds the WHOLE help once the key path no longer fits beside it
        # (settings_view._detail_clause), and at 100 columns that row is 94 cells
        # with `subagents.models.hi` (19) plus its separator taking 22 — 72 cells
        # of help. An earlier version measured 73 and shed the key path, which
        # the comment beside it wrongly claimed it did not; this is 71 and was
        # re-measured on a rendered frame at 80/100/140 rather than estimated.
        # At 80 the key path RENDERS (the row is 74 cells there and the rung
        # paints `help · clause · key`), so the earlier note that it was "still
        # shed" at that width was wrong in the safe direction — the frame is the
        # only thing that settles it, which is why the bounds below are measured
        # rather than derived.
        #
        # The pointer names the row (by the key the page greppable from, which is
        # also the spelling `lop config edit` takes) instead of saying "row
        # above": registry order is max_running, model_choice, lo, med, hi, so
        # "above" would point med at lo and hi at med — and `hi` is the row this
        # incident ran through.
        help="Bills at that model's rates; empty inherits. See subagents.model_choice",
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
        label="Smart hints",
        kind=Kind.BOOL,
        default=False,
        # Same precedent as `values.effort.auto` (`model/effort_classifier.py`):
        # default OFF, because an upgrade must never silently change behaviour or
        # spend. The help says what ON does rather than what the feature is: the
        # label already names the feature, and it now names it in the one term
        # the tips and `guide://classification` use (D11).
        help="Off: the prompt is unchanged. On: a decision model may add advisory resources.",
        choices=_bool_choices(
            "a decision model may add advisory resources",
            "the prompt stays exactly as it is",
        ),
    ),
    Setting(
        key="classification.vendor",
        path=("classification", "vendor"),
        section="classification",
        label="↳ vendor",
        kind=Kind.ENUM,
        default="auto",
        # `auto` is a real member here rather than the unset spelling
        # `model_effort` uses: the cascade's own default IS "first leg with a
        # usable credential", and storing that as an empty string would leave
        # the page unable to show which of the two the operator chose.
        help="Needs recommendations on. Pin one leg; auto prefers Radient.",
        choices=(
            Choice("auto", "auto", "first leg with a usable credential"),
            Choice("radient", "radient", "Radient's decision route"),
            Choice("typesafe", "typesafe", "TypeSafe's Jev endpoint"),
            Choice("openrouter", "openrouter", "OpenRouter's alpha decisions route"),
        ),
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.model",
        path=("classification", "model"),
        section="classification",
        label="↳ model",
        kind=Kind.TEXT,
        default="",
        help="Needs recommendations on. Vendor model id; empty uses its default.",
        empty_unsets=True,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.timeoutMs",
        path=("classification", "timeoutMs"),
        section="classification",
        label="↳ deadline (ms)",
        kind=Kind.INT,
        default=1500,
        # 0 is "use the default", not "no deadline": the reader refuses a
        # non-positive value rather than treating it as unlimited, so a hand-edit
        # cannot leave a turn parked on a vendor. The help says so.
        help="Needs recommendations on. Per-call deadline; 0 uses 1500 ms.",
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.waitMs",
        path=("classification", "waitMs"),
        section="classification",
        label="↳ wait budget (ms)",
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
        help="Needs recommendations on. How long a turn waits; 0 uses 50 ms.",
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.maxStateChars",
        path=("classification", "maxStateChars"),
        section="classification",
        label="↳ max state chars",
        kind=Kind.INT,
        default=6000,
        help="Needs recommendations on. State cap; 0 uses 6000 chars.",
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.maxCandidates",
        path=("classification", "maxCandidates"),
        section="classification",
        label="↳ max candidates",
        kind=Kind.INT,
        default=12,
        help="Needs recommendations on. Candidates per kind; 0 uses 12.",
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.maxRecommendations",
        path=("classification", "maxRecommendations"),
        section="classification",
        label="↳ max recommendations",
        kind=Kind.INT,
        default=3,
        help="Needs recommendations on. Per-message resources; 0 uses 3.",
        minimum=0,
        gated_by="classification.auto",
    ),
    Setting(
        key="classification.notice",
        path=("classification", "notice"),
        section="classification",
        label="↳ notice",
        kind=Kind.BOOL,
        default=True,
        # ON by default, unlike the master switch: the notice is the only place
        # the spend and the vendor are visible, and a user who turned the layer on
        # asked for that. It appears once per message and only when the model
        # actually recommended something.
        help="Needs recommendations on. Names vendor, resources, cost.",
        choices=_bool_choices("show the one-line notice", "stay silent"),
        gated_by="classification.auto",
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
        label="Where a fork opens",
        kind=Kind.ENUM,
        default="switch",
        help="Choose where the fork opens; unfinished work stays in the original.",
        choices=(
            Choice(
                "window",
                "new window",
                "open the fork elsewhere; this session keeps running",
            ),
            Choice("switch", "this terminal", "follow the fork here; return with /resume"),
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
        label="Where it opens under cmux",
        kind=Kind.ENUM,
        default="workspace",
        help="Under cmux, whether a fork gets its own workspace or a surface here.",
        choices=(
            Choice("workspace", "new workspace", "a sidebar row of its own"),
            Choice("surface", "new surface", "a tab in the current workspace"),
        ),
    ),
    # -- compaction ---------------------------------------------------------
    Setting(
        key="compaction.enabled",
        path=("compaction", "enabled"),
        section="compaction",
        label="Compaction",
        kind=Kind.BOOL,
        default=True,
        help="Summarise older history when the context fills.",
        choices=_bool_choices("compact automatically", "never compact"),
    ),
    Setting(
        key="compaction.strategy",
        path=("compaction", "strategy"),
        section="compaction",
        label="Strategy",
        kind=Kind.ENUM,
        default="auto",
        help="Which mechanism compacts. Auto picks per model.",
        choices=(
            Choice("auto", "auto", "snapcompact for vision models, else context-full"),
            Choice("context-full", "context-full", "summarise the whole context"),
            Choice("snapcompact", "snapcompact", "snapshot-based, keeps images out"),
            Choice("off", "off", "disable the pass"),
        ),
    ),
    Setting(
        key="compaction.threshold_percent",
        path=("compaction", "threshold_percent"),
        section="compaction",
        label="Threshold (% of window)",
        kind=Kind.FLOAT,
        default=0.80,
        help="Percentage trigger. 0.80 and 80 both mean 80%.",
        minimum=0.0,
        maximum=100.0,
    ),
    Setting(
        key="compaction.threshold_tokens",
        path=("compaction", "threshold_tokens"),
        section="compaction",
        label="Threshold (tokens)",
        kind=Kind.INT,
        default=600_000,
        help="Absolute trigger. The smaller of this and the percentage wins.",
        minimum=1,
    ),
    Setting(
        key="compaction.keep_recent_tokens",
        path=("compaction", "keep_recent_tokens"),
        section="compaction",
        label="Keep recent tokens",
        kind=Kind.INT,
        default=20_000,
        help="Recent history kept verbatim across a pass.",
        minimum=0,
    ),
    Setting(
        key="compaction.auto_continue",
        path=("compaction", "auto_continue"),
        section="compaction",
        label="Continue after compaction",
        kind=Kind.BOOL,
        default=True,
        help="Schedule a continuation prompt after a successful post-turn pass.",
        choices=_bool_choices("continue automatically", "stop after the pass"),
    ),
    Setting(
        key="compaction.mid_turn_enabled",
        path=("compaction", "mid_turn_enabled"),
        section="compaction",
        label="Mid-turn compaction",
        kind=Kind.BOOL,
        default=True,
        help="Allow a pass at safe tool-loop boundaries, not only between turns.",
        choices=_bool_choices("compact mid-turn", "only between turns"),
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
        label="Request size limit (bytes)",
        kind=Kind.INT,
        default=24_000_000,
        help=(
            "Hard ceiling on the request. Older screenshots are dropped from the"
            " context (never from the transcript) to stay under it. 0 disables."
        ),
        minimum=0,
    ),
    Setting(
        key="compaction.wire_bytes_trigger",
        path=("compaction", "wire_bytes_trigger"),
        section="compaction",
        label="Request size trigger (bytes)",
        kind=Kind.INT,
        default=16_000_000,
        help=(
            "Compact once the request passes this size, so a screenshot-heavy"
            " session summarises early instead of dropping frames. 0 disables."
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
        label="Web search",
        kind=Kind.BOOL,
        default=True,
        help="Offer the search tool and let it run; off refuses every call.",
        choices=_bool_choices("search available", "search disabled"),
    ),
    Setting(
        key="web_search.strategy",
        path=("web_search", "strategy"),
        section="web_search",
        label="Load balancing",
        kind=Kind.ENUM,
        default="round_robin",
        help="How the provider list is consumed.",
        choices=(
            Choice("round_robin", "round_robin", "rotate across providers"),
            Choice("ordered", "ordered", "top of the list first, fall through"),
        ),
    ),
    Setting(
        key="web_search.providers",
        path=("web_search", "providers"),
        section="web_search",
        label="Providers",
        kind=Kind.LIST,
        default=["duckduckgo", "tavily"],
        help="Comma-separated, in priority order.",
        members=(
            "duckduckgo",
            "tavily",
            "deepseek",
            "perplexity",
            "brave",
            "exa",
            "serpapi",
            "searxng",
        ),
    ),
    Setting(
        key="web_search.timeout_seconds",
        path=("web_search", "timeout_seconds"),
        section="web_search",
        label="Timeout (s)",
        kind=Kind.FLOAT,
        default=20.0,
        help="Per-provider request timeout. Clamped to 1-120 when read.",
        minimum=1.0,
        maximum=120.0,
    ),
    Setting(
        key="web_search.searxng_endpoint",
        path=("web_search", "searxng_endpoint"),
        section="web_search",
        label="SearXNG endpoint",
        kind=Kind.TEXT,
        default="",
        help="Base URL of a self-hosted SearXNG instance.",
    ),
    Setting(
        key="web_search.deepseek_evidence",
        path=("web_search", "deepseek_evidence"),
        section="web_search",
        label="DeepSeek page evidence",
        kind=Kind.BOOL,
        default=False,
        # Off by default: it is a SECOND model turn (measured 4-11s on top of the
        # search) that buys a verbatim quote and a relevance score per source,
        # for the "which page do I fetch next" decision. Only the deepseek
        # provider consumes it; every other provider already returns snippets.
        help="Adds a per-page quote and relevance score after a DeepSeek search.",
    ),
    Setting(
        key="web_search.read_enabled",
        path=("web_search", "read_enabled"),
        section="web_search",
        label="Read from search",
        kind=Kind.BOOL,
        default=True,
        # On by default because it is inert until used: the tool refuses (telling
        # the model to fetch instead) whenever no readable page context exists,
        # so a session that never uses it pays nothing but a tool schema.
        help="Offer web_read: answer from pages a search already retrieved, no refetch.",
    ),
    # -- web fetch ----------------------------------------------------------
    Setting(
        key="web_fetch.enabled",
        path=("web_fetch", "enabled"),
        section="web_tools",
        label="Web fetch",
        kind=Kind.BOOL,
        default=True,
        # 60 cells — see the note on `hosting`. No BACKTICKS: the footer is a
        # plain `Text`, so they render as literal characters, and this was the
        # only one of 57 help strings carrying any (design round 1, D5).
        help="Offer the fetch tool and read <url>; off refuses every call.",
        choices=_bool_choices("fetch available", "fetch disabled"),
    ),
    Setting(
        key="web_fetch.timeout_seconds",
        path=("web_fetch", "timeout_seconds"),
        section="web_fetch",
        label="Timeout (s)",
        kind=Kind.FLOAT,
        default=20.0,
        help="Per-request timeout.",
        minimum=1.0,
        maximum=300.0,
    ),
    Setting(
        key="web_fetch.max_bytes",
        path=("web_fetch", "max_bytes"),
        section="web_fetch",
        label="Download ceiling (bytes)",
        kind=Kind.INT,
        default=5 * 1024 * 1024,
        help="Enforced during streaming, so a huge page is cut off rather than buffered.",
        minimum=1024,
    ),
    Setting(
        key="web_fetch.max_redirects",
        path=("web_fetch", "max_redirects"),
        section="web_fetch",
        label="Max redirects",
        kind=Kind.INT,
        default=5,
        help="Redirect hops followed before giving up.",
        minimum=0,
        maximum=50,
    ),
    Setting(
        key="web_fetch.cache_ttl_seconds",
        path=("web_fetch", "cache_ttl_seconds"),
        section="web_fetch",
        label="Cache TTL (s)",
        kind=Kind.INT,
        default=900,
        help="0 disables the URL cache entirely.",
        minimum=0,
    ),
    Setting(
        key="web_fetch.allow_private",
        path=("web_fetch", "allow_private"),
        section="web_fetch",
        label="Allow private addresses",
        kind=Kind.BOOL,
        default=False,
        help="SSRF guard. On permits loopback, private and link-local targets.",
        choices=_bool_choices("allow private targets", "block private targets"),
    ),
    Setting(
        key="web_fetch.render_backend",
        path=("web_fetch", "render_backend"),
        section="web_fetch",
        label="HTML renderer",
        kind=Kind.ENUM,
        default="auto",
        help="Auto uses markdownify when the [fetch] extra is installed.",
        choices=(
            Choice("auto", "auto", "markdownify if available, else stdlib"),
            Choice("stdlib", "stdlib", "always the bundled renderer"),
        ),
    ),
    Setting(
        key="web_fetch.enrich",
        path=("web_fetch", "enrich"),
        section="web_fetch",
        label="Enrich before scraping",
        kind=Kind.BOOL,
        default=True,
        help="Try .md, llms.txt and content negotiation before scraping HTML.",
        choices=_bool_choices("try cleaner sources first", "scrape HTML directly"),
    ),
    Setting(
        key="web_fetch.max_attempts",
        path=("web_fetch", "max_attempts"),
        section="web_fetch",
        label="Attempts per hop",
        kind=Kind.INT,
        default=3,
        help="Retries share the call's timeout, so more attempts never take longer.",
        minimum=1,
        maximum=5,
    ),
    Setting(
        key="web_fetch.blocked_retry",
        path=("web_fetch", "blocked_retry"),
        section="web_fetch",
        label="Browser-profile retry",
        kind=Kind.BOOL,
        default=True,
        help="After a refusal, retry once with browser-shaped headers.",
        choices=_bool_choices("retry refusals once", "stay self-identifying"),
    ),
    # -- tools --------------------------------------------------------------
    # ``path`` mirrors ``tools.builtin.BASH_SHELL_PATH``; the two are pinned
    # together by ``test_bash_shell_row_shares_the_consumer_path`` rather than
    # imported, because this module must stay cheap for the CLI and
    # ``tools.builtin`` is not (see the module docstring on Textual).
    Setting(
        key="bash.shell",
        path=("bash", "shell"),
        section="tools",
        label="Bash interpreter",
        kind=Kind.TEXT,
        default="",
        help="Interpreter for the bash tool. Empty uses bash on PATH, else /bin/sh.",
        empty_unsets=True,
    ),
    # -- shell_environment ----------------------------------------------
    # ``path`` mirrors ``tools.shell_env.MODE_PATH`` and friends, pinned the same
    # way as the row above. The three rows are one policy and are read together
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
        label="Agent shell environment",
        kind=Kind.ENUM,
        # Pinned to ``shell_env.MODE_DEFAULT`` by
        # ``test_shell_environment_rows_share_the_reader_paths`` rather than
        # imported: the default must stay the permissive one, and the pin is
        # what makes flipping it a decision rather than an edit.
        default="inherit",
        help=(
            "'inherit' lets a command the agent runs see your environment "
            "(gh, aws, npm keep their tokens). 'allowlist' passes only "
            "PATH/HOME/SHELL/TERM/USER/LOGNAME plus the two lists beside this, "
            "so the provider key this session launched with is not handed to a "
            "command the model writes. Read once per session: changing it takes "
            "effect at the next launch."
        ),
        choices=(
            Choice("inherit", "inherit", "your environment, as today (default)"),
            Choice("allowlist", "allowlist", "the safe set plus the lists below, nothing else"),
        ),
    ),
    Setting(
        key="shell_environment.inherit",
        path=("shell_environment", "inherit"),
        section="shell_environment",
        label="…kept in allowlist mode",
        kind=Kind.LIST,
        default=[],
        help=(
            "Empty = the safe set alone. Names the allowlist mode keeps ON TOP "
            "of the safe set — e.g. LOCAL_OPERATOR_CONFIG_DIR for a shell that "
            "must still resolve $(lop secret get NAME), or LANG. Inert in "
            "inherit mode."
        ),
        # OPEN namespace: these are the operator's own variable names, so there
        # is no vocabulary for this repo to bound.
        placeholder="LOCAL_OPERATOR_CONFIG_DIR, LANG, …",
        empty_unsets=True,
    ),
    Setting(
        key="shell_environment.exclude",
        path=("shell_environment", "exclude"),
        section="shell_environment",
        label="…removed in both modes",
        kind=Kind.LIST,
        default=[],
        help=(
            "Empty = nothing removed. Names never handed to a command the agent "
            "runs, in either mode — including the session credential store's "
            "own injections, so this is how a variable is denied outright. "
            "Covers the bash and eval children only; MCP servers and the "
            "harness's own processes are not governed by it."
        ),
        placeholder="OPENROUTER_API_KEY, …",
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
                f"{name} endpoint",
                Kind.TEXT,
                endpoint,
                "HTTP(S) API root. Changing servers requires a new token through /login; "
                "existing tokens are never forwarded to another endpoint.",
                validate_value=validate_endpoint_setting,
            ),
            Setting(
                f"providers.{provider}.models",
                ("providers", provider, "models"),
                "local_providers",
                f"{name} model overrides",
                Kind.TEXT,
                DEFAULT_MODEL_OVERRIDES,
                'JSON: {"model-id":{"context_window":8192}}; server limits still apply.',
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
        label="Conversation length",
        kind=Kind.READONLY,
        default=100,
        help="Deprecated. Superseded by the compaction engine.",
    ),
    Setting(
        key="detail_length",
        path=("detail_length",),
        section="retired",
        label="Detail length",
        kind=Kind.READONLY,
        default=15,
        help="Deprecated. Superseded by the compaction engine.",
    ),
    Setting(
        key="max_learnings_history",
        path=("max_learnings_history",),
        section="retired",
        label="Max learnings history",
        kind=Kind.READONLY,
        default=50,
        help="Deprecated. Superseded by the compaction engine.",
    ),
    Setting(
        key="desktop.launch_command",
        path=("desktop", "launch_command"),
        section="desktop",
        label="launch command",
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
            "Empty = discover the app (local-operator-ui, then the macOS app). "
            "{session} is the session id."
        ),
        placeholder="local-operator-ui --open-session {session}",
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
        # same whichever writer arrives.
        return _keymap.normalize_key(text)
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
    if (
        setting.kind is Kind.HOTKEY
        and values is not None
        and isinstance(value, str)
        and value.strip()
    ):
        group_problem = _keymap.group_conflict(setting.key, value, values)
        if group_problem is not None:
            return group_problem
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
        # widget calls the same predicate, so the two cannot disagree.
        return _keymap.validate_key(value)
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
        # One normalization at the single point every write funnels through.
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


# ---------------------------------------------------------------------------
# The failover cascade
# ---------------------------------------------------------------------------
#
# `retry.fallbackChains` is `{chain key: [hop, ...]}` where a hop is either a
# "provider/model" string or a `{provider, model, effort}` mapping. The page
# edits it as two levels (chains, then hops within one chain), so the helpers
# below are the only place that shape is known outside `providers/failover.py`.


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
