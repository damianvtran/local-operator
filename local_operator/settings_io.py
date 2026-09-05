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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

import yaml

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from local_operator.config import ConfigManager


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
    #: :mod:`local_operator.config_watch`).
    LIVE = "live"
    #: Read when a session is built — a ``/new`` or ``/reload`` picks it up.
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
    members: tuple[str, ...] = ()
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
    Section(
        "model",
        "Model",
        Scope.NEW_LAUNCH,
        "The provider and model new launches boot on.",
    ),
    # Split out of ``model`` (review round 1, M3). The design left this key in
    # ``model`` and proposed documenting the discrepancy, which was defensible
    # while nothing read the scope aloud — but the config-change notice now
    # says "takes effect on /new" for every non-LIVE key, and for this one that
    # is FALSE: ``Session._apply_config_change`` rebinds the stream fn on it and
    # ``configure._openai_api_mode`` reads the rebound mapping when it builds
    # the next client. Scope is uniform within a section by construction, so
    # saying something true here means a section of its own, exactly as ``fork``
    # and ``web_tools`` are. ``hosting``/``model_name`` genuinely stay
    # NEW_LAUNCH: they are the session's identity, not a knob it re-reads.
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
    # Deliberately NOT live, and split from ``subagents`` for that reason (scope
    # is uniform within a section by construction). A tool-approval mode that
    # flipped under a running turn would be a security-relevant surprise; the
    # per-session ``/approvals`` toggle is the live control, and it WRITES this
    # default rather than following it.
    Section(
        "session",
        "Session",
        Scope.NEW_SESSIONS,
        "Approvals and autosave for sessions started from now on.",
    ),
    # LIVE: ``max_running`` is pushed into the running ``AsyncJobManager`` by
    # ``Session._apply_config_change`` (raising it lets the next launch through;
    # lowering it lets running jobs finish — nothing is evicted), and the
    # ``models.*`` tiers are read at every spawn.
    Section(
        "subagents",
        "Subagents",
        Scope.LIVE,
        "Concurrency cap and the model each effort tier runs on.",
    ),
    # Its own section rather than a row under "Session", and the reason is the
    # SCOPE: scope is uniform within a section by construction, "Session" is
    # NEW_SESSIONS, and these keys take effect on the very next /resume in this
    # same terminal. Filing a live key under a section labelled "new sessions"
    # is exactly the painted lie AGENTS.md warns about — split the section.
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
    # D3). Whether each tool is offered at all is decided when the tool
    # inventory is built, so these two flags cannot be LIVE — but reading order
    # is the hierarchy the user sees, and putting the master switches after
    # four tuning knobs left someone scanning for "is web search on?" finding
    # the answer next to the retired-keys graveyard.
    Section(
        "web_tools",
        "Web tools",
        Scope.NEW_SESSIONS,
        "Whether the search and fetch tools are offered to the model.",
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
    Section(
        "retired",
        "Retired",
        Scope.NEW_LAUNCH,
        "Keys that are read but no longer do anything.",
    ),
)


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


SETTINGS: tuple[Setting, ...] = (
    # -- model --------------------------------------------------------------
    Setting(
        key="hosting",
        path=("hosting",),
        section="model",
        label="Default provider",
        kind=Kind.TEXT,
        default="",
        help="Provider new launches boot on. Set here or by /model default.",
        empty_unsets=True,
    ),
    Setting(
        key="model_name",
        path=("model_name",),
        section="model",
        label="Default model",
        kind=Kind.TEXT,
        default="",
        help="Model id new launches boot on. Set here or by /model default.",
        empty_unsets=True,
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
    # -- session ------------------------------------------------------------
    Setting(
        key="tool_approval_mode",
        path=("tool_approval_mode",),
        section="session",
        label="Tool approval mode",
        kind=Kind.ENUM,
        default="ask",
        help="How a new interactive session treats write and exec tools.",
        choices=(
            Choice("ask", "ask", "prompt before write/exec tools"),
            Choice("auto", "auto", "run them without asking"),
        ),
    ),
    Setting(
        key="auto_save_conversation",
        path=("auto_save_conversation",),
        section="session",
        label="Auto-save conversation",
        kind=Kind.BOOL,
        default=False,
        help="Write the conversation to disk as it goes.",
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
        # NEW_SESSIONS, this key is LIVE) is sound and unaffected: the section
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
        help="Needs cleanup on. Keep the N most recently active; 0 = no cap.",
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
        key="subagents.models.lo",
        path=("subagents", "models", "lo"),
        section="subagents",
        label="Subagent model: lo",
        kind=Kind.TEXT,
        default="",
        help="provider/model for the lo effort tier. Empty keeps the parent's model.",
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.med",
        path=("subagents", "models", "med"),
        section="subagents",
        label="Subagent model: med",
        kind=Kind.TEXT,
        default="",
        help="provider/model for the med effort tier. Empty keeps the parent's model.",
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.hi",
        path=("subagents", "models", "hi"),
        section="subagents",
        label="Subagent model: hi",
        kind=Kind.TEXT,
        default="",
        help="provider/model for the hi effort tier. Empty keeps the parent's model.",
        empty_unsets=True,
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
    # The two ``enabled`` flags sit in ``web_tools`` (NEW_SESSIONS), apart from
    # the knobs that share their YAML block: they gate whether the tool exists
    # in the inventory, which is decided once at build.
    Setting(
        key="web_search.enabled",
        path=("web_search", "enabled"),
        section="web_tools",
        label="Web search",
        kind=Kind.BOOL,
        default=True,
        help="Expose the search tool to the agent.",
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
    # -- web fetch ----------------------------------------------------------
    Setting(
        key="web_fetch.enabled",
        path=("web_fetch", "enabled"),
        section="web_tools",
        label="Web fetch",
        kind=Kind.BOOL,
        default=True,
        help="Expose the fetch tool to the agent.",
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


def read_setting(manager: "ConfigManager", setting: Setting) -> Any:
    """The stored value for ``setting``, or its default when unset."""
    raw = _walk(manager.get_config().values, setting.path)
    if raw is _MISSING:
        return setting.default
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
    return text


def validate(setting: Setting, value: Any) -> str | None:
    """``None`` when ``value`` may be stored, else the reason it may not.

    Bounds are enforced HERE rather than left to the consumer's own clamping,
    because the consumers clamp SILENTLY (``coerce_search_settings`` pins the
    timeout to 1-120 on read). A page that accepted 500 and stored it would
    show 500 forever while the tool used 120 — the config and the behaviour
    disagreeing, with nothing on screen admitting it.
    """
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
        if value not in [choice.value for choice in choices]:
            return f"expected one of: {', '.join(str(c.label) for c in choices)}"
        return None
    if setting.kind is Kind.LIST:
        if not isinstance(value, list):
            return "expected a comma-separated list"
        unknown = [item for item in value if item not in setting.members]
        if unknown:
            return f"unknown: {', '.join(str(item) for item in unknown)}"
        if not value:
            return "at least one provider is required"
        return None
    if setting.kind is Kind.BOOL:
        return None if isinstance(value, bool) else "expected on or off"
    if setting.kind in (Kind.INT, Kind.FLOAT):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return "expected a number"
        if setting.minimum is not None and value < setting.minimum:
            return f"must be at least {_number(setting.minimum)}"
        if setting.maximum is not None and value > setting.maximum:
            return f"must be at most {_number(setting.maximum)}"
        return None
    if setting.kind is Kind.TEXT:
        return None if isinstance(value, str) else "expected text"
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
    problem = validate(setting, value)
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
