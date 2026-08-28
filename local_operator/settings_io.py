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
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from local_operator.config import ConfigManager


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

    #: Takes effect immediately in this running session.
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
    #: Inclusive bounds for INT/FLOAT. ``None`` on either side means unbounded.
    minimum: float | None = None
    maximum: float | None = None
    #: Members a LIST setting may contain, in the order they are offered.
    members: tuple[str, ...] = ()
    #: An empty text field CLEARS the key rather than storing "". Off for
    #: settings whose empty string is a real value (``searxng_endpoint``
    #: unset IS ""), on where empty means "no opinion" (``hosting``).
    empty_unsets: bool = False

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
    Section(
        "failover",
        "Failover and retry",
        Scope.NEW_SESSIONS,
        "What happens when a provider call fails or a quota runs out.",
    ),
    Section(
        "appearance",
        "Appearance",
        Scope.LIVE,
        "Theme and the terminal features the TUI is allowed to use.",
    ),
    Section(
        "session",
        "Session",
        Scope.NEW_SESSIONS,
        "Approvals, autosave, and how many background jobs may run.",
    ),
    Section(
        "compaction",
        "Compaction",
        Scope.NEW_SESSIONS,
        "When the conversation is summarised to reclaim context.",
    ),
    Section(
        "web_search",
        "Web search",
        Scope.NEW_SESSIONS,
        "Providers and load balancing for the search tool.",
    ),
    Section(
        "web_fetch",
        "Web fetch",
        Scope.NEW_SESSIONS,
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


SETTINGS: tuple[Setting, ...] = (
    # -- model --------------------------------------------------------------
    Setting(
        key="hosting",
        path=("hosting",),
        section="model",
        label="Default provider",
        kind=Kind.TEXT,
        default="",
        help="Provider new launches boot on. Written by /model default.",
        empty_unsets=True,
    ),
    Setting(
        key="model_name",
        path=("model_name",),
        section="model",
        label="Default model",
        kind=Kind.TEXT,
        default="",
        help="Model id new launches boot on. Written by /model default.",
        empty_unsets=True,
    ),
    Setting(
        key="providers.openai.api",
        path=("providers", "openai", "api"),
        section="model",
        label="OpenAI API surface",
        kind=Kind.ENUM,
        default="responses",
        help="Direct OpenAI GPT-5 calls use the Responses API unless opted out.",
        choices=(
            Choice("responses", "responses", "the public Responses API (default)"),
            Choice("chat_completions", "chat_completions", "explicit compatibility opt-out"),
        ),
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
        key="retry.usageReservePercent",
        path=("retry", "usageReservePercent"),
        section="failover",
        label="Usage reserve (%)",
        kind=Kind.FLOAT,
        default=10.0,
        help="Quota headroom kept in reserve before usage-aware fallback moves on.",
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
        kind=Kind.TEXT,
        default="",
        help="Colour ramp. /theme switches it live with an arrow-key preview.",
        empty_unsets=True,
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
    Setting(
        key="subagents.max_running",
        path=("subagents", "max_running"),
        section="session",
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
        section="session",
        label="Subagent model: lo",
        kind=Kind.TEXT,
        default="",
        help="provider/model for the lo effort tier. Empty keeps the parent's model.",
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.med",
        path=("subagents", "models", "med"),
        section="session",
        label="Subagent model: med",
        kind=Kind.TEXT,
        default="",
        help="provider/model for the med effort tier. Empty keeps the parent's model.",
        empty_unsets=True,
    ),
    Setting(
        key="subagents.models.hi",
        path=("subagents", "models", "hi"),
        section="session",
        label="Subagent model: hi",
        kind=Kind.TEXT,
        default="",
        help="provider/model for the hi effort tier. Empty keeps the parent's model.",
        empty_unsets=True,
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
    # -- web search ---------------------------------------------------------
    Setting(
        key="web_search.enabled",
        path=("web_search", "enabled"),
        section="web_search",
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
        section="web_fetch",
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
        key="session_retention_max_sessions",
        path=("session_retention_max_sessions",),
        section="retired",
        label="Session retention: max sessions",
        kind=Kind.READONLY,
        default=0,
        help="Retired. Transcripts are never deleted automatically at any value.",
    ),
    Setting(
        key="session_retention_max_bytes",
        path=("session_retention_max_bytes",),
        section="retired",
        label="Session retention: max bytes",
        kind=Kind.READONLY,
        default=0,
        help="Retired. Transcripts are never deleted automatically at any value.",
    ),
    Setting(
        key="session_retention_max_age_days",
        path=("session_retention_max_age_days",),
        section="retired",
        label="Session retention: max age (days)",
        kind=Kind.READONLY,
        default=0,
        help="Retired. Transcripts are never deleted automatically at any value.",
    ),
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
        if value not in [choice.value for choice in setting.choices]:
            return f"expected one of: {', '.join(str(c.label) for c in setting.choices)}"
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


def _store(manager: "ConfigManager", path: Sequence[str], value: Any) -> None:
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


# ---------------------------------------------------------------------------
# The failover cascade
# ---------------------------------------------------------------------------
#
# `retry.fallbackChains` is `{chain key: [hop, ...]}` where a hop is either a
# "provider/model" string or a `{provider, model, effort}` mapping. The page
# edits it as two levels (chains, then hops within one chain), so the helpers
# below are the only place that shape is known outside `providers/failover.py`.


def read_chains(manager: "ConfigManager") -> dict[str, list[str]]:
    """The cascade as ``{key: ["provider/model", ...]}``.

    Structured hops are flattened to their selector for DISPLAY only; the page
    never writes a hop it did not read, so an ``effort`` a user hand-wrote
    survives untouched unless they edit that exact hop. Malformed entries are
    dropped rather than rendered, mirroring ``_normalize_chains``: a chain the
    failover layer will ignore must not be shown as if it were live.
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


def write_chains(manager: "ConfigManager", chains: Mapping[str, Sequence[str]]) -> None:
    """Replace the cascade with ``chains``, dropping empty ones.

    An empty chain is dropped rather than stored because ``_normalize_chains``
    already drops it on read: keeping it would put a row in the file that the
    page shows and the failover layer does not have, which is the config and
    the behaviour disagreeing again.
    """
    stored = {
        key: [hop.split(" (")[0] for hop in hops]
        for key, hops in chains.items()
        if key.strip() and hops
    }
    _store(manager, ("retry", "fallbackChains"), stored)
    _invalidate_caches()


def validate_hop(text: str) -> str | None:
    """``None`` when ``text`` is a usable ``provider/model`` selector."""
    candidate = text.strip()
    if not candidate:
        return "expected provider/model"
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
