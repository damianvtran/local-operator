"""Configuration management for Local Operator.

This module handles reading and writing configuration settings from a YAML file.
It provides default configurations and methods to update them.
"""

import argparse
import logging
import os
import sys
import tempfile
from copy import deepcopy
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Dict

import yaml

from local_operator.web_defaults import (
    DEFAULT_WEB_FETCH_CONFIG,
    DEFAULT_WEB_SEARCH_CONFIG,
)

logger = logging.getLogger(__name__)


def _version_tuple(raw: str) -> tuple[int, ...]:
    """Parse a dotted version into ints for ordering.

    Only the LEADING digits of each segment count, and the rest of the segment
    is discarded: collecting every digit turned "1.2.3rc1" into (1, 2, 31),
    making a pre-release compare as NEWER than its own release and firing the
    "your config is newer" warning on the wrong versions. A pre-release sorting
    equal to its release is the right approximation here — this decides one
    advisory message, not resolution.

    Empty segments are DROPPED, not zeroed ("1..3" -> (1, 3)), and a segment
    with no leading digit ENDS the parse rather than contributing a 0
    ("v1.2.3" -> (0,)). Nothing raises: a version warning must never be the
    thing that stops the CLI from starting.
    """
    parts: list[int] = []
    for chunk in str(raw).split("."):
        chunk = chunk.strip()
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        if digits:
            parts.append(int(digits))
        if digits != chunk:
            # This segment was not purely numeric, so the version proper ends
            # here: "3rc1", "3-beta" and "dev4" all mark a pre-release suffix.
            # Stopping makes every pre-release form collapse to exactly its
            # release version instead of sorting above it, which is what the
            # one advisory message this feeds actually wants.
            break
    return tuple(parts) or (0,)


#: ``(the resolver it was read through, its answer)``. See :func:`_package_version`.
_PACKAGE_VERSION: "tuple[Any, str] | None" = None


def _package_version() -> str:
    """``version("local-operator")``, read once per process instead of per manager.

    WHY. ``importlib.metadata.version`` locates the distribution and parses its
    ``METADATA`` file through ``email.parser`` on EVERY call, and each manager
    asked for it twice. A runtime child builds five managers before it can
    publish (``spawn_owned_session``, ``Session._configured_max_running``, two
    ``read_model_choice`` calls, ``read_effort_tier_selectors``), so one session
    construction paid ten lookups: 76-82 ms of CPU of a ~440 ms construction
    (CPU-clock cProfile), on a host at load 100+ where one CPU millisecond costs
    10-17 ms of wall time.

    NOT STALE IN ANY WAY THAT MATTERS: the answer is the version of the code
    this process imported, which cannot change without a new process. An
    install that moves under a live process is a different question with its
    own detector (``session._process_boot_build``); this value feeds only the
    config file's advisory version stamp and the "your config is newer" warning.

    KEYED ON THE RESOLVER'S IDENTITY so a test that patches
    ``local_operator.config.version`` is answered by its patch rather than by a
    value an earlier test cached — the cache then behaves like the direct call.
    """
    global _PACKAGE_VERSION
    cached = _PACKAGE_VERSION
    if cached is not None and cached[0] is version:
        return cached[1]
    answer = version("local-operator")
    _PACKAGE_VERSION = (version, answer)
    return answer


#: One parsed ``config.yml`` per path, keyed by what ``fstat`` said about the
#: bytes it was parsed from. See :func:`_parse_config_stream`.
_PARSED: dict[str, tuple[tuple[int, ...], Any]] = {}

#: The C scanner when libyaml is compiled in (it is in the wheels we ship), else
#: the pure-Python one. Both construct from the same SAFE tag set and raise the
#: same ``yaml.YAMLError`` subclasses, so the caller's error path is unchanged;
#: equal output was checked on the operator's real config.yml.
_SAFE_LOADER: Any = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _stat_key(st: os.stat_result) -> tuple[int, ...]:
    return (st.st_ino, st.st_dev, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def config_file_key(path: Path) -> "tuple[int, ...] | None":
    """The identity :func:`_parse_config_stream` caches a parse under, or ``None``.

    Public because a pre-imported standby runtime records it when it warms and
    compares it before adopting a session (``session.runtime.standby``): a standby
    whose config moved since it warmed is discarded rather than served.
    """
    try:
        return _stat_key(os.stat(path))
    except OSError:
        return None


def _parse_config_stream(path: Path, stream: Any) -> Any:
    """``config.yml`` parsed — at most once per version of the file, per process.

    WHY. Every ``ConfigManager`` re-read and re-parsed the file, and a session's
    construction path builds five of them (see :func:`_package_version`). The
    pure-Python ``SafeLoader`` costs ~2 ms of CPU per parse on the operator's
    1.6 KB file and ~8 ms on a seeded one, and YAML was the largest single item
    in a runtime child's pre-publication CPU after imports: 185 of ~440 ms,
    measured. At this host's load (100 ms of CPU = 1.0-1.7 s of wall) that is
    seconds of "starting…" on every cold engage.

    THE INVALIDATION STORY. The key is ``(inode, device, size, mtime_ns,
    ctime_ns)`` from ``fstat`` on the OPEN descriptor, taken on every call, and
    a hit needs all five to match:

    * every writer in this codebase replaces the file atomically
      (``_write_config``: temp file + ``os.replace``), which gives the path a NEW
      inode, so a replaced file can never match the old key;
    * an in-place rewrite (an editor, ``>>``) moves ``mtime_ns`` and
      ``ctime_ns``; ``ctime`` cannot be set back by any process, even with
      ``os.utime``;
    * ``fstat`` on the descriptor we read from, not ``stat`` on the path, so the
      key always describes the file the bytes come from, even if the path is
      swapped between the open and the read.

    So the memo only ever answers for bytes the open file still holds: it
    removes the PARSE, never the freshness check. A second ``fstat`` after the
    read gates the store, so a write that lands DURING the read is not cached
    under the key of the bytes it replaced. A parse error propagates uncached:
    the caller moves a bad file aside and the next read must see what replaced it.

    CALLERS GET A DEEP COPY, because a manager merges defaults into the dict it
    loads, in place, and managers must never share state — see
    :func:`_fresh_default_config` for the incident behind that rule.
    """
    name = str(path)
    try:
        key: "tuple[int, ...] | None" = _stat_key(os.fstat(stream.fileno()))
    except (OSError, AttributeError, ValueError):
        key = None
    if key is not None:
        hit = _PARSED.get(name)
        if hit is not None and hit[0] == key:
            return deepcopy(hit[1])
    loaded = yaml.load(stream, Loader=_SAFE_LOADER)  # noqa: S506 — a SAFE loader
    if key is not None:
        try:
            unchanged = _stat_key(os.fstat(stream.fileno())) == key
        except (OSError, ValueError):
            unchanged = False
        if unchanged:
            _PARSED[name] = (key, deepcopy(loaded))
    return loaded


class Config:
    """Configuration settings for Local Operator.

    Attributes:
        version (str): Configuration schema version for compatibility
        metadata (Dict): Metadata about the configuration
        values (Dict): Configuration settings
            conversation_length (int): Number of conversation messages to retain
            detail_length (int): Maximum length of detailed conversation history
            hosting (str): AI model hosting provider
            model_name (str): Name of the AI model to use
            rag_enabled (bool): Whether RAG is enabled
            auto_save_conversation (bool): Whether to automatically save the conversation
            tool_approval_mode (str): Interactive tool-approval default, ask or auto
            shell_environment (Dict): What a child process the MODEL asks for may
                see of this process's own environment. mode is inherit (the
                default: a copy, today's behaviour) or allowlist (strict: only
                the SDK's safe set plus inherit); inherit extends that safe set,
                exclude removes names from both modes
    """

    version: str
    metadata: Dict[str, Any]
    values: Dict[str, Any]

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize the config with default or existing settings.

        Creates a new Config instance that manages configuration settings.
        If a config file exists at the specified path, loads settings from it.
        """
        # Set metadata first. The schema version is DELIBERATELY absent from
        # this constructor when the caller did not supply one — see
        # :meth:`__getattr__`, which resolves it on first read instead.
        if "version" in config_dict:
            self.version = config_dict["version"]
        self.metadata = config_dict.get(
            "metadata",
            {
                "created_at": "",
                "last_modified": "",
                "description": "Local Operator configuration file",
            },
        )

        # Set metadata values with defaults if not provided
        if not self.metadata["created_at"]:
            self.metadata["created_at"] = datetime.now().isoformat()
        if not self.metadata["last_modified"]:
            self.metadata["last_modified"] = datetime.now().isoformat()

        # Set config values
        self.values = {}
        for key, value in config_dict.get("values", {}).items():
            self.values[key] = value

    def __getattr__(self, name: str) -> Any:
        """Resolve ``version`` on FIRST READ, not at construction.

        WHY THIS IS AN ``__getattr__`` AND NOT A PROPERTY. ``import
        local_operator.config`` builds :data:`DEFAULT_CONFIG`, whose dict literal
        used to carry ``version=version("local-operator")`` evaluated right
        there. That single call is what made the module cost 117.6 ms of CPU for
        every ``lop`` verb — and, worse, what made it walk every ``sys.path``
        entry at IMPORT time, so a ``python -m``/pytest start from a directory
        with 30,000 entries (this machine's own attachments directory has 30,372)
        paid +95 ms and one at 137,050 entries paid +721 ms for a string only
        ``--version`` and a config WRITE ever print.

        A ``version`` property was the obvious shape and is the wrong one:
        ``_write_config`` persists ``vars(self.config)`` and
        :func:`_fresh_default_config` copies ``vars(DEFAULT_CONFIG)``, so the
        backing field would have to be spelled ``_version`` and every one of
        those call sites would have to learn about it — with a stray
        ``_version: ''`` key written into the operator's ``config.yml`` as the
        failure mode if one was missed. Leaving the attribute simply UNSET until
        it is asked for keeps ``vars()`` byte-identical to the old shape (the
        key appears once read, exactly as before) and keeps ``self.version = x``
        an ordinary assignment.

        Falls through to ``AttributeError`` for every other name, which is what
        ``copy.deepcopy`` and ``pickle`` probe for: this must stay a lookup hook
        for one attribute, not a catch-all that answers for the class.
        """
        if name == "version":
            value = _package_version()
            self.version = value
            return value
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    def get_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration value.

        Args:
            key (str): The configuration key to retrieve

        Returns:
            Any: The configuration value for the key, or default if not found
        """
        return self.values.get(key, default)

    def set_value(self, key: str, value: Any) -> None:
        """Set a specific configuration value.

        Args:
            key (str): The configuration key to set
            value (Any): The value to set for the key
        """
        self.values[key] = value


# Default configuration settings for Local Operator
#
# No ``"version"`` key: ``Config`` resolves the schema stamp from the installed
# distribution on FIRST READ (``Config.__getattr__`` / :func:`_package_version`).
# Spelling it out here would evaluate it while this module is imported, which is
# the whole cost this avoids — and the value it produces is the same one either
# way, so a config written from these defaults is stamped identically.
DEFAULT_CONFIG = Config(
    {
        "metadata": {
            "created_at": "",
            "last_modified": "",
            "description": "Local Operator configuration file",
        },
        "values": {
            "conversation_length": 100,
            "detail_length": 15,
            "max_learnings_history": 50,
            "hosting": "",
            "model_name": "",
            # The BIRTH-default reasoning effort for new conversations, alongside
            # the model pair above. ``""`` means "no opinion" — the model's own
            # documented default stands. Read at session build
            # (``session_factory._prepare``, ``bootstrap.resolve_model_configuration``)
            # and clamped to the chosen model's own ladder there: a rung the
            # model cannot express lands on its nearest rung rather than reaching
            # the wire, so a stored ``xhigh`` beside a model that stops at
            # ``high`` is survivable and re-applies ``xhigh`` on a later switch
            # back to a wider ladder. A BIRTH default only — a resumed
            # conversation's own stored selection outranks it.
            "model_effort": "",
            "auto_save_conversation": False,
            # The tool-approval mode a NEW interactive session opens in, written
            # by ``/approvals default <mode>`` and read by the TUI at mount.
            # ``ask`` (prompt before write/exec tools) or ``auto`` (run them
            # without asking). A STRING and not a bool because the command's
            # vocabulary is a mode: a bool would have to be translated in both
            # directions, and the translation is where "off" ends up meaning
            # "prompting is off" in one place and "auto is off" in another.
            #
            # Read at mount by the TUI, at boot by phone-started sessions
            # (``spawn_owned_session``), and — since this key seeds it — at boot
            # by a ``lop exec --control`` run (``exec_control.start_exec_control``),
            # whose value then keeps following the file. The plain headless path
            # keeps ``--yolo`` as its one control: without ``--control`` there
            # are no runtime gates to seed, so a saved file still cannot disarm
            # the gate of a ``lop exec`` running in CI, where nobody is watching
            # the tools it approves.
            "tool_approval_mode": "ask",
            # Direct OpenAI GPT-5 calls use the public Responses API by default.
            # Set `providers.openai.api` to `chat_completions` for an explicit
            # compatibility opt-out; other OpenAI-shaped providers never read it.
            #
            # `providers.anthropic.cache_ttl_1h_min_context_tokens`: once a
            # session's context reaches this many tokens, Anthropic requests
            # carry the 1-hour prompt-cache TTL instead of the default 5 minutes.
            # A 1h write costs 2× base (vs 1.25× for 5m), but a large context
            # that idles past 5 minutes — waiting on subagents, a wake, or the
            # user — otherwise rewrites the WHOLE prefix on its next call.
            # Measured over 24h on this harness's own traffic: 276 TTL-expiry
            # rewrites of >150k contexts cost 89.5M write tokens (~112M
            # base-equivalent), while the incremental writes on those contexts
            # were only 14.7M (~11M base-equivalent extra at 2×). 150k is the
            # size above which the rewrite dominates; 0 disables the feature.
            # `providers.openrouter.*`: the chat-completions `provider` routing
            # object (see `settings_io.py` for the per-key semantics). Every
            # default below is "no opinion" — the resolver emits NO `provider`
            # object at all until the user sets at least one preference, so
            # OpenRouter's sticky routing (which keeps a long DeepSeek
            # conversation's prompt cache warm on one host) stays untouched.
            # In particular `sort` must never gain an explicit default value:
            # an always-on sort is an always-on cold cache.
            "providers": {
                "openai": {"api": "responses", "use_max_context_window": True},
                "anthropic": {"cache_ttl_1h_min_context_tokens": 150_000},
                "openrouter": {
                    # The one HARNESS-side key in this block: read by
                    # `SessionStreamFn._affinity_enabled`, never by
                    # `_openrouter_provider_preferences`, so it does not make
                    # the shipped config express a wire-level opinion. On by
                    # default — reusing the host that served the last turn is
                    # what keeps a long conversation's prompt cache warm, and
                    # any explicit routing preference below turns it off.
                    "provider_affinity": True,
                    "sort": "",
                    "order": [],
                    "only": [],
                    "ignore": [],
                    # The four switches are stored as ENUM strings, "" being
                    # "no opinion" — the same vocabulary as `sort`. The
                    # resolver still tolerates the bool a hand-edited YAML
                    # produces (`zdr: true`), so a config written by an older
                    # build of this branch keeps meaning what it said.
                    "allow_fallbacks": "",
                    "require_parameters": "",
                    "data_collection": "",
                    "zdr": "",
                    "enforce_distillable_text": "",
                    "quantizations": [],
                    "max_price": "",
                    "preferred_min_throughput": 0.0,
                    "preferred_max_latency": 0.0,
                },
            },
            # One ordered cascade for every text-model call. Entries may be
            # "provider/model" strings or {provider, model, effort} mappings;
            # usage-aware switching is opt-in because it spends one lightweight
            # quota request at user-message boundaries.
            "retry": {
                "enabled": True,
                "maxRetries": 10,
                "baseDelayMs": 500,
                "modelFallback": True,
                "usageAwareFallback": False,
                "usageReservePercent": 10,
                "usageAwareAccountPick": True,
                "fallbackChains": {},
            },
            # Search is useful on first run without a credential: DuckDuckGo
            # and Tavily keyless are both bounded fallbacks, so the default
            # rotates between them rather than depending on one free service.
            "web_search": dict(DEFAULT_WEB_SEARCH_CONFIG),
            # Web fetch is on by default and useful on a bare install: HTML falls
            # back to a stdlib renderer when the [fetch] extra is absent, so the
            # tool never depends on an optional dependency being present.
            "web_fetch": dict(DEFAULT_WEB_FETCH_CONFIG),
            # Subagent controls. ``models`` maps the lo/med/hi effort tiers to
            # "provider/model" selectors; ``max_running`` caps how many
            # background jobs (subagents AND backgrounded bash, which share one
            # pool) may run concurrently per session. Absent by default so the
            # ceiling lives in one place — AsyncJobManager's own default —
            # rather than being duplicated into every generated config file.
            # Set it when the machine or the models in use want a different
            # ceiling than the built-in one.
            "subagents": {},
            # Session-store cleanup policy, OFF by default. Every automatic
            # deleter that ever lived under ``sessions/`` — the age/count/byte
            # ceilings, the empty-directory reaper, the #576 "unused session"
            # backfill, the #622 exit-path rmdir — has been removed after the
            # last of them deleted 225 of an operator's 244 named sessions.
            # ``session.cleanup`` is the ONE remaining policy and it does
            # nothing at all unless ``enabled`` is true; the limits below are
            # inert without it. Read and written through ``settings_io``'s
            # nested path (``("session", "cleanup", ...)``) and consumed via
            # ``ConfigManager.get_nested_value`` on the same path, so the
            # flat-vs-nested key mismatch that made the #576 opt-out a no-op
            # cannot recur. Semantics are documented on the settings rows and
            # in ``local_operator.session.cleanup``.
            "session": {
                "cleanup": {
                    "enabled": False,
                    "max_sessions": 0,
                    "max_inactive_days": 0,
                    "max_total_bytes": 0,
                    "remove_empty": False,
                },
            },
            # What a child process the MODEL asks for may see of this
            # process's own environment. ``mode`` is ``inherit`` (the default:
            # the child gets a copy, so an operator's own commands behave as
            # they do in their terminal) or ``allowlist`` (the strict mode a
            # server-owned deployment turns on: only the SDK's safe set plus
            # ``inherit`` reaches the child, and credential-shaped names have to
            # be named explicitly). ``inherit`` lists extra names the strict
            # mode keeps (``LOCAL_OPERATOR_CONFIG_DIR`` when a command in the
            # shell must still resolve ``$(lop secret get ...)``); ``exclude``
            # names variables NEITHER mode passes on, winning even over the
            # harness's own injections.
            #
            # WHY the strict mode exists: the harness reads its provider API
            # key out of its own environment, so the copy is a spend credential
            # in the hands of any command the model writes — and the runs that
            # matter are the ones fetching attacker-influenceable pages. The
            # default is deliberately the permissive one because which mode is
            # right is a property of the DEPLOYMENT, not something the process
            # can observe: a laptop session and a server-owned run look
            # identical from inside. Read live, per call, by
            # ``tools/shell_env.py`` — the one reader, shared by the bash tool
            # and the eval worker.
            "shell_environment": {
                "mode": "inherit",
                "inherit": [],
                "exclude": [],
            },
        },
    }
)


def _fresh_default_config() -> Config:
    """A private copy of the shipped defaults, safe to mutate.

    :data:`DEFAULT_CONFIG` is a module-level object holding two mutable dicts,
    and a manager that adopted it directly wrote THROUGH it: on a machine with
    no config file yet, ``set_config_value`` mutated the process's idea of the
    defaults, so every later ``ConfigManager`` in the same process started from
    the last write instead of from the shipped values. It surfaced as
    ``/approvals default auto`` in one session leaking into the next session
    built in the same process — an app that had never read the file believing
    the gate was disarmed.

    Deep, not shallow: ``metadata`` and ``values`` are both dicts, and
    ``Config.__init__`` aliases ``metadata`` straight through, so a shallow
    copy would leave the timestamp shared.
    """
    return Config(deepcopy(vars(DEFAULT_CONFIG)))


# Name of the YAML configuration file
CONFIG_FILE_NAME = "config.yml"


class ConfigManager:
    """Manages configuration settings for Local Operator.

    Handles reading and writing configuration settings to a YAML file,
    with fallback to default values if no config exists.

    Attributes:
        config_dir (Path): Directory where config file is stored
        config_file (Path): Path to the config.yml file
        config (dict): Current configuration settings
    """

    config_dir: Path
    config_file: Path
    config: Config

    def __init__(self, config_dir: Path) -> None:
        """Initialize the config manager with default or existing settings.

        Creates a new ConfigManager instance that manages configuration settings.
        If a config file exists at the specified path, loads settings from it.
        Otherwise creates a new config file with default settings.

        Args:
            config_dir (Path): Directory path where the config file should be stored

        The config file will be named according to CONFIG_FILE_NAME and stored
        in the specified directory. Configuration is loaded immediately upon
        initialization.
        """
        self.config_dir = config_dir
        self.config_file = self.config_dir / CONFIG_FILE_NAME
        self.config = self._load_config()

    def _handle_bad_config(self, detail: str) -> None:
        """Report an unreadable config.yml and move it aside to config.yml.bad.

        Backing the file up rather than deleting it keeps the user's edits
        recoverable, and renaming it (rather than leaving it) is what stops the
        very next launch from failing identically: a broken file that stays in
        place turns one bad edit into a permanent lockout. Best-effort \u2014 if the
        rename cannot happen (read-only dir), the load still degrades to
        defaults, which is the whole point of catching this.
        """
        from local_operator.cli_style import ERROR, WARNING, paint

        print(paint(f"Error: {detail}", ERROR, stream=sys.stderr), file=sys.stderr)
        # Timestamp the backup so a SECOND bad edit does not clobber the first:
        # a plain `.bad` suffix means two broken saves in a row silently lose
        # the earlier recoverable copy, defeating the point of keeping it. The
        # timestamp is second-resolution, which is finer than a human can make
        # two edits, so collisions do not happen in practice.
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = self.config_file.with_suffix(self.config_file.suffix + f".bad.{stamp}")
        try:
            self.config_file.replace(backup)
            print(
                paint(
                    f"Moved the invalid file to {backup} and starting with defaults. "
                    "Run `local-operator config create` to write a fresh one.",
                    WARNING,
                    stream=sys.stderr,
                ),
                file=sys.stderr,
            )
        except OSError:
            print(
                paint("Starting with default configuration.", WARNING, stream=sys.stderr),
                file=sys.stderr,
            )

    def _load_config(self) -> Config:
        """Load configuration from file or create with defaults if none exists.

        Returns:
            Config: The configuration object
        """
        if not self.config_file.exists():
            # 0700 at CREATION only (item 17): config.yml and the transcripts and
            # credentials beside it are the same sensitivity class as the log dir
            # (paths.ensure_log_dir), and the default 0755 exposed the directory
            # to every other account on a shared host. Never chmod an existing
            # dir on upgrade — a user may have widened it on purpose.
            self.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            return _fresh_default_config()

        with open(self.config_file, "r", encoding="utf-8") as f:
            # A hand-edited config.yml with a YAML syntax error, or one whose
            # top level parses to something other than a mapping (a bare list or
            # scalar), used to raise a raw traceback straight out of startup \u2014
            # the CLI died before it could say which file was wrong. Catch both:
            # name the path and the parse error on one line, move the bad file
            # aside to config.yml.bad so the next launch starts clean instead of
            # failing identically forever, and point at `config create`. stderr
            # because ConfigManager is built on the `exec --json` path, whose
            # stdout is the event stream.
            try:
                loaded = _parse_config_stream(self.config_file, f)
            except yaml.YAMLError as exc:
                self._handle_bad_config(f"could not parse {self.config_file}: {exc}")
                return _fresh_default_config()
            if loaded is not None and not isinstance(loaded, dict):
                self._handle_bad_config(
                    f"{self.config_file} is not a valid configuration mapping "
                    f"(top level is {type(loaded).__name__})"
                )
                return _fresh_default_config()
            config_dict = loaded or deepcopy(vars(DEFAULT_CONFIG))

            # Check if config version is older than current version
            config_version = config_dict.get("version", "0.0.0")
            current_version = _package_version()
            # Compare as version TUPLES, not strings: "1.10.0" > "1.9.0" is
            # False lexicographically, so the warning fired on the wrong set of
            # versions entirely. stderr because ConfigManager is constructed on
            # the `exec --json` path, whose stdout is the event stream.
            if _version_tuple(config_version) > _version_tuple(current_version):
                print(
                    f"\n\033[1;33mWarning: Your config file version ({config_version}) "
                    f"is newer than the current version ({current_version}). "
                    "Please upgrade to ensure compatibility.\033[0m",
                    file=sys.stderr,
                )

            # Fill in any missing values with defaults
            if "values" not in config_dict:
                config_dict["values"] = deepcopy(vars(DEFAULT_CONFIG)["values"])
            else:
                default_values = vars(DEFAULT_CONFIG)["values"]
                for key, value in default_values.items():
                    if key not in config_dict["values"]:
                        config_dict["values"][key] = deepcopy(value)

            return Config(config_dict)

    # LOADING IS READ-ONLY. A migration used to live here, run from
    # ``_load_config`` on every load that found a retired key — so ANY process
    # that merely constructed a ConfigManager on a config dir with this code
    # rewrote the file: an un-isolated probe script did exactly that to the
    # operator's live config while this change was still under review, and
    # an older runtime then read the rewritten file unguarded (PR #645,
    # round 5). Migrations live in ``local_operator.config_migrations`` and
    # run from ONE explicit startup seam (``cli.main``); a library caller
    # constructing this class cannot trigger them.

    def _write_config(self, config: Dict[str, Any]) -> None:
        """Write configuration to YAML file.

        Creates the config file first if it doesn't exist.

        Args:
            config (Dict[str, Any]): Configuration dictionary to write
        """
        if not self.config_file.exists():
            # 0700 at creation for the same reason as _load_config above (item 17).
            self.config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            self.config_file.touch()

        # Ensure version and metadata are included
        if "version" not in config:
            config["version"] = DEFAULT_CONFIG.version
        if "metadata" not in config:
            config["metadata"] = deepcopy(DEFAULT_CONFIG.metadata)

        # Ensure created_at and last_modified are included
        if "created_at" not in config["metadata"]:
            config["metadata"]["created_at"] = datetime.now().isoformat()

        config["metadata"]["last_modified"] = datetime.now().isoformat()

        # ATOMIC. This was a plain `open(..., "w")`, which truncates the file
        # before it writes a byte: a crash, a full disk, or a kill between the
        # truncate and the flush left config.yml empty or half-written, and the
        # next launch met it as an unreadable config (moved aside to
        # config.yml.bad) with every setting gone. The window was tolerable
        # while writes were rare CLI operations; `/settings` writes on every
        # Enter, so it is now on an interactive path a user drives dozens of
        # times a session.
        #
        # Temp file in the SAME directory — os.replace is only atomic within a
        # filesystem, and /tmp is routinely a different one. fsync before the
        # replace so the rename cannot be ordered ahead of the data on a crash,
        # leaving a correctly-named empty file.
        #
        # The EXISTING file's mode is carried onto the replacement. os.replace
        # swaps in the temp file's inode, so without this every write would
        # silently reset the mode to mkstemp's 0600 — and a user who widened
        # config.yml on purpose (a shared host, a group-readable checkout) would
        # find it narrowed again on the next toggle. Same rule `_load_config`
        # states for the directory: 0600 at CREATION only, never on upgrade.
        directory = self.config_file.parent
        try:
            preserve_mode = self.config_file.stat().st_mode & 0o777
        except OSError:
            preserve_mode = None
        handle, temp_path = tempfile.mkstemp(
            dir=str(directory), prefix=".config.", suffix=".yml.tmp"
        )
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as f:
                yaml.dump(config, f, default_flow_style=False)
                f.flush()
                os.fsync(f.fileno())
            if preserve_mode is not None:
                os.chmod(temp_path, preserve_mode)
            os.replace(temp_path, self.config_file)
            # The DIRECTORY, after the rename. Syncing the file's data (above)
            # only guarantees the bytes; the rename that gives them the config's
            # name lives in the parent directory's own metadata, so a crash
            # between the two can still surface the OLD file on a filesystem
            # that has not flushed the entry. Cheap here because config writes
            # are user-paced, not a hot loop.
            #
            # Best-effort: some filesystems (and every Windows path) refuse
            # O_RDONLY on a directory or its fsync. The replace has already
            # succeeded at that point, so failing the write over an
            # unavailable durability upgrade would turn a working save into an
            # error for no gain.
            try:
                dir_fd = os.open(str(directory), os.O_RDONLY)
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
            except OSError:
                pass
        except BaseException:
            # Leaving a stray .config.*.yml.tmp beside a config the user is
            # about to hand-edit is its own small confusion, and the failure
            # is re-raised either way — the caller reports it.
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            raise

    def get_config(self) -> Config:
        """Get the current configuration settings.

        Returns:
            Config: Current configuration settings
        """
        return self.config

    def reload(self) -> None:
        """Re-read the config from disk, replacing the in-memory copy.

        Exists for the first-run setup flow: the TUI's ``/login`` writes hosting
        and model to config.yml through its own manager, and the session factory
        captured a DIFFERENT manager instance at launch whose in-memory config
        still reads empty. Reloading that instance before the post-login session
        rebuild is what lets the new hosting actually take effect \u2014 without it
        the reload resolves the same empty config and drops straight back into
        the setup state.
        """
        self.config = self._load_config()

    def update_config(self, updates: Dict[str, Any], write: bool = True) -> None:
        """Update configuration with new values.

        Args:
            updates (Dict[str, Any]): Dictionary of configuration updates
        """
        # Update each field individually to work with Config class
        for key, value in updates.items():
            self.config.set_value(key, value)

        if write:
            self._write_config(vars(self.config))

    def update_config_from_args(self, args: argparse.Namespace) -> None:
        """Update configuration with values from command line arguments.

        Only updates values that were explicitly provided via CLI args.

        Args:
            args (argparse.Namespace): Parsed command line arguments
        """
        updates = {}
        if args.hosting:
            updates["hosting"] = args.hosting
        if args.model:
            updates["model_name"] = args.model

        self.update_config(updates, write=False)

    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        # A COPY, for the reason `_fresh_default_config` exists: adopting the
        # module-level object made the next `set_config_value` a write into the
        # process's defaults.
        self.config = _fresh_default_config()
        self._write_config(vars(self.config))

    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get a specific configuration variable.

        ``key`` is a TOP-LEVEL key of ``values`` and is looked up verbatim: a
        dotted string such as ``"session.cleanup.enabled"`` is NOT split into
        a nested walk, it is looked up as the literal key ``"session.cleanup.
        enabled"`` (which is how the ``display.*`` flags are stored). Code
        that consumes a genuinely nested setting must use
        :meth:`get_nested_value` with the same path tuple ``settings_io``
        writes, or it reads a key nothing ever writes — that mismatch is what
        turned the #576 reaper's opt-out toggle into a silent no-op.

        Args:
            key (str): The configuration key to retrieve
            default (Any, optional): Default value if key doesn't exist. Defaults to None.

        Returns:
            Any: The configuration value for the key, or default if not found
        """
        return self.config.get_value(key, default)

    def get_nested_value(self, path: tuple[str, ...], default: Any = None) -> Any:
        """Walk ``path`` through nested mappings under ``values``.

        The reader that pairs with ``settings_io.write_setting`` for a
        ``Setting`` whose ``path`` has more than one element. Both sides take
        the same tuple, so a consumer that spells its path as the registry
        does cannot disagree with the writer about where the value lives.
        A non-mapping partway down (a hand-edited ``session: "yes"``) reads
        as absent rather than raising, matching ``settings_io.read_setting``.
        """
        current: Any = self.config.values
        for part in path:
            if not isinstance(current, dict) or part not in current:
                return default
            current = current[part]
        return current

    def set_config_value(self, key: str, value: Any) -> None:
        """Set a specific configuration variable.

        Args:
            key (str): The configuration key to set
            value (Any): The value to set for the key
        """
        self.config.set_value(key, value)
        self._write_config(vars(self.config))
