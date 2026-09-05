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

from local_operator.web_fetch.models import DEFAULT_WEB_FETCH_CONFIG
from local_operator.web_search.models import DEFAULT_WEB_SEARCH_CONFIG

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
    """

    version: str
    metadata: Dict[str, Any]
    values: Dict[str, Any]

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize the config with default or existing settings.

        Creates a new Config instance that manages configuration settings.
        If a config file exists at the specified path, loads settings from it.
        """
        # Set version and metadata first
        self.version = config_dict.get("version", version("local-operator"))
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
DEFAULT_CONFIG = Config(
    {
        "version": version("local-operator"),
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
            "auto_save_conversation": False,
            # The tool-approval mode a NEW interactive session opens in, written
            # by ``/approvals default <mode>`` and read by the TUI at mount.
            # ``ask`` (prompt before write/exec tools) or ``auto`` (run them
            # without asking). A STRING and not a bool because the command's
            # vocabulary is a mode: a bool would have to be translated in both
            # directions, and the translation is where "off" ends up meaning
            # "prompting is off" in one place and "auto is off" in another.
            #
            # Read by the TUI only. The headless paths keep ``--yolo`` as their
            # one control: a saved file must not be able to disarm the gate of a
            # ``local-operator exec`` running in CI, where nobody is watching the
            # tools it approves.
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
            "providers": {
                "openai": {"api": "responses", "use_max_context_window": True},
                "anthropic": {"cache_ttl_1h_min_context_tokens": 150_000},
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
                loaded = yaml.safe_load(f)
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
            current_version = version("local-operator")
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

            if self._migrate_retired_session_cleanup_keys(config_dict["values"]):
                self._write_config(config_dict)

            return Config(config_dict)

    def _migrate_retired_session_cleanup_keys(self, values: Dict[str, Any]) -> bool:
        """Drop the keys of the removed session reapers; opt the user out of cleanup.

        One-time and idempotent: returns ``True`` only when it changed
        ``values``, which is the caller's cue to write the file. A config
        carrying none of the retired keys is untouched, so the ordinary load
        path costs nothing.

        The retired keys are the ceilings of the first eviction policy
        (``session_retention_max_*``) and the opt-out of the #576 unused-session
        reaper (``session.reap_unused``), which could be present in BOTH its
        nested form (what ``/settings`` wrote) and its flat-dotted form (what
        the reaper actually read) — the mismatch that made the toggle a no-op
        is why both spellings are handled. Every one of those mechanisms is
        gone, and leaving their keys behind would let a config file claim a
        protection ("reap_unused: false") that no longer means anything.

        A config that carried any of them belonged to a user who lived through
        the old reapers, so the migration writes ``session.cleanup.enabled:
        false`` EXPLICITLY rather than relying on the default — an explicit
        ``false`` survives a future change of default and is visible to anyone
        reading the file. An existing ``cleanup`` block is merged into, never
        replaced. ``config.yml`` is backed up beside itself before the rewrite
        so the user can see exactly what was removed.
        """
        retired_flat = (
            "session_retention_max_sessions",
            "session_retention_max_bytes",
            "session_retention_max_age_days",
            "session.reap_unused",
        )
        removed: list[str] = []
        for key in retired_flat:
            if key in values:
                del values[key]
                removed.append(key)
        session = values.get("session")
        if isinstance(session, dict) and "reap_unused" in session:
            del session["reap_unused"]
            removed.append("session.reap_unused (nested)")
        if not removed:
            return False

        if not isinstance(session, dict):
            session = {}
            values["session"] = session
        cleanup = session.get("cleanup")
        if not isinstance(cleanup, dict):
            cleanup = {}
            session["cleanup"] = cleanup
        cleanup.setdefault("enabled", False)

        # A store that predates the marker is the operator's real one; mark it
        # here so that IF they later enable cleanup it is eligible. Marking
        # does not enable anything — ``enabled`` was just pinned to false.
        from local_operator.session.cleanup import SESSIONS_DIRNAME, mark_store

        if (self.config_dir / SESSIONS_DIRNAME).is_dir():
            mark_store(self.config_dir / SESSIONS_DIRNAME)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup = self.config_file.with_name(
            f"{self.config_file.name}.pre-cleanup-migration.{stamp}"
        )
        try:
            backup.write_bytes(self.config_file.read_bytes())
        except OSError as exc:
            # Without a backup the migration must not rewrite the file: the
            # user could lose the record of what they had set. The retired
            # keys are inert either way, so leaving them costs nothing.
            logger.warning(
                "config migration: could not back up %s (%s); leaving it as is",
                self.config_file,
                exc,
            )
            return False
        logger.warning(
            "config migration: removed retired session-reaper keys %s from %s "
            "(backup at %s); session.cleanup.enabled is now explicitly false "
            "\u2014 no automatic session cleanup runs unless you turn it on in /settings",
            ", ".join(removed),
            self.config_file,
            backup,
        )
        return True

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
