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


#: The top-level keys ``Config`` models. EVERY setting is read from ``values``
#: (``ConfigManager.get_value`` / ``get_nested_value``), and ``version``/``metadata``
#: are the two the file carries beside it. A key outside this set is not read by
#: anything — which is worth stating in one place, because two behaviours hinge on
#: it: ``_load_config`` says so on stderr, and ``_write_config`` carries it through
#: rather than deleting it. Both were a bug: a hand-written
#: ``network: {advertise_hosts: ["203.0.113.7:4097"]}`` — the dotted key
#: ``network.advertise_hosts`` as YAML, which is how the mesh docs spell it — was
#: invisible to the reader AND erased by the next write, so an operator's declared
#: address did nothing and then vanished (leaving only the cleanup migration's
#: ``config.yml.pre-cleanup-migration.<stamp>`` backup behind).
_MODELLED_TOP_LEVEL = ("version", "metadata", "values")

#: ``(path, keys)`` already reported, so the warning below is once per process per
#: key set rather than once per ``ConfigManager`` — a session's construction path
#: builds five of them (see :func:`_package_version`), and five identical lines is
#: noise where one is information. The file's own ``stat`` is deliberately NOT part
#: of the key: an unrelated rewrite must not re-report a key the operator has not
#: touched yet, and once they move or delete it the warning stops on its own.
#: The key set is ``(path, string keys, unnameable keys)``: the third element is the
#: non-string keys this store cannot address, reported beside the others.
_UNMODELLED_WARNED: set[tuple[str, tuple[str, ...], tuple[str, ...]]] = set()

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


def read_config_values(config_dir: Path) -> "Dict[str, Any] | None":
    """The ``values`` mapping of ``config_dir``'s ``config.yml`` as it is on disk NOW.

    WHY THIS EXISTS beside :class:`ConfigManager`. A periodic background reader (the
    hub update runner re-reads its ``hub.*`` keys every tick) needs the file's current
    contents, and building a whole manager for that is the wrong tool: construction
    moves a config it cannot parse aside to ``config.yml.bad.<stamp>`` and prints to
    stderr, which is right at startup and wrong from a timer that can catch a
    non-atomic editor save mid-write. This reads through the same parse memo
    (:func:`_parse_config_stream`, so an unchanged file costs a ``fstat``) and
    NEVER writes, moves or reports anything.

    ``{}`` when there is no file or no ``values`` (nothing is set); ``None`` when the
    file cannot be read or parsed right now — the caller keeps what it had rather
    than treating a torn read as "everything unset".
    """
    path = Path(config_dir) / CONFIG_FILE_NAME
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as stream:
            parsed = _parse_config_stream(path, stream)
    except (OSError, yaml.YAMLError):
        return None
    if parsed is None:
        return {}
    if not isinstance(parsed, dict):
        return None
    values = parsed.get("values", {})
    return values if isinstance(values, dict) else {}


def _unmodelled_top_level(path: Path) -> Dict[str, Any]:
    """Top-level keys of the config file that this store has no field for.

    Read from the FILE, not from the live ``Config``, because ``Config`` holds only
    the modelled keys — that is what makes them unmodelled. An unreadable or
    unparseable file answers ``{}``: every caller is on a path that already reports
    that condition with its own message (``ConfigManager._load_config`` moves a bad
    file aside; ``_write_config`` is about to overwrite it), and a second complaint
    from here would be the louder of the two for no reason.
    """
    try:
        with open(path, "r", encoding="utf-8") as stream:
            parsed = _parse_config_stream(path, stream)
    except (OSError, yaml.YAMLError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {key: value for key, value in parsed.items() if key not in _MODELLED_TOP_LEVEL}


def _report_unmodelled_top_level(path: Path, config_dict: Dict[str, Any]) -> None:
    """Say ONCE per process that a top-level key is not a setting, and where one lives.

    The half of this that makes it a fix rather than a note: the key does nothing
    (every setting is read from ``values``), so an operator who wrote it — or who
    followed a sentence in the mesh docs that spells the path in dots — has to be
    TOLD, because the file gives no sign. ``_write_config`` keeps the key so the
    edit survives to be moved; this names it and its ``values:`` home, which is the
    spelling ``settings_io`` writes and ``get_nested_value`` reads.

    A library caller constructing ``ConfigManager`` in a loop reports once per key
    set (``_UNMODELLED_WARNED``), and the message is a WARNING rather than the
    ``print`` the version check beside it uses: this one belongs in the log the
    operator can find later, and the version check is about the file being NEWER
    than the build, which the user has to see while it happens.

    WHAT IT REPORTS, and the two decisions behind it. Non-string keys are NEVER sorted
    into the same pass as string ones: ``sorted`` over mixed keys raised ``TypeError:
    '<' not supported between instances of 'int' and 'str'`` inside
    ``ConfigManager.__init__``, which is every ``lop`` verb failing with a stack-trace
    panel on precisely the file class this mechanism exists to make readable (review
    round 1, B1) — so the string keys are filtered first, and the rest are ordered by
    ``repr`` beside them.

    Those non-string keys (YAML 1.1 parses ``2024:`` as an int, ``on:``/``yes:`` as a
    bool, and has dates, floats and null besides) cannot be given a ``values.`` home to
    move to, so they are named in their OWN SENTENCE with the only advice that applies —
    delete, or re-spell as a string — rather than as a clause inside the first sentence,
    where the key read as the subject of the store's behaviour and got no action (design
    review round 1, D2). A file where EVERY unmodelled key is one of those is therefore
    not silence either: it still gets that sentence.
    """
    unmodelled = sorted(
        key for key in config_dict if isinstance(key, str) and key not in _MODELLED_TOP_LEVEL
    )
    # A key this store cannot SPELL — no `values.` path exists to move it to — is still a
    # key somebody wrote, so a file where EVERY unmodelled key is one of those is not
    # silence. Its own sentence rather than a clause hung off the first one, and its own
    # action: the sibling sentence tells a string key where to move, which is advice
    # these cannot use.
    unnameable = sorted((key for key in config_dict if not isinstance(key, str)), key=repr)
    if not unmodelled and not unnameable:
        return
    seen = (str(path), tuple(unmodelled), tuple(repr(key) for key in unnameable))
    if seen in _UNMODELLED_WARNED:
        return
    _UNMODELLED_WARNED.add(seen)
    listed = ", ".join(repr(key) for key in unnameable)
    # NOT NAMED TWICE: the all-unnameable shape has already put the key in the first
    # sentence, so its second sentence takes `A non-string key`; the mixed shape has not,
    # so it names the key there. Two mentions in one warning read as two keys.
    if len(unnameable) == 1:
        subject = "A non-string key" if not unmodelled else f"Non-string key {listed}"
        unnamed = (
            f" {subject} cannot be a settings path at all — delete it, or re-spell it "
            "as a string if it was meant to be one."
        )
    elif unnameable:
        subject = "Non-string keys" if not unmodelled else f"Non-string keys {listed}"
        unnamed = (
            f" {subject} cannot be settings paths at all — delete them, or re-spell them "
            "as strings if that was the intent."
        )
    else:
        unnamed = ""
    if not unmodelled:
        logger.warning(
            "%s has top-level %s, which this store does not read: every setting lives "
            "under `values:`, so the key does nothing.%s",
            path,
            f"key {unnameable[0]!r}" if len(unnameable) == 1 else "keys " + listed,
            unnamed,
        )
        return
    homes = ", ".join(f"values.{key}" for key in unmodelled)
    named = f"key {unmodelled[0]}" if len(unmodelled) == 1 else "keys " + ", ".join(unmodelled)
    logger.warning(
        "%s has top-level %s, which this store does not read: every setting lives "
        "under `values:`, so the key does nothing. It is left in place rather than "
        "deleted — move it to %s (or set it in /settings) to make it take effect.%s",
        path,
        named,
        homes,
        unnamed,
    )


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
            # (``spawn_owned_session``), by a ``lop exec --control`` run
            # (``exec_control.start_exec_control``; its gates then keep
            # following the file), and — since 2026-09-28 — by the PLAIN
            # headless gate itself, the one a ``lop exec`` running in CI
            # decides on (``session_factory._approval_mode_is_auto``), which
            # previously took only ``--yolo``. A saved ``auto`` is the
            # operator's standing instruction that a run with nobody to ask
            # must not park or deny; ``--yolo`` still wins for one run.
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


def _dotted_write_refusal(key: str) -> "str | None":
    """Why ``key`` must not be written through :meth:`ConfigManager.set_config_value`.

    ``None`` means the write is fine. That is every non-dotted key, plus the
    declared FLAT-dotted ones (``display.shimmer``, ``keymap.*``), where the dot
    is part of the literal top-level name rather than a level of nesting.

    Everything else is a write that would land where no reader looks (#1920):
    a dotted key naming a declared NESTED setting (``subagents.models.hi``,
    whose real home is ``values.subagents.models.hi``), or naming nothing
    declared at all. ``Config.set_value`` is a plain ``dict.__setitem__``, so
    both were stored as a literal top-level key — the call returned, the file
    grew a line, and every reader (``get_nested_value``,
    ``settings_io.read_setting``, ``read_effort_tier_selectors``) walked right
    past it.

    The registry is asked, rather than the string being split, because the two
    cases are indistinguishable from the key alone and the registry is the
    authority on which is which (``Setting.path`` exists precisely because
    ``key`` and ``path`` differ for the flat-dotted flags). Splitting every
    dotted key would turn ``display.shimmer`` into a ``display:`` mapping that
    ``tui/settings.py`` does not read — one silent failure traded for another,
    which is ``settings_io``'s "THE ``display.*`` FLAT-KEY TRAP".

    An UNDECLARED literal flat key is refused too, and that is deliberate. The
    allowance is for keys the registry declares, not for any string containing a
    dot: a flat key that has been retired is exactly the shape a reader no
    longer looks at, which is the defect this guard exists to catch.
    ``session.reap_unused`` is the live example — the flat key the retired #576
    reaper read, still written by its migration straight into ``values`` — and a
    caller reaching for it through this writer is better told it is not a
    setting any more than handed a silent no-op.

    ``settings_io`` is imported function-locally: it imports this module, so a
    module-level import here would be a cycle. ``read_effort_tier_selectors``
    in ``harness/subagent.py`` is the precedent for the same move.
    """
    # A non-`str` key is not a dotted one — `"x" in 2024` would raise TypeError
    # from a guard that exists to explain a refusal. The repo pins the shape
    # (`test_a_non_string_top_level_key_cannot_take_the_store_down`): an int key
    # is stored verbatim and later reported by `_report_unmodelled_top_level`,
    # which is the surface that tells the user about it.
    if not isinstance(key, str) or "." not in key:
        return None

    from local_operator import settings_io

    setting = settings_io.BY_KEY.get(key)
    if setting is not None and setting.is_flat_dotted and key == setting.path[0]:
        return None

    if setting is not None:
        # Remedy first, mechanism last (design round 1, D1): this is public API
        # text whose one-line hosts — the TUI's detail row, the CLI's `Error:`
        # line — truncate at 34-98 cells, which sheds a trailing remedy entirely
        # and leaves the caller told what went wrong but not what to do. The
        # container is named rather than the root (D4): for
        # `subagents.models.hi` the leaf lives in `subagents.models`, and that is
        # the map a reader needs if they go to fix the YAML by hand.
        container = ".".join(setting.path[:-1])
        return (
            f"Use `lop config edit {key} <value>` — that key is a nested "
            f"setting, not a top-level one, so nothing was written. In Python, "
            f"`settings_io.write_setting(manager, "
            f'settings_io.resolve_key("{key}"), value)` does the same. Both '
            f"merge into the {container!r} mapping, which is where the runtime "
            f"reads it; a literal write would store the dotted name as a "
            f"top-level key that no reader looks at and report a success that "
            f"changed nothing."
        )
    return (
        f"Run `lop config list` for the real name, then `lop config edit <key> "
        f"<value>` — {key!r} is not a declared setting, so nothing was written. "
        f"A dotted key is stored literally at the top level, where no reader "
        f"looks."
    )


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
                    # `lop`, not `local-operator`: the launcher registered in
                    # [project.scripts] and the spelling the TUI's own hints use.
                    # The tree is split dead-even between the two (10/10 measured
                    # in design round 1, D3) — this is the newer copy, so it takes
                    # the correct side rather than propagating the older spelling.
                    "Run `lop config create` to write a fresh one.",
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

            _report_unmodelled_top_level(self.config_file, config_dict)

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

        # Every setting is read from `values`, so a key beside it is inert — and
        # `_write_config` writes the file back from `vars(self.config)`, which holds
        # only the modelled keys, so before this the inert key was also DELETED by
        # the next write. Read from DISK rather than remembered from load, because
        # the file is the only place that still has the key once `Config` has been
        # constructed, and because a key the operator removed by hand must stop being
        # carried through at the next write. The parse is the cached one
        # (`_parse_config_stream`), so a write pays a dict lookup when the file has
        # not moved since the read that preceded it, against the `yaml.dump` + fsync
        # it is about to do.
        document: Dict[str, Any] = dict(config)
        for key, value in _unmodelled_top_level(self.config_file).items():
            document.setdefault(key, value)

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
                yaml.dump(document, f, default_flow_style=False)
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

        The same two rules as :meth:`set_config_value`, for the same reasons: a
        key must be one this writer can place, and a WRITE must merge into what
        is on disk rather than over it. This is the second public whole-snapshot
        writer (``_write_config(vars(self.config))``), so it reverted concurrent
        edits in exactly the same way — and it is the one that made that
        reachable over HTTP, because ``app.state.config_manager`` is built once
        at startup and handed to every request unchanged, so a single
        ``PATCH /v1/config`` could undo a ``lop config edit`` typed in another
        terminal (#1920's vanished sibling; QA round 1's G1 reproduced that
        against the real app, and ``providers/controller.py`` holds a manager the
        same way).

        The reload is skipped when ``updates`` is empty, and that exception is
        load-bearing rather than an optimisation: an empty call is a FLUSH of the
        in-memory state, not a merge of new values. ``settings_io._delete``'s
        top-level branch deletes the key from the live mapping and then flushes
        it through this exact call, so reloading there would read the key back
        off disk and write it again — silently undoing every ``reset_setting``
        on a flat-dotted key.

        ``write=False`` still only mutates the in-memory copy, which is what
        :meth:`update_config_from_args` relies on to layer one run's CLI
        overrides over the file without persisting them.

        Args:
            updates (Dict[str, Any]): Dictionary of configuration updates
            write (bool): Whether to write the updated config to the config file

        Raises:
            ValueError: a key is dotted and is not a declared flat-dotted
                setting, so it could only be written inertly.
            settings_io.ConfigUnreadableError: ``write=True`` and ``config.yml``
                cannot be parsed, so no write may be based on it.
        """
        # Every key, before any mutation: a raise must not leave the caller
        # holding a half-applied update.
        for key in updates:
            refusal = _dotted_write_refusal(key)
            if refusal is not None:
                raise ValueError(refusal)

        if write and updates:
            # Function-local for the import-cycle reason `_dotted_write_refusal`
            # documents, and imported rather than re-derived so that "never base
            # a write on defaults" has exactly one implementation.
            from local_operator import settings_io

            settings_io._reload_before_write(self)

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

        ``key`` is a TOP-LEVEL key of ``values``. A key containing a dot is
        accepted only when the dot is part of the literal name — the declared
        flat-dotted settings, ``display.shimmer`` and friends. Any other dotted
        key raises :class:`ValueError` BEFORE anything is mutated, naming the
        route that does work; see :func:`_dotted_write_refusal` for why the
        refusal rather than a split-and-recurse, and why it is not routed
        through ``settings_io`` from here.

        Args:
            key (str): The configuration key to set
            value (Any): The value to set for the key

        Raises:
            ValueError: ``key`` is dotted and is not a declared flat-dotted
                setting, so a write here could only be inert.
            settings_io.ConfigUnreadableError: ``config.yml`` cannot be parsed,
                so no write may be based on it.
        """
        refusal = _dotted_write_refusal(key)
        if refusal is not None:
            raise ValueError(refusal)

        # Merge into what is on DISK, not into whatever snapshot this manager
        # happens to be holding. The write below dumps the whole in-memory
        # mapping, so a manager built before another writer's change silently
        # reverts it: three consecutive field writes through different managers,
        # with one of them stale, lose the other two (#1920's vanished sibling).
        #
        # The rule and its guard live in `settings_io`, where the facade's two
        # write primitives have carried them since review round 1 ("THE reason
        # this exists": a reload at the primitive cannot be forgotten by the
        # next entry point added, unlike one repeated at each facade method).
        # Imported rather than re-derived — a second spelling of "never degrade
        # to defaults as the base of a write" is exactly the drift that guard
        # exists to prevent — and function-locally, because `settings_io`
        # imports this module.
        from local_operator import settings_io

        settings_io._reload_before_write(self)

        self.config.set_value(key, value)
        self._write_config(vars(self.config))
