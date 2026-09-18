"""Which of this process's own environment a model-authored child may see.

Why this module exists
----------------------
``bash`` and ``eval`` both hand the model a real child process, and both used
to build that child's environment by copying the harness's own
(``os.environ.copy()`` / ``dict(os.environ)``) and layering the deliberate
injections on top. For a session on the operator's own laptop that is exactly
right — a command behaves the way it does in their terminal, and their
credentials, proxies and ``LANG`` are all there.

It is wrong for a run the operator is not watching, because the harness reads
its **provider API key out of its own environment**: a copy hands a spend
credential to any command the model writes, and the runs this matters for are
the ones whose whole job is to fetch and read attacker-influenceable public
pages. The composed path is attacker-controlled page content → model → shell →
credential, with no approval gate in the way on an unattended run. Nothing in
the shell even has to *print* the key for it to leak; ``env``, a traceback, an
``export -p`` or a subprocess's crash dump all carry it out.

This module is the ONE place that decision is made, so ``bash`` and ``eval``
cannot drift apart on it, and so the next tool that spawns a child has a
function to call rather than a pattern to copy.

The two modes
-------------
``inherit`` — the DEFAULT, and today's behaviour
    The child gets a copy of the harness's environment, minus ``exclude``. What
    an interactive operator wants: ``gh``, ``aws``, ``npm``, ``gcloud`` and
    anything else keep their tokens, endpoints and locale. Every existing
    session is unchanged until a deployment says otherwise.

``allowlist`` — the strict mode a server-owned deployment turns on
    The child starts from the SDK's own safe set (``HOME``, ``LOGNAME``,
    ``PATH``, ``SHELL``, ``TERM``, ``USER`` — the same list
    ``mcp.client.stdio.get_default_environment()`` copies) plus whatever the
    policy names in ``inherit``, and nothing else. Every credential the
    harness's own process holds is therefore absent unless the policy names it
    back, and a *credential-shaped* name the base set would have carried has to
    be named explicitly too.

The default is deliberately the permissive one. Which mode is right is a
property of the deployment — a run whose tools are not the operator's, on a
host whose environment the operator does not control — and it cannot be
inferred from anything the harness can observe: a laptop session and a
server-owned run look identical from inside the process. So ``inherit`` keeps
every existing session working, and the deployment that wants the strict mode
writes one key (see the ``shell_environment`` block in ``config.py``, or
``/settings`` → tools).

What the child may still legitimately see in either mode
-------------------------------------------------------
Both modes keep the harness's INTENTIONAL injections, and the strict mode
carries them through deliberately rather than by omission. What those are
differs per tool — the ``bash`` child gets ``NON_INTERACTIVE_ENV`` and the
session credential store's ``credential_env()``, while the ``eval`` worker gets
its own protocol channel — and each spawn site passes them as ``injections``
because they are values the harness decides to hand over, not names it happens
to have inherited:

* ``local_operator.tools.builtin.NON_INTERACTIVE_ENV`` (``CI``,
  ``LOCAL_OPERATOR_AGENT_SHELL``, the pager/editor overrides, ``TERM=dumb``) —
  the non-interactive contract the ``bash`` tool documents at its definition.
* the session credential store's ``credential_env()`` — the value the agent is
  meant to *use* without being able to *read* it. The store is the grant; this
  module does not second-guess it.
* the ``eval`` worker's scrub fd (``LOCAL_OPERATOR_EVAL_SCRUB_FD``), which is
  its own protocol channel and is popped from the environment by the worker.

``exclude`` still wins over all three — naming a variable there is an operator
saying "not even that".

What the strict mode does NOT close (read this before claiming it does)
----------------------------------------------------------------------
The strict mode removes the key from the child's **own** environment. It does
not make the key unreadable by the child, because the child runs as the same uid
as this process and the parent's environment stays readable from it:

* Darwin — ``ps eww -p $PPID`` prints the parent's environment, key and value.
* Linux — ``cat /proc/$PPID/environ`` does the same, and that file's mode is the
  same-uid read the model's child already has.

So the E2 exposure is NARROWED by this module, not closed: one command recovers
the credential from the process that spawned the shell. The root fix is not to
hold the provider key in the process environment at all (read it from the
credential store or a 0600 file at call time), or to run the child under a
different uid — neither of which belongs in a shell-environment policy. The
value this module does deliver, and the reason to keep it, is that the child's
own environment becomes a decision this repo makes rather than an accident of
how the harness was launched: deterministic, name-based, and reviewable.

The mode is resolved ONCE PER PROCESS
-------------------------------------
``load_policy()`` reads the policy on first use and memoises it for the life of
the process (the process is the session here). That is deliberate, and it is
the difference between a control and a suggestion: ``config.yml`` lives in a
directory the agent's own shell can write, as the same uid, so a policy re-read
per command is a policy the constrained party can LOWER mid-run — flip the mode
back to ``inherit`` from the shell this module was supposed to be constraining,
and the next command in the same session gets the credential back.
``shell_environment.mode`` is therefore a NEW-LAUNCH key, not a live one: a
write during a run takes effect at the next launch, and can never weaken the run
in flight.

The honest boundary of that, stated rather than implied: a deployment that wants
the strict mode must not let the run rewrite its own config *before* the first
tool call of that run. This module guarantees "the policy this process resolved
under", not "a policy nobody could change".

The credential-NAME machinery is insurance, not a filter
-------------------------------------------------------
:data:`CREDENTIAL_NAME_MARKERS`, :func:`provider_credential_names` and
:func:`is_credential_shaped` have exactly one enforcement role today: filtering
:data:`BASE_ALLOWLIST` so that a name added there for an unrelated reason cannot
become granted to every strict run, and building the DEBUG ``withheld`` line.
Suppression in the strict mode comes from **not granting** — everything unlisted
is dropped regardless of its shape — and the ``inherit`` mode deliberately never
consults them. So do not add a marker expecting it to close a leak: it will not,
and the next reader should not believe one is holding a strict run shut.

The same class, elsewhere in this harness (reported, not changed here)
---------------------------------------------------------------------
A config key that a running session re-reads is a key the agent's own tools can
raise or lower, because those tools can write the same ``config.yml``.
``approvals.tool_approval_mode`` is the one that matters most: its section is
``Scope.LIVE`` (``local_operator/settings_io.py``, ``SECTIONS``) and the
approval decision reads it per call, so a model that can write config can raise
its own approval mode — a wider hole than this module's, and one this module
neither creates nor closes. It is named here so the class is discoverable and so
nobody reads this module's once-per-process fix as if it covered the harness.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import yaml

from local_operator.config import CONFIG_FILE_NAME, ConfigManager
from local_operator.paths import config_dir

logger = logging.getLogger(__name__)

#: Where the policy lives under ``values``. Spelled ONCE here and pinned to the
#: ``settings_io`` rows by test
#: (``test_shell_environment_rows_share_the_reader_paths``) rather than imported
#: there, because ``settings_io`` must stay cheap for the CLI and this module is
#: not (same split as ``builtin.BASH_SHELL_PATH``).
MODE_PATH: tuple[str, ...] = ("shell_environment", "mode")
INHERIT_PATH: tuple[str, ...] = ("shell_environment", "inherit")
EXCLUDE_PATH: tuple[str, ...] = ("shell_environment", "exclude")

#: Inherit the parent environment (today's behaviour). The default.
MODE_INHERIT = "inherit"
#: Grant only the base set plus ``inherit``, and subtract credential shapes.
MODE_ALLOWLIST = "allowlist"
#: The registry default for ``shell_environment.mode``, quoted by the settings
#: row and by ``DEFAULT_CONFIG`` so the two cannot disagree about what "unset"
#: means.
MODE_DEFAULT = MODE_INHERIT

#: The SDK's own inherited-environment allowlist, restated rather than imported:
#: ``mcp`` is an optional dependency (the stdio transport imports it lazily, and
#: the bash tool must not pull it in on every command), and this list is the SDK
#: contract, not this repo's opinion of it. A test compares the two when ``mcp``
#: is importable, so a change to the SDK's set shows up as a failure here rather
#: than as a silent difference in what a child sees.
BASE_ALLOWLIST: tuple[str, ...] = ("HOME", "LOGNAME", "PATH", "SHELL", "TERM", "USER")

#: Substrings that mark an inherited variable as a credential BY NAME. The
#: provider keys have an authoritative list (see
#: :func:`provider_credential_names`) and this is the floor for everything else
#: a host might export — a service token, a broker password, an API key for a
#: tool this repo has never heard of.
#:
#: Chosen to be name-shaped rather than value-shaped on purpose: the decision is
#: made over a host environment that holds arbitrary secrets, and reading a
#: VALUE to decide would be both useless (nothing distinguishes a key from a
#: path) and a second place a secret is handled. Names are safe to log.
CREDENTIAL_NAME_MARKERS: tuple[str, ...] = (
    "API_KEY",
    "APIKEY",
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "PRIVATE_KEY",
    "CREDENTIAL",
    "_AUTH",
)


@dataclass(frozen=True)
class ShellEnvironmentPolicy:
    """The three stored keys, resolved and validated.

    A dataclass rather than a dict because the reader hands this straight to
    :func:`child_environment`, and an unvalidated mapping is how a mode typo
    would reach the spawn site as a string nothing matches.
    """

    mode: str = MODE_DEFAULT
    inherit: tuple[str, ...] = ()
    exclude: frozenset[str] = frozenset()


def _as_name_list(value: Any) -> tuple[str, ...]:
    """Coerce a stored list of names, tolerating the shapes a hand-edited file has.

    A ``config.yml`` written by an adapter or edited by a person is free to hold
    a comma-separated string where the settings row writes a list, and a strict
    deployment must not fall back to the permissive mode because of the TYPE of
    a value the operator did express. Blanks are dropped (a trailing comma is
    not a variable name) and order and duplicates do not matter.
    """
    if isinstance(value, str):
        parts: Sequence[Any] = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        return ()
    names: list[str] = []
    for part in parts:
        name = str(part).strip()
        if name and name not in names:
            names.append(name)
    return tuple(names)


def policy_from_values(mode: Any, inherit: Any, exclude: Any) -> ShellEnvironmentPolicy:
    """Resolve raw stored values into a policy, validating the mode.

    An absent or blank mode is "no opinion" and resolves to the default — a
    blank is what an unset field holds, and it is not an expression of intent
    the way a typo is.

    Anything else unrecognised resolves to the STRICT mode, with a warning
    naming the value. This is the one place in the module that fails closed, and
    for the reason the module exists: this key hardens a run nobody is watching,
    so a deployment typo (``"allow-list"``, ``"strict"``) must not silently mean
    "off" — that is the same silent-downgrade failure class as a security toggle
    that quietly stops applying. The warning is the other half: the deployment's
    own log names the value it has to fix.
    """
    resolved = MODE_DEFAULT
    if isinstance(mode, str) and mode.strip():
        candidate = mode.strip().lower()
        if candidate in (MODE_INHERIT, MODE_ALLOWLIST):
            resolved = candidate
        else:
            logger.warning(
                "shell_environment.mode %r is not one of %r/%r; using %r",
                mode,
                MODE_INHERIT,
                MODE_ALLOWLIST,
                MODE_ALLOWLIST,
            )
            resolved = MODE_ALLOWLIST
    return ShellEnvironmentPolicy(
        mode=resolved,
        inherit=_as_name_list(inherit),
        exclude=frozenset(_as_name_list(exclude)),
    )


#: Resolved policies, keyed by the config directory they were read from.
#:
#: WHY A CACHE, and why it is keyed by directory rather than global: the policy
#: must be resolved ONCE per process (see the module docstring) so that a write
#: during a run cannot LOWER it — but one process can serve more than one config
#: directory (a test suite, a ``--config-dir`` switch), and a single global would
#: hand the first directory's policy to every later one. The directory is the
#: identity of the policy, and it is what the tests isolate on.
_POLICY_CACHE: dict[str, ShellEnvironmentPolicy] = {}


def reset_policy_cache() -> None:
    """Forget every resolved policy. For TESTS and config-dir switches only.

    Production never calls this: clearing it is exactly the mid-run re-read that
    would let the constrained party lower its own policy. A test that rewrites
    ``config.yml`` under an already-resolved directory calls it to make the next
    ``load_policy()`` read again, and a caller that legitimately switches config
    directories gets a fresh resolution because the cache is keyed by directory.
    """
    _POLICY_CACHE.clear()


def load_policy() -> ShellEnvironmentPolicy:
    """The policy for this process, resolved on first use and then FROZEN.

    The resolution happens once per config directory per process; every later
    call returns the same object. That is the control, not an optimisation — a
    per-call read is a policy the agent's own shell can weaken between two
    commands (see the module docstring on the mode being a new-launch key).

    A read failure never raises: a command must still run when ``config.yml`` is
    unreadable or half written. What it resolves TO depends on whether there is a
    config to lose, and the asymmetry is deliberate:

    * No config file at all — there is no policy to honour, so the permissive
      default applies, which is the behaviour every session had before this
      module existed (DEBUG).
    * A config file that exists and cannot be read or parsed — a deployment that
      believes it is hardened must not silently lose the protection because a
      file is being rewritten under it, so the STRICT policy applies and a
      WARNING names the file and the failure (this is the one place the reader
      fails closed, matching what ``policy_from_values`` already does for an
      unrecognised mode).
    """
    key = str(config_dir())
    cached = _POLICY_CACHE.get(key)
    if cached is not None:
        return cached
    resolved = _read_policy()
    _POLICY_CACHE[key] = resolved
    return resolved


def _read_policy() -> ShellEnvironmentPolicy:
    """Read one policy off disk, applying the failure rule above.

    The validity PROBE comes first and is deliberate: ``ConfigManager`` does not
    raise on an unparseable file — it moves the file aside (``config.yml.bad.<ts>``)
    and continues with built-in defaults, so the read "succeeds" and every
    configured policy in it is gone. For a hardened deployment that is the silent
    downgrade this module exists to prevent, and it is invisible from the values
    alone. The probe answers only "is there a policy here that we failed to
    read?"; the VALUES still come from the same ``ConfigManager`` reader every
    other key uses, so this module cannot diverge from the config layer on what a
    stored value means.
    """
    config_file = config_dir() / CONFIG_FILE_NAME
    if _config_file_is_unreadable(config_file):
        logger.warning(
            "shell_environment policy could not be read from %s; using the STRICT "
            "policy %r: a config that cannot be read must not silently widen what a "
            "command the model runs may see. Fix the file, then restart the session.",
            config_file,
            MODE_ALLOWLIST,
        )
        return ShellEnvironmentPolicy(mode=MODE_ALLOWLIST)
    try:
        manager = ConfigManager(config_dir())
        mode = manager.get_nested_value(MODE_PATH, MODE_DEFAULT)
        inherit = manager.get_nested_value(INHERIT_PATH, ())
        exclude = manager.get_nested_value(EXCLUDE_PATH, ())
    except Exception as error:  # noqa: BLE001 — config trouble must never block a command
        if _config_file_exists(config_file):
            logger.warning(
                "shell_environment policy could not be read from %s (%s); using the "
                "STRICT policy %r: a config that cannot be read must not silently "
                "widen what a command the model runs may see. Fix the file, then "
                "restart the session.",
                config_dir() / CONFIG_FILE_NAME,
                error,
                MODE_ALLOWLIST,
            )
            return ShellEnvironmentPolicy(mode=MODE_ALLOWLIST)
        logger.debug(
            "no config file at %s; using the %s default",
            config_dir() / CONFIG_FILE_NAME,
            MODE_DEFAULT,
            exc_info=True,
        )
        return ShellEnvironmentPolicy()
    return policy_from_values(mode, inherit, exclude)


def _config_file_exists(path: Any) -> bool:
    """Whether a config file is present, without letting that check itself raise.

    A failure to even stat the path reads as "no file", so the permissive default
    applies: the strict branch is for a policy this deployment wrote and we then
    failed to read, not for a directory this process cannot see.
    """
    try:
        return path.exists()
    except Exception:  # noqa: BLE001 — see the docstring
        return False


def _config_file_is_unreadable(path: Any) -> bool:
    """Whether a config file is present and could not be read as a mapping.

    True only for the case that must fail closed: a file that EXISTS and that
    this process cannot read or cannot parse. False for an absent file (there is
    no policy to lose) and for an empty one (a deliberate way of saying nothing),
    and False when the check itself cannot stat the path — the strict branch is
    for a policy this deployment wrote and we failed to read, not for a
    directory this process cannot see at all.

    The parse here is a VALIDITY question, not a value read: PyYAML answering
    "this is not a mapping" is the same answer ``ConfigManager`` acts on when it
    quarantines the file, asked before that quarantine can take the policy away.
    """
    try:
        if not path.exists():
            return False
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — unreadable is the answer, not an exception
        return True
    try:
        parsed = yaml.safe_load(text)
    except Exception:  # noqa: BLE001 — unparseable is the answer
        return True
    return parsed is not None and not isinstance(parsed, dict)


_PROVIDER_KEY_NAMES: frozenset[str] | None = None


def provider_credential_names() -> frozenset[str]:
    """Every env var NAME this harness reads a provider key out of, upper-cased.

    DERIVED from the provider registry rather than restated here: a second list
    beside it is free to drift, and the one that drifts is the one that stops
    covering a provider added later. Both sources are read because both are used
    at runtime — ``PROVIDER_REGISTRY.env_keys`` (which answers ``None`` for the
    callable form, i.e. Anthropic's pick-between-two-vars resolver) and
    ``SupportedHostingProviders.requiredCredentials``, the table the CLI, the
    server schema and the setup prompt already read.

    Imported lazily and cached: only the strict mode asks, and the bash tool's
    path must not pay for the provider graph on a call that never reads it. A
    registry that cannot be imported leaves the name-SHAPE markers in
    :data:`CREDENTIAL_NAME_MARKERS` as the floor instead of disarming it — the
    markers cover every name this table could produce.
    """
    global _PROVIDER_KEY_NAMES
    if _PROVIDER_KEY_NAMES is not None:
        return _PROVIDER_KEY_NAMES
    names: set[str] = set()
    try:
        from local_operator.model.registry import SupportedHostingProviders
        from local_operator.providers.registry import PROVIDER_REGISTRY, env_key_name

        for definition in PROVIDER_REGISTRY:
            name = env_key_name(definition.id)
            if name:
                names.add(name.upper())
        for detail in SupportedHostingProviders:
            for name in getattr(detail, "requiredCredentials", ()) or ():
                if name:
                    names.add(name.upper())
    except Exception:  # noqa: BLE001 — an unimportable registry must not disarm the floor
        logger.debug(
            "provider registry unavailable; using credential name shapes only",
            exc_info=True,
        )
    _PROVIDER_KEY_NAMES = frozenset(names)
    return _PROVIDER_KEY_NAMES


def is_credential_shaped(name: str) -> bool:
    """Whether a variable NAME reads as a credential. Never reads the value."""
    upper = name.upper()
    if upper in provider_credential_names():
        return True
    return any(marker in upper for marker in CREDENTIAL_NAME_MARKERS)


def child_environment(
    policy: ShellEnvironmentPolicy | None = None,
    *,
    parent: Mapping[str, str] | None = None,
    injections: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment for a child the model asked to run.

    ``parent`` defaults to this process's environment — the thing being filtered
    — and is a parameter so a test can assert the policy over a fixture instead
    of over whatever the host happens to export.

    ``injections`` are the harness's own deliberate grants (see the module
    docstring): they are drawn from the harness's decisions rather than from the
    inherited environment, so they survive the strict mode by construction.

    ``exclude`` is subtracted LAST, so it wins over an injection too.
    """
    policy = policy if policy is not None else load_policy()
    source = os.environ if parent is None else parent
    if policy.mode == MODE_ALLOWLIST:
        # Two grants make up the allowlist. The SDK's base set first: those six
        # are what a child needs to BE a process (an interpreter to find, a home
        # to write, a shell to report) and none of them carries a credential.
        #
        # The base set is filtered for credential SHAPE rather than trusted
        # whole, because it is the half that grows without anyone asking this
        # question: a name added there for a perfectly good reason would
        # otherwise be granted to every strict run from then on, silently. A
        # credential-shaped base name still reaches the child when the policy
        # names it in ``inherit`` — which is a deliberate act with a diff behind
        # it, not a side effect of editing a constant here.
        granted = [name for name in BASE_ALLOWLIST if not is_credential_shaped(name)]
        granted += list(policy.inherit)
        env = {name: source[name] for name in granted if name in source}
        withheld = sorted(name for name in source if name not in env and is_credential_shaped(name))
        if withheld:
            # NAMES at DEBUG, never values: a deployment asking "why did my
            # command lose its token" needs to see the decision that was made,
            # and this is the only line that shows it.
            logger.debug("shell environment (allowlist): withheld %s", ", ".join(withheld))
    else:
        # Today's behaviour, and what an interactive operator wants. The parent
        # environment IS the explicit grant in this mode — it is the thing the
        # operator asked to keep — so the credential-shape filter above does not
        # apply here and must not: it would stop the mode from being the
        # behaviour it claims to be, and the E2 finding is about the strict mode
        # a deployment turns ON, not about redefining the operator's own laptop.
        env = dict(source)
    if injections:
        env.update({str(name): str(value) for name, value in injections.items()})
    for name in policy.exclude:
        env.pop(name, None)
    # A NAME LIST THAT MATCHED NOTHING IS A SILENT NO-OP, and a typo in one is the
    # same silent-downgrade class as an unrecognised mode. The two lists are
    # reported at different levels because their failure directions differ:
    # an ``exclude`` name that matches nothing means a DENIAL the deployment
    # believes it configured is not happening, so it warns; an ``inherit`` name
    # that matches nothing means a GRANT did not apply, which is usually benign
    # (a name that only exists in the deployment's own environment), so it is
    # debug. Neither line ever prints a value.
    unmatched_exclude = sorted(
        name for name in policy.exclude if name not in env and name not in source
    )
    if unmatched_exclude:
        logger.warning(
            "shell_environment.exclude names %s, which the environment being filtered "
            "does not define: a denial that matches nothing is silently a no-op "
            "(names are matched exactly and case-sensitively)",
            ", ".join(unmatched_exclude),
        )
    unmatched_inherit = sorted(name for name in policy.inherit if name not in source)
    if unmatched_inherit:
        logger.debug(
            "shell_environment.inherit names %s, which the environment being filtered "
            "does not define; nothing was granted for them",
            ", ".join(unmatched_inherit),
        )
    return env
