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
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from local_operator.config import ConfigManager
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


def load_policy() -> ShellEnvironmentPolicy:
    """Read the policy from config AT CALL TIME, the way ``bash.shell`` is read.

    A fresh ``ConfigManager(config_dir())`` per call is what makes the key
    honestly LIVE: a deployment writes it once before the run starts, and an
    operator flipping it in ``/settings`` sees it on the next command rather
    than after a session rebuild. Any read failure means the DEFAULT policy and
    never an exception — a command must still run when ``config.yml`` is
    unreadable or half written, and the default is the behaviour every session
    had before this module existed.
    """
    try:
        manager = ConfigManager(config_dir())
        mode = manager.get_nested_value(MODE_PATH, MODE_DEFAULT)
        inherit = manager.get_nested_value(INHERIT_PATH, ())
        exclude = manager.get_nested_value(EXCLUDE_PATH, ())
    except Exception:  # noqa: BLE001 — config trouble must never block a command
        logger.debug(
            "shell environment policy unreadable; using the %s default",
            MODE_DEFAULT,
            exc_info=True,
        )
        return ShellEnvironmentPolicy()
    return policy_from_values(mode, inherit, exclude)


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
    return env
