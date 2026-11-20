"""The credential-store-backed ``SecretResolver``.

WHY THIS SHAPE (2026-10-17, measured on the OSWorld session arm). ``resolve``
used to read the ENTIRE store first -- ``_stored_secrets`` enumerated every
non-provider row and fetched each one through the broker -- and then served
the handful of names the caller actually asked for. On the campaign's scratch
stores (130 non-provider rows) that is one broker round trip PER ROW, and an
episode boot resolves twice (infra refs, then secret refs), so ~260 round
trips before the first record line is even written. Measured on this host:
~5 ms per retrieve warm and unloaded, ~0.1-0.2 s under load, and every one of
the four concurrent session-arm runs of the 2026-10-17 18:36 wave spent ~11
minutes of boot SILENCE with stacks sitting in ``retrieve_secret ->
fetch_master_key -> recv_frame`` (``py-spy``-class dumps taken through a
SIGUSR1 sitecustomize rig). Each individual call is bounded by the broker's
own timeouts (``REQUEST_TIMEOUT_S``), but the AGGREGATE was not: a broker that
accepts and never answers makes the storm ~10 s x N, and the per-name errors
were swallowed, so the boot showed nothing at all -- the worst reading, an
apparent hang with no output. The resolver therefore (a) reads ONLY the
requested names, (b) checks a whole-resolution budget between fetches, and
(c) raises a NAMED failure when the budget expires instead of continuing
silently. ``run_episode.py`` routes the failure to a printed refusal that
exits ``EXIT_PREFLIGHT`` before any allocation: the infra-refs site catches
the ``ValueError`` directly, and the secret-refs site classifies the
diagnostic (this class is named in its ``_SECRET_DIAGNOSTICS`` tuple), so
both paths agree.

This is the SECOND runner module allowed to reach into the application (the
first is ``provider_client.py``, for the same reason: a real episode needs a
real credential, and the store is where operators keep them). Everything
else under ``runner/`` must stay free of the store so an episode's evidence
cannot silently depend on the operator's own configuration --
``tests/unit/evaluation/runner/test_isolation.py`` asserts both the
exception and the rule.

The store is the ENCRYPTED secret store (``local_operator.secrets``), whose
agent-class rows hold the names this resolver serves — an AWS key, an eval
harness token, anything an operator stored with ``lop secret set`` or moved out
of the plaintext file with ``lop secret migrate-env``. Reads go through the
store's agent namespace, so a ``LOP_PROVIDER_*`` row is deliberately NOT visible
here: the evaluation runner resolves the names its spec lists, and a provider
key is not one of them.

The legacy plaintext mapping (``~/.local-operator/credentials.env``) is NO LONGER
consulted (PR2a). It used to be a TRANSITION fallback for a name the store did
not hold; the plaintext file is no longer a credential source, so a ref the store
does not hold is reported missing rather than served from the file. This was the
last reader leg to leave, and its holder is now deleted outright (PR2b).

The store resolver never touches the environment: the runner's contract is that
the environment is reachable only through an explicit ``EnvSecretResolver`` over
names the caller listed, and a store resolver that quietly served ambient
variables would make "resolved from the store" a false claim in the operator's
own proof.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

from local_operator.evaluation.adapters.api import ResolvedSecret
from local_operator.evaluation.runner.secrets import build_resolved_secret

logger = logging.getLogger(__name__)

#: Default whole-resolution budget in ms; ``LOCAL_OPERATOR_SECRET_RESOLUTION_TIMEOUT_MS``
#: overrides, ``0`` disables. NOT a per-read timer: the store and broker calls
#: carry their own bounds, and what this deadline converts is "N individually
#: bounded calls that each go badly" into ONE bounded, named failure. The value
#: is deliberately far above every healthy measurement (whole-store scan was
#: sub-second warm; the requested-names read this module now performs measured
#: ~0.1 s): its job is to make an unbounded wait impossible, not to police a
#: slow one -- raise the knob if a store is merely slow, and the refusal
#: sentence says so.
DEFAULT_RESOLUTION_TIMEOUT_MS = 120_000.0

#: The environment knob, named in the refusal so the reader of a failed boot
#: learns the lever from the artifact itself rather than from a docstring.
RESOLUTION_TIMEOUT_ENV = "LOCAL_OPERATOR_SECRET_RESOLUTION_TIMEOUT_MS"


class SecretResolutionTimeout(ValueError):
    """The resolution did not finish inside its budget. Names the ref; never a value.

    A ``ValueError`` subclass ON PURPOSE: ``scripts/run_episode.py`` routes the
    failure from both of its resolution call sites (infra refs, secret refs) to
    a printed, named, pre-allocation ``EXIT_PREFLIGHT`` -- the loud refusal the
    silent storm lacked. The infra site catches the ``ValueError``; the
    secret-refs site classifies the diagnostic, which is why the class is named
    in the script's ``_SECRET_DIAGNOSTICS``. It deliberately does NOT subclass
    :class:`MissingSecret`: "the store did not answer in time" and "the store
    does not hold this name" are different problems with different fixes, and a
    reader must be able to tell them apart without reading this module.
    """

    def __init__(
        self, *, name: str, position: int, total: int, waited_s: float, budget_s: float
    ) -> None:
        super().__init__(
            f"secret resolution exceeded its {budget_s:g}s budget at ref {name!r} "
            f"({position} of {total}); {waited_s:.1f}s elapsed fetching values from the "
            f"credential store (and its broker). The run refuses before any allocation; "
            f"raise {RESOLUTION_TIMEOUT_ENV} if this store is merely slow."
        )
        self.name = name
        self.position = position
        self.total = total
        self.waited_s = waited_s
        self.budget_s = budget_s


def _resolution_timeout_s() -> float | None:
    """Resolve the whole-resolution budget in seconds (``None`` = off).

    ``0`` disables the bound (documented, deliberate). Anything else that
    cannot be read as a POSITIVE number is ANNOUNCED and falls back to the
    default: on the change that exists to end silent fallbacks, a typo'd
    override must not be ignored in silence -- and a NEGATIVE one must not be
    silently read as the off-switch (review NIT-1, 2026-10-17).
    """
    raw = os.environ.get(RESOLUTION_TIMEOUT_ENV)
    default_s = DEFAULT_RESOLUTION_TIMEOUT_MS / 1000.0
    if raw is None:
        return default_s
    try:
        ms = float(raw)
    except ValueError:
        ms = None
    if ms is not None and ms > 0:
        return ms / 1000.0
    if ms == 0.0:
        return None
    logger.warning(
        "ignoring %s=%r (expected a positive number of milliseconds, or 0 to "
        "disable); using the %gs default",
        RESOLUTION_TIMEOUT_ENV,
        raw[:64],
        default_s,
    )
    return default_s


class CredentialStoreResolver:
    """Resolve ``SecretRef`` names from the harness secret store.

    ``config_dir`` locates the encrypted store, or ``None`` for the
    HOME-derived default. It used to be a ``CredentialManager`` whose
    ``config_dir`` this read, and is now the bare PATH that class used to carry
    (PR2b) — the readers only ever needed the root, and ``ConfigManager`` owns it.

    The encrypted store is read in the AGENT namespace, so a
    ``LOP_PROVIDER_*`` row can never satisfy an evaluation ref.
    """

    def __init__(self, config_dir: Any) -> None:
        # Accepts the config ROOT (a ``Path``, the post-PR2b shape) or any object
        # that carries a ``config_dir`` — ``ConfigManager``, or an evaluation
        # host's stand-in. The attribute read is kept because a host builds this
        # before it needs the config stack, so this module imports no
        # configuration machinery at module scope.
        if hasattr(config_dir, "config_dir"):
            config_dir = config_dir.config_dir
        self._config_dir = config_dir

    def _stored_secrets(self, names: Sequence[str]) -> dict[str, str]:
        """The REQUESTED names' agent-class values, best-effort per name.

        ONLY ``names`` is read -- the old shape scanned every non-provider row
        (130 on the campaign's scratch stores) to serve ~18, which is the boot
        storm the module docstring measures. Provider-reserved names are
        refused before any read, exactly as the old enumeration filter refused
        them: they resolve as missing.

        Best-effort, matching the readers elsewhere: an absent, locked or
        damaged store is "this leg has nothing" rather than a failure, so a name
        the store cannot produce is reported missing rather than turning the
        resolve into an error. A single unreadable row is skipped, not fatal —
        one damaged record must not hide every other credential, the store's own
        ``list`` rule.

        The ONE failure that is NOT swallowed is the resolution budget expiring
        (:class:`SecretResolutionTimeout`), checked between fetches and raised
        OUTSIDE the per-name handler so it cannot read as "missing". A boot that
        cannot state which values it holds must fail with its reason, not wait
        silently -- measured: 4 concurrent boots spent ~11 minutes in this loop
        with no output.
        """
        from local_operator.secrets.access import retrieve_secret
        from local_operator.secrets.errors import SecretStoreError
        from local_operator.secrets.keys import store_path
        from local_operator.secrets.store import is_provider_secret_name

        base = self._config_dir
        # De-duplicated: a ref list that repeats a name must cost ONE broker
        # round trip, not one per appearance, and the values map is keyed by
        # name anyway.
        wanted = [name for name in dict.fromkeys(names) if not is_provider_secret_name(name)]
        values: dict[str, str] = {}
        if not wanted:
            return values
        try:
            if not store_path(base).exists():
                return {}
        except (SecretStoreError, OSError, ValueError):
            return {}
        budget_s = _resolution_timeout_s()
        started = time.monotonic()
        for position, name in enumerate(wanted, start=1):
            if budget_s is not None:
                elapsed = time.monotonic() - started
                if elapsed > budget_s:
                    raise SecretResolutionTimeout(
                        name=name,
                        position=position,
                        total=len(wanted),
                        waited_s=elapsed,
                        budget_s=budget_s,
                    )
            try:
                values[name] = retrieve_secret(name, base, role="agent").decode("utf-8", "replace")
            except (SecretStoreError, OSError, ValueError):
                continue
        return values

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        stored = self._stored_secrets(names)
        resolved: list[ResolvedSecret] = []
        for name in names:
            # The encrypted store, agent namespace. A name the store does not
            # hold resolves to an EMPTY value — the legacy plaintext fallback
            # this used to consult is GONE (PR2a), so a ref the operator never
            # stored is honestly missing rather than quietly served from a file
            # the store's own resolution no longer reads.
            resolved.append(build_resolved_secret(name, stored.get(name, "")))
        return tuple(resolved)


def _broker_client() -> Any | None:
    """The store's broker client IF this process has loaded it, else ``None``.

    Read from ``sys.modules`` rather than imported: a broker can only have
    been spawned by this process through that module, so "never loaded" is
    exactly "nothing to own" -- and an episode that never touches the store
    (``--no-store``) keeps importing nothing from the application, the rule
    this module's docstring states. Also the seam the scope's tests replace
    with a fake daemon.
    """

    return sys.modules.get("local_operator.secrets.client")


def _spawn_history(entry: Any) -> tuple[int, ...]:
    """Every pid a ``client.SpawnedBroker`` record covers, oldest first."""

    return tuple(getattr(entry, "pids", ())) or (entry.pid,)


@dataclass
class OwnedBrokerScope:
    """What :func:`owned_broker_scope` did at exit, for the host's own log."""

    #: ``(broker pid, outcome)`` per store this scope started a broker for; the
    #: outcome is ``client.settle_spawned_broker``'s token, or
    #: ``error: <ExceptionName>``. The pid is the daemon the outcome is ABOUT
    #: -- the one the socket's holder reported, which after a slow start is
    #: not necessarily the last one spawned -- and falls back to the latest
    #: spawn only when none of ours holds the socket any more.
    outcomes: list[tuple[int, str]] = field(default_factory=list)


@contextmanager
def owned_broker_scope() -> Iterator[OwnedBrokerScope]:
    """Own the lifetime of any secret broker started INSIDE this scope.

    WHY. ``CredentialStoreResolver.resolve`` (and the episode's own session)
    reach the store through its broker, which is started lazily and DETACHED so
    it outlives whoever started it -- right for an operator's store, whose
    sessions share one daemon. An episode runs against a per-episode scratch
    store, so the same laziness leaked one ``secrets.brokerd`` per episode
    (measured on v0.107.2: each orphan held its socket until the 30-minute
    idle timer; a campaign accumulates them). The episode host wraps its run in
    this scope so the broker it caused is gone when the episode is.

    THE RULE IS OWNERSHIP, NOT LOCATION -- nothing here asks whether the config
    root "looks like" scratch:

    - only a broker the store's client recorded THIS process spawning, and only
      one spawned AFTER the scope opened, is a candidate (the difference of two
      ``spawned_brokers()`` snapshots). A broker that was already listening --
      the operator's own, or another episode's -- was merely ATTACHED to: it
      never enters the registry, so it is never touched;
    - EVERY spawn inside the scope counts, not the last one. A daemon slower
      to listen than ``ensure_broker`` waits makes the next resolve spawn
      again, and the FIRST usually wins the socket; owning only the latest pid
      disowned exactly the daemon that then leaked (PR #4236 QA Q1). The claim
      handed to the stop is the scope's own spawns for that store -- the ones
      not already on record when it opened;
    - the stop itself (``client.settle_spawned_broker``) re-reads the daemon's
      live pid, and refuses if another session registered with it meanwhile or
      the store's unlock lives only in its memory. Its refusals are recorded
      as outcomes, not retried.

    Teardown hygiene never replaces the episode's result: a failing stop is
    logged and recorded, and the scope exits normally.
    """

    scope = OwnedBrokerScope()
    client = _broker_client()
    # No client loaded yet means nothing was spawned before the scope opened.
    before = client.spawned_brokers() if client is not None else {}
    try:
        yield scope
    finally:
        client = _broker_client()
        started = client.spawned_brokers() if client is not None else {}
        for key, entry in started.items():
            earlier = before.get(key)
            if client is None or earlier == entry:
                # Spawned before the scope opened: an outer owner's, not ours.
                continue
            # This scope's own spawns for the store: everything on record now
            # that was not on record when it opened.
            outer = set(_spawn_history(earlier)) if earlier is not None else set()
            owned = tuple(pid for pid in _spawn_history(entry) if pid not in outer)
            pid = entry.pid
            try:
                stop = client.settle_spawned_broker(entry.base, owned_pids=owned)
                outcome = stop.outcome
                if stop.pid is not None:
                    pid = stop.pid
            except Exception as error:  # noqa: BLE001 - teardown must not mask the episode
                outcome = f"error: {type(error).__name__}"
            scope.outcomes.append((pid, outcome))
            if outcome != "stopped":
                logger.warning(
                    "episode-started secret broker (pid %s) was not stopped: %s",
                    pid,
                    outcome,
                )
