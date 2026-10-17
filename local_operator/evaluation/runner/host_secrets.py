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
silently. Both ``run_episode.py`` call sites already print a ``ValueError``
and exit ``EXIT_PREFLIGHT``, so the failure is loud and pre-allocation.

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

import os
import time
from typing import Any, Sequence

from local_operator.evaluation.adapters.api import ResolvedSecret
from local_operator.evaluation.runner.secrets import build_resolved_secret

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

    A ``ValueError`` subclass ON PURPOSE: ``scripts/run_episode.py`` routes
    ``ValueError`` from both of its resolution call sites (infra refs, secret
    refs) to a printed sentence and ``EXIT_PREFLIGHT`` -- the loud pre-allocation
    refusal the silent storm lacked. It deliberately does NOT subclass
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
    """Resolve the whole-resolution budget in seconds (``None`` = off)."""
    raw = os.environ.get(RESOLUTION_TIMEOUT_ENV)
    if raw is not None:
        try:
            ms = float(raw)
        except ValueError:
            ms = DEFAULT_RESOLUTION_TIMEOUT_MS
    else:
        ms = DEFAULT_RESOLUTION_TIMEOUT_MS
    return None if ms <= 0 else ms / 1000.0


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
