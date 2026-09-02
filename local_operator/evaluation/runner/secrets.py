"""How the parent turns a spec's ``SecretRef`` names into bytes for the worker.

The worker is spawned with a stripped environment and learns credentials only
over the private RPC pipe (``ResetStartParams.secrets`` / ``BeginRescueParams
.secrets``), so SOMETHING in the parent has to resolve a name into a value.
That something is deliberately an injected ``SecretResolver`` rather than a
call the runner makes itself:

* ``episode.py`` must stay isolated from the application (no credential store,
  no config -- ``test_isolation.py``). A resolver is handed in, so the runner
  never learns where secrets come from.
* Tests need to hand the runner a known canary value and prove it never lands
  in a bundle, a diagnostic, or a log. ``StaticSecretResolver`` is that seam.
* An operator script must be able to say exactly which env vars are secrets.
  ``EnvSecretResolver`` reads from an EXPLICIT mapping it is given, never from
  ``os.environ`` implicitly, so a resolver can only ever surface names the
  caller chose to expose to it.

The credential-store-backed resolver lives in ``host_secrets.py``, which is
the one runner module besides ``provider_client.py`` allowed to import the
application; keeping it out of this file is what lets ``episode.py`` import
the protocol here without dragging the store along.

Failure shape: a resolver raises ``MissingSecret(name)`` (or its subclass
``UnusableSecret(name)`` for a value the wire model refuses) and NEVER
includes a value in any exception. The runner turns that into a pre-bundle
failure whose diagnostic names the ref, so the operator learns which
credential to provision without a value ever crossing into a durable record.
Every resolver MUST build its ``ResolvedSecret`` through
``build_resolved_secret``: pydantic's ``ValidationError`` quotes the offending
input, so an over-long value (``max_length=8192``; a cert chain does it)
constructed anywhere else would put the secret on stdout.
"""

from __future__ import annotations

from typing import Mapping, Protocol, Sequence, runtime_checkable

from pydantic import ValidationError

from local_operator.evaluation.adapters.api import ResolvedSecret


class MissingSecret(LookupError):
    """A ref the resolver cannot satisfy. ``args[0]`` is the NAME, never a value.

    A ``LookupError`` subclass so a caller that already handles ``KeyError``
    shaped failures from a mapping-backed resolver is not surprised, but a
    distinct type so the runner can route it to a pre-bundle failure without
    catching every lookup error the resolution path could raise.
    """

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:
        return f"missing secret {self.name}"


class UnusableSecret(MissingSecret):
    """A value the store holds but the wire model refuses (empty, too long).

    A ``MissingSecret`` subclass so every guard that routes a missing ref to a
    name-only failure routes this the same way; a distinct type so the message
    tells the operator to FIX the stored value rather than provision one. The
    underlying ``ValidationError`` is deliberately not chained: pydantic
    renders ``input_value=`` in its message, which is the value itself.
    """

    def __str__(self) -> str:
        return f"unusable secret {self.name}"


def build_resolved_secret(name: str, value: str) -> ResolvedSecret:
    """The ONLY sanctioned constructor for a ``ResolvedSecret`` in the parent.

    ``ResolvedSecret`` bounds ``value`` (``min_length=1``, ``max_length=8192``)
    and re-validates the name pattern; a violation raises pydantic's
    ``ValidationError`` whose message embeds ``input_value='<head>…<tail>'``.
    That message would ride ``_diagnostic`` into ``EpisodeOutcome.diagnostic``
    and the run's stdout. Translating it here to a name-only error is what
    keeps a PEM-shaped credential out of the durable record.
    """

    if not value:
        raise MissingSecret(name)
    try:
        return ResolvedSecret(name=name, value=value)
    except ValidationError:
        failure = UnusableSecret(name)
    # Raised OUTSIDE the except block on purpose. ``raise ... from None`` only
    # clears ``__cause__`` and sets ``__suppress_context__``; ``__context__``
    # would still reference the ValidationError -- and its input -- for any
    # logger or serializer that walks the chain rather than the traceback.
    raise failure


@runtime_checkable
class SecretResolver(Protocol):
    """Resolve ``SecretRef`` names to values for the private RPC pipe only.

    ``resolve`` is synchronous on purpose: every implementation here reads a
    local file or an in-memory mapping, and an async surface would invite a
    network-backed vault into the episode's critical path without the retry
    and timeout treatment such a call would need. Raise ``MissingSecret`` for
    any name that cannot be satisfied; return them in the order requested.
    """

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]: ...


class StaticSecretResolver:
    """A fixed name -> value mapping, for tests and scripted proofs.

    Values are copied at construction so a caller cannot mutate them after the
    runner has already canaried the evidence writer against them.
    """

    def __init__(self, values: Mapping[str, str]) -> None:
        self._values = dict(values)

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        return _resolve_from_mapping(self._values, names)


class EnvSecretResolver:
    """Resolve names from an explicit environment-shaped mapping.

    The mapping is a PARAMETER, not ``os.environ``: an operator script passes
    ``{name: os.environ[name] for name in names_it_was_told_about}`` (or the
    whole environment if it genuinely means to), so which variables are
    reachable is a visible decision at the call site rather than an ambient
    property of the process. An empty or missing variable is a missing
    secret -- ``ResolvedSecret.value`` requires at least one byte, and an
    empty credential is never what an operator meant.
    """

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._environ = environ

    def resolve(self, names: Sequence[str]) -> tuple[ResolvedSecret, ...]:
        return _resolve_from_mapping(self._environ, names)


def _resolve_from_mapping(
    values: Mapping[str, str], names: Sequence[str]
) -> tuple[ResolvedSecret, ...]:
    # A malformed entry fails here, in the parent, before anything has been
    # allocated -- and by name only (``build_resolved_secret``).
    return tuple(build_resolved_secret(name, values.get(name, "")) for name in names)
