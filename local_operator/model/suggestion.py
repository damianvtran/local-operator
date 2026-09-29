"""Would a suggested provider/model pair run on THIS machine? (model-suggestion §4.1)

A hub-published agent or team may carry a ``model_suggestion`` — ``{hosting,
model}`` — and the consume side must decide whether to honour it. The decision
is a property of the CONSUMING machine (credentials stay here; the hub has no
provider knowledge and deliberately never checks availability), so this module
is the single authority for that question, composing the existing ones:

- ``providers.registry.get_provider_definition`` — is the provider id known?
- ``ProviderController.is_usable`` — a stored credential, an environment key,
  or a provider that needs none (a local server).
- ``providers.local.configured_local_providers`` — the app's own record that
  the user pointed a local server somewhere. A local preset is *usable* with no
  credential by design, so presence alone says nothing about whether a server
  is there; only the ``providers.<id>.base_url`` write (``_configure_local``)
  proves the user opted in.
- ``model.discovery.offered_model_ids`` — "the same set the picker painted
  from". Its own contract says ``None`` means "cannot be enumerated offline"
  and callers must ACCEPT the pair rather than refuse it; this resolver
  honours exactly that. Fully offline, no network.

THE RULE: fail over only on positive evidence; uncertainty accepts. A check
that cannot be answered (an unreadable credential store, an unreadable
catalogue cache) is not evidence of absence, so it yields the pair rather than
refusing it — a suggestion silently dropped because a database was momentarily
locked would teach the user the feature is broken.

Every ``local_operator`` import is function-local: this module is reached from
``agents.import_agent`` and ``teams.import_hub_team``, both of which sit on
heavy import graphs, and the startup-cost guards pin what those graphs are
allowed to pull in. Nothing here touches the network or writes anything.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The suggestion broke its shape (not an object, or a member missing/blank)
#: before any question could be asked of a provider. Lenient by contract: a
#: hand-edited zip or team.yml may carry anything, and "malformed" is answered
#: with a notice, never an error.
REASON_INVALID = "invalid"

#: No such provider id exists on this machine.
REASON_UNKNOWN_PROVIDER = "unknown_provider"

#: The provider is known, but nothing here can run it (no credential, no env
#: key) — the "not logged in" case.
REASON_PROVIDER_UNAVAILABLE = "provider_unavailable"

#: A local preset whose endpoint the user never pointed anywhere — the "not
#: installed" case, arm 1.
REASON_LOCAL_NOT_CONFIGURED = "local_not_configured"

#: An enumerable catalogue (the registry's rows plus any cached listing,
#: including a cached local one) does not offer the requested id — the
#: "unknown id" case, and "not installed" arm 2.
REASON_UNKNOWN_MODEL = "unknown_model"


@dataclass(frozen=True)
class ModelNotice:
    """A suggestion this machine could not honour, carried as DATA.

    Non-blocking by construction: the import already succeeded by the time this
    exists, and every consumer renders it its own way — a CLI line via
    :meth:`describe`, a route payload via :meth:`as_payload`. Nothing here
    raises, and nothing here is shown to a model.
    """

    reason: str
    requested_hosting: str
    requested_model: str

    def as_payload(self) -> dict[str, Any]:
        """The shape the desktop routes put on their result payloads.

        Declared by ``server.models.schemas.ModelSuggestionNotice`` on the wire;
        ``requested`` is echoed so a caller can render what was asked for
        without re-deriving it from the listing.
        """

        return {
            "reason": self.reason,
            "requested": {"hosting": self.requested_hosting, "model": self.requested_model},
        }

    def describe(self) -> str:
        """The one-line CLI rendering (copy round 1, C2/C3/C7: landed wording)."""

        if self.reason == REASON_INVALID:
            # Never render the empty quote pairs a malformed payload leaves
            # (``''``): name back only the half that was readable, so the one
            # message whose job is to stay calm about a hand-edited file does
            # not read as a bug (copy round 1, C2).
            named = []
            if self.requested_hosting:
                named.append(f"hosting '{self.requested_hosting}'")
            if self.requested_model:
                named.append(f"model '{self.requested_model}'")
            subject = "The model suggestion" + (f" ({', '.join(named)})" if named else "")
            return (
                f"{subject} was malformed and was not applied. " "Using your default model instead."
            )
        where = {
            REASON_UNKNOWN_PROVIDER: (
                f"this machine does not know the provider '{self.requested_hosting}'"
            ),
            REASON_PROVIDER_UNAVAILABLE: f"not logged in to '{self.requested_hosting}'",
            REASON_LOCAL_NOT_CONFIGURED: (
                f"no local server is configured for '{self.requested_hosting}'"
            ),
            # "on this machine", not the dangling "here": the check is this
            # machine's cached catalogue, and the unknown-provider arm above
            # names the same subject, so the family stays parallel (copy
            # round 1, C3).
            REASON_UNKNOWN_MODEL: (
                f"the model is not offered by '{self.requested_hosting}' on this machine"
            ),
        }.get(self.reason, f"the suggestion could not be applied ({self.reason})")
        return (
            # "Model suggestion", the field's own name on every other surface
            # (copy round 1, C7): one concept, one name.
            f"Model suggestion '{self.requested_model}' (hosting "
            f"'{self.requested_hosting}') was not applied: {where}. "
            "Using your default model instead."
        )


@dataclass(frozen=True)
class SuggestionVerdict:
    """The answer: whether the pair runs here, why not, and the pair itself.

    ``hosting``/``model`` carry the submitted pair with surrounding whitespace
    removed — the pair to PROMOTE when ``available``, and the pair to name in a
    notice when not. They are empty strings only when the submission was not a
    readable pair at all (the ``invalid`` reason).
    """

    available: bool
    reason: str | None
    hosting: str | None
    model: str | None

    def notice(self) -> ModelNotice | None:
        """The carried notice for an unavailable verdict; ``None`` when it applies."""

        if self.available:
            return None
        return ModelNotice(
            reason=self.reason or REASON_INVALID,
            requested_hosting=self.hosting or "",
            requested_model=self.model or "",
        )


def _submitted_pair(suggestion: Any) -> tuple[str, str] | None:
    """The suggestion's trimmed ``(hosting, model)`` pair, or ``None`` when malformed.

    A blank member is malformed for the same reason a half pair is not
    expressible anywhere in this feature: "hosting with no model" would resolve
    to a different model on every machine, which is not a recommendation.
    """

    if not isinstance(suggestion, Mapping):
        return None
    hosting = suggestion.get("hosting")
    model = suggestion.get("model")
    if not isinstance(hosting, str) or not isinstance(model, str):
        return None
    hosting = hosting.strip()
    model = model.strip()
    if not hosting or not model:
        return None
    return hosting, model


def _best_effort_echo(suggestion: Any) -> tuple[str, str]:
    """Whatever of the submission can be named back, for an ``invalid`` notice."""

    if not isinstance(suggestion, Mapping):
        return "", ""
    hosting = suggestion.get("hosting")
    model = suggestion.get("model")
    return (
        hosting.strip() if isinstance(hosting, str) else "",
        model.strip() if isinstance(model, str) else "",
    )


def _provider_is_usable(hosting: str, auth_store: Any | None) -> bool:
    """``ProviderController.is_usable`` over an injected or short-lived store.

    ``auth_store=None`` means "construct and close locally" — the pattern
    ``mobile/peer_model.py::provider_usable_here`` uses for hosts that do not
    hold a controller of their own. Tests inject a fake through the same
    parameter.
    """

    from local_operator.paths import config_dir
    from local_operator.providers.controller import ProviderController

    if auth_store is None:
        from local_operator.providers.auth_store import AuthStore

        with closing(AuthStore()) as store:
            return ProviderController(store, config_dir()).is_usable(hosting)
    return ProviderController(auth_store, config_dir()).is_usable(hosting)


def _local_realm_configured(hosting: str) -> bool:
    """Whether a local preset has been pointed somewhere (uncertainty ⇒ True)."""

    from local_operator.providers.local import (
        LOCAL_PROVIDER_IDS,
        configured_local_providers,
    )

    if hosting not in LOCAL_PROVIDER_IDS:
        return True
    try:
        return hosting in configured_local_providers()
    except Exception:  # noqa: BLE001 — a config read that failed is not "not installed"
        return True


def _model_is_offered(hosting: str, model: str, cache_dir: Path | None) -> bool:
    """Whether ``model`` is in the set the picker would have painted for ``hosting``.

    ``offered_model_ids`` returning ``None`` means "cannot be enumerated
    offline" and is accepted by its own contract; a read that RAISES is equally
    not evidence of absence.
    """

    from local_operator.model.discovery import offered_model_ids
    from local_operator.model.ids import normalised_id

    try:
        offered = offered_model_ids(hosting, cache_dir=cache_dir)
    except Exception:  # noqa: BLE001 — an unreadable catalogue is not an answer
        return True
    if offered is None:
        return True
    return normalised_id(model) in {normalised_id(candidate) for candidate in offered}


def resolve_model_suggestion(
    suggestion: Any,
    *,
    auth_store: Any | None = None,
    cache_dir: Path | None = None,
) -> SuggestionVerdict:
    """Whether a hub ``model_suggestion`` can run here, and why not when it cannot.

    Checks run in the order the reason table lists (first match wins, after the
    shape check — a malformed suggestion cannot be asked about a provider at
    all): provider known → provider usable → local realm configured → model
    offered. All offline; ``cache_dir`` exists so tests can point the
    catalogue reader at a fixture cache instead of the machine's own.

    Never raises for anything the suggestion could contain and never writes.
    """

    pair = _submitted_pair(suggestion)
    if pair is None:
        hosting, model = _best_effort_echo(suggestion)
        return SuggestionVerdict(
            available=False, reason=REASON_INVALID, hosting=hosting, model=model
        )

    from local_operator.providers.registry import get_provider_definition

    hosting, model = pair
    if get_provider_definition(hosting) is None:
        return SuggestionVerdict(False, REASON_UNKNOWN_PROVIDER, hosting, model)

    try:
        usable = _provider_is_usable(hosting, auth_store)
    except Exception:  # noqa: BLE001 — an unreadable store is uncertainty, not absence
        usable = True
    if not usable:
        return SuggestionVerdict(False, REASON_PROVIDER_UNAVAILABLE, hosting, model)

    if not _local_realm_configured(hosting):
        return SuggestionVerdict(False, REASON_LOCAL_NOT_CONFIGURED, hosting, model)

    if not _model_is_offered(hosting, model, cache_dir):
        return SuggestionVerdict(False, REASON_UNKNOWN_MODEL, hosting, model)

    return SuggestionVerdict(True, None, hosting, model)
