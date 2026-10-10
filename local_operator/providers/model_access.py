"""Which providers a user can actually RUN a chat turn on — and which models are stranded.

Why this module exists: signing in to a provider never moved a default that was
already set. ``plan_login_defaults`` treated any registry-known hosting as
"working" (its test is registry membership, not credentials), so a user whose
config still named ``radient`` — from an earlier sign-in whose credential is now
gone — signed in to OpenAI or Anthropic and stayed on ``radient/auto``. Every
open session stayed there too: a conversation's journalled ``selected_model``
outranks config (``session_factory``/``cold_model``), so even a corrected config
would not move it. The user had a working key and every turn failed on a
provider they could not reach.

This module is the ONE place that answers "is this model reachable for this
user", for the login planner and the live-session re-home alike. It reuses
:meth:`ProviderController.usable_providers` as the predicate rather than adding a
fifth spelling of it (``usable_providers``, ``persisted_providers``,
``is_usable`` and ``view_rows().usable`` already exist), and narrows it for the
two questions the picker's wider answer is wrong for.

Why the picker's answer is too wide here. ``usable_providers`` counts every
keyless local server (``ollama``, ``lmstudio``, ``vllm`` ...) and the ``test``
provider as usable, always — right for a picker (show what could run), wrong for
"did this user previously have anything" and "is this default stranded":

* an empty account is NOT "has Ollama", or the first sign-in of a new user would
  never count as their first;
* an unconfigured keyless local (``ollama`` with no ``base_url``) has never
  served a chat, so a default naming one is STRANDED like any unreachable
  hosting: a later sign-in replaces it, which is the repair, not the surprise.
  A local server the user pointed somewhere (``base_url`` on record) is in the
  set like any other working choice, so it is neither stranded nor a target —
  see :func:`is_stranded`, which states the same rule at the predicate.

Pure of side effects apart from the credential-store read the controller does,
and imported lazily by its front ends (the TUI, the CLI and the server each pull
it in only at login time).
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Protocol

from local_operator.providers.registry import (
    credential_provider_id,
    get_provider_definition,
)

logger = logging.getLogger(__name__)


class AccessController(Protocol):
    """The slice of ``ProviderController`` this module reads.

    A Protocol rather than the concrete class because the hosts already treat a
    controller as duck-typed at this seam — the TUI and server pass reduced
    doubles through it in tests, and only two members are ever read. Typing them
    is what keeps a rename from silently reading as "signed out".
    """

    #: The config root the store-first readers resolve under. ``None`` means
    #: the HOME-derived default, exactly as on the real controller.
    config_dir: Path | None

    def usable_providers(self) -> set[str] | None: ...


def _borrowed_provider_keys(config_dir: Path | None) -> set[str]:
    """Provider ids this device may BORROW from a paired device, never local rows.

    A device paired to an owner that lent it a Radient bearer has no local
    credential row for ``radient`` (``radient_credentials._radient_auth_store``),
    yet its sessions run on it. Without this, ``usable_providers`` would call that
    device signed out of Radient and a sign-in to any other provider would strand,
    then re-home, a session that works. Both the planner and the re-home treat a
    borrowable key as accessible.

    Any failure answers "nothing borrowed": this only ever WIDENS what counts as
    accessible, and the caller's other evidence stands on its own. The identity
    is read without minting one (``placement_for_store`` loads, never creates), so
    a device that never joined a network pays one directory glob.
    """
    try:
        from local_operator.network.credentials.placement import (
            has_any_placement,
            placement_for_store,
        )

        if not has_any_placement(config_dir):
            return set()
        found = placement_for_store(config_dir)
        if found is None:
            return set()
        from local_operator.network.identity import load as load_identity

        identity = load_identity(config_dir)
        if identity is None:
            return set()
        _network_id, document = found
        return set(document.borrowable_keys(identity.device_id))
    except Exception:  # noqa: BLE001 - a placement fault must not break a login
        logger.debug("could not read credential placement", exc_info=True)
        return set()


def credentialed_chat_providers(
    controller: AccessController,
    *,
    config_values: Mapping[str, Any] | None = None,
) -> set[str] | None:
    """Providers this user holds a credential for AND can serve a chat turn.

    ``None`` is "cannot tell" (the store is unreadable) and is never "none":
    callers must treat it as *do not change anything*. It passes straight through
    from ``usable_providers``, whose narrow ``except`` is the contract.

    On top of ``usable_providers`` this drops:

    * providers that cannot serve CHAT at all (decision-only, speech-only,
      media-only) — a stored FAL or ElevenLabs key is not a model to run on;
    * keyless local providers, unless the user pointed one somewhere
      (``providers.<id>.base_url``, read from config alone through
      ``providers.local.configured_local_providers``) — see the module docstring;
    * the ``test`` provider, for the same reason.

    …and adds the keys a paired device may BORROW (:func:`_borrowed_provider_keys`).

    The call must stay on the thread that owns the controller's SQLite
    connection: ``usable_providers`` re-raises ``sqlite3.ProgrammingError`` for a
    connection used across threads, deliberately, so a worker-thread call here is a
    loud bug rather than a quiet "signed out".
    """
    usable = controller.usable_providers()
    if usable is None:
        return None
    from local_operator.providers.local import configured_local_providers

    configured_locals = configured_local_providers(config_values)
    credentialed: set[str] = set()
    for provider in usable:
        definition = get_provider_definition(provider)
        if definition is None:
            continue
        if definition.decision_only or definition.speech_only or definition.media_only:
            continue
        if definition.allows_missing_api_key and provider not in configured_locals:
            continue
        credentialed.add(provider)
    credentialed |= _borrowed_provider_keys(getattr(controller, "config_dir", None))
    return credentialed


def credentialed_chat_providers_here(
    *, config_dir: Path | None = None, config_values: Mapping[str, Any] | None = None
) -> set[str] | None:
    """:func:`credentialed_chat_providers` against THIS process's own store.

    For the callers with no controller of their own — the session owner, which
    re-checks access itself rather than trusting the front end that asked it to
    move (a sign-in in one window must not be able to switch a session on a
    claim the owner cannot confirm). The store is opened, read and CLOSED inside
    the call, like ``mobile.peer_model.provider_usable_here``; run it on a worker
    thread, since it is a SQLite read and the connection belongs to that thread.

    A store that cannot be opened is "cannot tell" (``None``), not an error.
    """
    import sqlite3

    try:
        with _own_controller(config_dir) as controller:
            return credentialed_chat_providers(controller, config_values=config_values)
    except sqlite3.ProgrammingError:
        # A connection crossing threads is a caller BUG, not an unreadable store:
        # ``usable_providers`` re-raises it before its degradation for the same
        # reason (D18), and a store fault must never dress itself as a plausible
        # "cannot tell".
        raise
    except (sqlite3.Error, OSError):
        # Same narrow pair ``usable_providers`` degrades on, one layer out: a
        # store that cannot even be OPENED (deleted, corrupt, unwritable) is as
        # unknowable as one that cannot be read, and must move nothing.
        return None


def usable_providers_here(*, config_dir: Path | None = None) -> set[str] | None:
    """``ProviderController.usable_providers`` against THIS process's own store.

    The WIDE access answer — keyless locals and ``test`` included — which is the
    predicate the published model-access claim is defined against (the TUI and
    the serve-side runtime both answer the band from it). Narrower questions use
    :func:`credentialed_chat_providers_here`; both open the store through
    :func:`_own_controller`, so they cannot drift on how it is found, closed or
    degraded: an unopenable store is ``None`` ("cannot tell"), never an empty
    set, and a cross-thread connection is re-raised as the caller bug it is.
    """
    import sqlite3

    try:
        with _own_controller(config_dir) as controller:
            return controller.usable_providers()
    except sqlite3.ProgrammingError:
        raise
    except (sqlite3.Error, OSError):
        return None


@contextmanager
def _own_controller(config_dir: Path | None) -> Iterator[Any]:
    """This process's own store, opened for ONE read and closed after it.

    The DB path is spelled from ``root`` explicitly, exactly as ``DesktopAuth``
    does: ``config_dir`` alone feeds only the store's env-override tier, so a
    caller whose root is not the ambient one would otherwise silently read the
    AMBIENT auth.db. One shape for both readers above, because "which store did
    this open" is the kind of thing a second copy gets wrong quietly.
    """
    from contextlib import closing

    from local_operator.paths import config_dir as resolve_config_dir
    from local_operator.providers.auth_store import AuthStore
    from local_operator.providers.controller import ProviderController

    root = config_dir if config_dir is not None else resolve_config_dir()
    with closing(AuthStore(root / "auth.db", config_dir=root)) as store:
        yield ProviderController(store, root)


def _storage_ids(providers: Collection[str]) -> set[str]:
    """The credential storage ids of ``providers`` (``xai-oauth`` -> ``xai``).

    A login FLAVOUR is a route to a credential, not a separate account: the set
    ``usable_providers`` returns holds the flavour ids a stored row makes usable,
    and a config hosting names the base provider. Comparing storage ids makes
    ``openai`` and ``openai-device`` one answer.
    """
    return {credential_provider_id(provider) for provider in providers}


def is_stranded(provider: str | None, accessible: Collection[str] | None) -> bool:
    """True when ``provider`` needs a credential this user does not have.

    Conservative in the one direction that matters: every doubt answers
    ``False``, because the consequence of ``True`` is a switched default or
    session.

    * ``accessible is None`` — unknowable, never stranded;
    * an empty provider, or one the registry does not own — that is the
      planner's separate "unusable hosting" repair, not this;
    * a provider whose STORAGE ID is in the set — signed in, env key, configured
      local server or a credential borrowed from a paired device.

    A keyless local server (``ollama``) is out of the set unless the user pointed
    one somewhere (``configured_local_providers``, read by
    ``credentialed_chat_providers``), and so is the ``test`` provider. That is
    the design's own rule rather than an oversight: *the picker* should offer
    them, but a default of ``ollama`` with no endpoint configured means no chat
    has ever run on it — nothing the user opted into, and a login to a working
    provider replacing it is the repair, not the surprise. A user who DID run
    one has a ``base_url``, which is their opt-in on record.
    """
    if accessible is None or not provider:
        return False
    if get_provider_definition(provider) is None:
        return False
    return credential_provider_id(provider) not in _storage_ids(accessible)


def is_accessible(provider: str | None, accessible: Collection[str] | None) -> bool:
    """True when ``provider`` is positively known reachable (the move TARGET test).

    The symmetric twin of :func:`is_stranded`, and deliberately so: a target must
    be PROVED reachable where a source must be proved stranded, and ``None``
    (unknowable) answers ``False`` in both, so an unreadable store moves nothing
    in either direction. Keyless providers follow the same rule here as there —
    a local server counts only once it is configured — because a re-home ONTO an
    arbitrary unconfigured port is exactly the kind of half-configured pair the
    planner refuses to write.
    """
    if accessible is None or not provider:
        return False
    if get_provider_definition(provider) is None:
        return False
    return credential_provider_id(provider) in _storage_ids(accessible)


def rehome_notice(old_label: str, new_label: str) -> str:
    """The transcript sentence for a session moved off an unreachable model.

    One spelling for the owner's transcript notice and the sign-in receipt, so
    what the user reads in the chat and in the settings toast cannot disagree.
    Names BOTH models: the user did not ask for this move, so it must say what
    left and what arrived.
    """
    old_provider = old_label.partition("/")[0] or old_label
    return f"Switched to {new_label} — not signed in to {old_provider}."


#: The owner's reply word for a session that could not be re-homed because it is
#: mid-turn. One spelling, because a CALLER classifies on it: the desktop pool
#: counts its deferrals by matching this exact string (the conversation itself is
#: told by :func:`rehome_deferred_notice`, which the owner emits).
REHOME_BUSY_REPLY = "kept: the session is working right now"


def rehome_deferred_notice(old_label: str, count: int = 1) -> str:
    """The sentence that tells a user their busy conversation did NOT move.

    A refusal nobody speaks is the silent variant of the bug the re-home exists
    to end: the receipts around the sign-in read as "fixed" while the
    conversation keeps using the model the app has just established this user
    cannot run. So the refusal is said, once, where it is recorded — with the one
    route out of the state named.

    "UNTIL THE TURN ENDS" IS A PROMISE THE OWNER KEEPS (round-2 M1/D6/Q2): every
    owner retries at its own turn end — the TUI in ``_settle_deferred_rehome``
    and a serve-side runtime in ``ServingSessionHandle._retry_deferred_rehome``
    — so the sentence describes a wait, not a dead end, on both surfaces. From
    the moment both retried, the word "current" only bought the row a one-word
    widow at 100 columns (design round 2, D8), so it is gone.

    Named per conversation rather than in the aggregate for ``count == 1`` (the
    common case); a multi-session receipt says the count without picking one
    conversation's model to speak for the others.
    """
    if count == 1:
        return (
            f"This conversation stays on {old_label} until the turn ends — /model switches it now."
        )
    return (
        f"{count} conversations stay on their model until their turns end — "
        "/model switches each one now."
    )


def rehome_still_stranded_notice(old_label: str) -> str:
    """The sentence for a deferred re-home that could not complete at turn end.

    The other half of :func:`rehome_deferred_notice`'s promise: a deferral the
    owner retries when the turn settles either moves the conversation or says it
    is still where it was, because a pending repair that can silently evaporate
    is the same silence one turn later. Spoken by BOTH owners — the TUI's
    ``_settle_deferred_rehome`` close-out and the runtime's
    ``_close_deferred_rehome`` — and only when the repair is still NEEDED: a
    conversation whose old provider works again, or that the user moved
    themselves, is closed in silence (nothing is wrong to complain about).
    """
    return f"This conversation is still on {old_label} — /model switches it when you are ready."


def is_first_provider_login(accessible: Collection[str] | None, provider: str) -> bool:
    """True when the provider just signed in to is the ONLY credentialed chat provider.

    The session re-home's gate (operator refinement, round 1): it exists for the
    state a session can be in BEFORE any provider login — started by an earlier
    sign-in, pinned to ``radient/auto``, with nothing behind it — and the FIRST
    provider login is the one that should move it. Every later login is a user
    adding a second provider to switch models with, and re-pointing their open
    conversations then is the surprise the planner's case-1 rule exists to
    prevent, wearing a session-shaped hat. So: re-home iff the credentialed chat
    providers OTHER than the one just added are empty.

    Radient counts like any other credentialed chat provider, deliberately: a
    prior Radient/web sign-in means this user already logged in to a provider,
    and a later OpenAI or Anthropic login therefore must NOT move sessions.

    ``None`` (unknowable store) answers ``False`` — every doubt in this module
    leans the same way, because the consequence of ``True`` is a switched model
    the user did not ask for.
    """
    if accessible is None:
        return False
    providers = _storage_ids(accessible)
    added = credential_provider_id(provider)
    return added in providers and not (providers - {added})
