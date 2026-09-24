"""ProviderController — the TUI's window into providers, credentials, models
and usage.

One narrow facade over the provider, credential, model and usage layers: it
is exactly what the slash-command UX needs and nothing the TUI must not
reach. The app gets ONE instance injected by ``cli.py``; it owns
no terminal I/O (login is invoked with caller-provided callbacks, the same
interactive-login control flow) and no session state.

Read/credential surfaces are sync and I/O-free. Anything that touches the
network (usage fetches, OAuth refresh during login) is async and belongs in
a ``run_worker`` so the Textual message thread never blocks.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import random
import sqlite3
import time
from typing import TYPE_CHECKING, Any, Callable, Collection, Mapping, Protocol

import httpx

from local_operator.harness.types import AbortSignal, ModelSpec
from local_operator.model.configure import (  # noqa: F401  (used by callers)
    build_model_spec,
)
from local_operator.model.discovery import (
    DiscoveredModel,
    available_models,
    cached_available_models,
    invalidate_listing,
    is_meta_route_id,
)
from local_operator.model.naming import model_label
from local_operator.model.registry import static_models
from local_operator.providers.qwencloud_console import (
    QWENCLOUD_CONSOLE_PROVIDER,
    QWENCLOUD_TICKET_SECRET_NAME,
)
from local_operator.providers.registry import (
    AGGREGATOR_PROVIDERS,
    PROVIDER_REGISTRY,
    ProviderDefinition,
    credential_file_names,
    credential_provider_id,
    get_provider_definition,
    is_decision_only,
    list_login_providers,
    resolve_env_key,
    stored_provider_env_keys,
)
from local_operator.providers.usage import (
    USAGE_PROVIDERS,
    UsageReport,
    fetch_usage,
    usage_kinds,
    usage_supported,
)
from local_operator.providers.usage_cache import (
    USAGE_ACCOUNT_MAX_FAILURES,
    USAGE_FAILURE_BACKOFF_MS,
    USAGE_REPORT_TTL_MS,
    USAGE_UNAVAILABLE_RETRY_MS,
    UsageCacheStore,
    account_backoff_ms,
    fingerprint_accounts,
    fingerprint_secret,
    provider_cache_key,
    report_identity_key,
)

if TYPE_CHECKING:  # auth_store stays off this module's runtime import graph
    from pathlib import Path

    from local_operator.providers.auth_store import OAuthAccess, StoredCredential
    from local_operator.providers.oauth.callback_server import LoginCallbacks

LoginCallbackFactory = Callable[[ProviderDefinition], "LoginCallbacks"]

logger = logging.getLogger("local_operator.providers.controller")

#: How long an empty refresh keeps deferring to old data before it is believed.
#: The empty-over-data heuristic reads a blank answer over non-empty history as
#: an outage — but a provider that GENUINELY went quota-less (plan lapsed,
#: account emptied) would otherwise be re-fetched on every cool-down forever,
#: because each ``write_failure`` keeps the old row alive. Once the last real
#: data is this old, consecutive empty answers are accepted as the truth and
#: negative-cached at full TTL. Half an hour: long enough to ride out any
#: plausible rate-limit window, short enough that a lapsed plan stops burning
#: a request per warm tick the same day.
EMPTY_OVER_DATA_ACCEPT_MS = 30 * 60_000

#: The listing TTL the ``/model`` picker asks :meth:`live_catalogue` for. The
#: user is asking NOW, and the fetch already runs off-loop behind rows painted
#: from the registry, so a short TTL costs nothing visible. Fifteen minutes is
#: short enough that a release announced during a working session shows on the
#: next open, and long enough that scrolling in and out of the picker does not
#: re-list nine providers. Boot and repaint paths keep the default 24h hard TTL
#: (with an hourly background refresh) because there a request IS visible.
PICKER_TTL_S = 15 * 60

#: Notes for the QwenCloud console-ticket states the user can ACT ON. All are
#: painted by ``usage_panel.py``'s body builder, which prefixes a two-cell
#: indent and then truncates to ``_body_content_width()``.
#:
#: **The budget is a ladder, not a number, and 47 cells is only its top rung.**
#: Measured by rendering the real panel, never derived from the width
#: constants — the app's own chrome means ``overlay.screen_size`` reports 58
#: for a 60-column terminal, so the arithmetic chain (panel 54, content 50,
#: body 49, less the two-cell indent) starts two cells lower than the terminal
#: width suggests and gives 49 where the frame gives 47.
#:
#: Rendered budget by terminal width: 60→47, 58→45, 55→42, 50→37, 45→32,
#: 40→27, and 36 and below→25, where ``PANEL_MIN_WIDTH`` floors the card. An
#: earlier 47-cell locked note therefore fit ONLY at 60+: it lost its remedy
#: at every width from 32 to 59, which is most of the range a split pane
#: actually gets. Sizing to the top rung is the same mistake as deriving it.
#:
#: The budget belongs to the REMEDY: a truncated command still looks like an
#: instruction and then fails when followed, which is worse than no command —
#: ``_fit_status_note``'s own ladder says so for the dead-grant note. So each
#: string here is sized to survive as far DOWN the ladder as its remedy can be
#: spelled, and the state is dropped before the command is. Pinned by
#: ``test_the_note_survives_truncation_at_narrow_widths``, which renders the
#: panel at each width rather than counting characters.
#:
#: No backticks: the panel's own remedy note is ``sign-in expired — /login
#: kimi``, and a backtick costs two cells that buy no clarity in a dim,
#: unstyled row.
QWENCLOUD_TICKET_LOCKED_NOTE = "locked — lop secret unlock"
QWENCLOUD_TICKET_ORPHAN_NOTE = "no ticket — lop qwencloud-ticket set"

#: The three states above are the user's own to fix. These two are the store
#: telling the operator something is WRONG with it, and both docstrings
#: (:class:`~local_operator.secrets.errors.InsecurePermissions`,
#: :class:`~local_operator.secrets.errors.BrokerIncompatible`) forbid
#: swallowing them: the first says the exposure already happened and the
#: operator needs to know, the second that "live but unusable" laundered into
#: "unreachable" is what silently disarmed the redaction notice. A silent
#: ``None`` here is exactly that laundering, one layer up.
#:
#: Both remedies were run against a throwaway store rather than assumed:
#: ``lop secret status`` prints the offending path with its ``chmod 0600``
#: fix, and ``lop secret broker restart`` exists for version skew and is what
#: the broker's own mismatch error names.
QWENCLOUD_TICKET_EXPOSED_NOTE = "unsafe modes — lop secret status"
QWENCLOUD_TICKET_BROKER_NOTE = "old broker — lop secret broker restart"


@dataclasses.dataclass(frozen=True)
class CatalogueEntry:
    """One offerable model, provider-qualified, with the provider's auth state.

    Deliberately a flat record of PRESENTABLE values rather than a `ModelInfo`:
    the TUI's picker needs a display label and two prices and must not have to
    know that a context window of `-1` means unknown while `0` also does. The
    normalization happens once, here, where the registry's conventions are known.
    """

    provider: str
    model_id: str
    label: str
    context_window: int
    input_price: float
    output_price: float
    connected: bool
    default_context_window: int | None = dataclasses.field(default=None, kw_only=True)
    max_context_window: int | None = dataclasses.field(default=None, kw_only=True)
    #: The listing's OWN human name for this model, before ``model_label``'s
    #: honesty rule decides whether it may stand alone. Empty when the source
    #: named nothing.
    #:
    #: WHY this is carried beside ``label`` rather than recovered from it. That
    #: rule (``naming._unambiguous_name``) returns "" for a RESELLER, so a
    #: reseller's ``label`` degrades to the selector — deliberately, because the
    #: two shipped aggregators share ~398 of ~400 names and a name alone cannot
    #: say which route is answering. That is right for the TUI, whose row paints
    #: a separate selector column, and wrong for a surface with two slots (name,
    #: provider) where the route is ALREADY spelled by the provider slot: there
    #: the same rule leaves the name slot holding a slug, which is how 916 of 996
    #: phone rows came to read ``anthropic/claude-opus-5`` where the desktop
    #: reads ``Claude Opus 5``.
    #:
    #: So the name is carried, not re-derived. A consumer that can disambiguate
    #: the route by other means renders this; one that cannot keeps using
    #: ``label``. Nothing about ``label`` or the ranking changes.
    listing_name: str = dataclasses.field(default="", kw_only=True)
    #: This provider RESELLS the model rather than serving it. The picker ranks
    #: the direct route first when the same model is reachable both ways.
    aggregated: bool = False
    #: This entry is a META-ROUTE: its price depends on the model it dispatches
    #: to, so there is no pair of numbers that describes it.
    #:
    #: WHY this travels when ``free`` deliberately does not. ``free`` stops at
    #: the boundary below because the price pair can already carry it: ``0.0``
    #: means stated-free and ``-1.0`` means unknown, so the flag would be a
    #: second spelling of a fact the floats already hold, free to drift from
    #: them. That argument does not extend here, and the reason is that the
    #: float vocabulary is FULL. A router has no price, so it can only arrive
    #: as ``-1.0``/``-1.0`` — which is already spoken for by "nobody quoted
    #: this", a genuinely different answer that must keep rendering as a blank
    #: cell. Encoding a fourth meaning would mean inventing a fourth sentinel
    #: (``-2.0``) and teaching every reader of these two floats about it, which
    #: is the "smuggle a new meaning into an existing channel" move that
    #: ``format_price_pair`` and ``_price`` both warn against at length.
    #:
    #: So it is carried as its own bit, the way ``DiscoveredModel.routed`` is,
    #: and the prices stay honestly unknown underneath it. Nothing but the
    #: display reads it.
    routed: bool = False
    #: The time-of-use schedule NAME the two prices above are quoted at, if any
    #: (``ModelInfo.time_of_use`` -> ``DiscoveredModel.time_of_use``). Carried so
    #: the picker can render the rate IN FORCE rather than the stored PEAK rate —
    #: a DeepSeek figure shown off-peak at the peak number is 2x the truth for
    #: ~79% of the week. ``None`` is every provider with no time-of-day structure,
    #: and it is what keeps such a row rendering byte-identically to before.
    time_of_use: str | None = None

    @property
    def selector(self) -> str:
        """``provider/id`` — what ``/model`` accepts."""
        return f"{self.provider}/{self.model_id}"


class ControllerAuthStore(Protocol):
    """The credential-store slice this facade uses.

    Structural so a host can supply its own store; every member is called
    exactly as declared here (no extra keywords), which keeps the contract
    small enough for lightweight stand-ins.
    """

    def list_credentials(
        self, provider: str | None = None
    ) -> list["StoredCredential"]: ...  # pragma: no cover

    def upsert_credential(
        self, provider: str, credential: dict[str, Any]
    ) -> "StoredCredential": ...  # pragma: no cover

    def delete_credentials_for_provider(
        self, provider: str, disabled_cause: str = ...
    ) -> int: ...  # pragma: no cover

    def disable_credential(self, credential_id: int, cause: str) -> None: ...  # pragma: no cover

    def active_local_credential(
        self, provider: str, endpoint: str
    ) -> "StoredCredential | None": ...  # pragma: no cover

    async def get_oauth_access(self, provider: str) -> "OAuthAccess | None": ...  # pragma: no cover

    async def list_oauth_accesses(
        self, provider: str
    ) -> list["OAuthAccess"]: ...  # pragma: no cover

    def list_oauth_identities(self, provider: str) -> list["OAuthAccess"]: ...  # pragma: no cover

    async def get_api_key(self, provider: str) -> str | None: ...  # pragma: no cover


def _chat_providers() -> list[ProviderDefinition]:
    """The registry rows that may contribute CHAT models to a catalogue.

    Decision-only providers (``registry.is_decision_only`` — TypeSafe's Jev) are
    dropped HERE, at the one place every catalogue is assembled, rather than
    inside the three builders below or in each consumer of them. A provider whose
    wire rejects ``chat/completions`` on every host must not appear as a model
    the user can pick: selecting one produced a session that could not answer a
    single turn, which is a dead end the picker cannot explain (see
    ``ProviderDefinition.decision_only``).

    It also keeps the LIVE fetch honest: :func:`available_models` would otherwise
    be asked to list a decision endpoint's models and offer them as chat models,
    spending a network round trip to build a list that must not be shown.

    LOGIN FLAVOURS ARE DROPPED HERE TOO, and this is the ONE place that rule
    lives: a flavour declares ``store_credentials_as`` naming the provider it
    actually authenticates, and it adds NO catalogue its base does not already
    add. The two resolve to the SAME credential and therefore the SAME listing —
    ``_listing_credential`` follows ``store_credentials_as`` for the api_key,
    the ``is_oauth`` flag and the account scope alike — so the base row already
    carries everything the flavour row would.

    WHAT THIS FIXES, measured on the operator's catalogue: the picker offered a
    duplicated Radient listing under two labels. A key-only install saw 444
    ``radient/`` rows AND 444 ``radient-key/`` rows, the identical listing twice;
    querying ``auto`` returned ``radient/auto`` beside ``radient-key/auto``. The
    requirement is a single ``radient/`` namespace whenever the account is usable
    in EITHER form.

    WHY IT IS NOT RADIENT-SPECIFIC, and why an OAuth-only gate would be wrong.
    The credential-view suppression in :meth:`usable_providers` (
    ``store_credentials_as`` and the storage id in ``oauth_providers``) only
    hides a flavour when an OAUTH row exists — so a key-only install, before
    this, was offered BOTH labels (measured 2026-09-22 with a seeded
    ``upsert_credential("radient-key", {"type": "api_key", ...})``). And the
    duplication is not a Radient peculiarity: every flavour in
    ``PROVIDER_REGISTRY`` carries it — ``openai``/``openai-device``,
    ``xai``/``xai-oauth``, ``zai``/``zai-oauth``,
    ``alibaba-token-plan``/``alibaba-token-plan-oauth`` as well. Those four are
    already de-duplicated on the DESKTOP picker by the connected-only filter, but
    the filter that does it is NOT "only the base is usable": the flavour is
    suppressed only when an OAUTH row exists, so a KEY-ONLY OpenAI install has
    BOTH ``openai`` and ``openai-device`` in ``usable_providers`` (measured
    2026-09-22: ``usable_openai_key_only`` has both, while adding an OAuth row
    drops ``openai-device``). What actually keeps the desktop clean on every
    install is this catalogue filter, which is the point — the four flavours still
    reached the LIVE catalogue and the phone's sheet, so naming radient here would
    have been a special case of a general rule. The flavour's target is always a
    non-decision-only registry row, so a flavour is dropped exactly when its base
    is in the chat providers — which is always.

    A caller that names a flavour explicitly is served the BASE catalogue ONLY IF
    IT ALSO NAMES THE BASE. A flavour-ONLY ``providers`` set matches no chat
    provider (the flavour is already gone from the registry above), so
    ``live_catalogue(providers={"radient-key"})`` returns ``rows: []`` and
    ``statuses: {}``. The alias is NOT resolved in the narrowing step: that
    argument is a credential-admission filter and an empty answer is the CORRECT
    one for a CHAT catalogue request naming only a row that can never feed one —
    the same answer an empty collection gets, honoured literally. No in-tree
    caller hits it: the phone passes :meth:`persisted_providers`, whose answer CAN
    name the flavour (a key-only install measures ``['radient', 'radient-key']``)
    but ALWAYS names the BASE alongside it — the stored row lands under the base,
    and the flavour is admitted only because that base row is present — so a
    flavour-ONLY set never arises. Adding a translation there would put a second
    spelling of ``store_credentials_as`` on the admission path, which is what this
    filter exists to avoid.

    A flavour IS a hosting that ``configure_model`` accepts — ``get_provider_definition``
    resolves all five, so a spec can be built and the sentence that used to stand
    here ("flavours are not hostings ``configure_model`` accepts, so nothing may
    run under one") was false. A session that names one RUNS: the spec carries the
    flavour's own id (measured: ``build_model_spec("radient-key", "auto")`` yields
    ``provider="radient-key"`` at ``https://api.radienthq.com/v1``) and the stream
    path resolves the credential through the auth store's own alias cascade, which
    is why ``get_api_key`` returns the same key under both spellings (measured:
    ``radient-key`` and ``radient`` both yield the stored key). Dropping the
    flavour ROW is therefore lossless for a different reason than "nothing can run
    under it": the base and the flavour resolve to ONE credential and therefore ONE
    listing, so the base row already carries every model the flavour row would.
    """
    chat = [definition for definition in PROVIDER_REGISTRY if not is_decision_only(definition.id)]
    ids = {definition.id for definition in chat}
    return [
        definition
        for definition in chat
        if not (definition.store_credentials_as and definition.store_credentials_as in ids)
    ]


class ProviderController:
    """Provider/model/credential/usage facade for interactive front ends."""

    def __init__(
        self,
        auth_store: ControllerAuthStore,
        config_dir: "Path | None" = None,
        *,
        login_callbacks: LoginCallbackFactory | None = None,
        usage_cache: UsageCacheStore | None = None,
    ) -> None:
        self.auth_store = auth_store
        # The config ROOT the store-first readers resolve under. Only the path is
        # kept: PR2b deleted the ``CredentialManager`` whose ``config_dir`` this
        # used to be read off, and a second carrier of a path ``ConfigManager``
        # already owns is the "second way of doing things" the working
        # principles call a defect.
        self.config_dir = config_dir
        # Terminal-bound login callbacks. The CLI's print/input callbacks are
        # used by default; an embedding host (e.g. a Textual app) injects
        # callbacks that yield the terminal before the flow runs.
        self._login_callbacks = login_callbacks
        # Shared on-disk usage cache (see providers.usage_cache). Built lazily
        # so a host that never asks for usage pays nothing, and injectable so
        # tests can aim it at a temp file. Deliberately SHARED across every
        # lop session on this machine: several terminals run at once, and one
        # process's refresh is every process's answer.
        self._usage_cache = usage_cache

    def set_login_callbacks(self, factory: "LoginCallbackFactory | None") -> None:
        """Install host-specific login callbacks after construction.

        The CLI builds this controller before it knows whether a TUI will run,
        and a TUI's callbacks are fundamentally different in kind rather than in
        wording: it renders the authorization URL into its own transcript and
        must NEVER read from the real stdin, because doing so either fights the
        app for the terminal or requires suspending it.
        """
        self._login_callbacks = factory

    # -- discovery ---------------------------------------------------------
    def login_providers(self) -> list[ProviderDefinition]:
        """Providers offering an interactive login, registry order."""
        return list_login_providers()

    def provider(self, provider_id: str) -> ProviderDefinition | None:
        return get_provider_definition(provider_id)

    def credentials(self) -> list["StoredCredential"]:
        """Every active stored credential (StoredCredential rows)."""
        return self.auth_store.list_credentials()

    def has_any_credential(self, provider: str) -> bool:
        """Whether ``provider`` (or its storage id) has a stored credential."""
        storage = credential_provider_id(provider)
        return any(c.provider == storage for c in self.auth_store.list_credentials(provider=None))

    def is_usable(self, provider: str) -> bool:
        """Whether ``provider`` has a credential the session could actually run on.

        Wider than :meth:`has_any_credential`, which only sees the AuthStore. A key
        in the ENVIRONMENT is a working credential by every measure that matters —
        it is the one the stream-time cascade resolves, so a session started that
        way runs fine — and reporting such a provider as "needs a login" was both
        wrong and unactionable, since there is no login to perform.

        ``allows_missing_api_key`` providers (a local Ollama) are usable with no
        credential at all, which is the whole point of running one.
        """
        definition = get_provider_definition(provider)
        if definition is not None and definition.allows_missing_api_key:
            return True
        if self.has_any_credential(provider):
            return True
        return bool(resolve_env_key(provider))

    def usable_providers(self) -> set[str] | None:
        """Every provider id a turn could run on now — or ``None`` when unknowable.

        ``None`` is not "none of them": it means the credential store could not be
        read at all, which is a different answer and the only one that is honest
        when SQLite is locked. A caller that filters on this set has to show
        EVERYTHING in that case, because an empty model list reads as "you own no
        models" — a claim the app cannot make when it read zero rows.

        One store scan for the whole registry, where :meth:`is_usable` costs one
        per provider: the catalogue asks this question about a dozen providers on
        the keystroke that opens the picker.

        The ``except`` is deliberately NARROW. It used to be a bare
        ``except Exception``, which caught ``sqlite3.ProgrammingError`` from
        calling a connection across threads and reported it as "store
        unreadable" — so :meth:`initial_catalogue` took its show-everything
        degradation and every model in the catalogue was labelled connected on a
        machine with no credentials at all (D18). A locked or corrupt store is an
        ENVIRONMENT fact this method must survive; a connection used from the
        wrong thread is a BUG in the caller, and a bug that dresses itself as a
        plausible degraded state is one nobody finds. Only the environment
        failures degrade here; everything else raises and gets fixed.
        """
        try:
            stored_rows = self.auth_store.list_credentials(provider=None)
            stored = {row.provider for row in stored_rows}
            oauth_providers = {
                row.provider for row in stored_rows if row.credential_type == "oauth"
            }
        except sqlite3.ProgrammingError:
            # Misuse of the sqlite API — a connection crossing threads, a closed
            # handle. Re-raised BEFORE the degradation below because it subclasses
            # `DatabaseError`, so any except clause broad enough to cover a corrupt
            # store would swallow it again.
            raise
        except (sqlite3.Error, OSError):
            # Locked, busy, corrupt, or unreadable on disk — genuinely unknowable.
            return None
        usable: set[str] = set()
        for definition in PROVIDER_REGISTRY:
            storage = credential_provider_id(definition.id)
            # If the user has an active OAuth sign-in under the base provider
            # (e.g. `radient` OAuth), suppress secondary legacy API-key flavours
            # (`radient-key`) from being treated as usable alternatives.
            if definition.store_credentials_as and storage in oauth_providers:
                continue
            if (
                definition.allows_missing_api_key  # a local Ollama needs no credential
                or storage in stored
                or resolve_env_key(definition.id)
            ):
                usable.add(definition.id)
        return usable

    def persisted_providers(self) -> set[str] | None:
        """Every provider id backed by a STORED credential — or ``None``, unknowable.

        :meth:`usable_providers` minus two rungs, and the difference is a
        security boundary rather than a nicety. That method counts a key in the
        ENVIRONMENT and an ``allows_missing_api_key`` local server as usable,
        which is right for a local session: the stream-time cascade resolves
        both, so a turn started that way runs. It is wrong for a REMOTE picker.
        The mobile daemon is typically launched by a service manager whose
        environment the phone's user never chose and cannot see, so an ambient
        ``ANTHROPIC_API_KEY`` inherited from a shell profile would silently add
        an account to a picker reachable over a tunnel — and a keyless local
        server would advertise models against an endpoint that may not even be
        running. Only what the owner deliberately persisted authorizes a listing
        there.

        BOTH persisted stores count, because the two sanctioned flows write
        different ones: ``/login`` writes the AuthStore (auth.db) and ``lop
        credential update`` writes a provider-class ROW in the encrypted secret
        store. A reader consulting only the first hides every API-key provider
        the owner configured by hand; only the second hides every OAuth login,
        which on a current install is most of them.

        The plaintext ``credentials.env`` file is NO LONGER READ here (PR2a);
        the provider-class store rows below are the consolidated source it used
        to supplement, so this method's answer is unchanged for any host whose
        keys live in the store or in auth.db.

        Everything else is deliberately IDENTICAL to ``usable_providers``,
        including the oauth-flavour suppression (an active ``radient`` OAuth
        sign-in hides the legacy ``radient-key`` flavour, so one account is not
        offered twice), the narrow ``sqlite3.ProgrammingError`` re-raise (a
        connection crossing threads is a caller BUG and must not dress itself as
        the unreadable-store degradation — see that method's note on D18), and
        the ``None`` return for an environment failure, which callers must read
        as "cannot tell", never as "none".
        """
        try:
            stored_rows = self.auth_store.list_credentials(provider=None)
            stored = {row.provider for row in stored_rows}
            oauth_providers = {
                row.provider for row in stored_rows if row.credential_type == "oauth"
            }
        except sqlite3.ProgrammingError:
            raise
        except (sqlite3.Error, OSError):
            return None
        # The store's provider-class rows, keyed by env-key NAME to match the
        # rung list ``credential_file_names`` returns. This is the consolidated
        # source; the plaintext file it used to union is gone.
        #
        # Through the controller's OWN root (R4): a controller built against a
        # non-default config root must consult the store under that root, not the
        # HOME-derived default.
        legacy: set[str] = set(stored_provider_env_keys(self.config_dir))
        persisted: set[str] = set()
        for definition in PROVIDER_REGISTRY:
            storage = credential_provider_id(definition.id)
            if definition.store_credentials_as and storage in oauth_providers:
                continue
            if storage in stored:
                persisted.add(definition.id)
                continue
            # ``credential_file_names`` rather than ``env_key_name`` because the
            # latter answers only the plain-string ``env_keys`` form and returns
            # ``None`` for the callable one — ``anthropic`` — so a key written by
            # ``lop credential update ANTHROPIC_API_KEY`` matched no rung here
            # and the phone served an empty sheet to an install the desktop
            # listed 18 models for. It is alias-aware for the same reason
            # ``resolve_env_key`` is: a login flavour declares no key name of its
            # own, but the provider it stores under does, and that name is what
            # the store holds.
            if any(name in legacy for name in credential_file_names(definition.id)):
                persisted.add(definition.id)
        return persisted

    def usage_enabled_providers(self) -> list[str]:
        """Provider ids with a live quota endpoint, sorted.

        "Has an endpoint", not "can reach it" — see :meth:`can_report_usage` for
        the question every UI surface actually asks.
        """
        return sorted(USAGE_PROVIDERS)

    def can_report_usage(self, provider: str) -> bool:
        """Whether the credentials ON HAND can reach ``provider``'s quota endpoint.

        ONE predicate, because there are three surfaces asking it — `/provider`'s
        "reports quota" list, bare `/usage`'s target list and `/usage <provider>`'s
        up-front warning — and any two of them disagreeing is the defect this
        replaces: with only ``ANTHROPIC_API_KEY`` set, `/provider` advertised
        anthropic, bare `/usage` rendered "no usage data", and `/usage anthropic`
        correctly said it needs a login.

        :meth:`is_usable` alone is too coarse here. It answers "is there any
        credential", including one resolved from the environment, but five of the
        eight usage providers are OAuth-only for usage — an API key cannot reach
        their endpoint at all, so an ``ANTHROPIC_API_KEY`` user holds a credential
        that runs the model and cannot read the quota. The API-key route has to
        exist (``usage_kinds(p)[1]``) before an env/API key counts.
        """
        if not self.is_usable(provider):
            return False
        return self.has_any_credential(provider) or usage_kinds(provider)[1]

    def usage_reportable_providers(self) -> list[str]:
        """Sorted provider ids that both have an endpoint and a way to reach it."""
        return [p for p in self.usage_enabled_providers() if self.can_report_usage(p)]

    # -- model -------------------------------------------------------------
    def resolve_model(self, provider: str, model_id: str) -> ModelSpec:
        """Build a ModelSpec for a provider/model pair (raises on unknown
        provider/hosting). Used by ``/model <provider>/<id>``."""
        return build_model_spec(provider, model_id)

    # -- login / logout ----------------------------------------------------
    def _default_callbacks(self, definition: ProviderDefinition) -> LoginCallbacks:
        """Fall back to the CLI's terminal print/input callbacks."""
        from local_operator.providers.auth_cli import _callbacks_interactive

        return _callbacks_interactive(definition)

    async def _configure_local(self, definition: ProviderDefinition) -> str:
        import httpx

        from local_operator.config import ConfigManager
        from local_operator.paths import config_dir
        from local_operator.providers.local import normalize_base_url, resolve_base_url
        from local_operator.providers.local_discovery import discover_local
        from local_operator.providers.oauth.callback_server import LoginCancelledError

        callbacks = (self._login_callbacks or self._default_callbacks)(definition)
        setup_input = callbacks.on_setup_input
        if setup_input is None:
            raise ValueError("Configure this server in the app with /login, or use /settings.")

        async def prompt(label: str, default: str, secret: bool) -> str | None:
            import inspect

            value = setup_input(label, default, secret)
            return await value if inspect.isawaitable(value) else value

        manager = ConfigManager(config_dir())
        previous = resolve_base_url(definition.id)
        endpoint_input = await prompt(
            "Server URL", previous if definition.id != "openai-compatible" else "", False
        )
        if endpoint_input is None:
            raise LoginCancelledError("Setup cancelled; previous configuration kept.")
        endpoint = normalize_base_url(endpoint_input)
        token = await prompt("API token (optional)", "", True)
        if token is None:
            raise LoginCancelledError("Setup cancelled; previous configuration kept.")
        clear_token = token == "-"
        if not token and endpoint == previous:
            token = await self.auth_store.get_api_key(definition.id) or ""
        elif token == "-":
            token = ""
        if callbacks.on_progress:
            callbacks.on_progress(f"Checking {endpoint} and model list…")

        def probe():
            with httpx.Client() as client:
                return discover_local(definition.id, endpoint, token or None, client, 5.0)

        manual = False
        try:
            models = await asyncio.to_thread(probe)
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {401, 403}:
                raise ValueError(
                    "Server rejected the token. Check its authentication settings "
                    "and try /login again; nothing changed."
                ) from None
            if error.response.status_code in {404, 405}:
                models, manual = [], True
            else:
                raise ValueError(
                    f"Server returned HTTP {error.response.status_code}. "
                    "Check the API root and server logs; nothing changed."
                ) from None
        except ValueError:
            raise ValueError(
                "Invalid model list. Check that this URL serves an "
                "OpenAI-compatible API, then retry /login; nothing changed."
            ) from None
        except httpx.HTTPError:
            raise ValueError(
                "Cannot reach the server. Start it in its own app, check the URL "
                "and port, then retry /login; nothing changed."
            ) from None
        if not models and not manual:
            raise ValueError(
                "Server is reachable but lists no chat models. Load a chat model "
                "in the server, then retry /login; nothing changed."
            )
        if callbacks.on_progress:
            callbacks.on_progress(
                "Model listing unavailable; enter an exact served model ID."
                if manual
                else "Available models: " + ", ".join(model.id for model in models[:30])
            )
        current = (
            manager.get_config_value("model_name", "")
            if manager.get_config_value("hosting") == definition.id
            else ""
        )
        model_id = await prompt("Model ID", current or (models[0].id if models else ""), False)
        if model_id is None:
            raise LoginCancelledError("Setup cancelled; previous configuration kept.")
        model_id = model_id.strip()
        if not model_id:
            raise ValueError("Enter the exact model ID served by this endpoint; nothing changed.")
        if models and model_id not in {model.id for model in models}:
            raise ValueError(
                "That model is not in the server's list. Refresh /login or choose "
                "one of the listed IDs; nothing changed."
            )
        if callbacks.on_progress:
            callbacks.on_progress(f"Selected model: {model_id}\nServer: {endpoint}")
        confirm = await prompt("Activate and save as default? (yes/no)", "yes", False)
        if confirm is None or confirm.strip().lower() != "yes":
            raise LoginCancelledError("Setup cancelled; previous configuration kept.")
        # All prompts and network reads finish before either durable write. A
        # cancelled or failed probe must never displace a working connection.
        providers = dict(manager.get_config_value("providers", {}) or {})
        settings = dict(providers.get(definition.id, {}) or {})
        settings["base_url"] = endpoint
        providers[definition.id] = settings
        if token:
            self.auth_store.upsert_credential(
                definition.id,
                {"key": token, "source": "login", "type": "api_key", "endpoint": endpoint},
            )
        elif clear_token:
            # Clearing a connection is not erasing its credential history. The
            # store's latest-token policy treats this tombstone as authoritative
            # instead of reviving a previously replaced key on the next lookup.
            for row in self.auth_store.list_credentials(definition.id):
                if row.data.get("endpoint") == endpoint:
                    self.auth_store.disable_credential(row.id, "local-token-cleared")
        manager.update_config(
            {"providers": providers, "hosting": definition.id, "model_name": model_id}
        )
        _invalidate_cached_listing(definition.id)
        return (
            f"Configured {definition.name}: {model_id}. Saved as the default. "
            "Use /model to switch models or /settings for model overrides."
        )

    async def login(
        self,
        provider_id: str,
        *,
        open_browser: Callable[[str], None] | None = None,
        signal: AbortSignal | None = None,
    ) -> str:
        """Run the provider's login flow and report a human summary.

        Must be called with the terminal yielded to the flow (a Textual app
        wraps this in ``App.suspend()``). Returns a message like
        ``Logged in to 'anthropic' (you@example.com).``; raises
        ``ValueError`` / ``LoginError`` on failure.

        ``signal`` makes a PENDING login cancellable. Every layer beneath this
        one already had the machinery -- ``OAuthCallbackFlow._await_code``
        races an abort watcher against its capture futures, the device flow
        polls it, and the registry's lazy thunks forward it -- but nothing ever
        constructed one, so all of it was unreachable and a user whose browser
        never came back had no way out but the 300 s
        ``DEFAULT_TIMEOUT_SECONDS``. This parameter is the missing end of that
        chain, not a new mechanism.

        Forwarded to EVERY login callable, including the paste-a-key providers
        and ``local_setup``, which ignore it: they all accept ``**kwargs``, and
        keeping the call shape uniform is what stops a host having to know
        which flavour of provider it is talking to before it can offer a
        cancel.
        """
        definition = get_provider_definition(provider_id)
        if definition is None:
            raise ValueError(f"Unknown provider: {provider_id}")
        if definition.local_setup:
            # Endpoint setup has no loopback listener and no timeout to escape:
            # it is a sequence of prompts, and its cancel is the host declining
            # one (each `prompt` returning None raises LoginCancelledError
            # already). The signal is accepted and dropped here rather than
            # threaded through, so a caller can pass one to any provider.
            return await self._configure_local(definition)
        if definition.login is None:
            raise ValueError(f"Provider '{provider_id}' has no interactive login.")

        factory = self._login_callbacks or self._default_callbacks
        callbacks = factory(definition)
        # Desktop hosts hand the URL to their own main-process browser opener.
        # Inject it per flow: changing webbrowser globally would steal another
        # session's concurrent login. Device flows only publish their URL.
        options: dict[str, Any] = {}
        if open_browser is not None and definition.callback_port is not None:
            options["open_browser"] = open_browser
        # Passed UNCONDITIONALLY, including when it is None, so there is exactly
        # one call shape. Every login callable in the registry takes it: the
        # lazy OAuth thunks name it explicitly, and `create_api_key_login`'s
        # paste-only login swallows it through `**_kwargs`.
        result = await definition.login(callbacks, signal=signal, **options)

        storage = definition.store_credentials_as or provider_id
        if isinstance(result, str):
            if result:
                self.auth_store.upsert_credential(
                    storage, {"key": result, "source": "login", "type": "api_key"}
                )
                _invalidate_cached_listing(storage)
                return f"Stored API key for '{storage}'."
            return f"Login for '{storage}' produced no key; nothing stored."

        result.setdefault("authorized_at", int(time.time() * 1000))
        self.auth_store.upsert_credential(storage, result)
        # The new credential may list DIFFERENT models than the one it replaced
        # -- a different account, a different plan, or a catalogue listed
        # anonymously before there was a credential at all. No TTL can observe
        # that, so the login event has to say so itself. Same hook as the CLI's
        # ``run_login``; the TUI's ``/login`` arrives here.
        _invalidate_cached_listing(storage)
        # The cached usage row is wrong for the same reason and one more: a
        # completed login is positive evidence this account's grant is alive,
        # which directly contradicts any ``credential_invalid`` verdict the row
        # still carries. Re-authenticating an ALREADY-stored OAUTH account keeps
        # the account fingerprint (and so the cache key) identical, so nothing
        # else in the system observes this event and only this drop clears the
        # verdict -- see ``UsageCacheStore.invalidate``. An api_key re-login
        # dedupes on nothing and inserts a second row, so there the key MOVES
        # and the drop is redundant rather than load-bearing (the orphaned row
        # is never served and the new key misses to a live fetch). The
        # asymmetry is spelled out in full on ``auth_cli._invalidate_cached_usage``.
        self.invalidate_cached_usage(storage)
        identity = result.get("email") or result.get("account_id") or result.get("org_name") or ""
        suffix = f" ({identity})" if identity else ""
        msg = f"Logged in to '{storage}'{suffix}."
        if result.get("grant_note"):
            msg += f" Note: {result['grant_note']}"
        return msg

    async def logout(self, provider_id: str) -> str:
        """Remove stored credentials (alias + storage id) for a provider."""
        definition = get_provider_definition(provider_id)
        if definition is None:
            raise ValueError(f"Unknown provider: {provider_id}")
        targets = {provider_id, definition.store_credentials_as or provider_id}
        removed = 0
        for target in sorted(targets):
            removed += self.auth_store.delete_credentials_for_provider(
                target, disabled_cause="logged-out"
            )
        if removed == 0:
            raise ValueError(f"No stored credentials for '{provider_id}'.")
        # Symmetrical with login: a catalogue fetched under the credential just
        # removed must not decide what the NEXT credential can select. One call
        # per STORAGE id: alias and storage id (``zai-oauth``/``zai``) resolve
        # to the same document set, so iterating both would glob twice.
        for storage_id in sorted({credential_provider_id(t) for t in targets}):
            _invalidate_cached_listing(storage_id)
        return f"Removed {removed} credential(s) for '{provider_id}'."

    # -- usage -------------------------------------------------------------
    async def fetch_usage(
        self,
        provider_ids: list[str] | None = None,
        *,
        force_refresh: bool = False,
    ) -> list[UsageReport]:
        """Fetch normalized usage reports for the requested (or all
        report-able) providers. Never raises: a provider with no reachable
        credential or endpoint is simply absent from the result, and one
        malformed provider never aborts the others.

        Two accelerators sit in front of the network (see
        :mod:`local_operator.providers.usage_cache`):

        - A **shared on-disk cache** keyed per provider + account set, so a
          fresh entry answers with no network at all — and because it is shared
          across every lop session on this machine, one session's refresh is
          every session's answer.
        - **Parallel fan-out** across providers. The old loop awaited each
          provider in turn, so the panel waited for the SUM of the round trips;
          now it waits for the slowest one.

        ``force_refresh`` bypasses the fresh-cache check (the panel's ``r``)
        but still writes its result back, so the next read — in this session or
        any other — is instant.
        """
        targets = provider_ids or []
        if not targets:
            # The SAME predicate `/provider` and `/usage <provider>` use. An env key
            # counts (it is the tier the stream cascade resolves), but only where an
            # API-key route to the quota endpoint exists — see
            # :meth:`can_report_usage` for why each half is needed.
            targets = self.usage_reportable_providers()
        # De-duplicate aliases that share a storage id (openai vs
        # openai-device; xai vs xai-oauth) so one request/one report per row.
        targets = self._dedupe_targets(targets)
        if not targets:
            return []
        reports: list[UsageReport] = []
        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *(
                    self._fetch_provider_cached(client, provider, force_refresh)
                    for provider in targets
                ),
                return_exceptions=True,
            )
        for result in results:
            # Isolate one broken provider: an exception here drops that row
            # rather than aborting the whole report (the old per-provider try).
            if isinstance(result, BaseException):
                continue
            reports.extend(result)
        return reports

    # -- usage cache plumbing ------------------------------------------------

    def close(self) -> None:
        """Release the shared usage cache handle (idempotent, never raises)."""
        if self._usage_cache is not None:
            try:
                self._usage_cache.close()
            except Exception:  # noqa: BLE001 — teardown, never fatal
                pass
            self._usage_cache = None

    def usage_cache_age_ms(self, provider: str) -> int | None:
        """Milliseconds since the cached usage row for ``provider`` was fetched.

        ``None`` when there is no cached row for the provider's CURRENT account
        set (never fetched, or the account set changed since). This is the
        question the TUI's background warmer asks before deciding whether to
        spend a refresh: a warm row means `/usage` will answer from disk, so
        the warmer only fires when the row is missing or going stale.

        Synchronous and cheap (one indexed SQLite read), safe to call from an
        interval callback.
        """
        cache = self._usage_cache_store()
        if cache is None:
            return None
        key = self._usage_cache_key(provider)
        fetched_at = cache.fetched_at_ms(key)
        if fetched_at is None or fetched_at <= 0:
            return None
        now_ms = int(time.time() * 1000)
        return max(0, now_ms - fetched_at)

    def cached_usage_reports(self, provider: str | None = None) -> list[UsageReport]:
        """The cached usage reports for ``provider`` (or all providers), any age.

        The panel's instant-open half: when a row exists, `/usage` can paint it
        immediately (its age stated in the title) while the fetch worker runs in
        the background to confirm or replace it. Reads the shared cache only —
        never crosses the network — so it is safe to call synchronously on the
        keystroke that opens the panel.
        """
        cache = self._usage_cache_store()
        if cache is None:
            return []
        targets = (
            [provider] if provider else self._dedupe_targets(self.usage_reportable_providers())
        )
        reports: list[UsageReport] = []
        for target in targets:
            key = self._usage_cache_key(target)
            try:
                cached = cache.get(key, include_expired=True)
            except Exception:  # noqa: BLE001 — a bad read is an empty open
                cached = None
            if cached:
                reports.extend(cached)
        return reports

    def invalidate_cached_usage(self, storage_id: str) -> None:
        """Best-effort drop of ``storage_id``'s cached usage row after a login.

        Never raises, for the reason ``_invalidate_cached_listing`` gives: a
        login that actually succeeded must not be reported as failed because a
        cache write went wrong afterwards. The next fetch re-derives the state
        from a live refresh either way; this only removes the stale answer that
        would otherwise be served ahead of it.

        Public because the shell ``local-operator login`` (``auth_cli.run_login``)
        has to drop the same row: it writes the credential through a bare
        ``AuthStore`` with no controller of its own, and a ``sign-in expired``
        verdict that ``/login`` clears but a shell login leaves standing for the
        cache TTL is the R1 asymmetry in its other entry point (#618 R11). The
        key is derived HERE rather than re-spelled by the CLI because it folds
        in the storage-id aliasing and the account fingerprint
        (:meth:`_usage_cache_key`); a second derivation would drift.
        """
        try:
            cache = self._usage_cache_store()
            if cache is not None:
                cache.invalidate(self._usage_cache_key(storage_id))
        except Exception:  # noqa: BLE001 — a stale row is not worth failing a login
            logger.debug("usage cache: post-login invalidate failed", exc_info=True)

    def _usage_cache_store(self) -> UsageCacheStore | None:
        """The shared usage cache, built lazily on first use.

        Lazy so a host that never asks for usage pays nothing. Injectable via
        the constructor so tests can aim it at a temp file instead of the real
        ``~/.local-operator/usage_cache.db``.
        """
        if self._usage_cache is None:
            try:
                self._usage_cache = UsageCacheStore()
            except Exception:  # noqa: BLE001 — no cache = live fetch, never fatal
                return None
        return self._usage_cache

    def _storage_id(self, provider: str) -> str:
        """The credential storage id for ``provider`` (aliases collapse).

        ``openai-device`` logs in under ``openai``, ``xai-oauth`` under ``xai``;
        the registry's ``store_credentials_as`` says so. Cache keys must follow
        the SAME aliasing or the two spellings of one account would hold two
        rows — one of them permanently stale.
        """
        definition = get_provider_definition(provider)
        return (definition.store_credentials_as or provider) if definition else provider

    def _usage_cache_key(self, provider: str) -> str:
        """The shared-cache key for ``provider``'s current account set."""
        return provider_cache_key(self._storage_id(provider), self._account_fingerprint(provider))

    def _account_fingerprint(self, provider: str) -> str:
        """A synchronous fingerprint of WHICH accounts ``provider`` would fetch.

        Built from the stored credential rows (identity keys for OAuth, a hash
        for API keys) plus any env key — no OAuth refresh, no network. Folding
        the account set into the cache key is what makes login/logout
        self-invalidating: the moment the set changes, the key changes and the
        stale row stops matching. See :mod:`usage_cache` for why the key names
        the account rather than the (rotating) access token.
        """
        storage = self._storage_id(provider)
        parts: list[str] = []
        # Cascade tiers 1/2 (runtime `--api-key`, models.yml pointer) can WIN
        # the fetch — `get_api_key` resolves them ahead of every stored row —
        # so they belong in the fingerprint too, or two sessions running on
        # different override keys would share one cache row and read each
        # other's numbers. `override_keys` is AuthStore's public accessor for
        # exactly this question; guarded because the narrow store protocol
        # does not require it, and a store without it has no overrides to name.
        override_keys = getattr(self.auth_store, "override_keys", None)
        if callable(override_keys):
            try:
                secrets = override_keys(provider)
            except Exception:  # noqa: BLE001 — overrides are optional context
                secrets = ()
            if isinstance(secrets, (list, tuple)):
                for secret in secrets:
                    parts.append(fingerprint_secret(str(secret)))
        try:
            rows = self.auth_store.list_credentials(storage)
        except Exception:  # noqa: BLE001 — an unreadable store fingerprints empty
            rows = []
        for row in rows:
            if getattr(row, "credential_type", None) == "oauth":
                identity = getattr(row, "identity_key", None)
                parts.append(identity or f"cred:{getattr(row, 'id', 0)}")
            else:
                data = getattr(row, "data", None) or {}
                key = data.get("key") if isinstance(data, dict) else None
                if key:
                    parts.append(fingerprint_secret(str(key)))
        try:
            env_key = resolve_env_key(storage)
        except Exception:  # noqa: BLE001
            env_key = None
        if env_key:
            parts.append(fingerprint_secret(env_key))
        return fingerprint_accounts(parts)

    async def _qwencloud_console_creds(
        self, provider: str
    ) -> tuple[dict[str, Any] | None, str | None]:
        """The console session cookie for QwenCloud Token Plan, and why not.

        Returns ``(creds, note)``. ``note`` is non-None only for a state the
        user can ACT ON, and it becomes :attr:`UsageReport.notes` so the
        provider block stays on screen with a reason instead of vanishing --
        the same move ``usage.py``'s xAI path makes for an answering account
        with no metered window. A silent ``None`` for a locked store would be
        the "plausible degraded state" failure this class already warns about
        at :class:`ProviderController`'s own 268-278.

        A SEPARATE namespace (``qwencloud-console``) rather than a row under
        ``alibaba-token-plan``, because :meth:`has_any_credential` matches on
        the provider column with no type or field filter: a row there would
        flip ``is_usable`` and then ``can_report_usage``, and local-operator
        would believe it could CHAT through alibaba-token-plan on a read-only
        console cookie. It would also join the cache fingerprint and sit one
        field name away from the API-key cascade's ``data["key"]`` read.

        ``qwencloud-console`` is deliberately NOT in ``PROVIDER_REGISTRY``;
        it is a row namespace, following the ``mcp-oauth`` precedent.

        The VALUE lives in the encrypted secret store; only its metadata is in
        ``auth.db``. The retrieval is guarded on ``store_path().exists()``
        because ``retrieve_secret`` against a base with no store SPAWNS A
        BROKER DAEMON before failing -- measured, with a stray process left
        behind, on the very common host that has never run ``lop secret set``.

        **Async for the retrieval hop, not for the arithmetic.** The blocking
        work is off-loaded to :func:`asyncio.to_thread` below; the overhead of
        the hop measured at noise (-0.1 ms against a warm store), and what it
        buys is the 10 s case. A broker that will not start makes
        ``retrieve_secret`` poll to ``STARTUP_TIMEOUT_S`` twice, and this runs
        inside the ``asyncio.gather`` that paints the panel on an auto-refresh
        the user never asked for -- so without the hop the whole TUI is frozen
        for it, which is the shape of blocking call ``client.py``'s own #401
        note records this codebase already freezing the TUI with. The freeze
        and the note are not alternatives: the hardened-locked path now
        RETURNS a note, so a user who sees it waited the full stall first.
        """
        if credential_provider_id(provider) != "alibaba-token-plan":
            return None, None
        try:
            rows = self.auth_store.list_credentials(QWENCLOUD_CONSOLE_PROVIDER)
        except Exception:  # noqa: BLE001 — an unreadable store has no ticket
            # The METADATA store, not the secret store. This is the hint path,
            # not the security gate: no row really does mean no ticket here.
            return None, None
        for row in rows:
            data = getattr(row, "data", None)
            if not isinstance(data, dict):
                continue
            if data.get("ticket"):
                # Pre-migration row with the value still in plaintext. A user
                # who has not run `lop qwencloud-ticket migrate` keeps a
                # working /usage; do not break them to make a point.
                return dict(data), None
            if not data.get("secret_name"):
                continue
            # Imported INSIDE the function, not at module scope.
            # `qwencloud_console.py` is stdlib-only so it is cycle-safe for
            # this module, and `access.py` uses the same idiom for the same
            # reason. A module-scope import here reintroduces the cycle the
            # stdlib-only import list exists to protect.
            from local_operator.secrets import access
            from local_operator.secrets.client import BrokerDenied, BrokerLocked
            from local_operator.secrets.errors import (
                BrokerIncompatible,
                InsecurePermissions,
                SecretNotFound,
                SecretStoreError,
            )
            from local_operator.secrets.keys import store_path

            if not store_path(None).exists():
                # No secret store on this host. Returning here is what keeps a
                # plain TUI user from spawning a daemon they have no use for
                # on every refresh.
                return None, None
            try:
                # The one blocking call on this path, and the reason the method
                # is async at all -- see the docstring. `store_path().exists()`
                # above stays on the loop: it is a single stat, and hopping for
                # it would cost more than it saves.
                value = await asyncio.to_thread(
                    access.retrieve_secret, QWENCLOUD_TICKET_SECRET_NAME, None
                )
            except (BrokerDenied, BrokerLocked):
                # Caught BEFORE SecretStoreError: both subclass it. The store's
                # own message on THIS path is the raw wire text ("no lop
                # session is registered with the broker") -- the re-wording
                # that names `lop secret unlock` lives in `master_key_for` and
                # does not fire on the retrieval path. So the actionable
                # message is ours.
                return None, QWENCLOUD_TICKET_LOCKED_NOTE
            except SecretNotFound:
                return None, QWENCLOUD_TICKET_ORPHAN_NOTE
            except InsecurePermissions:
                # The store is readable by another account. Its own docstring
                # calls this "a condition to stop on, not one to quietly
                # repair -- the exposure already happened and the operator
                # needs to know", so swallowing it into a vanished window is
                # precisely the prohibited behaviour. `lop secret status`
                # prints the offending path and its `chmod 0600` fix; verified
                # by running it against a 0644 throwaway store rather than
                # assumed.
                return None, QWENCLOUD_TICKET_EXPOSED_NOTE
            except BrokerIncompatible:
                # A daemon left running across a runtime update. This class
                # exists BECAUSE collapsing "live but unusable" into
                # "unreachable" silently disarmed a safety property (its own
                # round-4 Q4 note), and answering a silent None here repeats
                # that collapse one layer up. It is also self-inflicted and
                # trivially fixable, which is what makes it worth a row.
                return None, QWENCLOUD_TICKET_BROKER_NOTE
            except (SecretStoreError, OSError):
                # Deliberately still silent, and deliberately NOT given a note.
                # The two classes that dominate this clause are
                # `BrokerUnavailable` and `SecretCorrupt`, and neither has a
                # remedy this note could name:
                #
                # - `BrokerUnavailable` is transient by construction -- the
                #   daemon starts lazily and exits on its own idle timer, so
                #   "nothing answered" is usually a race the next auto-refresh
                #   wins. A note telling the user to act on it would ask them
                #   to fix something that has already fixed itself.
                # - `SecretCorrupt` has no repair verb at all. Measured against
                #   a store with one tampered record: `set` raises
                #   `SecretExists`, `update` and even `lop secret rm` raise
                #   `SecretCorrupt` and exit 2, and the record survives. Naming
                #   any of them would be a remedy that fails when followed,
                #   which `_fit_status_note` calls worse than no command.
                #
                # A note is a promise the user can act; where there is nothing
                # to run, silence is the honest answer.
                return None, None
            try:
                ticket = value.decode()
            except UnicodeDecodeError:
                # A value that cannot decode was never sendable as a `cookie:`
                # header, so it is the unreadable case, not a locked one.
                return None, None
            creds = dict(data)
            creds.pop("secret_name", None)
            creds.pop("length", None)
            creds["ticket"] = ticket
            return creds, None
        return None, None

    def _expected_oauth_identities(self, provider: str) -> list[str]:
        """Stored OAuth identities for ``provider``, including refresh-failed.

        The expected set for ``/usage`` is the logged-in rows, not the subset
        that happened to mint a bearer this cycle. A sibling enumerator
        (:meth:`AuthStore.list_oauth_identities`) is preferred because it
        never calls ``_ensure_oauth_fresh``. An empty return from that
        sibling is authoritative — a runtime/config override short-circuits
        to ``[]`` so stored identity does not apply, and falling through to
        ``list_credentials`` (which does not honour overrides) would paint
        those OAuth emails as last-known stubs and skip the API-key route
        the session is actually using. ``list_credentials`` is only the
        shim for a store protocol that has not grown the sibling.
        """
        storage = self._storage_id(provider)
        try:
            enumerator = self.auth_store.list_oauth_identities
        except AttributeError:
            enumerator = None
        if enumerator is not None:
            try:
                accesses = enumerator(storage)
            except Exception:  # noqa: BLE001 — identities are labels, never fatal
                return []
            identities: list[str] = []
            for access in accesses:
                label = (
                    getattr(access, "email", None)
                    or getattr(access, "account_id", None)
                    or (
                        f"cred:{getattr(access, 'credential_id', 0)}"
                        if getattr(access, "credential_id", 0)
                        else None
                    )
                )
                if label:
                    identities.append(str(label))
            return identities
        try:
            rows = self.auth_store.list_credentials(storage)
        except Exception:  # noqa: BLE001
            return []
        labels: list[str] = []
        for row in rows:
            if getattr(row, "credential_type", None) != "oauth":
                continue
            data = getattr(row, "data", None) or {}
            label = None
            if isinstance(data, dict):
                label = data.get("email") or data.get("account_id")
            label = label or getattr(row, "identity_key", None) or f"cred:{getattr(row, 'id', 0)}"
            if label and not str(label).startswith("oauth:"):
                labels.append(str(label))
            elif getattr(row, "id", 0):
                labels.append(f"cred:{row.id}")
        return labels

    @staticmethod
    def _account_in_backoff(previous: UsageReport | None, now_ms: int, *, force: bool) -> bool:
        """Whether this account should be served from last-good, not re-probed.

        ``r`` always retries. An unavailable account is skipped only until its
        scheduled retry (``next_probe_at_ms``, set to a jittered
        :data:`USAGE_UNAVAILABLE_RETRY_MS` when the ceiling trips), then probed
        like any other — so a provider that recovers is discovered without
        ``r``. Otherwise the per-account ``next_probe_at_ms`` is the gate;
        siblings that are fresh still refresh on the same provider lease.
        """
        if force or previous is None:
            return False
        # NOTE: ``credential_invalid`` deliberately does NOT gate here, and
        # that is load-bearing rather than an omission.
        #
        # Skipping the cycle for a dead grant looks like it saves the retry
        # budget, and measurably saves nothing: ``list_oauth_accesses`` is
        # awaited before this loop and refreshes EVERY row, so the refresh
        # POST is already spent by the time the gate is consulted, and the
        # usage probe is short-circuited separately by the
        # ``access.credential_invalid`` branch, so no usage request is made
        # either. Measured on three consecutive cycles with and without the
        # skip: 1 refresh POST and 0 usage probes, identically.
        #
        # What the skip did buy was a one-way latch. The only writer that
        # clears the flag is ``_mark_account_success``, which is reached only
        # through a fetch this gate prevented -- so after the user followed
        # the panel's own ``/login`` advice, every automatic poll re-rendered
        # ``sign-in expired`` against a freshly-minted valid grant, and only
        # ``r`` could break the loop. That is this defect with its polarity
        # reversed: a permanent message that the user cannot act their way
        # out of. The verdict must be re-derived from the live refresh each
        # cycle, never remembered as a terminal state.
        # Only a FAILURE cool-down skips the probe. An unavailable account is
        # no longer latched forever: ``_mark_account_failure`` schedules a
        # jittered ``USAGE_UNAVAILABLE_RETRY_MS`` retry when the ceiling trips,
        # so the account falls through to the same ``next_probe_at_ms`` gate as
        # a transient miss and is probed once that time passes. A successful
        # account leaves ``next_probe_at_ms`` unset so a sibling's shorter
        # backoff can expire the provider row without freezing the healthy
        # logins.
        if previous.consecutive_failures <= 0:
            return False
        nxt = previous.next_probe_at_ms
        return nxt is not None and nxt > now_ms

    def _reset_account_for_force(self, report: UsageReport) -> UsageReport:
        """``r`` clears the failure streak so a maxed-out account is retried.

        ``credential_invalid`` is deliberately NOT cleared here. The streak
        and the unavailable ceiling are guesses about a provider that may have
        recovered, so ``r`` is right to drop them; a dead grant is a verdict
        the IdP returned, and clearing it optimistically would blank the one
        line telling the user to re-login until the fetch re-derived it.
        ``_mark_account_success`` clears it when a bearer actually works.
        """
        report.consecutive_failures = 0
        report.usage_unavailable = False
        report.next_probe_at_ms = None
        return report

    def _mark_account_success(self, report: UsageReport, now_ms: int) -> UsageReport:
        """A live 200: clear the failure streak and schedule the next probe."""
        report.consecutive_failures = 0
        report.usage_unavailable = False
        report.next_probe_at_ms = None
        # A 200 proves the grant minted a working bearer, so whatever the
        # cached row said about it is stale by definition. Clearing here is
        # what makes the state self-healing after the user re-logs in.
        report.credential_invalid = False
        return report

    def _mark_account_invalid(
        self,
        previous: UsageReport | None,
        *,
        provider: str,
        identity: str,
        now_ms: int,
    ) -> UsageReport:
        """The grant is dead: state it, and stop spending retries on it.

        Deliberately NOT ``_mark_account_failure`` with an extra flag. That
        path exists to decide when a run of transient misses has gone on long
        enough to stop trusting the numbers, and every part of it is wrong
        here: the failure streak measures an outage's length, the exponential
        backoff schedules a retry that cannot succeed, and
        ``usage_unavailable`` renders as ``usage unavailable - last known 2d
        ago``, which tells the user to wait when they need to act.

        So the streak is left untouched and ``next_probe_at_ms`` stays None.
        The account is not in a cool-down -- there is simply nothing to
        re-probe until the credential is replaced, and
        ``_account_in_backoff`` skips it on that flag alone. Last-known limits
        stay on the report: they remain the last true reading of a login the
        user still owns.
        """
        report = (
            previous
            if previous is not None
            else UsageReport(provider=provider, fetched_at=now_ms, identity=identity)
        )
        report.credential_invalid = True
        report.usage_unavailable = False
        report.next_probe_at_ms = None
        return report

    def _mark_account_failure(
        self,
        previous: UsageReport | None,
        *,
        provider: str,
        identity: str,
        now_ms: int,
    ) -> UsageReport:
        """Keep last-good (if any) and bump this account's consecutive misses.

        After :data:`USAGE_ACCOUNT_MAX_FAILURES` the account is marked
        usage-unavailable and its next probe is scheduled on the jittered
        :data:`USAGE_UNAVAILABLE_RETRY_MS` cadence rather than latched dark:
        a provider whose quota resets or whose incident closes is discovered
        on its own, and the background warmer retries it every ~10 min instead
        of never. Last-good numbers, when they exist, stay on the report so
        the operator can still see they are logged in and what the last
        successful check said.

        A transient miss also CLEARS any dead-grant verdict, which is not the
        contradiction it first looks like. Reaching here means the store
        handed back a usable bearer and the usage endpoint was what failed --
        so the grant refreshed, which is direct evidence it is alive. Leaving
        the flag set let one endpoint blip re-latch a healthy credential as
        ``sign-in expired``, telling the user to re-authenticate when nothing
        was wrong with their login. A cycle that genuinely still sees
        ``invalid_grant`` sets it again on the spot.
        """
        failures = (previous.consecutive_failures if previous is not None else 0) + 1
        unavailable = failures >= USAGE_ACCOUNT_MAX_FAILURES
        if previous is not None:
            report = previous
        else:
            report = UsageReport(provider=provider, fetched_at=now_ms, identity=identity)
        report.consecutive_failures = failures
        report.usage_unavailable = unavailable
        report.credential_invalid = False
        report.next_probe_at_ms = (
            now_ms + self._jittered_unavailable_retry_ms()
            if unavailable
            else now_ms + account_backoff_ms(failures)
        )
        return report

    def _payload_expires_at_ms(self, reports: list[UsageReport], now_ms: int) -> int:
        """When the shared row should go stale so the next due account is probed.

        A mixed payload used to inherit the full 5-minute success TTL, which
        swallowed the 10 s / 20 s / … per-account backoff: the failed login
        sat un-retried until the whole set expired. Expiry is the soonest
        ``next_probe_at_ms`` still in the future. An unavailable account now
        carries a scheduled retry too, so a wholly-latched payload expires at
        the :data:`USAGE_UNAVAILABLE_RETRY_MS` cadence (~10 min) and the
        background warmer re-probes it — which is the point: recovery is
        discoverable without ``r``. The full jittered TTL is the fallback only
        when no account has a future probe scheduled at all.
        """
        soonest: int | None = None
        for report in reports:
            nxt = report.next_probe_at_ms
            if nxt is None or nxt <= now_ms:
                continue
            if soonest is None or nxt < soonest:
                soonest = nxt
        return soonest if soonest is not None else now_ms + self._jittered_ttl_ms()

    def _merge_account_reports(
        self,
        *,
        provider: str,
        expected: list[str],
        live: dict[str, UsageReport],
        previous: dict[str, UsageReport],
        now_ms: int,
        force: bool,
    ) -> list[UsageReport]:
        """Union of this fetch's successes and last-good for everyone else.

        A provider-level ``cache.set`` of only the accounts that succeeded
        this round is the #277 leftover that dropped a 429'd login from
        ``/usage``: the next warm read served a 3-account payload over a
        4-account login set. Expected order is the stored-row order so the
        panel does not reshuffle under the reader.
        """
        merged: list[UsageReport] = []
        seen: set[str] = set()
        for identity in expected:
            seen.add(identity)
            if identity in live:
                merged.append(self._settle_live_report(live[identity], now_ms))
                continue
            prior = previous.get(identity)
            if self._account_in_backoff(prior, now_ms, force=force) and prior is not None:
                # Still inside this account's cool-down (or already
                # unavailable): serve last-good without incrementing. A
                # force refresh never takes this branch.
                merged.append(prior)
                continue
            merged.append(
                self._mark_account_failure(
                    prior, provider=provider, identity=identity, now_ms=now_ms
                )
            )
        # An API-key report (or an identity the store no longer names) still
        # belongs on the panel if it succeeded this fetch; do not invent
        # stubs for leftovers that are no longer logged in.
        for identity, report in live.items():
            if identity in seen:
                continue
            merged.append(self._settle_live_report(report, now_ms))
        return merged

    def _settle_live_report(self, report: UsageReport, now_ms: int) -> UsageReport:
        """Finish a report this cycle produced, honouring a dead-grant verdict.

        Everything in ``live`` used to be a 200 by construction, so the merge
        could call ``_mark_account_success`` on all of it. A dead-grant entry
        is also produced by this cycle (it is a fresh verdict, not last-good)
        but it is the opposite of a success, and passing it through the
        success path would clear the very flag it was created to carry.

        A console-ticket note is the same shape of fresh non-success: it
        carries a scheduled re-probe so the note clears itself once the user
        runs the remedy, and ``_mark_account_success`` would null exactly that.
        """
        if report.credential_invalid:
            return report
        if report.notes and report.next_probe_at_ms is not None and not report.limits:
            return report
        return self._mark_account_success(report, now_ms)

    @staticmethod
    def _jittered_ttl_ms() -> int:
        """Base TTL spread ±25%, so several accounts/providers do not all expire
        into the same refresh window (the per-IP burst that earns a 429)."""
        jitter = USAGE_REPORT_TTL_MS * (random.random() * 0.5 - 0.25)
        return int(USAGE_REPORT_TTL_MS + jitter)

    @staticmethod
    def _jittered_unavailable_retry_ms() -> int:
        """Unavailable-retry cadence spread ±25%, the same way
        :func:`_jittered_ttl_ms` spreads the TTL, so N latched accounts do not
        re-probe in lockstep into the same per-IP 429."""
        jitter = USAGE_UNAVAILABLE_RETRY_MS * (random.random() * 0.5 - 0.25)
        return int(USAGE_UNAVAILABLE_RETRY_MS + jitter)

    async def _fetch_provider_cached(
        self,
        client: httpx.AsyncClient,
        provider: str,
        force_refresh: bool,
    ) -> list[UsageReport]:
        """One provider's reports, cache-first.

        Fast path: a fresh cache entry returns with no network at all. The
        slow path delegates to :meth:`_refresh_provider_usage`, where the
        cross-process lease ensures only one session on the machine actually
        crosses the network for a stale row — every other session serves the
        stale value while that one refreshes. That lease is the coordination;
        no in-process future map is needed on top of it.
        """
        cache = self._usage_cache_store()
        key = ""
        if cache is not None:
            key = self._usage_cache_key(provider)
            if not force_refresh:
                fresh = cache.get(key)
                if fresh is not None:
                    return fresh
        return await self._refresh_provider_usage(client, provider, key, cache, force_refresh)

    async def _refresh_provider_usage(
        self,
        client: httpx.AsyncClient,
        provider: str,
        key: str,
        cache: UsageCacheStore | None,
        force_refresh: bool,
    ) -> list[UsageReport]:
        """Actually cross the network for ``provider``, then settle the cache.

        Cross-process coordination lives here: when the cached row is stale,
        ONE session wins a lease and refreshes while the others serve the stale
        row rather than joining a synchronized fan-out (Anthropic/OpenAI
        rate-limit the usage endpoint per source IP). On failure the last good
        value is served with a short cool-down, so a blip never blanks the
        report.

        **An empty result never overwrites non-empty last-good data.** The
        fetchers signal transport/HTTP failure by returning ``None`` —
        ``_get_json`` swallows ``httpx.HTTPError``, non-200s (including 429)
        and bad JSON — so by the time a result reaches this function, "the
        endpoint is down" and "the account has no quota to report" are the
        same empty list. The one disambiguating fact on hand is history: a
        provider that HAD data a moment ago and reports none now is far more
        likely rate-limited than suddenly quota-less, so the empty answer is
        treated as a failure (last-good kept servable under a short cool-down,
        retried on the next poll). A provider with no history of data — or
        whose last answer was also empty — negative-caches the empty list at
        the full TTL, which is what stops the warmer from re-hitting endpoints
        that legitimately report nothing.
        """
        stale: list[UsageReport] | None = None
        lease_held = False
        #: The empty answer was BELIEVED (no history, or history too old), so
        #: the stale row must not be served over it — see the acceptance branch.
        accepted_empty = False
        if cache is not None and key:
            stale = cache.get(key, include_expired=True)
            # The lease only has something to protect when a stale value exists:
            # the loser serves it while the winner refreshes. With nothing on
            # hand every session must fetch anyway (the pre-cache behaviour).
            if not force_refresh and stale is not None:
                if not cache.try_lease(key):
                    # A peer session owns this refresh; its result lands in the
                    # same shared row. Serve what we have instead of doubling
                    # the fan-out.
                    return stale
                # Held ONLY when try_lease actually granted it: the force path
                # never takes the lease, and releasing one it does not hold
                # would free a concurrent warmer's lease (holder identity is
                # per-process, not per-coroutine).
                lease_held = True
        try:
            try:
                reports = await self._fetch_provider(
                    client, provider, previous=stale or [], force_refresh=force_refresh
                )
            except Exception:  # noqa: BLE001 — isolate a broken provider
                reports = []
            if cache is not None and key:
                if reports:
                    now_ms = int(time.time() * 1000)
                    # The merged payload already carries last-good for
                    # accounts that failed this round. Writing only the live
                    # successes is the #277 leftover that shrank a 4-account
                    # snapshot to 3 the next time one token 429'd.
                    cache.set(
                        key,
                        provider,
                        reports,
                        expires_at_ms=self._payload_expires_at_ms(reports, now_ms),
                    )
                elif stale:
                    # Truthiness, not `is not None`: only a NON-EMPTY history
                    # marks this empty answer as a probable outage. Keep the
                    # last good value servable through a short cool-down —
                    # unless the data is old enough that the "outage" reading
                    # has expired (EMPTY_OVER_DATA_ACCEPT_MS), in which case
                    # the empty answer is accepted and negative-cached so a
                    # genuinely quota-less provider stops being re-fetched on
                    # every cool-down forever.
                    now_ms = int(time.time() * 1000)
                    newest = max((int(r.fetched_at or 0) for r in stale), default=0)
                    if newest and now_ms - newest > EMPTY_OVER_DATA_ACCEPT_MS:
                        cache.set(key, provider, [], expires_at_ms=now_ms + self._jittered_ttl_ms())
                        accepted_empty = True
                    else:
                        cache.write_failure(key, provider)
                else:
                    # No history of data (or an empty one): negative-cache so
                    # the warmer stops re-hitting an endpoint that reports
                    # nothing. `r` (force_refresh) still bypasses this row.
                    now_ms = int(time.time() * 1000)
                    cache.set(key, provider, [], expires_at_ms=now_ms + self._jittered_ttl_ms())
                    accepted_empty = True
        finally:
            if lease_held and cache is not None and key:
                cache.release_lease(key)
        if reports:
            return reports
        if stale is not None and not accepted_empty:
            # A forced refresh that failed still shows the last good numbers
            # (their age is stated in the panel) rather than an empty card. An
            # ACCEPTED empty answer is not papered over with old data, though —
            # the cache just recorded "this provider reports nothing" and the
            # caller should say the same.
            return stale
        return []

    async def _fetch_provider(
        self,
        client: httpx.AsyncClient,
        provider: str,
        *,
        previous: list[UsageReport] | None = None,
        force_refresh: bool = False,
    ) -> list[UsageReport]:
        """Every logged-in account's usage for one provider.

        Quota is per ACCOUNT, so a provider with two logins has two answers.
        Asking :meth:`AuthStore.get_oauth_access` produced one of them, chosen
        by a round-robin that also moved between refreshes: a user with two
        Anthropic accounts saw a single block and could not tell which login it
        described, or that a second one was missing entirely.

        The expected set is the *stored* OAuth rows (blocked included;
        refresh-failed included). A live 200 replaces that account's last-good;
        a 429 / ``None`` / exception keeps the previous numbers and increments
        that account's failure count. The written cache payload is the union,
        never "whoever answered this round" — that shrink is how one 429
        dropped a fourth Anthropic login from ``/usage``.

        The API-key route stays a single report, and is only reached when no
        OAuth identity is stored. An API key is not an identity — the
        cascade's env/config tiers resolve one secret per provider — so fanning
        out there would report the same numbers twice.
        """
        if not usage_supported(provider):
            return []
        now_ms = int(time.time() * 1000)
        expected = self._expected_oauth_identities(provider)
        previous_by_id = {
            report_identity_key(report): report
            for report in (previous or [])
            if report_identity_key(report)
        }
        if force_refresh:
            # ``r`` retries every expected account, including ones that had
            # already hit the unavailable ceiling. Reset first so the
            # backoff gate does not skip them.
            previous_by_id = {
                key: self._reset_account_for_force(report) for key, report in previous_by_id.items()
            }
        live: dict[str, UsageReport] = {}
        if expected:
            # Bearer mint is still list_oauth_accesses (routing contract
            # unchanged). Identities with no bearer this cycle stay in
            # ``expected`` and fall through to last-good / unavailable.
            accesses_by_id: dict[str, Any] = {}
            try:
                accesses = await self.auth_store.list_oauth_accesses(provider)
            except Exception:  # noqa: BLE001 — no bearer is a per-account miss
                accesses = []
            for access in accesses:
                label = (
                    getattr(access, "email", None)
                    or getattr(access, "account_id", None)
                    or (
                        f"cred:{getattr(access, 'credential_id', 0)}"
                        if getattr(access, "credential_id", 0)
                        else None
                    )
                )
                if label:
                    accesses_by_id[str(label)] = access
            # The console ticket is ONE session for the whole account, not one
            # per login, so it is spent at most once per cycle however many
            # dead grants are expected. Attempting it per identity would send
            # a request each and land a report under each, and the panel
            # flattens limits with no dedup by id -- the same window would
            # render twice. Set on the ATTEMPT, not on success: a failure is a
            # property of the ticket, not of the identity that reached it.
            console_attempted = False
            for identity in expected:
                prior = previous_by_id.get(identity)
                if self._account_in_backoff(prior, now_ms, force=force_refresh):
                    continue
                access = accesses_by_id.get(identity)
                if access is None:
                    # The stored grant minted no bearer this cycle (expired,
                    # no refresh token, or the store omitted it). For QwenCloud
                    # that is the STEADY state, not a transient miss: the row's
                    # `expires` is in the past and neither ProviderDefinition
                    # declares a refresh_token, so `_ensure_oauth_fresh` can
                    # never revive it. A console session cookie stored
                    # separately can still answer, and this is the only place
                    # it is reachable -- with `expected` non-empty the API-key
                    # route below is dead code for this provider.
                    if console_attempted:
                        continue
                    console, console_note = await self._qwencloud_console_creds(provider)
                    if console is None:
                        # Set on the ATTEMPT, not on success -- the flag's own
                        # comment above says so, and the cost of not doing it
                        # is not one wasted call. A locked store whose broker
                        # cannot be started makes `retrieve_secret` poll to
                        # `STARTUP_TIMEOUT_S` twice (client.py:71,254-259):
                        # measured at 10 s per call, so leaving the flag False
                        # re-runs that for EVERY expected identity -- three
                        # dead grants froze the event loop for a measured 30 s
                        # inside the `asyncio.gather` that paints the panel.
                        console_attempted = True
                        if console_note:
                            # A state with a remedy keeps the provider block on
                            # screen with the reason, rather than letting it
                            # vanish into the generic empty string.
                            report = UsageReport(
                                provider=provider,
                                fetched_at=now_ms,
                                identity=identity,
                                notes=console_note,
                                # The remedy is a command the user runs in
                                # ANOTHER terminal, so the note has to clear
                                # itself once they have run it. Without a
                                # scheduled probe this payload expires on the
                                # full jittered 5-minute TTL and the note
                                # survives `lop secret unlock` until it lapses
                                # or the user presses `r` -- the "permanent
                                # message the user cannot act their way out of"
                                # polarity defect `_account_in_backoff` below
                                # names, which this codebase has already paid
                                # for once. Measured: re-probing an unlocked
                                # store costs one sub-millisecond retrieval.
                                next_probe_at_ms=now_ms + USAGE_FAILURE_BACKOFF_MS,
                            )
                            live[report_identity_key(report) or identity] = report
                        continue
                    console_attempted = True
                    try:
                        report = await self._fetch_one(
                            client, provider, access=None, extra_creds=console
                        )
                    except Exception:  # noqa: BLE001 — one bad account, not the provider
                        continue
                    if report is None:
                        continue
                    if not report.identity:
                        report.identity = identity
                    live[report_identity_key(report) or identity] = report
                    continue
                if access.credential_invalid:
                    # The store minted no bearer and said why: the grant is
                    # dead. Record that verdict instead of probing with an
                    # empty token, which would fail as a generic 401 and land
                    # this account back in the transient-failure path.
                    #
                    # Direct access, not ``getattr(..., False)``: every access
                    # here is an ``OAuthAccess`` row from ``list_oauth_accesses``,
                    # where the field always exists (default ``False``). The
                    # defensive spelling only ever covered test doubles that
                    # omitted the field, and ``configure.py`` reads the same
                    # field the same way (#618 R12).
                    live[identity] = self._mark_account_invalid(
                        prior, provider=provider, identity=identity, now_ms=now_ms
                    )
                    continue
                try:
                    report = await self._fetch_one(client, provider, access=access)
                except Exception:  # noqa: BLE001 — one bad account, not the provider
                    continue
                if report is None:
                    continue
                if not report.identity:
                    report.identity = identity
                live[report_identity_key(report) or identity] = report
            return self._merge_account_reports(
                provider=provider,
                expected=expected,
                live=live,
                previous=previous_by_id,
                now_ms=now_ms,
                force=force_refresh,
            )
        # The API-key route, reached when the provider has no stored OAuth
        # identity at all. The console ticket is threaded through HERE TOO, not
        # only at the dead-grant point above: spending it only inside
        # `if expected:` made the console window depend on a stored OAuth row
        # being present-but-dead. A user who runs `lop logout
        # alibaba-token-plan` drops that row while the api_key row remains, and
        # `/usage` then rendered NOTHING for a perfectly valid ticket -- the
        # silent-empty-table symptom this feature exists to fix, wearing the
        # costume of a plausible degraded state (controller.py:268-278).
        #
        # `_qwencloud_console_creds` is already guarded on the storage id and
        # returns None for every other provider, so this cannot widen any other
        # provider's fetch; verified by execution, not by reading.
        console, console_note = await self._qwencloud_console_creds(provider)
        try:
            report = await self._fetch_one(client, provider, access=None, extra_creds=console)
        except Exception:  # noqa: BLE001
            return []
        if report is None:
            if console_note:
                # Same short re-probe as the dead-grant route above: the remedy
                # runs in another terminal, so the note must clear itself
                # rather than outlive the fix on the full TTL.
                return [
                    UsageReport(
                        provider=provider,
                        fetched_at=now_ms,
                        notes=console_note,
                        next_probe_at_ms=now_ms + USAGE_FAILURE_BACKOFF_MS,
                    )
                ]
            return []
        report = self._mark_account_success(report, now_ms)
        if console_note and not report.notes:
            report.notes = console_note
            # AFTER `_mark_account_success`, which sets `next_probe_at_ms` to
            # None: attaching the note before it let the success path null the
            # schedule and re-armed the exact staleness defect the other two
            # note branches exist to avoid. Measured on the un-fixed order:
            # `next_probe_at_ms=None` and the payload expiring in 358890 ms --
            # the full jittered TTL, not 10 s -- so a user with a valid
            # `api_key` row plus a locked ticket read "run `lop secret unlock`",
            # ran it, and watched the note outlive the fix by ~6 minutes.
            # `_settle_live_report`'s guard does not cover this path.
            #
            # This report is a genuine 200 whose NOTE is stale-able, which is
            # why it needs the schedule despite being a success: the remedy
            # runs in another terminal and nothing else brings the panel back.
            report.next_probe_at_ms = now_ms + USAGE_FAILURE_BACKOFF_MS
        return [report]

    def _dedupe_targets(self, targets: list[str]) -> list[str]:
        """Keep one id per storage row so alias providers don't double-fetch."""
        seen: set[str] = set()
        ordered: list[str] = []
        for provider in targets:
            storage = credential_provider_id(provider)
            if storage in seen:
                continue
            seen.add(storage)
            ordered.append(provider)
        return ordered

    # -- catalogue ---------------------------------------------------------
    def static_catalogue(self) -> list[CatalogueEntry]:
        """Every model the SHIPPED registry knows, with each provider's auth state.

        Synchronous and I/O-free, which is the point: a picker has to paint on the
        keystroke that opened it. This is the first frame; :meth:`live_catalogue`
        replaces it when the network answers.
        """
        entries: list[CatalogueEntry] = []
        # ONE store read for the whole registry, and one that survives a store
        # that cannot be read: `is_usable` per provider would raise straight
        # through a picker that is about to paint. An unknowable state is
        # reported as connected — see :meth:`usable_providers` for why the
        # degradation runs that way and not toward an empty list.
        usable = self.usable_providers()
        for definition in _chat_providers():
            connected = usable is None or definition.id in usable
            for model_id, info in static_models(definition.id).items():
                entries.append(
                    CatalogueEntry(
                        provider=definition.id,
                        model_id=model_id,
                        label=model_label(definition.id, model_id, info.name or "").full,
                        listing_name=info.name or "",
                        context_window=max(0, info.context_window or 0),
                        default_context_window=info.default_context_window,
                        max_context_window=info.max_context_window,
                        input_price=_price(info.input_price, definition),
                        output_price=_price(info.output_price, definition),
                        connected=connected,
                        aggregated=definition.id in AGGREGATOR_PROVIDERS,
                        time_of_use=info.time_of_use,
                    )
                )
        return entries

    def initial_catalogue(self, *, cache_dir: Any = None) -> list[CatalogueEntry]:
        """First frame catalogue: every provider's rows, from its CACHED listing.

        Synchronous, non-blocking and NETWORK-FREE: the reader is
        :func:`cached_available_models`, which peeks at the document on disk and
        never takes a fetch lease, spawns a revalidation thread or issues a
        request. A picker has to paint on the keystroke that opened it.

        NOT I/O-FREE, and the distinction is measured rather than assumed: the
        frame reads config.yml ONCE (see ``values`` below) and, when a cached
        listing contributed a row the registry does not have, one keyless price
        document (:func:`_price_listing_only_rows`). Both are disk reads with no
        request behind them.

        WHY EVERY PROVIDER AND NOT JUST THE AGGREGATORS. This used to hand a
        direct provider the shipped static registry alone, so the first frame --
        and, for the desktop composer's inline ``/model `` argument list, the
        ONLY frame -- could not offer a model the provider lists and the registry
        does not carry. Aggregators were read through the cache because they ship
        no static rows at all; that was never a special rule, it was just the one
        provider class where the gap was fatal enough to be noticed.

        The policy lives where it already lived, in the reader: on a cold or
        unusable cache :func:`cached_available_models` falls back to the shipped
        rows, so the first frame is field-for-field the one this method always
        painted, and a local provider whose configured endpoint cannot be
        resolved contributes its shipped rows rather than raising (see that
        reader).

        WHAT THIS FRAME DOES NOT READ, stated because the earlier wording here
        claimed otherwise: the reader takes the PLAIN document name
        (``<credential>.listing``), never a credential-scoped one, so
        ``openai.oauth.<hash>.listing`` and ``kimi.oauth.listing`` do not reach
        frame one and the account-scoped prune inside ``_listing_replaces_static``
        is never entered from here. Deepseek is the only provider whose cached
        listing OWNS the set on this path. Sweeping the hashed documents instead
        would be wrong for a frame asked without an account: it could only offer
        whichever account's catalogue it happened to pick, and a first frame has
        no credential in hand to pick with.

        ORDER, which pickers rely on: within a provider the cached listing comes
        first (providers list newest-first) and the registry-only ids it did not
        mention follow; between providers it is the ``_chat_providers()``
        registry order. With nothing cached that is the registry's own dict order.
        """
        entries: list[CatalogueEntry] = []
        usable = self.usable_providers()
        # ONE config read for the whole frame, and only when a local provider is in
        # the registry at all: the endpoint resolution below reads config.yml per
        # provider when it is not handed one, and this method runs on the keystroke
        # that opens a picker -- and, on the desktop, on every keystroke typing a
        # `/model ` argument. Five reads turned the frame from 0.24 ms into 14.5 ms
        # (agent review round 2, R2-2).
        values: Mapping[str, Any] | None = None
        listed: list[tuple[ProviderDefinition, list[DiscoveredModel]]] = []
        for definition in _chat_providers():
            if definition.local_setup and values is None:
                from local_operator.providers.local import config_values

                values = config_values()
            models, _status = cached_available_models(
                definition.id, cache_dir=cache_dir, values=values
            )
            listed.append((definition, models))
        priced = _price_listing_only_rows(listed, cache_dir=cache_dir)
        for definition, models in listed:
            connected = usable is None or definition.id in usable
            for model in models:
                # The price chain's answer, where it has one, for a row the SHIPPED
                # registry does not describe (see `_price_listing_only_rows`).
                model = priced.get((definition.id, model.id), model)
                entries.append(
                    CatalogueEntry(
                        provider=definition.id,
                        model_id=model.id,
                        label=model_label(definition.id, model.id, model.name or "").full,
                        listing_name=model.name or "",
                        context_window=max(0, model.context_window),
                        default_context_window=model.default_context_window,
                        max_context_window=model.max_context_window,
                        input_price=_price(model.input_price, definition, free=model.free),
                        output_price=_price(model.output_price, definition, free=model.free),
                        connected=connected,
                        aggregated=definition.id in AGGREGATOR_PROVIDERS,
                        routed=model.routed,
                        time_of_use=model.time_of_use,
                    )
                )
        return entries

    def entry_for(
        self,
        provider: str,
        model_id: str,
        *,
        spec: ModelSpec | None = None,
    ) -> CatalogueEntry | None:
        """One entry for ``provider``/``model_id``, or ``None`` if unknown here.

        Exists for the model a session is ALREADY RUNNING. A picker must offer
        it whatever the catalogue says, and after an authoritative listing it
        may not be in the catalogue at all: the account's live list is allowed
        to prune bundled ids, so a session started on one of those ids had its
        own model disappear from the list while the status band still named it,
        and typing the id answered "no matching models".

        ``spec`` is the session's already-resolved active model. It matters for
        aggregators: they deliberately have no ENUMERABLE static catalogue, so
        ``static_models("openrouter")`` is empty even for a model the session
        is running. Re-listing here would put synchronous network/cache work
        back on the TUI thread; the spec already carries the name and context
        that session startup resolved. Prices are unknown on ``ModelSpec`` and
        stay unknown rather than being invented.

        Built here rather than in the caller because the normalization is this
        module's job (see :class:`CatalogueEntry`): a caller reaching into the
        registry itself would have to know that a context window of ``-1`` and
        ``0`` both mean unknown, and would spell the price rules a second time.
        ``None`` means neither the registry nor the active spec describes this
        pair, which is a real answer for a model an operator configured by hand.
        """
        definition = get_provider_definition(provider)
        if definition is None:
            return None
        info = static_models(definition.id).get(model_id)
        if info is not None:
            name = info.name or ""
            context_window = max(0, info.context_window or 0)
            input_price = _price(info.input_price, definition)
            output_price = _price(info.output_price, definition)
        elif spec is not None and spec.provider == definition.id and spec.model_id == model_id:
            name = spec.display_name
            context_window = max(0, spec.context_window)
            input_price = -1.0
            output_price = -1.0
        else:
            return None
        default_window = info.default_context_window if info is not None else None
        maximum_window = info.max_context_window if info is not None else None
        if spec is not None and (spec.provider, spec.model_id) == (definition.id, model_id):
            # The serving route beats public registry limits even for a fully
            # described model; the registry still supplies its known prices.
            context_window = max(0, spec.context_window)
            default_window = spec.default_context_window
            maximum_window = spec.max_context_window
        usable = self.usable_providers()
        return CatalogueEntry(
            provider=definition.id,
            model_id=model_id,
            label=model_label(definition.id, model_id, name).full,
            listing_name=name,
            context_window=context_window,
            default_context_window=default_window,
            max_context_window=maximum_window,
            input_price=input_price,
            output_price=output_price,
            connected=usable is None or definition.id in usable,
            aggregated=definition.id in AGGREGATOR_PROVIDERS,
            # From the id alone, because this branch has no listing row to read
            # a price off: it is the fallback for when the listing could not be
            # had at all. That makes it the path a user running `radient/auto`
            # on a cold cache actually takes, so leaving it False here would
            # blank the label on exactly the surface that reported the bug.
            # Provider-scoped: `ollama/auto` reaches this same branch (R1).
            routed=is_meta_route_id(model_id, definition.id),
            # ``info`` is None on the spec-only branch above, and a ``ModelSpec``
            # carries no price table: a spec-derived row has no schedule to
            # state, which is why this reads the registry row rather than any
            # price the row ended up carrying.
            time_of_use=info.time_of_use if info is not None else None,
        )

    async def live_catalogue(
        self,
        *,
        ttl_s: float | None = None,
        providers: Collection[str] | None = None,
    ) -> tuple[list[CatalogueEntry], dict[str, str]]:
        """The catalogue with each provider's LIVE listing layered over the registry.

        Returns ``(entries, statuses)`` where ``statuses`` maps provider id to one
        of discovery's status strings, so a UI can say "stale" or "login
        required" instead of implying the catalogue is complete.

        ``ttl_s`` is the hard TTL passed to discovery; the picker passes
        :data:`PICKER_TTL_S`, and ``None`` keeps discovery's default.

        Only providers with a credential are fetched. An unconnected provider still
        contributes its STATIC models — the question "what would I get if I logged
        in here" is precisely what a user cannot otherwise answer, and it was the
        reason a newly released model was undiscoverable.

        Each provider is isolated: discovery never raises by contract, but a
        credential resolution can (an OAuth refresh against a dead network), and
        one broken provider must not empty the whole list. Provider listings are
        fetched concurrently via :func:`asyncio.gather` so overall latency bounds
        to the single slowest provider rather than the sum of all provider round
        trips.

        ``providers`` narrows the registry to those ids and is how a caller that
        has already decided which accounts it may speak for expresses that. The
        mobile daemon is that caller: it admits only providers with a PERSISTED
        credential, and a listing for anything else would be a network call on
        behalf of an account its user never granted. ``None`` — the default and
        every existing caller — keeps the whole-registry behaviour unchanged,
        including the static rows an unconnected provider contributes so that
        "what would I get if I logged in here" stays answerable in the TUI. An
        empty collection is honoured literally: no provider is fetched and no
        status is reported, which is the correct answer for an owner who has not
        logged in to anything, not a reason to fall back to everything.
        """
        entries: list[CatalogueEntry] = []
        statuses: dict[str, str] = {}
        usable = self.usable_providers()
        # ``_chat_providers()`` filters FIRST, so an explicit ``providers`` set
        # naming a decision-only provider is honoured as "this catalogue request
        # mentions it" and still contributes no rows: the caller is asking for a
        # CHAT catalogue, which is the one thing such a provider can never feed.
        # An empty collection stays literally empty.
        registry = [
            definition
            for definition in _chat_providers()
            if providers is None or definition.id in providers
        ]
        oauth_context: dict[str, dict[str, int]] = {}

        async def _fetch_provider(
            definition: ProviderDefinition,
        ) -> tuple[ProviderDefinition, bool, list[DiscoveredModel], str]:
            # An explicit ``providers`` set is the CALLER's own credential
            # determination and outranks ``usable_providers`` for the ids in it.
            # It has to: ``usable_providers`` has no legacy ``credentials.env``
            # rung, so a provider configured with ``lop credential update`` came
            # back unconnected here, listed ANONYMOUSLY, and the phone's picker
            # then showed it empty — with a credential on disk the whole time.
            # Narrowing without this makes the narrowing itself lose rows.
            connected = (
                definition.id in providers
                if providers is not None
                else (usable is None or definition.id in usable)
            )
            api_key: str | None = None
            is_oauth = False
            account_id: str | None = None
            endpoint: str | None = None
            if definition.local_setup:
                from local_operator.providers.local import resolve_base_url

                try:
                    endpoint = resolve_base_url(definition.id)
                except ValueError:
                    return definition, False, [], "static"
                if not endpoint:
                    return definition, False, [], "static"
                active = self.auth_store.active_local_credential(definition.id, endpoint)
                api_key = str(active.data.get("key") or "") if active else None
            elif connected:
                try:
                    api_key, is_oauth, account_id = await self._listing_credential(definition.id)
                except Exception:  # noqa: BLE001 — one provider's auth is not fatal
                    api_key, is_oauth, account_id = None, False, None
            kwargs: dict[str, Any] = {
                "api_key": api_key,
                "is_oauth": is_oauth,
                "account_id": account_id,
            }
            if ttl_s is not None:
                kwargs["ttl_s"] = ttl_s
            if endpoint is not None:
                kwargs["base_url"] = endpoint
            # Off the event loop: discovery is synchronous httpx by design (it is
            # also called from the CLI and the server), and fetching on the loop
            # would freeze a TUI's repaint. Providers run concurrently.
            models, status = await asyncio.to_thread(available_models, definition.id, **kwargs)
            if definition.id == "openai" and is_oauth:
                # Price enrichment may know the API model, but not this route's
                # limits. Preserve even an unknown OAuth window through it.
                oauth_context[definition.id] = {row.id: row.context_window for row in models}
            if definition.local_setup:
                # Keyless readiness is not evidence a server is running. A
                # stale fallback remains selectable but must not claim a live
                # connection that the refresh just failed to establish.
                connected = status in {"ok", "cached"}
            return definition, connected, models, status

        results = await asyncio.gather(*[_fetch_provider(defn) for defn in registry])
        listed: list[tuple[ProviderDefinition, bool, list[DiscoveredModel]]] = []
        for definition, connected, models, status in results:
            statuses[definition.id] = status
            listed.append((definition, connected, models))

        # Prices for the rows no listing priced, from the same keyless chain the
        # status band resolves through (see :func:`_enrich_prices`). After the
        # listings rather than per provider so the two documents are read ONCE
        # for the whole catalogue, and off-loop for the same reason the listings
        # are: the OpenRouter document is ~120 KB of JSON.
        rows_by_provider = await asyncio.to_thread(
            _enrich_prices, [(definition, models) for definition, _connected, models in listed]
        )
        for definition, connected, _models in listed:
            for model in rows_by_provider[definition.id]:
                if definition.id in oauth_context:
                    model = dataclasses.replace(
                        model, context_window=oauth_context[definition.id].get(model.id, 0)
                    )
                entries.append(
                    CatalogueEntry(
                        provider=definition.id,
                        model_id=model.id,
                        label=model_label(definition.id, model.id, model.name or "").full,
                        listing_name=model.name or "",
                        context_window=max(0, model.context_window),
                        default_context_window=model.default_context_window,
                        max_context_window=model.max_context_window,
                        # ``free`` is consumed HERE and goes no further: a
                        # stated zero survives ``_price`` as ``0.0``, which is
                        # already the entry's way of saying free (an unknown is
                        # ``-1.0``). Carrying the flag onto the entry as well
                        # would be a second spelling of one fact, free to drift.
                        input_price=_price(model.input_price, definition, free=model.free),
                        output_price=_price(model.output_price, definition, free=model.free),
                        connected=connected,
                        aggregated=definition.id in AGGREGATOR_PROVIDERS,
                        routed=model.routed,
                        time_of_use=model.time_of_use,
                    )
                )
        return entries, statuses

    async def _listing_credential(self, provider: str) -> tuple[str | None, bool, str | None]:
        """``(secret, is_oauth, account_id)`` for a model listing call.

        The account id is part of OpenAI's ChatGPT authorization boundary; the
        current Codex catalogue rejects a subscription token without it. Other
        providers ignore the value, while ``is_oauth`` still selects Anthropic's
        bearer header instead of its API-key header.

        The lookup follows ``store_credentials_as``, because the credential a
        login flavour needs is not stored under its own id. ``openai-device``
        and ``xai-oauth`` are login flavours of ``openai`` and ``xai`` -- the
        registry says so, and ``discovery._static_rows`` already follows it for
        the bundled rows -- and the login writes ONE row, under the aliased
        name. Asking ``AuthStore`` for the literal id (its ``WHERE provider = ?``
        is exact) therefore found nothing, so the flavour listed anonymously:
        for ``openai-device`` that meant no OAuth, no account scope, no
        account-scoped catalogue, and the picker offering that logged-in
        ChatGPT account the bundled ``gpt-4o``/``o3`` rows under a second
        prefix -- the very ids this listing exists to stop presenting as
        current.
        """
        credential_id = credential_provider_id(provider)
        access = await self.auth_store.get_oauth_access(credential_id)
        if access is not None and access.kind == "oauth" and access.access_token:
            return access.access_token, True, access.account_id or access.org_id
        if access is not None and access.access_token:
            return access.access_token, False, None
        try:
            stored = await self.auth_store.get_api_key(credential_id)
        except Exception:  # noqa: BLE001 — a refresh failure just means no listing
            stored = None
        # The environment is the last tier of the same cascade the stream uses, so
        # a key set there has to reach the listing too: otherwise the provider a
        # session is ACTUALLY RUNNING ON is the one whose catalogue stays empty.
        return stored or resolve_env_key(credential_id), False, None

    async def _fetch_one(
        self,
        client: httpx.AsyncClient,
        provider: str,
        *,
        access: OAuthAccess | None,
        extra_creds: dict[str, Any] | None = None,
    ) -> UsageReport | None:
        """One report for one account.

        ``access`` is supplied by the caller rather than resolved here: which
        account this report is FOR is the caller's decision now that there can
        be several (see :meth:`_fetch_provider`). ``None`` means "no OAuth
        account" and selects the API-key route.
        """
        if not usage_supported(provider):
            return None
        access_token: str | None = None
        api_key: str | None = None
        account_id: str | None = None
        if access is not None and access.kind == "oauth":
            access_token = access.access_token
            account_id = access.account_id
        elif access is not None and access.access_token:
            api_key = access.access_token
        if access_token is None and api_key is None:
            # Only when no OAuth account was handed in: with one, falling back
            # to the cascade's key would silently report a DIFFERENT account's
            # numbers under this account's identity.
            try:
                api_key = await self.auth_store.get_api_key(provider)
            except Exception:  # noqa: BLE001 — a refresh failure is not fatal here
                api_key = None
        if not access_token and not api_key and not extra_creds:
            # A console credential is neither an access token nor an API key,
            # and is the only thing that can answer for an account whose OAuth
            # grant is dead.
            return None
        # BOTH are handed over, and the dispatcher picks the route each can reach.
        # Passing only one was how the API-key half of a dual-route provider became
        # unreachable: an OAuth token for Kimi went to the coding-plan endpoint, but
        # an API key went nowhere at all because this function had already decided
        # the request was an OAuth one.
        report = await fetch_usage(
            client,
            provider,
            api_key=api_key,
            access_token=access_token,
            account_id=account_id,
            # The raw row lets split-token providers (QwenCloud Token Plan:
            # sk-sp inference key vs. OAuth usage token) spend the right one,
            # where access_token is already the wire-mapped key.
            oauth_creds=access.raw if access is not None and access.kind == "oauth" else None,
            # A credential from outside the provider registry -- the QwenCloud
            # console session cookie. Kept out of `oauth_creds` because that is
            # the raw OAuth row, already read for its `access` field.
            extra_creds=extra_creds,
        )
        if report is not None and not report.identity and access is not None:
            # Whose account this is. The field existed and no fetcher ever set it, so
            # the TUI's annotation for it was unreachable — and it matters most
            # exactly where usage does: an operator with two accounts on one provider
            # needs to know which one the numbers describe.
            report.identity = getattr(access, "email", None) or access.account_id
        return report


def _invalidate_cached_listing(storage_id: str) -> None:
    """Best-effort listing drop after a credential change; never raises.

    An exception here would fail a login that actually succeeded, which is far
    worse than a stale list that the picker's TTL clears within the quarter
    hour anyway. The in-process resolver memo is dropped too: a status-band
    resolution that degraded BEFORE the credential arrived (no key →
    registry-only limits/price) is memoised per TTL bucket and would otherwise
    stay pinned for the rest of the bucket in a long-lived TUI. Same pairing
    the server's credential route performs for exactly this event.
    """
    try:
        invalidate_listing(storage_id)
    except Exception:  # noqa: BLE001 - never fail a successful login over a cache
        logger.debug("listing invalidation failed for %s", storage_id, exc_info=True)
    try:
        from local_operator.model.configure import invalidate_model_info_cache

        invalidate_model_info_cache()
    except Exception:  # noqa: BLE001 - same rule as the listing drop above
        logger.debug("model-info invalidation failed for %s", storage_id, exc_info=True)


def _enrich_prices(
    listed: list[tuple[ProviderDefinition, list[DiscoveredModel]]],
    *,
    cache_dir: Any = None,
) -> dict[str, list[DiscoveredModel]]:
    """Each provider's rows with price/limit HOLES filled from the keyless chain.

    WHY: the picker used to price a row from ``merge_models(registry, listing)``
    alone, while the status band priced the same model through the resolver's
    models.dev/OpenRouter leg. A direct-provider model the shipped registry did
    not carry therefore showed a blank price in the picker (``_price``'s unknown
    sentinel) and ``$10/50`` in the band the moment it was selected — the
    operator's ``claude-fable-5-1`` screenshot. Both surfaces now go through
    ``prices.price_row`` so they cannot drift again.

    CONSTRAINTS. (1) Disk only, one read per document: the models.dev projection
    is ~141 KB and the OpenRouter document ~120 KB; parsing either per row would
    turn 400 OpenRouter rows into seconds, and ``resolve_model_info`` per row is
    a three-leg memoised resolution that may fetch. The OpenRouter rows come
    straight from the ``openrouter`` provider's own listing, which this same
    ``live_catalogue`` call has just read under the picker's TTL — so no second
    document, no second request. (2) Only rows whose listing quoted NO money are
    touched, and only the money and the limits the listing left at zero: a price
    the provider's own listing stated is authoritative and never overridden.
    (3) Aggregator rows are never enriched — their listing IS the priced source —
    and a provider the chain does not map (``ollama``, ``radient``) is left as
    is, so a keyless provider's genuine ``free`` stays free (``_price``). An
    aggregator's ``:free`` routes therefore take their ``free`` flag straight
    from ``discovery._row_from_openai_entry``, the parser that saw the explicit
    ``0`` on the wire, and never pass through here at all.
    """
    from local_operator.model.prices import models_dev_providers, price_row

    models_dev = models_dev_providers(cache_dir=cache_dir)
    openrouter: list[DiscoveredModel] = next(
        (rows for definition, rows in listed if definition.id == "openrouter"), []
    )
    result: dict[str, list[DiscoveredModel]] = {}
    for definition, rows in listed:
        if (
            definition.local_setup
            or definition.id in AGGREGATOR_PROVIDERS
            or (models_dev is None and not openrouter)
        ):
            result[definition.id] = rows
            continue
        # The chain is keyed on the canonical provider, the same translation the
        # resolver applies: ``openai-device`` prices as ``openai``.
        canonical = credential_provider_id(definition.id)
        enriched: list[DiscoveredModel] = []
        for row in rows:
            if row.input_price > 0 or row.output_price > 0 or row.free:
                # ``free`` counts as priced: the listing already ANSWERED the
                # money question with a quoted zero, and re-asking the chain
                # could only replace that answer with a third party's rate.
                enriched.append(row)
                continue
            found = price_row(canonical, row.id, models_dev=models_dev, openrouter=openrouter)
            if found is None:
                enriched.append(row)
                continue
            if found.free:
                # A stated zero fills the HOLE without filling the prices: the
                # numbers stay 0.0 and the flag is what the picker reads. Kept
                # ahead of the positive-price test below because a free row has
                # no positive leg and would otherwise be dropped as unanswered.
                enriched.append(dataclasses.replace(row, free=True))
                continue
            if not (found.input_price > 0 or found.output_price > 0):
                enriched.append(row)
                continue
            enriched.append(
                dataclasses.replace(
                    row,
                    input_price=found.input_price,
                    output_price=found.output_price,
                    cache_read_price=row.cache_read_price or found.cache_read_price,
                    cache_write_price=row.cache_write_price or found.cache_write_price,
                    # Limits only where the listing gave none: the provider's
                    # own window is the right number for its endpoint.
                    context_window=row.context_window or found.context_window,
                    max_tokens=row.max_tokens or found.max_tokens,
                )
            )
        result[definition.id] = enriched
    return result


def _price_listing_only_rows(
    listed: list[tuple[ProviderDefinition, list[DiscoveredModel]]],
    *,
    cache_dir: Any = None,
) -> dict[tuple[str, str], DiscoveredModel]:
    """The keyless price chain, applied ONLY to rows a cached listing contributed.

    WHY ONLY THOSE. A first frame painted from the cache carries whatever money
    the provider's own listing stated, and for a model the shipped registry has
    never heard of that is nothing -- so the row paints ``-1.0/-1.0``, the
    picker's blank cell, where the live path paints the real rate. On the
    composer's inline ``/model `` list that blank is PERMANENT, because that
    surface never goes live, so nothing would ever fill it (agent review round 2,
    R2-3). The chain that fills it is the one ``live_catalogue`` already runs
    (:func:`_enrich_prices`): disk only, no request, one document read for the
    whole frame -- 0.9-1.0 ms here for the whole helper against a 138 KiB
    models.dev projection, 0.9-2.1 ms for the document read inside it (measured
    2026-09-24 on the fleet host: isolated cache, 25-call median after a
    warm-up; the range is three runs and the magnitude is rig-dependent). It is
    not entered at all when nothing needs it, so a cold cache costs this frame
    exactly what it cost before.

    WHY NOT EVERY ROW, which would be the smaller change: the SHIPPED rows are
    the frame's contract with the registry. With nothing cached, frame one is
    ``static_catalogue()`` field for field, and the live pass is what upgrades
    it; pricing rows the registry already describes here would silently break
    that equality -- including for shipped rows whose price the registry does not
    know, which today paint blank on frame one and on the live frame alike.
    Returns a ``(provider id, model id)`` keyed map, so the caller splices by
    identity and needs to know nothing about the chain's own ranking.
    """
    wanted: list[tuple[ProviderDefinition, list[DiscoveredModel]]] = []
    for definition, models in listed:
        shipped = static_models(credential_provider_id(definition.id))
        only = [row for row in models if row.id not in shipped]
        if only:
            wanted.append((definition, only))
    if not wanted:
        return {}
    return {
        (provider_id, row.id): row
        for provider_id, rows in _enrich_prices(wanted, cache_dir=cache_dir).items()
        for row in rows
    }


def _price(value: float | None, definition: ProviderDefinition, *, free: bool = False) -> float:
    """A per-million price, with UNKNOWN kept distinct from FREE.

    Discovery and the static registry both use ``0`` for "no price known", and the
    picker renders a genuine pair of zeroes as ``free`` — so passing an unknown
    through as zero advertises a paid model as costing nothing. Anthropic makes
    this immediate rather than theoretical: its listing carries no pricing at all,
    so every model it discovers that we did not already ship would read ``free``.

    ``-1`` is the unknown sentinel the picker blanks. Zero — and therefore the
    word ``free`` — survives in exactly two cases:

    * ``allows_missing_api_key``: a local Ollama really is free per token, and
      blanking that would hide the one thing that makes it interesting.
    * ``free``: a SOURCE stated the zero. That is a quoted price, not a silence,
      and repeating a quoted zero fabricates nothing. This is what makes the
      picker's ``free`` label reachable for the 18 ``:free`` OpenRouter routes,
      every one of which the listing prices at an explicit ``0``; before it, a
      stated zero collapsed into the unknown sentinel here and rendered as the
      same blank cell as a model nobody had priced.

    ``free`` is never derived from ``value`` — it arrives from the parser that
    read the wire (:attr:`DiscoveredModel.free`) — which is what keeps two rows
    that both reach here as ``0.0`` apart: a plan-billed row whose real cost is
    unknowable stays blank, because the plan catalogues do not set it (see
    ``prices._PLAN_BILLED_KEYS``), and so does a row nobody quoted at all.

    NOT the same ``-1.0`` as ``model.prices._STATED_ZERO``, which means the
    opposite — "models.dev stated this price and it is zero". That marker is
    module-private to ``prices`` and stripped back to ``0.0`` before any row
    reaches here, so the two never meet; they would collide silently if either
    one's reach were widened, hence the note on both.
    """
    if value is not None and value > 0:
        return float(value)
    return (
        0.0 if (free or (definition.local_setup and definition.id != "openai-compatible")) else -1.0
    )
