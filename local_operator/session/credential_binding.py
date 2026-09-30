"""The per-session credential binding: WHICH account is serving a session.

Nothing durable used to record which account served a session: resolution was
re-decided per borrow and per device, and the only "no switch" machinery was the
process-lifetime sticky pin — whose own docstring says *"There is nothing durable
to write and nothing to un-write"*. Two consequences (``docs/design/
mesh-credentials.md`` §2.4): a device that gains its own login mid-session
silently captured the session (local-first cascade), and the §5.5 move assertion
("binding travels byte-identical, re-resolves identically") had no mechanism to
assert against. This module is the durable half of the fix.

THE HOME IS A TRANSCRIPT CUSTOM ROW, and that is the design's own choice because
every path that carries a session carries the transcript: sync/move digest
``transcript.jsonl`` as an allow-list member, fork ``copyfile``s it
byte-for-byte on purpose, and ``Transcript.latest_custom`` reads the newest row
of a type with a backward scan ("each change appends a full snapshot"). A
sidecar would need its own entry in the copy set and its own retention class;
the row needs none.

THE WRITE RULE IS REPLACEMENT STATE, NEVER A DELTA — the same rule the spend
ledger states. Every successful serve is compared against the newest row FOR
THAT PROVIDER; a new row is appended iff there is none, or the effective
identity differs (``owner_device``, ``credential_id``, ``identity_label``,
``policy``). Old rows are never rewritten. On move/adopt and on fork nothing is
written at all: a move is not a credential event, and a fork's ``copyfile``
carries the row. No token event (re-grant, refresh, refusal) writes a row, and
no token material is ever stored here.

TWO DEFAULTS THIS BUILD CARRIES (manager-decided, pending the operator's
objection; both must stay visible in review):

* **D1** — any account change is RECORDED and surfaced by ONE operator-visible
  notice; the switch itself stays allowed exactly where the shipped local-first
  invariant permits it. Never silent, but never blocked. As built (slice B) the
  notice rides this recorder's ``on_change`` seam through
  ``Session.journal_credential_binding_change``, which persists one
  ``session_credential_binding_notice.v1`` row per change — operator-facing,
  never model context — with its sentence rendered in
  ``network/credentials/messages.py`` (the credential copy home).
* **D2** — a row with ``policy: owner`` is HONOURED by readers when present
  (``store.py``'s consult skips local for it), but no set-verb emits one in
  v1: the writer carries an existing row's policy forward and defaults to
  ``local-first``, so this build can only ever write the default.

THE CONSULT READS THROUGH THIS MODULE (slice B, design Q3.1): ``store.py``
installs :meth:`CredentialBindingRecorder.recall_for` as its reader — after the
local tiers miss, and to honour ``policy: owner`` — and the factory wires the
notice seam at attach time. The copy is NOT here: sentences are rendered in
``network/credentials/messages.py`` and the notice row is journaled by the
Session, so this module stays what it was — the row and its recorder. Read the
version gate as the forward-compat story: an unknown ``version`` is treated as
"no row" (never a defaulted value, never an error), and no reader or writer may
ever rewrite a row's bytes — sync digests compare transcript bytes, so a
normalising rewrite would corrupt a move.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from local_operator.network.credentials.types import is_synthetic_credential_id
from local_operator.session.spend import writer_stamp

if TYPE_CHECKING:  # pragma: no cover - typing only
    from local_operator.session.transcript import Transcript

logger = logging.getLogger(__name__)

#: The transcript custom type carrying the binding row. One row per serve
#: CHANGE, newest wins; the ``.v1`` suffix is what a future format bump claims.
SESSION_BINDING_CUSTOM_TYPE = "mesh_credential_binding.v1"

#: The row's details-shape identity. ``schema`` and ``version`` are both part of
#: the format: ``version`` gates interpretation (exact match, see
#: :meth:`CredentialBinding.from_details`), and ``schema`` is the human-readable
#: spelling of that same fact.
SCHEMA_ID = "lop.mesh.credential_binding.v1"
SCHEMA_VERSION = 1

#: The shipped resolution policy: local first, then the broker. The default and,
#: in this build, the only policy a writer can emit (D2 above).
POLICY_LOCAL_FIRST = "local-first"

#: "Broker to this row's owner even when local would answer" — honoured when a
#: row carries it (future/mixed builds write it), never written by this one.
POLICY_OWNER = "owner"

_POLICIES = frozenset({POLICY_LOCAL_FIRST, POLICY_OWNER})


def _as_int(value: Any) -> int | None:
    """``value`` as an ``int``, or ``None`` when it is not a plain number.

    ``bool`` is rejected on purpose: ``True`` is an ``int`` in Python, and a
    malformed row must not be able to contribute a silent ``1`` credential id.
    Strings are rejected too, exactly as the spend ledger's reader does (one
    spelling of "not a number" across the two record types): ``json`` writes a
    number as a number, so a quoted one is a malformed row, and coercing it
    would let a broken writer's ``"42"`` read as a confident binding.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _as_float(value: Any) -> float:
    """``value`` as a ``float``, or ``0.0`` for anything unreadable.

    ``bound_at`` is metadata rather than a gate, so a malformed one does not
    invalidate the row (unlike ``credential_id``): the record's meaning is the
    identity, and a timestamp is only what the reader sorts by second.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _identity_label(identity: Mapping[str, str] | None) -> str:
    """The operator-facing account string from a grant's identity dict.

    The derivation mirrors the usage path's "row's email, else account id"
    rule, but drops its last resort (``cred:<id>``): that is a placeholder for
    a surface inventing a name where none exists, and this row's contract is
    that an unknown label is OMITTED rather than fabricated — §2.1 forbids a
    pool holder's label from riding the record at all, so the absent form is
    the safe half of the same rule.
    """
    if not isinstance(identity, Mapping):
        return ""
    email = str(identity.get("email") or "")
    account_id = str(identity.get("account_id") or "")
    return email or account_id


@dataclass(frozen=True)
class CredentialBinding:
    """One ``mesh_credential_binding.v1`` row, as a value object.

    The fields are exactly the design's §2.4 sketch. What it does NOT carry is
    as load-bearing as what it does: no token material (never — the design's
    §0/§3.3 stance), no grant (grants are memory-only), no refusal state (that
    lives in ``placement.state.json``), and never a SYNTHETIC credential id
    (synthetics are negative by construction and are meaningless off the
    borrower; :func:`is_synthetic_credential_id` gates that at construction).
    """

    provider: str
    owner_device: str
    credential_id: int
    owner_device_name: str = ""
    identity_label: str = ""
    policy: str = POLICY_LOCAL_FIRST
    bound_at: float = field(default_factory=time.time)
    writer: str = ""

    def __post_init__(self) -> None:
        """Refuse the three shapes that must never reach a transcript row.

        Raising (rather than silently dropping) is for the WRITER's benefit —
        a caller building a row from a synthetic id has confused the borrower's
        alias with the owner's row and would otherwise write a plausible wrong
        answer. :meth:`from_details` catches the same ``ValueError`` and reads
        the row as absent.
        """
        if not isinstance(self.provider, str) or not self.provider:
            raise ValueError("a credential binding names the provider key it was resolved for")
        if not isinstance(self.owner_device, str) or not self.owner_device:
            raise ValueError("a credential binding names the device that owns the account")
        if isinstance(self.credential_id, bool) or not isinstance(self.credential_id, int):
            raise ValueError("credential_id must be an int")
        if is_synthetic_credential_id(self.credential_id):
            raise ValueError(
                "a synthetic credential id belongs to the borrower's view of a borrow and "
                "is meaningless off the owner; record the owner-side id instead"
            )
        if self.policy not in _POLICIES:
            raise ValueError(f"unknown policy {self.policy!r}")

    def to_details(self) -> dict[str, Any]:
        """The record's ``details`` payload.

        ``identity_label`` is OMITTED when empty (rather than written as ``""``)
        so a row never asserts a label it does not have; the reader restores the
        empty string. Every other field is always present — the schema is the
        contract, and a reader that needs a key to exist should never meet a
        row the writer chose to slim.
        """
        details: dict[str, Any] = {
            "schema": SCHEMA_ID,
            "version": SCHEMA_VERSION,
            "provider": self.provider,
            "owner_device": self.owner_device,
            "owner_device_name": self.owner_device_name,
            "credential_id": int(self.credential_id),
            "policy": self.policy,
            "bound_at": float(self.bound_at),
            "writer": self.writer or writer_stamp(),
        }
        if self.identity_label:
            details["identity_label"] = self.identity_label
        return details

    @classmethod
    def from_details(cls, details: Any) -> "CredentialBinding | None":
        """Recall a row, or ``None`` when there is nothing trustworthy.

        ``None`` — not a defaulted binding — for a missing, malformed,
        wrong-schema or unknown-version row, mirroring the spend ledger's rule:
        an unknown ``version`` is treated as ABSENT (resolution falls back to
        today's placement-driven path), never defaulted into an interpretation
        this build does not know, and never an error. The row's bytes stay
        untouched either way.
        """
        if not isinstance(details, Mapping):
            return None
        if details.get("schema") != SCHEMA_ID:
            return None
        if _as_int(details.get("version")) != SCHEMA_VERSION:
            return None
        provider = details.get("provider")
        owner_device = details.get("owner_device")
        if not isinstance(provider, str) or not provider:
            return None
        if not isinstance(owner_device, str) or not owner_device:
            return None
        credential_id = _as_int(details.get("credential_id"))
        if credential_id is None or is_synthetic_credential_id(credential_id):
            return None
        policy = str(details.get("policy") or "")
        if policy not in _POLICIES:
            return None
        try:
            return cls(
                provider=provider,
                owner_device=owner_device,
                credential_id=credential_id,
                owner_device_name=str(details.get("owner_device_name") or ""),
                identity_label=str(details.get("identity_label") or ""),
                policy=policy,
                bound_at=_as_float(details.get("bound_at")),
                writer=str(details.get("writer") or ""),
            )
        except ValueError:
            # Defensive: every field above is pre-validated, so this only fires
            # if the constructor's rules and this reader's rules ever drift.
            return None


def recall(transcript: "Transcript | None") -> CredentialBinding | None:
    """The newest binding row this transcript carries (any provider), or ``None``.

    One dict lookup on the index the transcript's constructor already built —
    no scan, no re-parse — mirroring :func:`local_operator.session.spend.recall`.
    """
    if transcript is None:
        return None
    try:
        details = transcript.latest_custom(SESSION_BINDING_CUSTOM_TYPE)
    except Exception:  # noqa: BLE001 — a missing index is "no row", not an error
        return None
    return CredentialBinding.from_details(details)


def recall_for(transcript: "Transcript | None", provider: str) -> CredentialBinding | None:
    """The newest binding row FOR ``provider``, or ``None``.

    Rows are per-provider facts, so the write rule's "no row for that provider"
    question cannot be answered by :func:`recall` alone (the newest row overall
    may belong to another provider). A backward walk over the in-memory entries
    is the cheap form: the newest row of this provider is almost always the last
    custom entry of this type, and the walk stops at the FIRST match for the
    provider — an unknown version there reads as absent and does NOT fall
    through to an older row, because "newest wins" is the whole read rule.
    """
    if transcript is None:
        return None
    try:
        entries = transcript.entries()
    except Exception:  # noqa: BLE001 — a missing index is "no row", not an error
        return None
    for entry in reversed(entries):
        try:
            payload = entry.payload
            if not isinstance(payload, Mapping):
                continue
            if payload.get("custom_type") != SESSION_BINDING_CUSTOM_TYPE:
                continue
            details = payload.get("details")
            if not isinstance(details, Mapping):
                continue
            if str(details.get("provider") or "") != provider:
                continue
        except AttributeError:  # a foreign entry shape is not this row's answer
            continue
        return CredentialBinding.from_details(details)
    return None


async def record(transcript: "Transcript", binding: CredentialBinding) -> None:
    """Append ``binding`` as the newest replacement-state row.

    ``preserve_mtime=True`` is REQUIRED, and it is honoured only because
    ``SESSION_BINDING_CUSTOM_TYPE`` is a member of
    ``Transcript.BOOKKEEPING_CUSTOM_TYPES``: this row is bookkeeping ABOUT the
    session, and without both halves every append would restamp the session as
    freshly worked in and age it toward deletion (the rule the spend record and
    the MCP-unavailable record state at their members' comments).
    """
    await transcript.append_custom(
        SESSION_BINDING_CUSTOM_TYPE, binding.to_details(), preserve_mtime=True
    )


def _binding_changed(previous: CredentialBinding, candidate: CredentialBinding) -> bool:
    """Whether ``candidate`` is news against ``previous`` (the write rule's diff).

    The four fields the write rule names, and only those: ``owner_device``
    (whose account serves), ``credential_id`` (which row of theirs — the §4.9
    sibling pick), ``identity_label`` (the operator-facing account string) and
    ``policy``. ``bound_at``/``writer``/``owner_device_name`` are provenance,
    not identity, so a rename or a retry never appends a row.
    """
    return (
        previous.owner_device != candidate.owner_device
        or previous.credential_id != candidate.credential_id
        or previous.identity_label != candidate.identity_label
        or previous.policy != candidate.policy
    )


class CredentialBindingRecorder:
    """Coalesced, best-effort writer of binding rows for ONE session.

    Shaped like the spend ledger's persist discipline (dirty flag + single
    flush task): a flurry of resolves in one turn appends at most one row per
    provider, the append runs on the loop as a task, and a failure is DEBUG —
    a lost row is never a failed turn. Two feeds report serves:

    * :meth:`observe_serve` — the mesh store's sink, invoked after a successful
      serve on EITHER of its branches; a borrow carries the owner's real
      ``credential_id`` from the grant ref (the only place it is in hand), and
      the store's local branch reports ``owner_device=""`` (this device).
    * :meth:`observe_local` — the plain-store direction, driven off the sticky
      read at the model-call boundary: an owner device that borrows nothing
      still has to record, because §5.5's owner→borrower move needs a source
      row on the device that held the account.

    Neither feed adds a round trip: every observation is built from data already
    in hand (a grant ref, or one in-process sticky read), and the write is
    coalesced and asynchronous. The recorder is created only where a credential
    placement document exists (see :func:`recorder_for_session`), so a 0-peer
    root never constructs one and its transcripts stay byte-identical.
    """

    def __init__(
        self,
        transcript: "Transcript",
        *,
        session_id: str,
        device_id: str,
        device_name: str = "",
        on_change: Callable[[CredentialBinding, "CredentialBinding | None"], None] | None = None,
    ) -> None:
        self._transcript = transcript
        #: The transcript must belong to a session; an empty id means there is
        #: no session scope to record against (a bare reader, a probe), so the
        #: recorder stays mute rather than filing rows against nothing.
        self.session_id = session_id
        #: How a LOCAL serve names the owning device (owner + name fields).
        self.device_id = device_id
        self.device_name = device_name
        #: The account-change NOTICE seam: called with ``(new, previous)``
        #: after each appended row, exactly once per append, and never allowed
        #: to fail the write that produced it. The factory installs the
        #: session's journaling hook after construction
        #: (:meth:`set_change_handler`); ``None`` — every other construction
        #: site — journals nothing.
        self._on_change = on_change
        #: The newest candidate per provider, waiting for the flush.
        self._observed: dict[str, CredentialBinding] = {}
        #: The newest KNOWN row per provider; membership of this dict answers
        #: "already read the transcript for this provider?" — a value of
        #: ``None`` is a checked absence, which is a fact worth caching too.
        self._rows: dict[str, CredentialBinding | None] = {}
        self._task: asyncio.Task[None] | None = None
        self._dirty = False

    def set_change_handler(
        self,
        handler: Callable[[CredentialBinding, "CredentialBinding | None"], None] | None,
    ) -> None:
        """Install (or clear) the account-change notice seam after construction.

        The factory builds the recorder BEFORE the Session exists, so the
        notice handler is bound at attach time (``attach_credential_binding``);
        tests that only exercise the write rule leave it unset and journal
        nothing. Same best-effort contract as the constructor's ``on_change``.
        """
        self._on_change = handler

    def recall_for(self, provider: str) -> CredentialBinding | None:
        """The newest row this session carries for ``provider``, read FRESH.

        THE CONSULT READ (installed as ``store.set_binding_reader``). Fresh
        rather than served from ``self._rows``: the transcript is the
        authority, and it is also the one surface carrying rows this process
        did not write — a moved session's destination reads the row the source
        recorded. The walk is over the in-memory entry list and stops at the
        first match for the provider, so a resolve adds no I/O.
        """
        return recall_for(self._transcript, provider)

    # -- the two feeds ------------------------------------------------------

    def observe_serve(
        self,
        *,
        provider: str,
        owner_device: str,
        credential_id: int | None,
        owner_device_name: str = "",
        identity: Mapping[str, str] | None = None,
    ) -> None:
        """Report one completed serve through the mesh store's sink.

        ``owner_device == ""`` is the store's LOCAL branch (its own wrapped
        store answered): the owner is then this device. A borrow passes the
        grant ref's owner and its identity dict.

        Never raises and never blocks: a malformed report is debug-logged and
        dropped, because this runs inside a serve that must not fail with it.
        """
        try:
            if owner_device:
                if credential_id is None:
                    # A borrow always carries the owner's real row id; a report
                    # without one is a wiring bug, and an empty id must never
                    # reach a row.
                    return
                candidate: CredentialBinding | None = CredentialBinding(
                    provider=str(provider),
                    owner_device=owner_device,
                    owner_device_name=owner_device_name,
                    credential_id=credential_id,
                    identity_label=_identity_label(identity),
                )
            else:
                candidate = self._local_binding(provider, credential_id)
                if candidate is None:
                    return
        except Exception:  # noqa: BLE001 — a malformed serve report must not fail a turn
            logger.debug("credential binding serve report rejected", exc_info=True)
            return
        self._observe(candidate)

    def observe_local(self, *, provider: str, credential_id: int | None) -> None:
        """Report one completed serve by THIS device's own credential.

        The plain-store / owner-device feed, driven off the model-call
        boundary's sticky read. A synthetic id means the store answered with a
        BORROW (its alias for one) — skipped here on purpose: the borrow is the
        store sink's fact, reported with the owner's real id, and a synthetic id
        must never enter a row.
        """
        try:
            candidate = self._local_binding(provider, credential_id)
        except Exception:  # noqa: BLE001 — a malformed report must not fail a turn
            logger.debug("credential binding local report rejected", exc_info=True)
            return
        if candidate is not None:
            self._observe(candidate)

    def _local_binding(self, provider: str, credential_id: int | None) -> CredentialBinding | None:
        """The candidate row for a local serve, or ``None`` when there is none.

        The identity label is left empty, and slice B did NOT change that: the
        account-change notice reads labels off the rows themselves (a borrow's
        row carries the grant's identity; a local row carries none), so nothing
        resolves a local side's label today — the notice names the device
        instead. The field is optional by schema, so the omission is honest
        rather than lossy, and a future reader that needs the label (a usage
        surface, say) must resolve it where the credential row is in hand.
        """
        if credential_id is None or isinstance(credential_id, bool):
            return None
        if not isinstance(credential_id, int) or is_synthetic_credential_id(credential_id):
            return None
        if not self.device_id:
            return None
        return CredentialBinding(
            provider=str(provider),
            owner_device=self.device_id,
            owner_device_name=self.device_name,
            credential_id=credential_id,
        )

    # -- coalescing (the spend ledger's persist discipline) ------------------

    def _observe(self, candidate: CredentialBinding) -> None:
        if not self.session_id:
            return
        self._observed[candidate.provider] = candidate
        self._dirty = True
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop: nothing to schedule; drain() flushes it directly
        try:
            if self._task is not None and not self._task.done():
                return
            self._task = loop.create_task(self._flush())
        except Exception:  # noqa: BLE001 — scheduling is best-effort; drain() still flushes
            logger.debug("credential binding scheduling failed", exc_info=True)

    async def _flush(self) -> None:
        """Write the newest observation, re-entering while new ones land.

        One task in flight at a time with a dirty flag, exactly like the spend
        record: the row is replacement state, so only the newest value matters
        and skipping an intermediate one loses nothing.
        """
        while True:
            self._dirty = False
            await self._write_observed()
            if not self._dirty:
                return

    async def _write_observed(self) -> None:
        observed, self._observed = self._observed, {}
        for provider, candidate in observed.items():
            previous: CredentialBinding | None = None
            try:
                if provider in self._rows:
                    previous = self._rows[provider]
                else:
                    previous = recall_for(self._transcript, provider)
                if previous is not None:
                    # Replacement state: the new snapshot carries the existing
                    # policy forward. This build cannot EMIT ``owner`` (D2); it
                    # must not DROP one a mixed fleet wrote.
                    candidate = replace(candidate, policy=previous.policy)
                    if not _binding_changed(previous, candidate):
                        self._rows[provider] = previous
                        continue
                await record(self._transcript, candidate)
                self._rows[provider] = candidate
            except Exception:  # noqa: BLE001 — a lost row is not a failed turn
                logger.debug("credential binding record write failed", exc_info=True)
                continue
            self._notify_change(candidate, previous)

    def _notify_change(
        self, binding: CredentialBinding, previous: CredentialBinding | None
    ) -> None:
        """Hand one appended row to the change callback, best-effort.

        ``previous is None`` marks the first serve (nothing to compare, so no
        notice will be due); every later call marks a real change.
        """
        callback = self._on_change
        if callback is None:
            return
        try:
            callback(binding, previous)
        except Exception:  # noqa: BLE001 — a notice must never fail the write
            logger.debug("credential binding change callback failed", exc_info=True)

    async def drain(self) -> None:
        """Flush pending observations and wait for the in-flight write.

        Registered as a dispose hook so the last turn's row is not lost to a
        teardown racing it; tests call it to settle a recorder deterministically
        (no sleeps, no polling). Never raises: teardown must not be masked by a
        bookkeeping write, exactly as a failed append must not fail a turn.
        """
        task = self._task
        if task is not None and not task.done():
            try:
                await task
            except Exception:  # noqa: BLE001
                logger.debug("credential binding drain failed", exc_info=True)
        if self._dirty:
            try:
                await self._flush()
            except Exception:  # noqa: BLE001
                logger.debug("credential binding drain failed", exc_info=True)


def recorder_for_session(
    transcript: "Transcript",
    *,
    config_dir: Path | None,
    session_id: str,
    on_change: Callable[[CredentialBinding, "CredentialBinding | None"], None] | None = None,
) -> CredentialBindingRecorder | None:
    """The recorder for a session on ``config_dir``, or ``None`` when ungateable.

    THE GATE IS NOT ``build_auth_store``'s PREDICATE — that asks "does this
    device BORROW?", and a pure owner device needs rows too (§5.5's
    owner→borrower direction). The gate is: this config root holds at least one
    credential placement document (an owner writes its own entry when it
    declares; a 0-peer root has none). Without one there is nothing to record
    against and the recorder is never created, which — with the store's own
    0-peer branch — is what keeps a networkless root's transcript
    byte-identical.

    The identity is required as well: a local serve names THIS device, and a
    root with a placement document but no keypair is a broken/moved state where
    the honest answer is to stay mute rather than write rows with an empty
    owner. Reads only; nothing is created or minted here.

    A ``None`` ``config_dir`` means "nothing to gate on", never "the default
    root": a session's recorder must read the root it was built for, and an
    ambient fallback here would read — and bind rows to — whichever config dir
    happened to be current.
    """
    from local_operator.network.credentials.placement import has_any_placement
    from local_operator.network.identity import load as load_identity

    if config_dir is None:
        return None
    root = Path(config_dir)
    if not has_any_placement(root):
        return None
    identity = load_identity(root)
    if identity is None:
        return None
    return CredentialBindingRecorder(
        transcript,
        session_id=session_id,
        device_id=identity.device_id,
        device_name=identity.name,
        on_change=on_change,
    )
