"""H5b retry state: durable decide intents for consumed human answers.

WHY THIS MODULE EXISTS. The decide executor (``.decide``) fires at the
CONSUMPTION seam — the durable append of the ask's response row — because that
is the only point where the receipt can describe what the record actually
holds. A network failure AT that moment must not silently drop the human's
decision, and it must not be papered over by an automatic re-POST later: the
only lawful retry is an EXPLICIT TRUSTED HUMAN action (the U4 card's Retry,
§2.5 ``card.not-sent``), against the SAME frozen record.

HONEST LIMITS (no assumed safety from prose):
- CRASH WINDOW, SHRUNK: the intent row is written SYNCHRONOUSLY immediately
  after the durable ask-response append, before any await — a task switch
  cannot land between the two writes. A HARD KILL still can (two stores, two
  roots; no cross-store transaction exists anywhere), and the window is closed
  by RECOVERY, not by pretending otherwise: ``retry_state_for`` reads "none"
  from the absent row, the card shows "Not sent — try again", and
  ``recover_from_durable_answer`` may re-record only what it RE-DERIVES from
  the DURABLE original answer record (S-3 remediation: the harness loads and
  validates the record itself — no caller scalars).
- GENERATION BINDING: ``BridgeRuntime.key()`` does NOT carry the generation
  (it is base+account+tenant). The generation is a SEPARATE explicit column,
  frozen from the ORIGINAL record's scope; a same-account re-enrollment keeps
  the same key, so retry/recovery compare this column and a row with an EMPTY
  generation (any pre-amend WIP row) is NON-AUTHORIZING — deliberately NO
  backfill/migration to the current generation.
- ORIGINAL SCOPE (S-3/S-5 remediation): the frozen row also carries the
  ask's ORIGINAL scope block (``scope_json``, as persisted in the ask log at
  issuance) and the consumption row id. Every transport door
  (``attempt_decide_intent`` / ``retry_from_human_action`` /
  ``recover_from_durable_answer``) runs the SAME authority validator before
  any POST: the LIVE shared fence (``authority.py``) plus scope/generation
  equality — an exported lower-level call cannot bypass them because there is
  no unchecked transport helper at all.
- SETTLEMENT IS CONDITIONAL: every state write is one guarded UPDATE;
  ``sent`` and its byte-identical ORIGINAL receipt are never downgraded or
  overwritten (m8: the first recorded outcome is authoritative "even if a deny
  or kill landed since", §6.2 :970–996), and a duplicate consumption of an
  already-recorded answer never schedules a new POST (S-5).
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .client import TeamBridgeClient

#: Our own schema generation; bump only for additive migrations.
DECIDE_INTENTS_SCHEMA_VERSION = 1

#: The four states an intent can hold. ``pending`` and ``signin`` are the
#: retry-eligible pool; ``sent`` and ``refused`` are terminal.
INTENT_STATES = ("pending", "sent", "refused", "signin")

#: Refusal codes that mean "nothing was recorded; try again later" (§6.5).
#: Everything else refuses permanently — including ``bridge_disabled`` and
#: ``request_revoked``, which are verdicts, not transport.
TRANSIENT_CODES = frozenset({"rate_limited", "messaging_unavailable", "bridge_unreachable"})


class DecideIntentsRefused(Exception):
    """The ledger cannot serve this request (damage, unknown row, bad state)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def decide_intents_db_path(config_dir: Path | str) -> Path:
    return Path(config_dir) / "team_bridge" / "decide_intents.db"


def scope_owner_key(scope: Mapping[str, Any]) -> str:
    """``BridgeRuntime.key()``'s exact construction, from an ORIGINAL scope.

    The recovery door builds the row key from the DURABLE scope block — never
    from the current runtime — so a re-enrolled account (same key) still finds
    its row and a changed account/org does not.
    """

    return json.dumps(
        [
            str(scope.get("base_url") or ""),
            str(scope.get("account_id") or ""),
            str(scope.get("tenant_id") or ""),
        ]
    )


@dataclass(frozen=True)
class DecideIntent:
    """One frozen human answer, durable before any transport runs."""

    intent_id: str
    owner_key: str
    ask_id: str
    question_id: str
    word: str
    #: The native enrollment generation this answer was frozen under. EMPTY on
    #: any pre-amend WIP row — non-authorizing (no backfill, no silent rebind).
    generation: str
    #: The ORIGINAL bridge scope JSON persisted with the ask at issuance; the
    #: recovery door re-validates it against the live binding. EMPTY on any
    #: pre-scope row — non-authorizing.
    scope_json: str
    #: The durable consumption identity (``store.response_row_id``) the answer
    #: was frozen from — the record selector recovery re-reads.
    consumption_row_id: str
    state: str
    attempts: int
    error_code: str | None
    retry_after: str | None
    receipt_json: str | None
    consumed_at: str | None
    created_at: str


def _row_to_intent(row: sqlite3.Row) -> DecideIntent:
    keys = row.keys()
    return DecideIntent(
        intent_id=str(row["intent_id"]),
        owner_key=str(row["owner_key"]),
        ask_id=str(row["ask_id"]),
        question_id=str(row["question_id"]),
        word=str(row["word"]),
        generation=(str(row["generation"]) if "generation" in keys else ""),
        scope_json=(str(row["scope_json"]) if "scope_json" in keys else ""),
        consumption_row_id=(str(row["consumption_row_id"]) if "consumption_row_id" in keys else ""),
        state=str(row["state"]),
        attempts=int(row["attempts"]),
        error_code=row["error_code"],
        retry_after=row["retry_after"],
        receipt_json=row["receipt_json"],
        consumed_at=row["consumed_at"],
        created_at=str(row["created_at"]),
    )


#: Columns every pre-existing row must carry to be AUTHORIZING at all. A
#: database lacking any of them predates the explicit original-scope binding
#: and is refused — re-running DDL is not repair (see ``outbox``'s law).
_REQUIRED_COLUMNS = frozenset({"generation", "scope_json", "consumption_row_id"})


class DecideIntents:
    """The durable decide-intent ledger. Every method opens a short connection.

    Mirrors :mod:`local_operator.team_bridge.outbox`'s storage law (one schema
    generation; a non-empty database without one is DAMAGE, and re-running DDL
    is not repair).
    """

    def __init__(self, config_dir: Path | str) -> None:
        self._path = decide_intents_db_path(config_dir)

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        existed = self._path.exists() and self._path.stat().st_size > 0
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if existed and not version:
            conn.close()
            raise DecideIntentsRefused(
                "intents_damaged",
                "a non-empty decide-intents database has no schema generation; "
                "refusing authority (re-running DDL is not repair)",
            )
        if version:
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(decide_intents)")}
            if columns and not _REQUIRED_COLUMNS <= columns:
                # A pre-amend WIP database (no scope/consumption binding):
                # refusing is the NON-AUTHORIZING answer — there is no
                # backfill and no silent rebind.
                conn.close()
                raise DecideIntentsRefused(
                    "intents_damaged",
                    "this decide-intents database predates the explicit original-scope "
                    "binding; refusing authority (no backfill)",
                )
        if not version:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS decide_intents ("
                " intent_id TEXT PRIMARY KEY,"
                " owner_key TEXT NOT NULL,"
                " ask_id TEXT NOT NULL,"
                " question_id TEXT NOT NULL,"
                " word TEXT NOT NULL,"
                # Generation binding is EXPLICIT (``BridgeRuntime.key()`` does
                # NOT carry it): a same-account re-enrollment yields the same
                # owner_key, so retry/recovery must compare this column. A row
                # with an EMPTY generation (any pre-amend WIP row) is
                # NON-AUTHORIZING — deliberately NO backfill/migration.
                " generation TEXT NOT NULL DEFAULT '',"
                # The ORIGINAL scope block, frozen from the ask log at
                # consumption; EMPTY is non-authorizing exactly like an empty
                # generation.
                " scope_json TEXT NOT NULL DEFAULT '',"
                " consumption_row_id TEXT NOT NULL DEFAULT '',"
                " state TEXT NOT NULL,"
                " error_code TEXT,"
                " retry_after TEXT,"
                " attempts INTEGER NOT NULL DEFAULT 0,"
                " receipt_json TEXT,"
                " consumed_at TEXT,"
                " created_at TEXT NOT NULL DEFAULT (datetime('now')),"
                " UNIQUE (owner_key, ask_id, question_id)"
                ")"
            )
            conn.execute(f"PRAGMA user_version = {DECIDE_INTENTS_SCHEMA_VERSION}")
            conn.commit()
        return conn

    def record(
        self,
        owner_key: str,
        *,
        ask_id: str,
        question_id: str,
        word: str,
        generation: str = "",
        scope_json: str = "",
        consumption_row_id: str = "",
        consumed_at: str | None = None,
    ) -> tuple[DecideIntent, bool]:
        """Record the frozen answer at consumption; idempotent on the triple.

        The FIRST record wins: a second call for the same
        ``(owner_key, ask_id, question_id)`` returns the existing row — an
        answer is not edited once taken (the answer surface has ``revise``,
        which lands BEFORE consumption; after it, nothing here may mint a new
        decision word from anything but the frozen record). Returns
        ``(intent, created)``: ``created`` is True ONLY when THIS call inserted
        the row — the consumption path schedules a POST for a newly created
        intent and for nothing else (S-5).
        """

        conn = self._connect()
        try:
            cursor = conn.execute(
                "INSERT OR IGNORE INTO decide_intents"
                " (intent_id, owner_key, ask_id, question_id, word, generation,"
                " scope_json, consumption_row_id, state, consumed_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
                (
                    str(uuid.uuid4()),
                    owner_key,
                    ask_id,
                    question_id,
                    word,
                    generation or "",
                    scope_json or "",
                    consumption_row_id or "",
                    consumed_at,
                ),
            )
            created = cursor.rowcount == 1
            conn.commit()
            row = conn.execute(
                "SELECT * FROM decide_intents WHERE owner_key = ? AND ask_id = ?"
                " AND question_id = ?",
                (owner_key, ask_id, question_id),
            ).fetchone()
            return _row_to_intent(row), created
        finally:
            conn.close()

    def get(self, intent_id: str) -> DecideIntent | None:
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT * FROM decide_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            return _row_to_intent(row) if row is not None else None
        finally:
            conn.close()

    def pending(self, owner_key: str) -> list[DecideIntent]:
        """The retry-eligible pool for one account+org (``pending``+``signin``)."""

        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT * FROM decide_intents WHERE owner_key = ?"
                " AND state IN ('pending', 'signin') ORDER BY created_at",
                (owner_key,),
            ).fetchall()
            return [_row_to_intent(row) for row in rows]
        finally:
            conn.close()

    def retry_state_for(
        self,
        owner_key: str,
        ask_id: str,
        question_id: str,
        *,
        current_generation: str | None = None,
    ) -> str:
        """The card's derivation (U4 source contract): one closed state string.

        ``none`` (no row — the crash case), ``pending``, ``signin``, ``sent``,
        ``refused``, or — when ``current_generation`` is supplied, as the card's
        read must — ``owner_changed`` for a row whose stored generation is
        EMPTY or different. The card renders ``none|pending|signin`` as the
        §10.3 state **"Not sent — try again"** with the Retry control; ``sent``
        as the recorded receipt (never a mark before a server receipt, D24);
        ``refused``/``owner_changed`` as terminal sentences.
        """

        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT state, generation FROM decide_intents WHERE owner_key = ?"
                " AND ask_id = ? AND question_id = ?",
                (owner_key, ask_id, question_id),
            ).fetchone()
            if row is None:
                return "none"
            if current_generation is not None:
                stored = str(row["generation"]) if "generation" in row.keys() else ""
                if not stored or stored != str(current_generation):
                    return "owner_changed"
            return str(row["state"])
        finally:
            conn.close()

    def _settle(self, intent_id: str, sql: str, params: tuple[Any, ...]) -> None:
        """Run ONE guarded UPDATE; a no-op row hits report rather than rewrite.

        Conditional settlement is the whole point (S-5): ``sent`` is never
        downgraded and its ORIGINAL receipt is never overwritten; a duplicate
        settle is idempotent. A zero-row update re-reads the row to tell
        "already in a terminal/other state" (no-op, documented per caller)
        from "no such row" (refused).
        """

        conn = self._connect()
        try:
            cursor = conn.execute(sql, params)
            if cursor.rowcount == 0:
                row = conn.execute(
                    "SELECT state FROM decide_intents WHERE intent_id = ?", (intent_id,)
                ).fetchone()
                if row is None:
                    raise DecideIntentsRefused("unknown_intent", f"no intent {intent_id!r}")
                return
            conn.commit()
        finally:
            conn.close()

    def mark_sent(self, intent_id: str, receipt: Any) -> None:
        """Settle ``sent`` with the server's ORIGINAL receipt.

        A row already settled ``sent`` keeps its receipt byte-identically:
        §6.2's m8 makes the first recorded outcome authoritative, "even if a
        deny or kill landed since", so a later attempt must never rewrite it.
        """

        self._settle(
            intent_id,
            "UPDATE decide_intents SET state = 'sent', receipt_json = ?,"
            " error_code = NULL, retry_after = NULL, attempts = attempts + 1"
            # ONLY from a retry-eligible state: a ``refused`` row is terminal
            # too (S-5) — a raced success must never convert it, and a settled
            # ``sent`` row keeps its ORIGINAL receipt byte-identically
            # (§6.2 m8: the first recorded outcome is authoritative).
            " WHERE intent_id = ? AND state IN ('pending', 'signin')",
            (json.dumps(receipt, sort_keys=True), intent_id),
        )

    def mark_transient(self, intent_id: str, code: str, retry_after: int | None = None) -> None:
        """Keep the intent in the retry pool; nothing was recorded server-side.

        Applied only to a ``pending`` row: a ``signin`` park keeps its signal,
        and a settled ``sent``/``refused`` row is NEVER downgraded by a later
        transport failure (S-5).
        """

        self._settle(
            intent_id,
            "UPDATE decide_intents SET state = 'pending', error_code = ?,"
            " retry_after = ?, attempts = attempts + 1"
            " WHERE intent_id = ? AND state = 'pending'",
            (code, None if retry_after is None else str(retry_after), intent_id),
        )

    def mark_signin(self, intent_id: str) -> None:
        """Park on the sign-in state; NO automatic retry from here."""

        self._settle(
            intent_id,
            "UPDATE decide_intents SET state = 'signin', error_code = 'signin_required',"
            " attempts = attempts + 1"
            " WHERE intent_id = ? AND state IN ('pending', 'signin')",
            (intent_id,),
        )

    def mark_refused(self, intent_id: str, code: str) -> None:
        """Settle ``refused`` (terminal). ``sent``/``refused`` are never overwritten."""

        self._settle(
            intent_id,
            "UPDATE decide_intents SET state = 'refused', error_code = ?,"
            " attempts = attempts + 1"
            " WHERE intent_id = ? AND state NOT IN ('sent', 'refused')",
            (code, intent_id),
        )


def _validated_for_transport(intent: DecideIntent, runtime: Any) -> None:
    """The SAME validator every transport door runs (S-3), before any POST:

    - the live shared authority fence (enabled/owner/generation/consent/
      binding/designation — ``authority.py``);
    - the frozen GENERATION is non-empty and equals the live binding;
    - the frozen ORIGINAL SCOPE is non-empty and names exactly this binding
      (``scope_matches_runtime``; a legacy/empty scope is non-authorizing).
    """

    from .authority import authority_fence, scope_matches_runtime

    refusal = authority_fence(runtime)
    if refusal is not None:
        raise DecideIntentsRefused(
            str(refusal.get("code") or "owner_changed"),
            str(refusal.get("error") or "the runtime is not authorized"),
        )
    generation = str(getattr(runtime.owner, "enrollment_generation", "") or "")
    if not generation or str(intent.generation or "") != generation:
        raise DecideIntentsRefused(
            "owner_changed",
            "the frozen intent's generation is empty or differs from the live "
            "binding — non-authorizing (no backfill, no silent rebind)",
        )
    try:
        scope = json.loads(intent.scope_json) if intent.scope_json else None
    except ValueError:
        scope = None
    if not scope_matches_runtime(scope, runtime):
        raise DecideIntentsRefused(
            "owner_changed",
            "the frozen intent carries no ORIGINAL scope for this binding — " "non-authorizing",
        )


async def attempt_decide_intent(
    client: TeamBridgeClient,
    intents: DecideIntents,
    intent: DecideIntent,
    *,
    runtime: Any,
) -> dict[str, Any] | None:
    """Dispatch ONE frozen intent and settle it on a definitive outcome.

    The decision word never leaves the closed executor: the intent's word is
    rebuilt through :func:`human_decision`-adjacent checks — ``human_decision``
    refuses anything outside approve|deny — and the transport call is
    :func:`decide_from_human_answer`; this function adds the SHARED validator
    (S-3: no unchecked transport door) and settling, not authority.

    ``sent``/``refused`` re-read as terminal here too: a lower-level call can
    never re-POST a settled receipt (S-5) — the m8 replay lane belongs to the
    human retry door, which refuses those states first.
    """

    from .client import BridgeRefusal, BridgeSignInRequired, BridgeUnreachable
    from .decide import NotAHumanDecision, decide_from_human_answer, human_decision

    _validated_for_transport(intent, runtime)
    current = intents.get(intent.intent_id)
    if current is None:
        raise DecideIntentsRefused("unknown_intent", f"no intent {intent.intent_id!r}")
    if current.state in ("sent", "refused"):
        raise DecideIntentsRefused(
            "not_retryable",
            f"intent {intent.intent_id!r} is {current.state!r}: the recorded outcome is "
            "terminal (m8) and no transport may re-POST it",
        )

    try:
        decision = human_decision(
            message_id=intent.question_id, ask_id=intent.ask_id, word=intent.word
        )
    except NotAHumanDecision as exc:
        # A corrupted row is refused, never sent — the closed set is the door.
        intents.mark_refused(intent.intent_id, "invalid_intent")
        raise DecideIntentsRefused("invalid_intent", str(exc)) from exc

    try:
        result = await decide_from_human_answer(
            client,
            decision,
            tenant_id=runtime.tenant_id,
            expected_account_id=getattr(runtime.owner, "account_id", None),
        )
    except BridgeSignInRequired:
        intents.mark_signin(intent.intent_id)
        return None
    except BridgeUnreachable:
        # Nothing answered — retryable, and only by an explicit human action.
        intents.mark_transient(intent.intent_id, "bridge_unreachable")
        return None
    except BridgeRefusal as refusal:
        code = str(getattr(refusal, "code", "refused"))
        if code in TRANSIENT_CODES:
            intents.mark_transient(
                intent.intent_id, code, getattr(refusal, "retry_after_seconds", None)
            )
        else:
            intents.mark_refused(intent.intent_id, code)
        return None

    receipt = result.get("request") if isinstance(result, Mapping) else result
    intents.mark_sent(intent.intent_id, receipt)
    return result


async def retry_from_human_action(
    client: TeamBridgeClient,
    intents: DecideIntents,
    intent_id: str,
    *,
    runtime: Any,
) -> dict[str, Any] | None:
    """THE retry door — callable ONLY from an explicit trusted human action.

    §2.5 ``card.not-sent`` requires a Retry control on the card; this is the
    function that control will call (the U4 surface itself is a later, frozen
    lane). The rules it enforces:

    - the live shared fence AND the frozen generation/scope validation run
      FIRST (S-1/S-3) — a drifted binding refuses before any traffic;
    - the intent must be retry-eligible (``pending`` after a transient, or
      ``signin`` after a sign-in gate) — ``sent``/``refused`` are terminal
      (the recorded original receipt is retrieved locally, never re-POSTed);
    - the replay is the SAME frozen record, verbatim; nothing is re-derived
      from the message text or the model;
    - no caller may be automatic: this module is not imported by any
      scheduler, timer, wake path or tool — reachability IS the enforcement,
      exactly as ``.decide`` documents for itself.
    """

    from .authority import authority_fence

    refusal = authority_fence(runtime)
    if refusal is not None:
        raise DecideIntentsRefused(
            str(refusal.get("code") or "owner_changed"),
            str(refusal.get("error") or "the runtime is not authorized"),
        )
    intent = intents.get(intent_id)
    if intent is None:
        raise DecideIntentsRefused("unknown_intent", f"no intent {intent_id!r}")
    if intent.state not in ("pending", "signin"):
        raise DecideIntentsRefused(
            "not_retryable",
            f"intent {intent_id!r} is {intent.state!r}: only a transient or sign-in"
            " state may be retried, and only by an explicit human action",
        )
    return await attempt_decide_intent(client, intents, intent, runtime=runtime)


def _load_consumed_response_row(runtime: Any, ask_id: str) -> Mapping[str, Any] | None:
    """The IMMUTABLE consumed ASK_RESPONSE details for ``ask_id``, or ``None``.

    S-7: the durable human receipt is the transcript row the CONSUMPTION
    append wrote — NOT the ask log's mutable ``answered`` fold (which can
    precede consumption and still be revised before it) and NOT a computed
    row id (a selector is not evidence). ``append_message`` persists a
    message's own id as the entry id, so the expected selector
    (``asks.store.response_row_id``) addresses the row exactly, and
    ``transcript.read_entry_by_id`` answers it purely (no directory creation).

    Verification is fail-closed HERE, before any insert or POST: the parsed
    row's id must equal the expected selector; the payload must be a custom
    ``ask_response`` row attributed to the HUMAN; and the details must name
    the SAME ask. A forged/foreign/missing/wrong-type row answers ``None``
    and the caller raises its typed refusal — never a current-generation
    substitute.
    """

    from local_operator.asks import store as asks_store
    from local_operator.harness.message_types import ASK_RESPONSE_MESSAGE_TYPE
    from local_operator.session import transcript as transcript_mod

    expected = asks_store.response_row_id(str(ask_id))
    directory = asks_store.session_dir(runtime.config_dir, runtime.session_id)
    entry = transcript_mod.read_entry_by_id(directory, expected)
    if entry is None or entry.id != expected or entry.type != transcript_mod.ENTRY_MESSAGE:
        return None
    payload = dict(entry.payload or {})
    if str(payload.get("custom_type") or "") != ASK_RESPONSE_MESSAGE_TYPE:
        return None
    if str(payload.get("attribution") or "") != "user":
        return None
    details = payload.get("details")
    if not isinstance(details, Mapping) or str(details.get("ask_id") or "") != str(ask_id):
        return None
    if str(details.get("bridge_provenance") or "") != "authority_grant":
        # R3-5: the marker is a CORRELATION marker — it records that the HARNESS
        # rendered this row under a confirmed admitted verdict, which is what
        # separates it from a fold line any same-uid writer appended. It is NOT
        # unforgeable by itself (the transcript file is writable by the same
        # user) and this door does not treat it as authority: the consumed
        # GRANT, the original scope block and the live fence are the authority,
        # and the marker only narrows which rows may even be considered. Rows
        # that predate the marker are non-authorizing (typed refusal upstream),
        # never backfilled.
        return None
    return dict(details)


async def recover_from_durable_answer(
    client: TeamBridgeClient,
    intents: DecideIntents,
    runtime: Any,
    *,
    ask_id: str,
    question_id: str,
) -> dict[str, Any] | None:
    """The Retry control's ONLY recovery entry (U4 source contract; visual later).

    NO CALLER SCALARS (S-3). The caller supplies a durable-record SELECTOR —
    ``(ask_id, question_id)`` — and the HARNESS loads and validates the
    original answer itself from the IMMUTABLE consumed transcript row (S-7):
    the row's existence under its EXPECTED selector, its custom type, its
    human attribution, the question binding, the ORIGINAL bridge scope block
    (S-3: a generic ``human_required`` flag is NOT provenance), the exact
    single approve|deny pick, and the scope's equality with the CURRENT live
    binding. An answered ask-log fold is mutable/unconsumed state and is NOT
    the proof; a computed row id is only a selector. Missing/legacy/empty
    scope or generation is non-authorizing and surfaces as a typed refusal —
    never a fresh mint under the current generation, never a backfill. Then
    the idempotent ``record`` and the unchanged human-retry door
    (pending/signin only; sent/refused terminal; m8 never downgrades). Never
    scheduled, never model-reachable.
    """

    from local_operator.asks import store as asks_store

    from .authority import authority_fence, scope_matches_runtime
    from .decide import NotAHumanDecision, human_decision

    refusal = authority_fence(runtime)
    if refusal is not None:
        raise DecideIntentsRefused(
            str(refusal.get("code") or "owner_changed"),
            str(refusal.get("error") or "the runtime is not authorized"),
        )
    details = _load_consumed_response_row(runtime, str(ask_id))
    if details is None:
        raise DecideIntentsRefused(
            "no_durable_answer",
            f"no CONSUMED human-answer row for {ask_id!r} in this session",
        )
    questions = [q for q in (details.get("questions") or []) if isinstance(q, Mapping)]
    matched = [q for q in questions if str(q.get("question_id") or "") == str(question_id)]
    if not matched:
        raise DecideIntentsRefused(
            "frozen_mismatch", f"the durable ask has no question {question_id!r}"
        )
    if len(matched) > 1:
        raise DecideIntentsRefused(
            "frozen_mismatch",
            f"the durable ask maps {question_id!r} to MORE than one question — ambiguous",
        )
    question = matched[0]
    # S-6: the queue keys its answer map by the question's IMMUTABLE ``id``
    # (``bridge-bmsg_1``), NOT by ``question_id`` (the server message id
    # ``bmsg_1``) — see asks/queue.py's answer projection. Indexing the
    # message id here read ZERO picks for every legitimately issued ask and
    # broke W1 recovery; message ids are never answer-map keys.
    question_uid = str(question.get("id") or "")
    if not question_uid:
        raise DecideIntentsRefused(
            "frozen_mismatch",
            "the durable ask's bridge question has no immutable id — the answer "
            "map cannot be indexed safely",
        )
    scope = question.get("bridge")
    if not isinstance(scope, Mapping):
        raise DecideIntentsRefused(
            "scope_missing",
            "the durable answer carries no ORIGINAL bridge scope — unrecoverable "
            "here; a human must approve a NEW request (no backfill from the "
            "current binding)",
        )
    if not scope_matches_runtime(scope, runtime):
        raise DecideIntentsRefused(
            "owner_changed",
            "the durable answer's ORIGINAL scope differs from the live binding — "
            "non-authorizing (no silent rebind)",
        )
    answers = details.get("answers") or {}
    picks = [str(v) for v in (answers.get(question_uid) or [])]
    if len(picks) != 1:
        raise DecideIntentsRefused(
            "not_human_decision",
            "the durable answer is not exactly one pick — nothing to replay",
        )
    try:
        wanted = human_decision(message_id=question_id, ask_id=ask_id, word=picks[0])
    except NotAHumanDecision as exc:
        raise DecideIntentsRefused("not_human_decision", str(exc)) from exc
    answered_at = details.get("at")
    intent, _created = intents.record(
        scope_owner_key(scope),
        ask_id=ask_id,
        question_id=question_id,
        word=wanted.word,
        generation=str(scope.get("generation") or ""),
        scope_json=json.dumps(dict(scope), sort_keys=True),
        consumption_row_id=asks_store.response_row_id(str(ask_id)),
        consumed_at=str(int(answered_at)) if answered_at else None,
    )
    if not intent.generation:
        raise DecideIntentsRefused(
            "owner_changed", "the frozen intent carries no generation — non-authorizing"
        )
    if intent.word != wanted.word:
        raise DecideIntentsRefused(
            "frozen_mismatch", "the frozen intent's word differs from the durable answer"
        )
    return await retry_from_human_action(client, intents, intent.intent_id, runtime=runtime)


__all__ = [
    "DECIDE_INTENTS_SCHEMA_VERSION",
    "INTENT_STATES",
    "TRANSIENT_CODES",
    "DecideIntent",
    "DecideIntents",
    "DecideIntentsRefused",
    "attempt_decide_intent",
    "decide_intents_db_path",
    "recover_from_durable_answer",
    "retry_from_human_action",
    "scope_owner_key",
]
