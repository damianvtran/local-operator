"""The ONE client module for Guard background ingest (RFC v0.9 §5.3, §6.2, §9.2', §23).

THE CONTRACT IS AGENT-SERVER A3's. radient-ml/agent-server PR #519 (A3) body
``## Wire contract`` is the harness-agnostic ingest contract; lop is its first
producer and a golden test there (``internal/guard/ingest_test.go``) verifies a
batch this module's :func:`build_payload` + :func:`sign` produced. This module
was written against the RFC before A3 opened and is aligned to it here (the
five "Deltas for lop" in that section), so a custom harness and lop send the
same bytes:

* route ``POST /v1/guard/ingest`` (JWT + an install-key-signed body), same
  ``{payload, alg, key_id, signature}`` envelope and canonical JSON as A2's
  ``POST /v1/guard/consents`` (sorted keys, no whitespace, printable ASCII),
  signed over :data:`BATCH_SIGNING_DOMAIN` + canonical payload -- a distinct
  domain, so a batch signature can never verify as a grant or vice versa;
* ``payload`` = ``{schema, tenant, install_id, batch_id, collection_rev,
  revocation_gen, nonce, ts, <family>: [...]}`` carrying ONLY the families the
  live consent covers (§6.2). A3 binds ``settings_rev``, ``ent_gen``,
  ``key_gen`` and the consent identity server-side, so the batch carries only
  ``collection_rev`` and ``revocation_gen``;
* ``batch_id`` is derived from the tenant and the sorted record ids, so a
  retry of the same records is the same id (A3 matches it on the payload
  without ``nonce``/``ts`` and answers ``replayed: true``);
* a 200 ``result`` is ``{batch_id, replayed, accepted, refused_fields,
  <family>: {status, code?, accepted, duplicate, rejected: [{ref, code}]}}``.
  ``refused_fields`` is the per-family ``guard_scope_not_consented``; a
  per-RECORD ``guard_day_closed`` / ``guard_record_conflict`` /
  ``guard_posture_not_designated`` lives inside that 200, and such a record can
  never count, so it is retired with the rest of the batch.

WHAT IS ON THE WIRE. Enums, a UTC day or ISO week, random record ids, a
randomised posture bit, the install id and the consent revisions. No path,
argument, prompt, name, forge id, title, timestamp finer than a day, or true
posture value. :func:`build_payload` refuses any family it does not know, and
every family record is re-checked against its closed key set before signing,
so a producer bug cannot widen what is sent.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

from .consent import canonical_json

BATCH_SCHEMA: Final = "guard_batch_v1"
BATCH_SIGNING_DOMAIN: Final = b"radient-guard-batch-v1\x00"

#: Per family per batch. The decisions outbox can hold at most
#: ``MAX_OUTBOX_ROWS`` = 2048 rows; 500 keeps one request small and drains a
#: full outbox in five requests. This is the DECISIONS cap and the default for
#: an unlisted family; :data:`FAMILY_CAPS` holds the per-family figures.
MAX_RECORDS_PER_FAMILY: Final = 500

FAMILY_DECISIONS: Final = "decisions"
FAMILY_TASK_TIME: Final = "task_time"
FAMILY_REVIEW_CYCLES: Final = "review_cycles"
FAMILY_POSTURE: Final = "posture"

#: The exact keys each family record may carry. Anything else is a bug.
FAMILY_KEYS: Final[Mapping[str, frozenset[str]]] = {
    FAMILY_DECISIONS: frozenset(
        {"record_id", "schema", "day", "outcome", "rule_class", "tool_class"}
    ),
    FAMILY_TASK_TIME: frozenset(
        {"record_id", "schema", "day", "workflow_class", "terminal_state", "active_duration_band"}
    ),
    FAMILY_REVIEW_CYCLES: frozenset(
        {"record_id", "schema", "day", "terminal_outcome", "rounds_band", "source"}
    ),
    FAMILY_POSTURE: frozenset({"schema", "fact", "bit", "week"}),
}

#: A3's per-batch record caps (PR #519 "Caps": 1024 decisions, 256 task_time,
#: 64 review_cycles, 16 posture, 256 KiB body; over any -> 413). The A3 text
#: says "lop's 500-per-family cap fits", which holds for decisions only: 500
#: task_time or review_cycles records would be a 413 every time. Each figure
#: here is at or under A3's; at ~200 bytes a record the largest batch
#: (500 + 256 + 64 + 6) stays well under 256 KiB.
FAMILY_CAPS: Final[Mapping[str, int]] = {
    FAMILY_DECISIONS: MAX_RECORDS_PER_FAMILY,
    FAMILY_TASK_TIME: 256,
    FAMILY_REVIEW_CYCLES: 64,
    FAMILY_POSTURE: 16,
}

# -- response classes ------------------------------------------------------
# Every code below is one A3 emits on POST /v1/guard/ingest (PR #519 "Batch-
# level refusals"). Codes the RFC draft named but A3 never emits
# (``guard_consent_revoked``, ``guard_settings_stale``, ``guard_namespace_reset``,
# ``day_closed``/``guard_batch_stale`` as batch codes) are gone: an unknown
# code is a backed-off retry, never a silent stop or a silent purge.

#: 409: a revoke landed first, on any surface. A STICKY stop: the consent is
#: recorded revoked (sticky, so no prompt re-queues on its own) and the queue
#: is discarded.
REVOKED_CODES: Final = frozenset({"guard_revocation_stale"})
#: The install's signing key was removed (Console "remove computer" or a
#: device revoke): it can never sign again. Every consent this install holds
#: stops, the dead key is retired, and the ONLY way back is a fresh,
#: person-admitted accept, which mints and enrols a NEW key. Never silently.
DEVICE_REVOKED_CODES: Final = frozenset({"guard_device_revoked"})
#: Authority no longer holds (org off, plan lapsed, membership gone, device
#: unknown, no live consent, a re-grant moved the binding): stop for this
#: consent, drop its queue, never re-send it under a later grant (§22).
PAUSED_CODES: Final = frozenset(
    {
        "guard_org_not_on",
        "guard_not_entitled",
        "guard_membership_inactive",
        "guard_device_unknown",
        "guard_device_signature_required",
        "guard_consent_not_live",
        "guard_binding_stale",
        "not_a_member",
    }
)
#: The records can never count: drop the batch (retire its records).
#: ``guard_invalid_request`` -- outside A3's grammar; ``guard_replay`` -- the
#: batch id was reused for a different record set (a producer bug; re-sending
#: the same bytes can only fail again); ``guard_scope_not_consented`` as a
#: WHOLE-batch 403 -- every family in it is outside consent ∩ ceiling (an org
#: narrowing), so those families' records can never count under this consent.
DROP_CODES: Final = frozenset(
    {"guard_invalid_request", "guard_replay", "guard_scope_not_consented"}
)
#: 413: over a cap. Split: send fewer records per family next time.
SPLIT_CODES: Final = frozenset({"guard_batch_too_large"})
#: 400: ``ts`` more than 15 min off. Re-sign now with the current clock; a
#: second refusal in a row means the local clock is off, and backs off.
RESIGN_CODES: Final = frozenset({"guard_request_expired"})
#: 404 ``guard_not_available``: no Guard on this deployment (A0), which is also
#: how a namespace loss surfaces. A3 says "no retry loop", so it is not a
#: retry at the normal cadence: the producers wait the full backoff cap.
#: It does NOT purge the queue -- a 404 during a deploy is not proof the
#: server's data is gone, and the queue's own 72 h TTL bounds what is kept.
UNAVAILABLE_CODES: Final = frozenset({"guard_not_available"})


class GuardIngestTransport(Protocol):
    def guard_ingest(self, envelope: Mapping[str, Any]) -> dict[str, Any]: ...


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def batch_id_for(tenant: str, families: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    """Stable per (tenant, record set): a retry of the same records is the same batch."""

    ids: list[str] = []
    for family in sorted(families):
        for item in families[family]:
            ident = item.get("record_id") or f"{item.get('fact')}@{item.get('week')}"
            ids.append(f"{family}:{ident}")
    material = canonical_json({"tenant": tenant, "records": sorted(ids)})
    return hashlib.sha256(b"radient-guard-batch-id-v1\x00" + material).hexdigest()[:32]


def build_payload(
    *,
    tenant: str,
    install_id: str,
    collection_rev: int,
    revocation_gen: int,
    families: Mapping[str, Sequence[Mapping[str, Any]]],
    ts: int,
    nonce: str | None = None,
) -> dict[str, Any]:
    """The signed payload. Refuses unknown families and unknown record keys."""

    payload: dict[str, Any] = {
        "schema": BATCH_SCHEMA,
        "tenant": tenant,
        "install_id": install_id,
        "collection_rev": int(collection_rev),
        "revocation_gen": int(revocation_gen),
        "nonce": nonce or secrets.token_hex(16),
        "ts": int(ts),
    }
    clean: dict[str, list[dict[str, Any]]] = {}
    for family, records in families.items():
        allowed = FAMILY_KEYS.get(family)
        if allowed is None:
            raise ValueError("unknown guard ingest family")
        rows = []
        for record in records:
            if not set(record) <= allowed:
                raise ValueError("guard ingest record carries a field outside its family")
            rows.append(dict(record))
        if rows:
            clean[family] = rows
    payload.update(clean)
    payload["batch_id"] = batch_id_for(tenant, clean)
    return payload


def sign(key: Any, payload: Mapping[str, Any]) -> dict[str, Any]:
    """``{payload, alg, key_id, signature}`` -- A2's envelope, the batch domain."""

    from local_operator.guard import install_key

    public = install_key.public_raw(key)
    return {
        "payload": dict(payload),
        "alg": "Ed25519",
        "key_id": hashlib.sha256(public).hexdigest()[:32],
        "signature": _b64(key.sign(BATCH_SIGNING_DOMAIN + canonical_json(dict(payload)))),
    }


@dataclass(frozen=True)
class IngestResult:
    """How the producers must treat one send."""

    #: accepted | revoked | device_revoked | paused | drop | split | resign
    #: | unavailable | retry
    kind: str
    code: str = ""
    refused_families: tuple[str, ...] = field(default_factory=tuple)
    #: Records A3 rejected one by one inside the 200 (``<family>.rejected[]``:
    #: ``guard_day_closed`` and the like). Counted for the log only; they are
    #: retired with the rest of the batch because they can never count.
    rejected: int = 0


def classify(code: str) -> str:
    if code in REVOKED_CODES:
        return "revoked"
    if code in DEVICE_REVOKED_CODES:
        return "device_revoked"
    if code in PAUSED_CODES:
        return "paused"
    if code in DROP_CODES:
        return "drop"
    if code in SPLIT_CODES:
        return "split"
    if code in RESIGN_CODES:
        return "resign"
    if code in UNAVAILABLE_CODES:
        return "unavailable"
    # 401, 429 guard_rate_limited, 5xx, transport errors and any code A3 does
    # not list: the records are still valid; back off and try again.
    return "retry"


def send(transport: GuardIngestTransport, envelope: Mapping[str, Any]) -> IngestResult:
    """Send one batch; never raises (a failure is a classified result)."""

    try:
        result = transport.guard_ingest(envelope)
    except Exception as exc:  # noqa: BLE001 -- classified, never propagated
        code = str(getattr(exc, "code", "") or "guard_ingest_failed")
        return IngestResult(classify(code), code)
    refused: tuple[str, ...] = ()
    rejected = 0
    if isinstance(result, Mapping):
        raw = result.get("refused_fields")
        if isinstance(raw, list):
            refused = tuple(sorted(str(item) for item in raw if item in FAMILY_KEYS))
        for family in FAMILY_KEYS:
            detail = result.get(family)
            if isinstance(detail, Mapping) and isinstance(detail.get("rejected"), list):
                rejected += len(detail["rejected"])
    return IngestResult("accepted", refused_families=refused, rejected=rejected)


__all__ = [
    "BATCH_SCHEMA",
    "BATCH_SIGNING_DOMAIN",
    "DEVICE_REVOKED_CODES",
    "DROP_CODES",
    "FAMILY_CAPS",
    "FAMILY_DECISIONS",
    "FAMILY_KEYS",
    "FAMILY_POSTURE",
    "FAMILY_REVIEW_CYCLES",
    "FAMILY_TASK_TIME",
    "MAX_RECORDS_PER_FAMILY",
    "PAUSED_CODES",
    "RESIGN_CODES",
    "REVOKED_CODES",
    "SPLIT_CODES",
    "UNAVAILABLE_CODES",
    "GuardIngestTransport",
    "IngestResult",
    "batch_id_for",
    "build_payload",
    "classify",
    "send",
    "sign",
]
