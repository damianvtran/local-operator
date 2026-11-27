"""The ONE client module for Guard background ingest (RFC v0.9 §5.3, §6.2, §9.2', §23).

STATUS OF THE CONTRACT. Agent-server slice A3 (ingest) is not open yet:
``gh pr list --repo radient-ml/agent-server --search guard`` on 2026-11-27
lists A0 #515 and A1 #516 (merged) and A2 #517 (consent routes) only. This
module therefore codes against the RFC's conformance shape and pins the parts
the RFC leaves open, so A3 can adopt them or change them HERE, in one place,
without touching the producers:

* route ``POST /v1/guard/ingest`` (JWT + an install-key-signed body), same
  ``{payload, alg, key_id, signature}`` envelope and canonical JSON as A2's
  ``POST /v1/guard/consents`` (sorted keys, no whitespace, printable ASCII),
  signed over :data:`BATCH_SIGNING_DOMAIN` + canonical payload -- a distinct
  domain, so a batch signature can never verify as a grant or vice versa;
* ``payload`` = ``{schema, tenant, install_id, batch_id, collection_rev,
  revocation_gen, nonce, ts, <family>: [...]}`` carrying ONLY the families the
  live consent covers (§6.2: "each field's scope ⊆ consent.scopes ∩ current
  ceiling"). The batch binds the consent's ``collection_rev`` and the
  ``revocation_gen`` the member was shown; ``settings_rev`` and ``ent_gen``
  are not known to lop (A2's status answer does not carry them), so A3 must
  bind those server-side or add them to the status answer -- an OPEN POINT
  for A3, stated in the PR;
* ``batch_id`` is derived from the tenant and the sorted record ids, so a
  retry of the same records is the same id (§6.3 idempotency) without any
  local in-flight state;
* a 2xx ``result`` is ``{batch_id, accepted, refused_fields?: [family]}``;
  ``refused_fields`` is the per-field ``guard_scope_not_consented`` refusal.

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
#: ``MAX_OUTBOX_ROWS`` = 2048 rows; 500 keeps one request small (well under
#: agent-server's body limit at ~120 bytes per record) and drains a full
#: outbox in five requests.
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

# -- response classes ------------------------------------------------------
#: A revoke landed first (A2 maps Console revoke to this): stop, record it.
REVOKED_CODES: Final = frozenset({"guard_revocation_stale", "guard_consent_revoked"})
#: Authority no longer holds (org off, plan lapsed, membership gone, device
#: revoked, stale binding): stop for this consent, drop its queue, never
#: re-send it under a later grant (§22).
PAUSED_CODES: Final = frozenset(
    {
        "guard_org_not_on",
        "guard_not_entitled",
        "guard_membership_inactive",
        "guard_device_revoked",
        "guard_device_unknown",
        "guard_device_signature_required",
        "guard_consent_not_live",
        "guard_settings_stale",
        "guard_binding_stale",
        "not_a_member",
    }
)
#: The server's namespace epoch moved (§11.1'/16' drill step 5d).
RESET_CODES: Final = frozenset({"guard_namespace_reset"})
#: The records can never count (day closed, too old, malformed): drop the batch.
DROP_CODES: Final = frozenset(
    {"day_closed", "guard_batch_stale", "guard_report_stale", "guard_invalid_request"}
)


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

    #: accepted | revoked | paused | reset | drop | retry
    kind: str
    code: str = ""
    refused_families: tuple[str, ...] = field(default_factory=tuple)


def classify(code: str) -> str:
    if code in REVOKED_CODES:
        return "revoked"
    if code in PAUSED_CODES:
        return "paused"
    if code in RESET_CODES:
        return "reset"
    if code in DROP_CODES:
        return "drop"
    # guard_not_available (404, namespace off), 429, 5xx and transport errors:
    # the records are still valid; back off and try again.
    return "retry"


def send(transport: GuardIngestTransport, envelope: Mapping[str, Any]) -> IngestResult:
    """Send one batch; never raises (a failure is a classified result)."""

    try:
        result = transport.guard_ingest(envelope)
    except Exception as exc:  # noqa: BLE001 -- classified, never propagated
        code = str(getattr(exc, "code", "") or "guard_ingest_failed")
        return IngestResult(classify(code), code)
    refused = ()
    if isinstance(result, Mapping):
        raw = result.get("refused_fields")
        if isinstance(raw, list):
            refused = tuple(sorted(str(item) for item in raw if item in FAMILY_KEYS))
    return IngestResult("accepted", refused_families=refused)


__all__ = [
    "BATCH_SCHEMA",
    "BATCH_SIGNING_DOMAIN",
    "DROP_CODES",
    "FAMILY_DECISIONS",
    "FAMILY_KEYS",
    "FAMILY_POSTURE",
    "FAMILY_REVIEW_CYCLES",
    "FAMILY_TASK_TIME",
    "MAX_RECORDS_PER_FAMILY",
    "PAUSED_CODES",
    "RESET_CODES",
    "REVOKED_CODES",
    "GuardIngestTransport",
    "IngestResult",
    "batch_id_for",
    "build_payload",
    "classify",
    "send",
    "sign",
]
