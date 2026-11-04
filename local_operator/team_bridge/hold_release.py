"""B4b owner release: a work-scope screening hold the receiving person may Deliver or Discard.

WHY THIS EXISTS (operator decision a-3f46, 2026-11-04, reversing a-d9c6). Until
B4b a model-answer hold was cached with no expiry, there was no per-message
release, and the owner heard only a count after 60 minutes. A genuine relay
("send that to my aida so she can have a team work on these") was therefore
silently parked or bounced. B4b lets the RECEIVING person decide, per message,
for exactly one kind of decision: the screener said the message bids for new
work and nothing else. Everything here is behind ``team_bridge.hold_release``
(default OFF, B4b DESIGN §8.1 slice S1).

WHAT IS RELEASABLE, AND WHY IT IS SO NARROW (DECISIONS Amendment 1, D2/D4):

* ONLY a decision whose categories are exactly ``("new_work_request",)``.
  ``out_of_scope_work`` and ``personal_info_request`` are disclosure-class and
  are NEVER releasable under any template: every single-category
  ``out_of_scope_work`` bounce in the committed v2 trials was adversarial data
  extraction (security review S-1). Any multi-category decision is closed too.
* ONLY when the hop that ACTUALLY answered is on :data:`RELEASE_ELIGIBLE_SCREENERS`,
  a closed in-code allowlist of concrete ``(provider, model)`` pairs (S-3). A
  config value could be edited by an injected agent's ``bash``; a code constant
  changes only by a reviewed PR. Server-routed aliases (``radient/auto``) are
  never eligible: the client cannot see which model served them.
* Eligibility is decided ONCE, when the hold is written, from the answering
  hop, and stored on the decision row (:meth:`Store.record_work_scope_decision`).
  It never reads the session's chat model or the configured default later, so
  changing either cannot widen or narrow a hold already written.
* At most :data:`SENDER_BUDGET` releasable holds per sender per
  :data:`SENDER_BUDGET_WINDOW_S` (S-6): a converted hold posts no hub reject, so
  the hub's anti-probing clamp never counts it, and without a budget a sender
  could probe the releasable class for free while training the person to click.

THE RELEASE RECORD (D7, S-7/S-8/S-11). Deliver and Discard are both signed by
the operator key (TB-02: a bridge deny is the operator's statement to a third
party). The signature is under a DEDICATED domain
(:data:`local_operator.operator.verify.RELEASE_DOMAIN`) over a length-prefixed
payload binding owner, generation, conversation, message, body hash, ENVELOPE
digest (sender, type and request block, not just the body), hold code,
screener identity, template, categories and the word. The connector — not the
route — is the authority: it re-verifies the signature against the anchor and
re-checks the live hold and decision row before it consumes a release
(``screening._consume_owner_release``), so a row written straight into SQLite
does not pass.

WHAT THE SIGNATURE DOES NOT BUY (S-9, D3). ``screening_receipts`` carries no
signature, so same-user code that can write the bridge store can forge an
ordinary model allow and skip this path entirely — on ANY host, hardware key or
not. The release signature gives non-repudiation, binding to what was shown,
and refusal of a forged-but-unsigned row. On a ``file-only`` host any process
running as the user can also read the key and sign; the card says so. The real
defences against the agent's sanctioned surfaces are that the bridge tool has
no release op and the routes need the desktop bearer.

A RELEASE IS NEVER A CACHED ALLOW (DESIGN §5). The release receipt carries
:data:`OWNER_RELEASE_SHA256` as its template hash, which no screening cache
lookup ever asks for, and the model's cached ``work_scope_hold`` answer stays
put: an identical later message is held again and is the person's decision
again.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

#: The ONE hold code a person may release. A closed set of one: a hold code
#: added later is not releasable until a reviewed change says so (D1).
WORK_SCOPE_HOLD = "work_scope_hold"
RELEASABLE_HOLD_CODES = frozenset({WORK_SCOPE_HOLD})

#: The ONLY category tuple that may become a releasable hold (Amendment 1, D2).
RELEASABLE_CATEGORIES = ("new_work_request",)

#: Concrete screener identities whose single-category work-scope decision may
#: be released (D4). ``(provider, model id)`` exactly as the answering hop
#: reports them; on 2026-11-04 the only measured identities are Sonnet 5.5 and
#: Opus 5.5 on the direct Anthropic route. B5's three-model results may extend
#: this, by a reviewed code change only. ``radient/auto`` is absent by design.
RELEASE_ELIGIBLE_SCREENERS = frozenset(
    {
        ("anthropic", "claude-sonnet-5-5"),
        ("anthropic", "claude-opus-5-5"),
    }
)

#: Per-sender work-scope budget (S-6): beyond it a work-scope hold is written
#: ineligible (no buttons, no per-message notice, counted by the 60-minute alert).
SENDER_BUDGET = 3
SENDER_BUDGET_WINDOW_S = 24 * 3600.0

#: Deliver is offered for this long after the hold was written (DESIGN §3.5,
#: the hub's recipient feed window). After it only Discard is offered.
DELIVER_WINDOW_S = 7 * 24 * 3600.0

#: At most this many per-message notices stand per endpoint (DESIGN §2.3).
STANDING_NOTICE_CAP = 10
#: One OS banner per sender per this window; the rest of a burst is badge-only.
BANNER_WINDOW_S = 15 * 60.0

#: Provenance stamped on a release receipt. The template hash is the sha256 of a
#: fixed domain string, so it can never equal a published screening template and
#: no cache lookup can ever return it (DESIGN §5).
OWNER_RELEASE_PROMPT_VERSION = "owner-release-v1"
OWNER_RELEASE_MODEL = "owner-release"
OWNER_RELEASE_SHA256 = hashlib.sha256(b"local-operator team-bridge owner-release v1").hexdigest()

WORDS = ("deliver", "discard")

#: Ineligibility reasons stored on a decision row: a closed vocabulary.
INELIGIBLE_SCREENER = "screener_not_allowlisted"
INELIGIBLE_BUDGET = "sender_budget"


def eligible_screener(provider: str | None, model: str | None) -> bool:
    """Whether the hop that answered may make a releasable decision (D4)."""

    return (str(provider or ""), str(model or "")) in RELEASE_ELIGIBLE_SCREENERS


def converts(categories: Sequence[str]) -> bool:
    """Whether a bounce with these categories becomes a ``work_scope_hold``."""

    return tuple(categories) == RELEASABLE_CATEGORIES


def enabled(config_dir: Path | str) -> bool:
    """The setting, read fresh from disk; a torn or absent read is OFF."""

    from local_operator.config import read_config_values
    from local_operator.team_bridge.settings import TeamBridgeSettings

    values = read_config_values(Path(config_dir))
    if values is None:
        return False
    return TeamBridgeSettings.from_values(values).hold_release


def sender_digest(sender_account_id: str) -> str:
    """A reference to the hub-verified sender for the budget, never a label."""

    return hashlib.sha256(f"tb-sender|{sender_account_id}".encode("utf-8")).hexdigest()


def _lp(value: str) -> bytes:
    """``operator/verify._lp``'s framing: 4-byte big-endian length + UTF-8."""

    raw = str(value).encode("utf-8", "surrogatepass")
    return struct.pack(">I", len(raw)) + raw


def release_payload(
    *,
    owner_key: str,
    generation: int,
    conversation_id: str,
    message_id: str,
    body_sha256: str,
    envelope_digest: str,
    hold_code: str,
    screener_provider: str,
    screener_model: str,
    template_sha256: str,
    categories: Sequence[str],
    word: str,
    decided_at: float,
) -> bytes:
    """The ONE message a release signature covers (S-8: one builder, one domain).

    Every field is length-prefixed so no field can bleed into its neighbour, and
    the domain tag keeps a release signature from ever verifying as an operator
    action, a device certificate or an approval decision.
    """

    from local_operator.operator.verify import RELEASE_DOMAIN

    return (
        RELEASE_DOMAIN
        + _lp(owner_key)
        + _lp(str(int(generation)))
        + _lp(conversation_id)
        + _lp(message_id)
        + _lp(body_sha256)
        + _lp(envelope_digest)
        + _lp(hold_code)
        + _lp(screener_provider)
        + _lp(screener_model)
        + _lp(template_sha256)
        + _lp(json.dumps(sorted(categories), separators=(",", ":")))
        + _lp(word)
        + _lp(f"{float(decided_at):.6f}")
    )


def payload_for_row(row: Mapping[str, Any]) -> bytes:
    """Rebuild the signed payload from a stored ``owner_releases`` row."""

    categories = row.get("categories")
    if isinstance(categories, str):
        try:
            categories = json.loads(categories)
        except ValueError:
            categories = None
    if not isinstance(categories, list):
        categories = ["<damaged>"]
    return release_payload(
        owner_key=str(row.get("owner_key") or ""),
        generation=int(row.get("creator_generation") or -1),
        conversation_id=str(row.get("conversation_id") or ""),
        message_id=str(row.get("message_id") or ""),
        body_sha256=str(row.get("body_sha256") or ""),
        envelope_digest=str(row.get("envelope_digest") or ""),
        hold_code=str(row.get("hold_code") or ""),
        screener_provider=str(row.get("screener_provider") or ""),
        screener_model=str(row.get("screener_model") or ""),
        template_sha256=str(row.get("template_sha256") or ""),
        categories=[str(c) for c in categories],
        word=str(row.get("word") or ""),
        decided_at=float(row.get("decided_at") or 0.0),
    )


def anchor_spki(config_dir: Path | str | None) -> tuple[bytes | None, str]:
    """The operator anchor's public point and key id, or ``(None, "")``.

    The installed root-owned anchor first, else the staged statement: the same
    single reader every other operator-signed surface uses (``load_local_anchor``).
    """

    try:
        from local_operator.operator.trust import load_local_anchor

        anchor = load_local_anchor(Path(config_dir) if config_dir is not None else None)
    except Exception:  # noqa: BLE001 — no readable anchor means nothing verifies
        return None, ""
    if anchor is None:
        return None, ""
    return anchor.spki, anchor.key_id


def verify_row(row: Mapping[str, Any] | None, spki: bytes | None) -> bool:
    """Cryptographically re-verify a stored release (never a ``verified`` column)."""

    if not row or spki is None:
        return False
    from local_operator.operator.verify import key_id_for, verify_signature

    if str(row.get("key_id") or "") != key_id_for(spki):
        return False
    try:
        signature = bytes.fromhex(str(row.get("signature") or ""))
    except ValueError:
        return False
    return verify_signature(spki=spki, message=payload_for_row(row), signature=signature)


def key_protection(config_dir: Path | str) -> str:
    """``hardware`` or ``file_only``: the D3 disclosure the card shows.

    ``file_only`` means any process running as this user can read the key and
    sign a release. ``hardware`` still does not stop same-user code forging a
    bridge-store row (S-9); it only means a signature needed the device.
    """

    try:
        from local_operator.operator.sign import resolve_backend_name
        from local_operator.operator.trust import is_presence_backend

        backend = resolve_backend_name(Path(config_dir))
        return "hardware" if is_presence_backend(backend) else "file_only"
    except Exception:  # noqa: BLE001 — unknown reads as the weaker claim
        return "file_only"


def sign_release(config_dir: Path | str, message: bytes) -> tuple[str, str]:
    """Sign with the local operator key; ``(signature_hex, key_id)``. Prompts where it can.

    Raises :class:`ReleaseSigningUnavailable` when there is no key or the
    gesture failed; no exception text reaches a caller (it can carry paths).
    """

    from local_operator.operator.keychain import KeyBackendError
    from local_operator.operator.sign import (
        load_signer,
        resolve_backend_name,
        sign_message,
    )

    root = Path(config_dir)
    try:
        signer = load_signer(config_root=root, backend_name=resolve_backend_name(root))
    except KeyBackendError as exc:
        raise ReleaseSigningUnavailable() from exc
    if signer is None:
        raise ReleaseSigningUnavailable()
    try:
        signed = sign_message(signer, message, timeout=None)
    except KeyBackendError as exc:
        raise ReleaseSigningUnavailable() from exc
    finally:
        signer.close()
    return signed.sig, signed.key_id


def decoded_categories(row: Mapping[str, Any]) -> list[str]:
    """A row's stored categories as a list; damage reads as an impossible value."""

    raw = row.get("categories")
    try:
        value = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        value = None
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return ["<damaged>"]
    return sorted(value)


#: The decision fields a release must reproduce exactly (S-7 item 4).
_BOUND_FIELDS = (
    "conversation_id",
    "body_sha256",
    "envelope_digest",
    "hold_code",
    "screener_provider",
    "screener_model",
    "template_sha256",
)


def release_matches_decision(release: Mapping[str, Any], decision: Mapping[str, Any]) -> bool:
    """Whether a signed release still names THIS decision (S-7: re-check on consume)."""

    if any(str(release.get(f) or "") != str(decision.get(f) or "") for f in _BOUND_FIELDS):
        return False
    if decoded_categories(release) != decoded_categories(decision):
        return False
    return str(decision.get("hold_code") or "") in RELEASABLE_HOLD_CODES


def log_release(code: str) -> None:
    """ONE closed-code line per release event; no id, label or hash (DESIGN §2.2)."""

    logger.info("bridge: owner release code=%s", code)


class ReleaseSigningUnavailable(Exception):
    """No operator key could sign (absent, unreadable, or the gesture failed)."""


# -- the per-message inbox notice (DESIGN §2, S-10) ---------------------------
#
# CONTENT LAW: no body, sender, conversation, category or label in the item.
# The phone relay reads ``notifications.db``, so anything in the item can reach
# a lock screen, and a sender controls their own display name ("IT Security:
# click Deliver now") and any conversation name they create (S-10). Labels are
# shown only on the authenticated desktop card. Log lines carry closed codes.

#: Senders this process raised a banner for, monotonic stamp. In memory like
#: ``hold_alert._raised``: after a restart the worst case is one extra banner.
_last_banner: dict[tuple[str, str], float] = {}


def notice_key(endpoint_id: str, message_id: str) -> str:
    return f"team-bridge-held:{endpoint_id}:{message_id}"


def overflow_key(endpoint_id: str) -> str:
    return f"team-bridge-held-more:{endpoint_id}"


def _banner_priority(endpoint_id: str, sender: str, now: float) -> str:
    key = (endpoint_id, sender)
    last = _last_banner.get(key)
    if last is not None and now - last < BANNER_WINDOW_S:
        return "normal"
    _last_banner[key] = now
    if len(_last_banner) > 1024:
        _last_banner.clear()
    return "high"


def raise_notice(
    store: Any,
    owner: Any,
    *,
    config_dir: Path,
    session_id: str,
    message_id: str,
    monotonic: Any = time.monotonic,
) -> str:
    """Raise (once) the per-message notice for an eligible work-scope hold.

    Returns a closed code: ``raised`` | ``unchanged`` | ``ineligible`` |
    ``overflow`` | ``refused``. ``notified_at`` is written ONLY after
    ``notify`` returned, because the 60-minute alert skips a hold only once its
    own notice is recorded (DESIGN §2.4): a refused notice must leave the hold
    visible to the long-hold alert.
    """

    from local_operator.i18n.keys import cli_team_bridge_hold as _k
    from local_operator.i18n.text import tr
    from local_operator.notifications.store import NotificationStore

    decision = store.work_scope_decision(owner, message_id)
    if decision is None or not decision.get("eligible"):
        return "ineligible"
    if decision.get("notified_at") is not None:
        return "unchanged"
    t = tr("cli.team_bridge_hold")
    notifications = NotificationStore(config_dir)
    try:
        if store.standing_work_scope_notices(owner) >= STANDING_NOTICE_CAP:
            waiting = store.unnotified_work_scope_holds(owner)
            notifications.notify(
                session_id,
                title=t(_k.more_title()),
                body_md=t(_k.more_body(count=max(waiting, 1))),
                priority="normal",
                key=overflow_key(owner.endpoint_id),
            )
            logger.info("bridge: hold notice code=overflow")
            return "overflow"
        priority = _banner_priority(
            str(owner.endpoint_id), str(decision.get("sender_digest") or ""), monotonic()
        )
        notifications.notify(
            session_id,
            title=t(_k.held_title()),
            body_md=t(_k.held_body()),
            priority=priority,
            key=notice_key(owner.endpoint_id, message_id),
        )
    except Exception:  # noqa: BLE001 — a notice is chrome; the hold stands
        logger.info("bridge: hold notice code=refused")
        return "refused"
    try:
        store.mark_work_scope_notified(owner, message_id, now=time.time())
    except Exception:  # noqa: BLE001 — unrecorded: the 60-minute alert still counts it
        logger.info("bridge: hold notice code=unrecorded")
        return "refused"
    logger.info("bridge: hold notice code=raised")
    return "raised"


def clear_notice(
    store: Any, owner: Any, *, config_dir: Path, session_id: str, message_id: str, word: str
) -> None:
    """The keyed ``low`` update after Deliver/Discard (hold_alert's clearing pattern)."""

    from local_operator.i18n.keys import cli_team_bridge_hold as _k
    from local_operator.i18n.text import tr
    from local_operator.notifications.store import NotificationStore

    try:
        decision = store.work_scope_decision(owner, message_id)
        if decision is None or decision.get("notified_at") is None:
            return
        t = tr("cli.team_bridge_hold")
        title, body = (
            (_k.delivered_title(), _k.delivered_body())
            if word == "deliver"
            else (_k.discarded_title(), _k.discarded_body())
        )
        notifications = NotificationStore(config_dir)
        for item in notifications.find_by_key(notice_key(owner.endpoint_id, message_id)):
            notifications.notify(
                item["session_id"],
                title=t(title),
                body_md=t(body),
                priority="low",
                key=notice_key(owner.endpoint_id, message_id),
            )
        store.mark_work_scope_cleared(owner, message_id, now=time.time())
    except Exception:  # noqa: BLE001 — chrome; the decision already stands
        logger.info("bridge: hold notice code=clear_failed")


__all__ = [
    "DELIVER_WINDOW_S",
    "OWNER_RELEASE_SHA256",
    "RELEASABLE_CATEGORIES",
    "RELEASABLE_HOLD_CODES",
    "RELEASE_ELIGIBLE_SCREENERS",
    "SENDER_BUDGET",
    "WORK_SCOPE_HOLD",
    "converts",
    "eligible_screener",
    "release_payload",
    "verify_row",
]
