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
* ONLY when the trusted answering hop is PREFERRED and its identity is on
  :data:`RELEASE_ELIGIBLE_SCREENERS`,
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
put: an identical later message (same body and envelope digest) is held again
with no model call, and gets its OWN decision row copied from the first and
charged to the sender budget (``screening._copy_work_scope_decision``), so it
is the person's decision again — with buttons and a notice, budget permitting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import struct
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from local_operator.team_bridge.model_boundary import SCREENING_EVALUATED_MODELS

logger = logging.getLogger(__name__)

#: The ONE hold code a person may release. A closed set of one: a hold code
#: added later is not releasable until a reviewed change says so (D1).
WORK_SCOPE_HOLD = "work_scope_hold"

#: Cached ANSWER holds the receiving person may REVIEW (``team_bridge.review_held``,
#: default on; operator report 2026-11-09). Each is a model answer the answer cache
#: replays forever so a re-draw cannot shop for ``deliver`` — which also meant it
#: had no exit. Reviewing one releases nothing by itself: Deliver/Discard is the
#: person's signed word through the same verified release path as a work-scope
#: hold. Unlike the work-scope class, eligibility needs no preferred screener and
#: spends no sender budget: those gates exist because a work-scope conversion
#: TRUSTS the screener's category to offer a release, while a review hold trusts
#: only the person, and a hold with no exit is the failure this closes.
REVIEW_HOLD_CODES = frozenset(
    {"model_hold", "model_refused", "reply_truncated", "bounce_without_category"}
)
RELEASABLE_HOLD_CODES = frozenset({WORK_SCOPE_HOLD}) | REVIEW_HOLD_CODES

#: The ONLY category tuple that may become a releasable hold (Amendment 1, D2).
RELEASABLE_CATEGORIES = ("new_work_request",)

#: Identity compatibility export; preference itself must come from the trusted
#: answering hop. The reviewed producer's table is the single source (DESIGN §3,
#: corrected Amendment 2 / #3444), including concrete Radient transport IDs.

RELEASE_ELIGIBLE_SCREENERS = frozenset(SCREENING_EVALUATED_MODELS)

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

#: The screener verdict a decision row records (code review #3443 m3): a
#: Discard and a setting-off settlement must reproduce what the screener
#: actually said, and only these two verdicts are ever converted.
SCREENER_VERDICTS = ("hold", "bounce")


def eligible_screener(provider: str | None, model: str | None) -> bool:
    """Whether the hop that answered may make a releasable decision (D4)."""

    from local_operator.team_bridge.model_boundary import is_evaluated_screening_model

    return is_evaluated_screening_model(str(provider or ""), str(model or ""))


def preferred_answer(provider: str | None, model: str | None, preferred: bool) -> bool:
    """A trusted preferred-stage answer, bound to its answering identity (a-3f46).

    DESIGN §3 / corrected Amendment 2: preferred Opus after Sonnet exhaustion
    qualifies. The receiver's ordinary default/fallback Opus does not. The caller
    must obtain ``preferred`` from the trusted chain's answering hop, never from
    a model name, a bare-runner attribute or the hop index. Missing provenance
    stays False. Reuse #3444's evaluated-pair table instead of a second allowlist.
    """

    from local_operator.team_bridge.model_boundary import is_evaluated_screening_model

    return preferred is True and is_evaluated_screening_model(str(provider or ""), str(model or ""))


# -- the release-only category check (Amendment 2; security review #3443 S-2) --
#
# ``parse_verdict``'s exact-token rule stays the screening rule (setting-off
# bytes depend on it). A RELEASE needs more: the REASON is model prose, and in a
# v2 hold a category token is incidental, so a second category named in words
# ("…or social engineering"), a hyphenated or capitalised spelling, a negated
# mention or a hidden format character would all pass the token rule. This
# check only ever REFUSES; it never makes something releasable that the token
# rule did not already name.

#: Prose stems that name another screening category or an injection/extraction
#: class. Matched as substrings of the NORMALISED reason (casefolded, runs of
#: whitespace/dashes as ``_``), so ``Prompt-Injection`` and ``prompt injection``
#: both hit ``prompt_inject``. Closed; an addition is a reviewed change.
_OTHER_CLASS_STEMS = (
    "prompt_inject",
    "inject",
    "social_engineer",
    "phish",
    "impersonat",
    "malicious",
    "malware",
    "credential",
    "password",
    "secret",
    "token",
    "api_key",
    "confidential",
    "personal_info",
    "private_data",
    "surveil",
    "exfiltrat",
    "harass",
    "spam",
    "scam",
    "fraud",
    "manipulat",
    "pressure_tactic",
    "out_of_scope",
    "jailbreak",
)

#: Words that negate the category when they precede it closely.
_NEGATORS = frozenset(
    {"not", "no", "isnt", "arent", "wasnt", "never", "nor", "neither", "without", "non"}
)
_NEGATION_WINDOW = 3


def _normalised_reason(reason: str) -> str:
    import re
    import unicodedata

    text = unicodedata.normalize("NFKC", reason).casefold()
    text = text.replace("\u2019", "").replace("'", "")
    return re.sub(r"[\s\-\u2010-\u2015]+", "_", text)


def strict_release_reason(reason: str) -> bool:
    """Whether a converted decision's REASON passes the release-only check.

    Refuses (``False``) when the reason carries any Unicode format character
    (zero-width and bidi controls never appear in an honest verdict), when it
    names any other screening category or a prose synonym of one, when the
    ``new_work_request`` mention is negated within three words, or when it does
    not name ``new_work_request`` at all once normalised.
    """

    import re
    import unicodedata

    from local_operator.team_bridge.types import SCREENING_CATEGORIES

    if not isinstance(reason, str) or not reason:
        return False
    if any(unicodedata.category(char) == "Cf" for char in reason):
        return False
    text = _normalised_reason(reason)
    target = RELEASABLE_CATEGORIES[0]
    if target not in text:
        return False
    if any(category in text for category in SCREENING_CATEGORIES if category != target):
        return False
    remainder = text.replace(target, "_")
    if any(stem in remainder for stem in _OTHER_CLASS_STEMS):
        return False
    for match in re.finditer(re.escape(target), text):
        before = [word for word in re.split(r"[^a-z0-9]+", text[: match.start()]) if word]
        if _NEGATORS.intersection(before[-_NEGATION_WINDOW:]):
            return False
    return True


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


def review_enabled(config_dir: Path | str) -> bool:
    """``team_bridge.review_held``: ON unless the file says a real ``false``."""

    from local_operator.config import read_config_values
    from local_operator.team_bridge.settings import (
        DEFAULT_REVIEW_HELD,
        TeamBridgeSettings,
    )

    values = read_config_values(Path(config_dir))
    if values is None:
        return DEFAULT_REVIEW_HELD
    return TeamBridgeSettings.from_values(values).review_held


def enabled_for(config_dir: Path | str, hold_code: str | None) -> bool:
    """Is the release path open for a decision/release of THIS hold code?

    Review codes follow ``review_held``; the work-scope code (and anything
    else) follows ``hold_release`` exactly as before.
    """

    if hold_code in REVIEW_HOLD_CODES:
        return review_enabled(config_dir)
    return enabled(config_dir)


def any_enabled(config_dir: Path | str) -> bool:
    """Whether ANY release path is open (the desktop list's ``enabled`` flag)."""

    return enabled(config_dir) or review_enabled(config_dir)


def trusted_decision(decision: Mapping[str, Any]) -> bool:
    """The screener-provenance gate a release needs for THIS decision's class.

    A work-scope decision must come from a preferred screener (Amendment 2); a
    review decision trusts only the person's signed word (see REVIEW_HOLD_CODES).
    """

    code = str(decision.get("hold_code") or "")
    return code in REVIEW_HOLD_CODES or bool(decision.get("screener_preferred"))


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
    screener_verdict: str,
    screener_hop: int,
    screener_preferred: bool,
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
        # Amendment 2 (S-1, m3): what the screener said and which hop said it,
        # so the record can never be presented for a fallback or a re-labelled
        # verdict.
        + _lp(screener_verdict)
        + _lp(str(int(screener_hop)))
        + _lp("1" if screener_preferred else "0")
        + _lp(word)
        + _lp(f"{float(decided_at):.6f}")
    )


def _int_or(value: Any, default: int) -> int:
    """``value`` as an int, ``default`` only when ABSENT (code review n1: ``0`` is a value)."""

    return default if value is None else int(value)


def _float_or(value: Any, default: float) -> float:
    return default if value is None else float(value)


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
        generation=_int_or(row.get("creator_generation"), -1),
        conversation_id=str(row.get("conversation_id") or ""),
        message_id=str(row.get("message_id") or ""),
        body_sha256=str(row.get("body_sha256") or ""),
        envelope_digest=str(row.get("envelope_digest") or ""),
        hold_code=str(row.get("hold_code") or ""),
        screener_provider=str(row.get("screener_provider") or ""),
        screener_model=str(row.get("screener_model") or ""),
        template_sha256=str(row.get("template_sha256") or ""),
        categories=[str(c) for c in categories],
        screener_verdict=str(row.get("screener_verdict") or ""),
        screener_hop=_int_or(row.get("screener_hop"), -1),
        screener_preferred=bool(row.get("screener_preferred")),
        word=str(row.get("word") or ""),
        decided_at=_float_or(row.get("decided_at"), 0.0),
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
    "screener_verdict",
    "screener_hop",
    "screener_preferred",
)


def release_matches_decision(release: Mapping[str, Any], decision: Mapping[str, Any]) -> bool:
    """Whether a signed release still names THIS decision (S-7: re-check on consume)."""

    if any(str(release.get(f) or "") != str(decision.get(f) or "") for f in _BOUND_FIELDS):
        return False
    if decoded_categories(release) != decoded_categories(decision):
        return False
    if not trusted_decision(decision):
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
    ``badge_only`` (a review hold: no inbox item) | ``overflow`` | ``refused``.
    ``notified_at`` is written ONLY after ``notify`` returned, because the
    60-minute alert skips a hold only once its own notice is recorded (DESIGN
    §2.4): a refused notice must leave the hold visible to the long-hold alert.
    """

    from local_operator.i18n.keys import cli_team_bridge_hold as _k
    from local_operator.i18n.text import tr
    from local_operator.notifications.store import NotificationStore

    decision = store.work_scope_decision(owner, message_id)
    if decision is None or not decision.get("eligible"):
        return "ineligible"
    if str(decision.get("hold_code") or "") in REVIEW_HOLD_CODES:
        # A review hold raises NO inbox item — neither the per-message notice nor
        # the overflow (operator report 2026-11-10). #3714 (a79c82fb) routed every
        # review hold through this notice, and with ``model_hold`` & co. common
        # that wrote one row per held message (~28 in minutes, the first per
        # sender at ``high``) for messages the Agent messages sidebar badge and
        # holds list already show. The review row itself (Deliver/Discard, the
        # badge count) is written by the screener and is untouched here.
        # ``notified_at`` stays NULL on purpose: the hold remains visible to the
        # ONE keyed 60-minute long-hold alert, exactly as before #3714, so a
        # review hold is never invisible to every alert.
        logger.info("bridge: hold notice code=badge_only")
        return "badge_only"
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


# -- retiring the review-hold notices #3714 wrote (operator report 2026-11-10) --
#
# WHY. Before ``raise_notice`` stopped writing them, every review hold left one
# inbox row (and, past the cap, the overflow row) that nothing would ever take
# down short of a Deliver/Discard per message. Stopping the producer does not
# clear an inbox that is already flooded, so the connector runs this once per
# process per endpoint (``connector.loop``), with or without a new hold.
#
# SELECTORS, and nothing wider. A row is dismissed only when its key is
# EXACTLY ``notice_key(endpoint, message_id)`` for a live-generation decision
# row whose ``hold_code`` is a review code and whose ``notified_at`` is set —
# i.e. a notice this producer recorded writing. A work-scope hold's notice uses
# the same key shape and is never touched, nor is any other key. The overflow
# row (``overflow_key(endpoint)``) is shared with the work-scope class, so it is
# dismissed only once the standing count is back under the cap, which is when
# its "N more" claim stopped being true anyway.
#
# DISMISS, NEVER DELETE. Rows stay readable under the dismissed filter, filed
# ``dismissed_via=REVIEW_NOTICE_RETIRED_VIA`` so they read as retired, not as
# the operator's own dismissal; a row the operator already dismissed is left
# exactly as they filed it. The decision row is only stamped ``cleared_at`` (its
# notice is gone), which frees the per-endpoint cap: Deliver/Discard, the
# holds list and the sidebar badge read the decision and the hold, never this.
# Idempotent: a second pass finds nothing live and nothing uncleared.

#: ``dismissed_via`` on a review-hold notice this module retired.
REVIEW_NOTICE_RETIRED_VIA = "bridge_review_retired"

#: Endpoints this process has already retired notices for (in memory like
#: ``_last_banner``): after a restart the pass runs once more and is a no-op.
_retired_endpoints: set[str] = set()


def retire_review_notices(store: Any, owner: Any, *, config_dir: Path) -> dict[str, int]:
    """Dismiss the review-hold inbox rows #3714 wrote; return closed counts.

    ``dismissed`` per-message rows, ``overflow`` rows, ``cleared`` decision
    rows. Raises on store errors; :func:`retire_review_notices_once` is the
    never-raising caller.
    """

    from local_operator.notifications.store import NotificationStore

    notifications = NotificationStore(config_dir)
    endpoint = str(owner.endpoint_id)
    out = {"dismissed": 0, "overflow": 0, "cleared": 0}

    def dismiss_live(key: str) -> int:
        live = [
            (str(item["id"]), int(item["revision"]))
            for item in notifications.find_by_key(key)
            if not item.get("dismissed")
        ]
        if not live:
            return 0
        result = notifications.apply("dismiss", ids=live, via=REVIEW_NOTICE_RETIRED_VIA)
        return len(result.changed)

    for decision in store.work_scope_decisions(owner):
        if str(decision.get("hold_code") or "") not in REVIEW_HOLD_CODES:
            continue
        if decision.get("notified_at") is None:
            continue
        message_id = str(decision["message_id"])
        out["dismissed"] += dismiss_live(notice_key(endpoint, message_id))
        if decision.get("cleared_at") is None:
            store.mark_work_scope_cleared(owner, message_id, now=time.time())
            out["cleared"] += 1
    if store.standing_work_scope_notices(owner) < STANDING_NOTICE_CAP:
        out["overflow"] = dismiss_live(overflow_key(endpoint))
    return out


def retire_review_notices_once(owner: Any, config_dir: Path) -> str:
    """Run :func:`retire_review_notices` once per endpoint per process; never raises."""

    from local_operator.team_bridge.store import Store

    endpoint = str(getattr(owner, "endpoint_id", ""))
    if endpoint in _retired_endpoints:
        return "done"
    try:
        counts = retire_review_notices(Store(config_dir), owner, config_dir=config_dir)
    except Exception:  # noqa: BLE001 — chrome; retried on the next process start
        logger.info("bridge: review notice retire code=failed")
        return "failed"
    _retired_endpoints.add(endpoint)
    logger.info(
        "bridge: review notice retire code=ok dismissed=%d overflow=%d cleared=%d",
        counts["dismissed"],
        counts["overflow"],
        counts["cleared"],
    )
    return "ok"


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
