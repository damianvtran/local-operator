"""Explicit signed recovery of ONE revoked finite share, never future consent.

The normal approval envelope is deliberately reused. Its operation-specific what
must be dispatched before onboarding/signing; neither an old onboarding approval
nor the bare share verb authorizes removing an owner-local tombstone.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

from local_operator.network import approvals, identity, onboard_approvals, store
from local_operator.network.credentials import consent, finite_journal, placement
from local_operator.network.types import MeshRefusal

OP = "credential_finite_share"
STEP = "credential_finite_share"


#: The EXACT payload key set a finite share carries, and the exact top-level ``what`` keys.
#: ONE definition read by both :func:`disclosure` and :func:`_apply` (#3070 review r3,
#: R3-1): disclosure used to read only the fields it names, so an approver signed a digest
#: covering an extra, never-shown field that apply refused only afterwards.
_FIELDS = frozenset(
    {
        "owner_device",
        "network_id",
        "recipient_device",
        "public_key_fingerprint",
        "key",
        "scope",
        "denies",
        "policy_revision",
        "guard",
    }
)
_WHAT = frozenset({OP, "anchor"})


def intent(record: dict[str, Any]) -> dict[str, Any] | None:
    what = record.get("what")
    return what.get(OP) if isinstance(what, dict) and isinstance(what.get(OP), dict) else None


def exact_shape(record: Mapping[str, Any]) -> bool:
    """True only for the EXACT shape this flow mints: ``what`` is ``{OP, "anchor"}`` and
    the payload carries exactly :data:`_FIELDS`.

    The ONE key-set predicate behind :func:`disclosure`, :func:`_apply` and the generic
    approvers' refusal (``approval_ops.finite_generic_refusal``), so none of them can
    accept a record another would refuse. Values are validated by each consumer.
    """
    what = record.get("what")
    data = intent(dict(record))
    return (
        data is not None and isinstance(what, dict) and set(what) == _WHAT and set(data) == _FIELDS
    )


def _recipient_label(record: dict[str, Any], data: dict[str, Any]) -> str:
    """``name (device id)`` when the card recorded the member's name, else the id alone.

    The NAME leads (design D1): a non-technical reader decides from the first line, and the
    sibling inheritance card already opens the same way.
    """
    raw_device = record.get("device")
    device: dict[str, Any] = raw_device if isinstance(raw_device, dict) else {}
    name = _plain(device.get("name"))
    recipient = str(data.get("recipient_device", ""))
    return f"{name} ({recipient})" if name else recipient


def summary(record: dict[str, Any]) -> str | None:
    data = intent(record)
    if data is None:
        return None
    return (
        f"Restore sharing of {data.get('key', '')!r} with {_recipient_label(record, data)} "
        f"({data.get('scope', '')} scope); this clears only the earlier withdrawals the card "
        "names. "
        "No automatic future sign-ins, and nothing is installed, joined or trusted."
    )


#: Bumped when a field of :func:`disclosure` changes meaning. The desktop echoes the
#: disclosure's DIGEST, so a renderer built for an older layout cannot have a newer one
#: signed in its name: the digest it echoes would not be the one the server recomputes.
DISCLOSURE_VERSION = 1


def disclosure(record: dict[str, Any]) -> dict[str, Any] | None:
    """Every fact the operator must see BEFORE signing, as DATA, from the signed record alone.

    WHY A DICT AND NOT LINES. The CLI and the desktop must read ONE source, or the two
    surfaces can disclose different things about the same signature. The desktop localizes
    its labels (i18n: no English and no CLI command may travel in here); the CLI renders
    this dict in :func:`details`. Every value below is either an identifier/fingerprint
    copied from the digest-bound record or a fixed machine token, so the dict is a pure
    function of ``request_digest``-bound fields (plus the record's own id) and its digest
    is stable for the life of the request.

    Full escaped metadata, never silent truncation: every signed withdrawal tuple,
    revision and reference is listed, and a payload this build cannot validate raises
    ``finite_disclosure_invalid`` (fail closed) so nothing undisclosable is ever signed.
    ``None`` when the record is not a finite-share request.
    """
    data = intent(record)
    if data is None:
        return None
    # Never sign what this build cannot state. The signature binds the WHOLE record, so a
    # field the disclosure does not show -- an extra payload key, or a sibling of the op
    # under ``what`` -- would be signed blind. Same key sets as _apply, so the two cannot
    # disagree about which records are describable.
    if not exact_shape(record):
        consent.refuse("finite_disclosure_invalid")
    try:
        for name in ("owner_device", "network_id", "recipient_device"):
            store.safe_component(data[name])
        if (
            data["scope"] not in {"device", "session"}
            or not isinstance(data["key"], str)
            or not data["key"]
            or len(data["key"].encode()) > 200
            or any(ord(ch) < 32 or 127 <= ord(ch) <= 159 for ch in data["key"])
        ):
            consent.refuse("finite_disclosure_invalid")
        denies = data["denies"]
        if not isinstance(denies, list) or not 0 < len(denies) <= finite_journal.MAX_DENIES:
            consent.refuse("finite_disclosure_invalid")
        guard = data["guard"]
        if guard:
            consent._exact(guard, {"id", "revision"}, {"id", "revision"})
        withdrawals = []
        for deny in denies:
            consent._selector(deny["selector"])
            consent._revision(deny["revision"])
            consent._ref(deny["approval_ref"])
            withdrawals.append(
                {
                    name: copy.deepcopy(deny[name])
                    for name in ("selector", "revision", "approval_ref")
                }
            )
        fingerprint = str(data["public_key_fingerprint"])
        raw_device = record.get("device")
        device: dict[str, Any] = raw_device if isinstance(raw_device, dict) else {}
        anchor = (record.get("what") or {}).get("anchor")
        principal = str(anchor.get("key_id") or "") if isinstance(anchor, dict) else ""
        result = {
            "version": DISCLOSURE_VERSION,
            "operation": OP,
            "approval_id": str(record.get("approval_id") or ""),
            "credential_key": data["key"],
            "scope": data["scope"],
            "owner_device": data["owner_device"],
            "network_id": data["network_id"],
            "recipient": {
                "device_id": data["recipient_device"],
                # Display-only and bounded. A card filed before #3002 (or for a member with
                # no name) carries none, and the UI shows the device id alone then.
                "name": _plain(device.get("name")),
                "key_fingerprint": fingerprint,
            },
            "signing_principal": principal,
            "policy_revision": consent._revision(data["policy_revision"]),
            "retained_fence": copy.deepcopy(guard) if guard else None,
            "withdrawals": withdrawals,
            # TWO different clocks, kept apart on purpose: the request window bounds how
            # long THIS CARD can be approved and applied; the grant lifetime is how long
            # the restored share lasts once applied. A finite share has no expiry of its
            # own -- it lasts until revoked -- and conflating the two is how an operator
            # reads "one hour" as the life of the permission.
            "request_window": {
                "created_at": record.get("created_at"),
                "expires_at": record.get("expires_at"),
            },
            "grant_lifetime": {"kind": "until_revoked"},
            "future_logins": False,
            "installs_or_joins": False,
        }
        if len(json.dumps(result, ensure_ascii=True).encode("utf-8")) > finite_journal.LIMIT:
            consent.refuse("finite_disclosure_invalid")
        return result
    except (KeyError, TypeError, ValueError, AttributeError):
        consent.refuse("finite_disclosure_invalid")


def _plain(text: Any, limit: int = 64) -> str:
    """A display-only label: no control characters, bounded. Never used as an identifier."""
    clean = "".join(ch for ch in str(text or "") if ord(ch) >= 32 and not 127 <= ord(ch) <= 159)
    return clean[:limit]


def _when(value: Any) -> str:
    # The zone is named (design N2): the card is read on other machines of the mesh too.
    return (
        time.strftime("%Y-%m-%d %H:%M %Z", time.localtime(float(value)))
        if isinstance(value, (int, float)) and value
        else "its set time"
    )


def _apply_argv(data: dict[str, Any], approval_id: str) -> list[str]:
    return [
        "lop",
        "network",
        "credential",
        "share",
        str(data.get("key", "")),
        "--with",
        str(data.get("recipient_device", "")),
        "--network",
        str(data.get("network_id", "")),
        "--scope",
        str(data.get("scope", "session")),
        "--approval",
        approval_id,
    ]


def _window_line(record: dict[str, Any], expires_at: Any, lifetime_of: str) -> str:
    """The request-window line, true for the card's CURRENT state (design #3002 r2, D8).

    Only a card still open to a decision or an apply may say it "can be approved and applied
    until" a time: a failed card said that 14 rows above "this card can't be used again".
    """
    if record.get("state") in (approvals.STATE_REQUESTED, approvals.STATE_APPROVED):
        return (
            f"Request window: this card can be approved and applied until {_when(expires_at)}. "
            "That window only limits how long this request stays answerable; it is not the "
            f"lifetime of the {lifetime_of}."
        )
    return f"Request window: closed; this card is {record.get('state', '')}."


def details(record: dict[str, Any]) -> list[str]:
    """The CLI's rendering of :func:`disclosure` -- the SAME facts the desktop renders.

    ORDER (design D2): who gets access and for how long come first, then the evidence a
    careful reader checks (withdrawals, revision, fingerprint, network, principal). Every
    fact stays; only the order changed. Lines are logical: the CLI wraps them to the
    terminal (design D3), the desktop renders the dict instead.

    Raises (fail closed) for a record this build cannot describe, so the CLI refuses to
    sign what it cannot state.
    """
    facts = disclosure(record)
    heading = summary(record)
    data = intent(record)
    if facts is None or heading is None or data is None:
        return []
    recipient = facts["recipient"]
    label = (
        f"{recipient['name']} ({recipient['device_id']})"
        if recipient["name"]
        else recipient["device_id"]
    )
    count = len(facts["withdrawals"])
    lines = [
        heading,
        f"Recipient device: {label}",
        "Duration: until you revoke it. The restored share has no automatic expiry.",
        _window_line(record, facts["request_window"]["expires_at"], "share"),
        # One plain line above the JSON (design D5), so "clears only the withdrawals listed
        # below" is understandable without parsing ``owner-revoke:<ns>``.
        f"What this undoes: {count} earlier "
        + ("withdrawal" if count == 1 else "withdrawals")
        + f" of {facts['credential_key']!r} for this device, listed exactly below.",
    ]
    for number, deny in enumerate(facts["withdrawals"], 1):
        lines.append(f"Withdrawal {number}: " + json.dumps(deny, sort_keys=True, ensure_ascii=True))
    if facts["retained_fence"]:
        lines.append(
            "Recover retained fence: " + json.dumps(facts["retained_fence"], ensure_ascii=True)
        )
    lines += [
        f"Signed policy revision: {facts['policy_revision']}",
        f"Recipient key fingerprint: {recipient['key_fingerprint']}",
        f"Network: {facts['network_id']}",
        f"Signing principal: your operator key {facts['signing_principal']} on this machine",
    ]
    if len("\n".join(lines).encode("utf-8")) > finite_journal.LIMIT:
        consent.refuse("finite_disclosure_invalid")
    return lines


def next_lines(record: dict[str, Any]) -> list[str]:
    """What the operator does next with this card, per state (UX U1, U5).

    A ``failed`` card names WHY it stopped and the actual next step from its durable
    receipt (:func:`approval_copy.last_failure`) -- before this, ``show`` printed
    ``failed`` and nothing else, and the reason lived only in ``--json``. The terminal
    command rides beside the plain sentence, never instead of it.
    """
    import shlex

    from local_operator.network.credentials import approval_copy

    data = intent(record)
    if data is None:
        return []
    approval_id = str(record.get("approval_id", ""))
    state = record.get("state")
    apply_cmd = shlex.join(_apply_argv(data, approval_id))
    # COMMANDS ON THEIR OWN LINES (design #3002 r2, D6; UX U7; QA Q7): a command inside a
    # sentence got hard-wrapped with the prose, and the pasted rows ran as separate, broken
    # commands. The prose ends at its label; the command follows alone, and the CLI never
    # wraps a line that carries one (``network.cli._wrap_lines``).
    if state == approvals.STATE_REQUESTED:
        return ["Approving records your decision only. Then run:", apply_cmd]
    if state == approvals.STATE_APPROVED:
        return ["Approved, not applied yet. Run:", apply_cmd]
    failure = approval_copy.last_failure(record)
    if failure is None:
        return []
    lines = [f"Why it stopped: {failure['reason']}", f"What to do: {failure['remedy']}"]
    if failure["retryable"]:
        lines += ["Apply again:", apply_cmd]
    else:
        lines += [
            NEW_REQUEST_LABEL,
            *_new_request_commands(data),
        ]
    return lines


def list_lines(record: dict[str, Any]) -> list[str]:
    """The ``approvals list`` block, in the SAME shape as the inheritance card (UX U6, D4).

    A header, the one-line summary, a failed card's reason and next step, and the pointer
    to the full disclosure -- not the whole block, so a list of several cards stays
    scannable. ``approvals show`` and the pre-sign print keep every fact.
    """
    from local_operator.network.credentials import approval_copy

    try:
        heading = summary(record)
    except MeshRefusal:
        heading = "A credential-share request this build cannot describe; do not approve it."
    approval_id = record.get("approval_id", "")
    lines = [f"approval {approval_id} — {record.get('state', '')}", f"  {heading}"]
    failure = approval_copy.last_failure(record)
    if failure is not None:
        lines += [f"  Why it stopped: {failure['reason']}", f"  What to do: {failure['remedy']}"]
    lines.append(f"  Full details: lop network approvals show {approval_id}")
    return lines


def _request_command(data: dict[str, Any]) -> str:
    import shlex

    return shlex.join(
        [
            "lop",
            "network",
            "credential",
            "share",
            str(data.get("key", "")),
            "--with",
            str(data.get("recipient_device", "")),
            "--network",
            str(data.get("network_id", "")),
            "--scope",
            str(data.get("scope", "session")),
            "--request-approval",
        ]
    )


#: What follows the request command (UX #3002 r3, U12; design N5). The request prints the new
#: card's own Approve and Apply commands, with its real id, so the recipe never asks the
#: operator to fill a ``<new-approval-id>`` template by hand.
NEW_REQUEST_LABEL = (
    "To file the new request from a terminal on this Mac, run this. It prints the Approve and "
    "Apply commands for the new card:"
)


def _new_request_commands(data: dict[str, Any]) -> list[str]:
    """The terminal route after a permanent refusal: the request command, whole, alone."""
    return [_request_command(data)]


def _refusal(code: str, expected: dict[str, Any], approval_id: str) -> MeshRefusal:
    """One apply refusal, as every surface shows it (design D9; UX U8).

    THE SENTENCE is ``approval_copy.refusal_sentence``: the same reason ``approvals show``
    prints for the stored code, then the next step, with NO terminal syntax -- it is also the
    desktop's 409 ``text`` (operator a-b815: connections must be possible without a command).
    THE COMMANDS ride beside it on ``.commands`` (with their ``.commands_label``); only the
    CLI prints them, each on its own line, and ``--json`` carries them as a list.
    """
    import shlex

    from local_operator.network.credentials import approval_copy

    refusal = MeshRefusal(code, approval_copy.refusal_sentence(code))
    if approval_copy.retryable(code):
        # TRANSIENT, not a changed intent (QA #3002 round 1, Q3; UX U2): the same signed
        # card applies once the other writer is done.
        label = "Apply again:"
        commands = [shlex.join(_apply_argv(expected, approval_id))]
    else:
        label = NEW_REQUEST_LABEL
        commands = _new_request_commands(expected)
    refusal.commands_label = label  # type: ignore[attr-defined]
    refusal.commands = commands  # type: ignore[attr-defined]
    return refusal


def apply(approval_id: str, expected: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    try:
        _refuse_spent(approval_id, root)
        return _apply(approval_id, expected, root)
    except MeshRefusal as exc:
        raise _refusal(exc.code, expected, approval_id) from exc


def _refuse_spent(approval_id: str, root: Path | None) -> None:
    """Refuse a ``failed`` card whose last failure was PERMANENT, before any new run (UX U1).

    Such a card cannot succeed again -- the policy revision it signed only moves forward,
    a changed recipient key does not change back -- so re-running Apply used to add one
    more ``failed`` receipt per attempt and repeat the same refusal. It now answers at once
    with the stored code, so the caller's label and the "new request is needed" remedy are
    the same words every time. A transient last failure is let through to retry.
    """
    from local_operator.network.credentials import approval_copy

    try:
        record = approvals.load_record(approval_id, root=placement._config_root(root))
    except MeshRefusal:
        return  # the apply's own load names a missing card
    failure = approval_copy.last_failure(record)
    if failure is not None and not failure["retryable"] and failure["code"]:
        raise MeshRefusal(failure["code"], approval_copy.reason_for(failure["code"]))


def _reject_future_overlap(rec: dict[str, Any], key: str) -> None:
    if not rec.get("enabled"):
        return
    from local_operator.network.credentials.types import credential_key_for_mcp

    # Derive exact signed selector IDs/URLs using the existing trusted MCP key
    # constructor. Broad selectors cannot prove exclusion here; do NOT guess a
    # native class/account from a key's spelling or disable other permissions.
    selected: set[str] = set()
    for kind in ("providers", "mcp"):
        subset = rec["inherit"][kind]
        if not subset["future_logins"]:
            continue
        if "selector" in subset:
            raise MeshRefusal(
                "finite_future_scope_conflict",
                "This broad future-sign-in permission cannot prove this key excluded. "
                "Exact selector-exclusion recovery is not supported yet; no permission changed.",
            )
        if kind == "providers":
            selected.update(subset["ids"])
        else:
            selected.update(credential_key_for_mcp(url) for url in subset["urls"])
    if key in selected:
        raise MeshRefusal(
            "finite_future_scope_conflict",
            "Clearing this withdrawal would also reauthorize an existing future-sign-in selection. "
            "Exact selector-exclusion recovery is not supported yet; no permission changed.",
        )


def _audit(event: str, data: dict[str, Any], approval_id: str, root: Path | None) -> None:
    from local_operator.network.audit import AuditEvent, AuditLog

    log = AuditLog(root=root)
    try:
        log.record(
            AuditEvent(
                event=event,
                actor=data["owner_device"],
                subject=data["recipient_device"],
                detail={
                    "credential_key": data["key"],
                    "act": data["owner_device"],
                    "sub": data["recipient_device"],
                    "scope": data["scope"],
                    "approval_id": approval_id,
                },
            )
        )
    finally:
        log.close()


def _subject(
    network: str, owner: str, recipient: str, key: str, root: Path | None
) -> dict[str, Any]:
    for item in (network, owner, recipient):
        store.safe_component(item)
    own = identity.load(root)
    record = store.load(network, root)
    member = record.member(recipient)
    if (
        own is None
        or own.device_id != owner
        or record.self_device_id != owner
        or record.trust != "active"
        or member is None
        or not member.active
        or member.device_id != recipient
    ):
        consent.refuse("finite_subject_mismatch")
    from local_operator.network.relay import set_member_capabilities

    # Preview the existing capability policy on a COPY before authority writes.
    # A final normal CLI step cannot discover a refused admin source only after
    # a finite holder was restored and its tombstone removed.
    set_member_capabilities(copy.deepcopy(record), device_id=recipient, grant=["broker_credential"])
    public = base64.urlsafe_b64decode(member.public_key + "=" * (-len(member.public_key) % 4))
    if len(public) != 32:
        consent.refuse("scope_unknown")
    doc = placement.PlacementDocument.load(network, root, self_device=owner)
    entry = doc.entry(key)
    if entry is None or entry.owner_device != owner:
        consent.refuse("not_owner")
    return {
        "owner_device": owner,
        "network_id": network,
        "recipient_device": recipient,
        "public_key_fingerprint": "sha256:" + hashlib.sha256(public).hexdigest(),
        "key": key,
    }


def _complete_capability(data: dict[str, Any], root: Path | None) -> None:
    from local_operator.network.relay import set_member_capabilities

    # Revoke legitimately removes this capability when its last key goes away.
    # Complete the existing local capability producer under its write lock, not
    # the CLI's relay-first variant (no socket/native I/O under journal locks).
    # The caller (apply) already holds this record lock ahead of placement in
    # the declared order; this nested acquisition is a same-thread RLock
    # re-entry, never a first acquisition under placement (R4-1).
    target = store.record_path(data["network_id"], root)
    with store._write_lock(target):
        if (
            _subject(
                data["network_id"],
                data["owner_device"],
                data["recipient_device"],
                data["key"],
                root,
            )["public_key_fingerprint"]
            != data["public_key_fingerprint"]
        ):
            consent.refuse("finite_subject_mismatch")
        record = store.load(data["network_id"], root)
        set_member_capabilities(
            record, device_id=data["recipient_device"], grant=["broker_credential"]
        )
        store.save(record, root)
        readback = store.load(data["network_id"], root).member(data["recipient_device"])
        if (
            readback is None
            or not readback.active
            or "broker_credential" not in readback.capabilities
        ):
            consent.refuse("finite_capability_unconfirmed")


def request(
    network: str, owner: str, recipient: str, key: str, scope: str, root: Path | None = None
) -> dict[str, Any]:
    subject = _subject(network, owner, recipient, key, root)
    if scope not in {"device", "session"} or (key == "github" and scope != "device"):
        consent.refuse("finite_scope_mismatch")
    authority = consent.ConsentDocument(network, owner, root)
    journal = finite_journal.Journal(network, owner, root)
    with authority.locked(), journal.locked():
        row = authority.load()
        rec = row["recipients"].get(recipient, {})
        _reject_future_overlap(rec, key)
        denies = [
            copy.deepcopy(d)
            for d in rec.get("denies", [])
            if d["selector"]["credential_key"] == key
        ]
        guard = journal.pending(recipient, key)
        if guard:
            # Retained original withdrawals remain signed/disclosed even when
            # failed policy compensation left no corresponding consent rows.
            if not authority.path.exists():
                consent.refuse("finite_policy_unknown")
            for original in guard["denies"]:
                if original not in denies:
                    denies.append(copy.deepcopy(original))
        if not denies:
            consent.refuse("finite_supersede_unneeded")
        subject.update(
            scope=scope,
            denies=denies,
            policy_revision=row["revision"],
            guard={"id": guard["id"], "revision": guard["revision"]} if guard else None,
        )
    anchor = approvals.local_anchor_trio(root)
    if anchor is None:
        consent.refuse("approval_anchor_unavailable")
    now = time.time()
    # Current member metadata only: NO invented SSH host/target or new approval
    # kind. This envelope authorizes the exclusive finite operation, not setup.
    result = approvals.create_request(
        kind=approvals.KIND_DEVICE_ONBOARD,
        request_id=approvals.new_request_id(),
        requested_by={"device_id": owner, "session_id": "", "surface": "cli"},
        # The member's display name rides the digest-bound device block so the disclosure can
        # name the recipient (QA #3002 round 1, Q2); display-only, never an identifier.
        device={
            "device_id": recipient,
            "name": _plain(getattr(store.load(network, root).member(recipient), "name", "")),
            "fingerprint": subject["public_key_fingerprint"],
        },
        what={
            OP: subject,
            "anchor": {k: anchor[k] for k in ("key_id", "spki_fp", "statement_digest")},
        },
        credential_ref=None,
        created_at=now,
        expires_at=now + 3600,
        root=root,
    )
    _audit("credential.finite_requested", subject, result["approval_id"], root)
    return result


def _apply(approval_id: str, expected: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = placement._config_root(root)
    source = approvals.load_record(approval_id, root=root)
    if source.get("state") == approvals.STATE_CONNECTED:
        raise MeshRefusal(
            "approval_already_connected",
            "This finite key/scope approval already finished; "
            "replay cannot clear a later withdrawal. "
            "Request a new exact finite-share approval; no install or connection ran.",
        )
    data = intent(source)
    if data is None or set(source.get("what", {})) != _WHAT:
        consent.refuse("finite_intent_missing")
    # The SAME key sets disclosure() refuses on (exact_shape): apply can never accept a
    # payload the approver was not shown, nor disclosure state one apply would reject.
    consent._exact(data, set(_FIELDS), set(_FIELDS))
    for field in ("owner_device", "network_id", "recipient_device", "key", "scope"):
        if data[field] != expected.get(field):
            consent.refuse("finite_subject_mismatch")
    current = _subject(
        data["network_id"], data["owner_device"], data["recipient_device"], data["key"], root
    )
    if current["public_key_fingerprint"] != data["public_key_fingerprint"]:
        consent.refuse("finite_subject_mismatch")
    run = onboard_approvals.begin_run(approval_id, root=root, operation=OP)
    try:
        result = _transact(approval_id, data, current, run, root)
    except Exception as exc:
        # DEFECT B (2026-10-26). ``begin_run`` moved the record to ``connecting`` with this
        # process's pid as the run lease. A refusal inside the transaction (a policy
        # revision that moved -- ``consent_conflict`` -- or withdrawals that changed --
        # ``finite_denial_mismatch``) used to leave it there; in the long-lived desktop
        # daemon the pid stays alive, so ``run_is_stale`` never fires and every retry read
        # "run in flight" until the one-hour window closed. Mirror inheritance's apply:
        # release the run as retry-eligible ``failed``. The release runs AFTER the
        # transaction's locks are dropped (``mark_failed`` takes the record flock, which a
        # second descriptor in this same process would wait on forever).
        _release_run(approval_id, run.run_id, getattr(exc, "code", type(exc).__name__), root)
        raise
    _audit("credential.finite_applied", data, approval_id, root)
    return result


def _release_run(approval_id: str, run_id: str, detail: str, root: Path | None) -> None:
    """``connecting -> failed`` for THIS run only; a no-op once the record moved on.

    A refusal raised after ``_mark_connected_locked`` (the post-publish expiry/subject
    re-checks) leaves a ``connected`` record whose compensation already ran; that record is
    terminal and must not be rewritten here. A record another run superseded is that run's.
    """
    try:
        record = approvals.load_record(approval_id, root=root)
        raw_run = record.get("run")
        run: dict[str, Any] = raw_run if isinstance(raw_run, dict) else {}
        if record.get("state") != approvals.STATE_CONNECTING or run.get("run_id") != run_id:
            return
        approvals.mark_failed(approval_id, run_id=run_id, step=STEP, detail=detail, root=root)
    except (MeshRefusal, OSError):
        # The original refusal is the one the caller must see; a release that lost a race
        # (a concurrent deny, a supersede) leaves the record in the state that race chose.
        pass


def _transact(
    approval_id: str,
    data: dict[str, Any],
    current: dict[str, Any],
    run: Any,
    root: Path,
) -> dict[str, Any]:
    """The locked finite transaction, run inside an open run lease (see ``_apply``)."""
    authority = consent.ConsentDocument(data["network_id"], data["owner_device"], root)
    projection_path = placement.placement_path(data["network_id"], root)
    journal = finite_journal.Journal(data["network_id"], data["owner_device"], root)
    # Writers below are existing factored LOCKED entrypoints. No await/network,
    # SSH/onboarding/copy work or reentrant record flock occurs in this transaction.
    #
    # LOCK ORDER (R4-1): the network record lock is taken BEFORE the placement
    # document lock and held across the transaction, matching relay pairing
    # admission (record -> placement, relay.py store.mutate -> placement.mutate).
    # _complete_capability later re-enters this same record lock from the same
    # thread (threading.RLock, store._write_lock) while placement is held; that
    # re-entry is safe only because the outer acquisition already ordered
    # record -> placement. Taking record for the first time under placement
    # (the previous shape) deadlocked against a concurrent pairing admission:
    # each thread held one lock and waited, unbounded, for the other.
    with (
        approvals._record_lock(approval_id, root),
        authority.locked(),
        journal.locked(),
        store._write_lock(store.record_path(data["network_id"], root)),
        placement._document_lock(projection_path, root=root),
    ):
        admitted = consent._admitted(approval_id, run.run_id, root)
        fresh = _subject(
            data["network_id"], data["owner_device"], data["recipient_device"], data["key"], root
        )
        if fresh != current:
            consent.refuse("finite_subject_mismatch")
        row = authority.load()
        if row["revision"] != data["policy_revision"]:
            consent.refuse("consent_conflict")
        rec = row["recipients"].get(data["recipient_device"])
        _reject_future_overlap(rec or {}, data["key"])
        actual = (
            [d for d in rec.get("denies", []) if d["selector"]["credential_key"] == data["key"]]
            if rec
            else []
        )
        guarded = journal.pending(data["recipient_device"], data["key"])
        retained = copy.deepcopy(actual)
        if guarded:
            for original in guarded["denies"]:
                if original not in retained:
                    retained.append(copy.deepcopy(original))
        if not retained or retained != data["denies"]:
            consent.refuse("finite_denial_mismatch")
        # No mutable holder/policy operation occurs until PENDING is durable,
        # fsynced and read back. A fresh signed request must bind the old fence.
        permit = journal._begin(data, approval_id, run.run_id, admitted["request_digest"])
        old_row = copy.deepcopy(row)
        projection = placement.PlacementDocument.load(
            data["network_id"], root, self_device=data["owner_device"]
        )
        old_projection = store._read_json(projection_path)
        try:
            # Keep deny effective while the finite projection/execution proof is
            # committed. Authority removal is LAST; no unverified receipt can
            # leave an accessible stale holder. Other keys' tombstones survive.
            entry = projection.grant(
                data["key"], data["recipient_device"], scope=data["scope"], by=data["owner_device"]
            )
            projection.save()
            checked = placement.PlacementDocument.load(data["network_id"], root).entry(data["key"])
            if checked is None or not checked.is_holder(data["recipient_device"]):
                consent.refuse("finite_write_unconfirmed")
            consent._admitted(approval_id, run.run_id, root)
            approvals._append_receipt_locked(
                approval_id,
                run_id=run.run_id,
                step=STEP,
                ok=True,
                detail="EXECUTED: exact finite-share permission; no future inheritance",
                digest=admitted["request_digest"],
                root=root,
            )
            approvals._mark_connected_locked(
                approval_id,
                run_id=run.run_id,
                step=STEP,
                detail=(
                    "Finite projection confirmed; apply must confirm withdrawal removal; "
                    "no automatic future-sign-in inheritance enabled"
                ),
                root=root,
            )
            if time.time() >= float(admitted["expires_at"]):
                consent.refuse("approval_expired")
            if (
                _subject(
                    data["network_id"],
                    data["owner_device"],
                    data["recipient_device"],
                    data["key"],
                    root,
                )
                != current
            ):
                consent.refuse("finite_subject_mismatch")
            rec["denies"] = [d for d in rec["denies"] if d not in actual]
            authority.save(row, data["policy_revision"])
            checked_policy = authority.load()
            if checked_policy != row:
                consent.refuse("finite_write_unconfirmed")
            checked = placement.PlacementDocument.load(data["network_id"], root).entry(data["key"])
            if checked is None or not checked.is_holder(data["recipient_device"]):
                consent.refuse("finite_write_unconfirmed")
            completed = approvals._load_raw(approval_id, root)
            if (
                completed["state"] != approvals.STATE_CONNECTED
                or completed["request_digest"] != admitted["request_digest"]
                or completed["run"]["run_id"] != run.run_id
                or time.time() >= float(completed["expires_at"])
                or not any(
                    receipt["step"] == STEP
                    and receipt["run_id"] == run.run_id
                    and receipt["ok"]
                    and receipt["digest"] == admitted["request_digest"]
                    for receipt in completed["receipts"]
                )
            ):
                consent.refuse("finite_write_unconfirmed")
            # Verify the same current member/key/capability after every publish;
            # connected/source receipts ALONE cannot clear a later fence.
            if (
                _subject(
                    data["network_id"],
                    data["owner_device"],
                    data["recipient_device"],
                    data["key"],
                    root,
                )
                != current
            ):
                consent.refuse("finite_subject_mismatch")
            _complete_capability(data, root)
            if (
                _subject(
                    data["network_id"],
                    data["owner_device"],
                    data["recipient_device"],
                    data["key"],
                    root,
                )
                != current
            ):
                consent.refuse("finite_subject_mismatch")
            if time.time() >= float(completed["expires_at"]):
                consent.refuse("approval_expired")
            journal._finish(permit)
            result = entry.to_json()
        except (MeshRefusal, OSError, ValueError):
            # Original valid tombstones are conservative even if restoring the
            # projection fails; do not report an incomplete prepared operation
            # as success or drop denial to fit bounds. CAS stays forward-only.
            try:
                now_row = authority.load()
                old_row["revision"] = now_row["revision"]
                authority.save(old_row, now_row["revision"])
            except (MeshRefusal, OSError, ValueError):
                # If the policy filesystem cannot accept compensation after its
                # last publish, persist a WITHHELD projection instead. Never
                # restore a formerly eligible holder with unconfirmed denies.
                withheld = placement.PlacementDocument.load(data["network_id"], root)
                withheld.revoke(data["key"], data["recipient_device"], by=data["owner_device"])
                withheld.save()
                checked = placement.PlacementDocument.load(data["network_id"], root).entry(
                    data["key"]
                )
                if checked is not None and checked.is_holder(data["recipient_device"]):
                    consent.refuse("finite_compensation_unconfirmed")
                consent.refuse("finite_projection_withheld")
            try:
                # Rollback is restoration, not a newly observed projection:
                # preserve its prior timestamp/revision and full document under
                # the existing private atomic writer while its flock is held.
                store._write_private_json(projection_path, old_projection)
                if store._read_json(projection_path) != old_projection:
                    consent.refuse("finite_projection_withheld")
            except OSError:
                # Persisted original deny still withholds any partial holder.
                consent.refuse("finite_projection_withheld")
            consent.refuse("finite_recovery_failed")
    return result
