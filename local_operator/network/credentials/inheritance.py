"""Operator authoring of ONE signed ``credential_inheritance`` permission (CLI only).

WHY THIS MODULE EXISTS. ``consent.provision_from_run`` records "this recipient may borrow
my future native OAuth logins", ``consent.verified_permission`` re-verifies it, and
``reconcile`` projects it every 15 s -- but nothing a person or agent could RUN ever
authored the signed record. This is that entry point, built on the same envelope the
finite-share recovery uses (``finite_share``): a ``device_onboard`` approval whose
``what`` carries ``{credential_inheritance: ..., anchor: ...}`` and NO install/connect
scope, so approving it signs exactly one thing and installs/joins/connects nothing.

THE FLOW (each step is a real, separately-gated entrypoint, never a shortcut):

1. ``request``  -- an agent or operator files the card. Nothing is signed or written but
   the request record; an agent can ask, it can never approve.
2. ``lop network approvals approve <id>`` -- the operator's own presence-gated signature.
   :func:`details` is printed to stderr BEFORE the signer runs.
3. ``apply``    -- ``begin_run`` -> ``consent.provision_from_run`` -> ``mark_connected``.
   One-shot per approval: the card ends ``connected`` (not a lying ``connecting`` that
   folds to ``expired`` after its 1 h window, though the permission itself survives).
4. ``status``   -- read-only, disk-derived; reports PERMISSION state, never delivery.

WHAT IT DECIDES (each a constraint, not an omission):

* Permission lifetime is the foundation's: UNTIL REVOKED. The 1 h request window only
  bounds how long the CARD may be answered. Both are said plainly, side by side, so
  neither is read as the other. No expiry, no wildcard (``eligible-owner-native``), no
  MCP, no ``approved-owner-servers`` is invented here.
* One explicit provider id, native OAuth only, bound to the exact provider ACCOUNT (and
  optionally org) read from this device's own inventory, and to the operator key that
  signs. A copied, borrowed, device-bound, static-key or account-less row is never a
  source (``reconcile.inventory`` rows; ``provider_ok``).
* ``broker_credential`` (the dial authority) is a PRECONDITION on both member rows, never
  an output: a missing one is refused with the exact ``reconcile.transport_gap`` remedy
  before any write. A signed approval of credentials is not a signed approval of dial
  authority.
* A sticky withdrawal wins. A recipient whose provider was revoked, or that has a retained
  finite-recovery fence, cannot be re-granted here: that needs its own specific
  superseding consent, which this slice does not offer (it never clears a deny).
* No credential material, no hash of one, no email, is read, stored, signed or printed:
  the record carries ids and public fingerprints only.
"""

from __future__ import annotations

import base64
import hashlib
import re
import shlex
import time
from pathlib import Path
from typing import Any

from local_operator.network import approvals, identity, store
from local_operator.network.credentials import consent, placement, reconcile
from local_operator.network.types import MeshRefusal

OP = "credential_inheritance"

#: The one ``what`` shape this authoring flow mints. Anything else carrying the key (a
#: supersede, a wildcard, an MCP selector, extra scope) is a record this build cannot
#: DESCRIBE, so it is a record this build will not sign, apply or render as approvable.
_ANCHOR_KEYS = {"key_id", "spki_fp", "statement_digest"}
_FINGERPRINT = re.compile(r"sha256:[0-9a-f]{64}\Z")

#: A state in which a request is still answerable/applicable (the window folded by
#: ``approvals.presented``). Used to find an already-open request for the same grant.
_OPEN_STATES = frozenset({approvals.STATE_REQUESTED, approvals.STATE_APPROVED})


def intent(record: dict[str, Any]) -> dict[str, Any] | None:
    what = record.get("what")
    return what.get(OP) if isinstance(what, dict) and isinstance(what.get(OP), dict) else None


def _config(root: Path | None) -> Path:
    """The config root the inventory lives under: the explicit one, else the ambient one.

    Every other reader in this module takes ``root=None`` to mean "the ambient root"
    (``store``/``placement``/``approvals``); the inventory needs a concrete path to open
    ``auth.db``, so it is resolved here rather than treated as "no inventory".
    """
    if root is not None:
        return Path(root)
    from local_operator.paths import config_dir

    return config_dir()


def _plain(text: Any, limit: int = 64) -> str:
    """A display-only label: no control characters, bounded. Never used as an identifier."""
    clean = "".join(ch for ch in str(text or "") if ord(ch) >= 32 and not 127 <= ord(ch) <= 159)
    return clean[:limit]


def _refuse(code: str, message: str) -> MeshRefusal:
    return MeshRefusal(code, message)


def _fingerprint(member: Any) -> str:
    try:
        public = base64.urlsafe_b64decode(member.public_key + "=" * (-len(member.public_key) % 4))
    except (ValueError, TypeError):
        raise _refuse(
            "inherit_subject_mismatch", "the recipient's stored key is unreadable"
        ) from None
    if len(public) != 32:
        raise _refuse("inherit_subject_mismatch", "the recipient's stored key has the wrong length")
    return "sha256:" + hashlib.sha256(public).hexdigest()


def _subject(network: str, owner: str, recipient: str, root: Path | None) -> dict[str, Any]:
    """The current owner/network/recipient triple and the recipient's key fingerprint.

    Reads only. Mirrors the checks ``consent.resolve_approved_scope`` repeats at apply, so a
    request that could never be applied is refused before an operator is asked to sign it.
    """
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
    ):
        raise _refuse(
            "inherit_subject_mismatch",
            "this device is not the owner of that active network, so it cannot author a "
            "credential permission there",
        )
    if (
        member is None
        or not member.active
        or member.lifecycle != "active"
        or member.removed_at
        or member.device_id != recipient
    ):
        raise _refuse(
            "inherit_subject_mismatch", "the recipient is not an active member of that network"
        )
    return {
        "record": record,
        "member": member,
        "owner_device": owner,
        "network_id": network,
        "recipient_device": recipient,
        "public_key_fingerprint": _fingerprint(member),
    }


def _require_transport(subject: dict[str, Any]) -> None:
    gap = reconcile.transport_gap(
        subject["record"], subject["network_id"], subject["recipient_device"]
    )
    if gap:
        raise _refuse(
            "inherit_transport_not_ready",
            "Not ready, and nothing was changed: "
            + gap.removeprefix("notready:")
            + ". A signed credential permission does not grant dial authority; grant it "
            "separately (it needs its own explicit yes), then try again.",
        )


def _native_rows(provider: str, root: Path | None) -> list[reconcile.NativeRow]:
    """This device's own, native, OAuth, enabled rows for ``provider`` (metadata only)."""
    inventory = reconcile.inventory(_config(root))
    if inventory is None:
        raise _refuse(
            "inherit_inventory_unreadable",
            "this device's login store could not be read (missing, locked or legacy); "
            "nothing was changed",
        )
    return [row for row in inventory.get(provider, []) if row.oauth and not row.copied]


def _eligible_providers(root: Path | None) -> list[str]:
    inventory = reconcile.inventory(_config(root))
    if inventory is None:
        return []
    return sorted(
        name
        for name, rows in inventory.items()
        if reconcile.provider_ok(name) and any(r.oauth and not r.copied for r in rows)
    )


def _choose_account(
    provider: str, account: str, org: str, rows: list[reconcile.NativeRow], root: Path | None
) -> tuple[str, str]:
    """Pick the exact (account_id, org_id) the permission will be bound to. Never guesses."""
    if not rows:
        known = _eligible_providers(root)
        raise _refuse(
            "inherit_no_native_login",
            f"This device has no native sign-in for {provider!r} to inherit from; nothing was "
            f"changed. Sign in to {provider} on this device first"
            + (f" (native logins here: {', '.join(known)})" if known else "")
            + ".",
        )
    if any(not r.marker for r in rows):
        raise _refuse(
            "inherit_login_ambiguous",
            f"A {provider} login on this device predates login stamping, so it cannot be "
            "matched to an account exactly. Sign in again, then retry; nothing was changed.",
        )
    accounts = sorted({r.context["account_id"] for r in rows if r.context.get("account_id")})
    if not accounts:
        raise _refuse(
            "inherit_account_unknown",
            f"The {provider} login on this device reports no account id, and a permission "
            "must name the exact account it covers. Nothing was changed.",
        )
    if not account:
        if len(accounts) > 1:
            raise _refuse(
                "inherit_account_required",
                f"{provider} is signed in under {len(accounts)} accounts here "
                f"({', '.join(accounts)}); choose one with --account. Nothing was changed.",
            )
        account = accounts[0]
    if account not in accounts:
        raise _refuse(
            "inherit_account_unknown",
            f"No native {provider} login on this device is for account {account!r} "
            f"(signed in: {', '.join(accounts)}). Nothing was changed.",
        )
    orgs = sorted(
        {
            r.context["org_id"]
            for r in rows
            if r.context.get("account_id") == account and r.context.get("org_id")
        }
    )
    if org and org not in orgs:
        raise _refuse(
            "inherit_org_unknown",
            f"The {provider} login for account {account!r} is not for org {org!r} "
            f"(orgs here: {', '.join(orgs) or 'none reported'}). Nothing was changed.",
        )
    for label, value in (("account", account), ("org", org)):
        if value and consent.REFERENCE.fullmatch(value) is None:
            raise _refuse(
                "inherit_scope_unrepresentable",
                f"This {label} id cannot be written into a signed permission "
                "(only letters, digits and _ . : / - are allowed, up to 256). Nothing was changed.",
            )
    return account, org


def _denies_for(rec: dict[str, Any] | None, provider: str) -> list[dict[str, Any]]:
    return [d for d in (rec or {}).get("denies", []) if d["selector"]["credential_key"] == provider]


def _existing_conflict(
    network: str, owner: str, recipient: str, provider: str, root: Path | None
) -> None:
    """Refuse, before anything is written, a grant that the foundation would mishandle.

    * a withdrawal (sticky deny) or a retained finite-recovery fence for this provider:
      re-granting needs a separate, specific superseding consent that this slice does not
      author -- it must never be cleared as a side effect of a fresh grant;
    * a different enabled standing permission for the recipient: ``provision_from_run``
      REPLACES a recipient's whole ``inherit`` set, so a second approval would silently
      drop the first one's coverage. Reduction is not widening, but it is not silent.
    """
    from local_operator.network.credentials import finite_journal

    path = placement.placement_path(network, root).with_name("consent.json")
    rec = None
    if path.exists():
        rec = consent.ConsentDocument(network, owner, root).load()["recipients"].get(recipient)
    if _denies_for(rec, provider):
        raise _refuse(
            "inherit_withdrawn",
            f"Sharing {provider} with this device was withdrawn (credential revoke) and the "
            "withdrawal stays in force. Re-granting it needs its own explicit superseding "
            "approval, which this command does not offer yet. Nothing was changed.",
        )
    if finite_journal.Journal(network, owner, root).peek(recipient, provider):
        raise _refuse(
            "inherit_recovery_pending",
            f"A finite-share recovery for {provider} on this device is unconfirmed; resolve it "
            "first. Nothing was changed.",
        )
    if rec is not None and rec.get("enabled") is True and "approval" in rec:
        covered = rec["inherit"]["providers"].get("ids", [])
        live = [p for p in covered if not _denies_for(rec, p)]
        if live:
            raise _refuse(
                "inherit_permission_exists",
                "This device already holds a standing inherit permission "
                f"(providers: {', '.join(live)}). A second approval would replace it, not add "
                "to it; revoke it first. Nothing was changed.",
            )


def _open_request(
    network: str, owner: str, recipient: str, provider: str, root: Path | None
) -> dict[str, Any] | None:
    for record in approvals.list_records(root=root):
        data = intent(record)
        if (
            data is not None
            and record.get("state") in _OPEN_STATES
            and data.get("owner_device") == owner
            and data.get("network_id") == network
            and data.get("recipient_device") == recipient
            and (data.get("inherit") or {}).get("providers", {}).get("ids") == [provider]
        ):
            return record
    return None


def request(
    network: str,
    owner: str,
    recipient: str,
    provider: str,
    *,
    account: str = "",
    org: str = "",
    root: Path | None = None,
) -> dict[str, Any]:
    """File the card for operator signature. Writes ONLY the request record."""
    provider = provider.strip().lower()
    if not reconcile.provider_ok(provider):
        raise _refuse(
            "inherit_not_applicable",
            f"{provider!r} is not a native OAuth login class that can be inherited "
            "(GitHub, MCP, device-bound and static-key credentials are explicit shares only); "
            "nothing was changed.",
        )
    subject = _subject(network, owner, recipient, root)
    _require_transport(subject)
    _existing_conflict(network, owner, recipient, provider, root)
    open_one = _open_request(network, owner, recipient, provider, root)
    if open_one is not None:
        raise _refuse(
            "inherit_request_open",
            f"Request {open_one['approval_id']} for this provider and device is already open "
            f"({open_one['state']}). Answer it, or deny it, before filing another.",
        )
    account, org = _choose_account(provider, account, org, _native_rows(provider, root), root)
    anchor = approvals.local_anchor_trio(root)
    if anchor is None:
        raise _refuse(
            "approval_anchor_unavailable",
            "this machine has no operator key to sign against; set up operator authority first",
        )
    scope: dict[str, Any] = {"principal_ref": anchor["key_id"], "account_id": account}
    if org:
        scope["org_id"] = org
    data = {
        "action": "grant",
        "owner_device": owner,
        "network_id": network,
        "recipient_device": recipient,
        "public_key_fingerprint": subject["public_key_fingerprint"],
        "inherit": {
            "providers": {"ids": [provider], "future_logins": True},
            # No MCP selection: ``mcp.urls == []`` leaves every MCP key refused by
            # ``verified_permission`` even if a future login existed.
            "mcp": {"urls": [], "future_logins": False},
            "origin": "owner-native-login",
        },
        "scope": scope,
    }
    now = time.time()
    result = approvals.create_request(
        kind=approvals.KIND_DEVICE_ONBOARD,
        request_id=approvals.new_request_id(),
        requested_by={"device_id": owner, "session_id": "", "surface": "cli"},
        device={
            "device_id": recipient,
            "name": _plain(subject["member"].name),
            "fingerprint": subject["public_key_fingerprint"],
        },
        what={OP: data, "anchor": {k: anchor[k] for k in sorted(_ANCHOR_KEYS)}},
        credential_ref=None,
        created_at=now,
        expires_at=now + approvals.DEFAULT_EXPIRY_S,
        root=root,
    )
    _audit("credential.inherit_requested", data, provider, result["approval_id"], root)
    return result


# ---------------------------------------------------------------------------
# Disclosure: derived from the SIGNED record at read time, never stored text
# ---------------------------------------------------------------------------


def _parse(record: dict[str, Any]) -> dict[str, Any]:
    """Validate that ``record`` is exactly the shape this flow mints, and unpack it.

    FAIL CLOSED and shared by every consumer (disclosure, apply, row extras): the
    disclosure a human signs against and the thing apply executes are parsed by ONE
    function, so they cannot describe different records. Anything it cannot describe is a
    record nobody is invited to approve.
    """
    what = record.get("what")
    data = intent(record)
    bad = _refuse(
        "inherit_record_unreadable",
        "This credential-permission request is not in a shape this build can describe, so it "
        "will not be approved or applied.",
    )
    if data is None or not isinstance(what, dict) or set(what) != {OP, "anchor"}:
        raise bad
    anchor = what["anchor"]
    if not isinstance(anchor, dict) or set(anchor) != _ANCHOR_KEYS:
        raise bad
    try:
        if (
            set(data)
            != {
                "action",
                "owner_device",
                "network_id",
                "recipient_device",
                "public_key_fingerprint",
                "inherit",
                "scope",
            }
            or data["action"] != "grant"
        ):
            raise bad
        for name in ("owner_device", "network_id", "recipient_device"):
            store.safe_component(data[name])
        inherit = data["inherit"]
        providers = inherit["providers"] if isinstance(inherit, dict) else None
        ids = providers.get("ids") if isinstance(providers, dict) else None
        if (
            not isinstance(ids, list)
            or len(ids) != 1
            or not isinstance(ids[0], str)
            or not reconcile.provider_ok(ids[0])
            or inherit
            != {
                "providers": {"ids": ids, "future_logins": True},
                "mcp": {"urls": [], "future_logins": False},
                "origin": "owner-native-login",
            }
        ):
            raise bad
        scope = data["scope"]
        if (
            not isinstance(scope, dict)
            or not {"principal_ref", "account_id"} <= set(scope)
            or set(scope) - {"principal_ref", "account_id", "org_id"}
            or scope["principal_ref"] != anchor["key_id"]
            or not _FINGERPRINT.fullmatch(str(data["public_key_fingerprint"]))
        ):
            raise bad
        for name in ("principal_ref", "account_id", "org_id"):
            if name in scope:
                consent._ref(scope[name])
        device = record.get("device") or {}
        if (
            device.get("device_id") != data["recipient_device"]
            or device.get("fingerprint") != data["public_key_fingerprint"]
        ):
            raise bad
        if (record.get("requested_by") or {}).get("device_id") != data["owner_device"]:
            raise bad
    except (KeyError, TypeError, AttributeError):
        raise bad from None
    return {
        "data": data,
        "provider": ids[0],
        "account": scope["account_id"],
        "org": scope.get("org_id", ""),
        "principal": scope["principal_ref"],
        "recipient_name": _plain(device.get("name")),
    }


def _revoke_command(parsed: dict[str, Any]) -> str:
    return shlex.join(
        [
            "lop",
            "network",
            "credential",
            "revoke",
            parsed["provider"],
            "--from",
            parsed["data"]["recipient_device"],
            "--network",
            parsed["data"]["network_id"],
        ]
    )


def _apply_command(parsed: dict[str, Any], approval_id: str) -> str:
    argv = [
        "lop",
        "network",
        "credential",
        "inherit",
        parsed["provider"],
        "--to",
        parsed["data"]["recipient_device"],
        "--network",
        parsed["data"]["network_id"],
        "--account",
        parsed["account"],
    ]
    if parsed["org"]:
        argv += ["--org", parsed["org"]]
    return shlex.join([*argv, "--approval", approval_id])


def _recipient_label(parsed: dict[str, Any]) -> str:
    name = parsed["recipient_name"]
    device = parsed["data"]["recipient_device"]
    return f"{name} ({device})" if name else device


def summary(record: dict[str, Any]) -> str | None:
    if intent(record) is None:
        return None
    parsed = _parse(record)
    account = parsed["account"] + (f" / org {parsed['org']}" if parsed["org"] else "")
    return (
        f"Let {_recipient_label(parsed)} borrow your {parsed['provider']} login for account "
        f"{account}: this sign-in and any later re-login of the SAME account, until you "
        "revoke it. Nothing is copied; the device gets short-lived access only."
    )


def details(record: dict[str, Any]) -> list[str]:
    """Every fact the operator must see BEFORE signing, from the signed record alone.

    Raises (fail closed) for a record this build cannot describe, so the CLI refuses to
    sign what it cannot state.
    """
    heading = summary(record)
    if heading is None:
        return []
    parsed = _parse(record)
    data = parsed["data"]
    expires = record.get("expires_at")
    window = (
        time.strftime("%Y-%m-%d %H:%M", time.localtime(float(expires)))
        if isinstance(expires, (int, float)) and expires
        else "its set time"
    )
    return [
        heading,
        f"Provider: {parsed['provider']} (a native sign-in on THIS device; copied, borrowed, "
        "device-bound, static-key and MCP logins are not covered)",
        f"Account: {parsed['account']}",
        "Org: " + (parsed["org"] or "not constrained (any org of this account)"),
        f"Recipient device: {_recipient_label(parsed)}",
        f"Recipient key fingerprint: {data['public_key_fingerprint']}",
        f"Network: {data['network_id']}",
        f"Signing principal: your operator key {parsed['principal']} on this machine",
        "Duration: UNTIL YOU REVOKE IT. This permission has no automatic expiry; a sign-in "
        "to the same account later is covered without asking again, a different account is not.",
        f"Request window: this CARD can be approved and applied until {window}. That one-hour "
        "window only limits how long this request stays answerable; it is NOT the lifetime "
        "of the permission.",
        "What the device receives: short-lived access minted here on request. The sign-in "
        "and its refresh token stay on this device; nothing is copied.",
        "Needs, and this does not grant: broker_credential on this device's own member row "
        f"and on the row for the recipient (lop network member grant {data['network_id']} "
        f"{data['recipient_device']} broker_credential).",
        f"Revoke at any time: {_revoke_command(parsed)}",
        f"Approving records your decision only. Then run: "
        f"{_apply_command(parsed, str(record.get('approval_id', '')))}",
    ]


def row_extras(record: dict[str, Any]) -> dict[str, Any] | None:
    """Optional, keyless extras for the desktop approval row (derived server-side only).

    A UI that does not know these keys ignores them. ``can_approve_here`` is False until
    the app can state what is signed; it carries the remedy rather than a blind Approve.
    """
    if intent(record) is None:
        return None
    try:
        heading: str | None = summary(record)
        lines = details(record)
    except MeshRefusal:
        heading = "A credential-permission request this build cannot describe. Do not approve it."
        lines = []
    return {
        "summary": heading,
        "details": lines,
        "can_approve_here": False,
        "approve_remedy": (
            "Review and approve it from a terminal on this machine: "
            f"lop network approvals approve {record.get('approval_id', '')}"
        ),
    }


# ---------------------------------------------------------------------------
# Apply
# ---------------------------------------------------------------------------


def _audit(
    event: str, data: dict[str, Any], provider: str, approval_id: str, root: Path | None
) -> None:
    from local_operator.network.audit import AuditEvent, AuditLog

    log = AuditLog(root=root)
    try:
        log.record(
            AuditEvent(
                event=event,
                actor=data["owner_device"],
                subject=data["recipient_device"],
                detail={
                    "credential_key": provider,
                    "act": data["owner_device"],
                    "sub": data["recipient_device"],
                    "approval_id": approval_id,
                },
            )
        )
    finally:
        log.close()


def apply(approval_id: str, expected: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    """Turn ONE approved, unexpired, verified card into the recorded permission.

    Every check that can fail runs BEFORE ``begin_run`` and before any policy write, so a
    refusal leaves the approval record, ``consent.json`` and placement byte-identical.
    """
    try:
        return _apply(approval_id, expected, root)
    except MeshRefusal as exc:
        labels = {
            "approval_expired": "The request window passed; no permission was recorded.",
            "approval_denied": "The approval was denied; no permission was recorded.",
            "approval_not_approved": "The request has not been approved; nothing was recorded.",
            "approval_already_connected": "This approval was already applied; it is one-shot.",
            "approval_record_tampered": "The approval no longer verifies; nothing was recorded.",
        }
        leading = labels.get(exc.code, "The permission was not recorded (" + exc.code + ").")
        # The label already says what the store's own sentence says for a replay: appending the
        # store text would print one fact twice. The store text is kept for every other code.
        if exc.code == "approval_already_connected":
            raise MeshRefusal(exc.code, leading + " Check the permission with --status.") from exc
        raise MeshRefusal(exc.code, leading + " " + str(exc)) from exc


def _matching_context(parsed: dict[str, Any], root: Path | None) -> dict[str, str]:
    """The native-selected context of the signed account, taken from this device's rows."""
    rows = _native_rows(parsed["provider"], root)
    chosen = [
        row
        for row in rows
        if row.context.get("account_id") == parsed["account"]
        and (not parsed["org"] or row.context.get("org_id") == parsed["org"])
    ]
    if not chosen:
        raise _refuse(
            "inherit_no_native_login",
            f"This device has no native {parsed['provider']} login for the account you "
            "approved, so nothing was recorded. Sign in to that account, or request a new "
            "permission for the account that is signed in.",
        )
    if any(not row.marker for row in chosen):
        raise _refuse(
            "inherit_login_ambiguous",
            "That login predates login stamping and cannot be matched exactly; sign in "
            "again. Nothing was recorded.",
        )
    context = {"account_id": parsed["account"]}
    if parsed["org"]:
        context["org_id"] = parsed["org"]
    return context


def _apply(approval_id: str, expected: dict[str, Any], root: Path | None) -> dict[str, Any]:
    source = approvals.load_record(approval_id, root=root)
    parsed = _parse(source)
    data = parsed["data"]
    if source.get("state") == approvals.STATE_CONNECTED:
        raise MeshRefusal(
            "approval_already_connected",
            "Nothing was changed.",
        )
    for field, value in (
        ("owner_device", data["owner_device"]),
        ("network_id", data["network_id"]),
        ("recipient_device", data["recipient_device"]),
        ("provider", parsed["provider"]),
    ):
        if expected.get(field) != value:
            raise _refuse(
                "inherit_subject_mismatch",
                f"The {field.replace('_', ' ')} on the command is not the one that was signed; "
                f"use: {_apply_command(parsed, approval_id)}",
            )
    # ``--account``/``--org`` are optional on the apply command, but a typed one must be the
    # signed one: the operator naming a different account than the card is a mistake to stop,
    # never one to quietly override (the card, not the command line, is what was signed).
    for field, value in (("account", parsed["account"]), ("org", parsed["org"])):
        if expected.get(field) and expected[field] != value:
            raise _refuse(
                "inherit_subject_mismatch",
                f"The {field} on the command is not the one that was signed; "
                f"use: {_apply_command(parsed, approval_id)}",
            )
    subject = _subject(data["network_id"], data["owner_device"], data["recipient_device"], root)
    if subject["public_key_fingerprint"] != data["public_key_fingerprint"]:
        raise _refuse(
            "inherit_subject_mismatch",
            "The recipient's key is not the one that was signed (it changed or was replaced); "
            "request a new permission.",
        )
    _require_transport(subject)
    _existing_conflict(
        data["network_id"], data["owner_device"], data["recipient_device"], parsed["provider"], root
    )
    context = _matching_context(parsed, root)
    expected_subject = {
        "owner_device": data["owner_device"],
        "network_id": data["network_id"],
        "recipient_device": data["recipient_device"],
        "public_key_fingerprint": data["public_key_fingerprint"],
        "native_context": context,
    }
    from local_operator.network import onboard_approvals

    # ``begin_run`` re-verifies decision, digest and signature and the request window; its
    # ``operation`` is what lets the generic-runner guard tell this caller from SSH onboarding.
    run = onboard_approvals.begin_run(approval_id, root=root, operation=OP)
    try:
        result = consent.provision_from_run(approval_id, run.run_id, expected_subject, root)
    except MeshRefusal as exc:
        # Retry-eligible ``failed`` (until the window ends) instead of an in-flight
        # ``connecting`` that would block a retry for 20 minutes.
        approvals.mark_failed(approval_id, run_id=run.run_id, step=OP, detail=exc.code, root=root)
        raise
    # Terminal and honest: the card says what happened. Without this the record would sit in
    # ``connecting`` and fold to ``expired`` after its hour although the permission lives on.
    approvals.mark_connected(
        approval_id,
        run_id=run.run_id,
        step=OP,
        detail="Permission recorded until revoked; delivery is the owner relay's tick",
        root=root,
    )
    _audit("credential.inherit_applied", data, parsed["provider"], approval_id, root)
    return {
        **result,
        "approval_id": approval_id,
        "state": approvals.STATE_CONNECTED,
        "provider": parsed["provider"],
        "account": parsed["account"],
        "recipient_device": data["recipient_device"],
    }


# ---------------------------------------------------------------------------
# Status: read-only, disk-derived PERMISSION state (never a delivery claim)
# ---------------------------------------------------------------------------


#: Words for the wire codes ``serve_gate`` answers. Only the ones that name a state a
#: person can act on get an action; a withdrawal and an unreadable store do not suggest a
#: refresh that would change nothing.
def _gate_words(code: str, parsed: dict[str, Any] | None, accounts: list[str]) -> str:
    if code == reconcile.CODE_MISMATCH:
        approved = parsed["account"] if parsed else "the approved account"
        return (
            f"notready:scope_mismatch (approved for account {approved}; signed in now as "
            f"{', '.join(accounts) or 'an unreported account'}. Sign in to the approved "
            "account again, or request a new permission for the account that is signed in)"
        )
    if code == reconcile.CODE_UNKNOWN:
        return (
            "notready:scope_unknown (the login here carries no account id or login stamp the "
            "approval can be matched to; sign in again so the provider reports an account)"
        )
    if code == reconcile.CODE_WITHDRAWN:
        return (
            "notready:approval_no_longer_valid (the approval is withdrawn, denied, tampered, "
            "or the recipient's key or membership changed; request a new permission)"
        )
    return "notready:approval_unverifiable (retrying each tick)"


def status(
    network: str, owner: str, recipient: str, provider: str, root: Path | None = None
) -> dict[str, Any]:
    """The permission's state, from this device's disk. Writes nothing, creates nothing."""
    provider = provider.strip().lower()
    out: dict[str, Any] = {
        "provider": provider,
        "recipient_device": recipient,
        "network_id": network,
        "scope": (
            "permission (read from this device's disk); delivery to the recipient is not "
            "observed here"
        ),
    }

    def done(state: str, *lines: str, **extra: Any) -> dict[str, Any]:
        out.update(state=state, lines=[*lines], **extra)
        return out

    if not reconcile.provider_ok(provider):
        return done(
            "not-applicable",
            f"{provider}: not applicable. This login class is shared only explicitly "
            "(lop network credential share); no inherit permission exists or can exist for "
            "it, and no sign-in or refresh changes that.",
        )
    subject = _subject(network, owner, recipient, root)
    path = placement.placement_path(network, root).with_name("consent.json")
    try:
        policy = consent.ConsentDocument(network, owner, root).load() if path.exists() else None
    except MeshRefusal:
        return done(
            "notready",
            "notready:consent_unreadable (the owner's permission file cannot be read; "
            "borrows stay refused until it can)",
        )
    rec = policy["recipients"].get(recipient) if policy else None
    if _denies_for(rec, provider):
        return done(
            "withdrawn",
            f"withdrawn: sharing {provider} with this device was revoked and the withdrawal "
            "stays in force. A new grant needs its own explicit superseding approval "
            "(not offered by this command yet).",
        )
    covered = (
        rec is not None
        and rec.get("enabled") is True
        and "approval" in rec
        and provider in rec["inherit"]["providers"].get("ids", [])
        and rec["inherit"]["providers"]["future_logins"] is True
    )
    if not covered:
        requested = _open_request(network, owner, recipient, provider, root)
        if requested is not None and requested["state"] == approvals.STATE_REQUESTED:
            return done(
                "awaiting-approval",
                f"awaiting-approval: request {requested['approval_id']} is waiting for your "
                f"signature (lop network approvals approve {requested['approval_id']}). "
                "Nothing is permitted yet.",
                approval_id=requested["approval_id"],
            )
        if requested is not None:
            parsed_open = _parse(requested)
            return done(
                "approved-not-applied",
                f"approved-not-applied: request {requested['approval_id']} is signed but the "
                f"permission is not recorded. Apply it: "
                f"{_apply_command(parsed_open, requested['approval_id'])}",
                approval_id=requested["approval_id"],
            )
        return done(
            "no-consent",
            f"no-consent: this device has no standing permission to borrow your {provider} "
            "login. Read-only work still runs there; anything that needs your "
            f"{provider} login will fail with an authentication gap until you grant it "
            f"(request: lop network credential inherit {shlex.quote(provider)} --to "
            f"{shlex.quote(recipient)} --request-approval).",
        )
    assert rec is not None  # ``covered`` above is only True for a present row (narrows the type)
    parsed = None
    try:
        parsed = _parse(approvals.load_record(rec["approval"]["ref"], root=root))
    except MeshRefusal:
        parsed = None
    holder_note = ""
    entry = placement.PlacementDocument.load(network, root, self_device=owner).entry(provider)
    held = entry.holder(recipient) if entry is not None else None
    if held is not None:
        holder_note = (
            "placement: derived from this permission (consent:...)"
            if reconcile.is_derived(held)
            else "placement: already an explicit share (the derived permission is not used)"
        )
    permission = (
        f"permission: until revoked, approval {rec['approval']['ref']}"
        + (
            f", account {parsed['account']}" + (f", org {parsed['org']}" if parsed["org"] else "")
            if parsed
            else ""
        )
        + "; revoke: "
        + (
            _revoke_command(parsed)
            if parsed
            else "lop network credential revoke "
            f"{shlex.quote(provider)} --from {shlex.quote(recipient)}"
        )
    )
    gap = reconcile.transport_gap(subject["record"], network, recipient)
    if gap:
        return done("notready", gap, permission)
    inventory = reconcile.inventory(_config(root))
    if inventory is None:
        return done(
            "notready",
            "notready:inventory_unreadable (retrying each tick; if it persists, check this "
            "device's auth.db is readable)",
            permission,
        )
    from local_operator.network.credentials.sync import MARK_LOCAL_ONLY, SyncState

    if SyncState.load(network, root).mark_for(provider) == MARK_LOCAL_ONLY:
        return done(
            "notready",
            "notready:marked_local_only (a local-only mark keeps this login on this device, "
            "so it is never offered)",
            permission,
        )
    native = [r for r in inventory.get(provider, []) if r.oauth and not r.copied]
    if not native:
        return done(
            "notready",
            f"notready:no_native_login (no native {provider} sign-in on this device; sign in "
            "to the approved account)",
            permission,
        )
    allowed, _excluded, code = reconcile.serve_gate(
        network, owner, recipient, provider, _config(root)
    )
    if not allowed:
        accounts = sorted({r.context["account_id"] for r in native if r.context.get("account_id")})
        return done("notready", _gate_words(code, parsed, accounts), permission)
    lines = [
        "ready: the permission verifies against a current native login. This is PERMISSION "
        "only; whether the recipient holds a working login now is not observed here.",
        permission,
    ]
    if holder_note:
        lines.append(holder_note)
    return done("ready", *lines)
