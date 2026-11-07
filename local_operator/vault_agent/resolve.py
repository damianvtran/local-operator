"""The one agent resolve flow: mint (route 27), prove, resolve (route 15).

Called by the ``secret`` tool's ``vault_resolve`` verb and nothing else. Each
step is a contract item (VAULT.md §4.1 step 4, §6.9); the order matters:

1. Re-check the gate, the cached grant and the seed BEFORE any network call,
   so a closed gate or a missing grant never mints a challenge it cannot use.
2. Mint a ``resolve`` challenge with the person's bearer and validate the
   answer's shape: ``tenant_id`` must be the owner's tenant (a challenge minted
   for another organization can never verify), a 64-hex local id, a 32-hex
   authority id, and ``ledger_version`` as a canonical decimal STRING.
3. Sign the agent-proof message over the commitment with the CURRENT Unix
   seconds (the authority's window is ±60 s inclusive).
4. One route-15 call. **Never retried**: the challenge is single-use, and a
   ``503`` can follow a burned row, so a retry is a new mint the agent decides
   on, not something this flow does behind its back.

The value is returned to the caller (the tool), which lands it in the session
credential store; nothing here logs or formats it.

REFUSALS ARE MODEL-FACING (English by design, like every tool result) and each
one names the remedy, because the agent has to tell its person what to do and
"something went wrong" leaves both of them stuck (design §7 risk 6).
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

from local_operator.vault_agent import eligibility, keys, wire
from local_operator.vault_agent.client import (
    AUTHORITY_CHALLENGE_RE,
    ITEM_ID_RE,
    LOCAL_CHALLENGE_RE,
    VAULT_ID_RE,
    VaultClient,
)

#: The one sentence for the dark state (production today: the delegation port is
#: unwired, so routes 27-30 answer 503 and route 1 says delegations_available
#: false). Not an error loop: the agent is told plainly and to stop.
UNAVAILABLE = (
    "Agent access to Radient Vault isn't available yet (the service answered "
    "'unavailable'). Do not retry now; tell the user it is not available yet."
)
NOT_CONNECTED = (
    "This agent has no live Radient Vault access. Ask the user to run "
    "`lop vault connect` in a terminal and approve it in the console, then try again."
)

_CONSOLE_ITEM_PATH = re.compile(r"/dashboard/vaults/(vlt_[0-9A-Za-z_-]+)/items/(vit_[0-9A-Za-z_-]+)/?$")


class VaultAgentRefusal(Exception):
    """A resolve that did not produce a value. ``str(exc)`` is the model message."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Resolved:
    value: str
    username: str | None
    expires_at: str | None


def parse_reference(vault_id: str | None, item_id: str | None) -> tuple[str, str]:
    """``(vault_id, item_id)`` from the two ids, or a console item URL in either.

    Agent access is manual-vault only, so in v1 the ids come from the person or a
    console link (``…/dashboard/vaults/<vlt_>/items/<vit_>``); the model passes
    them through and never invents either (VAULT.md §4).
    """

    for candidate in (item_id, vault_id):
        text = (candidate or "").strip()
        if text.startswith(("http://", "https://")):
            match = _CONSOLE_ITEM_PATH.search(urlsplit(text).path)
            if match:
                return match.group(1), match.group(2)
    vault = (vault_id or "").strip()
    item = (item_id or "").strip()
    if not VAULT_ID_RE.fullmatch(vault) or not ITEM_ID_RE.fullmatch(item):
        raise VaultAgentRefusal(
            "invalid_reference",
            "vault_resolve needs both ids the user shared: vault_id (vlt_...) and "
            "item_id (vit_...), or the console item link. Never guess an id.",
        )
    return vault, item


def default_client_factory(config_dir: Path) -> VaultClient | None:
    from local_operator.hub_org.eligibility import person_client

    resolved = person_client(config_dir)
    return None if resolved is None else VaultClient.for_person(resolved[0])


def _code(exc: BaseException) -> tuple[int | None, str | None, dict[str, Any]]:
    details = getattr(exc, "details", None)
    return (
        getattr(exc, "status_code", None),
        getattr(exc, "code", None),
        details if isinstance(details, dict) else {},
    )


def refusal_for(config_dir: Path, exc: BaseException) -> VaultAgentRefusal:
    """Map an upstream refusal to its model message, applying its side effect.

    The side effects are the point of mapping here rather than in the tool:
    ``delegation_required`` forgets the cached grant (revoked, expired, a
    re-enrolment or a key the authority does not hold all look the same, and all
    mean "connect again"), and a plan refusal closes the gate at once instead of
    waiting for the next boundary refresh.
    """

    status, code, details = _code(exc)
    if status == 503 or code == "vault_unavailable":
        return VaultAgentRefusal("unavailable", UNAVAILABLE)
    if code == "delegation_required":
        eligibility.clear_grant(config_dir)
        return VaultAgentRefusal("delegation_required", NOT_CONNECTED)
    if code == "assertion_invalid" and details.get("reason") == "proof_invalid":
        return VaultAgentRefusal(
            "proof_invalid",
            "Radient Vault rejected the agent proof as out of window or for the wrong "
            "endpoint. Ask the user to check this computer's clock is set automatically, "
            "then try once more.",
        )
    if code == "vault_no_access":
        return VaultAgentRefusal(
            "vault_no_access",
            "The user is no longer a member of that vault, so their agent cannot read "
            "it. Ask them to check they still have access in the console.",
        )
    if status == 401:
        return VaultAgentRefusal(
            "identity_required",
            "Radient refused the user's sign-in (signed out, or this machine's Team "
            "Bridge endpoint is paused). Ask the user to run `lop login radient` and "
            "check Team Bridge is on.",
        )
    if code in ("vault_not_available", "plan_required"):
        eligibility.close(config_dir, reason=code)
        return VaultAgentRefusal(
            code,
            "Radient Vault isn't available for this account's organization (it needs a "
            "Team plan). Use the local secret store instead.",
        )
    if code == "conversation_vaults_disabled":
        return VaultAgentRefusal(code, "Conversation vaults aren't available yet.")
    if status == 410 or code in ("secret_expired", "vault_item_expired"):
        return VaultAgentRefusal(
            code or "expired", "That Vault item has expired. Ask the user to share it again."
        )
    if status == 404:
        return VaultAgentRefusal(
            code or "not_found",
            "No such Vault item for this user (check both ids, or ask the user for the "
            "console link).",
        )
    if code == "rate_limited" or status == 429:
        return VaultAgentRefusal("rate_limited", "Radient Vault is rate limiting reads; wait a minute.")
    label = code or (f"HTTP {status}" if status else "no response")
    return VaultAgentRefusal(
        code or "error",
        f"Radient Vault did not return the item ({label}); the outcome is unknown. "
        "Do not retry in a loop.",
    )


def _validated_mint(answer: dict[str, Any], *, tenant_id: str) -> tuple[str, str, int]:
    challenge_id = answer.get("challenge_id")
    authority_id = answer.get("authority_challenge_id")
    if answer.get("tenant_id") != tenant_id:
        raise VaultAgentRefusal(
            "wrong_tenant",
            "That vault belongs to a different organization than this machine's Team "
            "Bridge endpoint, so this agent cannot read it.",
        )
    try:
        ledger = wire.parse_ledger_version(answer.get("ledger_version"))
    except ValueError:
        ledger = None
    if (
        not isinstance(challenge_id, str)
        or not LOCAL_CHALLENGE_RE.fullmatch(challenge_id)
        or not isinstance(authority_id, str)
        or not AUTHORITY_CHALLENGE_RE.fullmatch(authority_id)
        or ledger is None
    ):
        raise VaultAgentRefusal(
            "protocol_error",
            "Radient Vault answered in a shape this version of Local Operator does not "
            "understand. Ask the user to update (`lop update`).",
        )
    return challenge_id, authority_id, ledger


def resolve_item(
    config_dir: Path,
    *,
    vault_id: str,
    item_id: str,
    client_factory: Callable[[Path], VaultClient | None] = default_client_factory,
    clock: Callable[[], float] = time.time,
) -> Resolved:
    """Resolve one item as this machine's agent. Raises :class:`VaultAgentRefusal`."""

    config_dir = Path(config_dir)
    gate = eligibility.gate(config_dir)
    if gate is None:
        # Defence in depth: the verb is not on the schema when the gate is
        # closed, so reaching here means a stale schema or a forged call.
        raise VaultAgentRefusal(
            "not_available",
            "Radient Vault agent access is not available on this machine (it needs a "
            "signed-in Radient Team plan and Team Bridge). Use the local secret store.",
        )
    seed = keys.load_seed(gate.endpoint_id, config_dir)
    if seed is None:
        raise VaultAgentRefusal("not_connected", NOT_CONNECTED)
    fingerprint = keys.fingerprint_for_seed(seed)
    grant = eligibility.read_grant(
        config_dir, tenant_id=gate.tenant_id, endpoint_id=gate.endpoint_id, fingerprint=fingerprint
    )
    if grant is None:
        raise VaultAgentRefusal("not_connected", NOT_CONNECTED)
    client = client_factory(config_dir)
    if client is None:
        raise VaultAgentRefusal(
            "identity_required",
            "The user is not signed in to Radient on this machine. Ask them to run "
            "`lop login radient`.",
        )

    try:
        answer = client.mint_resolve(vault_id=vault_id, item_id=item_id)
    except VaultAgentRefusal:
        raise
    except Exception as exc:  # noqa: BLE001 - classified
        raise refusal_for(config_dir, exc) from None
    challenge_id, authority_id, ledger = _validated_mint(answer, tenant_id=gate.tenant_id)

    commitment = wire.resolve_commitment(
        tenant_id=gate.tenant_id,
        vault_id=vault_id,
        item_id=item_id,
        authority_challenge_id=authority_id,
        ledger_version=ledger,
    )
    timestamp = int(clock())
    message = wire.agent_proof_message(
        commitment=commitment,
        delegation_id=grant.delegation_id,
        endpoint_id=gate.endpoint_id,
        timestamp=timestamp,
    )
    proof = keys.sign_proof(seed, message)
    try:
        body = client.resolve_as_agent(
            vault_id=vault_id,
            item_id=item_id,
            endpoint_id=gate.endpoint_id,
            challenge_id=challenge_id,
            delegation_id=grant.delegation_id,
            timestamp=timestamp,
            proof=proof,
        )
    except Exception as exc:  # noqa: BLE001 - classified; never retried (single-use challenge)
        raise refusal_for(config_dir, exc) from None
    value = body.get("value")
    if not isinstance(value, str) or not value.strip():
        raise VaultAgentRefusal(
            "protocol_error", "Radient Vault returned no value for that item."
        )
    username = body.get("username")
    expires_at = body.get("expires_at")
    return Resolved(
        value=value,
        username=username if isinstance(username, str) and username else None,
        expires_at=expires_at if isinstance(expires_at, str) else None,
    )


def pick_up(
    client: VaultClient, *, tenant_id: str, endpoint_id: str, fingerprint: str
) -> eligibility.Grant | None:
    """One route-29 read (every page) and the §6.20 selection."""

    rows = client.list_delegations(tenant_id)
    return eligibility.select_grant(
        rows, tenant_id=tenant_id, endpoint_id=endpoint_id, fingerprint=fingerprint
    )
