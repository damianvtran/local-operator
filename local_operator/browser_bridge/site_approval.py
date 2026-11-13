"""Per-site browser approvals on a headless mesh node, decided by the operator's signature.

WHY THIS EXISTS. On a headless node (``lop browser setup --node``) the extension
runs in a Chrome nobody can see, so the ordinary approval — a person clicks
Allow in the extension popup — cannot happen: ``request_access`` used to queue a
prompt that sat unseen until its 10-minute TTL. The operator (D3: "ask me once
per site", default-deny, no loopback pre-approval) decides on THEIR device
instead, through the queued-ask card the node's session raises.

HOW THE DECISION IS PROVEN (design: PR-B, "signed per-site approval"):

* The card answer travels as the EXISTING signed ``decide`` frame, unchanged —
  ``b"lop-operator-v1\\0" ‖ lp("decide") ‖ lp(session_id) ‖
  lp(f"{ask_id}#{question_id}#{word}") ‖ lp(challenge)``, ES256/DER over SHA-256
  (``operator/verify.py:signed_message``) — so the Mac and the desktop app need
  no new signing code.
* The ask's ``question_id`` is :func:`site_question_id`, a digest over the node,
  the extension's own queue entry, the origin and the scope. The extension
  recomputes it from ITS entry, so a signature decides exactly one entry for
  exactly one origin and nothing else.
* The EXTENSION verifies the signature (``extension/src/signed-decision.ts``)
  against the operator anchor point that the one root writer
  (``lop browser setup --node``, ``node_host.py``) copies from
  ``/etc/local-operator/operators/<uid>.json`` into Chrome managed policy. The
  runtime's own verification is not enough on its own: the agent holds the
  bridge token, so an extension that accepted an unsigned "decide" would let it
  approve its own request with one bridge call. With the check in the
  extension, approving without the operator means rewriting root-owned policy —
  advisory on a node whose user has passwordless sudo (Damian's D2: "leave it,
  disclose it"), but a visible, audited act rather than a silent one.

``_lp`` is duplicated rather than imported from ``operator.verify`` (where it
is private) and its TypeScript twin lives in ``signed-decision.ts``; the golden
vector in ``tests/unit/browser_bridge/test_site_approval.py`` and
``extension/tests/signed-decision.test.mjs`` pins both to the same bytes.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from typing import Any

#: Every site-approval question id starts with this. The ``v1`` is the digest
#: layout below; a different layout gets a different prefix, never a silent
#: reinterpretation of an old id.
SITE_QUESTION_PREFIX = "site:v1:"

#: The one scope a node approval grants: the EXACT origin (scheme, host, port) —
#: the extension's ``"site"`` decision. D3 rules out anything broader, so the
#: scope is a constant bound into the digest, not a choice the card offers.
SITE_SCOPE = "origin"

#: The decision words, the same closed set ``team_bridge.decide`` accepts, so the
#: ask card (which renders ``approve``/``deny`` for any question carrying a
#: ``question_id``) needs nothing new.
SITE_WORDS = ("approve", "deny")

#: Managed-policy keys under ``3rdparty.extensions.<id>``. Mirrors
#: ``extension/src/signed-decision.ts`` and ``extension/managed_schema.json`` —
#: renaming one without the others silently turns signed approval off.
POLICY_HEADLESS_NODE = "headlessNode"
POLICY_ANCHOR_POINT = "operatorAnchorPoint"
POLICY_KEY_ID = "operatorKeyId"
POLICY_NODE_DEVICE_ID = "nodeDeviceId"

#: The ``approval`` value a node extension's ``request_access`` reports.
APPROVAL_SIGNED = "signed"
#: The node has no usable anchor in policy: nobody can decide, so say so
#: instead of queueing a prompt that expires unseen.
APPROVAL_NO_APPROVER = "no_approver"


def _lp(value: str) -> bytes:
    """4-byte big-endian length, then UTF-8 — ``operator.verify._lp``'s layout."""
    raw = value.encode("utf-8")
    return struct.pack(">I", len(raw)) + raw


def site_question_id(
    *, node_device_id: str, entry_id: str, origin: str, scope: str = SITE_SCOPE
) -> str:
    """The ask question id that binds a signed decision to ONE extension entry.

    Length-prefixed so no field can borrow bytes from its neighbour (an origin
    ending in a digit cannot be re-split against the entry id). ``entry_id`` is
    the extension's ``crypto.randomUUID`` — unguessable and single-use — and
    ``node_device_id`` stops a signature minted for one node's entry being
    presented to another node's extension.
    """
    digest = hashlib.sha256(
        _lp(node_device_id) + _lp(entry_id) + _lp(origin) + _lp(scope)
    ).hexdigest()
    return SITE_QUESTION_PREFIX + digest


def is_site_question_id(question_id: object) -> bool:
    return isinstance(question_id, str) and question_id.startswith(SITE_QUESTION_PREFIX)


def site_ask_question(*, origin: str, node_name: str, question_id: str) -> dict[str, Any]:
    """The queued-ask question a node raises for one origin.

    The TEXT is built here from the origin the extension reported, never from
    model input: the card is what the operator approves, and an agent-authored
    sentence could describe a different site than the one the digest binds
    (the extension enforces the bound origin regardless; this keeps the card
    honest too). ``human_required`` keeps the ask gate from ever answering it,
    and the non-empty ``question_id`` routes every answer through the verified
    ``bridge_decide`` door — a plain ``ask_respond`` is refused for it.
    """
    from local_operator.i18n.keys import browser_site_approval as _keys
    from local_operator.i18n.messages import render_resolved

    return {
        "id": "site-access",
        "question": render_resolved(_keys.question(origin=origin, node=node_name)),
        "options": [
            {"id": "approve", "label": render_resolved(_keys.option_approve())},
            {"id": "deny", "label": render_resolved(_keys.option_deny())},
        ],
        "multi": False,
        "secret": False,
        "human_required": True,
        "question_id": question_id,
    }


def policy_fields(
    anchor_spki: bytes | None, anchor_key_id: str, node_device_id: str
) -> dict[str, Any]:
    """The managed-policy keys setup writes for signed approval.

    ``headlessNode`` is written ALWAYS in node mode: it is what tells the
    extension there is no person at the popup, so a missing anchor becomes a
    typed ``no_approver`` instead of a prompt that waits ten minutes for a
    click that cannot come. The anchor keys are written only when the anchor
    is usable; the extension re-checks that ``key_id`` is the digest of the
    point, so a hand-edited pair that disagrees fails closed.
    """
    fields: dict[str, Any] = {POLICY_HEADLESS_NODE: True}
    if anchor_spki and anchor_key_id:
        fields[POLICY_ANCHOR_POINT] = anchor_spki.hex()
        fields[POLICY_KEY_ID] = anchor_key_id
        fields[POLICY_NODE_DEVICE_ID] = node_device_id
    return fields


def anchor_policy_fields() -> dict[str, Any]:
    """Derive :func:`policy_fields` from THIS host's root-owned operator anchor.

    The single source is ``trust.load_anchor`` — the same fail-closed reader
    the runtime's authority seam uses (root-owned, not group/world-writable,
    no symlinks) — so the extension trusts exactly the key the runtime does.
    An unusable anchor yields only ``headlessNode``.
    """
    from local_operator.network import identity
    from local_operator.operator.trust import load_anchor

    loaded = load_anchor()
    anchor = loaded.anchor if loaded.usable else None
    if anchor is None:
        return policy_fields(None, "", "")
    try:
        device = identity.load()
    except Exception:  # noqa: BLE001 — an unreadable identity binds to "" rather than failing setup
        device = None
    return policy_fields(anchor.spki, anchor.key_id, device.device_id if device else "")


def anchor_key_held_locally(config_root: Path, key_id: str) -> bool:
    """Whether THIS host holds the private half of the anchor key.

    A node that can sign for its own anchor has no independent approver: the
    agent could read the key file and approve itself, so the node is treated as
    having none (``no_approver``) rather than pretending the signature means a
    human decided. Only the ``file-only`` backend can exist on a Linux node, so
    that file is the one place to look; a key that fails to parse is not held.
    """
    if not key_id:
        return False
    from local_operator.operator.keychain import FileKeyBackend, default_file_path

    path = default_file_path(config_root)
    if not path.is_file():
        return False
    try:
        signer = FileKeyBackend(path).load()
    except Exception:  # noqa: BLE001 — unreadable/malformed: not a usable local key
        return False
    return signer is not None and signer.handle.key_id == key_id
