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
  extension, a site is approved only by a signature from the anchor key — the
  agent can neither forge one nor redirect one to another entry, origin or node.
  Loosening that without the operator means rewriting root-owned policy:
  advisory on a node whose user has passwordless sudo (Damian's D2: "leave it,
  disclose it"), a hard boundary otherwise.
* The SIGNER checks the card before it signs (:func:`verify_site_card`, called
  from ``asks/decision.py`` by both the TUI and the desktop answer route). The
  card's words are drawn on the node, inside a runtime the agent shares a uid
  with, and the signature covers only the opaque digest — so an agent appending
  to the ask log could publish a card that READS "allow origin A" while its
  ``question_id`` binds origin B (review R1-1). The question therefore carries
  its bound inputs as a structured ``site`` block; the signer recomputes the
  digest from that block and redraws the card's question and options with the
  same template, and refuses unless both match what it is about to show-and-
  sign. What the operator reads is then what the extension enforces.

  WHAT THIS DOES NOT COVER: a signer running a build from before this check
  (an older Mac or desktop) signs whatever card it is shown, so a node's cards
  are only trustworthy once the operator's own machine has updated. The
  ``node_name`` in the sentence is display-only (the digest binds the node's
  device id, which the extension checks against its own policy), so a forged
  name can mislabel which machine is asking, never which origin is granted or
  where. And a Mac and a node whose catalogues word the question differently
  (different releases) refuse an honest card — fail-closed, fixed by updating.

``_lp`` is duplicated rather than imported from ``operator.verify`` (where it
is private) and its TypeScript twin lives in ``signed-decision.ts``; the golden
vector in ``tests/unit/browser_bridge/test_site_approval.py`` and
``extension/tests/signed-decision.test.mjs`` pins both to the same bytes.
"""

from __future__ import annotations

import hashlib
import struct
from collections.abc import Mapping
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


def site_binding(
    *, node_device_id: str, entry_id: str, origin: str, node_name: str, scope: str = SITE_SCOPE
) -> dict[str, str]:
    """The structured ``site`` block a site question carries for its signer.

    Exactly the digest's inputs plus the display name, so a signer can rebuild
    both the ``question_id`` and the sentence (:func:`verify_site_card`) instead
    of trusting text drawn on the node.
    """
    return {
        "node_device_id": node_device_id,
        "entry_id": entry_id,
        "origin": origin,
        "scope": scope,
        "node_name": node_name,
    }


def site_ask_question(
    *, origin: str, node_name: str, node_device_id: str, entry_id: str, scope: str = SITE_SCOPE
) -> dict[str, Any]:
    """The queued-ask question a node raises for one origin.

    The TEXT is built here from the origin the extension reported, never from
    model input, and the digest is computed here from the same fields, so the
    two cannot disagree at the source. ``site`` carries those fields so the
    SIGNER can prove they still agree after the row has crossed the node's
    multi-writer ask log (:func:`verify_site_card`). ``human_required`` keeps
    the ask gate from ever answering it, and the non-empty ``question_id``
    routes every answer through the verified ``bridge_decide`` door — a plain
    ``ask_respond`` is refused for it.
    """
    binding = site_binding(
        node_device_id=node_device_id,
        entry_id=entry_id,
        origin=origin,
        node_name=node_name,
        scope=scope,
    )
    return {
        "id": "site-access",
        **_card_words(origin=origin, node_name=node_name, locale=None),
        "multi": False,
        "secret": False,
        "human_required": True,
        "question_id": site_question_id(
            node_device_id=node_device_id, entry_id=entry_id, origin=origin, scope=scope
        ),
        "site": binding,
    }


def _card_words(*, origin: str, node_name: str, locale: str | None) -> dict[str, Any]:
    """The card's question and options in ``locale`` (``None``: the config's).

    One template for the node that draws the card and the signer that redraws
    it, so the comparison in :func:`verify_site_card` is byte-for-byte.
    """
    from local_operator.i18n.keys import browser_site_approval as _keys
    from local_operator.i18n.messages import render, render_resolved

    def text(message: Any) -> str:
        if locale is None:
            return render_resolved(message)
        return render(message.code, dict(message.params), locale=locale)

    return {
        "question": text(_keys.question(origin=origin, node=node_name)),
        "options": [
            {"id": "approve", "label": text(_keys.option_approve())},
            {"id": "deny", "label": text(_keys.option_deny())},
        ],
    }


class SiteCardMismatch(ValueError):
    """The signer refused a site card it could not rebuild (review R1-1).

    A ``ValueError`` so every answer surface's existing refusal path (the
    desktop route's 409, the TUI's toast) reports it with no new branch; the
    message is the keyed, user-facing sentence and ``code`` is the stable id.
    """

    code = "site_card_mismatch"


def verify_site_card(question: Any) -> None:
    """Refuse (:class:`SiteCardMismatch`) a site card whose words or digest are
    not what its structured ``site`` block produces.

    Called by the SIGNER before it signs, never by the node. Three checks, all
    against the block: the ``question_id`` is the digest of its inputs; the
    question sentence is the code-built template for its origin and node name;
    the options are exactly ``approve``/``deny`` with the template's labels (a
    swapped pair would turn a tap on "Deny" into a signed approve). The node
    renders in ITS configured language, so any shipped locale's rendering is
    accepted — every one is code-built from the same fields, so whichever it
    is, it names the origin the digest binds.
    """
    from local_operator.i18n.keys import browser_site_approval as _keys
    from local_operator.i18n.messages import render_resolved
    from local_operator.i18n.resolve import SUPPORTED_LOCALES

    refusal = SiteCardMismatch(render_resolved(_keys.refused_card_mismatch()))
    site = question.get("site") if isinstance(question, Mapping) else None
    if not isinstance(site, Mapping):
        raise refusal
    raw = {
        key: site.get(key) for key in ("node_device_id", "entry_id", "origin", "scope", "node_name")
    }
    # ``node_device_id`` may be empty (setup binds "" when the node's identity
    # was unreadable, and the extension hashes the same ""); the rest may not.
    # The scope must be the one the card's sentence describes ("this exact
    # origin only"): a block naming another scope is not the card on screen.
    if not all(isinstance(value, str) for value in raw.values()):
        raise refusal
    fields: dict[str, str] = {key: str(value) for key, value in raw.items()}
    if not (fields["entry_id"] and fields["origin"]) or fields["scope"] != SITE_SCOPE:
        raise refusal
    expected_id = site_question_id(
        node_device_id=fields["node_device_id"],
        entry_id=fields["entry_id"],
        origin=fields["origin"],
        scope=fields["scope"],
    )
    if question.get("question_id") != expected_id:
        raise refusal
    shown = {
        "question": question.get("question"),
        "options": [
            {"id": option.get("id"), "label": option.get("label")}
            for option in (question.get("options") or ())
            if isinstance(option, Mapping)
        ],
    }
    for locale in SUPPORTED_LOCALES:
        if shown == _card_words(
            origin=fields["origin"], node_name=fields["node_name"], locale=locale
        ):
            return
    raise refusal


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


def anchor_policy_fields(config_root: Path | None = None) -> dict[str, Any]:
    """Derive :func:`policy_fields` from THIS host's root-owned operator anchor.

    The single source is ``trust.load_anchor`` — the same fail-closed reader
    the runtime's authority seam uses (root-owned, not group/world-writable,
    no symlinks) — so the extension trusts exactly the key the runtime does.
    An unusable anchor yields only ``headlessNode``.

    So does an anchor whose PRIVATE half this host holds
    (:func:`anchor_key_held_locally`, the rule the browser tool already applies
    before raising a card): the agent could read that key file and sign its
    own decision straight into the extension, so the policy must make the
    EXTENSION report ``no_approver`` too, not only the runtime (review R1-2).
    ``config_root`` defaults to the invoking user's config directory — setup
    runs as that user and only its privileged steps go through sudo.
    """
    from local_operator.network import identity
    from local_operator.operator.trust import load_anchor

    loaded = load_anchor()
    anchor = loaded.anchor if loaded.usable else None
    if anchor is None:
        return policy_fields(None, "", "")
    if config_root is None:
        from local_operator.paths import config_dir

        config_root = config_dir()
    if anchor_key_held_locally(config_root, anchor.key_id):
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
