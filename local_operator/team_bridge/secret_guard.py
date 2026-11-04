"""Refuse a Team Bridge message that carries a credential, BEFORE it is sent.

WHY THIS EXISTS (2026-11-04). An agent put AWS credentials into a
``channel_post`` body. Nothing on this side noticed; the post died at the AWS
WAF in front of ``api.radienthq.com`` (a ``~/.aws/credentials`` path in the body
matched ``GenericLFI_Body``), and the agent read the result as a transient
failure to retry. Had the WAF not fired, the keys would now sit in a
conversation every member — human and agent, on every device — can read, and
in every later model call that replays the history. A sent message cannot be
recalled, so the only place to stop this is before the first byte leaves.

WHAT IT MATCHES, and why it is not a second detector. The patterns are the
harness's one credential-shape table (:mod:`local_operator.redaction_shapes`),
the same one that masks tool results: an AWS access key id, a PEM private key
block, a credential-named assignment carrying a value (``password: …``,
``token=…``, ``AWS_SECRET_ACCESS_KEY=…``), issuer-prefixed tokens (``ghp_``,
``github_pat_``, ``xox[bp]-``, ``sk-``, ``AIza``), DSN passwords, bearer
headers, JWTs. Two bridge-local adjustments sit on top, both because a REFUSAL
has a different cost from a MASK — a mask that over-fires blurs a word in a
result; a refusal that over-fires blocks a message the agent meant to send:

* a CONFIDENCE FLOOR on the AWS rule. The shared rule matches ``AKIA``/``ASIA``
  followed by ANY upper-case run (it is tuned to mask fragments), so the word
  ``ASIAN`` matched it (measured). Here a key id must be the full 20-character
  shape, ``(AKIA|ASIA)[0-9A-Z]{16}``.
* ONE extra rule for the AWS secret access key under its CONSOLE and SDK
  labels (``Secret access key: …``, ``"SecretAccessKey": "…"`` — the shape
  ``aws sts assume-role`` prints). The shared table only knows the
  ``aws_secret_access_key=`` assignment spelling, and a bare 40-character
  value is indistinguishable from a SHA-1 or a git commit, so the label is the
  evidence: a 40-character value with no such label beside it passes.

A bare WORD passes: "my password manager", "reset your password", ``AKIA``
alone, a commit SHA. What is refused is a value standing where a credential
stands.

WHAT NEVER TRAVELS. The matched text: not in the result, the error, the log
line or an exception. Only the shape LABEL (the class) is named, because the
agent needs to know which part of its message to remove, and a label is not
the secret.
"""

from __future__ import annotations

import logging
import re
from re import Match
from typing import Any

from local_operator.redaction_shapes import scrub_shapes_with_hits

logger = logging.getLogger(__name__)

#: The stable machine code of this refusal. Callers and tests key on it.
SECRET_DETECTED_CODE = "secret_detected"

#: The full AWS access key id shape the floor admits. ``AKIA`` is a long-term
#: IAM user key, ``ASIA`` an STS session key; both are 20 characters.
_AWS_KEY_ID = re.compile(r"A(?:KIA|SIA)[0-9A-Z]{16}")

#: The AWS secret access key under the labels the shared table does not spell
#: (see the module docstring). Case-insensitive on the LABEL only: the value is
#: AWS's 40-character base64-alphabet secret, and the trailing lookahead stops a
#: longer token from matching on its first 40 characters.
_AWS_SECRET_UNDER_LABEL = re.compile(
    r"(?i:\b(?:aws[ _-]?)?secret[ _-]?access[ _-]?key)[\"']?\s*[:=]\s*[\"']?"
    r"[A-Za-z0-9/+]{40}(?![A-Za-z0-9/+=])"
)
_AWS_SECRET_LABEL = "aws-secret-access-key"

#: The sentence that says HOW a credential may travel. Kept apart from the rest
#: of the refusal on purpose: while Radient Vault handoff is not available to
#: agents (``guides/credentials/GUIDE.md``: agents cannot create or read Vault
#: items yet; Vault is also off in production), this says so and sends the agent
#: to its human. When a vault tool lands, THIS constant is the one to replace
#: with "package it with <tool>; the recipient retrieves it" — the refusal around
#: it does not change.
VAULT_HANDOFF_SENTENCE = (
    "Vault handoff is not available to agents yet, so do not paste the secret in "
    "any form; ask your human how your team shares secrets."
)


def _below_floor(label: str, value: str, match: Match[str]) -> bool:
    """The bridge's confidence floor over the shared table (``reject`` hook).

    Only the AWS key-id rule is floored; every other rule's own spelling is
    already the evidence it fires on. Quotes are stripped first because the
    shared rule tolerates a quote spliced into the id (``A'KIA…``), which is a
    real evasion shape, not a reason to let it through.
    """

    del match
    if label != "aws-access-key-id":
        return False
    return _AWS_KEY_ID.fullmatch(value.replace("'", "").replace('"', "")) is None


def credential_classes(text: str) -> list[str]:
    """The credential CLASSES present in ``text``, de-duplicated, in order.

    Labels only — never a value. Empty when the text carries none.
    """

    if not text:
        return []
    _, hits = scrub_shapes_with_hits(text, reject=_below_floor)
    seen: dict[str, None] = {}
    for hit in hits:
        seen.setdefault(hit.label, None)
    if _AWS_SECRET_UNDER_LABEL.search(text) is not None:
        seen.setdefault(_AWS_SECRET_LABEL, None)
    return list(seen)


def refuse_secret_in(op: str, text: Any) -> dict[str, Any] | None:
    """The tool's refusal when ``text`` carries a credential, else ``None``.

    Local and network-free by construction: the executor calls it before the
    admission gate (which may schedule a status read), before any name
    resolution and before a reply-guard slot is reserved, so a refused message
    leaves no trace anywhere but this result.
    """

    if not isinstance(text, str):
        return None
    classes = credential_classes(text)
    if not classes:
        return None
    named = ", ".join(classes)
    # Labels only: the log is read by humans and shipped in bug reports.
    logger.info("team bridge %s refused before send: credential shape(s) %s", op, named)
    return {
        "ok": False,
        "code": SECRET_DETECTED_CODE,
        "classes": classes,
        "error": (
            f"Nothing was sent: this {op} looks like it contains a credential ({named}). "
            "Credentials must never travel as Team Bridge message text. The way to "
            "share one is a Radient Vault credential: the sender packages it and the "
            f"recipient retrieves it. {VAULT_HANDOFF_SENTENCE} If it is not a "
            "credential, rephrase the message without that value."
        ),
    }
