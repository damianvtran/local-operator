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
* an ALL-DIGIT value never counts as a credential-named assignment. A prose
  count under a plural label (``tokens: 1234567890``, ``total tokens: …``) is
  a usage figure an agent reports all the time; the shared table's count
  judgement covers the identifier spelling (``max_tokens=``) but not a spaced
  prose label (review m3). The accepted cost: a purely numeric password under
  a ``password:`` label passes this guard.
* ONE extra rule for the AWS secret access key under the SDK and
  ``aws configure`` labels the shared table does not spell: run-together
  ``SecretAccessKey`` (``"SecretAccessKey": "…"`` is the shape ``aws sts
  assume-role`` prints) and the prompt form ``AWS Secret Access Key [None]:
  …``. The shared table owns ``aws_secret_access_key=`` and the spaced console
  label ``Secret access key: …`` (measured), and a bare 40-character
  value is indistinguishable from a SHA-1 or a git commit, so the label is the
  evidence: a 40-character value with no such label beside it passes.

A bare WORD passes: "my password manager", "reset your password", ``AKIA``
alone, a commit SHA. What is refused is a value standing where a credential
stands.

WHAT THIS GUARD IS, AND IS NOT (security review S-2). It stops an ACCIDENTAL
paste of a credential spelled the way a CLI, console, SDK or credentials file
spells one. It is not a control against a model trying to move a secret, or
prompt-injected into it: an encoded, split, re-cased or look-alike spelling has
no shape here and passes. That defence belongs to the hub, and to limiting
which credentials a session holds at all.

FAIL CLOSED. If the pattern table itself raises, the message is refused with
:data:`SCAN_UNAVAILABLE_CODE`, never sent unscanned: a guard that steps aside
when it breaks is a guard that is off exactly when nobody is watching it.

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
#: The refusal when the scan could not run (see "FAIL CLOSED" above).
SCAN_UNAVAILABLE_CODE = "secret_scan_unavailable"

#: The full AWS access key id shape the floor admits. ``AKIA`` is a long-term
#: IAM user key, ``ASIA`` an STS session key; both are 20 characters.
_AWS_KEY_ID = re.compile(r"A(?:KIA|SIA)[0-9A-Z]{16}")

#: The AWS secret access key under the labels the shared table does not spell
#: (see the module docstring). Case-insensitive on the LABEL only: the value is
#: AWS's 40-character base64-alphabet secret, and the trailing lookahead stops a
#: longer token from matching on its first 40 characters.
#: The optional bracketed group is ``aws configure``'s prompt default
#: (``[None]``, ``[****************ABCD]``), which sits between the label and
#: the colon (review m2).
_AWS_SECRET_UNDER_LABEL = re.compile(
    r"(?i:\b(?:aws[ _-]?)?secret[ _-]?access[ _-]?key)[\"']?"
    r"(?:\s*\[[^\]\n]{0,24}\])?\s*[:=]\s*[\"']?"
    r"(?P<value>[A-Za-z0-9/+]{40})(?![A-Za-z0-9/+=])"
)
_AWS_SECRET_LABEL = "aws-secret-access-key"

#: The sentence that says HOW a credential may travel — the WHOLE availability
#: claim lives here (review S-4), so no clause elsewhere states the handoff as
#: working today. Kept apart from the rest of the refusal on purpose: while
#: Radient Vault handoff is not available to
#: agents (``guides/credentials/GUIDE.md``: agents cannot create or read Vault
#: items yet; Vault is also off in production), this says so and sends the agent
#: to its human. When a vault tool lands, THIS constant is the one to replace
#: with "package it with <tool>; the recipient retrieves it" — the refusal around
#: it does not change.
VAULT_HANDOFF_SENTENCE = (  # i18n: ignore model-facing refusal sentence
    "The intended way to share one is a Radient Vault credential (the sender "
    "packages it, the recipient retrieves it), but that handoff is not available "
    "to agents yet: do not paste the secret in any form, and ask your human how "
    "your team shares secrets."
)


def _below_floor(label: str, value: str, match: Match[str]) -> bool:
    """The bridge's confidence floor over the shared table (``reject`` hook).

    Two rules are floored; every other rule's own spelling is already the
    evidence it fires on. The AWS key id must be the full 20-character shape;
    quotes are stripped before that check because the shared rule accepts a
    quote spliced into the id (``A'KIA…``) and the floor should not undo that.
    Stated limit (review S-2): the shared table only RUNS when its prefilter
    finds an anchor word, and a spliced id is no anchor, so a spliced id with
    no other anchor in the text is never scanned at all — the gap is the
    prefilter's, not this floor's. A credential-named assignment whose value
    is all digits is a count, not a credential (module docstring, m3).
    """

    del match
    if label == "credential-assignment":
        return value.strip().isdigit()
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
    # One class per credential (review n2): an ``AWS_SECRET_ACCESS_KEY=`` line
    # is already a shared-table hit on the same value, and naming it twice reads
    # as two secrets.
    already = {hit.value for hit in hits}
    for found in _AWS_SECRET_UNDER_LABEL.finditer(text):
        if found.group("value") not in already:
            seen.setdefault(_AWS_SECRET_LABEL, None)
            break
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
    try:
        classes = credential_classes(text)
    except Exception as failure:  # any scanner fault must refuse, never fail open
        # Type name only: an exception's text could quote the input.
        logger.warning(
            "team bridge %s refused before send: credential scan failed (%s)",
            op,
            type(failure).__name__,
        )
        return {
            "ok": False,
            "code": SCAN_UNAVAILABLE_CODE,
            "error": (
                f"Nothing was sent: this {op} could not be checked for credentials, "
                "so it was held back. Try again; if it keeps failing, tell your "
                "user what you were trying to send."
            ),
        }
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
            "Credentials must never travel as Team Bridge message text. "
            f"{VAULT_HANDOFF_SENTENCE} If it is not a credential, rephrase the "
            "message without that value."
        ),
    }
