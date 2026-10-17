"""Credential-shape redaction for imported text, with counts but never values.

The source stores are foreign: a conversation can contain a pasted secret that
this harness never registered, so exact-value scrubbing (``VariableStore``) is
not enough. We reuse the shipped shape table
(:mod:`local_operator.redaction_shapes`) and add the narrow cases the design
names that a provider-error path may not cover: a bare ``Bearer`` token and an
``ENV_ASSIGNMENT=value`` line.

Over-masking is a defect: every added rule is anchored to a credential-name
shape so ordinary prose and ``KEY=value`` configuration discussion survive.
Callers receive a COUNT and must never log the matched value.
"""

from __future__ import annotations

import re

from local_operator.redaction_shapes import REDACTION_MARKER, scrub_shapes_with_hits

#: ``Authorization: Bearer <token>`` with a token long enough to be a credential
#: (a short word after "bearer" is prose, not a secret).
_BEARER = re.compile(r"(?i)\b(bearer\s+)([A-Za-z0-9._~+/=-]{16,})")
#: ``export MONGO_DSN=...`` / ``API_TOKEN="..."`` / ``DB_PASSWORD=...``.
#: The name must carry a credential word; the value must be at least 8
#: non-space characters. Quoted values are matched through their closing quote.
_ENV_ASSIGN = re.compile(
    r"(?mi)^(\s*(?:export\s+)?"
    r"[A-Z][A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|DSN|"
    r"DATABASE_URL|CONNECTION_STRING|PRIVATE_KEY)[A-Z0-9_]*\s*=\s*)"
    r"(['\"]?)([^\s'\"#]{8,})\2"
)


def _env_replacement(match: re.Match[str]) -> str:
    name = match.group(1)
    value = match.group(3)
    # A CREDENTIALS_DIR/PATH/HOME names a configuration location, not a
    # secret, and an absolute/relative path is a path in any assignment.
    # Masking these rewrote ordinary setup instructions and shell profiles.
    if re.search(r"_(?:DIR|PATH|HOME)\s*=\s*$", name, re.I):
        return match.group(0)
    if value.startswith(("/", "./", "../", "~/")):
        return match.group(0)
    return match.group(1) + REDACTION_MARKER


def redact_text(text: str) -> tuple[str, int]:
    """Return ``(redacted, count)``; ``count`` is the number of distinct hits.

    ``scrub_shapes_with_hits`` already masks PEM blocks, DSNs, ``NAME=value``
    credential assignments and issuer-prefixed tokens. The local rules cover
    two shapes the design names as residual, and count only what they actually
    changed. The text is passed through the shape scrubber LAST so a bearer/env
    hit is still handled if the table already knows it.
    """
    if not text:
        return text, 0
    text, bearer_hits = _BEARER.subn(lambda match: match.group(1) + REDACTION_MARKER, text)
    before_env = text
    text = _ENV_ASSIGN.sub(_env_replacement, text)
    env_hits = sum(
        1 for match in _ENV_ASSIGN.finditer(before_env) if _env_replacement(match) != match.group(0)
    )
    # The shared shape table also reads a credential-named assignment whose
    # value is a PATH as a secret (``GOOGLE_APPLICATION_CREDENTIALS=~/...``).
    # Those matches are deliberately unchanged above; shield them from the
    # table too, then restore the exact bytes. A path is configuration, not a
    # credential value, and masking it breaks ordinary setup instructions.
    protected: list[str] = []

    def protect_path(match: re.Match[str]) -> str:
        if _env_replacement(match) != match.group(0):
            return match.group(0)
        token = f"__LOP_IMPORT_PATH_{len(protected)}__"
        protected.append(match.group(0))
        return token

    text = _ENV_ASSIGN.sub(protect_path, text)
    cleaned, shape_hits = scrub_shapes_with_hits(text)
    for index, original in enumerate(protected):
        cleaned = cleaned.replace(f"__LOP_IMPORT_PATH_{index}__", original)
    return cleaned, len(shape_hits) + bearer_hits + env_hits


def redact_value(value: object) -> tuple[object, int]:
    """Recursively redact strings inside JSON-ish tool arguments/results."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        total = 0
        out: list[object] = []
        for item in value:
            clean, count = redact_value(item)
            out.append(clean)
            total += count
        return out, total
    if isinstance(value, dict):
        total = 0
        out_dict: dict[str, object] = {}
        for key, item in value.items():
            clean_key, key_count = redact_text(key) if isinstance(key, str) else (key, 0)
            clean_item, item_count = redact_value(item)
            out_dict[clean_key] = clean_item
            total += key_count + item_count
        return out_dict, total
    return value, 0
