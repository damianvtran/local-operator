"""Lossless folding of a provider model id into a ``RouteIdentity`` field.

``RouteIdentity`` fields are ``StrictIdentifier`` values
(``[A-Za-z0-9][A-Za-z0-9_.:-]*``), so an aggregator model id such as
``deepseek/deepseek-v4-flash-vision-exp`` or ``moonshotai/kimi-k2:free``
cannot be sealed verbatim. Comparability (M1) depends on a route identity
meaning exactly one thing, so the fold has to be REVERSIBLE: two distinct
model ids must never seal to the same identifier, and a reader of a sealed
bundle must be able to recover the exact id the provider was asked for.

The scheme is a classic escape: ``_`` is the escape character, so it is
itself written ``__``; ``/`` becomes ``_s``; any other character outside the
identifier alphabet becomes ``_x<HH>`` (its UTF-8 bytes, lowercase hex).
Every other character -- letters, digits, ``.``, ``:``, ``-`` -- passes
through, which keeps the common OpenRouter shapes readable
(``deepseek_sdeepseek-v4-flash-vision-exp``, ``moonshotai_skimi-k2:free``).
Decoding reads left to right and is unambiguous because ``_`` only ever
introduces one of those three forms.

``scripts/run_episode.py`` also writes the original id into the manifest
metadata (``route_model_id``) so a human never has to decode by hand; the
fold is what makes the identity field itself honest when they do not.
"""

from __future__ import annotations

import re

_PASSTHROUGH = re.compile(r"[A-Za-z0-9.:-]")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_MAX_IDENTIFIER = 128


def fold_model_id(model_id: str) -> str:
    """Encode ``model_id`` as a ``StrictIdentifier``; ``unfold_model_id`` inverts it."""

    if not model_id:
        raise ValueError("model id must not be empty")
    out: list[str] = []
    for char in model_id:
        if char == "_":
            out.append("__")
        elif char == "/":
            out.append("_s")
        elif _PASSTHROUGH.fullmatch(char):
            out.append(char)
        else:
            out.append("".join(f"_x{byte:02x}" for byte in char.encode("utf-8")))
    folded = "".join(out)
    if not _IDENTIFIER.fullmatch(folded):
        # Only reachable when the id starts with a non-alphanumeric: the
        # identifier pattern forbids a leading ``_``/``.``/``:``/``-``, and no
        # provider ships such an id. Refuse rather than invent a prefix that
        # the decoder would then have to guess about.
        raise ValueError(f"model id {model_id!r} cannot start with {model_id[0]!r}")
    if len(folded) > _MAX_IDENTIFIER:
        raise ValueError(f"model id {model_id!r} folds to more than {_MAX_IDENTIFIER} characters")
    return folded


def unfold_model_id(folded: str) -> str:
    """Recover the exact model id ``fold_model_id`` encoded."""

    out: list[str] = []
    pending = bytearray()
    index = 0
    while index < len(folded):
        char = folded[index]
        if char != "_":
            if pending:
                out.append(pending.decode("utf-8"))
                pending.clear()
            out.append(char)
            index += 1
            continue
        code = folded[index + 1 : index + 2]
        if code == "_":
            decoded, width = "_", 2
        elif code == "s":
            decoded, width = "/", 2
        elif code == "x" and re.fullmatch(r"[0-9a-f]{2}", folded[index + 2 : index + 4] or ""):
            pending.append(int(folded[index + 2 : index + 4], 16))
            index += 4
            continue
        else:
            raise ValueError(f"malformed escape at offset {index} in {folded!r}")
        if pending:
            out.append(pending.decode("utf-8"))
            pending.clear()
        out.append(decoded)
        index += width
    if pending:
        out.append(pending.decode("utf-8"))
    return "".join(out)
