"""The Saver canonical byte form and ``source_digest`` (contract §3.1, rev4).

WHY BYTE-EXACT. The client verifies its chain by re-digesting its own upload
projection epoch by epoch (§3.2 step 1) and compares that with the
``source_digest`` agent-server computed over the same bytes. One differing byte
between this module and the Go writer truncates the chain on every turn and the
provider prefix cache never hits. So the rule is written down here completely,
and ``tests/unit/saver/test_canonical.py`` asserts it against agent-server's
golden vectors.

THE RULE (rev4, S2-2):

- objects: keys sorted by code point (identical to Go's ``sort.Strings`` over
  UTF-8 for every valid scalar value); ``,`` and ``:`` separators, no
  whitespace;
- strings: escape ONLY ``"`` → ``\\"``, ``\\`` → ``\\\\`` and the C0 controls —
  ``\\b \\f \\n \\r \\t`` in short form, every other byte below 0x20 as lowercase
  ``\\u00xx``. EVERYTHING else is raw UTF-8, including U+2028/U+2029, U+007F,
  ``/``, ``<``, ``>`` and ``&``. ``json.dumps(ensure_ascii=False)`` produces
  exactly this (rev3's U+2028 replacement is gone);
- numbers: the literal the writer chose. Values this client builds are ``int``
  and ``bool`` only (``image_ref`` sizes and dimensions), so no float formatting
  question arises on the upload path;
- ``source_digest`` = ``"sha256:"`` + lowercase hex of SHA-256 over the bytes.

The STRICT PARSER below is the response-side half: an untrusted Saver response
is decoded with duplicate keys, non-JSON constants, lone surrogates and nesting
beyond the contract's depth cap all refused, so two parsers can never read one
body two ways (§6.1 check 1, S-27).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

#: The contract's nesting cap (§3, gap 4): the outer array is level 1.
MAX_DEPTH = 128
#: Longest single string a decoded body may carry: the 8 MiB body cap itself.
#: Contract r5 (S3-6) removed the 1 MB per-string cap because one real tool
#: result may be multi-megabyte; the body cap and the response cap R are what
#: bound the work.
MAX_STRING_CHARS = 8 * 1024 * 1024
#: Most keys one object may carry (§3 structural caps).
MAX_KEYS_PER_OBJECT = 64


class CanonicalError(ValueError):
    """The value cannot be put in canonical form, or a body is not strict JSON."""


def _encode(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")  # i18n: ignore JSON literal token, not copy
    elif value is True:
        out.append("true")  # i18n: ignore JSON literal token, not copy
    elif value is False:
        out.append("false")  # i18n: ignore JSON literal token, not copy
    elif isinstance(value, str):
        # ensure_ascii=False is exactly the rev4 string rule: short escapes for
        # \b \f \n \r \t, \u00xx for the other C0 bytes, everything else raw.
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, int):
        out.append(str(value))
    elif isinstance(value, dict):
        out.append("{")
        for index, key in enumerate(sorted(value)):
            if not isinstance(key, str):
                raise CanonicalError("canonical: object keys must be strings")
            if index:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _encode(value[key], out)
        out.append("}")
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for index, item in enumerate(value):
            if index:
                out.append(",")
            _encode(item, out)
        out.append("]")
    else:
        # Floats never reach the upload (see the module docstring); refusing them
        # keeps a future caller from silently choosing Python's float repr for
        # bytes Go must reproduce.
        raise CanonicalError(f"canonical: unsupported type {type(value).__name__}")


def canonical_bytes(value: Any) -> bytes:
    """The canonical UTF-8 bytes of ``value`` (dicts, lists, str, int, bool, None)."""
    out: list[str] = []
    _encode(value, out)
    try:
        return "".join(out).encode("utf-8")
    except UnicodeEncodeError as exc:
        # A lone surrogate cannot be UTF-8; the server rejects it, so the client
        # must never digest or upload one.
        raise CanonicalError("canonical: lone surrogate") from exc


def digest(value: Any) -> str:
    """``source_digest`` of a projected span: ``"sha256:"`` + hex."""
    return digest_bytes(canonical_bytes(value))


def digest_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _reject_constant(name: str) -> Any:
    raise CanonicalError(f"non-JSON constant {name}")


def _object_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    if len(pairs) > MAX_KEYS_PER_OBJECT:
        raise CanonicalError("too many keys in one object")
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise CanonicalError("duplicate object key")
        out[key] = value
    return out


def _depth_scan(text: str) -> None:
    # json.loads recurses in C up to the interpreter limit, so a deeply nested
    # body would raise RecursionError instead of a clean refusal. Scan nesting
    # outside string literals first, with the contract's own cap.
    depth = 0
    in_string = escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > MAX_DEPTH:
                raise CanonicalError("nesting too deep")
        elif char in "]}":
            depth -= 1


def _check_values(value: Any) -> None:
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            if len(item) > MAX_STRING_CHARS:
                raise CanonicalError("string too long")
            try:
                item.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise CanonicalError("lone surrogate escape") from exc
        elif isinstance(item, dict):
            stack.extend(item.keys())
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)


def parse_strict(data: bytes) -> Any:
    """Decode ``data`` as strict JSON, or raise :class:`CanonicalError`.

    Refused, never repaired: invalid UTF-8, a byte-order mark, a non-JSON
    constant (``NaN``/``Infinity``), a duplicate key, a lone surrogate escape,
    nesting deeper than :data:`MAX_DEPTH`, trailing data, an over-long string or
    an object with more than :data:`MAX_KEYS_PER_OBJECT` keys.
    """
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CanonicalError("invalid UTF-8") from exc
    if text.startswith("\ufeff"):
        raise CanonicalError("byte-order mark")
    _depth_scan(text)
    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_hook,
            parse_constant=_reject_constant,
        )
    except CanonicalError:
        raise
    except (ValueError, RecursionError) as exc:
        raise CanonicalError("invalid JSON") from exc
    _check_values(value)
    return value
