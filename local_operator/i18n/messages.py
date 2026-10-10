"""The message envelope: ``{code, params, text}`` (RFC §2.6, decision 6).

Wire surfaces migrate ADDITIVELY to codes: a ``NoticeEvent``, an
``HTTPException`` detail or a wire frame gains ``code``/``params`` while ``text``
keeps being the rendered sentence, so a client that predates codes reads
exactly what it reads today. This module is the one constructor for that shape.

``text`` is resolved from the catalogue when the code is known there and falls
back to the CODE ITSELF when it is not. That fallback is deliberate: at adoption
time — and for any code a later string-extraction slice has not landed yet — a
missing catalogue entry must degrade to something a client can render, never
raise through a path whose whole reason for existing is that old readers cannot
parse codes. The checker, not this function, is where an unextracted literal
goes red.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from . import catalogues
from . import runtime as _runtime


#: The namespace is the code's dotted prefix: `wire.errors.model_unavailable`
#: lives in `catalogues/<locale>/wire.errors.json` (namespace = file boundary,
#: §2.2). Codes with no dot cannot address a namespace and therefore always
#: fall back to their own literal.
def _namespace_of(code: str) -> str | None:
    namespace, dot, _ = code.rpartition(".")
    return namespace if dot and namespace else None


@dataclass(frozen=True)
class Msg:
    """A message id plus its parameters — what generated key functions return."""

    code: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def envelope(self, *, locale: str | None = None) -> dict[str, Any]:
        """This message as ``{code, params, text}`` (see :func:`envelope`)."""
        return envelope(self.code, dict(self.params), locale=locale)


def msg(code: str, **params: Any) -> Msg:
    """Terse constructor for hand-written call sites and tests."""
    return Msg(code=code, params=params)


def envelope(
    code: str,
    params: Mapping[str, Any] | None = None,
    *,
    locale: str | None = None,
) -> dict[str, Any]:
    """``{code, params, text}`` for a message id.

    ``locale=None`` renders ``text`` in ``en`` — the default keeps this
    function free of configuration reads, so callers on hot paths choose when
    to pay for resolution. A caller that already knows the resolved locale
    passes it; anything unshipped or unknown renders from ``en``, which
    :func:`render` enforces by loading the ``en`` catalogue for absent locales.
    """
    params = dict(params or {})
    text = render(code, params, locale=locale)
    return {"code": code, "params": params, "text": text}


def render(code: str, params: Mapping[str, Any], *, locale: str | None = None) -> str:
    """The rendered sentence for ``code``, or the code itself when unknown."""
    namespace = _namespace_of(code)
    if namespace is None:
        return code
    requested = locale or "en"
    source = _lookup(requested, namespace, code)
    if source is None and requested != "en":
        source = _lookup("en", namespace, code)
    if source is None:
        return code
    try:
        return _runtime.render_message(source, params, requested)
    except _runtime.MessageError:
        # A catalogue that does not render is a build-time defect (the parity
        # check parses every message); at RUN time the additive contract wins —
        # the envelope still carries code and params, and `text` degrades to the
        # code rather than raising through an old client's read path.
        return code


def _lookup(locale: str, namespace: str, code: str) -> str | None:
    try:
        messages = catalogues.load_catalogue(locale, namespace)
    except (catalogues.CatalogueNotFound, catalogues.CatalogueInvalid):
        return None
    return messages.get(code)
