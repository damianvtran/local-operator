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

import functools
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping

from . import catalogues, resolve
from . import runtime as _runtime

_logger = logging.getLogger(__name__)


#: Resolution is FILE-derived, not string-derived: §2.2 keys are
#: `<namespace>.<area>.<element>[.<state>]`, so a code may carry several segments
#: after its namespace and "everything before the last dot" resolves the wrong
#: file for any key deeper than one segment. One message is DEFINED by exactly
#: one catalogue file (the collision rule), so the namespace is the longest
#: dot-prefix whose file CONTAINS the code: a nested namespace
#: (`wire.errors.auth`) addresses its own file, while a key living in the
#: shallower file is not shadowed by a deeper file that does not define it.
@functools.lru_cache(maxsize=4096)
def _namespace_for(root: str, code: str) -> str | None:
    """The namespace whose catalogue defines ``code``, or ``None``.

    ``root`` (the catalogue root, as a string) is part of the cache key so
    fixture trees resolve independently of the shipped corpus, and this sits
    on every render — the probe stats candidate files and parses the first
    that carries the key. A root's corpus is package data and immutable at
    runtime; a test that mutates a tree in place can call
    ``_namespace_for.cache_clear()``. The probe reads `en`, the authored
    source every other locale mirrors (parity is enforced), so discovery does
    not depend on which locale is being rendered.
    """
    segments = code.split(".")
    for cut in range(len(segments) - 1, 0, -1):
        namespace = ".".join(segments[:cut])
        try:
            messages = catalogues.load_catalogue("en", namespace)
        except (catalogues.CatalogueNotFound, catalogues.CatalogueInvalid):
            continue
        if code in messages:
            return namespace
    return None


@dataclass(frozen=True)
class Msg:
    """A message id plus its parameters — what generated key functions return."""

    code: str
    params: Mapping[str, Any] = field(default_factory=dict)

    def envelope(self, *, code: str | None = None, locale: str | None = None) -> dict[str, Any]:
        """This message as ``{code, params, text}`` (see :func:`envelope`).

        ``code`` overrides the id ON THE WIRE while ``params``/``text`` still
        come from THIS message. It exists for refusals whose wire code is a
        MACHINE category that an existing client switches on by equality —
        the desktop plane's vocabulary (``runtime_busy``, ``project_not_found``,
        ``session_is_remote``, read as ``error.code`` by the UI's
        ``RUNTIME_BUSY_CODE`` checks). Renaming those to message ids would
        break every released client build, so the token rides VERBATIM and
        only the sentence is catalogued; every other call site keeps the
        message id as the code (the default here).
        """
        body = envelope(self.code, dict(self.params), locale=locale)
        if code is not None:
            body["code"] = code
        return body


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
    :func:`render` enforces through the generated shipped set.
    """
    params = dict(params or {})
    text = render(code, params, locale=locale)
    return {"code": code, "params": params, "text": text}


@dataclass(frozen=True)
class RenderedMessage:
    """Internal raw result plus actual catalogue provenance, never wire metadata.

    A preference is not the direction of fallback copy. None means unknown
    code, failed render, or opaque passthrough prose: none can claim a catalogue
    language. No bidi controls are introduced here; only a paint adapter may
    project these bytes for display. This record is not an envelope/schema change.
    """

    text: str
    source_locale: str | None


def render(code: str, params: Mapping[str, Any], *, locale: str | None = None) -> str:
    """The rendered sentence for ``code``, or the code itself when unknown."""
    return render_record(code, params, locale=locale).text


def render_record(
    code: str, params: Mapping[str, Any], *, locale: str | None = None
) -> RenderedMessage:
    """Source-aware projection using the SAME cached lookup and ICU formatter.

    Raw render/envelope callers continue receiving exactly the old text. Typed
    numbers and dates reach ICU unchanged; annotation happens only afterwards.
    This deliberately does not generate parameter offsets or guess technical
    tokens. A span-aware ICU collector is a separate, larger adoption slice.
    """
    namespace = _namespace_for(str(catalogues.catalogue_root()), code)
    if namespace is None:
        return RenderedMessage(code, None)
    requested = resolve.eligible_locale(locale)
    source_locale = requested
    source = _lookup(requested, namespace, code)
    if source is None and requested != "en":
        source = _lookup("en", namespace, code)
        source_locale = "en"
    if source is None:
        return RenderedMessage(code, None)
    if source_locale != requested:
        _logger.debug(
            "i18n fallback code=%s requested=%s source=%s", code, requested, source_locale
        )
    # Formatting follows the source: Russian rules on English copy select
    # "one" for 21, producing "21 job" instead of the English "21 jobs".
    try:
        rendered = _runtime.render_message(source, params, source_locale)
        provenance = None if _runtime.is_passthrough_message(source) else source_locale
        return RenderedMessage(rendered, provenance)
    except _runtime.MessageError:
        # A catalogue that does not render is a build-time defect (the parity
        # check parses every message); at RUN time the additive contract wins —
        # the envelope still carries code and params, and `text` degrades to the
        # code rather than raising through an old client's read path.
        return RenderedMessage(code, None)


def _lookup(locale: str, namespace: str, code: str) -> str | None:
    try:
        messages = catalogues.load_catalogue(locale, namespace)
    except (catalogues.CatalogueNotFound, catalogues.CatalogueInvalid):
        return None
    return messages.get(code)
