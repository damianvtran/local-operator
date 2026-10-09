"""The pure supplement document assembler (lane C0; memo §2.6 "One assembler implementation").

WHY ONE ASSEMBLER. Four surfaces (desktop UI, relay web, native WebView, and later any
other) render the same component, and the memo's invariant is "identical bytes" per
surface: the theme guarantee, the CSP and the honesty check all rest on every host
mounting exactly the document this module builds. Both document routes (desktop
``GET /v1/desktop/sessions/{id}/supplements/{digest}/document`` and relay
``GET /api/sessions/{id}/supplements/{digest}/document``) call :func:`assemble_document`
and return ``{html}`` as JSON -- never ``text/html``, because no host may navigate a frame
to a URL (memo §4.1).

PURE. A function of (body, data, the vendored prelude): no I/O beyond reading the two
packaged prelude files once, no session, no clock. That is what makes the fixtures under
``tests/fixtures/supplements/documents`` byte-stable and checkable by every lane.

THE ASSEMBLED SHAPE (every byte is load-bearing; ``tests/unit/supplements`` pins it)::

    <!doctype html><html><head><meta charset="utf-8">
    <meta http-equiv="Content-Security-Policy" content="{CSP}">
    <style>{prelude.css}</style></head><body>
    <script type="application/json" id="lo-data">{merged <data> JSON}</script>
    <script>{prelude.js}</script>
    {component body}
    </body></html>

INJECTION ORDER (the memo's §2.6 sketch prints the prelude after the body, then its own
note says the opposite; the note is the contract). The prelude ``<script>`` sits BEFORE
the body so ``window.LO`` exists when the component's inline script runs, and AFTER the
``lo-data`` block because the prelude reads ``LO.data`` synchronously at load. Its message
listener is installed synchronously too, so the host's first theme ``postMessage`` -- sent
at frame ``load`` -- cannot be lost. The document stays hidden (``visibility:hidden`` in
the prelude CSS) until that first theme arrives, with a 400 ms ``prefers-color-scheme``
fallback as a last resort.

THE CSP is a ``<meta>`` and not a header on purpose: the tunnel gateway forwards a fixed
response-header set and overwrites CSP, and the iframe ``csp`` attribute is unimplemented
in Chromium (memo §2.7 "Tunnel header stripping", §4.1). The sandbox (``allow-scripts``
only, opaque origin) is the boundary; this policy closes the network (``connect-src
'none'``) which is what makes ``'unsafe-inline'`` safe here.

NOT HERE: extracting ``<data>`` from a generated component, validating it, or storing it
(validator and persistence, lane C1). The caller hands over the already-merged data.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib import resources
from typing import Any, Final, Mapping

#: Bump when ``prelude.css``/``prelude.js`` change, in the same commit. The prelude ships
#: INSIDE core so one fix reaches all four surfaces with no UI or mobile release; the
#: version is how a host or a bug report names which prelude produced a document, and
#: ``tests/unit/supplements/test_prelude.py`` pins the pair's digest to this number so an
#: edit that forgets to bump it fails.
PRELUDE_VERSION: Final = 1

#: The document CSP (memo §4.1), verbatim.
CONTENT_SECURITY_POLICY: Final = (
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
    "img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; "
    "form-action 'none'; base-uri 'none'"
)

#: The id of the JSON data block the prelude reads into ``LO.data``.
DATA_ELEMENT_ID: Final = "lo-data"

_PRELUDE_PACKAGE: Final = "local_operator.supplements"

#: ``<data>`` elements of a stored component. ``<data>`` is a real (inline) HTML element, so
#: the extractor must remove it from the body or it would paint its JSON as text.
_DATA_BLOCK_RE: Final = re.compile(r"<data>(.*?)</data>", re.DOTALL | re.IGNORECASE)


@cache
def prelude_css() -> str:
    """The vendored minified stylesheet (read once per process)."""
    return _read_prelude("prelude.css")


@cache
def prelude_js() -> str:
    """The vendored minified script (read once per process)."""
    return _read_prelude("prelude.js")


def _read_prelude(name: str) -> str:
    return (resources.files(_PRELUDE_PACKAGE) / "prelude" / name).read_text(encoding="utf-8")


def data_json(data: Mapping[str, Any] | None) -> str:
    """Serialise the merged ``<data>`` datasets for the ``lo-data`` block.

    ``<`` is written as ``\\u003c``: inside a ``<script>`` element the HTML parser ends the
    block at the first ``</script`` (and treats ``<!--`` specially) whatever the script's
    ``type``, so a dataset value containing either would let data break out of its block
    into markup. ``JSON.parse`` reads the escape back as the same character, so the data
    the prelude sees is unchanged. Key order is preserved and the separators are compact,
    so equal data gives equal bytes.
    """
    text = json.dumps(dict(data or {}), ensure_ascii=False, separators=(",", ":"))
    return text.replace("<", "\\u003c")


def split_component(blob: str) -> tuple[str, dict[str, Any]]:
    """Split a STORED component into ``(body, merged data)``.

    CONTRACT RESOLUTION (the memo is silent on where ``<data>`` lives at rest). The
    attachment blob is the generated component with its ``<component>``/``<html>``
    wrapper removed: zero or more ``<data>{json}</data>`` elements followed by the body
    markup. That keeps the digest -> document route self-sufficient (a row carries a
    digest and nothing else) and puts the datasets in the bytes the content hash covers,
    so the validator's provenance check and the host's rendered data cannot diverge.

    Each ``<data>`` JSON must be an object of datasets; several blocks merge in order, and
    a dataset id that appears twice with different content is an error (silently letting
    one win would change what the chart plots). Raises ``ValueError`` for either problem;
    the caller (the route, the validator) decides what that means for its surface.
    """
    merged: dict[str, Any] = {}
    for match in _DATA_BLOCK_RE.finditer(blob):
        try:
            block = json.loads(match.group(1))
        except json.JSONDecodeError as error:
            raise ValueError(f"<data> is not valid JSON: {error.msg}") from None
        if not isinstance(block, dict):
            raise ValueError("<data> must be a JSON object of datasets")
        for key, value in block.items():
            if key in merged and merged[key] != value:
                raise ValueError(f"dataset {key!r} is declared twice with different content")
            merged[key] = value
    return _DATA_BLOCK_RE.sub("", blob).strip("\n"), merged


def assemble_stored_component(blob: str) -> str:
    """:func:`assemble_document` over a stored blob; what both document routes serve."""
    body, data = split_component(blob)
    return assemble_document(body, data)


def assemble_document(body: str, data: Mapping[str, Any] | None = None) -> str:
    """Build the complete HTML document for one stored component body.

    ``body`` is the generated component markup WITHOUT the prelude (what
    ``AttachmentStore.put_bytes(raw, "text/html")`` holds). ``data`` is the merged
    ``<data>`` JSON object, ``{dataId: {title, columns, rows}}``.

    Built by concatenation, never ``str.format``/``replace``: the prelude and the body
    are arbitrary text full of braces and ``%``, and a substitution pass over them is how
    a document silently changes.
    """
    return "".join(
        (
            '<!doctype html><html><head><meta charset="utf-8">\n',
            '<meta http-equiv="Content-Security-Policy" content="',
            CONTENT_SECURITY_POLICY,
            '">\n<style>',
            prelude_css(),
            "</style></head><body>\n",
            '<script type="application/json" id="',
            DATA_ELEMENT_ID,
            '">',
            data_json(data),
            "</script>\n<script>",
            prelude_js(),
            "</script>\n",
            body,
            "\n</body></html>",
        )
    )
