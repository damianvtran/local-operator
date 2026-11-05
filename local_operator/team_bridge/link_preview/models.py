"""The wire shape of ONE link preview, shared by every surface that asks for one.

There is exactly one definition of these shapes, and it lives in core rather
than in the desktop route, because the mobile relay and the phone app call the
same resolution through the same backend. A second copy in a route module is how
two surfaces drift into different field names for the same card.

``siteName`` is camelCased on the wire while the Python attribute is snake_case:
the renderer's contract names it that way, and FastAPI serialises a response
model by its ``serialization_alias``.

Every field except ``kind`` is optional and omitted when absent
(``response_model_exclude_none``), so a refusal is a two-field document rather
than a card with blank strings a renderer might try to draw.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Why a URL produced no card. The renderer switches on these strings, so they
#: are a closed vocabulary: ``no-metadata`` (the page said nothing), ``local-or-
#: private`` (the address guard refused it, including on a redirect hop),
#: ``unsupported-scheme`` (not http/https), ``blocked-host`` (no host to judge, or
#: a chain we could not complete), ``content-type`` (wrong declared type),
#: ``not-html`` (declared as HTML but carries no markup), ``too-large`` (the page
#: declared more bytes than a card is worth reading) and ``cap`` (this message
#: already has :data:`~local_operator.team_bridge.link_preview.service.PER_MESSAGE_CAP`
#: previews).
AbsentReason = Literal[
    "no-metadata",
    "local-or-private",
    "unsupported-scheme",
    "blocked-host",
    "content-type",
    "too-large",
    "not-html",
    "cap",
]


class FaviconArtifact(BaseModel):
    """A stored favicon: the id to ask the artifact route for, and its real mime."""

    id: str
    mime: str


class ImageArtifact(BaseModel):
    """A stored ``og:image``, with the dimensions read from its own header.

    ``width``/``height`` may be ``null`` when the format's header does not carry
    them cheaply. They are never obtained by decoding pixels, and a renderer that
    needs the ratio must fall back to its own layout rather than fetch the bytes
    out-of-band.
    """

    id: str
    mime: str
    width: int | None = None
    height: int | None = None


class LinkPreview(BaseModel):
    """What a card renders. Nothing here is message text, sender or tenant."""

    url: str
    title: str = ""
    description: str = ""
    site_name: str = Field(default="", serialization_alias="siteName")
    favicon: FaviconArtifact | None = None
    image: ImageArtifact | None = None


class LinkPreviewResolution(BaseModel):
    """The one answer shape: a card, a coded absence, or a retryable error."""

    kind: Literal["ready", "absent", "error"]
    preview: LinkPreview | None = None
    reason: AbsentReason | None = None
    retryable: bool | None = None
