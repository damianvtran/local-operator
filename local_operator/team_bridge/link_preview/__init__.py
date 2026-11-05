"""Slack-style link previews, resolved in CORE so no third-party byte reaches a renderer.

WHY THIS LIVES HERE. The desktop surface, the mobile relay and the mobile app all
sit behind this backend, and the operator's constraint is absolute: metadata and
images are fetched by the server and handed over as artifacts the server has
already vouched for. A renderer that loaded a third-party ``og:image`` itself
would leak every correspondent's reading list to that origin and turn one
message into an arbitrary outbound request. So the transport is
:class:`~local_operator.web_fetch.service.WebFetchService` — the EXISTING hardened
fetcher, never a second HTTP client — and the renderer only ever sees a
content-addressed id it can ask this backend for.

WHAT A PREVIEW MAY CONTAIN. A title, a description, a site name, and two raster
artifacts. Nothing else is stored: no message text, no sender, no tenant. The
artifact store is content-addressed (sha256 of the bytes) and holds no sidecar
metadata at all, so there is nothing in it to leak beyond a public web image the
origin already served.

THE THREE PIECES. ``extract`` turns markup into those facts (declared-charset
decoding, bounded parse, magic-byte sniffing, header-only dimension reads);
``store`` is the bounded content-addressed artifact directory; ``service``
orchestrates one resolution under a single total deadline.
"""

from local_operator.team_bridge.link_preview.service import (
    PER_MESSAGE_CAP,
    TOTAL_DEADLINE_S,
    resolve_preview,
)
from local_operator.team_bridge.link_preview.store import (
    PreviewStore,
    InvalidArtifactId,
)

__all__ = [
    "PER_MESSAGE_CAP",
    "TOTAL_DEADLINE_S",
    "PreviewStore",
    "InvalidArtifactId",
    "resolve_preview",
]
