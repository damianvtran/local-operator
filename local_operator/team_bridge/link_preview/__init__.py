"""Slack-style link previews, resolved in CORE so no third-party byte reaches a renderer.

WHY THIS LIVES HERE. The desktop surface, the mobile relay and the mobile app all
sit behind this backend, and the operator's constraint is absolute: metadata and
images are fetched by the server and handed over as artifacts the server has
already vouched for. A renderer that loaded a third-party ``og:image`` itself
would leak every correspondent's reading list to that origin and turn one
message into an arbitrary outbound request. So the transport is
:class:`~local_operator.web_fetch.service.WebFetchService` — the EXISTING hardened
fetcher, never a second HTTP client — and the renderer only ever sees an artifact
id it can ask this backend for.

WHAT A PREVIEW MAY CONTAIN. A title, a description, a site name, and two raster
artifacts. Nothing else is stored: no message text, no sender, no tenant. The
artifact id is an opaque hash of the ACCOUNT, TENANT and canonical URL that
produced it (:mod:`~local_operator.team_bridge.link_preview.cache`), so the id
names a slot, not a body, and no scope can construct one that names another
scope's blob.

THE FOUR PIECES. ``extract`` turns markup into those facts (declared-charset
decoding, bounded parse, magic-byte sniffing, header-only dimension reads);
``store`` is the bounded blob directory; ``cache`` is the scoped, TTL'd, bounded
and single-flight memory of what each scope was told; ``service`` orchestrates one
resolution under a single total deadline.
"""

from local_operator.team_bridge.link_preview.cache import (
    PreviewCache,
    PreviewCacheStats,
    PreviewScope,
    get_preview_cache,
)
from local_operator.team_bridge.link_preview.service import (
    PER_MESSAGE_CAP,
    TOTAL_DEADLINE_S,
    resolve_preview,
)
from local_operator.team_bridge.link_preview.store import (
    InvalidArtifactId,
    PreviewStore,
)

__all__ = [
    "PER_MESSAGE_CAP",
    "TOTAL_DEADLINE_S",
    "InvalidArtifactId",
    "PreviewCache",
    "PreviewCacheStats",
    "PreviewScope",
    "PreviewStore",
    "get_preview_cache",
    "resolve_preview",
]
