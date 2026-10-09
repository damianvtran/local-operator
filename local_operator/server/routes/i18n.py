"""Message catalogues over HTTP: ``GET /v1/i18n/catalogues/{locale}/{namespace}``.

The served half of the hybrid delivery decision (RFC §2.1): core owns the
``wire.*`` namespaces and the backend serves them, gated by
``features.i18n`` on ``/v1/capabilities`` — a client that does not see the
capability never calls this route, and nothing changes for it.

Public on purpose, exactly like ``/v1/capabilities``: this is negotiation data
(the backend's own strings plus their hash), not user content, and a client
needs it before it has any session. The path components are grammar-validated
and root-anchored inside ``local_operator.i18n.catalogues`` — a traversal
attempt is a 404 like any other unknown name.

Shapes and caching, all part of the contract:

* ``result.messages`` — the flat message map, verbatim from the committed file;
* ``result.content_sha256`` and the ``ETag`` header — the SAME digest, the
  SHA-256 of the file BYTES (the ledger's ``source_sha256`` rule, §6). A client
  caches by hash and re-fetches on change;
* ``Cache-Control: no-cache`` — reuse must revalidate, because the hash is the
  cache key, not a timer (RFC §2.1's operational note);
* ``If-None-Match`` matching the current ETag short-circuits to 304.

The two sentences this route can emit carry ``# i18n: ignore`` markers: they
are wire-surface prose whose extraction belongs to the `wire.errors` slice, and
the marker is how the ratchet records that they are KNOWN, not missed.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse

from local_operator.i18n import catalogues
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["I18n"])


@router.get("/v1/i18n/catalogues/{locale}/{namespace}", response_model=CRUDResponse)
async def get_catalogue(locale: str, namespace: str, request: Request) -> Response:
    """One catalogue: messages + hash, with ETag revalidation."""
    try:
        sha = catalogues.catalogue_sha256(locale, namespace)
        messages = catalogues.load_catalogue(locale, namespace)
    except catalogues.CatalogueNotFound:
        # Same shape for "no such locale" and "no such namespace": the
        # namespace inventory is served, not discoverable, and a 404 that
        # distinguished them would describe the disk.
        detail = f"No catalogue for {locale}/{namespace}."  # i18n: ignore wire.errors slice owns it
        raise HTTPException(status_code=404, detail=detail) from None
    except catalogues.CatalogueInvalid as exc:
        # Committed data is validated by the `i18n` check at build time, so a
        # malformed file here is a deployed-artifact bug: fail loudly rather
        # than serve a half-typed map a renderer would trip over.
        raise HTTPException(status_code=500, detail=str(exc)) from None

    etag = f'"{sha}"'
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    body = CRUDResponse(
        status=200,
        message="Catalogue retrieved.",  # i18n: ignore wire.errors slice owns it
        result={"messages": messages, "content_sha256": sha},
    )
    return JSONResponse(body.model_dump(), headers=headers)
