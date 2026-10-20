"""Message catalogues over HTTP.

``GET /v1/i18n/catalogues/{locale}/{namespace}`` serves one namespace.
``GET /v1/i18n/catalogues/{locale}`` serves the per-locale eager bundle: every
``wire.*`` namespace that client fetches may consume at startup, flattened, with
a bundle hash/ETag and a manifest of the namespaces deliberately deferred
to the per-namespace route (``wire.settings`` today; the policy lives in
``local_operator.i18n.catalogues.DEFERRED_NAMESPACES``).

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

from local_operator.i18n import catalogues, resolve
from local_operator.i18n.http import etag_matches as _etag_matches
from local_operator.server.models.schemas import CRUDResponse

router = APIRouter(tags=["I18n"])


@router.get("/v1/i18n/catalogues/{locale}", response_model=CRUDResponse)
async def get_wire_bundle(locale: str, request: Request) -> Response:
    """One locale's eager ``wire.*`` messages plus the deferred manifest.

    Same public negotiation-data contract and the same eligibility rules as the
    per-namespace route: an ineligible or unshipped locale is a 404 BEFORE any
    disk work, so a conditional request cannot activate held bytes. A shipped
    locale with no wire files is a valid, empty answer, not a 404: the client
    then holds no catalogue and every identity degrades to the envelope's own
    ``text`` (design §6), which is what an empty translation set means.
    """
    if not resolve.is_shipped(locale) or resolve.eligible_locale(locale) != locale:
        detail = f"No catalogue for {locale}."  # i18n: ignore wire.errors slice owns it
        raise HTTPException(status_code=404, detail=detail)
    try:
        result = catalogues.wire_bundle(locale)
    except catalogues.CatalogueNotFound:
        # A shipped locale can still lose a file between the listing and the
        # read (a bad deployment). Same 404 shape as the sibling route: the
        # namespace inventory is not described through error bodies.
        detail = f"No catalogue for {locale}."  # i18n: ignore wire.errors slice owns it
        raise HTTPException(status_code=404, detail=detail) from None
    except catalogues.CatalogueInvalid as exc:
        # Committed data is validated by the `i18n` check at build time, so a
        # malformed file here is a deployed-artifact bug: fail loudly rather
        # than serve a half-typed map a renderer would trip over.
        raise HTTPException(status_code=500, detail=str(exc)) from None

    etag = f'"{result["content_sha256"]}"'
    headers = {"ETag": etag, "Cache-Control": "no-cache"}
    if _etag_matches(request.headers.get("if-none-match", ""), result["content_sha256"]):
        return Response(status_code=304, headers=headers)
    body = CRUDResponse(
        status=200,
        message="Catalogues retrieved.",  # i18n: ignore wire.errors slice owns it
        result=result,
    )
    return JSONResponse(body.model_dump(), headers=headers)


@router.get("/v1/i18n/catalogues/{locale}/{namespace}", response_model=CRUDResponse)
async def get_catalogue(locale: str, namespace: str, request: Request) -> Response:
    """One catalogue: messages + hash, with ETag revalidation."""
    # Eligibility precedes all disk/validator work: a 304 would activate a
    # client's held bytes just as surely as serving a new representation.
    if not resolve.is_shipped(locale) or resolve.eligible_locale(locale) != locale:
        detail = f"No catalogue for {locale}/{namespace}."  # i18n: ignore wire.errors slice owns it
        raise HTTPException(status_code=404, detail=detail)
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
    if _etag_matches(request.headers.get("if-none-match", ""), sha):
        return Response(status_code=304, headers=headers)
    body = CRUDResponse(
        status=200,
        message="Catalogue retrieved.",  # i18n: ignore wire.errors slice owns it
        result={"messages": messages, "content_sha256": sha},
    )
    return JSONResponse(body.model_dump(), headers=headers)
