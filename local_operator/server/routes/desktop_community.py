"""Desktop routes for the Local Operator Community: ``/v1/desktop/community/*``.

RFC v2.4 §7.2: the desktop renderer feature ``features/community`` reaches the
community through Electron IPC → these routes → the shared Python client
(``local_operator/community``). The routes are a thin, typed door:

* behind ``require_desktop`` (router dependency), so they inherit the desktop
  plane's bearer + Origin rule, the ``/v1/desktop/`` ``Cache-Control:
  no-store`` header and the app's 422 handler that keeps rejected request
  bodies out of validation responses (``server/app.py``
  ``desktop_validation_error``);
* exactly the member operations in ``community.member_ops.MEMBER_OPERATIONS``,
  registered from that table so the desktop and relay doors cannot drift;
* dark: every route answers ``404 {"code": "community_disabled"}`` unless
  ``LOCAL_OPERATOR_COMMUNITY`` is on (``community.community_enabled``).

OWNER GUARD. Every route takes an optional ``expected_account_id`` query
parameter: the account the renderer showed when the user acted. The client
compares it with the pinned Radient account before any request leaves and
answers ``409 account_mismatch`` on a difference (the Team Bridge desktop
routes' anti-replay receipt). Omitted, the pinned account acts.

REFUSALS are ``{"detail": {"code": ..., <facts>}}`` with the hub's own status
class (see ``community/errors.py``): ``refused`` carries ``category`` and
``reference``; ``rate_limited`` carries ``retry_after`` + ``scope`` and a
``Retry-After`` header; ``guidelines_changed`` carries the new
``guidelines_version``; ``moderation_unavailable`` and ``images_unavailable``
are distinct codes. No English sentence is authored here — the renderer owns
the localized copy — and nothing from the hub's ``message`` is echoed.

SUCCESS is the desktop envelope (``CRUDResponse``) with the typed model dumped
as ``result``.
"""

from typing import Annotated, Any, Callable

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel

from local_operator.community import community_enabled
from local_operator.community.member_ops import MEMBER_OPERATIONS, MemberOperation
from local_operator.config import ConfigManager
from local_operator.server.dependencies import get_config_manager
from local_operator.server.desktop import require_desktop
from local_operator.server.models.schemas import CRUDResponse
from local_operator.server.routes.desktop_sessions import reply

PREFIX = "/v1/desktop/community"

router = APIRouter(tags=["Desktop community"], dependencies=[Depends(require_desktop)])


def _client(config_manager: ConfigManager, expected_account_id: str | None) -> Any:
    """The transport, imported inside the handler (keeps httpx off app boot).

    Module-level so tests can substitute a client wired to a fake hub.
    """

    from local_operator.community.client import CommunityClient

    return CommunityClient(config_manager.config_dir, expected_account_id=expected_account_id)


def _refuse(error: Exception) -> HTTPException:
    from local_operator.community.member_ops import refusal

    status, body = refusal(error)  # type: ignore[arg-type]
    headers = None
    retry_after = body.get("retry_after")
    if status == 429 and isinstance(retry_after, int):
        headers = {"Retry-After": str(retry_after)}
    return HTTPException(status, body, headers=headers)


async def _serve(
    op: MemberOperation,
    request: Request,
    config_manager: ConfigManager,
    expected_account_id: str | None,
    *,
    query: BaseModel | None = None,
    body: BaseModel | None = None,
) -> CRUDResponse[Any]:
    from local_operator.community.errors import CommunityError
    from local_operator.community.member_ops import run

    if not community_enabled():
        raise HTTPException(404, {"code": "community_disabled"})
    client = _client(config_manager, expected_account_id)
    try:
        _status, result = await run(
            op.name, client, dict(request.path_params), query=query, body=body
        )
    except CommunityError as error:
        raise _refuse(error) from None
    return reply(result)


_ACCOUNT = Query(default=None, min_length=1, max_length=128)


def _endpoint(op: MemberOperation) -> Callable[..., Any]:
    """One FastAPI endpoint for one allowlisted operation.

    Built per row so FastAPI sees the row's real body/query models: that is
    what puts body validation (and therefore the scrubbed 422) in front of the
    handler, instead of hand-parsing JSON here.
    """

    if op.body is not None:
        body_model = op.body

        async def with_body(
            request: Request,
            body: body_model,  # type: ignore[valid-type]
            expected_account_id: str | None = _ACCOUNT,
            config_manager: ConfigManager = Depends(get_config_manager),
        ) -> CRUDResponse[Any]:
            return await _serve(op, request, config_manager, expected_account_id, body=body)

        return with_body
    if op.query is not None:
        query_model = op.query

        async def with_query(
            request: Request,
            # A query-parameter MODEL (FastAPI >= 0.115): validated like a
            # body, so a malformed cursor is the scrubbed 422 too, and the
            # model's ``extra="forbid"`` refuses unknown parameters.
            query: Annotated[query_model, Query()],  # type: ignore[valid-type]
            expected_account_id: str | None = _ACCOUNT,
            config_manager: ConfigManager = Depends(get_config_manager),
        ) -> CRUDResponse[Any]:
            return await _serve(op, request, config_manager, expected_account_id, query=query)

        return with_query

    async def bare(
        request: Request,
        expected_account_id: str | None = _ACCOUNT,
        config_manager: ConfigManager = Depends(get_config_manager),
    ) -> CRUDResponse[Any]:
        return await _serve(op, request, config_manager, expected_account_id)

    return bare


for _op in MEMBER_OPERATIONS:
    router.add_api_route(
        PREFIX + _op.path,
        _endpoint(_op),
        methods=[_op.method],
        response_model=CRUDResponse[Any],
        name=f"desktop_community_{_op.name}",
    )
