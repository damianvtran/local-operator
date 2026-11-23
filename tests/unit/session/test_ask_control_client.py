"""One-shot durable ask control never installs a partial display connection."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, Mock

import pytest

from local_operator.mobile.attach_client import AttachClient
from local_operator.session.attached import AttachedSession
from local_operator.session.owner import SessionOwner


def _facade(*, local=True, record=None):
    facade = AttachedSession.__new__(AttachedSession)
    facade._surface = "desktop"
    facade._session_id = "synthetic-ask-control"
    facade._client = cast(AttachClient, Mock(spec=AttachClient, connected=True))
    control = Mock(spec=AttachClient, connected=True)
    owner = Mock(spec=SessionOwner)
    owner.placement = SimpleNamespace(is_local=local)
    owner.locate.return_value = (record, None)
    owner.make_client.return_value = control
    facade._owner = cast(SessionOwner, owner)
    # Keep the mock handles separate: production methods retain their real
    # signatures while assertions and fault injection use the test doubles.
    ensure_bound = AsyncMock()
    connect_client = AsyncMock()
    facade._ensure_bound = ensure_bound
    facade._connect_client = connect_client
    return facade, control, owner, ensure_bound, connect_client


@pytest.mark.asyncio
async def test_live_owner_control_does_not_wait_for_or_replace_display():
    record = object()
    facade, control, owner, ensure_bound, connect_client = _facade(record=record)
    original = facade._client
    async with facade._ask_control_client() as client:
        assert client is control
        assert facade._client is original
        ensure_bound.assert_not_awaited()
        assert owner.make_client.call_args.kwargs == {
            "surface": "desktop",
            "frontend_state": False,
        }
        assert connect_client.call_args.args == (control, record)
    control.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("local,record", [(True, None), (False, object())])
async def test_cold_and_remote_owners_keep_normal_engage_path(local, record):
    facade, control, owner, ensure_bound, _ = _facade(local=local, record=record)
    async with facade._ask_control_client() as client:
        assert client is facade._client
    ensure_bound.assert_awaited_once()
    owner.make_client.assert_not_called()
    control.close.assert_not_called()
    if not local:
        owner.locate.assert_not_called()


@pytest.mark.asyncio
async def test_failed_control_connect_closes_socket_without_touching_display():
    facade, control, _, _, connect_client = _facade(record=object())
    original = facade._client
    connect_client.side_effect = ConnectionError("owner refused")
    with pytest.raises(ConnectionError, match="owner refused"):
        async with facade._ask_control_client():
            pytest.fail("a failed connection cannot dispatch")
    control.close.assert_called_once()
    assert facade._client is original
