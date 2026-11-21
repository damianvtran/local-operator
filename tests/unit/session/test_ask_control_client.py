"""One-shot durable ask control never installs a partial display connection."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from local_operator.session.attached import AttachedSession


def _facade(*, local=True, record=None):
    facade = AttachedSession.__new__(AttachedSession)
    facade._surface = "desktop"
    facade._session_id = "synthetic-ask-control"
    facade._client = SimpleNamespace(connected=True)
    control = SimpleNamespace(close=Mock(), connected=True)
    facade._owner = SimpleNamespace(
        placement=SimpleNamespace(is_local=local),
        locate=Mock(return_value=(record, None)),
        make_client=Mock(return_value=control),
    )
    facade._ensure_bound = AsyncMock()
    facade._connect_client = AsyncMock()
    return facade, control


@pytest.mark.asyncio
async def test_live_owner_control_does_not_wait_for_or_replace_display():
    record = object()
    facade, control = _facade(record=record)
    original = facade._client
    async with facade._ask_control_client() as client:
        assert client is control
        assert facade._client is original
        facade._ensure_bound.assert_not_awaited()
        assert facade._owner.make_client.call_args.kwargs == {
            "surface": "desktop",
            "frontend_state": False,
        }
        assert facade._connect_client.call_args.args == (control, record)
    control.close.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("local,record", [(True, None), (False, object())])
async def test_cold_and_remote_owners_keep_normal_engage_path(local, record):
    facade, control = _facade(local=local, record=record)
    async with facade._ask_control_client() as client:
        assert client is facade._client
    facade._ensure_bound.assert_awaited_once()
    facade._owner.make_client.assert_not_called()
    control.close.assert_not_called()
    if not local:
        facade._owner.locate.assert_not_called()


@pytest.mark.asyncio
async def test_failed_control_connect_closes_socket_without_touching_display():
    facade, control = _facade(record=object())
    original = facade._client
    facade._connect_client.side_effect = ConnectionError("owner refused")
    with pytest.raises(ConnectionError, match="owner refused"):
        async with facade._ask_control_client():
            pytest.fail("a failed connection cannot dispatch")
    control.close.assert_called_once()
    assert facade._client is original
