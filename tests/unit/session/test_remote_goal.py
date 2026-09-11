"""The compatibility goal setter must not invoke the new submit-and-goal RPC."""

from __future__ import annotations

import asyncio
from typing import Any, cast

import pytest

from local_operator.session.attached import AttachedSession


@pytest.mark.asyncio
async def test_remote_goal_setter_uses_consumed_receipt_and_explicit_clear():
    calls = []

    class Client:
        async def slash_result(self, command, text):
            calls.append((command, text))
            return {"kind": "notice", "data": {"type": "goal_set", "request": text}}

        async def slash(self, command, text):
            raise AssertionError("the legacy slash RPC would also submit a turn")

    remote = object.__new__(AttachedSession)
    remote._client = cast(Any, Client())
    assert remote.set_goal(" ship it ") == "ship it"
    await asyncio.sleep(0)
    assert remote.set_goal("") == ""
    await asyncio.sleep(0)
    assert calls == [("goal", "ship it"), ("goal", "clear")]
