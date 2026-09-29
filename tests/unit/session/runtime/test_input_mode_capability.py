"""The ``input-mode-v1`` attach-record advertisement, pinned at its four answers.

The mobile stream gates its sends on this string: PRESENT, the client may put
``input_mode``/``input_path`` on a prompt/steer frame knowing the owner lands
them on the durable user row; ABSENT, the client must not, because the owner
would silently drop them — which is the one failure a producer can neither see
nor recover from. So the token is pinned in four shapes:

* present when BOTH ops take the keyword (the production handle's shape);
* absent on a legacy handle (``FakeHandle``'s signature — every owner from
  before the carriage);
* absent on a ``**kwargs``-only acceptor, because the DISPATCH would not pass
  the keyword (it probes the literal parameter name), so advertising it would
  promise a row field the same build silently drops;
* absent when only ONE op takes it — one token gates both fields AND both
  send modes, so half a carriage must not read as a whole one.

The probe deliberately does not reuse ``_accepts_kw`` (which counts
VAR_KEYWORD as accepting); ``_takes_input_mode``'s docstring in
``session/runtime/server.py`` carries that reasoning and the ``**kwargs`` cell
below is the contract it exists for.
"""

from __future__ import annotations

import pytest

from local_operator.session.runtime.server import RuntimeServer
from local_operator.session.runtime.types import INPUT_MODE_CAPABILITY
from tests.unit.session.runtime.test_server import FakeHandle


class CarryingHandle(FakeHandle):
    """The production shape: both ops read the carriage keywords."""

    async def prompt(  # noqa: ANN001, ANN202
        self, text, images=None, command_id=None, input_mode=None, input_path=None
    ):
        return await self._record("prompt", text)

    async def steer(  # noqa: ANN001, ANN202
        self, text, images=None, command_id=None, input_mode=None, input_path=None
    ):
        return await self._record("steer", text)


class KwargsHandle(FakeHandle):
    """A ``**kwargs``-only acceptor: the dispatch does not pass the keyword."""

    async def prompt(self, text, images=None, command_id=None, **kwargs):  # noqa: ANN001, ANN202
        return await self._record("prompt", text)

    async def steer(self, text, images=None, **kwargs):  # noqa: ANN001, ANN202
        return await self._record("steer", text)


class HalfCarryingHandle(FakeHandle):
    """Only ``prompt`` reads the keyword; the token needs both ops."""

    async def prompt(  # noqa: ANN001, ANN202
        self, text, images=None, command_id=None, input_mode=None, input_path=None
    ):
        return await self._record("prompt", text)


def _capabilities(handle: FakeHandle) -> list[str]:
    return RuntimeServer(handle, kind="tui")._record.capabilities


@pytest.mark.asyncio
async def test_the_token_is_present_when_both_ops_take_the_keyword() -> None:
    assert INPUT_MODE_CAPABILITY in _capabilities(CarryingHandle())


@pytest.mark.asyncio
async def test_the_token_is_absent_on_a_legacy_handle() -> None:
    """``FakeHandle`` carries the pre-carriage signatures verbatim."""
    assert INPUT_MODE_CAPABILITY not in _capabilities(FakeHandle())


@pytest.mark.asyncio
async def test_the_token_is_absent_on_a_kwargs_only_handle() -> None:
    assert INPUT_MODE_CAPABILITY not in _capabilities(KwargsHandle())


@pytest.mark.asyncio
async def test_the_token_needs_both_ops() -> None:
    assert INPUT_MODE_CAPABILITY not in _capabilities(HalfCarryingHandle())
