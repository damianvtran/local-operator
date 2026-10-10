"""The edit declaration⇔wiring join, walked across every rung.

Rule (media wave-2 edit lane): a rung declares ``sources != NONE`` only in
the same change that wires its edit path, and the cascade's capability filter
pre-records every other rung's skip. This module drives each rung's executor
WITH A SOURCE against a transport that refuses everything, and asserts the
two drift directions cannot happen:

- a declared-capable rung must ATTEMPT an edit request — a ``RungSkipped``
  with the unsupported class, or no request at all, fails the test;
- a declared-``NONE`` rung must raise the unsupported skip and issue NO
  request (an incapable rung must not be called — the mutation-style
  negative the filter relies on).

Wire shapes are each rung's own test file's job; "attempted" is unambiguous
here because the mock refuses every request with a 400.
"""

from __future__ import annotations

import httpx
import pytest

from local_operator.artifacts.rung import CancelHandle, RungSkipped, SourceSupport
from local_operator.imagegen import ImageRoute, cascade
from local_operator.imagegen import rungs as image_rungs
from local_operator.imagegen import (
    rungs_google,
    rungs_openai_sub,
    rungs_openrouter,
    rungs_xai,
)

SOURCE = "data:image/png;base64,AAAA"


def _refusing_client() -> tuple[httpx.AsyncClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(400, json={"error": {"message": "refused for the test"}})

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), seen


async def _drive(route: str, client: httpx.AsyncClient) -> None:
    """Call ``route``'s executor with a source; raises whatever it raises."""
    if route == ImageRoute.RADIENT:
        await image_rungs.run_radient(
            prompt="edit",
            base_url="https://hub.test",
            credential="cred",
            num_images=1,
            image_size="square_hd",
            seed=None,
            strength=None,
            source_url=SOURCE,
            model=None,
            handle=CancelHandle(),
            emit=None,
            pause=None,
            client=client,
        )
    elif route == ImageRoute.FAL:
        await image_rungs.run_fal(
            prompt="edit",
            key="fk",
            num_images=1,
            image_size="square_hd",
            seed=None,
            strength=None,
            source_url=SOURCE,
            model=None,
            handle=CancelHandle(),
            emit=None,
            pause=None,
            base_url="https://queue.fal.test",
            client=client,
        )
    elif route == ImageRoute.OPENAI:
        await image_rungs.run_openai(
            prompt="edit",
            key="sk",
            num_images=1,
            image_size="square_hd",
            source_url=SOURCE,
            model=None,
            emit=None,
            pause=None,
            base_url="https://oai.test/v1",
            client=client,
        )
    elif route == ImageRoute.OPENAI_SUB:
        await rungs_openai_sub.run_openai_sub(
            prompt="edit",
            access_token="tok",
            account_id=None,
            num_images=1,
            image_size="square_hd",
            source_url=SOURCE,
            seed=None,
            model=None,
            emit=None,
            pause=None,
            client=client,
        )
    elif route == ImageRoute.GOOGLE:
        await rungs_google.run_google(
            prompt="edit",
            key="gk",
            num_images=1,
            image_size="square_hd",
            source_url=SOURCE,
            seed=None,
            model=None,
            emit=None,
            pause=None,
            client=client,
        )
    elif route == ImageRoute.XAI:
        await rungs_xai.run_xai(
            prompt="edit",
            key="xk",
            num_images=1,
            image_size="square_hd",
            source_url=SOURCE,
            seed=None,
            model=None,
            emit=None,
            pause=None,
            client=client,
        )
    elif route == ImageRoute.OPENROUTER:
        await rungs_openrouter.run_openrouter(
            prompt="edit",
            key="ork",
            num_images=1,
            image_size="square_hd",
            source_url=SOURCE,
            seed=None,
            model=None,
            emit=None,
            pause=None,
            client=client,
        )
    else:  # pragma: no cover - the order constant is the closed set
        raise AssertionError(f"no executor driven for {route}")


@pytest.mark.asyncio
async def test_the_edit_declaration_matches_the_wired_path_for_every_rung() -> None:
    for route in cascade.IMAGE_RUNG_ORDER:
        capable = cascade.RUNG_SPECS[route].sources is not SourceSupport.NONE
        client, seen = _refusing_client()
        raised: BaseException | None = None
        try:
            await _drive(route, client)
        except BaseException as exc:  # noqa: BLE001 - the join is what is asserted
            raised = exc
        finally:
            await client.aclose()
        if capable:
            assert not isinstance(
                raised, RungSkipped
            ), f"{route} declares sources != NONE but raised the skip: {raised}"
            assert raised is not None, f"{route} declared capable and the mock refused everything"
            assert seen, f"{route} declared capable but issued no request"
        else:
            assert isinstance(
                raised, RungSkipped
            ), f"{route} declares sources=NONE but did not raise the unsupported skip: {raised!r}"
            assert raised.reason_class == "unsupported"
            assert seen == [], f"{route} declares NONE yet issued a request"
