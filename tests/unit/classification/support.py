"""Test doubles and small builders for the classification suite.

Fixtures live in ``conftest.py``; this module holds the pieces that are imported
by name.

Two things every test here needs and none wants to re-derive:

* a :class:`~local_operator.credentials.CredentialManager` pointed at a temp dir
  (the real class — its env fallback is part of what the credential tests
  exercise, and ``tests/conftest.py`` already clears the ambient keys);
* a way to replace the three cascade legs without a socket, via
  ``vendors.VENDOR_CLASSES`` (see the ``install_legs`` fixture for why that is
  the single patch point).
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass, field
from typing import Any

from pydantic import SecretStr

from local_operator.classification.context import Candidate
from local_operator.classification.types import (
    Answer,
    DecisionRequest,
    DecisionResponse,
    DecisionVendorError,
)

#: The credential every armed test manager carries. Shared so a test that asserts
#: "the key is not in this message" can pass the same value to the scrubber.
TEST_KEY = "unit-test-key"


def candidate(
    name: str,
    *,
    kind: str = "skill",
    description: str = "does a thing",
    resource_url: str | None = None,
) -> Candidate:
    return Candidate(
        kind=kind,  # type: ignore[arg-type]  # tests deliberately pass odd kinds
        name=name,
        description=description,
        resource_url=resource_url or f"{kind}://{name}",
    )


def choice_response(question_id: str, choice: str, *, confidence: float = 1.0) -> DecisionResponse:
    """A one-answer choice response, as a leg would return it."""
    return DecisionResponse(
        vendor="stub",
        model="stub-model",
        answers={
            question_id: Answer(
                id=question_id,
                kind="choice",
                value=choice,
                probabilities={choice: confidence, "none": 1 - confidence},
                confidence=confidence,
            )
        },
        input_tokens=100,
        output_tokens=10,
        cost_usd=0.00002,
    )


@dataclass
class LegBehaviour:
    """What one stub leg does, and what it recorded.

    ``script`` is a queue: each call pops the next entry, which is either a
    :class:`DecisionResponse` to return or an exception to raise; the last entry
    repeats, so a test can describe "always fails" in one entry. An empty queue
    answers with an empty (but valid) response.
    """

    name: str
    credential: bool = True
    credential_raises: bool = False
    script: list[Any] = field(default_factory=list)
    delay_s: float = 0.0
    calls: list[DecisionRequest] = field(default_factory=list)
    credential_calls: int = 0
    clients: list[Any] = field(default_factory=list)

    def decision(self) -> Any:
        """The next scripted outcome, repeating the final entry."""
        if not self.script:
            return DecisionResponse(vendor=self.name, model="stub-model", answers={})
        if len(self.script) == 1:
            return self.script[0]
        return self.script.pop(0)


def leg_class(behaviour: LegBehaviour) -> type:
    """A cascade-leg class that answers from ``behaviour`` instead of the network."""

    class _StubLeg:
        def __init__(self, manager: Any, *, model: str = "", client: Any = None, **_: Any) -> None:
            self.name = behaviour.name
            self.model_id = model
            self.manager = manager
            self.client = client
            behaviour.clients.append(client)

        async def credential(self, manager: Any) -> SecretStr | None:
            behaviour.credential_calls += 1
            if behaviour.credential_raises:
                raise DecisionVendorError("credential resolution exploded", kind="transport")
            return SecretStr("stub-key") if behaviour.credential else None

        async def decide(self, request: DecisionRequest, *, timeout_s: float) -> DecisionResponse:
            behaviour.calls.append(request)
            if behaviour.delay_s:
                await asyncio.sleep(behaviour.delay_s)
            outcome = behaviour.decision()
            if isinstance(outcome, BaseException):
                raise outcome
            # The real legs report their own name in the response; the stub does
            # the same, so a test asserting `Recommendation.vendor` is testing the
            # service's plumbing rather than the fixture's default.
            return dataclasses.replace(outcome, vendor=behaviour.name)

    return _StubLeg
