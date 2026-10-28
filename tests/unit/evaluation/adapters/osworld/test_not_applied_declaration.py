"""The adapter declares a batch NOT APPLIED only when it has proved it.

WHY THIS FILE EXISTS. On the OSWorld 2.1 offline subset (arm main16) 10 of 20
episodes ended in "action bridge wedged by a transport error" on failures whose
own text read "guest action 1 was not attempted; no input was sent": the
provider refused before sending, but the failure crossed the worker as an
ambiguous mutation and the harness ended the episode. The harness now has a
declared type for that fact (``ActionNotAppliedError``); these tests pin that
the adapter raises it ONLY where non-application is proved -- never for an
ambiguous delivery, and never once any input of the batch went out, including
from an earlier provider call of the same batch (interleaved waits split one
batch into several provider calls).

They drive the REAL ``OSWorldV2Adapter.execute`` over a scripted provider; the
provider's own ``nothing_sent`` proof is pinned against real sockets in
``test_guest_loss.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from lop_osworld_v2_adapter import adapter as adapter_module
from lop_osworld_v2_adapter import guest_loss
from lop_osworld_v2_adapter.providers.aws import GuestExecutionError

from local_operator.evaluation.adapters.api import (
    ActionNotAppliedError,
    ObservationPhaseError,
)
from local_operator.evaluation.adapters.worker import _error_detail
from local_operator.evaluation.protocol import TypeAction, WaitAction
from tests.unit.evaluation.adapters.osworld.test_action_wait_order import (
    _click,
    _observation,
    _params,
    _StubBuilder,
)

_UNATTEMPTED = guest_loss.describe_unattempted_action(0) + "; supervisor recovery budget exhausted"


class _ScriptedProvider:
    """Answers each ``execute`` call from a script: ``None`` succeeds, an
    exception raises. Records every call so a test can see what went out."""

    def __init__(self, script: list[BaseException | None]) -> None:
        self.script = script
        self.calls: list[list[str]] = []

    async def execute(self, statements: list[str], *, settle: bool = True) -> None:
        del settle
        self.calls.append(list(statements))
        outcome = self.script.pop(0) if self.script else None
        if outcome is not None:
            raise outcome

    async def observe(self) -> dict[str, Any]:
        return {"screenshot": b""}


def _adapter(monkeypatch: pytest.MonkeyPatch, provider: _ScriptedProvider) -> Any:
    inst = adapter_module.OSWorldV2Adapter.__new__(adapter_module.OSWorldV2Adapter)
    inst._provider = provider  # type: ignore[attr-defined]
    inst._observation_builder = _StubBuilder()  # type: ignore[attr-defined]
    inst._current_observation = _observation("obs-1", 1)  # type: ignore[attr-defined]
    inst._sequence = 1  # type: ignore[attr-defined]
    inst._action_settle_policy = "throughput"  # type: ignore[attr-defined]

    async def no_sleep(seconds: float) -> None:
        del seconds

    monkeypatch.setattr(adapter_module.asyncio, "sleep", no_sleep)
    return inst


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(GuestExecutionError(_UNATTEMPTED, nothing_sent=True), id="unattempted"),
        pytest.param(
            GuestExecutionError(
                guest_loss.describe_failed_action(
                    0, guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "refused")
                ),
                delivery=guest_loss.DELIVERY_NOT_SENT,
                nothing_sent=True,
            ),
            id="first-statement-not-sent",
        ),
    ],
)
async def test_a_proved_unsent_first_call_is_declared_not_applied(
    monkeypatch: pytest.MonkeyPatch, error: GuestExecutionError
) -> None:
    provider = _ScriptedProvider([error])
    inst = _adapter(monkeypatch, provider)

    with pytest.raises(ActionNotAppliedError) as raised:
        await inst.execute(_params(_click()))

    # The provider's own sentence crosses unchanged; the cause chain keeps it.
    assert str(raised.value) == str(error)
    assert raised.value.__cause__ is error
    # And the worker maps it to the declared phase.
    detail = _error_detail(raised.value, "execute", "op-1", None)
    assert detail is not None and detail.phase == "not_applied"
    # No observation was built and the sequence did not move.
    assert inst._sequence == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        pytest.param(
            GuestExecutionError(
                guest_loss.describe_failed_action(
                    0, guest_loss.TransportVerdict(guest_loss.DELIVERY_AMBIGUOUS)
                ),
                delivery=guest_loss.DELIVERY_AMBIGUOUS,
            ),
            id="ambiguous",
        ),
        pytest.param(
            GuestExecutionError(
                guest_loss.describe_failed_action(
                    1, guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "refused")
                ),
                delivery=guest_loss.DELIVERY_NOT_SENT,
            ),
            id="not-sent-after-statement-1-succeeded",
        ),
        pytest.param(
            GuestExecutionError("guest action 1 failed (exit 1); input may be partial"),
            id="non-zero-exit",
        ),
        pytest.param(RuntimeError("no input was sent"), id="foreign-type-same-words"),
    ],
)
async def test_anything_unproved_stays_ambiguous(
    monkeypatch: pytest.MonkeyPatch, error: BaseException
) -> None:
    provider = _ScriptedProvider([error])
    inst = _adapter(monkeypatch, provider)

    with pytest.raises(BaseException) as raised:
        await inst.execute(_params(_click()))

    assert raised.value is error, "the provider's failure must propagate untouched"
    assert not isinstance(raised.value, (ActionNotAppliedError, ObservationPhaseError))
    detail = _error_detail(raised.value, "execute", "op-1", None)
    assert detail is not None and detail.phase == "unknown"


@pytest.mark.asyncio
async def test_a_proved_unsent_LATER_call_of_a_split_batch_is_not_declared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The batch-level guard: ``[click, wait, type]`` is two provider calls.

    The click's call succeeded -- input reached the guest -- so even a later
    call whose OWN proof says it sent nothing leaves the BATCH partially
    applied. Declaring it not applied would tell the model to re-issue a click
    that already landed.
    """

    unsent = GuestExecutionError(_UNATTEMPTED, nothing_sent=True)
    provider = _ScriptedProvider([None, unsent])
    inst = _adapter(monkeypatch, provider)

    with pytest.raises(GuestExecutionError) as raised:
        await inst.execute(
            _params(
                _click(),
                WaitAction(observation_id="obs-1", duration_ms=10),
                TypeAction(observation_id="obs-1", text="hello"),
            )
        )

    assert raised.value is unsent
    assert len(provider.calls) == 2 and provider.calls[0], "the click went out first"


@pytest.mark.asyncio
async def test_a_leading_wait_does_not_count_as_sent_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``[wait, click]``: the wait is a host-side sleep, so a proved-unsent
    click still leaves the environment untouched."""

    provider = _ScriptedProvider([GuestExecutionError(_UNATTEMPTED, nothing_sent=True)])
    inst = _adapter(monkeypatch, provider)

    with pytest.raises(ActionNotAppliedError):
        await inst.execute(_params(WaitAction(observation_id="obs-1", duration_ms=10), _click()))
