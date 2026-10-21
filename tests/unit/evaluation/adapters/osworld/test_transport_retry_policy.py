"""A guest request is retried only when the classifier PROVED it was never sent.

The policy is one function in the shared provider seam (``providers.base``),
deliberately not in the benchmark adapter and deliberately not a boolean flag a
caller can set: it takes the ``TransportVerdict`` produced by
``guest_loss.classify_transport_failure`` and schedules a retry only for
``DELIVERY_NOT_SENT`` -- positive evidence the TCP connection was never
established, so no byte of the request reached the guest. Everything that means
"sent" or "unknown" is refused here, which is what keeps a lost response from
being replayed onto a guest that may already have applied it.

The bound and the backoff are properties of this function, not of any call site,
so they are pinned here rather than inferred from a provider test.
"""

from __future__ import annotations

from lop_osworld_v2_adapter import guest_loss
from lop_osworld_v2_adapter.providers.base import (
    TRANSPORT_RETRY_ATTEMPTS,
    transport_retry_delay,
)

_NOT_SENT = guest_loss.TransportVerdict(guest_loss.DELIVERY_NOT_SENT, "connect_failed")


def test_only_a_provably_unsent_verdict_may_be_retried() -> None:
    """Every value that means sent-or-unknown is refused, whatever the cause."""

    for delivery in (guest_loss.DELIVERY_AMBIGUOUS, "sent", "unknown", ""):
        verdict = guest_loss.TransportVerdict(delivery)
        assert transport_retry_delay(verdict, failed_attempt=1, jitter=lambda: 0.5) is None


def test_a_not_sent_verdict_retries_to_the_bound_and_then_stops() -> None:
    """Jitter pinned at 0.5 means "no adjustment", so the schedule is exact."""

    delays = [
        transport_retry_delay(_NOT_SENT, failed_attempt=used, jitter=lambda: 0.5)
        for used in range(1, TRANSPORT_RETRY_ATTEMPTS + 1)
    ]
    assert TRANSPORT_RETRY_ATTEMPTS == 4
    assert delays == [0.5, 1.0, 2.0, None]
    assert delays[-1] is None, "the last permitted attempt must not schedule another"


def test_the_whole_intervention_is_bounded_to_a_few_seconds() -> None:
    """The worst-case spend, at maximum jitter, over the whole schedule."""

    total = 0.0
    for used in range(1, TRANSPORT_RETRY_ATTEMPTS):
        delay = transport_retry_delay(_NOT_SENT, failed_attempt=used, jitter=lambda: 1.0)
        assert delay is not None
        total += delay
    assert total <= 4.5


def test_a_hostile_jitter_generator_cannot_escape_the_bound() -> None:
    """A caller-supplied generator is clamped: this function owns the bound."""

    for sample in (-5.0, 5.0):
        delay = transport_retry_delay(_NOT_SENT, failed_attempt=1, jitter=lambda: sample)
        assert delay is not None
        assert 0.375 <= delay <= 0.625


def test_a_non_finite_jitter_sample_falls_back_to_no_adjustment() -> None:
    """A NaN is not a delay: it must not propagate into ``time.sleep``."""

    for sample in (float("nan"), float("inf")):
        delay = transport_retry_delay(_NOT_SENT, failed_attempt=1, jitter=lambda: sample)
        assert delay == 0.5


def test_the_attempt_ordinal_is_clamped_to_one() -> None:
    """A zero or negative ordinal must not double the delay backwards."""

    assert transport_retry_delay(_NOT_SENT, failed_attempt=0, jitter=lambda: 0.5) == 0.5
    assert transport_retry_delay(_NOT_SENT, failed_attempt=-3, jitter=lambda: 0.5) == 0.5
