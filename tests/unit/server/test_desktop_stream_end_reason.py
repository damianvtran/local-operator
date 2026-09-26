"""The per-stream teardown reason must survive to a log on a machine nobody configures.

WHY THIS FILE EXISTS, and why the assertion is about a LEVEL rather than a string.
The desktop stream's churn has two candidate drivers, and on the operator's own
machine they are **indistinguishable**:

* the **relief valve** disconnecting a viewer (a subscriber that fell behind), and
* the **cancelled-teardown ordering** that used to destroy the facade before the
  reconnect dwell could arm.

Both rotate the epoch through the same ``_detach``, and the shipped client names the
SSE close cause nowhere -- no ``closeReason``, no log line, nothing persisted. So
neither leaves a client-side trace, and a claim about which one dominated cannot be
made from any log this machine already has.

The reason vocabulary was already written and could not be read. Measured on the
operator's daemon: ``backend-service.log`` holds **0 ``[INFO] local_operator`` lines
against 9,687 WARNING+ ones**, because the daemon's console logging runs at
``LOG_LEVEL``'s default WARNING. There is no second place to look -- the daemon's
stdout and stderr both land in that one file, and ``serve-1111.log`` does not exist.

``server/retire.py`` records this exact problem and solved it the same way: "WARNING,
not INFO, because the daemon's console logging runs at LOG_LEVEL's default WARNING --
a line nobody sees is not a notice". These cases hold that property for the reason
line, and they are deliberately written against a **WARNING-level capture**, because
that is the daemon nobody configured.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any

import pytest

from local_operator.server.utils.desktop_sessions import DesktopSessions

pytestmark = pytest.mark.asyncio

LOGGER = "local_operator.server.utils.desktop_sessions"


async def _one_ended_stream(
    tmp_path: Path, caplog: Any, *, overflow: bool = False, burst: int = 0
) -> list[logging.LogRecord]:
    """Drive one stream to its end and return the reason lines it produced.

    The capture is set to **WARNING**, never to INFO: a reason recorded below the
    default level is invisible on the operator's daemon, which is the whole problem
    this change answers. A test that captured at INFO would pass against the broken
    level and prove nothing.
    """
    pool = DesktopSessions(tmp_path)
    sid = await pool.create(str(tmp_path))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        async with pool.session(sid) as bridge:
            sub = bridge.subscribe()
            stream = bridge.events(sub, epoch=bridge.epoch, after_seq=0)
            await anext(stream)  # the open frame: the handshake has begun
            for n in range(burst):
                bridge.publish("event", {"n": n})
            if overflow:
                # The relief valve, exactly as the bridge does it: it revokes its
                # own subscriber, which is what makes the reason `subscriber
                # overflow` rather than `client disconnect`.
                bridge._disconnect(sub)
            await stream.aclose()
        await asyncio.sleep(0)

    return [record for record in caplog.records if "desktop stream ended" in record.getMessage()]


async def test_an_ordinary_disconnect_reaches_a_default_configured_log(tmp_path, caplog):
    """The property the instrumentation exists for: WARNING, on the default level.

    An ordinary client disconnect is the shape the churn is made of, so it is the
    one that has to be visible. It was logged at INFO and reached nothing.
    """
    ended = await _one_ended_stream(tmp_path, caplog)

    assert ended, (
        "the reason line must reach a WARNING-level capture; at INFO it is dropped "
        "before the handler on the operator's daemon"
    )
    assert (
        ended[-1].levelno == logging.WARNING
    ), "and at WARNING, not INFO: the level is what decides whether anyone sees it"
    assert (
        "client disconnect" in ended[-1].getMessage()
    ), "with the vocabulary unchanged -- the words are what an operator greps for"


async def test_the_relief_valve_is_named_at_warning_too(tmp_path, caplog):
    """The candidate the line exists to separate, so it cannot stay invisible.

    `subscriber overflow` is one of the two possible drivers of the churn. A
    promotion that left it below the default level would leave the question open
    exactly where it matters, which is why this is its own case and not a parameter
    of the one above.
    """
    ended = await _one_ended_stream(tmp_path, caplog, overflow=True)

    assert ended, "the relief valve's own end must be logged"
    assert ended[-1].levelno == logging.WARNING
    assert "subscriber overflow" in ended[-1].getMessage(), (
        "and named as an overflow rather than as an ordinary disconnect: telling "
        "those two apart is the only reason this line is being promoted"
    )


async def test_the_reason_is_one_line_per_stream_end(tmp_path, caplog):
    """Bounded: the rate is the churn's rate, never the frame rate.

    The bridge publishes a burst before the stream ends, so a promotion that leaked
    into the per-frame path would show here as dozens of lines for one stream.
    """
    ended = await _one_ended_stream(tmp_path, caplog, burst=50)

    assert len(ended) == 1, (
        f"one line per stream END, not per frame; {len(ended)} lines for a single "
        "stream would make this instrumentation its own flood"
    )


async def test_the_capture_level_is_what_makes_it_readable(caplog):
    """The counterfactual, so the promotion cannot be reverted quietly.

    This is the operator's daemon in miniature: a WARNING-level capture keeps the
    promoted line and drops the INFO one. If a captured level below WARNING ever
    became the default, this case would be the place that says so.
    """
    logger = logging.getLogger(LOGGER)
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        logger.info("desktop stream ended for x (sub=deadbeef): client disconnect")
        logger.warning("desktop stream ended for x (sub=deadbeef): client disconnect")

    seen = [r for r in caplog.records if "desktop stream ended" in r.getMessage()]
    assert len(seen) == 1, (
        "a default-level capture keeps the WARNING and drops the INFO, which is the "
        "measurement this change responds to"
    )
    assert seen[0].levelno == logging.WARNING
