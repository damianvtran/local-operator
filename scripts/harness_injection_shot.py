"""Capture a resumed transcript that holds a leaked harness injection.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/harness_injection_shot.py out.svg

Drives the REAL ``OperatorApp`` (the one that loads ``local_operator.tcss``)
over a SEED OF THE REPORTED SHAPE — the four ``harness_injected`` rows a
compaction pass baked into session ``835fbcafdc27`` (each one the fallover
notice ``format_model_switch_message`` renders, rebuilt here through the real
producer) beside the assistant line that followed them — and then posts the
failover receipt the live path paints for that moment, so one frame answers
both halves of the report: the notice must not read as the user's own words, and
the moment is still announced.

This is a SHAPE, not the operator's transcript: the real session is replayed by
``scripts/legacy_notice_resume_shot.py`` against a copy of it, which is the
frame that shows the reported data. That script is also the one that covers the
legacy pre-stamp notices, which this seed does not carry.

TWO THINGS THE RECEIPT IS, and the caption on the PR must say both: it is the
``NoticeEvent`` ``configure.py::_on_route_change`` emits ("<reason> — falling
back to <selector>", plus ``_on_route_settle``'s "back to <primary>" on
recovery), and it is the account of the event only WHILE THE SESSION IS LIVE —
reopening the conversation carries no trace of it by design, which is why the
harness also tells the MODEL through the injected notice this fix stops from
surfacing as user text.

Run it against a pre-fix checkout for the before-frame (the leaked rows paint as
user bubbles there, one per notice, which is the screenshot the operator sent).

    git worktree add --detach /tmp/lo-before HEAD
    ln -s ~/local-operator/.venv /tmp/lo-before/.venv   # throwaway only
    cd /tmp/lo-before && env -u NO_COLOR TERM=xterm-256color \
        .venv/bin/python scripts/harness_injection_shot.py /tmp/before.svg
"""

import asyncio
import sys
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, ".")

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()

from local_operator.compaction.cutpoint import RENDERED_INJECTION_KEY  # noqa: E402
from local_operator.harness.types import Message, TextContent  # noqa: E402
from local_operator.incidents import format_model_switch_message  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import EffectiveModelChanged, NoticePosted  # noqa: E402
from local_operator.tui.widgets.transcript import (  # noqa: E402
    TranscriptView,
    UserBlock,
)
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: The four fallover hops the transcript carries, in order, rebuilt through the
#: producer that wrote them (``journal_model_switch(transient=True)``).
HOPS = (
    ("zai/glm-5.3", "anthropic quota exhausted (0% remaining)"),
    ("kimi/k3", "provider failure"),
    ("alibaba-token-plan/qwen3.8-max", "provider failure"),
    ("xai/grok-4.6", "provider failure"),
)


def _leaked_hops() -> list[Message]:
    """The rows the compaction rebuild baked in: plain user Messages, stamped."""
    return [
        Message(
            role="user",
            content=[
                TextContent(
                    text=format_model_switch_message(
                        new_label, "anthropic/claude-opus-5", reason=reason, transient=True
                    )
                )
            ],
            provider_payload={RENDERED_INJECTION_KEY: True},
        )
        for new_label, reason in HOPS
    ]


class _FallbackSession(FakeSession):
    """A session serving on a fallback, so the band can name the detour."""

    @property
    def model_label(self) -> str:
        return "anthropic/claude-opus-5"

    @property
    def model(self):
        return SimpleNamespace(
            provider="anthropic",
            model_id="claude-opus-5",
            display_name="Claude Opus 5",
            context_window=200_000,
            reasoning_effort=None,
            reasoning_efforts=(),
            reasoning=False,
        )

    @property
    def effective_model(self):
        return SimpleNamespace(
            provider="xai",
            model_id="grok-4.6",
            display_name="Grok 4.6",
            context_window=256_000,
            reasoning_effort=None,
            reasoning_efforts=(),
            reasoning=False,
        )

    @property
    def effective_model_label(self):
        return "xai/grok-4.6"


async def _pump_until(pilot: Any, predicate: Any, attempts: int = 300) -> None:
    """Pump frames until ``predicate`` holds, instead of guessing a count.

    The boot's resume replay is a worker, so the number of frames it needs is a
    property of the machine. A fixed count produced a frame that painted the
    history TWICE (the replay landing after an explicit projection) on one run
    and once on another — the same class of bet the tests in
    ``tests/unit/tui`` were converted away from.
    """
    for _ in range(attempts):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("the resumed replay never painted")


async def main() -> None:
    out = sys.argv[1]
    session = _FallbackSession()
    session._history = [
        Message.user("check the MiniMax subset state before the next batch"),
        *_leaked_hops(),
        Message.assistant("Checking the MiniMax subset state after the model switch."),
    ]
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(120, 46)) as pilot:
        await pilot.pause()
        # The replay the operator was looking at when he reported this: the
        # app's OWN resume path paints it, so the frame is the surface he saw
        # rather than a second projection the harness added.
        transcript = app.query_one(TranscriptView)
        await _pump_until(pilot, lambda: any(isinstance(b, UserBlock) for b in transcript.blocks()))
        # …and the receipt the LIVE path paints for the failover: the retry
        # notice, then the band's own effective-model edge.
        # The live failover receipt, exactly as ``configure.py::_on_route_change``
        # emits it (a ``NoticeEvent`` carrying "<reason> — falling back to
        # <selector>"). Nothing in this tree raises a retry notice for a
        # model fallback, which is why this is the NoticePosted shape.
        app.post_message(
            NoticePosted(
                "anthropic quota exhausted (0% remaining) — falling back to zai/glm-5.3",
                "warning",
            )
        )
        app.post_message(EffectiveModelChanged("xai", "grok-4.6", None, "provider failure", True))
        for _ in range(4):
            await pilot.pause()
        save_capture(app, out)


asyncio.run(main())
