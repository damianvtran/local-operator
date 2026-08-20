"""Capture the composer band while a provider fallback serves requests.

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/fallback_shot.py out.svg after
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python scripts/fallback_shot.py out.svg before

Drives the REAL ``OperatorApp`` (the one that loads ``local_operator.tcss``)
with a session whose effective-model surface follows the route edge, posts the
fallback notice and the ``EffectiveModelChanged`` message the way the live
event bridge does, and saves the settled frame.

``before`` skips the ``EffectiveModelChanged`` post, which reproduces the
pre-fix behaviour exactly — the notice printed, the band never moved — so the
pair shows the delta this feature exists to close: a band still naming the
selected model while every request goes to the fallback.
"""

import asyncio
import sys
from types import SimpleNamespace

sys.path.insert(0, ".")

from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.events import EffectiveModelChanged, NoticePosted  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402


class _FallbackSession(FakeSession):
    """A session mid-fallback: selection Kimi K3, requests served by GLM 5.3."""

    def __init__(self) -> None:
        super().__init__()
        # Declared on the class: the "after" frame assigns it, and pyright
        # rejects attribute-creation off an assignment.
        self._eff: object = None

    @property
    def model_label(self) -> str:
        return "kimi/kimi-k3"

    @property
    def model(self):
        return SimpleNamespace(
            provider="kimi",
            model_id="kimi-k3",
            display_name="Kimi K3",
            context_window=262_144,
            reasoning_effort=None,
            reasoning_efforts=(),
            reasoning=False,
        )

    @property
    def effective_model(self):
        return getattr(self, "_eff", None) or self.model

    @property
    def effective_model_label(self):
        spec = self.effective_model
        return f"{spec.provider}/{spec.model_id}"


async def main() -> None:
    out, mode = sys.argv[1], sys.argv[2]
    session = _FallbackSession()
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(100, 24)) as pilot:
        await pilot.pause()
        # The user's turn, then the failure narration the stream fn emits.
        from local_operator.tui.widgets.transcript import UserBlock

        app._append_block(UserBlock("Continue"))
        app.post_message(NoticePosted("provider failure — falling back to zai/glm-5.3", "warning"))
        await pilot.pause()
        if mode == "after":
            session._eff = SimpleNamespace(
                provider="zai",
                model_id="glm-5.3",
                display_name="GLM 5.3",
                context_window=1_000_000,
                reasoning_effort=None,
                reasoning_efforts=(),
                reasoning=False,
            )
            app.post_message(
                EffectiveModelChanged("zai", "glm-5.3", None, "provider failure", True)
            )
        await pilot.pause()
        await pilot.pause()
        app.save_screenshot(out)


asyncio.run(main())
