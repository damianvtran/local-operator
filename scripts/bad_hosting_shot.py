"""Capture the TUI frame a user gets when config names an unknown provider.

Drives the REAL ``OperatorApp`` (so ``local_operator.tcss`` is applied — the
lightweight test hosts declare no ``CSS_PATH`` and would show none of the
styling) with a session factory that fails exactly the way a corrupted
``hosting:`` value fails in production: by calling the real
``resolve_hosting_model`` against a config carrying the bad value, and letting
whatever it raises escape into the app's boot-failure handler.

Resolving for real rather than raising a hand-written error is the point. A
synthetic ``ValueError`` would prove only that the handler paints what it is
given; routing through the resolver is what makes the before/after pair
evidence about the DEFECT (which error class the boot path produces for a bad
hosting) instead of evidence about the painter.

The model name is deliberately a REAL one: the operator's corrupted config
paired ``hosting: anthropicxyq`` with a valid ``model_name``, so leaving it
empty would trip the unrelated "no default model for this hosting" branch and
capture the wrong failure.

Usage:
    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/bad_hosting_shot.py out.svg [hosting] [WIDTHxHEIGHT]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from local_operator.tui.app import OperatorApp  # noqa: E402

if TYPE_CHECKING:
    from local_operator.config import ConfigManager
from tests.unit.tui.test_app_pilot import FakeProviderController  # noqa: E402


class _Config:
    """Minimal ConfigManager stand-in holding the corrupted hosting value."""

    def __init__(self, hosting: str) -> None:
        self._values = {"hosting": hosting, "model_name": "claude-sonnet-4-5"}

    def get_config_value(self, key: str, default=None):
        return self._values.get(key, default)


async def main() -> None:
    out = sys.argv[1]
    hosting = sys.argv[2] if len(sys.argv) > 2 else "anthropicxyq"
    size = sys.argv[3] if len(sys.argv) > 3 else "100x30"
    width, height = (int(part) for part in size.split("x"))

    async def _factory():
        from local_operator.model.configure import configure_model
        from local_operator.session_factory import resolve_hosting_model

        # The production boot chain, in order: the preflight resolver, then the
        # model configuration that `create_session` reaches next. Before the
        # fix the resolver passes the garbage through and `configure_model`
        # raises the bare "Unsupported hosting platform" ValueError; after it,
        # the resolver classifies it first. Keeping BOTH calls here is what
        # lets one script capture both frames, and what makes the pair show
        # WHICH layer answered rather than just which text was painted.
        # Cast: the resolver only ever calls `get_config_value` on this, and a
        # real ConfigManager would write a config file the capture does not want.
        resolved, model = resolve_hosting_model(
            None,
            argparse.Namespace(hosting=None, model=None),
            cast("ConfigManager", _Config(hosting)),
        )
        configure_model(hosting=resolved, model_name=model, credential_manager=None)
        raise AssertionError("boot chain accepted a bad hosting value")

    app = OperatorApp(_factory, provider_controller=FakeProviderController())
    async with app.run_test(size=(width, height)) as pilot:
        # Poll until the boot task has actually settled: the band starts at
        # "connecting…" and a fixed sleep captures that transient rather than
        # the failure state under test.
        for _ in range(60):
            await pilot.pause()
            await asyncio.sleep(0.05)
            label = app._status._model_label if app._status is not None else ""
            if label != "connecting…" or app._setup_state:
                break
        await pilot.pause()
        app.save_screenshot(out)
        label = app._status._model_label if app._status is not None else "?"
        print(f"saved={out} setup_state={app._setup_state} band={label!r}")
        print(f"splash_notice={app._splash_notice!r}")


asyncio.run(main())
