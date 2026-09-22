"""Capture the /model picker with the query ``auto`` typed, for visual validation.

Run from a worktree root:

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/model_picker_auto_shot.py OUT.svg [COLSxROWS]

Seeds an isolated HOME (``probe_isolation``) with a copy of the cached Radient
and OpenRouter listings, signs in a Radient OAuth account and an OpenRouter key
so the picker offers the aggregator rows connected, then boots the REAL
``OperatorApp`` and drives the picker to the state a user reports: ``/model``
open with ``auto`` typed.

The rows are ranked by the production ``rank_rows`` over the production
``initial_catalogue``, so the frame is the product's order rather than a
fixture's. It is the capture behind the "Radient Auto comes first" change.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Resolve the operator's real cache BEFORE ``probe_isolation`` rewrites HOME to
# its sandbox, or the seed below would copy the sandbox's own (empty) cache.
_REAL_CACHE = Path(os.environ.get("HOME", "~")).expanduser() / ".local-operator" / "cache"

import scripts.probe_isolation  # noqa: E402,F401  (isolates HOME/config on import)
from local_operator.credentials import CredentialManager  # noqa: E402
from local_operator.paths import config_dir  # noqa: E402
from local_operator.providers.auth_store import AuthStore, default_db_path  # noqa: E402
from local_operator.providers.controller import ProviderController  # noqa: E402
from local_operator.tui.app import OperatorApp  # noqa: E402
from local_operator.tui.widgets.editor import Editor  # noqa: E402
from scripts.visual_capture import save_capture  # noqa: E402
from tests.unit.tui.test_app_pilot import FakeSession, _factory  # noqa: E402

#: Providers whose cached listings make the frame show the interesting case: two
#: aggregators holding rows whose query scores are close enough to expose the
#: tie-break.
SEEDED_LISTINGS = ("radient.listing.json", "openrouter.listing.json")


def seed() -> ProviderController:
    """A controller whose stores are populated the way a signed-in install is.

    The credential rows are written through ``upsert_credential`` with the
    STORAGE id (``radient`` for both the OAuth and the pasted-key form), which is
    what a real login or ``lop credential update`` writes — see ``AuthStore``.
    """
    cache = Path("~/.local-operator/cache").expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    for name in SEEDED_LISTINGS:
        source = _REAL_CACHE / name
        if source.exists():
            shutil.copy2(source, cache / name)
    store = AuthStore(default_db_path())
    store.upsert_credential(
        "radient",
        {"type": "oauth", "access": "capture", "refresh": "capture", "expires": 4102444800000},
    )
    store.upsert_credential("openrouter", {"type": "api_key", "key": "capture"})
    return ProviderController(store, CredentialManager(config_dir()))


async def main() -> None:
    out = sys.argv[1]
    size = (100, 30)
    if len(sys.argv) > 2:
        cols, rows = sys.argv[2].split("x")
        size = (int(cols), int(rows))

    controller = seed()
    app = OperatorApp(lambda: _factory(FakeSession()), provider_controller=controller)
    async with app.run_test(size=size) as pilot:
        for _ in range(60):
            await pilot.pause()
            if app._session is not None:
                break
        # The buffer is the single authority on which picker shows, so opening
        # through the app (and typing the query on the widget) is the real route
        # rather than poking a widget the editor would close again on the next
        # key.
        app._open_model_picker()
        for _ in range(20):
            await pilot.pause()
        picker = app.query_one(Editor).model_picker
        picker.set_query("auto")
        for _ in range(30):
            await pilot.pause()
        save_capture(app, out)
        print("query:", picker.query_text())
        print("matches:", [row.selector for row in picker.suggestions()][:8])

    controller.close()


asyncio.run(main())
