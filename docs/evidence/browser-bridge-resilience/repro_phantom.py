"""RC3/RC4 + the repair verb: a driven tab, its close, and a phantom cleaned.

Shows the user's exact complaint and its fix without needing a real browser:
a driven tab is advertised, `tab_closed` clears exactly that tab, and a
phantom left behind by a worker that died without announcing anything is
reconciled by `lop browser status --repair`.

    .venv/bin/python docs/evidence/browser-bridge-resilience/repro_phantom.py
"""

from __future__ import annotations

import asyncio
import json
import socket
import sys
import tempfile
import urllib.request
from pathlib import Path

# See repro_rc1.py: sys.path[0] is the SCRIPT's directory, so pin this repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from local_operator.browser_bridge import install as bridge_install  # noqa: E402
from local_operator.browser_bridge.daemon import create_app  # noqa: E402


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def main() -> int:
    import uvicorn

    root = Path(tempfile.mkdtemp(prefix="lop-phantom-"))
    port = free_port()
    app = create_app(port, root)
    service = app.state.bridge
    # An attached browser, deliberately unpaired (see repro_rc1.py: faking
    # paired without a pairing file makes the revocation watcher sever it).
    service.link.websocket = object()  # type: ignore[assignment]

    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    serving = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)

    def _health_sync() -> dict:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as response:
            return json.loads(response.read().decode())

    async def health() -> dict:
        # Off-thread: a blocking urlopen on the serving loop deadlocks.
        return await asyncio.to_thread(_health_sync)

    phantom = "http://127.0.0.1:8974/r3-notice-amber.svg"
    failures = 0

    service.link.note_driven("bridge:9:zz", phantom, "amber")
    state = await health()
    tabs = len(state["driven_tabs"])
    print(f"[1 driving]   current_url={state['current_url']}  driven_tabs={tabs}")
    failures += 0 if len(state["driven_tabs"]) == 1 else 1

    # A second session's tab: closing one must not blank the other (RC4).
    service.link.note_driven("bridge:10:yy", "https://example.com", "Example")
    service.link.note_closed("bridge:10:yy")
    state = await health()
    print(
        f"[2 one close] current_url={state['current_url']}  driven_tabs={len(state['driven_tabs'])}"
        "   <- the other session's tab survived"
    )
    failures += 0 if len(state["driven_tabs"]) == 1 else 1

    # RC3: the tab_closed the extension now sends clears the driven record.
    service.link.note_closed("bridge:9:zz")
    state = await health()
    tabs = len(state["driven_tabs"])
    print(f"[3 closed]    current_url={state['current_url']!r}  driven_tabs={tabs}")
    failures += 0 if state["current_url"] == "" else 1

    # The leak case: a worker that died without announcing anything, so the
    # daemon still advertises a tab whose browser is gone. This is the state
    # the user could not clear.
    service.link.note_driven("bridge:9:zz", phantom, "amber")
    service.link.websocket = None
    state = await health()
    print(f"[4 phantom]   current_url={state['current_url']}   <- advertised, browser gone")

    result = await asyncio.to_thread(bridge_install.repair, port, root)
    for line in result["steps"]:
        print(f"   repair: {line}")
    state = await health()
    print(f"[5 repaired]  current_url={state['current_url']!r}  ok={result['ok']}")
    failures += 0 if (state["current_url"] == "" and result["ok"]) else 1

    server.should_exit = True
    await serving

    if failures:
        print(f"\nRESULT: FAILED ({failures} check(s))")
        return 1
    print(
        "\nRESULT: PASS — a closed tab clears exactly its own driven record, and a phantom "
        "left by a dead worker is reconciled by --repair."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
