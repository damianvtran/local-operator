from __future__ import annotations

import asyncio
from dataclasses import dataclass
from types import SimpleNamespace

from local_operator.saver import command


@dataclass
class Result:
    kind: str
    text: str
    style: str


class Journal:
    def __init__(self) -> None:
        self.rows = []

    async def append_custom(self, kind, details):
        self.rows.append({"custom_type": kind, "details": details})

    def latest_custom(self, kind):
        return next((row for row in reversed(self.rows) if row["custom_type"] == kind), None)


async def main() -> None:
    session = SimpleNamespace(
        routing_settings={"radient": {"saver": {"mode": "opt_in"}}},
        radient_saver_choice=None,
        _transcript=Journal(),
        refresh_frontend_state=lambda: None,
    )
    command.signed_in = lambda: True
    # `saver_slash` is a coroutine since review M2 (the answer is published only
    # after the journal write lands), so every call is awaited; the bare calls
    # this script started with printed a coroutine's missing `.text` and died.
    print("$ /saver")
    print((await command.saver_slash(session, "", Result)).text)
    await asyncio.sleep(0)
    print("$ /saver on")
    print((await command.saver_slash(session, "on", Result)).text)
    await asyncio.sleep(0)
    print("$ /saver status")
    print((await command.saver_slash(session, "status", Result)).text)
    print("$ /radient saver off")
    print((await command.saver_slash(session, "off", Result)).text)
    await asyncio.sleep(0)
    command.signed_in = lambda: False
    print("$ /saver  # signed out")
    print((await command.saver_slash(session, "", Result)).text)
    print("journal rows:", len(session._transcript.rows))


asyncio.run(main())
