"""Capture the mesh credential-binding ACCOUNT-CHANGE notices, at any width.

Usage::

    env -u NO_COLOR TERM=xterm-256color .venv/bin/python \
        scripts/mesh_binding_notice_shot.py out.svg [WIDTHxHEIGHT] [STATE]

States:

``sibling``   M3 — the owner's sibling pick: §4.9's own sentence.
``capture``   M4 — a device using a loaned login gained its own; the "your
              login on this device" shape.
``move``      M6 — the serving account moved between devices.
``stack``     all three in one conversation: the frame where the notice rows
              must read as notices between real turns.
``off``       the same conversation with NO notice — the before frame.

WHY A RENDER RATHER THAN THE UNIT ASSERTIONS: the notice is a new operator-facing
transcript row, and the design round owns its surface and copy. The frames here
are exactly what the fold paints — the sentence rendered by
``network/credentials/messages.py`` (called for real below, never copied) inside
the standard ``NoticeBlock`` the MCP/credential notices already use.

Isolates HOME and the config dir before importing the app (``isolate_capture``)
so the frame never depends on the developer's own session state, and so a
capture cannot touch a live session.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.visual_capture import isolate_capture, save_capture  # noqa: E402

isolate_capture()  # BEFORE the app imports: it reads HOME/config at import time.

from local_operator.harness.message_types import (  # noqa: E402
    SESSION_BINDING_NOTICE_MESSAGE_TYPE,
)
from local_operator.harness.types import CustomMessage, Message  # noqa: E402
from local_operator.network.credentials.messages import (  # noqa: E402
    render_binding_change_notice,
)
from local_operator.session.credential_binding import CredentialBinding  # noqa: E402

#: This device's id in the fixture rows (the "your login on this device" arm
#: needs the recorder's own device, which the row never carries).
SELF = "d_00000000000000000000000000000011"
OWNER = "d_00000000000000000000000000000022"
OTHER = "d_00000000000000000000000000000033"


def _binding(**overrides: object) -> CredentialBinding:
    fields: dict[str, object] = {
        "provider": "openai",
        "owner_device": OWNER,
        "owner_device_name": "damian-mbp",
        "credential_id": 42,
        "identity_label": "you@example.com",
        "writer": "1000:1000",
    }
    fields.update(overrides)
    return CredentialBinding(**fields)  # type: ignore[arg-type]


#: The production sentences, produced by the production renderer.
SENTENCES = {
    "sibling": render_binding_change_notice(
        _binding(credential_id=43), _binding(), self_device=SELF
    ),
    "capture": render_binding_change_notice(
        _binding(
            owner_device=SELF, owner_device_name="my-laptop", credential_id=7, identity_label=""
        ),
        _binding(),
        self_device=SELF,
    ),
    "move": render_binding_change_notice(
        _binding(owner_device=OTHER, owner_device_name="box-2", credential_id=9, identity_label=""),
        _binding(),
        self_device=SELF,
    ),
}


def _notice(text: str) -> CustomMessage:
    """The row exactly as the Session journals it (details carry only the text here)."""
    return CustomMessage(
        custom_type=SESSION_BINDING_NOTICE_MESSAGE_TYPE,
        attribution="system",
        details={"text": text},
    )


def _conversation(state: str) -> list[Message | CustomMessage]:
    turns: list[Message | CustomMessage] = [
        Message.user("run the nightly audit against the mini's warehouse copy"),
        Message.assistant("Kicking that off — the warehouse read goes out over the mesh."),
    ]
    if state == "off":
        return turns
    if state == "stack":
        turns += [
            _notice(SENTENCES["sibling"]),
            Message.assistant("The audit finished; 12 rows flagged for review."),
            _notice(SENTENCES["capture"]),
            Message.user("also re-run the export with the fixed delimiter"),
            _notice(SENTENCES["move"]),
        ]
    else:
        turns += [
            _notice(SENTENCES[state]),
            Message.user("anything I need to do about that?"),
        ]
    return turns


async def main() -> None:
    out = Path(sys.argv[1]).resolve()
    size = sys.argv[2] if len(sys.argv) > 2 else "100x30"
    state = sys.argv[3] if len(sys.argv) > 3 else "stack"
    width, height = (int(part) for part in size.split("x"))

    from local_operator.tui.app import OperatorApp
    from tests.unit.tui.test_app_pilot import FakeSession, _factory

    session = FakeSession()
    session._history = _conversation(state)
    app = OperatorApp(lambda: _factory(session))
    async with app.run_test(size=(width, height)) as pilot:
        # The app's own boot replays ``settled_rows()`` (FakeSession returns the
        # seeded history), so the fold under test is the product's, not a
        # hand-called one.
        await pilot.pause()
        save_capture(app, out)
    print(f"wrote {out} ({state}, {size})")


if __name__ == "__main__":
    asyncio.run(main())
