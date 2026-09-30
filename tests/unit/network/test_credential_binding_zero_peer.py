"""U8: the 0-peer guard — a root with no placement document records nothing.

The one cell of the binding slice that must stay GREEN on the pre-slice base
(its siblings are red by construction, the module absent). It pins pre-existing
behaviour that must not regress: without a credential placement document,
``build_auth_store`` hands back the plain ``AuthStore``, no recorder exists to
create, and the session's transcript is byte-identical. The module import is
guarded precisely so the cell reads as a pass on both sides of the slice —
before it, there is nothing that could create a recorder; after it, the factory
must still decline.
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

from local_operator.harness.types import Message
from local_operator.network.credentials import store as mesh_store
from local_operator.session.transcript import Transcript

BARE_DEVICE = "d_00000000000000000000000000000001"


def _recorder_for(transcript: Transcript, root: Path) -> Any:
    """The recorder a session on ``root`` would get, or ``None``.

    ``None`` when the module is absent is not a stub: on the pre-slice base no
    code path could produce a recorder at all, which is the state this guard
    asserts must persist for a root that cannot record.
    """
    if importlib.util.find_spec("local_operator.session.credential_binding") is None:
        return None
    from local_operator.session.credential_binding import recorder_for_session

    return recorder_for_session(transcript, config_dir=root, session_id="sess-zero-peer")


def test_u8_no_placement_document_means_no_recorder_and_no_writes(tmp_path: Path) -> None:
    """U8: the 0-peer path is byte-identical — plain store, no recorder, no rows."""
    from local_operator.providers.auth_store import AuthStore

    root = tmp_path / "bare"
    root.mkdir()
    store = mesh_store.build_auth_store(root)
    assert type(store) is AuthStore
    store.close()

    transcript = Transcript(root / "sess")

    async def seed() -> bytes:
        await transcript.append_message(Message.user("hello"))
        return transcript.path.read_bytes()

    before = asyncio.run(seed())
    assert before
    assert _recorder_for(transcript, root) is None
    assert transcript.path.read_bytes() == before

    # A document with no identity is the same answer: a local row names THIS
    # device, and there is no device to name.
    from local_operator.network.credentials import placement as placement_mod

    document = placement_mod.PlacementDocument("n_zero_peer", root=root, written_by=BARE_DEVICE)
    document.declare(
        "openai",
        owner_device=BARE_DEVICE,
        owner_device_name="solo",
        provider="openai",
        by=BARE_DEVICE,
    )
    document.save()
    assert _recorder_for(transcript, root) is None
    assert transcript.path.read_bytes() == before
