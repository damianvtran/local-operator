"""``/api/sessions/{id}/checkpoints`` — one conversation's checkpoint manifest.

WHAT THIS IS. The relay half of ``GET /v1/desktop/sessions/{session_id}/checkpoints``
(the desktop rail's manifest, design D9): one entry per user turn and one per
completed agent turn, each with its outcome and, on completions, its naming
state. The derivation is ``local_operator/session/transcript_index.py`` — the
same function the desktop route calls — and this module serves it through the
desktop's OWN wire model (``server.models.desktop_sessions.CheckpointManifest``),
so the two surfaces cannot drift: the payload is validated as the very model
the desktop route declares as its ``response_model``, and the dump emits every
declared field.

WHY THE PHONE CANNOT DERIVE THIS FROM ITS TRANSCRIPT. The phone's projection is
a bounded tail WINDOW — older rows only arrive through a history page — so a
rail built from the frames a phone happens to hold would silently mark only the
tail of the conversation, which is worse than no rail. The manifest is derived
from the journal and covers EVERY turn, whether or not any frame ever loaded it.

THE EMPTY ANSWER AND THE FAILED ANSWER ARE DIFFERENT CLAIMS, and ``index.state``
is the manifest's own honest vocabulary — this module adds no flattening of its
own. ``ready`` with no checkpoints is a conversation with nothing written yet;
``building`` means a scan is in flight and the entries served are the previous
scan's (the rail paints them and polls — the desktop renderer's own discipline);
``error`` is a refresh that failed — a journal that fails to read after a
successful stat (the case the mobile suite pins: a ``chmod 000`` file) must
never render as "no checkpoints". The bound is exactly that, and the residual
is known: a journal that cannot be ``stat``'ed at all still maps to
``missing`` → ``ready`` + ``[]`` in the shared derivation (``probe_index``
catches any ``OSError``) — pre-existing, deferred: the fix changes the desktop
rail's and ``find``'s semantics and ships separately (recorded on PR #2068).
``unsupported`` cannot arise here: a conversation this daemon cannot see
locally is refused 404 before the derivation runs.

READ-ONLY, deliberately. Nothing here writes or dials a session, and the naming
warm (``sessions.checkpoints.warm``) is a desktop-plane spend this route
deliberately does not serve.

THREADING. The blocking steps (the probes, the scan) already run on worker
threads inside ``transcript_index``; this module's own work is the call plus a
model round-trip. It must be awaited ON the caller's loop — ``checkpoints_view``
keeps loop-bound in-flight and resident state — so never hand it to
``asyncio.to_thread``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from local_operator.server.models.desktop_sessions import CheckpointManifest


async def manifest_payload(config_dir: Path | str, session_id: str) -> dict[str, Any]:
    """One conversation's manifest, in the desktop's own wire shape (D9).

    Construction is the schedules relay's drift guard, applied to this row set:
    the answer is built as the very model the desktop route serves, so a change
    required to construct it fails loudly. The models ignore extras, though
    (construction alone cannot catch a defaulted addition or a rename — that
    lands as a default), so the quiet path is pinned instead of assumed:
    ``tests/unit/mobile/test_checkpoints_relay.py`` asserts the declared field
    sets, and a shared-wire change reds a test before a phone can read a
    default the desktop never serves.
    """
    from local_operator.session import transcript_index

    view = await transcript_index.checkpoints_view(config_dir, session_id)
    # JSON mode: match the desktop serializer for non-JSON-native fields.
    return CheckpointManifest.model_validate(view).model_dump(mode="json")
