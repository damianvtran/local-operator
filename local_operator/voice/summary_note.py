"""The turn note that asks the agent for a ``Spoken:`` paragraph.

While the voice slot is held, the provider request for a prompt that was
DICTATED carries one extra paragraph telling the agent the user is listening and
asking it to end its reply with a short ``Spoken:`` paragraph that ``speakable``
then selects (``docs/design/voice-spoken-summary.md``).

Model-facing prompt machinery, so it is English by design (AGENTS.md: model-
facing prompts are exempt from the keyed-copy rule), and ``Spoken:`` is a
protocol label that is never translated.

The note is NEVER stored. ``Session._with_voice_note`` appends it, as one
ephemeral message, to the request-local copy of the history before each
provider call, so no row, preview, replay or compaction ever contains it and
the cached prompt prefix is untouched.
"""

from __future__ import annotations

#: One paragraph, parenthesised like the other harness notes that ride a turn.
VOICE_SUMMARY_NOTE = (
    "(The user dictated this message by voice and is listening to your reply. "
    "End your reply with a final paragraph that starts with `Spoken:`. Write at "
    "most two plain sentences for the ear: no code, no URLs, no markdown, no "
    "list. Say the point of the reply, in the language the user is speaking. "
    "Keep the word `Spoken:` exactly as written. Do this even when the reply "
    "is short.)"
)
