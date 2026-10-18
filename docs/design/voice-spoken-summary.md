# Design: spoken summary for voice — option (i), agent-written

Status: **Proposed, exploratory.** Nothing here is approved for building. Source:
§11 of the architect's plan "Qwen voice without the browser tab" (2026-10-18),
which recommends option (i) over a small rewriting model (ii) and over a
frontend LLM paraphrase (iii). Base: `feat/qwen-voice` @ `fcdaf4555`; line
numbers below are against that base, so they drift as the page is edited.

The operator asked for this as a design note only. It gates no PR and changes
no contract.

## Problem

`speakable` speaks the first words of whatever the agent wrote. For a long
report that is a preamble — "Here is what I found" — and the useful sentence is
past the cap. The listener hears a cut and has to go to the screen for the
point.

Option (i): while the slot is held, the agent writes a short spoken summary in
the same message, and `speakable` speaks that paragraph instead of the whole
reply. The words stay the agent's own, so V8 ("the server resolves a source")
holds: the server still resolves a source reference, and now also selects a
tagged paragraph inside it. No second model, no language model in the rig's
loop (Q2), no paraphrase (Q3's rejected path).

## The turn note

While the slot is held, the backend appends a note to the turn text the model
receives:

> End with a paragraph that begins `Spoken:` — at most two sentences, written
> for the ear. It is read aloud; the rest of the message stays on screen.

- **Where.** The same seam `_append_busy_resume_note` uses
  (`local_operator/session/session.py:27040`): the note rides the turn's own
  text, not a system block. The note sits in the NEWEST message, so the cached
  prompt prefix is untouched. A system-block note would sit ahead of the
  messages and invalidate the whole conversation cache — the advisor's own
  measurement forbids that shape (`session.py:23280-23300`).
- **Who gets it.** Every turn admitted to the slot session: typed prompts,
  voice takes (they are prompts or steers through the same handlers), and
  wake/monitor deliveries. A subagent child never speaks, so `_job_id` set
  means no note.
- **The gate.** `voice/slot.py:is_voice_session` (`:125`) answers "is this
  session the voice session" with one small JSON read per turn. That is the
  same authority the routes use; a second source of truth is how stale slots
  happen.
- **What the operator sees.** The note is harness instruction, not the user's
  own words. It should be stripped from the rendered user row the way
  `<operator-references>` is (`local_operator/harness/rows.py:772`), while
  staying in the journal row the model replays. If the note is visible, a
  reader sees the harness talking to the model in their own message.
- **Cache cost.** About 50-100 extra output tokens per voiced turn (the
  plan's estimate, not measured here). It is output, not a new request, so it
  adds no latency to the reply.

## `speakable`: extracting the paragraph

`speakable()` (`local_operator/voice/speakable.py:134`) currently is
`cap(clean(text))`. Option (i) adds one step before `clean`:

1. **Select.** Find a paragraph whose first non-space token is `Spoken:`,
   accepting the bolded form (`**Spoken:**`) because models bold labels. The
   paragraph runs to the next blank line. When several qualify, the LAST one
   wins — a model that writes it twice means the later one.
2. **Fallback.** No paragraph, or an empty one, means today's path: `clean` and
   `cap` the whole message, exactly as shipped. An empty `Spoken:` is not
   silence; silence is the failure mode the gate announcement deliberately
   avoids.
3. **Still clean and cap.** The selected paragraph goes through the same
   `clean` (fences, URLs, markdown) and `cap`. A model that writes six
   sentences gets cut, and `truncated` (added to `VoiceText` in `qwen-voice.md`
   "Frozen contract") says so. The closing sentence and its arithmetic are
   unchanged, including its counting against the cap.
4. **The label is machinery.** `Spoken:` is a structural marker, not copy: it
   is never translated, never rendered, and never spoken. It is matched
   case-insensitively at a paragraph start only, so an ordinary sentence
   containing the word "spoken" is not a marker.

The paragraph stays in the message on screen. It is the agent's own words, and
hiding it would make the transcript differ from what was said; a surface that
wants to fold it can, later and on its own.

## Which design-doc rows this amends

All in `docs/design/qwen-voice.md`:

- **V8 — Who decides what is spoken** (`:181`): still server-side and still by
  source reference; the server now selects the agent's `Spoken:` paragraph
  inside that source before the cap. The fallback keeps today's behaviour for a
  source with no paragraph.
- **Q10 — No lop speech code** (`:82`): unchanged in substance. The wording
  gains "and selects the agent's `Spoken:` paragraph" beside "returns text".
- **Q3 — Verbatim speech goes through a token** (`:53`): the sentence "the only
  edits to spoken text are s2s's own `remove_markdown`/`remove_unspeechable`
  and lop's `speakable` cap" becomes "… and lop's `speakable` paragraph
  selection and cap".
- **Q7 — Filler is a closed set of phrases** (`:73`): **not** amended. The
  `Spoken:` paragraph is the agent's prose; the filler set stays closed and
  model-written filler still does not exist.
- **Settings row** (`:203`): `voice.max_spoken_chars` now caps the selected
  paragraph, and caps the whole message on the fallback path.
- **Frozen contract row** for `POST /v1/desktop/voice/text`: the shape does
  not change (`{text, spoken_chars, truncated}`); `truncated` now reports a cut
  of the paragraph or of the fallback.

## Open questions

1. **Every turn, or only final answers?** The note rides every turn in the slot
   session. A one-line answer pays for a `Spoken:` paragraph it does not need.
   Alternative: note only the last message of a turn, which needs the turn's
   end to be known at admission — the backend does not know it then.
2. **What the operator sees.** Strip the note from the rendered row (proposed),
   or show it? The `@path` precedent strips. Showing it costs a line of noise
   in every message; stripping makes the journal and the screen differ.
3. **Cache.** The note is append-only in the message tail, but confirm with a
   cache-hit measurement on a live session before shipping; the advisor's
   numbers are for a different call path.
4. **Multiple markers.** Last paragraph wins is proposed. Is the FIRST one
   better when the agent writes a summary early and then keeps working? The
   spoken text is fetched after the turn, so late usually means final.
5. **Empty `Spoken:`.** Fallback to the whole text (proposed), not silence.
   Silence leaves the operator with nothing after asking a question.
6. **Instruction adherence.** The model may ignore the note, write five
   sentences, or label something else `Spoken:`. The fallback and the cap cover
   each case, but no test can prove a model obeys; this is a quality question,
   not a contract one.
7. **Surfaces.** The note would apply to any surface in the slot session (TUI,
   CLI, mobile), not only the desktop. Is that wanted, or is the note
   desktop-only while the slot is what it is?
8. **Steers and wakes.** Both produce assistant text that `/voice/text` can
   resolve. Does a steer's answer carry the note too, or only prompts?
9. **Localisation.** `Spoken:` is a machine label. It must never be translated,
   and the note itself is model-facing English like other harness prompts.
10. **Cost of the gate read.** One slot-file read per turn is small but not
    free; confirm it is not on a hot path in a session that never uses voice.
