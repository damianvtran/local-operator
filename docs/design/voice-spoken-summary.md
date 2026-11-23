# Design: spoken summary for voice — option (i), agent-written

Status: **Approved to build 2026-11-23 (Ben).** Source: §11 of the architect's
plan "Qwen voice without the browser tab" (2026-10-18), which recommends option
(i) over a small rewriting model (ii) and over a frontend LLM paraphrase (iii).
Built as the server half in `feat/voice-spoken-summary` on `origin/main` @
`32270f862`; line numbers below are against that commit and drift as the code
is edited (names are the stable anchor).

The original note was design-only and changed no contract. The build narrows it
(see "Decisions") and adds ONE additive field, `spoken_source`.

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

When the slot is held and the prompt was dictated by voice, the backend appends
one paragraph, as its own message, to the request the model receives. It is
never stored. The exact text
(`local_operator/voice/summary_note.py`, `VOICE_SUMMARY_NOTE`):

> (The user dictated this message by voice and is listening to your reply. End
> your reply with a final paragraph that starts with `Spoken:`. Write at most
> two plain sentences for the ear: no code, no URLs, no markdown, no list. Say
> the point of the reply, in the language the user is speaking. Keep the word
> `Spoken:` exactly as written. Do this even when the reply is short.)

- **Where.** Two places. `Session.prompt` (beside the `input_mode` carriage) keeps
  the gate and records the id of the dictated user message it just built
  (`Session._voice_note_for`); it never edits the text, so the stored row is
  exactly the user's words. `Session._with_voice_note`, the last stage of
  `Session._compose_transform_context`, appends the note to the request-local copy
  of the history before EVERY provider call of the turn (the loop applies
  `transform_context` to a copy, `harness/loop.py` just before `convert_to_llm`;
  `team_bridge/request_body.py` is the precedent): when the newest user-role
  `Message` is the recorded one, it returns `[*messages, note]`, where `note` is a
  plain user `Message` flagged `_ephemeral`. A plain `Message` passes through
  `_default_convert_to_llm` unchanged, so there is no new carrier type. The input
  list is never mutated, and the stage returns its input untouched when nothing is
  armed.
- **Cache.** The note is the LAST message and `_ephemeral`, and the prompt-cache
  breakpoint code skips trailing ephemeral messages (`providers/clients.py`, both
  the Anthropic and the OpenAI-compatible encoders), so the marker stays on the
  persistent message before it and the prefix is byte-identical with and without
  the note. A system-block note would sit ahead of the messages and invalidate the
  whole conversation cache. (Offline encoding is pinned in
  `tests/unit/session/test_voice_summary_note.py`; a live cache-hit measurement
  was not done and stays a post-ship observation.) It is NOT appended in
  `_append_busy_resume_note`: that helper serves the wake/monitor/catch-up
  deliveries, which the narrowed gate excludes.
- **Disarming.** `_voice_note_for` is cleared in the `finally` of the `prompt` that
  set it, so a wake or monitor turn that follows (no `prompt`, and the dictated
  message would otherwise still be the newest user message) gets nothing. A newer
  user message displaces the note too: a steer typed or spoken mid-turn becomes the
  newest user message, so later provider calls of that turn go without the note, and
  the reply to the steer falls back to the whole reply unless the agent writes a
  `Spoken:` paragraph unprompted. The id lives in memory only: a turn resumed
  after a restart has none, so it gets no note and falls back the same way.
- **Who gets it (narrowed).** ONLY a prompt carrying `input_mode="dictated"` (set
  by the voice utterance route, `server/routes/desktop_voice.py:940`) in the
  session that holds the slot. A typed prompt, a wake, a monitor delivery, a
  steer, a harness-injected turn and a subagent turn get none: nobody is
  listening to those, and a one-line typed answer should not pay for a spoken
  paragraph. A dictation that is a mixed typed-and-spoken draft (`mixed`) is
  also left alone.
- **The gate.** `voice/slot.py:is_voice_session` (`:125`) — the same authority
  the routes use. One small JSON read per DICTATED prompt, run off the session
  loop with `asyncio.to_thread`; nothing is read for any other prompt.
- **What the operator sees.** Exactly what they said, everywhere, because there
  is nothing to hide: the journal row, the replay, the live `message_start`, the
  session preview (`session/preview.py`), the desktop and phone history folds, the
  TUI row, naming and compaction all read the stored row, which has no note.
  `harness/rows.py` is unchanged by this feature.
- **No new setting.** The gate is the slot plus `input_mode`; there is no switch.
- **Cost.** About 50-100 extra output tokens per voiced turn (the plan's
  estimate, not measured here); output, not a new request.

## `speakable`: extracting the paragraph

`speakable()` (`local_operator/voice/speakable.py:325` on `32270f862`) was
`cap(clean(text))`. Option (i) adds one step before `clean`, `select_spoken`:

1. **Select.** Find a paragraph whose first non-space token is `Spoken:`,
   accepting the bolded form (`**Spoken:**`) because models bold labels. At most
   three spaces of indent are allowed, as in Markdown: a block indented four
   spaces or a tab is a code block that may quote the protocol and is not a
   marker, and neither is a quoted block (`> Spoken:`; the `>` is not
   whitespace). The paragraph runs to the next blank line (`\n` or `\r\n`). When
   several qualify, the LAST one wins — a model that writes it twice means the
   later one. A paragraph that is the label ALONE (`**Spoken:**`, summary in the
   NEXT paragraph) takes the paragraph immediately after it as the summary: the
   label is machinery and is never itself what is spoken.
2. **Fallback.** No paragraph, or an empty one, means today's path: `clean` and
   `cap` the whole message, exactly as shipped. An empty `Spoken:` is not
   silence; silence is the failure mode the gate announcement deliberately
   avoids. "Empty" includes a paragraph that `clean` reduces to nothing, and the
   LAST marker paragraph is the one judged: an empty final `Spoken:` falls back
   to the whole reply rather than reaching back to an earlier one, with that
   label-only paragraph dropped from the spoken text (the word "Spoken" is never
   said). Selection is
   a PURE function of the message text, because `/voice/text` and every
   `/voice/speech` segment re-resolve the source and must agree.
3. **Still clean and cap.** The selected paragraph goes through the same
   `clean` (fences, URLs, markdown) and `cap`. A model that writes six
   sentences gets cut, and `truncated` (added to `VoiceText` in `qwen-voice.md`
   "Frozen contract") says so. The closing sentence and its arithmetic are
   unchanged, including its counting against the cap.
4. **The label is machinery.** `Spoken:` is a structural marker, not copy: it
   is never translated and never spoken (the selected paragraph has it
   removed). It is matched case-insensitively at a paragraph start only, so an
   ordinary sentence containing the word "spoken" is not a marker. Code fences
   are masked before the scan: `Spoken:` inside a fence is not a marker and a
   blank line inside one does not end a paragraph. A trailing partial message
   (the UI streams the paragraph last) falls back only when the cut precedes the
   label; one cut after it ("Spoken: Done and") selects the partial paragraph as
   it stands. The final read happens once the turn is done, so a settled message
   is never partial.
5. **Which sources.** Message sources only. A gate (a pending approval card) is
   never narrowed: its text can quote model-written arguments, and a paragraph
   in it that opened with the label must not stand in for the whole card.
6. **`spoken_source`.** `VoiceText` gains `spoken_source: "summary" | "reply"`
   (a missing field reads as `"reply"`). `text`, `spoken_chars`, `truncated` and
   `segments` keep their meaning; `truncated` says the cap cut what was spoken,
   summary or reply. `full_text` is unchanged (the whole reply, label included:
   the transcript is not edited). `/voice/speech` returns audio and needs no
   field. The `voice` capability stays `2`.

The paragraph stays in the message on screen. It is the agent's own words, and
hiding it would make the transcript differ from what was said; a surface that
wants to fold it can, later and on its own.

## Which design-doc rows this amends

All in `docs/design/qwen-voice.md` (line numbers at `32270f862`, verified with
`git show 32270f862:docs/design/qwen-voice.md`; this build records the amendment
here rather than editing those rows, and adds only the `spoken_source` bullet beside
the `/voice/text` contract row. Q3 and Q10 are already marked retired/superseded on
the cloud path at `:16-17`, so their amendments are for the record):

- **V8 — Who decides what is spoken** (`:213`): still server-side and still by
  source reference; the server now selects the agent's `Spoken:` paragraph
  inside that source before the cap. The fallback keeps today's behaviour for a
  source with no paragraph.
- **Q10 — No lop speech code** (`:103`): unchanged in substance. The wording
  gains "and selects the agent's `Spoken:` paragraph" beside "returns text".
- **Q3 — Verbatim speech goes through a token** (`:70`): the sentence "the only
  edits to spoken text are s2s's own `remove_markdown`/`remove_unspeechable`
  and lop's `speakable` cap" becomes "… and lop's `speakable` paragraph
  selection and cap".
- **Q7 — Filler is a closed set of phrases** (`:90`): **not** amended. The
  `Spoken:` paragraph is the agent's prose; the filler set stays closed and
  model-written filler still does not exist.
- **Settings row** (`:290`): `voice.max_spoken_chars` now caps the selected
  paragraph, and caps the whole message on the fallback path.
- **Frozen contract row** for `POST /v1/desktop/voice/text`: gains the additive
  `spoken_source` (the row already carries `segments` and `full_text`);
  `truncated` now reports a cut of the paragraph or of the fallback.

## Decisions

Ben, 2026-11-23 13:20 EST, answers to the questions put to him (verbatim):

| Question | Answer |
|---|---|
| `vs_visible` | "Show it at the end" |
| `vs_engine` | "The agent writes it in its reply" |
| `vs_history` | "Summary + link to the full reply" |
| `vs_replay` | "Yes" |
| `vs_fallback` | "First 800 characters, then 'the rest is on screen'" |
| `vs_speak_button` | "Yes, whole reply" |
| `vs_build` | "Yes, build" |

How the build reads them:

- **`vs_engine`** is option (i) only: the agent writes a `Spoken:` paragraph. No
  second model.
- **`vs_visible`** the paragraph stays at the end of the message on screen.
- **`vs_fallback`** is the existing path: no usable paragraph means the whole
  reply, cleaned and capped at `voice.max_spoken_chars` (default 800, range
  100-4000), then the spoken closing sentence. That setting also guards the
  summary.
- **`vs_history` / `vs_replay`** are client work (the History panel's summary
  plus link to the full reply, and replaying the spoken text): the server half
  supplies them with `spoken_source` and `full_text` (already shipped, #4491).
- **`vs_speak_button`**: the per-message Speak button path
  (`/v1/agents/{id}/speech`) is NOT touched; it reads the whole reply.

Narrowing against the original proposal:

1. The note goes to **dictated prompts with the slot held** only (was: every turn
   admitted to the slot session, including typed prompts, wakes and monitors).
2. The contract gains **`spoken_source`** (was: "the shape does not change").
3. The note is NEVER STORED: it is appended to the provider request only (see
   "The note is never stored" below). Review round 1 found that a stored note
   showed in every surface that reads the row, and stripping it per surface does
   not hold.
4. Gates are never narrowed (new; not in the original).

## The note is never stored (decision, review round 1)

The first build wrote the note into the user row and hid it with a byte-exact
strip in `user_row_text`. Review and QA round 1 showed that the strip covered only
the surfaces that call `user_row_text`: the desktop history and snapshot showed the
note in every dictated bubble (M1/Q1), the session preview showed it raw (M2), the
TUI live `message_start` painted it (Q2), and compaction re-read old notes (m5).

The architect recommended option B, and the operator's lean agreed: **stop storing
the note** (relayed in the round-1 remediation brief, `arch-note-hiding`). A
request-time ephemeral tail message needs no strip anywhere, because no row ever
contains the note; there is no per-surface list to forget a surface from. Nothing
released ever stored the note, so there is no old row to strip, and the
`user_row_text` change was reverted entirely.

Risks accepted, in the open:

- **Injected `Spoken:`.** Tool or web output that ends with its own paragraph
  opening with `Spoken:` can become the spoken text on a dictated turn when the
  agent quotes or ends with it. The gate-narrowing guard stays (a gate card is
  never narrowed) and selection takes the LAST marker of the agent's own message
  only, but a reply that ends with such a paragraph is spoken as written.
- **A steer drops the note** (see "Disarming"): the rest of that turn is
  answered without the request, so it falls back to the whole reply unless the
  agent writes the paragraph anyway.
- **A turn resumed mid-turn after a restart has no note** (the id is in memory):
  fallback to the whole reply.
- **Fallback.** Whenever no `Spoken:` paragraph appears (the model ignored the
  note, a steer displaced it, the turn was resumed) the whole reply is cleaned and
  capped exactly as before the feature (`spoken_source: "reply"`).
- **Deferred (Q4).** A `Spoken:` paragraph that is only code or a URL speaks
  "(code block omitted)" or "link"; the note asks for plain sentences, and no code
  guards this.

## Open questions — answered

1. **Every turn, or only final answers?** Neither: only dictated turns in the
   voice session (Decisions 1). A dictated one-line question still gets the
   note ("even when the reply is short"); the cost is a short paragraph.
2. **What the operator sees.** Nothing different: the note is never stored, so no
   row, preview or live event contains it (see "The turn note").
3. **Cache.** The note is an ephemeral last message and the cache breakpoint skips
   it. A live cache-hit measurement was NOT done here; it remains a post-ship
   observation.
4. **Multiple markers.** LAST wins (Decisions; the spoken text is fetched after
   the turn, so late means final).
5. **Empty `Spoken:`.** Falls back to the whole reply, not silence.
6. **Instruction adherence.** A quality question, not a contract one: the
   fallback and the cap cover a model that ignores, over-writes or mislabels.
7. **Surfaces.** The gate is the voice slot plus `input_mode="dictated"`, so a
   surface (desktop, phone) gets the note only for a dictated prompt into the one
   session that holds the slot; typed prompts on any surface get nothing.
8. **Steers and wakes.** Neither carries the note (Decisions 1); their answers are
   still selectable if the agent writes a `Spoken:` paragraph unprompted. A steer
   that lands mid-way through a dictated turn also drops the note for that turn's
   remaining provider calls (the steer is the newest user message).
9. **Localisation.** `Spoken:` is a machine label, never translated. The note is
   model-facing English prompt machinery, which the i18n rule exempts (AGENTS.md:
   model-facing prompts stay English); it asks for the paragraph in the user's
   language, and no `wire.*` key is involved.
10. **Cost of the gate read.** One slot-file read per DICTATED prompt, off the
    loop; none for typed prompts, wakes or sessions that never use voice.
