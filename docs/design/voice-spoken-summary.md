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
one paragraph to the turn text the model receives. The exact text
(`local_operator/voice/summary_note.py`, `VOICE_SUMMARY_NOTE`):

> (The user dictated this message by voice and is listening to your reply. End
> your reply with a final paragraph that starts with `Spoken:`. Write at most
> two plain sentences for the ear: no code, no URLs, no markdown, no list. Say
> the point of the reply, in the language the user is speaking. Keep the word
> `Spoken:` exactly as written. Do this even when the reply is short.)

- **Where.** `Session.prompt`, beside the `input_mode` carriage
  (`local_operator/session/session.py:11134` on `32270f862`), AFTER `@path`
  expansion: the note rides the NEWEST user message, not a system block, so the
  cached prompt prefix is untouched. A system-block note would sit ahead of the
  messages and invalidate the whole conversation cache. (The original note cited
  the advisor's own measurement of that; it was not re-measured here.) It is NOT
  appended in `_append_busy_resume_note` (`session.py:38626`): that helper serves
  the wake/monitor/catch-up deliveries, which the narrowed gate excludes.
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
- **What the operator sees.** Nothing. `harness/rows.py:user_row_text` (`:954`)
  strips the note by its exact bytes, as it does an `<operator-references>`
  block, so every surface that paints a user row (TUI, phone, naming, resume)
  shows what the person said, while the stored journal row keeps the note for
  the model to replay. The strip matches the COMPLETE note, so a user who quotes
  a fragment of it loses nothing from their row.
- **No new setting.** The gate is the slot plus `input_mode`; there is no switch.
- **Cost.** About 50-100 extra output tokens per voiced turn (the plan's
  estimate, not measured here); output, not a new request.

## `speakable`: extracting the paragraph

`speakable()` (`local_operator/voice/speakable.py:325` on `32270f862`) was
`cap(clean(text))`. Option (i) adds one step before `clean`, `select_spoken`:

1. **Select.** Find a paragraph whose first non-space token is `Spoken:`,
   accepting the bolded form (`**Spoken:**`) because models bold labels. The
   paragraph runs to the next blank line. When several qualify, the LAST one
   wins — a model that writes it twice means the later one.
2. **Fallback.** No paragraph, or an empty one, means today's path: `clean` and
   `cap` the whole message, exactly as shipped. An empty `Spoken:` is not
   silence; silence is the failure mode the gate announcement deliberately
   avoids. "Empty" includes a paragraph that `clean` reduces to nothing, and the
   LAST marker paragraph is the one judged: an empty final `Spoken:` falls back
   to the whole reply rather than reaching back to an earlier one. Selection is
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
   (the UI streams the paragraph last) has no marker yet, so it falls back.
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

All in `docs/design/qwen-voice.md` (line numbers at `32270f862`; this build records the
amendment here rather than editing those rows, and adds only the `spoken_source` bullet
beside the `/voice/text` contract row):

- **V8 — Who decides what is spoken** (`:216`): still server-side and still by
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
- **Settings row** (`:293`): `voice.max_spoken_chars` now caps the selected
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
3. The note is stripped from rendered rows by exact bytes in `user_row_text`.
4. Gates are never narrowed (new; not in the original).

## Open questions — answered

1. **Every turn, or only final answers?** Neither: only dictated turns in the
   voice session (Decisions 1). A dictated one-line question still gets the
   note ("even when the reply is short"); the cost is a short paragraph.
2. **What the operator sees.** Stripped from every rendered row by exact bytes;
   the journal keeps it (see "The turn note").
3. **Cache.** The note is appended to the newest message only. A live
   cache-hit measurement was NOT done here; it remains a post-ship observation.
4. **Multiple markers.** LAST wins (Decisions; the spoken text is fetched after
   the turn, so late means final).
5. **Empty `Spoken:`.** Falls back to the whole reply, not silence.
6. **Instruction adherence.** A quality question, not a contract one: the
   fallback and the cap cover a model that ignores, over-writes or mislabels.
7. **Surfaces.** The gate is the voice slot plus `input_mode="dictated"`, so a
   surface (desktop, phone) gets the note only for a dictated prompt into the one
   session that holds the slot; typed prompts on any surface get nothing.
8. **Steers and wakes.** Neither carries the note (Decisions 1); their answers are
   still selectable if the agent writes a `Spoken:` paragraph unprompted.
9. **Localisation.** `Spoken:` is a machine label, never translated. The note is
   model-facing English prompt machinery, which the i18n rule exempts (AGENTS.md:
   model-facing prompts stay English); it asks for the paragraph in the user's
   language, and no `wire.*` key is involved.
10. **Cost of the gate read.** One slot-file read per DICTATED prompt, off the
    loop; none for typed prompts, wakes or sessions that never use voice.
