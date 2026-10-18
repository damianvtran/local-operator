# LOH-5 — Transcription cleanup (fillers and non-text cues)

Status: proposed for implementation · 2026-10-18
Owner: LOH-5 delivery manager (`e627d01cffba`) · Requested by: hackathon intake (`b290eb6429d7`)
Scope source: operator brief, 2026-10-18. This note is the gate before code.

## 0. Decision summary

Clean the transcript **once, in the shared speech-to-text (STT) layer**, before any
surface sees it. Two levers exist, and the evidence says they are not equal:

1. **Provider-native cleanup is the primary lever.** ElevenLabs Scribe v2 now has a
   `no_verbatim` request field that removes filler words, false starts, disfluencies
   and non-speech sounds at the model, plus `tag_audio_events=false` to suppress
   `(laughter)`-style cues. It is free (no surcharge is documented), language-aware,
   and our live probe shows it removes English/French fillers while preserving
   Portuguese `um` (a real word) and Arabic `أمي` (mother). It is **under-inclusive**
   for the CJK/Hindi samples we probed: Japanese `えーと`, Chinese `嗯` and Hindi
   `अरे` survived. Under-cleaning is the safe failure direction.
2. **ElevenLabs `transcript_edit` is the opt-in multilingual lever.** It takes a
   natural-language instruction and returns `edited_transcript` **alongside the raw
   `text`**, so the raw copy is preserved. It removed the CJK/Hindi fillers our
   `no_verbatim` probe left, and preserved Portuguese/Arabic real words. It is
   experimental, costs **+30%** on the base transcription (minimum 10 s billed), and
   its notion of "filler" is not ours: it removed the Hindi vocative particle `अरे`
   in our negative-control probe. Therefore it is opt-in, never the default.
3. **Client-side heuristics are a fallback, not the primary path.** A blanket ASCII
   regex is unsafe; the design uses a small per-language rule table with no rules for
   languages where the candidate token is a real word, a closed cue lexicon instead
   of generic bracketed-text removal, and a hard no-op on unknown languages. In v1
   this covers English and French only; CJK, Arabic, Hindi/Urdu and other scripts are
   deliberately untouched by client rules. This is the only lever that helps the
   **Radient** rung, which is the rung this machine actually runs today.

The first implementation slice is the shared cleanup module plus the provider-native
ElevenLabs request fields, provenance and raw-text carrying, with tests. It does not
add a settings surface yet; the mode is a request field with a safe default. The
settings key, the conservative client rules and the opt-in `edit` mode follow in
separate slices so each can be reviewed on its own evidence.

## 1. Scope boundary

In scope: cleaning the text returned by the STT cascade before it reaches a surface.

Explicitly out of scope:

* Audio generation, sound effects, TTS, or any change to the voicing path.
* Streaming transcription. The pipeline is request/response; the phone records a
  clip and uploads it (see §2). No streaming assembly exists to clean.
* Re-transcribing, diarisation, timestamps, speaker labels or translation.
* Persisting raw transcripts in a new durable store. The design carries the raw
  text through the response and preserves it where the provider returns it; the
  existing transcript journal is not repurposed in v1.
* A per-team policy server. The setting is per installation/user; a team default
  can ride an existing team brief later without a new mechanism here.

## 2. The pipeline as it actually is (evidence, 2026-10-18)

### 2.1 Map

```text
phone composer (MediaRecorder, <=120 s)
  POST /api/transcribe  (mobile/daemon.py)
    -> mobile/stt.py::transcribe_audio        availability, then dispatch
      -> clients/stt.py::transcribe_with_backend
         -> Radient rung:   RadientClient.create_transcription
                            (clients/radient.py, POST /tools/transcriptions)
         -> ElevenLabs/OpenAI rung:
                            stt/cascade.py::transcribe_backend
                              -> _run_elevenlabs_rung -> stt/clients.py::ElevenLabsSttClient
                              -> _run_openai_rung     -> stt/clients.py::OpenAiSttClient
         -> response {text, provider, model, path}
  composer joins result.text into the draft (web/src/lib/dictation.ts::joinDraft)
  send envelope carries input_mode/input_path (silent provenance)

desktop/API caller
  POST /v1/stt/transcriptions  (server/routes/stt.py)
    -> stt/cascade.py::transcribe_audio       whole cascade: Radient -> ElevenLabs -> OpenAI
       -> _run_radient_rung / _run_elevenlabs_rung / _run_openai_rung
    -> 200 {text, path, attempts, duration_s}

legacy caller
  POST /v1/transcriptions  (server/routes/transcription.py)
    -> Radient-only path, unchanged by the cascade work
```

There is **no streaming transcript** and no server-side transcript assembly. The
phone's draft is the first durable copy; the send envelope's `input_mode`/`input_path`
is the provenance record (docs/design/mobile-stt.md §2.1). The sidecar
(`stt/sidecar.py`) writes `stt_transcript_v1` custom entries only for the
`model_audio_sidecar` rung, which is not this path.

### 2.2 The ElevenLabs client today

`stt/clients.py::ElevenLabsSttClient.transcribe` sends exactly:

* `model_id=scribe_v2` (default constant `ELEVENLABS_STT_MODEL`)
* `language_code` when the caller supplied one
* `file`

It deliberately sends **no prompt**: the previous research pass recorded that Scribe
had no prompt field, and `transcript_edit` was considered a billed post-edit out of
scope for v1. It reads only `payload["text"]`; `language_code`, `words`, and
`edited_transcript` are discarded. `tag_audio_events` and `no_verbatim` are not sent.

The model name needs one correction: **ElevenLabs v3 is a text-to-speech model, not
the STT model.** This pipeline calls `scribe_v2` for transcription. If a hackathon
demo used "ElevenLabs v3" to generate the audio, that is the voicing side; the
transcript still comes from Scribe v2 (or another rung).

### 2.3 Which rung this machine actually runs

Read-only probe of the provider auth store (`~/.local-operator/auth.db`), names only:

| Provider row | Credential | Meaning for the STT cascade |
|---|---|---|
| `radient` | OAuth | Rung 1 is available. |
| `openai` | OAuth (ChatGPT) | Not valid at `/v1/audio/transcriptions`; rung 3 needs an `openai-key` **API key** row, which is absent. |
| `elevenlabs` | **no row** | Rung 2 is **not advertised** on this machine. |

The harness secret store has an `ELEVENLABS_API_KEY`, but that is a different store
from the provider auth store the resolver reads; a key there does not light the
ElevenLabs rung. The resolver's own rule is "persisted provider rows only". So the
live dictation path on this machine is **Radient**, and the ElevenLabs knobs only
take effect after a user stores an ElevenLabs key through the login flow.

### 2.4 Real dictated transcripts on this machine (read-only)

Scan of the 600 most recently modified session transcripts under
`~/.local-operator/sessions` (1,456 transcripts, 5.8 GB in total; the scan read
`transcript.jsonl` bytes, never modified them):

* 20 user messages carry `input_mode: "dictated"`.
* 19 of 20 have **no** `input_path` (written before the path carriage); the one
  recorded path is `provider_stt_radient`.
* A conservative token scan found **90 filler-like tokens** (upper bound: the scan
  also counted `er`, `mm`, `eh`) across 15 of the 20 messages. Manual inspection of
  25 contexts confirmed genuine `uh`/`um` hesitations and false starts
  (`un-untrusted`, `te- like`).
* One message contains the non-text cue `[lip smack]`, in the same message that
  describes this workstream. No other bracket cue matched the cue lexicon.
* No transcript content is reproduced in this note.

This is direct evidence of the operator's complaint on the **Radient** rung, and it
is why the client fallback is not optional for the current installation: an
ElevenLabs-only fix would not change a single transcript this machine produced.

### 2.5 Live ElevenLabs probes

Audio: macOS `say` generated seven short clips (English, Portuguese, Arabic,
Japanese, Hindi, French, Chinese), converted to 16-bit WAV. These are synthetic
probes, not user recordings; the transcripts below are the API's actual output, not
hand-written fixtures. Two calls per clip (`no_verbatim` absent vs `true`), plus four
`transcript_edit` calls. Key supplied from the encrypted store; never printed.

| Clip | Provider output with `no_verbatim=false` | With `no_verbatim=true` |
|---|---|---|
| en | `Um, so I, uh, think we should ship the change. Uh, can you check the build?` | `So I think we should ship the change. Can you check the build?` |
| fr | `Euh, je ne sais pas` | `Je ne sais pas` |
| pt | `Um livro, por favor. Um momento` | `Um livro, por favor. Um momento.` (both `Um` preserved) |
| ar | `أمي في البيت وأمي تحب القراءة` | `أمي في البيت، وأمي تحب القراءة.` (both `أمي` preserved) |
| ja | `えーと、今日はいい天気ですね。` | `えーと、今日はいい天気ですね。` (filler preserved) |
| zh | `嗯，我知道了。` | `嗯，我知道了。` (filler preserved) |
| hi | `अरे वह आया।` | `अरे! वह आया।` (particle preserved) |

`transcript_edit` instruction: *"Remove filler words, hesitations and non-speech
sounds, but keep every other word, punctuation, and the original language exactly as
spoken."*

| Clip | `text` (raw) | `edited_transcript.text` |
|---|---|---|
| en | `Um, so I, uh, think we should ship the change. Uh, can you check the build?` | `So I think we should ship the change. Can you check the build?` |
| ja | `えーと、今日はいい天気ですね。` | `今日はいい天気ですね。` |
| zh | `嗯，我知道了。` | `我知道了。` |
| pt | `Um livro, por favor. Um momento` | `Um livro, por favor. Um momento` (unchanged) |
| ar | `أمي في البيت وأمي تحب القراءة` | `أمي في البيت وأمي تحب القراءة` (unchanged) |
| hi | `अरे वह आया।` | `वह आया।` (**`अरे` removed**) |
| fr | `Euh, je ne sais pas` | `Je ne sais pas` |

Two response-shape facts from the probes:

* `no_verbatim=true` returns **only** the cleaned `text` and cleaned `words`; there is
  no raw copy in the response. `tag_audio_events=false` was sent and no
  `audio_event` word type appeared.
* `transcript_edit` returns the raw `text`/`words` plus `edited_transcript`
  (`kind: "transcript"` on success), so raw preservation is possible on that path.
* The returned `language_code` is ISO-639-3 (`eng`, `por`, `ara`, `jpn`, `hin`,
  `fra`, `zho`) — useful for provenance and for gating client rules.

### 2.6 ElevenLabs documentation evidence

From the live API reference and guides fetched 2026-10-18:

* `no_verbatim` (boolean): "If true, the transcription will not have any filler
  words, false starts and non-speech sounds. Only supported with scribe_v2 model."
* `tag_audio_events` (boolean): whether to tag audio events like `(laughter)`,
  `(footsteps)`. No default is stated in the reference; send `false` explicitly.
* `transcript_edit` (string, max 2000 chars): natural-language instruction applied to
  the finished transcript; `edited_transcript` returned alongside the original.
  Cannot be combined with `entity_detection`, `entity_redaction` or
  `use_multi_channel`. **+30% surcharge, minimum 10 s billed, experimental.**
* ElevenLabs' own transcript-editing guide says dedicated parameters are "cheaper
  and more predictable than an edit instruction" and names `no_verbatim` for filler
  words and disfluencies — i.e. use `no_verbatim` first, reserve editing for what it
  cannot do.

## 3. Lever evaluation

| Lever | Quality / language safety | Coverage | Cost / latency | Raw copy | Verdict |
|---|---|---|---|---|---|
| Provider `no_verbatim` (Scribe v2) | Model-native, context-aware. Probe: removes en/fr fillers; preserves pt `um`, ar `أمي`; leaves ja/zh/hi fillers. | ElevenLabs rung only | Free; no documented surcharge; one request | No (provider removes it) | **Default where available** |
| Provider `transcript_edit` | Model-native rewrite. Probe: removes ja/zh/hi fillers, preserves pt/ar words, but removed hi `अरे` (valid vocative). | ElevenLabs rung only | +30%, min 10 s; extra latency grows with transcript | **Yes** (raw `text` + edited) | **Opt-in** for languages `no_verbatim` leaves |
| Client rules | Deterministic and auditable, but language-blind regexes are unsafe. V1: en/fr token sets only, closed cue lexicon, quote/code guard, unknown language no-op. | All rungs (Radient/OpenAI today) | CPU-only | **Yes** (`raw_text` + removed spans) | **Fallback / opt-in; default only where the provider has no native knob** |

No blanket ASCII regex. No Unicode normalisation (`unicodedata.normalize` is not
called; NFC/NFD changes are forbidden). No removal of combining marks, bidi controls,
zero-width joiners, Arabic diacritics, or any script's characters except the exact
matched filler/cue lexeme. The engine never rewrites, reorders or re-encodes text; it
deletes matched spans from the original string and returns the original string
untouched when no rule applies.

## 4. Design

### 4.1 One cleanup point

Add `local_operator/stt/cleanup.py` (pure, no I/O, import-light) and call it from the
cascade executor after the serving rung returns, before `SttOutcome` is built. Both
the phone route (`transcribe_backend`) and the desktop route (`transcribe_audio`)
therefore get identical behaviour. The legacy Radient-only route is wired to the same
pure function in a later slice if it still has callers; the phone does not use it.

Why not in the phone client: the phone is one surface, the daemon is the one place
that holds provider keys and sees the provider's raw response, and a client-side
implementation would diverge from the desktop route immediately.

### 4.2 Cleanup modes

Request field `cleanup` (string; absent = `auto`), validated against a closed
vocabulary:

| Mode | ElevenLabs | Radient / OpenAI | Raw copy |
|---|---|---|---|
| `auto` (default) | `no_verbatim=true`, `tag_audio_events=false`; then client rules for residual cues/fillers | client rules | provider text; client rules preserve their input as `raw_text` |
| `provider` | `no_verbatim=true`, `tag_audio_events=false`; no client rules | no-op | none on ElevenLabs; raw == text elsewhere |
| `client` | `no_verbatim=false`, `tag_audio_events=false`; client rules | client rules | **yes** |
| `off` | send neither field (verbatim) | no-op | **yes** (raw == text) |
| `edit` | `no_verbatim=false`, `tag_audio_events=false`, `transcript_edit=<strict instruction>` | client rules (fallback) | **yes** on ElevenLabs (`text` + `edited_transcript`) |

`edit` is the only mode that uses the experimental, surcharged API; it must be
documented as such wherever it is offered. An invalid `cleanup` value is a 422
(client error), never a silent fallback to a different cleanup.

### 4.3 Client rules v1 (conservative)

Rules are keyed by a normalised language code (ISO-639-1 or the provider's
ISO-639-3 returned as `language_code`). If the language is unknown or not in the
table, the token rules are a **no-op**; cue removal still applies because the cue
labels are provider vocabulary, not natural language.

Token sets (v1):

* `en`: `uh`, `um`, `uhm`, `erm`
* `fr`: `euh`

Deliberately absent: `um` for Portuguese (numeral/article), `أم`/`أمي` for Arabic
(mother), `अरे` for Hindi (vocative), `嗯`/`呃`/`那个` for Chinese, `えーと`/`あの` for
Japanese, `음`/`어` for Korean, `eh` for Spanish/Portuguese, `er`/`mm` for English.
Those require a model-assisted pass (`edit`) or a later, evidence-backed rule.

Deletion is span-based on the original string:

* match only at a word boundary (`(?<!\w)token(?!\w)`, Unicode-aware), not inside
  `album`, `sum`, `hum`, `documentum`, or a hyphenated compound;
* skip all-uppercase tokens (`UM`, `UH`) so abbreviations survive;
* skip matches inside backtick/code spans and straight/curly double-quoted spans;
* remove the token plus only the punctuation/space that existed to separate it
  (`Um, so` -> `so`; `I, uh, think` -> `I think`; `Okay. Um. Let's go.` ->
  `Okay. Let's go.`); never remove unrelated punctuation, collapse unrelated
  whitespace, or change case;
* return `removed` spans in raw-string offsets for audit.

Cue removal uses a **closed lexicon**, not generic brackets: `(laughter)`,
`[laughter]`, `(laughs)`, `(sighs)`, `(coughs)`, `(clears throat)`, `(footsteps)`,
`(applause)`, `(music)`, `(noise)`, `(static)`, `(breath)`, `(lip smack)`,
`(lip smacking)`, `(sniffs)`, `(whistles)`, `(silence)`, `(pause)`, and their
`[...]` forms. It never removes `[inaudible]`, `[sic]`, `[REDACTED]`,
`[CREDIT_CARD]` or any entity label — those are information, not noise.

Idempotency: cleaning already-clean text returns it byte-for-byte; applying the
rules twice equals applying them once.

### 4.4 Provenance and raw preservation

`SttOutcome` and the wire payload gain:

* `cleanup`: `"none" | "elevenlabs.no_verbatim" | "elevenlabs.transcript_edit" |
  "elevenlabs.transcript_edit_failed" | "client.rules.v1"` and joined forms such as
  `"elevenlabs.no_verbatim+client.rules.v1"`;
* `raw_text`: the text before **client** rules; `None` when the provider itself
  removed the fillers and returned no raw copy;
* `language`: the provider-detected language code when it reports one;
* `removed`: ordered spans `(start, end, kind, token)` for audit; empty when nothing
  was removed.

The rule is: **our layer never destroys the only copy.** If client rules run, the
input is returned as `raw_text`. If the provider removes at source (`no_verbatim`),
there is no raw copy to preserve — that is an upstream contract, not our deletion;
`cleanup=off` is the documented way to obtain a verbatim transcript. `edit` returns
the raw provider transcript by API contract, so that mode satisfies the rule even
when it removes the most.

No new durable store is introduced. The response carries the raw copy to the caller;
the phone client may keep it in memory for an undo affordance later, and the desktop
caller may persist it. The existing journal already stores the **sent** (cleaned)
message and its `input_path`; changing that is a separate data-retention decision.

### 4.5 Configuration

The full design adds one `/settings` key under the existing `speech` section:

`speech.transcription.cleanup` — enum `auto` (default), `provider`, `client`, `off`,
`edit`; scope LIVE (the route reads it per request); per user/installation via
`config.yml`. The first implementation slice uses the same vocabulary as a request
field with a module-level default so the API is usable before the settings surface
lands. A team-level default is out of scope; a team brief can select the per-user
default later without a new config mechanism.

## 5. Test matrix

### 5.1 Unit — provider request shape

| Case | Expectation |
|---|---|
| `auto` on ElevenLabs | multipart has `no_verbatim=true`, `tag_audio_events=false`, `model_id=scribe_v2`; no `prompt`; no `transcript_edit` |
| `provider` | same as `auto`, no client rules run |
| `client` | `no_verbatim=false`, `tag_audio_events=false`; client rules run |
| `off` | neither `no_verbatim` nor `tag_audio_events` sent; text returned byte-identical |
| `edit` | `transcript_edit` present (<=2000 chars), `no_verbatim=false`, `tag_audio_events=false`; `edited_transcript.text` chosen; raw `text` carried |
| `edit` failure | `edited_transcript.kind == "error"` does not fail the request; raw text returned, provenance `...edit_failed` |
| invalid mode | 422 at the route; no provider call |
| language passthrough | `language_code` still sent when supplied |
| no ElevenLabs key | existing "No ElevenLabs API key is stored." unchanged |

### 5.2 Unit — client rules

Fixtures per language, each with a positive and a negative control:

| Language | Input | Expectation |
|---|---|---|
| en | `Um, so I, uh, think we should ship.` | `So I think we should ship.` (exact string pinned by the implementation; no filler left) |
| en negative | `The album is in the sum. Hum a tune.` | byte-identical |
| en abbreviations | `UM and UH are universities.` | byte-identical (all-caps guard) |
| en quote | `He said "um" and left.` | byte-identical inside quotes |
| en code | `` `um = 1` `` / fenced block | byte-identical |
| fr | `Euh, je ne sais pas.` | `Je ne sais pas.` |
| pt | `Um livro, por favor. Um momento.` | byte-identical |
| ar | `أمي في البيت، وأمي تحب القراءة.` | byte-identical; bidi marks/diacritics preserved |
| hi | `अरे, वह आया।` | byte-identical |
| ur | `امی گھر میں ہیں۔` | byte-identical; RTL preserved |
| ja | `えーと、今日はいい天気ですね。` | byte-identical |
| zh | `嗯，我知道了。` | byte-identical |
| mixed | `Um, but in Portuguese um livro means a book.` | the first `Um` may be removed only if language is `en`; the Portuguese `um` must remain — this is a **known limitation** of a single language code and is covered by a test that documents the chosen behaviour |
| cue | `I agree. [lip smack] Let's go.` | `I agree. Let's go.` |
| cue negative | `The label says [inaudible].` | byte-identical |
| idempotency | clean(clean(x)) == clean(x) | true for every fixture |
| no normalization | accented/composed text unchanged outside removed spans | byte-identical codepoints |
| disabled | `cleanup=off` | byte-identical to provider text for all fixtures |

### 5.3 Integration / real path

* `POST /v1/stt/transcriptions` with a fake ElevenLabs transport: response has the
  cleaned text, `cleanup` token, `raw_text` (or null), `language`, and the existing
  `path`/`attempts` fields unchanged.
* `POST /api/transcribe` on the mobile daemon: same fields; composer's `joinDraft`
  still receives a string; no phone code change required in slice 1.
* Cascade fall-forward: ElevenLabs refuses -> OpenAI serves -> cleanup rules apply
  to the OpenAI text; provenance names the serving rung, not ElevenLabs.
* `cleanup=off` returns exactly the provider's text, byte-for-byte.
* Legacy route: unchanged in slice 1, or wired to the same pure function if its
  callers still matter (decide in the slice PR, with a test that the legacy response
  shape is unchanged except for additive fields).

### 5.4 Live evidence (QA, not unit)

* Re-run the seven-language probe with the shipped client against the real
  ElevenLabs API; record `text`, `cleanup`, `raw_text`, `language` for `auto`,
  `off`, `provider`, `client` and `edit`.
* Replay the real dictated transcripts found on this machine (read-only) through
  `client.rules.v1`; report before/after counts without reproducing content.
* Negative controls: Portuguese `Um livro`, Arabic `أمي`, Hindi `अरे`, Japanese
  `えーと`, Chinese `嗯`, accented French, and a mixed-script sentence must not be
  damaged by `client` mode. If any is, the rule is removed, not waived.
* `edit` mode must be measured for cost/latency and its +30% surcharge disclosed in
  the report; it is never enabled by default.

## 6. Slice plan (serial; one PR per slice)

1. **This design note** — pipeline findings, lever evaluation, test matrix, slice
   plan. Docs-only PR. No product code.
2. **Shared cleanup module + provider-native ElevenLabs request fields.** Add
   `stt/cleanup.py` with the mode resolver and provider request options,
   `no_verbatim`/`tag_audio_events`, `SttOutcome` fields (`raw_text`, `cleanup`,
   `language`, `removed`), request field `cleanup` with `auto` default, both routes'
   plumbing, unit tests and fake-transport integration tests. No settings key, no
   client token rules, no `edit` yet (mode `client` behaves as `off` until slice 3;
   `edit` is rejected as unsupported until slice 5). This is the first
   implementation slice.
3. **Conservative client rules v1.** `en`/`fr` token sets, closed cue lexicon,
   quote/code/all-caps guards, span deletion, multilingual fixtures and negative
   controls, idempotency, `off` byte-equality. Wire into `auto`/`client` for the
   Radient and OpenAI rungs (the current live path).
4. **Settings surface.** `speech.transcription.cleanup` in `/settings` (enum,
   LIVE), i18n source + generated keys, consumer-default test, route reads the
   config, per-user opt-out. Design/UX review only if a visible control is added;
   a settings row alone is a settings change, and the `/settings` frame is
   inspected as a rendered still.
5. **Opt-in `edit` mode.** Expose it behind the setting with the +30% and
   experimental warning in the copy; live probe evidence; document the Hindi
   `अरे` negative control and the raw-preservation guarantee.
6. **Release handoff.** No version bump in any PR; hand the merged PRs to the
   current release-window owner. `lop-update` is the owner's action, not this
   workstream's.

Slices 2 and 3 are the minimum that changes what this machine's users see: slice 2
makes the ElevenLabs rung correct when it is used, slice 3 cleans the Radient rung
that is used today. Slice 4 makes the behaviour optional per user. Slice 5 is the
multilingual completeness path and stays opt-in because its evidence includes a
real negative control.

## 7. Risks and open questions

1. **No ElevenLabs credential is persisted on this machine.** Until a user stores
   one, the provider lever is dormant. Do not copy the harness secret-store key into
   the provider auth store; that is a credential decision for the user. The report
   to the hackathon manager must say this plainly, because it contradicts the
   assumption that the current dictation path is ElevenLabs.
2. **`no_verbatim` is under-inclusive in CJK/Hindi.** A user who wants those fillers
   removed must choose `edit` and accept +30% and the experimental API. The Hindi
   `अरे` result shows `edit` can remove a valid particle; the raw copy is the
   mitigation, and the mode is opt-in.
3. **Provider defaults are not contractual.** `tag_audio_events` has no documented
   default, so it is always sent explicitly. If ElevenLabs changes `no_verbatim`
   semantics or model support, the wire test and live probe catch it; the cleanup
   mode remains `off`-able.
4. **Code-switching with a single language code.** Client token rules cannot tell an
   English `um` from a Portuguese `um` inside one transcript. V1 restricts client
   rules to languages where the token is not lexical and documents the mixed-language
   limitation; the provider-native lever is language-aware and is preferred.
5. **Raw retention.** `no_verbatim` gives no raw copy. The design preserves raw where
   the provider returns it and documents the upstream limitation; a future durable
   raw store is a data-retention decision, not part of this workstream.
6. **The legacy Radient route.** If desktop callers still use
   `POST /v1/transcriptions`, they bypass the cascade and would not get cleanup until
   the legacy route is wired. Slice 3 must either wire it or state explicitly that it
   is deprecated; no silent divergence.
7. **Phone UI.** No toggle is proposed in the first slices. The setting is global
   per user/installation; a per-message toggle would need a design/UX round and is
   out of scope until the behaviour is proven.

## Appendix A — Re-runnable evidence commands (read-only)

```sh
# Provider rows (names only; never prints key material)
cd ~/local-operator
.venv/bin/python - <<'PY'
from pathlib import Path
from local_operator.providers.auth_store import AuthStore
p = Path.home() / ".local-operator" / "auth.db"
s = AuthStore(p, config_dir=p.parent, read_only=True)
for c in s.list_credentials():
    print(c.provider, c.credential_type, c.disabled_cause)
s.close()
PY

# Dictated messages: count only, no content
# (scan ~/.local-operator/sessions/*/transcript.jsonl for entries whose payload
#  message has input_mode == "dictated"; do not print message bodies)

# ElevenLabs live probe
# POST https://api.elevenlabs.io/v1/speech-to-text
#   model_id=scribe_v2, tag_audio_events=false, [no_verbatim=true]
#   header xi-api-key: <from the encrypted store>
# Compare text/text with no_verbatim absent vs true; for transcript_edit add
#   transcript_edit="Remove filler words, hesitations and non-speech sounds, but
#   keep every other word, punctuation, and the original language exactly as spoken."
```

## Appendix B — Source references

* `local_operator/stt/clients.py` — `ElevenLabsSttClient`, `OpenAiSttClient`
* `local_operator/stt/cascade.py` — resolver, executor, per-rung runners
* `local_operator/stt/__init__.py` — `SttOutcome`, `AudioPath`, attempt types
* `local_operator/clients/stt.py` — mobile dispatch table and `SttOutcome` mirror
* `local_operator/mobile/stt.py` — phone availability and dispatch
* `local_operator/mobile/daemon.py` — `POST /api/transcribe`
* `local_operator/server/routes/stt.py` — `POST /v1/stt/transcriptions`
* `local_operator/server/routes/transcription.py` — legacy Radient route
* `local_operator/stt/sidecar.py` — `stt_transcript_v1` precedent for durable records
* `docs/design/mobile-stt.md` — input-mode-v1 provenance contract
* ElevenLabs API reference (fetched 2026-10-18): `no_verbatim`, `tag_audio_events`,
  `transcript_edit`, response `edited_transcript`
* ElevenLabs transcript-editing guide (fetched 2026-10-18): "use `no_verbatim` to
  remove filler words and disfluencies"; +30% surcharge, experimental
