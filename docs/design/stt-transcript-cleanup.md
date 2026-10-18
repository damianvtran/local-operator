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
   deliberately untouched by client rules. This is the only lever that can help the
   **Radient** rung, and it needs a language hint to remove fillers; the phone web
   client supplies `navigator.language` in slice 3, normalised to the primary
   subtag before it reaches Radient (the daemon already accepts the form field).
   Without a hint the token rules no-op, while the closed cue lexicon still removes
   non-text cues because cue labels are provider vocabulary.

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
  also counted `er`, `mm`, `eh`) across **19 of the 20** messages. Manual
  inspection of 25 contexts confirmed genuine hesitations and false starts; only
  their types are reported here, never the transcript text.
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
hand-written fixtures. Two calls per clip (`no_verbatim` absent vs `true`), plus seven
`transcript_edit` calls (three in the first pass, four negative controls in the
second). Key supplied from the encrypted store; never printed.

| Clip | Provider output with `no_verbatim` absent | With `no_verbatim=true` |
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
  `(footsteps)`. No default is stated in the reference: cleanup modes that enable
  provider options send `false` explicitly. `cleanup=off` sends neither cleanup
  field (see §4.2), so verbatim output may include cues; that is the cost of a
  byte-identical disabled path.
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
matched filler/cue lexeme. The engine deletes matched spans from the original
string; the only non-deletion edit is the sentence-initial capitalization rule in
§4.3 (a removed sentence-initial filler leaves the next word's first letter
upper-cased). It never otherwise rewrites, reorders or re-encodes text, and returns
the original string untouched when no rule applies.

## 4. Design

### 4.1 One cleanup point

Add `local_operator/stt/cleanup.py` (pure, no I/O, import-light). The functions
are called from **every path that serves the phone and the cascade**:

* `stt/cascade.py::_run_radient_rung` — the desktop cascade's Radient rung;
* `clients/stt.py::_transcribe_radient_sync` — the phone's live Radient adapter,
  which does not go through the cascade at all;
* `stt/cascade.py::_run_elevenlabs_rung` and `_run_openai_rung` — provider options
  before the call, then client rules after it **only where the mode permits**
  (Radient/OpenAI; `auto` on ElevenLabs is provider-native only, per §4.2).

This is what makes the phone and desktop answers identical. The legacy
Radient-only route (`server/routes/transcription.py`) is outside this set until
slice 3, which must either wire it to the same pure function or declare it
deprecated; it is not allowed to drift silently.

Why the pure function lives in `stt/cleanup.py` and not in the phone client: the
phone is one surface, the daemon holds the provider keys and sees the provider's
raw response, and a second implementation in the web client would diverge from
the desktop route immediately.

### 4.2 Cleanup modes

Request field `cleanup` (string; absent = `auto`), validated against a closed
vocabulary:

| Mode | ElevenLabs | Radient / OpenAI | Raw copy |
|---|---|---|---|
| `auto` (default) | `no_verbatim=true`, `tag_audio_events=false`; **provider-native only**, no client rules on this rung | client rules | **none** on ElevenLabs (`raw_text=null`, `raw_unavailable=provider_no_verbatim`); client rules preserve their input as `raw_text` elsewhere |
| `provider` | `no_verbatim=true`, `tag_audio_events=false`; no client rules | no-op | none on ElevenLabs; raw == text elsewhere |
| `client` | `no_verbatim=false`, `tag_audio_events=false`; client rules | client rules | **yes** |
| `off` | send neither field (verbatim; may include audio-event cues) | no-op | **yes** (raw == text) |
| `edit` | `no_verbatim=false`, `tag_audio_events=false`, `transcript_edit=<strict instruction>` | client rules (fallback) | **yes** on ElevenLabs (`text` + `edited_transcript`) |

`edit` is the only mode that uses the experimental, surcharged API; it must be
documented as such wherever it is offered. An invalid `cleanup` value is a 422
(client error), never a silent fallback to a different cleanup.

### 4.3 Client rules v1 (conservative)

Rules are keyed by a normalised language code. **Language source, in order:**
(1) the caller-supplied `language` form field, which is authoritative; (2) the
provider's `language_code` **only** when its `language_probability` clears a
threshold (0.8) and the transcript's script is compatible with the rule set;
(3) otherwise unknown -> the token rules are a **no-op**. Cue removal still
applies because the cue labels are provider vocabulary, not natural language.
The probability gate is not optional: the live Hindi probe returned `hin` once
and `por` twice for the same voice, so a bare label is not trustworthy. On the
Radient path the response may omit language entirely; slice 3 therefore has the
phone web client send `navigator.language` as the caller-supplied hint (the daemon
form field already exists). The hint is **normalised at the daemon boundary to the
primary subtag** (`en-US` -> `en`, `zh-Hans` -> `zh`) before it reaches either the
rule key or Radient's `language` field, whose documented shape is ISO-639-1. The
provider's ISO-639-3 labels map to the same keys (`eng` -> `en`, `fra` -> `fr`); an
unmapped or unknown value is a no-op. The wire guard is stricter than the rule
key: `fil-PH` normalises to the three-letter `fil`, which is not ISO-639-1, so the
Radient `language` field is forwarded only when the value matches `^[a-z]{2}$`;
otherwise the field is omitted and the provider auto-detects. The hinted
`/api/transcribe` test asserts the Radient wire value is `en`, not `en-US`.
Without any hint, token rules do nothing; the cue lexicon still runs because its
labels are provider vocabulary.

Token sets (v1):

* `en`: `uh`, `uhm`, `erm`
* `fr`: `euh`

`um` is deliberately NOT in the English set: it is a real Portuguese
numeral/article, and a single detected language code cannot distinguish an English
filler `Um,` from Portuguese `Um livro` inside a code-switched transcript. English
`um` is cleaned by the provider on the ElevenLabs rung; on the Radient/OpenAI rungs
it remains until a model-assisted pass. Also deliberately absent: `أم`/`أمي` for
Arabic (mother), `अरे` for Hindi (vocative), `嗯`/`呃`/`那个` for Chinese,
`えーと`/`あの` for Japanese, `음`/`어` for Korean, `eh` for Spanish/Portuguese,
`er`/`mm` for English. Those require `edit` or a later, evidence-backed rule.

Deletion is span-based on the original string:

* match only at a **code-point boundary** computed with `unicodedata.category`:
  the adjacent character is not in a letter (`L*`), mark (`M*`), number (`N*`) or
  connector-punctuation (`Pc`) category, is not `-`, and is not ZWJ/ZWNJ. This
  keeps `album`, `sum`, `hum`, `documentum` and `uh-huh` intact without the
  `\w`-class traps: decomposed accents (base + combining mark), ZWJ/ZWNJ joins,
  RLM/LRM controls and Arabic diacritics are handled by category, not by an
  ASCII assumption. No normalization is performed; RLM/LRM and marks are
  preserved even when an adjacent filler is removed;
* skip all-uppercase tokens (`UM`, `UH`) so abbreviations survive;
* skip matches inside backtick/code spans and straight/curly double-quoted spans;
* remove the token plus only the punctuation/space that existed to separate it
  (`Uh, so` -> `So`; `I, uh, think` -> `I think`; `Okay. Uh. Let's go.` ->
  `Okay. Let's go.`). Sentence-initial removal keeps sentence case by
  upper-casing the first letter of the following word **only** when the removed
  token began a sentence and that word was lowercase; no other case change, no
  unrelated punctuation removal, no unrelated whitespace collapse;
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
  "elevenlabs.transcript_edit_failed" | "client.rules.v1"`; if a later slice
  combines levers, a joined form such as `"elevenlabs.no_verbatim+client.rules.v1"`;
* `raw_text`: the text before **client** rules; `None` when the provider itself
  removed the fillers and returned no raw copy. `raw_unavailable` names the reason
  (`provider_no_verbatim` today); `None` must never be read as "nothing was
  removed";
* `language`: the provider-detected language code when it reports one;
* `removed`: ordered spans `(start, end, kind, token)` in `raw_text` coordinates
  for audit; empty when **our** layer removed nothing. A provider-native removal
  can leave this empty while `raw_unavailable` is set: the spans are unknown
  upstream, not absent.

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

`speech.transcription.cleanup` — enum `auto` (default), `provider`, `client`, `off`
through slice 4; `edit` is added to the enum only in slice 5, when the code can
accept it. Scope LIVE (the route reads it per request); per user/installation via
`config.yml`. The first implementation slice uses the same vocabulary as a request
field with a module-level default so the API is usable before the settings surface
lands. A team-level default is out of scope; a team brief can select the per-user
default later without a new config mechanism.

## 5. Test matrix

### 5.1 Unit — provider request shape

| Case | Expectation |
|---|---|
| `auto` on ElevenLabs | multipart has `no_verbatim=true`, `tag_audio_events=false`, `model_id=scribe_v2`; no `prompt`; no `transcript_edit` |
| `auto` provenance | `outcome.cleanup == "elevenlabs.no_verbatim"`; `outcome.raw_text is None`; `outcome.raw_unavailable == "provider_no_verbatim"`; `outcome.language` is the provider's `language_code` |
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
| en | `Uh, so I, uh, think we should ship.` | `So I think we should ship.` (exact string pinned by the implementation; no filler left) |
| en negative | `The album is in the sum. Hum a tune.` | byte-identical |
| en `um` | `Um, the album is in the sum.` | byte-identical — `um` is not in the English set (Portuguese collision) |
| en hyphen | `uh-huh, that works` | byte-identical (hyphen guard) |
| NFD accent | `uh\u0301` | byte-identical; the combining mark blocks the token boundary (no split) |
| ZWJ/ZWNJ | `uh\u200d` / `\u200cuh` | byte-identical; joiners preserved |
| RTL control | `U+200FuhU+200F` (RLM immediately before and after `uh`, no spaces) | `uh` removed; both U+200F controls preserved; no separator whitespace involved |
| Arabic diacritic | `أُمّي` (with harakat) | byte-identical; no mark or letter removed |
| en abbreviations | `UM and UH are universities.` | byte-identical (all-caps guard) |
| en quote | `He said "uh" and left.` | byte-identical inside quotes |
| en code | `` `uh = 1` `` / fenced block | byte-identical |
| fr | `Euh, je ne sais pas.` | `Je ne sais pas.` |
| pt | `Um livro, por favor. Um momento.` | byte-identical |
| ar | `أمي في البيت، وأمي تحب القراءة.` | byte-identical; bidi marks/diacritics preserved |
| hi | `अरे, वह आया।` | byte-identical |
| ur | `امی گھر میں ہیں۔` | byte-identical; RTL preserved |
| ja | `えーと、今日はいい天気ですね。` | byte-identical |
| zh | `嗯，我知道了。` | byte-identical |
| mixed | `Um, but in Portuguese um livro means a book.` | byte-identical in `client` mode: `um` is absent from the English set, so neither occurrence is touched. Provider `no_verbatim` cleans the English filler on the ElevenLabs rung; this is the documented code-switching limitation. |
| cue | `I agree. [lip smack] Let's go.` | `I agree. Let's go.` |
| cue negative | `The label says [inaudible].` | byte-identical |
| idempotency | clean(clean(x)) == clean(x) | true for every fixture |
| no normalization | accented/composed text unchanged outside removed spans | byte-identical codepoints |
| disabled | `cleanup=off` | byte-identical to provider text for all fixtures |

### 5.3 Integration / real path

* `POST /v1/stt/transcriptions` with a fake ElevenLabs transport: response has the
  cleaned text, `cleanup` token, `raw_text` (or null), `language`, and the existing
  `path`/`attempts` fields unchanged.
* `POST /api/transcribe` on the mobile daemon: same fields, carried through the
  mobile `SttOutcome` mirror, `_adopt_outcome`, and the daemon JSON; composer's
  `joinDraft` still receives a string; no phone web-client change in slice 2.
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
   `no_verbatim`/`tag_audio_events`, the **cascade** `SttOutcome` fields
   (`raw_text`, `raw_unavailable`, `cleanup`, `language`, `removed`), the matching
   **mobile mirror** in `clients/stt.py::SttOutcome`, its `_adopt_outcome` copy, and
   the daemon's `/api/transcribe` JSON, request field `cleanup` with `auto` default,
   both routes' plumbing, unit tests and fake-transport integration tests. No
   settings key, no client token rules, no `edit` yet: `client` and `edit` are
   recognized by the closed vocabulary and rejected with a typed 422 until slices 3
   and 5. This is the first implementation slice.
3. **Conservative client rules v1.** `en`/`fr` token sets, closed cue lexicon,
   quote/code/all-caps guards, span deletion, multilingual fixtures and negative
   controls, idempotency, `off` byte-equality. Wire into `auto`/`client` for **both
   Radient adapters** (`_run_radient_rung` and `_transcribe_radient_sync`) and the
   OpenAI rung; have the phone web client send `navigator.language`, normalised to
   the primary subtag before it reaches either consumer; forward it to Radient's
   `language` field only when the normalised value matches `^[a-z]{2}$`, otherwise
   omit the field; carry `language_probability` so the language gate is real, and
   no-op the token rules (not cue removal) when the caller supplied no language and
   the provider's label is weak. Add a
   `POST /api/transcribe` Radient-rung integration test for both the hinted and
   unhinted cases.
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
makes the ElevenLabs rung correct when it is used; slice 3 cleans the Radient rung
that is used today by having the phone send its language hint and applying the
same pure rules, with cue removal working even when no hint is present. Slice 4
makes the behaviour optional per user. Slice 5 is the multilingual completeness path
and stays opt-in because its evidence includes a real negative control.

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
   default, so every cleanup mode that doesn't disable the provider sends it
   explicitly; `off` sends neither field to stay byte-identical to today, so
   verbatim output may include cues. If ElevenLabs changes `no_verbatim` semantics
   or model support, the wire test and live probe catch it; the cleanup mode
   remains `off`-able.
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

# Dictated messages: counts only, no content
.venv/bin/python - <<'PY'
import json, os
root = os.path.expanduser("~/.local-operator/sessions")
files = sorted(
    (os.path.getmtime(os.path.join(root, d, "transcript.jsonl")), os.path.join(root, d, "transcript.jsonl"))
    for d in os.listdir(root)
    if os.path.isfile(os.path.join(root, d, "transcript.jsonl"))
)[-600:]
n = with_path = 0
for _, path in files:
    with open(path, "rb") as handle:
        for line in handle:
            if b'"input_mode"' not in line or b"dictated" not in line:
                continue
            entry = json.loads(line)
            payload = entry.get("payload") or {}
            message = payload.get("message") or payload
            if message.get("input_mode") == "dictated":
                n += 1
                with_path += bool(message.get("input_path"))
print({"dictated": n, "with_input_path": with_path, "scanned": len(files)})
PY

# ElevenLabs live probe (synthetic audio; this reproduces the `en` row only,
# as a smoke test of the request shape — §2.5 is a sample, not a whole-table
# replay. Repeat with the other `say` voices/texts to reproduce the rest.)
# Writes only into a temporary directory; the repository is untouched.
TMP="$(mktemp -d)"
REPO="$HOME/local-operator"
trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
say -v Samantha -o en.aiff "Um, so I, uh, think we should ship the change. Uh, can you check the build?"
afconvert -f WAVE -d LEI16 en.aiff en.wav
"$REPO/.venv/bin/python" - <<'PY'
import json, subprocess
import httpx

key = subprocess.check_output(["lop", "secret", "get", "ELEVENLABS_API_KEY"], text=True).strip()
for no_verbatim in (False, True):
    data = {"model_id": "scribe_v2", "tag_audio_events": "false"}
    if no_verbatim:
        data["no_verbatim"] = "true"
    with open("en.wav", "rb") as audio:
        response = httpx.post(
            "https://api.elevenlabs.io/v1/speech-to-text",
            headers={"xi-api-key": key},
            files={"file": ("en.wav", audio, "audio/wav")},
            data=data,
            timeout=90,
        )
    body = response.json()
    print(json.dumps({"no_verbatim": no_verbatim, "text": body.get("text")}, ensure_ascii=False))
PY
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
