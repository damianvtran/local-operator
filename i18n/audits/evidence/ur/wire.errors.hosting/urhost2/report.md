# `wire.errors.hosting` · Urdu (`ur`) — independent audit (reader 2)

| | |
|---|---|
| Namespace | `wire.errors.hosting` (new, **1 key**) |
| Locale | `ur` |
| Round | contribution audit (not a TUI-chrome round) |
| Reader role | **second** independent reader |
| Independence | I did not translate the key and did not author the caller; I did **not** read the other reader's verdict or receipt before this verdict was fixed. I read only the author's record packet and my own files. |
| Auditor | `i18n-auditor`, session `25304c88781e` |
| Model | **requested** (host system prompt): `anthropic/claude-sonnet-5-5`; **served: UNKNOWN** (not exposed; an earlier session-state snapshot exposed only the route `radient/auto`; no fallback is claimed) |
| Date | 2026-11-03 |

## Verdict

**`passed` — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.**

The one key renders the owned condition correctly, adds no remedy or detail, is a
complete statement with the Urdu terminal full stop, and follows the shipped
`wire.errors` pattern for "not configured".

```
copy=passed   native/renderer=FALSE(NOT RUN)   runtime locale eligibility=NOT CLAIMED
```

## Inputs and hashes (computed in this session, stdlib `hashlib`)

| Input | Path | SHA-256 | Matches supplied |
|---|---|---|---|
| English source | `…/sessions/3ef6295a5545/scratchpad/hosting-source-v1/wire.errors.hosting.json` (82 B) | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` | **yes** |
| Context sidecar | `…/hosting-source-v1/wire.errors.hosting.context.json` (854 B) | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` | **yes** |
| Urdu target | `…/sessions/1955dea69023/scratchpad/hosting-ur/targets/ur/wire.errors.hosting.json` (113 B) | `8f7977e2ce06a53267f0672f50605e9e5a09b5c52e85b58619acbb90dfbdd732` | **yes** |
| Author record (reference) | `…/hosting-ur/record.json` | `ca191e5a491a9f0d8044ca3311f568ccfc07d94ecdb5b6b6dc6ba838779ed7b2` | — |

A second en copy at `~/local-operator/.worktrees/startup-owned-copy/…/en/wire.errors.hosting.json`
hashes to the same `458d8f8b…` (readable worktree copy; the session copy was read directly).

The checked-out catalogue tree has **no** `wire.errors.hosting.json` for `en` or `ur`
— consistent with an uncommitted candidate, not shipped. (Source-side observation; no
Git command was run.)

## The key

- English: **`Hosting platform is not configured.`**
- Urdu: **`ہوسٹنگ پلیٹ فارم ترتیب نہیں دیا گیا۔`**

### Mechanical (source-only, actual values)

| Check | Result |
|---|---|
| Key parity | 1 / 1 / 1 — `wire.errors.hosting.not_configured` in en, context and ur; no keys added or dropped |
| ICU arguments | **none in either file** (no `{}`) → placeholder parity and CLDR plural categories are **N/A by reason**, not skipped silently |
| Encoding | valid UTF-8 JSON; 36 code points; **NFC-normalised** (`NFD` identical, no combining marks) |
| `Cf` / manual bidi | **none**; no ZWNJ/ZWJ/RLM/LRM/FSI/PDI |
| Latin text | **0 ASCII letters**; no Latin token, so no English fallback and no bidi island in the copy |
| Script fidelity | clean Urdu letters (`ہ U+06C1`, `ٹ U+0679`, `گ U+06AF`, `ی U+06CC`, `ں U+06BA`); no Arabic-variant `ي/ك/ه`, no tatweel, no Arabic comma |
| Terminal punctuation | last code point `U+06D4` — the Urdu full stop `۔`; the only punctuation in the value |

### Accuracy and idiom

- Meaning: "the hosting platform is not configured" — the condition the preflight
  resolved, with no provider named and no remedy added (the surrounding refusal owns
  the remedy, as the sidecar requires).
- Shape: a complete impersonal statement; it survives the mid-sentence splice after a
  colon and keeps its terminal full stop. Urdu has no case, so the English capital is
  not a content change.
- Precedent verified in the shipped corpus: `wire.errors.desktop.aida_no_provider`
  (ur) = `… کوئی پرووائیڈر ترتیب نہیں دیا گیا، …` — the same `ترتیب نہیں دیا گیا`
  construction for "not configured". `ترتیب` is also the guide's settled word family
  (`ترتیبات` = settings).
- `ہوسٹنگ` extends the corpus's existing `ہوسٹ` family (`ہوسٹ`, `ہوسٹ کیے`), in Urdu
  script as the loanword policy allows; `پلیٹ فارم` is the ordinary Urdu word for
  platform. Agreement is masculine (`دیا گیا`), correct for `پلیٹ فارم` as the head of
  the compound.
- `پرووائیڈر` is deliberately **not** used: it is the corpus word for the provider
  *choice* (`wire.settings.hosting.label`), while this sentence names the platform.
  The sidecar explicitly forbids importing the surrounding frame's wording into the
  clause, so mirroring the English noun is the fiducial choice. (Other locales may
  legitimately choose the provider word; that is not a uniformity requirement.)

### Consumer and scope guards

- Display path: `sdk.py::startup_cause_text` renders this code through
  `messages.render(..., locale=startup_refusal_locale())` after testing exact type and
  the catalogued English source, so the clause is locale-aware and the English
  exception text is not spliced in.
- `session_factory.HostingNotConfiguredError("Hosting platform is not configured.")`
  (`session_factory.py:816`) stays English by contract and is untouched; this key is
  display copy only. No caller, API error, catalogue, ledger or generated file was
  modified by me.

## Advisories (non-gating)

1. **New term, no corpus precedent.** `ہوسٹنگ پلیٹ فارم` has zero matches in the
   shipped `ur` catalogues (verified by search); the `ہوسٹ` family does exist. First
   use is acceptable and idiomatic, but the wave that owns
   `i18n/style-guides/ur.md` may want a glossary line for "hosting platform" so the
   next namespace does not coin a different compound (e.g. `میزبانی`).
2. **Runtime eligibility not claimed.** Whether `ur` is shipped for this surface is a
   programme/ledger fact, outside this copy audit.

## Native / renderer holds (NOT claimed)

- The clause's real rendering — the refusal frame's colon splice, terminal width and
  bidi/shaping — is **NOT RUN**; it belongs to the consumer/renderer proof, not to this
  source-only audit. The copy itself carries no Latin island and no mixed-direction
  punctuation, so it introduces no bidi hazard that the frame would have to solve.
- No pseudolocale, formatter, ICU parser, `check.py`, generation, build, app or native
  run was executed; all checks above are static plus stdlib hashing/normalisation.

## Artifacts

- `report.md` (this file) and `record.json` in
  `/Users/damian/.local-operator/sessions/25304c88781e/scratchpad/hosting-ur-audit-reader2/`
  (hashes in `SHA256SUMS`).
- Author packet: `/Users/damian/.local-operator/sessions/1955dea69023/scratchpad/hosting-ur/`.
- Source packet: `/Users/damian/.local-operator/sessions/3ef6295a5545/scratchpad/hosting-source-v1/`.
- Preserved: the finished TUI-chrome R1/R2 records are untouched.
