# Independent audit — `hi` · `wire.errors.hosting` (reader2)

- **Audit id:** `hi-hosting-r1-786d3d9f` · **date:** 2026-11-03
- **Auditor:** `i18n-auditor`, session `786d3d9ff0bb`
- **Model requested:** `radient/auto` (session route)
- **Model served:** **UNKNOWN** — the auto route does not disclose the served provider/model; not inferred, no fallback claimed. This session ran no model swap of its own.
- **Independence:** not the translator (`b60b4bdc8c47`), not the code author (`94dcef624dec`). I read my own prior reports and this packet's author notes only; no other reader's verdict was read before this one.

**Verdict: `passed`** — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT on the one key.
Scope: the display copy only. This is **not** frame/page clearance, not a shipped-state claim, and not a native/render proof.

## 1. Inputs, hashed here

| role | path | sha256 |
|---|---|---|
| en | `…/sessions/3ef6295a5545/scratchpad/hosting-source-v1/wire.errors.hosting.json` | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` |
| context | `…/hosting-source-v1/wire.errors.hosting.context.json` | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` |
| target hi | `…/sessions/b60b4bdc8c47/scratchpad/hosting-hi/target/hi/wire.errors.hosting.json` | `22989094e5c759832fbd1e22220b68d0b5a7165c9dca9df6690b587d5add8f2e` |

All three recomputed with stdlib `hashlib.sha256` over the exact bytes; all match the manager's values. The namespace is new, uncommitted, and **absent at base `87ca846245b2`** — nothing here is shipped or active.

## 2. The copy

| | |
|---|---|
| EN | `Hosting platform is not configured.` |
| HI | `होस्टिंग प्लेटफ़ॉर्म कॉन्फ़िगर नहीं है।` |

## 3. Mechanical checks (computed, not asserted)

- Key parity 1:1 with English; single key, same id, no additions or drops.
- Valid JSON, two-space indent, one trailing newline, UTF-8, no BOM, no CR.
- `unicodedata.is_normalized("NFC", value)` → **True**; no `Cf` (ZWJ/ZWNJ/LRM/RLM/bidi marks), no U+2028/U+2029.
- Zero ICU arguments: no braces, no `#`, no plural/select, so **placeholder and CLDR checks are N/A, with reason** — the English entry carries no parameters and the sidecar's note says "No placeholders and no dynamic detail".
- No markup, code spans, flags or identifiers in either string.
- No ASCII letters in the target: **no English fallback** in the value.
- Terminator: U+0964 DEVANAGARI DANDA, no ASCII period, no trailing space — matching the `hi` guide ("danda for full stops in prose").
- Sidecar carries `role`, `tone`, `note` only; **no `maxChars`/`maxCells`**, so no width budget exists to check for this key.

## 4. Meaning and composition

The sentence names the owned condition exactly: the hosting platform is not configured. It adds no remedy, no path, no `config.yml` instruction, no provider name and no "none" qualifier — the surrounding refusal frame supplies those, and the sidecar explicitly forbids restating them here. Read after the frame's colon it is a complete statement:

`… यह रूट एक शुरू नहीं कर सकता: होस्टिंग प्लेटफ़ॉर्म कॉन्फ़िगर नहीं है। उस रूट के लिए …`

The sentence carries its own full stop and ends the clause before the frame's remedy sentence, which is what the sidecar asks for ("must read as a complete statement, so keep its terminal period"). Hindi has no letter case, so the English capitalisation requirement is satisfied by starting with the term.

## 5. Terminology (judged against the actual corpus, not uniform wording)

| corpus key | shipped `hi` | term |
|---|---|---|
| `cli.global_help.hosting` | `उपयोग करने के लिए होस्टिंग प्लेटफ़ॉर्म (…)` | exact `होस्टिंग प्लेटफ़ॉर्म` |
| `cli.exec_worker_help.hosting` | `होस्टिंग प्लेटफ़ॉर्म ओवरराइड` | exact |
| `cli.credential_help.credential` | `विभिन्न होस्टिंग प्लेटफ़ॉर्म के लिए …` | exact |
| `wire.settings.hosting.label` | `डिफ़ॉल्ट प्रोवाइडर` | *provider* is that field's noun |
| `wire.errors.desktop.aida_no_provider` | `अभी कोई प्रोवाइडर कॉन्फ़िगर नहीं है, …` | same predicate shape |

`होस्टिंग प्लेटफ़ॉर्म` is byte-identical to three already-shipped `hi` strings (NFC, same nukta order), so this is the settled repository term, not an invention. `कॉन्फ़िगर नहीं है।` reproduces the corpus's own state pattern. The `vi` file's choice of a provider word instead of "platform" is not a reason to change `hi`: the English subject here is the platform, and this corpus has a dedicated term for it. Using `प्रोवाइडर` would narrow the stated cause to a different noun. No finding.

## 6. Advisories (non-gating, recorded for the manager)

- **Glossary proposal:** no `i18n/glossary.md` row exists for `hosting platform`; the author proposes `होस्टिंग प्लेटफ़ॉर्म`. The evidence above supports it, but adding the row is a repo change outside this packet.
- **Legacy exception stays English by contract.** `session_factory.HostingNotConfiguredError.__str__` is not this key's consumer text; this packet neither edits it nor claims it is translated. I did not read the Python module (out of scope, and no code read is needed to judge this display string); the sidecar states the split, and the display key is the one audited.
- **No frame clearance:** other copy in the SDK refusal frame has its own audits; this verdict covers one key.

## 7. Native / renderer holds (separate, not claimed)

- Rendered terminal or browser output: **NOT RUN**. This is a source-only verdict; it makes no pixel, width, shaping or real-fit claim.
- Pseudo-localisation: **NOT RUN** (no generator, none permitted here).
- RTL: **N/A** — `hi` is LTR; no bidi controls present.
- Width: **N/A** (no `maxCells`/`maxChars` in the sidecar); the value is 41 codepoints and composes into a sentence, not a fixed-width cell.

## 8. Not run, and nothing written

No process execution of any kind: no bash, subprocess, tests, catalogue validator, generator, formatter, build, install, app, native render, CI or git. No repo, ledger, generated key, baseline or caller edit; the target was read only. Only `read`/`grep`/`glob` and pure-stdlib `eval` (`json`, `hashlib`, `unicodedata`) were used, and only for hashes, parity and Unicode checks. The finished TUI R1/R2 records were not opened for modification and remain untouched.
