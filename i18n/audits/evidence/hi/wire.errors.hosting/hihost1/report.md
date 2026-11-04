# hi · `wire.errors.hosting` · independent audit (reader 1)

**Verdict: PASS — 1/1 key, 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT, no advisory changes requested.**

Key: `wire.errors.hosting.not_configured`
EN: `Hosting platform is not configured.`
HI: **`होस्टिंग प्लेटफ़ॉर्म कॉन्फ़िगर नहीं है।`**
Target sha256 `22989094e5c759832fbd1e22220b68d0b5a7165c9dca9df6690b587d5add8f2e` (156 B) — matches the binding.

## Identity and provenance

Auditor `i18n-auditor`, session `b858f6df97a0`; requested route `radient/auto`, **served model UNKNOWN**
(the session-state header exposes only `radient/auto`; nothing inferred, no fallback claimed).
Independent of translator session `b60b4bdc8c47` and of code author `94dcef624dec`; no translation/MT
service used; no self-translation; no repo, target, ledger, generated or baseline edit. I did not read
any other reader's verdict on this key.

Hashes computed here with stdlib `hashlib` over the exact bytes:

| input | path | sha256 | bytes |
|---|---|---|---|
| English | `…/hosting-source-v1/wire.errors.hosting.json` | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` | 82 |
| context | `…/hosting-source-v1/wire.errors.hosting.context.json` | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` | 854 |
| target (hi) | `…/hosting-hi/target/hi/wire.errors.hosting.json` | `22989094e5c759832fbd1e22220b68d0b5a7165c9dca9df6690b587d5add8f2e` | 156 |

Namespace state: new English namespace, **uncommitted and absent at base `87ca846245b2`, not shipped** —
taken from the dispatch binding; not re-verified because Git/process execution is out of bounds for this
audit. No activation or shipped-state claim is made.

## What the copy has to do (context read)

Role: owned cause clause composed into startup refusals. Tone: diagnostic; **names the condition, never
restates the remedy**. The preflight resolved no provider at all, the surrounding refusal supplies the
remedy, and the clause is spliced mid-sentence after a colon, so it must read as a **complete statement
with its terminal period**. No placeholders, no dynamic detail. The English sentence is also the legacy
`session_factory.HostingNotConfiguredError` text, which stays English by contract and is **not** this
key's consumer text (verified in the tree at `session_factory.py:816`; untouched).

## Audit against those requirements

- **Condition named, not an opaque external error:** the string says the hosting platform is not
  configured — the owned preflight condition. Nothing about an unknown failure, no exception class, no
  path. PASS.
- **No added remedy or detail:** no "configure it", no `config.yml`, no provider list, no "no provider
  was found" cause detail beyond what English carries. PASS.
- **Complete statement after a colon:** `… : होस्टिंग प्लेटफ़ॉर्म कॉन्फ़िगर नहीं है।` composes cleanly into
  the already-shipped frame `wire.errors.sdk_startup.attach_hosting_unconfigured` (“…यह रूट एक शुरू नहीं
  कर सकता: {error} इस रूट के लिए होस्टिंग कॉन्फ़िगर करें…”), which then starts a fresh sentence. PASS.
- **Locale punctuation and case:** terminal **danda U+0964**, not an ASCII period; four single spaces, no
  double or edge spaces; Hindi has no letter case, so the English capitalisation requirement is met by
  beginning with the term. PASS.
- **Terminology, judged against the corpus (not uniform wording):** `होस्टिंग प्लेटफ़ॉर्म` is the repository's
  settled Hindi rendering of the concept — verified here in `cli.global_help.hosting` (“उपयोग करने के लिए
  होस्टिंग प्लेटफ़ॉर्म…”), `cli.exec_worker_help.hosting` (“होस्टिंग प्लेटफ़ॉर्म ओवरराइड”) and
  `cli.credential_help.credential` (“विभिन्न होस्टिंग प्लेटफ़ॉर्म के लिए…”). `कॉन्फ़िगर नहीं है` mirrors the
  existing error pattern exactly (`wire.errors.media.radient_key_missing`: “सर्वर पर Radient API कुंजी
  कॉन्फ़िगर नहीं है।”). `wire.settings.hosting.label` renders the *setting field* as “डिफ़ॉल्ट प्रोवाइडर”, a
  different noun for a different thing; the English source itself says “platform”, so forcing “प्रोवाइडर”
  here would diverge from the source and from the corpus. PASS.
- **No English fallback:** no Latin characters at all; `होस्टिंग`, `प्लेटफ़ॉर्म`, `कॉन्फ़िगर` are settled
  Devanagari loans, which the hi guide's loan rule allows. PASS.
- **Native idiom:** one plain diagnostic clause, polite/neutral register, no honorifics, no exclamation,
  no calque word order. PASS.

## Mechanical

One key in English, context and target, identical key ID; valid JSON (two-space indent, trailing newline,
no BOM). **No ICU arguments, no markup, no code spans** — so placeholder identity and CLDR plural
categories are **N/A with reason** (a literal sentence with zero args; no `{…}` present). The string is
NFC- and NFD-stable; **0 `Cf`**, **0 bidi controls**, 0 ZWJ/ZWNJ.

## Holds kept separate (nothing claimed here)

- **Native / rendered consumer proof: NOT RUN** — held for the separate consumer/renderer proof.
- **Pseudo-localisation: NOT RUN**; **RTL: N/A** (hi is LTR).
- **Formatter / cell measurement: not applicable and NOT RUN** — the sidecar carries no `maxCells`; this
  audit makes no width claim.
- **Repo validator / scanner / generation / tests / CI: NOT RUN** (forbidden by the brief).
- **Activation / shipped state: not claimed** (uncommitted namespace, absent at base per the binding).

## Advisories (no copy change requested)

1. **Consumer wiring, caller-owned:** the display path must pass this keyed sentence, not
   `str(HostingNotConfiguredError)` — the exception's message is English by contract. Caller/API changes
   are outside this audit's permission; flagged only so the English-free acceptance is not claimed from
   the key alone.
2. **Glossary:** the author proposes a row `hosting platform → होस्टिंग प्लेटफ़ॉर्म`; the corpus already
   supports it, but recording it is a glossary decision for the manager, not this audit.
3. **Provenance:** source-side namespace absence at base and uncommitted status rest on the dispatch
   binding; this audit executed no Git.

## Preserved

The completed `tui.settings_chrome` R1 and R2 records, reports and companions were not touched. No PR
comment, activation, ledger or frame clearance.
