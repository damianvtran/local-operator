# zh-CN `wire.errors.hosting` — independent audit, reader 1

**Verdict: passed** — 1/1 key, no findings (0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT).

Auditor `i18n-auditor`, session `e55f54b6a26e`; route/observed label **`radient/auto`** from my own completed `payload.usage` (`provider: radient`, `model_id: auto`); the upstream model behind that label is **UNKNOWN** and is not inferred; no fallback asserted. Translator session `7fce46e29bee`; code author `94dcef624dec` — both distinct from this reader. I did not translate this string, wrote no target bytes, and read no other reader's verdict before recording this one.

## Inputs (recomputed here)

| input | path | sha256 |
|---|---|---|
| English | hosting-source-v1/wire.errors.hosting.json | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` |
| context | hosting-source-v1/wire.errors.hosting.context.json | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` |
| zh-CN target | hosting-zh-CN/wire.errors.hosting.json | `b4cd49fd638d7aa7b9f6142a61d465918d69734c7c970025da9121f18907207e` |

All three match the values supplied to me. **Namespace absence at base `87ca846245b2` is supplied, not verified here** — this pass has no Git access; it is therefore treated as an uncommitted, unshipped candidate throughout.

## The string

| key | English | zh-CN |
|---|---|---|
| `wire.errors.hosting.not_configured` | Hosting platform is not configured. | **托管平台未配置。** |

## Mechanical (stdlib only, executed this pass)

Single key in all three files, identical key name and order; JSON parses; NFC-normalised; **no `Cf` characters**; no bidi controls; **no ASCII letters**, no braces, no placeholders; no edge whitespace; terminal character is U+3002 `。`; 8 characters / **16 cells**. The context entry declares `role`, `tone`, `note` and **no `maxCells`**, so there is no width budget to test against and none is claimed.

ICU/CLDR: **not applicable, with reason** — the message is a plain literal with zero arguments; there is no plural/select arm and no CLDR category to satisfy. Placeholder parity is vacuous (0 = 0).

## Native semantic audit

- **Condition only.** It names the state and stops. No remedy, no "check your config", no provider list, no dynamic detail — matching the sidecar's rule that the surrounding refusal supplies the remedy.
- **Reads as a complete statement after a colon.** `……无法启动一个：托管平台未配置。` — subject `托管平台`, predicate `未配置`, full stop. It does not depend on the frame's wording being restated, which is exactly what the note forbids.
- **Terminology from the shipped corpus, not invented.** `托管平台` is the established rendering for hosting platform (`cli.global_help.hosting` 使用的托管平台, `cli.global_help.propagated_hosting` 本次运行的托管平台, `cli.exec_worker_help.hosting` 覆盖托管平台); `未配置` is how the corpus states "not configured" (`wire.errors.media.radient_key_missing` 服务器未配置…, `tui.authoring.invalid_effort` 该模型档位未配置。). Subject-then-predicate mirrors the English topic-first clause and the corpus shape.
- **Punctuation and case.** Full-width `。` is correct for a full sentence (a short label would drop it; this is a statement, so it keeps it). Capitalisation of the English sentence-initial `H` has no Chinese counterpart; introducing no case marker is correct, and the sidecar's "keep its capitalization" requirement is English-specific rather than a zh-CN obligation.
- **Provider vs platform.** The English names the *platform*, and the zh target follows the source with `托管平台`. The Vietnamese packet's "provider" choice is a different locale's call against the same source; it is not evidence that zh-CN should drift, and the corpus settles zh-CN as platform.
- **No English fallback inside this value.** The value contains no Latin text. Separately and by contract, `HostingNotConfiguredError`'s own string stays English and the surrounding refusal frame is caller-owned; neither is claimed localized by this key, and neither was edited.

## Advisories and separate holds (not cleared here)

1. **Composed rendering with the calling frame: NOT RUN.** The colon splice was assessed by reading the frame's contract and the source, not by rendering a refusal.
2. **Pseudo-localization, native fit in the real terminal, rendered frames: NOT RUN** — separate consumer/renderer proof.
3. **RTL: not applicable to zh-CN.**
4. **Second independent reader: separate gate.** This pass is reader 1 only.
5. **Activation / shipped state / page-level English-free: NOT CLAIMED.** This is one display key in one uncommitted namespace.

## Provenance and constraint compliance

No translation written, no repo/target/ledger/generated/baseline edit, no Git, no process or subprocess, no test, scanner, formatter, project import, native run, install or CI. Tool use was reads plus pure-standard-library hashing/parsing (`hashlib`, `json`, `unicodedata`). Completed TUI R1/R2 records and earlier SDK/policy bundles were not touched.
