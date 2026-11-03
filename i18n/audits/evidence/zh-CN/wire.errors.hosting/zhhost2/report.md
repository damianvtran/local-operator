# zh-CN `wire.errors.hosting` — independent audit (reader 2)

**Verdict: PASS.** One key, no BLOCKER, no MAJOR, no MINOR, no NIT. Two non-gating advisories are recorded below.

| key | English | zh-CN | cells |
|---|---|---|---|
| `wire.errors.hosting.not_configured` | Hosting platform is not configured. | **托管平台未配置。** | 16 |

## What was checked, and how

- **Mechanical, by reading the bytes**: one key in English, context and target, same key name and order; JSON parses; NFC-stable; **zero** `Cf` code points; no ASCII letters or digits; no braces, backticks or placeholders; no edge whitespace; terminal stop is U+3002.
- **Argument/plural structure**: the source and the target are both argument-free, so placeholder parity is vacuous and **CLDR categories are not applicable** — there is no `plural`/`select` in either file and nothing to enumerate for zh-CN. Stated as N/A with that reason rather than claimed as a passed plural check.
- **Condition meaning**: the clause names the condition and nothing else — no remedy, no host detail, no dynamic part. It matches the English one-for-one.
- **Composed use**: after the refusal's colon it reads as a complete statement (`…无法启动一个：托管平台未配置。`), which is what the sidecar requires; the terminal period is kept, and "capitalization" has no zh-CN counterpart, so no case marker was invented.
- **Locale typography**: full-width sentence stop, no half-width punctuation, no space before the stop, no space injected anywhere in an 8-character sentence.
- **No English fallback**: the value contains no Latin at all; it is not an English duplicate.
- **Width**: 16 cells, 8 characters. The context declares **no** `maxCells`, so there is no cap to exceed here; rendered fit remains a consumer/renderer question, not something this source pass can settle.

## Terminology, derived from the shipped corpus rather than from the packet

- `托管平台` for *hosting platform* is already shipped: `cli.global_help.hosting` (使用的托管平台), `cli.global_help.propagated_hosting` (本次运行的托管平台), `cli.exec_worker_help.hosting` (覆盖托管平台), and the same phrase in `cli.global_help.description`. The author's claim checks out; I re-derived it by searching the zh-CN catalogue tree myself.
- `未配置` as the predicate for "not configured" is also shipped: `tui.authoring.invalid_effort` ("该模型档位未配置。") and `wire.errors.media.radient_key_missing` ("服务器未配置Radient API密钥。").
- The subject-first shape `托管平台未配置。` follows `tui.authoring.invalid_effort`'s clause shape, so it is corpus-consistent native copy, not a calque.
- Provider versus platform stays distinct in zh-CN exactly as the corpus has it (`提供商` for provider in `wire.errors.desktop.aida_no_provider`; `托管平台` here). No uniform-wording assumption was applied across locales.

## Consumer evidence (why this clause is display copy, and what it does not claim)

In the owned worktree the refusal composes this key rather than splicing the exception text: `startup_cause_text` loads the **English** catalogue value for the message, and returns `render(message.code, message.params, locale=startup_refusal_locale())` **only** when `type(error) is HostingNotConfiguredError` **and** `str(error)` equals that English source exactly; every other exception, including a subclass, passes its own text through untouched (`sdk.py:603-632`). Two consequences follow, and both are consistent with the brief:

1. The ownership test reads the **English** catalogue, so translating this value cannot break or weaken the ownership check.
2. `HostingNotConfiguredError`'s own `str`/`args` remain English by contract; this key is display copy for the owned condition, not a localisation of the exception, and only exact-type matches on this path ever reach it.

## Advisories (non-gating, no change requested to the target)

- **A1 — contract boundary, not a defect.** No claim is made that the legacy exception is localised, or that a refusal carrying a foreign/subclass error is English-free. That is the designed boundary in the consumer above; it is a consumer/contract fact for the delivery record, not something a zh-CN wording change can fix.
- **A2 — author-record typo only.** The author's rationale writes "休 adds a temporal 'yet'" where the character discussed is `尚` (`托管平台尚未配置。`). The delivered copy is unaffected; correct the record if it is quoted downstream.

## Evidence and holds

Computed here with stdlib `hashlib`/`json`/`unicodedata` only (no processes, no repo imports, no test or formatter runs):

| input | sha256 | derivation |
|---|---|---|
| English | `458d8f8be304428ccb6a328ce22e5a86515e9f44f869f0138a5cc6dfde964b63` | computed over the frozen file, and independently over the owned worktree copy — identical |
| context | `ed0ae50fa465a010a2907bc680771dae80bc716d3a186b0af37e31e0d7063cf9` | computed over the frozen file |
| zh-CN target | `b4cd49fd638d7aa7b9f6142a61d465918d69734c7c970025da9121f18907207e` | computed over the delivered file; matches the value the author reported |

Namespace state: new English namespace, uncommitted in the owned worktree and absent at base `87ca846245b2` — taken from the supplied binding, **not** independently verified here (this pass has no VCS access). Not shipped, and no shipped/activation claim is made.

**Not run, and not implied:** repo catalogue validator, scanner, generation, production formatter/renderer, native app, pseudo-localisation, RTL (N/A for zh-CN), CI. No rendered frame exists for this string in any locale. Native/UI fit and any English-free claim for the composed refusal remain with the consumer/renderer proof, not with this audit.

No repo, ledger, generated-key, baseline, source, context, caller or earlier-bundle edit was made; the completed TUI R1/R2 records and every prior SDK/policy/census bundle are untouched.
