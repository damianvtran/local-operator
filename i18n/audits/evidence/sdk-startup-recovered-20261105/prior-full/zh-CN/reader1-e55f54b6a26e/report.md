# Independent zh-CN audit — wire.errors.sdk_startup (reader 1)

**Verdict: passed** — 4/4 keys, no BLOCKER/MAJOR. Two MINOR wording findings (F1, F2), both actionable, neither gating.

Auditor `i18n-auditor`, session `e55f54b6a26e`, exposed served model label **`radient/auto`** from my own completed `payload.usage` (provider `radient`, `model_id` `auto`). That label is an auto-router label: I do not claim a concrete upstream model, and I assert no fallback event for this turn. Translator session `7fce46e29bee`; coder `eae0961e4e78`. I did not translate these strings and read no other new reader's report before recording this verdict.

Scope: staged catalogue only, source pin `0e7cde33ca8008d561e74f4aed774aa5f76ff7d5`. No repo edit, ledger, generation, activation, source-factory/API change, PR, build, install, scanner or full suite.

## Inputs (verified, not taken from narrative)

Canonical manifest: 13/13 hashes recomputed and matched. EN, context, style guide and glossary snapshots equal the pinned git blobs byte-for-byte. zh-CN target is absent at the pin (staged only) — verified with `git cat-file -e`.

- en `5390ecb2413e0978c34541a9a0db6c2a004ae86446d1c0fa0d2e697e4f9ca6bd`
- context `ee0bbde5e99de22c04368e6bbfedf913d758ef6ba5075984ddf7d795b1ee7822`
- zh-CN target `fcff2fac99196a9e4dbd03957fb959c2af69013b1471532f819ddbc705293e02`

## This namespace is not sdk_spec (checked, not assumed)

Pinned `sdk.py:584-598` `startup_refusal_text` reads the config's `language`, resolves it (`resolve_language`, default `en`) and renders that locale — it does NOT force `locale="en"`. All four refusals route through it (`sdk.py` 621-646 hosting, 664-676 runtime-missing after `dispose`, 667-673 profile, 686-695 inventory). Executed read-only: `resolve_language("zh-CN") -> zh-CN`, `resolve_language("EN") -> en`, `resolve_language("en") -> en`. So a root really configured `zh-CN` reaches this catalogue with no producer change, and the default English path is unchanged (executed control: locale `en` renders the English source unchanged for all 4 keys). The historical 21-producer English-pinned `SessionSpecError` contract belongs to `sdk_spec`; I did not apply it here.

## Mechanical (executed, isolated HOME, 256 MiB / 90 s)

Exact 4 keys, English order, none added/dropped; argument tuples identical to source (`resume`+`error`; `resume`; `profile`; `spec_tools`+`policy_tools`), all `simple`; no plural/select arms (zh-CN needs none here); `Cf=0`; no manual bidi controls; no `请` where English has no "please"; no width caps in context. Opaque values (`{braces}`, quotes, `%`, `<tag>`, backslash, newline, bidi) pass through byte-identically and are never re-parsed. `mode=''own''` renders `mode='own'`. Negative controls: malformed ICU rejected; added placeholder rejected; missing argument refused (`MessageFormatError`); plural-shape change rejected.

## Per-key verdict

1. `attach_hosting_unconfigured` — **pass**. Facts and remedy preserved: no live runtime serving `{resume}`, this root cannot start one, `{error}` opaque, configure hosting via `config.yml`, then engage the id, attach once a runtime exists. remedy order and technical tokens (`config.yml`, `spawn_session()/deliver()`, `ID`) intact. F2 wording note below.
2. `attach_runtime_missing` — **pass**. Viewer/own distinction preserved (`mode='own'` in-process), engage-then-attach order preserved, disposal lifecycle untouched (catalogue-only).
3. `profile_attach_failed` — **pass**. Faithful, terse; `{profile}` repr-quoting untouched. F1 term note below.
4. `tool_inventory_mismatch` — **pass**. “different inventories” and the state-it-once constraint preserved; `spec.tools`, `ApprovalPolicy.declared(...)`, both tuple placeholders intact; `范围` for “reach” is acceptable.

## Findings

### F1 — MINOR — `附接` is a coinage; the carried corpus says `附加`
Keys: `attach_hosting_unconfigured` (“…再附接”), `attach_runtime_missing` (“…再附接”), `profile_attach_failed` (“无法附接配置档{profile}”).
Measured on the pinned zh-CN catalogues: `附接` **0 occurrences / 0 files**; `附加` **10 occurrences / 8 files**, used for exactly this verb in `cli.exec_help.*` (“附加可复用角色/专家”, “附加已保存的团队”) and `cli.global_help.agent`. The translation is internally consistent (all three sites use the same word) but introduces a second term for an operation the CLI already names. Understandable, not misleading → MINOR, not a register failure.
Suggested: `附加` — “待运行时存在后再附加”, “无法附加配置档{profile}”.

### F2 — MINOR — `启动` carries both “start a runtime” and “engage the id”
`attach_hosting_unconfigured`: “且此根目录无法启动一个：{error}…然后用spawn_session()/deliver()启动该ID…” — the same verb is used for what the root cannot do and for the remedy, so a native reader can trip on “cannot start one … then start the id”. English keeps “start one” and “engage the id” distinct. `attach_runtime_missing` uses `启动` for “engage it” alone, where it is less exposed but still conflates spawn and deliver.
Suggested (one line, either fix): rename the first clause to “且此根目录无法新建一个运行时”, keeping `启动` for engage; or keep the first clause and render engage as “驱动该ID”.

## Cross-packet alignment notes (not findings against this packet)

- `profile` term: this packet’s `配置档` matches the corpus (`配置档` 9 occurrences / 4 files; `配置档案` 0). The already-audited `wire.errors.sdk_spec` uses `配置档案` once — **my earlier sdk_spec pass accepted that without a corpus count; that was a miss**. Alignment belongs to the single remediation wave, with `配置档` as the corpus-backed form.
- `sdk_spec` uses the same `附接` coinage, so F1 should be fixed in both namespaces in that one wave.
- **Embedded English (source-side, not fixable here):** `{error}` is `str(error)` of `HostingNotConfiguredError("Hosting platform is not configured.")` raised at pinned `session_factory.py:816`. A zh-CN user will therefore read one English clause inside this otherwise-Chinese refusal. Preserving the placeholder is correct; making the whole refusal Chinese needs a new owned message plus a producer change, which this packet does not make and must not be waived. This key alone does not meet the operator’s “no English anywhere” bar.
- Next-sentence boundary: English relies on the opaque error’s own terminal period, and Chinese continues with `为` after that Latin `.` with no space — which the zh-CN guide requires (no space between CJK and Latin runs). I considered appending `。` and rejected it: it would double the punctuation for this producer and the placeholder contract forbids assuming the error’s punctuation. Current form is the best catalogue-only option; residual risk recorded.

## Evidence produced vs not run

Produced: canonical hash verification, pinned source/caller inspection, scoped parser/render/argument checks with four negative controls, a default-English 4/4 rendering control, `resolve_language` execution, corpus term counts.
**NOT RUN:** rendered native fit (no width caps exist; no app/TUI frame), pseudo-localization, RTL (not applicable to zh-CN), live exception text, production `startup_refusal_text` execution (it is a closure inside `create_session`; I executed its locale-resolution function, not the closure), desktop/mobile sink audit (the architect assessment says no current sink; I did not re-derive it), second independent reader, activation or global UI clearance. Passing linguistic audit is not release eligibility.
