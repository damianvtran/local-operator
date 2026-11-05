# Vietnamese (vi) translation audit — `wire.errors.sdk_startup` — independent reader 2

- Audit ID: `vi-sdkstartup-r2-20261103`
- Locale: `vi` · Namespace (file group): `wire.errors.sdk_startup` · Keys: 4
- Date: 2026-11-03
- Auditor: agent `i18n-auditor`, session `7188e3347110`
- Model: requested/inherited `radient/auto` (session header). Served model id is not exposed to this session: **UNKNOWN**. No fallback was configured or observed by me.
- Independence: translator/author session is `15986c0af719`; this audit ran in `7188e3347110`. I did not read any other reader's report before forming this verdict, and I produced no translation text myself.
- Tool restriction honoured: `read`/`grep`/`glob`/`write` plus stdlib-only `eval` (hashlib/json/unicode/os). **Not run:** the repo i18n validator, any project import, any subprocess, tests, native rendering. Mechanical checks below were performed independently in Python; none of it is validator output.

## Scope

Binding: `~/.local-operator/sessions/3ef6295a5545/scratchpad/vi-startup-remedy-binding.json`. Delta = one key, `wire.errors.sdk_startup.attach_hosting_unconfigured`; the other three values are direct neighbours whose identity is proven, not re-audited as a corpus.

## 1. Input integrity (all four hashes match the binding)

| Input | SHA-256 | Binding | Match |
|---|---|---|---|
| EN source `catalogues/en/wire.errors.sdk_startup.json` | `5390ecb2…f9ca6bd` | `5390ecb2…f9ca6bd` | yes |
| EN context `…sdk_startup.context.json` | `82f193fd…3369bb13e` | same | yes |
| Target `…/sdk-startup-vi-remedy-fix/target/vi/wire.errors.sdk_startup.json` | `7c9a6bfb…c8c99ef95e` | same | yes |
| Baseline vi in worktree | `ac685815…4300abd` | same | yes |

All four decode as UTF-8 and parse as JSON (4 keys each; key sets identical EN/new, old/new).

## 2. Mechanical checks — all 4 keys

| Key | Placeholders EN → new | Braces | Apostrophe escaping | NFC | Ctrl/ZW/double-space | Chars EN/old/new |
|---|---|---|---|---|---|---|
| `attach_hosting_unconfigured` | `{resume}`,`{error}` → identical | balanced, no nesting, no `plural`/`select`/`#` | 0 / 0 | yes | none | 210 / 247 / 252 |
| `attach_runtime_missing` | `{resume}` → identical | balanced | 4 / 4 (`mode=''own''` preserved) | yes | none | 173 / 189 / 189 |
| `profile_attach_failed` | `{profile}` → identical | balanced | 0 / 0 | yes | none | 34 / 30 / 30 |
| `tool_inventory_mismatch` | `{spec_tools}`,`{policy_tools}` → identical | balanced | EN 2 / vi 0 (possessive recast, no ICU text dropped) | yes | none | 123 / 140 / 140 |

No plural/select messages exist in this namespace, so CLDR category completeness is vacuous here; no ICU structure was altered by the delta. Backticked code, `config.yml`, `spec.tools`, `ApprovalPolicy.declared(...)`, `spawn_session()/deliver()`, `mode=''own''`, `spec_tools`, `policy_tools`, `id`, `resume`, `error`, `profile` are byte-identical to the baseline.

## 3. Delta proof

Whole-file level, not just per-key: `new_file == old_file.replace("Hãy cấu hình hosting", "Hãy cấu hình nhà cung cấp")` is **True**, and the needle occurs exactly once in the baseline. So the only textual change in the file is that one literal substitution; every other byte, including all three sibling values, is identical. Per-codepoint diff of the key's value shows a single contiguous edit `hosting → nhà cung cấp`. Residual English `hosting` in the value: **none** (the remaining match in the file is the key *name*, a technical identifier that must stay exact).

## 4. Accuracy and native wording of the change

`Hãy cấu hình nhà cung cấp cho thư mục gốc này (config.yml của nó), rồi kích hoạt id bằng spawn_session()/deliver() và attach khi đã có runtime.`

- Meaning is preserved: the refusal reason, the remedy target ("this root", "its config.yml"), and the follow-up sequence (`engage` → `attach once a runtime exists`) are unchanged. No negation/polarity or scope drift; the causal colon and the `{error}` slot keep the same position and spacing.
- Register and idiom: `Hãy …` imperative matches the sibling remedy in `attach_runtime_missing` (`hãy mở nó …`); lowercase diagnostic opening before the colon is retained; punctuation is ASCII `,` `:` `;` `(...)` as in the rest of the vi corpus (no French-style spacing, correct for Vietnamese).
- Terminology is the house term, not a fresh coinage: `nhà cung cấp` is used in 74 vi values, and the product's own `hosting` config axis is localized exactly this way — `wire.settings.hosting.label` = "Nhà cung cấp mặc định", `wire.settings.hosting.help` = "Nhà cung cấp cho các cuộc trò chuyện mới", `cli.global_help.hosting` = "Nhà cung cấp sẽ dùng (…)"; `cli.exec_worker_help.hosting` = "Ghi đè nhà cung cấp". The identical collocation `Cấu hình nhà cung cấp …` already exists (`wire.slash.search` = "Cấu hình nhà cung cấp tìm kiếm web và cân bằng tải"). The rejected alternatives are unused in the corpus: `dịch vụ lưu trữ` 0 values, `nhà cung cấp dịch vụ` 0 values — so keeping English `hosting` or inventing a periphrastic form would both have been glossary/consistency violations.

### Exception classification, verified rather than asserted

The operator rule is "no English in supported UI except exact technical identifiers/params". Classifying the remaining Latin tokens as legitimate technical exceptions is correct on evidence, not convenience:

| Token | Values using it | Classification |
|---|---|---|
| `runtime` | 32 | settled kept term; `môi trường chạy` appears in 0 values, so a Vietnamese substitution would be a corpus break |
| `attach` | 7 (incl. this file) | API action name, same class as `spawn_session()`/`deliver()` |
| `id`, `config.yml`, `spec.tools`, `ApprovalPolicy.declared(...)`, `spec_tools`, `policy_tools`, `mode=''own''`, `resume`, `error`, `profile` | params/paths/code | exact params and code, must stay |
| `hosting` | 3 values after this fix: `cli.config_help.edit_key` (config-key enumeration), `wire.errors.sdk_spec` ×2 (`hosting/model` compound config-key pair) | defensible as config-key tokens; **not** the prose usage this delta removes |

So `attach_hosting_unconfigured` is a genuine removal (human remedy prose), not an over-broad purge that would have damaged a config key or parameter.

## 5. Fit

The context sidecar carries `role`/`tone`/`note` only — **no `maxCells`/`maxChars` hint exists** for these keys. The string is a multi-line refusal message, not a button/tab/status slot; display length grew 247 → 252 chars (+5), bytes +8. Rendered width, wrapping and terminal cell behaviour were **not** measured (no rendering tool in scope), so I make no claim about pixels or overflow.

## 6. Findings

- **F1 — NIT (pre-existing, unchanged by this delta, non-gating).** `wire.errors.sdk_startup.tool_inventory_mismatch`: `… nêu các danh mục khác nhau; hãy nêu phạm vi của lần chạy một lần duy nhất`. Trailing `một lần duy nhất` sits next to `lần chạy`, so it can be parsed as "the single run" instead of EN "state the run's reach once". Suggested wording: `hãy nêu phạm vi của lần chạy đúng một lần`. Outside the delta; belongs to any future pass on this group, not a gate on this change.

No BLOCKER, no MAJOR.

## 7. Verdict

**`passed`** for `vi` / `wire.errors.sdk_startup` on the audited head: the remedy no longer carries prose English `hosting`, the replacement is literal and exactly bounded, `nhà cung cấp` is the corpus-sanctioned rendering of the `hosting` axis, ICU arg names/types, quoting and every technical token are intact, and no structural risk was introduced. Unresolved blocker/major: 0. NIT F1 recorded above is pre-existing and non-gating.

Not exercised (explicit holds): repo validator, ICU runtime rendering, pseudo-locale expansion, RTL (not applicable to vi), terminal-width/rendering fit, git range-diff CLI, ledger write. No repo or ledger file was edited by this audit.
