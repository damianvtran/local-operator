# Independent translation audit — zh-CN / cli.exec_remote_help / zh24a1

- Round: 1
- Audited revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (branch `i18n/zh-cn-core-rebased`)
- Auditor: translation-auditor (lopdev team), session `0c7aebe462e8`, served model `deepseek/deepseek-flash`; no fallback observed. Independent of the translation: these catalogues were not authored in this session and were not edited by this audit.
- Keys read: 11 of 11 English keys (full namespace, every value read in context)
- Key parity: 0 missing, 0 extra; key order: identical
- Verdict: **clean-with-findings**
- Findings: 1 minor, 0 major, 0 blocker

## Method

String-level audit against the English source and the English context sidecar, using the repository's own ICU runtime (`local_operator.i18n.runtime`) as the renderer of record. Mechanical checks were executed, not eyeballed: key parity and order; NFC; control, bidi and zero-width code points; full-width Latin/digit homoglyphs; the style guide's no-space-between-CJK-and-Latin rule; ICU argument names and kinds; plural selector categories and `#` binding; verbatim survival of flags, commands, env vars, URLs, paths and product names; and display-cell widths against every sidecar `maxCells`.

Every value was then read semantically against its context note: what the control does, who the actor is, what the surrounding labels say, and whether a native speaker would write the sentence. Candidate findings were cross-checked against the five shipped locales and against the existing zh-CN catalogues before being recorded, so a divergence that is this locale's established practice is reported as an observation, not a defect.

## Mechanical results (executed)

- Key parity: 0 missing, 0 extra; key order: identical
- Non-NFC values: 0; control chars: 0; bidi controls: 0; zero-width: 0; full-width Latin/digits: 0; CJK↔Latin spaces: 0
- ICU argument mismatches: 0; plural-structure errors: 0; select arguments present: 0
- Technical tokens dropped: 0
- Width: 0 budgeted key(s), 0 over budget
- Render: 11 render sample(s), 0 failure(s)

## Findings

### ZH-01 — minor — `cli.exec_remote_help.idempotency_key`

- Current: `复用此幂等键而不生成新键，使已提交运行的重试不会运行两次`
- Proposed: `要复用的幂等键（而非新生成的键），以免已提交的运行在重试时执行两次`
- Reason: argparse value help: the English names the value the flag takes ("Idempotency key to reuse…") and every shipped locale keeps that noun phrase (es "Clave de idempotencia que reutilizar…", fr "Clé d’idempotence à réutiliser…"); the Chinese opens with an imperative ("复用此幂等键…"), so it tells the reader what to do instead of describing the argument.

## Observations (checked, no action requested)

- All 11 values are `--remote` argparse help. Flags (`--remote`, `--repo`, `--sha`, `--idempotency-key`), `Git`, `URL`, `JSON` and the `s`/`m`/`l` value set survive verbatim; no ICU arguments exist in the namespace and none was introduced.
- 第1档 for "tier 1" matches this locale's established 档/档位 rendering for tier (tui.authoring.effort 模型档位, wire.settings.subagents.description 各档位).
- 服务器…准入默认值 for "the server's admission default" is jargon in English too and is corroborated by fr "valeur d’admission du serveur" and ru "значение допуска сервера"; left unchanged.

## Caveats

- This is a string-level audit. No native terminal, relay SPA, Electron or rendered-frame pass was performed here, so truncation in a real paint, font-cell width and interaction behaviour are not claimed.
- zh-CN remains in `SECURITY_HELD_LOCALES` (`scripts/i18n/generate.py`) and out of `data/shipped.json`; this record changes no eligibility, ledger row or other locale.

## Evidence files

- `record.json`
- `receipt.json`
- `report.md`
- `audit.log`
- `mechanical-raw.json`
- `runtime-mechanical.json`
- `pseudo-evidence.json`
- `cli.exec_remote_help.pairs.txt`
- `cli.exec_remote_help.compact.txt`
- `audit.py.txt`
