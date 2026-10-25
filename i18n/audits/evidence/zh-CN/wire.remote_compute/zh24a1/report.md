# Independent translation audit — zh-CN / wire.remote_compute / zh24a1

- Round: 1
- Audited revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10` (branch `i18n/zh-cn-core-rebased`)
- Auditor: translation-auditor (lopdev team), session `0c7aebe462e8`, served model `deepseek/deepseek-flash`; no fallback observed. Independent of the translation: these catalogues were not authored in this session and were not edited by this audit.
- Keys read: 43 of 43 English keys (full namespace, every value read in context)
- Key parity: 0 missing, 0 extra; key order: identical
- Verdict: **clean-with-findings**
- Findings: 5 minor, 0 major, 0 blocker

## Method

String-level audit against the English source and the English context sidecar, using the repository's own ICU runtime (`local_operator.i18n.runtime`) as the renderer of record. Mechanical checks were executed, not eyeballed: key parity and order; NFC; control, bidi and zero-width code points; full-width Latin/digit homoglyphs; the style guide's no-space-between-CJK-and-Latin rule; ICU argument names and kinds; plural selector categories and `#` binding; verbatim survival of flags, commands, env vars, URLs, paths and product names; and display-cell widths against every sidecar `maxCells`.

Every value was then read semantically against its context note: what the control does, who the actor is, what the surrounding labels say, and whether a native speaker would write the sentence. Candidate findings were cross-checked against the five shipped locales and against the existing zh-CN catalogues before being recorded, so a divergence that is this locale's established practice is reported as an observation, not a defect.

## Mechanical results (executed)

- Key parity: 0 missing, 0 extra; key order: identical
- Non-NFC values: 0; control chars: 0; bidi controls: 0; zero-width: 0; full-width Latin/digits: 0; CJK↔Latin spaces: 0
- ICU argument mismatches: 0; plural-structure errors: 0; select arguments present: 0
- Technical tokens dropped: 0
- Width: 0 budgeted key(s), 0 over budget
- Render: 43 render sample(s), 0 failure(s)

## Findings

### ZH-08 — minor — `wire.remote_compute.needs_prompt`

- Current: `--remote需要提示词：将任务作为唯一的定位参数传入。`
- Proposed: `--remote需要提示词：将任务作为唯一的位置参数传入。`
- Reason: 位置参数 is the established term in this locale (cli.send_help.message: "单个位置参数就是消息") and the standard Chinese term for a positional argument; 定位参数 reads as a "locating parameter" and names no concept a CLI user knows.

### ZH-09 — minor — `wire.remote_compute.api_override`

- Current: `由于设置了LOP_REMOTE_COMPUTE_API_URL，远程计算正将你的Radient登录信息和请求发送到{host}，而不是默认的Radient API。除非你控制该主机，否则取消该设置。`
- Proposed: `由于设置了LOP_REMOTE_COMPUTE_API_URL，远程计算正将你的Radient登录信息和请求发送到{host}，而不是默认的Radient API。除非你控制该主机，否则取消设置该变量。`
- Reason: "Unset it" unsets the environment variable named in the previous sentence; 取消该设置 is vague about the object and uses 取消 (this locale's word for "cancel") for an unset, while every shipped locale names the variable (es "Quita esa variable", fr "Supprimez cette variable", ru "Снимите эту переменную").

### ZH-10 — minor — `wire.remote_compute.bad_sha`

- Current: `--sha值必须是完整的40字符提交SHA：分支名或缩写哈希可能解析为与本次运行提交时不同的提交。`
- Proposed: `--sha值必须是完整的40字符提交SHA：分支名或缩写哈希可能解析成另一个提交，而不是本次运行提交时指定的那个提交。`
- Reason: The English says the resolved commit can differ from the one this run was submitted for; the current clause strings 与…提交时不同 together in a way that has no clean parse, and it drops the "the one … submitted for" comparison that carries the warning (es "del que se envió esta ejecución", fr "celui soumis").

### ZH-11 — minor — `wire.remote_compute.retrying`

- Current: `Radient报告此运行出错并正在重试，因此继续监视。`
- Proposed: `Radient报告此运行出错并正在重试，因此本次监视继续。`
- Reason: "so this watch continues" is a statement about the running command; subjectless 继续监视 reads as an instruction to the reader. The same namespace states the fact with a subject elsewhere (仍在监视, 本命令已停止监视), and every shipped locale keeps the third person (fr "le suivi continue", ru "слежение продолжается").

### ZH-12 — minor — `wire.remote_compute.finished_unknown, wire.remote_compute.status_unknown`

- Current: `远程运行{run_id}已不再运行，但Radient未报告此版本能够识别的结果。 / 状态：{status}（此版本不认识的状态；仍在监视）`
- Proposed: `远程运行{run_id}已不再运行，但Radient未报告此构建能够识别的结果。 / 状态：{status}（此构建不认识的状态；仍在监视）`
- Reason: The same English noun "build" appears three times in this namespace and is rendered 构建 once (unreadable_envelope: 此lop构建) and 版本 twice; this locale's own CLI copy uses 构建 for a build and 版本 for a version (cli.services_help.status_description distinguishes 同版本重建/版本号 from 构建), and every shipped locale uses one word throughout (es versión, ru сборка). Unify on 构建.

## Observations (checked, no action requested)

- All 43 values are CLI runtime messages; flags, env vars, `git@host:path`, loopback hosts, `MiB`-class units and the `lop exec --remote-*` command lines survive byte-identically, and all 28 ICU argument occurrences across 23 keys ({run_id}, {network_id}, {idempotency_key}, {max_bytes} twice, {status}, {host}, {flag}, …) are present with the same names and kinds as English.
- remote compute → 远程计算 and remote run → 远程运行 are uniform across all 43 values and agree with cli.exec_remote_help; the relay vocabulary (中继) does not appear here.
- idempotency key → 幂等键 is uniform (5 values); it is an auth-adjacent technical term rather than an LLM token, so the glossary's token/令牌 split does not apply.
- Retry/admission vocabulary is consistent: 接纳 for admitted, 额度 for compute credit, 容量 for capacity.

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
- `wire.remote_compute.pairs.txt`
- `wire.remote_compute.compact.txt`
- `audit.py.txt`
