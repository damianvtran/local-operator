# Independent zh-CN audit — wire.slash — zh24s1

- Locale: `zh-CN` (held; not in `shipped.json`; the security hold is unchanged)
- Namespace: `wire.slash`
- Revision audited: `f286318c89053e5e00cbb7823f6f29cd6a06d205` (current bytes; catalogue blob `96198518bd7ec811904760f7e83770f230b275ce`)
- Catalogue bytes introduced at: `4840abdeb650a44db3bd9b1f564b24b6546c099a` (held candidate); carried and attested at `00b3446cf3df78805988e25862133419a235e181`; byte-identical at the audited HEAD
- Auditor: translation-auditor (lopdev team); session `4ecbbf4d3281`; actual model `deepseek/deepseek-flash`; fallback: none
- Independent of the translator: yes (the translator commit is `4840abdeb650a44db3bd9b1f564b24b6546c099a`, authored by Damian Tran; the translator subagent session/model are not recorded in it)
- Keys in namespace: **52**; keys read: **52**
- Delta keys: **0** — this is a full-namespace read of the carried bytes, not a delta-only pass
- Verdict: **clean_with_followups** (status `passed`); unresolved blocker/major: **0**; minor: **6**; nit: **1**

## The previously withheld key

The prior row withheld this namespace at 51/52 because `wire.slash.credential` could not be read under ordinary tool-output redaction, and the translator explicitly refused a redaction bypass (commit `4840abdeb650a44db3bd9b1f564b24b6546c099a`).

The key was read in this audit through the sanctioned public-copy route after the redaction-policy correction (`6bfac3fccb`, `docs/security/redaction-policy.md`): `execute_read_public_copy` served the committed HEAD blob, recorded the read before returning it, and wrote the audit log under the auditor's scratchpad config dir.

- Route digest (sha256 of the keyed JSON body): `0617711757169777c98ba763258cc09bc997914643a5602071cc66c082abc8dd`
- Route log: `/Users/damian/.local-operator/sessions/4ecbbf4d3281/scratchpad/zh-slash-audit/config/audit/public-copy-reads.jsonl` — sha256 `97a4f1e153a73868a672bdb6ed3a26a48317c5f4d20babfe5ad527be57784980`
- Bundle copy: `route-read-log.jsonl` (three recorded reads, all the same digest)

**This is a fresh independent audit of the current bytes, not an upgrade or relabel of the old failed row.** The old 51/52 row is not being converted in place; this bundle is a new audit id, `zh24s1`, with its own hashes and provenance.

### Faithfulness of the withheld key

- English: `Type or paste a secret after a space; masked`
- zh-CN: `空格后输入或粘贴密钥；显示为掩码`

The translation is faithful: 空格后输入或粘贴密钥 keeps the imperative ("type or paste a secret"), the space instruction, and 显示为掩码 preserves the "masked" guarantee; the semicolon is full-width, and 密钥 is the locale's established term for a secret/key value (`tui.welcome` "粘贴密钥", `wire.errors.auth` "API密钥", while stored credentials are 凭据 in `wire.slash.accounts`). No placeholder, plural or width issue. **No finding.**

## Findings

### ZH-S1 — minor — accuracy_and_glossary

- Key: `wire.slash.agent`
- Current: `列出智能体、切换其类别，或让本会话以智能体身份回复`
- Proposed: `列出智能体、切换其自主级别，或以智能体身份对本会话发言`
- Width: current 50 cells, proposed 54 cells, English 69 cells.
- Reason: 类别 is the generic word for category; the locale's established rendering for an agent's class is 自主级别 (tui.authoring.action_class, wire.settings.proactive.title). The second clause also shifts the actor: the English says the user speaks to the session as an agent, while the translation says the session replies as one.

### ZH-S2 — minor — glossary_consistency

- Key: `wire.slash.highlights`
- Current: `取消、重试或调整回答下方的补充内容`
- Proposed: `取消、重试或调整回答下方的亮点`
- Width: current 34 cells, proposed 30 cells, English 53 cells.
- Reason: The Highlights feature is rendered 亮点 in all nine wire.settings.supplements* keys; 补充内容 ('supplementary content') is a different term and creates glossary drift for the same English feature name.

### ZH-S3 — minor — glossary_consistency

- Key: `wire.slash.team`
- Current: `列出团队、查看团队组织图，或向其负责人发送请求`
- Proposed: `列出团队、查看团队组织图，或向其管理者发送请求`
- Width: current 46 cells, proposed 46 cells, English 69 cells.
- Reason: manager is 管理者 in the team surfaces (tui.small.org_chart_view.manager_legend, tui.team_authoring.manager, cli.exec_help.team); 负责人 is the locale's rendering for owner (tui.projects.form.owner_placeholder), so this is term drift.

### ZH-S4 — minor — glossary_consistency

- Key: `wire.slash.context`
- Current: `显示提示词、工具定义与消息的token用量`
- Proposed: `显示提示词、工具schema与消息的token用量`
- Width: current 37 cells, proposed 39 cells, English 48 cells.
- Reason: tool-schema is a token category; the locale keeps schema as English in wire.settings.tools_defer.help ('把罕用工具的schema...'), while 工具定义 renders tool definitions and drops the schema term.

### ZH-S5 — minor — glossary_consistency

- Key: `wire.slash.failovers`
- Current: `查看模型回退顺序和当前使用的模型`
- Proposed: `查看模型故障转移级联和当前提供服务的模型`
- Width: current 32 cells, proposed 40 cells, English 51 cells.
- Reason: failover cascade is 故障转移级联 in wire.settings.failover.title and wire.settings.retry_fallbackchains.label; 回退顺序 drops both the established term and 级联, and 当前使用的模型 is less precise than 当前提供服务的模型 for 'what is serving'.

### ZH-S6 — minor — glossary_consistency

- Key: `wire.slash.hub`
- Current: `打开Hub链接，设置各链接的subscribe和auto-push策略`
- Proposed: `打开Hub链接，设置各链接的订阅和自动push策略`
- Width: current 49 cells, proposed 43 cells, English 56 cells.
- Reason: The same policy concepts are 订阅 and 自动push in tui.hub_links (policy.subscribe.label; mark.healthy), while this command description leaves them English; the command-help surface keeps auto_push only as a literal token.

### ZH-S7 — nit — typography

- Key: `wire.slash.stop`
- Current: `结束本会话、按名称/pid指定的会话或全部 — /resume可重新打开`
- Proposed: `结束本会话、按名称/pid指定的会话或全部——/resume可重新打开`
- Width: current 58 cells, proposed 57 cells, English 66 cells.
- Reason: The locale uses the Chinese double em dash —— in prose (mobile.setup.activating) and full-width punctuation; a spaced Latin em dash reads translated.



## Reviewed, not findings

- `wire.slash.loop`'s ASCII comma is inside the literal `<goal>, <n>` syntax; `wire.slash.mobile` and `wire.slash.project` use ASCII commas only between literal subcommand tokens. The style guide allows Latin punctuation inside Latin/code fragments.
- `wire.slash.resume`'s ASCII parentheses around `id` were reviewed as an argument hint, not prose punctuation; the English source uses the same form.
- `auto` (`wire.slash.language`), `mesh` (`wire.slash.network`), `token` (`wire.slash.analytics`, `wire.slash.context`), `Hub`, `MCP`, `OAuth`, `Radient`, `PyPI`, `id`/`ID`, and the literal flags/subcommands are kept English exactly as the style guide and glossary direct.
- The catalogue uses no 你/您 pronouns; the style guide's 你 address rule is not violated (verb-forward phrasing is used instead).
- No string is materially longer than its English source. Two rows exceed the approximate 55-cell `/help` pin — `wire.slash.stop` (58 cells vs English 66) and `wire.slash.fork` (59 vs 74) — but both are shorter than English, so they inherit rather than add the English fit risk. A rendered check is not claimed.

## Mechanical results

- Key parity: en 52 / zh 52; missing 0; extra 0; order equal.
- ICU argument parity (names and kinds): 52 keys checked, 0 failures.
- Message parse/render: 52 zh messages and 52 en messages rendered with empty parameters, 0 failures.
- Plural structure: 0 plural keys; failures 0. zh-CN's other-only arms are expected by the style guide (none are present here).
- Hard technical tokens (backticks, flags, angle tokens, URLs, paths, %s/%d): 0 failures.
- Width: 52 rows measured; 0 materially longer than English; 2 above the approximate 55-cell `/help` pin but shorter than English.
- Punctuation: reviewed; one nit (`wire.slash.stop` em dash). Latin/CJK spacing: 0 failures; 您 occurrences: 0.
- Numbers: no formatted number/date literals to check; 0 failures.
- Encoding: non-NFC 0; bidi controls 0. RTL checks: not applicable (LTR locale).
- Route: `local_operator/i18n/catalogues/zh-CN/wire.slash.json` key `wire.slash.credential` served with digest `0617711757169777c98ba763258cc09bc997914643a5602071cc66c082abc8dd`; audit-log sha256 `97a4f1e153a73868a672bdb6ed3a26a48317c5f4d20babfe5ad527be57784980`.

## Regression read

All 52 keys were read side by side with the English source and the English context sidecar (the sidecar covers `wire.slash.hub` only). The target catalogue is byte-identical to the held candidate at `4840abdeb650a44db3bd9b1f564b24b6546c099a` (sha256 `a3de10768a538e054afc35ca0aff44c29bbaa042e10e10a8913781d19de87b11`), and the English/context sources are unchanged (sha256 `b20f4be0dae6446424862ba227313b6c3384bd115a033edc57c0a289bfe74202` / `32b3475a5808048a6eeaae462fee0feb55405fca77026e41b9609a516d74a215`); no key regressed at the byte level. The six minor findings and one nit are term-consistency and typography issues in the carried text, not changes introduced by a later edit. The previously withheld key is correct and is not a finding.

## Verdict

Per namespace: **wire.slash — passed with follow-ups** (`clean_with_followups`): 52/52 keys read; 0 blocker; 0 major; 6 minor; 1 nit.
Overall: **clean_with_followups** on the audited bytes. This audit does not clear the zh-CN security hold, does not make the locale shippable, and does not edit the ledger, allowance, shipped set or eligibility files.

## Limitations

No native TUI screenshot, relay or desktop render is claimed. Width is measured in cells (CJK/full-width = 2) against the approximate `/help` pin; the +25% growth figure in `pseudo-evidence.json` is a synthetic probe, not a rendered frame. zh-CN remains held and is not made shippable by this audit.
