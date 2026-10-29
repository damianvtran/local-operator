# zh-CN carried-namespace attestation — zhca1

- Revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Auditor: translation-auditor (lopdev team), session `d9e49690d15d`, model `deepseek/deepseek-flash`

## Totals

- Prior ledger rows examined: 72
- Byte-carried: 64 (63 attested + 1 withheld)
- Carried and attested: 63
- Withheld (bytes carried, prior audit failed): 1
- Not carried (hash differs, fresh audit required): 8
- Keys mechanically checked (all rows): 2619
- Keys sample-read: 390

## Not carried

### cli.sessions_help
- source: prior `81b18d3d6b7e10419262b2ea97a4dfc423779be6d82b872dbf04209704c2bfe3` -> current `a250a185a780ab330187560c23dfe5e8d95956da24d417b22e1fc8bb121c2a6a`
- target: held `3b859a0e1f9f5a244b41e14aab9ea4be5319dabed46caaa0b825be5fca410cac` -> current `d8e41d97fdbfdf890e14088ebdd44f87aeb3c2efb49a6401b430f45e51914bf8`
- fresh audit required

### tui.authoring
- source: prior `d1a49c668d1049700396c33b7af3c905d0f02b30a2cbf55aab7ea91b40c36cb4` -> current `7431acf3288cc3766cf37135badb29511737120ab52fc518bf514311ee488d13`
- target: held `d233ce0bfe19bfbc60495f39a87ea9a6165578bf3efda85b9ca94c485320d3e9` -> current `b26f3ab6cddbf93beff849976a2f453feaa4ee4a828474a2f30e37e9410c7470`
- fresh audit required

### tui.notices
- source: prior `2eaf4097197ee0257cfcf7d9d0f450e06fb496db67a92edb21e6f6b1122aaef1` -> current `4a159689628113c32a2df75a9e3d98d74aab790c9946848307bb267318731510`
- target: held `035b9df1c4d6c2e124c7ffab6f2e49eb310f6508f08a6957ba042d3964e90ebe` -> current `40af781ff575e4571fbb4caa7f2ae55443f1bdb2c344c2e04137d59198c77c61`
- fresh audit required

### tui.small
- source: prior `7cd74d45ada43efceba846a9c5d29a0aa90e0567e1f137778300e1742a6adfec` -> current `16d49c4ba2a514bbde7da494443abade26a3b28c0389e6bca124d00aa7d0ed75`
- target: held `a05ee9e08e560ddb5bbaf999875e20e53a9eb7e58d73ffbc1712c119ac8a92d7` -> current `5b7e05905402a7713dee3dd0ffa3df8048bae23e6b18108cfcec1c6cd4567ca6`
- fresh audit required

### wire.errors.desktop
- source: prior `30312c2052a41da6f580f00dce8daf9babbc0b25c0462c1f2658817cd51f391c` -> current `9a164be42a80d9bf0f60ad2a9f1d76fd1601855a4d19ffac3bf35d6bef11f76d`
- target: held `cf2fbad491db3452ce9ef80f9764e1e4937a352ed66eb2f99cc1d151e2f2daf1` -> current `a8cb0db6bbc8a33ef041c082d68b9c06dd0b3168a303e447e0f40e0334fc89f4`
- fresh audit required

### wire.memory
- source: prior `8732c8a8ab1572a39bf81ff94d80c8ea02f146d7da933db6eb41ee3c73062a00` -> current `319e23e6e4a6abee4e9549655142bebb5402563902c6f0497f3ee761fb53aec7`
- target: held `304e44b834272be678bdcacc57b48b18810690e9a2d283156d419f5c58098fc1` -> current `4e39d2bacb0ea88a16f7de720a5ae2208a580ea789ecc135a1c4368d2f3d1e28`
- fresh audit required

### wire.settings
- source: prior `d1bc117e26fd79c55663c7ba1972304509a42370213f90924b0b9dc6ec81f63a` -> current `c1e237c4dd3c2ea236f9e39d15cf4d0b34a231841357aee5a389f34201a73582`
- target: held `b5bcb8ebfaa674d31b155ed64130f4a7a256c40b2c7833cfa1f901d12c5c8138` -> current `d0df5e3e6975719b39d3e81171b2d9416844a1df8a5498f81f398476eb284b55`
- fresh audit required

### wire.settings.hub_defaults
- source: prior `9e785003bab2f7c01f32bae9b2cb114a240a82bac509c05bc55266585a41241d` -> current `8720b5719479b638eab24b081f76f97c44ce8d8554cb1b8280f6980182b6740c`
- target: held `2ba72847cff17089eaf641666b2ea7602a4e27bbc585a97da01d1e400ef25197` -> current `b547b9730d78a81e840746de723e8a44597a3dcb175cd7e0200a143bb4972c13`
- fresh audit required

## Withheld

### wire.slash
- current hashes equal the held candidate, but the prior row is `failed`
- zh-CN core held-candidate wave 2026-10-19. Translator i18n-translator openai/gpt-6.1-sol (launched anthropic/claude-sonnet-5-5; provider quota fallback). Preserved 51 values byte-identical to held/i18n-w3-zh-candidate-4773bb87; 1 new key wire.slash.hub. Full audit coverage 51/52: wire.slash.credential is unreadable under ordinary tool-output redaction and was NOT extracted, reread or bypassed; the 51 readable values and the new hub key are clean (0 findings). Namespace WITHHELD at 51/52; status failed until an approved security disposition covers the unreadable value. Independent auditor i18n-auditor openai/gpt-6.1-sol, 2026-10-19.

## Findings

### ZHC-01 — minor — 'credit' drifts between 点数, 积分 and 余额

The same English billing term is rendered three ways: 点数 in mobile.setup and relay.shell.usage.unit_credits, 积分 in tui.usage.paid.continue, and bare 余额 (dropping the credit noun) in the two speech errors. 点数 is the dominant rendering and matches 'Add credit'/'account credit'; standardise on 点数, keeping 余额 only as the compound 点数余额.

- `tui.usage.paid.continue`
  - current: `包含的配额已用尽；将继续使用积分`
  - proposed: `包含的配额已用尽；将继续使用点数`
- `wire.errors.media.speech_radient_credit_low`
  - current: `Radient余额不足以生成语音。在Radient Console充值后继续。`
  - proposed: `Radient点数余额不足，无法生成语音。请在Radient Console充值后继续。`
- `wire.errors.media.speech_vendor_credit_low`
  - current: `{vendor}余额不足以生成语音。在{vendor}充值后继续。`
  - proposed: `{vendor}点数余额不足，无法生成语音。请在{vendor}充值后继续。`
- `mobile.setup.balance`
  - current: `自行前往https://console.radienthq.com/dashboard/tunnels充值；设置绝不会购买点数。之后重试。`
  - proposed: no change (reference side)
- `mobile.setup.consent`
  - current: `激活将为此Radient账户开通月费为USD {amount}的隧道订阅，预留手机地址，并安装和启动后台连接器。此操作不会购买点数。`
  - proposed: no change (reference side)
- `mobile.setup.title_balance`
  - current: `手机访问需要账户点数`
  - proposed: no change (reference side)
- `relay.shell.usage.unit_credits`
  - current: `{count, plural, other {点数}}`
  - proposed: no change (reference side)

### ZHC-02 — minor — 'quota' drifts between 配额 and 额度, and one sentence is translated twice

The same English sentence family ('Included quota used; credits will continue.') is 包含的配额已用尽；将继续使用积分 in tui.usage and 已用完内含额度；将继续使用点数 in relay.shell. 配额 is the dominant rendering (wire.settings, wire.slash.usage), so relay.shell should use 配额 and 包含的.

- `relay.shell.usage.continue`
  - current: `已用完内含额度；将继续使用点数。`
  - proposed: `包含的配额已用尽；将继续使用点数。`
- `tui.usage.paid.continue`
  - current: `包含的配额已用尽；将继续使用积分`
  - proposed: `包含的配额已用尽；将继续使用点数`

### ZHC-03 — minor — Aida 'check-in' drifts between 问候 and 回访

The carried cadence namespace says 问候 (wire.aida_cadence), while the TUI status surface says 回访 (tui.app.aida). The settings namespace that is not carried (wire.settings) says 检查. One feature term should have one rendering; 问候 matches the cadence labels that name the setting.

- `tui.app.aida.next_checkin`
  - current: `下次回访{when}`
  - proposed: `下次问候{when}`
- `tui.app.aida.extras_used`
  - current: `今日额外回访已用：{used} / {maximum}`
  - proposed: `今日额外问候已用：{used} / {maximum}`
- `wire.aida_cadence.interval_hours.label`
  - current: `问候间隔（小时）`
  - proposed: no change (reference side)
- `wire.aida_cadence.interval_hours.help`
  - current: `0保留每日本地问候时间。1–168按经过的小时数计时；4表示每四小时问候一次。更改会保留已到期应执行的问候。`
  - proposed: no change (reference side)

### ZHC-04 — minor — The Library feature is 资料库, 资源库 and 资产库

tui.library consistently calls the feature 资料库; the slash help calls the same Library 资产库, and the (not-carried) desktop errors call it 资源库. One feature name is needed; 资料库 is the rendering the TUI title and the Librarian settings both use.

- `wire.slash.library.help`
  - current: `浏览全局资产库`
  - proposed: `浏览全局资料库`
- `tui.library.title`
  - current: `资料库`
  - proposed: no change (reference side)
- `tui.library.loading`
  - current: `正在读取资料库……`
  - proposed: no change (reference side)

### ZHC-05 — minor — 'key agent' rendered as an AI agent (智能体)

'the macOS key agent' is the signed helper app lop-keyagent.app (local_operator/update.py), not an LLM agent. 智能体 is this locale's fixed rendering of 'agent' in the AI sense (glossary), so 密钥智能体 misstates what the wheel carries; use 密钥代理应用.

- `cli.update_help.from_snapshot`
  - current: `将本地源码树（目录或当前仓库的git引用）安装为独立安装代次。在macOS上，恰好对应已发布版本的引用会安装PyPI上发布的wheel，这是唯一包含macOS密钥智能体的构建；其他引用均从源码构建。`
  - proposed: `将本地源码树（目录或当前仓库的git引用）安装为独立安装代次。在macOS上，恰好对应已发布版本的引用会安装PyPI上发布的wheel，这是唯一包含macOS密钥代理应用的构建；其他引用均从源码构建。`

### ZHC-06 — nit — Bare Latin 'operator' left in a prose sentence

Everywhere else the person is 操作员 and the product is Local Operator. '运行operator的工作目录' reads as an untranslated token; name the product or the role.

- `cli.global_help.run_in`
  - current: `运行operator的工作目录。  必须是有效目录。`
  - proposed: `运行Local Operator的工作目录。  必须是有效目录。`

### ZHC-07 — nit — ASCII colon in Chinese prose

The zh-CN style guide requires full-width punctuation in prose. These are the only two Chinese-prose strings in the carried corpus that use an ASCII colon and a following space.

- `tui.projects.form.readback_hint`
  - current: `{label}提示: {hint}`
  - proposed: `{label}提示：{hint}`
- `tui.projects.form.readback_error`
  - current: `{label}错误: {error}`
  - proposed: `{label}错误：{error}`

### ZHC-08 — nit — Plural counter spacing is split '# 个' vs '#个'

Four carried strings put a space between # and the classifier while 26 do not. The style guide's plural example shows a space, but its punctuation rule forbids a space between Chinese and digit runs; either way the corpus should not be split. Proposed: follow the majority and the punctuation rule.

- `tui.status.agents`
  - current: `{count, plural, other {# 个智能体}}`
  - proposed: `{count, plural, other {#个智能体}}`
- `tui.status.jobs`
  - current: `{count, plural, other {# 个任务}}`
  - proposed: `{count, plural, other {#个任务}}`
- `tui.transcript.attach.receipt`
  - current: `↑ 已附加 {count, plural, other {# 张图像}}`
  - proposed: `↑ 已附加 {count, plural, other {#张图像}}`
- `tui.transcript.catchup.missed`
  - current: `{count, plural, other {# 次错过的唤醒}}`
  - proposed: `{count, plural, other {#次错过的唤醒}}`

### ZHC-09 — nit — 'provider census' rendered with 普查

Here 'census' is the enumeration of configured providers that the credential store is needed for; 普查 (a statistical census) is not the idiomatic software term. 清点 keeps the counting sense without the demographic register.

- `wire.errors.auth.store_unreadable`
  - current: `无法读取凭据存储——提供商普查需要读取它。`
  - proposed: `无法读取凭据存储——提供商清点需要读取它。`

### ZHC-10 — nit — Projects summary is 24% wider than English in cells

{open}项未完成 / {total}项 is 26 cells against the English 21, in a narrow summary row with no declared maxCells. The shorter form keeps the open/total meaning and fits; flag for a rendered check rather than assert a break.

- `tui.projects.render.open_over_total`
  - current: `{open}项未完成 / {total}项`
  - proposed: `{open}/{total}项未完成`

## Scope

Byte carry against the held-candidate audit row plus a sampled semantic read. No re-translation, native/rendered UI evidence or design sign-off is claimed. A namespace whose hash differs is NOT CARRIED and needs a fresh audit.
