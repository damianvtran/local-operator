# Attestation — zh-CN / cli.update_help / zhca1

- **Verdict: carried**
- Attested revision: `c4ba95919dbea27fcf990101f9e48ed21e977e10`; held candidate: `ebb70df3a887aa7e9b448560e0546c2cf817b133`
- Prior audit: `i18n-auditor` on 2026-10-19, model `anthropic/claude-sonnet-5-5`, result `passed`, 0 finding(s), from the held-candidate ledger row
- Key count: 6 (en) / 6 (zh)

## Hash equality proof

- Source: prior row `78da39b393ee77900255ddfdf92a17220a6eedbecba340e7d925a2fbdf01abf9` = held `78da39b393ee77900255ddfdf92a17220a6eedbecba340e7d925a2fbdf01abf9` = current `78da39b393ee77900255ddfdf92a17220a6eedbecba340e7d925a2fbdf01abf9` — **equal**
- Target: held `b59b3b7c5da4535a4543b56ad54a6129f9abdcf761f7ef4e6b531b208277d790` = current `b59b3b7c5da4535a4543b56ad54a6129f9abdcf761f7ef4e6b531b208277d790` — **equal** (the prior ledger row does not record a target hash)
- Context sidecar: equal (`46dfc07c168a78805b437a4561dc0a8f47b005a48176b424b5aca47373cdaa5f`)
- Whole target file byte-identical to held candidate: **true**
- Policy inputs (glossary and `i18n/style-guides/zh-CN.md`) are byte-identical to the held candidate.

## Mechanical checks

- Key set equal: `True`; order equal: `True`; duplicates: 0 en / 0 zh
- ICU: 6 messages checked, 0 parse error(s), 0 argument-tuple mismatch(es), 0 plural-category failure(s)
- Technical tokens: 0 literal failure(s), 0 kept-English identifier(s) missing, 1 ALL-CAPS emphasis token(s) rendered lexically; ICU number/date skeletons: 0 key(s) checked, 0 failure(s)
- Width: 0 declared `maxCells` key(s), 0 over budget

## Stratified sample

6/6 keys read (100.0%), including all 4 messages over 120 characters and all 0 plural messages.

## Findings

- **ZHC-05 (minor)** `cli.update_help.from_snapshot` — 'the macOS key agent' is the signed helper app lop-keyagent.app (local_operator/update.py), not an LLM agent. 智能体 is this locale's fixed rendering of 'agent' in the AI sense (glossary), so 密钥智能体 misstates what the wheel carries; use 密钥代理应用.
  - current: `将本地源码树（目录或当前仓库的git引用）安装为独立安装代次。在macOS上，恰好对应已发布版本的引用会安装PyPI上发布的wheel，这是唯一包含macOS密钥智能体的构建；其他引用均从源码构建。`
  - proposed: `将本地源码树（目录或当前仓库的git引用）安装为独立安装代次。在macOS上，恰好对应已发布版本的引用会安装PyPI上发布的wheel，这是唯一包含macOS密钥代理应用的构建；其他引用均从源码构建。`

## Scope

No native TUI, relay, desktop, rendered-fit or design evidence is claimed. This is a byte-carry attestation with a sampled semantic read.
