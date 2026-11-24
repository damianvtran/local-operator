# Independent Simplified Chinese audit A — PASSED

Audit notes, NOT execution log. Completed after admission repair; prior incomplete attempt `../zhA26/record.json` remains untouched and is not retrospectively a linguistic pass.

Auditor: i18n-auditor / model-notice-zhCN-audit-A, session 26adce0a0cfa. Actual current host-declared model anthropic/claude-opus-5-5. Earlier admission-only attempt reported openai/gpt-6.1-sol fallback; the current resumed host declares anthropic/claude-opus-5-5, which is the source for this completed audit's model identity. No MT API/service used. Independent from translator session 58dee179aac0 (author record zha58/record.json); also not source author a97c7f76d5ee. Same model family, where applicable, does not imply same actor. Independent B audit remains separate/pending.

## All four native-copy decisions

| Key | Current text | Independent judgment |
|---|---|---|
| persist | /model default将当前模型保存为新会话的默认模型 | PASS. 当前模型 states what is saved; 新会话 excludes switching this session. 保存为…默认模型 is idiomatic product wording, not a literal boot metaphor. |
| settings | /settings也可修改新会话的默认模型 | PASS. Explicit future-session default and a standalone usable route. 也可 naturally preserves “too”; it does not require an unavailable /model command when this is the only displayed route. |
| saved | /model saved将本会话切换至已保存的默认模型 | PASS. 本会话 + 切换至 distinguishes THIS-session switching from saving. 已保存的默认模型 preserves the previously saved default, without implying persistence of the current model. |
| with_model | 模型：{label} — {routes} | PASS. Human label translated per glossary; opaque ID and already-localized clauses each interpolated exactly once. Full-width colon is native Chinese punctuation. |

No BLOCKER, MAJOR, MINOR or NIT copy findings; no suggested wording needed. Register is concise declarative behavior explanation rather than a blocking/error directive, so the style-guide 请 rule does not require adding 请. 模型 and 会话 match glossary; no gratuitous pronouns; no spaces inserted at Chinese/Latin boundaries; protected ASCII commands retain their single internal space. No label/command retranslation. No negation, destructive-action or permission polarity changes.

## Actual mechanical execution

`validate.py.txt` was copied into THIS NEW completed bundle and actually executed here (the earlier attempt's script was not executed there). With primary .venv, PYTHONPATH=this worktree, env -i, scratch HOME/config, TERM=xterm-256color, no inherited CMUX/LOP variables: exit 0. Module resolved this worktree's local_operator/__init__.py. Official `scripts.i18n.check.check_catalogues()` scoped through namespace/locale enumeration to ONLY tui.model_persistence + en/zh-CN returned []. This retains official catalogue JSON/shape/context/ICU/argument/plural checks, without running global/base/coverage gates. Exact ordered key set: four. First three have no arguments; envelope has label and routes as simple/string arguments, once each. Zero plural/select arms in both languages, so CLDR branch expansion is inapplicable. Protected commands byte-equal. Additional direct checks pass.

`control.py.txt` independently executed: exit 0. `messages.render_record(..., locale='zh-CN')` returns all four actual Chinese results with source_locale='zh-CN'; verifies NFC, no bidi controls and no added markup/backtick spans. Logs contain actual stdout, not author conclusions.

## Fit: measured, qualified

Clause cell widths: persist 46, settings 33, saved 42. Envelope template 24 cells including literal placeholder syntax (not a display cap). No maxCells/maxChars exists; the legacy 43-cell footer budget is explicitly inapplicable.

Plain Rich controls cover ordinary/missing-model/settings-only route sets, short 3-character and long 80-character opaque labels, at 80/100/120 columns. Short-label ordinary output wraps to 2 rows at all three widths; long-label ordinary output to 3/3/2 rows. All measured lines fit selected widths. Settings-only remains meaningful. Pseudo expansion control preserves arguments and exact commands.

Important integration observation, not a new copy finding: generic Rich wrapping splits `/model saved` after `/model` for short-label ordinary notices at 80/100, and splits `/model default` with some long-label budgets. Same-code English control also splits a two-word command (short label width 100; long label width 100). Therefore bare Rich controls do NOT prove NoticeBlock command-boundary handling. Long IDs also wrap; bytes remain preserved in rendered string. Root must inspect real NoticeBlock budgets/pristine and populated transcript families; no all-width guarantee or native integration/terminal shaping certification claimed. No invented fit cap, no product redesign proposed.

## Scope and provenance

Reached committed HEAD 87d87ec8b02ef3c279fda673302ce775f4a0329e. English source authored at c40586cd0ae23ca25a6c82c26c4f867271c4e01f; source/context/target hashes unchanged before and after. Approved public-copy target read succeeded after the private admission commit. Earlier admission access blocker resolved; it was not a translation finding. No target/source/context edits, no ledger append/stage/commit/push/generation/full suite/native app dispatch. Receipt is append-ready only; root serializes history later.
