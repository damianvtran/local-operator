# Vietnamese independent safety-copy audit — first instance

**Verdict: passed.** All 17 keys, all select arms and fallbacks, and the Vietnamese `other` plural were read and audited. No BLOCKER, MAJOR, MINOR or NIT translation finding. This is one independent pass, not the required second pass or aggregate release approval.

Auditor: `i18n-auditor`, task `terminal-status-vi-audit`, session `7a2548e2c7b1`, actual model `openai/gpt-6.1-sol` from the host Model header; no fallback used. Independent of translator session `56c3029571f6`; same model family is not the same agent instance. No translation/MT service used, no catalogue edited.

Frozen repository head: `2930492e05a6968a2e1adfe34a091b94802bacd0`.

## Inputs and method

Read Vietnamese style guide and glossary, English source and complete context sidecar, then the candidate. Input byte hashes were independently calculated and asserted by `audit_vi.py`:

- English: `52a712a441652702614192715b94d271bdc4653497a9d7065c82d362786d9326`
- Context: `5119ae9388854ba8c6bbca489034d75d145426ad13d12361468cc6a6d25fc525`
- Candidate Vietnamese: `963b8bf337981a328bb80b265cb49d033db10c7afb2838c4ec8d5f05df20c1d1`

Actual repository `scripts/i18n/check.py::check_catalogues()` ran on a scratch-only en/context/vi fixture under isolated HOME/config, inherited environment stripped with `env -i`, and bytecode writes disabled. Normal admission returned `[]` before and after mutations. Rejected negative controls: missing key, extra key, renamed placeholder, changed placeholder type, invalid Vietnamese `one` category, malformed ICU. Independently inspected recursive AST select-arm sets and placeholder kinds, duplicate-free JSON and exact key sets, NFC and no bidi controls/newlines. Existing `retired_label` value is byte-for-value unchanged.

`vi-first-audit.log` is actual command output. This report is **audit_notes_not_execution_log**. `vi-first-render-cases.json` contains all 58 actual ICU text-formatting cases, key/parameters/current text/cell count/character count; it is not rendered UI or runtime proof.

## Complete per-key audit

Keys below are relative to `wire.session_status.`. The case JSON is the complete current-text listing, including every branch.

| Key | Current text or form | Audit reasoning | Result |
|---|---|---|---|
| retired_label | `{base} — đã ngừng để cập nhật` | Preserves `{base}` verbatim and update purpose. Existing value unchanged; new formatter does not use this legacy composition path. | PASS |
| interrupted | `Gián đoạn` | Natural terse interrupted outcome; does not assert a user cancellation. | PASS |
| unseen_interruption | `Gián đoạn — chưa đọc` | Read-state separate from interrupted outcome, not an extra failure. | PASS |
| retired | `Gián đoạn — đã ngừng để cập nhật` | Explicit update purpose; no attribution to operator or history deletion. | PASS |
| unseen_retirement | `Gián đoạn — chưa đọc — đã ngừng để cập nhật` | Preserves both unread state and update-specific cause. | PASS |
| signal_stop | `Gián đoạn — đã dừng bằng tín hiệu` plus optional ` qua {command}` | Signal retained explicitly. `none` yields no attribution; other inserts whole exact command bytes. | PASS |
| unseen_signal_stop | Above with `— chưa đọc` | Same two arms checked independently; unread marker not confused with stop signal. | PASS |
| forced_kill | `Gián đoạn — bị buộc kết thúc` plus optional ` qua {command}` | Explicit forced termination distinct from signal stop; no literal murder metaphor. | PASS |
| unseen_forced_kill | Above with `— chưa đọc` | Both arms preserve force and unread status separately. | PASS |
| complete | `Hoàn tất` | Concise idiomatic completion, no extra success conditions. | PASS |
| unseen_completion | `Hoàn tất — chưa đọc` | Completion remains complete; only reading is pending. | PASS |
| recent | `Gần đây` | Correct temporal recency. | PASS |
| dormant | `Đã dừng ({count, plural, other {# lượt thức dậy tạm ngưng}})` | `thức dậy` follows glossary; classifier `lượt` counts occurrences; `tạm ngưng` is suspended/dormant, not canceled, deleted or promised to fire. All count values retain digits. | PASS |
| error | `Lỗi` | Correct generic error for absent diagnostic, no fabricated cause. | PASS |
| unseen_error | `Lỗi — chưa đọc` | Unread is independent of error outcome. | PASS |
| error_cause | Full cause select, listed below | Every stable machine arm unchanged; no English diagnostic is injected. | PASS |
| unseen_error_cause | Same cause select with `Lỗi — chưa đọc — …`; fallback `Lỗi — chưa đọc` | Every arm audited separately against ordinary counterpart; no dropped negation or unread-marker branch. | PASS |

## Complete cause-branch audit (both cause keys)

The current ordinary text is shown in full. Every unseen branch has the same cause wording after `Lỗi — chưa đọc — `; both fallback forms are stated explicitly. There are 13 arms in each key: twelve named causes plus `other`.

| Stable arm | Current ordinary Vietnamese text | Meaning/safety audit | Result |
|---|---|---|---|
| user_stop | Lỗi — người dùng đã dừng | The only branch explicitly blaming a user stop; correctly says user. | PASS |
| runtime_stall_bound | Lỗi — runtime tự kết thúc khi chạm ngưỡng đình trệ | Self-termination and stall bound retained; does not name an external kill or invent duration. | PASS |
| continuation_limit | Lỗi — đã chạm giới hạn tiếp tục; tin nhắn đang chờ bị loại bỏ | Continuation limit and pending-message loss both present; not deletion of session history. | PASS |
| output_limit | Lỗi — đã chạm giới hạn đầu ra của mô hình; không thể tiếp tục | Model output limit and inability to continue retained with explicit negation. | PASS |
| runtime_retired | Lỗi — đã ngừng để cập nhật | Explicit update retirement, no operator attribution. | PASS |
| runtime_shutdown | Lỗi — runtime bị kết thúc trong lượt | Runtime terminated during the turn; does not invent a cause/actor. | PASS |
| runtime_overdue | Lỗi — runtime rời đi để cập nhật sau ngưỡng tiến triển | Departure for update after progress bound distinct from stall-bound self-ending. Terse technical label, not a positive progress/success claim; no invented timeout. | PASS |
| runtime_update_failed | Lỗi — cập nhật chưa hoàn tất; runtime giữ nguyên bản đã nạp | Failed-to-finish update plus continued loaded build retained; does not claim upgrade success. | PASS |
| runtime_killed | Lỗi — runtime biến mất mà không thoát đúng cách; không có yêu cầu dừng | Both unclean disappearance and no stop request explicit; not forced operator stop. | PASS |
| install_mid_update | Lỗi — bản cài đặt bị thay thế trong lượt | Installation replaced during turn, not history replaced or a completed update guarantee. | PASS |
| owner_lost | Lỗi — runtime ngừng phản hồi trong lượt | Loss of runtime response, not asserted death/deletion. | PASS |
| disposed | Lỗi — phiên bị giải phóng trong lượt | Disposed runtime session: `giải phóng` does not say stored history deleted (`xóa`). | PASS |
| other | Lỗi | Generic fallback; unseen fallback is Lỗi — chưa đọc. | PASS |

## Branch/count coverage and fit

- Four command selects × `none`, `/stop --all`, `lop sessions stop synthetic-123`: 12 actual cases. Exact technical bytes checked, no `none` or dangling connector leaked. Attribution connector `qua` reads naturally with a literal command.
- Two cause selects × all 13 arms plus a synthetic unknown selector: 28 actual cases. Unknown selector ICU fallback alone is not proof of external-diagnostic application behavior.
- Dormant counts: 0, 1, 2, 5, 21, 55, 1,000,000 and 1.5: 8 cases. Vietnamese `other` only; large/fraction format outputs `1.000.000` and `1,5` per locale formatter. No plural `one` arm needed.
- Remaining ten keys: one case each, including legacy base placeholder.
- All Vietnamese strings are NFC; Latin script with precomposed Vietnamese letters, not wide CJK glyphs or decomposed combining marks. No maxCells/maxChars hints exist in this sidecar and it explicitly sets no fixed cap. The longest tested fixed no-command label is 81 cells (`unseen_error_cause/runtime_killed`); a long injected command can extend further. No overflow defect is asserted without a cap or a rendered surface. Narrow-layout/pseudo/visual proof is outside this audit.
- `runtime` stays English per vi style guide; `mô hình`, `phiên`, `thức dậy` conform to glossary. Em dashes are spaced, sentence case preserved, no added courtesy/register.

## Observed validator gaps — not candidate findings

1. The real catalogue validator accepted a mutated cause select missing `user_stop` (`[]`). Independent AST comparison rejected its missing arm. Candidate itself contains every arm in both cause keys.
2. The actual `_check_plural_categories` helper accepted synthetic Russian `other` only while its own required category table contains `one/few/many/other`: it rejects unknown categories but does not enforce all non-other categories. This is an observed validator limitation, not a new Vietnamese defect. Vietnamese table is exactly `['other']`, and the target has that complete category set.

No target fix is requested for these pre-existing mechanical-gate limitations. No translation finding has been padded from them.

## Boundaries and retention

No catalogue, repository history, generated asset, build, install, server or PR mutation. No full suite, visual/runtime flow, pseudo-localization, terminal shaping or native-human certification. Unknown external diagnostic legacy-English behavior remains intentionally **UNRESOLVED** in the separate programme scope.

Final receipt is append-compatible and can be dry-run only here. Coordinator must retain this report, record, actual log, case JSON and validator script under `i18n/audits/evidence/vi/wire.session_status/ts1a7a/` before applying the receipt. Scratch is only a handoff, not a claim that evidence is tracked or merged. Second independent audit still required.
