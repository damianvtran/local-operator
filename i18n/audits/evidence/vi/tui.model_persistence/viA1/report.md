# Vietnamese independent audit A — tui.model_persistence

Date: 2026-11-22. Scope: all four keys at b1c5ca8be245dcc3f9cae4463d9a5553b3e0c01f. Verdict: **passed**, zero BLOCKER/MAJOR, no findings (including no padded minor/nit findings).

Auditor: i18n-auditor, session fcb8a76716b3, actual model openai/gpt-6.1-sol. Host switched from anthropic/claude-opus-5-5 for provider quota failure before this assessment. Independent of translator edc50a6f3d77; translator actual model openai/gpt-6.1-sol and fallback are preserved from viau1/record.json. Same model family does not mean same translator session. No MT/API/service was used.

## Complete native assessment, in source order

1. `persist`: `/model default lưu lựa chọn này cho các phiên mới`. Accurate: “lưu lựa chọn này” saves the current choice; “cho các phiên mới” binds its use to future new sessions, not an immediate switch. Natural concise instructional copy. `phiên` follows the glossary. Exact ASCII command retained.
2. `settings`: `/settings cũng sửa mô hình mặc định khi mở phiên`. “cũng sửa” conveys the alternate editing route; “mô hình mặc định khi mở phiên” identifies the startup default, not the current model. Terse but idiomatic in this command-as-subject construction. It is also a complete usable clause when shown alone: “cũng” retains the source's “too” without inventing another available command. No forbidden /model route added for hosting dead ends.
3. `saved`: `chuyển phiên này sang mô hình mặc định đã lưu bằng /model saved`. Explicit “phiên này” makes the THIS-session switch unambiguous; “đã lưu” identifies the existing saved default, rather than saving the present choice. Natural “chuyển … sang … bằng …” construction. `mô hình` and `phiên` match the glossary; the command remains literal.
4. `with_model`: `mô hình: {label} — {routes}`. Human prefix translated per glossary; sentence-case lower-case fits the source notice. Opaque label and already-localized route clauses each remain a single simple-string argument, unchanged and exactly once. Source colon/em-dash spacing retained, consistent with Vietnamese guide.

All four use modern terse product voice, no unnecessary pronouns or “vui lòng”, no inappropriate formal register, no ASCII ellipsis, no incorrect tone-mark normalization. Meaning was independently reasoned from English and context after reading the Vietnamese style guide and glossary. This is A only, not the later independent B approval.

## Executed mechanical evidence

Official `scripts.i18n.check.check_catalogues()` ran with only namespace/locale enumeration narrowed to en/vi + tui.model_persistence; all checker internals unchanged. Actual result `[]`; valid key/context/ICU/CLDR checks. Exact ordered four-key parity independently checked; zero plural/select/selectordinal structures, so no invented locale plural branches. Simple argument names/types match; command bytes, absent Rich/backtick tags, NFC and absence of bidi controls checked independently. Missing-key and renamed-argument counterfactuals produced official failures. Missing runtime label/routes each raised MessageFormatError. Unknown-code API's documented code-passthrough was checked.

81 plain Rich wrapping cases passed: English controls first, Vietnamese second, then synthetic expanded-English human-only copy; ordinary, missing-model and absent-hosting route compositions; short/ordinary/long opaque IDs; 80/100/120 columns. Full row content and cell counts are in audit.log. Clause-only no-model output exercised; hosting-dead-end composition contains /settings only. No maxCells/maxChars exists in this namespace; the 43-cell legacy footer cap does not apply. Source Vietnamese clause cell lengths are 49, 48, 63. No overflow finding is made: these are wrapping notices.

### Actual successful command

From the worktree root (scratch HOME created beforehand):

```sh
env -i HOME="$LOCAL_OPERATOR_SCRATCHPAD/viA1-home" \
 LOCAL_OPERATOR_CONFIG_DIR="$LOCAL_OPERATOR_SCRATCHPAD/viA1-home/config" \
 PATH="$PATH" PYTHONPATH="$PWD" TERM=xterm-256color \
 /Users/damian/local-operator/.venv/bin/python \
 i18n/audits/evidence/vi/tui.model_persistence/viA1/script.py.txt \
 > i18n/audits/evidence/vi/tui.model_persistence/viA1/audit.log 2>&1
```

Exit 0. Log proves primary interpreter, all imported validation/runtime module paths in this worktree, isolated HOME/config, CMUX absent, all three hashes unchanged BEFORE/AFTER. Git revision read under normal HOME. `script.py.txt` is the executed reproduction source; `audit.log` is actual output. This report is audit notes, not execution output.

## Harness correction preserved separately

The first independent fixture incorrectly expected unknown message IDs to raise. It failed before the wrap matrix. Inspection of `messages.py:161–163` established documented code passthrough; corrected the fixture and reran successfully. Original failure and source remain `harness-failure.log` and `failed-script.py.txt`; this was not a translation defect. An earlier read-only discovery command also tried nonexistent render.py (exit 1); no check execution or copy judgement relied on it.

## Limits and authority

Plain Rich only, not styled NoticeBlock/native font, actual pristine/nonempty transcript geometry, app startup, screenshots, RTL certificate, UX/design or UI approval. No consumer integration or release/eligibility claim: consumer may still be English. No source/translation changes, shared ledger/history mutation, staging or commit. Append-ready receipt is supplied for the manager; evidence exists in the manager-owned worktree and is not yet claimed committed or on main.
