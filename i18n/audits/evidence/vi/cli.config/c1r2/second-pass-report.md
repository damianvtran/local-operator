# vi C1 — independent full second-pass audit (cli.config.* 15, tui.session_move.* 7)

Auditor: i18n-auditor (session 56bda4ce6da8), actual served model anthropic/claude-sonnet-5-5.
Independent of translator (openai/gpt-6.1-sol). Did not read other auditors' reports or v2-changes notes.
Inputs: c1-prep/keys.json, c1-prep/vi.v2.json (22 keys, key sets identical). Style guide ~/local-operator/i18n/style-guides/vi.md + en.md + glossary.md.
Shipped baseline: .worktrees/i18n-c1 @ b188079f32e6 (vi catalogue has no cli.config / tui.session_move namespace yet — the candidate is new; terminology was checked against the other 147 shipped vi namespaces by counting uses).

## Verdict: PASSED — 0 BLOCKER, 0 MAJOR, 2 MINOR, 4 NIT

## Mechanical (python3, all 22 keys)
- Repo ICU parser (local_operator.i18n.runtime.parse_message / message_arguments / plural_selectors / render_message, locale vi): every value parses; argument (name,kind) tuples identical to en for all 22; no plural/select (correct: vi = other only); sample render OK.
- Backticked spans identical to en (cannot_wait, deadline_exceeded). Typed tokens `/approvals auto`, `/move`, `--to`, `--keep`, `--queue`, `--drop-session-roles`, `--wait` identical; `local` verbatim in `<thiết bị ngang hàng|local>`; `|` and angle brackets kept.
- NFC: all values NFC. No "...", no "!", no double/edge whitespace.
- Residual Latin (outside code/placeholders): only `auto`, `shell`, `CLI`, provider names (kept per guide / shipped: "shell" used the same way in 7+ shipped values; "CLI" is on the kept list). No stray English.
- Fit: no maxCells (one-line, may wrap); list_title 17 cells (<30).

## Safety lines
- cli.config.running_sessions_unchanged — all four claims present: (1) running sessions unchanged "Các phiên đang chạy không thay đổi"; (2) a config write cannot loosen them "ghi cấu hình không thể nới lỏng yêu cầu phê duyệt của các phiên này" (same "nới lỏng yêu cầu phê duyệt" wording as shipped relay/CLI help); (3) type `/approvals auto` in each session you want ungated "gõ /approvals auto trong từng phiên muốn bỏ yêu cầu phê duyệt công cụ"; (4) new sessions open at auto "Phiên mới bắt đầu ở chế độ auto". Calm, no alarm. OK.
- tui.session_move.deadline_exceeded — outcome stated as UNKNOWN ("chưa rõ kết quả chuyển"), neither failed nor succeeded; re-run command verbatim with {session}/{to}; rendered sample: "…chạy lại `lop sessions move abc123 --to laptop` để biết…". Consistent with use site (session_move.py marks it changed=True; app.py comment says move may have gone through). OK.
- cannot_wait: backticked command byte-identical incl. <id> <to> <seconds>. OK.

## Per-key table (all 22; OK = meaning, register, terms, placeholders)
cli.config.created OK · opened OK (same noun "tệp cấu hình" as created; matches shipped cli.config_help) · cleared OK · updated OK (drops "Successfully", per en style guide "no successfully"; idiomatic "Đã cập nhật" — 7 shipped uses) · running_sessions_unchanged OK · list_title OK (sentence case per guide) · list_description OK (colon, no space before) · list_unrecognised MINOR m1 · list_desc_hosting OK · list_desc_model_name OK (N1) · list_desc_conversation_length OK · list_desc_detail_length OK · list_desc_max_learnings_history OK · list_desc_classification_notice OK (N2) · list_desc_auto_save_conversation OK · ambiguous OK · to_needs_device OK (N4) · cannot_wait OK · unsupported_flag OK · cli_unavailable OK · deadline_exceeded OK (N3) · cli_no_answer OK.

## Terminology vs shipped (counts of real usage, 147 vi namespaces)
- compaction → "bộ máy nén": 3 shipped uses (the three wire.settings.*.help for the same retired settings) — identical wording; glossary term "nén" (37 uses). Consistent. Note: "bộ máy nén" literally "compression engine"; acceptable and matches shipped, so not flagged.
- DEPRECATED → "ĐÃ LỖI THỜI" ("Đã lỗi thời" in shipped retired-setting help, 5 uses). Consistent. (Shipped title for the retired group is "Đã ngừng dùng" — a different word for a different UI element; not a violation.)
- session phiên; device/peer "thiết bị ngang hàng" (11 shipped incl. cli.sessions_help.move_to); approvals "phê duyệt" (67); config key "khóa cấu hình" (cli.config_help.edit_key/edit_value — the sibling help for the same command); "Gợi ý thông minh" (4, matches settings page); tool "công cụ"; default "mặc định" (95); "tệp cấu hình" (4); "thư mục làm việc" (11); "đường dẫn" (18); "ví dụ:" (6); "Mô tả:" (9); "lệnh gọi" for call (40); "tin nhắn"/"cuộc trò chuyện" consistent.

## Findings
MINOR m1 — cli.config.list_unrecognised
  Current: "khóa không xác định; không có gì đọc khóa này"
  Problem: "không có gì đọc" ("nothing reads") is a word-for-word calque of an English idiom; reads stilted (Vietnamese "không có gì" + active verb reads like "nothing is there reading it"). Also "không xác định" ≈ "unspecified/undetermined" more than "not recognised" (shipped uses "không nhận ra" for that sense: "không nhận ra" ×2 in wire.errors.desktop).
  Suggest: "khóa không được nhận ra; không có thành phần nào đọc khóa này" (or shorter "khóa không được nhận ra; chương trình không đọc khóa này"). Not gating — fully intelligible, and "không xác định" has 11 shipped uses in similar roles.

MINOR m2 — cli.config.cleared (polish, judge optional)
  Current: "Đã xóa giá trị của {key}: cài đặt này lại dùng giá trị mặc định"
  Problem: none in meaning; "lại dùng" is slightly terse. Suggest "Đã xóa giá trị của {key}: cài đặt này trở về giá trị mặc định" (shipped uses "chuyển về mặc định"/"trở về"). Optional; leave as is if the wave is frozen.

NIT n1 — cli.config.list_desc_model_name: "dùng để tương tác" is faithful but vague; "dùng cho các cuộc trò chuyện" would match shipped model_name.help ("Mô hình cho các cuộc trò chuyện mới"). Keep if source fidelity preferred.
NIT n2 — cli.config.list_desc_classification_notice: wording ("không hiện thông báo trong cuộc trò chuyện") differs from the shipped settings-page help for the same setting ("im lặng trong cuộc trò chuyện; lệnh gọi nằm trong nhật ký"). Candidate is MORE faithful (includes cost, session log); the settings page is the one that is lossy. No action on C1; flag for a later settings-namespace sweep.
NIT n3 — tui.session_move.deadline_exceeded: "kết quả chuyển" / "việc chuyển đã đến bước nào" — "chuyển" as a bare noun is elliptical; "kết quả của việc chuyển" is smoother. "hết thời gian chờ" reads as the wait timing out, which is accurate and compatible with the unknown-outcome statement.
NIT n4 — tui.session_move.to_needs_device / ambiguous: multi-word placeholder "<thiết bị ngang hàng>" contains spaces inside the angle brackets; acceptable (context permits; matches shipped term), shorter "<thiết bị|local>" would scan better as a usage string.

## Scope observation (not a translation finding, outside the 22 keys)
tui/app.py ~L37151–37181 (HEAD b188079f32e6) still builds the surrounding /move refusal in hard-coded English: f"Could not move {session_id}: {message}.{tail}" and tails " You are back on {session_id} — it did not move.", " You are back on … this move may already have gone through, so check before asking again.", " Nothing changed.", fallback "the move was refused". After `/move … --to` fails in vi, the translated clause will therefore be wrapped in English. This directly bears on the operator's "no English anywhere" criterion; it is un-keyed, so no auditor can pass it. Recommend manager confirm a slice/key covers it. Not counted against C1.

## Commands run
- python3 key-set/placeholder/backtick/token/NFC/ASCII-residue checks over keys.json vs vi.v2.json (22 keys)
- python3 with local_operator.i18n.runtime parse_message/message_arguments/plural_selectors/render_message (repo ICU parser) on all 22
- grep/regex counts across .worktrees/i18n-c1/local_operator/i18n/catalogues/vi/*.json for the terms above
- read cli.py (config edit/list sites), tui/session_move.py, tui/app.py refusal sites
- scripts/i18n/check.py not run: it validates shipped catalogues only (vi has no cli.config / tui.session_move namespaces yet), so it cannot see the candidate.
- Repo ledger/evidence record NOT written (task asked for scratchpad report only; no tracked commit made).

## Input hashes (sha256)
- keys.json ac35dc123a4241a3c1fdbd0f4f012636d9706695bacfb3601aa06eed2cae1ffc
- vi.v2.json 95b3f7ab7dca0e7c794ab6d5b61877d049cdb3d55a3dc3c5538b42043dbf0738
