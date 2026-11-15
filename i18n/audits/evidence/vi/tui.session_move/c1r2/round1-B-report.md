# c1-au-vi-B — independent audit, locale vi, 22 keys (cli.config.* 15, tui.session_move.* 7)
Auditor: i18n-auditor (c1-au-vi-B), model anthropic/claude-sonnet-5-5. Translator: openai/gpt-6.1-sol.
Verdict: clean-with-findings — 0 BLOCKER, 0 MAJOR, 3 MINOR, 2 NIT (passed-eligible).

Mechanical (repo .venv python + rich.cells.cell_len): key set/order identical (22/22); no empties; none English-identical; all NFC; placeholders identical per key; backtick spans byte-identical; /approvals auto, /move, --to, --keep, --queue, --drop-session-roles, --wait, `local`, `|` verbatim; angle-bracket counts match; list_title 17 cells; longest line 209 cells (wraps by design).

Safety claims preserved: running_sessions_unchanged (running sessions unchanged / config write cannot loosen / per-session /approvals auto / new sessions at auto); cleared = back to default; deadline_exceeded says neither failed nor succeeded (see m2).

m1 MINOR cli.config.list_desc_{conversation_length,detail_length,max_learnings_history}: "thay bằng cơ chế nén" vs shipped wire.settings.*.help "Được thay thế bởi bộ máy nén" -> use "[ĐÃ LỖI THỜI — được thay bằng bộ máy nén]".
m2 MINOR tui.session_move.deadline_exceeded: "chưa chuyển xong … để xem tiến độ" implies still running; outcome is unknown -> "việc chuyển không hoàn tất trong thời gian cho phép; chạy lại `lop sessions move {session} --to {to}` để biết nó đã đến bước nào".
m3 MINOR tui.session_move.cannot_wait: "trình bao" appears nowhere in shipped vi; shipped keeps "shell" -> "chạy từ shell:".
n1 NIT cli.config.list_unrecognised: "không thành phần nào đọc khóa này" stiff -> "khóa không xác định; không có gì đọc khóa này".
n2 NIT cli.config.list_desc_hosting: "Nền tảng cung cấp AI" -> "Nền tảng nhà cung cấp AI (ví dụ: …)" (shipped "nhà cung cấp").
Register fine: no bạn, no "!", sentence case, "tệp cấu hình" consistent with cli.config_help, success lines terse.
