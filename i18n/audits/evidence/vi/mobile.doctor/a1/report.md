# vi / mobile.doctor — independent translation audit a1

- Verdict: **changes_requested** — blocker 0 · major 1 · minor 0 · nit 1 (unresolved blocker/major: 1)
- Scope: complete NEW namespace, all 23 keys judged
- Revision: a3722be7dadacaf140a2587a0100bd3b26c2ab50 (branch feat/mobile-doctor, worktree .worktrees/mobile-doctor)
- Source sha256 (en): `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`
- Target sha256 (vi): `2642bc0c30738625eb8b2c99b4410f8d1a9ef1bbef7f20d6ea61bdd4c4042314`
- Translator: i18n-translator subagent, session 398cdb26808f (English source drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14
- Auditor: session 2c1a73eaa6cd, anthropic/claude-opus-5-5, 2026-11-14 — independent; edited no catalogue value

## Findings

### VI-D1 — major (unresolved)
Keys: `mobile.doctor.files.ok`, `mobile.doctor.files.near`

«đang mở {open} tệp trên giới hạn {limit}» reads most naturally as “{open} files open ABOVE the limit {limit}” («trên giới hạn» collocates as over/above the limit; the «X trên Y» = “X out of Y” reading needs the two numbers adjacent). On files.ok — a PASS line pasted into support threads — that inverts the meaning into an alarm; in files.near it contradicts the following «gần chạm giới hạn» (“close to the limit”).

Suggested fix: files.ok: «Dịch vụ truy cập điện thoại đang mở {open} tệp (giới hạn {limit}).» · files.near: «Dịch vụ truy cập điện thoại đang mở {open} tệp (giới hạn {limit}), gần chạm giới hạn, nên các kết nối mới từ điện thoại và việc mở cơ sở dữ liệu có thể thất bại.» (alternative: «{open}/{limit} tệp»). Placeholders unchanged.

### VI-D2 — nit (accepted)
Keys: `mobile.doctor.daemon.not_installed`, `mobile.doctor.daemon.not_running`, `mobile.doctor.daemon.state_not_running`, `mobile.doctor.files.not_running`

«chưa chạy» (not yet running) in the two daemon sentences vs «không chạy» in the label and files.not_running. Both are correct Vietnamese and «chưa» fits a service that is expected to start; noted only for consistency.

Suggested fix: Optional: none required.


## Verified

- “phone service” rendered consistently as «dịch vụ truy cập điện thoại» in all 13 sentences that name it
- no command named in any value; tone calm and factual; gate_open states the unprotected portal plainly without alarmism
- Radient / Local Operator kept as product names; {state}/{reason} left as Latin machine words («đang ở trạng thái {state}»)
- «tunnel» / «trình kết nối» match mobile.setup; «không phản hồi» matches the phone app's own vi label «Không phản hồi» (relay.chrome.session.subtitle.not_answering)
- tunnel.duplicates: vi has only the plural category other; «{count} tunnel» is grammatical for any count
- quotes “ ” per style guide; NFC verified

## Translator's open questions

 Q6 (asleep): «đang ngủ» is the ordinary Vietnamese for a sleeping computer — accepted. Q1–Q5 concern other locales.

## Method and limits

- Every key read against the English value and its context note; glossary `i18n/glossary.md` and `i18n/style-guides/vi.md` applied.
- `audit.log`: placeholder-set parity, NFC and product-renderer output (`local_operator.i18n.runtime.render_message`) for every key with sample arguments, run with `.venv/bin/python`; 0 placeholder/NFC defects.
- LLM review, not native-speaker certification. Text renders only: no terminal, desktop or device frame captured. The phone app's own catalogue (separate repository) was not available, so its exact wording for the “asleep” state was not compared.
- Did not run generate.py or check.py (lane instruction).
