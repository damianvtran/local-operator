# i18n audit a1 — vi / tunnel.status (PR #3931)

Head audited: e95a77a5f3ac5266d24c2bc317a8710d26817d60 (worktree .worktrees/mobile-tunnel-resilience, round 2). Date 2026-11-13.
Auditor: i18n-auditor (lopdev team), independent `task` subagent, session 5469a14c00f5; did not translate or edit any value it judged; model anthropic/claude-opus-5-5 (harness system prompt of session 5469a14c00f5 (`Model: anthropic/claude-opus-5-5`); no provider fallback was disclosed during the audit).
Translator: coder subagent (lopdev team) remediating PR #3931 round 1 — not an i18n-translator lane; its session id is named in neither the manager brief nor commit e95a77a5f3ac; anthropic/claude-opus-5-5 (as stated by the manager brief; not independently verified); 2026-11-13. Not an i18n-translator lane — recorded as found.

**Verdict: changes_requested** — 2 unresolved blocker/major.

NFC, bạn-free phrasing, trình kết nối matches corpus. Fails on tunnel terminology and the gateway/port collision.

Read: docs/I18N.md L262-310, i18n/glossary.md, i18n/style-guides/vi.md, en + context sidecar, all 13 target values; corpus cross-checks in audit.log.

## Findings

### VI-01 — MAJOR
- Keys: `create.conflict_unmatched`
- Current: đường hầm (×2)
- Problem: Shipped vi corpus keeps «tunnel» (16 occurrences, 0 «đường hầm»), including mobile.setup in the same `lop tunnel` flow («tunnel hiện có»). Two names for one product noun.
- Suggested: «Radient cho biết đã có một tunnel cho yêu cầu này, nhưng danh sách tunnel của lần đăng nhập này không có bản ghi khớp duy nhất. Hãy kiểm tra lop tunnel list, rồi kết nối bằng lop tunnel connect --tunnel-id <id>.»

### VI-02 — MAJOR
- Keys: `gateway.listening_http_status, gateway.listening_timed_out, gateway.listening_unreadable`
- Current: cổng chuyển tiếp (gateway)
- Problem: «cổng» is also "port": http_status reads "listening on the relay port … another program holds the port, or the relay port has trouble", collapsing the diagnosis (port occupant vs unhealthy gateway) the sentence exists to separate.
- Suggested: http_status: «Có chương trình đang lắng nghe trên cổng 127.0.0.1:{port} của gateway chuyển tiếp, nhưng nó trả lời kiểm tra tình trạng bằng HTTP {status}: có thể một chương trình khác đang chiếm cổng, hoặc gateway đang gặp sự cố. Cài đặt lại sẽ không giúp được.» · timed_out/unreadable: begin «Gateway chuyển tiếp cục bộ đang lắng nghe trên 127.0.0.1:{port}, …» (rest unchanged); record «gateway» kept-English in the vi guide.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame, no RTL render (ur). Judgement by reasoning only; no MT service used.

This bundle is STAGED in session scratch; it is not on any branch until the implementer commits it at i18n/audits/evidence/vi/tunnel.status/a1/.
