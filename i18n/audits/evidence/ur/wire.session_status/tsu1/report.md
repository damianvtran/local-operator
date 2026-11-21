# First independent Urdu safety-copy audit

Auditor: i18n-auditor (lopdev; terminal-status-ur-audit; first independent instance); session `9075da90042e`; actual model `openai/gpt-6.1-sol`. No fallback. Independent from translator session `7b97856c53df`; no translation edits, translation/MT service, repository writes or history application.

**Verdict: passed — 0 BLOCKER, 0 MAJOR, 0 MINOR, 0 NIT.** This is ONE of the required two independent safety-copy passes. No second-pass, visual/runtime app, mixed-direction rendered-frame, pseudo or shaping certification is claimed.

Frozen worktree: `2930492e05a6968a2e1adfe34a091b94802bacd0`. Scope: all 17 supplied keys, all command and cause select branches and fallbacks. Logical Unicode storage and ICU output inspected; runtime formatting is not app/render proof.

## Mechanical execution

`audit.log` is actual execution output, not notes. `mechanical.json` retains 59 formatted cases. Scoped repo `scripts.i18n.check.check_catalogues()` normal admission returns `[]` using a scratch en+context+ur fixture and the checkout interpreter. No full suite, build, generator, install or server ran.

Counterfactuals: dropped key, added key, renamed placeholder, broken ICU closing brace, and missing other all rejected. Independently checked duplicate JSON keys, exact arguments/types and selector codes, Urdu CLDR one/other completeness, and absence of manual bidi controls. Empty command sentinel none omits attribution; `/stop --all` and `lop sessions stop abc123` survive exactly in other.

Observed validator gap: removing `one` from an existing plural while retaining `other` is accepted by the repo catalogue validator. Supplemental completeness check rejects it. Actual target contains both categories and formats correctly; this gap is not a new translation blocker.

Harness correction disclosure: attempt 1 initially removed only an ordinary closing parenthesis, not ICU syntax; the control was corrected to remove the ICU closing brace. Attempt 2 used numeric equality for Decimal(1.0); this was corrected to use actual CLDR category because visible fractional precision chooses other. Both original attempt logs are retained; neither failed attempt is claimed as evidence of a target defect.

## Per-key native/semantic audit

### `wire.session_status.retired_label` — PASS

Existing frozen value preserved byte-for-byte: {base} stays verbatim and اپ ڈیٹ کے لیے preserves update purpose. ریٹائر کیا گیا is an established loan/calque, not the new formatter wording; no new finding against this unchanged, unused-by-new-formatter key. Rendered with a deliberately Latin base to verify interpolation, not to claim the old composer was localized.

- Params `{"base": "Interrupted"}` → Interrupted — اپ ڈیٹ کے لیے ریٹائر کیا گیا

### `wire.session_status.interrupted` — PASS

درمیان میں رکا is natural terse masculine wording for an implied سیشن; interruption is not completion or deletion.

- Params `{}` → درمیان میں رکا

### `wire.session_status.unseen_interruption` — PASS

Outcome remains interruption. کیفیت is feminine, so نہیں پڑھی گئی agrees correctly; ابھی marks not-yet-read, not unknown outcome. No gender mismatch.

- Params `{}` → درمیان میں رکا — کیفیت ابھی نہیں پڑھی گئی

### `wire.session_status.retired` — PASS

رن ٹائم اپ ڈیٹ کے لیے بند ہوا makes update purpose explicit and does not attribute the stop to the operator.

- Params `{}` → درمیان میں رکا — رن ٹائم اپ ڈیٹ کے لیے بند ہوا

### `wire.session_status.unseen_retirement` — PASS

Same update-driven runtime departure plus a separate unread-status clause; کیفیت/پڑھی گئی agreement correct.

- Params `{}` → درمیان میں رکا — رن ٹائم اپ ڈیٹ کے لیے بند ہوا؛ کیفیت ابھی نہیں پڑھی گئی

### `wire.session_status.signal_stop` — PASS

سگنل سے روکا گیا denotes a signal stop, not a forced kill. Empty none arm has no invented attribution. other adds کمانڈ: followed by the exact command bytes.

- Params `{"command": "none"}` → درمیان میں رکا — سگنل سے روکا گیا
- Params `{"command": "/stop --all"}` → درمیان میں رکا — سگنل سے روکا گیا؛ کمانڈ: /stop --all
- Params `{"command": "lop sessions stop abc123"}` → درمیان میں رکا — سگنل سے روکا گیا؛ کمانڈ: lop sessions stop abc123

### `wire.session_status.unseen_signal_stop` — PASS

Signal outcome and unread-status are separate. Both command arms retain the same behavior as the seen key; command island is unchanged in logical storage.

- Params `{"command": "none"}` → درمیان میں رکا — سگنل سے روکا گیا؛ کیفیت ابھی نہیں پڑھی گئی
- Params `{"command": "/stop --all"}` → درمیان میں رکا — سگنل سے روکا گیا؛ کیفیت ابھی نہیں پڑھی گئی؛ کمانڈ: /stop --all
- Params `{"command": "lop sessions stop abc123"}` → درمیان میں رکا — سگنل سے روکا گیا؛ کیفیت ابھی نہیں پڑھی گئی؛ کمانڈ: lop sessions stop abc123

### `wire.session_status.forced_kill` — PASS

زبردستی بند کیا گیا explicitly adds force and therefore remains distinct from سگنل سے روکا گیا. It describes runtime/session termination, not history deletion; none omits attribution.

- Params `{"command": "none"}` → درمیان میں رکا — زبردستی بند کیا گیا
- Params `{"command": "/stop --all"}` → درمیان میں رکا — زبردستی بند کیا گیا؛ کمانڈ: /stop --all
- Params `{"command": "lop sessions stop abc123"}` → درمیان میں رکا — زبردستی بند کیا گیا؛ کمانڈ: lop sessions stop abc123

### `wire.session_status.unseen_forced_kill` — PASS

Forced termination remains explicit; unread clause neither negates nor replaces it. Exact technical command is preserved in other and omitted in none.

- Params `{"command": "none"}` → درمیان میں رکا — زبردستی بند کیا گیا؛ کیفیت ابھی نہیں پڑھی گئی
- Params `{"command": "/stop --all"}` → درمیان میں رکا — زبردستی بند کیا گیا؛ کیفیت ابھی نہیں پڑھی گئی؛ کمانڈ: /stop --all
- Params `{"command": "lop sessions stop abc123"}` → درمیان میں رکا — زبردستی بند کیا گیا؛ کیفیت ابھی نہیں پڑھی گئی؛ کمانڈ: lop sessions stop abc123

### `wire.session_status.complete` — PASS

مکمل is a native terse status; no added claim about success details or read state.

- Params `{}` → مکمل

### `wire.session_status.unseen_completion` — PASS

مکمل retains completion and کیفیت ابھی نہیں پڑھی گئی independently marks unread status; feminine agreement is correct.

- Params `{}` → مکمل — کیفیت ابھی نہیں پڑھی گئی

### `wire.session_status.recent` — PASS

حالیہ naturally names recency, without asserting running/completed/error state.

- Params `{}` → حالیہ

### `wire.session_status.dormant` — PASS

رکا ہوا describes the stopped session. بیداری/بیداریاں follows the glossary and غیر فعال means inactive, not cancelled or scheduled to fire. Urdu one/other both exist. Integer 1 uses one; 0/2/5/21/55/large/fraction use other. Decimal 1.0 uses other by actual CLDR visible-fraction semantics, unlike integer 1. No outcome/read-state confusion.

- Params `{"count": 0}`; CLDR `other` → رکا ہوا (0 بیداریاں غیر فعال)
- Params `{"count": 1}`; CLDR `one` → رکا ہوا (1 بیداری غیر فعال)
- Params `{"count": 2}`; CLDR `other` → رکا ہوا (2 بیداریاں غیر فعال)
- Params `{"count": 5}`; CLDR `other` → رکا ہوا (5 بیداریاں غیر فعال)
- Params `{"count": 21}`; CLDR `other` → رکا ہوا (21 بیداریاں غیر فعال)
- Params `{"count": 55}`; CLDR `other` → رکا ہوا (55 بیداریاں غیر فعال)
- Params `{"count": 1000000}`; CLDR `other` → رکا ہوا (1,000,000 بیداریاں غیر فعال)
- Params `{"count": "1.5"}`; CLDR `other` → رکا ہوا (1.5 بیداریاں غیر فعال)
- Params `{"count": "1.0"}`; CLDR `other` → رکا ہوا (1 بیداریاں غیر فعال)

### `wire.session_status.error` — PASS

خرابی is an idiomatic product error label with no invented diagnosis.

- Params `{}` → خرابی

### `wire.session_status.unseen_error` — PASS

خرابی retains error; کیفیت ابھی نہیں پڑھی گئی marks unread status rather than a second/new error.

- Params `{}` → خرابی — کیفیت ابھی نہیں پڑھی گئی

### `wire.session_status.error_cause` — PASS

All 13 machine selector arms preserve their English meanings and codes; detailed branch judgments below. Unknown cause exercises other, not an invented translated external diagnostic.

- Params `{"cause": "user_stop"}` → خرابی — صارف نے روکا
- Params `{"cause": "runtime_stall_bound"}` → خرابی — پیش رفت رکے رہنے کی حد کے بعد رن ٹائم خود بند ہوا
- Params `{"cause": "continuation_limit"}` → خرابی — جاری رکھنے کی حد پوری؛ زیرِ انتظار پیغام چھوڑ دیا گیا
- Params `{"cause": "output_limit"}` → خرابی — ماڈل کے آؤٹ پٹ کی حد پوری؛ جاری نہ رہ سکا
- Params `{"cause": "runtime_retired"}` → خرابی — رن ٹائم اپ ڈیٹ کے لیے بند ہوا
- Params `{"cause": "runtime_shutdown"}` → خرابی — باری کے دوران رن ٹائم بند ہو گیا
- Params `{"cause": "runtime_overdue"}` → خرابی — پیش رفت کی حد کے بعد رن ٹائم اپ ڈیٹ کے لیے بند ہوا
- Params `{"cause": "runtime_update_failed"}` → خرابی — اپ ڈیٹ مکمل نہیں ہوئی؛ رن ٹائم نے پہلے سے لوڈ شدہ بلڈ برقرار رکھی
- Params `{"cause": "runtime_killed"}` → خرابی — رن ٹائم باقاعدہ بند ہوئے بغیر غائب ہو گیا؛ روکنے کی درخواست نہیں کی گئی تھی
- Params `{"cause": "install_mid_update"}` → خرابی — باری کے دوران تنصیب بدل دی گئی
- Params `{"cause": "owner_lost"}` → خرابی — باری کے دوران رن ٹائم نے جواب دینا بند کر دیا
- Params `{"cause": "disposed"}` → خرابی — باری کے دوران سیشن ختم کر دیا گیا
- Params `{"cause": "other"}` → خرابی
- Params `{"cause": "external_unknown"}` → خرابی

Branch meaning review:
- `user_stop`: صارف نے روکا explicitly attributes the stop to the user; not confused with update retirement.
- `runtime_stall_bound`: پیش رفت رکے رہنے کی حد ... خود بند ہوا preserves a stalled-progress bound and runtime self-termination; خود retains agency.
- `continuation_limit`: جاری رکھنے کی حد پوری preserves continuation limit; زیرِ انتظار پیغام چھوڑ دیا گیا preserves loss of the pending message, not merely delay.
- `output_limit`: ماڈل کے آؤٹ پٹ کی حد پوری identifies model-output limit; جاری نہ رہ سکا retains inability to continue.
- `runtime_retired`: رن ٹائم اپ ڈیٹ کے لیے بند ہوا preserves update-driven retirement without user blame.
- `runtime_shutdown`: باری کے دوران ... بند ہو گیا preserves termination during the turn; does not add a user request or update reason.
- `runtime_overdue`: پیش رفت کی حد کے بعد ... اپ ڈیٹ کے لیے بند ہوا preserves progress-bound update departure, distinct from stall-bound self-exit.
- `runtime_update_failed`: اپ ڈیٹ مکمل نہیں ہوئی preserves failed/nonfinished update; پہلے سے لوڈ شدہ بلڈ برقرار رکھی explicitly retains the loaded build. بلڈ feminine agreement is idiomatic here; no claim that a new build loaded.
- `runtime_killed`: باقاعدہ بند ہوئے بغیر غائب ہو گیا conveys disappearance without orderly exit; روکنے کی درخواست نہیں کی گئی تھی explicitly retains no stop requested. Not falsely attributed to operator.
- `install_mid_update`: باری کے دوران تنصیب بدل دی گئی preserves replacement of installation during the turn, not a completed update claim.
- `owner_lost`: باری کے دوران ... جواب دینا بند کر دیا preserves nonresponsiveness during the turn, without asserting forced kill.
- `disposed`: سیشن ختم کر دیا گیا says the session was ended/disposed during the turn. It does not say حذف, history, transcript or stored records deleted.
- `other`: Only خرابی (plus separate unread clause in unseen key); no fabricated cause or diagnostic.

### `wire.session_status.unseen_error_cause` — PASS

All 13 cause arms retain the same outcome as error_cause and add a separate grammatically feminine unread-status clause. other remains an unread error without attribution. Detailed branch judgments below.

- Params `{"cause": "user_stop"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ صارف نے روکا
- Params `{"cause": "runtime_stall_bound"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ پیش رفت رکے رہنے کی حد کے بعد رن ٹائم خود بند ہوا
- Params `{"cause": "continuation_limit"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ جاری رکھنے کی حد پوری؛ زیرِ انتظار پیغام چھوڑ دیا گیا
- Params `{"cause": "output_limit"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ ماڈل کے آؤٹ پٹ کی حد پوری؛ جاری نہ رہ سکا
- Params `{"cause": "runtime_retired"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ رن ٹائم اپ ڈیٹ کے لیے بند ہوا
- Params `{"cause": "runtime_shutdown"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ باری کے دوران رن ٹائم بند ہو گیا
- Params `{"cause": "runtime_overdue"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ پیش رفت کی حد کے بعد رن ٹائم اپ ڈیٹ کے لیے بند ہوا
- Params `{"cause": "runtime_update_failed"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ اپ ڈیٹ مکمل نہیں ہوئی؛ رن ٹائم نے پہلے سے لوڈ شدہ بلڈ برقرار رکھی
- Params `{"cause": "runtime_killed"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ رن ٹائم باقاعدہ بند ہوئے بغیر غائب ہو گیا؛ روکنے کی درخواست نہیں کی گئی تھی
- Params `{"cause": "install_mid_update"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ باری کے دوران تنصیب بدل دی گئی
- Params `{"cause": "owner_lost"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ باری کے دوران رن ٹائم نے جواب دینا بند کر دیا
- Params `{"cause": "disposed"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی؛ باری کے دوران سیشن ختم کر دیا گیا
- Params `{"cause": "other"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی
- Params `{"cause": "external_unknown"}` → خرابی — کیفیت ابھی نہیں پڑھی گئی

Branch meaning review:
- `user_stop`: صارف نے روکا explicitly attributes the stop to the user; not confused with update retirement.
- `runtime_stall_bound`: پیش رفت رکے رہنے کی حد ... خود بند ہوا preserves a stalled-progress bound and runtime self-termination; خود retains agency.
- `continuation_limit`: جاری رکھنے کی حد پوری preserves continuation limit; زیرِ انتظار پیغام چھوڑ دیا گیا preserves loss of the pending message, not merely delay.
- `output_limit`: ماڈل کے آؤٹ پٹ کی حد پوری identifies model-output limit; جاری نہ رہ سکا retains inability to continue.
- `runtime_retired`: رن ٹائم اپ ڈیٹ کے لیے بند ہوا preserves update-driven retirement without user blame.
- `runtime_shutdown`: باری کے دوران ... بند ہو گیا preserves termination during the turn; does not add a user request or update reason.
- `runtime_overdue`: پیش رفت کی حد کے بعد ... اپ ڈیٹ کے لیے بند ہوا preserves progress-bound update departure, distinct from stall-bound self-exit.
- `runtime_update_failed`: اپ ڈیٹ مکمل نہیں ہوئی preserves failed/nonfinished update; پہلے سے لوڈ شدہ بلڈ برقرار رکھی explicitly retains the loaded build. بلڈ feminine agreement is idiomatic here; no claim that a new build loaded.
- `runtime_killed`: باقاعدہ بند ہوئے بغیر غائب ہو گیا conveys disappearance without orderly exit; روکنے کی درخواست نہیں کی گئی تھی explicitly retains no stop requested. Not falsely attributed to operator.
- `install_mid_update`: باری کے دوران تنصیب بدل دی گئی preserves replacement of installation during the turn, not a completed update claim.
- `owner_lost`: باری کے دوران ... جواب دینا بند کر دیا preserves nonresponsiveness during the turn, without asserting forced kill.
- `disposed`: سیشن ختم کر دیا گیا says the session was ended/disposed during the turn. It does not say حذف, history, transcript or stored records deleted.
- `other`: Only خرابی (plus separate unread clause in unseen key); no fabricated cause or diagnostic.

## Register, terminology, fit and RTL

Natural educated Urdu, terse status register; no inappropriate familiar address, padding honorifics or Hindi punctuation. سیشن, بیداری, ماڈل and other governed terms align with glossary; رن ٹائم and آؤٹ پٹ are normal technical loans. Unread clauses consistently use feminine کیفیت ... پڑھی گئی. Semicolon ۔/؛ usage separates clauses without moving Latin tokens. Technical command bytes are parameters, not translated literals.

No fixed maxCells/maxChars exists for these keys in the English sidecar; all are one logical line. Unseen cause labels are necessarily long, so narrow-sidebar wrapping/truncation and terminal glyph shaping remain rendered-evidence concerns, not measured overflow findings. No manual bidi marks appear. Logical Latin command islands keep exact order; no arrows or directional glyph hints were introduced. No mixed-direction visual/render proof was attempted.

Unknown nonempty external diagnostics intentionally retain legacy English outside the namespace; still an UNRESOLVED programme follow-up, not localized or waived by this audit. Existing retired_label is unchanged and unused by the new formatter.

## Evidence handling

This bundle is scratch-only per task prohibition on repo writes. Intended canonical import destination: `i18n/audits/evidence/ur/wire.session_status/tsu1/`. The proposed ledger receipt note references that destination; it is NOT claimed tracked or on main. Requesting manager must import the byte-identical bundle and append the receipt, preserving existing dated bundles/history. Audit notes in this report are not execution logs.
