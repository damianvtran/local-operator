# Second independent safety audit — ur: PR #4245 working-set transfer copy

Auditor: task lane (lopdev), session 1946542ef662, model **anthropic/claude-opus-5-5**, read from the session header. No fallback.
Independence: I did not author or remediate any audited bytes. I am a different agent from the translators (1768f308a175, 04a689e5090b/b57675c8318a, b7523be856f2, b404dd5868ee, the mechanical-fix job 19556dc3b91a / a69e0e74b) and from the first auditor 482db48afe47 (evidence mt1). I share their model, anthropic/claude-opus-5-5, so this audit is agent-independent only.

Audited head `4ba2f42185ccb51b5e5e452cb261b6cbdc13ef10`; scope: the 22 new cli keys and 6 wire.settings keys this PR adds against 696a18338. The rest of both namespaces is carried by earlier receipts.
Hashes: en cli `6ba71eab00d799d4ff17af1841bb319a15ec0e9c9d684656144c1b1799ab6050`, en wire.settings `562655dc455f9a7dd09cf27ed53608be9795989669c97e070d071a9c57dacda1`, target cli `32d7f97683c25a0103f23e059482f606dba6ea751fd4295c53d8554d7262c112`, target wire.settings `167dcbba61fde9feb128a005fac5e5b6d3613b76943e49ace915aaf5a31ee858`.

## Safety meaning
The safety meaning holds in every key, and the bindings were checked against the en context notes. receipt_sent, receipt_sent_row and receipt_sent_unknown name {device} as the receiver. receipt_sent_unknown says the files WERE sent, and does not say they stayed. receipt_transferred and receipt_restored_row are written from the receiving side ('here'), with {source} as the origin. receipt_pending says {source} still has the files. receipt_left_row and reason.kept_as_confirmed keep the files on the source. import_not_granted keeps the grant remedy correct: the files stay HERE, only the conversation moves, and the command runs on {device} (the destination), verbatim. reason.import_not_granted and destination_declined name the destination. probe_refused_by_policy says 'policy', not a permission. The wire.settings mesh_* keys keep the max-bytes rule (refused, nothing changes) and the roots rule (hidden, system, PATH and node_modules folders never used; unset = restore in place or clone only beside your repositories).

Plurals: receipt_transferred, receipt_pending and receipt_sent were rendered at 0, 1, 2, 5 and 21 (audit.log). Every branch renders its count.

## Findings — 0 blocker, 0 major, 5 minor, 1 nit (non-gating; proposed for deferral in the PR thread)

- **T-ALL-1** (minor) — `cli.sessions_move_workset.receipt_pending`: 'relay' is kept in Latin script. The glossary row 'relay' says translate (relais · relé · 中继 · ретранслятор · chuyển tiếp · ریلے · रिले), and fr follows it. Fix: use the glossary term for this locale
- **T-ALL-2** (minor) — `cli.sessions_move_workset.reason.too_many_worktrees`: 'worktrees' is translated as a calque (arbres de travail / árboles de trabajo / 工作树 / рабочих деревьев / cây làm việc / ورک ٹری / वर्कट्री), but every other key in this namespace keeps Latin 'worktree'. Glossary consistency. Fix: use worktree(s) as in brings_summary and receipt_*
- **T-ur-1** (minor) — `cli.sessions_move_workset.receipt_transferred`: The plural verb 'ہوئے' sits outside the plural block, so count 1 renders '1 worktree یہاں بحال ہوئے' (verified in audit.log). Fix: one {# worktree یہاں بحال ہوا} other {# worktrees یہاں بحال ہوئے}
- **T-ur-2** (minor) — `wire.settings.mesh_move_working_set_roots.help`: 'کاموں سے الگ کیے گئے' reads 'separated by tasks/works', not 'by commas'. The example '~/code, ~/work' shows the format, so the setting stays usable. Fix: کوموں (،) سے الگ کیے گئے
- **T-ur-3** (minor) — `cli.sessions_move_workset.import_not_granted`: 'اس ڈیوائس کو اپنی ورکنگ فائلیں بھیجنے کی اجازت نہیں دی': the reflexive 'اپنی' is ambiguous, and the recipient ('to it') is dropped. The keeping device ('یہیں') and the device to run the grant on ('{device} پر') are both explicit, so the remedy is intact. Fix: اس ڈیوائس کو اسے ورکنگ فائلیں بھیجنے کی اجازت نہیں دی
- **T-ur-4** (nit) — `reason.primary_absent / reason.snapshot_failed / restore_path_refused / roots.*`: The repository term has three spellings: Latin 'repository', 'ریپوزٹری', and 'ریپوزیٹری' (existing keys). Fix: ریپوزیٹری

## Verdict: PASS at the audited bytes — 0 blocker / 0 major.
This is a linguistic audit only. No native-speaker review, rendered frame or RTL paint proof is claimed. No wording was changed.
