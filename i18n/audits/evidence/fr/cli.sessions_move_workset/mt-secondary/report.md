# Second independent safety audit — fr: PR #4245 working-set transfer copy

Auditor: task lane (lopdev), session 1946542ef662, model **anthropic/claude-opus-5-5**, read from the session header. No fallback.
Independence: I did not author or remediate any audited bytes. I am a different agent from the translators (1768f308a175, 04a689e5090b/b57675c8318a, b7523be856f2, b404dd5868ee, the mechanical-fix job 19556dc3b91a / a69e0e74b) and from the first auditor 482db48afe47 (evidence mt1). I share their model, anthropic/claude-opus-5-5, so this audit is agent-independent only.

Audited head `4ba2f42185ccb51b5e5e452cb261b6cbdc13ef10`; scope: the 22 new cli keys and 6 wire.settings keys this PR adds against 696a18338. The rest of both namespaces is carried by earlier receipts.
Hashes: en cli `6ba71eab00d799d4ff17af1841bb319a15ec0e9c9d684656144c1b1799ab6050`, en wire.settings `562655dc455f9a7dd09cf27ed53608be9795989669c97e070d071a9c57dacda1`, target cli `224786f6570eb964ff7cc34e71988c332a30b1280a8b56c8ef82224132d4d40f`, target wire.settings `887445f38b7241b6f4d98fa64ce428ffd46b5ea62bc800ceac8386b846946071`.

## Safety meaning
The safety meaning holds in every key, and the bindings were checked against the en context notes. receipt_sent, receipt_sent_row and receipt_sent_unknown name {device} as the receiver. receipt_sent_unknown says the files WERE sent, and does not say they stayed. receipt_transferred and receipt_restored_row are written from the receiving side ('here'), with {source} as the origin. receipt_pending says {source} still has the files. receipt_left_row and reason.kept_as_confirmed keep the files on the source. import_not_granted keeps the grant remedy correct: the files stay HERE, only the conversation moves, and the command runs on {device} (the destination), verbatim. reason.import_not_granted and destination_declined name the destination. probe_refused_by_policy says 'policy', not a permission. The wire.settings mesh_* keys keep the max-bytes rule (refused, nothing changes) and the roots rule (hidden, system, PATH and node_modules folders never used; unset = restore in place or clone only beside your repositories).

Plurals: receipt_transferred, receipt_pending and receipt_sent were rendered at 0, 1, 2, 5 and 21 (audit.log). Every branch renders its count.

## Findings — 0 blocker, 0 major, 1 minor, 0 nit (non-gating; proposed for deferral in the PR thread)

- **T-ALL-2** (minor) — `cli.sessions_move_workset.reason.too_many_worktrees`: 'worktrees' is translated as a calque (arbres de travail / árboles de trabajo / 工作树 / рабочих деревьев / cây làm việc / ورک ٹری / वर्कट्री), but every other key in this namespace keeps Latin 'worktree'. Glossary consistency. Fix: use worktree(s) as in brings_summary and receipt_*

## Verdict: PASS at the audited bytes — 0 blocker / 0 major.
This is a linguistic audit only. No native-speaker review, rendered frame or RTL paint proof is claimed. No wording was changed.
