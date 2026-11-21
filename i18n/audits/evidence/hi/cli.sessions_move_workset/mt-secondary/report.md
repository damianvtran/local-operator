# Second independent safety audit — hi: PR #4245 working-set transfer copy

Auditor: task lane (lopdev), session 1946542ef662, model **anthropic/claude-opus-5-5**, read from the session header. No fallback.
Independence: I did not author or remediate any audited bytes. I am a different agent from the translators (1768f308a175, 04a689e5090b/b57675c8318a, b7523be856f2, b404dd5868ee, the mechanical-fix job 19556dc3b91a / a69e0e74b) and from the first auditor 482db48afe47 (evidence mt1). I share their model, anthropic/claude-opus-5-5, so this audit is agent-independent only.

Audited head `4ba2f42185ccb51b5e5e452cb261b6cbdc13ef10`; scope: the 22 new cli keys and 6 wire.settings keys this PR adds against 696a18338. The rest of both namespaces is carried by earlier receipts.
Hashes: en cli `6ba71eab00d799d4ff17af1841bb319a15ec0e9c9d684656144c1b1799ab6050`, en wire.settings `562655dc455f9a7dd09cf27ed53608be9795989669c97e070d071a9c57dacda1`, target cli `c88840be4aafce014c7d91ccf7451a825ca94eba3cf6ca97be0a4077415e203a`, target wire.settings `d2bd83636903d7aeabab744f57fa2af83da033432432a26a27065d2a5ad83090`.

## Safety meaning
The safety meaning holds in every key, and the bindings were checked against the en context notes. receipt_sent, receipt_sent_row and receipt_sent_unknown name {device} as the receiver. receipt_sent_unknown says the files WERE sent, and does not say they stayed. receipt_transferred and receipt_restored_row are written from the receiving side ('here'), with {source} as the origin. receipt_pending says {source} still has the files. receipt_left_row and reason.kept_as_confirmed keep the files on the source. import_not_granted keeps the grant remedy correct: the files stay HERE, only the conversation moves, and the command runs on {device} (the destination), verbatim. reason.import_not_granted and destination_declined name the destination. probe_refused_by_policy says 'policy', not a permission. The wire.settings mesh_* keys keep the max-bytes rule (refused, nothing changes) and the roots rule (hidden, system, PATH and node_modules folders never used; unset = restore in place or clone only beside your repositories).

Plurals: receipt_transferred, receipt_pending and receipt_sent were rendered at 0, 1, 2, 5 and 21 (audit.log). Every branch renders its count.

## Findings — 0 blocker, 0 major, 4 minor, 1 nit (non-gating; proposed for deferral in the PR thread)

- **T-ALL-1** (minor) — `cli.sessions_move_workset.receipt_pending`: 'relay' is kept in Latin script. The glossary row 'relay' says translate (relais · relé · 中继 · ретранслятор · chuyển tiếp · ریلے · रिले), and fr follows it. Fix: use the glossary term for this locale
- **T-ALL-2** (minor) — `cli.sessions_move_workset.reason.too_many_worktrees`: 'worktrees' is translated as a calque (arbres de travail / árboles de trabajo / 工作树 / рабочих деревьев / cây làm việc / ورک ٹری / वर्कट्री), but every other key in this namespace keeps Latin 'worktree'. Glossary consistency. Fix: use worktree(s) as in brings_summary and receipt_*
- **T-hi-1** (minor) — `cli.sessions_move_workset.receipt_transferred`: The plural verb 'हुए' sits outside the plural block, so count 1 renders '1 worktree यहाँ रीस्टोर हुए' (verified in audit.log). Fix: one {# worktree यहाँ रीस्टोर हुआ} other {# worktrees यहाँ रीस्टोर हुए}
- **T-hi-2** (minor) — `cli.sessions_move_workset.import_not_granted`: 'इस डिवाइस को अपनी वर्किंग फ़ाइलें भेजने की अनुमति': the reflexive 'अपनी' is ambiguous, and the recipient ('to it') is dropped. 'यहीं रहेंगी' and 'run on {device}' are explicit, so the remedy is intact. Fix: इस डिवाइस को उसे वर्किंग फ़ाइलें भेजने की अनुमति
- **T-hi-3** (nit) — `wire.settings.mesh_move_working_set_roots.help; reason.primary_absent/snapshot_failed`: The help text uses 'बहाल' where the receipts use 'रीस्टोर', and Latin 'repository' appears beside 'रिपॉज़िटरी'. Fix: रीस्टोर / रिपॉज़िटरी

## Verdict: PASS at the audited bytes — 0 blocker / 0 major.
This is a linguistic audit only. No native-speaker review, rendered frame or RTL paint proof is claimed. No wording was changed.
