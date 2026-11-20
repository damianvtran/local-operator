# Second independent safety audit — ur: cli.sessions_move_workset (65) + tui.session_move_workset (11)

Auditor: task lane (lopdev, move-working-set), session 1946542ef662, model **anthropic/claude-opus-5-5**, read from the session header; no provider fallback observed.
Independence: No authorship or remediation of any audited catalogue/context bytes; a different agent/session from every translator (dc12534ff2ca, 8966047148fb, ddcac83c68a5, d84367798211) and from the first auditor 482db48afe47. SAME MODEL (anthropic/claude-opus-5-5) as translators dc12534ff2ca/8966047148fb/ddcac83c68a5 and the first auditor: agent-independent only, NOT model-independent.

Audited head: `c5bc5896186cad939fa813d321c525f78ad4e8cf`. en sha256: cli `cede2b5900e0a1884b33b2a525a25aac1db76e39962e7b7fe0d8b50a49cac4dd`, tui `966bc7111580843e30b5f6e845a7bc46f7f7fb6cfc52c51c130b56ea92bc933a`; target sha256: cli `94dffee72b7946ecf19311680d0ee7cd96dc879e2633098d7cbe8baf3c89bfc7`, tui `ef94361e2912a8267223656723e79e48d7fd94b0737272735590679f70b6f9f3`.

Classification: high-visibility/safety copy (consent to send working files to another device, or to leave them behind). This is the SECOND independent audit required by docs/I18N.md step 3. The first audit is `mw1` + `mw2` (session 482db48afe47). Every one of the 76 keys was read against the en source and its context note, en meaning first.

## Safety meaning

For which device receives or keeps the files: {device} is the destination in the sends/probe copy; {source} (or, in the receipt_* and unsupported* keys, {device} bound to the keeper, which was verified in local_operator/network/workset_text.py) is the keeper. Each translation keeps that binding, and no locale calls {source} 'this device' or 'here'.
ur: warning_sends 'وہاں بھیج دی جائیں گی'; warning_stays 'ابھی … ساتھ نہیں جاتیں۔ وہ {source} پر رہتی ہیں'; withheld 'روکا گیا' = stopped/held (unambiguous); checkbox_sends 'مجھے یقین ہے کہ {device} کو یہ فائلیں ملنی چاہییں'; the refusals say 'کچھ منتقل نہیں ہوا' plus the remedy; RTL punctuation uses ۔ and ، with no danda; placeholders, flags and [y/N] are Latin runs.

Plural branches were rendered at 0, 1, 2, 5, 21 and 1 000 000 through local_operator.i18n.runtime (see audit.log). Every branch rendered and carried its count.
Truncation: the TUI checkbox wraps (`_WrappingCheckbox`), so it never ellipsizes. The longest button label in any locale is fr confirm_without at 37 cells (en: 30), inside an 86-column card. I found no fit risk, but this is a linguistic estimate: no rendered frame was taken.

## Findings (this audit) — 0 blocker, 0 major, 0 minor, 3 nit; all non-gating, proposed for deferral in the PR thread

- **S-ur-1** (nit) — `cli.sessions_move_workset.probe_remedy`: ڈیوائس is feminine elsewhere in this namespace ('دوسری ڈیوائس … نہیں کر سکتی', reason.unsupported_by_peer) but masculine here ('رسائی دینے والے ڈیوائس'). Fix: رسائی دینے والی ڈیوائس پر یہ چلائیں
- **S-ur-2** (nit) — `cli.sessions_move_workset.reason.destination_conflict`: 'یہ راستہ' = 'this path'; en says 'that path'. Folds into the first audit's راستہ→پاتھ minor. Fix: وہ پاتھ منزل پر پہلے سے استعمال میں ہے
- **S-ALL-1** (nit) — `tui.session_move_workset.checkbox_without`: en 'I understand ANY working files stay on {source}' is shown when the files could not be listed; the translation says 'the working files stay', presupposing files exist. What the box commits the user to (whatever exists stays on {source}) is intact. Fix: مجھے معلوم ہے کہ جو بھی ورکنگ فائلیں ہیں وہ {source} پر ہی رہیں گی

I concur with the first audit's non-gating findings: اسناد vs کریڈینشل, راستہ vs پاتھ.

## Verdict: PASS at the audited bytes — 0 blocker / 0 major unresolved.

This is a linguistic/copy audit only. No native-speaker review, rendered TUI/desktop frame or RTL paint proof is claimed. I changed no wording, and translations were not edited.
