# Second independent safety audit — hi: cli.sessions_move_workset (65) + tui.session_move_workset (11)

Auditor: task lane (lopdev, move-working-set), session 1946542ef662, model **anthropic/claude-opus-5-5**, read from the session header; no provider fallback observed.
Independence: No authorship or remediation of any audited catalogue/context bytes; a different agent/session from every translator (dc12534ff2ca, 8966047148fb, ddcac83c68a5, d84367798211) and from the first auditor 482db48afe47. SAME MODEL (anthropic/claude-opus-5-5) as translators dc12534ff2ca/8966047148fb/ddcac83c68a5 and the first auditor: agent-independent only, NOT model-independent.

Audited head: `c5bc5896186cad939fa813d321c525f78ad4e8cf`. en sha256: cli `cede2b5900e0a1884b33b2a525a25aac1db76e39962e7b7fe0d8b50a49cac4dd`, tui `966bc7111580843e30b5f6e845a7bc46f7f7fb6cfc52c51c130b56ea92bc933a`; target sha256: cli `2279517b85d27307d2e8efd8c7d42e88fd2734167b588e508b5c6eb811b093c9`, tui `b3bf08112dabd6bbd429937f26fe1c15eac328a6d7d0674ec9530a440936eb5d`.

Classification: high-visibility/safety copy (consent to send working files to another device, or to leave them behind). This is the SECOND independent audit required by docs/I18N.md step 3. The first audit is `mw1` + `mw2` (session 482db48afe47). Every one of the 76 keys was read against the en source and its context note, en meaning first.

## Safety meaning

For which device receives or keeps the files: {device} is the destination in the sends/probe copy; {source} (or, in the receipt_* and unsupported* keys, {device} bound to the keeper, which was verified in local_operator/network/workset_text.py) is the keeper. Each translation keeps that binding, and no locale calls {source} 'this device' or 'here'.
hi: warning_sends 'वहाँ भेज दी जाएँगी'; warning_stays 'अभी … साथ नहीं जातीं। वे {source} पर ही रहती हैं'; withheld 'रोका गया' (unambiguous); checkbox_sends 'मुझे भरोसा है कि {device} को ये फ़ाइलें मिलनी चाहिए'; the refusals say 'कुछ नहीं ले जाया गया' plus the remedy; danda is used; probe_asked 'सिर्फ़ … कोई पाथ या ब्रांच नहीं'.

Plural branches were rendered at 0, 1, 2, 5, 21 and 1 000 000 through local_operator.i18n.runtime (see audit.log). Every branch rendered and carried its count.
Truncation: the TUI checkbox wraps (`_WrappingCheckbox`), so it never ellipsizes. The longest button label in any locale is fr confirm_without at 37 cells (en: 30), inside an 86-column card. I found no fit risk, but this is a linguistic estimate: no rendered frame was taken.

## Findings (this audit) — 0 blocker, 0 major, 0 minor, 3 nit; all non-gating, proposed for deferral in the PR thread

- **S-hi-1** (nit) — `tui.session_move_workset.title_sends vs cli.sessions_move_workset.prompt_sends`: The TUI title uses '{device} पर भेजें' and the CLI prompt uses '{device} को भेजें' for the same action. Both are grammatical, but they are inconsistent. Fix: use '{device} को भेजें' in title_sends
- **S-hi-2** (nit) — `cli.sessions_move_workset.brings_summary, receipt_not_transferred`: CLDR hi 'one' covers 0, so count 0 renders singular ('0 worktree … पर ही है'). That is CLDR-correct and acceptable; recorded only because it was exercised. Fix: none
- **S-ALL-1** (nit) — `tui.session_move_workset.checkbox_without`: en 'I understand ANY working files stay on {source}' is shown when the files could not be listed; the translation says 'the working files stay', presupposing files exist. What the box commits the user to (whatever exists stays on {source}) is intact. Fix: मुझे पता है कि जो भी वर्किंग फ़ाइलें हैं वे {source} पर ही रहेंगी

I concur with the first audit's non-gating findings: 'मूव' noun loan nit.

## Verdict: PASS at the audited bytes — 0 blocker / 0 major unresolved.

This is a linguistic/copy audit only. No native-speaker review, rendered TUI/desktop frame or RTL paint proof is claimed. I changed no wording, and translations were not edited.
