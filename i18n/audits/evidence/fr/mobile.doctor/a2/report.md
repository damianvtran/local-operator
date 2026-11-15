# fr / mobile.doctor — independent translation audit a2 (remediation delta)

- Verdict: **passed** — blocker 0 · major 0 · minor 0 · nit 0 new (unresolved blocker/major: 0)
- Scope: round 2, remediation delta only — 1 changed key(s): `mobile.doctor.daemon.not_running`; the other keys were confirmed byte-identical to the a1-audited values and not re-audited
- Revision: a3722be7dadacaf140a2587a0100bd3b26c2ab50 + uncommitted working-tree remediation (branch feat/mobile-doctor, worktree .worktrees/mobile-doctor, node lop-remote-node-2)
- Source sha256 (en): `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; context `c6baa99bc539ef2d2e03bf7b6b11e83edac37139eede8db82034f5fad98d4614`
- Target sha256 (fr): `b42ef71a8c742deeed0ec9926f7b51177b37fecde464f03a69a739bc4bac17ae` (a1 audited `e1926069a8840b344faf7ffb29566c73b14d64aad2503a3700cead5f94f0c0ad`)
- Prior audit: a1 `i18n/audits/evidence/fr/mobile.doctor/a1/record.json` — passed (auditor session 5e5b8d070b27)
- Translator: i18n-translator subagent, session 8dedc962f71a (remediation of a1 findings); original values by session 398cdb26808f (English source drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14
- Auditor: session 24933ff37418, anthropic/claude-opus-5-5, 2026-11-14 — independent of the translators and the round-1 auditor; edited no catalogue value. Same model as the translator (disclosed).

## Prior findings

| id | severity | round-2 status |
|---|---|---|
| FR-D1 | minor | resolved — daemon.not_running now «jusqu’au démarrage du service» |
| FR-D2 | nit | carried (open non-blocking in a1; key unchanged) |

## New findings

None.

## Changed keys

| key | English | a1 value | a2 value | judgement |
|---|---|---|---|---|
| mobile.doctor.daemon.not_running | The phone service is set up but not running; the phone app shows this computer as asleep until it starts. | Le service d’accès téléphone est configuré mais n’est pas lancé ; l’application mobile affiche cet ordinateur en veille jusqu’à son démarrage. | Le service d’accès téléphone est configuré mais n’est pas lancé ; l’application mobile affiche cet ordinateur en veille jusqu’au démarrage du service. | ok — Resolves FR-D1. «jusqu’au démarrage du service» names the service explicitly, so it can only mean the phone service starting, never the computer. «du service» refers back unambiguously to «Le service d’accès téléphone» at the head of the sentence; the repetition is acceptable for a diagnostic line. Typography kept: U+202F before «;», apostrophes U+2019, final period mirrors English. |

## Verified

- placeholder sets identical to English for every changed key; NFC; key set and order identical (audit.log)
- only the 1 remediation key(s) differ from the a1-audited file; every other key byte-identical
- each changed value matches the a1 suggested fix in meaning; no command named; calm factual tone kept
- U+202F before «;», U+2019 apostrophes only, final period mirrors English

## Ledger

- `i18n/audit-history/fr/mobile.doctor.json`: history sha `7eb0a1167884026de004986675b432a89a042b3ff21e7114a52c9d9805a184e6` → `82c30c4d0658575870e63fe7f70918d320d66ae2e7f5aded0d0fd033cc9ccc6c` (append --dry-run then --apply, exit 0; dryrun.log)

## Method and limits

- Independent LLM review on anthropic/claude-opus-5-5, not native-speaker certification; translator and auditor ran the same model (disclosed).
- Text renders via the product renderer only (audit.log); no terminal, desktop or device frame.
- Round-2 scope: only the remediation delta was judged; unchanged keys were confirmed byte-identical to the a1-audited values, not re-audited.
- generate.py / check.py not run by this lane; nothing committed or pushed.
