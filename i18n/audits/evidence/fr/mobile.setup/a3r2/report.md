# Independent translation re-audit — fr / mobile.setup — a3r2

- PR: #3958 (branch fix/tunnel-remedy-no-commands, stacked on #3945), head 14b76fafa9dcdb3bcaa38e2430f4ea0a591fb3d8; scope: the 12 new `*_plain` keys (values from 01113e5aa, remediated in 14b76faf).
- Prior round: a3 fr audit by session 9054cd229d3f FAILED on FR3-01 (MAJOR, U+0020 before `;` in account_plain, verification_plain, failed_plain) with minors FR3-02..04; no receipt was appended for it.
- Translator: coder subagent of host Local Operator session 25fe8dfbf610 (01113e5aa); FR3 remediation in 14b76faf by a #3958 remediation lane whose session was not disclosed to this auditor; model not disclosed (recorded null).
- Auditor: translation-auditor (lopdev team), independent Local Operator session 375851fcc4e9; neither the translator/coder nor the previous fr auditor; edited no catalogue value; model anthropic/claude-opus-5-5 (session system prompt; no provider fallback during the audit).
- Date: 2026-11-14. Result: **PASSED** — 0 blocker / 0 major; 2 nits, none gating.

## Method
Read side by side with the en source and context sidecar, i18n/glossary.md and i18n/style-guides/fr.md, and against #3945's shipped fr phone-access copy (titles, repoint, consent, reservation keys). Byte checks: U+202F before every `; : ! ?` and inside « » (1 each in account/verification/failed, none needed elsewhere), U+2019 apostrophes only (no U+0027), no placeholders in source or target, NFC, no bidi controls, no command/flag/slash/backtick text (setup._SHELL_COMMAND plus slash/flag scan), vous register only. No maxCells on these keys (wrapping card prose). Runtime: after the receipt and regeneration, set_active_locale('fr') activates fr and setup.plain_remedy({'step': s}) equals the fr catalogue value for all 12 steps (0/12 before the receipt — fr fell back to en).

## Prior findings verified
- FR3-01 fixed: account_plain, verification_plain, failed_plain carry U+202F (bytes e2 80 af) before `;`.
- FR3-02 fixed: stopped_plain "quand vous voudrez".
- FR3-03 fixed: cloudflared_plain "un utilitaire appelé cloudflared".
- FR3-04 fixed: supervisor_plain "maintenir l’accès téléphone actif en arrière-plan".

## Locale notes
vous register throughout; "accès téléphone", "interface téléphone", "service en arrière-plan", "configuration", "approbation", "agent", "compte Radient", "connectez-vous" match the namespace's titles and #3945 copy; "adresse de téléphone" matches mobile.setup.repoint (the older reservation keys' "adresse du/de votre téléphone" drift is pre-existing and not gated here).

## Findings
- FR4-N1 (nit) `mobile.setup.verification_plain`: "Revérifiez dans un instant" → "Vérifiez de nouveau dans un instant". "Revérifier" is attested but colloquial; the analytic form reads more natural in UI prose.
- FR4-N2 (nit) `mobile.setup.failed_plain`: "ne s’est pas terminée … les étapes terminées" echoes the same root; "n’a pas abouti … les étapes déjà effectuées restent enregistrées" would avoid it. Optional — the sibling `mobile.setup.failed` uses the same pair, so keeping it is consistent.

## Keys read (12)
| key | en | fr |
|---|---|---|
| `mobile.setup.stopped_plain` | Phone access is turned off on this computer. Turn it back on whenever you want to reach this computer from your phone. | L’accès téléphone est désactivé sur cet ordinateur. Réactivez-le quand vous voudrez accéder à cet ordinateur depuis votre téléphone. |
| `mobile.setup.owner_missing_plain` | The Radient account that set up phone access is no longer signed in on this computer. Sign in with that account to bring phone access back. | Le compte Radient qui a configuré l’accès téléphone n’est plus connecté sur cet ordinateur. Connectez-vous avec ce compte pour rétablir l’accès téléphone. |
| `mobile.setup.login_plain` | Sign in to Radient with the account that owns phone access, then try again. | Connectez-vous à Radient avec le compte propriétaire de l’accès téléphone, puis réessayez. |
| `mobile.setup.choose_account_plain` | Choose the Radient account phone access should use. Setup never moves an existing phone address to a different account. | Choisissez le compte Radient que l’accès téléphone doit utiliser. La configuration ne transfère jamais une adresse de téléphone existante vers un autre compte. |
| `mobile.setup.account_plain` | Radient could not confirm the account and phone address. Check your internet connection and try again; nothing was changed. | Radient n’a pas pu confirmer le compte et l’adresse de téléphone. Vérifiez votre connexion internet et réessayez ; rien n’a été modifié. |
| `mobile.setup.cloudflared_plain` | Phone access needs a helper app called cloudflared, which is not installed on this computer. Ask your agent to install it (it asks for your approval first), then try again. | L’accès téléphone a besoin d’un utilitaire appelé cloudflared, qui n’est pas installé sur cet ordinateur. Demandez à votre agent de l’installer (il vous demandera d’abord votre approbation), puis réessayez. |
| `mobile.setup.mobile_bundle_plain` | The phone interface is not prepared on this computer yet. Ask your agent to prepare it (it asks for your approval first), then try again. Nothing was changed. | L’interface téléphone n’est pas encore préparée sur cet ordinateur. Demandez à votre agent de la préparer (il vous demandera d’abord votre approbation), puis réessayez. Rien n’a été modifié. |
| `mobile.setup.supervisor_plain` | This computer cannot keep phone access running in the background yet. Ask your agent to set up a background service for it, then try again. Nothing was changed. | Cet ordinateur ne peut pas encore maintenir l’accès téléphone actif en arrière-plan. Demandez à votre agent de configurer un service en arrière-plan pour cela, puis réessayez. Rien n’a été modifié. |
| `mobile.setup.apply_plain` | Everything is ready. Turn on phone access to finish. | Tout est prêt. Activez l’accès téléphone pour terminer. |
| `mobile.setup.verification_plain` | The connection is not confirmed yet. Check again in a moment; completed steps stay saved. | La connexion n’est pas encore confirmée. Revérifiez dans un instant ; les étapes terminées restent enregistrées. |
| `mobile.setup.failed_plain` | Phone access setup did not finish. Try again; completed steps stay saved. | La configuration de l’accès téléphone ne s’est pas terminée. Réessayez ; les étapes terminées restent enregistrées. |
| `mobile.setup.complete_plain` | Open the address above on your phone and sign in with the same Radient account. | Ouvrez l’adresse ci-dessus sur votre téléphone et connectez-vous avec le même compte Radient. |
