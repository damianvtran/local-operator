# i18n audit a3 — fr / tunnel.status (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 b368acfb0cdee89c8de5a210fc9956251701a9f3f73c88b1096611d0c1f9063f (21 keys); target sha256 6ec2bf10d1ba2f03ab7949c43ea75c44ad6375e9492549e51bc56ec63e8cd243. Prior rounds on disk: a1, a2.

**Verdict: passed** — 0 unresolved blocker/major; 0 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 8 keys new in 1fce7ecb1f5f were judged in full; the other keys were re-read for consistency only (already passed in earlier rounds; unchanged in this commit).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/fr.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | fr | judgement |
|---|---|---|---|
| `tunnel.status.tunnel_inactive.label` | tunnel not active | tunnel inactif | OK — «tunnel inactif». |
| `tunnel.status.tunnel_inactive.detail` | Radient has paused this tunnel, but its billing is in order. Reconnect phone access in Local Operator on the computer running it; it also keeps retrying by itself. If it stays paused, review the tunnel at {url}. | Radient a suspendu ce tunnel, mais sa facturation est en règle. Reconnectez l’accès téléphone dans Local Operator sur l’ordinateur qui l’héberge ; il continue aussi de réessayer de lui-même. S’il reste suspendu, vérifiez le tunnel sur {url}. | OK — «suspendu … mais sa facturation est en règle»; reconnect in Local Operator; U+202F before ;; {url} verbatim; vous. |
| `tunnel.status.tunnel_disabled.label` | tunnel turned off in the console | tunnel désactivé dans la console | OK. |
| `tunnel.status.tunnel_disabled.detail` | This tunnel is turned off in the Radient console, so remote access stays off until it is turned back on there: {url}. It reconnects by itself once it is on. | Ce tunnel est désactivé dans la console Radient, donc l’accès à distance reste coupé tant que vous ne le réactivez pas sur {url}. Il se reconnecte de lui-même dès qu’il est réactivé. | OK — console-only fix; «tant que vous ne le réactivez pas sur {url}» natural; auto-reconnect. |
| `tunnel.status.billing_inactive.label` | billing not activated | facturation non activée | OK. |
| `tunnel.status.billing_inactive.detail` | Billing for this tunnel has not been activated, so Radient will not connect it. Open phone access setup in Local Operator on the computer running it and accept the monthly price. | La facturation de ce tunnel n’a pas été activée, donc Radient ne le connectera pas. Ouvrez la configuration de l’accès téléphone dans Local Operator sur l’ordinateur qui l’héberge et acceptez le prix mensuel. | OK — «acceptez le prix mensuel» matches corpus «prix mensuel accepté». |
| `tunnel.status.billing_lapsed.label` | billing lapsed | facturation interrompue | OK — «facturation interrompue» is the right call (stopped, not expired); answers translator question: endorsed. |
| `tunnel.status.billing_lapsed.detail` | This tunnel's billing lapsed because the Radient balance ran out, so Radient will not connect it. Add credit at {url}; it reconnects by itself once the balance is restored. | La facturation de ce tunnel est interrompue, car le solde Radient est épuisé ; Radient ne le connectera donc pas. Ajoutez du crédit sur {url} ; il se reconnecte de lui-même dès que le solde est rétabli. | OK — cause «le solde Radient est épuisé», fix «Ajoutez du crédit sur {url}»; U+202F x2. |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

None.

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
