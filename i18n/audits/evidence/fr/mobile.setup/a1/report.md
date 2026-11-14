# i18n audit a1 — fr / mobile.setup (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 6883f92e94dfcec4be0af6041a0531c974fbf18294f7094a132cc555a2ad2065 (86 keys); target sha256 e980df13bc600a38ee50479d0c23e60cb981254aa390457182ee21058844c4ec. Prior rounds on disk: frbk1, r1.

**Verdict: passed** — 0 unresolved blocker/major; 1 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 7 keys new in 1fce7ecb1f5f were judged in full; the other keys were re-read for consistency only (already passed in earlier rounds; unchanged in this commit).

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/fr.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | fr | judgement |
|---|---|---|---|
| `mobile.setup.disabled` | This tunnel is turned off in the Radient console. Turn it back on at https://console.radienthq.com/dashboard/tunnels; it reconnects by itself, and setup will not override that choice. | Ce tunnel est désactivé dans la console Radient. Réactivez-le sur https://console.radienthq.com/dashboard/tunnels ; il se reconnecte de lui-même, et la configuration ne passera pas outre ce choix. | OK — «ne passera pas outre ce choix» idiomatic; URL verbatim; U+202F. |
| `mobile.setup.billing_lapsed` | This tunnel's billing lapsed because the Radient balance ran out. Add credit at https://console.radienthq.com/dashboard/tunnels; setup never purchases credit, and the tunnel reconnects by itself once the balance is restored. | La facturation de ce tunnel est interrompue, car le solde Radient est épuisé. Ajoutez du crédit sur https://console.radienthq.com/dashboard/tunnels ; la configuration n’achète jamais de crédit, et le tunnel se reconnecte de lui-même dès que le solde est rétabli. | OK — «n’achète jamais de crédit» matches mobile.setup.balance. |
| `mobile.setup.reconnect` | Radient has paused this tunnel, but its billing is in order. Reconnect to bring phone access back; nothing is charged. | Radient a suspendu ce tunnel, mais sa facturation est en règle. Reconnectez-le pour rétablir l’accès téléphone ; rien n’est facturé. | OK — «rien n’est facturé». |
| `mobile.setup.network` | This computer could not reach Radient (its network may have changed or dropped). Phone access reconnects by itself once the connection is back; nothing needs doing. | Cet ordinateur n’a pas pu joindre Radient (son réseau a peut-être changé ou été coupé). L’accès téléphone se reconnecte de lui-même dès que la connexion revient ; vous n’avez rien à faire. | OK — «vous n’avez rien à faire». |
| `mobile.setup.radient_unavailable` | Radient answered with a temporary server error. Nothing is wrong with this computer or its sign-in; phone access reconnects by itself once Radient recovers. | Radient a répondu par une erreur serveur temporaire. Ni cet ordinateur ni sa connexion ne sont en cause ; l’accès téléphone se reconnecte de lui-même dès que Radient est rétabli. | OK; FR-N1 nit («sa connexion» can be read as network connection). |
| `mobile.setup.reconnecting` | Phone access is reconnecting in the background after your sign-in. Nothing is charged; this clears by itself in a minute or two. | L’accès téléphone se reconnecte en arrière-plan après votre connexion. Rien n’est facturé ; ce message disparaîtra de lui-même d’ici une à deux minutes. | OK — «Rien n’est facturé»; «ce message disparaîtra de lui-même» natural. |
| `mobile.setup.login_offer` | Reach this computer from the phone app: run /mobile enable. Any monthly price is shown to you before billing starts. | Accédez à cet ordinateur depuis l’application mobile : exécutez /mobile enable. Tout prix mensuel vous est présenté avant le début de la facturation. | OK — `/mobile enable` verbatim; U+202F before :; «Tout prix mensuel vous est présenté avant le début de la facturation». |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| FR-N1 | nit | `mobile.setup.radient_unavailable` | «Ni cet ordinateur ni sa connexion» — «connexion» here can be read as the network connection rather than the Radient sign-in (either reading is true for a 5xx, so meaning is not harmed). | «Ni cet ordinateur ni votre connexion à Radient ne sont en cause». |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
