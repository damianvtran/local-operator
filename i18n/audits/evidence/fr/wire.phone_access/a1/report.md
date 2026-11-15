# i18n audit a1 — fr / wire.phone_access (branch fix/tunnel-remedies-auto-up @ 1fce7ecb1f5f)

Head audited: 1fce7ecb1f5fb587e7b91ef2f9c7fe658521e0d0 (worktree .worktrees/tunnel-remedies-auto-up, branch fix/tunnel-remedies-auto-up). Date 2026-11-14.
Auditor: i18n-auditor, independent `task` subagent on node lop-remote-node-2, session 110287e9c4f3; not the translator (7af08d89b259) nor the English-drafting coder (b0300e2c2c7f); did not edit any value; model anthropic/claude-opus-5-5 (harness system prompt of session 110287e9c4f3 (`Model: anthropic/claude-opus-5-5`); no provider fallback disclosed during the audit).
Translator: i18n-translator subagent, Local Operator session 7af08d89b259 (values in commit 1fce7ecb1f5f; English source drafted by coder session b0300e2c2c7f); model anthropic/claude-opus-5-5 (as stated by the coordinator brief; not independently verified); 2026-11-14.
Source: en sha256 2979b226db50db768fcad68f5f7376854f2185b08e6b0af08eb8e72bdb470f58 (4 keys); target sha256 d00794f915939230570c0c8474043e3c4bc4c75996ed90bd7862485db13c91a3. Prior rounds on disk: none.

**Verdict: passed** — 0 unresolved blocker/major; 2 minor/nit finding(s).

Scope: the whole namespace at the new English source hash. The 4 keys new in 1fce7ecb1f5f were judged in full; the namespace is new; every key is new.

Read: docs/I18N.md 'Add or extend a locale', i18n/glossary.md (no row for phone access/tunnel/billing/credit), i18n/style-guides/fr.md, en + context sidecar, corpus precedents (mobile.setup.balance/consent/billing_activated, tunnel.status.server_unavailable.*).

## Per-key judgement (new keys)

| key | en | fr | judgement |
|---|---|---|---|
| `wire.phone_access.title` | Phone access | Accès téléphone | OK — «Accès téléphone» matches corpus. |
| `wire.phone_access.description` | How this computer keeps its Radient tunnel for the phone app working. Changes apply to the next sign-in. | Comment cet ordinateur maintient son tunnel Radient pour l’application mobile. Les modifications s’appliquent à la prochaine connexion. | OK; FR-N2 nit («maintient» drops «working»). |
| `wire.phone_access.auto_setup.label` | Reconnect phone access after signing in | Reconnecter l’accès téléphone après la connexion | OK — infinitive toggle label. |
| `wire.phone_access.auto_setup.help` | When you sign in to Radient on a computer that already has phone access, reconnect or repair its tunnel in the background. It never activates billing or spends credit; any price or missing credit is shown to you instead. | Quand vous vous connectez à Radient sur un ordinateur qui dispose déjà de l’accès téléphone, son tunnel est reconnecté ou réparé en arrière-plan. Cela n’active jamais la facturation et ne dépense jamais de crédit ; tout prix ou crédit manquant vous est présenté à la place. | OK — «n’active jamais la facturation et ne dépense jamais de crédit» intact; FR-N3 nit on last clause. |

## Safety meanings

No terminal command is named in any new key (only `/mobile enable`, verbatim, in mobile.setup.login_offer). Money clauses survive where the English has them (nothing charged; setup never purchases credit; never activates billing or spends credit; price shown before billing) — regex presence logged per key in audit.log, meaning judged above. {url} and the console URL are byte-identical to en.

## Findings

| ID | severity | keys | issue | suggested fix |
|---|---|---|---|---|
| FR-N2 | nit | `wire.phone_access.description` | «maintient son tunnel Radient» drops 'working'. | «Comment cet ordinateur maintient en service son tunnel Radient pour l’application mobile.» |
| FR-N3 | nit | `wire.phone_access.auto_setup.help` | «tout prix ou crédit manquant vous est présenté» — 'presenting missing credit' is slightly odd. | «tout prix à accepter ou tout manque de crédit vous est signalé à la place». |

Minor/nit findings are advisory and do not block; the coordinator may batch them into a translator revision.

## Not verified

No native-speaker review, no rendered terminal/desktop/phone frame. Judgement by reasoning only; no MT service used. Translator model identity is as stated by the coordinator, not independently verified.
