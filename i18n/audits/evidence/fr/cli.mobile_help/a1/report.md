# fr/cli.mobile_help — independent i18n audit a1

**Verdict: PASSED.** blocker 0 · major 0 · minor 1 · nit 1; unresolved blocker/major: 0.

- Scope: whole namespace (17 keys) at the new English source hash: the 2 NEW keys (cli.mobile_help.doctor, cli.mobile_help.doctor_json) judged in full; the 15 previously audited keys re-read for consistency with the new keys and with mobile.doctor, and any defect seen is recorded.
- Revision: `a3722be7dadacaf140a2587a0100bd3b26c2ab50` (feat/mobile-doctor, worktree mobile-doctor, node lop-remote-node-2).
- English sha256 `3e1aea77751a6a1286cd0bc525163ac6e90da2f81fd19fff1fc23ae247133267`; context `dab85ceb9e4acf145e434179d866848912a0831bbf827ea196f82c28f855a869`; target `98392767f98a8f09b2c1e29f405ee68bba0692c0bf87462da5423e3da1daafd5`.
- Translator: i18n-translator subagent, session 398cdb26808f (English drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14.
- Auditor: i18n-auditor, independent `task` subagent (C4 i18n-auditor es fr ru), session 5e5b8d070b27; did not translate or edit any value it judged; anthropic/claude-opus-5-5 (served model reported by the session; no provider fallback observed). Same model family as the translator (disclosed).

## Findings

### FR-H1 — minor
- Keys: cli.mobile_help.logs
- Issue: Pre-existing value: «Suivre les logs du daemon et des runtimes en continu» describes --follow; the bare `logs` command prints the last --lines lines (default 100) and exits unless --follow is given (cli.py logs_parser). «Tail» here means «show the end of».
- Suggested fix: «Afficher la fin des logs du daemon et des runtimes».

### FR-H2 — nit
- Keys: cli.mobile_help.password
- Issue: Pre-existing value: «faire tourner le mot de passe» is a calque of «rotate»; understood by technical readers.
- Suggested fix: Optional: «Afficher ou renouveler le mot de passe du portail».


## Translator questions

- Q5 (ru/es daemon.state_not_running fragment): n/a to es/ru only; fr «arrêté» nit FR-D2.
- Q6 («asleep» renderings): «en veille» idiomatic for a sleeping device; not verified against the phone app's own label.

## Per-key judgement

| key | English | fr | judgement |
|---|---|---|---|
| cli.mobile_help.mobile | Phone access: control this machine's lop sessions | Accès téléphone : piloter les sessions lop de cette machine | ok |
| cli.mobile_help.install | Install the daemon, password and LaunchAgent | Installer le daemon, le mot de passe et le LaunchAgent | ok |
| cli.mobile_help.status | Daemon health, gate and live sessions | État du daemon, du contrôle d’accès et des sessions actives | ok |
| cli.mobile_help.daemon_action | {action, select, start{Start} stop{Stop} restart{Restart} other{{action}}} the daemon | {action, select, start{Démarrer} stop{Arrêter} restart{Redémarrer} other{{action}}} le daemon | ok |
| cli.mobile_help.logs | Tail the daemon's and the runtimes' logs | Suivre les logs du daemon et des runtimes en continu | FR-H1 |
| cli.mobile_help.logs_lines | Lines per file (up to two are read) | Lignes par fichier (deux fichiers au plus sont lus) | ok |
| cli.mobile_help.logs_follow | Follow by name, so a log created or rotated while you watch is seen (a descriptor-bound follow misses both) | Suivre par nom, pour voir un log créé ou pivoté pendant la surveillance (un suivi lié à un descripteur rate les deux) | ok |
| cli.mobile_help.password | Show or rotate the portal password | Afficher ou faire tourner le mot de passe du portail | FR-H2 |
| cli.mobile_help.devices | List this computer's push devices, or revoke/unrevoke one | Lister les appareils push de cet ordinateur, ou révoquer ou réautoriser l’un d’eux | ok |
| cli.mobile_help.devices_command | list (default), revoke <device_id>, or unrevoke <device_id> | list (par défaut), revoke <device_id>, ou unrevoke <device_id> | ok |
| cli.mobile_help.devices_device_id | the device_id `lop mobile devices` lists; required by revoke/unrevoke | le device_id que liste `lop mobile devices` ; requis par revoke/unrevoke | ok |
| cli.mobile_help.devices_port | the mobile daemon's port (default: the daemon's own default) | le port du daemon mobile (par défaut : celui du daemon) | ok |
| cli.mobile_help.uninstall | Remove the LaunchAgent | Retirer le LaunchAgent | ok |
| cli.mobile_help.uninstall_purge | Also delete the password | Supprimer aussi le mot de passe | ok |
| cli.mobile_help.serve | Run the daemon (foreground) | Exécuter le daemon (au premier plan) | ok |
| cli.mobile_help.doctor | Read-only diagnostics for phone access (safe to share) | Diagnostic en lecture seule de l’accès téléphone (partageable sans risque) | ok |
| cli.mobile_help.doctor_json | Print the diagnostics as JSON | Afficher le diagnostic au format JSON | ok |

## Mechanical checks

- PASS — key parity (same set and order)
- PASS — ICU argument names identical
- PASS — select branch labels identical
- PASS — no command named that English does not name
- PASS — final period mirrors English
- PASS — fr: U+202F before ; : and inside « » ; apostrophe U+2019 only
- PASS — rendered through Msg.envelope(locale=fr) without error (17/17 keys present in render)
- PASS — no silent English fallback in render

## Not verified

- Native-speaker review; terminal/desktop frames (text render via the runtime only, see rendered.txt); the phone app's own labels (separate repo).
