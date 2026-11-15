# fr/mobile.doctor — independent i18n audit a1

**Verdict: PASSED.** blocker 0 · major 0 · minor 1 · nit 1; unresolved blocker/major: 0.

- Scope: complete NEW namespace, all 23 keys judged semantically.
- Revision: `a3722be7dadacaf140a2587a0100bd3b26c2ab50` (feat/mobile-doctor, worktree mobile-doctor, node lop-remote-node-2).
- English sha256 `0ca00b276fdea066228394e3fee336042b1e149c9e7df3ea31be11a24f0beb72`; context `c6baa99bc539ef2d2e03bf7b6b11e83edac37139eede8db82034f5fad98d4614`; target `e1926069a8840b344faf7ffb29566c73b14d64aad2503a3700cead5f94f0c0ad`.
- Translator: i18n-translator subagent, session 398cdb26808f (English drafted by coder session b0300e2c2c7f), anthropic/claude-opus-5-5, 2026-11-14.
- Auditor: i18n-auditor, independent `task` subagent (C4 i18n-auditor es fr ru), session 5e5b8d070b27; did not translate or edit any value it judged; anthropic/claude-opus-5-5 (served model reported by the session; no provider fallback observed). Same model family as the translator (disclosed).

## Findings

### FR-D1 — minor
- Keys: mobile.doctor.daemon.not_running
- Issue: «affiche cet ordinateur en veille jusqu’à son démarrage»: the possessive «son» binds most naturally to the nearest noun «cet ordinateur», so it reads as «until the computer starts» and can suggest rebooting the computer instead of starting the phone service.
- Suggested fix: «…l’application mobile affiche cet ordinateur en veille jusqu’au démarrage du service.»

### FR-D2 — nit
- Keys: mobile.doctor.daemon.state_not_running
- Issue: «arrêté» (stopped) for «not running», while the namespace otherwise says «n’est pas lancé»; correct for a state label, slightly different nuance (a never-started service is not «arrêté»).
- Suggested fix: Optional: «non lancé».


## Translator questions

- Q5 (ru/es daemon.state_not_running fragment): n/a to es/ru only; fr «arrêté» nit FR-D2.
- Q6 («asleep» renderings): «en veille» idiomatic for a sleeping device; not verified against the phone app's own label.

## Per-key judgement

| key | English | fr | judgement |
|---|---|---|---|
| mobile.doctor.versions.ok | Local Operator {lop} on Python {python} ({platform}). | Local Operator {lop} sur Python {python} ({platform}). | ok |
| mobile.doctor.daemon.not_installed | The phone service is not set up on this computer, so it is not running. | Le service d’accès téléphone n’est pas configuré sur cet ordinateur, il n’est donc pas lancé. | ok |
| mobile.doctor.daemon.not_running | The phone service is set up but not running; the phone app shows this computer as asleep until it starts. | Le service d’accès téléphone est configuré mais n’est pas lancé ; l’application mobile affiche cet ordinateur en veille jusqu’à son démarrage. | FR-D1 |
| mobile.doctor.daemon.ok | The phone service is running and answered in {ms} ms. | Le service d’accès téléphone est lancé et a répondu en {ms} ms. | ok |
| mobile.doctor.daemon.slow | The phone service is running but took {ms} ms to answer, which the phone app may show as not answering. | Le service d’accès téléphone est lancé mais a mis {ms} ms à répondre ; l’application mobile peut alors indiquer que cet ordinateur ne répond pas. | ok |
| mobile.doctor.daemon.gate_open | The phone service answered without asking for its password: the phone portal is unprotected. | Le service d’accès téléphone a répondu sans demander son mot de passe : le portail téléphone n’est pas protégé. | ok |
| mobile.doctor.daemon.state_not_running | not running | arrêté | FR-D2 |
| mobile.doctor.files.ok | The phone service has {open} of {limit} files open. | Le service d’accès téléphone a {open} fichiers ouverts sur {limit}. | ok |
| mobile.doctor.files.near | The phone service has {open} of {limit} files open, close to its limit, so new phone connections and database opens can fail. | Le service d’accès téléphone a {open} fichiers ouverts sur {limit}, près de sa limite : les nouvelles connexions du téléphone et les ouvertures de base de données peuvent échouer. | ok |
| mobile.doctor.files.unknown | The phone service's open files could not be counted on this computer. | Les fichiers ouverts du service d’accès téléphone n’ont pas pu être comptés sur cet ordinateur. | ok |
| mobile.doctor.files.status_line | {open} open of {limit} | {open} ouverts sur {limit} | ok |
| mobile.doctor.logs.clean | No out-of-files or database-open errors in the last day of logs. | Aucune erreur « trop de fichiers ouverts » ni erreur d’ouverture de base de données dans les logs de la dernière journée. | ok |
| mobile.doctor.logs.errors | The last day of logs shows out-of-files errors ({emfile}) and database-open errors ({sqlite}); the phone service may be running out of files. | Les logs de la dernière journée montrent des erreurs « trop de fichiers ouverts » ({emfile}) et des erreurs d’ouverture de base de données ({sqlite}) ; le service d’accès téléphone approche peut-être de sa limite de fichiers ouverts. | ok |
| mobile.doctor.tunnel.not_configured | Phone access through Radient is not set up on this computer. | L’accès téléphone via Radient n’est pas configuré sur cet ordinateur. | ok |
| mobile.doctor.tunnel.ok | The tunnel is set up and its connector is {state}. | Le tunnel est configuré et son connecteur est à l’état {state}. | ok |
| mobile.doctor.tunnel.problem | The tunnel is set up but its connector is {state} ({reason}). | Le tunnel est configuré mais son connecteur est à l’état {state} ({reason}). | ok |
| mobile.doctor.tunnel.single | This Radient account has one tunnel for this computer. | Ce compte Radient a un seul tunnel pour cet ordinateur. | ok |
| mobile.doctor.tunnel.duplicates | This Radient account has {count} tunnels for this computer; the phone app may be connecting to an old one. | Ce compte Radient a {count} tunnels pour cet ordinateur ; l’application mobile se connecte peut-être à un ancien tunnel. | ok |
| mobile.doctor.tunnel.list_unchecked | The account's tunnel list could not be read, so duplicate tunnels were not checked. | La liste des tunnels du compte n’a pas pu être lue ; les tunnels en double n’ont donc pas été vérifiés. | ok |
| mobile.doctor.login.ok | Signed in to Radient. | Connecté à Radient. | ok |
| mobile.doctor.login.missing | Not signed in to Radient. | Non connecté à Radient. | ok |
| mobile.doctor.files.not_running | The phone service is not running, so it has no open files to count. | Le service d’accès téléphone n’est pas lancé ; il n’a donc aucun fichier ouvert à compter. | ok |
| mobile.doctor.daemon.not_answering | The phone service is running but not answering; the phone app shows this computer as not answering. It may have run out of files. | Le service d’accès téléphone est lancé mais ne répond pas ; l’application mobile affiche cet ordinateur comme ne répondant pas. Il a peut-être atteint sa limite de fichiers ouverts. | ok |

## Mechanical checks

- PASS — key parity (same set and order)
- PASS — ICU argument names identical
- PASS — select branch labels identical
- PASS — no command named that English does not name
- PASS — final period mirrors English
- PASS — fr: U+202F before ; : and inside « » ; apostrophe U+2019 only
- PASS — rendered through Msg.envelope(locale=fr) without error (23/23 keys present in render)
- PASS — no silent English fallback in render

## Not verified

- Native-speaker review; terminal/desktop frames (text render via the runtime only, see rendered.txt); the phone app's own labels (separate repo).
