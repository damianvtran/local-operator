# fr mobile.doctor audit a6, PR #4216 round 2

Verdict: PASSED (0 blocker / 0 major / 0 minor / 0 nit).

Scope: 4 keys changed by 193e08e33. English source sha256 8e22fc91780c70e0750c05c92a8478db0ca00d738981ae20063028fe33b0ddcc.

Select arms complete; U+202F before ':' and '»', U+2019 apostrophes kept; «redémarrez-le» agrees with masculine programme/processus; «La plupart sont détenus par» agrees with descripteurs (m.pl.).

## Renders (synthetic params: node, pid 4242, systemctl --user restart dev.service)

- `near` / `command`: Votre compte utilise 8100 des 8192 surveillances de fichiers autorisées sur cet ordinateur, près de la limite. La plupart appartiennent à node (processus 4242) : redémarrez-le avec systemctl --user restart dev.service, ou demandez à un administrateur d’augmenter fs.inotify.max_user_watches.
- `near` / `program`: Votre compte utilise 8100 des 8192 surveillances de fichiers autorisées sur cet ordinateur, près de la limite. La plupart appartiennent à node (processus 4242) : redémarrez ce programme, ou demandez à un administrateur d’augmenter fs.inotify.max_user_watches.
- `near` / `other`: Votre compte utilise 8100 des 8192 surveillances de fichiers autorisées sur cet ordinateur, près de la limite. Demandez à un administrateur d’augmenter fs.inotify.max_user_watches.
- `at_limit` / `command`: Votre compte a épuisé ses surveillances de fichiers sur cet ordinateur (8100 sur 8192) : les programmes qui surveillent des fichiers échouent avec « No space left on device ». La plupart appartiennent à node (processus 4242) : redémarrez-le avec systemctl --user restart dev.service, ou demandez à un administrateur d’augmenter fs.inotify.max_user_watches.
- `at_limit` / `program`: Votre compte a épuisé ses surveillances de fichiers sur cet ordinateur (8100 sur 8192) : les programmes qui surveillent des fichiers échouent avec « No space left on device ». La plupart appartiennent à node (processus 4242) : redémarrez ce programme, ou demandez à un administrateur d’augmenter fs.inotify.max_user_watches.
- `at_limit` / `other`: Votre compte a épuisé ses surveillances de fichiers sur cet ordinateur (8100 sur 8192) : les programmes qui surveillent des fichiers échouent avec « No space left on device ». Demandez à un administrateur d’augmenter fs.inotify.max_user_watches.
- `instances_near` / `command`: Votre compte utilise 126 des 128 descripteurs de surveillance autorisés sur cet ordinateur, près de la limite. La plupart sont détenus par node (processus 4242) : redémarrez-le avec systemctl --user restart dev.service, ou demandez à un administrateur d’augmenter fs.inotify.max_user_instances.
- `instances_near` / `program`: Votre compte utilise 126 des 128 descripteurs de surveillance autorisés sur cet ordinateur, près de la limite. La plupart sont détenus par node (processus 4242) : redémarrez ce programme, ou demandez à un administrateur d’augmenter fs.inotify.max_user_instances.
- `instances_near` / `other`: Votre compte utilise 126 des 128 descripteurs de surveillance autorisés sur cet ordinateur, près de la limite. Demandez à un administrateur d’augmenter fs.inotify.max_user_instances.
- `instances_at_limit` / `command`: Votre compte a épuisé ses descripteurs de surveillance sur cet ordinateur (126 sur 128) : les programmes qui surveillent des fichiers échouent avec « Too many open files ». La plupart sont détenus par node (processus 4242) : redémarrez-le avec systemctl --user restart dev.service, ou demandez à un administrateur d’augmenter fs.inotify.max_user_instances.
- `instances_at_limit` / `program`: Votre compte a épuisé ses descripteurs de surveillance sur cet ordinateur (126 sur 128) : les programmes qui surveillent des fichiers échouent avec « Too many open files ». La plupart sont détenus par node (processus 4242) : redémarrez ce programme, ou demandez à un administrateur d’augmenter fs.inotify.max_user_instances.
- `instances_at_limit` / `other`: Votre compte a épuisé ses descripteurs de surveillance sur cet ordinateur (126 sur 128) : les programmes qui surveillent des fichiers échouent avec « Too many open files ». Demandez à un administrateur d’augmenter fs.inotify.max_user_instances.
