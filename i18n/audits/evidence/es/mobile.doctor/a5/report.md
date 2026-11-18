# es mobile.doctor audit a5, PR #4216 round 2

Verdict: PASSED (0 blocker / 0 major / 0 minor / 0 nit).

Scope: 4 keys changed by 193e08e33. English source sha256 8e22fc91780c70e0750c05c92a8478db0ca00d738981ae20063028fe33b0ddcc.

Select arms complete; «reinícialo» (tú) matches round 1; «La mayoría pertenecen» kept from round 1 (accepted ad sensum agreement); «La mayoría los tiene {name}» idiomatic for instances.

## Renders (synthetic params: node, pid 4242, systemctl --user restart dev.service)

- `near` / `command`: Tu cuenta usa 8100 de 8192 vigilancias de archivos permitidas en este equipo, cerca del límite. La mayoría pertenecen a node (proceso 4242): reinícialo con systemctl --user restart dev.service o pide a un administrador que aumente fs.inotify.max_user_watches.
- `near` / `program`: Tu cuenta usa 8100 de 8192 vigilancias de archivos permitidas en este equipo, cerca del límite. La mayoría pertenecen a node (proceso 4242): reinicia ese programa o pide a un administrador que aumente fs.inotify.max_user_watches.
- `near` / `other`: Tu cuenta usa 8100 de 8192 vigilancias de archivos permitidas en este equipo, cerca del límite. Pide a un administrador que aumente fs.inotify.max_user_watches.
- `at_limit` / `command`: Tu cuenta agotó sus vigilancias de archivos en este equipo (8100 de 8192), así que los programas que vigilan archivos fallan con “No space left on device”. La mayoría pertenecen a node (proceso 4242): reinícialo con systemctl --user restart dev.service o pide a un administrador que aumente fs.inotify.max_user_watches.
- `at_limit` / `program`: Tu cuenta agotó sus vigilancias de archivos en este equipo (8100 de 8192), así que los programas que vigilan archivos fallan con “No space left on device”. La mayoría pertenecen a node (proceso 4242): reinicia ese programa o pide a un administrador que aumente fs.inotify.max_user_watches.
- `at_limit` / `other`: Tu cuenta agotó sus vigilancias de archivos en este equipo (8100 de 8192), así que los programas que vigilan archivos fallan con “No space left on device”. Pide a un administrador que aumente fs.inotify.max_user_watches.
- `instances_near` / `command`: Tu cuenta usa 126 de 128 identificadores de vigilancia permitidos en este equipo, cerca del límite. La mayoría los tiene node (proceso 4242): reinícialo con systemctl --user restart dev.service o pide a un administrador que aumente fs.inotify.max_user_instances.
- `instances_near` / `program`: Tu cuenta usa 126 de 128 identificadores de vigilancia permitidos en este equipo, cerca del límite. La mayoría los tiene node (proceso 4242): reinicia ese programa o pide a un administrador que aumente fs.inotify.max_user_instances.
- `instances_near` / `other`: Tu cuenta usa 126 de 128 identificadores de vigilancia permitidos en este equipo, cerca del límite. Pide a un administrador que aumente fs.inotify.max_user_instances.
- `instances_at_limit` / `command`: Tu cuenta agotó sus identificadores de vigilancia en este equipo (126 de 128), así que los programas que vigilan archivos fallan con “Too many open files”. La mayoría los tiene node (proceso 4242): reinícialo con systemctl --user restart dev.service o pide a un administrador que aumente fs.inotify.max_user_instances.
- `instances_at_limit` / `program`: Tu cuenta agotó sus identificadores de vigilancia en este equipo (126 de 128), así que los programas que vigilan archivos fallan con “Too many open files”. La mayoría los tiene node (proceso 4242): reinicia ese programa o pide a un administrador que aumente fs.inotify.max_user_instances.
- `instances_at_limit` / `other`: Tu cuenta agotó sus identificadores de vigilancia en este equipo (126 de 128), así que los programas que vigilan archivos fallan con “Too many open files”. Pide a un administrador que aumente fs.inotify.max_user_instances.
