# zh-CN mobile.doctor audit a6, PR #4216 round 2

Verdict: PASSED (0 blocker / 0 major / 0 minor / 0 nit).

Scope: 4 keys changed by 193e08e33. English source sha256 8e22fc91780c70e0750c05c92a8478db0ca00d738981ae20063028fe33b0ddcc.

Select arms complete; 其中大部分属于 / 其中…占用最多 retained from round 1; half-width spaces around {command} and sysctl names kept; full-width punctuation.

## Renders (synthetic params: node, pid 4242, systemctl --user restart dev.service)

- `near` / `command`: 你的账户在此电脑上使用了8100个文件监视（上限8192个），已接近上限。其中大部分属于node（进程4242）：请运行 systemctl --user restart dev.service 重启它，或请管理员调高 fs.inotify.max_user_watches。
- `near` / `program`: 你的账户在此电脑上使用了8100个文件监视（上限8192个），已接近上限。其中大部分属于node（进程4242）：请重启该程序，或请管理员调高 fs.inotify.max_user_watches。
- `near` / `other`: 你的账户在此电脑上使用了8100个文件监视（上限8192个），已接近上限。请管理员调高 fs.inotify.max_user_watches。
- `at_limit` / `command`: 你的账户已用完此电脑上的文件监视（8100个，上限8192个），因此监视文件的程序会报错“No space left on device”。其中大部分属于node（进程4242）：请运行 systemctl --user restart dev.service 重启它，或请管理员调高 fs.inotify.max_user_watches。
- `at_limit` / `program`: 你的账户已用完此电脑上的文件监视（8100个，上限8192个），因此监视文件的程序会报错“No space left on device”。其中大部分属于node（进程4242）：请重启该程序，或请管理员调高 fs.inotify.max_user_watches。
- `at_limit` / `other`: 你的账户已用完此电脑上的文件监视（8100个，上限8192个），因此监视文件的程序会报错“No space left on device”。请管理员调高 fs.inotify.max_user_watches。
- `instances_near` / `command`: 你的账户在此电脑上使用了126个监视句柄（上限128个），已接近上限。其中node（进程4242）占用最多：请运行 systemctl --user restart dev.service 重启它，或请管理员调高 fs.inotify.max_user_instances。
- `instances_near` / `program`: 你的账户在此电脑上使用了126个监视句柄（上限128个），已接近上限。其中node（进程4242）占用最多：请重启该程序，或请管理员调高 fs.inotify.max_user_instances。
- `instances_near` / `other`: 你的账户在此电脑上使用了126个监视句柄（上限128个），已接近上限。请管理员调高 fs.inotify.max_user_instances。
- `instances_at_limit` / `command`: 你的账户已用完此电脑上的监视句柄（126个，上限128个），因此监视文件的程序会报错“Too many open files”。其中node（进程4242）占用最多：请运行 systemctl --user restart dev.service 重启它，或请管理员调高 fs.inotify.max_user_instances。
- `instances_at_limit` / `program`: 你的账户已用完此电脑上的监视句柄（126个，上限128个），因此监视文件的程序会报错“Too many open files”。其中node（进程4242）占用最多：请重启该程序，或请管理员调高 fs.inotify.max_user_instances。
- `instances_at_limit` / `other`: 你的账户已用完此电脑上的监视句柄（126个，上限128个），因此监视文件的程序会报错“Too many open files”。请管理员调高 fs.inotify.max_user_instances。
