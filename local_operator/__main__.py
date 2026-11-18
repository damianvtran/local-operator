"""``python -m local_operator``: run the ``lop`` CLI as a module.

WHY THIS FILE EXISTS: ``update_bootstrap.controller`` launches the controller
preparation and the post-flip rolling-update tail as
``<generation python> -B -P -m local_operator update ...`` (see
``update_bootstrap._module_command``). It runs the generation's own
interpreter rather than a console-script shim, because the shim's path and
shebang are install-layout details the controller must not depend on. Without
this module that launch dies with "No module named local_operator.__main__",
so every rolling update stops after the core flip: no service refresh, no
rollout, no prune. Removing or renaming this file breaks rolling updates;
``tests/unit/test_main_module.py`` guards it.

The import stays inside the guard so a plain ``import local_operator.__main__``
(e.g. by ``importlib.util.find_spec`` callers or tooling) does not load the
whole CLI; ``cli.main`` returns the process exit code, which ``SystemExit``
passes to the interpreter exactly as the ``lop`` console script does.
"""

if __name__ == "__main__":
    from local_operator.cli import main

    raise SystemExit(main())
