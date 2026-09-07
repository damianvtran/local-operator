"""How this project re-enters a Python interpreter to run its own modules.

WHY THIS EXISTS
---------------
Every ``[sys.executable, "-m", "local_operator...."]`` spawn in this codebase
shared one latent defect: ``-m`` puts the child's WORKING DIRECTORY on
``sys.path[0]``, *ahead of* site-packages. None of those spawns pass ``cwd=``,
so the child inherits the parent's directory — and when that directory is a
checkout of this project (the normal case for anyone developing it, and the
exact case for an agent session whose cwd is a worktree), the child imports the
CHECKOUT rather than the installed distribution it was supposed to run.

Measured on a real uv-tool install, same interpreter, only cwd differing::

    cd <a worktree whose pyproject says 0.51.0>
      imports: <worktree>/local_operator/__init__.py       stamp 0.51.0@ad6db35
    same command, isolation flag added:
      imports: <uv tool>/…/site-packages/local_operator/…  stamp 0.51.5@ad6db35

The parent TUI is immune by accident rather than by design: it launches through
the console script ``~/.local/bin/lop``, whose ``sys.path[0]`` is ``bin/``.
Only the children were exposed — which is how live runtime daemons ended up
pinned to a superseded build, serving a session whose viewer was current and
missing a fix the user could see was absent.

WHY ``-P`` AND NOT ``PYTHONSAFEPATH=1``
---------------------------------------
The two are equivalent *for the process being started*, and the environment
variable looks tidier because these call sites already build an ``env`` dict.
It is nonetheless the wrong tool, for a reason that stays invisible until
something downstream breaks: **the variable is inherited by the entire process
subtree; the flag is not.** Measured, not assumed::

    env PYTHONSAFEPATH=1 python child.py  ->  grandchild: SAFEPATH=1, path0 stripped
    python -P child.py                    ->  grandchild: SAFEPATH=None, path0 ''

That difference decides correctness here. A spawned runtime goes on to run the
agent's ``bash`` tool, which copies ``os.environ`` into every command it
executes. With the variable set, every ``python`` the user's agent runs — in
the user's own project, against the user's own code — silently loses
``sys.path[0]`` and stops importing modules sitting beside the script::

    cd <a user project>; python runner.py  ->  ModuleNotFoundError: No module named 'usermod'

Ordinary Python semantics would break in the user's workspace, caused by an
isolation flag that was only ever meant to protect OUR import. The flag form
corrects our child and changes nothing for anything that child launches, so the
blast radius is exactly the one process we intended to fix.

WHY NOT ``cwd=``
----------------
Pointing the child at a neutral directory would also stop the bad import, and
is worse: the session's working directory is a user-facing contract. It is what
the agent operates on, what its tools resolve relative paths against, and what
the status bar shows. Changing it would trade a silent version skew for a
silent behaviour change. ``-P`` removes only the implicit ``sys.path[0]``
injection — precisely the defect, and nothing else.

WHERE THE CWD IS STILL WANTED
-----------------------------
One child legitimately needs it. The ``eval`` worker executes USER code, and a
cell doing ``import my_module`` beside the session's cwd must keep working. That
worker re-inserts its own cwd *after* the harness modules have already resolved
— see :mod:`local_operator.tools.eval_worker`. The ordering is the entire point:
our modules bind to the install, user code still reaches the workspace.
"""

from __future__ import annotations

import sys

#: Interpreter flag suppressing the implicit ``sys.path[0]`` entry (the cwd for
#: ``-m``/``-c``, the script's own directory for a path argument). CPython 3.11+;
#: this project requires >=3.12, so it is always available and needs no guard.
SAFE_PATH_FLAG = "-P"


def python_argv(*args: str) -> list[str]:
    """``[sys.executable, "-P", *args]`` — how this project re-enters Python.

    Use for every self-spawn (``-m local_operator...``, ``-c`` snippets) so a
    child cannot import a checkout that merely happens to be the cwd. Callers
    pass their own ``-u``/``-m``/arguments as usual; this only guarantees the
    isolation flag is present and FIRST, since interpreter options are
    recognised only before ``-m``/``-c``.

    ``sys.executable`` is read at call time and never cached: a runtime is
    started by whichever interpreter is current, so an upgrade that replaces
    the install under a long-lived parent is picked up by its next child.
    """
    return [sys.executable, SAFE_PATH_FLAG, *args]
