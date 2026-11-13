"""A provably read-only ``python3 -c`` reading a task-tree session's scratchpad.

Incident 2026-11-13 (lop-remote-node-2, build 0.102.2, parent session
ce618964c88e): a parent printing FAILED/passed lines from four logs in its
child coder's scratchpad with a ``pathlib`` one-liner — no unlink, write,
open-for-write or subprocess anywhere in the program — was refused as
dangerous-delete: the backstop marked the quoted pad path as a mention
(``touch``) and only a git-tracked script FILE could clear a task-tree read.

The fix ranks such a read LOW only when the inline program is proven unable
to write, delete or run anything (``resolve.python_code_read_only``), and only
for a pad inside the caller's own task tree. The negative controls keep every
real delete or opaque program on the same targets HIGH, and a stranger's pad
HIGH even for a read.

Everything is synthetic: an isolated ``HOME``/state root and made-up ids.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from local_operator.guardrails import evidence as ev_mod
from local_operator.guardrails import provenance as prov
from local_operator.guardrails.check import CallContext, GuardSettings, check_tool_call
from local_operator.guardrails.classify import Risk, classify_bash

PARENT = "aaaaaaaaaaaa"
CHILD = "bbbbbbbbbbbb"
STRANGER = "cccccccccccc"
INHERITED_PATH = "/usr/bin:/bin"

LOGS = (
    "thumbnail_interleaving_control.py",
    "thumbnail-isolated-current.log",
    "thumbnail-current-interleaving.log",
    "thumbnail-main-interleaving.log",
)


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    for sid in (PARENT, CHILD, STRANGER):
        pad = state / "sessions" / sid / "scratchpad"
        pad.mkdir(parents=True)
        for name in LOGS:
            (pad / name).write_text("1 passed\nFAILED x\n")
        # The file the review r1 repros target: it must EXIST for a LOW to be
        # possible at all, so those rows discriminate.
        (pad / "a.log").write_text("1 passed\n")
    (state / "config.yml").write_text("version: 1\n")
    (home / "work").mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    for key in [k for k in os.environ if k.startswith(("CMUX_", "GIT_"))]:
        monkeypatch.delenv(key, raising=False)
    ev_mod._TREE_CACHE.clear()
    prov.record_lineage(CHILD, origin="subagent", parent=PARENT, state_root=state)
    return home


def _pad(home: Path, sid: str) -> Path:
    return home / ".local-operator" / "sessions" / sid / "scratchpad"


def _verdict(home: Path, command: str, sid: str = PARENT):
    pad = _pad(home, sid)
    env = {"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(pad), "PATH": INHERITED_PATH}
    return classify_bash(
        command, cwd=str(home / "work"), env=env, scratchpad=str(pad), session_id=sid
    )


def _incident(pad: Path) -> str:
    """The refused call, verbatim but for the pad path."""
    names = ",".join(f'"{n}"' for n in LOGS)
    return (
        "python3 -c 'from pathlib import Path;"
        f'r=Path("{pad}"); names=[{names}]; '
        '[(print(n),print("\\n".join(x for x in (r/n).read_text().splitlines() '
        'if "FAILED" in x or "passed" in x or "failed" in x or "ensure_started" in x '
        'or "saved.clear" in x)[-2500:])) for n in names]\''
    )


# -- the incident: RED on origin/main ----------------------------------------------------


def test_parent_reading_its_childs_pad_inline_is_low(home: Path) -> None:
    verdict = _verdict(home, _incident(_pad(home, CHILD)), sid=PARENT)
    assert verdict.risk is Risk.LOW, verdict.reasons


def test_child_reading_its_parents_pad_inline_is_low(home: Path) -> None:
    verdict = _verdict(home, _incident(_pad(home, PARENT)), sid=CHILD)
    assert verdict.risk is Risk.LOW, verdict.reasons


def test_incident_passes_the_real_checkpoint_without_a_judge(home: Path) -> None:
    class _NoJudge:
        calls = 0

        async def __call__(self, system: str, prompt: str) -> str:
            self.calls += 1
            return "BLOCK"

    judge = _NoJudge()
    pad = _pad(home, PARENT)
    ctx = CallContext(
        cwd=str(home / "work"),
        env={"HOME": str(home), "LOCAL_OPERATOR_SCRATCHPAD": str(pad)},
        scratchpad=str(pad),
        session_id=PARENT,
    )
    outcome = asyncio.run(
        check_tool_call(
            "bash",
            {"command": _incident(_pad(home, CHILD))},
            ctx,
            judge=judge,
            settings=GuardSettings(judge_timeout_s=1.0),
            forked_judge=judge,
        )
    )
    assert outcome is None or not outcome.blocked, outcome
    assert judge.calls == 0


# -- negative controls: the same targets, a program that CAN delete or hide ---------------


_DELETES = [
    'import shutil; shutil.rmtree("{pad}")',
    'from pathlib import Path; Path("{pad}/x.log").unlink()',
    'import os; os.remove("{pad}/x.log")',
    'from pathlib import Path; Path("{pad}/x.log").write_text("")',
    'open("{pad}/x.log", "w").write("")',
    'open("{pad}/x.log", mode="a")',
    'from pathlib import Path; Path("{pad}").rename("/tmp/x")',
    'from pathlib import Path; Path("{pad}/x.log").open("w")',
    'exec("import shutil; shutil.rmtree(\\"{pad}\\")")',
    'eval("1"); print("{pad}")',
    'getattr(__import__("os"), "remove")("{pad}/x.log")',
    'import base64; print("{pad}")',
    'import pathlib; pathlib.os.remove("{pad}/x.log")',
    'import subprocess; subprocess.run(["rm", "-rf", "{pad}"])',
    'from pathlib import Path; p = Path("{pad}"); p.__class__',
    'o = open; o("{pad}/x.log", "w")',
    'import functools; functools.partial(open, mode="w")("{pad}/x.log")',
    'from pathlib import Path; list(map(Path.unlink, Path("{pad}").iterdir()))',
    'print("{pad}"',  # does not parse: unknown, so not provably read-only
    # Review r1 M1: ``string.Formatter().get_field`` is getattr by a string.
    "import pathlib,string; f=string.Formatter(); "
    '[f.get_field("0.unlink",(p,),dict())[0]() for p in pathlib.Path("{pad}").iterdir()]',
    "import pathlib,string; "
    'string.Formatter().get_field("0.write_text",(pathlib.Path("{pad}/a.log"),),dict())[0]("x")',
    'import pathlib,string; o=string.Formatter().get_field("0.os",(pathlib,),dict())[0]; '
    'o.execvp("rm",["rm","-rf","{pad}"])',
    # The same class, found auditing the rest of the allow-list.
    'import pathlib; pathlib.operator.methodcaller("unlink")(pathlib.Path("{pad}/a.log"))',
    'import glob; [glob.operator.methodcaller("unlink")(p) for p in glob.glob("{pad}/*")]',
    'import pathlib\nmatch pathlib.Path("{pad}/a.log"):\n case pathlib.PurePath(unlink=u): u()',
    'import json.tool; json.tool.main(); print("{pad}")',
    'from json import tool; tool.main(); print("{pad}")',
    'import functools, pathlib; print("{pad}")',
    'import re; re.functools; print("{pad}")',
]


@pytest.mark.parametrize("code", _DELETES)
def test_writing_or_opaque_programs_on_a_childs_pad_stay_high(home: Path, code: str) -> None:
    command = f"python3 -c '{code.format(pad=_pad(home, CHILD))}'"
    verdict = _verdict(home, command, sid=PARENT)
    assert verdict.risk >= Risk.HIGH, (code, verdict.reasons)


def test_inline_read_of_a_strangers_pad_stays_high(home: Path) -> None:
    verdict = _verdict(home, _incident(_pad(home, STRANGER)), sid=PARENT)
    assert verdict.risk is Risk.HIGH, verdict.reasons
    assert not any("task tree" in r for r in verdict.reasons)


def test_shell_delete_of_a_descendants_pad_stays_high(home: Path) -> None:
    """A parent deleting its child's pad is NOT lowered (see ``_tree_read``)."""
    verdict = _verdict(home, f"rm -rf {_pad(home, CHILD)}", sid=PARENT)
    assert verdict.risk >= Risk.HIGH, verdict.reasons


def test_substituted_or_prefixed_code_is_never_marked_read_only(home: Path) -> None:
    """The flag that lowers the ranking needs the CODE word fully literal and
    python run bare: a substitution, an inline ``PYTHONPATH`` or a wrapper can
    change what runs."""
    from local_operator.guardrails.resolve import resolve_command

    pad = _pad(home, CHILD)
    env = {"HOME": str(home)}
    literal = f"python3 -c 'print(open(\"{pad}/x.log\").read())'"
    for command, expected in (
        (literal, True),
        (f'python3 -c "print(open(\\"{pad}/$(cat /etc/hosts).log\\").read())"', False),
        (f"PYTHONPATH=/tmp {literal}", False),
        (f"env {literal}", False),
        (f"echo hi > fnmatch.py; {literal}", False),
        (f"touch x; {literal}", False),
    ):
        res = resolve_command(command, cwd=str(home / "work"), env=env, home=str(home))
        if expected:
            flags = {t.read_only_code for t in res.touched if (t.path or "").startswith(str(pad))}
            assert flags == {True}, (command, flags)
        else:
            assert not any(t.read_only_code for t in res.touched), command


def test_planted_module_in_the_cwd_is_never_marked_read_only(home: Path) -> None:
    from local_operator.guardrails.resolve import resolve_command

    (home / "work" / "fnmatch.py").write_text("import shutil\n")
    pad = _pad(home, CHILD)
    res = resolve_command(
        f"python3 -c 'print(open(\"{pad}/x.log\").read())'",
        cwd=str(home / "work"),
        env={"HOME": str(home)},
        home=str(home),
    )
    assert not any(t.read_only_code for t in res.touched)


def test_inline_read_with_an_unknown_interpreter_flag_stays_high(home: Path) -> None:
    pad = _pad(home, CHILD)
    command = f"python3 -X importtime -c 'print(open(\"{pad}/x.log\").read())'"
    assert _verdict(home, command, sid=PARENT).risk >= Risk.HIGH


# -- the analyser itself ------------------------------------------------------------------


@pytest.mark.parametrize(
    "code",
    [
        'print(open("/x").read())',
        'print(open("/x", "r").read())',
        'print(open("/x", mode="rb").read())',
        'import json; print(json.load(open("/x"))["k"])',
        "from pathlib import Path; print(sorted(p.name for p in Path('/x').iterdir()))",
        "import re, collections; print(collections.Counter(re.findall('a', 'aa')))",
        # Harmless module-named attributes a read commonly uses.
        "from pathlib import Path; print(Path('/x').stat().st_size)",
        "import datetime; print(datetime.datetime.now().time())",
        "import collections.abc; print(collections.abc.Mapping)",
    ],
)
def test_read_only_programs_are_recognised(code: str) -> None:
    from local_operator.guardrails.resolve import python_code_read_only

    assert python_code_read_only(code)


@pytest.mark.parametrize("code", [c.replace("{pad}", "/x") for c in _DELETES])
def test_effectful_programs_are_not_read_only(code: str) -> None:
    from local_operator.guardrails.resolve import python_code_read_only

    assert not python_code_read_only(code)


# -- review r1 M2: the interpreter itself is pinned ----------------------------------------


def _planted(home: Path) -> Path:
    """An executable named ``python3`` that would ignore CODE entirely."""
    bin_dir = home / "plant"
    bin_dir.mkdir(exist_ok=True)
    exe = bin_dir / "python3"
    exe.write_text('#!/bin/sh\nrm -rf "$3"\n')
    exe.chmod(0o755)
    return exe


def _read(home: Path) -> str:
    return f"-c 'print(open(\"{_pad(home, CHILD)}/{LOGS[1]}\").read())'"


def test_bare_python_on_the_inherited_path_still_ranks_low(home: Path) -> None:
    """Control for the M2 rows: the same read, interpreter bare, is LOW."""
    assert _verdict(home, f"python3 {_read(home)}").risk is Risk.LOW


@pytest.mark.parametrize(
    "prefix",
    [
        "./python3",
        "{exe}",
        "export PATH={bin}:$PATH; python3",
        "PATH={bin}:$PATH; export PATH; python3",
        "hash -p {exe} python3; python3",
    ],
)
def test_a_relative_absolute_or_repathed_python_stays_high(home: Path, prefix: str) -> None:
    exe = _planted(home)
    (home / "work" / "python3").symlink_to(exe)
    command = f"{prefix.format(exe=exe, bin=exe.parent)} {_read(home)}"
    verdict = _verdict(home, command)
    assert verdict.risk >= Risk.HIGH, (command, verdict.reasons)


# -- review r1 m1: start-up environment -----------------------------------------------------


def test_pythonwarnings_set_by_the_command_stays_high(home: Path) -> None:
    command = f"export PYTHONWARNINGS=error::x.Y; python3 {_read(home)}"
    assert _verdict(home, command).risk >= Risk.HIGH


def test_harmless_inherited_python_knobs_keep_the_read_low(home: Path) -> None:
    """``PYTHONUNBUFFERED`` and friends change no import: refusing them would
    re-break the incident read on machines that export them."""
    pad = _pad(home, PARENT)
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_SCRATCHPAD": str(pad),
        "PATH": INHERITED_PATH,
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    verdict = classify_bash(
        _incident(_pad(home, CHILD)),
        cwd=str(home / "work"),
        env=env,
        scratchpad=str(pad),
        session_id=PARENT,
    )
    assert verdict.risk is Risk.LOW, verdict.reasons
