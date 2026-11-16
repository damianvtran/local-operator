"""Prove fresh-directory containment using synthetic HOME/state and real shells.

No fixture launches a browser, interpreter script, network request or delete.
The resolver/preview negatives are read-only strings, never executed.
"""

import subprocess
from pathlib import Path

import pytest

from local_operator.guardrails.check import CallContext, GuardSettings
from local_operator.guardrails.classify import Risk, classify_bash
from local_operator.guardrails.preview import preview_tool_call
from local_operator.guardrails.resolve import resolve_command

SID = "aaaaaaaaaaaa"
HEADER = "set -euo pipefail\n"
ASSIGN = 'ROOT=$(mktemp -d "$LOCAL_OPERATOR_SCRATCHPAD/mac-proof.XXXXXXXX")\n'
WRITE = 'printf proof > "$ROOT/source/proof.txt"\n'


@pytest.fixture
def tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    home = (tmp_path / "home").resolve()
    state = home / ".local-operator"
    pad = state / "sessions" / SID / "scratchpad"
    pad.mkdir(parents=True)
    (state / "config.yml").write_text("version: 1\n")
    cwd = home / "repo"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_CONFIG_DIR": str(state),
        "LOCAL_OPERATOR_SCRATCHPAD": str(pad),
        "PATH": "/usr/bin:/bin",
    }
    return home, pad, cwd, env


def _resolve(tree, command):
    home, _, cwd, env = tree
    return resolve_command(command, cwd=str(cwd), home=str(home), env=env)


def _rank(tree, command):
    _, pad, cwd, env = tree
    return classify_bash(command, cwd=str(cwd), env=env, scratchpad=str(pad), session_id=SID)


def test_fail_closed_scratch_template_keeps_parent(tree) -> None:
    _, pad, _, _ = tree
    command = HEADER + ASSIGN + """SRC="$ROOT/source"
mkdir -p "$SRC/scripts"
for f in package.json scripts/x.mjs; do git show "SHA:$f" > "$SRC/$f"; done
"""
    result = _resolve(tree, command)
    writes = [target for target in result.targets if target.op == "overwrite"]
    assert len(writes) == 2
    assert all(target.path == str(pad) for target in writes)
    assert all(target.opaque_contained and target.opaque and not target.temp for target in writes)
    assert _rank(tree, command).risk == Risk.LOW
    # A symbolic dirname is not a check-time creation-ledger path.
    assert not any(c.path and "mac-proof." in c.path for c in result.created)


def test_preview_and_real_shell_write_agree(tree) -> None:
    home, pad, cwd, env = tree
    command = HEADER + ASSIGN + 'mkdir -p "$ROOT/source"\n' + WRITE
    report = preview_tool_call(
        "bash",
        {"command": command},
        CallContext(cwd=str(cwd), env=env, scratchpad=str(pad), session_id=SID),
        settings=GuardSettings(enabled=False),
    )
    assert report.builtin_risk == "LOW"
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    files = list(pad.glob("mac-proof.*/source/proof.txt"))
    assert len(files) == 1
    assert files[0].read_text() == "proof"
    assert not (home / "source").exists()


@pytest.mark.parametrize(
    "command",
    [
        ASSIGN + WRITE,
        "set -e\n" + ASSIGN + WRITE,
        "set -euo pipefail && " + ASSIGN + WRITE,
        HEADER + ASSIGN.rstrip() + " || true\n" + WRITE,
        HEADER + "if " + ASSIGN + "then :; fi\n" + WRITE,
        HEADER + "! " + ASSIGN + WRITE,
        HEADER + ASSIGN.rstrip() + " | cat\n" + WRITE,
        HEADER + "set +e\n" + ASSIGN + WRITE,
        HEADER + ASSIGN + 'C=command; S=set; "$C" "$S" +e\n' + WRITE,
        HEADER + ASSIGN + 'C=command; U=unset; "$C" "$U" ROOT\n' + WRITE,
        HEADER + ASSIGN + 'C=command; E=export; "$C" "$E" ROOT=\n' + WRITE,
        HEADER + ASSIGN + 'C=command; R=read; "$C" "$R" ROOT\n' + WRITE,
        HEADER + "ROOT=$(printf arbitrary)\n" + WRITE,
        HEADER + ASSIGN + 'ROOT=""\n' + WRITE,
        HEADER + ASSIGN + "unset ROOT\n" + WRITE,
        HEADER + ASSIGN + "ROOT=$(arbitrary)\n" + WRITE,
        HEADER + ASSIGN + 'printf proof > "$ROOT/../config.yml"\n',
        HEADER + ASSIGN + 'printf proof > "${ROOT}old/source"\n',
        HEADER + ASSIGN + 'printf proof > "$ROOT/$(arbitrary)"\n',
        HEADER + 'ROOT=$(mktemp -u -d "$LOCAL_OPERATOR_SCRATCHPAD/p.XXXXXXXX")\n' + WRITE,
        HEADER + 'ROOT=$(mktemp -d -p "$LOCAL_OPERATOR_SCRATCHPAD" p.XXXXXXXX)\n' + WRITE,
        HEADER + 'ROOT=$(mktemp -d "$LOCAL_OPERATOR_SCRATCHPAD/../p.XXXXXXXX")\n' + WRITE,
        HEADER + 'ROOT=$(mktemp -d "$LOCAL_OPERATOR_SCRATCHPAD/p.XXXXXXXX") OTHER=x\n' + WRITE,
        HEADER + ASSIGN + 'ln -s "$HOME/.local-operator" "$ROOT/source"\n' + WRITE,
        HEADER + ASSIGN + 'mv "$HOME/.local-operator" "$ROOT/source"\n' + WRITE,
        HEADER + ASSIGN + 'node unknown.mjs "$ROOT"\n' + WRITE,
        HEADER + ASSIGN + 'X=$(node unknown.mjs "$ROOT")\n' + WRITE,
        HEADER + ASSIGN + "PATH=/untrusted\n" + WRITE,
        HEADER + 'ln -sfn "$HOME" "$LOCAL_OPERATOR_SCRATCHPAD"\n' + ASSIGN + WRITE,
    ],
)
def test_unsupported_or_invalidated_provenance_never_lowers(tree, command) -> None:
    assert _rank(tree, command).risk >= Risk.HIGH
    writes = [t for t in _resolve(tree, command).targets if t.op == "overwrite"]
    assert writes and not all(t.opaque_contained for t in writes)


def test_symlink_parent_is_not_pinned(tree) -> None:
    home, pad, _, _ = tree
    (pad / "link").symlink_to(home, target_is_directory=True)
    command = HEADER + ASSIGN.replace("/mac-proof.", "/link/mac-proof.") + WRITE
    assert _rank(tree, command).risk >= Risk.HIGH
    assert not any(t.opaque_contained for t in _resolve(tree, command).targets)


def test_reassigned_scratchpad_cannot_grant_freshness(tree) -> None:
    command = HEADER + 'LOCAL_OPERATOR_SCRATCHPAD="$HOME"\n' + ASSIGN + WRITE
    assert _rank(tree, command).risk >= Risk.HIGH


@pytest.mark.parametrize(
    "operand", ["$HOME", "$HOME/.local-operator", "$HOME/.local-operator/config.yml"]
)
@pytest.mark.parametrize("program", ["rm -rf", "mv"])
def test_protected_removal_and_move_remain_guarded(tree, operand, program) -> None:
    command = HEADER + ASSIGN + f'{program} "{operand}" "$ROOT/moved"\n'
    assert _rank(tree, command).risk >= Risk.HIGH


@pytest.mark.parametrize("program", ["node unknown.mjs --out", "python3 unknown.py --out"])
def test_unknown_interpreter_argv_into_state_stays_guarded(tree, program) -> None:
    command = HEADER + ASSIGN + f'{program} "$HOME/.local-operator/sessions/{SID}/asks.jsonl"\n'
    assert _rank(tree, command).risk >= Risk.HIGH


@pytest.mark.parametrize("flag", ["--output ", "--output=", "-o ", "-o"])
def test_curl_output_protected_destination_stays_critical(tree, flag) -> None:
    command = (
        HEADER
        + ASSIGN
        + f'curl -fsSL {flag}"$HOME/.local-operator/sessions/{SID}/asks.jsonl" '
        + "https://example.invalid/x\n"
    )
    assert _rank(tree, command).risk == Risk.CRITICAL


def test_failed_mktemp_exits_before_empty_root_can_write(tree) -> None:
    _, pad, cwd, env = tree
    # Even a same-named failing mktemp function cannot reach the sentinel.
    # The function is a runtime failure injector, never a static exemption.
    marker = pad / "must-not-exist"
    command = "mktemp() { return 23; }\n" + HEADER + ASSIGN + f'printf unsafe > "{marker}"\n'
    completed = subprocess.run(
        ["/bin/bash", "-c", command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 23
    assert not marker.exists()
    # Function definitions are outside the statically proved shape.
    assert not any(t.opaque_contained for t in _resolve(tree, command + WRITE).targets)


@pytest.mark.parametrize(
    "mutation",
    [
        'printf -v ROOT %s "$HOME/.local-operator"',
        'COPY="$ROOT/source"; printf -v COPY %s "$HOME/.local-operator"',
        'declare -n REF=ROOT; REF="$HOME/.local-operator"',
        'typeset -n REF=ROOT; REF="$HOME/.local-operator"',
        "declare -i ROOT=0",
        "printf '%n' ROOT",
        'FORMAT=%n; printf "$FORMAT" ROOT',
        'builtin printf -v ROOT %s "$HOME/.local-operator"',
        'command printf -v ROOT %s "$HOME/.local-operator"',
        "read -a ROOT",
        "let ROOT=0",
        "eval 'ROOT=0'",
        "set -- ROOT",
        'alias printf="printf -v ROOT"',
        "echo $((ROOT=0))",
        'echo "${ROOT:=0}"',
    ],
)
def test_shell_mutators_end_all_fresh_binding_proofs(tree, mutation) -> None:
    # Never execute these protected-target probes. Copies must be invalidated
    # too: modeling only the directly named variable misses nameref aliases.
    destination = "COPY" if mutation.startswith("COPY=") else "ROOT"
    command = HEADER + ASSIGN + mutation + f'\nprintf proof > "${destination}/config.yml"\n'
    result = _resolve(tree, command)
    assert not any(t.opaque_contained for t in result.targets if t.op == "overwrite")
    # Literal zero assignments are accurately resolved as ordinary paths; the
    # invariant is lost freshness, not forcing harmless literal paths HIGH.
    if mutation not in ("declare -i ROOT=0", "eval 'ROOT=0'"):
        assert _rank(tree, command).risk >= Risk.HIGH


def test_inherited_path_shadow_is_not_a_fresh_directory_creator(tree) -> None:
    home, pad, cwd, env = tree
    bin_dir = home / "bin"
    bin_dir.mkdir()
    shadow = bin_dir / "mktemp"
    shadow.write_text('#!/bin/bash\nprintf "%s\\n" "$HOME/.local-operator"\n')
    shadow.chmod(0o755)
    env["PATH"] = str(bin_dir) + ":/usr/bin:/bin"
    command = HEADER + ASSIGN + 'printf proof > "$ROOT/config.yml"\n'
    assert _rank(tree, command).risk >= Risk.HIGH
    assert not any(t.opaque_contained for t in _resolve(tree, command).targets)
    # A display-only execution verifies this is a successful impostor, not a
    # hypothetical failed mktemp. The protected overwrite is never dispatched.
    completed = subprocess.run(
        ["/bin/bash", "-c", HEADER + ASSIGN + 'printf "%s\\n" "$ROOT"'],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0
    assert completed.stdout.strip() == str(home / ".local-operator")
    assert not list(pad.glob("mac-proof.*"))
    assert (home / ".local-operator/config.yml").read_text() == "version: 1\n"


@pytest.mark.parametrize(
    "prefix",
    [
        "mktemp() { printf '%s' 0; }\n",
        "alias mktemp='printf 0'\n",
        'printf fake > "$LOCAL_OPERATOR_SCRATCHPAD/mktemp"\n',
    ],
)
def test_prior_shell_or_executable_shadowing_ends_fresh_scope(tree, prefix) -> None:
    command = HEADER + prefix + ASSIGN + WRITE
    assert _rank(tree, command).risk >= Risk.HIGH
    assert not any(t.opaque_contained for t in _resolve(tree, command).targets)


@pytest.mark.parametrize("key", ["BASH_ENV", "ENV", "BASH_FUNC_mktemp%%"])
def test_inherited_shell_startup_or_function_prevents_identity_proof(tree, key) -> None:
    tree[3][key] = "() { printf fake; }" if key.startswith("BASH_FUNC_") else "/untrusted"
    command = HEADER + ASSIGN + 'printf proof > "$ROOT/config.yml"\n'
    assert _rank(tree, command).risk >= Risk.HIGH
    assert not any(t.opaque_contained for t in _resolve(tree, command).targets)


@pytest.mark.parametrize("entry", ["", ".", "bin"])
def test_relative_path_winner_uses_modeled_cwd(tree, entry) -> None:
    _, _, cwd, env = tree
    directory = cwd / "bin" if entry == "bin" else cwd
    directory.mkdir(exist_ok=True)
    shadow = directory / "mktemp"
    shadow.write_text('#!/bin/bash\nprintf "%s\\n" "$HOME/.local-operator"\n')
    shadow.chmod(0o755)
    env["PATH"] = entry + ":/usr/bin:/bin"
    command = HEADER + ASSIGN + 'printf proof > "$ROOT/config.yml"\n'
    assert _rank(tree, command).risk >= Risk.HIGH
    assert not any(t.opaque_contained for t in _resolve(tree, command).targets)


def test_system_mktemp_in_ordinary_inherited_path_is_supported(tree) -> None:
    home, _, _, env = tree
    empty_bin = home / "empty-bin"
    empty_bin.mkdir()
    env["PATH"] = str(empty_bin) + ":/usr/bin:/bin"
    command = HEADER + ASSIGN + 'mkdir -p "$ROOT/source"\n' + WRITE
    assert _rank(tree, command).risk == Risk.LOW


def test_legacy_default_mktemp_remains_supported(tree) -> None:
    result = _resolve(tree, 'ROOT=$(mktemp -d)\nprintf proof > "$ROOT/proof"')
    assert any(t.temp for t in result.targets)
