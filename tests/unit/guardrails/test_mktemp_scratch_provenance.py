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
        # Rebase onto #4147, which added ``NAME+=`` tracking: appending to a
        # fresh binding (or appending one) must drop the proof, never extend it.
        HEADER + ASSIGN + "ROOT+=/../..\n" + WRITE,
        HEADER + ASSIGN + "ROOT+=old\n" + WRITE,
        HEADER
        + 'OTHER=x\nOTHER+=$(mktemp -d "$LOCAL_OPERATOR_SCRATCHPAD/p.XXXXXXXX")\n'
        + 'printf proof > "$OTHER/source"\n',
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


@pytest.mark.parametrize("binding", ["ROOT", "COPY"])
@pytest.mark.parametrize(
    "mutation",
    [
        'echo "$[{binding}=0]"',
        'printf %s "$[{binding}=0]"',
        'echo "$(({binding}=0))"',
        'printf %s "$(({binding}=0))"',
        "(({binding}=0)) || true",
    ],
)
def test_arithmetic_binding_mutation_cannot_hide_numeric_symlink(tree, binding, mutation) -> None:
    home, _, cwd, env = tree
    state = home / ".local-operator"
    (cwd / "0").symlink_to(state, target_is_directory=True)
    prefix = HEADER + ASSIGN
    if binding == "COPY":
        prefix += 'COPY="$ROOT"\n'
    prefix += mutation.format(binding=binding) + "\n"
    command = prefix + f'printf proof > "${binding}/config.yml"\n'
    # Preview the dangerous suffix only; execute a display-only prefix to prove
    # Bash really changes the binding to the numeric symlink destination.
    report = preview_tool_call(
        "bash",
        {"command": command},
        CallContext(cwd=str(cwd), env=env, scratchpad=str(tree[1]), session_id=SID),
        settings=GuardSettings(enabled=False),
    )
    assert report.builtin_risk in {"HIGH", "CRITICAL"}
    assert not any(t.opaque_contained for t in _resolve(tree, command).targets)
    completed = subprocess.run(
        ["/bin/bash", "-c", prefix + f'printf "\\nresolved=%s\\n" "${binding}"'],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.endswith("\nresolved=0\n")
    assert (cwd / "0/config.yml").resolve() == state / "config.yml"
    assert (state / "config.yml").read_text() == "version: 1\n"


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


# PR #4116 review B1: FRESH_DIR_MARK is the byte \x03, which a command can
# write itself (ANSI-C ``$'\x03'`` or a raw U+0003). Freshness must come from
# the walker's own proof, never from the byte, so each forged binding ranks
# exactly as main does: a write through a link into the state root is CRITICAL.
_PAD = '"$LOCAL_OPERATOR_SCRATCHPAD/'
_FORGED = {
    "ln-config": 'ln -s "$HOME/.local-operator" ' + _PAD + "n\"$'\\x03'\n"
    "R=" + _PAD + "n\"$'\\x03'\n"
    'printf x > "$R/config.yml"\n',
    "asks": "R=" + _PAD + "\"$'\\x03'\n" + f'printf x > "$R/sessions/{SID}/asks.jsonl"\n',
    "rm-sessions": "R=" + _PAD + "\"$'\\x03'\n" + 'rm -rf "$R/sessions"\n',
    "raw-u0003-config": "R=" + _PAD + 'q\x03"\nprintf x > "$R/config.yml"\n',
    "raw-u0003-rm": "R=" + _PAD + '\x03"\nrm -rf "$R/sessions"\n',
}


@pytest.mark.parametrize("header", ["", HEADER], ids=["bare", "header"])
@pytest.mark.parametrize("case", sorted(_FORGED))
def test_forged_fresh_mark_never_lowers_critical(tree, case, header) -> None:
    home, pad, cwd, env = tree
    # The "pre-planted" rows: a link left by an earlier call, named so the
    # forged value resolves through it into the state root.
    (pad / "\x03").symlink_to(home / ".local-operator")
    (pad / "q\x03").symlink_to(home / ".local-operator")
    command = header + _FORGED[case]
    assert _rank(tree, command).risk == Risk.CRITICAL
    report = preview_tool_call(
        "bash",
        {"command": command},
        CallContext(cwd=str(cwd), env=env, scratchpad=str(pad), session_id=SID),
        settings=GuardSettings(enabled=False),
    )
    assert report.builtin_risk == "CRITICAL"


def test_forged_mark_in_inherited_env_is_not_fresh(tree) -> None:
    # Out-of-band provenance: even an inherited value carrying the byte is a
    # plain string, so it never grants containment.
    _, pad, _, env = tree
    env["R"] = str(pad) + "/\x03"
    result = _resolve(tree, HEADER + 'printf x > "$R/config.yml"\n')
    assert not any(t.opaque_contained for t in result.targets)


def test_genuine_fresh_copy_still_keeps_provenance(tree) -> None:
    # Control for B1: a walker-proved binding copied by plain assignment keeps
    # the intended LOW; only forged marks lost it.
    command = HEADER + ASSIGN + 'COPY="$ROOT"\nmkdir -p "$COPY/s"\nprintf x > "$COPY/s/f"\n'
    assert _rank(tree, command).risk == Risk.LOW


# PR #4116 review M1: mkdir is found on PATH, so it gets mktemp's root-owned
# identity check; read-only git can run repo-configured programs, so it ends
# the proof for every later segment.
def test_shadowed_mkdir_ends_fresh_scope(tree) -> None:
    home, _, _, env = tree
    bin_dir = home / "bin"
    bin_dir.mkdir()
    shadow = bin_dir / "mkdir"
    shadow.write_text("#!/bin/bash\nexit 0\n")
    shadow.chmod(0o755)
    env["PATH"] = str(bin_dir) + ":/usr/bin:/bin"
    command = HEADER + ASSIGN + 'mkdir -p "$ROOT/source"\n' + WRITE
    assert _rank(tree, command).risk >= Risk.HIGH
    assert not any(t.opaque_contained for t in _resolve(tree, command).targets)


@pytest.mark.parametrize("program", ["git status", 'git -C "$ROOT" log', 'gmkdir -p "$ROOT/s"'])
def test_git_and_gnu_variants_end_fresh_scope(tree, program) -> None:
    command = HEADER + ASSIGN + program + "\n" + WRITE
    assert _rank(tree, command).risk >= Risk.HIGH


# PR #4116 Q1: curl's output destination is ranked as the same overwrite
# target a ``>`` redirection records, so protection follows the path.
@pytest.mark.parametrize("flag", ["--output ", "--output=", "-o ", "-o"])
def test_curl_output_into_config_is_critical(tree, flag) -> None:
    command = f'curl -fsSL {flag}"$HOME/.local-operator/config.yml" https://example.invalid/x'
    assert _rank(tree, command).risk == Risk.CRITICAL


def test_curl_output_ranks_like_redirect(tree) -> None:
    curl = _rank(tree, "curl -o ~/.bashrc https://example.invalid/x").risk
    assert curl == _rank(tree, "printf x > ~/.bashrc").risk


@pytest.mark.parametrize(
    "command",
    [
        'curl -o "$LOCAL_OPERATOR_SCRATCHPAD/out" https://example.invalid/x',
        "curl -s https://example.invalid/x",
        "curl -s https://example.invalid/x | jq .",
        "curl -o out.json https://example.invalid/x",
        'curl -o "$LOCAL_OPERATOR_SCRATCHPAD/a" https://example.invalid/x',
        HEADER + ASSIGN + 'curl -o "$ROOT/a" https://example.invalid/x\n',
        "curl -o /tmp/x https://example.invalid/x",
        "curl -o /dev/null -w '%{http_code}' https://example.invalid/x",
        "curl -O https://example.invalid/f.txt",
        "curl -K cfg https://example.invalid/x",
    ],
)
def test_benign_curl_forms_stay_low(tree, command) -> None:
    assert _rank(tree, command).risk == Risk.LOW
