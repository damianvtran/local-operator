"""Incident-shape tests for the dangerous-delete classifier.

The 2026-11-04 incident: an unattended session ran
``rm -rf ~/.local-operator/$(echo)/ 2>/dev/null; ...; rm -rf $W/scratchpad/realproof/ds*``
under auto approval. ``$(echo)`` expanded to nothing, the first segment became
``rm -rf ~/.local-operator//`` and the whole state root (config, secrets, agents,
projects, memory) was deleted. Every test here is a shape of that failure, or a
deletion the guard must keep letting through without any model call.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from local_operator.guardrails.classify import Risk, classify_bash


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    state = home / ".local-operator"
    (state / "sessions" / "own" / "scratchpad" / "realproof").mkdir(parents=True)
    (state / "sessions" / "other").mkdir(parents=True)
    (state / "secrets").mkdir()
    (state / "config.yml").write_text("x: 1\n")
    (home / "repo" / ".git").mkdir(parents=True)
    (home / "repo" / "pkg" / "build").mkdir(parents=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(state))
    return home


def _classify(home: Path, command: str, *, cwd: Path | None = None, env: dict | None = None):
    state = home / ".local-operator"
    base_env = {
        "HOME": str(home),
        "LOCAL_OPERATOR_SCRATCHPAD": str(state / "sessions" / "own" / "scratchpad"),
    }
    base_env.update(env or {})
    return classify_bash(
        command,
        cwd=str(cwd or home / "repo"),
        env=base_env,
        scratchpad=str(state / "sessions" / "own" / "scratchpad"),
        session_id="own",
    )


# -- CRITICAL: hard refusal, no model call ----------------------------------


def test_incident_echo_substitution_is_critical(home: Path) -> None:
    state = home / ".local-operator"
    verdict = _classify(
        home,
        "rm -rf ~/.local-operator/$(echo)/ 2>/dev/null; W=$HOME/.local-operator/sessions/own; "
        "rm -rf $W/scratchpad/realproof/ds*; ls",
    )
    assert verdict.risk is Risk.CRITICAL
    assert str(state) in [t.path for t in verdict.critical_targets]
    assert any("expanded empty" in r for r in verdict.reasons)


def test_unset_var_slash_is_critical(home: Path) -> None:
    assert _classify(home, "rm -rf $UNSET/").risk is Risk.CRITICAL


def test_quoted_home_is_critical(home: Path) -> None:
    assert _classify(home, 'rm -rf "$HOME"/').risk is Risk.CRITICAL


def test_cd_then_relative_is_critical(home: Path) -> None:
    assert _classify(home, "cd ~ && rm -rf .local-operator").risk is Risk.CRITICAL


def test_dotdot_normalises(home: Path) -> None:
    assert _classify(home, "rm -rf ~/.local-operator/sessions/..").risk is Risk.CRITICAL


def test_assignment_tracked(home: Path) -> None:
    assert _classify(home, "W=~/.local-operator; rm -rf $W").risk is Risk.CRITICAL


def test_export_assignment_tracked(home: Path) -> None:
    assert _classify(home, "export W=$HOME/.local-operator && rm -r $W").risk is Risk.CRITICAL


def test_find_delete_state_root(home: Path) -> None:
    assert _classify(home, "find ~/.local-operator -name '*.yml' -delete").risk is Risk.CRITICAL


def test_find_exec_rm_state_root(home: Path) -> None:
    verdict = _classify(home, "find ~/.local-operator -type f -exec rm {} +")
    assert verdict.risk is Risk.CRITICAL


def test_mv_away_config_yml(home: Path) -> None:
    assert _classify(home, "mv ~/.local-operator/config.yml /tmp/").risk is Risk.CRITICAL


def test_rsync_delete_into_home(home: Path) -> None:
    assert _classify(home, "rsync -a --delete /tmp/empty/ ~/").risk is Risk.CRITICAL


def test_bash_c_recursion(home: Path) -> None:
    assert _classify(home, "bash -c 'rm -rf ~/.local-operator'").risk is Risk.CRITICAL


def test_sudo_timeout_wrapped(home: Path) -> None:
    assert _classify(home, "sudo timeout 30 rm -rf ~/.local-operator").risk is Risk.CRITICAL


def test_glob_star_in_state_root_cwd(home: Path) -> None:
    verdict = _classify(home, "rm -rf *", cwd=home / ".local-operator")
    assert verdict.risk is Risk.CRITICAL


def test_generations_dir_critical(home: Path) -> None:
    assert _classify(home, "rm -rf ~/.local/share/lop/generations").risk is Risk.CRITICAL


def test_secrets_child_critical(home: Path) -> None:
    assert _classify(home, "rm ~/.local-operator/secrets/store.db").risk is Risk.CRITICAL


def test_truncate_config_yml_critical(home: Path) -> None:
    assert _classify(home, "truncate -s0 ~/.local-operator/config.yml").risk is Risk.CRITICAL


def test_redirect_onto_config_yml_critical(home: Path) -> None:
    assert _classify(home, "echo '' > ~/.local-operator/config.yml").risk is Risk.CRITICAL


def test_shred_secrets_critical(home: Path) -> None:
    assert _classify(home, "shred -u ~/.local-operator/secrets/key").risk is Risk.CRITICAL


def test_root_is_critical(home: Path) -> None:
    assert _classify(home, "rm -rf /").risk is Risk.CRITICAL


# -- HIGH: forked judge ------------------------------------------------------


def test_date_substitution_one_word_opaque_high(home: Path) -> None:
    verdict = _classify(home, "rm -rf ~/.local-operator/sessions/own/scratchpad/$(date +%s)")
    # Under the caller's own scratchpad the opaque suffix cannot climb out.
    assert verdict.risk is Risk.LOW
    verdict = _classify(home, "rm -rf ~/work/$(date +%s)")
    assert verdict.risk is Risk.HIGH
    assert any("opaque" in r for r in verdict.reasons)


def test_other_session_dir_high(home: Path) -> None:
    assert _classify(home, "rm -rf ~/.local-operator/sessions/other").risk is Risk.HIGH


def test_state_root_child_high(home: Path) -> None:
    assert _classify(home, "rm -rf ~/.local-operator/projects").risk is Risk.HIGH


def test_home_child_high(home: Path) -> None:
    assert _classify(home, "rm -rf ~/projects-dir").risk is Risk.HIGH


def test_repo_root_high(home: Path) -> None:
    assert _classify(home, f"rm -rf {home}/repo").risk is Risk.HIGH


def test_git_clean_x_high(home: Path) -> None:
    assert _classify(home, "git clean -fdx").risk is Risk.HIGH


def test_unknown_cwd_relative_high(home: Path) -> None:
    assert _classify(home, "cd $(mktemp -d) && rm -rf stuff").risk is Risk.HIGH


# -- LOW: ordinary deletes, zero judge calls ----------------------------------


def test_own_scratchpad_glob_low(home: Path) -> None:
    verdict = _classify(
        home, 'rm -rf "$LOCAL_OPERATOR_SCRATCHPAD"/realproof/ds*; ls $LOCAL_OPERATOR_SCRATCHPAD'
    )
    assert verdict.risk is Risk.LOW


def test_node_modules_low(home: Path) -> None:
    assert _classify(home, "rm -rf node_modules").risk is Risk.LOW


def test_tmp_low(home: Path) -> None:
    assert _classify(home, "rm /tmp/x").risk is Risk.LOW


def test_build_subdir_low(home: Path) -> None:
    assert _classify(home, "rm -rf build/", cwd=home / "repo" / "pkg").risk is Risk.LOW


def test_single_quoted_home_literal_low(home: Path) -> None:
    # '$HOME' in single quotes is a literal file name under the cwd, not $HOME.
    assert _classify(home, "rm -f '$HOME'").risk is Risk.LOW


def test_non_delete_commands_low(home: Path) -> None:
    for command in ("ls -la ~", "cat ~/.local-operator/config.yml", "git status", "echo rm -rf /"):
        assert _classify(home, command).risk is Risk.LOW, command


def test_rm_in_quoted_argument_low(home: Path) -> None:
    assert _classify(home, "git commit -m 'rm -rf ~ is bad'").risk is Risk.LOW


def test_mv_within_repo_low(home: Path) -> None:
    assert _classify(home, "mv src/a.py src/b.py").risk is Risk.LOW


def test_redirect_to_ordinary_file_low(home: Path) -> None:
    assert _classify(home, "echo hi > out.txt").risk is Risk.LOW


def test_git_clean_n_low(home: Path) -> None:
    assert _classify(home, "git clean -n").risk is Risk.LOW


def test_arbitrary_substitution_in_own_scratchpad_high(home: Path) -> None:
    # A body that could print `../..` climbs out of the pad: not contained.
    verdict = _classify(home, 'rm -rf "$LOCAL_OPERATOR_SCRATCHPAD"/$(cat name.txt)')
    assert verdict.risk is Risk.HIGH


def test_command_substitution_body_is_classified(home: Path) -> None:
    assert _classify(home, "echo $(rm -rf ~/.local-operator)").risk is Risk.CRITICAL


def test_xargs_rm_is_opaque_high(home: Path) -> None:
    assert _classify(home, "ls | xargs rm -rf").risk is Risk.HIGH


def test_subshell_group_still_resolves(home: Path) -> None:
    assert _classify(home, "( cd ~ && rm -rf .local-operator )").risk is Risk.CRITICAL
