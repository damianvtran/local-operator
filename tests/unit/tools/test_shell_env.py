"""What a child process the MODEL asked for actually receives.

Behavioural, and deliberately over the REAL handlers: ``execute_bash`` (the real
``bash`` tool, spawning the real interpreter) and the real ``eval`` worker
subprocess. The property this file exists for is "which NAMES are in the child's
environment", and a test of the policy dataclass alone would stay green with the
policy wired to nothing — which is exactly the bug class that let a provider key
sit in a model-authored shell in the first place.

Every assertion here is over NAMES (or over booleans and counts computed inside
the child), never over a variable's value: a test that printed the environment
would put a credential in the transcript, which is the exposure being closed.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from local_operator.config import ConfigManager
from local_operator.harness.types import AbortSignal, ToolContext
from local_operator.tools import builtin
from local_operator.tools import eval as eval_tool
from local_operator.tools import shell_env
from local_operator.variables import VariableStore

#: The name of a provider key, as the E2 finding observed it in a lop child's
#: environment. Its VALUE here is a sentinel: nothing in this file depends on
#: what a real key looks like, only on the name being absent or present.
PROVIDER_KEY_NAME = "OPENROUTER_API_KEY"
PROVIDER_KEY_SENTINEL = "sentinel-not-a-real-key"
#: A non-credential name that the parent exports and the strict mode must drop
#: on its own (it is neither in the base set nor credential-shaped).
UNGRANTED_NAME = "LOP_E2_UNGRANTED"
UNGRANTED_SENTINEL = "1"
#: A credential-SHAPED name that is not a provider key: the name-shape floor,
#: not the provider table, is what has to catch this one.
SERVICE_TOKEN_NAME = "MINERVA_E2_SERVICE_TOKEN"
SERVICE_TOKEN_SENTINEL = "sentinel-not-a-real-token"


def _store_policy(config_dir: Path, **values: object) -> None:
    """Write ``shell_environment`` through the real ``ConfigManager``.

    The same path a deployment takes (an adapter writes the per-run
    ``config.yml``), so the reader under test resolves it the way it will in
    production rather than through an injected policy object.
    """
    stored = {"mode": "inherit", "inherit": [], "exclude": []}
    stored.update(values)
    ConfigManager(config_dir).set_config_value("shell_environment", stored)
    # The policy is memoised per config directory for the life of the process
    # (that memo IS the control — see the module docstring on M2). Tests use a
    # fresh directory each, but a test that stores twice must be able to see its
    # second write, so the test-only reset lives here. No production path calls
    # it: clearing it mid-run is exactly the weakening the memo prevents.
    shell_env.reset_policy_cache()


def _stdout_block(result) -> str:
    """The stdout section of a tool result, without the wrapper or the exit line.

    The result wraps both streams with ``--- stdout ---``/``--- stderr ---`` and
    may append an exit-code or ``result:`` line; parsing the block rather than
    the whole payload is what keeps a header out of a name set and out of a JSON
    parse.
    """
    assert not result.is_error, result.text
    text = result.text
    start = text.find("--- stdout ---")
    end = text.find("--- stderr ---")
    return text[start + len("--- stdout ---") : end if end > start else None]


def _child_environment_names(result) -> set[str]:
    """The child's variable NAMES, from a command that prints names only."""
    return {line.strip() for line in _stdout_block(result).splitlines() if line.strip()}


def _env_names_command() -> str:
    """A command that prints the child's variable names and nothing else.

    ``cut -d= -f1`` is the whole reason this is safe to assert on: a value can
    never reach the result, whatever the policy did.
    """
    return "env | cut -d= -f1"


def _session_store() -> VariableStore:
    """A store holding one session credential, the way an operator grants one."""
    store = VariableStore(cwd="/tmp", env={})
    result = store.store_credential("github token", "ghp_sentinel_1", "command")
    assert result.ok is True
    return store


@pytest.fixture
def config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A REAL config dir, pointed at by the environment the reader uses."""
    monkeypatch.setenv("LOCAL_OPERATOR_CONFIG_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _sentinel_parent_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The parent environment a strict mode has to defend against.

    The two non-interactive markers are DELETED from the parent on purpose: the
    harness's own shell exports them, so a test that left them there would
    assert the host rather than the tool's injection.
    """
    monkeypatch.setenv(PROVIDER_KEY_NAME, PROVIDER_KEY_SENTINEL)
    monkeypatch.setenv(UNGRANTED_NAME, UNGRANTED_SENTINEL)
    monkeypatch.setenv(SERVICE_TOKEN_NAME, SERVICE_TOKEN_SENTINEL)
    monkeypatch.delenv("LOCAL_OPERATOR_AGENT_SHELL", raising=False)
    monkeypatch.delenv("CI", raising=False)


@pytest_asyncio.fixture(autouse=True)
async def _clean_kernel_registry():
    """Kill any eval worker a test leaves behind (it outlives the test)."""
    eval_tool._KERNELS.clear()
    eval_tool._LOST_KERNELS.clear()
    eval_tool._ACTIVE_KERNELS.clear()
    eval_tool._CLOSE_ON_RETURN.clear()
    yield
    for kernel in list(eval_tool._KERNELS.values()):
        await eval_tool._close_kernel(kernel)
    eval_tool._KERNELS.clear()
    for task in list(eval_tool._CLOSING):
        task.cancel()


# ---------------------------------------------------------------------------
# bash — the real handler, the real interpreter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inherit_mode_is_unchanged(config_dir: Path, tmp_path: Path) -> None:
    """The default keeps today's behaviour: the child sees the parent env.

    This is the promise that makes the strict mode safe to ship — an
    interactive operator's own commands must not change until a deployment says
    so, and this is the assertion that holds that.
    """
    _store_policy(config_dir, mode="inherit")
    context = ToolContext(cwd=str(tmp_path), variables=_session_store())

    result = await builtin.execute_bash(
        "bash-env", {"command": _env_names_command()}, AbortSignal(), None, context
    )
    names = _child_environment_names(result)

    assert PROVIDER_KEY_NAME in names
    assert UNGRANTED_NAME in names
    assert SERVICE_TOKEN_NAME in names
    # …and the two intentional injections are there in this mode too, which is
    # what makes the allowlist assertions below meaningful rather than a
    # different baseline.
    assert "LOCAL_OPERATOR_AGENT_SHELL" in names
    assert "CI" in names
    assert "GITHUB_TOKEN" in names


@pytest.mark.asyncio
async def test_allowlist_mode_gives_the_child_only_the_grants(
    config_dir: Path, tmp_path: Path
) -> None:
    """The strict mode: base set + ``inherit`` + injections, and nothing else."""
    _store_policy(config_dir, mode="allowlist")
    context = ToolContext(cwd=str(tmp_path), variables=_session_store())

    result = await builtin.execute_bash(
        "bash-env",
        {"command": _env_names_command()},
        AbortSignal(),
        None,
        context,
    )
    names = _child_environment_names(result)

    # The exposure: the provider key the harness itself launched with is not in
    # the child a model-written command runs in.
    assert PROVIDER_KEY_NAME not in names
    # The name-shape floor, not the provider table: a service token this repo
    # has never heard of goes too.
    assert SERVICE_TOKEN_NAME not in names
    # A plain parent variable is not granted either — the mode grants NAMES, and
    # this one was never named.
    assert UNGRANTED_NAME not in names
    # What the child legitimately keeps: the safe set where the PARENT had it, and
    # the harness's own injections.
    #
    # Relative to ``os.environ`` on purpose. The base set is a filter over the parent
    # environment (``shell_env.BASE_ALLOWLIST``), not a set the child invents: a runner
    # that launches pytest with ``env -i`` — the repo's isolated pattern for keeping a
    # test run out of the operator's real home — has no ``SHELL``, and asserting the
    # name absolutely would fail for the runner's environment rather than for the code
    # under test. Asserting the intersection checks the property that matters: every
    # base name the parent had survives, and none the parent lacked appears.
    base_names = {"HOME", "PATH", "SHELL", "TERM", "LOGNAME", "USER"}
    assert (base_names & set(os.environ)) <= names
    assert "LOCAL_OPERATOR_AGENT_SHELL" in names
    assert "CI" in names
    assert "TERM" in names
    assert "GITHUB_TOKEN" in names


@pytest.mark.asyncio
async def test_allowlist_mode_granted_names_reach_the_child(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``inherit`` is a real grant, and it is the knob a deployment needs.

    ``LOCAL_OPERATOR_CONFIG_DIR`` is the motivating case: without it a shell in
    the strict mode cannot resolve ``$(lop secret get NAME)`` for a store held
    outside the default home.
    """
    monkeypatch.setenv("LOP_E2_GRANTED", "granted-value")
    _store_policy(config_dir, mode="allowlist", inherit=["LOP_E2_GRANTED"])
    context = ToolContext(cwd=str(tmp_path))

    result = await builtin.execute_bash(
        "bash-env",
        {"command": "printf 'granted=%s\\n' \"$LOP_E2_GRANTED\""},
        AbortSignal(),
        None,
        context,
    )

    assert not result.is_error, result.text
    assert "granted=granted-value" in result.text


@pytest.mark.asyncio
async def test_exclude_removes_a_name_in_both_modes(
    config_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``exclude`` is the operator's veto, and it wins over the injections too."""
    monkeypatch.setenv("LOP_E2_KEPT", "kept-value")
    _store_policy(config_dir, mode="inherit", exclude=["LOP_E2_KEPT", "GITHUB_TOKEN"])
    context = ToolContext(cwd=str(tmp_path), variables=_session_store())

    result = await builtin.execute_bash(
        "bash-env", {"command": _env_names_command()}, AbortSignal(), None, context
    )
    names = _child_environment_names(result)

    assert "LOP_E2_KEPT" not in names
    # The veto outranks the session credential store: naming a variable in
    # `exclude` is an operator saying "not even that".
    assert "GITHUB_TOKEN" not in names
    assert PROVIDER_KEY_NAME in names


# ---------------------------------------------------------------------------
# eval — the real worker subprocess
# ---------------------------------------------------------------------------


async def _eval_env_report(context: ToolContext, code: str) -> dict[str, Any]:
    tool = eval_tool.build_eval_tool()
    result = await tool.execute(  # type: ignore[operator]
        "call-1", {"code": code}, None, None, context
    )
    lines = [line for line in _stdout_block(result).splitlines() if line.strip()]
    report = json.loads(lines[-1])
    assert isinstance(report, dict)
    return report


_ENV_REPORT_CODE = (
    "import json, os; print(json.dumps({"
    "'names': len(os.environ),"
    f"'provider_key': {PROVIDER_KEY_NAME!r} in os.environ,"
    "'path': 'PATH' in os.environ, 'home': 'HOME' in os.environ,"
    "'agent_shell': os.environ.get('LOCAL_OPERATOR_AGENT_SHELL'),"
    "'ungranted': os.environ.get('LOP_E2_UNGRANTED'),"
    "}))"
)


@pytest.mark.asyncio
async def test_inherit_mode_eval_worker_keeps_the_parent_environment(
    config_dir: Path, tmp_path: Path
) -> None:
    """``inherit``: an eval cell reads the environment it always could."""
    _store_policy(config_dir, mode="inherit")
    context = ToolContext(cwd=str(tmp_path), session_id="eval-env-inherit")

    report = await _eval_env_report(context, _ENV_REPORT_CODE)

    assert report["provider_key"] is True
    assert report["path"] is True
    assert report["ungranted"] == UNGRANTED_SENTINEL


@pytest.mark.asyncio
async def test_allowlist_mode_eval_worker_loses_the_provider_key(
    config_dir: Path, tmp_path: Path
) -> None:
    """The eval worker is not a smaller exposure than a shell: the model writes
    the Python, so it can read ``os.environ`` and hand it to any subprocess the
    cell spawns. The same policy therefore governs the spawn.
    """
    _store_policy(config_dir, mode="allowlist")
    context = ToolContext(cwd=str(tmp_path), session_id="eval-env-allowlist")

    report = await _eval_env_report(context, _ENV_REPORT_CODE)

    assert report["provider_key"] is False
    assert report["ungranted"] is None
    assert report["path"] is True
    assert report["home"] is True
    # The eval worker has never received ``NON_INTERACTIVE_ENV`` (that contract
    # belongs to the bash tool, and this change does not widen the worker's
    # injections), so the strict mode leaves it with the safe set and the scrub
    # fd alone. Asserted rather than left implicit: a deployment reading "the
    # strict mode gives the child the safe set plus the injections" should see
    # which injections that actually is on this path.
    assert report["agent_shell"] is None
    assert report["names"] > 0


@pytest.mark.asyncio
async def test_allowlist_mode_still_runs_a_real_subprocess_from_a_cell(
    config_dir: Path, tmp_path: Path
) -> None:
    """The strict mode must not break the tool: a cell that spawns a command
    still works, because PATH is part of the safe set — and the grandchild
    inherits the same strict environment, which is the property that matters,
    since the model can spawn from a cell exactly as it can from bash.
    """
    _store_policy(config_dir, mode="allowlist")
    context = ToolContext(cwd=str(tmp_path), session_id="eval-env-subprocess")

    cell = (
        "import json, subprocess, sys\n"
        "probe = \"import os, sys; print('yes' if sys.argv[1] in os.environ else 'no')\"\n"
        "out = subprocess.run([sys.executable, '-c', probe, "
        + repr(PROVIDER_KEY_NAME)
        + "], capture_output=True, text=True)\n"
        "print(json.dumps({'sees_key': out.stdout.strip() == 'yes',"
        " 'returncode': out.returncode}))\n"
    )
    report = await _eval_env_report(context, cell)

    assert report["returncode"] == 0
    assert report["sees_key"] is False


# ---------------------------------------------------------------------------
# the policy reader itself
# ---------------------------------------------------------------------------


def test_absent_mode_defaults_to_inherit(config_dir: Path) -> None:
    assert shell_env.load_policy().mode == shell_env.MODE_INHERIT


def test_blank_mode_is_no_opinion_and_stays_inherit(config_dir: Path) -> None:
    """A blank value is what an unset field holds, not an intent to harden."""
    _store_policy(config_dir, mode="   ")
    assert shell_env.load_policy().mode == shell_env.MODE_INHERIT


def test_mode_is_read_case_insensitively(config_dir: Path) -> None:
    _store_policy(config_dir, mode=" ALLOWLIST ")
    assert shell_env.load_policy().mode == shell_env.MODE_ALLOWLIST


def test_unrecognised_mode_fails_closed_and_says_so(
    config_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A typo must not silently mean "off".

    This key exists to harden a run nobody is watching, so the strict mode is
    the safe reading of a value the reader cannot parse — and the WARNING naming
    the value is how the deployment finds out.
    """
    _store_policy(config_dir, mode="allow-list")

    with caplog.at_level("WARNING"):
        policy = shell_env.load_policy()

    assert policy.mode == shell_env.MODE_ALLOWLIST
    assert "allow-list" in caplog.text


def test_provider_key_names_cover_the_registry_without_being_restated() -> None:
    """The provider table is READ, not copied — the failure mode of a second
    list is that it stops covering the provider added later."""
    from local_operator.model.registry import SupportedHostingProviders

    names = shell_env.provider_credential_names()
    for detail in SupportedHostingProviders:
        for name in detail.requiredCredentials or ():
            assert name.upper() in names, name
    assert "OPENROUTER_API_KEY" in names
    # The callable form is why the shape markers are needed beside the table:
    # Anthropic's resolver picks between two names and answers ``None`` to a
    # name-only question, so its key names come from the second source.
    assert shell_env.is_credential_shaped("ANTHROPIC_OAUTH_TOKEN") is True
    assert shell_env.is_credential_shaped("PATH") is False
    assert shell_env.is_credential_shaped("LANG") is False


def test_base_allowlist_matches_the_sdk() -> None:
    """The SDK's safe set is the contract this module restates; a change there
    must show up here rather than as a silent difference in a child's env."""
    mcp_stdio = pytest.importorskip("mcp.client.stdio")

    assert set(shell_env.BASE_ALLOWLIST) == set(mcp_stdio.DEFAULT_INHERITED_ENV_VARS)


def test_child_environment_subtracts_exclude_after_injections() -> None:
    """The ordering is the contract: exclude is applied LAST, so it can deny an
    injection as well as an inherited name."""
    policy = shell_env.ShellEnvironmentPolicy(
        mode=shell_env.MODE_ALLOWLIST, exclude=frozenset({"GITHUB_TOKEN"})
    )

    env = shell_env.child_environment(
        policy,
        parent={"PATH": "/usr/bin", "GITHUB_TOKEN": "inherited"},
        injections={"GITHUB_TOKEN": "from-the-store", "CI": "1"},
    )

    assert env == {"PATH": "/usr/bin", "CI": "1"}


# ---------------------------------------------------------------------------
# The policy cannot be LOWERED mid-run (review round 1, M2)
# ---------------------------------------------------------------------------
#
# ``config.yml`` sits in a directory the agent's own shell can write, as the
# same uid. A policy re-read per command is therefore a policy the constrained
# party can turn off between two commands: flip the mode back to ``inherit``
# from the very shell the strict mode was constraining and the next command in
# the same session gets the provider key back. These tests hold the fix to what
# it claims — resolved once, and never lowerable by a write during the run —
# and they use the REAL handlers, because the claim is about what a second
# command actually receives.


@pytest.mark.asyncio
async def test_a_rewrite_of_the_runs_own_config_cannot_lower_the_policy(
    config_dir: Path, tmp_path: Path
) -> None:
    """The reproduced flip: strict, rewrite, and the next command is still strict.

    Written DIRECTLY through ``ConfigManager`` rather than through
    ``_store_policy``, and that is the point of the test: the helper clears the
    memo, and clearing the memo is precisely what the model's own shell cannot
    do.
    """
    _store_policy(config_dir, mode="allowlist")
    context = ToolContext(cwd=str(tmp_path), variables=_session_store())

    first = await builtin.execute_bash(
        "bash-env", {"command": _env_names_command()}, AbortSignal(), None, context
    )
    assert PROVIDER_KEY_NAME not in _child_environment_names(
        first
    ), "the strict policy must not hand the key to the first command"

    # The model's own shell rewrites the run's config, the way the review
    # reproduced it (``sed -i '' 's/mode: allowlist/mode: inherit/'``).
    ConfigManager(config_dir).set_config_value("shell_environment", {"mode": "inherit"})
    assert (
        shell_env.load_policy().mode == shell_env.MODE_ALLOWLIST
    ), "a write during the run must not lower the resolved policy"

    second = await builtin.execute_bash(
        "bash-env", {"command": _env_names_command()}, AbortSignal(), None, context
    )
    names = _child_environment_names(second)
    assert PROVIDER_KEY_NAME not in names, (
        "the SECOND command in the same session must not get the key back: the "
        "policy is resolved per process, and a rewrite cannot weaken a run in flight"
    )
    assert SERVICE_TOKEN_NAME not in names


def test_a_missing_config_keeps_the_permissive_default(config_dir: Path) -> None:
    """No policy file means no policy: the pre-change behaviour, unchanged."""
    shell_env.reset_policy_cache()
    assert shell_env.load_policy().mode == shell_env.MODE_DEFAULT


def test_an_existing_but_unreadable_config_fails_closed_and_warns(
    config_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A hardened deployment must not lose the protection because of a bad read.

    A half-written or unparseable ``config.yml`` used to fall back to the
    PERMISSIVE default with a DEBUG line, so a deployment that believed it was
    hardened silently was not. The asymmetry is now explicit: no file at all is
    "no opinion" (above), a file that exists and cannot be read is "protect
    until told otherwise", and the operator's log says so.
    """
    (config_dir / "config.yml").write_text(
        "values: [this is not a mapping: {{{\n", encoding="utf-8"
    )
    shell_env.reset_policy_cache()
    with caplog.at_level(logging.WARNING, logger=shell_env.logger.name):
        policy = shell_env.load_policy()

    assert (
        policy.mode == shell_env.MODE_ALLOWLIST
    ), "an unreadable config must resolve to the strict policy, not to the permissive one"
    warnings = [record for record in caplog.records if record.levelno >= logging.WARNING]
    assert any(
        "shell_environment policy could not be read" in r.getMessage() for r in warnings
    ), "the operator's log must name the failure; a silent downgrade is the bug"


@pytest.mark.asyncio
async def test_a_strict_run_still_works_after_an_unreadable_config(
    config_dir: Path, tmp_path: Path
) -> None:
    """Failing closed must still run the command: strict, not broken."""
    (config_dir / "config.yml").write_text("values: [broken\n", encoding="utf-8")
    shell_env.reset_policy_cache()
    context = ToolContext(cwd=str(tmp_path), variables=_session_store())

    result = await builtin.execute_bash(
        "bash-env", {"command": _env_names_command()}, AbortSignal(), None, context
    )

    names = _child_environment_names(result)
    assert "PATH" in names, "a command with a strict environment still has to run"
    assert PROVIDER_KEY_NAME not in names


def test_a_denial_name_that_matches_nothing_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A typo in ``exclude`` is a denial that silently is not happening."""
    policy = shell_env.ShellEnvironmentPolicy(
        mode=shell_env.MODE_ALLOWLIST,
        exclude=frozenset({"LOP_E2_ABSENT_NAME"}),
    )
    with caplog.at_level(logging.WARNING, logger=shell_env.logger.name):
        shell_env.child_environment(policy, parent={"PATH": "/usr/bin"})

    messages = [r.getMessage() for r in caplog.records]
    assert any(
        "exclude" in m and "LOP_E2_ABSENT_NAME" in m for m in messages
    ), "an exclude name that matched nothing must be reported at WARNING"


def test_a_grant_name_that_matches_nothing_is_reported_at_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The grant side is reported too, but quietly: a missing name is often benign."""
    policy = shell_env.ShellEnvironmentPolicy(
        mode=shell_env.MODE_ALLOWLIST,
        inherit=("LOP_E2_ABSENT_GRANT",),
    )
    with caplog.at_level(logging.DEBUG, logger=shell_env.logger.name):
        shell_env.child_environment(policy, parent={"PATH": "/usr/bin"})

    debug_messages = [r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG]
    assert any(
        "inherit" in m and "LOP_E2_ABSENT_GRANT" in m for m in debug_messages
    ), "an inherit name that matched nothing must be reported at DEBUG"


def test_the_name_machinery_has_no_enforcement_role_in_suppression() -> None:
    """The markers are insurance, pinned so a reader cannot mistake them.

    Strict-mode suppression comes from NOT GRANTING: an unlisted name is dropped
    whatever its shape is. This test fixes that reading — a name that is neither
    credential-shaped nor granted is still absent — so nobody closes a leak by
    adding a marker to ``CREDENTIAL_NAME_MARKERS`` and believing it filtered
    something.
    """
    policy = shell_env.ShellEnvironmentPolicy(mode=shell_env.MODE_ALLOWLIST)
    plain_ungranted = "LOP_E2_PLAIN_UNGRANTED"

    assert not shell_env.is_credential_shaped(plain_ungranted)
    env = shell_env.child_environment(
        policy, parent={"PATH": "/usr/bin", plain_ungranted: "1", PROVIDER_KEY_NAME: "s"}
    )
    assert plain_ungranted not in env
    assert PROVIDER_KEY_NAME not in env


def test_denying_an_injection_counts_as_a_denial_that_happened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``exclude`` exists to deny what the harness hands over; that is not a miss.

    The unmatched-name report measures membership BEFORE the denial runs, so a
    name that was present only because the harness injected it still counts as
    matched. Asking after the pop reported the loudest WORKING denial — denying
    ``CI`` or the session credential, which is what the README and the settings
    help text advertise — as a denial that "matches nothing" and is "silently a
    no-op".
    """
    policy = shell_env.ShellEnvironmentPolicy(
        mode=shell_env.MODE_ALLOWLIST,
        exclude=frozenset({"CI"}),
    )
    with caplog.at_level(logging.WARNING, logger=shell_env.logger.name):
        env = shell_env.child_environment(
            policy,
            parent={"PATH": "/usr/bin"},
            injections={"CI": "1", "LOCAL_OPERATOR_AGENT_SHELL": "1"},
        )

    assert "CI" not in env, "the denial must still apply"
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any("exclude" in m for m in warnings), (
        "a denial that removed an injected name DID happen; reporting it as unmatched "
        f"is a false alarm: {warnings}"
    )


def test_denying_a_parent_defined_name_is_a_denial_that_happened(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The README's own sample config must not raise a false alarm.

    ``mode: allowlist`` with ``exclude: [GH_TOKEN]`` is the sample this repo
    documents. The name is defined in the PARENT and never granted to the child,
    so it is not in the child's names — and reporting that as "the environment
    being filtered does not define it" is both false and noisy on a correct
    config. Membership is measured against the child's names plus the parent's
    plus the injections, so a denial of anything the harness could have handed
    over counts as matched, while a genuine typo still warns.
    """
    policy = shell_env.ShellEnvironmentPolicy(
        mode=shell_env.MODE_ALLOWLIST,
        exclude=frozenset({"GH_TOKEN"}),
    )
    with caplog.at_level(logging.WARNING, logger=shell_env.logger.name):
        env = shell_env.child_environment(
            policy, parent={"PATH": "/usr/bin", "GH_TOKEN": "sentinel-not-a-token"}
        )

    assert "GH_TOKEN" not in env
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert not any(
        "GH_TOKEN" in m for m in warnings
    ), f"a denial of a parent-defined name is not a miss: {warnings}"

    # And the typo case still warns: the fix must not silence the real signal.
    caplog.clear()
    typo = shell_env.ShellEnvironmentPolicy(
        mode=shell_env.MODE_ALLOWLIST, exclude=frozenset({"GH_TOKEN"})
    )
    with caplog.at_level(logging.WARNING, logger=shell_env.logger.name):
        shell_env.child_environment(typo, parent={"PATH": "/usr/bin"})
    assert any(
        "GH_TOKEN" in r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING
    ), "a name nothing defines must still warn"
