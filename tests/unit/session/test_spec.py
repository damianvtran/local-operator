"""``local_operator.session.spec`` — the SDK's value objects, and the pins.

The parity test is the load-bearing one: ``SessionSpec.to_namespace()`` must
reproduce, field for field, the narrow namespace
``exec_mode._make_default_session_factory`` hands ``create_session``. Every
other property of the SDK ("the same machinery as ``lop exec``") rests on that
construction being the same construction; if the two namespaces can drift, that
claim stops being true silently, which is why the pin lives here rather than in
a review checklist.

The isolation tests around ``SessionRoots`` cover the defaults the facade
enforces: durable roots unless explicitly waived, and uid-default detection
answered against the passwd home rather than ``$HOME`` (an isolated run's
``$HOME`` lies — the same reasoning as ``browser_bridge/install.py``'s
``label()``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from local_operator.session.spec import (
    ApprovalPolicy,
    SessionRoots,
    SessionSpec,
    SessionSpecError,
    VolatileRootError,
    is_volatile_root,
    uid_home_dir,
)

REPO = Path(__file__).resolve().parents[3]

#: Dump the exact module set a virgin interpreter ends up with after importing
#: one target — same probe as ``tests/unit/test_import_graph.py``, copied
#: deliberately: an in-process ``sys.modules`` check is worthless after pytest
#: has imported half the tree.
_PROBE = """
import json, importlib, sys
importlib.import_module(sys.argv[1])
print(json.dumps(sorted(sys.modules)))
"""


def _imported_modules(target: str) -> set[str]:
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE, target],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        env={**os.environ, "LOCAL_OPERATOR_ALLOW_NESTED_SESSION": "1"},
    )
    assert proc.returncode == 0, f"importing {target} failed:\n{proc.stderr[-3000:]}"
    return set(json.loads(proc.stdout.strip().splitlines()[-1]))


async def _approve(tool_name: str, description: str) -> bool:
    """A correctly-shaped gate for the policy tests (async, two-argument)."""
    return True


# --- namespace parity with exec -------------------------------------------------


def _exec_namespace(monkeypatch: pytest.MonkeyPatch, exec_args: object) -> dict[str, object]:
    """Build exec's narrow session namespace, captured on its way to the factory.

    Same capture technique as ``tests/unit/test_exec_mode.py``'s stamp tests:
    the managers are replaced so no root is touched, and ``create_session`` is
    replaced so the namespace can be read without building anything.
    """
    seen: dict[str, object] = {}

    def fake_create_session(session_args, *managers, **kwargs):
        seen.update(vars(session_args))
        return None

    from local_operator import exec_mode

    monkeypatch.setattr("local_operator.config.ConfigManager", lambda *a, **k: object())
    monkeypatch.setattr("local_operator.agents.AgentRegistry", lambda *a, **k: object())
    monkeypatch.setattr("local_operator.session_factory.create_session", fake_create_session)
    exec_mode._make_default_session_factory(exec_args)()  # type: ignore[arg-type]
    assert seen, "exec's factory never reached create_session"
    return seen


#: exec's one INTERNAL session-args field the published SDK namespace does not
#: carry: only ``lop exec --team`` has a launch-time team to suggest a model,
#: and the SDK spec deliberately has no team surface (adding a permanently-None
#: ninth field to a published adapter shape is the silent surface change the
#: eight-field pin exists to stop). Pinned by NAME here so a second silent
#: exec-only field still fails this file.
_EXEC_ONLY_ARG = "team_model_suggestion"


def _exec_only_fields(seen: dict[str, object]) -> dict[str, object]:
    """``seen`` without exec's internal field, asserting that field is present and inert.

    The fixture builds runs without a ``--team``, so the value must be ``None``;
    anything else means an exec default changed under the tests that trust it.
    """
    shared = dict(seen)
    assert _EXEC_ONLY_ARG in seen, "exec stopped carrying its internal field at all"
    assert shared.pop(_EXEC_ONLY_ARG, None) is None
    return shared


def test_to_namespace_matches_the_exec_factory_field_for_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SDK and ``lop exec`` must hand ``create_session`` the same namespace.

    Field-for-field equality for the same inputs on every PINNED field, both
    for a fully-populated run and for the all-defaults run — the second catches
    a drift in DEFAULTS (a field exec later defaults differently) that a
    populated-only comparison would miss. The one deliberate exception is
    exec's internal ``team_model_suggestion`` (``_EXEC_ONLY_ARG``): a team
    launch's model suggestion, which the SDK cannot produce. The exception is
    pinned by name and by value, so it cannot widen unnoticed.
    """
    from local_operator import exec_mode

    # Annotated ``Any``: the kwargs mix str and bool, and ``**`` expansion of a
    # ``str | bool`` dict cannot satisfy either ``ExecArgs``' or ``SessionSpec``'s
    # per-field annotations for a checker.
    populated: dict[str, Any] = dict(
        hosting="openrouter",
        model="deepseek/deepseek-v4.1-flash",
        resume="abc123",
        workstream=True,
    )
    exec_seen = _exec_namespace(monkeypatch, exec_mode.ExecArgs(**populated))
    ours = vars(SessionSpec(**populated).to_namespace())
    assert ours == _exec_only_fields(exec_seen)

    defaults = _exec_namespace(monkeypatch, exec_mode.ExecArgs())
    assert vars(SessionSpec().to_namespace()) == _exec_only_fields(defaults)


def test_to_namespace_has_exactly_the_eight_pinned_fields() -> None:
    """No ninth field sneaks in, and none of the eight disappears.

    The count is pinned because both halves of the parity test above would keep
    passing if exec and the SDK grew the same new field together — which is
    fine for exec, and a silent surface change for a published adapter. Adding
    a field here is a deliberate act with a test to update.
    """
    assert set(vars(SessionSpec().to_namespace())) == {
        "hosting",
        "model",
        "agent_name",
        "agent_id",
        "yolo",
        "train",
        "resume",
        "workstream",
    }


def test_to_runner_args_is_accepted_by_the_exec_startup_helpers() -> None:
    """The adapter feeds ``resolve_startup`` / ``declared_tool_inventory`` as-is.

    This is the other half of the No namespace coupling mitigation: the
    runner-args namespace must carry every key the exec helpers read, in the
    shapes they read them — ``tools`` as exec's comma string, ``loop`` and
    ``clear_goal`` present-and-off rather than absent.
    """
    from local_operator.exec_startup import declared_tool_inventory, resolve_startup

    spec = SessionSpec(tools=("read", "grep"), name="nightly", goal="finish the audit")
    args = spec.to_runner_args()
    # resolve_startup validates every selector before any construction; it must
    # accept this namespace without falling onto a getattr default for a key
    # that should have been answered.
    assert resolve_startup(args) is None
    assert args.tools == "read,grep"
    assert args.clear_goal is False and args.loop is None and args.control is False
    # The team key is PRESENT and None (Q-MAJOR-3's fix): the helpers' getattr
    # reads are answered, and a set key is what lets resolve_startup validate —
    # and, beside a profile, refuse — a team when one IS named.
    assert args.team is None

    class SessionStub:
        attached_profile_tools = ()
        unresolved_declared_tools = ()

    assert declared_tool_inventory(SessionStub(), args) == ("read", "grep")


def test_to_runner_args_forwards_a_named_team_for_the_helpers() -> None:
    """The team reaches ``resolve_startup``; before Q-MAJOR-3's fix it was dropped.

    Team-only opened with no team attached and ``--team X --profile Y`` opened
    instead of being refused, because the runner-args namespace never carried
    the key (proven to one key on #2050's re-review). The spec field is the
    source; this cell pins the adapter's half of the fix.
    """
    spec = SessionSpec(team="lopdev")
    assert spec.to_runner_args().team == "lopdev"
    assert "team" not in vars(
        spec.to_namespace()
    ), "the eight-field narrow namespace stays pinned to exec's factory shape"


def test_with_resume_returns_a_copy_and_refuses_empty() -> None:
    spec = SessionSpec(hosting="test", model="mock")
    resumed = spec.with_resume("sess-1")
    assert resumed.resume == "sess-1"
    assert spec.resume is None, "with_resume must not mutate the original"
    with pytest.raises(SessionSpecError):
        spec.with_resume("")


@pytest.mark.parametrize(
    "kwargs, message_part",
    [
        ({"agent_name": "a", "agent_id": "b"}, "mutually exclusive"),
        ({"tools": ()}, "at least one tool"),
        ({"resume": ""}, "resume"),
        ({"yolo": True, "approvals": ApprovalPolicy.refuse()}, "unreachable"),
        ({"yolo": True, "approvals": ApprovalPolicy.callback(_approve)}, "unreachable"),
    ],
)
def test_spec_validation_refuses_before_anything_is_built(
    kwargs: dict[str, Any], message_part: str
) -> None:
    with pytest.raises(SessionSpecError) as excinfo:
        SessionSpec(**kwargs)
    assert message_part in str(excinfo.value)


def test_spec_field_surface_is_pinned() -> None:
    """The published dataclass surface: PR 2's consumers code against these names."""
    names = {field.name for field in dataclasses.fields(SessionSpec)}
    assert names == {
        "hosting",
        "model",
        "agent_name",
        "agent_id",
        "yolo",
        "train",
        "resume",
        "workstream",
        "birth_effort",
        "team",
        "profile",
        "tools",
        "approvals",
        "name",
        "goal",
        "notifications",
        # The output-contract fields (additive; applied post-open through
        # ``Session.set_output_contract``, never through the namespace).
        "output_format",
        "output_schema",
        "output_retries",
    }


# --- approval policy ------------------------------------------------------------


def test_approval_policy_presets_and_validation() -> None:
    assert ApprovalPolicy.refuse().mode == "refuse"
    assert ApprovalPolicy.auto().mode == "auto"
    declared = ApprovalPolicy.declared(["read", "grep", "read"])
    assert declared.mode == "declared"
    assert declared.tools == ("read", "grep"), "duplicates collapse, order kept"
    assert declared.stands_as_approval is True
    assert ApprovalPolicy.refuse().stands_as_approval is False

    with pytest.raises(SessionSpecError):
        ApprovalPolicy.declared([])
    with pytest.raises(SessionSpecError):
        ApprovalPolicy(mode="refuse", tools=("read",))
    with pytest.raises(SessionSpecError):
        ApprovalPolicy(mode="callback")  # no handler

    assert ApprovalPolicy.callback(_approve).handler is _approve


# --- roots ----------------------------------------------------------------------


def test_roots_resolve_and_expose_the_environment_they_imply(tmp_path: Path) -> None:
    roots = SessionRoots(
        config_dir=tmp_path / "cfg",
        agent_home=tmp_path / "ah",
        cwd=tmp_path / "work",
        allow_volatile=True,
    )
    env = roots.to_env()
    assert env["LOCAL_OPERATOR_CONFIG_DIR"] == str((tmp_path / "cfg").resolve())
    assert env["LOCAL_OPERATOR_HOME"] == str((tmp_path / "ah").resolve())
    assert roots.same_as(
        SessionRoots(config_dir=tmp_path / "cfg", agent_home=tmp_path / "other", cwd=tmp_path)
    ), "the config dir is the root identity"


def test_volatile_roots_are_refused_by_default(tmp_path: Path) -> None:
    """``$TMPDIR``-rooted stores can be purged under a live run — the failure
    that deleted a paid pilot's rescue root. The default refuses; the opt-out
    is one greppable word."""
    roots = SessionRoots(config_dir=tmp_path / "cfg", agent_home=tmp_path / "ah", cwd=tmp_path)
    with pytest.raises(VolatileRootError) as excinfo:
        roots.assert_durable()
    assert "allow_volatile=True" in str(excinfo.value)

    # The opt-out is honoured, and a durable location passes without it.
    SessionRoots(
        config_dir=tmp_path / "cfg", agent_home=tmp_path / "ah", cwd=tmp_path, allow_volatile=True
    ).assert_durable()
    assert is_volatile_root(tmp_path)
    assert is_volatile_root(Path("/tmp/x")) or is_volatile_root(Path("/private/tmp/x"))


def test_a_durable_home_path_is_not_volatile_or_this_test_says_why() -> None:
    """The predicate's negative leg, against the running uid's real home.

    Skipped where even the uid home is tmp-ish (a container with a wiped
    ``/tmp`` home), because there the honest answer is "cannot test this here",
    not a false pass.
    """
    home = uid_home_dir()
    if is_volatile_root(home):
        pytest.skip(f"uid home {home} is itself under a volatile root")
    assert not is_volatile_root(home / ".cache" / "local-operator")


def test_uid_home_dir_reads_the_passwd_entry_not_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``$HOME`` lies under isolation; the passwd answer cannot be redirected."""
    if sys.platform == "win32":  # pragma: no cover
        pytest.skip("no passwd database on Windows")
    monkeypatch.setenv("HOME", str(tmp_path / "redirected"))
    assert uid_home_dir() != tmp_path / "redirected"


# --- import contract -------------------------------------------------------------


def test_importing_spec_leaves_the_engine_off_the_graph() -> None:
    """``spec.py`` is the import-cheap half — stdlib plus ``paths`` only.

    The guard mirrors ``tests/unit/test_import_graph.py``: a fresh subprocess,
    because by test time everything is already imported in-process.
    """
    modules = _imported_modules("local_operator.session.spec")
    for banned in (
        "local_operator.session_factory",
        "local_operator.session.session",
        "local_operator.session.runtime.launch",
        "local_operator.session.runtime.serving",
        "local_operator.harness.types",
        "local_operator.harness.approval",
        "local_operator.providers",
    ):
        offenders = sorted(m for m in modules if m == banned or m.startswith(banned + "."))
        assert not offenders, f"{banned} is back on local_operator.session.spec's import path"
    assert "local_operator.paths" in modules


# --- output contract fields --------------------------------------------------------


def test_output_contract_field_refusals() -> None:
    """The three fields are cheaply validated where they are written: a spec
    that cannot possibly work never reaches construction."""
    with pytest.raises(SessionSpecError, match="output_format must be one of"):
        SessionSpec(output_format="xml")
    with pytest.raises(SessionSpecError, match="output_schema requires output_format"):
        SessionSpec(output_schema={"type": "object"})
    with pytest.raises(SessionSpecError, match="output_retries requires output_format"):
        SessionSpec(output_retries=1)
    for retries in (-1, 6):
        with pytest.raises(SessionSpecError, match="output_retries must be between 0 and 5"):
            SessionSpec(output_format="json", output_retries=retries)


def test_output_contract_fields_default_off_and_survive_with_resume() -> None:
    spec = SessionSpec(output_format="json", output_schema={"type": "object"}, output_retries=1)
    resumed = spec.with_resume("abc123def456")
    assert resumed.output_format == "json"
    assert resumed.output_schema == {"type": "object"}
    assert resumed.output_retries == 1
    # Default off, and the eight-field namespace pin is untouched by the three
    # new fields (they are applied post-open, like tools/goal/name).
    assert SessionSpec().output_format is None
    assert "output_format" not in vars(SessionSpec().to_namespace())
    assert "output_format" not in vars(spec.to_runner_args())


def test_the_spec_format_vocabulary_matches_the_contracts() -> None:
    """The tuple duplicated in ``spec.py`` (stdlib-only by design) is pinned to
    the contract's own list, so the duplication cannot drift."""
    from local_operator.output_contract import OUTPUT_FORMATS
    from local_operator.session import spec as spec_module

    assert spec_module._OUTPUT_FORMATS == OUTPUT_FORMATS
