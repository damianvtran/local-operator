"""Tier-1 static parsing of OSWorld V2 task modules.

These tests are the executable spec of the two-tier trust boundary: every
field must resolve from a static ``ast`` parse with NO task code executing,
and a field that is not statically resolvable must raise ``TaskParseError``
rather than fall back to import (executing unknown code to decide whether to
spend money is the failure the boundary exists to prevent). The ``instruction``
field is the ONE documented deferral: an unresolvable instruction records
None with ``instruction_static=False`` and the environment supplies the real
text at reset (``providers/aws.py``'s post-setup accessor read), because the
harness makes no decision on that text.
"""

from __future__ import annotations

import pytest
from lop_osworld_v2_adapter import taskfile
from lop_osworld_v2_adapter.taskfile import TaskParseError

from tests.unit.evaluation.adapters.osworld import fixtures


def test_plain_task_parses_every_field() -> None:
    descriptor = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/t.py")
    assert descriptor.task_id == "task_plain"
    assert descriptor.instruction == "Open the text editor and write hello."
    assert descriptor.config == ({"type": "launch", "app": "gedit"},)
    assert descriptor.related_apps == ("gedit",)
    assert descriptor.proxy is False
    assert descriptor.has_evaluator()
    assert len(descriptor.source_sha256) == 64


def test_proxy_flag_is_read() -> None:
    descriptor = taskfile.load_static(fixtures.PROXY.encode(), module_name="tasks/t.py")
    assert descriptor.proxy is True


def test_llm_simulator_is_read() -> None:
    descriptor = taskfile.load_static(fixtures.LLM_SIMULATOR.encode(), module_name="tasks/t.py")
    assert descriptor.user_simulator == {"type": "llm", "provider": "openai", "model": "gpt-4o"}


def test_custom_provisioning_fields_are_read() -> None:
    descriptor = taskfile.load_static(fixtures.CUSTOM_INSTANCE.encode(), module_name="tasks/t.py")
    assert descriptor.image == "ami-0123456789abcdef0"
    assert descriptor.instance_type == "t3.2xlarge"
    assert descriptor.volume_size == 100


def test_no_evaluator_task_reports_no_evaluator() -> None:
    descriptor = taskfile.load_static(fixtures.NO_EVALUATOR.encode(), module_name="tasks/t.py")
    assert not descriptor.has_evaluator()


def test_evaluate_override_counts_as_an_evaluator() -> None:
    """108 of 108 pinned V2 tasks score through an ``evaluate`` override and
    none declares an ``evaluator`` dict; ``has_evaluator`` must see them."""

    descriptor = taskfile.load_static(fixtures.EVALUATE_OVERRIDE.encode(), module_name="t.py")
    assert descriptor.evaluate_override is True
    assert descriptor.evaluator is None
    assert descriptor.has_evaluator() is True


def test_an_evaluate_outside_the_task_class_does_not_count() -> None:
    descriptor = taskfile.load_static(fixtures.EVALUATE_ELSEWHERE.encode(), module_name="t.py")
    assert descriptor.evaluate_override is False
    assert descriptor.has_evaluator() is False


def test_module_constants_class_attributes_fstrings_and_strip_fold_statically() -> None:
    """29 of 108 pinned tasks bind a field through exactly these shapes."""

    descriptor = taskfile.load_static(fixtures.FOLDED.encode(), module_name="t.py")
    assert descriptor.task_id == "013"
    assert descriptor.instruction == (
        "First line. Watch tutorial reaper-ducking-050 and then finish."
    )
    assert descriptor.instruction_static is True
    assert descriptor.user_simulator == {
        "type": "llm",
        "model": "gpt-4o",
        "instruction": "First line. Watch tutorial reaper-ducking-050 and then finish.",
    }


def test_an_unresolvable_instruction_defers_instead_of_folding_a_prefix() -> None:
    """An imported interpolation defers the instruction; no skeleton is kept.

    The old contract folded this to the literal parts
    (``"Open https://streamview./watch?v=1 and summarise."``). That prefix is
    a fabricated instruction: it is not what the task says, and a consumer
    cannot tell it from a real one. The deferred spelling -- None, with
    ``instruction_static`` False -- is the only honest static statement.
    """

    descriptor = taskfile.load_static(fixtures.PARTIAL_INSTRUCTION.encode(), module_name="t.py")
    assert descriptor.instruction_static is False
    assert descriptor.instruction is None


@pytest.mark.parametrize(
    ("source", "expected_config"),
    [
        (fixtures.DEFERRED_HELPER_INSTRUCTION, ({"type": "launch", "app": "gedit"},)),
        (fixtures.DEFERRED_CALL_INSTRUCTION, ()),
    ],
)
def test_a_helper_expression_instruction_defers_without_executing(
    source: str, expected_config: tuple[object, ...]
) -> None:
    """BinOp and call instructions defer, and the module never runs.

    ``DEFERRED_HELPER_INSTRUCTION``'s class body raises when evaluated, so a
    parser that imported the module to find out what the expression means
    would crash here rather than defer. The rest of the parse completes:
    the decision fields next to the deferred instruction are still read.
    """

    descriptor = taskfile.load_static(source.encode(), module_name="t.py")
    assert descriptor.instruction is None
    assert descriptor.instruction_static is False
    assert descriptor.config == expected_config
    # Never the stringified absence: ``str(None)`` would fabricate the text
    # "None" and hand a consumer exactly the fake instruction deferral exists
    # to avoid.
    assert descriptor.instruction != "None"


def test_an_unresolved_policy_field_refuses_next_to_a_deferred_instruction() -> None:
    """Deferral is scoped to ``instruction``: policy stays strict.

    ``platform`` here is an environment lookup, not a closed static shape, so
    the whole module refuses BEFORE allocation even though the instruction
    defers -- the same decision the field would force on its own.
    """

    with pytest.raises(TaskParseError, match="platform"):
        taskfile.load_static(
            fixtures.DEFERRED_INSTRUCTION_UNKNOWN_POLICY.encode(), module_name="t.py"
        )


def test_a_field_referencing_the_deferred_instruction_also_refuses() -> None:
    """Anything that READS the instruction symbol is strict, not deferred.

    The simulator config cannot resolve ``instruction`` (it deferred), and a
    simulator is a pre-allocation decision, so it refuses -- deferral never
    propagates into a decision field.
    """

    with pytest.raises(TaskParseError, match="user_simulator"):
        taskfile.load_static(fixtures.DEFERRED_INSTRUCTION_REFERENCED.encode(), module_name="t.py")


@pytest.mark.parametrize(
    ("source", "field"),
    [
        (fixtures.PARTIAL_ID, "id"),
        (fixtures.PARTIAL_SIMULATOR, "user_simulator"),
        (fixtures.CYCLIC, "id"),
    ],
)
def test_a_decision_field_that_cannot_fully_resolve_is_refused(source: str, field: str) -> None:
    with pytest.raises(taskfile.TaskParseError, match=field):
        taskfile.load_static(source.encode(), module_name="t.py")


def test_source_sha256_binds_the_exact_bytes() -> None:
    a = taskfile.load_static(fixtures.PLAIN.encode(), module_name="tasks/a.py")
    b = taskfile.load_static(fixtures.PLAIN.encode() + b"\n", module_name="tasks/b.py")
    assert a.source_sha256 != b.source_sha256


def test_non_literal_field_raises_instead_of_importing() -> None:
    # A field whose value is a call is not statically resolvable. The parser
    # must raise, never execute the module to resolve it.
    with pytest.raises(TaskParseError):
        taskfile.load_static(fixtures.NON_LITERAL.encode(), module_name="tasks/t.py")


def test_module_without_task_class_raises() -> None:
    with pytest.raises(TaskParseError):
        taskfile.load_static(b"x = 1\n", module_name="tasks/t.py")


def test_source_text_is_carried_for_requirement_derivation() -> None:
    descriptor = taskfile.load_static(fixtures.GITLAB.encode(), module_name="tasks/t.py")
    assert "controllers" in descriptor.source_text
    assert "gitlab" in descriptor.source_text


def test_infeasible_task_is_detected() -> None:
    """Round-1 review F4: the exclusion must be enforced, not just documented."""
    descriptor = taskfile.load_static(fixtures.INFEASIBLE.encode(), module_name="tasks/t.py")
    assert descriptor.is_infeasible()


def test_ordinary_tasks_are_not_flagged_infeasible() -> None:
    for source in (fixtures.PLAIN, fixtures.PROXY, fixtures.CLOCK, fixtures.NO_EVALUATOR):
        descriptor = taskfile.load_static(source.encode(), module_name="tasks/t.py")
        assert not descriptor.is_infeasible()
