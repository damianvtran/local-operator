"""Tier-1 static parsing of OSWorld V2 task modules.

These tests are the executable spec of the two-tier trust boundary: every
field must resolve from a static ``ast`` parse with NO task code executing,
and a field that is not statically resolvable must raise ``TaskParseError``
rather than fall back to import (executing unknown code to decide whether to
spend money is the failure the boundary exists to prevent).
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
