import asyncio
import io
import sys
from datetime import datetime

import pytest

from local_operator.agents import AgentData
from local_operator.console import (
    condense_logging,
    format_agent_output,
    format_error_output,
    format_success_output,
    log_retry_error,
    print_agent_response,
    print_cli_banner,
    print_execution_section,
    print_task_interrupted,
    spinner,
)
from local_operator.types import ActionType


@pytest.fixture
def mock_config_manager():
    """Fixture that creates a mock ConfigManager."""

    class MockConfigManager:
        def __init__(self):
            self.config_dir = None
            self.config_file = None
            self.config = None

        def get_config_value(self, key):
            config = {
                "hosting": "test-host",
                "model_name": "test-model",
                "conversation_length": 10,
                "detail_length": 5,
            }
            return config.get(key)

        def get_config(self):
            return self.config

        def update_config(self, updates, write=True):
            pass

        def reset_to_defaults(self):
            pass

        def set_config_value(self, key, value):
            pass

    return MockConfigManager()


def test_print_cli_banner_basic(monkeypatch, mock_config_manager):
    # Capture stdout
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    # Override config values for this test
    def mock_get_config_value(key):
        return None

    mock_config_manager.get_config_value = mock_get_config_value

    # Call function
    print_cli_banner(mock_config_manager, current_agent=None, training_mode=False)
    result = output.getvalue()

    # Check basic banner elements are present
    assert "Local Executor Agent CLI" in result
    assert "You are interacting with a helpful CLI agent" in result
    assert "Type 'exit' or 'quit' to quit" in result
    assert "Press Ctrl+C to interrupt current task" in result

    # Check no hosting/model info shown when not configured
    assert "Using hosting:" not in result
    assert "Using model:" not in result


def test_print_cli_banner_with_config(monkeypatch, mock_config_manager):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    print_cli_banner(mock_config_manager, current_agent=None, training_mode=False)
    result = output.getvalue()

    # Check config values are displayed
    assert "Using hosting: test-host" in result
    assert "Using model: test-model" in result


def test_print_cli_banner_debug_mode(monkeypatch, mock_config_manager):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("LOCAL_OPERATOR_DEBUG", "true")

    print_cli_banner(mock_config_manager, current_agent=None, training_mode=False)
    result = output.getvalue()

    # Check debug info is shown
    assert "[DEBUG MODE]" in result
    assert "Configuration" in result
    assert "Conversation Length: 10" in result
    assert "Detail Length: 5" in result


def test_print_cli_banner_with_agent(monkeypatch, mock_config_manager):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    # Create mock agent metadata
    agent = AgentData(
        id="test-agent-id",
        name="Test Agent",
        created_date=datetime.now(),
        version="1.0.0",
        security_prompt="",
        hosting="",
        model="",
    )

    print_cli_banner(mock_config_manager, current_agent=agent, training_mode=False)
    result = output.getvalue()

    # Check agent info is displayed
    assert "Current agent: Test Agent" in result
    assert "Agent ID: test-agent-id" in result
    assert "Training Mode" not in result

    # Check other banner elements are still present
    assert "Local Executor Agent CLI" in result
    assert "Using hosting: test-host" in result
    assert "Using model: test-model" in result


def test_print_cli_banner_with_agent_and_config(monkeypatch, mock_config_manager):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    # Create mock agent metadata with custom hosting and model
    agent = AgentData(
        id="test-agent-id",
        name="Test Agent",
        created_date=datetime.now(),
        version="1.0.0",
        security_prompt="",
        hosting="custom-host",
        model="custom-model",
    )

    print_cli_banner(mock_config_manager, current_agent=agent, training_mode=False)
    result = output.getvalue()

    # Check agent info is displayed
    assert "Current agent: Test Agent" in result
    assert "Agent ID: test-agent-id" in result
    assert "Training Mode" not in result

    # Check other banner elements are still present
    assert "Local Executor Agent CLI" in result

    # Check that agent's hosting and model override config values
    assert "Using hosting: custom-host" in result
    assert "Using model: custom-model" in result


def test_print_cli_banner_with_agent_and_training(monkeypatch, mock_config_manager):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    monkeypatch.setenv("LOCAL_OPERATOR_DEBUG", "true")

    # Create mock agent metadata
    agent = AgentData(
        id="test-agent-id",
        name="Test Agent",
        created_date=datetime.now(),
        version="1.0.0",
        security_prompt="Security prompt",
        hosting="test-host",
        model="test-model",
    )

    print_cli_banner(mock_config_manager, current_agent=agent, training_mode=True)
    result = output.getvalue()

    # Check agent info and training mode are displayed
    assert "Current agent: Test Agent" in result
    assert "Agent ID: test-agent-id" in result
    assert "Training Mode" in result

    # Check other banner elements are still present
    assert "Local Executor Agent CLI" in result
    assert "Using hosting: test-host" in result
    assert "Using model: test-model" in result
    assert "Security prompt" in result


@pytest.mark.asyncio
async def test_spinner_cancellation(monkeypatch):
    # Capture sys.stdout output
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    # Start the spinner and let it run briefly
    task = asyncio.create_task(spinner("Testing spinner"))
    await asyncio.sleep(0.25)
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    # The spinner should clear the line on cancellation by writing "\r" at the end.
    assert output.getvalue().endswith("\r")


def test_log_retry_error_with_attempts(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    error = Exception("Retry error")
    # For attempt 0 (i.e. retry 1 of 3) the extra message should be printed.
    log_retry_error(error, attempt=0, max_retries=3)
    result = output.getvalue()

    assert "✗ Error during execution (attempt 1):" in result
    assert "Retry error" in result
    assert "Attempting to fix the error" in result


def test_log_retry_error_without_extra_message(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    error = Exception("Final error")
    # For attempt = max_retries - 1 (attempt 3 of 3) extra message should not be printed.
    log_retry_error(error, attempt=2, max_retries=3)
    result = output.getvalue()

    assert "✗ Error during execution (attempt 3):" in result
    assert "Final error" in result
    assert "Attempting to fix the error" not in result


def test_format_agent_output() -> None:
    """
    Test the format_agent_output function to ensure it correctly strips control tags
    and removes empty lines from the agent's raw text output.
    """
    # Prepare raw text with control tokens and blank lines.
    raw_text = "Hello\n[ASK]World\n\n[DONE]Goodbye"
    formatted = format_agent_output(raw_text)
    expected_output = "Hello\nWorld\nGoodbye"
    assert formatted == expected_output


def test_format_error_output():
    error = Exception("Operation failed")
    max_retries = 3
    output = format_error_output(error, max_retries)
    assert f"✗ Code Execution Failed after {max_retries} attempts" in output
    assert "Operation failed" in output


def test_format_success_output():
    stdout_text = "Output text"
    stderr_text = "Error text"
    log_text = "Log text"
    output = format_success_output((stdout_text, stderr_text, log_text))
    assert "✓ Code Execution Complete" in output
    assert "│ Output:" in output
    assert stdout_text in output
    assert "│ Error/Warning Output:" in output
    assert stderr_text in output
    assert "│ Log Output:" in output
    assert log_text in output


def test_format_success_output_no_logs():
    stdout_text = "Output text"
    stderr_text = "Error text"
    log_text = ""
    output = format_success_output((stdout_text, stderr_text, log_text))
    assert "✓ Code Execution Complete" in output
    assert "│ Output:" in output
    assert stdout_text in output
    assert "│ Error/Warning Output:" in output
    assert stderr_text in output
    assert "│ Log Output:" not in output


@pytest.mark.parametrize(
    "action_type, step, expected_output",
    [
        (ActionType.CODE, 1, "Executing Code (Step 1)"),
        (ActionType.WRITE, 2, "Executing Write (Step 2)"),
        (ActionType.EDIT, 3, "Executing Edit (Step 3)"),
        (ActionType.READ, 4, "Executing Read (Step 4)"),
        (ActionType.DONE, 5, "Executing Done (Step 5)"),
        (ActionType.ASK, 6, "Executing Ask (Step 6)"),
        (ActionType.BYE, 7, "Executing Bye (Step 7)"),
    ],
)
def test_print_execution_section_header(
    monkeypatch: pytest.MonkeyPatch, action_type: ActionType, step: int, expected_output: str
) -> None:
    output: io.StringIO = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    print_execution_section("header", step=step, action=action_type)
    result: str = output.getvalue()
    assert expected_output in result
    assert "╭─" in result


def test_print_execution_section_code(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    test_code = "print('Hello')"
    print_execution_section("code", content=test_code, action=ActionType.CODE)
    result = output.getvalue()
    assert "Executing:" in result
    assert "print('Hello')" in result


def test_print_execution_section_result(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    test_result = "Result is 42"
    print_execution_section("result", content=test_result, action=ActionType.CODE)
    result = output.getvalue()
    assert "Result:" in result
    assert "Result is 42" in result


def test_print_execution_section_footer(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    print_execution_section("footer", action=ActionType.CODE)
    result = output.getvalue()
    assert "╰" in result


def test_print_task_interrupted(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    print_task_interrupted()
    result = output.getvalue()
    assert "Task Interrupted" in result
    assert "User requested to stop current task" in result
    assert "╭─" in result
    assert "╰" in result


def test_print_agent_response(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    test_step = 3
    test_content = "This is a test response"
    print_agent_response(test_step, test_content)

    result = output.getvalue()
    assert f"Agent Response (Step {test_step})" in result
    assert test_content in result
    assert "╭─" in result
    assert "╰" in result


def test_print_agent_response_multiline(monkeypatch):
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)

    test_step = 1
    test_content = "Line 1\nLine 2\nLine 3"
    print_agent_response(test_step, test_content)

    result = output.getvalue()
    assert f"Agent Response (Step {test_step})" in result
    assert "Line 1" in result
    assert "Line 2" in result
    assert "Line 3" in result
    assert "╭─" in result
    assert "╰" in result


condense_test_case_console_output = """
Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT.
"""

condense_test_case_console_expected = """
Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
Please also refer to the documentation for alternative solver options:
    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
  n_iter_i = _check_optimize_result(
STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT. (3 identical multi-line blocks)"""

long_line_output = "line1\n" * 2000
long_line_expected = "...(1000 previous lines removed)\n" + "line1\n" * 1000


@pytest.mark.parametrize(
    "log_output, expected, test_id",
    [
        (
            "line1\nline1\nline2",
            "line1 (2 identical lines)\nline2",
            "test_consecutive_lines",
        ),
        (
            "line1\nline2\nline2\nline2",
            "line1\nline2 (3 identical lines)",
            "test_multiple_consecutive_lines",
        ),
        (
            "line1\nline2\nline3",
            "line1\nline2\nline3",
            "test_no_consecutive_lines",
        ),
        (
            "",
            "",
            "test_empty_string",
        ),
        (
            "line1",
            "line1",
            "test_single_line",
        ),
        (
            "line1\nline1",
            "line1 (2 identical lines)",
            "test_two_identical_lines",
        ),
        (
            "line1\nline1\nline1\nline1",
            "line1 (4 identical lines)",
            "test_multiple_identical_lines",
        ),
        (
            "line1\nline1\nline2\nline2\nline3",
            "line1 (2 identical lines)\nline2 (2 identical lines)\nline3",
            "test_mixed_consecutive_lines",
        ),
        (
            "pattern1\npattern2\npattern1\npattern2",
            "pattern1\npattern2 (2 identical multi-line blocks)",
            "test_repeating_pattern",
        ),
        (
            "pattern1\npattern2\npattern3\npattern1\npattern2\npattern3\npattern1\npattern2\n"
            "pattern3",
            "pattern1\npattern2\npattern3 (3 identical multi-line blocks)",
            "test_multiple_repeating_patterns",
        ),
        (
            "line1\npattern1\npattern2\npattern1\npattern2\nline2",
            "line1\npattern1\npattern2 (2 identical multi-line blocks)\nline2",
            "test_mixed_lines_and_patterns",
        ),
        (
            condense_test_case_console_output,
            condense_test_case_console_expected,
            "test_complex_console_output",
        ),
        (
            long_line_output,
            long_line_expected,
            "test_long_line_output",
        ),
    ],
    ids=lambda x: x[2] if isinstance(x, tuple) else x,
)
def test_condense_logging(log_output: str, expected: str, test_id: str) -> None:
    """
    Test the condense_logging function with various inputs and expected outputs.

    Args:
        log_output: The input log output string.
        expected: The expected condensed log output string.
        test_id: The ID of the test case.
    """
    result = condense_logging(log_output)
    assert result == expected
