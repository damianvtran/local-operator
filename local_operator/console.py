import asyncio
import itertools
import os
import sys
from contextlib import asynccontextmanager
from enum import Enum
from typing import Any, AsyncGenerator, Awaitable, Callable, List, TypeVar

from local_operator.agents import AgentData
from local_operator.config import ConfigManager
from local_operator.types import ActionType


class ExecutionSection(Enum):
    """Enum for execution section types.

    This enum defines the different sections that can be printed during the execution of a task.
    Each section type corresponds to a specific part of the task's output or action.

    Attributes:
        HEADER: Indicates the header section, which includes the step number and action.
        CODE: Indicates the code section, which displays the code to be executed.
        RESULT: Indicates the result section, which shows the output of the executed code.
        FOOTER: Indicates the footer section, which marks the end of the task.
        TOKEN_USAGE: Indicates the token usage section, which provides details on token consumption.
        WRITE: Indicates the write section, which shows file writing operations.
        EDIT: Indicates the edit section, which details file editing operations.
        READ: Indicates the read section, which shows file reading operations.
    """

    HEADER = "header"
    CODE = "code"
    RESULT = "result"
    FOOTER = "footer"
    TOKEN_USAGE = "token_usage"
    WRITE = "write"
    EDIT = "edit"
    READ = "read"


class VerbosityLevel(int, Enum):
    """The level of detail to output from the operator in the CLI."""

    QUIET = 0
    INFO = 1
    VERBOSE = 2
    DEBUG = 3


def wrap_text_to_width(text: str, max_width: int, first_line_prefix: str = "") -> list[str]:
    """
    Wrap text to a specified width, handling a prefix on the first line.

    Args:
        text (str): The text to wrap
        max_width (int): Maximum width for each line
        first_line_prefix (str): Prefix to add to first line, reducing its available width

    Returns:
        list[str]: List of wrapped lines
    """
    words = text.split()
    lines = []
    current_line = []
    current_length = len(first_line_prefix) if first_line_prefix else 0

    for word in words:
        # +1 for the space between words
        if current_length + len(word) + 1 <= max_width:
            current_line.append(word)
            current_length += len(word) + 1
        else:
            lines.append(" ".join(current_line))
            current_line = [word]
            current_length = len(word)

    if current_line:
        lines.append(" ".join(current_line))

    return lines


def print_cli_banner(
    config_manager: ConfigManager, current_agent: AgentData | None, training_mode: bool
) -> None:
    """
    Print the banner for the chat CLI.

    This function prints a banner for the Local Executor Agent CLI, including details about the
    current agent, hosting, model, and configuration options. It also indicates if the CLI is in
    debug or training mode.

    Args:
        config_manager (ConfigManager): The configuration manager to retrieve settings.
        current_agent (AgentData | None): The current agent data, if available.
        training_mode (bool): Whether the CLI is in training mode.
    """
    debug_mode = os.getenv("LOCAL_OPERATOR_DEBUG", "false").lower() == "true"

    hosting = config_manager.get_config_value("hosting")
    model = config_manager.get_config_value("model_name")

    if current_agent:
        if current_agent.hosting:
            hosting = current_agent.hosting
        if current_agent.model:
            model = current_agent.model

    debug_indicator = " [DEBUG MODE]" if debug_mode else ""
    print("\033[1;36m╭──────────────────────────────────────────────────╮\033[0m")
    print(f"\033[1;36m│ Local Executor Agent CLI{debug_indicator:<25}│\033[0m")
    print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
    print("\033[1;36m│ You are interacting with a helpful CLI agent     │\033[0m")
    print("\033[1;36m│ that can execute tasks locally on your device    │\033[0m")
    print("\033[1;36m│ by running Python code.                          │\033[0m")
    print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
    if current_agent:
        agent_name = f"Current agent: {current_agent.name}"
        padding = 49 - len(agent_name)
        print(f"\033[1;36m│ {agent_name}{' ' * padding}│\033[0m")
        agent_id = f"Agent ID: {current_agent.id}"
        padding = 49 - len(agent_id)
        print(f"\033[1;36m│ {agent_id}{' ' * padding}│\033[0m")
        if training_mode:
            training_text = "** Training Mode **"
            padding = 49 - len(training_text)
            print(f"\033[1;36m│ {training_text}{' ' * padding}│\033[0m")
        print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
    if hosting:
        hosting_text = f"Using hosting: {hosting}"
        padding = 49 - len(hosting_text)
        print(f"\033[1;36m│ {hosting_text}{' ' * padding}│\033[0m")
    if model:
        model_text = f"Using model: {model}"
        padding = 49 - len(model_text)
        print(f"\033[1;36m│ {model_text}{' ' * padding}│\033[0m")
    autosave_enabled = config_manager.get_config_value("auto_save_conversation", False)
    autosave_text = f"Autosave: {'Enabled' if autosave_enabled else 'Disabled'}"
    padding = 49 - len(autosave_text)
    print(f"\033[1;36m│ {autosave_text}{' ' * padding}│\033[0m")
    print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
    print("\033[1;36m│ Type 'exit' or 'quit' to quit                    │\033[0m")
    print("\033[1;36m│ Press Ctrl+C to interrupt current task           │\033[0m")
    print("\033[1;36m╰──────────────────────────────────────────────────╯\033[0m\n")

    # Print configuration options
    if debug_mode:
        print("\033[1;36m╭─ Configuration ────────────────────────────────\033[0m")
        print(f"\033[1;36m│\033[0m Hosting: {config_manager.get_config_value('hosting')}")
        print(f"\033[1;36m│\033[0m Model: {config_manager.get_config_value('model_name')}")
        conv_len = config_manager.get_config_value("conversation_length")
        detail_len = config_manager.get_config_value("detail_length")
        print(f"\033[1;36m│\033[0m Conversation Length: {conv_len}")
        print(f"\033[1;36m│\033[0m Detail Length: {detail_len}")
        print(f"\033[1;36m│\033[0m Training Mode: {training_mode}")
        if current_agent and current_agent.security_prompt:
            security_prompt = current_agent.security_prompt
            lines = wrap_text_to_width(security_prompt, 49, "Security Prompt: ")

            print(f"\033[1;36m│\033[0m Security Prompt: {lines[0]}")
            for line in lines[1:]:
                padding = " " * len("Security Prompt: ")
                print(f"\033[1;36m│\033[0m {padding}{line}")
        print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")


async def spinner(text: str):
    """
    Asynchronously display a rotating spinner with the provided text.

    This coroutine continuously displays a rotating spinner in the terminal alongside the given
    text, updating every 0.1 seconds. If the spinner is cancelled via asyncio.CancelledError, it
    clears the spinner display and exits gracefully.

    Args:
        text (str): The message to display alongside the spinner.
    """
    spinner_cycle = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
    while True:
        sys.stdout.write(f"\r\033[1;36m{next(spinner_cycle)} {text}\033[0m")
        sys.stdout.flush()
        try:
            await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            sys.stdout.write("\r")
            break


T = TypeVar("T")


@asynccontextmanager
async def spinner_context(
    message: str, verbosity_level: VerbosityLevel
) -> AsyncGenerator[None, None]:
    """Context manager for displaying a spinner during async operations.

    Args:
        message: The message to display alongside the spinner
        verbosity_level: The verbosity level to use for the spinner

    Yields:
        None

    Example:
        ```python
        async with spinner_context("Processing data"):
            result = await some_long_running_operation()
        ```
    """
    spinner_task = None
    if verbosity_level >= VerbosityLevel.VERBOSE:
        spinner_task = asyncio.create_task(spinner(message))
    try:
        yield
    finally:
        if spinner_task:
            spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass


async def with_spinner(
    message: str,
    verbosity_level: VerbosityLevel,
    coro_func: Callable[..., Awaitable[T]],
    *args,
    **kwargs,
) -> T:
    """Execute a coroutine function with a spinner.

    Args:
        message: The message to display alongside the spinner
        coro_func: The coroutine function to execute
        *args: Positional arguments to pass to the coroutine function
        **kwargs: Keyword arguments to pass to the coroutine function

    Returns:
        The result of the coroutine function

    Example:
        ```python
        result = await with_spinner(
            "Processing data",
            process_data,
            data_input
        )
        ```
    """
    async with spinner_context(message, verbosity_level):
        return await coro_func(*args, **kwargs)


def log_action_error(error: Exception, action: str, verbosity_level: VerbosityLevel) -> None:
    """Log an error that occurred during an action, including the traceback.

    Args:
        error (Exception): The error that occurred.
        action (str): The action that occurred.
    """
    if verbosity_level < VerbosityLevel.VERBOSE:
        return

    error_str = str(error)
    print(f"\n\033[1;31m✗ Error during {action}:\033[0m")
    print("\033[1;34m╞══════════════════════════════════════════════════╡\033[0m")
    print(f"\033[1;36m│ Error:\033[0m\n{error_str}")
    print("\033[1;34m╞══════════════════════════════════════════════════╡\033[0m")


def log_retry_error(
    error: Exception, attempt: int, max_retries: int, verbosity_level: VerbosityLevel
) -> None:
    """
    Print a formatted error message for a given retry attempt.

    Args:
        error (Exception): The error that occurred.
        attempt (int): The current retry attempt number.
        max_retries (int): The maximum number of retry attempts allowed.
        verbosity_level (VerbosityLevel): The verbosity level to use for the section.
    """
    if verbosity_level < VerbosityLevel.VERBOSE:
        return

    error_str = str(error)
    print(f"\n\033[1;31m✗ Error during execution (attempt {attempt + 1}):\033[0m")
    print("\033[1;34m╞══════════════════════════════════════════════════╡\033[0m")
    print(f"\033[1;36m│ Error:\033[0m\n{error_str}")
    if attempt < max_retries - 1:
        print("\033[1;36m│\033[0m \033[1;33mAttempting to fix the error...\033[0m")


def format_agent_output(text: str) -> str:
    """
    Format agent output by stripping control tags.

    Args:
        text (str): Raw agent output text.

    Returns:
        str: The formatted text.
    """
    output = text.replace("[ASK]", "").replace("[DONE]", "").replace("[BYE]", "").strip()
    # Remove any empty (or whitespace-only) lines.
    lines = [line for line in output.split("\n") if line.strip()]
    return "\n".join(lines)


def format_error_output(error: Exception, max_retries: int) -> str:
    """Format error output message with ANSI color codes.

    Args:
        error (Exception): The error to format
        max_retries (int): Number of retry attempts made

    Returns:
        str: Formatted error message string
    """
    error_str = str(error)
    return (
        f"\n\033[1;31m✗ Code Execution Failed after {max_retries} attempts\033[0m\n"
        f"\033[1;34m╞══════════════════════════════════════════════════╡\n"
        f"\033[1;36m│ Error:\033[0m\n{error_str}"
    )


def format_success_output(output: tuple[str, str, str]) -> str:
    """Format successful execution output with ANSI color codes.

    Args:
        output (tuple[str, str, str]): Tuple containing (stdout output, stderr output, log output)

    Returns:
        str: Formatted string with colored success message and execution output
    """
    stdout, stderr, log_output = output
    print_str = (
        "\n\033[1;32m✓ Code Execution Complete\033[0m\n"
        "\033[1;34m╞══════════════════════════════════════════════════╡\n"
        f"\033[1;36m│ Output:\033[0m\n{stdout}\n"
        f"\033[1;36m│ Error/Warning Output:\033[0m\n{stderr}"
    )

    if log_output:
        print_str += f"\n\033[1;36m│ Log Output:\033[0m\n{log_output}"

    return print_str


def print_agent_response(step: int, content: str, verbosity_level: VerbosityLevel) -> None:
    """
    Print the agent's response with formatted styling.

    Args:
        step (int): The current step number
        content (str): The agent's response content to display
        verbosity_level (VerbosityLevel): The verbosity level to use for the section.
    """
    if verbosity_level < VerbosityLevel.INFO:
        return

    print(f"\n\033[1;36m╭─ Agent Response (Step {step}) ──────────────────────────\033[0m")
    print(content)
    print("\033[1;36m╰══════════════════════════════════════════════════╯\033[0m")


def print_execution_section(
    section: ExecutionSection | str,
    verbosity_level: VerbosityLevel,
    *,
    step: int | None = None,
    content: str = "",
    data: dict[str, Any] | None = None,
    file_path: str | None = None,
    replacements: list[dict[str, str]] | None = None,
    action: ActionType | None = None,
) -> None:
    """
    Print a section of the execution output.

    Parameters:
        section (ExecutionSection | str): One of ExecutionSection values or their
        string equivalents:
            - HEADER: Prints a header with the step number; requires 'step'.
            - CODE: Prints the code to be executed; requires 'content'.
            - RESULT: Prints the result of code execution; requires 'content'.
            - FOOTER: Prints a footer.
            - TOKEN_USAGE: Prints the token usage for the current session.
            - WRITE: Prints the content of a file to be written.
            - EDIT: Prints the content of a file to be edited.
            - READ: Prints the content of a file to be read.
        verbosity_level (VerbosityLevel): The verbosity level to use for the section.
        step (int, optional): The step number (required for "header").
        content (str, optional): The content to be printed for the "code" or "result" sections.
        data (dict[str, Any], optional): Data to be printed for the "token_usage" section.
        file_path (str, optional): The path to the file to be read or written.
        replacements (list[dict[str, str]], optional): The replacements to be made in
        the "edit" section.
        action (ActionType, optional): The action to be printed for the "header" section.
    """
    if verbosity_level < VerbosityLevel.VERBOSE:
        return

    if isinstance(section, str):
        try:
            section = ExecutionSection(section)
        except ValueError:
            raise ValueError("Unknown section type. Choose from: header, code, result, footer.")

    if section == ExecutionSection.HEADER:
        action_str = str(action).title() if action else ""
        if step is None:
            raise ValueError("Step must be provided for header section.")
        print(f"\n\033[1;36m╭─ Executing {action_str} (Step {step}) ──────────────────\033[0m")
    elif section == ExecutionSection.CODE:
        print("\n\033[1;36m│ Executing:\033[0m")
        print(content)
    elif section == ExecutionSection.WRITE:
        print(f"\n\033[1;36m│ Writing to file: {file_path}\033[0m")
        print(content)
    elif section == ExecutionSection.EDIT:
        print(f"\n\033[1;36m│ Editing file: {file_path}\033[0m")
        print("\n\033[1;36m│ Replacements:\033[0m")
        for replacement in replacements or []:
            print(f"\n\033[1;36m│ {replacement['find']} -> {replacement['replace']}\033[0m")
    elif section == ExecutionSection.READ:
        print(f"\n\033[1;36m│ Reading file: {file_path}\033[0m")
    elif section == ExecutionSection.RESULT:
        print("\n\033[1;36m│ Result:\033[0m " + content)
    elif section == ExecutionSection.TOKEN_USAGE:
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary.")

            prompt_tokens = data.get("prompt_tokens", 0)
            completion_tokens = data.get("completion_tokens", 0)
            total_cost = data.get("cost", 0.0)

            if prompt_tokens == 0 and completion_tokens == 0 and total_cost == 0.0:
                print(
                    "\n\033[1;36m│ Session Usage: \033[0m\033[1;33mToken usage data "
                    "unavailable.\033[0m"
                )
            else:
                cost_str = f"Cost: ${total_cost:.4f} USD 💰  " if total_cost > 0 else ""
                print(
                    "\n\033[1;36m│ Session Usage: \033[0m"
                    f"\033[1;33mPrompt: {prompt_tokens} ⬆️  Completion: {completion_tokens} ⬇️  "
                    f"{cost_str}\033[0m"
                )

        except Exception:
            # Don't display if there is no token usage data
            pass
    elif section == ExecutionSection.FOOTER:
        print("\033[1;36m╰══════════════════════════════════════════════════╯\033[0m")


def print_task_interrupted(verbosity_level: VerbosityLevel) -> None:
    """
    Print a section indicating that the task was interrupted.
    """
    if verbosity_level < VerbosityLevel.INFO:
        return

    print("\n\033[1;33m╭─ Task Interrupted ───────────────────────────────────\033[0m")
    print("\033[1;33m│ User requested to stop current task\033[0m")
    print("\033[1;33m╰══════════════════════════════════════════════════╯\033[0m\n")


def condense_logging(log_output: str, max_lines: int = 8000) -> str:
    """Condense the logging output to a more concise format.

    This function takes a string of logging output and condenses identical lines,
    replacing them with a single line indicating the number of repetitions.
    It also identifies and condenses multi-line patterns that repeat throughout the output.
    If the number of lines exceeds max_lines, it truncates the beginning of the output
    and adds a message indicating the number of removed lines.

    Args:
        log_output (str): The logging output to condense.
        max_lines (int, optional): The maximum number of lines to show in the condensed output.
            Defaults to 8000.

    Returns:
        str: The condensed logging output.
    """
    if not log_output:
        return log_output

    lines: List[str] = log_output.splitlines()

    # First pass: identify consecutive identical lines
    i: int = 0
    condensed_lines: List[str] = []
    while i < len(lines):
        line: str = lines[i]
        count: int = 1

        # Count consecutive identical lines
        while i + count < len(lines) and lines[i + count] == line:
            count += 1

        if count > 1 and line.strip() != "":
            condensed_lines.append(f"{line} ({count} identical lines)")
            i += count
        else:
            # Look for multi-line patterns
            pattern_found: bool = False

            # Try patterns of different lengths (2 to 10 lines)
            for pattern_length in range(2, min(11, len(lines) - i + 1)):
                pattern: List[str] = lines[i : i + pattern_length]

                # Check if this pattern repeats
                repeats: int = 0
                j: int = i
                while j <= len(lines) - pattern_length:
                    if lines[j : j + pattern_length] == pattern:
                        repeats += 1
                        j += pattern_length
                    else:
                        break

                if repeats > 1:
                    # Found a repeating multi-line pattern
                    for k in range(pattern_length - 1):
                        condensed_lines.append(pattern[k])

                    # Add the last line of the pattern with the count
                    condensed_lines.append(
                        f"{pattern[pattern_length - 1]} ({repeats} identical multi-line blocks)"
                    )

                    i += pattern_length * repeats
                    pattern_found = True
                    break

            if not pattern_found:
                condensed_lines.append(line)
                i += 1

    # Truncate if necessary
    num_condensed_lines: int = len(condensed_lines)
    if num_condensed_lines > max_lines:
        lines_removed: int = num_condensed_lines - max_lines
        condensed_lines = condensed_lines[-max_lines:]
        condensed_lines.insert(0, f"...({lines_removed} previous lines removed)")

    return "\n".join(condensed_lines)
