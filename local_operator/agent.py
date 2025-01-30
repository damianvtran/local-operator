import importlib.metadata
import io
import os
import platform
import readline
import signal
import sys
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Union

from langchain.schema import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tiktoken import encoding_for_model

from local_operator.credentials import CredentialManager
from local_operator.prompts import BaseSystemPrompt


class ProcessResponseStatus(Enum):
    """Status codes for process_response results."""

    SUCCESS = "success"
    CANCELLED = "cancelled"
    ERROR = "error"
    INTERRUPTED = "interrupted"


class ProcessResponseOutput:
    """Output structure for process_response results.

    Attributes:
        status (ProcessResponseStatus): Status of the response processing
        message (str): Descriptive message about the processing result
    """

    def __init__(self, status: ProcessResponseStatus, message: str):
        self.status = status
        self.message = message


class ConversationRole(Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class LocalCodeExecutor:
    context: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    model: ChatOpenAI | ChatOllama | ChatAnthropic
    step_counter: int
    max_conversation_history: int
    detail_conversation_length: int
    interrupted: bool

    """A class to handle local Python code execution with safety checks and context management.

    Attributes:
        context (dict): A dictionary to maintain execution context between code blocks
        conversation_history (list): A list of message dictionaries tracking the conversation
        model: The language model used for code analysis and safety checks
        step_counter (int): A counter to track the current step in sequential execution
        max_conversation_history (int): The maximum number of messages to keep in
            the conversation history.  This doesn't include the system prompt.
        detail_conversation_length (int): The number of messages to keep in full detail in the
            conversation history.  Every step before this except the system prompt will be
            summarized.
        interrupted (bool): Flag indicating if execution was interrupted
    """

    def __init__(
        self,
        model: ChatOpenAI | ChatOllama | ChatAnthropic,
        max_conversation_history: int = 100,
        detail_conversation_length: int = 10,
    ):
        """Initialize the LocalCodeExecutor with a language model.

        Args:
            model: The language model instance to use for code analysis
            max_conversation_history: The maximum number of messages to keep in
            the conversation history.  This doesn't include the system prompt.
            detail_conversation_length: The number of messages to keep in full detail in the
            conversation history.  Every step before this except the system prompt will be
            summarized.
        """
        self.context = {}
        self.conversation_history = []
        self.model = model
        self.max_conversation_history = max_conversation_history
        self.detail_conversation_length = detail_conversation_length
        self.reset_step_counter()
        self.interrupted = False

    def reset_step_counter(self):
        """Reset the step counter."""
        self.step_counter = 1

    def _append_to_history(
        self, role: ConversationRole, content: str, should_summarize: str = "True"
    ) -> None:
        """Append a message to conversation history and maintain length limit.

        Args:
            role (str): The role of the message sender (user/assistant/system)
            content (str): The message content
            should_summarize (str): Whether to summarize the message in the future.
            This can be set to False for messages that are already sufficiently
            summarized.
        """
        self.conversation_history.append(
            {
                "role": role.value,
                "content": content,
                "summarized": "False",
                "should_summarize": should_summarize,
            }
        )
        self._limit_conversation_history()

    async def _summarize_old_steps(self) -> None:
        """Summarize old conversation steps beyond the detail conversation length.
        Only summarizes steps that haven't been summarized yet."""
        if len(self.conversation_history) <= 1:  # Just system prompt or empty
            return

        # Calculate which messages need summarizing
        history_to_summarize = self.conversation_history[1 : -self.detail_conversation_length]

        for msg in history_to_summarize:
            # Skip messages that are already sufficiently concise/summarized
            if msg.get("should_summarize") == "False":
                continue

            if msg.get("summarized") == "True":
                continue

            # Leave the user prompts intact
            if msg.get("role") == ConversationRole.USER.value:
                continue

            summary = await self._summarize_conversation_step(msg)
            msg["content"] = summary
            msg["summarized"] = "True"

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from text using markdown-style syntax.
        Handles nested code blocks by matching outermost ```python enclosures.

        Args:
            text (str): The text containing potential code blocks

        Returns:
            list: A list of extracted code blocks as strings
        """
        blocks = []
        current_pos = 0

        while True:
            # Find start of next ```python block
            start = text.find("```python", current_pos)
            if start == -1:
                break

            # Find matching end block by counting nested blocks
            nested_count = 1
            pos = start + 9  # Length of ```python

            while nested_count > 0 and pos < len(text):
                if (
                    text[pos:].startswith("```")
                    and not text[pos:].startswith("```\n")
                    and not pos + 3 == len(text)
                ):
                    nested_count += 1
                    pos += 9
                elif text[pos:].startswith("```"):
                    nested_count -= 1
                    pos += 3
                else:
                    pos += 1

            if nested_count == 0:
                # Extract the block content between the outermost delimiters
                block = text[start + 9 : pos - 3].strip()

                # Validate block is not just comments/diffs
                is_comment = True
                for line in block.split("\n"):
                    trimmed_line = line.strip()
                    if not (
                        trimmed_line.startswith("//")
                        or trimmed_line.startswith("/*")
                        or trimmed_line.startswith("#")
                        or trimmed_line.startswith("+")
                        or trimmed_line.startswith("-")
                        or trimmed_line.startswith("<<<<<<<")
                        or trimmed_line.startswith(">>>>>>>")
                        or trimmed_line.startswith("=======")
                        or trimmed_line.startswith("```")
                    ):
                        is_comment = False
                        break

                if not is_comment:
                    blocks.append(block)

                current_pos = pos
            else:
                # No matching end found, move past this start marker
                current_pos = start + 9

        return blocks

    async def invoke_model(self, messages: List[Dict[str, str]]) -> BaseMessage:
        """Invoke the language model with a list of messages."""
        if isinstance(self.model, ChatAnthropic):
            # Anthropic models expect a single message, so combine the conversation history
            combined_message = ""
            for msg in messages:
                role_prefix = (
                    "Human: "
                    if msg["role"] == ConversationRole.USER.value
                    else (
                        "Assistant: "
                        if msg["role"] == ConversationRole.ASSISTANT.value
                        else "System: "
                    )
                )
                combined_message += f"{role_prefix}{msg['content']}\n\n"

            # Remove trailing newlines and invoke the model
            combined_message = combined_message.strip()
            return await self.model.ainvoke(combined_message)
        else:
            return await self.model.ainvoke(messages)

    async def check_code_safety(self, code: str) -> bool:
        """Analyze code for potentially dangerous operations using the language model.

        Args:
            code (str): The Python code to analyze

        Returns:
            bool: True if dangerous operations are detected, False otherwise
        """
        safety_check_prompt = """
        You are a code safety and security checker.

        You will be given a code snippet and asked to check if it contains any dangerous operations.

        🚫 Respond "yes" if contains:
        - High risk file deletion
        - Suspicious package installs
        - High risk system commands execution
        - Sensitive system access
        - Risky network operations

        ✅ Respond "no" if none detected

        Format: "yes" or "no"
        """

        safety_history = [
            {"role": ConversationRole.SYSTEM.value, "content": safety_check_prompt},
            {
                "role": ConversationRole.USER.value,
                "content": f"Determine if the following code is safe: {code}",
            },
        ]

        response = await self.invoke_model(safety_history)

        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        return "yes" in response_content.strip().lower()

    async def execute_code(self, code: str, max_retries: int = 2) -> str:
        """Execute Python code with safety checks and context management.

        Args:
            code (str): The Python code to execute
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Execution result message or error message
        """
        # First check code safety
        if await self._check_and_confirm_safety(code):
            return "Code execution canceled by user"

        # Try initial execution
        try:
            return await self._execute_with_output(code)
        except Exception as initial_error:
            return await self._handle_execution_error(initial_error, max_retries)

    async def _check_and_confirm_safety(self, code: str) -> bool:
        """Check code safety and get user confirmation if needed.

        Returns True if execution should be cancelled."""
        if await self.check_code_safety(code):
            confirm = input("Warning: Potentially dangerous operation detected. Proceed? (y/n): ")
            if confirm.lower() != "y":
                msg = (
                    "I've identified that this is a dangerous operation. "
                    "Let's stop this task for now, I will provide further instructions shortly."
                )
                self._append_to_history(ConversationRole.USER, msg)
                return True
        return False

    async def _execute_with_output(self, code: str) -> str:
        """Execute code and capture stdout/stderr output.

        Args:
            code (str): The Python code to execute
            timeout (int, optional): Maximum execution time in seconds. Defaults to 30.

        Returns:
            str: Formatted string containing execution output and any error messages

        Raises:
            Exception: Re-raises any exceptions that occur during code execution
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        new_stdout, new_stderr = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = new_stdout, new_stderr

        try:
            await self._run_code(code)
            output, error_output = self._capture_and_record_output(new_stdout, new_stderr)
            return self._format_success_output((output, error_output))
        except Exception as e:
            output, error_output = self._capture_and_record_output(new_stdout, new_stderr)
            error_msg = (
                f"Code execution error:\n{str(e)}\n"
                f"Output:\n{output}\n"
                f"Error output:\n{error_output}"
            )
            self._append_to_history(ConversationRole.SYSTEM, error_msg)
            raise e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            new_stdout.close()
            new_stderr.close()

    async def _run_code(self, code: str) -> None:
        """Run code in the main thread.

        Args:
            code (str): The Python code to execute
            timeout (int): Unused parameter kept for compatibility

        Raises:
            Exception: Any exceptions raised during code execution
        """
        old_stdin = sys.stdin

        try:
            # Redirect stdin to /dev/null to ignore input requests
            with open(os.devnull) as devnull:
                sys.stdin = devnull
                exec(code, self.context)
        except Exception as e:
            raise e
        finally:
            sys.stdin = old_stdin

    def _capture_and_record_output(
        self, stdout: io.StringIO, stderr: io.StringIO
    ) -> tuple[str, str]:
        """Capture stdout/stderr output and record it in conversation history.

        Args:
            stdout (io.StringIO): Buffer containing standard output
            stderr (io.StringIO): Buffer containing error output

        Returns:
            tuple[str, str]: Tuple containing (stdout output, stderr output)
        """
        stdout.flush()
        stderr.flush()
        output = stdout.getvalue() or "[No output]"
        error_output = stderr.getvalue() or "[No error output]"

        self.context["last_code_output"] = output
        self.context["last_code_error"] = error_output
        self._append_to_history(
            ConversationRole.SYSTEM,
            f"Code execution output:\n{output}\nError output:\n{error_output}",
        )

        return output, error_output

    def _format_success_output(self, output: tuple[str, str]) -> str:
        """Format successful execution output with ANSI color codes.

        Args:
            output (tuple[str, str]): Tuple containing (stdout output, stderr output)

        Returns:
            str: Formatted string with colored success message and execution output
        """
        stdout, stderr = output
        return (
            "\n\033[1;32m✓ Code Execution Complete\033[0m\n"
            "\033[1;34m╞══════════════════════════════════════════╡\n"
            f"\033[1;36m│ Output:\033[0m\n{stdout}\n"
            f"\033[1;36m│ Error Output:\033[0m\n{stderr}"
        )

    async def _handle_execution_error(self, initial_error: Exception, max_retries: int) -> str:
        """Handle code execution errors with retry logic.

        Args:
            initial_error (Exception): The original error that occurred
            code (str): The Python code that failed
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Final execution output or formatted error message
        """
        self._record_initial_error(initial_error)
        self._log_error_and_retry_message(initial_error)

        for attempt in range(max_retries):
            try:
                new_code = await self._get_corrected_code()
                if new_code:
                    return await self._execute_with_output(new_code[0])
            except Exception as retry_error:
                self._record_retry_error(retry_error, attempt)
                self._log_retry_error(retry_error, attempt, max_retries)

        return self._format_error_output(initial_error, max_retries)

    def _record_initial_error(self, error: Exception) -> None:
        """Record the initial execution error in conversation history.

        Args:
            error (Exception): The error that occurred during initial execution
        """
        msg = (
            f"The initial execution failed with error: {str(error)}. "
            "Review the code and make corrections to run successfully."
        )
        self._append_to_history(ConversationRole.USER, msg)

    def _record_retry_error(self, error: Exception, attempt: int) -> None:
        """Record retry attempt errors in conversation history.

        Args:
            error (Exception): The error that occurred during retry
            attempt (int): The current retry attempt number
        """
        msg = (
            f"The code execution failed with error: {str(error)}. "
            "Please review and make corrections to the code to fix this error and try again."
        )
        self._append_to_history(ConversationRole.USER, msg)

    def _log_error_and_retry_message(self, error: Exception) -> None:
        """Print formatted error message and retry notification.

        Args:
            error (Exception): The error to display
        """
        print("\n\033[1;31m✗ Error during execution:\033[0m")
        print("\033[1;34m╞══════════════════════════════════════════╡")
        print(f"\033[1;36m│ Error:\033[0m\n{str(error)}")
        print("\033[1;34m╞══════════════════════════════════════════╡")
        print("\033[1;36m│ Attempting to fix the error...\033[0m")
        print("\033[1;34m╰══════════════════════════════════════════╯\033[0m")

    def _log_retry_error(self, error: Exception, attempt: int, max_retries: int) -> None:
        """Print formatted retry error message.

        Args:
            error (Exception): The error that occurred during retry
            attempt (int): Current retry attempt number
            max_retries (int): Maximum number of retry attempts
        """
        print(f"\n\033[1;31m✗ Error during execution (attempt {attempt + 1}):\033[0m")
        print("\033[1;34m╞══════════════════════════════════════════╡")
        print(f"\033[1;36m│ Error:\033[0m\n{str(error)}")
        if attempt < max_retries - 1:
            print("\033[1;36m│\033[0m \033[1;33mAnother attempt will be made...\033[0m")

    async def _get_corrected_code(self) -> List[str]:
        """Get corrected code from the language model.

        Returns:
            List[str]: List of extracted code blocks from model response
        """
        response = await self.invoke_model(self.conversation_history)
        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        return self.extract_code_blocks(response_content)

    def _format_error_output(self, error: Exception, max_retries: int) -> str:
        """Format error output message with ANSI color codes.

        Args:
            error (Exception): The error to format
            max_retries (int): Number of retry attempts made

        Returns:
            str: Formatted error message string
        """
        return (
            f"\n\033[1;31m✗ Code Execution Failed after {max_retries} attempts\033[0m\n"
            f"\033[1;34m╞══════════════════════════════════════════╡\n"
            f"\033[1;36m│ Error:\033[0m\n{str(error)}"
        )

    def _format_agent_output(self, text: str) -> str:
        """Format agent output with colored sidebar and indentation.

        Args:
            text (str): Raw agent output text

        Returns:
            str: Formatted text with colored sidebar and control tags removed
        """
        # Add colored sidebar to each line and remove control tags
        lines = [f"\033[1;36m│\033[0m {line}" for line in text.split("\n")]
        output = "\n".join(lines)
        output = output.replace("[ASK]", "").replace("[DONE]", "").replace("[BYE]", "").strip()

        # Remove trailing empty lines
        lines = [line for line in output.split("\n") if line.strip()]

        return "\n".join(lines)

    async def process_response(self, response: str) -> ProcessResponseOutput:
        """Process model response, extracting and executing any code blocks.

        Args:
            response (str): The model's response containing potential code blocks
        """
        if self.interrupted:
            print("\n\033[1;33m╭─ Task Interrupted ────────────────────────────\033[0m")
            print("\033[1;33m│ User requested to stop current task\033[0m")
            print("\033[1;33m╰──────────────────────────────────────────────────\033[0m\n")
            self.interrupted = False
            return ProcessResponseOutput(
                status=ProcessResponseStatus.INTERRUPTED,
                message="Task interrupted by user",
            )

        formatted_response = self._format_agent_output(response)
        print(
            f"\n\033[1;36m╭─ Agent Response (Step {self.step_counter}) "
            f"───────────────────────\033[0m"
        )
        print(formatted_response)
        print("\033[1;36m╰──────────────────────────────────────────────────\033[0m")

        self._append_to_history(ConversationRole.ASSISTANT, response)

        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            print(
                f"\n\033[1;36m╭─ Executing Code Blocks (Step {self.step_counter}) "
                f"───────────────\033[0m"
            )
            for code in code_blocks:
                print("\n\033[1;36m│ Executing:\033[0m\n{}".format(code))
                result = await self.execute_code(code)

                if "code execution cancelled by user" in result:
                    return ProcessResponseOutput(
                        status=ProcessResponseStatus.CANCELLED,
                        message="Code execution cancelled by user",
                    )

                print("\033[1;36m│ Result:\033[0m {}".format(result))

                self.context["last_code_result"] = result
            print("\033[1;36m╰──────────────────────────────────────────────────\033[0m")

            self._append_to_history(
                ConversationRole.SYSTEM,
                f"Current working directory: {os.getcwd()}",
                should_summarize="False",
            )

            self.step_counter += 1

        # Summarize old steps to reduce token usage
        await self._summarize_old_steps()

        return ProcessResponseOutput(
            status=ProcessResponseStatus.SUCCESS,
            message="Code execution complete",
        )

    def _limit_conversation_history(self) -> None:
        """Limit the conversation history to the maximum number of messages."""
        if len(self.conversation_history) > self.max_conversation_history:
            # Keep the first message (system prompt) and the most recent messages
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[
                -self.max_conversation_history + 1 :
            ]

    async def _summarize_conversation_step(self, msg: dict[str, str]) -> str:
        """Summarize the conversation step by invoking the model to generate a concise summary.

        Args:
            step_number (int): The step number to summarize

        Returns:
            str: A concise summary of the critical information from this step
        """
        summary_prompt = """
        You are a conversation summarizer. Your task is to summarize what happened in the given
        conversation step in a single concise sentence. Focus only on capturing critical details
        that may be relevant for future reference, such as:
        - Key actions taken
        - Important changes made
        - Significant results or outcomes
        - Any errors or issues encountered

        Format your response as a single sentence with the format:
        "[SUMMARY] {summary}"
        """

        step_info = "Please summarize the following conversation step:\n" + "\n".join(
            f"{msg['role']}: {msg['content']}"
        )

        summary_history = [
            {"role": ConversationRole.SYSTEM.value, "content": summary_prompt},
            {"role": ConversationRole.USER.value, "content": step_info},
        ]

        response = await self.invoke_model(summary_history)
        return response.content if isinstance(response.content, str) else str(response.content)


class CliOperator:
    """A command-line interface for interacting with language models.

    Attributes:
        model: The configured ChatOpenAI or ChatOllama instance
        executor: LocalCodeExecutor instance for handling code execution
    """

    def __init__(
        self,
        credential_manager: CredentialManager,
        model_instance: Union[ChatOpenAI, ChatOllama, ChatAnthropic],
    ):
        """Initialize the CLI by loading credentials or prompting for them.

        Args:
            hosting (str): Hosting platform (deepseek, openai, or ollama)
            model (str): Model name to use
        """
        self.credential_manager = credential_manager
        self.model = model_instance
        self.executor = LocalCodeExecutor(self.model)

        self._load_input_history()
        self._setup_interrupt_handler()

    def _setup_interrupt_handler(self) -> None:
        """Set up the interrupt handler for Ctrl+C."""

        def handle_interrupt(signum, frame):
            if self.executor.interrupted:
                # Pass through SIGINT if already interrupted
                signal.default_int_handler(signum, frame)
            self.executor.interrupted = True
            print(
                "\033[33m⚠️  Received interrupt signal, execution will"
                " stop after current step\033[0m"
            )

        signal.signal(signal.SIGINT, handle_interrupt)

    def _save_input_history(self) -> None:
        """Save input history to file."""
        history_file = Path.home() / ".local-operator" / "input_history.txt"
        history_file.parent.mkdir(parents=True, exist_ok=True)
        readline.write_history_file(str(history_file))

    def _load_input_history(self) -> None:
        """Load input history from file."""
        history_file = Path.home() / ".local-operator" / "input_history.txt"

        if history_file.exists():
            readline.read_history_file(str(history_file))

    def _get_input_with_history(self, prompt: str) -> str:
        """Get user input with history navigation using up/down arrows."""
        try:
            # Get user input with history navigation
            user_input = input(prompt)

            if user_input == "exit" or user_input == "quit":
                return user_input

            self._save_input_history()

            return user_input
        except KeyboardInterrupt:
            return "exit"

    def _agent_is_done(self, response) -> bool:
        """Check if the agent has completed its task."""
        if response is None:
            return False

        return "[DONE]" in response.content.strip().splitlines()[
            -1
        ].strip() or self._agent_should_exit(response)

    def _agent_requires_user_input(self, response) -> bool:
        """Check if the agent requires user input."""
        if response is None:
            return False

        return "[ASK]" in response.content.strip().splitlines()[-1].strip()

    def _agent_should_exit(self, response) -> bool:
        """Check if the agent should exit."""
        if response is None:
            return False

        return "[BYE]" in response.content.strip().splitlines()[-1].strip()

    def _get_installed_packages(self) -> str:
        """Get installed packages for the system prompt context."""

        # Filter to show only commonly used packages and require that the model
        # check for any other packages as needed.
        key_packages = {
            "numpy",
            "pandas",
            "torch",
            "tensorflow",
            "scikit-learn",
            "matplotlib",
            "seaborn",
            "requests",
            "pillow",
            "pip",
            "setuptools",
            "wheel",
            "langchain",
            "plotly",
            "scipy",
            "statsmodels",
            "tqdm",
        }

        installed_packages = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]

        # Filter and sort with priority for key packages
        filtered_packages = sorted(
            (pkg for pkg in installed_packages if pkg.lower() in key_packages),
            key=lambda x: (x.lower() not in key_packages, x.lower()),
        )

        # Add count of non-critical packages
        other_count = len(installed_packages) - len(filtered_packages)
        package_str = ", ".join(filtered_packages[:15])  # Show first 15 matches
        if other_count > 0:
            package_str += f" + {other_count} others"

        return package_str

    def _setup_prompt(self) -> None:
        """Setup the prompt for the agent."""

        base_system_prompt = BaseSystemPrompt
        user_system_prompt = Path.home() / ".local-operator" / "system_prompt.md"
        if user_system_prompt.exists():
            user_system_prompt = user_system_prompt.read_text()
        else:
            user_system_prompt = ""

        system_details = {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "home_directory": os.path.expanduser("~"),
        }
        system_details_str = "\n".join(f"{key}: {value}" for key, value in system_details.items())

        installed_packages_str = self._get_installed_packages()

        base_system_prompt = (
            base_system_prompt.replace("{{system_details_str}}", system_details_str)
            .replace("{{installed_packages_str}}", installed_packages_str)
            .replace("{{user_system_prompt}}", user_system_prompt)
        )

        self.executor.conversation_history = [
            {
                "role": ConversationRole.SYSTEM.value,
                "content": base_system_prompt,
            }
        ]

    async def chat(self) -> None:
        """Run the interactive chat interface with code execution capabilities."""
        debug_indicator = (
            " [DEBUG MODE]" if os.getenv("LOCAL_OPERATOR_DEBUG", "false").lower() == "true" else ""
        )
        print("\033[1;36m╭──────────────────────────────────────────────────╮\033[0m")
        print(f"\033[1;36m│ Local Executor Agent CLI{debug_indicator:<25}│\033[0m")
        print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
        print("\033[1;36m│ You are interacting with a helpful CLI agent     │\033[0m")
        print("\033[1;36m│ that can execute tasks locally on your device    │\033[0m")
        print("\033[1;36m│ by running Python code.                          │\033[0m")
        print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
        print("\033[1;36m│ Type 'exit' or 'quit' to quit                    │\033[0m")
        print("\033[1;36m│ Press Ctrl+C to interrupt current task           │\033[0m")
        print("\033[1;36m╰──────────────────────────────────────────────────╯\033[0m\n")

        self._setup_prompt()

        while True:

            prompt = f"You ({os.getcwd()}): > "
            user_input = self._get_input_with_history(prompt)

            if not user_input.strip():
                continue

            if user_input.lower() == "exit" or user_input.lower() == "quit":
                break

            self.executor.conversation_history.append(
                {"role": ConversationRole.USER.value, "content": user_input}
            )

            response = None
            self.executor.reset_step_counter()

            while (
                not self._agent_is_done(response)
                and not self._agent_requires_user_input(response)
                and not self.executor.interrupted
            ):
                if self.model is None:
                    raise ValueError("Model is not initialized")

                response = await self.executor.invoke_model(self.executor.conversation_history)

                response_content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                result = await self.executor.process_response(response_content)

                # Break out of the agent flow if the user cancels the code execution
                if (
                    result.status == ProcessResponseStatus.CANCELLED
                    or result.status == ProcessResponseStatus.INTERRUPTED
                ):
                    break

            if os.environ.get("LOCAL_OPERATOR_DEBUG") == "true":
                tokenizer = encoding_for_model("gpt-4o")
                total_tokens = sum(
                    len(tokenizer.encode(entry["content"]))
                    for entry in self.executor.conversation_history
                )

                print("\n\033[1;35m╭─ Debug: Conversation History ───────────────────────\033[0m")
                print(f"\033[1;35m│ Total tokens: {total_tokens}\033[0m")
                for i, entry in enumerate(self.executor.conversation_history, 1):
                    role = entry["role"]
                    content = entry["content"]
                    print(f"\033[1;35m│ {i}. {role.capitalize()}:\033[0m")
                    for line in content.split("\n"):
                        print(f"\033[1;35m│   {line}\033[0m")
                print("\033[1;35m╰──────────────────────────────────────────────────\033[0m\n")

            # Check if the last line of the response contains "[BYE]" to exit
            if self._agent_should_exit(response):
                break

            # Print the last assistant message if the agent is asking for user input
            if response and self._agent_requires_user_input(response):
                response_content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                ).replace("ASK", "")
                print("\n\033[1;36m╭─ Agent Question Requires Input ────────────────\033[0m")
                print(f"\033[1;36m│\033[0m {response_content}")
                print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")
