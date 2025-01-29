import importlib.metadata
import io
import os
import platform
import readline
import sys
import threading
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


class ProcessResponseOutput:
    """Output structure for process_response results.

    Attributes:
        status (ProcessResponseStatus): Status of the response processing
        message (str): Descriptive message about the processing result
    """

    def __init__(self, status: ProcessResponseStatus, message: str):
        self.status = status
        self.message = message


class LocalCodeExecutor:
    context: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    model: ChatOpenAI | ChatOllama | ChatAnthropic
    step_counter: int

    """A class to handle local Python code execution with safety checks and context management.

    Attributes:
        context (dict): A dictionary to maintain execution context between code blocks
        conversation_history (list): A list of message dictionaries tracking the conversation
        model: The language model used for code analysis and safety checks
        step_counter (int): A counter to track the current step in sequential execution
    """

    def __init__(self, model: ChatOpenAI | ChatOllama | ChatAnthropic):
        """Initialize the LocalCodeExecutor with a language model.

        Args:
            model: The language model instance to use for code analysis
        """
        self.context = {}
        self.conversation_history = []
        self.model = model
        self.reset_step_counter()

    def reset_step_counter(self):
        """Reset the step counter."""
        self.step_counter = 1

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
                if text[pos:].startswith("```") and not text[pos:].startswith("```\n"):
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
                    if msg["role"] == "user"
                    else "Assistant: " if msg["role"] == "assistant" else "System: "
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
        safety_check_prompt = f"""
        Analyze the following Python code for potentially dangerous operations:
        {code}

        Respond with only "yes" if the code contains dangerous operations that could:
        - Delete or modify files
        - Install or update packages that might be harmful or unsafe
        - Execute unsafe system commands
        - Access sensitive system resources
        - Perform network operations that expose the system or user data to the internet
        - Otherwise compromise system security

        Respond with only "no" if the code appears safe to execute.
        """

        self.conversation_history.append({"role": "user", "content": safety_check_prompt})
        response = await self.invoke_model(self.conversation_history)
        self.conversation_history.pop()

        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        return "yes" in response_content.strip().lower()

    async def execute_code(self, code: str, max_retries: int = 2, timeout: int = 30) -> str:
        """Execute Python code with safety checks and context management.

        Args:
            code (str): The Python code to execute
            max_retries (int): Maximum number of retry attempts
            timeout (int): Maximum execution time in seconds

        Returns:
            str: Execution result message or error message
        """

        async def _execute_with_timeout(code_to_execute: str) -> None:
            """Helper function to execute code with timeout."""
            result = {"success": False, "error": None}
            event = threading.Event()

            def run_code():
                try:
                    exec(code_to_execute, self.context)
                    result["success"] = True
                except Exception as e:
                    result["error"] = e
                finally:
                    event.set()

            exec_thread = threading.Thread(target=run_code, daemon=True)
            exec_thread.start()

            # Wait for either the event or timeout
            event_occurred = event.wait(timeout)

            if not event_occurred:
                # If timeout occurred, check if thread completed anyway
                if not result["success"]:
                    # Clean up the thread before raising timeout
                    raise TimeoutError(f"Code execution timed out after {timeout} seconds")

            if not result["success"] and result["error"]:
                raise result["error"]
            elif not result["success"]:
                raise TimeoutError("Code execution failed")

        async def _capture_output(code_to_execute: str) -> str:
            """Helper function to capture and return execution output."""
            old_stdout, old_stderr = sys.stdout, sys.stderr
            new_stdout, new_stderr = io.StringIO(), io.StringIO()
            sys.stdout, sys.stderr = new_stdout, new_stderr

            def _get_outputs() -> tuple[str, str]:
                """Helper to get and format stdout/stderr outputs."""
                new_stdout.flush()
                new_stderr.flush()
                output = new_stdout.getvalue() or "[No output]"
                error_output = new_stderr.getvalue() or "[No error output]"
                return output, error_output

            def _log_outputs(output: str, error_output: str) -> None:
                """Helper to log outputs to context and history."""
                self.context["last_code_output"] = output
                self.conversation_history.append(
                    {
                        "role": "system",
                        "content": (
                            "Code execution output:\n"
                            f"{output}\n"
                            "Error output:\n"
                            f"{error_output}"
                        ),
                    }
                )

            try:
                await _execute_with_timeout(code_to_execute)
                output, error_output = _get_outputs()
                _log_outputs(output, error_output)

                return (
                    "\n\033[1;32m✓ Code Execution Successful\033[0m\n"
                    "\033[1;34m╞══════════════════════════════════════════╡\n"
                    f"\033[1;36m│ Output:\033[0m\n{output}\n"
                    f"\033[1;36m│ Error Output:\033[0m\n{error_output}"
                )
            except Exception as e:
                output, error_output = _get_outputs()
                _log_outputs(output, error_output)
                raise e
            finally:
                sys.stdout, sys.stderr = old_stdout, old_stderr
                new_stdout.close()
                new_stderr.close()

        async def _handle_error(error: Exception, attempt: int | None = None) -> str:
            """Helper function to handle execution errors."""
            error_message = str(error)
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": (
                        f"The code execution failed with error: {error_message}. "
                        "Please review and make corrections to the code to fix this error."
                    ),
                }
            )

            if attempt is not None:
                return (
                    f"\n\033[1;31m✗ Code Execution Failed after {attempt + 1} attempts\033[0m\n"
                    f"\033[1;34m╞══════════════════════════════════════════╡\n"
                    f"\033[1;36m│ Error:\033[0m\n{error_message}"
                )

            return (
                "\n\033[1;31m✗ Code Execution Failed\033[0m\n"
                f"\033[1;34m╞══════════════════════════════════════════╡\n"
                f"\033[1;36m│ Error:\033[0m\n{error_message}"
            )

        # Main execution flow
        try:
            if await self.check_code_safety(code):
                confirm = input(
                    "Warning: Potentially dangerous operation detected. Proceed? (y/n): "
                )
                if confirm.lower() != "y":
                    self.conversation_history.append(
                        {
                            "role": "user",
                            "content": "I've identified that this is a dangerous"
                            " operation.  Let's stop this task for now, I will"
                            " provide further instructions shortly.",
                        }
                    )
                    return "Code execution canceled by user"

            return await _capture_output(code)

        except Exception as initial_error:
            error_message = str(initial_error)
            self.conversation_history.append(
                {
                    "role": "user",
                    "content": (
                        f"The initial execution failed with error: {error_message}. "
                        "Review the code and make corrections to run successfully."
                    ),
                }
            )

            print("\n\033[1;31m✗ Error during execution:\033[0m")
            print("\033[1;34m╞══════════════════════════════════════════╡")
            print(f"\033[1;36m│ Error:\033[0m\n{error_message}")
            print("\033[1;34m╞══════════════════════════════════════════╡")
            print("\033[1;36m│ Attempting to fix the error...\033[0m")
            print("\033[1;34m╰══════════════════════════════════════════╯\033[0m")

            for attempt in range(max_retries):
                try:
                    response = await self.invoke_model(self.conversation_history)
                    response_content = (
                        response.content
                        if isinstance(response.content, str)
                        else str(response.content)
                    )
                    new_code = self.extract_code_blocks(response_content)
                    if new_code:
                        return await _capture_output(new_code[0])
                except Exception as retry_error:
                    print(f"\n\033[1;31m✗ Error during execution (attempt {attempt + 1}):\033[0m")
                    print("\033[1;34m╞══════════════════════════════════════════╡")
                    print(f"\033[1;36m│ Error:\033[0m\n{str(retry_error)}")
                    if attempt < max_retries - 1:
                        print("\033[1;36m│\033[0m \033[1;33mAnother attempt will be made...\033[0m")

            return await _handle_error(initial_error, attempt=max_retries)

    def _format_agent_output(self, text: str) -> str:
        """Format agent output with colored sidebar and indentation."""

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
        formatted_response = self._format_agent_output(response)
        print(
            f"\n\033[1;36m╭─ Agent Response (Step {self.step_counter}) "
            f"───────────────────────\033[0m"
        )
        print(formatted_response)
        print("\033[1;36m╰──────────────────────────────────────────────────\033[0m")

        self.conversation_history.append({"role": "assistant", "content": response})

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

            self.conversation_history.append(
                {"role": "system", "content": f"Current working directory: {os.getcwd()}"}
            )

            self.step_counter += 1

        return ProcessResponseOutput(
            status=ProcessResponseStatus.SUCCESS,
            message="Code execution successful",
        )


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

        installed_packages = importlib.metadata.distributions()
        installed_packages_str = ", ".join(
            package.metadata["Name"] for package in installed_packages
        )

        base_system_prompt = (
            base_system_prompt.replace("{{system_details_str}}", system_details_str)
            .replace("{{installed_packages_str}}", installed_packages_str)
            .replace("{{user_system_prompt}}", user_system_prompt)
        )

        self.executor.conversation_history = [
            {
                "role": "system",
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
        print("\033[1;36m╰──────────────────────────────────────────────────╯\033[0m\n")

        self._setup_prompt()

        while True:

            prompt = f"You ({os.getcwd()}): > "
            user_input = self._get_input_with_history(prompt)

            if not user_input.strip():
                continue

            if user_input.lower() == "exit" or user_input.lower() == "quit":
                break

            self.executor.conversation_history.append({"role": "user", "content": user_input})

            response = None
            self.executor.reset_step_counter()

            while not self._agent_is_done(response) and not self._agent_requires_user_input(
                response
            ):
                if self.model is None:
                    raise ValueError("Model is not initialized")

                response = await self.executor.invoke_model(self.executor.conversation_history)

                response_content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                result = await self.executor.process_response(response_content)

                # Break out of the agent flow if the user cancels the code execution
                if result.status == ProcessResponseStatus.CANCELLED:
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
