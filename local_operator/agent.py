import importlib.metadata
import io
import os
import platform
import re
import readline
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from local_operator.credentials import CredentialManager


class LocalCodeExecutor:
    context: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    model: ChatOpenAI | ChatOllama
    step_counter: int

    """A class to handle local Python code execution with safety checks and context management.

    Attributes:
        context (dict): A dictionary to maintain execution context between code blocks
        conversation_history (list): A list of message dictionaries tracking the conversation
        model: The language model used for code analysis and safety checks
        step_counter (int): A counter to track the current step in sequential execution
    """

    def __init__(self, model):
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

        Args:
            text (str): The text containing potential code blocks

        Returns:
            list: A list of extracted code blocks as strings
        """
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        return pattern.findall(text)

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
        response = await self.model.ainvoke(self.conversation_history)
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
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout

            try:
                await _execute_with_timeout(code_to_execute)
                # Flush the output buffer to ensure all content is captured
                new_stdout.flush()
                output = new_stdout.getvalue()

                # Ensure the output is captured even if empty
                if not output:
                    output = "[No output]"

                self.context["last_code_output"] = output
                self.conversation_history.append(
                    {"role": "system", "content": f"Code execution output:\n{output}"}
                )
                return (
                    "\n\033[1;32m✓ Code Execution Successful\033[0m\n"
                    "\033[1;34m╞══════════════════════════════════════════╡\n"
                    f"\033[1;36m│ Output:\033[0m\n{output}"
                )
            finally:
                # Restore stdout and ensure the buffer is flushed
                sys.stdout = old_stdout
                new_stdout.close()

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

            for attempt in range(max_retries):
                try:
                    response = await self.model.ainvoke(self.conversation_history)
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
        return "\n".join(f"\033[1;36m│\033[0m {line}" for line in text.split("\n"))

    async def process_response(self, response: str) -> None:
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
                print("\033[1;36m│ Result:\033[0m {}".format(result))

                self.context["last_code_result"] = result
            print("\033[1;36m╰──────────────────────────────────────────────────\033[0m")

            self.conversation_history.append(
                {"role": "system", "content": f"Current working directory: {os.getcwd()}"}
            )

            self.step_counter += 1


class CliOperator:
    """A command-line interface for interacting with language models.

    Attributes:
        model: The configured ChatOpenAI or ChatOllama instance
        executor: LocalCodeExecutor instance for handling code execution
    """

    def __init__(self, hosting: str, model: str, credential_manager: CredentialManager):
        """Initialize the CLI by loading credentials or prompting for them.

        Args:
            hosting (str): Hosting platform (deepseek, openai, or ollama)
            model (str): Model name to use
        """
        if not hosting:
            raise ValueError("Hosting is required")
        if not model:
            raise ValueError("Model is required")

        self.credential_manager = credential_manager

        # Configure model based on hosting
        if hosting == "deepseek":
            base_url = "https://api.deepseek.com/v1"
            api_key = self.credential_manager.get_api_key("DEEPSEEK_API_KEY")
            if not api_key:
                api_key = self.credential_manager.prompt_for_api_key("DEEPSEEK_API_KEY")
            self.model = ChatOpenAI(
                api_key=SecretStr(api_key),
                temperature=0.5,
                base_url=base_url,
                model=model,
            )
        elif hosting == "openai":
            base_url = "https://api.openai.com"
            api_key = self.credential_manager.get_api_key("OPENAI_API_KEY")
            if not api_key:
                api_key = self.credential_manager.prompt_for_api_key("OPENAI_API_KEY")
            self.model = ChatOpenAI(
                api_key=SecretStr(api_key),
                temperature=0.5,
                base_url=base_url,
                model=model,
            )
        elif hosting == "ollama":
            self.model = ChatOllama(
                model=model,
                temperature=0.5,
            )
        elif hosting == "noop":
            # Useful for testing, will create a dummy operator
            self.model = None

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

        return "DONE" in response.content.strip().splitlines()[
            -1
        ].strip() or self._agent_should_exit(response)

    def _agent_should_exit(self, response) -> bool:
        """Check if the agent should exit."""
        if response is None:
            return False

        return "Bye!" in response.content.strip().splitlines()[-1].strip()

    def _setup_prompt(self) -> None:
        """Setup the prompt for the agent."""

        system_details = {
            "os": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.machine(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "home_directory": os.path.expanduser("~"),
        }

        installed_packages = importlib.metadata.distributions()
        installed_packages_str = ", ".join(
            package.metadata["Name"] for package in installed_packages
        )

        self.executor.conversation_history = [
            {
                "role": "system",
                "content": f"""
                You are Local Operator - a Python code execution agent that
                runs securely on the user's local machine. Your primary function
                is to execute Python code safely and efficiently to help users
                accomplish their tasks.

                Core Principles:
                1. Safety First: Never execute harmful or destructive code.
                   Always validate code safety before execution.
                2. Step-by-Step Execution: Break tasks into logical steps,
                   executing one step at a time.
                3. Context Awareness: Maintain context between steps and across
                   sessions.
                4. Minimal Output: Keep responses concise and focused on
                   executable code.
                5. Data Verification: When uncertain about information, write code
                   to fetch data rather than making assumptions.
                6. Research: Write code to fetch data from the internet in preliminary
                   steps before proceeding to more complex tasks.

                Execution Rules:
                - Always output Python code in ```python``` blocks
                - Include package installation commands when needed
                - Validate code safety before execution
                - Print results in human-readable format
                - Handle one step per response
                - Mark final step with "DONE" on a new line after code block
                - Maintain secure execution environment
                - Exit with "Bye!" when user requests to quit
                - When uncertain about system state or data, write code to:
                  * Verify file existence
                  * Check directory contents
                  * Validate system information
                  * Confirm package versions
                  * Fetch data from the internet
                - Only ask for clarification as a last resort when code cannot
                  retrieve the required information

                Task Handling Guidelines:
                1. Analyze user request and break into logical steps
                2. Each step is separate from the others, so the execution of one
                   step can be put into the context of the next step.
                3. For each step:
                   - Generate minimal required code
                   - Include necessary package installations
                   - Add clear output formatting
                   - Validate code safety
                   - When uncertain, write verification code before proceeding
                   - Use research steps as necessary to find information needed to
                     proceed to the next step.
                4. After execution:
                   - Analyze results
                   - Determine next steps
                   - Continue until task completion
                5. Mark final step with "DONE" on the last line of the response, only if
                   there are no other steps that should be executed to better complete
                   the task. Ensure that "DONE" is the last word that is generated after
                   all other content.

                System Context:
                - OS: {system_details['os']} {system_details['release']}
                - Architecture: {system_details['architecture']}
                - Processor: {system_details['processor']}
                - Python Environment: {os.getcwd()}
                - Home Directory: {system_details['home_directory']}
                - Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

                Installed Packages:
                {installed_packages_str}

                Remember:
                - Always prioritize safety and security
                - Maintain context between steps
                - Keep responses minimal and focused
                - Handle one step at a time
                - Mark completion with "DONE"
                - Exit with "Bye!" when requested
                - When uncertain, write code to verify information
                - Only ask for clarification when code cannot retrieve needed data
                """,
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

            while not self._agent_is_done(response):
                if self.model is None:
                    raise ValueError("Model is not initialized")

                response = self.model.invoke(self.executor.conversation_history)
                response_content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )
                await self.executor.process_response(response_content)

            if os.environ.get("LOCAL_OPERATOR_DEBUG") == "true":
                print("\n\033[1;35m╭─ Debug: Conversation History ───────────────────────\033[0m")
                for i, entry in enumerate(self.executor.conversation_history, 1):
                    role = entry["role"]
                    content = entry["content"]
                    print(f"\033[1;35m│ {i}. {role.capitalize()}:\033[0m")
                    for line in content.split("\n"):
                        print(f"\033[1;35m│   {line}\033[0m")
                print("\033[1;35m╰──────────────────────────────────────────────────\033[0m\n")

            # Check if the last line of the response contains "Bye!" to exit
            if self._agent_should_exit(response):
                break
