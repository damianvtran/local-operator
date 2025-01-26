import re
import io
import sys
import os
import platform
import threading
import pkg_resources
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import SecretStr
import readline
from typing import Any

from local_operator.credentials import CredentialManager


class LocalCodeExecutor:
    context: dict[str, Any]
    conversation_history: list[dict[str, str]]
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

    def extract_code_blocks(self, text):
        """Extract Python code blocks from text using markdown-style syntax.

        Args:
            text (str): The text containing potential code blocks

        Returns:
            list: A list of extracted code blocks as strings
        """
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    async def check_code_safety(self, code):
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
        response = self.model.invoke(self.conversation_history)
        self.conversation_history.pop()

        return "yes" in response.content.strip().lower()

    async def execute_code(self, code, max_retries=2, timeout=30):
        """Execute Python code with safety checks and context management.

        Args:
            code (str): The Python code to execute
            max_retries (int): Maximum number of retry attempts
            timeout (int): Maximum execution time in seconds

        Returns:
            str: Execution result message or error message
        """

        async def _execute_with_timeout(code_to_execute):
            """Helper function to execute code with timeout."""
            result = {"success": False, "error": None}

            def run_code():
                try:
                    exec(code_to_execute, self.context)
                    result["success"] = True
                except Exception as e:
                    result["error"] = e
                    result["success"] = False

            exec_thread = threading.Thread(target=run_code)
            exec_thread.start()
            exec_thread.join(timeout=timeout)

            if exec_thread.is_alive():
                raise TimeoutError(f"Code execution timed out after {timeout} seconds")

            if not result["success"] and result["error"]:
                raise result["error"]
            elif not result["success"]:
                raise TimeoutError("Code execution failed")

        async def _capture_output(code_to_execute):
            """Helper function to capture and return execution output."""
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout

            try:
                await _execute_with_timeout(code_to_execute)
                output = new_stdout.getvalue()
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
                sys.stdout = old_stdout

        async def _handle_error(error, attempt=None):
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
                    response = self.model.invoke(self.conversation_history)
                    new_code = self.extract_code_blocks(response.content)
                    if new_code:
                        return await _capture_output(new_code[0])
                except Exception as retry_error:
                    print(f"\n\033[1;31m✗ Error during execution (attempt {attempt + 1}):\033[0m")
                    print("\033[1;34m╞══════════════════════════════════════════╡")
                    print(f"\033[1;36m│ Error:\033[0m\n{str(retry_error)}")
                    if attempt < max_retries - 1:
                        print("\033[1;36m│\033[0m \033[1;33mAnother attempt will be made...\033[0m")

            return await _handle_error(initial_error, attempt=max_retries)

    def _format_agent_output(self, text):
        """Format agent output with colored sidebar and indentation."""
        lines = text.split("\n")
        formatted_lines = []
        for line in lines:
            formatted_lines.append(f"\033[1;36m│\033[0m {line}")
        return "\n".join(formatted_lines)

    async def process_response(self, response):
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

        self.conversation_history.append({"role": "agent", "content": response})

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

    def __init__(self, hosting: str, model: str):
        """Initialize the CLI by loading credentials or prompting for them.

        Args:
            hosting (str): Hosting platform (deepseek, openai, or ollama)
            model (str): Model name to use
        """
        if not hosting:
            raise ValueError("Hosting is required")
        if not model:
            raise ValueError("Model is required")

        credential_manager = CredentialManager()

        # Configure model based on hosting
        if hosting == "deepseek":
            base_url = "https://api.deepseek.com/v1"
            api_key = credential_manager.get_api_key("DEEPSEEK_API_KEY")
            if not api_key:
                api_key = credential_manager.prompt_for_api_key("DEEPSEEK_API_KEY")
            self.model = ChatOpenAI(
                api_key=SecretStr(api_key),
                temperature=0.5,
                base_url=base_url,
                model=model,
            )
        elif hosting == "openai":
            base_url = "https://api.openai.com"
            api_key = credential_manager.get_api_key("OPENAI_API_KEY")
            if not api_key:
                api_key = credential_manager.prompt_for_api_key("OPENAI_API_KEY")
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

        self.executor = LocalCodeExecutor(self.model)
        self.input_history = []  # Store user input history
        self.history_index = 0  # Track current position in history

    def _get_input_with_history(self, prompt: str) -> str:
        """Get user input with history navigation using up/down arrows."""

        def completer(text, state):
            # Filter history based on current input
            options = [i for i in self.input_history if i.startswith(text)]
            if state < len(options):
                return options[state]
            return None

        readline.set_completer(completer)
        readline.parse_and_bind("tab: complete")

        # Set up history navigation
        def pre_input_hook():
            if len(self.input_history) > 0:
                # Add all history items
                for item in self.input_history:
                    readline.add_history(item)

                # Add empty string to represent current position
                readline.add_history("")
                readline.set_history_length(len(self.input_history) + 1)

        readline.set_pre_input_hook(pre_input_hook)

        try:
            # Save cursor position
            print("\033[s", end="")
            user_input = input(prompt)
            # Restore cursor position and clear line
            print("\033[u\033[K", end="")

            if user_input and (not self.input_history or user_input != self.input_history[-1]):
                self.input_history.append(user_input)
            return user_input
        except KeyboardInterrupt:
            return "exit"

    def _agent_is_done(self, response):
        """Check if the agent has completed its task."""
        if response is None:
            return False

        return "DONE" in response.content.strip().splitlines()[
            -1
        ].strip() or self._agent_should_exit(response)

    def _agent_should_exit(self, response):
        """Check if the agent should exit."""
        if response is None:
            return False

        return "Bye!" in response.content.strip().splitlines()[-1].strip()

    def _setup_prompt(self):
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

        installed_packages = pkg_resources.working_set
        installed_packages_str = ""
        for package in installed_packages:
            installed_packages_str += f"{package.project_name}=={package.version}\n"

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

    async def chat(self):
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
            user_input = self._get_input_with_history(
                f"\033[1m\033[94mYou ({os.getcwd()}):\033[0m \033[1m>\033[0m "
            )

            if not user_input.strip():
                continue

            if user_input.lower() == "exit" or user_input.lower() == "quit":
                break

            self.executor.conversation_history.append({"role": "user", "content": user_input})

            response = None
            self.executor.reset_step_counter()

            while not self._agent_is_done(response):
                response = self.model.invoke(self.executor.conversation_history)
                await self.executor.process_response(response.content)

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
