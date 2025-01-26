import re
import io
import sys
import os
import threading
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core.messages import BaseMessage
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

        self.executor.conversation_history = [
            {
                "role": "system",
                "content": f"""
                You are a Python code execution assistant. You strictly run Python code locally.
                You are able to run python code on the local machine and install required packages
                to run the code.  Think about the code that you will need to run in order
                to acheive the user's goals and output this code in ```python``` code blocks.
                Output only one step at a time.  If the step is not the final step, do not
                include DONE in that response.  If the step is the final step, include DONE
                in that response.  Include DONE outside of the code block, on a new line after
                the code block.  The operator will request you to keep running steps until
                you have completed the task.  At each step you will need to figure out
                what the next step is in real time based on the previous output.  You
                will be able to use the output of the previous step to determine the next
                step.

                For user commands, the following rules apply:
                - Plan the sequence of events needed to achieve the user's goals.  Break
                  down complex tasks into sequential steps.  Output only one step at a time.
                - For each step, generate only the code needed for that step.
                - After completing a step, analyze the results and determine next steps.
                - When you believe the task is complete, include "DONE" as the last line
                  in your response.  This will end the loop for that task and the next
                  user input will be a new task.
                - Don't re-do any DONE steps from conversation history unless explicitly
                  requested by the user.  Pay attention to more recent commands from
                  the user over older commands, but use the older commands for context.
                - Keep your responses concise and to the point.
                - Analyze and execute python code blocks.
                - Any python code blocks in your response will be interpreted as code to execute.
                  These must all have the ```python``` syntax.
                - Provide all code in a single code block instead of multiple code blocks.
                - In the code block, focus on printing the results in a human readable format in the
                  terminal such that it is easy to understand and follow.
                - Add the installation of required packages in the code blocks for any
                  that are not the standard library (for example, pandas, numpy, and others)
                - Validate python code safety first
                - Never execute harmful python code
                - Maintain secure execution.
                - Don't write any other text in your response.
                - If the user indicates that they want to exit, respond with a goodbye message
                  and then on a separate line, "Bye!"
                You only execute python code and no other code.

                You are currently in the following directory: {os.getcwd()}
                """,
            }
        ]

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
