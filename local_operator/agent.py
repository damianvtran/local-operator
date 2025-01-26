import os
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import readline  # Added for input history support


class CredentialManager:
    """Manages API credentials storage and retrieval."""

    def __init__(self):
        self.config_dir = Path.home() / ".local-operator"
        self.config_file = self.config_dir / "config.env"
        self._ensure_config_exists()
        # Load environment variables from config file
        load_dotenv(self.config_file)

    def _ensure_config_exists(self):
        """Ensure config directory and file exist, prompt for API key if needed."""
        self.config_dir.mkdir(exist_ok=True, mode=0o700)
        if not self.config_file.exists():
            print("\033[1;36m╭──────────────────────────────────────────────────╮\033[0m")
            print("\033[1;36m│ DeepSeek API Key Setup                           │\033[0m")
            print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
            print("\033[1;36m│ API key not found. Let's set it up.              │\033[0m")
            print("\033[1;36m╰──────────────────────────────────────────────────╯\033[0m")

            api_key = input("\033[1m\033[94mPlease enter your DeepSeek API key: \033[0m").strip()
            if not api_key:
                raise ValueError("\033[1;31mAPI key is required to use this application\033[0m")

            with open(self.config_file, "w") as f:
                f.write(f"DEEPSEEK_API_KEY={api_key}\n")
            self.config_file.chmod(0o600)

            print("\n\033[1;32m✓ API key successfully saved!\033[0m")

    def get_api_key(self, key: str):
        """Retrieve the API key from config file.

        Args:
            key (str): The environment variable key to retrieve

        Returns:
            str: The API key value
        """
        return os.getenv(key)

    def prompt_for_api_key(self, key: str) -> str:
        """Prompt the user to enter an API key if not present in environment.

        Args:
            key (str): The environment variable key to check

        Returns:
            str: The API key value
        """
        api_key = self.get_api_key(key)
        if not api_key:
            print(f"{key} not found in configuration.")
            api_key = input(f"Please enter your {key}: ").strip()
            if not api_key:
                raise ValueError(f"{key} is required to use this application")

            # Save the new API key to config file
            with open(self.config_file, "a") as f:
                f.write(f"\n{key}={api_key}\n")
            self.config_file.chmod(0o600)

            # Reload environment variables
            load_dotenv(self.config_file, override=True)

        return api_key


class LocalCodeExecutor:
    """A class to handle local Python code execution with safety checks and context management.

    Attributes:
        context (dict): A dictionary to maintain execution context between code blocks
        conversation_history (list): A list of message dictionaries tracking the conversation
        model: The language model used for code analysis and safety checks
    """

    def __init__(self, model):
        """Initialize the LocalCodeExecutor with a language model.

        Args:
            model: The language model instance to use for code analysis
        """
        self.context = {}
        self.conversation_history = []
        self.model = model

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
        - Execute system commands
        - Access sensitive system resources
        - Perform network operations
        - Otherwise compromise system security

        Respond with only "no" if the code appears safe to execute.
        """

        self.conversation_history.append({"role": "user", "content": safety_check_prompt})
        response = self.model.invoke(self.conversation_history)
        self.conversation_history.pop()

        return "yes" in response.content.strip().lower()

    async def execute_code(self, code):
        """Execute Python code with safety checks and context management.

        Args:
            code (str): The Python code to execute

        Returns:
            str: Execution result message or error message
        """
        try:
            is_dangerous = await self.check_code_safety(code)
            if is_dangerous:
                confirm = input(
                    "Warning: Potentially dangerous operation detected. Proceed? (y/n): "
                )
                if confirm.lower() != "y":
                    return "Code execution canceled by user"

            exec(code, self.context)
            return "Code executed successfully"
        except Exception as e:
            return f"Error executing code: {str(e)}"

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
        print("\n\033[1;36m╭─ Agent Response ────────────────────────────────\033[0m")
        print(formatted_response)
        print("\033[1;36m╰──────────────────────────────────────────────────\033[0m")

        self.conversation_history.append({"role": "assistant", "content": response})

        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            print("\n\033[1;36m╭─ Executing Code Blocks ─────────────────────────\033[0m")
            for code in code_blocks:
                print("\n\033[1;36m│ Executing:\033[0m\n{}".format(code))
                result = await self.execute_code(code)
                print("\033[1;36m│ Result:\033[0m {}".format(result))

                self.conversation_history.append(
                    {"role": "system", "content": "Code execution result:\n{}".format(result)}
                )
                self.context["last_code_result"] = result
            print("\033[1;36m╰──────────────────────────────────────────────────\033[0m")


class CliOperator:
    """A command-line interface for interacting with DeepSeek's language model.

    Attributes:
        model: The configured ChatOpenAI instance for DeepSeek
        executor: LocalCodeExecutor instance for handling code execution
    """

    def __init__(self):
        """Initialize the CLI by loading credentials or prompting for them."""
        credential_manager = CredentialManager()

        # Try to get API key from environment first, then from config file
        api_key = credential_manager.get_api_key("DEEPSEEK_API_KEY")
        if not api_key:
            api_key = credential_manager.prompt_for_api_key("DEEPSEEK_API_KEY")

        self.model = ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.5,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        self.executor = LocalCodeExecutor(self.model)
        self.input_history = []  # Store user input history
        self.history_index = 0  # Track current position in history

    def _get_input_with_history(self, prompt):
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
            nonlocal self
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

    async def chat(self):
        """Run the interactive chat interface with code execution capabilities."""
        print("\033[1;36m╭──────────────────────────────────────────────────╮\033[0m")
        print("\033[1;36m│ Local Executor Agent CLI                         │\033[0m")
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
                "content": """
                You are a Python code execution assistant. You strictly run Python code locally.
                You are able to run python code on the local machine.
                Keep your responses concise and to the point. Your functions:
                - Analyze and execute python code blocks.
                - Any python code blocks in your response will be interpreted as code to execute.
                - Provide all code in a single code block instead of multiple code blocks.
                - Validate python code safety first
                - Never execute harmful python code
                - Maintain secure execution.
                - Don't write any other text in your response.
                You only execute python code and no other code.
                """,
            }
        ]

        while True:
            user_input = self._get_input_with_history("\033[1m\033[94mYou:\033[0m \033[1m>\033[0m ")
            if not user_input.strip():
                continue
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                break

            self.executor.conversation_history.append({"role": "user", "content": user_input})
            response = self.model.invoke(self.executor.conversation_history)
            await self.executor.process_response(response.content)
