import importlib.metadata
import os
import platform
import readline
import signal
from enum import Enum
from pathlib import Path
from typing import Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from tiktoken import encoding_for_model

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
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


class OperatorType(Enum):
    CLI = "cli"
    SERVER = "server"


def get_installed_packages_str() -> str:
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


def create_system_prompt() -> str:
    """Create the system prompt for the agent."""

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

    installed_packages_str = get_installed_packages_str()

    base_system_prompt = (
        base_system_prompt.replace("{{system_details_str}}", system_details_str)
        .replace("{{installed_packages_str}}", installed_packages_str)
        .replace("{{user_system_prompt}}", user_system_prompt)
    )

    return base_system_prompt


class Operator:
    """Environment manager for interacting with language models.

    Attributes:
        model: The configured ChatOpenAI or ChatOllama instance
        executor: LocalCodeExecutor instance for handling code execution
        config_manager: ConfigManager instance for managing configuration
        credential_manager: CredentialManager instance for managing credentials
        executor_is_processing: Whether the executor is processing a response
    """

    credential_manager: CredentialManager
    config_manager: ConfigManager
    model: Union[ChatOpenAI, ChatOllama, ChatAnthropic]
    executor: LocalCodeExecutor
    executor_is_processing: bool
    type: OperatorType

    def __init__(
        self,
        executor: LocalCodeExecutor,
        credential_manager: CredentialManager,
        model_instance: Union[ChatOpenAI, ChatOllama, ChatAnthropic],
        config_manager: ConfigManager,
        type: OperatorType,
    ):
        """Initialize the CLI by loading credentials or prompting for them.

        Args:
            hosting (str): Hosting platform (deepseek, openai, or ollama)
            model (str): Model name to use
        """
        self.credential_manager = credential_manager
        self.config_manager = config_manager
        self.model = model_instance
        self.executor = executor
        self.executor_is_processing = False
        self.type = type

        if self.type == OperatorType.CLI:
            self._load_input_history()
            self._setup_interrupt_handler()

    def _setup_interrupt_handler(self) -> None:
        """Set up the interrupt handler for Ctrl+C."""

        def handle_interrupt(signum, frame):
            if self.executor.interrupted or not self.executor_is_processing:
                # Pass through SIGINT if already interrupted or the
                # executor is not processing a response
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

    def _print_banner(self) -> None:
        """Print the banner for the chat CLI."""
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
        hosting = self.config_manager.get_config_value("hosting")
        model = self.config_manager.get_config_value("model_name")
        if hosting:
            hosting_text = f"Using hosting: {hosting}"
            padding = 49 - len(hosting_text)
            print(f"\033[1;36m│ {hosting_text}{' ' * padding}│\033[0m")
        if model:
            model_text = f"Using model: {model}"
            padding = 49 - len(model_text)
            print(f"\033[1;36m│ {model_text}{' ' * padding}│\033[0m")
        if hosting or model:
            print("\033[1;36m│──────────────────────────────────────────────────│\033[0m")
        print("\033[1;36m│ Type 'exit' or 'quit' to quit                    │\033[0m")
        print("\033[1;36m│ Press Ctrl+C to interrupt current task           │\033[0m")
        print("\033[1;36m╰──────────────────────────────────────────────────╯\033[0m\n")

        # Print configuration options
        if os.getenv("LOCAL_OPERATOR_DEBUG", "false").lower() == "true":
            print("\033[1;36m╭─ Configuration ────────────────────────────────\033[0m")
            print(f"\033[1;36m│\033[0m Hosting: {self.config_manager.get_config_value('hosting')}")
            print(f"\033[1;36m│\033[0m Model: {self.config_manager.get_config_value('model_name')}")
            conv_len = self.config_manager.get_config_value("conversation_length")
            detail_len = self.config_manager.get_config_value("detail_length")
            print(f"\033[1;36m│\033[0m Conversation Length: {conv_len}")
            print(f"\033[1;36m│\033[0m Detail Length: {detail_len}")
            print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")

    async def handle_user_input(self, user_input: str) -> BaseMessage | None:
        """Process user input and generate agent responses.

        This method handles the core interaction loop between the user and agent:
        1. Adds user input to conversation history
        2. Resets agent state for new interaction
        3. Repeatedly generates and processes agent responses until:
           - Agent indicates completion
           - Agent requires more user input
           - User interrupts execution
           - Code execution is cancelled

        Args:
            user_input: The text input provided by the user

        Raises:
            ValueError: If the model is not properly initialized
        """
        self.executor.conversation_history.append(
            {"role": ConversationRole.USER.value, "content": user_input}
        )

        response = None
        self.executor.reset_step_counter()
        self.executor_is_processing = True

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
            self.print_conversation_history()

        return response

    def print_conversation_history(self) -> None:
        """Print the conversation history for debugging."""
        tokenizer = encoding_for_model("gpt-4o")
        total_tokens = sum(
            len(tokenizer.encode(entry["content"])) for entry in self.executor.conversation_history
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

    async def chat(self) -> None:
        """Run the interactive chat interface with code execution capabilities.

        This method implements the main chat loop that:
        1. Displays a command prompt showing the current working directory
        2. Accepts user input with command history support
        3. Processes input through the language model
        4. Executes any generated code
        5. Displays debug information if enabled
        6. Handles special commands like 'exit'/'quit'
        7. Continues until explicitly terminated or [BYE] received

        The chat maintains conversation history and system context between interactions.
        Debug mode can be enabled by setting LOCAL_OPERATOR_DEBUG=true environment variable.

        Special keywords in model responses:
        - [ASK]: Model needs additional user input
        - [DONE]: Model has completed its task
        - [BYE]: Gracefully exit the chat session
        """
        self._print_banner()

        self.executor.conversation_history = [
            {
                "role": ConversationRole.SYSTEM.value,
                "content": create_system_prompt(),
            }
        ]

        while True:
            self.executor_is_processing = False

            prompt = f"You ({os.getcwd()}): > "
            user_input = self._get_input_with_history(prompt)

            if not user_input.strip():
                continue

            if user_input.lower() == "exit" or user_input.lower() == "quit":
                break

            response = await self.handle_user_input(user_input)

            # Check if the last line of the response contains "[BYE]" to exit
            if self._agent_should_exit(response):
                break

            # Print the last assistant message if the agent is asking for user input
            if response and self._agent_requires_user_input(response):
                response_content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                ).replace("[ASK]", "")
                print("\n\033[1;36m╭─ Agent Question Requires Input ────────────────\033[0m")
                print(f"\033[1;36m│\033[0m {response_content}")
                print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")
