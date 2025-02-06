import asyncio
import os
import readline
import signal
from enum import Enum
from pathlib import Path

from langchain_core.messages import BaseMessage
from pydantic import ValidationError
from tiktoken import encoding_for_model

import local_operator.tools as tools
from local_operator.config import ConfigManager
from local_operator.console import print_cli_banner, spinner
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
from local_operator.model import ModelType
from local_operator.prompts import create_system_prompt
from local_operator.types import ResponseJsonSchema


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
    model: ModelType
    executor: LocalCodeExecutor
    executor_is_processing: bool
    type: OperatorType

    def __init__(
        self,
        executor: LocalCodeExecutor,
        credential_manager: CredentialManager,
        model_instance: ModelType,
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

    def _agent_is_done(self, response: ResponseJsonSchema | None) -> bool:
        """Check if the agent has completed its task."""
        if response is None:
            return False

        return response.action == "DONE" or self._agent_should_exit(response)

    def _agent_requires_user_input(self, response: ResponseJsonSchema | None) -> bool:
        """Check if the agent requires user input."""
        if response is None:
            return False

        return response.action == "ASK"

    def _agent_should_exit(self, response: ResponseJsonSchema | None) -> bool:
        """Check if the agent should exit."""
        if response is None:
            return False

        return response.action == "BYE"

    def _process_json_response(self, response_str: str) -> ResponseJsonSchema:
        """Process the JSON response from the model."""
        response_content = response_str
        if response_content.startswith("```json"):
            response_content = response_content[7:]
        if response_content.endswith("```"):
            response_content = response_content[:-3]

        # Validate the JSON response
        response_json = ResponseJsonSchema.model_validate_json(response_content)

        return response_json

    async def handle_user_input(self, user_input: str) -> ResponseJsonSchema | None:
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

        response_json: ResponseJsonSchema | None = None
        response: BaseMessage | None = None
        self.executor.reset_step_counter()
        self.executor_is_processing = True

        while (
            not self._agent_is_done(response_json)
            and not self._agent_requires_user_input(response_json)
            and not self.executor.interrupted
        ):
            if self.model is None:
                raise ValueError("Model is not initialized")

            spinner_task = asyncio.create_task(spinner("Generating response"))
            try:
                response = await self.executor.invoke_model(self.executor.conversation_history)
            finally:
                spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass

            response_content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )

            try:
                response_json = self._process_json_response(response_content)
            except ValidationError:
                self.executor.conversation_history.append(
                    {
                        "role": ConversationRole.SYSTEM.value,
                        "content": "Invalid JSON response.  Please try again and "
                        "generate a valid JSON response that exactly matches the JSON "
                        "schema.",
                    }
                )
                continue

            result = await self.executor.process_response(response_json)

            # Break out of the agent flow if the user cancels the code execution
            if (
                result.status == ProcessResponseStatus.CANCELLED
                or result.status == ProcessResponseStatus.INTERRUPTED
            ):
                break

        if os.environ.get("LOCAL_OPERATOR_DEBUG") == "true":
            self.print_conversation_history()

        return response_json

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
        print_cli_banner(self.config_manager)

        self.executor.conversation_history = [
            {
                "role": ConversationRole.SYSTEM.value,
                "content": create_system_prompt(tools),
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

            response_json = await self.handle_user_input(user_input)

            # Check if the last line of the response contains "[BYE]" to exit
            if self._agent_should_exit(response_json):
                break

            # Print the last assistant message if the agent is asking for user input
            if response_json and self._agent_requires_user_input(response_json):
                response_content = response_json.response
                print("\n\033[1;36m╭─ Agent Question Requires Input ────────────────\033[0m")
                print(f"\033[1;36m│\033[0m {response_content}")
                print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")
