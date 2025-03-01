import asyncio
import os
import readline
import signal
from enum import Enum
from pathlib import Path

from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from local_operator.agents import AgentData, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import format_agent_output, print_cli_banner, spinner
from local_operator.credentials import CredentialManager
from local_operator.executor import (
    CodeExecutionResult,
    LocalCodeExecutor,
    process_json_response,
)
from local_operator.model.configure import ModelConfiguration
from local_operator.prompts import (
    PlanSystemPrompt,
    PlanUserPrompt,
    create_system_prompt,
)
from local_operator.types import (
    ConversationRecord,
    ConversationRole,
    ProcessResponseStatus,
    ResponseJsonSchema,
)


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
        agent_registry: AgentRegistry instance for managing agents
        current_agent: The current agent to use for this session
        training_mode: Whether the operator is in training mode.  If True, the operator will save
        the conversation history to the agent's directory after each completed task.  This
        allows the agent to learn from its experiences and improve its performance over time.
        Omit this flag to have the agent not store the conversation history, thus resetting it
        after each session.
    """

    credential_manager: CredentialManager
    config_manager: ConfigManager
    model_configuration: ModelConfiguration
    executor: LocalCodeExecutor
    executor_is_processing: bool
    type: OperatorType
    agent_registry: AgentRegistry
    current_agent: AgentData | None
    training_mode: bool

    def __init__(
        self,
        executor: LocalCodeExecutor,
        credential_manager: CredentialManager,
        model_configuration: ModelConfiguration,
        config_manager: ConfigManager,
        type: OperatorType,
        agent_registry: AgentRegistry,
        current_agent: AgentData | None,
        training_mode: bool,
    ):
        """Initialize the Operator with required components.

        Args:
            executor (LocalCodeExecutor): Executor instance for handling code execution
            credential_manager (CredentialManager): Manager for handling credentials
            model_configuration (ModelConfiguration): The configured language model instance
            config_manager (ConfigManager): Manager for handling configuration
            type (OperatorType): Type of operator (CLI or Server)
            agent_registry (AgentRegistry): Registry for managing AI agents
            current_agent (AgentData | None): The current agent to use for this session
            training_mode (bool): Whether the operator is in training mode.
                If True, the operator will save the conversation history to the agent's directory
                after each completed task. This allows the agent to learn from its experiences
                and improve its performance over time.
                Omit this flag to have the agent not store the conversation history, thus
                resetting it after each session.

        The Operator class serves as the main interface for interacting with language models,
        managing configuration, credentials, and code execution. It handles both CLI and
        server-based operation modes.
        """
        self.credential_manager = credential_manager
        self.config_manager = config_manager
        self.model_configuration = model_configuration
        self.executor = executor
        self.executor_is_processing = False
        self.type = type
        self.agent_registry = agent_registry
        self.current_agent = current_agent
        self.training_mode = training_mode

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

    async def generate_plan(self) -> str:
        """Generate a plan for the agent to follow.

        This method constructs a conversation with the agent to generate a plan. It
        starts by creating a system prompt based on the available tools and the
        predefined plan system prompt. The method then appends the current
        conversation history and a user prompt to the messages list. The agent is
        invoked to generate a response, which is checked for a skip planning
        directive. If the directive is found, the method sets a default plan and
        returns an empty string. Otherwise, it updates the conversation history
        with the agent's response and a user instruction to proceed according to
        the plan. The plan is also set in the executor and added to the code
        history.

        Returns:
            str: The generated plan or an empty string if planning is skipped.
        """
        system_prompt = create_system_prompt(self.executor.tool_registry, PlanSystemPrompt)

        messages = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=system_prompt,
                is_system_prompt=True,
            ),
        ]

        messages.extend(self.executor.conversation_history[1:])

        messages.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=PlanUserPrompt,
            )
        )

        response = await self.executor.invoke_model(messages)

        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

        if "[SKIP_PLANNING]" in response_content:
            self.executor.set_current_plan("No plan required, proceed with execution.")
            return ""

        self.executor.conversation_history.extend(
            [
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=response_content,
                    should_summarize=False,
                ),
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "Please proceed according to the plan.  Choose appropriate actions "
                        "and follow the JSON schema for your response.  Do not include any "
                        "other text or comments aside from the JSON object."
                    ),
                    should_summarize=False,
                ),
            ]
        )

        self.executor.set_current_plan(response_content)
        self.executor.add_to_code_history(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message=response_content,
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.SUCCESS,
            ),
            None,
        )

        return response_content

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
            ConversationRecord(
                role=ConversationRole.USER,
                content=user_input,
                should_summarize=False,
            )
        )
        self.executor.add_to_code_history(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message=user_input,
                role=ConversationRole.USER,
                status=ProcessResponseStatus.SUCCESS,
            ),
            None,
        )

        self.executor.reset_learnings()

        response_json: ResponseJsonSchema | None = None
        response: BaseMessage | None = None
        self.executor.reset_step_counter()
        self.executor_is_processing = True
        self.executor.update_ephemeral_messages()

        spinner_task = asyncio.create_task(spinner("Generating plan"))
        try:
            plan = await self.generate_plan()

            if plan:
                formatted_plan = format_agent_output(plan)
                print("\n\033[1;36m╭─ Agent Plan ──────────────────────────────────────\033[0m")
                print(f"\033[1;36m│\033[0m {formatted_plan}")
                print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")
        finally:
            spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass

        while (
            not self._agent_is_done(response_json)
            and not self._agent_requires_user_input(response_json)
            and not self.executor.interrupted
        ):
            if self.model_configuration is None:
                raise ValueError("Model is not initialized")

            # Add environment details, etc.
            self.executor.update_ephemeral_messages()

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
                response_json = process_json_response(response_content)
            except ValidationError as e:
                error_details = "\n".join(
                    f"Error {i+1}:\n"
                    f"  Location: {' -> '.join(str(loc) for loc in err['loc'])}\n"
                    f"  Type: {err['type']}\n"
                    f"  Message: {err['msg']}"
                    for i, err in enumerate(e.errors())
                )

                self.executor.conversation_history.extend(
                    [
                        ConversationRecord(
                            role=ConversationRole.ASSISTANT,
                            content=response_content,
                            should_summarize=True,
                        ),
                        ConversationRecord(
                            role=ConversationRole.SYSTEM,
                            content=(
                                "[SYSTEM] Your attempted response failed JSON schema validation. "
                                "Please review the validation errors and generate a valid "
                                "response:\n\n"
                                f"{error_details}\n\n"
                                "Your response must exactly match the expected JSON schema "
                                "structure. Please reformat your response to continue with "
                                "the task.  Do not include any other text or comments aside "
                                "from the JSON object."
                            ),
                            should_summarize=True,
                        ),
                    ]
                )
                continue

            result = await self.executor.process_response(response_json)

            # Break out of the agent flow if the user cancels the code execution
            if (
                result.status == ProcessResponseStatus.CANCELLED
                or result.status == ProcessResponseStatus.INTERRUPTED
            ):
                break

        # Save the conversation history if an agent is being used and training mode is enabled
        if self.training_mode and self.current_agent:
            self.agent_registry.save_agent_conversation(
                self.current_agent.id, self.executor.conversation_history
            )

        if os.environ.get("LOCAL_OPERATOR_DEBUG") == "true":
            self.print_conversation_history()

        return response_json

    def print_conversation_history(self) -> None:
        """Print the conversation history for debugging."""
        total_tokens = self.executor.get_invoke_token_count(self.executor.conversation_history)

        print("\n\033[1;35m╭─ Debug: Conversation History ───────────────────────\033[0m")
        print(f"\033[1;35m│ Message tokens: {total_tokens}                       \033[0m")
        print(f"\033[1;35m│ Session tokens: {self.executor.get_session_token_usage()}\033[0m")
        for i, entry in enumerate(self.executor.conversation_history, 1):
            role = entry.role
            content = entry.content
            print(f"\033[1;35m│ {i}. {role.value.capitalize()}:\033[0m")
            for line in content.split("\n"):
                print(f"\033[1;35m│   {line}\033[0m")
        print("\033[1;35m╰──────────────────────────────────────────────────\033[0m\n")

    async def execute_single_command(self, command: str) -> ResponseJsonSchema | None:
        """Execute a single command in non-interactive mode.

        This method is used for one-off command execution rather than interactive chat.
        It initializes a fresh conversation history (if not already initialized),
        processes the command through the language model, and returns the result.

        Args:
            command (str): The command/instruction to execute

        Returns:
            ResponseJsonSchema | None: The processed response from the language model,
                or None if no valid response was generated
        """
        try:
            self.executor.initialize_conversation_history()
        except ValueError:
            # Conversation history already initialized
            pass

        result = await self.handle_user_input(command)
        return result

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
        print_cli_banner(self.config_manager, self.current_agent, self.training_mode)

        try:
            self.executor.initialize_conversation_history()
        except ValueError:
            # Conversation history already initialized
            pass

        while True:
            self.executor_is_processing = False
            self.executor.interrupted = False

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
