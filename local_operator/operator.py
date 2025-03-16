import logging
import os
import platform
import signal
import uuid
from enum import Enum
from pathlib import Path
from typing import List

from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from local_operator.agents import AgentData, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import (
    VerbosityLevel,
    format_agent_output,
    print_cli_banner,
    spinner_context,
)
from local_operator.credentials import CredentialManager
from local_operator.executor import (
    CodeExecutionResult,
    LocalCodeExecutor,
    process_json_response,
)
from local_operator.helpers import (
    clean_json_response,
    clean_plain_text_response,
    remove_think_tags,
)
from local_operator.model.configure import ModelConfiguration
from local_operator.notebook import save_code_history_to_notebook
from local_operator.prompts import (
    JsonResponseFormatPrompt,
    PlanSystemPrompt,
    PlanUserPrompt,
    ReflectionUserPrompt,
    RequestClassificationSystemPrompt,
    RequestType,
    apply_attachments_to_prompt,
    create_system_prompt,
    get_request_type_instructions,
)
from local_operator.types import (
    ActionType,
    ConversationRecord,
    ConversationRole,
    ProcessResponseStatus,
    RequestClassification,
    ResponseJsonSchema,
)

# Use pyreadline3 on Windows, standard readline on other OS
if platform.system() == "Windows":
    from pyreadline3 import Readline

    readline = Readline()
else:
    import readline


class OperatorType(Enum):
    CLI = "cli"
    SERVER = "server"


def process_classification_json(response_content: str) -> RequestClassification:
    """Process and validate a JSON response string from the language model into a
    RequestClassification.

    Args:
        response_content (str): Raw response string from the model, which may be wrapped in
            markdown-style JSON code block delimiters (```json) or provided as a plain JSON object.

    Returns:
        RequestClassification: Validated classification object containing the model's output.
            See RequestClassification class for the expected schema.

    Raises:
        ValidationError: If the JSON response does not match the expected schema.
        ValueError: If no valid JSON object can be extracted from the response.
    """
    response_content = clean_json_response(response_content)

    # Validate the JSON response
    classification = RequestClassification.model_validate_json(response_content)

    return classification


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
    auto_save_conversation: bool
    verbosity_level: VerbosityLevel
    persist_agent_conversation: bool

    def __init__(
        self,
        executor: LocalCodeExecutor,
        credential_manager: CredentialManager,
        model_configuration: ModelConfiguration,
        config_manager: ConfigManager,
        type: OperatorType,
        agent_registry: AgentRegistry,
        current_agent: AgentData | None,
        auto_save_conversation: bool = False,
        verbosity_level: VerbosityLevel = VerbosityLevel.VERBOSE,
        persist_agent_conversation: bool = False,
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
            auto_save_conversation (bool): Whether to automatically save the conversation
                and improve its performance over time.
                Omit this flag to have the agent not store the conversation history, thus
                resetting it after each session.
            auto_save_conversation (bool): Whether to automatically save the conversation
                history to the agent's directory after each completed task.
            verbosity_level (VerbosityLevel): The verbosity level to use for the operator.
            persist_agent_conversation (bool): Whether to persist the agent's conversation
                history to the agent's directory after each completed task.

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
        self.auto_save_conversation = auto_save_conversation
        self.verbosity_level = verbosity_level
        self.persist_agent_conversation = persist_agent_conversation
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

            if self.verbosity_level >= VerbosityLevel.INFO:
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

    async def classify_request(
        self, user_input: str, max_attempts: int = 3, max_conversation_depth: int = 8
    ) -> RequestClassification:
        """Classify the user request into a category.

        This method constructs a conversation with the agent to classify the request type.
        It prompts the agent to analyze the user input and categorize it based on the type
        of task, whether it requires planning, and the relative effort level.

        Args:
            user_input: The text input provided by the user
            max_attempts: Maximum number of attempts to get valid classification, defaults to 3
            max_conversation_depth: Maximum number of messages to include in the conversation
                context, defaults to 8

        Returns:
            RequestClassification: The classification of the user request

        Raises:
            ValueError: If unable to get valid classification after max attempts
        """
        messages = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=RequestClassificationSystemPrompt,
                is_system_prompt=True,
            ),
        ]

        if len(self.executor.conversation_history) + 1 > max_conversation_depth:
            messages.append(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        f"... The conversation history before this message has been truncated "
                        f"to the last {max_conversation_depth} messages.  Please review the "
                        "following messages in the sequence and respond with the request "
                        "type in the required JSON format."
                    ),
                )
            )

            messages.extend(self.executor.conversation_history[-max_conversation_depth:])
        else:
            messages.extend(self.executor.conversation_history[1:])

        messages.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=user_input,
            ),
        )

        attempt = 0
        last_error = None

        while attempt < max_attempts:
            try:
                response = await self.executor.invoke_model(messages)
                response_content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )

                classification = process_classification_json(response_content)

                self.executor.conversation_history.append(
                    ConversationRecord(
                        role=ConversationRole.ASSISTANT,
                        content=(
                            f"Here is the classification for your request: {response_content}"
                        ),
                        should_summarize=False,
                        ephemeral=True,
                    )
                )

                if classification.type != RequestType.CONTINUE:
                    self.executor.set_instruction_details(response_content)

                return classification

            except ValidationError as e:
                attempt += 1
                last_error = str(e)

                if attempt < max_attempts:
                    error_message = (
                        f"The response you provided wasn't valid JSON. Error: {last_error}. "
                        "Please provide a valid JSON response matching the "
                        "RequestClassification schema."
                    )
                    messages.append(
                        ConversationRecord(
                            role=ConversationRole.USER,
                            content=error_message,
                        )
                    )
                    continue

        self.executor.set_instruction_details("")

        raise ValueError(
            f"Failed to get valid classification after {max_attempts} attempts. "
            f"Last error: {last_error}"
        )

    async def generate_plan(self, current_task_classification: RequestClassification) -> str:
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
        # Clear any existing plans from the previous invocation
        if current_task_classification.type != RequestType.CONTINUE:
            self.executor.set_current_plan("")

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

        # Remove think tags for reasoning models
        response_content = remove_think_tags(response_content)

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
                        "Please proceed according to your plan.  Choose appropriate actions "
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
                files=[],
            ),
            None,
        )

        # Save the conversation history and code execution history to the agent registry
        # if the persist_conversation flag is set.
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                self.current_agent.id,
                self.executor.conversation_history,
                self.executor.code_history,
            )

        return response_content

    async def generate_reflection(self) -> str:
        """Generate a reflection for the agent.

        This method constructs a conversation with the agent to generate a reflection.
        It starts by creating a system prompt based on the available tools and the
        predefined reflection system prompt. The method then appends the current
        """
        system_prompt = create_system_prompt(self.executor.tool_registry, "")

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
                content=ReflectionUserPrompt,
            )
        )

        response = await self.executor.invoke_model(messages)

        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

        # Remove think tags for reasoning models
        response_content = remove_think_tags(response_content)

        # Clean the response content
        response_content = clean_plain_text_response(response_content)

        self.executor.conversation_history.extend(
            [
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=response_content,
                    should_summarize=True,
                ),
            ]
        )

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
                files=[],
            ),
            None,
        )

        # Save the conversation history and code execution history to the agent registry
        # if the persist_conversation flag is set.
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                self.current_agent.id,
                self.executor.conversation_history,
                self.executor.code_history,
            )

        return response_content

    def add_task_instructions(self, request_classification: RequestClassification) -> None:
        """
        Add the task instructions as an ephemeral message to help the agent
        prioritize the information and the task at hand.
        """
        task_instructions = """
This is a {request_type} message, here are some guidelines for how to respond:

# Task Instructions

{task_instructions}
        """.format(
            request_type=request_classification.type,
            task_instructions=get_request_type_instructions(
                RequestType(request_classification.type)
            ),
        )

        self.executor.conversation_history.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=task_instructions,
                is_system_prompt=False,
                ephemeral=request_classification.type == RequestType.CONVERSATION,
            )
        )

    async def handle_user_input(
        self, user_input: str, user_message_id: str | None = None, attachments: List[str] = []
    ) -> ResponseJsonSchema | None:
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

        self.executor.update_ephemeral_messages()

        user_input_with_attachments = apply_attachments_to_prompt(user_input, attachments)

        self.executor.add_to_code_history(
            CodeExecutionResult(
                id=user_message_id if user_message_id else str(uuid.uuid4()),
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message=user_input,
                files=attachments,
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

        # Classify the user's request to determine the type of task at hand and if
        # planning is required.
        async with spinner_context(
            "Interpreting your message",
            verbosity_level=self.verbosity_level,
        ):
            classification = await self.classify_request(user_input)

        # Add the task instructions as an ephemeral message to help the agent
        # prioritize the information and the task at hand.
        self.add_task_instructions(classification)

        # Add the user's request after the task instructions
        self.executor.conversation_history.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=user_input_with_attachments,
                files=attachments,
                should_summarize=False,
            )
        )

        # Perform planning for more complex tasks
        if classification.planning_required:
            async with spinner_context(
                "Coming up with a plan",
                verbosity_level=self.verbosity_level,
            ):
                plan = await self.generate_plan(classification)

                if plan and self.verbosity_level >= VerbosityLevel.VERBOSE:
                    formatted_plan = format_agent_output(plan)
                    print("\n\033[1;36m╭─ Agent Plan ──────────────────────────────────────\033[0m")
                    print(f"\033[1;36m│\033[0m {formatted_plan}")
                    print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")
        elif classification.type != RequestType.CONTINUE:
            self.executor.set_current_plan("")

        if self.verbosity_level >= VerbosityLevel.VERBOSE:
            print("\n")

        while (
            not self._agent_is_done(response_json)
            and not self._agent_requires_user_input(response_json)
            and not self.executor.interrupted
        ):
            if self.model_configuration is None:
                raise ValueError("Model is not initialized")

            async with spinner_context(
                "Formulating a response",
                verbosity_level=self.verbosity_level,
            ):
                response = await self.executor.invoke_model(self.executor.conversation_history)

            response_content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )

            try:
                response_json = process_json_response(response_content)
            except ValidationError as e:
                logging.error(f"JSON validation error: {e}")

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
                            role=ConversationRole.USER,
                            content=(
                                "Your attempted response failed JSON schema validation. "
                                "Please review the validation errors and generate a valid "
                                "response:\n\n"
                                f"{error_details}\n\n"
                                "Your response must exactly match the expected JSON format: "
                                f"{JsonResponseFormatPrompt}"
                            ),
                            should_summarize=True,
                        ),
                    ]
                )
                continue

            result = await self.executor.process_response(response_json)

            # Update the "Agent Heads Up Display"

            if (
                response_json.action != ActionType.DONE
                and response_json.action != ActionType.ASK
                and response_json.action != ActionType.BYE
            ):
                self.executor.update_ephemeral_messages()

                # Reflect on the results of the last operation
                async with spinner_context(
                    "Reflecting on the last step",
                    verbosity_level=self.verbosity_level,
                ):
                    reflection = await self.generate_reflection()

                    if reflection and self.verbosity_level >= VerbosityLevel.VERBOSE:
                        formatted_reflection = format_agent_output(reflection)
                        print(
                            "\n\033[1;36m╭─ Agent Reflection ──────────────────────────────\033[0m"
                        )
                        print(f"\033[1;36m│\033[0m {formatted_reflection}")
                        print(
                            "\033[1;36m╰──────────────────────────────────────────────────\033[0m\n"
                        )

            # Auto-save on each step if enabled
            if self.auto_save_conversation:
                try:
                    self.handle_autosave(
                        self.agent_registry.config_dir,
                        self.executor.conversation_history,
                        self.executor.code_history,
                    )
                except Exception as e:
                    error_str = str(e)

                    if self.verbosity_level >= VerbosityLevel.INFO:
                        print(
                            "\n\033[1;31m✗ Error encountered while auto-saving conversation:\033[0m"
                        )
                        print(f"\033[1;36m│ Error Details:\033[0m\n{error_str}")

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
        print_cli_banner(
            self.config_manager, self.current_agent, self.executor.persist_conversation
        )

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
            if (
                response_json
                and self._agent_requires_user_input(response_json)
                and self.verbosity_level >= VerbosityLevel.QUIET
            ):
                response_content = response_json.response
                print("\n\033[1;36m╭─ Agent Question Requires Input ────────────────\033[0m")
                print(f"\033[1;36m│\033[0m {response_content}")
                print("\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")

    def handle_autosave(
        self,
        config_dir: Path,
        conversation: List[ConversationRecord],
        execution_history: List[CodeExecutionResult],
    ) -> None:
        """
        Update the autosave agent's conversation and execution history.

        This method persists the provided conversation and execution history
        by utilizing the agent registry to update the autosave agent's data.
        This ensures that the current state of the interaction is preserved.

        Args:
            conversation (List[ConversationRecord]): The list of conversation records
                to be saved. Each record represents a turn in the conversation.
            execution_history (List[CodeExecutionResult]): The list of code execution
                results to be saved. Each result represents the outcome of a code
                execution attempt.
            config_dir (Path): The directory to save the autosave notebook to.
        Raises:
            KeyError: If the autosave agent does not exist in the agent registry.
        """
        self.agent_registry.update_autosave_conversation(conversation, execution_history)

        notebook_path = config_dir / "autosave.ipynb"

        save_code_history_to_notebook(
            code_history=execution_history,
            model_configuration=self.model_configuration,
            max_conversation_history=self.config_manager.get_config_value(
                "max_conversation_history", 100
            ),
            detail_conversation_length=self.config_manager.get_config_value(
                "detail_conversation_length", 35
            ),
            max_learnings_history=self.config_manager.get_config_value("max_learnings_history", 50),
            file_path=notebook_path,
        )
