import logging
import os
import platform
import re
import signal
import uuid
from enum import Enum
from pathlib import Path
from typing import AsyncGenerator, List

from langchain_core.messages import BaseMessage
from pydantic import ValidationError

from local_operator.agents import AgentData, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import (
    VerbosityLevel,
    print_agent_response,
    print_cli_banner,
    spinner_context,
)
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.executor import (
    CodeExecutionResult,
    LocalCodeExecutor,
    process_json_response,
)
from local_operator.helpers import clean_plain_text_response, remove_think_tags
from local_operator.model.configure import ModelConfiguration
from local_operator.notebook import save_code_history_to_notebook
from local_operator.prompts import (
    FinalResponseInstructions,
    JsonResponseFormatSchema,
    PlanSystemPrompt,
    PlanUserPrompt,
    ReflectionUserPrompt,
    RequestClassificationSystemPrompt,
    RequestClassificationUserPrompt,
    RequestType,
    TaskInstructionsPrompt,
    apply_attachments_to_prompt,
    create_action_interpreter_prompt,
    create_system_prompt,
    get_request_type_instructions,
)
from local_operator.types import (
    ActionType,
    ConversationRecord,
    ConversationRole,
    ExecutionType,
    ProcessResponseOutput,
    ProcessResponseStatus,
    RelativeEffortLevel,
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


def process_classification_response(response_content: str) -> RequestClassification:
    """Process and validate a response string from the language model into a
    RequestClassification.

    Args:
        response_content (str): Raw response string from the model, which may contain XML tags
            like <type>, <planning_required>, <relative_effort>, and <subject_change>.

    Returns:
        RequestClassification: Validated classification object containing the model's output.
            See RequestClassification class for the expected schema.

    Raises:
        ValidationError: If the extracted data does not match the expected schema.
        ValueError: If no valid classification data can be extracted from the response.
    """
    # Extract values from XML-like tags if present
    classification_data = {}

    # Look for each expected tag in the response
    for tag in ["type", "planning_required", "relative_effort", "subject_change"]:
        start_tag = f"<{tag}>"
        end_tag = f"</{tag}>"

        if start_tag in response_content and end_tag in response_content:
            start_idx = response_content.find(start_tag) + len(start_tag)
            end_idx = response_content.find(end_tag)
            if start_idx < end_idx:
                value = response_content[start_idx:end_idx].strip().lower()

                # Convert string boolean values to actual booleans
                if value == "true":
                    value = True
                elif value == "false":
                    value = False

                classification_data[tag] = value

    # If we found XML tags, create a classification from the extracted data
    if classification_data:
        return RequestClassification.model_validate(classification_data)

    return RequestClassification(
        type=RequestType.CONTINUE,
        planning_required=False,
        relative_effort=RelativeEffortLevel.LOW,
        subject_change=False,
    )


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
    env_config: EnvConfig

    def __init__(
        self,
        executor: LocalCodeExecutor,
        credential_manager: CredentialManager,
        model_configuration: ModelConfiguration,
        config_manager: ConfigManager,
        type: OperatorType,
        agent_registry: AgentRegistry,
        current_agent: AgentData | None,
        env_config: EnvConfig,
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
            env_config (EnvConfig): The environment configuration instance.

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
        self.env_config = env_config
        # Set the delegate callback for DELEGATE actions
        self.executor.delegate_callback = self.delegate_to_agent
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

        if len(self.executor.agent_state.conversation) + 1 > max_conversation_depth:
            messages.append(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        f"... The conversation history before this message has been truncated "
                        f"to the last {max_conversation_depth} messages.  Please review the "
                        "following messages in the sequence and respond with the request "
                        "type with the required request classification XML tags."
                    ),
                )
            )

            messages.extend(self.executor.agent_state.conversation[-max_conversation_depth:])
        else:
            messages.extend(self.executor.agent_state.conversation[1:])

        messages.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=RequestClassificationUserPrompt.format(user_message=user_input),
            ),
        )

        attempt = 0
        last_error = None

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                files=[],
                execution_type=ExecutionType.CLASSIFICATION,
            )
        )

        while attempt < max_attempts:
            try:
                response = await self.executor.invoke_model(messages)
                response_content = (
                    response.content if isinstance(response.content, str) else str(response.content)
                )

                classification = process_classification_response(response_content)

                if classification.type != RequestType.CONTINUE:
                    self.executor.set_instruction_details(response_content)

                return classification

            except ValidationError as e:
                attempt += 1
                last_error = str(e)

                if attempt < max_attempts:
                    error_message = (
                        "The response you provided didn't have the required XML tags. "
                        f"Error: {last_error}. Please provide a valid XML response matching "
                        "the required classification schema."
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

    async def generate_plan(
        self, current_task_classification: RequestClassification
    ) -> AsyncGenerator[str, None]:
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
            AsyncGenerator[str, None]: The generated plan or an empty string if planning is skipped.
        """
        # Clear any existing plans from the previous invocation
        if current_task_classification.type != RequestType.CONTINUE:
            self.executor.set_current_plan("")

        system_prompt = create_system_prompt(
            tool_registry=self.executor.tool_registry,
            response_format=PlanSystemPrompt,
            agent_system_prompt=self.executor.agent_state.agent_system_prompt,
        )

        messages = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=system_prompt,
                is_system_prompt=True,
                should_cache=True,
            ),
        ]

        messages.extend(self.executor.agent_state.conversation[1:])

        messages.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=PlanUserPrompt,
            )
        )

        new_message = self.executor.add_to_code_history(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.SUCCESS,
                files=[],
                execution_type=ExecutionType.PLAN,
                is_streamable=True,
                is_complete=False,
            ),
            None,
            current_task_classification,
        )

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                id=new_message.id,
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                files=[],
                execution_type=ExecutionType.PLAN,
                is_streamable=True,
                is_complete=False,
            )
        )

        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

        async for chunk in self.executor.stream_model(messages):
            chunk_content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            new_message.message += chunk_content

            await self.executor.broadcast_message_update(
                new_message.id,
                new_message,
            )

            yield chunk_content

        # Remove think tags for reasoning models
        new_message.message = remove_think_tags(new_message.message)

        self.executor.agent_state.conversation.extend(
            [
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=new_message.message,
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

        new_message.is_complete = True

        await self.executor.update_code_history(
            new_message.id,
            new_message,
        )

        await self.executor.broadcast_message_update(
            new_message.id,
            new_message,
        )

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                id=new_message.id,
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                files=[],
                execution_type=ExecutionType.PLAN,
                is_streamable=True,
                is_complete=True,
            )
        )

        self.executor.set_current_plan(new_message.message)

        # Save the conversation history and code execution history to the agent registry
        # if the persist_conversation flag is set.
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

    async def generate_reflection(
        self, current_task_classification: RequestClassification
    ) -> AsyncGenerator[str, None]:
        """Generate a reflection for the agent.

        This method constructs a conversation with the agent to generate a reflection.
        It starts by creating a system prompt based on the available tools and the
        predefined reflection system prompt. The method then appends the current
        """
        system_prompt = create_system_prompt(
            tool_registry=self.executor.tool_registry,
            response_format="",
            agent_system_prompt=self.executor.agent_state.agent_system_prompt,
        )

        messages = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=system_prompt,
                is_system_prompt=True,
                should_cache=True,
            ),
        ]

        messages.extend(self.executor.agent_state.conversation[1:])

        messages.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=ReflectionUserPrompt,
            )
        )

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                files=[],
                execution_type=ExecutionType.REFLECTION,
            )
        )

        response_content = ""

        async for chunk in self.executor.stream_model(messages):
            chunk_content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            response_content += chunk_content
            yield chunk_content

        # Remove think tags for reasoning models
        response_content = remove_think_tags(response_content)

        # Clean the response content
        response_content = clean_plain_text_response(response_content)

        self.executor.agent_state.conversation.extend(
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
                execution_type=ExecutionType.REFLECTION,
            ),
            None,
            current_task_classification,
        )

        # Save the conversation history and code execution history to the agent registry
        # if the persist_conversation flag is set.
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

    async def generate_response(
        self, result: ResponseJsonSchema, current_task_classification: RequestClassification
    ) -> AsyncGenerator[str, None]:
        """Generate a final response for the user based on the conversation history.

        This method constructs a conversation with the agent to generate a well-structured
        final response. It adds an ephemeral message with guidelines on how to summarize
        and format the response appropriately for the user.

        Args:
            current_task_classification: Classification of the current request/task

        Returns:
            AsyncGenerator[str, str]: The generated response content

        Raises:
            Exception: If there's an error during model invocation
        """
        # Create a copy of the conversation history
        messages = list(self.executor.agent_state.conversation)

        messages.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=FinalResponseInstructions,
                ephemeral=True,
            )
        )

        new_message = self.executor.add_to_code_history(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.SUCCESS,
                files=[],
                execution_type=ExecutionType.RESPONSE,
                is_streamable=True,
                is_complete=False,
            ),
            None,
            current_task_classification,
        )

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                id=new_message.id,
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                files=[],
                execution_type=ExecutionType.RESPONSE,
                is_streamable=True,
                is_complete=False,
            )
        )

        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

        # Invoke the model to generate the response
        async for chunk in self.executor.stream_model(messages):
            chunk_content = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
            new_message.message += chunk_content

            await self.executor.broadcast_message_update(
                new_message.id,
                new_message,
            )

            yield chunk_content

        # Add content to the response if it is empty to prevent invoke errors
        # from the source.
        if not new_message.message:
            new_message.message = "No response from the agent"

        # Clean up the response
        new_message.message = remove_think_tags(new_message.message)

        # Add the response to conversation history
        self.executor.agent_state.conversation.extend(
            [
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=new_message.message,
                    should_summarize=True,
                ),
            ]
        )

        # Update the code history
        new_message.is_complete = True

        await self.executor.update_code_history(
            new_message.id,
            new_message,
        )

        await self.executor.broadcast_message_update(
            new_message.id,
            new_message,
        )

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                id=new_message.id,
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                files=[],
                execution_type=ExecutionType.RESPONSE,
                is_streamable=True,
                is_complete=True,
            )
        )

        # Persist conversation if enabled
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

    def add_task_instructions(self, request_classification: RequestClassification) -> None:
        """
        Add the task instructions as an ephemeral message to help the agent
        prioritize the information and the task at hand.
        """
        classification_str = ""

        for key, value in request_classification.model_dump().items():
            classification_str += f"<{key}>{value}</{key}>\n"

        task_instructions = TaskInstructionsPrompt.format(
            request_type=request_classification.type,
            request_classification=classification_str,
            task_instructions=get_request_type_instructions(
                RequestType(request_classification.type)
            ),
        )

        self.executor.agent_state.conversation.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=task_instructions,
                is_system_prompt=False,
                ephemeral=request_classification.type == RequestType.CONVERSATION,
                should_cache=True,
            )
        )

    async def interpret_action_response(
        self, response_content: str, max_attempts: int = 3
    ) -> ResponseJsonSchema:
        """Interpret the action response from the agent."""

        messages = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=create_action_interpreter_prompt(tool_registry=self.executor.tool_registry),
                is_system_prompt=True,
                should_cache=True,
            ),
            ConversationRecord(
                role=ConversationRole.USER,
                content=response_content,
                should_summarize=True,
            ),
        ]

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message="Cleaning up the action response",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.IN_PROGRESS,
                files=[],
                execution_type=ExecutionType.PRE_ACTION,
            )
        )

        response_json = None
        json_response_content = None
        attempts = 0

        while attempts < max_attempts and response_json is None:
            attempts += 1
            json_response = await self.executor.invoke_model(messages)

            json_response_content = (
                json_response.content
                if isinstance(json_response.content, str)
                else str(json_response.content)
            )

            try:
                if not json_response_content:
                    raise ValueError("JSON response content is empty")

                response_json = process_json_response(json_response_content)
            except Exception as e:
                logging.error(f"JSON validation error (attempt {attempts}/{max_attempts}): {e}")

                if attempts >= max_attempts:
                    break

                if isinstance(e, ValidationError):
                    error_details = "\n".join(
                        f"Error {i+1}:\n"
                        f"  Location: {' -> '.join(str(loc) for loc in err['loc'])}\n"
                        f"  Type: {err['type']}\n"
                        f"  Message: {err['msg']}"
                        for i, err in enumerate(e.errors())
                    )
                else:
                    error_details = str(e)

                messages.extend(
                    [
                        ConversationRecord(
                            role=ConversationRole.ASSISTANT,
                            content=json_response_content,
                            should_summarize=True,
                        ),
                        ConversationRecord(
                            role=ConversationRole.USER,
                            content=(
                                "Your interpretation of the action response failed JSON schema "
                                f"validation (attempt {attempts}/{max_attempts}). "
                                "Please review the validation errors and generate a "
                                "valid response:\n\n"
                                f"{error_details}\n\n"
                                "Your response must exactly match the expected JSON format: "
                                f"{JsonResponseFormatSchema}\n\n"
                                "Check for syntax errors, unescaped quotes or special characters, "
                                "invalid formatting, or other formatting issues that may cause "
                                "the error."
                            ),
                            should_summarize=True,
                        ),
                    ]
                )

        if not response_json:
            raise ValueError(
                f"Failed to generate valid JSON response after {max_attempts} "
                f'attempts.\n\nFailed response: "{json_response_content}"'
            )

        return response_json

    def process_early_response(
        self,
        response_content: str,
        response_json: ResponseJsonSchema,
        classification: RequestClassification,
    ) -> tuple[str, ProcessResponseOutput]:
        """Process an early response from the agent."""

        final_response = response_content

        # Add final response to code history
        self.executor.add_to_code_history(
            CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                message=final_response,
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.SUCCESS,
                files=[],
                execution_type=ExecutionType.RESPONSE,
                action=response_json.action,
                task_classification=classification.type,
            ),
            None,
            classification,
        )

        # Persist conversation if enabled
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

        return final_response, ProcessResponseOutput(
            status=ProcessResponseStatus.SUCCESS,
            message=final_response,
        )

    async def handle_user_input(
        self, user_input: str, user_message_id: str | None = None, attachments: List[str] = []
    ) -> tuple[ResponseJsonSchema | None, str]:
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

        Returns:
            tuple[ResponseJsonSchema | None, str]: The processed response from
                the language model, and the final response from the agent
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
                execution_type=ExecutionType.USER_INPUT,
            ),
            None,
            None,
        )

        response_json: ResponseJsonSchema | None = None
        response: BaseMessage | None = None
        final_response: str = ""

        self.executor.reset_step_counter()
        self.executor_is_processing = True

        # Classify the user's request to determine the type of task at hand and if
        # planning is required.
        async with spinner_context(
            "Interpreting your message",
            verbosity_level=self.verbosity_level,
        ):
            classification = await self.classify_request(user_input)

        if classification.subject_change:
            self.executor.set_current_plan("")
            self.executor.reset_learnings()

            # Add a breakpoint to steer the conversation inertia
            self.executor.agent_state.conversation.append(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "I would like to change the subject.  Please stop the current task "
                        "and pay attention to my new message."
                    ),
                    should_summarize=False,
                )
            )

        # Add the task instructions as an ephemeral message to help the agent
        # prioritize the information and the task at hand.
        self.add_task_instructions(classification)

        # Add the user's request after the task instructions
        self.executor.agent_state.conversation.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=user_input_with_attachments,
                files=attachments,
                should_summarize=False,
            )
        )

        # Perform planning for more complex tasks
        if classification.planning_required:
            plan = ""

            if self.verbosity_level >= VerbosityLevel.VERBOSE:
                print("\n\033[1;36m╭─ Agent Plan ──────────────────────────────────────\033[0m")

            async for chunk in self.generate_plan(classification):
                plan += chunk

                if self.verbosity_level >= VerbosityLevel.VERBOSE:
                    print(chunk, end="", flush=True)

            if self.verbosity_level >= VerbosityLevel.VERBOSE:
                print("\n\033[1;36m╰──────────────────────────────────────────────────\033[0m\n")

        elif classification.type != RequestType.CONTINUE:
            self.executor.set_current_plan("")

        while (
            not self._agent_is_done(response_json)
            and not self._agent_requires_user_input(response_json)
            and not self.executor.interrupted
        ):
            if self.model_configuration is None:
                raise ValueError("Model is not initialized")

            await self.executor.update_job_execution_state(
                CodeExecutionResult(
                    stdout="",
                    stderr="",
                    logging="",
                    formatted_print="",
                    code="",
                    message="Thinking about my next action",
                    role=ConversationRole.ASSISTANT,
                    status=ProcessResponseStatus.IN_PROGRESS,
                    files=[],
                    execution_type=ExecutionType.ACTION,
                )
            )

            if self.verbosity_level >= VerbosityLevel.VERBOSE:
                print("\n")

            async with spinner_context(
                "Formulating a response",
                verbosity_level=self.verbosity_level,
            ):
                response = await self.executor.invoke_model(self.executor.agent_state.conversation)

            response_content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )

            self.executor.append_to_history(
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=response_content,
                    should_summarize=True,
                )
            )

            response_json = await self.interpret_action_response(response_content)

            is_terminal_response = (
                response_json.action == ActionType.DONE
                or response_json.action == ActionType.ASK
                or response_json.action == ActionType.BYE
            )

            # Check if the response contains an action tag
            has_action_tag = re.search(r"<action>([^<]+)</action>", response_content) is not None

            if self.executor.step_counter == 1 and is_terminal_response and not has_action_tag:
                # This is a single response from the agent, such as from a conversation
                # or other response that doesn't require an action
                final_response, result = self.process_early_response(
                    response_content, response_json, classification
                )

                print_agent_response(
                    self.executor.step_counter, final_response, self.verbosity_level
                )
            else:
                result = await self.executor.process_response(response_json, classification)

                # Update the "Agent Heads Up Display"
                if not is_terminal_response:
                    self.executor.update_ephemeral_messages()

                    reflection = ""

                    # Only generate reflection if the last action was CODE
                    if response_json.action == ActionType.CODE:
                        if self.verbosity_level >= VerbosityLevel.VERBOSE:
                            print(
                                "\n\033[1;36m╭─ Agent Reflection "
                                "──────────────────────────────\033[0m"
                            )

                        # Reflect on the results of the last operation
                        async for chunk in self.generate_reflection(classification):
                            reflection += chunk
                            if self.verbosity_level >= VerbosityLevel.VERBOSE:
                                print(chunk, end="", flush=True)

                        if self.verbosity_level >= VerbosityLevel.VERBOSE:
                            print(
                                "\n\033[1;36m╰────────────────────────────────"
                                "──────────────────\033[0m\n"
                            )

                else:
                    final_response = ""

                    if self.verbosity_level >= VerbosityLevel.VERBOSE:
                        print(
                            f"\n\033[1;36m╭─ Agent Response (Step {self.executor.step_counter}) "
                            "──────────────────────────\033[0m"
                        )

                    async for chunk in self.generate_response(response_json, classification):
                        final_response += chunk

                        if self.verbosity_level >= VerbosityLevel.VERBOSE:
                            print(chunk, end="", flush=True)

                    if self.verbosity_level >= VerbosityLevel.VERBOSE:
                        print(
                            "\n\033[1;"
                            "36m╰══════════════════════════════════════════════════╯\033[0m"
                        )

            # Auto-save on each step if enabled
            if self.auto_save_conversation:
                try:
                    self.handle_autosave(
                        self.agent_registry.config_dir,
                        self.executor.agent_state.conversation,
                        self.executor.agent_state.execution_history,
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

        return response_json, final_response

    async def delegate_to_agent(self, agent_name: str, message: str):
        """
        Handle delegation to another agent for the DELEGATE action.

        Args:
            agent_name (str): The name of the agent to delegate to.
            message (str): The message to send to the delegated agent.

        Returns:
            ProcessResponseOutput: The result of the delegated agent's response.
        """
        # Import locally to avoid circular import
        from local_operator.bootstrap import initialize_operator

        # Find the agent by name in the registry
        agent = None
        for a in self.agent_registry.list_agents():
            if a.name == agent_name:
                agent = a
                break
        if agent is None:
            return ProcessResponseOutput(
                status=ProcessResponseStatus.ERROR,
                message=f"Agent '{agent_name}' not found for delegation.",
            )

        # Create a new Operator for the target agent
        try:
            delegated_operator = initialize_operator(
                operator_type=self.type,
                config_manager=self.config_manager,
                credential_manager=self.credential_manager,
                agent_registry=self.agent_registry,
                env_config=self.env_config,
                request_hosting=agent.hosting,
                request_model=agent.model,
                current_agent=agent,
                persist_conversation=False,
                auto_save_conversation=False,
                verbosity_level=VerbosityLevel.QUIET,  # Silence delegation logging
            )
        except Exception as e:
            return ProcessResponseOutput(
                status=ProcessResponseStatus.ERROR,
                message=f"Failed to initialize delegated agent '{agent_name}': {e}",
            )

        # Send the message to the delegated agent and get the response
        try:
            _, final_response = await delegated_operator.handle_user_input(message)
            return ProcessResponseOutput(
                status=ProcessResponseStatus.SUCCESS,
                message=final_response,
            )
        except Exception as e:
            return ProcessResponseOutput(
                status=ProcessResponseStatus.ERROR,
                message=f"Delegation to agent '{agent_name}' failed: {e}",
            )

    def print_conversation_history(self) -> None:
        """Print the conversation history for debugging."""
        total_tokens = self.executor.get_invoke_token_count(self.executor.agent_state.conversation)

        print("\n\033[1;35m╭─ Debug: Conversation History ───────────────────────\033[0m")
        print(f"\033[1;35m│ Message tokens: {total_tokens}                       \033[0m")
        print(f"\033[1;35m│ Session tokens: {self.executor.get_session_token_usage()}\033[0m")
        for i, entry in enumerate(self.executor.agent_state.conversation, 1):
            role = entry.role
            content = entry.content
            print(f"\033[1;35m│ {i}. {role.value.capitalize()}:\033[0m")
            for line in content.split("\n"):
                print(f"\033[1;35m│   {line}\033[0m")
        print("\033[1;35m╰──────────────────────────────────────────────────\033[0m\n")

    async def execute_single_command(
        self, command: str
    ) -> tuple[ResponseJsonSchema | None, str | None]:
        """Execute a single command in non-interactive mode.

        This method is used for one-off command execution rather than interactive chat.
        It initializes a fresh conversation history (if not already initialized),
        processes the command through the language model, and returns the result.

        Args:
            command (str): The command/instruction to execute

        Returns:
            tuple[ResponseJsonSchema | None, str]: The processed response from
                the language model, and the final response from the agent
        """
        try:
            self.executor.initialize_conversation_history()
        except ValueError:
            # Conversation history already initialized
            pass

        return await self.handle_user_input(command)

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

            response_json, final_response = await self.handle_user_input(user_input)

            # Check if the last line of the response contains "[BYE]" to exit
            if self._agent_should_exit(response_json):
                break

            # Print the last assistant message if the agent is asking for user input
            if (
                response_json
                and self._agent_requires_user_input(response_json)
                and self.verbosity_level >= VerbosityLevel.QUIET
            ):
                print("\n\033[1;36m╭─ Agent Question Requires Input ────────────────\033[0m")
                print(f"\033[1;36m│\033[0m {final_response}")
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
