import logging
import os
import platform
import re
import signal
import uuid
from enum import Enum
from pathlib import Path
from typing import List

from pydantic import ValidationError

from local_operator.agents import AgentData, AgentRegistry
from local_operator.config import ConfigManager
from local_operator.console import VerbosityLevel, print_cli_banner, spinner_context
from local_operator.credentials import CredentialManager
from local_operator.env import EnvConfig
from local_operator.executor import CodeExecutionResult, LocalCodeExecutor
from local_operator.helpers import parse_agent_action_xml
from local_operator.model.configure import ModelConfiguration
from local_operator.notebook import save_code_history_to_notebook
from local_operator.prompts import (
    PlanUserPrompt,
    RequestClassificationSystemPrompt,
    RequestClassificationUserPrompt,
    RequestType,
    TaskInstructionsPrompt,
    apply_attachments_to_prompt,
    get_request_type_instructions,
)
from local_operator.stream import stream_action_buffer
from local_operator.types import (
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
                        f"<system>The conversation history before this message has been truncated "
                        f"to the last {max_conversation_depth} messages.  Please review the "
                        "following messages in the sequence and respond with the request "
                        "type with the required request classification XML tags.</system>"
                    ),
                    should_summarize=False,
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
                        "<system>The response you provided didn't have the required XML tags. "
                        f"Error: {last_error}. Please provide a valid XML response matching "
                        "the required classification schema.</system>"
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
            AsyncGenerator[str, None]: The generated plan or an empty string if planning is skipped.
        """
        # Clear any existing plans from the previous invocation
        if current_task_classification.type != RequestType.CONTINUE:
            self.executor.set_current_plan("")

        self.executor.agent_state.conversation.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=f"<system>{PlanUserPrompt}</system>",
                should_summarize=False,
            )
        )

        _, response_content, _ = await self.invoke_and_process_response(
            self.executor.agent_state.conversation,
            current_task_classification,
        )

        self.executor.set_current_plan(response_content)

        self.executor.agent_state.conversation.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=(
                    "<system>Please proceed according to your plan. "
                    "Choose appropriate actions "
                    "and follow the action XML schema if you need to take "
                    "actions.  If you do not need to take any actions, do not include "
                    "an action in your response.</system>"
                ),
                should_summarize=False,
            ),
        )

        # Save the conversation history and code execution history to the agent registry
        # if the persist_conversation flag is set.
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

        return response_content

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
                content=f"<system>{task_instructions}</system>",
                is_system_prompt=False,
                ephemeral=request_classification.type == RequestType.CONVERSATION,
                should_cache=True,
            )
        )

    def interpret_action_response(self, response_content: str) -> ResponseJsonSchema:
        """Interpret the action response from the agent using a custom XML parser."""
        response_json_dict = None
        parsed_response_schema = None

        try:
            # Step 1: Parse the XML-like string into a dictionary
            response_json_dict = parse_agent_action_xml(response_content)

            if not response_json_dict or not response_json_dict.get("action"):
                # If action is missing, it's a critical parsing failure or invalid XML.
                # The schema validation below would also catch a missing 'action',
                # but an early check can be useful.
                raise ValueError(
                    "Failed to parse critical 'action' tag from response or response is empty."
                )

            # Step 2: Validate the dictionary against the Pydantic schema
            # This will raise ValidationError if the dict doesn't match the schema
            parsed_response_schema = ResponseJsonSchema.model_validate(response_json_dict)

        except ValidationError as ve:
            logging.error(f"Pydantic validation error for parsed XML: {ve}")
            # Potentially log response_json_dict here for debugging if needed
            error_details = "\n".join(
                f"Error {i+1}:\n"
                f"  Location: {' -> '.join(str(loc) for loc in err['loc'])}\n"
                f"  Type: {err['type']}\n"
                f"  Message: {err['msg']}"
                for i, err in enumerate(ve.errors())
            )
            raise ValueError(
                "Parsed agent response failed schema validation.\n"
                f'Original response content: "{response_content[:500]}..."\n'  # Log snippet
                "Parsed dictionary (first level keys): "
                f"{list(response_json_dict.keys()) if response_json_dict else 'None'}\n"
                f"Validation Errors:\n{error_details}"
            ) from ve
        except Exception as e:
            # Catch other potential errors during parsing or validation
            logging.error(f"Error interpreting action response: {e}")
            raise ValueError(
                "Failed to interpret action response due to an unexpected error.\n"
                f'Original response content: "{response_content[:500]}..."\n'
                f"Error: {e}"
            ) from e

        if not parsed_response_schema:
            # This case should ideally be caught by the exceptions above,
            # but as a fallback.
            raise ValueError(
                "Failed to generate a valid response schema from the agent's output.\n"
                f'Original response content: "{response_content[:500]}..."'
            )

        return parsed_response_schema

    def process_text_response(
        self,
        response_content: str,
    ) -> tuple[str, ProcessResponseOutput]:
        """Process an early response from the agent."""

        final_response = response_content

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

    def _has_action_tag(self, response_content: str) -> bool:
        """Check if the response content contains an action tag."""
        return re.search(r"<action>([^<]+)</action>", response_content) is not None

    async def invoke_and_process_response(
        self,
        messages: list[ConversationRecord],
        classification: RequestClassification,
    ) -> tuple[ResponseJsonSchema | None, str, ProcessResponseOutput]:
        """Invoke the model and process the response with streaming support."""

        # Initialize streaming state
        accumulated_text = ""
        finished = False

        # Create a new message in code history for streaming
        new_message = self.executor.add_to_code_history(
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
                execution_type=ExecutionType.ACTION,
                is_streamable=True,
                is_complete=False,
            ),
            None,
            classification,
        )

        await self.executor.update_job_execution_state(
            CodeExecutionResult(
                id=new_message.id,
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

        # Persist conversation if enabled
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

        # Retry loop for parsing the agent's action response
        attempts = 0
        max_attempts = 3
        response_json = None
        final_response_content = ""
        result = new_message.model_copy()

        while attempts < max_attempts:
            attempts += 1

            # Reset streaming state for retries
            if attempts > 1:
                accumulated_text = ""
                await self.executor.update_code_history(
                    new_message.id,
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
                        execution_type=ExecutionType.ACTION,
                        is_streamable=True,
                        is_complete=False,
                    ),
                )
                finished = False

                # Persist conversation if enabled
                if self.persist_agent_conversation and self.agent_registry and self.current_agent:
                    self.agent_registry.update_agent_state(
                        agent_id=self.current_agent.id,
                        agent_state=self.executor.agent_state,
                    )

            try:
                # Stream the model response
                if self.verbosity_level >= VerbosityLevel.VERBOSE:
                    print(
                        "\n\033[1;36m╭─ Agent Response ──────────────────────────────────────\033[0m"  # noqa: E501
                    )

                action_response_started = False

                async for chunk in self.executor.stream_model(messages):
                    chunk_content = (
                        chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                    )

                    # Accumulate the text
                    accumulated_text += chunk_content

                    # Process the accumulated text through the stream buffer
                    finished, result = stream_action_buffer(accumulated_text)

                    if self.verbosity_level >= VerbosityLevel.VERBOSE:
                        # Check if we're in an action response by looking for action_response tags
                        in_action_response = (
                            "<action_response>" in accumulated_text
                            and "</action_response>" not in accumulated_text
                        )

                        if not action_response_started and in_action_response:
                            action_response_started = True

                            split_text = accumulated_text.split("<action_response>")
                            pre_action_response_text = split_text[0].rstrip()
                            accumulated_text = split_text[1].lstrip()

                            print(pre_action_response_text)
                            print(
                                "\033[1;36m╰──────────────────────────────────────────────────\033[0m\n"  # noqa: E501
                            )
                            print(
                                "\n\n\033[1;36m╭─ Agent Action ──────────────────────────────\033[0m"  # noqa: E501
                            )
                            print(accumulated_text)
                        else:
                            print(chunk_content, end="", flush=True)

                    # Update the message in code history
                    new_message.message = result.message
                    new_message.content = result.content or ""
                    new_message.code = result.code or ""
                    new_message.replacements = result.replacements or ""
                    new_message.files = result.files if result.files else new_message.files
                    new_message.learnings = result.learnings or ""
                    new_message.agent = result.agent or ""
                    new_message.action = result.action
                    new_message.file_path = result.file_path
                    new_message.thinking = result.thinking or ""

                    # Broadcast the update
                    await self.executor.broadcast_message_update(
                        new_message.id,
                        new_message,
                    )

                    # If we've finished processing the action_response, break
                    if finished:
                        break

                if self.verbosity_level >= VerbosityLevel.VERBOSE:
                    print(
                        "\n\033[1;36m╰──────────────────────────────────────────────────\033[0m\n"
                    )

                # Get the final response content
                final_response_content = accumulated_text

                # Check if there is an action request from the agent
                if not self._has_action_tag(final_response_content) and not result.action:
                    self.executor.append_to_history(
                        ConversationRecord(
                            role=ConversationRole.ASSISTANT,
                            content=final_response_content,
                            should_summarize=True,
                        )
                    )

                    # Update final state
                    new_message.is_complete = True
                    new_message.status = ProcessResponseStatus.SUCCESS

                    await self.executor.update_code_history(
                        new_message.id,
                        new_message,
                    )

                    await self.executor.broadcast_message_update(
                        new_message.id,
                        new_message,
                    )

                    return (
                        None,
                        final_response_content,
                        ProcessResponseOutput(
                            status=ProcessResponseStatus.SUCCESS,
                            message=final_response_content,
                        ),
                    )

                # Try to interpret the action response
                self.executor.append_to_history(
                    ConversationRecord(
                        role=ConversationRole.ASSISTANT,
                        content=final_response_content,
                        should_summarize=True,
                    )
                )

                # Create ResponseJsonSchema from streamed result
                response_json = self.interpret_action_response(final_response_content)

                break  # Successfully parsed

            except Exception as e:
                logging.error(
                    "Failed to interpret action response "
                    f"(attempt {attempts}/{max_attempts}): {e}"
                )
                if attempts >= max_attempts:
                    # Persist the last failing response before raising
                    self.executor.append_to_history(
                        ConversationRecord(
                            role=ConversationRole.ASSISTANT,
                            content=final_response_content,
                            should_summarize=True,
                        )
                    )

                    # Update final state as error
                    new_message.is_complete = True
                    new_message.status = ProcessResponseStatus.ERROR
                    new_message.stderr = str(e)

                    await self.executor.update_code_history(
                        new_message.id,
                        new_message,
                    )

                    await self.executor.broadcast_message_update(
                        new_message.id,
                        new_message,
                    )

                    if (
                        self.persist_agent_conversation
                        and self.agent_registry
                        and self.current_agent
                    ):
                        self.agent_registry.update_agent_state(
                            agent_id=self.current_agent.id,
                            agent_state=self.executor.agent_state,
                        )

                    raise Exception(
                        f"Failed to interpret action response after {max_attempts} attempts. "
                        f"Last error: {e}"
                    ) from e

                # Prepare to re-prompt the agent
                error_message_for_agent = (
                    f"<system>Your previous action response was not parsable or failed validation. "
                    f"Error: {str(e)}\n\nPlease try again, ensuring your response is valid XML "
                    "matching the required action schema. "
                    f"Review the schema in the system prompt if needed. "
                    "The problematic response started with: "
                    f"{final_response_content[:200]}..."
                    "</system>"
                )
                self.executor.append_to_history(
                    ConversationRecord(
                        role=ConversationRole.USER,
                        content=error_message_for_agent,
                        should_summarize=False,
                    )
                )

                # Update messages for retry
                messages = self.executor.agent_state.conversation

        if response_json is None:
            raise ValueError("Failed to generate a valid response after repeated retries.")

        # Process the response
        result_output, execution_result = await self.executor.process_response(
            response_json, classification
        )

        if execution_result:
            # Apply the execution result to the new message
            execution_result.id = new_message.id
            execution_result.is_complete = True
            execution_result.timestamp = new_message.timestamp
            execution_result.replacements = new_message.replacements

            if not execution_result.message:
                execution_result.message = new_message.message

            new_message = execution_result

        await self.executor.update_code_history(
            new_message.id,
            new_message,
        )

        await self.executor.broadcast_message_update(
            new_message.id,
            new_message,
        )

        # Persist conversation if enabled
        if self.persist_agent_conversation and self.agent_registry and self.current_agent:
            self.agent_registry.update_agent_state(
                agent_id=self.current_agent.id,
                agent_state=self.executor.agent_state,
            )

        return response_json, final_response_content, result_output

    async def handle_user_input(
        self,
        user_input: str,
        user_message_id: str | None = None,
        attachments: List[str] = [],
        additional_instructions: str | None = None,
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
            user_message_id: The ID of the user message, will be generated
            if not provided
            attachments: A list of attachments to include in the user input
            additional_instructions: Additional instructions to include in the
            user input.  These will not be shown in the execution history but
            will be used as context in the conversation history.

        Returns:
            tuple[ResponseJsonSchema | None, str]: The processed response from
                the language model, and the final response from the agent
        """

        self.executor.update_ephemeral_messages()

        user_input_with_attachments = apply_attachments_to_prompt(user_input, attachments)

        if additional_instructions:
            user_input_with_attachments += (
                f"\n\n## Additional Instructions\n\n{additional_instructions}"
            )

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
                        "<system>I would like to change the subject.  Please stop the current task "
                        "and pay attention to my new message.  Don't acknowledge "
                        "this message directly.</system>"
                    ),
                    should_summarize=False,
                )
            )

        # Add the task instructions as an ephemeral message to help the agent
        # prioritize the information and the task at hand.
        if classification.type != RequestType.CONTINUE:
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
            await self.generate_plan(classification)
        elif classification.type != RequestType.CONTINUE:
            self.executor.set_current_plan("")

        while (
            not self._agent_is_done(response_json)
            and not final_response
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

            # Process and handle any actions from the agent
            response_json, response_content, result = await self.invoke_and_process_response(
                self.executor.agent_state.conversation,
                classification,
            )

            if response_json is None:
                # If there is no action request, process the response as a text response
                final_response, result = self.process_text_response(response_content)
            else:
                # Update the "Agent Heads Up Display"
                self.executor.update_ephemeral_messages()

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
            # Unpack the tuple returned by initialize_operator
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

    async def process_message_for_agent(
        self,
        agent_id: uuid.UUID,
        message_content: str,
        schedule_id: uuid.UUID | None = None,
    ) -> None:
        """
        Process a message for a specific agent, typically for scheduled tasks.
        This method sets up the context for the target agent and runs the message.
        """
        try:
            logging.info(f"Processing scheduled task for agent {agent_id}")

            # Add a system message to indicate this is a scheduled task
            scheduled_task_indicator = (
                f"This is an automated task triggered by schedule ID: {schedule_id}. "
                f"Execute the following prompt: {message_content}"
            )

            # Use handle_user_input to process the task
            # We might need a more direct way to inject and process non-interactive tasks
            # For now, this simulates user input.
            await self.handle_user_input(scheduled_task_indicator)

            logging.info(f"Finished processing scheduled task for agent {agent_id}")

        except Exception as e:
            logging.error(f"Error processing message for agent {agent_id}: {str(e)}")
