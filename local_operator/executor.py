import asyncio
import base64
import builtins
import difflib
import inspect
import io
import logging
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from enum import Enum
from logging import StreamHandler
from multiprocessing import Queue
from pathlib import Path
from traceback import format_exception
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Tuple

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tiktoken import encoding_for_model

from local_operator.agents import AgentData, AgentRegistry
from local_operator.console import (
    ExecutionSection,
    VerbosityLevel,
    condense_logging,
    format_error_output,
    format_success_output,
    log_action_error,
    log_retry_error,
    print_execution_section,
    print_task_interrupted,
    spinner_context,
)
from local_operator.helpers import clean_plain_text_response, process_json_response
from local_operator.model.configure import ModelConfiguration, calculate_cost
from local_operator.prompts import (
    AgentHeadsUpDisplayPrompt,
    MessageSummarySystemPrompt,
    SafetyCheckConversationPrompt,
    SafetyCheckSystemPrompt,
    SafetyCheckUserPrompt,
    create_system_prompt,
)
from local_operator.tools.general import ToolRegistry, list_working_directory
from local_operator.types import (
    ActionType,
    AgentState,
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    ExecutionType,
    ProcessResponseOutput,
    ProcessResponseStatus,
    RequestClassification,
    ResponseJsonSchema,
)

CHARS_PER_TOKEN = 4
MAX_FILE_READ_TOKENS = 50000
MAX_FILE_READ_SIZE_BYTES = CHARS_PER_TOKEN * MAX_FILE_READ_TOKENS
"""The maximum file size to read in bytes.

This is used to prevent reading large files into context, which can cause
context overflow errors for LLM APIs.
"""


class ExecutorInitError(Exception):
    """Raised when the executor fails to initialize properly."""

    def __init__(self, message: str = "Failed to initialize executor"):
        self.message = message
        super().__init__(self.message)


class ConfirmSafetyResult(Enum):
    """Result of the safety check."""

    SAFE = "safe"  # Code is safe, no further action needed
    UNSAFE = "unsafe"  # Code is unsafe, execution should be cancelled
    OVERRIDE = "override"  # Code is unsafe, but a user security override allows it
    CONVERSATION_CONFIRM = (
        "conversation_confirm"  # Safety needs to be confirmed in further conversation with the user
    )


def get_confirm_safety_result(response_content: str) -> ConfirmSafetyResult:
    """Get the result of the safety check from the response content."""
    if not response_content:
        return ConfirmSafetyResult.SAFE

    content_lower = response_content.lower()
    if "[override]" in content_lower:
        return ConfirmSafetyResult.OVERRIDE
    elif "[unsafe]" in content_lower:
        return ConfirmSafetyResult.UNSAFE
    else:
        return ConfirmSafetyResult.SAFE


def get_context_vars_str(context_vars: Dict[str, Any]) -> str:
    """Get the context variables as a string, limiting each value to 1000 lines.

    This function converts a dictionary of context variables into a string
    representation, limiting the output to a maximum of 1000 lines per value
    to prevent excessive output. It also ignores built-in variables and other
    common uninteresting variables.

    Args:
        context_vars (Dict[str, Any]): A dictionary of context variables.

    Returns:
        str: A string representation of the context variables, with each value
              limited to a maximum of 1000 lines.
    """
    context_vars_str = ""
    ignored_keys = {"__builtins__", "__doc__", "__file__", "__name__", "__package__"}

    for key, value in context_vars.items():
        if key in ignored_keys:
            continue

        value_str = str(value)
        formatted_value_str = value_str

        if inspect.iscoroutine(value):
            formatted_value_str = f"<Coroutine object {value.__name__} (not awaited)>"
        elif inspect.iscoroutinefunction(value):
            # For async function definitions, format them like other callables
            try:
                doc = value.__doc__ or "No description available"
                doc = doc.split("\n")[0].strip()
                sig = inspect.signature(value)
                args = [
                    f"{p.name}: {p.annotation.__name__ if hasattr(p.annotation, '__name__') else str(p.annotation)}"  # noqa: E501
                    for p in sig.parameters.values()
                ]
                return_type = (
                    sig.return_annotation.__name__
                    if hasattr(sig.return_annotation, "__name__")
                    else str(sig.return_annotation)
                )
                formatted_value_str = f"async def {key}({', '.join(args)}) -> {return_type}: {doc}"
            except ValueError:
                formatted_value_str = f"<AsyncFunction {key}>"
        elif callable(value):
            try:
                doc = value.__doc__ or "No description available"
                doc = doc.split("\n")[0].strip()
                sig = inspect.signature(value)
                args = [
                    f"{p.name}: {p.annotation.__name__ if hasattr(p.annotation, '__name__') else str(p.annotation)}"  # noqa: E501
                    for p in sig.parameters.values()
                ]
                return_type = (
                    sig.return_annotation.__name__
                    if hasattr(sig.return_annotation, "__name__")
                    else str(sig.return_annotation)
                )
                # No async_prefix here as iscoroutinefunction is handled above
                formatted_value_str = f"def {key}({', '.join(args)}) -> {return_type}: {doc}"
            except ValueError:
                formatted_value_str = f"<Function {key}>"

        if len(formatted_value_str) > 10000:
            formatted_value_str = (
                f"{formatted_value_str[:10000]} ... (truncated due to length limits)"
            )

        entry = f"{key}: {formatted_value_str}\n"
        context_vars_str += entry

    return context_vars_str


def annotate_code(code: str, error_line: int | None = None) -> str | None:
    """Annotate the code with line numbers and content lengths.

    This function takes a string of code, splits it into lines, and then
    prepends each line with its line number and character length.

    Args:
        code (str): The code to annotate.
        error_line (int | None): The line number where the error occurred, if any.

    Returns:
        str: The annotated code, with each line prepended by its line number
             and character length, or None if the input code is empty.

    Example:
        >>> code = "def foo():\\n    print('bar')\\n"
        >>> annotate_code(code)
        '   1 |     9 | def foo():\\n   2 |    15 |     print('bar')\\n'
    """
    if not code:
        return None

    lines = code.splitlines(keepends=True)
    annotated_code = ""

    for i, line in enumerate(lines):
        line_number = i + 1
        line_length = len(line) - (len(line.rstrip("\r\n")) - len(line))

        if error_line is not None:
            if line_number == error_line:
                error_indicator = " err >> | "
            else:
                error_indicator = "        | "
        else:
            error_indicator = ""

        annotated_code += f"{error_indicator}{line_number:>4} | {line_length:>4} | {line}"

    return annotated_code


class ExecutorTokenMetrics(BaseModel):
    """Tracks token usage and cost metrics for model executions.

    Attributes:
        total_prompt_tokens (int): Total number of tokens used in prompts across all invocations.
        total_completion_tokens (int): Total number of tokens generated in completions.
        total_cost (float): Total monetary cost of all model invocations.
    """

    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost: float = 0.0


class CodeExecutionError(Exception):
    """
    Exception raised when code execution fails.

    Attributes:
        message (str): The error message.
        code (str): The code that caused the error.
    """

    def __init__(self, message: str, code: str):
        """
        Initializes a new instance of the CodeExecutionError class.

        Args:
            message (str): The error message.
            code (str): The code that caused the error.
        """
        self.message = message
        self.code = code
        super().__init__(self.message)

    def agent_info_str(self) -> str:
        """
        Returns a string representation of the error, including annotated code for debugging.

        This method extracts the line number from the traceback where the error occurred within
        the agent-generated code, annotates the code with this information, and formats the
        error message and annotated code into a string suitable for displaying to the agent.

        Returns:
            str: A formatted string containing the error message and annotated code,
                 structured with XML-like tags for easy parsing. Includes:
                 - The error message itself.
                 - A legend explaining the annotation format.
                 - The annotated code block, highlighting the error location.
        """
        lineno: int | None = None
        tb = self.__traceback__
        while tb is not None:
            if tb.tb_frame.f_code.co_filename == "<agent_generated_code>":
                lineno = tb.tb_lineno
                break
            tb = tb.tb_next

        error_string = self.message
        annotated_code = annotate_code(self.code, error_line=lineno)
        traceback_str = "".join(format_exception(self))

        error_info = (
            "<error_message>\n"
            + f"{error_string}\n"
            + "</error_message>\n"
            + "<error_traceback>\n"
            + f"{traceback_str}\n"
            + "</error_traceback>\n"
            + "<agent_generated_code>\n"
            + "<legend>\n"
            + "Error Indicator |Line | Length | Content\n"
            + "</legend>\n"
            + "<code_block>\n"
            + f"{annotated_code}\n"
            + "</code_block>\n"
            + "</agent_generated_code>\n"
        )

        return error_info


class LocalCodeExecutor:
    """A class to handle local Python code execution with safety checks and context management.

    This executor manages the execution of Python code blocks, maintains conversation history,
    tracks execution context, and handles interactions with language models. It provides
    safety checks, error handling, and context persistence between executions.

    Attributes:
        context (Dict[str, Any]): A dictionary to maintain execution context between code blocks.
        model_configuration (ModelConfiguration): Configuration for the language model used.
        step_counter (int): A counter to track the current step in sequential execution.
        max_conversation_history (int): The maximum number of messages to keep in the
            conversation history. This does not include the system prompt.
        detail_conversation_length (int): The number of messages to keep in full detail in the
            conversation history. Every step before this, except the system prompt, will be
            summarized.
        interrupted (bool): Flag indicating if execution was interrupted.
        can_prompt_user (bool): Informs the executor about whether the end user has access to the
            terminal (True), or is consuming the service from some remote source where they
            cannot respond via the terminal (False).
        token_metrics (ExecutorTokenMetrics): Tracks token usage and cost metrics for model calls.
        agent (AgentData | None): The agent data for the current conversation.
        agent_registry (AgentRegistry | None): The agent registry for the current conversation.
        tool_registry (ToolRegistry | None): The tool registry for the current conversation.
        persist_conversation (bool): Whether to persist the conversation history and code
            execution history to the agent registry on each step.
        agent_state (AgentState): Contains the agent's state including conversation history,
            execution history, learnings, current plan, and instruction details.
        status_queue (Queue | None): A queue for status updates if this is part
            of a running job for a server operator.
        delegate_callback (Optional[Callable[[str, str], Awaitable[ProcessResponseOutput]]]):
            Callback for handling DELEGATE actions, set by the Operator.
    """

    context: Dict[str, Any]
    model_configuration: ModelConfiguration
    step_counter: int
    max_conversation_history: int
    detail_conversation_length: int
    interrupted: bool
    can_prompt_user: bool
    token_metrics: ExecutorTokenMetrics
    agent: AgentData | None
    agent_registry: AgentRegistry | None
    tool_registry: ToolRegistry | None
    persist_conversation: bool
    agent_state: AgentState
    status_queue: Optional[Queue] = None  # type: ignore
    delegate_callback: Optional[Callable[[str, str], Awaitable[ProcessResponseOutput]]] = None

    def __init__(
        self,
        model_configuration: ModelConfiguration,
        max_conversation_history: int = 100,
        detail_conversation_length: int = 10,
        can_prompt_user: bool = True,
        agent: AgentData | None = None,
        agent_state: AgentState = AgentState(
            version="",
            conversation=[],
            execution_history=[],
            learnings=[],
            current_plan=None,
            instruction_details=None,
            agent_system_prompt=None,
        ),
        agent_registry: AgentRegistry | None = None,
        max_learnings_history: int = 50,
        verbosity_level: VerbosityLevel = VerbosityLevel.VERBOSE,
        persist_conversation: bool = False,
        job_id: Optional[str] = None,
    ):
        """Initialize the LocalCodeExecutor with a language model.

        Args:
            model_configuration: Configuration for the language model to use
            max_conversation_history: Maximum number of messages to keep in the
                conversation history, excluding the system prompt
            detail_conversation_length: Number of recent messages to keep in full detail, with
                earlier messages being summarized (except system prompt). Set to -1 to keep all
                messages in full detail.
            can_prompt_user: Whether the end user has terminal access (True) or is using a remote
                interface without terminal access (False)
            agent: Optional agent data for the current conversation
            agent_state: Optional initial state for the agent, including conversation history,
                execution history, learnings, current plan, and instruction details
            agent_registry: Optional registry for managing agent data and state persistence
            max_learnings_history: Maximum number of learnings to retain in history
            verbosity_level: Controls the level of detail in executor output
            persist_conversation: Whether to automatically persist conversation and execution
                history to the agent registry after each step
            job_id: Optional identifier for the current job being processed
        """
        self.context = {"__builtins__": builtins}
        self.model_configuration = model_configuration
        self.agent_state = agent_state
        self.max_conversation_history = max_conversation_history
        self.detail_conversation_length = detail_conversation_length
        self.can_prompt_user = can_prompt_user
        self.token_metrics = ExecutorTokenMetrics()
        self.agent = agent
        self.interrupted = False
        self.max_learnings_history = max_learnings_history
        self.verbosity_level = verbosity_level
        self.agent_registry = agent_registry
        self.persist_conversation = persist_conversation
        self.job_id = job_id

        # Load agent context if agent and agent_registry are provided
        if self.agent and self.agent_registry:
            try:
                agent_context = self.agent_registry.load_agent_context(self.agent.id)
                if agent_context is not None:
                    self.context = agent_context
                    # Ensure builtins are always present in the context
                    self.context["__builtins__"] = builtins
            except Exception as e:
                print(f"Failed to load agent context: {str(e)}")

        self.reset_step_counter()

    def reset_step_counter(self):
        """Reset the step counter."""
        self.step_counter = 1

    def append_to_history(
        self,
        new_record: ConversationRecord,
    ) -> None:
        """Append a message to conversation history and maintain length limit.

        This method adds a new conversation record to the history and ensures the total history
        length stays within the configured maximum by calling _limit_conversation_history().

        Args:
            new_record (ConversationRecord): The conversation record to append, containing:
                role: The role of the message sender (user/assistant/system)
                content: The message content
                should_summarize: Whether to summarize this message in the future
                ephemeral: Whether this message is temporary/ephemeral

        The method updates self.agent_state.conversation in-place.
        """
        if not new_record.timestamp:
            new_record.timestamp = datetime.now()

        self.agent_state.conversation.append(new_record)
        self._limit_conversation_history()

    async def _summarize_old_steps(self, min_token_threshold: int = 1000) -> None:
        """
        Summarize old conversation steps beyond the detail conversation length.

        This method summarizes messages in the conversation history that are beyond the
        `detail_conversation_length` limit and have not been summarized yet. It ensures that
        only messages that need summarization are processed, and updates their content with
        a concise summary.

        Returns:
            None

        Raises:
            ValueError: If the conversation record is not of the expected type.
        """
        if len(self.agent_state.conversation) <= 1:  # Just system prompt or empty
            return

        if self.detail_conversation_length == -1:
            return

        # Calculate which messages need summarizing
        history_to_summarize = self.agent_state.conversation[1 : -self.detail_conversation_length]

        for msg in history_to_summarize:
            # Skip messages that are already sufficiently concise/summarized
            if not msg.should_summarize or msg.summarized:
                continue

            summary = await self._summarize_conversation_step(msg, min_token_threshold)
            msg.content = summary
            msg.summarized = True

    def get_model_name(self) -> str:
        """Get the name of the model being used.

        Returns:
            str: The lowercase name of the model. For OpenAI models, returns the model_name
                attribute. For other models, returns the string representation of the model.
        """
        if isinstance(self.model_configuration.instance, ChatOpenAI):
            return self.model_configuration.instance.model_name.lower()
        else:
            return str(self.model_configuration.instance.model).lower()

    def get_token_metrics(self) -> ExecutorTokenMetrics:
        """Get the total token metrics for the current session."""
        return self.token_metrics

    def get_invoke_token_count(self, messages: List[ConversationRecord]) -> int:
        """Calculate the total number of tokens in a list of conversation messages.

        Uses the appropriate tokenizer for the current model to count tokens. Falls back
        to the GPT-4 tokenizer if the model-specific tokenizer is not available.

        Args:
            messages: List of conversation message dictionaries, each containing a "content" key
                with the message text.

        Returns:
            int: Total number of tokens across all messages.
        """
        tokenizer = None
        try:
            tokenizer = encoding_for_model(self.get_model_name())
        except Exception:
            tokenizer = encoding_for_model("gpt-4o")

        total_tokens = 0
        for entry in messages:
            if entry.content:  # Ensure content is not None or empty before encoding
                total_tokens += len(tokenizer.encode(entry.content))
        return total_tokens

    def get_session_token_usage(self) -> int:
        """Get the total token count for the current session."""
        return self.token_metrics.total_prompt_tokens + self.token_metrics.total_completion_tokens

    def initialize_conversation_history(
        self, new_conversation_history: List[ConversationRecord] = [], overwrite: bool = False
    ) -> None:
        """Initialize the conversation history with a system prompt.

        The system prompt is always included as the first message in the history.
        If an existing conversation history is provided, it is appended to the
        system prompt, excluding the first message of the provided history (assumed
        to be a redundant system prompt).

        Args:
            new_conversation_history (List[ConversationRecord], optional):
                A list of existing conversation records to initialize the history with.
                Defaults to an empty list.
            overwrite (bool, optional): Whether to overwrite the existing conversation history.
                Defaults to False.
        """
        if overwrite:
            self.agent_state.conversation = []

        if len(self.agent_state.conversation) != 0:
            raise ValueError("Conversation history already initialized")

        system_prompt = create_system_prompt(
            tool_registry=self.tool_registry,
            agent_system_prompt=self.agent_state.agent_system_prompt,
            agent=self.agent,
        )

        history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=system_prompt,
                is_system_prompt=True,
            ),
        ]

        if len(new_conversation_history) == 0:
            self.agent_state.conversation = history
        else:
            # Remove the system prompt from the loaded history if it exists
            filtered_history = [
                record for record in new_conversation_history if not record.is_system_prompt
            ]
            self.agent_state.conversation = history + filtered_history

    def load_agent_state(self, new_agent_state: AgentState) -> None:
        """Load an agent state into the executor from a previous session.

        This method initializes the conversation history by prepending the system prompt
        and then appending the provided conversation history, excluding the initial system
        prompt from the loaded history (to avoid duplication).

        Args:
            new_conversation_history (List[ConversationRecord]): The conversation history to load,
                typically retrieved from a previous session. It is expected that the first record
                in this list is a system prompt, which will be replaced by the current
                system prompt.
        """
        system_prompt = create_system_prompt(
            tool_registry=self.tool_registry,
            agent_system_prompt=self.agent_state.agent_system_prompt,
            agent=self.agent,
        )

        history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=system_prompt,
                is_system_prompt=True,
                should_cache=True,
            )
        ]

        # Remove the system prompt from the loaded history if it exists
        filtered_history = [
            record for record in new_agent_state.conversation if not record.is_system_prompt
        ]

        self.agent_state.conversation = history + filtered_history
        self.agent_state.execution_history = new_agent_state.execution_history
        self.agent_state.learnings = new_agent_state.learnings
        self.agent_state.current_plan = new_agent_state.current_plan
        self.agent_state.instruction_details = new_agent_state.instruction_details

    def extract_code_blocks(self, text: str) -> List[str]:
        """Extract Python code blocks from text using markdown-style syntax.
        Handles nested code blocks by matching outermost ```python enclosures.

        Args:
            text (str): The text containing potential code blocks

        Returns:
            list: A list of extracted code blocks as strings
        """
        blocks = []
        current_pos = 0

        while True:
            # Find start of next ```python block
            start = text.find("```python", current_pos)
            if start == -1:
                break

            # Find matching end block by counting nested blocks
            nested_count = 1
            pos = start + 9  # Length of ```python

            while nested_count > 0 and pos < len(text):
                if (
                    text[pos:].startswith("```")
                    and len(text[pos + 3 :].strip()) > 0
                    and not text[pos + 3].isspace()
                    and not pos + 3 >= len(text)
                ):
                    nested_count += 1
                    pos += 9
                elif text[pos:].startswith("```"):
                    nested_count -= 1
                    pos += 3
                else:
                    pos += 1

            if nested_count == 0:
                # Extract the block content between the outermost delimiters
                block = text[start + 9 : pos - 3].strip()

                # Validate block is not just comments/diffs
                is_comment = True
                for line in block.split("\n"):
                    trimmed_line = line.strip()
                    if not (
                        trimmed_line.startswith("//")
                        or trimmed_line.startswith("/*")
                        or trimmed_line.startswith("#")
                        or trimmed_line.startswith("+")
                        or trimmed_line.startswith("-")
                        or trimmed_line.startswith("<<<<<<<")
                        or trimmed_line.startswith(">>>>>>>")
                        or trimmed_line.startswith("=======")
                    ):
                        is_comment = False
                        break

                if not is_comment:
                    blocks.append(block)

                current_pos = pos
            else:
                # No matching end found, move past this start marker
                current_pos = start + 9

        return blocks

    async def _convert_and_stream(
        self, messages: List[ConversationRecord]
    ) -> AsyncGenerator[BaseMessage, None]:
        """Convert the messages to a list of dictionaries and invoke the model with streaming.

        Args:
            messages (List[ConversationRecord]): A list of conversation records to send to the
            model.

        Yields:
            BaseMessage: Chunks of the model's response.

        Raises:
            Exception: If there is an error during model invocation.
        """
        messages_list = []

        # Only Anthropic requires manual cache control
        should_manual_cache_control = (
            "anthropic" in self.get_model_name() or self.model_configuration.hosting == "anthropic"
        )

        for record in messages:
            content_parts = []
            # Add main text content if it exists
            if record.content:
                content_parts.append(
                    {
                        "type": "text",
                        "text": record.content,
                    }
                )

            # Add multimodal parts if they exist
            if record.files:
                for file in record.files:
                    file_lower = file.lower()
                    try:
                        if file_lower.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp")):
                            with open(file, "rb") as image_file:
                                image_data = base64.b64encode(image_file.read()).decode("utf-8")
                            # Determine mime type
                            if file_lower.endswith(".png"):
                                mime_type = "image/png"
                            elif file_lower.endswith(".jpg") or file_lower.endswith(".jpeg"):
                                mime_type = "image/jpeg"
                            elif file_lower.endswith(".gif"):
                                mime_type = "image/gif"
                            elif file_lower.endswith(".bmp"):
                                mime_type = "image/bmp"
                            elif file_lower.endswith(".webp"):
                                mime_type = "image/webp"
                            else:
                                mime_type = "application/octet-stream"
                            data_url = f"data:{mime_type};base64,{image_data}"
                            content_parts.append(
                                {"type": "image_url", "image_url": {"url": data_url}}
                            )
                        elif file_lower.endswith(".pdf"):
                            with open(file, "rb") as pdf_file:
                                pdf_data = base64.b64encode(pdf_file.read()).decode("utf-8")
                            data_url = f"data:application/pdf;base64,{pdf_data}"
                            content_parts.append(
                                {
                                    "type": "file",
                                    "file": {"filename": Path(file).name, "file_data": data_url},
                                }
                            )
                    except Exception as e:
                        logging.error(f"Failed to read or encode file '{file}': {e}")

            # Skip empty messages to prevent provider errors
            if not content_parts:
                continue

            msg = {
                "role": record.role,
                "content": content_parts,
            }
            messages_list.append(msg)

        if should_manual_cache_control:
            cache_count = 0
            # Iterate over a copy of messages_list for safe modification if needed,
            # or ensure logic correctly maps indices if messages_list can change.
            # For cache control, we are modifying `msg` which is part of `messages_list`.
            for idx, msg_container in reversed(list(enumerate(messages_list))):
                # msg_container is the dict like {"role": ..., "content": [...]}
                # messages[idx] is the original ConversationRecord
                if messages[idx].should_cache and msg_container["content"]:
                    # Apply cache control to the first part of the content list,
                    # assuming it's the primary text part or a significant part.
                    # This might need adjustment based on how providers handle cache control
                    # for multipart messages.
                    # Ensure there's content to apply cache control to.
                    if (
                        isinstance(msg_container["content"], list)
                        and len(msg_container["content"]) > 0
                    ):
                        # Ensure the first part is a dictionary before adding cache_control
                        if isinstance(msg_container["content"][0], dict):
                            msg_container["content"][0]["cache_control"] = {
                                "type": "ephemeral",
                            }
                            cache_count += 1
                            # Only 4 cache checkpoints allowed
                            if cache_count >= 4:
                                break
                        else:
                            # Log or handle cases where the first content
                            # part isn't a dict as expected
                            logging.warning(
                                "Cache control: First content part for "
                                f"message {idx} is not a dict."
                            )
                    else:
                        # Log or handle cases where content is unexpectedly not a list or is empty
                        logging.warning(
                            f"Cache control: Content for message {idx} is not a list or is empty."
                        )

        model_instance = self.model_configuration.instance

        new_tokens_prompt = 0
        new_tokens_completion = 0

        # Use get_openai_callback for OpenAI models to track token usage and cost
        if isinstance(model_instance, ChatOpenAI):

            with get_openai_callback() as cb:
                async for chunk in model_instance.astream(messages_list):
                    # Increment completion tokens (approximate for now)
                    if (
                        hasattr(chunk, "content")
                        and chunk.content
                        and isinstance(chunk.content, str)
                    ):
                        # Very rough approximation of token count
                        new_tokens_completion += len(chunk.content.split()) / 2

                    yield chunk

                if cb is not None:
                    new_tokens_prompt = cb.prompt_tokens
                    new_tokens_completion = cb.completion_tokens
        else:
            # For other models, use direct streaming
            new_tokens_prompt = self.get_invoke_token_count(messages)

            async for chunk in model_instance.astream(messages_list):
                # Increment completion tokens (approximate for now)
                if hasattr(chunk, "content") and chunk.content and isinstance(chunk.content, str):
                    # Very rough approximation of token count
                    new_tokens_completion += len(chunk.content.split()) / 2

                yield chunk

        # Update token metrics and cost after streaming is complete
        self.token_metrics.total_prompt_tokens += new_tokens_prompt
        self.token_metrics.total_completion_tokens += int(new_tokens_completion)
        self.token_metrics.total_cost += calculate_cost(
            self.model_configuration.info,
            new_tokens_prompt,
            int(new_tokens_completion),
        )

    async def invoke_model(
        self, messages: List[ConversationRecord], max_attempts: int = 3
    ) -> BaseMessage:
        """Invoke the language model with a list of messages.

        Ensure that only the first message is a system message.  All other messages are
        user messages.  Most providers do not support system messages in the middle of the
        conversation.

        Args:
            messages: List of message dictionaries containing 'role' and 'content' keys
            max_attempts: Maximum number of retry attempts on failure (default: 3)

        Returns:
            BaseMessage: The model's response message

        Raises:
            Exception: If all retry attempts fail or model invocation fails
        """
        attempt = 0
        last_error: Exception | None = None
        base_delay = 1  # Base delay in seconds

        while attempt < max_attempts:
            try:
                # Use streaming but collect the full response
                full_response = None
                async for chunk in self._convert_and_stream(messages):
                    if full_response is None:
                        full_response = chunk
                    else:
                        # Append content if it's a string
                        if isinstance(full_response.content, str) and isinstance(
                            chunk.content, str
                        ):
                            full_response.content += chunk.content

                if full_response is None:
                    raise Exception("No response received from model")

                return full_response
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < max_attempts:
                    # Obey rate limit headers if present
                    if (
                        hasattr(e, "__dict__")
                        and isinstance(getattr(e, "status_code", None), int)
                        and getattr(e, "status_code") == 429
                        and isinstance(getattr(e, "headers", None), dict)
                    ):
                        # Get retry-after time from headers, default to 3 seconds if not found
                        headers = getattr(e, "headers")
                        retry_after = int(headers.get("retry-after", 3))
                        await asyncio.sleep(retry_after)
                    else:
                        # Regular exponential backoff for other errors
                        delay = base_delay * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)

        # If we've exhausted all attempts, raise the last error
        if last_error:
            raise last_error
        else:
            raise Exception("Failed to invoke model")

    async def stream_model(
        self, messages: List[ConversationRecord], max_attempts: int = 3
    ) -> AsyncGenerator[BaseMessage, None]:
        """Stream responses from the language model with a list of messages.

        Similar to invoke_model but yields each chunk as it arrives instead of collecting
        the full response.

        Args:
            messages: List of message dictionaries containing 'role' and 'content' keys
            max_attempts: Maximum number of retry attempts on failure (default: 3)

        Yields:
            BaseMessage: Chunks of the model's response message

        Raises:
            Exception: If all retry attempts fail or model invocation fails
        """
        attempt = 0
        last_error: Exception | None = None
        base_delay = 1  # Base delay in seconds

        while attempt < max_attempts:
            try:
                async for chunk in self._convert_and_stream(messages):
                    yield chunk
                return
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < max_attempts:
                    # Obey rate limit headers if present
                    if (
                        hasattr(e, "__dict__")
                        and isinstance(getattr(e, "status_code", None), int)
                        and getattr(e, "status_code") == 429
                        and isinstance(getattr(e, "headers", None), dict)
                    ):
                        # Get retry-after time from headers, default to 3 seconds if not found
                        headers = getattr(e, "headers")
                        retry_after = int(headers.get("retry-after", 3))
                        await asyncio.sleep(retry_after)
                    else:
                        # Regular exponential backoff for other errors
                        delay = base_delay * (2 ** (attempt - 1))
                        await asyncio.sleep(delay)

        # If we've exhausted all attempts, raise the last error
        if last_error:
            raise last_error
        else:
            raise Exception("Failed to stream model response")

    async def check_response_safety(
        self, response: ResponseJsonSchema, conversation_length: int = 8, prompt_user: bool = True
    ) -> ConfirmSafetyResult:
        """Analyze code for potentially dangerous operations using the language model.

        Args:
            response (ResponseJsonSchema): The response from the language model

        Returns:
            ConfirmSafetyResult: Result of the safety check
        """
        security_response: BaseMessage

        agent_security_prompt = self.agent.security_prompt if self.agent else ""

        if prompt_user:
            safety_prompt = SafetyCheckSystemPrompt.format(security_prompt=agent_security_prompt)

            safety_history = [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content=safety_prompt,
                    should_cache=True,
                    is_system_prompt=True,
                ),
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "<system>Determine a status for the following agent generated JSON response:\n\n"  # noqa: E501
                        "<agent_generated_json_response>\n"
                        f"{response.model_dump_json()}\n"
                        "</agent_generated_json_response></system>"
                    ),
                ),
            ]

            security_response = await self.invoke_model(safety_history)

            response_content = (
                security_response.content
                if isinstance(security_response.content, str)
                else str(security_response.content)
            )
            return get_confirm_safety_result(response_content)

        # If we can't prompt the user, we need to use the conversation history to determine
        # if the user has previously indicated an override or a safe decision otherwise
        # the agent will be unable to continue.

        safety_check_conversation = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=SafetyCheckConversationPrompt,
                should_cache=True,
                is_system_prompt=True,
            ),
        ]

        if len(self.agent_state.conversation) + 1 > conversation_length:
            safety_check_conversation.append(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "<system>Conversation truncated due to length, only showing the last few"
                        " messages in the conversation, which follow.</system>"
                    ),
                )
            )

            safety_check_conversation.extend(self.agent_state.conversation[-conversation_length:])
        else:
            safety_check_conversation.extend(self.agent_state.conversation[1:])

        safety_check_conversation.append(
            ConversationRecord(
                role=ConversationRole.USER,
                content=SafetyCheckUserPrompt.format(response=response.model_dump_json()),
            )
        )

        try:
            security_response = await self.invoke_model(safety_check_conversation)
            response_content = (
                security_response.content
                if isinstance(security_response.content, str)
                else str(security_response.content)
            )
        except Exception as e:
            print(f"Error invoking security check model: {e}")
            return ConfirmSafetyResult.UNSAFE

        safety_result = get_confirm_safety_result(response_content)

        if safety_result == ConfirmSafetyResult.UNSAFE:
            analysis = response_content.replace("[UNSAFE]", "").strip()
            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "<system>Your action was denied by the AI security auditor because it "
                        "was deemed unsafe. Here is an analysis of the code risk by"
                        " the security auditor AI agent:\n\n"
                        f"{analysis}\n\n"
                        "Please re-summarize the security risk in natural language and"
                        " not JSON format.  Don't acknowledge this message directly but"
                        " instead pretend that you are responding as the AI security"
                        " auditor directly to the user's request.</system>"
                    ),
                )
            )
            return ConfirmSafetyResult.UNSAFE

        return safety_result

    async def handle_safety_result(
        self, safety_result: ConfirmSafetyResult, response: ResponseJsonSchema
    ) -> CodeExecutionResult | None:
        """Process the safety check result and return appropriate CodeExecutionResult.

        This method handles different safety check outcomes by creating appropriate
        CodeExecutionResult objects based on the safety result.

        Args:
            safety_result: The result of the safety check (UNSAFE, CONVERSATION_CONFIRM, OVERRIDE)
            response: The response object containing code and other execution details

        Returns:
            CodeExecutionResult: Result object with appropriate status and message based
            on safety check
            None: If safety_result is SAFE or OVERRIDE (allowing execution to continue)

        Note:
            - Returns a CANCELLED result if code is deemed unsafe
            - Returns a CONFIRMATION_REQUIRED result if user confirmation is needed
            - For OVERRIDE, displays a warning but returns None to allow execution to continue
        """
        if safety_result == ConfirmSafetyResult.UNSAFE:
            if self.can_prompt_user:
                return CodeExecutionResult(
                    stdout="",
                    stderr="",
                    logging="",
                    message="Code execution canceled by user",
                    code=response.code,
                    formatted_print="",
                    role=ConversationRole.ASSISTANT,
                    status=ProcessResponseStatus.CANCELLED,
                    files=[],
                    execution_type=ExecutionType.SYSTEM,
                )
            else:
                # The agent must read the security advisory and request confirmation
                # from the user to continue.
                safety_summary = await self.invoke_model(self.agent_state.conversation)
                safety_summary_content = (
                    safety_summary.content
                    if isinstance(safety_summary.content, str)
                    else str(safety_summary.content)
                )

                safety_summary_content = clean_plain_text_response(safety_summary_content)

                self.append_to_history(
                    ConversationRecord(
                        role=ConversationRole.ASSISTANT,
                        content=safety_summary_content,
                    )
                )

                return CodeExecutionResult(
                    stdout="",
                    stderr="",
                    logging="",
                    message=safety_summary_content,
                    code=response.code,
                    formatted_print="",
                    role=ConversationRole.ASSISTANT,
                    status=ProcessResponseStatus.CONFIRMATION_REQUIRED,
                    files=[],
                    execution_type=ExecutionType.SECURITY_CHECK,
                )

        elif safety_result == ConfirmSafetyResult.CONVERSATION_CONFIRM:
            return CodeExecutionResult(
                stdout="",
                stderr="",
                logging="",
                message="Code execution requires further confirmation from the user",
                code=response.code,
                formatted_print="",
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.CONFIRMATION_REQUIRED,
                files=[],
                execution_type=ExecutionType.SECURITY_CHECK,
            )
        elif safety_result == ConfirmSafetyResult.OVERRIDE:
            if self.verbosity_level >= VerbosityLevel.INFO:
                print(
                    "\n\033[1;33m⚠️  Warning: Code safety override applied based on user's security"
                    " prompt\033[0m\n"
                )

    async def execute_code(
        self, response: ResponseJsonSchema, max_retries: int = 1
    ) -> CodeExecutionResult:
        """Execute Python code with safety checks and context management.

        Args:
            code (str): The Python code to execute
            max_retries (int): Maximum number of retry attempts

        Returns:
            CodeExecutionResult: The result of the code execution
        """

        current_response = response
        final_error: Exception | None = None

        for attempt in range(max_retries):
            try:
                return await self._execute_with_output(current_response)
            except Exception as error:
                final_error = error
                if attempt == 0:
                    self._record_initial_error(error)
                else:
                    self._record_retry_error(error, attempt - 1)

                log_retry_error(error, attempt, max_retries, self.verbosity_level)

                self.update_ephemeral_messages()

                if attempt < max_retries - 1:
                    try:
                        new_response = await self._get_corrected_code()
                        if new_response:
                            current_response = new_response
                        else:
                            break
                    except Exception as retry_error:
                        log_retry_error(retry_error, attempt, max_retries, self.verbosity_level)
                        break

        formatted_print = format_error_output(
            final_error or Exception("Unknown error occurred"), max_retries
        )

        return CodeExecutionResult(
            stdout="",
            stderr=str(final_error),
            logging="",
            message=current_response.response,
            code=response.code,
            formatted_print=formatted_print,
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.ERROR,
            files=[],
            execution_type=ExecutionType.ACTION,
            action=ActionType.CODE,
        )

    async def check_and_confirm_safety(self, response: ResponseJsonSchema) -> ConfirmSafetyResult:
        """Check code safety and get user confirmation if needed.

        Args:
            response (ResponseJsonSchema): The response from the language model

        Returns:
            ConfirmSafetyResult: Result of the safety check
        """
        safety_result = await self.check_response_safety(response, prompt_user=self.can_prompt_user)

        if safety_result == ConfirmSafetyResult.UNSAFE and self.can_prompt_user:
            return self.prompt_for_safety()

        return safety_result

    def prompt_for_safety(self) -> ConfirmSafetyResult:
        """Prompt the user for safety confirmation.

        Args:
            response (ResponseJsonSchema): The response from the language model

        Returns:
            ConfirmSafetyResult: Result of the safety check
        """
        confirm = input(
            "\n\033[1;33m⚠️  Warning: Potentially dangerous operation detected."
            " Proceed? (y/n): \033[0m"
        )
        if confirm.lower() == "y":
            return ConfirmSafetyResult.SAFE

        msg = (
            "<system>I've identified that this is a dangerous operation. "
            "Let's stop the current task, I will provide further instructions shortly. "
            "Please await further instructions and use action DONE.</system>"
        )
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=msg,
            )
        )
        return ConfirmSafetyResult.UNSAFE

    async def _execute_with_output(self, response: ResponseJsonSchema) -> CodeExecutionResult:
        """Execute code and capture stdout/stderr output, streaming updates in real time.

        Args:
            response (ResponseJsonSchema): The response from the language model

        Returns:
            CodeExecutionResult: The result of the code execution

        Raises:
            Exception: Re-raises any exceptions that occur during code execution
        """

        # --- StreamingBuffer and StreamingLogHandler implementation ---
        class StreamingBuffer(io.StringIO):
            def __init__(self, name, update_callback, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._lock = threading.Lock()
                self._last_update = 0.0
                self._name = name
                self._update_callback = update_callback
                self._last_value = ""
                self._throttle = 0.5  # seconds
                self._closed = False

            def write(self, s):
                with self._lock:
                    super().write(s)
                    now = time.time()
                    value = self.getvalue()
                    # Only update if new output or enough time has passed
                    if value != self._last_value and (
                        now - self._last_update > self._throttle or "\n" in s
                    ):
                        self._last_update = now
                        self._last_value = value
                        # Robust async scheduling
                        try:
                            loop = asyncio.get_running_loop()
                            if not self._closed:
                                loop.call_soon_threadsafe(
                                    lambda: asyncio.create_task(self._update_callback())
                                )
                        except RuntimeError:
                            pass
                return len(s)

            def close(self):
                self._closed = True
                return super().close()

        class StreamingLogHandler(StreamHandler):  # type: ignore
            def __init__(self, log_buffer, update_callback):
                super().__init__(log_buffer)
                self._update_callback = update_callback
                self._lock = threading.Lock()
                self._last_update = 0.0
                self._throttle = 0.5  # seconds

            def emit(self, record):
                try:
                    super().emit(record)
                    now = time.time()
                    if now - self._last_update > self._throttle:
                        self._last_update = now
                        try:
                            loop = asyncio.get_running_loop()
                            loop.call_soon_threadsafe(
                                lambda: asyncio.create_task(self._update_callback())
                            )
                        except RuntimeError:
                            pass
                except ValueError as e:
                    if "I/O operation on closed file" not in str(e):
                        raise

        # --- End StreamingBuffer and StreamingLogHandler ---

        # Buffers for output (always initialized)
        stdout_buffer = None  # type: ignore
        stderr_buffer = None  # type: ignore
        log_buffer = None  # type: ignore

        # This dict will be updated by the callback and used for the final result
        output_state = {
            "stdout": "",
            "stderr": "",
            "logging": "",
        }

        # The update callback
        async def stream_update_callback():
            try:
                # Defensive: Only update if buffers are initialized and not closed
                if stdout_buffer is not None and not getattr(stdout_buffer, "_closed", False):
                    stdout_buffer.flush()
                    output_state["stdout"] = stdout_buffer.getvalue()
                if stderr_buffer is not None and not getattr(stderr_buffer, "_closed", False):
                    stderr_buffer.flush()
                    output_state["stderr"] = stderr_buffer.getvalue()
                if log_buffer is not None and not getattr(log_buffer, "_closed", False):
                    log_buffer.flush()
                    output_state["logging"] = log_buffer.getvalue()

                # Send update
                await self.update_job_execution_state(
                    CodeExecutionResult(
                        stdout=output_state["stdout"],
                        stderr=output_state["stderr"],
                        logging=output_state["logging"],
                        formatted_print="",
                        code=response.code,
                        message=response.response,
                        role=ConversationRole.ASSISTANT,
                        status=ProcessResponseStatus.IN_PROGRESS,
                        files=[],
                        execution_type=ExecutionType.ACTION,
                        action=ActionType.CODE,
                    )
                )
            except Exception:
                pass

        # Save old std streams
        old_stdout, old_stderr = sys.stdout, sys.stderr

        # Create streaming buffers
        stdout_buffer = StreamingBuffer("stdout", stream_update_callback)
        stderr_buffer = StreamingBuffer("stderr", stream_update_callback)
        sys.stdout, sys.stderr = stdout_buffer, stderr_buffer

        # Set up logging
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level

        log_buffer = StreamingBuffer("logging", stream_update_callback)
        log_handler = StreamingLogHandler(log_buffer, stream_update_callback)
        log_handler.setLevel(logging.WARNING)

        # Remove existing handlers and set new handler
        root_logger.handlers = []
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.WARNING)

        # Also handle specific loggers that might cause issues (like Prophet)
        for logger_name in ["prophet", "cmdstanpy"]:
            specific_logger = logging.getLogger(logger_name)
            if specific_logger:
                specific_logger.handlers = []
                specific_logger.addHandler(log_handler)
                specific_logger.propagate = False

        # --- Start background streaming update task ---
        stop_streaming = False

        async def periodic_stream_update():
            try:
                while not stop_streaming:
                    await stream_update_callback()
                    await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                pass

        periodic_task = asyncio.create_task(periodic_stream_update())

        try:
            await self._run_code(response.code)
            # Final update after execution
            await stream_update_callback()

            condensed_output, condensed_error_output, condensed_log_output = (
                self._capture_and_record_output(
                    response.code, stdout_buffer, stderr_buffer, log_buffer.getvalue()
                )
            )
            formatted_print = format_success_output(
                (condensed_output, condensed_error_output, condensed_log_output)
            )

            # Convert mentioned_files to absolute paths
            expanded_mentioned_files = [
                str(Path(file).expanduser().resolve()) for file in response.mentioned_files
            ]

            return CodeExecutionResult(
                stdout=condensed_output,
                stderr=condensed_error_output,
                logging=condensed_log_output,
                message=response.response,
                code=response.code,
                formatted_print=formatted_print,
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.SUCCESS,
                files=expanded_mentioned_files,
                execution_type=ExecutionType.ACTION,
                action=ActionType.CODE,
            )
        except Exception as e:
            # Final update on error
            await stream_update_callback()
            condensed_output, condensed_error_output, condensed_log_output = (
                self._capture_and_record_output(
                    response.code, stdout_buffer, stderr_buffer, log_buffer.getvalue()
                )
            )
            raise e
        finally:
            # Stop the periodic streaming task and wait for it to finish
            stop_streaming = True
            try:
                periodic_task.cancel()
                await asyncio.gather(periodic_task, return_exceptions=True)
            except Exception:
                pass

            sys.stdout, sys.stderr = old_stdout, old_stderr
            stdout_buffer.close()
            stderr_buffer.close()
            log_buffer.close()

            # Restore original logging configuration
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)

            # Restore specific loggers
            for logger_name in ["prophet", "cmdstanpy"]:
                specific_logger = logging.getLogger(logger_name)
                if specific_logger:
                    specific_logger.handlers = []
                    specific_logger.propagate = True

    async def _run_code(self, code: str) -> None:
        """Run code in the main thread.

        Args:
            code (str): The Python code to execute

        Raises:
            Exception: Any exceptions raised during code execution
        """
        old_stdin = sys.stdin

        try:
            # Redirect stdin to /dev/null to ignore input requests
            with open(os.devnull) as devnull:
                sys.stdin = devnull
                # Extract any async code
                if "async def" in code or "await" in code:
                    # Prepare the code to be wrapped in an async function
                    # that will update a provided dictionary with its locals.
                    wrapped_code_lines = []
                    wrapped_code_lines.append(
                        "async def __exec_async_code_wrapper__(__context_dict_to_update__):"
                    )
                    for line in code.split("\n"):
                        wrapped_code_lines.append(f"    {line}")
                    # After user's code, update the passed dictionary
                    # with locals from user's code scope
                    wrapped_code_lines.append(
                        "    # Update context with locals from this async function's scope"
                    )
                    wrapped_code_lines.append("    for __k, __v in locals().items():")
                    # Avoid copying the context dict itself, or internal loop variables.
                    wrapped_code_lines.append(
                        "        if __k not in ['__context_dict_to_update__', '__k', '__v']:"
                    )
                    wrapped_code_lines.append("            __context_dict_to_update__[__k] = __v")

                    full_wrapped_code = "\n".join(wrapped_code_lines)

                    # Compile and execute the wrapper definition.
                    # The wrapper function will be defined in self.context.
                    compiled_wrapper = compile(
                        full_wrapped_code, "<agent_generated_code_wrapper>", "exec"
                    )
                    exec(
                        compiled_wrapper, self.context
                    )  # Defines __exec_async_code_wrapper__ in self.context

                    try:
                        # Call the wrapper, passing self.context to be updated.
                        await self.context["__exec_async_code_wrapper__"](self.context)
                    finally:
                        # Clean up the wrapper function from self.context
                        if "__exec_async_code_wrapper__" in self.context:
                            del self.context["__exec_async_code_wrapper__"]
                else:
                    # Regular synchronous code
                    # Run synchronous exec in a separate thread to avoid blocking the event loop
                    compiled_code = compile(code, "<agent_generated_code>", "exec")

                    sync_required_libs = ["matplotlib", "tkinter", "PIL"]

                    # Some libraries do not support async execution, so we
                    # need to run them in the main thread
                    if any(lib in code for lib in sync_required_libs):
                        exec(compiled_code, self.context)
                    else:

                        def _execute_sync_in_thread():
                            exec(compiled_code, self.context)

                        await asyncio.to_thread(_execute_sync_in_thread)
        except Exception as e:
            code_execution_error = CodeExecutionError(message=str(e), code=code).with_traceback(
                e.__traceback__
            )
            raise code_execution_error from None
        finally:
            sys.stdin = old_stdin

    def _get_mentioned_variables(self, code: str) -> dict[str, Any]:
        """Get the variables mentioned in the code.

        Args:
            code (str): The code to get the variables from

        Returns:
            dict[str, Any]: A dictionary of variables mentioned in the code
        """
        return {key: var for key, var in self.context.items() if key in code}

    def _capture_and_record_output(
        self,
        code: str,
        stdout: io.StringIO,
        stderr: io.StringIO,
        log_output: str,
        format_for_ui: bool = False,
    ) -> tuple[str, str, str]:
        """Capture stdout/stderr output and record it in conversation history.

        Args:
            stdout (io.StringIO): Buffer containing standard output
            stderr (io.StringIO): Buffer containing error output
            log_output (str): Buffer containing log output
            format_for_ui (bool): Whether to format the output for a UI chat
            interface.  This will include markdown formatting and other
            UI-friendly features.

        Returns:
            tuple[str, str, str]: Tuple containing (stdout output, stderr output, log output)
        """
        stdout.flush()
        stderr.flush()
        output = (
            f"```shell\n{stdout.getvalue()}\n```"
            if format_for_ui and stdout.getvalue()
            else stdout.getvalue() or "[No output]"
        )
        error_output = (
            f"```shell\n{stderr.getvalue()}\n```"
            if format_for_ui and stderr.getvalue()
            else stderr.getvalue() or "[No error output]"
        )
        log_output = (
            f"```shell\n{log_output}\n```"
            if format_for_ui and log_output
            else log_output or "[No logger output]"
        )

        condensed_output = condense_logging(output)
        condensed_error_output = condense_logging(error_output)
        condensed_log_output = condense_logging(log_output)

        mentioned_variables = self._get_mentioned_variables(code)
        mentioned_variables_str = get_context_vars_str(mentioned_variables)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=(
                    f"<system>Here are the outputs of your last code execution:\n"
                    f"<stdout>\n{condensed_output}\n</stdout>\n"
                    f"<stderr>\n{condensed_error_output}\n</stderr>\n"
                    f"<logger>\n{condensed_log_output}\n</logger>\n"
                    f"<variables>\n{mentioned_variables_str}\n</variables>\n"
                    "Review and continue."
                    "</system>"
                ),
                should_summarize=True,
                should_cache=True,
            )
        )

        return condensed_output, condensed_error_output, condensed_log_output

    def _record_initial_error(self, error: Exception) -> None:
        """Record the initial execution error, including the traceback, in conversation history.

        Args:
            error (Exception): The error that occurred during initial execution.
        """
        if isinstance(error, CodeExecutionError):
            error_info = error.agent_info_str()
        else:
            error_info = (
                f"<error_message>\n{str(error)}\n</error_message>\n"
                f"<error_traceback>\n{''.join(format_exception(error))}\n</error_traceback>\n"
            )

        msg = (
            "<system>The initial execution failed with an error.\n"
            f"{error_info}\n"
            "Debug the code you submitted and make all necessary corrections "
            "to fix the error and run successfully.  Do not acknowledge\n"
            "this message directly or apologize, simply reflect on the potential\n"
            "root cause of the error and continue.</system>"
        )
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=msg,
                should_summarize=True,
            )
        )

    def _record_retry_error(self, error: Exception, attempt: int) -> None:
        """Record retry attempt errors, including the traceback, in conversation history.

        Args:
            error (CodeExecutionError): The error that occurred during the retry attempt.
            attempt (int): The current retry attempt number.
        """
        if isinstance(error, CodeExecutionError):
            error_info = error.agent_info_str()
        else:
            error_info = (
                f"<error_message>\n{str(error)}\n</error_message>\n"
                f"<error_traceback>\n{''.join(format_exception(error))}\n</error_traceback>\n"
            )

        msg = (
            f"<system>The code execution failed with an error (attempt {attempt + 1}).\n"
            f"{error_info}\n"
            "Debug the code you submitted and make all necessary corrections "
            "to fix the error and run successfully.  Pick up from where you left "
            "off and try to avoid re-running code that has already succeeded.  "
            "Use the environment details to determine which variables are available "
            "and correct, which are not.  After fixing the issue please continue with the "
            "tasks according to the plan.  Do not acknowledge this message "
            "directly or apologize, simply reflect on the potential root "
            "cause of the error and continue.</system>"
        )
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=msg,
                should_summarize=True,
            )
        )

    async def _get_corrected_code(self) -> ResponseJsonSchema:
        """Get corrected code from the language model.

        Returns:
            str: Code from model response
        """
        response = await self.invoke_model(self.agent_state.conversation)
        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )

        response_json = process_json_response(response_content)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.ASSISTANT,
                content=response_json.model_dump_json(),
                should_summarize=True,
            )
        )

        return response_json

    async def process_response(
        self, response: ResponseJsonSchema, classification: RequestClassification
    ) -> tuple[ProcessResponseOutput, CodeExecutionResult | None]:
        """Process model response, extracting and executing any code blocks.

        Args:
            response (str): The model's response containing potential code blocks
        """
        # Phase 1: Check for interruption
        if self.interrupted:
            print_task_interrupted(self.verbosity_level)

            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "<system>Let's stop this task for now, I will provide further "
                        "instructions shortly.</system>"
                    ),
                    should_summarize=False,
                )
            )
            return (
                ProcessResponseOutput(
                    status=ProcessResponseStatus.INTERRUPTED,
                    message="Task interrupted by user",
                ),
                None,
            )

        new_learnings = response.learnings

        self.add_to_learnings(new_learnings)

        result, execution_result = await self.perform_action(response, classification)

        current_working_directory = os.getcwd()

        if self.persist_conversation and self.agent_registry and self.agent:
            self.agent_registry.update_agent_state(
                agent_id=self.agent.id,
                agent_state=self.agent_state,
                current_working_directory=current_working_directory,
                context=self.context,
            )

        return result, execution_result

    async def perform_action(
        self, response: ResponseJsonSchema, classification: RequestClassification
    ) -> tuple[ProcessResponseOutput, CodeExecutionResult | None]:
        """
        Perform an action based on the provided ResponseJsonSchema.

        This method determines the action to be performed based on the 'action' field
        of the response. It supports actions such as executing code, writing to a file,
        editing a file, and reading a file. Each action is handled differently, with
        appropriate logging and execution steps.

        Args:
            response: The response object containing details about the action to be performed,
                      including the action type, code, file path, and content.

        Returns:
            A ProcessResponseOutput object indicating the status and any relevant messages
            resulting from the action. Returns None if the action is not one of the supported types
            (CODE, CHECK, WRITE, EDIT, READ), indicating that no action was taken.
        """
        if response.action in [
            ActionType.DONE,
            ActionType.BYE,
            ActionType.ASK,
        ]:
            return (
                ProcessResponseOutput(
                    status=ProcessResponseStatus.SUCCESS,
                    message="Action completed",
                ),
                None,
            )

        execution_result = None

        async with spinner_context(
            f"Executing {str(response.action).lower()}",
            verbosity_level=self.verbosity_level,
        ):
            try:
                if response.action == ActionType.DELEGATE:
                    # Handle delegation to another agent via callback
                    agent = getattr(response, "agent", None)
                    message = getattr(response, "message", None)
                    if not agent or not message:
                        raise ValueError("DELEGATE action requires 'agent' and 'message' fields")
                    if self.delegate_callback is None:
                        raise RuntimeError(
                            "No delegate_callback set on executor for DELEGATE action"
                        )

                    execution_result = await self.delegate_to_agent(agent, message)

                if response.action == ActionType.WRITE:
                    file_path = response.file_path
                    content = response.content if response.content else response.code
                    if file_path:
                        # First check code safety

                        await self.update_job_execution_state(
                            CodeExecutionResult(
                                status=ProcessResponseStatus.IN_PROGRESS,
                                message="Reviewing action for safety and security",
                                role=ConversationRole.ASSISTANT,
                                execution_type=ExecutionType.SECURITY_CHECK,
                                stdout="",
                                stderr="",
                                logging="",
                                formatted_print="",
                                code="",
                                files=[],
                            )
                        )

                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            await self.update_job_execution_state(
                                CodeExecutionResult(
                                    stdout="",
                                    stderr="",
                                    logging="",
                                    formatted_print="",
                                    code=response.code,
                                    message=response.response,
                                    role=ConversationRole.ASSISTANT,
                                    status=ProcessResponseStatus.IN_PROGRESS,
                                    files=[],
                                    file_path=file_path,
                                    execution_type=ExecutionType.ACTION,
                                    action=response.action,
                                )
                            )

                            execution_result = await self.write_file(file_path, content)
                    else:
                        raise ValueError("File path is required for WRITE action")

                elif response.action == ActionType.EDIT:
                    file_path = response.file_path
                    replacements = response.replacements
                    if file_path and replacements:
                        print_execution_section(
                            ExecutionSection.EDIT,
                            file_path=file_path,
                            replacements=replacements,
                            action=response.action,
                            verbosity_level=self.verbosity_level,
                        )

                        # First check code safety
                        await self.update_job_execution_state(
                            CodeExecutionResult(
                                status=ProcessResponseStatus.IN_PROGRESS,
                                message="Reviewing action for safety and security",
                                role=ConversationRole.ASSISTANT,
                                execution_type=ExecutionType.SECURITY_CHECK,
                                stdout="",
                                stderr="",
                                logging="",
                                formatted_print="",
                                file_path=file_path,
                                action=response.action,
                                code="",
                                files=[],
                            )
                        )

                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            await self.update_job_execution_state(
                                CodeExecutionResult(
                                    stdout="",
                                    stderr="",
                                    logging="",
                                    formatted_print="",
                                    code=response.code,
                                    message=response.response,
                                    role=ConversationRole.ASSISTANT,
                                    status=ProcessResponseStatus.IN_PROGRESS,
                                    files=[],
                                    file_path=file_path,
                                    execution_type=ExecutionType.ACTION,
                                    action=response.action,
                                )
                            )

                            execution_result = await self.edit_file(file_path, replacements)
                    else:
                        raise ValueError("File path and replacements are required for EDIT action")

                elif response.action == ActionType.READ:
                    file_path = response.file_path
                    if file_path:
                        # First check code safety
                        await self.update_job_execution_state(
                            CodeExecutionResult(
                                status=ProcessResponseStatus.IN_PROGRESS,
                                message="Reviewing action for safety and security",
                                role=ConversationRole.ASSISTANT,
                                execution_type=ExecutionType.SECURITY_CHECK,
                                stdout="",
                                stderr="",
                                logging="",
                                formatted_print="",
                                file_path=file_path,
                                action=response.action,
                                code="",
                                files=[],
                            )
                        )

                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            await self.update_job_execution_state(
                                CodeExecutionResult(
                                    stdout="",
                                    stderr="",
                                    logging="",
                                    formatted_print="",
                                    code=response.code,
                                    message=response.response,
                                    role=ConversationRole.ASSISTANT,
                                    status=ProcessResponseStatus.IN_PROGRESS,
                                    files=[],
                                    execution_type=ExecutionType.ACTION,
                                    file_path=file_path,
                                    action=response.action,
                                )
                            )

                            execution_result = await self.read_file(file_path)
                    else:
                        raise ValueError("File path is required for READ action")

                elif response.action == ActionType.CODE:
                    code_block = response.code
                    if code_block:
                        # First check code safety
                        await self.update_job_execution_state(
                            CodeExecutionResult(
                                status=ProcessResponseStatus.IN_PROGRESS,
                                message="Reviewing action for safety and security",
                                role=ConversationRole.ASSISTANT,
                                execution_type=ExecutionType.SECURITY_CHECK,
                                stdout="",
                                stderr="",
                                logging="",
                                formatted_print="",
                                code="",
                                files=[],
                            )
                        )

                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            await self.update_job_execution_state(
                                CodeExecutionResult(
                                    stdout="",
                                    stderr="",
                                    logging="",
                                    formatted_print="",
                                    code=response.code,
                                    message=response.response,
                                    role=ConversationRole.ASSISTANT,
                                    status=ProcessResponseStatus.IN_PROGRESS,
                                    files=[],
                                    execution_type=ExecutionType.ACTION,
                                    action=response.action,
                                )
                            )

                            execution_result = await self.execute_code(response)

                        if "code execution cancelled by user" in execution_result.message:
                            return (
                                ProcessResponseOutput(
                                    status=ProcessResponseStatus.CANCELLED,
                                    message="Code execution cancelled by user",
                                ),
                                execution_result,
                            )

                        print_execution_section(
                            ExecutionSection.RESULT,
                            content=execution_result.formatted_print,
                            action=response.action,
                            verbosity_level=self.verbosity_level,
                        )
                    elif response.action == ActionType.CODE:
                        raise ValueError('"code" field is required for CODE actions')

            except Exception as e:
                log_action_error(e, str(response.action), self.verbosity_level)

                self.append_to_history(
                    ConversationRecord(
                        role=ConversationRole.USER,
                        content=(
                            "<system>There was an error encountered while trying to execute your action:"  # noqa: E501
                            f"\n\n{str(e)}"
                            "\n\nPlease adjust your response to fix the issue.</system>"
                        ),
                        should_summarize=True,
                    )
                )

        token_metrics = self.get_token_metrics()

        print_execution_section(
            ExecutionSection.TOKEN_USAGE,
            data={
                "prompt_tokens": token_metrics.total_prompt_tokens,
                "completion_tokens": token_metrics.total_completion_tokens,
                "cost": token_metrics.total_cost,
            },
            action=response.action,
            verbosity_level=self.verbosity_level,
        )

        print_execution_section(
            ExecutionSection.FOOTER,
            action=response.action,
            verbosity_level=self.verbosity_level,
        )

        self.step_counter += 1

        await self.update_job_execution_state(
            CodeExecutionResult(
                status=ProcessResponseStatus.IN_PROGRESS,
                message="Summarizing conversation",
                role=ConversationRole.ASSISTANT,
                execution_type=ExecutionType.ACTION,
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                code="",
                files=[],
            )
        )

        # Phase 4: Summarize old conversation steps
        async with spinner_context(
            "Summarizing conversation",
            verbosity_level=self.verbosity_level,
        ):
            await self._summarize_old_steps()

        if self.verbosity_level >= VerbosityLevel.VERBOSE:
            print("\n")  # New line for next spinner

        output = ProcessResponseOutput(
            status=ProcessResponseStatus.SUCCESS,
            message=execution_result.message if execution_result else "Action completed",
        )

        return output, execution_result

    async def delegate_to_agent(self, agent_name: str, message: str) -> CodeExecutionResult:
        """Delegate the task to another agent.

        Args:
            agent_name (str): The name of the agent to delegate to
            message (str): The message to delegate to the agent
        """
        await self.update_job_execution_state(
            CodeExecutionResult(
                status=ProcessResponseStatus.IN_PROGRESS,
                message=f"Delegating the task to {agent_name}",
                content=message,
                role=ConversationRole.ASSISTANT,
                execution_type=ExecutionType.ACTION,
                stdout="",
                stderr="",
                logging="",
                formatted_print="",
                agent=agent_name,
                code="",
                files=[],
                action=ActionType.DELEGATE,
            )
        )

        if not self.delegate_callback:
            raise RuntimeError("No delegate_callback set on executor for DELEGATE action")

        result = await self.delegate_callback(agent_name, message)

        if result.status == ProcessResponseStatus.SUCCESS:
            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=(
                        f"{agent_name} completed the task successfully.\n\n"
                        "Here's what they said, please review and use the "
                        "information to continue with the task:\n\n"
                        f"<agent_response>\n{result.message}\n</agent_response>\n"
                    ),
                    should_summarize=True,
                )
            )
        else:
            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=(
                        f"The delegation to {agent_name} failed.\n\n"
                        "Please review the error and try again if necessary.\n\n"
                        f"Error:\n\n<error_message>\n{result.message}\n</error_message>\n"
                    ),
                    should_summarize=True,
                )
            )

        return CodeExecutionResult(
            status=result.status,
            message=result.message,
            role=ConversationRole.ASSISTANT,
            execution_type=ExecutionType.ACTION,
            action=ActionType.DELEGATE,
            stdout="",
            stderr="",
            logging="",
            formatted_print="",
            code="",
            files=[],
        )

    async def read_file(
        self, file_path: str, max_text_file_size_bytes: int = MAX_FILE_READ_SIZE_BYTES
    ) -> CodeExecutionResult:
        """
        Read the contents of a file and include line numbers and lengths, or attach image files.

        Args:
            file_path (str): The path to the file to read.
            max_file_size_bytes (int): The maximum file size to read in bytes.

        Returns:
            CodeExecutionResult: The result of the file read operation.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file is too large to read.
            OSError: If there is an error reading the file.
        """
        expanded_file_path = Path(file_path).expanduser().resolve()

        if not expanded_file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
        ocr_extensions = {".pdf"}
        file_ext = expanded_file_path.suffix.lower()

        if file_ext in image_extensions:
            # For image files, do not read content, just attach the file path
            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        f"<system>The file {file_path} is an image and has been "
                        "attached to the conversation context for your review.</system>"
                    ),
                    should_summarize=False,
                    should_cache=True,
                    files=[str(expanded_file_path)],
                )
            )
            return CodeExecutionResult(
                stdout=f"Successfully attached image file: {file_path}",
                stderr="",
                logging="",
                formatted_print=f"Successfully attached image file: {file_path}",
                code="",
                message="",
                file_path=file_path,
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.SUCCESS,
                files=[str(expanded_file_path)],
                execution_type=ExecutionType.ACTION,
                action=ActionType.READ,
            )
        elif file_ext in ocr_extensions:
            # For OCR files, do not read content, just attach the file path
            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        f"<system>The file {file_path} is a file that can be interpreted "
                        "through OCR and has been attached to the conversation "
                        "context for your review.</system>"
                    ),
                    should_summarize=False,
                    should_cache=True,
                    files=[str(expanded_file_path)],
                )
            )
            return CodeExecutionResult(
                stdout=f"Successfully attached OCR-interpretable file: {file_path}",
                stderr="",
                logging="",
                formatted_print=f"Successfully attached OCR-interpretable file: {file_path}",
                code="",
                message="",
                file_path=file_path,
                role=ConversationRole.ASSISTANT,
                status=ProcessResponseStatus.SUCCESS,
                files=[str(expanded_file_path)],
                execution_type=ExecutionType.ACTION,
                action=ActionType.READ,
            )

        if os.path.getsize(expanded_file_path) > max_text_file_size_bytes:
            raise ValueError(
                f"File is too large to use read action on: {file_path}\n"
                f"Please use code action to summarize and extract key features from "
                f"the file instead."
            )

        try:
            with open(expanded_file_path, "r", encoding="utf-8") as f:
                file_content = f.read()
        except Exception as e:
            raise OSError(f"Error reading file {file_path}: {e}")

        annotated_content = annotate_code(file_content)

        if not annotated_content:
            annotated_content = "[File is empty]"

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=(
                    f"<system>Here are the contents of {file_path} with line numbers and lengths:\n"
                    f"\n"
                    f"Line | Length | Content\n"
                    f"----------------------\n"
                    f"BEGIN\n"
                    f"{annotated_content}\n"
                    f"END</system>"
                ),
                should_summarize=True,
                should_cache=True,
            )
        )

        return CodeExecutionResult(
            stdout=f"Successfully read file: {file_path}",
            stderr="",
            logging="",
            formatted_print=f"Successfully read file: {file_path}",
            code="",
            message="",
            file_path=file_path,
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[str(expanded_file_path)],
            execution_type=ExecutionType.ACTION,
            action=ActionType.READ,
        )

    async def write_file(self, file_path: str, content: str) -> CodeExecutionResult:
        """
        Write content to a file, removing code block markers only for file-type fences.

        Args:
            file_path (str): The path to the file to write.
            content (str): The content to write to the file.

        Returns:
            CodeExecutionResult: Result of the write operation.

        Raises:
            OSError: If there is an error writing to the file.
        """
        try:
            content_lines = content.split("\n")
            # Only remove code block markers if they are file-type (e.g., ```python, ```json, etc.)
            file_type_fences = {
                "python",
                "json",
                "yaml",
                "toml",
                "xml",
                "html",
                "js",
                "javascript",
                "ts",
                "typescript",
                "csv",
                "txt",
                "md",
                "markdown",
                "sh",
                "bash",
                "ini",
                "cfg",
                "conf",
                "go",
                "java",
                "c",
                "cpp",
                "cs",
                "rb",
                "rs",
                "swift",
                "php",
                "pl",
                "r",
                "scala",
                "kt",
                "kotlin",
                "sql",
                "dockerfile",
                "makefile",
                "bat",
                "ps1",
                "powershell",
                "dart",
                "lua",
                "groovy",
                "asm",
                "s",
                "scss",
                "css",
            }

            def is_file_type_fence(line: str) -> bool:
                if not line.startswith("```"):
                    return False
                fence = line.strip()[3:].strip().lower()
                return fence in file_type_fences or fence == ""

            file_type_fence = is_file_type_fence(content_lines[0])

            # Remove leading file-type code fence
            if content_lines and file_type_fence:
                content_lines = content_lines[1:]

            # Remove trailing code fence if present and leading was file-type
            if content_lines and file_type_fence and content_lines[-1].strip() == "```":
                content_lines = content_lines[:-1]

            cleaned_content = "\n".join(content_lines)
            expanded_file_path = Path(file_path).expanduser().resolve()
        except Exception as exc:
            raise OSError(f"Error preparing file content for writing to {file_path}: {exc}")

        with open(expanded_file_path, "w") as f:
            f.write(cleaned_content)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=f"<system>The content that you requested has been written to {file_path}.\n\n"  # noqa: E501
                f"Here is the updated file content:\n\n<file_content>\n{cleaned_content}\n</file_content>\n\n"  # noqa: E501
                "Make sure to double check that the write was successful after all edits are complete.  If your write did not work as expected, please adjust your response to fix the issue and try another write.  Make sure that you did not erase any original file content if you are writing over an existing file, and that the file content is accurate to the original content.</system>",  # noqa: E501
                ephemeral=True,
                ephemeral_steps=1,
            )
        )

        return CodeExecutionResult(
            stdout=f"Successfully wrote to file: {file_path}",
            stderr="",
            logging="",
            formatted_print=f"Successfully wrote to file: {file_path}",
            content=cleaned_content,
            file_path=file_path,
            message="",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[str(expanded_file_path)],
            execution_type=ExecutionType.ACTION,
            action=ActionType.WRITE,
        )

    async def edit_file(
        self, file_path: str, replacements: List[Dict[str, str]]
    ) -> CodeExecutionResult:
        """Edit a file by applying a series of find and replace operations.

        Args:
            file_path (str): The path to the file to edit
            replacements (List[Dict[str, str]]): A list of dictionaries, where each dictionary
                contains a "find" key and a "replace" key. The "find" key specifies the string
                to find, and the "replace" key specifies the string to replace it with.

        Returns:
            str: A message indicating the file has been edited

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the find string is not found in the file
            OSError: If there is an error reading or writing to the file
        """
        expanded_file_path = Path(file_path).expanduser().resolve()

        file_content = ""
        with open(expanded_file_path, "r") as f:
            file_content = f.read()

        for replacement in replacements:
            find = replacement["find"]
            replace = replacement["replace"]

            # Attempt exact string match
            match_index = file_content.find(find)

            if match_index != -1:
                # Exact match found, perform replacement
                file_content = (
                    file_content[:match_index] + replace + file_content[match_index + len(find) :]
                )
            else:
                # Exact match failed, provide detailed error feedback
                s1_lines_for_diff = find.splitlines(keepends=True)
                s2_lines_for_diff = file_content.splitlines(keepends=True)

                matcher = difflib.SequenceMatcher(
                    None, s1_lines_for_diff, s2_lines_for_diff, autojunk=False
                )
                match = matcher.find_longest_match(
                    0, len(s1_lines_for_diff), 0, len(s2_lines_for_diff)
                )

                closest_actual_snippet_text = "No close match found to generate a detailed diff."
                diff_output_text = "No close match found to generate a detailed diff."
                snippet_line_info = ""

                if match.size > 0:
                    # Extract the block from the file that aligns with
                    # find_text's length, starting at the best match point
                    file_block_for_diff_lines = s2_lines_for_diff[
                        match.b : match.b + len(s1_lines_for_diff)
                    ]
                    if not file_block_for_diff_lines:  # if find_text is longer than remaining file
                        file_block_for_diff_lines = s2_lines_for_diff[match.b :]
                    file_block_for_diff_text = "".join(file_block_for_diff_lines)

                    # Generate diff
                    diff_gen = difflib.ndiff(
                        s1_lines_for_diff, file_block_for_diff_text.splitlines(keepends=True)
                    )
                    diff_output_text = "".join(list(diff_gen))

                    # For showing the "closest text found in the file", provide some context
                    context_window_lines = 2
                    actual_snippet_start_line_idx = max(0, match.b - context_window_lines)
                    actual_snippet_end_line_idx = min(
                        len(s2_lines_for_diff),
                        match.b + len(file_block_for_diff_lines) + context_window_lines,
                    )

                    closest_snippet_lines_with_context = s2_lines_for_diff[
                        actual_snippet_start_line_idx:actual_snippet_end_line_idx
                    ]
                    closest_actual_snippet_text = "".join(closest_snippet_lines_with_context)

                    snippet_start_human_line = actual_snippet_start_line_idx + 1
                    snippet_end_human_line = actual_snippet_end_line_idx
                    snippet_line_info = (
                        f"(lines approx. {snippet_start_human_line}-{snippet_end_human_line})"
                    )

                    error_message_parts = [
                        f"The SEARCH block for file '{file_path}' was not found.",
                        f"\nYour provided SEARCH block:\n---\n{find}\n---\n",
                        f"The closest text found in the file {snippet_line_info} was:\n---\n{closest_actual_snippet_text}\n---\n",  # noqa: E501
                        "Differences (+ your search / - actual file content / ? suggestions):\n---",
                        diff_output_text,
                        "---\n",
                        "Please review these differences carefully and provide a corrected SEARCH block.",  # noqa: E501
                    ]
                    error_message = "\n".join(error_message_parts)
                else:
                    error_message = (
                        f"The SEARCH block for file '{file_path}' was not found.\n\n"
                        f"Your provided SEARCH block:\n---\n{find}\n---\n\n"
                        "No close match could be identified in the file to provide a detailed diff."
                    )
                raise ValueError(error_message)

        with open(expanded_file_path, "w") as f:
            f.write(file_content)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=(
                    f"<system>Your edits have been applied to the file: {file_path}\n\n"
                    f"Here is the updated file content:\n\n<file_content>\n{file_content}\n</file_content>\n\n"  # noqa: E501
                    "Make sure to double check that the edits were successful after all edits are complete.  If your edit did not work as expected, please adjust your response to fix the issue and try another edit.  If you are unable to fix the issue, please try to WRITE the file from scratch instead, making sure to stay accurate to the original file content.</system>"  # noqa: E501
                ),
                ephemeral=True,
                ephemeral_steps=1,
            )
        )

        return CodeExecutionResult(
            stdout=f"Successfully edited file: {file_path}",
            stderr="",
            logging="",
            formatted_print=f"Successfully edited file: {file_path}",
            file_path=file_path,
            message="",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[str(expanded_file_path)],
            execution_type=ExecutionType.ACTION,
            action=ActionType.EDIT,
        )

    def _limit_conversation_history(self) -> None:
        """Limit the conversation history to the maximum number of messages."""

        # Limit in chunks of half the max conversation history to reduce
        # cache breaking
        chunk_size = self.max_conversation_history // 2

        if len(self.agent_state.conversation) - 1 > self.max_conversation_history:
            # Keep the first message (system prompt) and the most recent messages
            self.agent_state.conversation = [
                self.agent_state.conversation[0],
                ConversationRecord(
                    role=ConversationRole.USER,
                    content="<system>Some conversation history has been truncated for brevity.</system>",  # noqa: E501
                    should_summarize=False,
                ),
            ] + self.agent_state.conversation[-chunk_size:]

    async def _summarize_conversation_step(
        self, msg: ConversationRecord, min_token_threshold: int = 500
    ) -> str:
        """
        Summarize the conversation step by invoking the model to generate a concise summary,
        but only if the message content exceeds 500 tokens.

        Args:
            msg (ConversationRecord): The conversation record to summarize.
            min_token_threshold (int): The minimum number of tokens to consider summarizing.

        Returns:
            str: A concise summary of the critical information from the conversation step,
                 or the original content if it is already concise.

        Raises:
            ValueError: If the conversation record is not of the expected type.
        """
        # Calculate token count for the message content
        tokenizer = encoding_for_model("gpt-4o")

        token_count = len(tokenizer.encode(msg.content or ""))

        if token_count <= min_token_threshold:
            return msg.content

        step_info = (
            "Please summarize the following conversation step:\n"
            f"<role>{msg.role}</role>\n<message>{msg.content}</message>"
        )

        summary_history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=MessageSummarySystemPrompt,
                is_system_prompt=True,
                should_cache=True,
            ),
            ConversationRecord(role=ConversationRole.USER, content=step_info),
        ]

        response = await self.invoke_model(summary_history)
        return response.content if isinstance(response.content, str) else str(response.content)

    def set_tool_registry(self, tool_registry: ToolRegistry) -> None:
        """Set the tool registry for the current conversation."""
        self.tool_registry = tool_registry
        self.context["tools"] = tool_registry
        # Ensure builtins are always present in the context
        self.context["__builtins__"] = builtins

    def get_conversation_history(self) -> list[ConversationRecord]:
        """Get the conversation history as a list of dictionaries.

        Returns:
            list[ConversationRecord]: The conversation history as a list of ConversationRecord
        """
        return self.agent_state.conversation

    def remove_ephemeral_messages(self) -> None:
        """Remove ephemeral messages from the conversation history.

        Only removes ephemeral messages with ephemeral_steps=0. Messages with
        ephemeral_steps>0 will remain in the conversation history until their
        steps count reaches 0.
        """
        self.agent_state.conversation = [
            msg
            for msg in self.agent_state.conversation
            if not (msg.ephemeral and getattr(msg, "ephemeral_steps", 0) <= 0)
        ]

    def format_directory_tree(self, directory_index: Dict[str, List[Tuple[str, str, int]]]) -> str:
        """Format a directory index into a human-readable tree structure.

        Creates a formatted tree representation of files and directories with icons
        and human-readable file sizes.

        Args:
            directory_index: Dictionary mapping directoryths to lists of
                (filename, file_type, size) tuples

        Returns:
            str: Formatted directory tree string with icons, file types, and human-readable sizes

        Example:
            >>> index = {".": [("test.py", "code", 1024)]}
            >>> format_directory_tree(index)
            '📁 ./\n  📄 test.py (code, 1.0KB)\n'
        """
        # File type to icon mapping
        FILE_TYPE_ICONS = {
            "code": "📄",
            "doc": "📝",
            "image": "🖼️",
            "config": "🔑",
            "data": "📊",
            "other": "📎",
        }

        # Constants for limiting output
        MAX_FILES_PER_DIR = 30
        MAX_TOTAL_FILES = 300

        directory_tree_str = ""
        total_files = 0

        for path, files in directory_index.items():
            # Add directory name with forward slash
            directory_tree_str += f"📁 {path}/\n"

            # Add files under directory (limited to MAX_FILES_PER_DIR)
            file_list = list(files)
            shown_files = file_list[:MAX_FILES_PER_DIR]
            has_more_files = len(file_list) > MAX_FILES_PER_DIR

            for filename, file_type, size in shown_files:
                # Format size to be human readable
                size_str = self._format_file_size(size)

                # Get icon based on file type
                icon = FILE_TYPE_ICONS.get(file_type, "📎")

                # Add indented file info
                directory_tree_str += f"  {icon} {filename} ({file_type}, {size_str})\n"

                total_files += 1
                if total_files >= MAX_TOTAL_FILES:
                    directory_tree_str += "\n... and more files\n"
                    break

            if has_more_files:
                remaining_files = len(file_list) - MAX_FILES_PER_DIR
                directory_tree_str += f"  ... and {remaining_files} more files\n"

            if total_files >= MAX_TOTAL_FILES:
                break

        if total_files == 0:
            directory_tree_str = "No files in the current directory"

        return directory_tree_str

    def _format_file_size(self, size: int) -> str:
        """Convert file size in bytes to a human-readable format.

        Args:
            size: File size in bytes

        Returns:
            str: Human-readable file size (e.g., "1.5KB", "2.0MB")
        """
        if size < 1024:
            return f"{size}B"
        elif size < 1024 * 1024:
            return f"{size/1024:.1f}KB"
        else:
            return f"{size/(1024*1024):.1f}MB"

    def get_environment_details(self) -> str:
        """Get detailed information about the current execution environment.

        Collects and formats information about the current working directory,
        git repository status, directory structure, and available execution context
        variables.

        Returns:
            str: Formatted string containing environment details
        """
        try:
            cwd = os.getcwd()
        except FileNotFoundError:
            cwd = "Unknown or deleted directory, please move to a valid directory"

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        current_time_zone = datetime.now().astimezone().tzname()
        git_status = self._get_git_status()
        directory_tree = self.format_directory_tree(list_working_directory())

        return f"""
Current working directory: {cwd}
Current time: {current_time}
Current time zone: {current_time_zone}
<git_status>
{git_status}
</git_status>
<directory_tree>
{directory_tree}
</directory_tree>
        """

    def _get_git_status(self) -> str:
        """Get the current git repository status.

        Returns:
            str: Git status output, a message indicating no git repository, or that git
            is not installed
        """
        try:
            # Check if git is available on the system
            if sys.platform == "win32":
                # On Windows, use where command
                try:
                    path = subprocess.check_output(["where", "git"], stderr=subprocess.DEVNULL)
                    if not path:
                        return "Git is not available on this system"
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return "Git is not available on this system"
            else:
                # On Unix-like systems, use which command
                try:
                    path = subprocess.check_output(["which", "git"], stderr=subprocess.DEVNULL)
                    if not path:
                        return "Git is not available on this system"
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return "Git is not available on this system"

            # If git exists, check repository status
            try:
                return (
                    subprocess.check_output(["git", "status"], stderr=subprocess.DEVNULL)
                    .decode()
                    .strip()
                ) or "Clean working directory"
            except subprocess.CalledProcessError:
                return "Not a git repository"
        except (subprocess.CalledProcessError, FileNotFoundError):
            return "Git is not available on this system"

    def reset_learnings(self) -> None:
        """Reset the learnings list."""
        self.agent_state.learnings = []

    def add_to_learnings(self, learning: str) -> None:
        """Add a unique learning to the learnings list.

        Maintains a maximum number of unique learnings by removing the oldest learning
        when the list would exceed the maximum length. Prevents duplicate entries.

        Args:
            learning: The learning to add to the list, if it's not already present.
        """
        if not learning or learning in self.agent_state.learnings:
            return

        self.agent_state.learnings.append(learning)
        if len(self.agent_state.learnings) > self.max_learnings_history:
            self.agent_state.learnings.pop(0)

    def get_learning_details(self) -> str:
        """Get the learning details from the current conversation.

        Returns:
            str: Formatted string containing learning details
        """
        return "\n".join([f"- {learning}" for learning in self.agent_state.learnings])

    def get_available_agents_str(self) -> str:
        """Get the available agents for the current conversation.

        Returns:
            str: Formatted string containing available agents
        """
        if not self.agent_registry:
            return "No agents found."

        try:
            agents = self.agent_registry.list_agents()
        except AttributeError as e:
            raise AttributeError(
                "The provided agent_manager does not have a list_agents() method."
            ) from e
        except Exception as e:
            raise Exception(f"Failed to retrieve agents: {str(e)}") from e

        if not agents:
            return "No agents found."

        lines = []
        for agent in agents:
            name = getattr(agent, "name", "")
            description = getattr(agent, "description", "")

            if not name:
                # All agents should have a name
                continue

            if not description:
                # All agents should have a description
                continue

            lines.append(f"{name}: {description}")

        return "\n".join(lines)

    def get_agent_info(self) -> str:
        """Get the agent info for the current conversation.

        Returns:
            str: Formatted string containing agent info
        """
        if not self.agent:
            return ""

        agent_info = f"""
Agent ID: {self.agent.id}
Agent Name: {self.agent.name}
Agent Description: {self.agent.description}
Agent Created At: {self.agent.created_date}
"""
        return agent_info

    def update_ephemeral_messages(self) -> None:
        """Add environment details and other ephemeral messages to the conversation history.

        This method performs three main tasks:
        1. Decrements ephemeral_steps for ephemeral messages with ephemeral_steps>0
        2. Removes any messages marked as ephemeral with ephemeral_steps=0
        3. Appends the current environment details as a system message to provide context

        Ephemeral messages are identified by having an 'ephemeral' field set to 'true' in their
        dictionary representation. These messages are meant to be temporary and are removed
        before the next model invocation, unless they have ephemeral_steps>0, in which case
        they will remain for that many steps before being removed.

        The method updates self.agent_state.conversation in-place.
        """

        # Remove ephemeral messages with ephemeral_steps=0
        self.remove_ephemeral_messages()

        # Decrement ephemeral_steps for messages with ephemeral_steps>0
        for msg in self.agent_state.conversation:
            if (
                msg.ephemeral
                and hasattr(msg, "ephemeral_steps")
                and msg.ephemeral_steps is not None
                and msg.ephemeral_steps > 0
            ):
                msg.ephemeral_steps = msg.ephemeral_steps - 1

        self.add_ephemeral_hud_message()

    def create_hud_message(self) -> str:
        """Create the agent heads up display message.

        This method adds the agent heads up display to the conversation history.
        """
        # Add environment details to the latest message
        environment_details = self.get_environment_details()

        # Add learning details to the latest message
        learning_details = self.get_learning_details()

        # Add current plan details to the latest message
        current_plan_details = self.get_current_plan_details()

        # Add instruction details to the latest message
        instruction_details = self.get_instruction_details()

        # Add agent info such as ID, name, description, and created date
        agent_info = self.get_agent_info()

        # Add available agents to the latest message
        available_agents = self.get_available_agents_str()

        # Format active schedules
        active_schedules_details_list = []
        if self.agent_state.schedules:
            for schedule in self.agent_state.schedules:
                if schedule.is_active:
                    start_info = (
                        f" starting at {schedule.start_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                        if schedule.start_time_utc
                        else ""
                    )
                    end_info = (
                        f" ending at {schedule.end_time_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}"
                        if schedule.end_time_utc
                        else ""
                    )
                    active_schedules_details_list.append(
                        f'- ID: {schedule.id}, Prompt: "{schedule.prompt}", Runs every {schedule.interval} {schedule.unit.value}{start_info}{end_info}'  # noqa: E501
                    )

        active_schedules_details = (
            "\n".join(active_schedules_details_list)
            if active_schedules_details_list
            else "No active schedules."
        )

        # "Heads up display" for the agent
        hud_message = AgentHeadsUpDisplayPrompt.format(
            environment_details=environment_details,
            learning_details=learning_details,
            current_plan_details=current_plan_details,
            instruction_details=instruction_details,
            available_agents=available_agents,
            active_schedules_details=active_schedules_details,
            agent_info=agent_info,
        )

        return hud_message

    def add_ephemeral_hud_message(self) -> None:
        """Add the ephemeral HUD message to the conversation history.

        This method adds the ephemeral HUD message to the conversation history.
        """
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=f"<system>{self.create_hud_message()}</system>",
                ephemeral=True,
                ephemeral_steps=0,
            )
        )

    def set_current_plan(self, plan: str) -> None:
        """Set the current plan for the agent.

        Args:
            plan (str): The current plan for the agent
        """
        self.current_plan = plan

    def get_current_plan_details(self) -> str:
        """Get the current plan details for the agent.

        Returns:
            str: Formatted string containing current plan details
        """
        return self.agent_state.current_plan or ""

    def set_instruction_details(self, instruction_details: str) -> None:
        """Set the instruction details for the agent.

        Args:
            instruction_details (str): The instruction details for the agent
        """
        self.agent_state.instruction_details = instruction_details

    def get_instruction_details(self) -> str:
        """Get the instruction details for the agent.

        Returns:
            str: Formatted string containing instruction details
        """
        return self.agent_state.instruction_details or ""

    def add_to_code_history(
        self,
        execution_result: CodeExecutionResult,
        response: ResponseJsonSchema | None,
        classification: RequestClassification | None,
    ) -> CodeExecutionResult:
        """Add a code execution result to the code history.

        Args:
            execution_result (CodeExecutionResult): The execution result to add
            response (ResponseJsonSchema | None): The response from the model
            classification (RequestClassification | None): The classification of the task

        Returns:
            CodeExecutionResult: The updated code execution result
        """
        new_code_record = execution_result

        if response and not new_code_record.message:
            new_code_record.message = response.response

        if not new_code_record.timestamp:
            new_code_record.timestamp = datetime.now()

        if classification and not new_code_record.task_classification:
            new_code_record.task_classification = classification.type

        self.agent_state.execution_history.append(new_code_record)

        return new_code_record

    async def update_code_history(self, id: str, new_code_record: CodeExecutionResult) -> None:
        """Update the code history.

        Args:
            id (str): The id of the code execution result to update
            new_code_record (CodeExecutionResult): The new code execution result to
            update the code history with.
        """
        for index, record in enumerate(self.agent_state.execution_history):
            if record.id == id:
                self.agent_state.execution_history[index] = new_code_record
                break

    async def broadcast_message_update(self, id: str, new_code_record: CodeExecutionResult) -> None:
        """Broadcast the update via WebSocket if available.

        Args:
            id (str): The id of the code execution result to update
            new_code_record (CodeExecutionResult): The new code execution result to
            update the code history with.
        """
        try:
            if self.status_queue:
                self.status_queue.put(("message_update", id, new_code_record))
        except Exception as e:
            print(f"Failed to broadcast execution state update via WebSocket: {e}")

    async def update_job_execution_state(self, new_code_record: CodeExecutionResult) -> None:
        """Update the job execution state.

        Args:
            new_code_record (CodeExecutionResult): The new code execution result to
            update the job execution state with.
        """
        # Set streamable flag based on execution type
        if new_code_record.execution_type in [
            ExecutionType.PLAN,
            ExecutionType.REFLECTION,
            ExecutionType.RESPONSE,
        ]:
            new_code_record.is_streamable = True

        # Set complete flag based on status
        if new_code_record.status not in [
            ProcessResponseStatus.IN_PROGRESS,
            ProcessResponseStatus.NONE,
        ]:
            new_code_record.is_complete = True

        if self.job_id:
            # Update job execution state if job manager and job ID are provided
            try:
                # If we're in a multiprocessing context with a status queue
                if self.status_queue:
                    # Send execution state update through the queue to the parent process
                    self.status_queue.put(("execution_update", self.job_id, new_code_record))
            except Exception as e:
                print(f"Failed to update job execution state: {e}")

    def tool_execution_callback(self, tool_name: str, tool_result: Any) -> None:
        """Callback for tool execution.  Bridges the boundary between the job execution
        environment and the server environment.

        Args:
            tool_name (str): The name of the tool that was executed
            tool_result (str): The result of the tool execution
        """

        if tool_name == "schedule_task":
            if self.agent_registry and self.agent:
                self.agent_state.schedules.append(tool_result)
                self.agent_registry.save_agent_state(self.agent.id, self.agent_state)
        elif tool_name == "stop_schedule":
            if self.agent_registry and self.agent:
                self.agent_state.schedules = [
                    schedule
                    for schedule in self.agent_state.schedules
                    if schedule.id != tool_result
                ]
                self.agent_registry.save_agent_state(self.agent.id, self.agent_state)
        else:
            print(f"Unknown tool name: {tool_name}")
