import asyncio
import inspect
import io
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from traceback import format_exception
from typing import Any, Dict, List, Tuple

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
    format_agent_output,
    format_error_output,
    format_success_output,
    log_action_error,
    log_retry_error,
    print_agent_response,
    print_execution_section,
    print_task_interrupted,
    spinner_context,
)
from local_operator.helpers import (
    clean_json_response,
    clean_plain_text_response,
    remove_think_tags,
)
from local_operator.model.configure import ModelConfiguration, calculate_cost
from local_operator.prompts import (
    SafetyCheckConversationPrompt,
    SafetyCheckSystemPrompt,
    SafetyCheckUserPrompt,
    create_system_prompt,
)
from local_operator.tools import ToolRegistry, list_working_directory
from local_operator.types import (
    ActionType,
    CodeExecutionResult,
    ConversationRecord,
    ConversationRole,
    ProcessResponseOutput,
    ProcessResponseStatus,
    ResponseJsonSchema,
)

MAX_FILE_READ_SIZE_BYTES = 1024 * 24
"""The maximum file size to read in bytes.

This is used to prevent reading large files into context, which can cause
context overflow errors for LLM APIs.
"""

FILE_WRITE_EQUIVALENT_TEMPLATE = """
write_file_content = \"\"\"{content}
\"\"\"

with open("{file_path}", "w") as f:
    f.write(write_file_content)

    print(f"Successfully wrote to file: {file_path}")
"""
"""
This template provides an equivalent code representation for a file write operation.

It's used to generate a code snippet that mirrors the action of writing content to a file,
so that it can be run in a notebook in the notebook export functionality.
"""

FILE_EDIT_EQUIVALENT_TEMPLATE = """
# Read the original content of the file
with open("{file_path}", "r") as f:
    original_content = f.read()

replacements = {replacements}

# Perform the replacements
for replacement in replacements:
    find = replacement["find"]
    replace = replacement["replace"]

    if find not in original_content:
        raise ValueError(f"Find string '{{find}}' not found in file {{file_path}}")

    original_content = original_content.replace(find, replace, 1)

# Write the modified content back to the file
with open("{file_path}", "w") as f:
    f.write(original_content)

print(f"Successfully edited file: {file_path}")
"""
"""
This template provides an equivalent code representation for a file edit operation.

It's used to generate a code snippet that mirrors the action of editing a file,
so that it can be run in a notebook in the notebook export functionality.
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


def process_json_response(response_str: str) -> ResponseJsonSchema:
    """Process and validate a JSON response string from the language model.

    Args:
        response_str (str): Raw response string from the model, which may be wrapped in
            markdown-style JSON code block delimiters (```json) or provided as a plain JSON object.

    Returns:
        ResponseJsonSchema: Validated response object containing the model's output.
            See ResponseJsonSchema class for the expected schema.

    Raises:
        ValidationError: If the JSON response does not match the expected schema.
        ValueError: If no valid JSON object can be extracted from the response.
    """
    response_content = clean_json_response(response_str)

    # Validate the JSON response
    response_json = ResponseJsonSchema.model_validate_json(response_content)

    return response_json


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

        if callable(value):
            try:
                doc = value.__doc__ or "No description available"
                # Get first line of docstring
                doc = doc.split("\n")[0].strip()

                sig = inspect.signature(value)
                args = []
                for p in sig.parameters.values():
                    arg_type = (
                        p.annotation.__name__
                        if hasattr(p.annotation, "__name__")
                        else str(p.annotation)
                    )
                    args.append(f"{p.name}: {arg_type}")

                return_type = (
                    sig.return_annotation.__name__
                    if hasattr(sig.return_annotation, "__name__")
                    else str(sig.return_annotation)
                )

                # Check if function is async
                is_async = inspect.iscoroutinefunction(value)
                async_prefix = "async " if is_async else ""

                formatted_value_str = (
                    f"{async_prefix}{key}({', '.join(args)}) -> {return_type}: {doc}"
                )
            except ValueError:
                formatted_value_str = value_str

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
    Attributes:
        context (Dict[str, Any]): A dictionary to maintain execution context between code blocks.
        conversation_history (List[ConversationRecord]): A list of
        message records tracking the conversation.
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
        token_metrics (ExecutorTokenMetrics): Tracks token usage and
        cost metrics for model executions.
        agent (AgentData | None): The agent data for the current conversation.
        agent_registry (AgentRegistry | None): The agent registry for the current conversation.
        tool_registry (ToolRegistry | None): The tool registry for the current conversation.
        learnings (List[str]): A list of learnings from the current conversation.
        current_plan (str | None): The current plan for the agent.
        code_history (List[CodeExecutionResult]): A list of code execution results.
        persist_conversation (bool): Whether to persist the conversation history and code
            execution history to the agent registry on each step.
        instruction_details (str | None): A set of instructions for the agent based on
            the classification of the user's request.
    """

    context: Dict[str, Any]
    conversation_history: List[ConversationRecord]
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
    learnings: List[str]
    current_plan: str | None
    code_history: List[CodeExecutionResult]
    persist_conversation: bool
    instruction_details: str | None

    def __init__(
        self,
        model_configuration: ModelConfiguration,
        max_conversation_history: int = 100,
        detail_conversation_length: int = 10,
        can_prompt_user: bool = True,
        conversation_history: List[ConversationRecord] = [],
        agent: AgentData | None = None,
        max_learnings_history: int = 50,
        verbosity_level: VerbosityLevel = VerbosityLevel.VERBOSE,
        agent_registry: AgentRegistry | None = None,
        persist_conversation: bool = False,
    ):
        """Initialize the LocalCodeExecutor with a language model.

        Args:
            model_configuration (ModelConfiguration): The model configuration to use
            max_conversation_history (int): The maximum number of messages to keep in
                the conversation history. This doesn't include the system prompt.
            detail_conversation_length (int): The number of messages to keep in full detail in the
                conversation history. Every step before this except the system prompt will be
                summarized. Set to -1 to keep all messages in full detail.
            can_prompt_user (bool): Informs the executor about whether
            the end user has access to the
                terminal (True), or is consuming the service from some remote source where they
                cannot respond via the terminal (False).
            conversation_history (List[ConversationRecord]): A list of
            message records tracking the conversation.
            agent (AgentData | None): The agent data for the current conversation.
            max_learnings_history (int): The maximum number of learnings to keep in history.
            verbosity_level (ExecutorVerbosityLevel): The level of detail to output
            from the executor.
            agent_registry (AgentRegistry | None): The agent registry for the current conversation.
            persist_conversation (bool): Whether to persist the conversation history and code
                execution history to the agent registry on each step.
        """
        self.context = {}
        self.model_configuration = model_configuration
        self.conversation_history = conversation_history
        self.max_conversation_history = max_conversation_history
        self.detail_conversation_length = detail_conversation_length
        self.can_prompt_user = can_prompt_user
        self.token_metrics = ExecutorTokenMetrics()
        self.agent = agent
        self.interrupted = False
        self.learnings = []
        self.current_plan = None
        self.max_learnings_history = max_learnings_history
        self.code_history = []
        self.verbosity_level = verbosity_level
        self.agent_registry = agent_registry
        self.persist_conversation = persist_conversation
        self.instruction_details = None

        # Load agent context if agent and agent_registry are provided
        if self.agent and self.agent_registry:
            try:
                agent_context = self.agent_registry.load_agent_context(self.agent.id)
                if agent_context is not None:
                    self.context = agent_context
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

        The method updates self.conversation_history in-place.
        """
        if not new_record.timestamp:
            new_record.timestamp = datetime.now()

        self.conversation_history.append(new_record)
        self._limit_conversation_history()

    async def _summarize_old_steps(self) -> None:
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
        if len(self.conversation_history) <= 1:  # Just system prompt or empty
            return

        if self.detail_conversation_length == -1:
            return

        # Calculate which messages need summarizing
        history_to_summarize = self.conversation_history[1 : -self.detail_conversation_length]

        for msg in history_to_summarize:
            # Skip messages that are already sufficiently concise/summarized
            if not msg.should_summarize or msg.summarized:
                continue

            summary = await self._summarize_conversation_step(msg)
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

        return sum(len(tokenizer.encode(entry.content)) for entry in messages)

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
            self.conversation_history = []

        if len(self.conversation_history) != 0:
            raise ValueError("Conversation history already initialized")

        system_prompt = create_system_prompt(self.tool_registry)

        history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=system_prompt,
                is_system_prompt=True,
            )
        ]

        if len(new_conversation_history) == 0:
            self.conversation_history = history
        else:
            # Remove the system prompt from the loaded history if it exists
            filtered_history = [
                record for record in new_conversation_history if not record.is_system_prompt
            ]
            self.conversation_history = history + filtered_history

    def load_conversation_history(self, new_conversation_history: List[ConversationRecord]) -> None:
        """Load a conversation history into the executor from a previous session.

        This method initializes the conversation history by prepending the system prompt
        and then appending the provided conversation history, excluding the initial system
        prompt from the loaded history (to avoid duplication).

        Args:
            new_conversation_history (List[ConversationRecord]): The conversation history to load,
                typically retrieved from a previous session. It is expected that the first record
                in this list is a system prompt, which will be replaced by the current
                system prompt.
        """
        system_prompt = create_system_prompt(self.tool_registry)

        history = [
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=system_prompt,
                is_system_prompt=True,
            )
        ]

        # Remove the system prompt from the loaded history if it exists
        filtered_history = [
            record for record in new_conversation_history if not record.is_system_prompt
        ]

        self.conversation_history = history + filtered_history

    def load_execution_history(self, new_execution_history: List[CodeExecutionResult]) -> None:
        """Load a code execution history into the executor from a previous session.

        This method initializes the code execution history by prepending the system prompt
        and then appending the provided code execution history, excluding the initial system
        prompt.
        """
        self.code_history = new_execution_history

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

    async def _convert_and_invoke(self, messages: List[ConversationRecord]) -> BaseMessage:
        """Convert the messages to a list of dictionaries and invoke the model.

        Args:
            messages (List[ConversationRecord]): A list of conversation records to send to the
            model.

        Returns:
            BaseMessage: The model's response.

        Raises:
            Exception: If there is an error during model invocation.
        """
        messages_list = [msg.dict() for msg in messages]
        model_instance = self.model_configuration.instance

        new_tokens_prompt = 0
        new_tokens_completion = 0

        # Use get_openai_callback for OpenAI models to track token usage and cost
        if isinstance(model_instance, ChatOpenAI):
            with get_openai_callback() as cb:
                response = await model_instance.ainvoke(messages_list)
                if cb is not None:
                    new_tokens_prompt = cb.prompt_tokens
                    new_tokens_completion = cb.completion_tokens
        else:
            # For other models, invoke the model directly
            new_tokens_prompt = self.get_invoke_token_count(messages)
            new_tokens_completion = 0
            response = await model_instance.ainvoke(messages_list)

        self.token_metrics.total_prompt_tokens += new_tokens_prompt
        self.token_metrics.total_completion_tokens += new_tokens_completion

        # Use the lookup table to get the cost per million tokens since the openai callback
        # doesn't always return cost information.
        self.token_metrics.total_cost += calculate_cost(
            self.model_configuration.info,
            new_tokens_prompt,
            new_tokens_completion,
        )

        return response

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
                return await self._convert_and_invoke(messages)
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
                continue

        # If we've exhausted all attempts, raise the last error
        if last_error:
            raise last_error
        else:
            raise Exception("Failed to invoke model")

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
                ),
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "Determine a status for the following agent generated JSON response:\n\n"
                        "<agent_generated_json_response>\n"
                        f"{response.model_dump_json()}\n"
                        "</agent_generated_json_response>"
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
            ),
        ]

        if len(self.conversation_history) + 1 > conversation_length:
            safety_check_conversation.append(
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=(
                        "Conversation truncated due to length, only showing the last few"
                        " messages in the conversation, which follow."
                    ),
                )
            )

            safety_check_conversation.extend(self.conversation_history[-conversation_length:])
        else:
            safety_check_conversation.extend(self.conversation_history[1:])

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
                        f"Your action was denied by the AI security auditor because it "
                        "was deemed unsafe. Here is an analysis of the code risk by"
                        " the security auditor AI agent:\n\n"
                        f"{analysis}\n\n"
                        "Please re-summarize the security risk in natural language and"
                        " not JSON format.  Don't acknowledge this message directly but"
                        " instead pretend that you are responding as the AI security"
                        " auditor directly to the user's request."
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
                )
            else:
                # The agent must read the security advisory and request confirmation
                # from the user to continue.
                safety_summary = await self.invoke_model(self.conversation_history)
                safety_summary_content = (
                    safety_summary.content
                    if isinstance(safety_summary.content, str)
                    else str(safety_summary.content)
                )

                safety_summary_content = remove_think_tags(safety_summary_content)
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

                log_retry_error(error, attempt, max_retries)

                self.update_ephemeral_messages()

                if attempt < max_retries - 1:
                    try:
                        new_response = await self._get_corrected_code()
                        if new_response:
                            current_response = new_response
                        else:
                            break
                    except Exception as retry_error:
                        log_retry_error(retry_error, attempt, max_retries)
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
            "I've identified that this is a dangerous operation. "
            "Let's stop the current task, I will provide further instructions shortly. "
            "Please await further instructions and use action DONE."
        )
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=msg,
            )
        )
        return ConfirmSafetyResult.UNSAFE

    async def _execute_with_output(self, response: ResponseJsonSchema) -> CodeExecutionResult:
        """Execute code and capture stdout/stderr output.

        Args:
            response (ResponseJsonSchema): The response from the language model

        Returns:
            CodeExecutionResult: The result of the code execution

        Raises:
            Exception: Re-raises any exceptions that occur during code execution
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        new_stdout, new_stderr = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = new_stdout, new_stderr

        # Get root logger and store original handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers.copy()
        original_level = root_logger.level

        # Create a custom handler that safely handles closed file errors
        class SafeStreamHandler(logging.StreamHandler[io.StringIO]):
            def emit(self, record):
                try:
                    super().emit(record)
                except ValueError as e:
                    if "I/O operation on closed file" not in str(e):
                        raise

        # Remove existing handlers and set new handler
        root_logger.handlers = []
        log_capture = io.StringIO()
        log_handler = SafeStreamHandler(log_capture)
        log_handler.setLevel(logging.WARNING)
        root_logger.addHandler(log_handler)
        root_logger.setLevel(logging.WARNING)

        # Also handle specific loggers that might cause issues (like Prophet)
        for logger_name in ["prophet", "cmdstanpy"]:
            specific_logger = logging.getLogger(logger_name)
            if specific_logger:
                specific_logger.handlers = []
                specific_logger.addHandler(log_handler)
                specific_logger.propagate = False

        try:
            await self._run_code(response.code)
            log_output = log_capture.getvalue()

            condensed_output, condensed_error_output, condensed_log_output = (
                self._capture_and_record_output(new_stdout, new_stderr, log_output)
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
            )
        except Exception as e:
            # Add captured log output to error output if any
            log_output = log_capture.getvalue()
            condensed_output, condensed_error_output, condensed_log_output = (
                self._capture_and_record_output(new_stdout, new_stderr, log_output)
            )
            raise e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            new_stdout.close()
            new_stderr.close()
            log_capture.close()

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
                    # Create an async function from the code
                    async_code = "async def __temp_async_fn():\n" + "\n".join(
                        f"    {line}" for line in code.split("\n")
                    )
                    # Add code to get and run the coroutine
                    async_code += "\n__temp_coro = __temp_async_fn()"
                    try:
                        # Execute the async function definition
                        compiled_code = compile(async_code, "<agent_generated_code>", "exec")
                        exec(compiled_code, self.context)
                        # Run the coroutine
                        await self.context["__temp_coro"]
                    finally:
                        # Clean up even if there was an error
                        if "__temp_async_fn" in self.context:
                            del self.context["__temp_async_fn"]
                        if "__temp_coro" in self.context:
                            del self.context["__temp_coro"]
                else:
                    # Regular synchronous code
                    compiled_code = compile(code, "<agent_generated_code>", "exec")
                    exec(compiled_code, self.context)
        except Exception as e:
            code_execution_error = CodeExecutionError(message=str(e), code=code).with_traceback(
                e.__traceback__
            )
            raise code_execution_error from None
        finally:
            sys.stdin = old_stdin

    def _capture_and_record_output(
        self, stdout: io.StringIO, stderr: io.StringIO, log_output: str, format_for_ui: bool = False
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

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=f"Here are the results of your last code execution:\n"
                f"<stdout>\n{condensed_output}\n</stdout>\n"
                f"<stderr>\n{condensed_error_output}\n</stderr>\n"
                f"<logger>\n{condensed_log_output}\n</logger>\n"
                "Please review the results and and determine what to do next.",
                should_summarize=True,
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
            f"The initial execution failed with an error.\n"
            f"{error_info}\n"
            "Debug the code you submitted and make all necessary corrections "
            "to fix the error and run successfully."
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
            f"The code execution failed with an error (attempt {attempt + 1}).\n"
            f"{error_info}\n"
            "Debug the code you submitted and make all necessary corrections "
            "to fix the error and run successfully.  Pick up from where you left "
            "off and try to avoid re-running code that has already succeeded.  "
            "Use the environment details to determine which variables are available "
            "and correct, which are not.  After fixing the issue please continue with the "
            "tasks according to the plan."
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
        response = await self.invoke_model(self.conversation_history)
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

    async def process_response(self, response: ResponseJsonSchema) -> ProcessResponseOutput:
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
                    content="Let's stop this task for now, I will provide further "
                    "instructions shortly.",
                    should_summarize=False,
                )
            )
            return ProcessResponseOutput(
                status=ProcessResponseStatus.INTERRUPTED,
                message="Task interrupted by user",
            )

        plain_text_response = response.response
        new_learnings = response.learnings

        self.add_to_learnings(new_learnings)

        # Phase 2: Display agent response
        formatted_response = format_agent_output(plain_text_response)

        print_agent_response(self.step_counter, formatted_response, self.verbosity_level)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.ASSISTANT,
                content=response.model_dump_json(),
                should_summarize=True,
            )
        )

        result = await self.perform_action(response)

        current_working_directory = os.getcwd()

        if self.persist_conversation and self.agent_registry and self.agent:
            self.agent_registry.update_agent_state(
                agent_id=self.agent.id,
                conversation_history=self.conversation_history,
                code_history=self.code_history,
                current_working_directory=current_working_directory,
                context=self.context,
            )

        return result

    async def perform_action(self, response: ResponseJsonSchema) -> ProcessResponseOutput:
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
            self.add_to_code_history(
                CodeExecutionResult(
                    stdout="",
                    stderr="",
                    logging="",
                    formatted_print=response.response,
                    code="",
                    message=response.response,
                    role=ConversationRole.ASSISTANT,
                    status=ProcessResponseStatus.SUCCESS,
                    files=[],
                ),
                response,
            )

            return ProcessResponseOutput(
                status=ProcessResponseStatus.SUCCESS,
                message="Action completed",
            )

        print_execution_section(
            ExecutionSection.HEADER,
            step=self.step_counter,
            action=response.action,
            verbosity_level=self.verbosity_level,
        )

        execution_result = None

        async with spinner_context(
            f"Executing {str(response.action).lower()}",
            verbosity_level=self.verbosity_level,
        ):
            try:
                if response.action == ActionType.WRITE:
                    file_path = response.file_path
                    content = response.content if response.content else response.code
                    if file_path:
                        print_execution_section(
                            ExecutionSection.WRITE,
                            file_path=file_path,
                            content=content,
                            action=response.action,
                            verbosity_level=self.verbosity_level,
                        )

                        # First check code safety
                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            execution_result = await self.write_file(file_path, content)

                        print_execution_section(
                            ExecutionSection.RESULT,
                            content=execution_result.formatted_print,
                            action=response.action,
                            verbosity_level=self.verbosity_level,
                        )
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
                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            execution_result = await self.edit_file(file_path, replacements)

                        print_execution_section(
                            ExecutionSection.RESULT,
                            content=execution_result.formatted_print,
                            action=response.action,
                            verbosity_level=self.verbosity_level,
                        )
                    else:
                        raise ValueError("File path and replacements are required for EDIT action")

                elif response.action == ActionType.READ:
                    file_path = response.file_path
                    if file_path:
                        print_execution_section(
                            ExecutionSection.READ,
                            file_path=file_path,
                            action=response.action,
                            verbosity_level=self.verbosity_level,
                        )

                        # First check code safety
                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            execution_result = await self.read_file(file_path)
                    else:
                        raise ValueError("File path is required for READ action")

                else:
                    code_block = response.code
                    if code_block:
                        print_execution_section(
                            ExecutionSection.CODE,
                            content=code_block,
                            action=response.action,
                            verbosity_level=self.verbosity_level,
                        )

                        # First check code safety
                        safety_result = await self.check_and_confirm_safety(response)
                        execution_result = await self.handle_safety_result(safety_result, response)

                        if not execution_result:
                            execution_result = await self.execute_code(response)

                        if "code execution cancelled by user" in execution_result.message:
                            return ProcessResponseOutput(
                                status=ProcessResponseStatus.CANCELLED,
                                message="Code execution cancelled by user",
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
                            f"There was an error encountered while trying to execute your action:"
                            f"\n\n{str(e)}"
                            "\n\nPlease adjust your response to fix the issue."
                        ),
                        should_summarize=True,
                    )
                )

        if execution_result:
            self.add_to_code_history(execution_result, response)

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

        return output

    async def read_file(
        self, file_path: str, max_file_size_bytes: int = MAX_FILE_READ_SIZE_BYTES
    ) -> CodeExecutionResult:
        """Read the contents of a file and include line numbers and lengths.

        Args:
            file_path (str): The path to the file to read
            max_file_size_bytes (int): The maximum file size to read in bytes

        Returns:
            str: A message indicating the file has been read

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file is too large to read
            OSError: If there is an error reading the file
        """
        expanded_file_path = Path(file_path).expanduser().resolve()

        if os.path.getsize(expanded_file_path) > max_file_size_bytes:
            raise ValueError(
                f"File is too large to use read action on: {file_path}\n"
                f"Please use code action to summarize and extract key features from "
                f"the file instead."
            )

        with open(expanded_file_path, "r", encoding="utf-8") as f:
            file_content = f.read()

        annotated_content = annotate_code(file_content)

        if not annotated_content:
            annotated_content = "[File is empty]"

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=(
                    f"Here are the contents of {file_path} with line numbers and lengths:\n"
                    f"\n"
                    f"Line | Length | Content\n"
                    f"----------------------\n"
                    f"BEGIN\n"
                    f"{annotated_content}\n"
                    f"END"
                ),
                should_summarize=True,
            )
        )

        return CodeExecutionResult(
            stdout=f"Successfully read file: {file_path}",
            stderr="",
            logging="",
            formatted_print=f"Successfully read file: {file_path}",
            code="",
            message="",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[str(expanded_file_path)],
        )

    async def write_file(self, file_path: str, content: str) -> CodeExecutionResult:
        """Write content to a file.

        Args:
            file_path (str): The path to the file to write
            content (str): The content to write to the file

        Returns:
            str: A message indicating the file has been written

        Raises:
            OSError: If there is an error writing to the file
        """
        # Remove code block markers if present
        content_lines = content.split("\n")
        if len(content_lines) > 0 and content_lines[0].startswith("```"):
            content_lines = content_lines[1:]
        if len(content_lines) > 0 and content_lines[-1].startswith("```"):
            content_lines = content_lines[:-1]

        cleaned_content = "\n".join(content_lines)

        expanded_file_path = Path(file_path).expanduser().resolve()

        with open(expanded_file_path, "w") as f:
            f.write(cleaned_content)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=f"The content that you requested has been written to {file_path}.",
                should_summarize=True,
            )
        )

        equivalent_code = FILE_WRITE_EQUIVALENT_TEMPLATE.format(
            file_path=file_path, content=cleaned_content
        )

        return CodeExecutionResult(
            stdout=f"Successfully wrote to file: {file_path}",
            stderr="",
            logging="",
            formatted_print=f"Successfully wrote to file: {file_path}",
            code=equivalent_code,
            message="",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[str(expanded_file_path)],
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

            if find not in file_content:
                raise ValueError(f"Find string '{find}' not found in file {file_path}")

            file_content = file_content.replace(find, replace, 1)

        with open(expanded_file_path, "w") as f:
            f.write(file_content)

        equivalent_code = FILE_EDIT_EQUIVALENT_TEMPLATE.format(
            file_path=file_path, replacements=json.dumps(replacements)
        )

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=(
                    f"Your edits have been applied to the file: {file_path}\n\n"
                    "Here are the contents of the edited file with line numbers and "
                    "lengths, please review and determine if your edit worked as expected:\n\n"
                    "Line | Length | Content\n"
                    "----------------------\n"
                    "BEGIN\n"
                    f"{annotate_code(file_content)}\n"
                    "END"
                ),
                should_summarize=True,
            )
        )

        return CodeExecutionResult(
            stdout=f"Successfully edited file: {file_path}",
            stderr="",
            logging="",
            formatted_print=f"Successfully edited file: {file_path}",
            code=equivalent_code,
            message="",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
            files=[str(expanded_file_path)],
        )

    def _limit_conversation_history(self) -> None:
        """Limit the conversation history to the maximum number of messages."""
        if len(self.conversation_history) - 1 > self.max_conversation_history:
            # Keep the first message (system prompt) and the most recent messages
            self.conversation_history = [
                self.conversation_history[0],
                ConversationRecord(
                    role=ConversationRole.USER,
                    content="[Some conversation history has been truncated for brevity]",
                    should_summarize=False,
                ),
            ] + self.conversation_history[-self.max_conversation_history :]

    async def _summarize_conversation_step(self, msg: ConversationRecord) -> str:
        """
        Summarize the conversation step by invoking the model to generate a concise summary.

        Args:
            msg (ConversationRecord): The conversation record to summarize.

        Returns:
            str: A concise summary of the critical information from the conversation step.
                 The summary includes key actions, important changes, significant results,
                 errors or issues, key identifiers, transformations, and data structures.

        Raises:
            ValueError: If the conversation record is not of the expected type.
        """
        summary_prompt = """
        You are a conversation summarizer. Your task is to summarize what happened in the given
        conversation step in a single concise sentence. Focus only on capturing critical details
        that may be relevant for future reference, such as:
        - Key actions taken
        - Important changes made
        - Significant results or outcomes
        - Any errors or issues encountered
        - Key variable names, headers, or other identifiers
        - Transformations or calculations performed that need to be remembered for
          later reference
        - Shapes and dimensions of data structures
        - Key numbers or values

        Format your response as a single sentence with the format:
        "[SUMMARY] {summary}"
        """

        step_info = "Please summarize the following conversation step:\n" + "\n".join(
            f"{msg.role}: {msg.content}"
        )

        summary_history = [
            ConversationRecord(role=ConversationRole.SYSTEM, content=summary_prompt),
            ConversationRecord(role=ConversationRole.USER, content=step_info),
        ]

        response = await self.invoke_model(summary_history)
        return response.content if isinstance(response.content, str) else str(response.content)

    def set_tool_registry(self, tool_registry: ToolRegistry) -> None:
        """Set the tool registry for the current conversation."""
        self.tool_registry = tool_registry
        self.context["tools"] = tool_registry

    def get_conversation_history(self) -> list[ConversationRecord]:
        """Get the conversation history as a list of dictionaries.

        Returns:
            list[ConversationRecord]: The conversation history as a list of ConversationRecord
        """
        return self.conversation_history

    def remove_ephemeral_messages(self) -> None:
        """Remove ephemeral messages from the conversation history."""
        self.conversation_history = [msg for msg in self.conversation_history if not msg.ephemeral]

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
        git_status = self._get_git_status()
        directory_tree = self.format_directory_tree(list_working_directory())
        context_vars = get_context_vars_str(self.context)

        return f"""
<environment_details>
Current working directory: {cwd}
Current time: {current_time}
<git_status>
{git_status}
</git_status>
<directory_tree>
{directory_tree}
</directory_tree>
<execution_context_variables>
{context_vars}
</execution_context_variables>
</environment_details>
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
        self.learnings = []

    def add_to_learnings(self, learning: str) -> None:
        """Add a unique learning to the learnings list.

        Maintains a maximum number of unique learnings by removing the oldest learning
        when the list would exceed the maximum length. Prevents duplicate entries.

        Args:
            learning: The learning to add to the list, if it's not already present.
        """
        if not learning or learning in self.learnings:
            return

        self.learnings.append(learning)
        if len(self.learnings) > self.max_learnings_history:
            self.learnings.pop(0)

    def get_learning_details(self) -> str:
        """Get the learning details from the current conversation.

        Returns:
            str: Formatted string containing learning details
        """
        template = f"""
        <learning_details>
        {"\n".join([f"- {learning}" for learning in self.learnings])}
        </learning_details>
        """
        return template

    def update_ephemeral_messages(self) -> None:
        """Add environment details and other ephemeral messages to the conversation history.

        This method performs two main tasks:
        1. Removes any messages marked as ephemeral (temporary) from the conversation history
        2. Appends the current environment details as a system message to provide context

        Ephemeral messages are identified by having an 'ephemeral' field set to 'true' in their
        dictionary representation. These messages are meant to be temporary and are removed
        before the next model invocation.

        The method updates self.executor.conversation_history in-place.
        """

        # Remove ephemeral messages from conversation history
        self.remove_ephemeral_messages()

        # Add environment details to the latest message
        environment_details = self.get_environment_details()

        # Add learning details to the latest message
        learning_details = self.get_learning_details()

        # Add current plan details to the latest message
        current_plan_details = self.get_current_plan_details()

        # Add instruction details to the latest message
        instruction_details = self.get_instruction_details()

        # "Heads up display" for the agent
        hud_message = f"""
# Agent Heads Up Display

This is your "heads up display" to help you understand the current state of the
conversation and the environment.  It is a message that is ephemeral and moves up
closer to the top of the conversation history to give you the most relevant information
at each point in time as you complete each task.  It will update and move forward after
each action.

Use this information to help you complete the user's request.

## Environment Details
This is information about the files, variables, and other details about the current
state of the environment.  Use these in this and future steps as needed instead of
re-writing code.

### About Environment Details
- git_status: this is the current git status of the working directory
- directory_tree: this is a tree of the current working directory.  You can use this
  to see what files and directories are available to you right here.
- execution_context_variables: this is a list of variables that are available for use
  in the current execution context.  You can use them in this step or future steps in
  the python code that you write to complete tasks.

{environment_details}

## Learning Details
This is a notepad of things that you have learned from previous conversations.
{learning_details}

## Current Plan
This is the current and original plan that you made based on the user's request.
Follow it closely and accurately and make sure that you are making progress towards it.
{current_plan_details}

## Instruction Details
This is a set of guidelines about how to best complete the current task or respond to
the user's request.  You should take them into account as you work on the current task.
{instruction_details}
        """

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=hud_message,
                should_summarize=False,
                ephemeral=True,
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
        template = f"""
        <current_plan>
        {self.current_plan}
        </current_plan>
        """
        return template

    def set_instruction_details(self, instruction_details: str) -> None:
        """Set the instruction details for the agent.

        Args:
            instruction_details (str): The instruction details for the agent
        """
        self.instruction_details = instruction_details

    def get_instruction_details(self) -> str:
        """Get the instruction details for the agent.

        Returns:
            str: Formatted string containing instruction details
        """
        template = f"""
        <instruction_details>
        {self.instruction_details}
        </instruction_details>
        """
        return template

    def add_to_code_history(
        self, execution_result: CodeExecutionResult, response: ResponseJsonSchema | None
    ) -> None:
        """Add a code execution result to the code history.

        Args:
            execution_result (CodeExecutionResult): The execution result to add
            response (ResponseJsonSchema | None): The response from the model
        """
        new_code_record = execution_result

        if response and not new_code_record.message:
            new_code_record.message = response.response

        if not new_code_record.timestamp:
            new_code_record.timestamp = datetime.now()

        self.code_history.append(new_code_record)
