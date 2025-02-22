import asyncio
import inspect
import io
import os
import sys
import traceback
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List

from langchain_community.callbacks.manager import get_openai_callback
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from tiktoken import encoding_for_model

from local_operator.agents import AgentData
from local_operator.console import (
    ExecutionSection,
    format_agent_output,
    format_error_output,
    format_success_output,
    log_action_error,
    log_error_and_retry_message,
    log_retry_error,
    print_agent_response,
    print_execution_section,
    print_task_interrupted,
    spinner,
)
from local_operator.model.configure import ModelConfiguration, calculate_cost
from local_operator.prompts import (
    SafetyCheckSystemPrompt,
    SafetyCheckUserPrompt,
    create_system_prompt,
)
from local_operator.tools import ToolRegistry
from local_operator.types import (
    ActionType,
    ConversationRecord,
    ConversationRole,
    ResponseJsonSchema,
)


class ExecutorInitError(Exception):
    """Raised when the executor fails to initialize properly."""

    def __init__(self, message: str = "Failed to initialize executor"):
        self.message = message
        super().__init__(self.message)


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
            markdown-style JSON code block delimiters (```json).

    Returns:
        ResponseJsonSchema: Validated response object containing the model's output.
            See ResponseJsonSchema class for the expected schema.

    Raises:
        ValidationError: If the JSON response does not match the expected schema.
    """
    response_content = response_str
    start_tag = "```json"
    end_tag = "```"

    start_index = response_content.find(start_tag)
    if start_index != -1:
        response_content = response_content[start_index + len(start_tag) :]

    if response_content.endswith(end_tag):
        response_content = response_content[: -len(end_tag)]

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


class LocalCodeExecutor:
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
    tool_registry: ToolRegistry | None

    """A class to handle local Python code execution with safety checks and context management.

    Attributes:
        context (dict): A dictionary to maintain execution context between code blocks
        conversation_history (list): A list of message dictionaries tracking the conversation
        model: The language model used for code analysis and safety checks
        step_counter (int): A counter to track the current step in sequential execution
        max_conversation_history (int): The maximum number of messages to keep in
            the conversation history.  This doesn't include the system prompt.
        detail_conversation_length (int): The number of messages to keep in full detail in the
            conversation history.  Every step before this except the system prompt will be
            summarized.
        interrupted (bool): Flag indicating if execution was interrupted
        can_prompt_user (bool): Informs the executor about whether the end user has access to the
            terminal (True), or is consuming the service from some remote source where they
            cannot respond via the terminal (False).
        token_metrics (ExecutorTokenMetrics): Tracks token usage and cost metrics for model
        executions
        agent (AgentData | None): The agent data for the current conversation
    """

    def __init__(
        self,
        model_configuration: ModelConfiguration,
        max_conversation_history: int = 100,
        detail_conversation_length: int = 10,
        can_prompt_user: bool = True,
        conversation_history: List[ConversationRecord] = [],
        agent: AgentData | None = None,
    ):
        """Initialize the LocalCodeExecutor with a language model.

        Args:
            model: The language model instance to use for code analysis
            max_conversation_history: The maximum number of messages to keep in
                the conversation history.  This doesn't include the system prompt.
            detail_conversation_length: The number of messages to keep in full detail in the
                conversation history.  Every step before this except the system prompt will be
                summarized.  Set to -1 to keep all messages in full detail.
            can_prompt_user: Informs the executor about whether the end user has access to the
                terminal (True), or is consuming the service from some remote source where they
                cannot respond via the terminal (False).
            conversation_history: A list of message dictionaries tracking the conversation.
            agent: The agent data for the current conversation.
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
        self.conversation_history.append(new_record)
        self._limit_conversation_history()

    async def _summarize_old_steps(self) -> None:
        """Summarize old conversation steps beyond the detail conversation length.
        Only summarizes steps that haven't been summarized yet."""
        if len(self.conversation_history) <= 1:  # Just system prompt or empty
            return

        if self.detail_conversation_length == -1:
            return

        # Calculate which messages need summarizing
        history_to_summarize = self.conversation_history[1 : -self.detail_conversation_length]

        for msg in history_to_summarize:
            # Skip messages that are already sufficiently concise/summarized
            if not msg.should_summarize:
                continue

            if msg.summarized:
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
        self, new_conversation_history: List[ConversationRecord] = []
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
        """
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

        This method handles invoking different types of language models with appropriate formatting:
        - For Anthropic models: Combines messages into a single string with role prefixes
        - For OpenAI reasoning models (o1/o3): Combines messages for chain-of-thought reasoning
        - For Google Gemini models: Converts system messages to human messages
        - For other models: Passes messages directly

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
                model_name = self.get_model_name()

                if "claude" in model_name:
                    # Anthropic models expect a single message, so combine the conversation history
                    combined_message = ""
                    for msg in messages:
                        role_prefix = (
                            "Human: "
                            if msg.role == ConversationRole.USER
                            else (
                                "Assistant: "
                                if msg.role == ConversationRole.ASSISTANT
                                else "System: "
                            )
                        )
                        combined_message += f"{role_prefix}{msg.content}\n\n"
                    combined_message = combined_message.strip()
                    response = await self._convert_and_invoke(
                        [ConversationRecord(role=ConversationRole.USER, content=combined_message)]
                    )
                else:
                    if "o1" in model_name or "o3" in model_name:
                        # OpenAI reasoning models (o1 and o3) expect a combined prompt
                        # for chain-of-thought reasoning.
                        combined_message = ""
                        for msg in messages:
                            role_prefix = (
                                "User: "
                                if msg.role == ConversationRole.USER
                                else (
                                    "Assistant: "
                                    if msg.role == ConversationRole.ASSISTANT
                                    else "System: "
                                )
                            )
                            combined_message += f"{role_prefix}{msg.content}\n\n"
                        combined_message = combined_message.strip()
                        response = await self._convert_and_invoke(
                            [
                                ConversationRecord(
                                    role=ConversationRole.USER, content=combined_message
                                )
                            ]
                        )
                    elif "gemini" in model_name or "mistral" in model_name:
                        # Convert system messages to human messages for Google Gemini
                        # or Mistral models.
                        for msg in messages[1:]:
                            if msg.role == ConversationRole.SYSTEM:
                                msg.role = ConversationRole.USER
                        response = await self._convert_and_invoke(messages)
                    else:
                        response = await self._convert_and_invoke(messages)

                return response

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

    async def check_code_safety(self, code: str) -> ConfirmSafetyResult:
        """Analyze code for potentially dangerous operations using the language model.

        Args:
            code (str): The Python code to analyze

        Returns:
            ConfirmSafetyResult: Result of the safety check
        """
        response: BaseMessage

        agent_security_prompt = self.agent.security_prompt if self.agent else ""

        if self.can_prompt_user:
            safety_prompt = SafetyCheckSystemPrompt.replace(
                "{{security_prompt}}", agent_security_prompt
            )

            safety_history = [
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content=safety_prompt,
                ),
                ConversationRecord(
                    role=ConversationRole.USER,
                    content=f"Determine a status for the following code:\n\n{code}",
                ),
            ]

            response = await self.invoke_model(safety_history)

            response_content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            return get_confirm_safety_result(response_content)

        # If we can't prompt the user, we need to use the conversation history to determine
        # if the user has previously indicated a decision.
        safety_prompt = SafetyCheckUserPrompt.replace("{{code}}", code).replace(
            "{{security_prompt}}", agent_security_prompt
        )
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=safety_prompt,
            )
        )
        response = await self.invoke_model(self.conversation_history)
        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        self.conversation_history.pop()

        safety_result = get_confirm_safety_result(response_content)

        if safety_result == ConfirmSafetyResult.UNSAFE:
            analysis = response_content.replace("[UNSAFE]", "").strip()
            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.ASSISTANT,
                    content=f"The code is unsafe. Here is an analysis of the code risk: {analysis}",
                )
            )
            return ConfirmSafetyResult.UNSAFE

        return safety_result

    async def execute_code(self, code: str, max_retries: int = 2) -> str:
        """Execute Python code with safety checks and context management.

        Args:
            code (str): The Python code to execute
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Execution result message or error message
        """
        # First check code safety
        safety_result = await self._check_and_confirm_safety(code)
        if safety_result == ConfirmSafetyResult.UNSAFE:
            return "Code execution canceled by user"
        elif safety_result == ConfirmSafetyResult.CONVERSATION_CONFIRM:
            return "Code execution requires further confirmation from the user"
        elif safety_result == ConfirmSafetyResult.OVERRIDE:
            print(
                "\n\033[1;33m⚠️  Warning: Code safety override applied based on user's security"
                " prompt\033[0m\n"
            )

        # Try initial execution
        try:
            return await self._execute_with_output(code)
        except Exception as initial_error:
            return await self._handle_execution_error(initial_error, max_retries)

    async def _check_and_confirm_safety(self, code: str) -> ConfirmSafetyResult:
        """Check code safety and get user confirmation if needed.

        Returns:
            ConfirmSafetyResult: Result of the safety check
        """
        safety_result = await self.check_code_safety(code)

        if safety_result == ConfirmSafetyResult.UNSAFE:
            if self.can_prompt_user:
                confirm = input(
                    "\n\033[1;33m⚠️  Warning: Potentially dangerous operation detected."
                    " Proceed? (y/n): \033[0m"
                )
                if confirm.lower() == "y":
                    return ConfirmSafetyResult.SAFE

                msg = (
                    "I've identified that this is a dangerous operation. "
                    "Let's stop this task for now, I will provide further instructions shortly. "
                    "Action DONE."
                )
                self.append_to_history(
                    ConversationRecord(
                        role=ConversationRole.USER,
                        content=msg,
                    )
                )
                return ConfirmSafetyResult.UNSAFE
            else:
                # If we can't prompt the user, we need to add our question to the conversation
                # history and end the task, waiting for the user's next input to determine
                # whether to execute or not.  On the next iteration, check_code_safety will
                # return a different value based on the user's response.
                msg = (
                    "I've identified that this is a potentially dangerous operation. "
                    "Do you want me to proceed, find another way, or stop this task?"
                )
                self.append_to_history(
                    ConversationRecord(
                        role=ConversationRole.ASSISTANT,
                        content=msg,
                    )
                )
                return ConfirmSafetyResult.CONVERSATION_CONFIRM
        return safety_result

    async def _execute_with_output(self, code: str) -> str:
        """Execute code and capture stdout/stderr output.

        Args:
            code (str): The Python code to execute
            timeout (int, optional): Maximum execution time in seconds. Defaults to 30.

        Returns:
            str: Formatted string containing execution output and any error messages

        Raises:
            Exception: Re-raises any exceptions that occur during code execution
        """
        old_stdout, old_stderr = sys.stdout, sys.stderr
        new_stdout, new_stderr = io.StringIO(), io.StringIO()
        sys.stdout, sys.stderr = new_stdout, new_stderr

        try:
            await self._run_code(code)
            output, error_output = self._capture_and_record_output(new_stdout, new_stderr)
            return format_success_output((output, error_output))
        except Exception as e:
            output, error_output = self._capture_and_record_output(new_stdout, new_stderr)
            raise e
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            new_stdout.close()
            new_stderr.close()

    async def _run_code(self, code: str) -> None:
        """Run code in the main thread.

        Args:
            code (str): The Python code to execute
            timeout (int): Unused parameter kept for compatibility

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
                        exec(async_code, self.context)
                        # Run the coroutine
                        await self.context["__temp_coro"]
                    finally:
                        # Clean up even if there was an error
                        if "__temp_async_fn" in self.context:
                            del self.context["__temp_async_fn"]
                        if "__temp_coro" in self.context:
                            try:
                                # Just try to await any remaining coroutine
                                await self.context["__temp_coro"]
                            except Exception:
                                pass  # Ignore errors from cleanup await
                            del self.context["__temp_coro"]
                else:
                    # Regular synchronous code
                    exec(code, self.context)
        except Exception as e:
            raise e
        finally:
            sys.stdin = old_stdin

    def _capture_and_record_output(
        self, stdout: io.StringIO, stderr: io.StringIO, format_for_ui: bool = False
    ) -> tuple[str, str]:
        """Capture stdout/stderr output and record it in conversation history.

        Args:
            stdout (io.StringIO): Buffer containing standard output
            stderr (io.StringIO): Buffer containing error output
            format_for_ui (bool): Whether to format the output for a UI chat
            interface.  This will include markdown formatting and other
            UI-friendly features.

        Returns:
            tuple[str, str]: Tuple containing (stdout output, stderr output)
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

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=f"Code execution output:\n{output}\n" f"Error output:\n{error_output}",
                should_summarize=True,
            )
        )

        return output, error_output

    async def _handle_execution_error(self, initial_error: Exception, max_retries: int) -> str:
        """Handle code execution errors with retry logic.

        Args:
            initial_error (Exception): The original error that occurred
            code (str): The Python code that failed
            max_retries (int): Maximum number of retry attempts

        Returns:
            str: Final execution output or formatted error message
        """
        self._record_initial_error(initial_error)
        log_error_and_retry_message(initial_error)

        for attempt in range(max_retries):
            try:
                new_code = await self._get_corrected_code()
                if new_code:
                    return await self._execute_with_output(new_code)
            except Exception as retry_error:
                self._record_retry_error(retry_error, attempt)
                log_retry_error(retry_error, attempt, max_retries)

        return format_error_output(initial_error, max_retries)

    def _record_initial_error(self, error: Exception) -> None:
        """Record the initial execution error, including the traceback, in conversation history.

        Args:
            error (Exception): The error that occurred during initial execution.
        """
        traceback_str = traceback.format_exc()

        msg = (
            f"The initial execution failed with error: {str(error)}.\n"
            f"Traceback:\n{traceback_str}\n"
            "Review the code and make corrections to run successfully."
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
            error (Exception): The error that occurred during the retry attempt.
            attempt (int): The current retry attempt number.
        """
        traceback_str = traceback.format_exc()
        msg = (
            f"The code execution failed with error (attempt {attempt + 1}): {str(error)}.\n"
            f"Traceback:\n{traceback_str}\n"
            "Please review and make corrections to the code to fix this error and try again."
        )
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.USER,
                content=msg,
                should_summarize=True,
            )
        )

    async def _get_corrected_code(self) -> str:
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

        return response_json.code

    async def process_response(self, response: ResponseJsonSchema) -> ProcessResponseOutput:
        """Process model response, extracting and executing any code blocks.

        Args:
            response (str): The model's response containing potential code blocks
        """
        # Phase 1: Check for interruption
        if self.interrupted:
            print_task_interrupted()
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

        # Phase 2: Display agent response
        formatted_response = format_agent_output(plain_text_response)
        print_agent_response(self.step_counter, formatted_response)
        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.ASSISTANT,
                content=response.model_dump_json(),
                should_summarize=True,
            )
        )

        result = await self.perform_action(response)

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
            return ProcessResponseOutput(
                status=ProcessResponseStatus.SUCCESS,
                message="Action completed",
            )

        print_execution_section(
            ExecutionSection.HEADER, step=self.step_counter, action=response.action
        )
        spinner_task = asyncio.create_task(spinner(f"Executing {str(response.action).lower()}"))

        result_message = ""

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
                    )

                    result_message = await self.write_file(file_path, content)
                    print_execution_section(
                        ExecutionSection.RESULT, content=result_message, action=response.action
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
                    )

                    result_message = await self.edit_file(file_path, replacements)

                    print_execution_section(
                        ExecutionSection.RESULT, content=result_message, action=response.action
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
                    )
                    result_message = await self.read_file(file_path)
                else:
                    raise ValueError("File path is required for READ action")

            else:
                code_block = response.code
                if code_block:
                    print_execution_section(
                        ExecutionSection.CODE, content=code_block, action=response.action
                    )

                    result_message = await self.execute_code(code_block)

                    if "code execution cancelled by user" in result_message:
                        return ProcessResponseOutput(
                            status=ProcessResponseStatus.CANCELLED,
                            message="Code execution cancelled by user",
                        )

                    print_execution_section(
                        ExecutionSection.RESULT, content=result_message, action=response.action
                    )
                elif response.action == ActionType.CHECK or response.action == ActionType.CODE:
                    raise ValueError('"code" field is required for CODE or CHECK actions')
        except Exception as e:
            log_action_error(e, str(response.action))
            self.append_to_history(
                ConversationRecord(
                    role=ConversationRole.SYSTEM,
                    content=f"Error: {str(e)}",
                    should_summarize=True,
                )
            )
        finally:
            spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass

        token_metrics = self.get_token_metrics()
        print_execution_section(
            ExecutionSection.TOKEN_USAGE,
            data={
                "prompt_tokens": token_metrics.total_prompt_tokens,
                "completion_tokens": token_metrics.total_completion_tokens,
                "cost": token_metrics.total_cost,
            },
            action=response.action,
        )

        print_execution_section(ExecutionSection.FOOTER, action=response.action)
        self.step_counter += 1

        # Phase 4: Summarize old conversation steps
        spinner_task = asyncio.create_task(spinner("Summarizing conversation"))
        try:
            await self._summarize_old_steps()
        finally:
            print("\n")  # New line for next spinner
            spinner_task.cancel()
            try:
                await spinner_task
            except asyncio.CancelledError:
                pass

        output = ProcessResponseOutput(
            status=ProcessResponseStatus.SUCCESS,
            message=result_message if result_message else "Action completed",
        )

        return output

    async def read_file(self, file_path: str) -> str:
        """Read the contents of a file and include line numbers and lengths.

        Args:
            file_path (str): The path to the file to read

        Returns:
            str: A message indicating the file has been read

        Raises:
            FileNotFoundError: If the file does not exist
            OSError: If there is an error reading the file
        """
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        file_content = ""
        for i, line in enumerate(lines):
            line_number = i + 1
            line_length = len(line.rstrip("\n"))
            file_content += f"{line_number:4d} | {line_length:4d} | {line}"

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=(
                    f"Contents of {file_path}:\n\nLine | Length | Content\n"
                    f"BEGIN\n{file_content}\nEND"
                ),
                should_summarize=True,
            )
        )

        return f"Successfully read file: {file_path}"

    async def write_file(self, file_path: str, content: str) -> str:
        """Write content to a file.

        Args:
            file_path (str): The path to the file to write
            content (str): The content to write to the file

        Returns:
            str: A message indicating the file has been written

        Raises:
            OSError: If there is an error writing to the file
        """
        with open(file_path, "w") as f:
            f.write(content)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=f"Successfully wrote to file: {file_path}",
                should_summarize=True,
            )
        )

        return f"Successfully wrote to file: {file_path}"

    async def edit_file(self, file_path: str, replacements: List[Dict[str, str]]) -> str:
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
        original_content = ""
        with open(file_path, "r") as f:
            original_content = f.read()

        for replacement in replacements:
            find = replacement["find"]
            replace = replacement["replace"]

            if find not in original_content:
                raise ValueError(f"Find string '{find}' not found in file {file_path}")

            original_content = original_content.replace(find, replace, 1)

        with open(file_path, "w") as f:
            f.write(original_content)

        self.append_to_history(
            ConversationRecord(
                role=ConversationRole.SYSTEM,
                content=f"Successfully edited file: {file_path}",
                should_summarize=True,
            )
        )
        return f"Successfully edited file: {file_path}"

    def _limit_conversation_history(self) -> None:
        """Limit the conversation history to the maximum number of messages."""
        if len(self.conversation_history) > self.max_conversation_history:
            # Keep the first message (system prompt) and the most recent messages
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[
                -self.max_conversation_history + 1 :
            ]

    async def _summarize_conversation_step(self, msg: ConversationRecord) -> str:
        """Summarize the conversation step by invoking the model to generate a concise summary.

        Args:
            step_number (int): The step number to summarize

        Returns:
            str: A concise summary of the critical information from this step
        """
        summary_prompt = """
        You are a conversation summarizer. Your task is to summarize what happened in the given
        conversation step in a single concise sentence. Focus only on capturing critical details
        that may be relevant for future reference, such as:
        - Key actions taken
        - Important changes made
        - Significant results or outcomes
        - Any errors or issues encountered

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

    def get_conversation_working_directory(self) -> Path | None:
        """Get the working directory from the conversation history.

        Searches through the conversation history in reverse order to find the most recent
        system message containing a working directory specification. This is used to maintain
        context about which directory the conversation is operating in.

        Returns:
            Path | None: The working directory path extracted from the most recent system
                        message that specifies it, or None if no working directory is found
                        in any system message.
        """
        for msg in reversed(self.conversation_history):
            if msg.role == ConversationRole.SYSTEM:
                content = msg.content
                lower_content = content.lower()
                if lower_content.startswith("current working directory:"):
                    return Path(lower_content.split("current working directory:", 1)[1].strip())
        return None

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
