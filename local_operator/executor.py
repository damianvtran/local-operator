import asyncio
import io
import os
import sys
from enum import Enum
from typing import Any, Dict, List

from langchain.schema import BaseMessage
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from local_operator.console import (
    ExecutionSection,
    format_agent_output,
    format_error_output,
    format_success_output,
    log_error_and_retry_message,
    log_retry_error,
    print_agent_response,
    print_execution_section,
    print_task_interrupted,
    spinner,
)
from local_operator.model import ModelType
from local_operator.prompts import SafetyCheckSystemPrompt, SafetyCheckUserPrompt
from local_operator.types import ConversationRole, ResponseJsonSchema


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
    CONVERSATION_CONFIRM = (
        "conversation_confirm"  # Safety needs to be confirmed in further conversation with the user
    )


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
    if response_content.startswith("```json"):
        response_content = response_content[7:]
    if response_content.endswith("```"):
        response_content = response_content[:-3]

    # Validate the JSON response
    response_json = ResponseJsonSchema.model_validate_json(response_content)

    return response_json


class LocalCodeExecutor:
    context: Dict[str, Any]
    conversation_history: List[Dict[str, str]]
    model: ModelType
    step_counter: int
    max_conversation_history: int
    detail_conversation_length: int
    interrupted: bool
    can_prompt_user: bool

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
    """

    def __init__(
        self,
        model: ModelType,
        max_conversation_history: int = 100,
        detail_conversation_length: int = 10,
        can_prompt_user: bool = True,
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
        """
        self.context = {}
        self.conversation_history = []
        self.model = model
        self.max_conversation_history = max_conversation_history
        self.detail_conversation_length = detail_conversation_length
        self.can_prompt_user = can_prompt_user
        self.reset_step_counter()
        self.interrupted = False

    def reset_step_counter(self):
        """Reset the step counter."""
        self.step_counter = 1

    def _append_to_history(
        self, role: ConversationRole, content: str, should_summarize: str = "True"
    ) -> None:
        """Append a message to conversation history and maintain length limit.

        Args:
            role (str): The role of the message sender (user/assistant/system)
            content (str): The message content
            should_summarize (str): Whether to summarize the message in the future.
            This can be set to False for messages that are already sufficiently
            summarized.
        """
        self.conversation_history.append(
            {
                "role": role.value,
                "content": content,
                "summarized": "False",
                "should_summarize": should_summarize,
            }
        )
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
            if msg.get("should_summarize") == "False":
                continue

            if msg.get("summarized") == "True":
                continue

            # Leave the user prompts intact
            if msg.get("role") == ConversationRole.USER.value:
                continue

            summary = await self._summarize_conversation_step(msg)
            msg["content"] = summary
            msg["summarized"] = "True"

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

    async def invoke_model(
        self, messages: List[Dict[str, str]], max_attempts: int = 3
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

        while attempt < max_attempts:
            try:
                if isinstance(self.model, ChatAnthropic):
                    # Anthropic models expect a single message, so combine the conversation history
                    combined_message = ""
                    for msg in messages:
                        role_prefix = (
                            "Human: "
                            if msg["role"] == ConversationRole.USER.value
                            else (
                                "Assistant: "
                                if msg["role"] == ConversationRole.ASSISTANT.value
                                else "System: "
                            )
                        )
                        combined_message += f"{role_prefix}{msg['content']}\n\n"
                    combined_message = combined_message.strip()
                    return await self.model.ainvoke(combined_message)
                elif isinstance(self.model, ChatOpenAI) and (
                    self.model.model_name.lower().startswith("o1")
                    or self.model.model_name.lower().startswith("o3")
                ):
                    # OpenAI reasoning models (o1 and o3) expect a combined prompt
                    # for chain-of-thought reasoning.
                    combined_message = ""
                    for msg in messages:
                        role_prefix = (
                            "User: "
                            if msg["role"] == ConversationRole.USER.value
                            else (
                                "Assistant: "
                                if msg["role"] == ConversationRole.ASSISTANT.value
                                else "System: "
                            )
                        )
                        combined_message += f"{role_prefix}{msg['content']}\n\n"
                    combined_message = combined_message.strip()
                    return await self.model.ainvoke(combined_message)
                elif isinstance(self.model, ChatGoogleGenerativeAI):
                    # Convert system messages to human messages for Google Gemini
                    for msg in messages[1:]:
                        if msg["role"] == ConversationRole.SYSTEM.value:
                            msg["role"] = ConversationRole.USER.value
                    return await self.model.ainvoke(messages)
                else:
                    return await self.model.ainvoke(messages)
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt < max_attempts:
                    await asyncio.sleep(1)  # Add a small delay between retries
                continue

        # If we've exhausted all attempts, raise the last error
        if last_error:
            raise last_error
        else:
            raise Exception("Failed to invoke model")

    async def check_code_safety(self, code: str) -> bool:
        """Analyze code for potentially dangerous operations using the language model.

        Args:
            code (str): The Python code to analyze

        Returns:
            bool: True if dangerous operations are detected, False otherwise
        """
        response: BaseMessage

        if self.can_prompt_user:
            safety_history = [
                {"role": ConversationRole.SYSTEM.value, "content": SafetyCheckSystemPrompt},
                {
                    "role": ConversationRole.USER.value,
                    "content": f"Determine if the following code is safe: {code}",
                },
            ]

            response = await self.invoke_model(safety_history)

            response_content = (
                response.content if isinstance(response.content, str) else str(response.content)
            )
            return "[UNSAFE]" in response_content

        # If we can't prompt the user, we need to use the conversation history to determine
        # if the user has previously indicated a decision.
        safety_prompt = SafetyCheckUserPrompt.replace("{{code}}", code)
        self._append_to_history(
            ConversationRole.USER,
            safety_prompt,
        )
        response = await self.invoke_model(self.conversation_history)
        response_content = (
            response.content if isinstance(response.content, str) else str(response.content)
        )
        self.conversation_history.pop()

        if "[UNSAFE]" in response_content:
            analysis = response_content.replace("[UNSAFE]", "").strip()
            self._append_to_history(
                ConversationRole.ASSISTANT,
                f"The code is unsafe. Here is an analysis of the code risk: {analysis}",
            )
            return True

        return False

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
        if await self.check_code_safety(code):
            if self.can_prompt_user:
                confirm = input(
                    "Warning: Potentially dangerous operation detected. Proceed? (y/n): "
                )
                if confirm.lower() == "y":
                    return ConfirmSafetyResult.SAFE

                msg = (
                    "I've identified that this is a dangerous operation. "
                    "Let's stop this task for now, I will provide further instructions shortly. "
                    "Action DONE."
                )
                self._append_to_history(ConversationRole.USER, msg)
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
                self._append_to_history(ConversationRole.ASSISTANT, msg)
                return ConfirmSafetyResult.CONVERSATION_CONFIRM
        return ConfirmSafetyResult.SAFE

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
            error_msg = (
                f"Code execution error:\n{str(e)}\n"
                f"Output:\n{output}\n"
                f"Error output:\n{error_output}"
            )
            self._append_to_history(ConversationRole.SYSTEM, error_msg)
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
        self, stdout: io.StringIO, stderr: io.StringIO
    ) -> tuple[str, str]:
        """Capture stdout/stderr output and record it in conversation history.

        Args:
            stdout (io.StringIO): Buffer containing standard output
            stderr (io.StringIO): Buffer containing error output

        Returns:
            tuple[str, str]: Tuple containing (stdout output, stderr output)
        """
        stdout.flush()
        stderr.flush()
        output = stdout.getvalue() or "[No output]"
        error_output = stderr.getvalue() or "[No error output]"

        self.context["last_code_output"] = output
        self.context["last_code_error"] = error_output
        self._append_to_history(
            ConversationRole.SYSTEM,
            f"Code execution output:\n{output}\nError output:\n{error_output}",
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
        """Record the initial execution error in conversation history.

        Args:
            error (Exception): The error that occurred during initial execution
        """
        msg = (
            f"The initial execution failed with error: {str(error)}. "
            "Review the code and make corrections to run successfully."
        )
        self._append_to_history(ConversationRole.USER, msg)

    def _record_retry_error(self, error: Exception, attempt: int) -> None:
        """Record retry attempt errors in conversation history.

        Args:
            error (Exception): The error that occurred during retry
            attempt (int): The current retry attempt number
        """
        msg = (
            f"The code execution failed with error (attempt {attempt + 1}): {str(error)}. "
            "Please review and make corrections to the code to fix this error and try again."
        )
        self._append_to_history(ConversationRole.USER, msg)

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
        return response_json.code

    async def process_response(self, response: ResponseJsonSchema) -> ProcessResponseOutput:
        """Process model response, extracting and executing any code blocks.

        Args:
            response (str): The model's response containing potential code blocks
        """
        # Phase 1: Check for interruption
        if self.interrupted:
            print_task_interrupted()
            self.interrupted = False
            return ProcessResponseOutput(
                status=ProcessResponseStatus.INTERRUPTED,
                message="Task interrupted by user",
            )

        plain_text_response = response.response

        # Phase 2: Display agent response
        formatted_response = format_agent_output(plain_text_response)
        print_agent_response(self.step_counter, formatted_response)
        self._append_to_history(ConversationRole.ASSISTANT, response.model_dump_json())

        # Extract code blocks from the agent response
        code_block = response.code
        if code_block:
            print_execution_section(ExecutionSection.HEADER, step=self.step_counter)
            print_execution_section(ExecutionSection.CODE, content=code_block)

            # Phase 3: Execute the code block
            spinner_task = asyncio.create_task(spinner("Executing code"))
            try:
                result = await self.execute_code(code_block)
            finally:
                spinner_task.cancel()
                try:
                    await spinner_task
                except asyncio.CancelledError:
                    pass

            if "code execution cancelled by user" in result:
                return ProcessResponseOutput(
                    status=ProcessResponseStatus.CANCELLED,
                    message="Code execution cancelled by user",
                )

            print_execution_section(ExecutionSection.RESULT, content=result)
            self.context["last_code_result"] = result

            print_execution_section(ExecutionSection.FOOTER)
            self._append_to_history(
                ConversationRole.SYSTEM,
                f"Current working directory: {os.getcwd()}",
                should_summarize="False",
            )
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

        return ProcessResponseOutput(
            status=ProcessResponseStatus.SUCCESS,
            message="Code execution complete",
        )

    def _limit_conversation_history(self) -> None:
        """Limit the conversation history to the maximum number of messages."""
        if len(self.conversation_history) > self.max_conversation_history:
            # Keep the first message (system prompt) and the most recent messages
            self.conversation_history = [self.conversation_history[0]] + self.conversation_history[
                -self.max_conversation_history + 1 :
            ]

    async def _summarize_conversation_step(self, msg: dict[str, str]) -> str:
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
            f"{msg['role']}: {msg['content']}"
        )

        summary_history = [
            {"role": ConversationRole.SYSTEM.value, "content": summary_prompt},
            {"role": ConversationRole.USER.value, "content": step_info},
        ]

        response = await self.invoke_model(summary_history)
        return response.content if isinstance(response.content, str) else str(response.content)
