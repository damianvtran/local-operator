import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr


class LocalCodeExecutor:
    """A class to handle local Python code execution with safety checks and context management.

    Attributes:
        context (dict): A dictionary to maintain execution context between code blocks
        conversation_history (list): A list of message dictionaries tracking the conversation
        model: The language model used for code analysis and safety checks
    """

    def __init__(self, model):
        """Initialize the LocalCodeExecutor with a language model.

        Args:
            model: The language model instance to use for code analysis
        """
        self.context = {}
        self.conversation_history = []
        self.model = model

    def extract_code_blocks(self, text):
        """Extract Python code blocks from text using markdown-style syntax.

        Args:
            text (str): The text containing potential code blocks

        Returns:
            list: A list of extracted code blocks as strings
        """
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    async def check_code_safety(self, code):
        """Analyze code for potentially dangerous operations using the language model.

        Args:
            code (str): The Python code to analyze

        Returns:
            bool: True if dangerous operations are detected, False otherwise
        """
        safety_check_prompt = f"""
        Analyze the following Python code for potentially dangerous operations:
        {code}

        Respond with only "yes" if the code contains dangerous operations that could:
        - Delete or modify files
        - Execute system commands
        - Access sensitive system resources
        - Perform network operations
        - Otherwise compromise system security

        Respond with only "no" if the code appears safe to execute.
        """

        self.conversation_history.append({"role": "user", "content": safety_check_prompt})
        response = self.model.invoke(self.conversation_history)
        self.conversation_history.pop()

        return response.content.strip().lower() == "yes"

    async def execute_code(self, code):
        """Execute Python code with safety checks and context management.

        Args:
            code (str): The Python code to execute

        Returns:
            str: Execution result message or error message
        """
        try:
            is_dangerous = await self.check_code_safety(code)
            if is_dangerous:
                confirm = input(
                    "Warning: Potentially dangerous operation detected. Proceed? (y/n): "
                )
                if confirm.lower() != "y":
                    return "Code execution canceled by user"

            exec(code, self.context)
            return "Code executed successfully"
        except Exception as e:
            return f"Error executing code: {str(e)}"

    async def process_response(self, response):
        """Process model response, extracting and executing any code blocks.

        Args:
            response (str): The model's response containing potential code blocks
        """
        print("\nModel Response:")
        print(response)

        self.conversation_history.append({"role": "assistant", "content": response})

        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            print("\nExecuting code blocks...")
            for code in code_blocks:
                print(f"\nExecuting:\n{code}")
                result = await self.execute_code(code)
                print(f"Result: {result}")

                self.conversation_history.append(
                    {"role": "system", "content": f"Code execution result:\n{result}"}
                )
                self.context["last_code_result"] = result


class CliOperator:
    """A command-line interface for interacting with DeepSeek's language model.

    Attributes:
        model: The configured ChatOpenAI instance for DeepSeek
        executor: LocalCodeExecutor instance for handling code execution
    """

    def __init__(self):
        """Initialize the CLI by loading environment variables and setting up the model."""
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env")

        self.model = ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.5,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        self.executor = LocalCodeExecutor(self.model)

    async def chat(self):
        """Run the interactive chat interface with code execution capabilities."""
        print("Local Executor Agent CLI")
        print(
            "You are interacting with a helpful CLI agent that can execute tasks locally "
            "on your device by running Python code."
        )
        print(
            "The agent will carefully analyze and execute code blocks, explaining any "
            "errors that occur."
        )
        print(
            "It will prompt you for confirmation before executing potentially dangerous "
            "or risky operations."
        )
        print("Type 'exit' or 'quit' to quit\n")

        self.executor.conversation_history = [
            {
                "role": "system",
                "content": "You are a Python code execution assistant. You strictly run "
                "Python code locally. You are able to run code on the local machine. "
                "Your functions: 1) Analyze and execute code blocks when requested 2) "
                "Validate code safety first 3) Explain code behavior and results 4) "
                "Never execute harmful code 5) Maintain secure execution. You only "
                "execute Python code.",
            }
        ]

        while True:
            user_input = input("\033[1m\033[94mYou:\033[0m \033[1m>\033[0m ")
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                break

            self.executor.conversation_history.append({"role": "user", "content": user_input})
            response = self.model.invoke(self.executor.conversation_history)
            await self.executor.process_response(response.content)
