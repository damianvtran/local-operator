import os
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
import asyncio


class LocalCodeExecutor:
    def __init__(self, model):
        self.context = {}
        self.conversation_history = []
        self.model = model

    def extract_code_blocks(self, text):
        """Extract Python code blocks marked with ```python``` syntax"""
        pattern = r"```python\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
        return matches

    async def check_code_safety(self, code):
        """Ask the model if the code contains dangerous operations"""
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

        # Add safety check to conversation history
        self.conversation_history.append({"role": "user", "content": safety_check_prompt})
        response = self.model.invoke(self.conversation_history)
        self.conversation_history.pop()  # Remove safety check from history

        return response.content.strip().lower() == "yes"

    async def execute_code(self, code):
        """Execute code in a safe environment and capture output"""
        try:
            # Check code safety with the model
            is_dangerous = await self.check_code_safety(code)
            if is_dangerous:
                confirm = input(
                    "Warning: Potentially dangerous operation detected. Proceed? (y/n): "
                )
                if confirm.lower() != "y":
                    return "Code execution canceled by user"

            # Execute code with access to context
            exec(code, self.context)
            return "Code executed successfully"
        except Exception as e:
            return f"Error executing code: {str(e)}"

    async def process_response(self, response):
        """Process model response, extract and execute code blocks"""
        print("\nModel Response:")
        print(response)

        # Add response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})

        # Extract and execute code blocks
        code_blocks = self.extract_code_blocks(response)
        if code_blocks:
            print("\nExecuting code blocks...")
            for code in code_blocks:
                print(f"\nExecuting:\n{code}")
                result = await self.execute_code(code)
                print(f"Result: {result}")

                # Add code execution result to conversation history
                self.conversation_history.append(
                    {"role": "system", "content": f"Code execution result:\n{result}"}
                )

                # Add result to context for next interaction
                self.context["last_code_result"] = result


class DeepSeekCLI:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in .env")

        # Use LangChain's ChatOpenAI implementation for DeepSeek
        self.model = ChatOpenAI(
            api_key=SecretStr(api_key),
            temperature=0.5,
            base_url="https://api.deepseek.com/v1",
            model="deepseek-chat",
        )
        self.executor = LocalCodeExecutor(self.model)

    async def chat(self):
        print("DeepSeek Local Code Executor CLI")
        print("You are interacting with a helpful CLI agent that can execute Python code locally.")
        print(
            "The agent will carefully analyze and execute code blocks, explaining any "
            "errors that occur."
        )
        print("Type 'exit' or 'quit' to quit\n")

        # Initialize system message only once
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
            user_input = input("You: ")
            if user_input.lower() == "exit" or user_input.lower() == "quit":
                break

            # Add user input to conversation history
            self.executor.conversation_history.append({"role": "user", "content": user_input})

            # Get model response using existing conversation history
            response = self.model.invoke(self.executor.conversation_history)

            # Process response and execute any code
            await self.executor.process_response(response.content)


if __name__ == "__main__":
    try:
        cli = DeepSeekCLI()

        asyncio.run(cli.chat())
    except Exception as e:
        print(f"Error: {str(e)}")
