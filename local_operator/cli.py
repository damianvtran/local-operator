"""
Main entry point for the Local Operator CLI application.

This script initializes and runs the DeepSeekCLI interface, which provides:
- Interactive chat with AI assistant
- Safe execution of Python code blocks
- Context-aware conversation history
- Built-in safety checks for code execution

The application uses asyncio for asynchronous operation and includes
error handling for graceful failure.

Example Usage:
    python main.py
"""

from local_operator.agent import CliOperator
import asyncio


def main():
    try:
        # Initialize the CLI interface
        operator = CliOperator()

        # Start the async chat interface
        asyncio.run(operator.chat())
    except Exception as e:
        # Handle any unexpected errors gracefully
        print(f"Error: {str(e)}")
        print("Please check your .env configuration and internet connection.")


if __name__ == "__main__":
    main()
