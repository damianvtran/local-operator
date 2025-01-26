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
    python main.py --hosting deepseek --model deepseek-chat
    python main.py --hosting openai --model gpt-4
    python main.py --hosting ollama --model llama2
"""

from local_operator.agent import CliOperator
import asyncio
import argparse


def main():
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Local Operator CLI")
        parser.add_argument(
            "--hosting",
            type=str,
            choices=["deepseek", "openai", "ollama"],
            default="deepseek",
            help="Hosting platform to use (deepseek, openai, or ollama)",
        )
        parser.add_argument(
            "--model",
            type=str,
            default="deepseek-chat",
            help="Model to use (e.g., deepseek-chat, gpt-4o, qwen2.5:14b)",
        )
        args = parser.parse_args()

        # Initialize the CLI interface with hosting and model parameters
        operator = CliOperator(hosting=args.hosting, model=args.model)

        # Start the async chat interface
        asyncio.run(operator.chat())
    except Exception as e:
        # Handle any unexpected errors gracefully
        print(f"Error: {str(e)}")
        print("Please check your .env configuration and internet connection.")


if __name__ == "__main__":
    main()
