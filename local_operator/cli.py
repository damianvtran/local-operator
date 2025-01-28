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

import argparse
import asyncio
import os
import traceback
from pathlib import Path

from local_operator.agent import CliOperator
from local_operator.credentials import CredentialManager
from local_operator.model import configure_model


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
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode for verbose output",
        )

        args = parser.parse_args()

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"

        config_dir = Path.home() / ".local-operator"
        credential_manager = CredentialManager(config_dir)

        model_instance = configure_model(args.hosting, args.model, credential_manager)

        # Initialize the CLI interface with hosting and model parameters
        operator = CliOperator(
            hosting=args.hosting,
            model=args.model,
            credential_manager=credential_manager,
            model_instance=model_instance,
        )

        # Start the async chat interface
        asyncio.run(operator.chat())
    except Exception as e:
        print(f"\n\033[1;31mError: {str(e)}\033[0m")
        print("\033[1;34m╭─ Stack Trace ────────────────────────────────────\033[0m")
        traceback.print_exc()
        print("\033[1;34m╰──────────────────────────────────────────────────\033[0m")
        print("\n\033[1;33mPlease check your .env configuration and internet connection.\033[0m")


if __name__ == "__main__":
    main()
