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


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    Returns:
        argparse.ArgumentParser: The CLI argument parser
    """
    parser = argparse.ArgumentParser(description="Local Operator CLI")
    parser.add_argument(
        "--hosting",
        type=str,
        choices=["deepseek", "openai", "anthropic", "ollama", "kimi", "alibaba"],
        default="deepseek",
        help="Hosting platform to use (deepseek, openai, anthropic, ollama, kimi, or alibaba)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Model to use (e.g., deepseek-chat, gpt-4o, qwen2.5:14b, "
        "claude-3-5-sonnet-20240620, moonshot-v1-32k, qwen-plus)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose output",
    )
    subparsers = parser.add_subparsers(dest="subcommand")
    credential_parser = subparsers.add_parser(
        "credential", help="Manage API keys and credentials for different hosting platforms"
    )
    credential_parser.add_argument(
        "--key",
        type=str,
        required=True,
        help="Credential key to update (e.g., DEEPSEEK_API_KEY, "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY, ALIBABA_CLOUD_API_KEY)",
    )
    return parser


def credential_command(args: argparse.Namespace) -> int:
    credential_manager = CredentialManager(Path.home() / ".local-operator")
    credential_manager.prompt_for_credential(args.key, reason="update requested")
    return 0


def main() -> int:
    try:
        parser = build_cli_parser()

        args = parser.parse_args()

        if args.subcommand == "credential":
            return credential_command(args)

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"

        config_dir = Path.home() / ".local-operator"
        credential_manager = CredentialManager(config_dir)

        model_instance = configure_model(args.hosting, args.model, credential_manager)

        if not model_instance:
            error_msg = (
                f"\n\033[1;31mError: Model not found for hosting: "
                f"{args.hosting} and model: {args.model}\033[0m"
            )
            print(error_msg)
            return -1

        operator = CliOperator(
            credential_manager=credential_manager,
            model_instance=model_instance,
        )

        # Start the async chat interface
        asyncio.run(operator.chat())

        return 0
    except Exception as e:
        print(f"\n\033[1;31mError: {str(e)}\033[0m")
        print("\033[1;34m╭─ Stack Trace ────────────────────────────────────\033[0m")
        traceback.print_exc()
        print("\033[1;34m╰──────────────────────────────────────────────────\033[0m")
        print("\n\033[1;33mPlease check your .env configuration and internet connection.\033[0m")
        return -1


if __name__ == "__main__":
    exit(main())
