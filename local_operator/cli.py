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
from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.model import configure_model

CLI_DESCRIPTION = """
    Local Operator CLI - An intelligent command-line environment for agentic
    AI models to perform tasks on the local device.

    Supports multiple hosting platforms including DeepSeek, OpenAI, Anthropic, Ollama, Kimi
    and Alibaba. Features include interactive chat, safe code execution,
    context-aware conversation history, and built-in safety checks.

    Configure your preferred model and hosting platform via command line arguments. Your
    configuration file is located at ~/.local-operator/config.yml and can be edited directly.
"""


def build_cli_parser() -> argparse.ArgumentParser:
    """
    Build and return the CLI argument parser.

    Returns:
        argparse.ArgumentParser: The CLI argument parser
    """
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)

    parser.add_argument(
        "--hosting",
        type=str,
        choices=["deepseek", "openai", "anthropic", "ollama", "kimi", "alibaba"],
        help="Hosting platform to use (deepseek, openai, anthropic, ollama, kimi, or alibaba)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., deepseek-chat, gpt-4o, qwen2.5:14b, "
        "claude-3-5-sonnet-20240620, moonshot-v1-32k, qwen-plus)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose output",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # Credential command
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

    # Config command
    config_parser = subparsers.add_parser("config", help="Manage configuration settings")
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_subparsers.add_parser("create", help="Create a new configuration file")

    return parser


def credential_command(args: argparse.Namespace) -> int:
    credential_manager = CredentialManager(Path.home() / ".local-operator")
    credential_manager.prompt_for_credential(args.key, reason="update requested")
    return 0


def config_create_command() -> int:
    """Create a new configuration file."""
    config_manager = ConfigManager(Path.home() / ".local-operator")
    config_manager._write_config(vars(config_manager.config))
    print("Created new configuration file at ~/.local-operator/config.yml")
    return 0


def main() -> int:
    try:
        parser = build_cli_parser()
        args = parser.parse_args()

        if args.subcommand == "credential":
            return credential_command(args)
        elif args.subcommand == "config":
            if args.config_command == "create":
                return config_create_command()
            else:
                parser.error(f"Invalid config command: {args.config_command}")

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"

        config_dir = Path.home() / ".local-operator"

        config_manager = ConfigManager(config_dir)
        credential_manager = CredentialManager(config_dir)

        # Override config with CLI args where provided
        config_manager.update_config_from_args(args)

        # Get final config values
        hosting = config_manager.get_config_value("hosting")
        model = config_manager.get_config_value("model_name")

        model_instance = configure_model(hosting, model, credential_manager)

        if not model_instance:
            error_msg = (
                f"\n\033[1;31mError: Model not found for hosting: "
                f"{hosting} and model: {model}\033[0m"
            )
            print(error_msg)
            return -1

        operator = CliOperator(
            credential_manager=credential_manager,
            config_manager=config_manager,
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
        print("\n\033[1;33mPlease review and correct the error to continue.\033[0m")
        return -1


if __name__ == "__main__":
    exit(main())
