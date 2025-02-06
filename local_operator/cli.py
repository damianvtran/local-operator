"""
Main entry point for the Local Operator CLI application.

This script initializes the DeepSeekCLI interface for interactive chat or,
when the "serve" subcommand is used, starts up the FastAPI server to handle HTTP requests.

The application uses asyncio for asynchronous operation and includes
error handling for graceful failure.

Example Usage:
    python main.py --hosting deepseek --model deepseek-chat
    python main.py --hosting openai --model gpt-4
    python main.py --hosting ollama --model llama2
    python main.py exec "write a hello world program" --hosting ollama --model llama2
"""

import argparse
import asyncio
import os
import traceback
from importlib.metadata import version
from pathlib import Path

import uvicorn

from local_operator.config import ConfigManager
from local_operator.credentials import CredentialManager
from local_operator.executor import LocalCodeExecutor
from local_operator.model import configure_model
from local_operator.operator import (
    ConversationRole,
    Operator,
    OperatorType,
    create_system_prompt,
)

CLI_DESCRIPTION = """
    Local Operator - An environment for agentic AI models to perform tasks on the local device.

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
    # Create parent parser with common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for verbose output",
    )

    # Main parser
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION, parents=[parent_parser])

    parser.add_argument(
        "--version",
        action="version",
        version=f"v{version('local-operator')}",
        help="Show program's version number and exit",
    )
    parser.add_argument(
        "--hosting",
        type=str,
        choices=[
            "deepseek",
            "openai",
            "anthropic",
            "ollama",
            "kimi",
            "alibaba",
            "google",
            "test",
        ],
        help="Hosting platform to use (deepseek, openai, anthropic, ollama, kimi, alibaba, "
        "google, test)",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., deepseek-chat, gpt-4o, qwen2.5:14b, "
        "claude-3-5-sonnet-20240620, moonshot-v1-32k, qwen-plus, gemini-2.0-flash, test-model)",
    )
    subparsers = parser.add_subparsers(dest="subcommand")

    # Credential command
    credential_parser = subparsers.add_parser(
        "credential",
        help="Manage API keys and credentials for different hosting platforms",
        parents=[parent_parser],
    )
    credential_parser.add_argument(
        "--key",
        type=str,
        required=True,
        help="Credential key to update (e.g., DEEPSEEK_API_KEY, "
        "OPENAI_API_KEY, ANTHROPIC_API_KEY, KIMI_API_KEY, ALIBABA_CLOUD_API_KEY, "
        "GOOGLE_AI_STUDIO_API_KEY)",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config", help="Manage configuration settings", parents=[parent_parser]
    )
    config_subparsers = config_parser.add_subparsers(dest="config_command")
    config_subparsers.add_parser(
        "create", help="Create a new configuration file", parents=[parent_parser]
    )

    # Serve command to start the API server
    serve_parser = subparsers.add_parser(
        "serve", help="Start the FastAPI server", parents=[parent_parser]
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address for the server (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for the server (default: 8080)",
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot reload for the server",
    )

    # Exec command for single execution mode
    exec_parser = subparsers.add_parser(
        "exec",
        help="Execute a single command without starting interactive mode",
        parents=[parent_parser],
    )
    exec_parser.add_argument(
        "command",
        type=str,
        help="The command to execute",
    )

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


def serve_command(host: str, port: int, reload: bool) -> int:
    """
    Start the FastAPI server using uvicorn.
    """
    print(f"Starting server at http://{host}:{port}")
    uvicorn.run("local_operator.server:app", host=host, port=port, reload=reload)
    return 0


def main() -> int:
    try:
        parser = build_cli_parser()
        args = parser.parse_args()

        os.environ["LOCAL_OPERATOR_DEBUG"] = "true" if args.debug else "false"

        if args.subcommand == "credential":
            return credential_command(args)
        elif args.subcommand == "config":
            if args.config_command == "create":
                return config_create_command()
            else:
                parser.error(f"Invalid config command: {args.config_command}")
        elif args.subcommand == "serve":
            # Use the provided host, port, and reload options for serving the API.
            return serve_command(args.host, args.port, args.reload)

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

        executor = LocalCodeExecutor(model_instance)

        operator = Operator(
            executor=executor,
            credential_manager=credential_manager,
            config_manager=config_manager,
            model_instance=model_instance,
            type=OperatorType.CLI,
        )

        # Start the async chat interface or execute single command
        if args.subcommand == "exec":
            system_prompt = create_system_prompt()
            operator.executor.conversation_history = [
                {"role": ConversationRole.SYSTEM.value, "content": system_prompt}
            ]
            message = asyncio.run(operator.handle_user_input(args.command))
            if message:
                print(message.response)
            return 0
        else:
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
