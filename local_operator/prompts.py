import importlib.metadata
import inspect
import os
import platform
from pathlib import Path
from types import ModuleType


def get_installed_packages_str() -> str:
    """Get installed packages for the system prompt context."""

    # Filter to show only commonly used packages and require that the model
    # check for any other packages as needed.
    key_packages = {
        "numpy",
        "pandas",
        "torch",
        "tensorflow",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "requests",
        "pillow",
        "pip",
        "setuptools",
        "wheel",
        "langchain",
        "plotly",
        "scipy",
        "statsmodels",
        "tqdm",
    }

    installed_packages = [dist.metadata["Name"] for dist in importlib.metadata.distributions()]

    # Filter and sort with priority for key packages
    filtered_packages = sorted(
        (pkg for pkg in installed_packages if pkg.lower() in key_packages),
        key=lambda x: (x.lower() not in key_packages, x.lower()),
    )

    # Add count of non-critical packages
    other_count = len(installed_packages) - len(filtered_packages)
    package_str = ", ".join(filtered_packages[:15])  # Show first 15 matches
    if other_count > 0:
        package_str += f" + {other_count} others"

    return package_str


def get_tools_str(tools_module: ModuleType | None = None) -> str:
    """Get formatted string describing available tool functions.

    Args:
        tools_module: Optional module containing tool functions to document

    Returns:
        Formatted string describing the tools, or empty string if no tools module provided
    """
    if not tools_module:
        return ""

    # Get list of builtin functions/types to exclude
    builtin_names = set(dir(__builtins__))
    builtin_names.update(["dict", "list", "set", "tuple", "Path"])

    tools_list: list[str] = []
    for name in dir(tools_module):
        # Skip private functions and builtins
        if name.startswith("_") or name in builtin_names:
            continue

        tool = getattr(tools_module, name)
        if callable(tool):
            doc = tool.__doc__ or "No description available"
            # Get first line of docstring
            doc = doc.split("\n")[0].strip()

            sig = inspect.signature(tool)
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
            is_async = inspect.iscoroutinefunction(tool)
            async_prefix = "async " if is_async else ""

            tools_list.append(f"- {async_prefix}{name}({', '.join(args)}) -> {return_type}: {doc}")
    return "\n".join(tools_list)


def create_system_prompt(tools_module: ModuleType | None = None) -> str:
    """Create the system prompt for the agent."""

    base_system_prompt = BaseSystemPrompt
    user_system_prompt = Path.home() / ".local-operator" / "system_prompt.md"
    if user_system_prompt.exists():
        user_system_prompt = user_system_prompt.read_text()
    else:
        user_system_prompt = ""

    system_details = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "home_directory": os.path.expanduser("~"),
    }
    system_details_str = "\n".join(f"{key}: {value}" for key, value in system_details.items())

    installed_packages_str = get_installed_packages_str()

    base_system_prompt = (
        base_system_prompt.replace("{{system_details_str}}", system_details_str)
        .replace("{{installed_packages_str}}", installed_packages_str)
        .replace("{{user_system_prompt}}", user_system_prompt)
    )

    tools_str = get_tools_str(tools_module)
    base_system_prompt = base_system_prompt.replace("{{tools_str}}", tools_str)

    return base_system_prompt


BaseSystemPrompt: str = """
You are Local Operator – a secure Python agent that runs code locally using your filesystem, Python
environment, and internet access. Your mission is to autonomously achieve user goals with strict
safety and verification.

You are conversing with both a user and the system (which executes your code). Do not ask for
confirmation before running code; if the code is unsafe, the system will verify your intent.

Core Principles:
- 🔒 Pre-validate safety and system impact.
- 🐍 Use a single Python block per step (output via print()).
- 🔄 Chain steps using previous stdout/stderr.
- 📦 Environment: {{system_details_str}} | {{installed_packages_str}}
- 🛠️ Auto-install missing packages via subprocess.
- 🔍 Verify state/data with code execution.
- 📝 Plan your steps and verify your progress.
- 🤖 Run methods that don't require user input automatically.

Response Flow:
1. Generate minimal Python code for the current step.
2. Include pip installs if needed (check via importlib).
3. Print clear, human-readable verification.
4. Return an action:
   - CONTINUE: proceed to the next step.
   - CHECK: validate previous outputs.
   - DONE: finish or cancel the task.
   - ASK: request additional details.
   - BYE: end the session.

Tool Use:
Available functions:
<tools_list>
{{tools_str}}
</tools_list>
Import them from local_operator.tools. Use await for async functions (do not call asyncio.run()).
For Playwright, use its async version.

Additional User Info:
<user_system_prompt>
{{user_system_prompt}}
</user_system_prompt>

Critical Constraints:
- No combined steps or assumptions.
- Always check paths, network, and installs first.
- Never repeat questions.
- Use sys.executable for installs.
- Test and verify that you achieve the user's goal correctly.
- System code execution printing to console consumes tokens.  Do not print more than
  10000 tokens at once in the code output.

Response Format:
Respond strictly in JSON following this schema:
{
  "plan": "Set of goals to achieve the user's goal",
  "previous_step_success": true | false,
  "previous_goal": "Your goal from the previous step",
  "current_goal": "Your goal for the current step",
  "next_goal": "Your goal for the next step",
  "response": "Natural language response to the user's goal",
  "code": "Code to achieve the user's goal, must be valid Python code",
  "learnings": "Aggregated information learned so far from previous steps",
  "action": "CONTINUE | CHECK | DONE | ASK | BYE"
}
Include all fields (use empty values if not applicable) and no additional text.
"""

SafetyCheckSystemPrompt: str = """
You are a code safety and security checker.

You will be given a code snippet and asked to check if it contains any dangerous operations.

🚫 Respond "[UNSAFE]" if contains:
- High risk file deletion
- Suspicious package installs
- High risk system commands execution
- Sensitive system access
- Risky network operations

✅ Respond "[SAFE]" if no risks detected
"""

SafetyCheckUserPrompt: str = """
Please review the following code snippet and determine if it contains any dangerous operations:

{{code}}

🚫 The code is unsafe if it contains:
- High risk file deletion
- Suspicious package installs
- High risk system commands execution
- Sensitive system access
- Risky network operations

If the code is unsafe, respond with an analysis of the code risk and put [UNSAFE] at the end of
your response.

✅ Respond "[SAFE]" if no risks detected
"""
