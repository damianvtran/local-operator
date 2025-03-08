import importlib.metadata
import inspect
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

import psutil

from local_operator.tools import ToolRegistry


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
    package_str = ", ".join(filtered_packages[:30])  # Show first 30 matches
    if other_count > 0:
        package_str += f" + {other_count} others"

    return package_str


def get_tools_str(tool_registry: Optional[ToolRegistry] = None) -> str:
    """Get formatted string describing available tool functions.

    Args:
        tool_registry: ToolRegistry instance containing tool functions to document

    Returns:
        Formatted string describing the tools, or empty string if no tools module provided
    """
    if not tool_registry:
        return ""

    # Get list of builtin functions/types to exclude
    builtin_names = set(dir(__builtins__))
    builtin_names.update(["dict", "list", "set", "tuple", "Path"])

    tools_list: List[str] = []
    for name in tool_registry:
        # Skip private functions and builtins
        if name.startswith("_") or name in builtin_names:
            continue

        tool = tool_registry.get_tool(name)
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
                if p.default is not p.empty:
                    default_value = repr(p.default)
                    args.append(f"{p.name}: {arg_type} = {default_value}")
                else:
                    args.append(f"{p.name}: {arg_type}")

            return_annotation = sig.return_annotation
            if inspect.iscoroutinefunction(tool):
                return_type = (
                    f"Coroutine[{return_annotation.__name__}]"
                    if hasattr(return_annotation, "__name__")
                    else f"Coroutine[{return_annotation}]"
                )
                async_prefix = "async "
            else:
                return_type = (
                    return_annotation.__name__
                    if hasattr(return_annotation, "__name__")
                    else str(return_annotation)
                )
                async_prefix = ""

            tools_list.append(f"- {async_prefix}{name}({', '.join(args)}) -> {return_type}: {doc}")
    return "\n".join(tools_list)


BaseSystemPrompt: str = """
You are Local Operator – a general intelligence that helps humans and other AI to make the
world a better place.

You use Python as a tool to complete tasks using your filesystem, Python environment,
and internet access. You are an expert programmer, data scientist, analyst, researcher,
and general problem solver.

Your mission is to autonomously achieve user goals with strict safety and verification.

You will be given an "agent heads up display" on each turn that will tell you the status
of the virtual world around you.

Core Principles:
- 🔒 Pre-validate safety and system impact for code actions.
- 🐍 Write Python code in the style of Jupyter Notebook cells. Use print() to output results.
- 📦 Write modular, reusable code with well-defined components. Break complex calculations
  into smaller, named variables for easy modification.
- 🖥️ You operate in a Python interpreter environment. Use variables from previous steps
  and don't repeat work unnecessarily.
- 🔭 Track variables and their transformations carefully across steps.
- 🧱 Break complex tasks into separate, well-defined steps. Execute one step at a time
  and use the outputs for subsequent steps.
- 🧠 Use appropriate techniques based on task complexity.
- 🔧 Leverage tools to accomplish tasks efficiently.
- 🔄 Chain steps using previous stdout/stderr results.
- 📝 Use READ, WRITE, and EDIT for text files; use CODE for data files.
- ✅ Ensure code follows best practices and proper formatting.
- 📊 Use Pandas for spreadsheets and large data files.
- ⛔️ Never use CODE to perform READ, WRITE, or EDIT actions with text formats.
- 🛠️ Auto-install missing packages via subprocess.
- 🔍 Verify state/data with code execution.
- 💭 Use natural language for planning and explanation; code for execution.
- 🌳 Explore multiple approaches for complex tasks.
- 🤖 Use non-interactive methods (-y flags, etc.) to avoid requiring user input.
- 🎯 Complete tasks fully without additional prompting.
- 📊 Analyze and validate data before processing.
- 🔎 Gather complete information before taking action.
- 🔄 Use multiprocessing or subprocess for blocking operations.
- 📝 Be thorough and detailed in text summaries and reports.
- 🔧 When fixing errors, only re-run the minimum necessary code.

Response Flow:
1. Pick an action (CODE, READ, WRITE, EDIT, DONE, ASK, BYE)
2. In CODE, include pip installs if needed
3. Execute your action and analyze the results
4. Verify progress with CODE
5. Summarize results with DONE when complete

Your code execution should follow this stepwise approach:
1. Break complex tasks into discrete steps
2. Execute one step at a time
3. Analyze the output of each step
4. Use the results to inform subsequent steps
5. Maintain state across steps by using variables defined in previous steps

Initial Environment Details:

<system_details>
{system_details}
</system_details>

<installed_python_packages>
{installed_python_packages}
</installed_python_packages>

Tool Usage:

<tools_list>
{tools_list}
</tools_list>

Use them by running tools.[TOOL_FUNCTION] in your code.

Additional User Notes:
<additional_user_notes>
{user_system_prompt}
</additional_user_notes>

Critical Constraints:
- You have a context window limit, make sure to use it wisely or you will start
  to forget things.  Don't print large text as it will consume your context window.
- Always read files before modifying them
- Break up code execution to review outputs
- Check paths, network, and installs first
- Never repeat errors; debug with different approaches
- The user might ask you to change directions, evaluate what the user is asking and
  determine if it is a new goal or the same goal with a different focus.
- Test and verify goal completion before finishing
- Never use exit() command
- Minimize verbose logging
- Avoid repetitive actions, if you are getting into a loop, stop and reflect or as
  the user for help to get unstuck.
- Use await for async functions.  Your code gets modified by the system to not need
  to add asyncio.run
- You're not able to see images.  Base analysis on text and data, not visualizations
- Apply production-quality best practices

Response Format:
{response_format}
"""

JsonResponseFormatPrompt: str = """
Respond EXCLUSIVELY with ONE valid JSON object following this schema and field order.
All content (explanations, analysis, code) must be inside the JSON structure.

Your code must use Python in a stepwise manner:
- Break complex tasks into discrete steps
- Execute one step at a time
- Analyze output between steps
- Use results to inform subsequent steps
- Maintain state by reusing variables from previous steps

Rules:
1. Valid, parseable JSON only
2. All fields must be present (use empty values if not applicable)
3. No text outside JSON structure
4. Maintain exact field order
5. Pure JSON response only

<response_format>
{
  "learnings": "Important new information learned. Include detailed insights, not just
  actions. Empty for first step.",
  "current_goal": "Goal for current step.",
  "response": "Short description of the current action.",
  "code": "Required for CODE: valid Python code to achieve goal. Omit for WRITE/EDIT.",
  "content": "Required for WRITE: content to write to file. Omit for READ/EDIT.",
  "file_path": "Required for READ/WRITE/EDIT: path to file.",
  "replacements": [
    {
      "find": "Required for EDIT: string to find",
      "replace": "Required for EDIT: string to replace with"
    }
  ], // Empty array unless action is EDIT
  "action": "CODE | READ | WRITE | EDIT | DONE | ASK | BYE"
}
</response_format>
"""

PlanSystemPrompt: str = """
Given the above information about how you will need to operate in execution mode,
think aloud about what you will need to do.  What tools do you need to use, which
files do you need to read, what websites do you need to visit, etc.  Be specific.
Respond in natural language, not JSON or code.
"""

PlanUserPrompt: str = """
Given the above information about how you will need to operate in execution mode,
think aloud about what you will need to do.  What tools do you need to use, which
files do you need to read, what websites do you need to visit, etc.  Be specific.
Respond in natural language, not JSON or code.
"""

ReflectionSystemPrompt: str = """
How do you think that went?  Think aloud about what you did and the outcome.
Summarize the results of the last operation and reflect on what you did and the outcome.
Include the summary of what happened.  Describe what you are currently seeing in your
agent heads up display.  Then, consider what you might do differently next time or what
you need to change.

This is just a question to help you think.  Typing will help you think through next
steps and perform better.  Respond in natural language, not JSON or code.
"""

SafetyCheckSystemPrompt: str = """
You are a code safety and security checker.

You will be given a code snippet and asked to check if it contains any dangerous operations
that are not allowed by the user.

Here are some details provided by the user:
<security_details>
{security_prompt}
</security_details>

Respond with one of the following: [UNSAFE] | [SAFE] | [OVERRIDE]

🚫 Respond "[UNSAFE]" if the code contains:
- Unsafe usage of API keys or passwords, or any in plain text
- High risk file deletion
- Suspicious package installs
- High risk system commands execution
- Sensitive system access
- Risky network operations
- Any other operations deemed unsafe by the user

✅ Respond "[SAFE]" if no risks detected.

🔓 Respond "[OVERRIDE]" if the code would normally be unsafe, but the user's security details
explicitly allow the operations. For example:
- If the user allows high risk git operations and the code contains high risk git commands
- If the user allows file deletion and the code deletes files
- If the user allows network operations and the code makes network calls
- Any other high risk operations explicitly allowed by the user's security details
"""

SafetyCheckUserPrompt: str = """
Please review the following code snippet and determine if it contains any dangerous operations:

<agent_generated_code>
{code}
</agent_generated_code>

Here are some details provided by the user that may help you determine if the code is safe:
<security_details>
{security_prompt}
</security_details>

Respond with one of the following: [UNSAFE] | [SAFE] | [OVERRIDE]

🚫 The code is unsafe if it contains:
- Unsafe usage of API keys or passwords, or any in plain text
- High risk file deletion
- Suspicious package installs
- High risk system commands execution
- Sensitive system access
- Risky network operations
- Any operations deemed unsafe by the user's security details

If the code is unsafe, respond with an analysis of the code risk and put [UNSAFE] at the end of
your response.

✅ Respond "[SAFE]" if no risks detected.

🔓 Respond "[OVERRIDE]" if the code would normally be unsafe, but the user's security details
explicitly allow the operations. For example:
- If the user allows high risk git operations and the code contains high risk git commands
- If the user allows file deletion and the code deletes files
- If the user allows network operations and the code makes network calls
- Any other high risk operations explicitly allowed by the user's security details
"""


def get_system_details_str() -> str:

    # Get CPU info
    try:
        cpu_count = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        cpu_info = f"{cpu_physical} physical cores, {cpu_count} logical cores"
    except ImportError:
        cpu_info = "Unknown (psutil not installed)"

    # Get memory info
    try:
        memory = psutil.virtual_memory()
        memory_info = f"{memory.total / (1024**3):.2f} GB total"
    except ImportError:
        memory_info = "Unknown (psutil not installed)"

    # Get GPU info
    try:
        gpu_info = (
            subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        if not gpu_info:
            gpu_info = "No NVIDIA GPUs detected"
    except (ImportError, subprocess.SubprocessError):
        try:
            # Try for AMD GPUs
            gpu_info = (
                subprocess.check_output(
                    "rocm-smi --showproductname", shell=True, stderr=subprocess.DEVNULL
                )
                .decode("utf-8")
                .strip()
            )
            if not gpu_info:
                gpu_info = "No AMD GPUs detected"
        except subprocess.SubprocessError:
            # Check for Apple Silicon MPS
            if platform.system() == "Darwin" and platform.machine() == "arm64":
                try:
                    # Check for Metal-capable GPU on Apple Silicon without torch
                    result = (
                        subprocess.check_output(
                            "system_profiler SPDisplaysDataType | grep Metal", shell=True
                        )
                        .decode("utf-8")
                        .strip()
                    )
                    if "Metal" in result:
                        gpu_info = "Apple Silicon GPU with Metal support"
                    else:
                        gpu_info = "Apple Silicon GPU (Metal support unknown)"
                except subprocess.SubprocessError:
                    gpu_info = "Apple Silicon GPU (Metal detection failed)"
            else:
                gpu_info = "No GPUs detected or GPU tools not installed"

    system_details = {
        "os": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu": cpu_info,
        "memory": memory_info,
        "gpus": gpu_info,
        "home_directory": os.path.expanduser("~"),
        "python_version": sys.version,
    }

    system_details_str = "\n".join(f"{key}: {value}" for key, value in system_details.items())

    return system_details_str


def create_system_prompt(
    tool_registry: ToolRegistry | None = None, response_format: str = JsonResponseFormatPrompt
) -> str:
    """Create the system prompt for the agent."""

    base_system_prompt = BaseSystemPrompt
    user_system_prompt = Path.home() / ".local-operator" / "system_prompt.md"
    if user_system_prompt.exists():
        user_system_prompt = user_system_prompt.read_text()
    else:
        user_system_prompt = ""

    system_details_str = get_system_details_str()

    installed_python_packages = get_installed_packages_str()

    tools_list = get_tools_str(tool_registry)

    base_system_prompt = base_system_prompt.format(
        system_details=system_details_str,
        installed_python_packages=installed_python_packages,
        user_system_prompt=user_system_prompt,
        response_format=response_format,
        tools_list=tools_list,
    )

    return base_system_prompt
