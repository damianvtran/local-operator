import os
import platform
import subprocess
from typing import Optional
from unittest.mock import patch

import psutil

from local_operator.prompts import (
    ActionResponseFormatPrompt,
    apply_attachments_to_prompt,
    create_system_prompt,
    get_system_details_str,
    get_tools_str,
)
from local_operator.tools import ToolRegistry


def test_create_system_prompt():
    # Mock system details
    mock_system = {
        "system": "TestOS",
        "release": "1.0",
        "version": "1.0.0",
        "machine": "x86_64",
        "processor": "Intel",
    }

    mock_home = "/home/test"
    mock_packages = "numpy, pandas + 10 others"

    with (
        patch.multiple(
            platform,
            system=lambda: mock_system["system"],
            release=lambda: mock_system["release"],
            version=lambda: mock_system["version"],
            machine=lambda: mock_system["machine"],
            processor=lambda: mock_system["processor"],
        ),
        patch("os.path.expanduser", return_value=mock_home),
        patch("local_operator.prompts.get_installed_packages_str", return_value=mock_packages),
        patch("pathlib.Path.exists", return_value=False),
    ):

        result = create_system_prompt(
            tool_registry=None,
            response_format=ActionResponseFormatPrompt,
            agent_system_prompt="Test agent system prompt",
        )

        # Verify system details are included
        assert mock_system["system"] in result
        assert mock_system["release"] in result
        assert mock_system["version"] in result
        assert mock_system["machine"] in result
        assert mock_system["processor"] in result
        assert mock_home in result

        # Verify packages are included
        assert mock_packages in result

        # Verify core sections exist
        assert "Core Principles" in result
        assert "Response Flow" in result
        assert "Response Format" in result

        # Verify agent system prompt is included
        assert "Test agent system prompt" in result


def test_get_tools_str():
    from pydantic import BaseModel, Field

    class TestModel(BaseModel):
        """A test model for documentation."""

        name: str = Field(description="The name field")
        value: int = Field(description="The value field")
        optional: Optional[bool] = Field(False, description="An optional boolean field")

    test_cases = [
        {"name": "No registry provided", "registry": None, "expected": ""},
        {
            "name": "Empty registry (0 tools)",
            "registry": ToolRegistry(),
            "expected": "",
        },
        {
            "name": "One tool registry",
            "registry": ToolRegistry(),
            "expected": "- test_func(param1: str, param2: int) -> bool: Test function description",
        },
        {
            "name": "Two tool registry",
            "registry": ToolRegistry(),
            "expected": (
                "- test_func(param1: str, param2: int) -> bool: Test function description\n"
                "- other_func(name: str) -> str: Another test function"
            ),
        },
        {
            "name": "Async tool registry",
            "registry": ToolRegistry(),
            "expected": "- async async_func(url: str) -> Coroutine[str]: Async test function",
        },
        {
            "name": "Default init registry",
            "registry": ToolRegistry(),
            "expected": (
                """
- async get_page_html_content(url: str) -> Coroutine[str]: Browse to a URL using Playwright to render JavaScript and return the full HTML page content.  Use this for any URL that you want to get the full HTML content of for scraping and understanding the HTML format of the page.
- async get_page_text_content(url: str) -> Coroutine[str]: Browse to a URL using Playwright to render JavaScript and extract clean text content.  Use this for any URL that you want to read the content for, for research purposes. Extracts text from semantic elements like headings, paragraphs, lists etc. and returns a cleaned text representation of the page content.
- list_working_directory(max_depth: int = 3) -> Dict: List the files in the current directory showing files and their metadata.

## Response Type Formats

### Dict
Custom return type (print the output to the console to read and interpret in following steps)
""".strip()  # noqa: E501
            ),
        },
        {
            "name": "Function with default argument",
            "registry": ToolRegistry(),
            "expected": "- func_with_default_arg(arg: str = 'default') -> str: "
            "Function with default argument",
        },
        {
            "name": "Function with Pydantic return type",
            "registry": ToolRegistry(),
            "expected": (
                """- pydantic_return_func() -> TestModel: Function returning a Pydantic model

## Response Type Formats

### TestModel
A test model for documentation.
```json
{
  "name": "string value",
  "value": 0,
  "optional": null
}
```

Fields:
- `name` (string): The name field
- `optional` (Optional[boolean]): An optional boolean field
- `value` (integer): The value field"""
            ),
        },
    ]

    # Set up test functions
    def test_func(param1: str, param2: int) -> bool:
        """Test function description"""
        return True

    test_func.__name__ = "test_func"
    test_func.__doc__ = "Test function description"

    def other_func(name: str) -> str:
        """Another test function"""
        return name

    other_func.__name__ = "other_func"
    other_func.__doc__ = "Another test function"

    async def async_func(url: str) -> str:
        """Async test function"""
        return url

    async_func.__name__ = "async_func"
    async_func.__doc__ = "Async test function"

    def func_with_default_arg(arg: str = "default") -> str:
        """Function with default argument"""
        return arg

    func_with_default_arg.__name__ = "func_with_default_arg"
    func_with_default_arg.__doc__ = "Function with default argument"

    def pydantic_return_func() -> TestModel:
        """Function returning a Pydantic model"""
        return TestModel(name="test", value=42, optional=True)

    pydantic_return_func.__name__ = "pydantic_return_func"
    pydantic_return_func.__doc__ = "Function returning a Pydantic model"

    # Configure the one tool registry
    test_cases[2]["registry"].add_tool("test_func", test_func)

    # Configure the two tool registry
    test_cases[3]["registry"].add_tool("test_func", test_func)
    test_cases[3]["registry"].add_tool("other_func", other_func)

    # Configure the async tool registry
    test_cases[4]["registry"].add_tool("async_func", async_func)

    # Configure the default init registry
    test_cases[5]["registry"].init_tools()

    # Configure the function with default argument
    test_cases[6]["registry"].add_tool("func_with_default_arg", func_with_default_arg)

    # Configure the function with Pydantic return type
    test_cases[7]["registry"].add_tool("pydantic_return_func", pydantic_return_func)

    # Run test cases
    for case in test_cases:
        result = get_tools_str(case["registry"])
        result_lines = sorted(result.split("\n")) if result else []
        expected_lines = sorted(case["expected"].split("\n")) if case["expected"] else []
        assert (
            result_lines == expected_lines
        ), f"Failed test case: {case['name']}\nExpected: {case['expected']}\nGot: {result}"


def test_get_system_details_str(monkeypatch):
    """Test the get_system_details_str function returns expected system information."""
    # Mock platform functions
    monkeypatch.setattr(platform, "system", lambda: "TestOS")
    monkeypatch.setattr(platform, "release", lambda: "1.0")
    monkeypatch.setattr(platform, "version", lambda: "Test Version")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "processor", lambda: "TestProcessor")

    # Mock psutil functions
    class MockVirtualMemory:
        def __init__(self):
            self.total = 8 * 1024**3  # 8 GB

    monkeypatch.setattr(psutil, "cpu_count", lambda logical: 8 if logical else 4)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: MockVirtualMemory())

    # Mock subprocess for GPU detection
    def mock_check_output(cmd, shell, stderr=None):
        if "nvidia-smi" in cmd:
            return b"GPU 0: Test NVIDIA GPU"
        raise subprocess.SubprocessError()

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    # Mock os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path: "/home/testuser")

    # Get the system details
    result = get_system_details_str()

    # Check that the result contains expected information
    assert "os: TestOS" in result
    assert "release: 1.0" in result
    assert "version: Test Version" in result
    assert "architecture: x86_64" in result
    assert "machine: x86_64" in result
    assert "processor: TestProcessor" in result
    assert "cpu: 4 physical cores, 8 logical cores" in result
    assert "memory: 8.00 GB total" in result
    assert "gpus: GPU 0: Test NVIDIA GPU" in result
    assert "home_directory: /home/testuser" in result


def test_get_system_details_str_fallbacks(monkeypatch):
    """Test the get_system_details_str function handles failures gracefully."""
    # Mock platform functions
    monkeypatch.setattr(platform, "system", lambda: "TestOS")
    monkeypatch.setattr(platform, "release", lambda: "1.0")
    monkeypatch.setattr(platform, "version", lambda: "Test Version")
    monkeypatch.setattr(platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(platform, "processor", lambda: "TestProcessor")

    # Mock psutil functions to raise ImportError
    def raise_import_error(*args, **kwargs):
        raise ImportError("psutil not available")

    monkeypatch.setattr(psutil, "cpu_count", raise_import_error)
    monkeypatch.setattr(psutil, "virtual_memory", raise_import_error)

    # Mock subprocess to always fail
    def mock_failed_check_output(cmd, shell, stderr=None):
        raise subprocess.SubprocessError()

    monkeypatch.setattr(subprocess, "check_output", mock_failed_check_output)

    # Mock os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda path: "/home/testuser")

    # Get the system details
    result = get_system_details_str()

    # Check that the result contains fallback information
    assert "os: TestOS" in result
    assert "cpu: Unknown (psutil not installed)" in result
    assert "memory: Unknown (psutil not installed)" in result
    assert "gpus: No GPUs detected or GPU tools not installed" in result
    assert "home_directory: /home/testuser" in result


def test_get_system_details_str_apple_silicon(monkeypatch):
    """Test the get_system_details_str function detects Apple Silicon GPUs."""
    # Mock platform functions for Apple Silicon
    monkeypatch.setattr(platform, "system", lambda: "Darwin")
    monkeypatch.setattr(platform, "machine", lambda: "arm64")

    # Mock platform.processor to avoid subprocess call
    monkeypatch.setattr(platform, "processor", lambda: "Apple M1")

    # Mock subprocess for Metal detection
    def mock_check_output(cmd, shell=False, stderr=None, text=None, encoding=None):
        if isinstance(cmd, list) and cmd[0] == "uname":
            return "arm"
        if "nvidia-smi" in cmd or "rocm-smi" in cmd:
            raise subprocess.SubprocessError()
        if "system_profiler" in cmd and "Metal" in cmd:
            return b"Metal: Supported"
        raise subprocess.SubprocessError()

    monkeypatch.setattr(subprocess, "check_output", mock_check_output)

    # Get the system details
    result = get_system_details_str()

    # Check that the result contains Apple Silicon GPU information
    assert "gpus: Apple Silicon GPU with Metal support" in result


def test_apply_attachments_to_prompt():
    """Test the apply_attachments_to_prompt function adds attachments section to prompt."""
    # Test case 1: No attachments
    prompt = "Analyze this data"
    result = apply_attachments_to_prompt(prompt, None)
    assert result == prompt

    # Test case 2: Empty attachments list
    result = apply_attachments_to_prompt(prompt, [])
    assert result == prompt

    # Test case 3: With attachments
    attachments = ["file1.txt", "file2.pdf", "https://example.com/data.csv"]
    result = apply_attachments_to_prompt(prompt, attachments)

    # Verify the result contains the original prompt
    assert prompt in result

    # Verify the result contains the attachments section header
    assert "## Attachments" in result
    assert "Please use the following files" in result

    # Verify each attachment is listed
    for attachment in attachments:
        assert attachment in result

    # Verify the numbering format (1. file1.txt, etc.)
    assert "1. file1.txt" in result
    assert "2. file2.pdf" in result
    assert "3. https://example.com/data.csv" in result
