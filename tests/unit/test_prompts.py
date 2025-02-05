import platform
from unittest.mock import MagicMock, patch

from local_operator.prompts import create_system_prompt, get_tools_str


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

        result = create_system_prompt()

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
        assert "Core Principles:" in result
        assert "Response Flow:" in result
        assert "Response Format:" in result


def test_get_tools_str():
    test_cases = [
        {"name": "No module provided", "module": None, "expected": ""},
        {
            "name": "Empty module (0 functions)",
            "module": MagicMock(__dir__=MagicMock(return_value=[])),
            "expected": "",
        },
        {
            "name": "One function module",
            "module": MagicMock(),
            "expected": "- test_func(param1: str, param2: int) -> bool: Test function description",
        },
        {
            "name": "Two function module",
            "module": MagicMock(),
            "expected": (
                "- test_func(param1: str, param2: int) -> bool: Test function description\n"
                "- other_func(name: str) -> str: Another test function"
            ),
        },
        {
            "name": "Async function module",
            "module": MagicMock(),
            "expected": "- async async_func(url: str) -> str: Async test function",
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

    # Configure the one function module
    test_cases[2]["module"].test_func = test_func
    test_cases[2]["module"].__dir__ = MagicMock(return_value=["test_func"])

    # Configure the two function module
    test_cases[3]["module"].test_func = test_func
    test_cases[3]["module"].other_func = other_func
    test_cases[3]["module"].__dir__ = MagicMock(
        return_value=["test_func", "other_func", "_private"]
    )

    # Configure the async function module
    test_cases[4]["module"].async_func = async_func
    test_cases[4]["module"].__dir__ = MagicMock(return_value=["async_func"])

    # Run test cases
    for case in test_cases:
        result = get_tools_str(case["module"])
        result_lines = sorted(result.split("\n")) if result else []
        expected_lines = sorted(case["expected"].split("\n")) if case["expected"] else []
        assert (
            result_lines == expected_lines
        ), f"Failed test case: {case['name']}\nExpected: {case['expected']}\nGot: {result}"
