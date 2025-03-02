import json
import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock

import pytest

from local_operator.executor import CodeExecutionResult
from local_operator.notebook import save_code_history_to_notebook
from local_operator.types import ConversationRole, ProcessResponseStatus


@pytest.fixture
def tmp_path() -> Generator[Path, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


def test_save_code_history_to_notebook(tmp_path: Path) -> None:
    """
    Test the save_code_history_to_notebook tool to verify that the code execution history
    is saved to an IPython notebook file.
    """
    file_path = tmp_path / "notebook.ipynb"
    code_history = [
        CodeExecutionResult(
            stdout="",
            stderr="",
            logging="",
            message="Please print hello world",
            code="",
            formatted_print="",
            role=ConversationRole.USER,
            status=ProcessResponseStatus.SUCCESS,
        ),
        CodeExecutionResult(
            stdout="",
            stderr="",
            logging="",
            message="Ok, the plan is that I will print hello world",
            code="",
            formatted_print="",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
        ),
        CodeExecutionResult(
            stdout="",
            stderr="Failed to print 'Lorem ipsum dolor sit amet!'",
            logging="",
            formatted_print="",
            code="print('Lorem ipsum dolor sit amet!')",
            message="I will now print 'Lorem ipsum dolor sit amet!'",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.ERROR,
        ),
        CodeExecutionResult(
            stdout="Hello, world!\n",
            stderr="",
            logging="",
            formatted_print="",
            code="print('Hello, world!')",
            message="I will now print 'Hello, world!'",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
        ),
        CodeExecutionResult(
            stdout="/path/to/cwd\n",
            stderr="",
            logging="",
            formatted_print="",
            code="import os\nprint(os.getcwd())",
            message="I will now print the current working directory",
            role=ConversationRole.ASSISTANT,
            status=ProcessResponseStatus.SUCCESS,
        ),
    ]

    model_configuration = MagicMock()
    model_configuration.name = "test_model"
    model_configuration.hosting = "test_hosting"

    save_code_history_to_notebook(
        code_history=code_history,
        model_configuration=model_configuration,
        max_conversation_history=100,
        detail_conversation_length=10,
        max_learnings_history=50,
        file_path=file_path,
    )

    assert file_path.exists(), "Notebook file was not created"

    with open(file_path, "r", encoding="utf-8") as f:
        notebook_data = json.load(f)

    assert "cells" in notebook_data, "Notebook does not contain cells"

    expected_cells = [
        {
            "cell_type": "markdown",
            "source_contains": "Local Operator Conversation Notebook",
            "description": "First cell (title)",
        },
        {
            "cell_type": "markdown",
            "source_contains": "Please print hello world",
            "description": "First cell (user message)",
        },
        {
            "cell_type": "markdown",
            "source_contains": "Ok, the plan is that I will print hello world",
            "description": "Second cell (assistant message)",
        },
        {
            "cell_type": "markdown",
            "source_contains": "I will now print 'Lorem ipsum dolor sit amet!'",
            "description": "Third cell (response)",
        },
        {
            "cell_type": "code",
            "source_contains": "print('Lorem ipsum dolor sit amet!')",
            "output_contains": "Failed to print 'Lorem ipsum dolor sit amet!'",
            "description": "Fourth cell (error code)",
            "should_skip_execution": True,
        },
        {
            "cell_type": "markdown",
            "source_contains": "I will now print 'Hello, world!'",
            "description": "Fifth cell (response)",
        },
        {
            "cell_type": "code",
            "source_contains": "print('Hello, world!')",
            "output_contains": "Hello, world!",
            "description": "Sixth cell (code)",
        },
        {
            "cell_type": "markdown",
            "source_contains": "I will now print the current working directory",
            "description": "Seventh cell (response)",
        },
        {
            "cell_type": "code",
            "source_contains": "import os",
            "output_contains": "/path/to/cwd",
            "description": "Eighth cell (code)",
        },
    ]

    assert len(notebook_data["cells"]) == len(expected_cells), (
        f"Notebook contains {len(notebook_data['cells'])} cells, " f"expected {len(expected_cells)}"
    )

    for i, expected in enumerate(expected_cells):
        cell = notebook_data["cells"][i]
        assert (
            cell["cell_type"] == expected["cell_type"]
        ), f"{expected['description']} is not a {expected['cell_type']} cell"

        if "source" in expected:
            assert (
                cell["source"] == expected["source"]
            ), f"{expected['description']} source is incorrect"

        if "source_contains" in expected:
            assert expected["source_contains"] in "".join(
                cell["source"]
            ), f"{expected['description']} source code does not contain expected content"

        if "output_contains" in expected and cell["cell_type"] == "code":
            assert expected["output_contains"] in "".join(
                cell["outputs"][0]["text"]
            ), f"{expected['description']} output is incorrect"

        if "should_skip_execution" in expected and cell["cell_type"] == "code":
            assert (
                cell["metadata"]["skip_execution"] == expected["should_skip_execution"]
            ), f"{expected['description']} should_skip_execution is incorrect"
