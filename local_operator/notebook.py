"""Module for creating and manipulating IPython notebooks with Local Operator.

This module provides functionalities to create, modify, and save IPython notebooks
in the .ipynb format. It supports adding various cell types (e.g., markdown, code)
and populating them with content, including code execution results, conversation
history, and other relevant metadata. The generated notebooks can be used for
review, sharing, or reproduction of interactions and processes.
"""

import importlib.metadata
import json
from datetime import datetime
from pathlib import Path
from typing import List

from local_operator.executor import CodeExecutionResult
from local_operator.model.configure import ModelConfiguration
from local_operator.types import ConversationRole, ProcessResponseStatus

TitleCellTemplate = """
# 🤖 Local Operator Conversation Notebook 📓

This notebook contains the exported conversation and code execution history from a
<a href='https://local-operator.com'>Local Operator</a> agent session.

## 📊 Session Information

<table style='width: 80%; border-collapse: collapse;'>
  <tr><td style='padding: 8px; font-weight: bold;'>📅 Date and Time</td>
  <td>{current_time}</td></tr>
  <tr><td style='padding: 8px; font-weight: bold;'>🔢 Local Operator Version</td>
  <td>{version}</td></tr>
  <tr><td style='padding: 8px; font-weight: bold;'>🧠 Model</td>
  <td>{model_info}</td></tr>
  <tr><td style='padding: 8px; font-weight: bold;'>☁️ Hosting</td>
  <td>{hosting_info}</td></tr>
  <tr><td style='padding: 8px; font-weight: bold;'>💬 Max Conversation History</td>
  <td>{max_conversation_history}</td></tr>
  <tr><td style='padding: 8px; font-weight: bold;'>📜 Detailed Conversation Length</td>
  <td>{detail_conversation_length}</td></tr>
  <tr><td style='padding: 8px; font-weight: bold;'>📚 Learning History Length</td>
  <td>{max_learnings_history}</td></tr>
</table>

💡 **Tip:** To reproduce this conversation, you can run Local Operator with the
same configuration settings listed above.
"""
"""
Template for the title cell of the conversation notebook.

This template is a markdown string that includes placeholders for dynamic information
such as the current date and time, Local Operator version, model information, hosting
information, and conversation history lengths.  It provides a summary of the session
from which the notebook was generated.
"""


def save_code_history_to_notebook(
    code_history: List[CodeExecutionResult],
    model_configuration: ModelConfiguration,
    max_conversation_history: int,
    detail_conversation_length: int,
    max_learnings_history: int,
    file_path: Path,
) -> None:
    """Save the code execution history to an IPython notebook file (.ipynb).

    This function retrieves the code blocks and their execution results, formats them
    as notebook cells, and saves them to a .ipynb file in JSON format.

    Args:
        code_history: A list of CodeExecutionResult objects representing the code execution history.
        model_configuration: The ModelConfiguration object containing model information.
        max_conversation_history: The maximum number of conversation turns to include
        in the notebook.
        detail_conversation_length: The number of recent conversation turns to include in detail.
        max_learnings_history: The maximum number of learnings to include in the notebook.
        file_path (Path): The path to save the notebook to.

    Raises:
        ValueError: If the file_path is empty.
        Exception: If there is an error during notebook creation or file saving.
    """
    if not file_path:
        raise ValueError("File path is required")

    notebook_content = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    # Add title and description as the first cell
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Get version information
    try:
        version = importlib.metadata.version("local-operator")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        version = "unknown"

    # Get model information
    model_info = model_configuration.name

    # Get hosting information
    hosting_info = model_configuration.hosting
    title_cell_content = TitleCellTemplate.format(
        current_time=current_time,
        version=version,
        model_info=model_info,
        hosting_info=hosting_info,
        max_conversation_history=max_conversation_history,
        detail_conversation_length=detail_conversation_length,
        max_learnings_history=max_learnings_history,
    )

    notebook_content["cells"].append(
        {"cell_type": "markdown", "metadata": {}, "source": title_cell_content}
    )

    for code_result in code_history:
        # Add agent response as a markdown cell
        if code_result.message:
            # Add role-specific icons and formatting
            if code_result.role == ConversationRole.ASSISTANT:
                prefix = "🤖 **Assistant**: "
            elif code_result.role == ConversationRole.USER:
                prefix = "👤 **User**: "
            else:
                prefix = ""

            message_with_prefix = f"{prefix}{code_result.message}"
            notebook_content["cells"].append(
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": message_with_prefix.splitlines(keepends=True),
                }
            )

        cell_source = code_result.code

        # Don't add a code cell if the history record is only a user/assistant message
        if not cell_source:
            continue

        cell_output = ""
        if code_result.stdout:
            cell_output += f"Output:\n{code_result.stdout}\n"
        if code_result.stderr:
            cell_output += f"Errors:\n{code_result.stderr}\n"
        if code_result.logging:
            cell_output += f"Logging:\n{code_result.logging}\n"

        metadata = {}
        if code_result.status != ProcessResponseStatus.SUCCESS:
            metadata["skip_execution"] = True

        notebook_content["cells"].append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": metadata,
                "outputs": [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": cell_output.splitlines(keepends=True),
                    }
                ],
                "source": cell_source.splitlines(keepends=True),
            }
        )

    # Save the notebook to a file
    with open(file_path, "w") as f:
        json.dump(notebook_content, f, indent=1)
