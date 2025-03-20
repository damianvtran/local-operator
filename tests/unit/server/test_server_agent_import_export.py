"""
Tests for the agent import and export endpoints of the FastAPI server.

This module contains tests for agent import and export functionality.
"""

import io
import zipfile
from unittest.mock import patch

import pytest
import yaml
from fastapi import UploadFile

from local_operator.agents import AgentEditFields, AgentRegistry


@pytest.mark.asyncio
async def test_import_agent_success(test_app_client, dummy_registry: AgentRegistry, tmp_path):
    """Test importing an agent from a ZIP file."""
    # Create a mock agent.yml file
    agent_data = {
        "id": "old-id",
        "name": "Test Import Agent",
        "created_date": "2024-01-01T00:00:00Z",
        "version": "0.2.16",
        "security_prompt": "Test security prompt",
        "hosting": "openrouter",
        "model": "openai/gpt-4o-mini",
        "description": "A test agent for import",
        "last_message": "",
        "last_message_datetime": "2024-01-01T00:00:00Z",
        "current_working_directory": "/some/old/path",
    }

    # Create a temporary ZIP file with agent.yml
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("agent.yml", yaml.dump(agent_data))
        zip_file.writestr("conversation.jsonl", "")
        zip_file.writestr("execution_history.jsonl", "")
        zip_file.writestr("learnings.jsonl", "")

    # Reset the buffer position to the beginning
    zip_buffer.seek(0)

    # Create a mock UploadFile
    upload_file = UploadFile(
        filename="agent.zip",
        file=zip_buffer,
    )

    # Mock the file read method to return the ZIP content
    with patch.object(upload_file, "read", return_value=zip_buffer.getvalue()):
        pass

    # Send the request with the mock file
    with patch("fastapi.File", return_value=upload_file):
        response = await test_app_client.post(
            "/v1/agents/import",
            files={"file": ("agent.zip", zip_buffer, "application/zip")},
        )

    # Check the response
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == 201
    assert data["message"] == "Agent imported successfully"

    # Verify the agent was created with a new ID and the correct working directory
    result = data["result"]
    assert result["id"] != "old-id"
    assert result["name"] == "Test Import Agent"
    assert result["current_working_directory"] == "~/local-operator-home"

    # Verify the agent exists in the registry
    agent = dummy_registry.get_agent(result["id"])
    assert agent is not None
    assert agent.name == "Test Import Agent"
    assert agent.current_working_directory == "~/local-operator-home"


@pytest.mark.asyncio
async def test_import_agent_missing_yml(test_app_client, dummy_registry: AgentRegistry):
    """Test importing an agent with a missing agent.yml file."""
    # Create a temporary ZIP file without agent.yml
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr("conversation.jsonl", "")
        zip_file.writestr("execution_history.jsonl", "")
        zip_file.writestr("learnings.jsonl", "")

    # Reset the buffer position to the beginning
    zip_buffer.seek(0)

    # Create a mock UploadFile
    upload_file = UploadFile(
        filename="agent.zip",
        file=zip_buffer,
    )

    # Mock the file read method to return the ZIP content
    with patch.object(upload_file, "read", return_value=zip_buffer.getvalue()):
        pass

    # Send the request with the mock file
    with patch("fastapi.File", return_value=upload_file):
        response = await test_app_client.post(
            "/v1/agents/import",
            files={"file": ("agent.zip", zip_buffer, "application/zip")},
        )

    # Check the response
    assert response.status_code == 400
    data = response.json()
    assert "Missing agent.yml in ZIP file" in data["detail"]


@pytest.mark.asyncio
async def test_import_agent_invalid_zip(test_app_client, dummy_registry: AgentRegistry):
    """Test importing an agent with an invalid ZIP file."""
    # Create an invalid ZIP file (just some random bytes)
    invalid_zip = io.BytesIO(b"not a zip file")

    # Send the request with the invalid file
    response = await test_app_client.post(
        "/v1/agents/import",
        files={"file": ("agent.zip", invalid_zip, "application/zip")},
    )

    # Check the response
    assert response.status_code == 400
    data = response.json()
    assert "Invalid ZIP file" in data["detail"]


@pytest.mark.asyncio
async def test_export_agent_success(test_app_client, dummy_registry: AgentRegistry, tmp_path):
    """Test exporting an agent to a ZIP file."""
    # Create a test agent
    agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Export Test Agent",
            security_prompt="Test security prompt",
            hosting="openrouter",
            model="openai/gpt-4o-mini",
            description="A test agent for export",
            current_working_directory="~/local-operator-home",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=20,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        )
    )

    # Save some state files for the agent
    agent_dir = dummy_registry.agents_dir / agent.id
    with open(agent_dir / "test_file.txt", "w") as f:
        f.write("Test content")

    # Export the agent
    response = await test_app_client.get(f"/v1/agents/{agent.id}/export")

    # Check the response
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    assert response.headers["content-disposition"] == 'attachment; filename="Export_Test_Agent.zip"'

    # Verify the ZIP file content
    zip_content = response.content
    zip_buffer = io.BytesIO(zip_content)
    with zipfile.ZipFile(zip_buffer, "r") as zip_file:
        # Check that the expected files are in the ZIP
        file_list = zip_file.namelist()
        assert "agent.yml" in file_list
        assert "test_file.txt" in file_list

        # Check the content of agent.yml
        agent_yml_content = zip_file.read("agent.yml").decode("utf-8")
        agent_data = yaml.safe_load(agent_yml_content)
        # The ID is a UUID, so we just check that it exists and is a string
        assert isinstance(agent_data["id"], str)
        assert agent_data["name"] == "Export Test Agent"

        # Check the content of the test file
        test_file_content = zip_file.read("test_file.txt").decode("utf-8")
        assert test_file_content == "Test content"


@pytest.mark.asyncio
async def test_export_agent_not_found(test_app_client, dummy_registry: AgentRegistry):
    """Test exporting a non-existent agent."""
    # Try to export a non-existent agent
    response = await test_app_client.get("/v1/agents/non-existent-id/export")

    # Check the response
    assert response.status_code == 404
    data = response.json()
    assert "Agent with ID non-existent-id not found" in data["detail"]


@pytest.mark.asyncio
async def test_import_export_roundtrip(test_app_client, dummy_registry: AgentRegistry, tmp_path):
    """Test a full import-export roundtrip to ensure data integrity."""
    # Create a test agent
    original_agent = dummy_registry.create_agent(
        AgentEditFields(
            name="Roundtrip Test Agent",
            security_prompt="Test security prompt",
            hosting="openrouter",
            model="openai/gpt-4o-mini",
            description="A test agent for roundtrip testing",
            current_working_directory="~/local-operator-home",
            last_message="",
            temperature=0.7,
            top_p=1.0,
            top_k=20,
            max_tokens=2048,
            stop=None,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            seed=None,
        )
    )

    # Save some state files for the agent
    agent_dir = dummy_registry.agents_dir / original_agent.id
    with open(agent_dir / "test_file.txt", "w") as f:
        f.write("Test content for roundtrip")

    # Export the agent
    export_response = await test_app_client.get(f"/v1/agents/{original_agent.id}/export")
    assert export_response.status_code == 200

    # Get the ZIP content
    zip_content = export_response.content

    # Now import the exported ZIP
    import_response = await test_app_client.post(
        "/v1/agents/import",
        files={"file": ("agent.zip", io.BytesIO(zip_content), "application/zip")},
    )

    # Check the import response
    assert import_response.status_code == 201
    import_data = import_response.json()
    imported_agent_id = import_data["result"]["id"]

    # Verify the imported agent has the same data (except ID and working directory)
    imported_agent = dummy_registry.get_agent(imported_agent_id)
    assert imported_agent.name == original_agent.name
    assert imported_agent.description == original_agent.description
    assert imported_agent.security_prompt == original_agent.security_prompt
    assert imported_agent.hosting == original_agent.hosting
    assert imported_agent.model == original_agent.model
    assert imported_agent.current_working_directory == "~/local-operator-home"

    # Verify the test file was imported correctly
    imported_agent_dir = dummy_registry.agents_dir / imported_agent.id
    with open(imported_agent_dir / "test_file.txt", "r") as f:
        content = f.read()
        assert content == "Test content for roundtrip"
